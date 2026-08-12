from __future__ import annotations

import gc
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from rlhf_dpo.config import Settings
from rlhf_dpo.data.preferences import load_prefs
from rlhf_dpo.utils import (
    batch_iter,
    build_lm,
    place_model,
    build_tokenizer,
    completion_logprobs,
    encode_pair,
    get_device,
    load_checkpoint,
    save_checkpoint,
    set_seed,
)


def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """Standard DPO loss (Rafailov et al., 2023)."""
    # Keep the preference margin math in fp32 for QLoRA stability.
    policy_chosen_logps = policy_chosen_logps.float()
    policy_rejected_logps = policy_rejected_logps.float()
    ref_chosen_logps = ref_chosen_logps.float()
    ref_rejected_logps = ref_rejected_logps.float()
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = ref_chosen_logps - ref_rejected_logps
    logits = beta * (pi_logratios - ref_logratios)
    return -F.logsigmoid(logits).mean()


def _resolve_dpo_max_steps(settings: Settings, n_batches: int) -> int:
    """Cap DPO steps for QLoRA unless the user set DPO_MAX_STEPS explicitly."""
    configured = int(getattr(settings, "dpo_max_steps", 0) or 0)
    if configured > 0:
        return min(configured, n_batches)
    if getattr(settings, "load_in_4bit", False):
        # ~1500 steps * ~6-12s ~= a few hours on a 3080 instead of ~30h.
        return min(1500, n_batches)
    return n_batches


def train_dpo(
    settings: Settings,
    data_dir: Path | None = None,
    sft_ckpt: Path | None = None,
    out: Path | None = None,
) -> Path:
    """Direct Preference Optimization from an SFT reference policy."""
    set_seed(settings.seed)
    device = get_device(settings)
    data_dir = data_dir or settings.data_dir
    out = out or (settings.ckpt_dir / "dpo.pt")
    sft_ckpt = sft_ckpt or (settings.ckpt_dir / "sft.pt")

    tokenizer = build_tokenizer(data_dir, settings)
    prefs = load_prefs(data_dir / "train_prefs.json")
    # Upweight safety pairs. Keep this light for QLoRA so wall-clock stays sane.
    safety = [p for p in prefs if getattr(p, "domain", "") == "safety"]
    if getattr(settings, "load_in_4bit", False):
        # +25% safety instead of doubling the whole safety slice.
        extra = safety[: max(1, len(safety) // 4)]
        prefs = list(prefs) + extra
    else:
        prefs = list(prefs) + safety

    # Build reference, cache logprobs once, then free it (halves DPO compute + VRAM).
    ref = place_model(build_lm(settings, tokenizer), device)
    if sft_ckpt.exists():
        load_checkpoint(ref, sft_ckpt, device)
    for p in ref.parameters():
        p.requires_grad_(False)
    ref.eval()

    print(f"DPO: caching reference logprobs for {len(prefs)} pairs...")
    ref_chosen: list[float] = []
    ref_rejected: list[float] = []
    encoded: list[tuple[torch.Tensor, torch.Tensor, int, torch.Tensor, torch.Tensor, int]] = []
    with torch.no_grad():
        for pair in tqdm(prefs, desc="dpo-cache-ref", leave=False):
            ci, cm, cp = encode_pair(tokenizer, pair.prompt, pair.chosen, settings.max_seq_len)
            ri, rm, rp = encode_pair(tokenizer, pair.prompt, pair.rejected, settings.max_seq_len)
            c = ci.unsqueeze(0).to(device)
            cm_b = cm.unsqueeze(0).to(device)
            r = ri.unsqueeze(0).to(device)
            rm_b = rm.unsqueeze(0).to(device)
            ref_chosen.append(float(completion_logprobs(ref, c, cm_b, cp).item()))
            ref_rejected.append(float(completion_logprobs(ref, r, rm_b, rp).item()))
            encoded.append((ci, cm, cp, ri, rm, rp))

    del ref
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    policy = place_model(build_lm(settings, tokenizer), device)
    if sft_ckpt.exists():
        load_checkpoint(policy, sft_ckpt, device)

    dpo_lr = settings.lr * getattr(settings, "dpo_lr_mult", 0.25)
    if getattr(settings, "load_in_4bit", False):
        dpo_lr = min(dpo_lr, 2e-5)
    trainable = [p for p in policy.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("DPO: no trainable parameters (LoRA adapters missing?)")
    opt = torch.optim.AdamW(trainable, lr=dpo_lr)

    batches = list(batch_iter(list(range(len(prefs))), settings.batch_size, shuffle=True, seed=settings.seed))
    max_steps = _resolve_dpo_max_steps(settings, len(batches))
    print(
        f"DPO lr={dpo_lr:.2e} trainable_tensors={len(trainable)} "
        f"load_in_4bit={getattr(settings, 'load_in_4bit', False)} "
        f"steps={max_steps}/{len(batches)} (pairs={len(prefs)})"
    )

    policy.train()
    for epoch in range(settings.dpo_epochs):
        total = 0.0
        steps = 0
        epoch_batches = list(
            batch_iter(list(range(len(prefs))), settings.batch_size, shuffle=True, seed=settings.seed + epoch)
        )[:max_steps]
        for index_batch in tqdm(
            epoch_batches,
            desc=f"dpo {epoch+1}/{settings.dpo_epochs}",
            leave=False,
        ):
            c_ids, c_mask, c_plen = [], [], []
            r_ids, r_mask, r_plen = [], [], []
            ref_c_vals, ref_r_vals = [], []
            for idx in index_batch:
                ci, cm, cp, ri, rm, rp = encoded[idx]
                c_ids.append(ci)
                c_mask.append(cm)
                c_plen.append(cp)
                r_ids.append(ri)
                r_mask.append(rm)
                r_plen.append(rp)
                ref_c_vals.append(ref_chosen[idx])
                ref_r_vals.append(ref_rejected[idx])

            c = torch.stack(c_ids).to(device)
            cm = torch.stack(c_mask).to(device)
            r = torch.stack(r_ids).to(device)
            rm = torch.stack(r_mask).to(device)
            cp = torch.tensor(c_plen, device=device)
            rp = torch.tensor(r_plen, device=device)
            ref_c = torch.tensor(ref_c_vals, device=device)
            ref_r = torch.tensor(ref_r_vals, device=device)

            policy_c = completion_logprobs(policy, c, cm, cp)
            policy_r = completion_logprobs(policy, r, rm, rp)

            loss = dpo_loss(policy_c, policy_r, ref_c, ref_r, settings.beta)
            if not torch.isfinite(loss):
                tqdm.write(f"DPO skip non-finite loss={float(loss.detach().cpu())}")
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            total += float(loss.item())
            steps += 1
        tqdm.write(f"DPO epoch {epoch+1}: loss={total / max(steps, 1):.4f} steps={steps}")

    save_checkpoint(policy, out)
    del policy
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out
