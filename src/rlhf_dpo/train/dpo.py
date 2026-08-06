from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from rlhf_dpo.config import Settings
from rlhf_dpo.data.preferences import load_prefs
from rlhf_dpo.utils import (
    batch_iter,
    build_lm,
    build_tokenizer,
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
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = ref_chosen_logps - ref_rejected_logps
    logits = beta * (pi_logratios - ref_logratios)
    return -F.logsigmoid(logits).mean()


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

    tokenizer = build_tokenizer()
    policy = build_lm(settings, tokenizer).to(device)
    ref = build_lm(settings, tokenizer).to(device)
    if sft_ckpt.exists():
        load_checkpoint(policy, sft_ckpt, device)
        load_checkpoint(ref, sft_ckpt, device)
    for p in ref.parameters():
        p.requires_grad_(False)
    ref.eval()

    prefs = load_prefs(data_dir / "train_prefs.json")
    opt = torch.optim.AdamW(policy.parameters(), lr=settings.lr)

    policy.train()
    for epoch in range(settings.dpo_epochs):
        total = 0.0
        steps = 0
        for batch in tqdm(
            list(batch_iter(prefs, settings.batch_size, shuffle=True, seed=settings.seed + epoch)),
            desc=f"dpo:{epoch+1}/{settings.dpo_epochs}",
            leave=False,
        ):
            c_ids, c_mask, r_ids, r_mask = [], [], [], []
            for p in batch:
                ci, cm = encode_pair(tokenizer, p.prompt, p.chosen, settings.max_seq_len)
                ri, rm = encode_pair(tokenizer, p.prompt, p.rejected, settings.max_seq_len)
                c_ids.append(ci)
                c_mask.append(cm)
                r_ids.append(ri)
                r_mask.append(rm)
            c = torch.stack(c_ids).to(device)
            cm = torch.stack(c_mask).to(device)
            r = torch.stack(r_ids).to(device)
            rm = torch.stack(r_mask).to(device)

            policy_c = policy.logprobs(c, cm)
            policy_r = policy.logprobs(r, rm)
            with torch.no_grad():
                ref_c = ref.logprobs(c, cm)
                ref_r = ref.logprobs(r, rm)

            loss = dpo_loss(policy_c, policy_r, ref_c, ref_r, settings.beta)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
            total += float(loss.item())
            steps += 1
        tqdm.write(f"DPO epoch {epoch+1}: loss={total / max(steps, 1):.4f}")

    save_checkpoint(policy, out)
    return out
