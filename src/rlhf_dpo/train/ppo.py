from __future__ import annotations

import gc
import json
from pathlib import Path

import torch
from tqdm import tqdm

from rlhf_dpo.config import Settings
from rlhf_dpo.data.preferences import load_prefs
from rlhf_dpo.utils import (
    build_lm,
    build_reward_model,
    build_tokenizer,
    completion_logprob_mean,
    encode_pair,
    get_device,
    is_quantized_module,
    load_checkpoint,
    place_model,
    save_checkpoint,
    set_seed,
)


def _precompute_pref_rewards(
    settings: Settings,
    tokenizer,
    prefs,
    reward_ckpt: Path,
    device: torch.device,
) -> list[tuple[float, float]]:
    """Score chosen/rejected once, then free the reward model (VRAM-friendly for QLoRA)."""
    rm = place_model(build_reward_model(settings, tokenizer), device)
    if reward_ckpt.exists():
        load_checkpoint(rm, reward_ckpt, device)
    rm.eval()
    for p in rm.parameters():
        p.requires_grad_(False)

    scored: list[tuple[float, float]] = []
    with torch.no_grad():
        for pair in tqdm(prefs, desc="ppo-cache-rewards", leave=False):
            c_ids, c_mask, _ = encode_pair(tokenizer, pair.prompt, pair.chosen, settings.max_seq_len)
            r_ids, r_mask, _ = encode_pair(tokenizer, pair.prompt, pair.rejected, settings.max_seq_len)
            rc = float(rm(c_ids.unsqueeze(0).to(device), c_mask.unsqueeze(0).to(device)).item())
            rr = float(rm(r_ids.unsqueeze(0).to(device), r_mask.unsqueeze(0).to(device)).item())
            scored.append((rc, rr))

    del rm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return scored


def train_ppo(
    settings: Settings,
    data_dir: Path | None = None,
    sft_ckpt: Path | None = None,
    reward_ckpt: Path | None = None,
    out: Path | None = None,
) -> Path:
    """
    Lightweight PPO-style RLHF loop (offline preference rollouts).

    Uses preference-pair completions as on-policy stand-ins (chosen/rejected),
    scores with the reward model, and takes a clipped policy-gradient step with a
    KL penalty toward the frozen SFT reference. This is a KL-tuned RLHF setup that
    avoids free-form collapse on a tiny LM.
    """
    set_seed(settings.seed)
    device = get_device(settings)
    data_dir = data_dir or settings.data_dir
    out = out or (settings.ckpt_dir / "ppo.pt")
    sft_ckpt = sft_ckpt or (settings.ckpt_dir / "sft.pt")
    reward_ckpt = reward_ckpt or (settings.ckpt_dir / "reward.pt")

    tokenizer = build_tokenizer(data_dir, settings)
    prefs = load_prefs(data_dir / "train_prefs.json")

    # For QLoRA / large HF models, cache RM scores first so PPO only keeps policy+ref in VRAM.
    # (Avoids 3x Mistral-7B-4bit copies, which will not fit a 10GB 3080.)
    probe = place_model(build_lm(settings, tokenizer), device)
    use_reward_cache = bool(getattr(settings, "load_in_4bit", False)) or is_quantized_module(probe)
    del probe
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    reward_cache: list[tuple[float, float]] | None = None
    rm = None
    if use_reward_cache:
        print("PPO: caching reward-model scores then freeing RM (QLoRA/VRAM mode)")
        reward_cache = _precompute_pref_rewards(settings, tokenizer, prefs, reward_ckpt, device)
    else:
        rm = place_model(build_reward_model(settings, tokenizer), device)
        if reward_ckpt.exists():
            load_checkpoint(rm, reward_ckpt, device)
        for p in rm.parameters():
            p.requires_grad_(False)
        rm.eval()

    policy = place_model(build_lm(settings, tokenizer), device)
    ref = place_model(build_lm(settings, tokenizer), device)

    if sft_ckpt.exists():
        load_checkpoint(policy, sft_ckpt, device)
        load_checkpoint(ref, sft_ckpt, device)

    for p in ref.parameters():
        p.requires_grad_(False)
    ref.eval()

    trainable = [p for p in policy.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("PPO: no trainable parameters (LoRA adapters missing?)")
    ppo_lr = settings.lr * 0.08
    if getattr(settings, "load_in_4bit", False):
        ppo_lr = min(ppo_lr, 1e-5)
    opt = torch.optim.AdamW(trainable, lr=ppo_lr)
    print(f"PPO lr={ppo_lr:.2e} trainable_tensors={len(trainable)}")

    norm_path = reward_ckpt.with_suffix(".norm.json")
    r_mean, r_std = 0.0, 1.0
    if norm_path.exists():
        norm = json.loads(norm_path.read_text(encoding="utf-8"))
        r_mean = float(norm.get("mean", 0.0))
        r_std = max(float(norm.get("std", 1.0)), settings.reward_norm_eps)
    run_mean, run_var, run_n = r_mean, r_std ** 2, 1.0

    running_reward = 0.0
    running_kl = 0.0
    bs = settings.ppo_batch_size
    ppo_steps = int(getattr(settings, "ppo_max_steps", 0) or 0)
    if ppo_steps <= 0:
        ppo_steps = settings.ppo_steps
        if getattr(settings, "load_in_4bit", False):
            ppo_steps = min(ppo_steps, 600)
    print(f"PPO steps={ppo_steps} (configured ppo_steps={settings.ppo_steps})")
    for step in tqdm(range(ppo_steps), desc="ppo"):
        batch_idxs = [((step * bs + i) % len(prefs)) for i in range(bs)]
        batch = [prefs[i] for i in batch_idxs]

        # Preference-pair rollouts (chosen-heavy) with KL to SFT reference.
        ids_list, mask_list, plen_list, rewards = [], [], [], []
        with torch.no_grad():
            for i, (pair, pref_i) in enumerate(zip(batch, batch_idxs)):
                use_chosen = (i % 3) != 0
                resp = pair.chosen if use_chosen else pair.rejected
                ids, mask, plen = encode_pair(tokenizer, pair.prompt, resp, settings.max_seq_len)
                ids_b = ids.unsqueeze(0).to(device)
                mask_b = mask.unsqueeze(0).to(device)
                if reward_cache is not None:
                    rc, rr = reward_cache[pref_i]
                    reward = rc if use_chosen else rr
                else:
                    assert rm is not None
                    reward = float(rm(ids_b, mask_b).item())
                ids_list.append(ids)
                mask_list.append(mask)
                plen_list.append(plen)
                rewards.append(reward)

        ids_b = torch.stack(ids_list).to(device)
        mask_b = torch.stack(mask_list).to(device)
        plen_t = torch.tensor(plen_list, device=device)
        reward_t = torch.tensor(rewards, device=device)

        batch_mean = float(reward_t.mean().item())
        batch_var = float(reward_t.var(unbiased=False).item()) if reward_t.numel() > 1 else 0.0
        n_batch = float(reward_t.numel())
        run_n += n_batch
        delta = batch_mean - run_mean
        run_mean += delta * (n_batch / run_n)
        run_var = ((run_var * (run_n - n_batch)) + batch_var * n_batch) / run_n
        r_std = max(run_var ** 0.5, settings.reward_norm_eps)
        reward_t = (reward_t - run_mean) / r_std

        # old_logp from current weights (no deepcopy — required for quantized models)
        with torch.no_grad():
            old_logp = completion_logprob_mean(policy, ids_b, mask_b, plen_t)
            ref_logp = completion_logprob_mean(ref, ids_b, mask_b, plen_t)

        policy.train()
        new_logp = completion_logprob_mean(policy, ids_b, mask_b, plen_t)
        if not new_logp.requires_grad:
            raise RuntimeError(
                "PPO: policy logprobs have no grad (enable_input_require_grads / "
                "gradient checkpointing). Try GRADIENT_CHECKPOINTING=false or update peft."
            )
        kl = (new_logp - ref_logp).clamp(-2.0, 2.0)
        shaped = reward_t - settings.ppo_kl_coef * kl.detach()
        advantage = shaped - shaped.mean()
        advantage = advantage / (advantage.std(unbiased=False) + 1e-6)

        ratio = torch.exp((new_logp - old_logp.detach()).clamp(-2.0, 2.0))
        unclipped = ratio * advantage
        clipped = torch.clamp(ratio, 1.0 - settings.ppo_clip, 1.0 + settings.ppo_clip) * advantage
        loss = -torch.min(unclipped, clipped).mean()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 0.5)
        opt.step()

        running_reward = 0.9 * running_reward + 0.1 * float(reward_t.mean().item())
        running_kl = 0.9 * running_kl + 0.1 * float(kl.mean().item())
        if (step + 1) % 50 == 0:
            tqdm.write(
                f"PPO step {step+1}: loss={float(loss.item()):.4f} "
                f"reward_ema={running_reward:.3f} kl_ema={running_kl:.3f}"
            )

    save_checkpoint(policy, out)
    return out
