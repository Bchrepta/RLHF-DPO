from __future__ import annotations

import copy
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from rlhf_dpo.config import Settings
from rlhf_dpo.utils import (
    build_lm,
    build_reward_model,
    build_tokenizer,
    encode_pair,
    encode_prompt,
    get_device,
    load_checkpoint,
    save_checkpoint,
    set_seed,
)


def train_ppo(
    settings: Settings,
    data_dir: Path | None = None,
    sft_ckpt: Path | None = None,
    reward_ckpt: Path | None = None,
    out: Path | None = None,
) -> Path:
    """
    Lightweight PPO-style RLHF loop.

    For each step: sample a prompt, generate a completion from the policy,
    score with the reward model, and take a clipped policy-gradient step with
    a KL penalty toward the frozen SFT reference (classic RLHF objective).
    """
    set_seed(settings.seed)
    device = get_device(settings)
    data_dir = data_dir or settings.data_dir
    out = out or (settings.ckpt_dir / "ppo.pt")
    sft_ckpt = sft_ckpt or (settings.ckpt_dir / "sft.pt")
    reward_ckpt = reward_ckpt or (settings.ckpt_dir / "reward.pt")

    tokenizer = build_tokenizer()
    policy = build_lm(settings, tokenizer).to(device)
    ref = build_lm(settings, tokenizer).to(device)
    rm = build_reward_model(settings, tokenizer).to(device)

    if sft_ckpt.exists():
        load_checkpoint(policy, sft_ckpt, device)
        load_checkpoint(ref, sft_ckpt, device)
    if reward_ckpt.exists():
        load_checkpoint(rm, reward_ckpt, device)

    for p in ref.parameters():
        p.requires_grad_(False)
    for p in rm.parameters():
        p.requires_grad_(False)
    ref.eval()
    rm.eval()

    import json

    prompts = json.loads((data_dir / "prompts.json").read_text(encoding="utf-8"))
    opt = torch.optim.AdamW(policy.parameters(), lr=settings.lr * 0.5)

    running_reward = 0.0
    for step in tqdm(range(settings.ppo_steps), desc="ppo"):
        prompt = prompts[step % len(prompts)]
        prompt_ids = encode_prompt(tokenizer, prompt, settings.max_seq_len // 2).unsqueeze(0).to(device)

        # Capture old logprob for clipped surrogate by freezing a snapshot.
        old_policy = copy.deepcopy(policy)
        old_policy.eval()
        for p in old_policy.parameters():
            p.requires_grad_(False)

        with torch.no_grad():
            gen = policy.generate(
                prompt_ids,
                max_new_tokens=min(24, settings.max_seq_len - prompt_ids.size(1)),
                temperature=0.9,
                eos_id=tokenizer.eos_id,
            )
            # Decode response portion for reward scoring.
            full_text = tokenizer.decode(gen[0].tolist(), skip_special=True)
            response = full_text[len(prompt) :].strip() or full_text
            ids, mask = encode_pair(tokenizer, prompt, response, settings.max_seq_len)
            ids = ids.unsqueeze(0).to(device)
            mask = mask.unsqueeze(0).to(device)
            reward = rm(ids, mask)
            old_logp = old_policy.logprobs(ids, mask)
            ref_logp = ref.logprobs(ids, mask)

        policy.train()
        new_logp = policy.logprobs(ids, mask)
        # KL approx vs reference
        kl = new_logp - ref_logp
        advantage = (reward - settings.ppo_kl_coef * kl).detach()

        ratio = torch.exp(new_logp - old_logp.detach())
        unclipped = ratio * advantage
        clipped = torch.clamp(ratio, 1.0 - settings.ppo_clip, 1.0 + settings.ppo_clip) * advantage
        loss = -torch.min(unclipped, clipped).mean()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()

        running_reward = 0.9 * running_reward + 0.1 * float(reward.mean().item())
        if (step + 1) % 30 == 0:
            tqdm.write(
                f"PPO step {step+1}: loss={float(loss.item()):.4f} "
                f"reward_ema={running_reward:.3f} kl={float(kl.mean().item()):.3f}"
            )

    save_checkpoint(policy, out)
    return out
