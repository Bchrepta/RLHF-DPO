from __future__ import annotations

import copy
import json
from pathlib import Path

import torch
from tqdm import tqdm

from rlhf_dpo.config import Settings
from rlhf_dpo.utils import (
    build_lm,
    build_reward_model,
    build_tokenizer,
    completion_logprobs,
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

    Sample on-policy completions, score with the reward model, and take a clipped
    policy-gradient step with a KL penalty toward the frozen SFT reference.
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

    prompts = json.loads((data_dir / "prompts.json").read_text(encoding="utf-8"))
    opt = torch.optim.AdamW(policy.parameters(), lr=settings.lr * 0.5)

    running_reward = 0.0
    running_kl = 0.0
    bs = settings.ppo_batch_size
    for step in tqdm(range(settings.ppo_steps), desc="ppo"):
        batch_prompts = [prompts[(step * bs + i) % len(prompts)] for i in range(bs)]

        # Freeze a snapshot for the clipped ratio (PPO).
        old_policy = copy.deepcopy(policy)
        old_policy.eval()
        for p in old_policy.parameters():
            p.requires_grad_(False)

        ids_list, mask_list, plen_list, rewards = [], [], [], []
        policy.eval()
        with torch.no_grad():
            for prompt in batch_prompts:
                prompt_ids = encode_prompt(tokenizer, prompt, settings.max_seq_len // 2).unsqueeze(0).to(device)
                max_new = min(32, settings.max_seq_len - prompt_ids.size(1) - 1)
                gen = policy.generate(
                    prompt_ids,
                    max_new_tokens=max_new,
                    temperature=0.9,
                    eos_id=tokenizer.eos_id,
                )
                # Decode only generated continuation after SEP.
                full = tokenizer.decode(gen[0].tolist(), skip_special=False)
                # Split on sep token text if present.
                sep = tokenizer.sep_token
                if sep in full:
                    response = full.split(sep, 1)[1]
                    response = response.replace(tokenizer.eos_token, "").replace(tokenizer.pad_token, "").strip()
                else:
                    response = tokenizer.decode(gen[0].tolist(), skip_special=True)
                    response = response[len(prompt) :].strip()
                if not response:
                    response = "..."
                ids, mask, plen = encode_pair(tokenizer, prompt, response, settings.max_seq_len)
                ids_b = ids.unsqueeze(0).to(device)
                mask_b = mask.unsqueeze(0).to(device)
                reward = float(rm(ids_b, mask_b).item())
                ids_list.append(ids)
                mask_list.append(mask)
                plen_list.append(plen)
                rewards.append(reward)

        ids_b = torch.stack(ids_list).to(device)
        mask_b = torch.stack(mask_list).to(device)
        plen_t = torch.tensor(plen_list, device=device)
        reward_t = torch.tensor(rewards, device=device)

        with torch.no_grad():
            old_logp = completion_logprobs(old_policy, ids_b, mask_b, plen_t)
            ref_logp = completion_logprobs(ref, ids_b, mask_b, plen_t)

        policy.train()
        new_logp = completion_logprobs(policy, ids_b, mask_b, plen_t)
        # Approx KL(π || π_ref) per sequence (nats).
        kl = new_logp - ref_logp
        # Advantage: reward − β KL, centered in-batch.
        shaped = reward_t - settings.ppo_kl_coef * kl.detach()
        advantage = shaped - shaped.mean()

        ratio = torch.exp(new_logp - old_logp.detach())
        unclipped = ratio * advantage
        clipped = torch.clamp(ratio, 1.0 - settings.ppo_clip, 1.0 + settings.ppo_clip) * advantage
        loss = -torch.min(unclipped, clipped).mean()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()

        running_reward = 0.9 * running_reward + 0.1 * float(reward_t.mean().item())
        running_kl = 0.9 * running_kl + 0.1 * float(kl.mean().item())
        if (step + 1) % 25 == 0:
            tqdm.write(
                f"PPO step {step+1}: loss={float(loss.item()):.4f} "
                f"reward_ema={running_reward:.3f} kl_ema={running_kl:.3f}"
            )

    save_checkpoint(policy, out)
    return out
