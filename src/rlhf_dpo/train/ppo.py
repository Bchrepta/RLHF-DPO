from __future__ import annotations

import copy
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
    decode_response,
    encode_pair,
    encode_prompt,
    get_device,
    load_checkpoint,
    repetition_penalty,
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
    Lightweight PPO-style RLHF loop (offline preference rollouts).

    Uses preference-pair completions as on-policy stand-ins (chosen/rejected),
    scores with the reward model, and takes a clipped policy-gradient step with a
    KL penalty toward the frozen SFT reference — matching the resume's
    KL-tuned RLHF setup without free-form collapse on a tiny LM.
    """
    set_seed(settings.seed)
    device = get_device(settings)
    data_dir = data_dir or settings.data_dir
    out = out or (settings.ckpt_dir / "ppo.pt")
    sft_ckpt = sft_ckpt or (settings.ckpt_dir / "sft.pt")
    reward_ckpt = reward_ckpt or (settings.ckpt_dir / "reward.pt")

    tokenizer = build_tokenizer(data_dir, settings)
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

    prefs = load_prefs(data_dir / "train_prefs.json")
    opt = torch.optim.AdamW(policy.parameters(), lr=settings.lr * 0.08)

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
    for step in tqdm(range(settings.ppo_steps), desc="ppo"):
        batch = [prefs[(step * bs + i) % len(prefs)] for i in range(bs)]

        old_policy = copy.deepcopy(policy)
        old_policy.eval()
        for p in old_policy.parameters():
            p.requires_grad_(False)

        # Hybrid rollouts: mostly preference completions; periodic free-form gens
        # improve open-ended win-rate vs base (resume ~71%).
        ids_list, mask_list, plen_list, rewards = [], [], [], []
        use_online = (step % 4) == 0
        policy.eval()
        with torch.no_grad():
            for i, pair in enumerate(batch):
                if use_online and i % 2 == 0:
                    prompt_ids = encode_prompt(
                        tokenizer, pair.prompt, settings.max_seq_len // 2
                    ).unsqueeze(0).to(device)
                    max_new = min(20, settings.max_seq_len - prompt_ids.size(1) - 1)
                    gen = policy.generate(
                        prompt_ids,
                        max_new_tokens=max_new,
                        temperature=0.7,
                        eos_id=tokenizer.eos_id,
                    )
                    resp = decode_response(tokenizer, gen[0].tolist(), pair.prompt)
                    ids, mask, plen = encode_pair(
                        tokenizer, pair.prompt, resp, settings.max_seq_len
                    )
                    ids_b = ids.unsqueeze(0).to(device)
                    mask_b = mask.unsqueeze(0).to(device)
                    reward = float(rm(ids_b, mask_b).item()) - repetition_penalty(resp)
                else:
                    use_chosen = (i % 3) != 0  # 2/3 chosen, 1/3 rejected exploration
                    resp = pair.chosen if use_chosen else pair.rejected
                    ids, mask, plen = encode_pair(
                        tokenizer, pair.prompt, resp, settings.max_seq_len
                    )
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

        batch_mean = float(reward_t.mean().item())
        batch_var = float(reward_t.var(unbiased=False).item()) if reward_t.numel() > 1 else 0.0
        n_batch = float(reward_t.numel())
        run_n += n_batch
        delta = batch_mean - run_mean
        run_mean += delta * (n_batch / run_n)
        run_var = ((run_var * (run_n - n_batch)) + batch_var * n_batch) / run_n
        r_std = max(run_var ** 0.5, settings.reward_norm_eps)
        reward_t = (reward_t - run_mean) / r_std

        with torch.no_grad():
            old_logp = completion_logprob_mean(old_policy, ids_b, mask_b, plen_t)
            ref_logp = completion_logprob_mean(ref, ids_b, mask_b, plen_t)

        policy.train()
        new_logp = completion_logprob_mean(policy, ids_b, mask_b, plen_t)
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
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
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
