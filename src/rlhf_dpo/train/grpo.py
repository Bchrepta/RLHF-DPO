"""Group Relative Policy Optimization (DeepSeekMath-style, no critic)."""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path

import torch
from tqdm import tqdm

from rlhf_dpo.config import Settings
from rlhf_dpo.data.preferences import PreferencePair, load_prefs
from rlhf_dpo.utils import (
    assert_trainable_grads,
    build_lm,
    build_reward_model,
    build_tokenizer,
    completion_logprob_mean,
    decode_response,
    disable_gradient_checkpointing,
    encode_pair,
    encode_prompt,
    free_cuda,
    get_device,
    load_checkpoint,
    place_model,
    save_checkpoint,
    set_seed,
)


def _precompute_pref_rewards(
    settings: Settings,
    tokenizer,
    prefs: list[PreferencePair],
    reward_ckpt: Path,
    device: torch.device,
) -> list[tuple[float, float]]:
    free_cuda()
    print("GRPO: loading reward model (inference-only) to cache scores...")
    rm = place_model(build_reward_model(settings, tokenizer, for_inference=True), device)
    if reward_ckpt.exists():
        load_checkpoint(rm, reward_ckpt, device)
    rm.eval()

    scored: list[tuple[float, float]] = []
    with torch.no_grad():
        for pair in tqdm(prefs, desc="grpo-cache-rewards", leave=False):
            c_ids, c_mask, _ = encode_pair(tokenizer, pair.prompt, pair.chosen, settings.max_seq_len)
            r_ids, r_mask, _ = encode_pair(tokenizer, pair.prompt, pair.rejected, settings.max_seq_len)
            rc = float(rm(c_ids.unsqueeze(0).to(device), c_mask.unsqueeze(0).to(device)).item())
            rr = float(rm(r_ids.unsqueeze(0).to(device), r_mask.unsqueeze(0).to(device)).item())
            scored.append((rc, rr))

    free_cuda(rm)
    return scored


def _precompute_ref_logprobs(
    settings: Settings,
    tokenizer,
    pools: dict[str, list[tuple[str, float]]],
    sft_ckpt: Path,
    device: torch.device,
) -> dict[tuple[str, str], float]:
    """
    Score every unique (prompt, response) under the frozen SFT policy, then free it.

    Lets QLoRA GRPO train with only the trainable policy resident (no policy+ref pair).
    """
    free_cuda()
    print("GRPO: caching SFT reference logprobs (inference-only), then freeing ref...")
    ref = place_model(build_lm(settings, tokenizer, for_inference=True), device)
    if sft_ckpt.exists():
        load_checkpoint(ref, sft_ckpt, device)
    ref.eval()

    cache: dict[tuple[str, str], float] = {}
    items = [(p, text) for p, cands in pools.items() for text, _ in cands]
    with torch.no_grad():
        for prompt, text in tqdm(items, desc="grpo-cache-ref-lp", leave=False):
            key = (prompt, text)
            if key in cache:
                continue
            ids, mask, plen = encode_pair(tokenizer, prompt, text, settings.max_seq_len)
            lp = float(
                completion_logprob_mean(
                    ref, ids.unsqueeze(0).to(device), mask.unsqueeze(0).to(device), plen
                ).item()
            )
            cache[key] = lp

    free_cuda(ref)
    print(f"GRPO: cached {len(cache)} reference logprobs")
    return cache


def _build_prompt_pools(
    prefs: list[PreferencePair],
    reward_cache: list[tuple[float, float]] | None,
) -> dict[str, list[tuple[str, float]]]:
    """Unique (response, reward) candidates per prompt for offline GRPO groups."""
    pools: dict[str, list[tuple[str, float]]] = defaultdict(list)
    seen: dict[str, set[str]] = defaultdict(set)
    for i, pair in enumerate(prefs):
        rc, rr = (0.0, 0.0) if reward_cache is None else reward_cache[i]
        for text, reward in ((pair.chosen, rc), (pair.rejected, rr)):
            if text in seen[pair.prompt]:
                continue
            seen[pair.prompt].add(text)
            pools[pair.prompt].append((text, float(reward)))
    return dict(pools)


def _group_advantages(rewards: torch.Tensor, eps: float) -> torch.Tensor:
    """A_i = (r_i - mean(r)) / std(r) within the group (GRPO)."""
    if rewards.numel() < 2:
        return rewards * 0.0
    mean = rewards.mean()
    std = rewards.std(unbiased=False).clamp_min(eps)
    return (rewards - mean) / std


def train_grpo(
    settings: Settings,
    data_dir: Path | None = None,
    sft_ckpt: Path | None = None,
    reward_ckpt: Path | None = None,
    out: Path | None = None,
) -> Path:
    """
    GRPO: sample a group of G answers per prompt, z-score rewards within the group,
    then take a clipped policy-gradient step with KL to the SFT reference.

    QLoRA / 10GB path (default when LOAD_IN_4BIT=true):
      1) cache RM scores → free
      2) cache SFT ref logprobs for all pool answers → free
      3) train with **only the policy** in VRAM (KL uses the cache)

    Non-4bit / plenty of VRAM: keep a live frozen ref beside the policy.
    """
    set_seed(settings.seed)
    device = get_device(settings)
    data_dir = data_dir or settings.data_dir
    out = out or (settings.ckpt_dir / "grpo.pt")
    sft_ckpt = sft_ckpt or (settings.ckpt_dir / "sft.pt")
    reward_ckpt = reward_ckpt or (settings.ckpt_dir / "reward.pt")

    tokenizer = build_tokenizer(data_dir, settings)
    prefs = load_prefs(data_dir / "train_prefs.json")
    group_size = max(2, int(getattr(settings, "grpo_group_size", 4)))
    steps = int(getattr(settings, "grpo_steps", 800))
    clip_eps = float(getattr(settings, "grpo_clip", settings.ppo_clip))
    kl_coef = float(getattr(settings, "grpo_kl_coef", settings.ppo_kl_coef))
    online = bool(getattr(settings, "grpo_online", False))
    quantized = bool(getattr(settings, "load_in_4bit", False))
    # On 4-bit, default offline unless explicitly online.
    if quantized and not getattr(settings, "grpo_online", False):
        online = False
    # Never hold two 7B QLoRA copies: cache SFT logprobs, train with policy only.
    cache_ref = quantized or bool(getattr(settings, "grpo_cache_ref", False))

    reward_cache: list[tuple[float, float]] | None = None
    rm = None
    if quantized or not online:
        print("GRPO: caching reward-model scores then freeing RM")
        reward_cache = _precompute_pref_rewards(settings, tokenizer, prefs, reward_ckpt, device)
    else:
        free_cuda()
        rm = place_model(build_reward_model(settings, tokenizer, for_inference=True), device)
        if reward_ckpt.exists():
            load_checkpoint(rm, reward_ckpt, device)
        for p in rm.parameters():
            p.requires_grad_(False)
        rm.eval()

    pools = _build_prompt_pools(prefs, reward_cache)
    prompt_list = [p for p, cand in pools.items() if len(cand) >= 2]
    if not prompt_list:
        raise RuntimeError("GRPO: need at least one prompt with >=2 scored responses")

    ref_lp_cache: dict[tuple[str, str], float] | None = None
    ref = None
    if cache_ref:
        ref_lp_cache = _precompute_ref_logprobs(settings, tokenizer, pools, sft_ckpt, device)
    else:
        free_cuda()
        print("GRPO: loading frozen reference (inference-only)...")
        ref = place_model(build_lm(settings, tokenizer, for_inference=True), device)
        if sft_ckpt.exists():
            load_checkpoint(ref, sft_ckpt, device)
        for p in ref.parameters():
            p.requires_grad_(False)
        ref.eval()

    free_cuda()
    print("GRPO: loading trainable policy (only large model kept in VRAM)..." if cache_ref else "GRPO: loading trainable policy...")
    policy = place_model(build_lm(settings, tokenizer, for_inference=False), device)
    if sft_ckpt.exists():
        load_checkpoint(policy, sft_ckpt, device)

    disable_gradient_checkpointing(policy)
    trainable = [p for p in policy.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("GRPO: no trainable parameters (LoRA adapters missing?)")

    lr = settings.lr * 0.08
    if quantized:
        lr = min(lr, 1e-5)
    opt = torch.optim.AdamW(trainable, lr=lr)
    print(
        f"GRPO lr={lr:.2e} group_size={group_size} steps={steps} "
        f"online={online} cache_ref={cache_ref} prompts={len(prompt_list)} "
        f"trainable={len(trainable)}"
    )

    rng = random.Random(settings.seed + 17)
    running_reward = 0.0
    running_adv = 0.0
    running_kl = 0.0
    eps = float(settings.reward_norm_eps)

    for step in tqdm(range(steps), desc="grpo"):
        prompt = prompt_list[step % len(prompt_list)]
        candidates = pools[prompt]

        responses: list[str] = []
        rewards: list[float] = []

        take = min(group_size, len(candidates))
        seeded = rng.sample(candidates, k=take)
        for text, reward in seeded:
            responses.append(text)
            rewards.append(reward)

        if online and len(responses) < group_size:
            policy.eval()
            pids = encode_prompt(tokenizer, prompt, settings.max_seq_len // 2).unsqueeze(0).to(device)
            max_new = min(24, settings.max_seq_len - pids.size(1) - 1)
            while len(responses) < group_size:
                with torch.no_grad():
                    gen = policy.generate(
                        pids,
                        max_new_tokens=max(4, max_new),
                        temperature=0.8,
                        eos_id=tokenizer.eos_id,
                    )
                text = decode_response(tokenizer, gen[0].tolist(), prompt)
                if not text or text in responses:
                    text = f"{text} ({len(responses)})".strip()
                responses.append(text)
                if rm is not None:
                    ids, mask, _ = encode_pair(tokenizer, prompt, text, settings.max_seq_len)
                    with torch.no_grad():
                        rewards.append(
                            float(rm(ids.unsqueeze(0).to(device), mask.unsqueeze(0).to(device)).item())
                        )
                else:
                    rewards.append(0.0)

        responses = responses[:group_size]
        rewards = rewards[:group_size]
        if len(responses) < 2:
            continue

        ids_list, mask_list, plen_list = [], [], []
        for text in responses:
            ids, mask, plen = encode_pair(tokenizer, prompt, text, settings.max_seq_len)
            ids_list.append(ids)
            mask_list.append(mask)
            plen_list.append(plen)

        ids_b = torch.stack(ids_list).to(device)
        mask_b = torch.stack(mask_list).to(device)
        plen_t = torch.tensor(plen_list, device=device)
        reward_t = torch.tensor(rewards, device=device, dtype=torch.float32)
        advantage = _group_advantages(reward_t, eps)

        policy.eval()
        with torch.no_grad():
            old_logp = completion_logprob_mean(policy, ids_b, mask_b, plen_t)
            if ref_lp_cache is not None:
                ref_vals = [ref_lp_cache[(prompt, text)] for text in responses]
                ref_logp = torch.tensor(ref_vals, device=device, dtype=old_logp.dtype)
            else:
                assert ref is not None
                ref_logp = completion_logprob_mean(ref, ids_b, mask_b, plen_t)

        new_logp = completion_logprob_mean(policy, ids_b, mask_b, plen_t)
        if not new_logp.requires_grad:
            raise RuntimeError(
                "GRPO: policy logprobs have no grad. Try GRADIENT_CHECKPOINTING=false."
            )

        log_ratio = (ref_logp - new_logp).clamp(-5.0, 5.0)
        kl = (torch.exp(log_ratio) - log_ratio - 1.0).clamp_min(0.0)

        ratio = torch.exp((new_logp - old_logp.detach()).clamp(-2.0, 2.0))
        unclipped = ratio * advantage
        clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantage
        pg_loss = -torch.min(unclipped, clipped).mean()
        loss = pg_loss + kl_coef * kl.mean()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        if step == 0 and float(advantage.detach().abs().max()) > 1e-6:
            assert_trainable_grads(trainable, "GRPO")
        torch.nn.utils.clip_grad_norm_(trainable, 0.5)
        opt.step()

        running_reward = 0.9 * running_reward + 0.1 * float(reward_t.mean().item())
        running_adv = 0.9 * running_adv + 0.1 * float(advantage.detach().abs().mean().item())
        running_kl = 0.9 * running_kl + 0.1 * float(kl.detach().mean().item())
        if (step + 1) % 50 == 0:
            with torch.no_grad():
                ratio_mean = float(ratio.detach().mean().item())
            tqdm.write(
                f"GRPO step {step+1}: loss={float(loss.item()):.4f} "
                f"pg={float(pg_loss.item()):.4f} "
                f"reward_ema={running_reward:.3f} |A|_ema={running_adv:.3f} "
                f"kl_ema={running_kl:.4f} ratio={ratio_mean:.3f}"
            )

    save_checkpoint(policy, out)
    return out
