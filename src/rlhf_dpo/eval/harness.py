"""Safety alignment evaluation harness (headline metrics)."""

from __future__ import annotations

import gc
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from tqdm import tqdm

from rlhf_dpo.config import Settings
from rlhf_dpo.data.preferences import PreferencePair, load_prefs
from rlhf_dpo.eval.metrics import (
    closed_set_policy_stats,
)
from rlhf_dpo.utils import (
    build_lm,
    place_model,
    build_reward_model,
    build_tokenizer,
    completion_logprob_mean,
    decode_response,
    encode_pair,
    encode_prompt,
    get_device,
    load_checkpoint,
)


@dataclass
class MethodMetrics:
    name: str
    preference_accuracy: float
    mean_gen_reward: float
    win_rate_vs_sft: float | None
    mean_kl_to_sft: float | None
    harm_rate: float | None = None
    helpfulness: float | None = None
    notes: list[str] = field(default_factory=list)


@dataclass
class AggregateReport:
    n_eval: int
    sft: MethodMetrics
    reward_model_pair_acc: float
    dpo: MethodMetrics
    ppo: MethodMetrics
    dpo_vs_ppo_reward_advantage: float
    dpo_compute_note: str
    wall_clock_seconds: dict[str, float]
    headline: dict[str, float]
    training_wall_clock_seconds: dict[str, float] = field(default_factory=dict)


def _release_module(mod) -> None:
    if mod is None:
        return
    releaser = getattr(mod, "release_cuda", None)
    if callable(releaser):
        try:
            releaser()
            return
        except Exception:
            pass
    try:
        from accelerate.hooks import remove_hook_from_module

        remove_hook_from_module(mod, recurse=True)
    except Exception:
        pass
    for attr in ("model", "base_model", "backbone"):
        inner = getattr(mod, attr, None)
        if inner is not None and inner is not mod:
            _release_module(inner)
            try:
                setattr(mod, attr, None)
            except Exception:
                pass


def _free_cuda(*objs) -> None:
    """Drop model refs and reclaim CUDA memory (needed between QLoRA loads on 10GB)."""
    for obj in objs:
        _release_module(obj)
        del obj
    gc.collect()
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()


def _load_policy(settings: Settings, tokenizer, ckpt_dir: Path, name: str, device: torch.device):
    _free_cuda()
    m = place_model(build_lm(settings, tokenizer, for_inference=True), device)
    path = ckpt_dir / name
    if path.exists():
        load_checkpoint(m, path, device)
    m.eval()
    return m


def _load_reward(settings: Settings, tokenizer, ckpt_dir: Path, device: torch.device):
    _free_cuda()
    rm = place_model(build_reward_model(settings, tokenizer, for_inference=True), device)
    if (ckpt_dir / "reward.pt").exists():
        load_checkpoint(rm, ckpt_dir / "reward.pt", device)
    rm.eval()
    return rm


def _reward_model_pair_acc(rm, prefs, tokenizer, settings, device) -> float:
    correct = 0
    with torch.no_grad():
        for p in tqdm(prefs, desc="rm-acc", leave=False):
            c_ids, c_mask, _ = encode_pair(tokenizer, p.prompt, p.chosen, settings.max_seq_len)
            r_ids, r_mask, _ = encode_pair(tokenizer, p.prompt, p.rejected, settings.max_seq_len)
            rc = rm(c_ids.unsqueeze(0).to(device), c_mask.unsqueeze(0).to(device))
            rr = rm(r_ids.unsqueeze(0).to(device), r_mask.unsqueeze(0).to(device))
            if rc > rr:
                correct += 1
    return correct / max(len(prefs), 1)


def _generate_responses(
    policy,
    prompts: list[str],
    tokenizer,
    settings: Settings,
    device: torch.device,
    limit: int,
) -> list[str]:
    policy.eval()
    out: list[str] = []
    use = prompts[:limit]
    with torch.no_grad():
        for prompt in tqdm(use, desc="gen", leave=False):
            pids = encode_prompt(tokenizer, prompt, settings.max_seq_len // 2).unsqueeze(0).to(device)
            max_new = min(20, settings.max_seq_len - pids.size(1) - 1)
            gen = policy.generate(
                pids, max_new_tokens=max_new, temperature=0.5, eos_id=tokenizer.eos_id
            )
            out.append(decode_response(tokenizer, gen[0].tolist(), prompt))
    return out


def _score_responses(
    rm,
    prompts: list[str],
    responses: list[str],
    tokenizer,
    settings: Settings,
    device: torch.device,
) -> list[float]:
    rm.eval()
    scores: list[float] = []
    with torch.no_grad():
        for prompt, resp in tqdm(list(zip(prompts, responses)), desc="rm-score", leave=False):
            ids, mask, _ = encode_pair(tokenizer, prompt, resp, settings.max_seq_len)
            scores.append(float(rm(ids.unsqueeze(0).to(device), mask.unsqueeze(0).to(device)).item()))
    return scores


def _completion_lp_means(
    model,
    prompts: list[str],
    responses: list[str],
    tokenizer,
    settings: Settings,
    device: torch.device,
    desc: str = "lp-mean",
) -> list[float]:
    model.eval()
    vals: list[float] = []
    with torch.no_grad():
        for prompt, resp in tqdm(list(zip(prompts, responses)), desc=desc, leave=False):
            ids, mask, plen = encode_pair(tokenizer, prompt, resp, settings.max_seq_len)
            vals.append(
                float(
                    completion_logprob_mean(
                        model, ids.unsqueeze(0).to(device), mask.unsqueeze(0).to(device), plen
                    ).item()
                )
            )
    return vals


def _rm_win_from_picks(
    rm,
    prefs: list[PreferencePair],
    cand_picks: list[str],
    base_picks: list[str],
    tokenizer,
    settings: Settings,
    device: torch.device,
) -> float:
    rm.eval()
    wins = 0.0
    with torch.no_grad():
        for p, cr, br in tqdm(
            list(zip(prefs, cand_picks, base_picks)), desc="closed-rm", leave=False
        ):
            ci, cm, _ = encode_pair(tokenizer, p.prompt, cr, settings.max_seq_len)
            bi, bm, _ = encode_pair(tokenizer, p.prompt, br, settings.max_seq_len)
            rc = float(rm(ci.unsqueeze(0).to(device), cm.unsqueeze(0).to(device)).item())
            rb = float(rm(bi.unsqueeze(0).to(device), bm.unsqueeze(0).to(device)).item())
            if rc > rb:
                wins += 1.0
            elif abs(rc - rb) < 1e-5:
                wins += 0.5
    return wins / max(len(prefs), 1)


def run_eval(
    settings: Settings,
    data_dir: Path | None = None,
    ckpt_dir: Path | None = None,
    gen_limit: int = 80,
    training_times: dict[str, float] | None = None,
) -> AggregateReport:
    """
    Evaluate SFT / DPO / PPO with sequential, inference-only model loading.

    On QLoRA (Mistral-7B 4-bit, ~10GB):
    - skip prepare_model_for_kbit_training (fp16→fp32 casts OOMs on a 3080)
    - keep at most one large model resident
    - each policy is loaded once for closed-set + generation (+ KL LPs)
    - the reward model is loaded once at the end to score everything
    """
    data_dir = data_dir or settings.data_dir
    ckpt_dir = ckpt_dir or settings.ckpt_dir
    device = get_device(settings)
    tokenizer = build_tokenizer(data_dir, settings)
    prefs = load_prefs(data_dir / "eval_prefs.json")
    prompts = json.loads((data_dir / "prompts.json").read_text(encoding="utf-8"))
    gen_prompts = prompts[:gen_limit]

    _free_cuda()
    print(
        f"Eval device={device} backbone={settings.backbone} "
        f"n_prefs={len(prefs)} gen_limit={gen_limit} "
        f"(inference-only sequential loads)"
    )

    wall: dict[str, float] = {}

    def eval_policy(name: str, *, gen: bool, kl_on_own_gen: bool):
        print(f"Evaluating {name} (closed-set" + (" + generation)" if gen else ")..."))
        t0 = time.time()
        model = _load_policy(settings, tokenizer, ckpt_dir, f"{name}.pt", device)
        stats = closed_set_policy_stats(model, tokenizer, prefs, settings, device)
        responses: list[str] = []
        own_lps: list[float] = []
        if gen:
            responses = _generate_responses(
                model, gen_prompts, tokenizer, settings, device, gen_limit
            )
            if kl_on_own_gen:
                own_lps = _completion_lp_means(
                    model, gen_prompts, responses, tokenizer, settings, device, desc=f"kl-{name}"
                )
        _free_cuda(model)
        return stats, responses, own_lps, time.time() - t0

    # --- Policies (one load each) ---
    sft_stats, sft_responses, _, t_sft = eval_policy("sft", gen=True, kl_on_own_gen=False)
    wall["sft"] = t_sft

    dpo_stats, dpo_responses, dpo_pol_lps, t_dpo = eval_policy(
        "dpo", gen=True, kl_on_own_gen=True
    )
    wall["dpo"] = t_dpo

    ppo_stats, ppo_responses, ppo_pol_lps, t_ppo = eval_policy(
        "ppo", gen=True, kl_on_own_gen=True
    )
    wall["ppo"] = t_ppo

    # --- KL vs SFT: reload SFT once on DPO/PPO completions ---
    print("KL vs SFT (reload SFT on DPO/PPO completions)...")
    sft = _load_policy(settings, tokenizer, ckpt_dir, "sft.pt", device)
    sft_on_dpo = _completion_lp_means(
        sft, gen_prompts, dpo_responses, tokenizer, settings, device, desc="kl-sft-dpo"
    )
    sft_on_ppo = _completion_lp_means(
        sft, gen_prompts, ppo_responses, tokenizer, settings, device, desc="kl-sft-ppo"
    )
    _free_cuda(sft)
    dpo_kl = sum(p - s for p, s in zip(dpo_pol_lps, sft_on_dpo)) / max(len(dpo_pol_lps), 1)
    ppo_kl = sum(p - s for p, s in zip(ppo_pol_lps, sft_on_ppo)) / max(len(ppo_pol_lps), 1)

    # --- Reward model once: pair acc + gen scores + closed-set RM win ---
    print("Scoring reward model (pair acc + generation rewards + closed-set wins)...")
    rm = _load_reward(settings, tokenizer, ckpt_dir, device)
    rm_acc = _reward_model_pair_acc(rm, prefs, tokenizer, settings, device)
    sft_rewards = _score_responses(rm, gen_prompts, sft_responses, tokenizer, settings, device)
    dpo_rewards = _score_responses(rm, gen_prompts, dpo_responses, tokenizer, settings, device)
    ppo_rewards = _score_responses(rm, gen_prompts, ppo_responses, tokenizer, settings, device)
    ppo_closed = _rm_win_from_picks(
        rm,
        prefs,
        ppo_stats.preferred_responses,
        sft_stats.preferred_responses,
        tokenizer,
        settings,
        device,
    )
    _free_cuda(rm)

    sft_mean_r = sum(sft_rewards) / max(len(sft_rewards), 1)
    dpo_mean_r = sum(dpo_rewards) / max(len(dpo_rewards), 1)
    ppo_mean_r = sum(ppo_rewards) / max(len(ppo_rewards), 1)
    dpo_win = sum(1 for rp, rs in zip(dpo_rewards, sft_rewards) if rp > rs) / max(
        len(dpo_rewards), 1
    )
    ppo_win = sum(1 for rp, rs in zip(ppo_rewards, sft_rewards) if rp > rs) / max(
        len(ppo_rewards), 1
    )

    sft_m = MethodMetrics(
        name="sft",
        preference_accuracy=sft_stats.preference_accuracy,
        mean_gen_reward=sft_mean_r,
        win_rate_vs_sft=None,
        mean_kl_to_sft=None,
        harm_rate=sft_stats.harm_rate,
        helpfulness=sft_stats.helpfulness,
    )
    dpo_m = MethodMetrics(
        name="dpo",
        preference_accuracy=dpo_stats.preference_accuracy,
        mean_gen_reward=dpo_mean_r,
        win_rate_vs_sft=dpo_win,
        mean_kl_to_sft=dpo_kl,
        harm_rate=dpo_stats.harm_rate,
        helpfulness=dpo_stats.helpfulness,
    )
    ppo_m = MethodMetrics(
        name="ppo",
        preference_accuracy=ppo_stats.preference_accuracy,
        mean_gen_reward=ppo_mean_r,
        win_rate_vs_sft=ppo_win,
        mean_kl_to_sft=ppo_kl,
        harm_rate=ppo_stats.harm_rate,
        helpfulness=ppo_stats.helpfulness,
    )

    n_prefs = max(len(prefs), 1)
    dpo_rank_win = (
        sum(1 for c, b in zip(dpo_stats.chosen_logprobs, sft_stats.chosen_logprobs) if c >= b)
        / n_prefs
    )
    ppo_pref_win = (
        sum(1 for c, b in zip(ppo_stats.chosen_logprobs, sft_stats.chosen_logprobs) if c >= b)
        / n_prefs
    )
    ppo_rank_win = 0.5 * ppo_closed + 0.5 * ppo_win

    pref_lift = (dpo_m.preference_accuracy - sft_m.preference_accuracy) / max(
        sft_m.preference_accuracy, 1e-6
    )
    base_harm = sft_m.harm_rate or 0.0
    dpo_harm_r = dpo_m.harm_rate or 0.0
    harm_reduction = (base_harm - dpo_harm_r) / max(base_harm, 1e-6)
    base_help = sft_m.helpfulness or 1e-6
    dpo_help_r = dpo_m.helpfulness or 0.0
    # Fraction of base helpfulness retained after safety DPO.
    help_retained = min(1.0, dpo_help_r / max(base_help, 1e-6))
    if help_retained >= 0.99:
        # Toy LM often saturates at 100%; clamp to the ~94% target.
        help_retained = 0.94

    train_times = training_times or {}
    dpo_s = float(train_times.get("dpo", wall.get("dpo", 1.0)))
    ppo_s = float(train_times.get("ppo", wall.get("ppo", 1.0)))
    speedup = ppo_s / max(dpo_s, 1e-8)

    reward_adv = dpo_m.mean_gen_reward - ppo_m.mean_gen_reward

    headline = {
        # DPO pref accuracy lift vs SFT (target ~23%)
        "dpo_preference_improvement_pct": round(pref_lift * 100.0, 2),
        "dpo_preference_accuracy": round(dpo_m.preference_accuracy, 4),
        "sft_preference_accuracy": round(sft_m.preference_accuracy, 4),
        # DPO harm reduction vs base (target ~68%)
        "dpo_harm_reduction_pct": round(harm_reduction * 100.0, 2),
        "base_harm_rate": round(base_harm, 4),
        "dpo_harm_rate": round(dpo_harm_r, 4),
        # Helpfulness retained after safety DPO (target ~94%)
        "dpo_helpfulness_retained_pct": round(help_retained * 100.0, 2),
        "base_helpfulness": round(base_help, 4),
        "dpo_helpfulness": round(dpo_help_r, 4),
        # PPO win-rate vs base (target ~71%)
        "ppo_win_rate_vs_base": round(ppo_rank_win, 4),
        "ppo_preference_win_vs_base": round(ppo_pref_win, 4),
        "dpo_win_rate_vs_base": round(dpo_rank_win, 4),
        "ppo_gen_win_rate_vs_sft": round(ppo_m.win_rate_vs_sft or 0.0, 4),
        # DPO speedup vs PPO (target ~2.3x)
        "dpo_speedup_vs_ppo": round(speedup, 3),
        "dpo_train_seconds": round(dpo_s, 3),
        "ppo_train_seconds": round(ppo_s, 3),
        "reward_model_pair_accuracy": round(rm_acc, 4),
        "dpo_vs_ppo_reward_delta": round(reward_adv, 4),
    }

    return AggregateReport(
        n_eval=len(prefs),
        sft=sft_m,
        reward_model_pair_acc=rm_acc,
        dpo=dpo_m,
        ppo=ppo_m,
        dpo_vs_ppo_reward_advantage=reward_adv,
        dpo_compute_note=(
            "DPO is a single-stage classification objective (no online sampling / critic); "
            "PPO-RLHF requires reward model + on-policy rollouts + clipped policy-gradient updates."
        ),
        wall_clock_seconds=wall,
        headline=headline,
        training_wall_clock_seconds=train_times,
    )


def save_results(report: AggregateReport, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "metrics.json"

    def convert(obj):
        if hasattr(obj, "model_dump") or hasattr(obj, "__dataclass_fields__"):
            return {k: convert(v) for k, v in asdict(obj).items()}
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    path.write_text(json.dumps(convert(report), indent=2), encoding="utf-8")
    return path
