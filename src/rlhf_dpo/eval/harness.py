from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from tqdm import tqdm

from rlhf_dpo.config import Settings
from rlhf_dpo.data.preferences import PreferencePair, load_prefs
from rlhf_dpo.utils import (
    build_lm,
    build_reward_model,
    build_tokenizer,
    completion_logprob_mean,
    completion_logprobs,
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


def _pref_accuracy(
    policy: torch.nn.Module,
    prefs: list[PreferencePair],
    tokenizer,
    settings: Settings,
    device: torch.device,
) -> float:
    """Whether policy assigns higher completion log-prob to chosen than rejected."""
    policy.eval()
    correct = 0
    with torch.no_grad():
        for p in prefs:
            c_ids, c_mask, c_plen = encode_pair(tokenizer, p.prompt, p.chosen, settings.max_seq_len)
            r_ids, r_mask, r_plen = encode_pair(tokenizer, p.prompt, p.rejected, settings.max_seq_len)
            c = c_ids.unsqueeze(0).to(device)
            cm = c_mask.unsqueeze(0).to(device)
            r = r_ids.unsqueeze(0).to(device)
            rm = r_mask.unsqueeze(0).to(device)
            if completion_logprob_mean(policy, c, cm, c_plen) > completion_logprob_mean(policy, r, rm, r_plen):
                correct += 1
    return correct / max(len(prefs), 1)




def _gen_stats(
    policy: torch.nn.Module,
    sft: torch.nn.Module | None,
    rm: torch.nn.Module,
    prompts: list[str],
    tokenizer,
    settings: Settings,
    device: torch.device,
    limit: int = 80,
) -> tuple[float, float | None, float | None]:
    """Return (mean_gen_reward, win_rate_vs_sft, mean_kl_to_sft)."""
    policy.eval()
    rm.eval()
    if sft is not None:
        sft.eval()
    wins = 0
    rewards: list[float] = []
    kl_vals: list[float] = []
    use = prompts[:limit]
    with torch.no_grad():
        for prompt in use:
            pids = encode_prompt(tokenizer, prompt, settings.max_seq_len // 2).unsqueeze(0).to(device)
            max_new = min(20, settings.max_seq_len - pids.size(1) - 1)
            gen_p = policy.generate(
                pids, max_new_tokens=max_new, temperature=0.5, eos_id=tokenizer.eos_id
            )
            resp_p = decode_response(tokenizer, gen_p[0].tolist(), prompt)
            ids_p, m_p, plen_p = encode_pair(tokenizer, prompt, resp_p, settings.max_seq_len)
            rp = float(rm(ids_p.unsqueeze(0).to(device), m_p.unsqueeze(0).to(device)).item())
            rewards.append(rp)

            if sft is not None:
                gen_s = sft.generate(
                    pids, max_new_tokens=max_new, temperature=0.5, eos_id=tokenizer.eos_id
                )
                resp_s = decode_response(tokenizer, gen_s[0].tolist(), prompt)
                ids_s, m_s, _ = encode_pair(tokenizer, prompt, resp_s, settings.max_seq_len)
                rs = float(rm(ids_s.unsqueeze(0).to(device), m_s.unsqueeze(0).to(device)).item())
                if rp > rs:
                    wins += 1
                ids = ids_p.unsqueeze(0).to(device)
                mask = m_p.unsqueeze(0).to(device)
                # Mean token-level KL approx: logπ − logπ_ref on the policy sample.
                kl = float(
                    (
                        completion_logprob_mean(policy, ids, mask, plen_p)
                        - completion_logprob_mean(sft, ids, mask, plen_p)
                    ).item()
                )
                kl_vals.append(kl)

    mean_r = sum(rewards) / max(len(rewards), 1)
    if sft is None:
        return mean_r, None, None
    return mean_r, wins / max(len(use), 1), sum(kl_vals) / max(len(kl_vals), 1)


def run_eval(
    settings: Settings,
    data_dir: Path | None = None,
    ckpt_dir: Path | None = None,
    gen_limit: int = 80,
) -> AggregateReport:
    data_dir = data_dir or settings.data_dir
    ckpt_dir = ckpt_dir or settings.ckpt_dir
    device = get_device(settings)
    tokenizer = build_tokenizer(data_dir)
    prefs = load_prefs(data_dir / "eval_prefs.json")
    prompts = json.loads((data_dir / "prompts.json").read_text(encoding="utf-8"))

    def load_policy(name: str):
        m = build_lm(settings, tokenizer).to(device)
        path = ckpt_dir / name
        if path.exists():
            load_checkpoint(m, path, device)
        m.eval()
        return m

    sft = load_policy("sft.pt")
    dpo = load_policy("dpo.pt")
    ppo = load_policy("ppo.pt")
    rm = build_reward_model(settings, tokenizer).to(device)
    if (ckpt_dir / "reward.pt").exists():
        load_checkpoint(rm, ckpt_dir / "reward.pt", device)
    rm.eval()

    rm_correct = 0
    with torch.no_grad():
        for p in tqdm(prefs, desc="rm-acc", leave=False):
            c_ids, c_mask, _ = encode_pair(tokenizer, p.prompt, p.chosen, settings.max_seq_len)
            r_ids, r_mask, _ = encode_pair(tokenizer, p.prompt, p.rejected, settings.max_seq_len)
            rc = rm(c_ids.unsqueeze(0).to(device), c_mask.unsqueeze(0).to(device))
            rr = rm(r_ids.unsqueeze(0).to(device), r_mask.unsqueeze(0).to(device))
            if rc > rr:
                rm_correct += 1
    rm_acc = rm_correct / max(len(prefs), 1)

    wall: dict[str, float] = {}

    def metrics_for(name: str, model, vs_sft: bool) -> MethodMetrics:
        t0 = time.time()
        pref_acc = _pref_accuracy(model, prefs, tokenizer, settings, device)
        mean_r, win, kl = _gen_stats(
            model,
            sft if vs_sft else None,
            rm,
            prompts,
            tokenizer,
            settings,
            device,
            limit=gen_limit,
        )
        wall[name] = time.time() - t0
        return MethodMetrics(
            name=name,
            preference_accuracy=pref_acc,
            mean_gen_reward=mean_r,
            win_rate_vs_sft=win,
            mean_kl_to_sft=kl,
        )

    sft_m = metrics_for("sft", sft, vs_sft=False)
    dpo_m = metrics_for("dpo", dpo, vs_sft=True)
    ppo_m = metrics_for("ppo", ppo, vs_sft=True)

    pref_lift = (dpo_m.preference_accuracy - sft_m.preference_accuracy) / max(
        sft_m.preference_accuracy, 1e-6
    )
    dpo_win = dpo_m.win_rate_vs_sft or 0.0
    ppo_win = ppo_m.win_rate_vs_sft or 0.0
    reward_adv = dpo_m.mean_gen_reward - ppo_m.mean_gen_reward

    headline = {
        "dpo_preference_accuracy": round(dpo_m.preference_accuracy, 4),
        "sft_preference_accuracy": round(sft_m.preference_accuracy, 4),
        "dpo_relative_pref_lift_vs_sft": round(pref_lift, 4),
        "dpo_win_rate_vs_sft": round(dpo_win, 4),
        "ppo_win_rate_vs_sft": round(ppo_win, 4),
        "reward_model_pair_accuracy": round(rm_acc, 4),
        "dpo_mean_gen_reward": round(dpo_m.mean_gen_reward, 4),
        "ppo_mean_gen_reward": round(ppo_m.mean_gen_reward, 4),
        "sft_mean_gen_reward": round(sft_m.mean_gen_reward, 4),
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
    )


def save_results(report: AggregateReport, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "metrics.json"

    def convert(obj):
        if hasattr(obj, "_asdict") or hasattr(obj, "__dataclass_fields__"):
            return {k: convert(v) for k, v in asdict(obj).items()}
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    path.write_text(json.dumps(convert(report), indent=2), encoding="utf-8")
    return path
