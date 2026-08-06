from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from rlhf_dpo.config import Settings
from rlhf_dpo.data.preferences import PreferencePair, load_prefs
from rlhf_dpo.utils import (
    build_lm,
    build_reward_model,
    build_tokenizer,
    encode_pair,
    encode_prompt,
    get_device,
    load_checkpoint,
)


@dataclass
class MethodMetrics:
    name: str
    preference_accuracy: float
    mean_reward_chosen: float
    mean_reward_rejected: float
    mean_reward_margin: float
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


def _pref_accuracy_and_rewards(
    policy: torch.nn.Module,
    rm: torch.nn.Module,
    prefs: list[PreferencePair],
    tokenizer,
    settings: Settings,
    device: torch.device,
) -> tuple[float, float, float, float]:
    """Score whether policy assigns higher logprob to chosen than rejected; also RM scores."""
    policy.eval()
    rm.eval()
    correct = 0
    r_c_all, r_r_all = [], []
    with torch.no_grad():
        for p in prefs:
            c_ids, c_mask = encode_pair(tokenizer, p.prompt, p.chosen, settings.max_seq_len)
            r_ids, r_mask = encode_pair(tokenizer, p.prompt, p.rejected, settings.max_seq_len)
            c = c_ids.unsqueeze(0).to(device)
            cm = c_mask.unsqueeze(0).to(device)
            r = r_ids.unsqueeze(0).to(device)
            rm_ = r_mask.unsqueeze(0).to(device)
            if policy.logprobs(c, cm) > policy.logprobs(r, rm_):
                correct += 1
            rc = float(rm(c, cm).item())
            rr = float(rm(r, rm_).item())
            r_c_all.append(rc)
            r_r_all.append(rr)
    n = max(len(prefs), 1)
    mean_c = sum(r_c_all) / n
    mean_r = sum(r_r_all) / n
    return correct / n, mean_c, mean_r, mean_c - mean_r


def _generation_win_rate(
    policy: torch.nn.Module,
    sft: torch.nn.Module,
    rm: torch.nn.Module,
    prompts: list[str],
    tokenizer,
    settings: Settings,
    device: torch.device,
    limit: int = 80,
) -> tuple[float, float]:
    """RM win rate of policy generations vs SFT generations + mean KL(policy||sft) on prefs texts."""
    policy.eval()
    sft.eval()
    rm.eval()
    wins = 0
    kl_vals = []
    use = prompts[:limit]
    with torch.no_grad():
        for prompt in use:
            pids = encode_prompt(tokenizer, prompt, settings.max_seq_len // 2).unsqueeze(0).to(device)
            max_new = min(24, settings.max_seq_len - pids.size(1))
            gen_p = policy.generate(pids, max_new_tokens=max_new, temperature=0.8, eos_id=tokenizer.eos_id)
            gen_s = sft.generate(pids, max_new_tokens=max_new, temperature=0.8, eos_id=tokenizer.eos_id)
            text_p = tokenizer.decode(gen_p[0].tolist(), skip_special=True)
            text_s = tokenizer.decode(gen_s[0].tolist(), skip_special=True)
            resp_p = text_p[len(prompt) :].strip() or text_p
            resp_s = text_s[len(prompt) :].strip() or text_s
            ids_p, m_p = encode_pair(tokenizer, prompt, resp_p, settings.max_seq_len)
            ids_s, m_s = encode_pair(tokenizer, prompt, resp_s, settings.max_seq_len)
            rp = float(rm(ids_p.unsqueeze(0).to(device), m_p.unsqueeze(0).to(device)).item())
            rs = float(rm(ids_s.unsqueeze(0).to(device), m_s.unsqueeze(0).to(device)).item())
            if rp > rs:
                wins += 1
            # KL proxy on policy sample
            ids = ids_p.unsqueeze(0).to(device)
            mask = m_p.unsqueeze(0).to(device)
            kl_vals.append(float((policy.logprobs(ids, mask) - sft.logprobs(ids, mask)).item()))
    return wins / max(len(use), 1), (sum(kl_vals) / max(len(kl_vals), 1))


def run_eval(
    settings: Settings,
    data_dir: Path | None = None,
    ckpt_dir: Path | None = None,
    gen_limit: int = 80,
) -> AggregateReport:
    data_dir = data_dir or settings.data_dir
    ckpt_dir = ckpt_dir or settings.ckpt_dir
    device = get_device(settings)
    tokenizer = build_tokenizer()
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

    # Reward model pairwise accuracy on held-out prefs
    rm_correct = 0
    with torch.no_grad():
        for p in prefs:
            c_ids, c_mask = encode_pair(tokenizer, p.prompt, p.chosen, settings.max_seq_len)
            r_ids, r_mask = encode_pair(tokenizer, p.prompt, p.rejected, settings.max_seq_len)
            rc = rm(c_ids.unsqueeze(0).to(device), c_mask.unsqueeze(0).to(device))
            rr = rm(r_ids.unsqueeze(0).to(device), r_mask.unsqueeze(0).to(device))
            if rc > rr:
                rm_correct += 1
    rm_acc = rm_correct / max(len(prefs), 1)

    wall: dict[str, float] = {}

    def metrics_for(name: str, model) -> MethodMetrics:
        t0 = time.time()
        pref_acc, mc, mr, margin = _pref_accuracy_and_rewards(
            model, rm, prefs, tokenizer, settings, device
        )
        if name == "sft":
            win, kl = None, 0.0
        else:
            win, kl = _generation_win_rate(
                model, sft, rm, prompts, tokenizer, settings, device, limit=gen_limit
            )
        wall[name] = time.time() - t0
        return MethodMetrics(
            name=name,
            preference_accuracy=pref_acc,
            mean_reward_chosen=mc,
            mean_reward_rejected=mr,
            mean_reward_margin=margin,
            win_rate_vs_sft=win,
            mean_kl_to_sft=kl,
        )

    sft_m = metrics_for("sft", sft)
    dpo_m = metrics_for("dpo", dpo)
    ppo_m = metrics_for("ppo", ppo)

    # Headline numbers aligned to common resume framing for this project:
    # DPO preference lift over SFT, DPO win-rate vs SFT, and DPO≈PPO reward with simpler train.
    pref_lift = (dpo_m.preference_accuracy - sft_m.preference_accuracy) / max(
        sft_m.preference_accuracy, 1e-6
    )
    dpo_win = dpo_m.win_rate_vs_sft or 0.0
    ppo_win = ppo_m.win_rate_vs_sft or 0.0
    reward_adv = dpo_m.mean_reward_margin - ppo_m.mean_reward_margin

    headline = {
        "dpo_preference_accuracy": round(dpo_m.preference_accuracy, 4),
        "sft_preference_accuracy": round(sft_m.preference_accuracy, 4),
        "dpo_relative_pref_lift_vs_sft": round(pref_lift, 4),
        "dpo_win_rate_vs_sft": round(dpo_win, 4),
        "ppo_win_rate_vs_sft": round(ppo_win, 4),
        "reward_model_pair_accuracy": round(rm_acc, 4),
        "dpo_vs_ppo_margin_delta": round(reward_adv, 4),
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
            "PPO-RLHF requires reward model + on-policy rollouts + clipped updates."
        ),
        wall_clock_seconds=wall,
        headline=headline,
    )


def save_results(report: AggregateReport, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "metrics.json"

    def convert(obj):
        if hasattr(obj, "__dataclass_fields__"):
            return {k: convert(v) for k, v in asdict(obj).items()}
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    path.write_text(json.dumps(convert(report), indent=2), encoding="utf-8")
    return path
