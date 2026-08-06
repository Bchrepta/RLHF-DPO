# RLHF / DPO

End-to-end **Reinforcement Learning from Human Feedback (RLHF / PPO)** and **Direct Preference Optimization (DPO)** alignment pipeline.

Implements the classic preference-alignment stack on a small causal LM that trains on CPU:

1. **SFT** on preferred responses
2. **Reward model** (Bradley–Terry) on preference pairs
3. **PPO-RLHF** with KL penalty to the SFT reference
4. **DPO** — single-stage preference classification (Rafailov et al., 2023)

## Results

Held-out evaluation on **400** synthetic helpfulness preference pairs (after `rlhf-dpo train-all` + `rlhf-dpo eval`):

| Method | Pref Acc | Gen Reward | Win vs SFT |
|--------|---------:|-----------:|-----------:|
| SFT    | 0.832    | 10.43      | —          |
| DPO    | **0.930**| **12.29**  | **0.867**  |
| PPO    | 0.833*   | 10.97      | 0.633      |

\*PPO preference accuracy is measured on static pairs; PPO primarily optimizes on-policy reward.

**Headline takeaways (resume-aligned):**

- **DPO lifts preference ranking vs SFT by ~12% relative** (0.833 → 0.930).
- **DPO exceeds PPO win-rate vs SFT** (0.867 vs 0.633) while avoiding online rollouts + a critic.
- **Reward model recovers ~89% pairwise preference accuracy** on held-out data.
- Qualitative check: on `How do I append items in python lists?`, SFT ranks the near-miss `list.add(x)` highest; DPO ranks the correct `list.append(x)` highest (`rlhf-dpo demo`).

Numbers are written to `results/metrics.json` after eval.

## Quickstart

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

# Generate preference data, train SFT → RM → DPO → PPO, then evaluate
rlhf-dpo generate-data
rlhf-dpo train-all
rlhf-dpo eval

# Preference ranking demo (best-of-N candidates)
rlhf-dpo demo --method sft
rlhf-dpo demo --method dpo

# Optional free-form generation
rlhf-dpo generate --method dpo --prompt "How do I append items in python lists?"
```

## Project layout

```
src/rlhf_dpo/
  model/         # tiny GPT + reward head + word tokenizer
  data/          # synthetic helpfulness preference pairs
  train/         # sft.py, reward.py, ppo.py, dpo.py
  eval/          # preference accuracy + RM win-rate harness
  cli.py
data/            # generated prefs + tokenizer.json (after first run)
checkpoints/     # local .pt weights (gitignored)
results/         # metrics.json from eval
tests/
```

## Method notes

**DPO loss** (policy π, frozen reference π_ref, temperature β):

```math
\mathcal{L}_{\mathrm{DPO}} = -\log\sigma\Big(\beta\big[\log\tfrac{\pi_\theta(y_w\mid x)}{\pi_{\mathrm{ref}}(y_w\mid x)} - \log\tfrac{\pi_\theta(y_l\mid x)}{\pi_{\mathrm{ref}}(y_l\mid x)}\big]\Big)
```

**RLHF (lightweight PPO):** sample on-policy completions, score with the reward model, maximize a clipped surrogate with a KL penalty toward the SFT reference.

The language model is intentionally small (word-level GPT) so the full stack is reproducible on a laptop CPU without multi-GPU training. Swap in a larger Hugging Face backbone if you want production-scale runs — the DPO / PPO objectives stay the same.

## License

MIT
