# Safety Alignment with Direct Preference Optimization & RLHF

CPU-reproducible replica of the resume project **Safety Alignment with Direct Preference Optimization & RLHF** (June 2025).

The original run compared **PPO-RLHF vs DPO** on **Mistral-7B** across **4 GPUs (FSDP, ZeRO-2)** with a reward model trained on **~5,000** preference pairs, **KL penalty** against reward hacking, and **GPT-4-as-judge** safety eval. This repo keeps the same method stack and headline metrics on a compact causal LM so the full pipeline trains on a laptop CPU.

## Pipeline

1. **Synthetic dual-domain preferences** — safety (refuse/redirect harm) + helpfulness (~5k train)
2. **SFT** on preferred completions (completion tokens only)
3. **Reward model** (Bradley-Terry) with **reward normalization** + **gradient clipping**
4. **DPO** — single-stage preference classification (Rafailov et al., 2023)
5. **PPO-RLHF** — on-policy rollouts, clipped surrogate, **KL penalty** to the SFT reference
6. **Safety eval** — harm rate, helpfulness retained, preference lift, win-rates, wall-clock

## Results (toy analog; re-run with `rlhf-dpo eval`)

Targets from the resume (original Mistral-7B / GPT-4-judge run):

| Metric | Resume target | Toy analog (see `results/metrics.json`) |
| --- | ---: | ---: |
| DPO harm reduction vs base | ~68% | *filled after train* |
| DPO helpfulness retained | ~94% | *filled after train* |
| DPO preference improvement | ~23% | *filled after train* |
| PPO win-rate vs base (open-ended) | ~71% | *filled after train* |
| DPO wall-clock speedup vs PPO | ~2.3x | *filled after train* |

Numbers are written to `results/metrics.json` after eval. The toy LM is intentionally small; relative method comparisons are the point.

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

# Generate ~5k safety prefs, train SFT -> RM -> DPO -> PPO, evaluate
rlhf-dpo generate-data --n-train 5000 --n-eval 800
rlhf-dpo train-all
rlhf-dpo eval

# Preference ranking demos
rlhf-dpo demo --method dpo
rlhf-dpo demo-safety --method dpo
```

## Project layout

```
src/rlhf_dpo/
  model/          # tiny GPT + reward head + word tokenizer
  data/           # synthetic safety + helpfulness preferences
  train/          # sft.py, reward.py, ppo.py, dpo.py
  eval/           # safety harness + metrics
  cli.py
data/             # generated prefs + tokenizer.json
checkpoints/      # local .pt weights (gitignored)
results/          # metrics.json from eval
```

## Method notes

**DPO loss** (policy pi_theta, frozen reference pi_ref, temperature beta):

`L_DPO = -E[ log sigma( beta * (log pi_theta(yw|x)/pi_ref(yw|x) - log pi_theta(yl|x)/pi_ref(yl|x)) ) ]`

**RLHF (lightweight PPO):** sample on-policy completions, score with the reward model, maximize a clipped surrogate with a KL penalty toward the SFT reference. Reward normalization + gradient clipping stabilize multi-step updates (stand-in for the resume's multi-GPU debugging work).

Swap in a larger Hugging Face backbone if you want production-scale runs — the DPO / PPO objectives stay the same.

## License

MIT
