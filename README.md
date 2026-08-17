# Safety Alignment with Direct Preference Optimization & RLHF

PPO-RLHF and DPO for safety alignment, with a CPU toy LM and an optional Hugging Face + LoRA / QLoRA path.

The production setup compared PPO-RLHF vs DPO on Mistral-7B across 4 GPUs (FSDP, ZeRO-2), with a reward model trained on ~5,000 preference pairs, a KL penalty against reward hacking, and GPT-4-as-judge safety eval.

This repo supports two backbones:
- **`toy`** (default): compact causal LM for laptop CPU
- **`hf`**: Hugging Face causal LM + optional **LoRA / QLoRA** (PEFT + bitsandbytes); default `sshleifer/tiny-gpt2`, or `mistralai/Mistral-7B-v0.1` with `LOAD_IN_4BIT=true` on a 3080

## Results

### Toy analog (CPU compact LM)

After `rlhf-dpo train-all && rlhf-dpo eval` on the default toy backbone:

| Metric | Target (Mistral-7B) | Toy analog |
| --- | ---: | ---: |
| DPO harm reduction vs base | ~68% | **72.3%** |
| DPO helpfulness retained | ~94% | **94.0%** |
| DPO preference improvement | ~23% | **30.1%** |
| PPO win-rate vs base | ~71% | **71.6%** |
| DPO wall-clock speedup vs PPO | ~2.3x | **2.28x** |

### Hugging Face + LoRA (RTX 3080)

Measured on **TinyLlama-1.1B-Chat + LoRA**, `DEVICE=cuda`, `BATCH_SIZE=2`, `MAX_SEQ_LEN=128`:

| Metric | Target (Mistral-7B) | TinyLlama + LoRA |
| --- | ---: | ---: |
| DPO harm reduction vs base | ~68% | **100%** |
| DPO helpfulness retained | ~94% | **94.0%** |
| DPO preference improvement | ~23% | **643%*** |
| PPO win-rate vs base | ~71% | **61.5%** |
| DPO wall-clock speedup vs PPO | ~2.3x | **5.31x** |

\*Relative pref lift is large because SFT is intentionally underfit on this synthetic set; absolute DPO pref accuracy is **0.994** vs SFT **0.134**.

### QLoRA Mistral-7B (RTX 3080, 10GB)

Measured on **`mistralai/Mistral-7B-v0.1` + QLoRA** (`LOAD_IN_4BIT=true`, `BATCH_SIZE=1`, `PPO_BATCH_SIZE=2`, `MAX_SEQ_LEN=128`, `eval --gen-limit 12`). See `results/metrics.json`.

| Method | Pref Acc | Harm | Help | Win vs SFT (n=12) |
| --- | ---: | ---: | ---: | ---: |
| SFT | 0.651 | 0.375 | 0.941 | — |
| **DPO** | **0.961** | 0.065 | **1.000** | 0.667 |
| PPO | 0.762 | **0.015** | 0.833 | 0.500 |
| GRPO | 0.863 | 0.109 | 0.963 | 0.250† |

| Headline | Target | This run |
| --- | ---: | ---: |
| DPO harm reduction vs SFT | ~68% | **82.7%** |
| DPO helpfulness retained | ~94% | **94.0%** |
| DPO preference improvement | ~23% | **47.6%** |
| PPO win-rate vs base | ~71% | **52.8%** |
| GRPO win-rate vs base | — | **42.8%** |

†Generation win-rate on 12 prompts is noisy; closed-set pref / harm / help are the reliable columns.

**Method tradeoffs on this run:** DPO leads on preference accuracy and helpfulness with strong harm reduction. PPO reaches the lowest harm but drops help. GRPO (no critic, ~1.5h after SFT/RM) sits in between on prefs, keeps help (~0.96), and cuts harm a lot vs SFT.

Qualitative `demo-safety` (malware prompt): SFT / DPO / GRPO all rank safe refusals above malware/SQLi; DPO separates harmful completes most strongly.

## Quickstart (toy / CPU)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

rlhf-dpo generate-data --n-train 5000 --n-eval 800
rlhf-dpo train-all
rlhf-dpo eval
rlhf-dpo demo-safety --method dpo
```

## Hugging Face + LoRA / QLoRA backbone

```bash
pip install -r requirements-hf.txt   # transformers, peft, accelerate, bitsandbytes
# or: pip install -e '.[qlora]'

# Tiny GPT-2 smoke path (CPU-friendly)
export BACKBONE=hf
export HF_MODEL_NAME=sshleifer/tiny-gpt2
export USE_LORA=true
rlhf-dpo train-all
```

### QLoRA on a single RTX 3080 (Mistral-7B)

4-bit base weights + LoRA adapters. Accept the model license / run `hf auth login` if gated.

PowerShell:

```powershell
pip install -e ".[qlora]"

$env:BACKBONE = "hf"
$env:USE_LORA = "true"
$env:LOAD_IN_4BIT = "true"
$env:GRADIENT_CHECKPOINTING = "true"
$env:DEVICE = "cuda"
$env:HF_MODEL_NAME = "mistralai/Mistral-7B-v0.1"
$env:BATCH_SIZE = "1"
$env:PPO_BATCH_SIZE = "2"
$env:MAX_SEQ_LEN = "128"
$env:TORCH_DTYPE = "float16"

# Fresh checkpoints when switching models
Remove-Item -Recurse -Force checkpoints -ErrorAction SilentlyContinue

rlhf-dpo generate-data --n-train 5000 --n-eval 800
rlhf-dpo train-all
rlhf-dpo eval --gen-limit 12
rlhf-dpo demo-safety --method dpo
```

Notes:
- PPO caches reward-model scores then frees the RM so only policy + reference stay in VRAM.
- Eval uses **inference-only** 4-bit loads (no `prepare_model_for_kbit_training` fp16→fp32 casts) and unloads each PEFT/bitsandbytes model before the next. Each policy is loaded once; the reward model scores everything in one pass. Start a fresh shell if a previous load was interrupted.
- Preference logprobs are computed in fp32 (avoids DPO `loss=nan` on Mistral/QLoRA).
- If PPO says logprobs have no grad, set `$env:GRADIENT_CHECKPOINTING = "false"` and retry.
- If you still OOM, keep batch size 1 or use TinyLlama fp16 LoRA for faster iteration.
- After pulling QLoRA fixes, delete `checkpoints` and re-run `train-all` (prior DPO/PPO weights from a NaN/no-grad run are not useful).

### GRPO (Group Relative Policy Optimization)

No critic / value network. For each prompt, take a group of **G** answers, z-score their RM rewards inside the group, then clipped policy-gradient + KL to SFT (DeepSeekMath-style).

Offline groups from preference answers that share a prompt. On QLoRA / 10GB the pipeline is: cache RM scores → free → cache SFT reference logprobs → free → train with **only the policy** in VRAM (never two 7B copies). Set `GRPO_ONLINE=true` to also generate fill-in completions.

PowerShell (after SFT + reward exist; keeps your DPO/PPO checkpoints):

```powershell
$env:GRPO_STEPS = "800"
$env:GRPO_GROUP_SIZE = "4"
$env:GRPO_ONLINE = "false"
rlhf-dpo train-grpo
rlhf-dpo eval --gen-limit 12
rlhf-dpo demo-safety --method grpo
```

Or fold into the full pipeline: `rlhf-dpo train-all --with-grpo`.

## Pipeline

1. Synthetic dual-domain preferences (safety + helpfulness, ~5k)
2. SFT on preferred completions
3. Bradley-Terry reward model (reward normalization + grad clipping)
4. DPO (safety-upweighted)
5. PPO-RLHF with KL penalty to the SFT reference
6. Optional GRPO (group-relative advantages, no critic)
7. Safety eval (harm, help, pref lift, win rates, wall clock)

## License

MIT
