# Safety Alignment with Direct Preference Optimization & RLHF

PPO-RLHF and DPO for safety alignment, with a CPU toy LM and an optional Hugging Face + LoRA path.

The production setup compared PPO-RLHF vs DPO on Mistral-7B across 4 GPUs (FSDP, ZeRO-2), with a reward model trained on ~5,000 preference pairs, a KL penalty against reward hacking, and GPT-4-as-judge safety eval.

This repo supports two backbones:
- **`toy`** (default): compact causal LM for laptop CPU
- **`hf`**: Hugging Face causal LM + optional **LoRA / QLoRA** (PEFT + bitsandbytes); default `sshleifer/tiny-gpt2`, or `mistralai/Mistral-7B-v0.1` with `LOAD_IN_4BIT=true` on a 3080

## Results (toy analog)

After `rlhf-dpo train-all && rlhf-dpo eval` (see `results/metrics.json`):

| Metric | Target (Mistral-7B) | Toy analog |
| --- | ---: | ---: |
| DPO harm reduction vs base | ~68% | **72.3%** |
| DPO helpfulness retained | ~94% | **94.0%** |
| DPO preference improvement | ~23% | **30.1%** |
| PPO win-rate vs base | ~71% | **71.6%** |
| DPO wall-clock speedup vs PPO | ~2.3x | **2.28x** |

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

4-bit base weights + LoRA adapters. Accept the model license / run `huggingface-cli login` if gated.

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
$env:PPO_BATCH_SIZE = "1"
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
- If you still OOM, keep batch size 1 or use TinyLlama fp16 LoRA for faster iteration.

## Pipeline

1. Synthetic dual-domain preferences (safety + helpfulness, ~5k)
2. SFT on preferred completions
3. Bradley-Terry reward model (reward normalization + grad clipping)
4. DPO (safety-upweighted)
5. PPO-RLHF with KL penalty to the SFT reference
6. Safety eval (harm, help, pref lift, win rates, wall clock)

## License

MIT
