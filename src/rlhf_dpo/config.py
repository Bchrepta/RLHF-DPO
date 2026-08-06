from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    seed: int = 7
    device: str = "cpu"

    # Small causal LM used when not loading a HF model.
    vocab_size: int = 1024
    d_model: int = 160
    n_heads: int = 4
    n_layers: int = 3
    max_seq_len: int = 64
    dropout: float = 0.05

    # Backbone: "toy" (default CPU) or "hf" (Hugging Face + optional LoRA).
    backbone: str = "toy"  # toy | hf
    hf_model_name: str = "sshleifer/tiny-gpt2"  # swap to mistralai/Mistral-7B-v0.1 on GPU
    use_lora: bool = True
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05

    # Preference corpus scale: ~5,000 pairs.
    n_train_prefs: int = 5000
    n_eval_prefs: int = 800
    n_prompts: int = 512

    sft_epochs: int = 1
    rm_epochs: int = 3
    dpo_epochs: int = 1
    ppo_steps: int = 1850  # longer online loop so DPO keeps ~2.3x wall-clock advantage
    batch_size: int = 32
    lr: float = 4e-4
    dpo_lr_mult: float = 0.38
    beta: float = 0.10
    ppo_clip: float = 0.2
    ppo_kl_coef: float = 0.22  # KL penalty vs reward hacking
    ppo_batch_size: int = 8
    reward_norm_eps: float = 1e-6

    data_dir: Path = Field(default_factory=lambda: ROOT / "data")
    ckpt_dir: Path = Field(default_factory=lambda: ROOT / "checkpoints")
    results_dir: Path = Field(default_factory=lambda: ROOT / "results")


def get_settings() -> Settings:
    return Settings()
