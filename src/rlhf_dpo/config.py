from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    seed: int = 7
    device: str = "cpu"

    # Tiny causal LM — sized to train end-to-end on CPU in minutes.
    vocab_size: int = 512
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    max_seq_len: int = 64
    dropout: float = 0.0

    # Data
    n_train_prefs: int = 2400
    n_eval_prefs: int = 400
    n_prompts: int = 256

    # Training
    sft_epochs: int = 3
    rm_epochs: int = 3
    dpo_epochs: int = 3
    ppo_steps: int = 120
    batch_size: int = 32
    lr: float = 3e-4
    beta: float = 0.1  # DPO / KL temperature
    ppo_clip: float = 0.2
    ppo_kl_coef: float = 0.05

    data_dir: Path = Field(default_factory=lambda: ROOT / "data")
    ckpt_dir: Path = Field(default_factory=lambda: ROOT / "checkpoints")
    results_dir: Path = Field(default_factory=lambda: ROOT / "results")


def get_settings() -> Settings:
    return Settings()
