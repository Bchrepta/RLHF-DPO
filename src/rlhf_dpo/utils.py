from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch

from rlhf_dpo.config import Settings
from rlhf_dpo.model import CausalLM, CharTokenizer, RewardModel


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device(settings: Settings) -> torch.device:
    if settings.device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_tokenizer() -> CharTokenizer:
    return CharTokenizer()


def build_lm(settings: Settings, tokenizer: CharTokenizer) -> CausalLM:
    return CausalLM(
        vocab_size=max(settings.vocab_size, tokenizer.vocab_size),
        d_model=settings.d_model,
        n_heads=settings.n_heads,
        n_layers=settings.n_layers,
        max_seq_len=settings.max_seq_len,
        dropout=settings.dropout,
    )


def build_reward_model(settings: Settings, tokenizer: CharTokenizer) -> RewardModel:
    return RewardModel(build_lm(settings, tokenizer))


def encode_pair(
    tokenizer: CharTokenizer,
    prompt: str,
    response: str,
    max_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (input_ids, attention_mask) for prompt+response."""
    text = f"{prompt} {response}"
    ids = tokenizer.encode(text, add_special=True, max_len=max_len)
    attn = [0 if i == tokenizer.pad_id else 1 for i in ids]
    return torch.tensor(ids, dtype=torch.long), torch.tensor(attn, dtype=torch.long)


def encode_prompt(tokenizer: CharTokenizer, prompt: str, max_len: int) -> torch.Tensor:
    # Leave room for generation; do not pad to max for generation prompts.
    ids = tokenizer.encode(prompt, add_special=True)
    # Drop trailing eos so generation continues.
    if ids and ids[-1] == tokenizer.eos_id:
        ids = ids[:-1]
    return torch.tensor(ids[:max_len], dtype=torch.long)


def save_checkpoint(model: torch.nn.Module, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_checkpoint(model: torch.nn.Module, path: Path, device: torch.device) -> None:
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state)


def batch_iter(items: list, batch_size: int, shuffle: bool = True, seed: int = 0):
    idxs = list(range(len(items)))
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(idxs)
    for start in range(0, len(idxs), batch_size):
        chunk = idxs[start : start + batch_size]
        yield [items[i] for i in chunk]
