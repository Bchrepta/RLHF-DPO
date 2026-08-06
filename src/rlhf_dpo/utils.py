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
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """
    Encode prompt/response as ``[BOS] prompt [SEP] response [EOS]``.

    Returns ``(input_ids, attention_mask, prompt_len)`` where ``prompt_len`` is the
    number of tokens up to and including ``[SEP]`` (completion starts after that).
    """
    p = tokenizer.encode(prompt, add_special=False)
    r = tokenizer.encode(response, add_special=False)
    ids = [tokenizer.bos_id] + p + [tokenizer.sep_id] + r + [tokenizer.eos_id]
    prompt_len = 1 + len(p) + 1  # bos + prompt + sep
    if len(ids) > max_len:
        ids = ids[:max_len]
        # Keep at least one completion token when possible.
        prompt_len = min(prompt_len, max_len - 1)
    if len(ids) < max_len:
        ids = ids + [tokenizer.pad_id] * (max_len - len(ids))
    attn = [0 if i == tokenizer.pad_id else 1 for i in ids]
    return (
        torch.tensor(ids, dtype=torch.long),
        torch.tensor(attn, dtype=torch.long),
        prompt_len,
    )


def encode_prompt(tokenizer: CharTokenizer, prompt: str, max_len: int) -> torch.Tensor:
    """Leave room for generation: ``[BOS] prompt [SEP]`` (no EOS)."""
    p = tokenizer.encode(prompt, add_special=False)
    ids = [tokenizer.bos_id] + p + [tokenizer.sep_id]
    if len(ids) > max_len:
        ids = ids[:max_len]
    return torch.tensor(ids[:max_len], dtype=torch.long)


def completion_logprobs(
    model: torch.nn.Module,
    ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_len: int | torch.Tensor,
) -> torch.Tensor:
    """Sum of token log-probs on the response span only (after ``[SEP]``)."""
    logits, _ = model(ids[:, :-1])
    logp = torch.nn.functional.log_softmax(logits, dim=-1)
    target = ids[:, 1:]
    token_lp = logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    # target index t corresponds to ids[:, t+1]; keep tokens with position >= prompt_len
    b, t = token_lp.shape
    positions = torch.arange(1, t + 1, device=ids.device).unsqueeze(0).expand(b, -1)
    if isinstance(prompt_len, int):
        comp = positions >= prompt_len
    else:
        comp = positions >= prompt_len.unsqueeze(1)
    pad = attention_mask[:, 1:].bool()
    mask = comp & pad
    return (token_lp * mask.float()).sum(dim=-1)


def completion_logprob_mean(
    model: torch.nn.Module,
    ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_len: int | torch.Tensor,
) -> torch.Tensor:
    """Length-normalized completion log-prob."""
    total = completion_logprobs(model, ids, attention_mask, prompt_len)
    b, t = ids.shape[0], ids.shape[1] - 1
    positions = torch.arange(1, t + 1, device=ids.device).unsqueeze(0).expand(b, -1)
    if isinstance(prompt_len, int):
        comp = positions >= prompt_len
    else:
        comp = positions >= prompt_len.unsqueeze(1)
    n = (comp & attention_mask[:, 1:].bool()).float().sum(dim=-1).clamp_min(1.0)
    return total / n


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
