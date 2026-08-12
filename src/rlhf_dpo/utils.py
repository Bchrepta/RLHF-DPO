from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch

from rlhf_dpo.config import Settings
from rlhf_dpo.model import CausalLM, RewardModel, WordTokenizer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device(settings: Settings) -> torch.device:
    """Resolve training device. `auto` picks CUDA when available."""
    want = (settings.device or "auto").lower()
    if want in {"cuda", "auto"} and torch.cuda.is_available():
        return torch.device("cuda")
    if want == "cuda" and not torch.cuda.is_available():
        print("DEVICE=cuda requested but CUDA is unavailable; falling back to CPU.")
    return torch.device("cpu")


def build_tokenizer(data_dir: Path | None = None, settings: Settings | None = None):
    settings = settings or Settings()
    if settings.backbone == "hf":
        from rlhf_dpo.model.hf_backbone import HFTokenizerAdapter

        return HFTokenizerAdapter(settings.hf_model_name, max_seq_len=settings.max_seq_len)
    tok = WordTokenizer()
    if data_dir is not None:
        vocab_path = Path(data_dir) / "tokenizer.json"
        if vocab_path.exists():
            tok.load_state_dict(json.loads(vocab_path.read_text(encoding="utf-8")))
            return tok
    return tok


def is_quantized_module(model: torch.nn.Module) -> bool:
    """True when the module (or nested HF backbone) uses bitsandbytes 4-bit weights."""
    if getattr(model, "_quantized", False):
        return True
    backbone = getattr(model, "backbone", None)
    if backbone is not None and getattr(backbone, "_quantized", False):
        return True
    return False


def place_model(model: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    """
    Move a model to device, skipping .to() for 4-bit modules (already device-mapped).

    Reward heads on quantized backbones are still moved explicitly.
    """
    if is_quantized_module(model):
        v_head = getattr(model, "v_head", None)
        if v_head is not None:
            model.v_head = v_head.to(device=device, dtype=torch.float32)
        backbone = getattr(model, "backbone", None)
        if backbone is not None:
            bv = getattr(backbone, "v_head", None)
            if bv is not None:
                backbone.v_head = bv.to(device=device, dtype=torch.float32)
        return model
    return model.to(device)


def build_lm(settings: Settings, tokenizer) -> torch.nn.Module:
    if settings.backbone == "hf":
        from rlhf_dpo.model.hf_backbone import HFCausalLM

        return HFCausalLM(
            settings.hf_model_name,
            use_lora=settings.use_lora,
            lora_r=settings.lora_r,
            lora_alpha=settings.lora_alpha,
            lora_dropout=settings.lora_dropout,
            max_seq_len=settings.max_seq_len,
            torch_dtype=getattr(settings, "torch_dtype", "float16"),
            load_in_4bit=bool(getattr(settings, "load_in_4bit", False)),
            gradient_checkpointing=bool(getattr(settings, "gradient_checkpointing", False)),
        )
    vocab = getattr(tokenizer, "vocab_size", settings.vocab_size)
    return CausalLM(
        vocab_size=max(settings.vocab_size, vocab),
        d_model=settings.d_model,
        n_heads=settings.n_heads,
        n_layers=settings.n_layers,
        max_seq_len=settings.max_seq_len,
        dropout=settings.dropout,
    )


def build_reward_model(settings: Settings, tokenizer) -> torch.nn.Module:
    backbone = build_lm(settings, tokenizer)
    if settings.backbone == "hf":
        from rlhf_dpo.model.hf_backbone import HFRewardModel

        return HFRewardModel(backbone)  # type: ignore[arg-type]
    return RewardModel(backbone)  # type: ignore[arg-type]


def _sep_id(tokenizer) -> int:
    return int(getattr(tokenizer, "sep_id", getattr(tokenizer, "eos_id")))


def encode_pair(
    tokenizer,
    prompt: str,
    response: str,
    max_len: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Encode as [BOS] prompt [SEP] response [EOS]; return ids, mask, prompt_len."""
    p = tokenizer.encode(prompt, add_special=False)
    r = tokenizer.encode(response, add_special=False)
    bos = int(tokenizer.bos_id) if tokenizer.bos_id is not None else int(tokenizer.eos_id)
    sep = _sep_id(tokenizer)
    eos = int(tokenizer.eos_id)
    pad = int(tokenizer.pad_id)
    ids = [bos] + p + [sep] + r + [eos]
    prompt_len = 1 + len(p) + 1
    if len(ids) > max_len:
        ids = ids[:max_len]
        prompt_len = min(prompt_len, max_len - 1)
    if len(ids) < max_len:
        ids = ids + [pad] * (max_len - len(ids))
    attn = [0 if i == pad else 1 for i in ids]
    return (
        torch.tensor(ids, dtype=torch.long),
        torch.tensor(attn, dtype=torch.long),
        prompt_len,
    )


def encode_prompt(tokenizer, prompt: str, max_len: int) -> torch.Tensor:
    """[BOS] prompt [SEP] for continuation generation."""
    p = tokenizer.encode(prompt, add_special=False)
    bos = int(tokenizer.bos_id) if tokenizer.bos_id is not None else int(tokenizer.eos_id)
    sep = _sep_id(tokenizer)
    ids = [bos] + p + [sep]
    if len(ids) > max_len:
        ids = ids[:max_len]
    return torch.tensor(ids, dtype=torch.long)


def completion_logprobs(
    model: torch.nn.Module,
    ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_len: int | torch.Tensor,
) -> torch.Tensor:
    # Pass attention mask when the wrapper supports it (HF path).
    try:
        logits, _ = model(ids[:, :-1], attention_mask=attention_mask[:, :-1])
    except TypeError:
        logits, _ = model(ids[:, :-1])
    # fp16/bf16 log_softmax over large vocabs (Mistral) easily overflows -> NaNs in DPO.
    logp = torch.nn.functional.log_softmax(logits.float(), dim=-1)
    target = ids[:, 1:]
    token_lp = logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)
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
    total = completion_logprobs(model, ids, attention_mask, prompt_len)
    b, t = ids.shape[0], ids.shape[1] - 1
    positions = torch.arange(1, t + 1, device=ids.device).unsqueeze(0).expand(b, -1)
    if isinstance(prompt_len, int):
        comp = positions >= prompt_len
    else:
        comp = positions >= prompt_len.unsqueeze(1)
    n = (comp & attention_mask[:, 1:].bool()).float().sum(dim=-1).clamp_min(1.0)
    return total / n


def decode_response(tokenizer, gen_ids: list[int], prompt: str = "") -> str:
    if hasattr(tokenizer, "tok"):
        text = tokenizer.decode(gen_ids)
        if prompt and text.startswith(prompt):
            text = text[len(prompt) :].strip()
        return text or "..."
    tokens = []
    for i in gen_ids:
        if i == tokenizer.eos_id:
            break
        if i in (tokenizer.pad_id, tokenizer.bos_id, tokenizer.unk_id):
            continue
        if 0 <= i < len(tokenizer.id_to_token):
            tokens.append(tokenizer.id_to_token[i])
    sep = getattr(tokenizer, "sep_token", None)
    if sep and sep in tokens:
        idx = tokens.index(sep)
        tokens = tokens[idx + 1 :]
    text = " ".join(tokens).strip()
    return text or "..."


def repetition_penalty(text: str) -> float:
    toks = text.split()
    if len(toks) < 2:
        return 0.0
    uniq = len(set(toks))
    ratio = uniq / len(toks)
    return max(0.0, 1.0 - ratio) * 5.0



def disable_gradient_checkpointing(model: torch.nn.Module) -> None:
    """Turn off gradient checkpointing for custom logprob training loops."""
    candidates = [model]
    inner = getattr(model, "model", None)
    if inner is not None:
        candidates.append(inner)
    for mod in candidates:
        if hasattr(mod, "gradient_checkpointing_disable"):
            try:
                mod.gradient_checkpointing_disable()
            except Exception:
                pass
        cfg = getattr(mod, "config", None)
        if cfg is not None and hasattr(cfg, "use_cache"):
            cfg.use_cache = False
        if hasattr(mod, "enable_input_require_grads"):
            try:
                mod.enable_input_require_grads()
            except Exception:
                pass


def assert_trainable_grads(params: list[torch.nn.Parameter], what: str) -> None:
    """Fail fast when a backward pass produced no LoRA/adapter gradients."""
    with_grad = [p for p in params if p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0]
    if not with_grad:
        raise RuntimeError(
            f"{what}: backward produced no trainable gradients. "
            "For QLoRA PPO/DPO set GRADIENT_CHECKPOINTING=false or update to a build that disables "
            "checkpointing during preference logprob training."
        )

def save_checkpoint(model: torch.nn.Module, path: Path) -> None:
    """Save full state_dict, or trainable-only weights for quantized/QLoRA models."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if is_quantized_module(model):
        trainable = {
            name: param.detach().cpu().clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        torch.save({"format": "trainable_only", "trainable": trainable}, path)
        return
    torch.save(model.state_dict(), path)


def load_checkpoint(model: torch.nn.Module, path: Path, device: torch.device) -> None:
    try:
        state = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(path, map_location=device)

    if isinstance(state, dict) and state.get("format") == "trainable_only":
        current = model.state_dict()
        for key, value in state.get("trainable", {}).items():
            if key in current:
                current[key] = value.to(device=current[key].device, dtype=current[key].dtype)
        missing, unexpected = model.load_state_dict(current, strict=False)
    else:
        missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        # LoRA / head mismatches are OK when warm-starting RM from SFT policy weights
        pass


def batch_iter(items: list, batch_size: int, shuffle: bool = True, seed: int = 0):
    idxs = list(range(len(items)))
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(idxs)
    for start in range(0, len(idxs), batch_size):
        chunk = idxs[start : start + batch_size]
        yield [items[i] for i in chunk]
