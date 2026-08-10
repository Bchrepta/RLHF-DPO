"""Hugging Face causal LM + optional LoRA / QLoRA adapters (e.g. Mistral-7B)."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def _require_transformers():
    try:
        import transformers  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Hugging Face backbone requires extras: pip install 'rlhf-dpo[hf]' "
            "(transformers, peft, accelerate)."
        ) from exc


def _require_bitsandbytes():
    try:
        import bitsandbytes  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "4-bit / QLoRA requires bitsandbytes: pip install 'rlhf-dpo[qlora]' "
            "or pip install bitsandbytes"
        ) from exc


def _resolve_dtype(torch_dtype: str) -> torch.dtype:
    if not torch.cuda.is_available() or torch_dtype == "float32":
        return torch.float32
    if torch_dtype == "bfloat16":
        return torch.bfloat16
    return torch.float16


class HFCausalLM(nn.Module):
    """
    Thin wrapper so HF models share the toy CausalLM training surface:
    forward(idx, targets=None) -> (logits, loss), generate(...), logprobs(...).
    """

    def __init__(
        self,
        model_name: str = "sshleifer/tiny-gpt2",
        *,
        use_lora: bool = True,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        max_seq_len: int = 128,
        torch_dtype: str = "float16",
        load_in_4bit: bool = False,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        _require_transformers()
        from transformers import AutoConfig, AutoModelForCausalLM

        self.model_name = model_name
        self.max_seq_len = max_seq_len
        self._quantized = bool(load_in_4bit)
        self._lora = False

        config = AutoConfig.from_pretrained(model_name)
        # Keep n_positions / max_position_embeddings aligned when possible
        if hasattr(config, "n_positions"):
            config.n_positions = max(getattr(config, "n_positions", max_seq_len), max_seq_len)

        load_kwargs: dict[str, Any] = {"config": config}
        if load_in_4bit:
            if not torch.cuda.is_available():
                raise RuntimeError("LOAD_IN_4BIT / QLoRA requires a CUDA GPU.")
            _require_bitsandbytes()
            from transformers import BitsAndBytesConfig

            # compute_dtype: fp16 on RTX 30-series; bf16 on Ampere+ data-center GPUs if requested
            compute = _resolve_dtype(torch_dtype)
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=compute,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            load_kwargs["device_map"] = {"": 0}
        else:
            load_kwargs["torch_dtype"] = _resolve_dtype(torch_dtype)

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        self.hidden_size = int(getattr(config, "n_embd", getattr(config, "hidden_size", 768)))
        self.vocab_size = int(config.vocab_size)

        if gradient_checkpointing and hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()
            if hasattr(self.model, "config"):
                self.model.config.use_cache = False

        if use_lora:
            try:
                from peft import LoraConfig, TaskType, get_peft_model

                if load_in_4bit:
                    from peft import prepare_model_for_kbit_training

                    self.model = prepare_model_for_kbit_training(
                        self.model,
                        use_gradient_checkpointing=gradient_checkpointing,
                    )

                # GPT-2 style + Llama/Mistral attention (+ optional MLP) targets
                targets = [
                    "c_attn",
                    "c_proj",
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ]
                lora = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    r=lora_r,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    target_modules=targets,
                    bias="none",
                )
                self.model = get_peft_model(self.model, lora)
                self._lora = True
            except Exception as exc:
                if load_in_4bit:
                    # Full finetune of a 4-bit base is not viable; surface the error.
                    raise RuntimeError(
                        f"Failed to attach LoRA for QLoRA ({type(exc).__name__}: {exc})"
                    ) from exc
                # Fall back to full fine-tune if peft/target modules mismatch
                self._lora = False

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        out = self.model(input_ids=idx, labels=targets)
        logits = out.logits
        loss = out.loss if targets is not None else None
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        eos_id: int | None = None,
    ) -> torch.Tensor:
        self.eval()
        # Prefer HF generate for quality; keep a simple loop fallback.
        try:
            # Avoid transformers warning: model generation_config often sets max_length=2048
            # while we only want max_new_tokens.
            gen_cfg = getattr(self.model, "generation_config", None)
            if gen_cfg is not None and getattr(gen_cfg, "max_length", None) is not None:
                gen_cfg.max_length = None
            gen = self.model.generate(
                input_ids=idx,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=max(temperature, 1e-5),
                eos_token_id=eos_id,
                pad_token_id=eos_id if eos_id is not None else 0,
            )
            return gen
        except Exception:
            for _ in range(max_new_tokens):
                idx_cond = idx[:, -self.max_seq_len :]
                logits, _ = self(idx_cond)
                logits = logits[:, -1, :] / max(temperature, 1e-5)
                probs = F.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
                idx = torch.cat([idx, next_id], dim=1)
                if eos_id is not None and bool((next_id == eos_id).all()):
                    break
            return idx

    def logprobs(self, idx: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        logits, _ = self(idx[:, :-1])
        logp = F.log_softmax(logits, dim=-1)
        target = idx[:, 1:]
        token_logp = logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)
        if attention_mask is None:
            return token_logp.sum(dim=-1)
        mask = attention_mask[:, 1:].float()
        return (token_logp * mask).sum(dim=-1)


class HFRewardModel(nn.Module):
    """Scalar reward head on pooled last hidden states of an HF backbone."""

    def __init__(self, backbone: HFCausalLM) -> None:
        super().__init__()
        self.backbone = backbone
        # Keep the value head in fp32 even when the backbone is 4-bit / fp16.
        self.v_head = nn.Linear(backbone.hidden_size, 1).float()

    def _base(self) -> Any:
        m = self.backbone.model
        # peft wraps .base_model.model or .model
        if hasattr(m, "base_model"):
            return m.base_model.model if hasattr(m.base_model, "model") else m.base_model
        return m

    def forward(self, idx: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        base = self._base()
        # Prefer transformer body without LM head
        if hasattr(base, "transformer"):
            out = base.transformer(input_ids=idx, attention_mask=attention_mask)
            hidden = out.last_hidden_state
        elif hasattr(base, "model"):
            out = base.model(input_ids=idx, attention_mask=attention_mask)
            hidden = out.last_hidden_state
        else:
            out = base(input_ids=idx, attention_mask=attention_mask, output_hidden_states=True)
            hidden = out.hidden_states[-1]
        if attention_mask is None:
            pooled = hidden.mean(dim=1)
        else:
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return self.v_head(pooled.float()).squeeze(-1)


class HFTokenizerAdapter:
    """Minimal adapter so encode_pair / demos work with AutoTokenizer."""

    def __init__(self, model_name: str = "sshleifer/tiny-gpt2", max_seq_len: int = 128) -> None:
        _require_transformers()
        from transformers import AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(model_name)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.max_seq_len = max_seq_len
        self.bos_id = self.tok.bos_token_id if self.tok.bos_token_id is not None else self.tok.eos_token_id
        self.eos_id = self.tok.eos_token_id
        self.pad_id = self.tok.pad_token_id
        self.unk_id = self.tok.unk_token_id if self.tok.unk_token_id is not None else self.pad_id
        # WordTokenizer compatibility
        self.sep_token = self.tok.sep_token or "[SEP]"
        self.sep_id = self.tok.sep_token_id if self.tok.sep_token_id is not None else self.eos_id
        self.vocab_size = len(self.tok)
        self.id_to_token = {i: t for t, i in self.tok.get_vocab().items()}

    def encode(self, text: str, add_special: bool = False) -> list[int]:
        return self.tok.encode(text, add_special_tokens=add_special)

    def decode(self, ids: list[int]) -> str:
        return self.tok.decode(ids, skip_special_tokens=True)

    def state_dict(self) -> dict:
        return {"model_name": getattr(self.tok, "name_or_path", "hf"), "type": "hf"}

    def load_state_dict(self, data: dict) -> None:
        return

    def build_from_texts(self, texts: list[str], min_count: int = 1, max_vocab: int = 4096) -> None:
        return
