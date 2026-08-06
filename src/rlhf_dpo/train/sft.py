from __future__ import annotations

from pathlib import Path

import torch
from tqdm import tqdm

from rlhf_dpo.config import Settings
from rlhf_dpo.data.preferences import PreferencePair, load_prefs
from rlhf_dpo.utils import (
    batch_iter,
    build_lm,
    build_tokenizer,
    encode_pair,
    get_device,
    save_checkpoint,
    set_seed,
)


def train_sft(settings: Settings, data_dir: Path | None = None, out: Path | None = None) -> Path:
    """Supervised fine-tune on chosen (preferred) responses."""
    set_seed(settings.seed)
    device = get_device(settings)
    data_dir = data_dir or settings.data_dir
    out = out or (settings.ckpt_dir / "sft.pt")

    tokenizer = build_tokenizer()
    model = build_lm(settings, tokenizer).to(device)
    prefs = load_prefs(data_dir / "train_prefs.json")
    opt = torch.optim.AdamW(model.parameters(), lr=settings.lr)

    model.train()
    for epoch in range(settings.sft_epochs):
        total = 0.0
        n = 0
        for batch in tqdm(
            list(batch_iter(prefs, settings.batch_size, shuffle=True, seed=settings.seed + epoch)),
            desc=f"sft:{epoch+1}/{settings.sft_epochs}",
            leave=False,
        ):
            ids_list = []
            for p in batch:
                ids, _ = encode_pair(tokenizer, p.prompt, p.chosen, settings.max_seq_len)
                ids_list.append(ids)
            ids = torch.stack(ids_list).to(device)
            # Standard LM: predict next token; ignore pads.
            targets = ids.clone()
            targets[targets == tokenizer.pad_id] = -100
            # Also ignore the pure-prompt prefix roughly by not masking — tiny data ok.
            _, loss = model(ids, targets)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.item())
            n += 1
        tqdm.write(f"SFT epoch {epoch+1}: loss={total / max(n, 1):.4f}")

    save_checkpoint(model, out)
    return out


def sft_demo_loss(prefs: list[PreferencePair], settings: Settings) -> float:
    """Tiny helper for tests."""
    device = get_device(settings)
    tokenizer = build_tokenizer()
    model = build_lm(settings, tokenizer).to(device)
    ids, _ = encode_pair(tokenizer, prefs[0].prompt, prefs[0].chosen, settings.max_seq_len)
    ids = ids.unsqueeze(0).to(device)
    targets = ids.clone()
    targets[targets == tokenizer.pad_id] = -100
    _, loss = model(ids, targets)
    return float(loss.item())
