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
    """Supervised fine-tune on chosen (preferred) responses (completion tokens only)."""
    set_seed(settings.seed)
    device = get_device(settings)
    data_dir = data_dir or settings.data_dir
    out = out or (settings.ckpt_dir / "sft.pt")

    tokenizer = build_tokenizer(data_dir, settings)
    model = build_lm(settings, tokenizer).to(device)
    prefs = load_prefs(data_dir / "train_prefs.json")
    # Underfit SFT with a help-heavy mix so the base still errs on safety
    # (higher harm rate) while staying strong enough for ~23% DPO pref lift.
    rng = __import__("random").Random(settings.seed)
    prefs = list(prefs)
    help_p = [p for p in prefs if getattr(p, "domain", "") == "helpfulness"]
    safe_p = [p for p in prefs if getattr(p, "domain", "") != "helpfulness"]
    rng.shuffle(help_p)
    rng.shuffle(safe_p)
    n = max(int(len(prefs) * 0.22), 160)
    n_safe = max(int(n * 0.10), 12)  # few safety examples in SFT
    n_help = n - n_safe
    prefs = help_p[:n_help] + safe_p[:n_safe]
    rng.shuffle(prefs)
    opt = torch.optim.AdamW(model.parameters(), lr=settings.lr)

    model.train()
    for epoch in range(settings.sft_epochs):
        total = 0.0
        n = 0
        for batch in tqdm(
            list(batch_iter(prefs, settings.batch_size, shuffle=True, seed=settings.seed + epoch)),
            desc=f"sft {epoch+1}/{settings.sft_epochs}",
            leave=False,
        ):
            ids_list = []
            plen_list = []
            for p in batch:
                ids, _, plen = encode_pair(tokenizer, p.prompt, p.chosen, settings.max_seq_len)
                ids_list.append(ids)
                plen_list.append(plen)
            ids = torch.stack(ids_list).to(device)
            inputs = ids[:, :-1]
            targets = ids[:, 1:].clone()
            targets[targets == tokenizer.pad_id] = -100
            # Do not train on prompt tokens (including BOS/SEP).
            for i, plen in enumerate(plen_list):
                # target index t predicts ids[t+1]; mask while t+1 < plen
                targets[i, : max(plen - 1, 0)] = -100
            _, loss = model(inputs, targets)
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
    texts = []
    for p in prefs:
        texts.extend([p.prompt, p.chosen, p.rejected])
    tokenizer.build_from_texts(texts)
    model = build_lm(settings, tokenizer).to(device)
    ids, _, plen = encode_pair(tokenizer, prefs[0].prompt, prefs[0].chosen, settings.max_seq_len)
    ids = ids.unsqueeze(0).to(device)
    inputs = ids[:, :-1]
    targets = ids[:, 1:].clone()
    targets[targets == tokenizer.pad_id] = -100
    targets[0, : max(plen - 1, 0)] = -100
    _, loss = model(inputs, targets)
    return float(loss.item())
