from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from rlhf_dpo.config import Settings
from rlhf_dpo.data.preferences import load_prefs
from rlhf_dpo.utils import (
    batch_iter,
    build_reward_model,
    build_tokenizer,
    encode_pair,
    get_device,
    load_checkpoint,
    save_checkpoint,
    set_seed,
)


def train_reward_model(
    settings: Settings,
    data_dir: Path | None = None,
    sft_ckpt: Path | None = None,
    out: Path | None = None,
) -> Path:
    """Train a Bradley-Terry reward model on preference pairs."""
    set_seed(settings.seed)
    device = get_device(settings)
    data_dir = data_dir or settings.data_dir
    out = out or (settings.ckpt_dir / "reward.pt")
    sft_ckpt = sft_ckpt or (settings.ckpt_dir / "sft.pt")

    tokenizer = build_tokenizer(data_dir)
    rm = build_reward_model(settings, tokenizer).to(device)
    if sft_ckpt.exists():
        load_checkpoint(rm.backbone, sft_ckpt, device)

    prefs = load_prefs(data_dir / "train_prefs.json")
    opt = torch.optim.AdamW(rm.parameters(), lr=settings.lr)

    rm.train()
    for epoch in range(settings.rm_epochs):
        total = 0.0
        correct = 0
        n = 0
        for batch in tqdm(
            list(batch_iter(prefs, settings.batch_size, shuffle=True, seed=settings.seed + epoch)),
            desc=f"rm {epoch+1}/{settings.rm_epochs}",
            leave=False,
        ):
            chosen_ids, chosen_mask, rejected_ids, rejected_mask = [], [], [], []
            for p in batch:
                c_ids, c_mask, _ = encode_pair(tokenizer, p.prompt, p.chosen, settings.max_seq_len)
                r_ids, r_mask, _ = encode_pair(tokenizer, p.prompt, p.rejected, settings.max_seq_len)
                chosen_ids.append(c_ids)
                chosen_mask.append(c_mask)
                rejected_ids.append(r_ids)
                rejected_mask.append(r_mask)
            c = torch.stack(chosen_ids).to(device)
            cm = torch.stack(chosen_mask).to(device)
            r = torch.stack(rejected_ids).to(device)
            rm_m = torch.stack(rejected_mask).to(device)
            r_chosen = rm(c, cm)
            r_rejected = rm(r, rm_m)
            # Bradley-Terry: -log σ(r_c − r_r)
            loss = -F.logsigmoid(r_chosen - r_rejected).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(rm.parameters(), 1.0)
            opt.step()
            total += float(loss.item())
            correct += int((r_chosen > r_rejected).sum().item())
            n += len(batch)
        tqdm.write(
            f"RM epoch {epoch+1}: loss={total / max(len(prefs) // settings.batch_size, 1):.4f} "
            f"pair_acc={correct / max(n, 1):.3f}"
        )

    save_checkpoint(rm, out)
    return out
