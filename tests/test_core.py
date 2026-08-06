from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from rlhf_dpo.config import Settings
from rlhf_dpo.data.preferences import generate_preference_dataset, write_dataset
from rlhf_dpo.train.dpo import dpo_loss
from rlhf_dpo.utils import build_lm, build_tokenizer, encode_pair, set_seed


def test_tokenizer_roundtrip():
    tok = build_tokenizer()
    text = "How do I use git?"
    ids = tok.encode(text, add_special=True)
    assert tok.bos_id in ids and tok.eos_id in ids
    decoded = tok.decode(ids)
    assert "git" in decoded


def test_preference_dataset_shapes():
    train, eval_pairs, prompts = generate_preference_dataset(32, 16, seed=1)
    assert len(train) == 32
    assert len(eval_pairs) == 16
    assert all(p.chosen != p.rejected for p in train)
    assert len(prompts) > 0


def test_causal_lm_forward():
    set_seed(0)
    settings = Settings(d_model=64, n_heads=4, n_layers=2, max_seq_len=32)
    tok = build_tokenizer()
    model = build_lm(settings, tok)
    ids = torch.randint(0, tok.vocab_size, (2, 16))
    logits, loss = model(ids, ids)
    assert logits.shape == (2, 16, max(settings.vocab_size, tok.vocab_size))
    assert loss is not None
    assert torch.isfinite(loss)


def test_dpo_loss_prefers_correct_direction():
    # Higher chosen logps relative to ref should yield lower loss than the reverse.
    beta = 0.1
    good = dpo_loss(
        policy_chosen_logps=torch.tensor([2.0]),
        policy_rejected_logps=torch.tensor([0.0]),
        ref_chosen_logps=torch.tensor([1.0]),
        ref_rejected_logps=torch.tensor([1.0]),
        beta=beta,
    )
    bad = dpo_loss(
        policy_chosen_logps=torch.tensor([0.0]),
        policy_rejected_logps=torch.tensor([2.0]),
        ref_chosen_logps=torch.tensor([1.0]),
        ref_rejected_logps=torch.tensor([1.0]),
        beta=beta,
    )
    assert good < bad


def test_write_dataset_and_encode(tmp_path: Path):
    meta = write_dataset(tmp_path, n_train=20, n_eval=10, seed=3)
    assert meta["n_train"] == 20
    assert (tmp_path / "train_prefs.json").exists()
    tok = build_tokenizer()
    ids, mask, plen = encode_pair(tok, "prompt?", "chosen answer", max_len=32)
    assert ids.shape[0] == 32
    assert mask.sum() > 0
    assert 1 < plen < 32
    assert ids[plen - 1].item() == tok.sep_id


def test_tiny_train_smoke():
    """End-to-end smoke: tiny data, 1 epoch each stage, eval runs."""
    from rlhf_dpo.train.dpo import train_dpo
    from rlhf_dpo.train.ppo import train_ppo
    from rlhf_dpo.train.reward import train_reward_model
    from rlhf_dpo.train.sft import train_sft
    from rlhf_dpo.eval.harness import run_eval

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        settings = Settings(
            seed=0,
            d_model=64,
            n_heads=4,
            n_layers=2,
            max_seq_len=48,
            n_train_prefs=64,
            n_eval_prefs=24,
            sft_epochs=1,
            rm_epochs=1,
            dpo_epochs=1,
            ppo_steps=4,
            ppo_batch_size=2,
            batch_size=16,
            data_dir=root / "data",
            ckpt_dir=root / "ckpt",
            results_dir=root / "results",
        )
        write_dataset(settings.data_dir, settings.n_train_prefs, settings.n_eval_prefs, settings.seed)
        train_sft(settings)
        train_reward_model(settings)
        train_dpo(settings)
        train_ppo(settings)
        report = run_eval(settings, gen_limit=8)
        assert report.n_eval == 24
        assert 0.0 <= report.dpo.preference_accuracy <= 1.0
        assert report.headline["dpo_win_rate_vs_sft"] is not None
