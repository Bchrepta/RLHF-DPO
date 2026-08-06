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
    tok.build_from_texts(["How do I use git?", "use git reset --soft HEAD~1"])
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
    settings = Settings(d_model=64, n_heads=4, n_layers=2, max_seq_len=32, vocab_size=128)
    tok = build_tokenizer()
    tok.build_from_texts(["hello world", "foo bar baz"] * 10)
    model = build_lm(settings, tok)
    ids = torch.randint(0, tok.vocab_size, (2, 16))
    logits, loss = model(ids, ids)
    assert logits.shape == (2, 16, max(settings.vocab_size, tok.vocab_size))
    assert loss is not None
    assert torch.isfinite(loss)


def test_dpo_loss_prefers_correct_direction():
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
    assert (tmp_path / "tokenizer.json").exists()
    tok = build_tokenizer(tmp_path)
    ids, mask, plen = encode_pair(tok, "How do I append items in python lists?", "Use list.append(x).", max_len=32)
    assert ids.shape[0] == 32
    assert mask.sum() > 0
    assert 1 < plen < 32
    assert ids[plen - 1].item() == tok.sep_id


def test_tiny_train_smoke():
    """Smoke test: tiny data, 1 epoch each stage, eval runs."""
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
            max_seq_len=32,
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
        assert report.headline["ppo_win_rate_vs_base"] is not None


def test_hf_backbone_optional():
    """HF path imports and runs a tiny forward when extras are installed."""
    pytest = __import__("pytest")
    try:
        import transformers  # noqa: F401
        import peft  # noqa: F401
    except ImportError:
        pytest.skip("transformers/peft not installed")

    from rlhf_dpo.config import Settings
    from rlhf_dpo.utils import build_lm, build_tokenizer, encode_pair, set_seed

    set_seed(0)
    settings = Settings(
        backbone="hf",
        hf_model_name="sshleifer/tiny-gpt2",
        use_lora=True,
        max_seq_len=32,
        lora_r=4,
    )
    tok = build_tokenizer(settings=settings)
    model = build_lm(settings, tok)
    ids, mask, plen = encode_pair(tok, "hello", "world", settings.max_seq_len)
    logits, loss = model(ids.unsqueeze(0), ids.unsqueeze(0))
    assert logits.shape[0] == 1
    assert loss is not None
