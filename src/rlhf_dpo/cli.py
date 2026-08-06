from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from rlhf_dpo.config import get_settings
from rlhf_dpo.data.preferences import write_dataset
from rlhf_dpo.eval.harness import run_eval, save_results
from rlhf_dpo.train.dpo import train_dpo
from rlhf_dpo.train.ppo import train_ppo
from rlhf_dpo.train.reward import train_reward_model
from rlhf_dpo.train.sft import train_sft
from rlhf_dpo.utils import (
    build_lm,
    build_tokenizer,
    encode_prompt,
    get_device,
    load_checkpoint,
)

app = typer.Typer(add_completion=False, no_args_is_help=True)
console = Console()


@app.command("generate-data")
def generate_data(
    n_train: int = typer.Option(2400, help="Training preference pairs"),
    n_eval: int = typer.Option(400, help="Eval preference pairs"),
    seed: int = typer.Option(7, help="RNG seed"),
    out: Optional[Path] = typer.Option(None, help="Output directory"),
) -> None:
    """Generate synthetic helpfulness preference pairs."""
    settings = get_settings()
    meta = write_dataset(out or settings.data_dir, n_train=n_train, n_eval=n_eval, seed=seed)
    console.print("[green]Wrote dataset[/green]", meta)


@app.command("train-sft")
def train_sft_cmd() -> None:
    settings = get_settings()
    if not (settings.data_dir / "train_prefs.json").exists():
        write_dataset(settings.data_dir, settings.n_train_prefs, settings.n_eval_prefs, settings.seed)
    path = train_sft(settings)
    console.print(f"[green]Saved SFT checkpoint[/green] {path}")


@app.command("train-reward")
def train_reward_cmd() -> None:
    settings = get_settings()
    path = train_reward_model(settings)
    console.print(f"[green]Saved reward model[/green] {path}")


@app.command("train-dpo")
def train_dpo_cmd() -> None:
    settings = get_settings()
    path = train_dpo(settings)
    console.print(f"[green]Saved DPO policy[/green] {path}")


@app.command("train-ppo")
def train_ppo_cmd() -> None:
    settings = get_settings()
    path = train_ppo(settings)
    console.print(f"[green]Saved PPO policy[/green] {path}")


@app.command("train-all")
def train_all() -> None:
    """Run full pipeline: data → SFT → reward model → DPO → PPO."""
    settings = get_settings()
    if not (settings.data_dir / "train_prefs.json").exists():
        write_dataset(settings.data_dir, settings.n_train_prefs, settings.n_eval_prefs, settings.seed)
        console.print("[cyan]Generated preference dataset[/cyan]")
    console.print("[bold]1/4 SFT[/bold]")
    train_sft(settings)
    console.print("[bold]2/4 Reward model[/bold]")
    train_reward_model(settings)
    console.print("[bold]3/4 DPO[/bold]")
    train_dpo(settings)
    console.print("[bold]4/4 PPO-RLHF[/bold]")
    train_ppo(settings)
    console.print("[green]Training complete.[/green]")


@app.command("eval")
def eval_cmd(
    gen_limit: int = typer.Option(80, help="Prompts for generation win-rate"),
) -> None:
    """Evaluate SFT vs DPO vs PPO on held-out preferences + RM win-rate."""
    settings = get_settings()
    required = ["sft.pt", "reward.pt", "dpo.pt", "ppo.pt"]
    missing = [n for n in required if not (settings.ckpt_dir / n).exists()]
    if missing:
        console.print(f"[yellow]Missing checkpoints {missing}; running train-all first...[/yellow]")
        train_all()

    report = run_eval(settings, gen_limit=gen_limit)
    path = save_results(report, settings.results_dir)

    table = Table(title="RLHF / DPO Evaluation")
    table.add_column("Method")
    table.add_column("Pref Acc", justify="right")
    table.add_column("Reward Margin", justify="right")
    table.add_column("Win vs SFT", justify="right")
    table.add_column("KL→SFT", justify="right")

    for m in (report.sft, report.dpo, report.ppo):
        table.add_row(
            m.name.upper(),
            f"{m.preference_accuracy:.3f}",
            f"{m.mean_reward_margin:.3f}",
            "-" if m.win_rate_vs_sft is None else f"{m.win_rate_vs_sft:.3f}",
            "-" if m.mean_kl_to_sft is None else f"{m.mean_kl_to_sft:.3f}",
        )
    console.print(table)
    console.print(
        f"Reward-model pair accuracy (eval)={report.reward_model_pair_acc:.3f} | "
        f"DPO win vs SFT={report.headline['dpo_win_rate_vs_sft']:.3f} | "
        f"PPO win vs SFT={report.headline['ppo_win_rate_vs_sft']:.3f}"
    )
    console.print(
        f"DPO relative preference lift vs SFT="
        f"{report.headline['dpo_relative_pref_lift_vs_sft']*100:.1f}%"
    )
    console.print(f"Wrote {path}")


@app.command("demo")
def demo(
    prompt: str = typer.Option(
        "How do I append items in python lists?",
        help="Prompt to complete with each policy",
    ),
    method: str = typer.Option("dpo", help="sft|dpo|ppo"),
) -> None:
    """Generate a qualitative completion from a trained policy."""
    settings = get_settings()
    device = get_device(settings)
    tokenizer = build_tokenizer()
    model = build_lm(settings, tokenizer).to(device)
    ckpt = settings.ckpt_dir / f"{method}.pt"
    if not ckpt.exists():
        console.print(f"[red]Missing {ckpt}; run train-all or train-{method}[/red]")
        raise typer.Exit(1)
    load_checkpoint(model, ckpt, device)
    model.eval()
    ids = encode_prompt(tokenizer, prompt, settings.max_seq_len // 2).unsqueeze(0).to(device)
    with __import__("torch").no_grad():
        out = model.generate(ids, max_new_tokens=40, temperature=0.7, eos_id=tokenizer.eos_id)
    text = tokenizer.decode(out[0].tolist(), skip_special=True)
    console.rule(method.upper())
    console.print(text)


@app.command("compare")
def compare(
    prompt: str = typer.Option("How do I read a csv in pandas?", help="Shared prompt"),
) -> None:
    """Side-by-side SFT / DPO / PPO generations."""
    for method in ("sft", "dpo", "ppo"):
        try:
            demo(prompt=prompt, method=method)
        except typer.Exit:
            console.print(f"[dim]skip {method}[/dim]")


if __name__ == "__main__":
    app()
