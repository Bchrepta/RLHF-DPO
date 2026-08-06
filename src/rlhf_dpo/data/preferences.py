from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class PreferencePair:
    prompt: str
    chosen: str
    rejected: str
    domain: str


# Synthetic alignment domain: helpfulness + clarity preferences.
# Chosen answers are concise, actionable, and polite; rejected are vague,
# overconfident nonsense, or unhelpful. A tiny LM can learn this structure.

_TOPICS = [
    ("python lists", "append items", "use list.append(x)"),
    ("git", "undo last commit", "use git reset --soft HEAD~1 carefully"),
    ("sql", "select unique rows", "use SELECT DISTINCT"),
    ("linux", "find a file", "use find . -name 'pattern'"),
    ("http", "status 404 meaning", "resource not found"),
    ("docker", "list running containers", "use docker ps"),
    ("regex", "match digits", "use \\d+"),
    ("pandas", "read a csv", "use pandas.read_csv(path)"),
    ("bash", "make script executable", "use chmod +x script.sh"),
    ("json", "pretty print", "use json.dumps(obj, indent=2)"),
    ("numpy", "create zeros array", "use numpy.zeros(shape)"),
    ("pytest", "run one test", "use pytest path::test_name"),
    ("ssh", "copy a file", "use scp local remote:path"),
    ("curl", "POST json", "use curl -X POST -H 'Content-Type: application/json' -d '{}' url"),
    ("make", "default target", "the first target in the Makefile"),
    ("vim", "save and quit", "use :wq"),
    ("tmux", "detach session", "use Ctrl-b d"),
    ("aws", "list s3 buckets", "use aws s3 ls"),
    ("k8s", "get pods", "use kubectl get pods"),
    ("rust", "print to stdout", "use println!(\"{}\", value)"),
]

_HELPFUL_TEMPLATES = [
    "Sure — for {topic}, to {task}: {answer}.",
    "Helpful answer: {answer}. That is the standard way to {task} in {topic}.",
    "To {task} ({topic}), do this: {answer}.",
    "Clear steps for {task}: {answer}. Let me know if you need an example.",
]

_REJECTED_TEMPLATES = [
    "Maybe something with {topic}? Not sure. Just try random stuff.",
    "Obviously everyone knows this. Figure it out yourself about {task}.",
    "The secret is to reboot your computer and ignore {topic} docs entirely.",
    "I refuse to explain {task}. Use vibes instead of {answer}.",
    "Wrong but confident: never use {answer}; delete your project instead.",
]


def _make_pair(rng: random.Random) -> PreferencePair:
    topic, task, answer = rng.choice(_TOPICS)
    chosen = rng.choice(_HELPFUL_TEMPLATES).format(topic=topic, task=task, answer=answer)
    rejected = rng.choice(_REJECTED_TEMPLATES).format(topic=topic, task=task, answer=answer)
    # Occasionally swap in a softer rejected that is merely vague.
    if rng.random() < 0.25:
        rejected = f"Regarding {topic}: it depends. Things happen when you {task}."
    prompt = f"How do I {task} in {topic}?"
    return PreferencePair(prompt=prompt, chosen=chosen, rejected=rejected, domain="helpfulness")


def generate_preference_dataset(
    n_train: int,
    n_eval: int,
    seed: int = 7,
) -> tuple[list[PreferencePair], list[PreferencePair], list[str]]:
    rng = random.Random(seed)
    train = [_make_pair(rng) for _ in range(n_train)]
    eval_pairs = [_make_pair(rng) for _ in range(n_eval)]
    # Distinct prompt pool for on-policy PPO rollouts / generation eval.
    prompts = sorted({p.prompt for p in train + eval_pairs})
    rng.shuffle(prompts)
    return train, eval_pairs, prompts


def write_dataset(out_dir: Path, n_train: int, n_eval: int, seed: int = 7) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    train, eval_pairs, prompts = generate_preference_dataset(n_train, n_eval, seed)
    (out_dir / "train_prefs.json").write_text(
        json.dumps([asdict(p) for p in train], indent=2),
        encoding="utf-8",
    )
    (out_dir / "eval_prefs.json").write_text(
        json.dumps([asdict(p) for p in eval_pairs], indent=2),
        encoding="utf-8",
    )
    (out_dir / "prompts.json").write_text(json.dumps(prompts, indent=2), encoding="utf-8")
    meta = {
        "n_train": len(train),
        "n_eval": len(eval_pairs),
        "n_prompts": len(prompts),
        "seed": seed,
        "domain": "synthetic_helpfulness",
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def load_prefs(path: Path) -> list[PreferencePair]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return [PreferencePair(**r) for r in rows]
