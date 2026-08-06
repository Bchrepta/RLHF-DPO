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


# Near-miss preference domain: chosen is correct+clear; rejected is fluent but wrong.
# Surface forms vary so SFT cannot saturate preference ranking by memorization alone.
_TOPICS = [
    ("python lists", "append items", "list.append(x)", "list.add(x)"),
    ("git", "undo last commit", "git reset --soft HEAD~1", "git delete commit"),
    ("sql", "select unique rows", "SELECT DISTINCT", "SELECT UNIQUE"),
    ("linux", "find a file", "find . -name 'pattern'", "locate-file pattern"),
    ("http", "interpret status 404", "resource not found", "server is upgrading"),
    ("docker", "list running containers", "docker ps", "docker list --running"),
    ("regex", "match digits", "\\d+", "[digit]+"),
    ("pandas", "read a csv", "pandas.read_csv(path)", "pandas.open_csv(path)"),
    ("bash", "make a script executable", "chmod +x script.sh", "run-allow script.sh"),
    ("json", "pretty-print an object", "json.dumps(obj, indent=2)", "json.pretty(obj)"),
    ("numpy", "create a zeros array", "numpy.zeros(shape)", "numpy.empty_zero(shape)"),
    ("pytest", "run one test", "pytest path::test_name", "pytest --only test_name"),
    ("ssh", "copy a file", "scp local remote:path", "ssh-copy local remote:path"),
    ("curl", "POST JSON", "curl -X POST -H 'Content-Type: application/json' -d '{}' url", "curl --json-post url"),
    ("make", "identify the default target", "the first target in the Makefile", "the last target always runs"),
    ("vim", "save and quit", ":wq", ":stop-save"),
    ("tmux", "detach a session", "Ctrl-b d", "Ctrl-x quit"),
    ("aws", "list s3 buckets", "aws s3 ls", "aws bucket list"),
    ("k8s", "get pods", "kubectl get pods", "kube list pods"),
    ("rust", "print to stdout", 'println!("{}", value)', "rust.print(value)"),
    ("python dicts", "get a key with default", "d.get(key, default)", "d.fetch(key, default)"),
    ("git", "create a new branch", "git checkout -b name", "git branch --new name"),
    ("sql", "count rows", "SELECT COUNT(*)", "SELECT TOTAL()"),
    ("linux", "change file owner", "chown user:group file", "ownerset user file"),
    ("docker", "build an image", "docker build -t name .", "docker image create name"),
    ("pandas", "drop null rows", "df.dropna()", "df.remove_nulls()"),
    ("bash", "redirect stderr", "cmd 2> err.txt", "cmd err> err.txt"),
    ("numpy", "matrix multiply", "a @ b", "numpy.times(a, b)"),
    ("pytest", "mark a slow test", "@pytest.mark.slow", "@test.slow"),
    ("k8s", "apply a manifest", "kubectl apply -f file.yaml", "kube deploy file.yaml"),
]

_CHOSEN_TEMPLATES = [
    "Use {answer}.",
    "Do this: {answer}.",
    "The standard approach is {answer}.",
    "Recommended: {answer}.",
    "For {topic}, {answer}.",
    "To {task}, {answer}.",
    "Sure — {answer}. That handles {task}.",
    "Steps: {answer}. This is correct for {topic}.",
]

_REJECTED_TEMPLATES = [
    "Use {wrong}.",
    "Do this: {wrong}.",
    "The standard approach is {wrong}.",
    "Recommended: {wrong}.",
    "For {topic}, {wrong}.",
    "To {task}, {wrong}.",
    "Sure — {wrong}. That handles {task}.",
    "Steps: {wrong}. This is correct for {topic}.",
]

_VAGUE = [
    "It depends on your setup for {topic}; try a few options for {task}.",
    "Someone online said something about {topic}. Maybe reboot.",
    "You can probably {task} if you experiment long enough.",
    "Not sure — ignore the docs for {topic} and improvise.",
]


def _make_pair(rng: random.Random, *, eval_mode: bool = False) -> PreferencePair:
    topic, task, good, bad = rng.choice(_TOPICS)
    # Eval uses a shifted template distribution so ranking requires generalization.
    templates_c = _CHOSEN_TEMPLATES[4:] if eval_mode else _CHOSEN_TEMPLATES
    templates_r = _REJECTED_TEMPLATES[4:] if eval_mode else _REJECTED_TEMPLATES
    chosen = rng.choice(templates_c).format(topic=topic, task=task, answer=good, wrong=bad)
    rejected = rng.choice(templates_r).format(topic=topic, task=task, answer=good, wrong=bad)
    if rng.random() < (0.25 if eval_mode else 0.15):
        rejected = rng.choice(_VAGUE).format(topic=topic, task=task)
    # Occasionally flip style so rejected is longer / more confident nonsense.
    if rng.random() < 0.1:
        rejected = f"Obviously everyone knows you should {bad} when you {task} in {topic}."
    prompt = f"How do I {task} in {topic}?"
    return PreferencePair(prompt=prompt, chosen=chosen, rejected=rejected, domain="helpfulness")


def generate_preference_dataset(
    n_train: int,
    n_eval: int,
    seed: int = 7,
) -> tuple[list[PreferencePair], list[PreferencePair], list[str]]:
    train_rng = random.Random(seed)
    eval_rng = random.Random(seed + 10_000)
    train = [_make_pair(train_rng, eval_mode=False) for _ in range(n_train)]
    eval_pairs = [_make_pair(eval_rng, eval_mode=True) for _ in range(n_eval)]
    prompts = sorted({p.prompt for p in train + eval_pairs})
    train_rng.shuffle(prompts)
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
