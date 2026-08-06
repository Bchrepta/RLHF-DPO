from __future__ import annotations

from collections import Counter
from dataclasses import dataclass


@dataclass
class WordTokenizer:
    """Whitespace tokenizer with a frozen vocab built from preference text."""

    pad_token: str = "<pad>"
    bos_token: str = "<bos>"
    eos_token: str = "<eos>"
    sep_token: str = "<sep>"
    unk_token: str = "<unk>"

    def __post_init__(self) -> None:
        # Placeholder until build_from_texts / load_vocab.
        specials = [
            self.pad_token,
            self.bos_token,
            self.eos_token,
            self.sep_token,
            self.unk_token,
        ]
        self.id_to_token = list(specials)
        self.token_to_id = {t: i for i, t in enumerate(self.id_to_token)}
        self.pad_id = self.token_to_id[self.pad_token]
        self.bos_id = self.token_to_id[self.bos_token]
        self.eos_id = self.token_to_id[self.eos_token]
        self.sep_id = self.token_to_id[self.sep_token]
        self.unk_id = self.token_to_id[self.unk_token]

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_token)

    @staticmethod
    def tokenize(text: str) -> list[str]:
        # Keep punctuation attached loosely by splitting on whitespace only.
        return [t for t in text.strip().split() if t]

    def build_from_texts(self, texts: list[str], min_count: int = 1, max_vocab: int = 2048) -> None:
        counts: Counter[str] = Counter()
        for text in texts:
            counts.update(self.tokenize(text))
        specials = [
            self.pad_token,
            self.bos_token,
            self.eos_token,
            self.sep_token,
            self.unk_token,
        ]
        words = [w for w, c in counts.most_common(max_vocab) if c >= min_count]
        self.id_to_token = specials + words
        self.token_to_id = {t: i for i, t in enumerate(self.id_to_token)}
        self.pad_id = self.token_to_id[self.pad_token]
        self.bos_id = self.token_to_id[self.bos_token]
        self.eos_id = self.token_to_id[self.eos_token]
        self.sep_id = self.token_to_id[self.sep_token]
        self.unk_id = self.token_to_id[self.unk_token]

    def encode(self, text: str, add_special: bool = True, max_len: int | None = None) -> list[int]:
        ids = [self.token_to_id.get(t, self.unk_id) for t in self.tokenize(text)]
        if add_special:
            ids = [self.bos_id] + ids + [self.eos_id]
        if max_len is not None:
            ids = ids[:max_len]
            if len(ids) < max_len:
                ids = ids + [self.pad_id] * (max_len - len(ids))
        return ids

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        special = {
            self.pad_id,
            self.bos_id,
            self.eos_id,
            self.sep_id,
            self.unk_id,
        }
        out: list[str] = []
        for i in ids:
            if skip_special and i in special:
                if i == self.eos_id:
                    break
                continue
            if not skip_special and 0 <= i < len(self.id_to_token):
                out.append(self.id_to_token[i])
            elif 0 <= i < len(self.id_to_token):
                out.append(self.id_to_token[i])
        if skip_special:
            return " ".join(out)
        # Preserve token boundaries for SEP splitting.
        return " ".join(out)

    def state_dict(self) -> dict:
        return {"id_to_token": self.id_to_token}

    def load_state_dict(self, state: dict) -> None:
        self.id_to_token = list(state["id_to_token"])
        self.token_to_id = {t: i for i, t in enumerate(self.id_to_token)}
        self.pad_id = self.token_to_id[self.pad_token]
        self.bos_id = self.token_to_id[self.bos_token]
        self.eos_id = self.token_to_id[self.eos_token]
        self.sep_id = self.token_to_id[self.sep_token]
        self.unk_id = self.token_to_id[self.unk_token]
