from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CharTokenizer:
    """Fixed printable-ASCII tokenizer with special tokens."""

    pad_token: str = "<pad>"
    bos_token: str = "<bos>"
    eos_token: str = "<eos>"
    sep_token: str = "<sep>"
    unk_token: str = "<unk>"

    def __post_init__(self) -> None:
        specials = [
            self.pad_token,
            self.bos_token,
            self.eos_token,
            self.sep_token,
            self.unk_token,
        ]
        # Keep enough room under vocab_size=512 for ASCII printable + specials.
        chars = [chr(i) for i in range(32, 127)]
        self.id_to_token = specials + chars
        self.token_to_id = {t: i for i, t in enumerate(self.id_to_token)}
        self.pad_id = self.token_to_id[self.pad_token]
        self.bos_id = self.token_to_id[self.bos_token]
        self.eos_id = self.token_to_id[self.eos_token]
        self.sep_id = self.token_to_id[self.sep_token]
        self.unk_id = self.token_to_id[self.unk_token]

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_token)

    def encode(self, text: str, add_special: bool = True, max_len: int | None = None) -> list[int]:
        ids = [self.token_to_id.get(ch, self.unk_id) for ch in text]
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
            if 0 <= i < len(self.id_to_token):
                out.append(self.id_to_token[i])
        return "".join(out)
