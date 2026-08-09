"""Character tokenization and next-token language-model datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
from torch.utils.data import DataLoader, Dataset

from .config import DataConfig

DEMO_TEXT = ("""
The small machine studied the stars and wrote down what it saw.
It learned that every careful question creates another useful question.
Science begins with observation, continues with a model, and improves by test.
When a prediction fails, we revise the idea instead of hiding the evidence.
Data can guide discovery, but judgment gives the numbers meaning.
The laboratory was quiet; the experiment was ready; the next result was unknown.
""".strip() + "\n") * 40


class CharacterTokenizer:
    """Deterministic character tokenizer with an unknown-character token."""

    UNK = "<unk>"

    def __init__(self, vocabulary: Sequence[str]) -> None:
        unique = list(dict.fromkeys(vocabulary))
        if self.UNK not in unique:
            unique.insert(0, self.UNK)
        self.itos = unique
        self.stoi = {token: index for index, token in enumerate(self.itos)}

    @classmethod
    def from_text(cls, text: str) -> "CharacterTokenizer":
        if not text:
            raise ValueError("cannot build a tokenizer from empty text")
        return cls([cls.UNK, *sorted(set(text))])

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def encode(self, text: str) -> list[int]:
        unknown = self.stoi[self.UNK]
        return [self.stoi.get(character, unknown) for character in text]

    def decode(self, token_ids: Sequence[int]) -> str:
        pieces = []
        for token_id in token_ids:
            token = self.itos[int(token_id)]
            pieces.append("�" if token == self.UNK else token)
        return "".join(pieces)

    def state_dict(self) -> dict[str, list[str]]:
        return {"itos": self.itos}

    @classmethod
    def from_state_dict(cls, state: dict[str, list[str]]) -> "CharacterTokenizer":
        return cls(state["itos"])


class TokenDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, tokens: Sequence[int], block_size: int) -> None:
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        self.block_size = block_size
        if len(self.tokens) <= block_size:
            raise ValueError(
                f"split has {len(self.tokens)} tokens but needs more than {block_size}"
            )

    def __len__(self) -> int:
        return len(self.tokens) - self.block_size

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        chunk = self.tokens[index : index + self.block_size + 1]
        return chunk[:-1], chunk[1:]


def load_text(path: str | Path | None) -> str:
    if path is None:
        return DEMO_TEXT
    corpus_path = Path(path)
    if not corpus_path.is_file():
        raise FileNotFoundError(
            f"text corpus not found: {corpus_path}. Run `python scripts/prepare_data.py` "
            "to prepare the default TinyStories corpus, or pass `--data PATH`."
        )
    text = corpus_path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"text corpus is empty: {path}")
    return text


def build_dataloaders(
    text: str,
    tokenizer: CharacterTokenizer,
    block_size: int,
    config: DataConfig,
    seed: int,
) -> tuple[DataLoader, DataLoader]:
    token_ids = tokenizer.encode(text)
    split_index = int(len(token_ids) * config.train_fraction)
    train_dataset = TokenDataset(token_ids[:split_index], block_size)
    val_dataset = TokenDataset(token_ids[split_index:], block_size)
    generator = torch.Generator().manual_seed(seed)
    common = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(
        train_dataset, shuffle=True, generator=generator, drop_last=True, **common
    )
    val_loader = DataLoader(val_dataset, shuffle=False, drop_last=False, **common)
    if len(train_loader) == 0:
        raise ValueError("training split is too small for the selected batch_size")
    return train_loader, val_loader
