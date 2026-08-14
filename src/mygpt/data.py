"""Tokenized next-token language-model datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
from torch.utils.data import DataLoader, Dataset

from .config import DataConfig
from .tokenizer import BPETokenizer

DEMO_TEXT = ("""
The small machine studied the stars and wrote down what it saw.
It learned that every careful question creates another useful question.
Science begins with observation, continues with a model, and improves by test.
When a prediction fails, we revise the idea instead of hiding the evidence.
Data can guide discovery, but judgment gives the numbers meaning.
The laboratory was quiet; the experiment was ready; the next result was unknown.
""".strip() + "\n") * 40


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
    tokenizer: BPETokenizer,
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
