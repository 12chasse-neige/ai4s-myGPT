"""Byte-level BPE tokenization for pretraining and instruction tuning."""

from __future__ import annotations

import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Iterable, Sequence

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

UNK_TOKEN = "<unk>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
PAD_TOKEN = "<pad>"
SPECIAL_TOKENS = (UNK_TOKEN, BOS_TOKEN, EOS_TOKEN, PAD_TOKEN)


class BPETokenizer:
    """Byte-level BPE tokenizer with fixed special-token IDs."""

    kind = "bpe"

    def __init__(self, backend: Tokenizer) -> None:
        self.backend = backend
        missing = [
            token for token in SPECIAL_TOKENS if backend.token_to_id(token) is None
        ]
        if missing:
            raise ValueError(f"BPE tokenizer is missing special tokens: {missing}")

    @classmethod
    def train(
        cls,
        files: Sequence[str | Path],
        *,
        vocab_size: int = 10_000,
        min_frequency: int = 2,
    ) -> "BPETokenizer":
        if vocab_size < len(SPECIAL_TOKENS) + 256:
            raise ValueError("BPE vocab_size must leave room for the byte alphabet")
        backend = Tokenizer(BPE(unk_token=UNK_TOKEN))
        backend.pre_tokenizer = ByteLevel(add_prefix_space=False)
        backend.decoder = ByteLevelDecoder()
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=list(SPECIAL_TOKENS),
            initial_alphabet=ByteLevel.alphabet(),
            show_progress=False,
        )
        backend.train([str(Path(path)) for path in files], trainer)
        return cls(backend)

    @classmethod
    def train_from_iterator(
        cls,
        texts: Iterable[str],
        *,
        vocab_size: int = 512,
        min_frequency: int = 1,
    ) -> "BPETokenizer":
        backend = Tokenizer(BPE(unk_token=UNK_TOKEN))
        backend.pre_tokenizer = ByteLevel(add_prefix_space=False)
        backend.decoder = ByteLevelDecoder()
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=list(SPECIAL_TOKENS),
            initial_alphabet=ByteLevel.alphabet(),
            show_progress=False,
        )
        backend.train_from_iterator(texts, trainer=trainer)
        return cls(backend)

    @classmethod
    def from_file(cls, path: str | Path) -> "BPETokenizer":
        return cls(Tokenizer.from_file(str(path)))

    @classmethod
    def from_state_dict(cls, state: dict[str, object]) -> "BPETokenizer":
        if state.get("type") != cls.kind:
            raise ValueError("checkpoint does not contain a BPE tokenizer")
        serialized = state.get("json")
        if not isinstance(serialized, str):
            raise ValueError("BPE tokenizer state has no serialized JSON")
        return cls(Tokenizer.from_str(serialized))

    @property
    def vocab_size(self) -> int:
        return self.backend.get_vocab_size()

    def _special_id(self, token: str) -> int:
        token_id = self.backend.token_to_id(token)
        if token_id is None:
            raise ValueError(f"BPE tokenizer does not define {token}")
        return token_id

    @property
    def eos_id(self) -> int:
        return self._special_id(EOS_TOKEN)

    @property
    def pad_id(self) -> int:
        return self._special_id(PAD_TOKEN)

    def encode(self, text: str) -> list[int]:
        return self.backend.encode(text, add_special_tokens=False).ids

    def decode(
        self, token_ids: Sequence[int], *, skip_special_tokens: bool = False
    ) -> str:
        return self.backend.decode(
            [int(token_id) for token_id in token_ids],
            skip_special_tokens=skip_special_tokens,
        )

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with NamedTemporaryFile(
                "w",
                dir=destination.parent,
                prefix=f".{destination.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary_path = Path(handle.name)
            self.backend.save(str(temporary_path))
            os.replace(temporary_path, destination)
        except BaseException:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            raise

    def state_dict(self) -> dict[str, object]:
        return {"type": self.kind, "json": self.backend.to_str()}
