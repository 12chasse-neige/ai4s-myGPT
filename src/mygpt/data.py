"""Tokenized next-token language-model datasets."""

from __future__ import annotations

from array import array
import codecs
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Sequence

import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler

from .config import DataConfig
from .tokenizer import BPETokenizer, EOS_TOKEN


TOKEN_CACHE_VERSION = 1
TOKENIZE_CHUNK_BYTES = 4 * 1024 * 1024
TOKENIZE_PROGRESS_BYTES = 128 * 1024 * 1024

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


class MemoryMappedTokenDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """A view over one split of a disk-backed token tensor."""

    def __init__(
        self,
        tokens: torch.Tensor,
        start: int,
        end: int,
        block_size: int,
    ) -> None:
        self.tokens = tokens
        self.start = start
        self.end = end
        self.block_size = block_size
        if end - start <= block_size:
            raise ValueError(
                f"split has {end - start} tokens but needs more than {block_size}"
            )

    def __len__(self) -> int:
        return self.end - self.start - self.block_size

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        offset = self.start + index
        chunk = self.tokens[offset : offset + self.block_size + 1].long()
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


def validate_text_file(path: str | Path) -> Path:
    corpus_path = Path(path)
    if not corpus_path.is_file():
        raise FileNotFoundError(
            f"text corpus not found: {corpus_path}. Run `python scripts/prepare_data.py` "
            "to prepare the default TinyStories corpus, or pass `--data PATH`."
        )
    if corpus_path.stat().st_size == 0:
        raise ValueError(f"text corpus is empty: {path}")
    return corpus_path


def text_file_contains(path: str | Path, needle: str) -> bool:
    """Search a UTF-8 text file without loading it all into memory."""
    overlap = max(0, len(needle.encode("utf-8")) - 1)
    previous = b""
    encoded_needle = needle.encode("utf-8")
    with validate_text_file(path).open("rb") as handle:
        while chunk := handle.read(TOKENIZE_CHUNK_BYTES):
            combined = previous + chunk
            if encoded_needle in combined:
                return True
            previous = combined[-overlap:] if overlap else b""
    return False


def _tokenizer_fingerprint(tokenizer: BPETokenizer) -> str:
    serialized = tokenizer.state_dict()["json"]
    assert isinstance(serialized, str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _cache_metadata(
    corpus_path: Path,
    tokenizer: BPETokenizer,
    *,
    dtype_name: str,
    token_count: int,
) -> dict[str, object]:
    source = corpus_path.stat()
    return {
        "version": TOKEN_CACHE_VERSION,
        "source_path": str(corpus_path.resolve()),
        "source_size": source.st_size,
        "source_mtime_ns": source.st_mtime_ns,
        "tokenizer_sha256": _tokenizer_fingerprint(tokenizer),
        "vocab_size": tokenizer.vocab_size,
        "dtype": dtype_name,
        "byte_order": sys.byteorder,
        "token_count": token_count,
    }


def _cache_is_current(
    metadata: object,
    corpus_path: Path,
    tokenizer: BPETokenizer,
    cache_path: Path,
) -> bool:
    if not isinstance(metadata, dict):
        return False
    dtype_sizes = {"uint16": 2, "int32": 4}
    dtype_name = metadata.get("dtype")
    token_count = metadata.get("token_count")
    if dtype_name not in dtype_sizes or not isinstance(token_count, int):
        return False
    expected = _cache_metadata(
        corpus_path, tokenizer, dtype_name=str(dtype_name), token_count=token_count
    )
    return (
        all(metadata.get(key) == value for key, value in expected.items())
        and cache_path.is_file()
        and cache_path.stat().st_size == token_count * dtype_sizes[str(dtype_name)]
    )


def _encode_chunks(
    corpus_path: Path,
    tokenizer: BPETokenizer,
    destination: Path,
    *,
    array_code: str,
) -> int:
    """Encode chunks ending at EOS boundaries and stream IDs to disk."""
    source_size = corpus_path.stat().st_size
    decoder = codecs.getincrementaldecoder("utf-8")()
    carry = ""
    token_count = 0
    bytes_read = 0
    next_progress = TOKENIZE_PROGRESS_BYTES

    print(
        f"building token cache from {corpus_path} "
        f"({source_size:,} bytes; chunked, memory bounded)",
        flush=True,
    )
    with corpus_path.open("rb") as source, destination.open("wb") as output:
        while raw := source.read(TOKENIZE_CHUNK_BYTES):
            bytes_read += len(raw)
            decoded = carry + decoder.decode(raw, final=False)
            boundary = decoded.rfind(EOS_TOKEN)
            if boundary >= 0:
                boundary += len(EOS_TOKEN)
                piece, carry = decoded[:boundary], decoded[boundary:]
                token_ids = tokenizer.encode(piece)
                array(array_code, token_ids).tofile(output)
                token_count += len(token_ids)
            else:
                carry = decoded

            if bytes_read >= next_progress:
                percent = 100.0 * bytes_read / source_size
                print(
                    f"token cache progress={percent:5.1f}% "
                    f"bytes={bytes_read:,}/{source_size:,} tokens={token_count:,}",
                    flush=True,
                )
                next_progress += TOKENIZE_PROGRESS_BYTES

        carry += decoder.decode(b"", final=True)
        if carry:
            token_ids = tokenizer.encode(carry)
            array(array_code, token_ids).tofile(output)
            token_count += len(token_ids)
        output.flush()
        os.fsync(output.fileno())

    print(f"token cache encoded tokens={token_count:,}", flush=True)
    return token_count


def prepare_token_cache(
    corpus_path: str | Path,
    tokenizer: BPETokenizer,
    cache_path: str | Path | None = None,
) -> tuple[Path, dict[str, object]]:
    """Create or reuse an atomic, memory-bounded binary token cache."""
    corpus = validate_text_file(corpus_path)
    cache = Path(cache_path) if cache_path is not None else Path(f"{corpus}.tokens.bin")
    metadata_path = Path(f"{cache}.json")

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        metadata = None
    if _cache_is_current(metadata, corpus, tokenizer, cache):
        assert isinstance(metadata, dict)
        print(
            f"loaded token cache {cache} tokens={int(metadata['token_count']):,}",
            flush=True,
        )
        return cache, metadata

    if tokenizer.vocab_size <= 2**16:
        dtype_name, array_code = "uint16", "H"
    elif tokenizer.vocab_size <= 2**31:
        dtype_name, array_code = "int32", "I"
    else:
        raise ValueError("tokenizer vocabulary is too large for the token cache")

    cache.parent.mkdir(parents=True, exist_ok=True)
    temporary_cache = cache.with_name(f".{cache.name}.part")
    temporary_metadata = metadata_path.with_name(f".{metadata_path.name}.part")
    try:
        token_count = _encode_chunks(
            corpus, tokenizer, temporary_cache, array_code=array_code
        )
        metadata = _cache_metadata(
            corpus,
            tokenizer,
            dtype_name=dtype_name,
            token_count=token_count,
        )
        temporary_metadata.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary_cache, cache)
        os.replace(temporary_metadata, metadata_path)
    except BaseException:
        temporary_cache.unlink(missing_ok=True)
        temporary_metadata.unlink(missing_ok=True)
        raise

    print(f"saved token cache to {cache}", flush=True)
    return cache, metadata


def _build_loaders(
    train_dataset: Dataset[tuple[torch.Tensor, torch.Tensor]],
    val_dataset: Dataset[tuple[torch.Tensor, torch.Tensor]],
    config: DataConfig,
    seed: int,
) -> tuple[DataLoader, DataLoader]:
    generator = torch.Generator().manual_seed(seed)
    common = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    sampler = RandomSampler(
        train_dataset,
        replacement=True,
        num_samples=len(train_dataset),
        generator=generator,
    )
    train_loader = DataLoader(
        train_dataset, sampler=sampler, drop_last=True, **common
    )
    val_loader = DataLoader(val_dataset, shuffle=False, drop_last=False, **common)
    if len(train_loader) == 0:
        raise ValueError("training split is too small for the selected batch_size")
    return train_loader, val_loader


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
    return _build_loaders(train_dataset, val_dataset, config, seed)


def build_file_dataloaders(
    path: str | Path,
    tokenizer: BPETokenizer,
    block_size: int,
    config: DataConfig,
    seed: int,
    cache_path: str | Path | None = None,
) -> tuple[DataLoader, DataLoader]:
    cache, metadata = prepare_token_cache(path, tokenizer, cache_path)
    token_count = int(metadata["token_count"])
    dtype_name = str(metadata["dtype"])
    dtype = {"uint16": torch.uint16, "int32": torch.int32}[dtype_name]
    tokens = torch.from_file(
        str(cache), shared=False, size=token_count, dtype=dtype
    )
    split_index = int(token_count * config.train_fraction)
    train_dataset = MemoryMappedTokenDataset(tokens, 0, split_index, block_size)
    val_dataset = MemoryMappedTokenDataset(
        tokens, split_index, token_count, block_size
    )
    return _build_loaders(train_dataset, val_dataset, config, seed)
