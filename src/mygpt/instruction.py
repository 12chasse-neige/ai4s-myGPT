"""Stanford Alpaca formatting and response-only instruction datasets."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import partial
import json
from pathlib import Path
import random
from typing import Mapping, Sequence

import torch
from torch.utils.data import DataLoader, Dataset

from .config import InstructionDataConfig
from .data import CharacterTokenizer

IGNORE_INDEX = -100
PROMPT_TEMPLATE_VERSION = "stanford_alpaca_v1"

PROMPT_WITH_INPUT = (
    "Below is an instruction that describes a task, paired with an input that "
    "provides further context. Write a response that appropriately completes "
    "the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}"
    "\n\n### Response:\n"
)
PROMPT_WITHOUT_INPUT = (
    "Below is an instruction that describes a task. Write a response that "
    "appropriately completes the request.\n\n### Instruction:\n{instruction}"
    "\n\n### Response:\n"
)


def normalize_field(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def format_alpaca_prompt(instruction: str, input_text: str = "") -> str:
    """Format an instruction with the original Stanford Alpaca template."""
    instruction = normalize_field(instruction)
    input_text = normalize_field(input_text)
    if not instruction:
        raise ValueError("instruction cannot be empty")
    if input_text:
        return PROMPT_WITH_INPUT.format(instruction=instruction, input=input_text)
    return PROMPT_WITHOUT_INPUT.format(instruction=instruction)


def load_alpaca_records(path: str | Path) -> list[dict[str, str]]:
    """Read and validate structured Stanford Alpaca records."""
    source = Path(path)
    with source.open(encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list) or not records:
        raise ValueError("Stanford Alpaca data must be a non-empty JSON list")
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError(f"record {index} must be a JSON object")
        for field in ("instruction", "input", "output"):
            if not isinstance(record.get(field), str):
                raise ValueError(f"record {index} must contain string field {field!r}")
    return records


def split_alpaca_records(
    records: Sequence[dict[str, str]],
    train_fraction: float,
    seed: int,
    max_records: int | None = None,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Shuffle, optionally limit, and split records deterministically."""
    indices = list(range(len(records)))
    random.Random(seed).shuffle(indices)
    if max_records is not None:
        indices = indices[:max_records]
    if len(indices) < 2:
        raise ValueError("instruction tuning requires at least two records")
    train_size = min(len(indices) - 1, max(1, int(len(indices) * train_fraction)))
    train = [records[index] for index in indices[:train_size]]
    validation = [records[index] for index in indices[train_size:]]
    return train, validation


def instruction_characters(records: Sequence[dict[str, str]]) -> list[str]:
    """Return sorted characters needed to encode a collection of records."""
    characters: set[str] = set()
    for record in records:
        characters.update(format_alpaca_prompt(record["instruction"], record["input"]))
        characters.update(normalize_field(record["output"]))
    return sorted(characters)


def missing_instruction_characters(
    records: Sequence[dict[str, str]], tokenizer: CharacterTokenizer
) -> list[str]:
    return [
        character
        for character in instruction_characters(records)
        if character not in tokenizer.stoi
    ]


@dataclass
class InstructionStats:
    total_records: int = 0
    kept_records: int = 0
    empty_responses: int = 0
    prompts_too_long: int = 0
    inputs_truncated: int = 0
    responses_truncated: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class EncodedInstruction:
    tokens: torch.Tensor
    targets: torch.Tensor
    input_truncated: bool
    response_truncated: bool


def fit_alpaca_prompt(
    instruction: str,
    input_text: str,
    max_length: int,
    *,
    reserve_tokens: int = 1,
) -> tuple[str, bool] | None:
    """Fit a complete instruction by trimming or removing optional input."""
    instruction = normalize_field(instruction)
    input_text = normalize_field(input_text)
    minimal_prompt = format_alpaca_prompt(instruction)
    if len(minimal_prompt) + reserve_tokens > max_length:
        return None
    if not input_text:
        return minimal_prompt, False

    complete_prompt = format_alpaca_prompt(instruction, input_text)
    if len(complete_prompt) + reserve_tokens <= max_length:
        return complete_prompt, False

    input_prefix, input_suffix = PROMPT_WITH_INPUT.split("{input}")
    fixed_prefix = input_prefix.format(instruction=instruction)
    available = max_length - reserve_tokens - len(fixed_prefix) - len(input_suffix)
    if available <= 0:
        return minimal_prompt, True
    return fixed_prefix + input_text[:available] + input_suffix, True


def encode_alpaca_record(
    record: Mapping[str, str],
    tokenizer: CharacterTokenizer,
    max_length: int,
) -> EncodedInstruction | None:
    """Encode one record with prompt masking and response-preserving truncation."""
    response = normalize_field(record["output"])
    if not response:
        return None
    fitted = fit_alpaca_prompt(
        record["instruction"], record["input"], max_length, reserve_tokens=1
    )
    if fitted is None:
        return None
    prompt, input_truncated = fitted
    prompt_ids = tokenizer.encode(prompt)
    response_ids = tokenizer.encode(response)
    response_budget = max_length - len(prompt_ids)
    if response_budget <= 0:
        return None
    response_truncated = len(response_ids) > response_budget
    response_ids = response_ids[:response_budget]
    tokens = prompt_ids + response_ids
    targets = [IGNORE_INDEX] * (len(prompt_ids) - 1) + response_ids + [tokenizer.eos_id]
    if len(tokens) != len(targets):
        raise AssertionError("instruction inputs and targets are misaligned")
    return EncodedInstruction(
        tokens=torch.tensor(tokens, dtype=torch.long),
        targets=torch.tensor(targets, dtype=torch.long),
        input_truncated=input_truncated,
        response_truncated=response_truncated,
    )


class InstructionDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        records: Sequence[dict[str, str]],
        tokenizer: CharacterTokenizer,
        max_length: int,
    ) -> None:
        self.samples: list[EncodedInstruction] = []
        self.stats = InstructionStats(total_records=len(records))
        for record in records:
            if not normalize_field(record["output"]):
                self.stats.empty_responses += 1
                continue
            sample = encode_alpaca_record(record, tokenizer, max_length)
            if sample is None:
                self.stats.prompts_too_long += 1
                continue
            self.samples.append(sample)
            self.stats.kept_records += 1
            self.stats.inputs_truncated += int(sample.input_truncated)
            self.stats.responses_truncated += int(sample.response_truncated)
        if not self.samples:
            raise ValueError("no instruction records fit the selected context length")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        return sample.tokens, sample.targets


def collate_instruction_batch(
    batch: Sequence[tuple[torch.Tensor, torch.Tensor]], pad_id: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-pad variable-length instruction examples."""
    max_length = max(len(tokens) for tokens, _ in batch)
    tokens = torch.full((len(batch), max_length), pad_id, dtype=torch.long)
    targets = torch.full(
        (len(batch), max_length), IGNORE_INDEX, dtype=torch.long
    )
    for row, (sample_tokens, sample_targets) in enumerate(batch):
        tokens[row, : len(sample_tokens)] = sample_tokens
        targets[row, : len(sample_targets)] = sample_targets
    return tokens, targets


def build_instruction_dataloaders(
    train_records: Sequence[dict[str, str]],
    validation_records: Sequence[dict[str, str]],
    tokenizer: CharacterTokenizer,
    block_size: int,
    config: InstructionDataConfig,
    seed: int,
) -> tuple[DataLoader, DataLoader, dict[str, object]]:
    """Build deterministic SFT loaders and return preprocessing statistics."""
    max_length = config.max_seq_length or block_size
    if max_length > block_size:
        raise ValueError(
            f"max_seq_length {max_length} exceeds checkpoint block_size {block_size}"
        )
    train_dataset = InstructionDataset(train_records, tokenizer, max_length)
    validation_dataset = InstructionDataset(validation_records, tokenizer, max_length)
    generator = torch.Generator().manual_seed(seed)
    common = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "collate_fn": partial(collate_instruction_batch, pad_id=tokenizer.pad_id),
    }
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=generator,
        drop_last=len(train_dataset) >= config.batch_size,
        **common,
    )
    validation_loader = DataLoader(
        validation_dataset, shuffle=False, drop_last=False, **common
    )
    stats: dict[str, object] = {
        "max_seq_length": max_length,
        "train": train_dataset.stats.to_dict(),
        "validation": validation_dataset.stats.to_dict(),
    }
    return train_loader, validation_loader, stats
