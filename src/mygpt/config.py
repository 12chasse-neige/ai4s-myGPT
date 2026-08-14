"""Typed experiment configuration loaded from YAML."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    vocab_size: int = 0
    block_size: int = 128
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.1
    bias: bool = True

    def validate(self) -> None:
        if self.vocab_size < 0:
            raise ValueError("vocab_size cannot be negative")
        if min(self.block_size, self.n_layer, self.n_head, self.n_embd) <= 0:
            raise ValueError("model dimensions must be positive")
        if self.n_embd % self.n_head:
            raise ValueError("n_embd must be divisible by n_head")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")


@dataclass
class DataConfig:
    path: str | None = "outputs/data/tinystories.txt"
    train_fraction: float = 0.9
    batch_size: int = 32
    num_workers: int = 0

    def validate(self) -> None:
        if not 0.5 <= self.train_fraction < 1.0:
            raise ValueError("train_fraction must be in [0.5, 1.0)")
        if self.batch_size <= 0 or self.num_workers < 0:
            raise ValueError("batch_size must be positive and num_workers non-negative")


@dataclass
class TokenizerConfig:
    type: str = "bpe"
    path: str = "outputs/tokenizers/tinystories-10k.json"
    vocab_size: int = 10_000
    min_frequency: int = 2

    def validate(self) -> None:
        if self.type != "bpe":
            raise ValueError("tokenizer.type must be bpe")
        if not self.path:
            raise ValueError("tokenizer.path is required for BPE")
        if self.vocab_size <= 260:
            raise ValueError("tokenizer.vocab_size must exceed the byte alphabet")
        if self.min_frequency <= 0:
            raise ValueError("tokenizer.min_frequency must be positive")


@dataclass
class InstructionDataConfig:
    path: str = "outputs/data/stanford_alpaca.json"
    train_fraction: float = 0.95
    batch_size: int = 8
    num_workers: int = 0
    max_seq_length: int | None = None
    max_records: int | None = None

    def validate(self) -> None:
        if not 0.5 <= self.train_fraction < 1.0:
            raise ValueError("train_fraction must be in [0.5, 1.0)")
        if self.batch_size <= 0 or self.num_workers < 0:
            raise ValueError("batch_size must be positive and num_workers non-negative")
        if self.max_seq_length is not None and self.max_seq_length <= 1:
            raise ValueError("max_seq_length must be greater than one")
        if self.max_records is not None and self.max_records < 2:
            raise ValueError("max_records must be at least two")


@dataclass
class TrainingConfig:
    seed: int = 42
    max_steps: int = 1000
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    warmup_steps: int = 100
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    eval_interval: int = 100
    eval_batches: int = 20
    log_interval: int = 10
    output_dir: str = "outputs/default"
    device: str = "auto"

    def validate(self) -> None:
        positive = (
            self.max_steps,
            self.learning_rate,
            self.eval_interval,
            self.eval_batches,
            self.log_interval,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("steps, rates, and intervals must be positive")
        if not 0 <= self.warmup_steps < self.max_steps:
            raise ValueError("warmup_steps must be in [0, max_steps)")
        if not 0 <= self.min_learning_rate <= self.learning_rate:
            raise ValueError("min_learning_rate must be between 0 and learning_rate")
        if self.device not in {"auto", "cpu", "mps", "cuda"}:
            raise ValueError("device must be auto, cpu, mps, or cuda")


@dataclass
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "ExperimentConfig":
        config = cls(
            model=ModelConfig(**values.get("model", {})),
            data=DataConfig(**values.get("data", {})),
            tokenizer=TokenizerConfig(**values.get("tokenizer", {})),
            training=TrainingConfig(**values.get("training", {})),
        )
        config.validate()
        return config

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            values = yaml.safe_load(handle) or {}
        if not isinstance(values, dict):
            raise ValueError("configuration root must be a mapping")
        return cls.from_dict(values)

    def validate(self) -> None:
        self.model.validate()
        self.data.validate()
        self.tokenizer.validate()
        self.training.validate()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SFTConfig:
    pretrained_checkpoint: str = "outputs/gpt-pretrain/best.pt"
    data: InstructionDataConfig = field(default_factory=InstructionDataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "SFTConfig":
        config = cls(
            pretrained_checkpoint=values.get(
                "pretrained_checkpoint", "outputs/gpt-pretrain/best.pt"
            ),
            data=InstructionDataConfig(**values.get("data", {})),
            training=TrainingConfig(**values.get("training", {})),
        )
        config.validate()
        return config

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SFTConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            values = yaml.safe_load(handle) or {}
        if not isinstance(values, dict):
            raise ValueError("configuration root must be a mapping")
        return cls.from_dict(values)

    def validate(self) -> None:
        if not self.pretrained_checkpoint:
            raise ValueError("pretrained_checkpoint cannot be empty")
        self.data.validate()
        self.training.validate()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
