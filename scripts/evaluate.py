#!/usr/bin/env python3
"""Evaluate a myGPT checkpoint on its validation corpus."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from mygpt.checkpoint import load_checkpoint
from mygpt.config import ExperimentConfig, SFTConfig
from mygpt.data import CharacterTokenizer, build_dataloaders, load_text
from mygpt.instruction import (
    build_instruction_dataloaders,
    load_alpaca_records,
    missing_instruction_characters,
    split_alpaca_records,
)
from mygpt.model import GPT
from mygpt.trainer import evaluate, select_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--checkpoint", type=Path, required=True, help="checkpoint to evaluate"
    )
    parser.add_argument("--data", type=Path, help="override the checkpoint corpus")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "mps", "cuda"),
        default="auto",
        help="device used for evaluation",
    )
    parser.add_argument(
        "--batches", type=int, help="maximum validation batches; all when omitted"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = load_checkpoint(args.checkpoint)
    config = ExperimentConfig.from_dict(checkpoint["config"])
    config.training.device = args.device
    tokenizer = CharacterTokenizer.from_state_dict(checkpoint["tokenizer"])
    if checkpoint.get("training_stage") == "sft":
        if "sft_config" not in checkpoint:
            raise ValueError("SFT checkpoint has no saved configuration")
        sft_config = SFTConfig.from_dict(checkpoint["sft_config"])
        if args.data:
            sft_config.data.path = str(args.data)
        records = load_alpaca_records(sft_config.data.path)
        train_records, validation_records = split_alpaca_records(
            records,
            sft_config.data.train_fraction,
            sft_config.training.seed,
            sft_config.data.max_records,
        )
        missing = missing_instruction_characters(
            [*train_records, *validation_records], tokenizer
        )
        if missing:
            raise ValueError(
                f"evaluation data contains {len(missing)} characters absent from "
                "the checkpoint tokenizer"
            )
        _, val_loader, _ = build_instruction_dataloaders(
            train_records,
            validation_records,
            tokenizer,
            config.model.block_size,
            sft_config.data,
            sft_config.training.seed,
        )
        metric_name = "response_validation_loss"
    else:
        if args.data:
            config.data.path = str(args.data)
        text = load_text(config.data.path)
        _, val_loader = build_dataloaders(
            text, tokenizer, config.model.block_size, config.data, config.training.seed
        )
        metric_name = "validation_loss"
    device = select_device(args.device)
    model = GPT(config.model).to(device)
    model.load_state_dict(checkpoint["model"])
    loss = evaluate(model, val_loader, device, args.batches)
    print(f"{metric_name}={loss:.4f} perplexity={math.exp(loss):.2f}")


if __name__ == "__main__":
    main()
