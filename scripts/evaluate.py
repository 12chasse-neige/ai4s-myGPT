#!/usr/bin/env python3
"""Evaluate a myGPT checkpoint on its validation corpus."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from mygpt.checkpoint import load_checkpoint
from mygpt.config import ExperimentConfig
from mygpt.data import CharacterTokenizer, build_dataloaders, load_text
from mygpt.model import GPT
from mygpt.trainer import evaluate, select_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, help="override the checkpoint corpus")
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--batches", type=int, help="maximum validation batches")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = load_checkpoint(args.checkpoint)
    config = ExperimentConfig.from_dict(checkpoint["config"])
    config.training.device = args.device
    if args.data:
        config.data.path = str(args.data)
    tokenizer = CharacterTokenizer.from_state_dict(checkpoint["tokenizer"])
    text = load_text(config.data.path)
    _, val_loader = build_dataloaders(
        text, tokenizer, config.model.block_size, config.data, config.training.seed
    )
    device = select_device(args.device)
    model = GPT(config.model).to(device)
    model.load_state_dict(checkpoint["model"])
    loss = evaluate(model, val_loader, device, args.batches)
    print(f"validation_loss={loss:.4f} perplexity={math.exp(loss):.2f}")


if __name__ == "__main__":
    main()

