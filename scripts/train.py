#!/usr/bin/env python3
"""Train a character-level GPT model."""

from __future__ import annotations

import argparse
from pathlib import Path

from mygpt.checkpoint import load_checkpoint
from mygpt.config import ExperimentConfig
from mygpt.data import CharacterTokenizer, build_dataloaders, load_text
from mygpt.model import GPT
from mygpt.trainer import train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/gpt_small.yaml"),
        help="experiment configuration file",
    )
    parser.add_argument("--data", type=Path, help="override data.path")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "mps", "cuda"),
        help="override training.device",
    )
    parser.add_argument("--max-steps", type=int, help="override training.max_steps")
    parser.add_argument("--resume", type=Path, help="resume from a checkpoint")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ExperimentConfig.from_yaml(args.config)
    if args.data:
        config.data.path = str(args.data)
    if args.device:
        config.training.device = args.device
    if args.max_steps is not None:
        config.training.max_steps = args.max_steps
        config.training.warmup_steps = min(config.training.warmup_steps, max(0, args.max_steps - 1))
    config.validate()

    text = load_text(config.data.path)
    start_step, optimizer_state, best_val_loss = 0, None, float("inf")
    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        tokenizer = CharacterTokenizer.from_state_dict(checkpoint["tokenizer"])
    else:
        checkpoint = None
        tokenizer = CharacterTokenizer.from_text(text)
    config.model.vocab_size = tokenizer.vocab_size
    train_loader, val_loader = build_dataloaders(
        text, tokenizer, config.model.block_size, config.data, config.training.seed
    )
    model = GPT(config.model)
    if checkpoint:
        model.load_state_dict(checkpoint["model"])
        optimizer_state = checkpoint.get("optimizer")
        start_step = int(checkpoint.get("step", 0))
        best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
        print(f"resuming {args.resume} from step {start_step}")
    train(
        model,
        train_loader,
        val_loader,
        tokenizer,
        config,
        start_step=start_step,
        optimizer_state=optimizer_state,
        best_val_loss=best_val_loss,
    )


if __name__ == "__main__":
    main()
