#!/usr/bin/env python3
"""Train a BPE-tokenized GPT model."""

from __future__ import annotations

import argparse
from pathlib import Path

from mygpt.checkpoint import load_checkpoint
from mygpt.config import ExperimentConfig
from mygpt.data import (
    build_dataloaders,
    build_file_dataloaders,
    load_text,
    text_file_contains,
    validate_text_file,
)
from mygpt.model import GPT
from mygpt.tokenizer import BPETokenizer, EOS_TOKEN
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
        config.training.warmup_steps = min(
            config.training.warmup_steps, max(0, args.max_steps - 1)
        )
    config.validate()

    if config.data.path is None:
        text = load_text(None)
        corpus_path = None
    else:
        text = None
        corpus_path = validate_text_file(config.data.path)
    start_step, optimizer_state, best_val_loss = 0, None, float("inf")
    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        tokenizer = BPETokenizer.from_state_dict(checkpoint["tokenizer"])
    else:
        checkpoint = None
        has_eos = (
            EOS_TOKEN in text
            if text is not None
            else text_file_contains(corpus_path, EOS_TOKEN)
        )
        if not has_eos:
            raise ValueError(
                f"training corpus has no {EOS_TOKEN} story boundaries; regenerate it "
                "with `python scripts/prepare_data.py --tinystories`"
            )
        tokenizer_path = Path(config.tokenizer.path)
        if tokenizer_path.is_file():
            tokenizer = BPETokenizer.from_file(tokenizer_path)
            print(f"loaded BPE tokenizer from {tokenizer_path}")
        else:
            if corpus_path is None:
                tokenizer = BPETokenizer.train_from_iterator(
                    [text],
                    vocab_size=config.tokenizer.vocab_size,
                    min_frequency=config.tokenizer.min_frequency,
                )
            else:
                tokenizer = BPETokenizer.train(
                    [corpus_path],
                    vocab_size=config.tokenizer.vocab_size,
                    min_frequency=config.tokenizer.min_frequency,
                )
            tokenizer.save(tokenizer_path)
            print(
                f"trained BPE tokenizer vocabulary={tokenizer.vocab_size} "
                f"and saved it to {tokenizer_path}"
            )
    config.model.vocab_size = tokenizer.vocab_size
    if corpus_path is None:
        assert text is not None
        train_loader, val_loader = build_dataloaders(
            text, tokenizer, config.model.block_size, config.data, config.training.seed
        )
    else:
        train_loader, val_loader = build_file_dataloaders(
            corpus_path,
            tokenizer,
            config.model.block_size,
            config.data,
            config.training.seed,
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
