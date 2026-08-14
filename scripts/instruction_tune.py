#!/usr/bin/env python3
"""Supervised instruction tuning for a pretrained myGPT checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mygpt.checkpoint import load_checkpoint
from mygpt.config import DataConfig, ExperimentConfig, SFTConfig
from mygpt.instruction import (
    PROMPT_TEMPLATE_VERSION,
    build_instruction_dataloaders,
    load_alpaca_records,
    split_alpaca_records,
)
from mygpt.model import GPT
from mygpt.tokenizer import BPETokenizer
from mygpt.trainer import train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/gpt_sft.yaml"),
        help="instruction-tuning configuration file",
    )
    initialization = parser.add_mutually_exclusive_group()
    initialization.add_argument(
        "--checkpoint", type=Path, help="override the pretrained checkpoint"
    )
    initialization.add_argument(
        "--resume", type=Path, help="resume an existing SFT checkpoint"
    )
    parser.add_argument("--data", type=Path, help="override the Alpaca JSON path")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "mps", "cuda"),
        help="override training.device",
    )
    parser.add_argument("--max-steps", type=int, help="override training.max_steps")
    parser.add_argument(
        "--max-records", type=int, help="limit records before the deterministic split"
    )
    parser.add_argument(
        "--output-dir", type=Path, help="override training.output_dir"
    )
    return parser.parse_args()


def apply_overrides(config: SFTConfig, args: argparse.Namespace) -> None:
    if args.checkpoint:
        config.pretrained_checkpoint = str(args.checkpoint)
    if args.data:
        config.data.path = str(args.data)
    if args.device:
        config.training.device = args.device
    if args.max_steps is not None:
        config.training.max_steps = args.max_steps
        config.training.warmup_steps = min(
            config.training.warmup_steps, max(0, args.max_steps - 1)
        )
    if args.max_records is not None:
        config.data.max_records = args.max_records
    if args.output_dir:
        config.training.output_dir = str(args.output_dir)
    config.validate()


def standard_experiment_config(
    model_config, tokenizer_config, sft_config: SFTConfig
) -> ExperimentConfig:
    return ExperimentConfig(
        model=model_config,
        data=DataConfig(
            path=sft_config.data.path,
            train_fraction=sft_config.data.train_fraction,
            batch_size=sft_config.data.batch_size,
            num_workers=sft_config.data.num_workers,
        ),
        tokenizer=tokenizer_config,
        training=sft_config.training,
    )


def main() -> None:
    args = parse_args()
    start_step = 0
    optimizer_state = None
    best_val_loss = float("inf")
    history: list[dict[str, float]] = []

    if args.resume:
        if args.data or args.max_records is not None:
            raise ValueError("--data and --max-records cannot change when resuming SFT")
        checkpoint = load_checkpoint(args.resume)
        if checkpoint.get("training_stage") != "sft":
            raise ValueError(f"not an SFT checkpoint: {args.resume}")
        if "sft_config" not in checkpoint:
            raise ValueError(f"SFT checkpoint has no saved configuration: {args.resume}")
        sft_config = SFTConfig.from_dict(checkpoint["sft_config"])
        apply_overrides(sft_config, args)
        start_step = int(checkpoint.get("step", 0))
        if sft_config.training.max_steps <= start_step:
            raise ValueError(
                f"max_steps must exceed resumed step {start_step}; "
                f"got {sft_config.training.max_steps}"
            )
        optimizer_state = checkpoint.get("optimizer")
        best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
        history = list(checkpoint.get("history", []))
        tokenizer = BPETokenizer.from_state_dict(checkpoint["tokenizer"])
        experiment_config = ExperimentConfig.from_dict(checkpoint["config"])
        experiment_config.training = sft_config.training
        model = GPT(experiment_config.model)
        model.load_state_dict(checkpoint["model"])
        source_checkpoint = checkpoint.get("source_checkpoint")
        source_step = checkpoint.get("source_step")
        added_tokens = list(checkpoint.get("added_tokens", []))
        print(f"resuming SFT {args.resume} from step {start_step}")
    else:
        sft_config = SFTConfig.from_yaml(args.config)
        apply_overrides(sft_config, args)
        source_path = Path(sft_config.pretrained_checkpoint)
        checkpoint = load_checkpoint(source_path)
        if checkpoint.get("training_stage") == "sft":
            raise ValueError("use --resume for an SFT checkpoint")
        pretrained_config = ExperimentConfig.from_dict(checkpoint["config"])
        tokenizer = BPETokenizer.from_state_dict(checkpoint["tokenizer"])
        source_checkpoint = str(source_path)
        source_step = int(checkpoint.get("step", 0))
        model = GPT(pretrained_config.model)
        model.load_state_dict(checkpoint["model"])

    records = load_alpaca_records(sft_config.data.path)
    train_records, validation_records = split_alpaca_records(
        records,
        sft_config.data.train_fraction,
        sft_config.training.seed,
        sft_config.data.max_records,
    )
    if not args.resume:
        added_tokens = []
        experiment_config = standard_experiment_config(
            model.config, pretrained_config.tokenizer, sft_config
        )

    experiment_config.model.vocab_size = tokenizer.vocab_size
    experiment_config.data.path = sft_config.data.path
    experiment_config.data.train_fraction = sft_config.data.train_fraction
    experiment_config.data.batch_size = sft_config.data.batch_size
    experiment_config.data.num_workers = sft_config.data.num_workers
    experiment_config.training = sft_config.training
    experiment_config.validate()

    train_loader, validation_loader, preprocessing_stats = (
        build_instruction_dataloaders(
            train_records,
            validation_records,
            tokenizer,
            experiment_config.model.block_size,
            sft_config.data,
            sft_config.training.seed,
        )
    )
    print(f"preprocessing={json.dumps(preprocessing_stats, sort_keys=True)}")
    print(
        f"source={source_checkpoint} source_step={source_step} "
        f"vocabulary={tokenizer.vocab_size} added_tokens={len(added_tokens)}"
    )

    metadata = {
        "training_stage": "sft",
        "source_checkpoint": source_checkpoint,
        "source_step": source_step,
        "prompt_template": PROMPT_TEMPLATE_VERSION,
        "split_seed": sft_config.training.seed,
        "preprocessing": preprocessing_stats,
        "added_tokens": added_tokens,
        "sft_config": sft_config.to_dict(),
    }
    train(
        model,
        train_loader,
        validation_loader,
        tokenizer,
        experiment_config,
        start_step=start_step,
        optimizer_state=optimizer_state,
        best_val_loss=best_val_loss,
        history=history,
        checkpoint_metadata=metadata,
    )


if __name__ == "__main__":
    main()
