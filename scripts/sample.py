#!/usr/bin/env python3
"""Generate text from a trained myGPT checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from mygpt.checkpoint import load_checkpoint
from mygpt.config import ExperimentConfig
from mygpt.generation import generate_text
from mygpt.model import GPT
from mygpt.tokenizer import BPETokenizer
from mygpt.trainer import select_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--checkpoint", type=Path, required=True, help="checkpoint used for generation"
    )
    parser.add_argument("--prompt", default="The ", help="text that starts the generation")
    parser.add_argument(
        "--max-new-tokens", type=int, default=120, help="maximum tokens to generate"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.8, help="sampling temperature"
    )
    parser.add_argument(
        "--top-k", type=int, default=40, help="sample from the top-k token candidates"
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "mps", "cuda"),
        default="auto",
        help="device used for generation",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = select_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint, device)
    config = ExperimentConfig.from_dict(checkpoint["config"])
    tokenizer = BPETokenizer.from_state_dict(checkpoint["tokenizer"])
    model = GPT(config.model).to(device)
    model.load_state_dict(checkpoint["model"])
    text = generate_text(
        model,
        tokenizer,
        args.prompt,
        args.max_new_tokens,
        args.temperature,
        args.top_k,
        device,
        tokenizer.eos_id,
    )
    print(text)


if __name__ == "__main__":
    main()
