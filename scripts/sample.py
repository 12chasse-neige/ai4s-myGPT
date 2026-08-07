#!/usr/bin/env python3
"""Generate text from a trained myGPT checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from mygpt.checkpoint import load_checkpoint
from mygpt.config import ExperimentConfig
from mygpt.data import CharacterTokenizer
from mygpt.generation import generate_text
from mygpt.model import GPT
from mygpt.trainer import select_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompt", default="The ")
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = select_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint, device)
    config = ExperimentConfig.from_dict(checkpoint["config"])
    tokenizer = CharacterTokenizer.from_state_dict(checkpoint["tokenizer"])
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
    )
    print(text)


if __name__ == "__main__":
    main()

