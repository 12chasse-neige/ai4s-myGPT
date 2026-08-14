#!/usr/bin/env python3
"""Generate a response from an instruction-tuned myGPT checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from mygpt.checkpoint import load_checkpoint
from mygpt.config import ExperimentConfig
from mygpt.generation import generate_tokens
from mygpt.instruction import fit_alpaca_prompt
from mygpt.model import GPT
from mygpt.tokenizer import BPETokenizer
from mygpt.trainer import select_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="SFT checkpoint used for generation",
    )
    parser.add_argument("--instruction", required=True, help="task for the model")
    parser.add_argument("--input", default="", help="optional task context")
    parser.add_argument(
        "--max-new-tokens", type=int, default=256, help="maximum response tokens"
    )
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "mps", "cuda"),
        default="auto",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = select_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint, device)
    if checkpoint.get("training_stage") != "sft":
        raise ValueError(f"not an SFT checkpoint: {args.checkpoint}")
    config = ExperimentConfig.from_dict(checkpoint["config"])
    tokenizer = BPETokenizer.from_state_dict(checkpoint["tokenizer"])
    fitted = fit_alpaca_prompt(
        args.instruction,
        args.input,
        config.model.block_size,
        reserve_tokens=0,
        tokenizer=tokenizer,
    )
    if fitted is None:
        raise ValueError("instruction does not fit the checkpoint context length")
    prompt, _ = fitted
    prompt_tokens = tokenizer.encode(prompt)
    tokens = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
    model = GPT(config.model).to(device)
    model.load_state_dict(checkpoint["model"])
    generated = generate_tokens(
        model,
        tokens,
        args.max_new_tokens,
        args.temperature,
        args.top_k,
        tokenizer.eos_id,
    )
    response_ids = generated[0, len(prompt_tokens) :].tolist()
    print(tokenizer.decode(response_ids, skip_special_tokens=True).strip())


if __name__ == "__main__":
    main()
