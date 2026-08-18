#!/usr/bin/env python3
"""Evaluate a myGPT checkpoint on the complete WikiText test corpus."""

from __future__ import annotations

import argparse
from pathlib import Path

from mygpt.evaluation import (
    evaluate_token_ids,
    load_model_for_evaluation,
    write_json_atomic,
)


DEFAULT_DATA = Path("outputs/data/wikitext-2-raw-v1-test.txt")
DEFAULT_OUTPUT = Path("outputs/evaluation/wikitext.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--checkpoint", type=Path, required=True, help="BPE checkpoint to evaluate"
    )
    parser.add_argument(
        "--data", type=Path, default=DEFAULT_DATA, help="raw WikiText text export"
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "mps", "cuda"),
        default="auto",
        help="device used for evaluation",
    )
    parser.add_argument(
        "--stride",
        type=int,
        help="new target tokens per window; half the checkpoint context when omitted",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        help="evaluate only this many leading tokens for a smoke test",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="context windows evaluated per model forward pass",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="print progress after this many windows; zero disables progress",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT, help="JSON report path"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_tokens is not None and args.max_tokens < 2:
        raise ValueError("max_tokens must be at least two")
    if not args.data.is_file():
        raise FileNotFoundError(
            f"WikiText data not found: {args.data}. Run "
            "`python scripts/prepare_data.py --wikitext` first."
        )
    text = args.data.read_text(encoding="utf-8")
    if not text:
        raise ValueError(f"WikiText data is empty: {args.data}")

    loaded = load_model_for_evaluation(args.checkpoint, args.device)
    token_ids = loaded.tokenizer.encode(text)
    full_token_count = len(token_ids)
    if args.max_tokens is not None:
        token_ids = token_ids[: args.max_tokens]
    print(
        f"device={loaded.device} parameters={loaded.model.num_parameters():,} "
        f"tokens={len(token_ids):,}/{full_token_count:,}",
        flush=True,
    )
    metrics = evaluate_token_ids(
        loaded.model,
        token_ids,
        loaded.device,
        stride=args.stride,
        batch_size=args.batch_size,
        progress_every=args.progress_every,
    )
    report = {
        "benchmark": "wikitext-2-raw-v1-test",
        "dataset_path": str(args.data.resolve()),
        "evaluation_scope": "full_test" if args.max_tokens is None else "prefix_smoke_test",
        "full_corpus_tokens": full_token_count,
        "model": loaded.metadata,
        "metrics": metrics,
    }
    write_json_atomic(args.output, report)
    print(
        f"average_nll={float(metrics['average_nll']):.4f} "
        f"perplexity={float(metrics['perplexity']):.2f} "
        f"bits_per_token={float(metrics['bits_per_token']):.4f}"
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
