#!/usr/bin/env python3
"""Write the bundled demo corpus or normalize a supplied UTF-8 text file."""

from __future__ import annotations

import argparse
from pathlib import Path

from mygpt.data import DEMO_TEXT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="optional source text file")
    parser.add_argument("--output", type=Path, default=Path("outputs/data/demo.txt"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    text = args.input.read_text(encoding="utf-8") if args.input else DEMO_TEXT
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if not text.strip():
        raise ValueError("input corpus is empty")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    print(f"wrote {len(text):,} characters to {args.output}")


if __name__ == "__main__":
    main()

