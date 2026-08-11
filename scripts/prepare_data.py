#!/usr/bin/env python3
"""Prepare a local UTF-8 corpus from TinyStories or a supplied text file."""

from __future__ import annotations

import argparse
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Iterable, Mapping

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT.parent / "outputs"
DEFAULT_DATASET = "roneneldan/TinyStories"
DEFAULT_MAX_STORIES = 100_000
DEFAULT_OUTPUT = OUTPUT_DIR / "data" / "tinystories.txt"


def normalize_text(text: str) -> str:
    """Normalize newlines and remove surrounding whitespace."""
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def load_tinystories(
    dataset_name: str = DEFAULT_DATASET,
    split: str = "train",
    max_stories: int = DEFAULT_MAX_STORIES,
) -> Iterable[str]:
    """Stream a bounded number of non-empty stories from Hugging Face."""
    if max_stories <= 0:
        raise ValueError("max_stories must be positive")
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError(
            'TinyStories preparation requires the "datasets" package; '
            'install the project with `python -m pip install -e ".[test]"`.'
        ) from error

    dataset = load_dataset(dataset_name, split=split, streaming=True)
    written = 0
    for record in dataset:
        if not isinstance(record, Mapping) or not isinstance(record.get("text"), str):
            raise ValueError(f'{dataset_name!r} records must contain a string "text" field')
        story = normalize_text(record["text"])
        if not story:
            continue
        yield story
        written += 1
        if written >= max_stories:
            break


def write_stories(stories: Iterable[str], output: Path) -> tuple[int, int]:
    """Write stories separated by blank lines and return story/character counts."""
    output.parent.mkdir(parents=True, exist_ok=True)
    story_count = 0
    character_count = 0
    temporary_output: Path | None = None
    try:
        with NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="\n",
            dir=output.parent,
            prefix=f".{output.name}.",
            delete=False,
        ) as handle:
            temporary_output = Path(handle.name)
            for story in stories:
                normalized = normalize_text(story)
                if not normalized:
                    continue
                if story_count:
                    handle.write("\n\n")
                    character_count += 2
                handle.write(normalized)
                story_count += 1
                character_count += len(normalized)
            if story_count == 0:
                raise ValueError("input corpus is empty")
            handle.write("\n")
            character_count += 1
        temporary_output.replace(output)
    except BaseException:
        if temporary_output is not None:
            temporary_output.unlink(missing_ok=True)
        raise
    return story_count, character_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="normalize this text file instead of downloading TinyStories",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT, help="destination corpus file"
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Hugging Face dataset ID")
    parser.add_argument("--split", default="train", help="Hugging Face dataset split")
    parser.add_argument(
        "--max-stories",
        type=int,
        default=DEFAULT_MAX_STORIES,
        help="maximum stories to stream",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.input:
        stories = [args.input.read_text(encoding="utf-8")]
        source = str(args.input)
    else:
        stories = load_tinystories(args.dataset, args.split, args.max_stories)
        source = f"{args.dataset}:{args.split}"
    story_count, character_count = write_stories(stories, args.output)
    print(
        f"wrote {story_count:,} stories ({character_count:,} characters) "
        f"from {source} to {args.output}"
    )
    quit


if __name__ == "__main__":
    main()
