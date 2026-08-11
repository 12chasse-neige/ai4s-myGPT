#!/usr/bin/env python3
"""Fetch training data or prepare a local UTF-8 text corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
from tempfile import NamedTemporaryFile
from typing import Iterable, Mapping, Sequence
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT.parent / "outputs"
DEFAULT_DATASET = "roneneldan/TinyStories"
DEFAULT_MAX_STORIES = 100_000
DEFAULT_OUTPUT = OUTPUT_DIR / "data" / "tinystories.txt"
DEFAULT_ALPACA_OUTPUT = OUTPUT_DIR / "data" / "stanford_alpaca.json"
STANFORD_ALPACA_URL = (
    "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/"
    "main/alpaca_data.json"
)
ALPACA_FIELDS = ("instruction", "input", "output")


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


def validate_stanford_alpaca(records: object) -> int:
    """Validate Stanford Alpaca's instruction/input/output JSON schema."""
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise ValueError("Stanford Alpaca data must be a JSON list")
    if not records:
        raise ValueError("Stanford Alpaca data is empty")

    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError(f"Stanford Alpaca record {index} must be a JSON object")
        for field in ALPACA_FIELDS:
            if not isinstance(record.get(field), str):
                raise ValueError(
                    f"Stanford Alpaca record {index} must contain a string {field!r} field"
                )
        if not normalize_text(record["instruction"]):
            raise ValueError(f"Stanford Alpaca record {index} has an empty instruction")
    return len(records)


def load_stanford_alpaca(path: Path = DEFAULT_ALPACA_OUTPUT) -> list[dict[str, str]]:
    """Read and validate a local Stanford Alpaca JSON file."""
    try:
        with path.open(encoding="utf-8") as handle:
            records = json.load(handle)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid Stanford Alpaca JSON in {path}: {error}") from error

    validate_stanford_alpaca(records)
    return records


def fetch_stanford_alpaca(
    output: Path = DEFAULT_ALPACA_OUTPUT,
    url: str = STANFORD_ALPACA_URL,
    timeout: float = 120.0,
) -> tuple[int, int]:
    """Download, validate, and atomically save the Stanford Alpaca dataset."""
    if timeout <= 0:
        raise ValueError("timeout must be positive")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output: Path | None = None
    try:
        with NamedTemporaryFile(
            "wb",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".part",
            delete=False,
        ) as handle:
            temporary_output = Path(handle.name)
            request = Request(
                url,
                headers={
                    "Accept": "application/json",
                    "User-Agent": "ai4s-mygpt/0.1",
                },
            )
            with urlopen(request, timeout=timeout) as response:
                shutil.copyfileobj(response, handle)

        record_count = len(load_stanford_alpaca(temporary_output))
        byte_count = temporary_output.stat().st_size
        temporary_output.replace(output)
    except BaseException:
        if temporary_output is not None:
            temporary_output.unlink(missing_ok=True)
        raise
    return record_count, byte_count


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
    sources = parser.add_mutually_exclusive_group()
    sources.add_argument(
        "--input",
        type=Path,
        help="normalize this text file instead of downloading TinyStories",
    )
    sources.add_argument(
        "--tinystories",
        action="store_true",
        default=True,
        help="fetch TinyStories (also the default when no source is selected)",
    )
    sources.add_argument(
        "--stanford-alpaca",
        dest="tinystories",
        action="store_false",
        default=argparse.SUPPRESS,
        help="fetch Stanford Alpaca JSON for later instruction tuning",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="destination file (chosen from the selected data source by default)",
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
    if not args.input and not args.tinystories:
        output = args.output or DEFAULT_ALPACA_OUTPUT
        record_count, byte_count = fetch_stanford_alpaca(output)
        print(
            f"wrote {record_count:,} instruction records ({byte_count:,} bytes) "
            f"from {STANFORD_ALPACA_URL} to {output}"
        )
        return

    if args.input:
        stories = [args.input.read_text(encoding="utf-8")]
        source = str(args.input)
    else:
        stories = load_tinystories(args.dataset, args.split, args.max_stories)
        source = f"{args.dataset}:{args.split}"
    output = args.output or DEFAULT_OUTPUT
    story_count, character_count = write_stories(stories, output)
    print(
        f"wrote {story_count:,} stories ({character_count:,} characters) "
        f"from {source} to {output}"
    )


if __name__ == "__main__":
    main()
