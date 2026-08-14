#!/usr/bin/env python3
"""Fetch training or evaluation data, or prepare a local UTF-8 corpus."""

from __future__ import annotations

import argparse
import json
from numbers import Integral
from pathlib import Path
import shutil
from tempfile import NamedTemporaryFile
from typing import Callable, Iterable, Mapping, Sequence
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT.parent / "outputs"
DEFAULT_DATASET = "roneneldan/TinyStories"
DEFAULT_MAX_STORIES = 2_000_000
DEFAULT_OUTPUT = OUTPUT_DIR / "data" / "tinystories.txt"
DEFAULT_ALPACA_OUTPUT = OUTPUT_DIR / "data" / "stanford_alpaca.json"
DEFAULT_WIKITEXT_DATASET = "Salesforce/wikitext"
DEFAULT_WIKITEXT_CONFIG = "wikitext-2-raw-v1"
DEFAULT_WIKITEXT_SPLIT = "test"
DEFAULT_WIKITEXT_OUTPUT = (
    OUTPUT_DIR / "data" / f"{DEFAULT_WIKITEXT_CONFIG}-{DEFAULT_WIKITEXT_SPLIT}.txt"
)
DEFAULT_MMLU_DATASET = "cais/mmlu"
DEFAULT_MMLU_CONFIG = "all"
DEFAULT_MMLU_SPLIT = "test"
DEFAULT_MMLU_OUTPUT = (
    OUTPUT_DIR / "data" / f"mmlu-{DEFAULT_MMLU_CONFIG}-{DEFAULT_MMLU_SPLIT}.json"
)
STANFORD_ALPACA_URL = (
    "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/"
    "main/alpaca_data.json"
)
ALPACA_FIELDS = ("instruction", "input", "output")
MMLU_FIELDS = ("question", "subject", "choices", "answer")
STORY_END_TOKEN = "<eos>"


def normalize_text(text: str) -> str:
    """Normalize newlines and remove surrounding whitespace."""
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def load_huggingface_split(
    dataset_name: str,
    split: str,
    config_name: str | None = None,
) -> Iterable[Mapping[str, object]]:
    """Stream one Hugging Face dataset split without loading it into memory."""
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError(
            'Hugging Face dataset preparation requires the "datasets" package; '
            'install the project with `python -m pip install -e ".[test]"`.'
        ) from error

    if config_name is None:
        return load_dataset(dataset_name, split=split, streaming=True)
    return load_dataset(dataset_name, config_name, split=split, streaming=True)


def load_tinystories(
    dataset_name: str = DEFAULT_DATASET,
    split: str = "train",
    max_stories: int = DEFAULT_MAX_STORIES,
) -> Iterable[str]:
    """Stream a bounded number of non-empty stories from Hugging Face."""
    if max_stories <= 0:
        raise ValueError("max_stories must be positive")
    dataset = load_huggingface_split(dataset_name, split)
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


def load_wikitext(
    dataset_name: str = DEFAULT_WIKITEXT_DATASET,
    config_name: str = DEFAULT_WIKITEXT_CONFIG,
    split: str = DEFAULT_WIKITEXT_SPLIT,
) -> Iterable[str]:
    """Stream raw WikiText records for language-model evaluation."""
    dataset = load_huggingface_split(dataset_name, split, config_name)
    for index, record in enumerate(dataset):
        if not isinstance(record, Mapping) or not isinstance(record.get("text"), str):
            raise ValueError(
                f'{dataset_name!r} record {index} must contain a string "text" field'
            )
        yield record["text"].replace("\r\n", "\n").replace("\r", "\n")


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


def validate_mmlu_record(
    record: object, index: int = 0
) -> dict[str, str | int | list[str]]:
    """Validate and normalize one MMLU multiple-choice record."""
    if not isinstance(record, Mapping):
        raise ValueError(f"MMLU record {index} must be a JSON object")
    for field in MMLU_FIELDS:
        if field not in record:
            raise ValueError(f"MMLU record {index} is missing the {field!r} field")

    question = record["question"]
    subject = record["subject"]
    choices = record["choices"]
    answer = record["answer"]
    if not isinstance(question, str) or not normalize_text(question):
        raise ValueError(f"MMLU record {index} must contain a non-empty question")
    if not isinstance(subject, str) or not normalize_text(subject):
        raise ValueError(f"MMLU record {index} must contain a non-empty subject")
    if (
        not isinstance(choices, Sequence)
        or isinstance(choices, (str, bytes))
        or len(choices) != 4
        or any(
            not isinstance(choice, str) or not normalize_text(choice)
            for choice in choices
        )
    ):
        raise ValueError(f"MMLU record {index} must contain exactly four string choices")
    if (
        isinstance(answer, bool)
        or not isinstance(answer, Integral)
        or not 0 <= answer < 4
    ):
        raise ValueError(f"MMLU record {index} answer must be an integer from 0 to 3")

    return {
        "question": question,
        "subject": subject,
        "choices": list(choices),
        "answer": int(answer),
    }


def write_json_records(
    records: Iterable[object],
    output: Path,
    validate: Callable[[object, int], Mapping[str, object]],
) -> tuple[int, int]:
    """Validate and atomically write a streamed JSON array."""
    output.parent.mkdir(parents=True, exist_ok=True)
    record_count = 0
    temporary_output: Path | None = None
    try:
        with NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="\n",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".part",
            delete=False,
        ) as handle:
            temporary_output = Path(handle.name)
            handle.write("[\n")
            for index, record in enumerate(records):
                validated = validate(record, index)
                if record_count:
                    handle.write(",\n")
                json.dump(validated, handle, ensure_ascii=False)
                record_count += 1
            if record_count == 0:
                raise ValueError("input records are empty")
            handle.write("\n]\n")
        byte_count = temporary_output.stat().st_size
        temporary_output.replace(output)
    except BaseException:
        if temporary_output is not None:
            temporary_output.unlink(missing_ok=True)
        raise
    return record_count, byte_count


def fetch_wikitext(
    output: Path = DEFAULT_WIKITEXT_OUTPUT,
    dataset_name: str = DEFAULT_WIKITEXT_DATASET,
    config_name: str = DEFAULT_WIKITEXT_CONFIG,
    split: str = DEFAULT_WIKITEXT_SPLIT,
) -> tuple[int, int]:
    """Fetch and atomically save a raw WikiText split as UTF-8 text."""
    return write_text_lines(load_wikitext(dataset_name, config_name, split), output)


def fetch_mmlu(
    output: Path = DEFAULT_MMLU_OUTPUT,
    dataset_name: str = DEFAULT_MMLU_DATASET,
    config_name: str = DEFAULT_MMLU_CONFIG,
    split: str = DEFAULT_MMLU_SPLIT,
) -> tuple[int, int]:
    """Fetch and atomically save a structured MMLU split as JSON."""
    records = load_huggingface_split(dataset_name, split, config_name)
    return write_json_records(records, output, validate_mmlu_record)


def write_stories(stories: Iterable[str], output: Path) -> tuple[int, int]:
    """Write stories with an explicit BPE end-of-sequence marker."""
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
                handle.write(normalized)
                handle.write(f"\n{STORY_END_TOKEN}\n")
                story_count += 1
                character_count += len(normalized) + len(STORY_END_TOKEN) + 2
            if story_count == 0:
                raise ValueError("input corpus is empty")
        temporary_output.replace(output)
    except BaseException:
        if temporary_output is not None:
            temporary_output.unlink(missing_ok=True)
        raise
    return story_count, character_count


def write_text_lines(lines: Iterable[str], output: Path) -> tuple[int, int]:
    """Atomically write line records while preserving empty WikiText rows."""
    output.parent.mkdir(parents=True, exist_ok=True)
    line_count = 0
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
            for line in lines:
                normalized = line.replace("\r\n", "\n").replace("\r", "\n")
                handle.write(normalized)
                character_count += len(normalized)
                if not normalized.endswith("\n"):
                    handle.write("\n")
                    character_count += 1
                line_count += 1
            if line_count == 0:
                raise ValueError("input text records are empty")
        temporary_output.replace(output)
    except BaseException:
        if temporary_output is not None:
            temporary_output.unlink(missing_ok=True)
        raise
    return line_count, character_count


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
        action="store_const",
        const="tinystories",
        dest="source",
        default=argparse.SUPPRESS,
        help="fetch TinyStories (also the default when no source is selected)",
    )
    sources.add_argument(
        "--stanford-alpaca",
        action="store_const",
        const="stanford-alpaca",
        dest="source",
        default=argparse.SUPPRESS,
        help="fetch Stanford Alpaca JSON for later instruction tuning",
    )
    sources.add_argument(
        "--wikitext",
        action="store_const",
        const="wikitext",
        dest="source",
        default=argparse.SUPPRESS,
        help="fetch raw WikiText-2 for language-model evaluation",
    )
    sources.add_argument(
        "--mmlu",
        action="store_const",
        const="mmlu",
        dest="source",
        default=argparse.SUPPRESS,
        help="fetch structured MMLU questions for reasoning evaluation",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="destination file (chosen from the selected data source by default)",
    )
    parser.add_argument(
        "--dataset",
        help="override the selected source's default Hugging Face dataset ID",
    )
    parser.add_argument(
        "--config",
        help="override the WikiText or MMLU Hugging Face dataset configuration",
    )
    parser.add_argument(
        "--split",
        help="override the source split (TinyStories: train; WikiText/MMLU: test)",
    )
    parser.add_argument(
        "--max-stories",
        type=int,
        default=DEFAULT_MAX_STORIES,
        help="maximum stories to stream",
    )
    args = parser.parse_args()
    if not hasattr(args, "source"):
        args.source = "tinystories"
    return args


def main() -> None:
    args = parse_args()
    if args.input:
        stories = [args.input.read_text(encoding="utf-8")]
        output = args.output or DEFAULT_OUTPUT
        story_count, character_count = write_stories(stories, output)
        print(
            f"wrote {story_count:,} text records ({character_count:,} characters) "
            f"from {args.input} to {output}"
        )
        return

    if args.source == "stanford-alpaca":
        output = args.output or DEFAULT_ALPACA_OUTPUT
        record_count, byte_count = fetch_stanford_alpaca(output)
        print(
            f"wrote {record_count:,} instruction records ({byte_count:,} bytes) "
            f"from {STANFORD_ALPACA_URL} to {output}"
        )
        return

    if args.source == "wikitext":
        dataset_name = args.dataset or DEFAULT_WIKITEXT_DATASET
        config_name = args.config or DEFAULT_WIKITEXT_CONFIG
        split = args.split or DEFAULT_WIKITEXT_SPLIT
        default_output = OUTPUT_DIR / "data" / f"{DEFAULT_WIKITEXT_CONFIG}-{split}.txt"
        output = args.output or default_output
        record_count, character_count = fetch_wikitext(
            output, dataset_name, config_name, split
        )
        print(
            f"wrote {record_count:,} text records ({character_count:,} characters) "
            f"from {dataset_name}/{config_name}:{split} to {output}"
        )
        return

    if args.source == "mmlu":
        dataset_name = args.dataset or DEFAULT_MMLU_DATASET
        config_name = args.config or DEFAULT_MMLU_CONFIG
        split = args.split or DEFAULT_MMLU_SPLIT
        default_output = OUTPUT_DIR / "data" / f"mmlu-{DEFAULT_MMLU_CONFIG}-{split}.json"
        output = args.output or default_output
        record_count, byte_count = fetch_mmlu(output, dataset_name, config_name, split)
        print(
            f"wrote {record_count:,} MMLU records ({byte_count:,} bytes) "
            f"from {dataset_name}/{config_name}:{split} to {output}"
        )
        return

    dataset_name = args.dataset or DEFAULT_DATASET
    split = args.split or "train"
    stories = load_tinystories(dataset_name, split, args.max_stories)
    output = args.output or DEFAULT_OUTPUT
    story_count, character_count = write_stories(stories, output)
    print(
        f"wrote {story_count:,} stories ({character_count:,} characters) "
        f"from {dataset_name}:{split} to {output}"
    )


if __name__ == "__main__":
    main()
