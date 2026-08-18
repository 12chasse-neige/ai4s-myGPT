#!/usr/bin/env python3
"""Evaluate a myGPT checkpoint on MMLU multiple-choice questions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mygpt.evaluation import (
    evaluate_mmlu_records,
    group_mmlu_by_subject,
    load_mmlu_records,
    load_model_for_evaluation,
    write_json_atomic,
    write_text_atomic,
)


DEFAULT_DATA = Path("outputs/data/mmlu-all-test.json")
DEFAULT_OUTPUT = Path("outputs/evaluation/mmlu.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--checkpoint", type=Path, required=True, help="BPE checkpoint to evaluate"
    )
    parser.add_argument(
        "--data", type=Path, default=DEFAULT_DATA, help="MMLU test JSON export"
    )
    parser.add_argument(
        "--few-shot-data",
        type=Path,
        help="MMLU dev JSON export used for subject-matched demonstrations",
    )
    parser.add_argument(
        "--shots", type=int, default=0, help="few-shot demonstrations per subject"
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        help="evaluate only these underscore-separated MMLU subject names",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="evaluate only the first N selected questions for a smoke test",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "mps", "cuda"),
        default="auto",
        help="device used for evaluation",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="right-padded prompts evaluated per model forward pass",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="print progress after this many questions; zero disables progress",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT, help="summary JSON report path"
    )
    parser.add_argument(
        "--predictions-output",
        type=Path,
        help="optional JSONL path for per-question predictions and scores",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.shots < 0:
        raise ValueError("shots cannot be negative")
    if args.shots and args.few_shot_data is None:
        raise ValueError(
            "--shots requires --few-shot-data. Fetch it with `python "
            "scripts/prepare_data.py --mmlu --split dev`."
        )
    if args.limit is not None and args.limit <= 0:
        raise ValueError("limit must be positive")

    records = load_mmlu_records(args.data)
    selected_subjects = set(args.subjects or ())
    if selected_subjects:
        available_subjects = {str(record["subject"]) for record in records}
        missing = selected_subjects - available_subjects
        if missing:
            raise ValueError(f"unknown MMLU subjects: {sorted(missing)}")
        records = [
            record for record in records if record["subject"] in selected_subjects
        ]
    selected_total = len(records)
    if args.limit is not None:
        records = records[: args.limit]
    if not records:
        raise ValueError("no MMLU questions were selected")

    demonstrations = {}
    if args.few_shot_data is not None:
        demonstrations = group_mmlu_by_subject(
            load_mmlu_records(args.few_shot_data)
        )

    loaded = load_model_for_evaluation(args.checkpoint, args.device)
    print(
        f"device={loaded.device} parameters={loaded.model.num_parameters():,} "
        f"questions={len(records):,}/{selected_total:,} shots={args.shots}",
        flush=True,
    )
    summary, predictions = evaluate_mmlu_records(
        loaded.model,
        loaded.tokenizer,
        records,
        loaded.device,
        demonstrations_by_subject=demonstrations,
        shots=args.shots,
        batch_size=args.batch_size,
        progress_every=args.progress_every,
    )
    report = {
        **summary,
        "dataset_path": str(args.data.resolve()),
        "few_shot_dataset_path": (
            str(args.few_shot_data.resolve()) if args.few_shot_data else None
        ),
        "evaluation_scope": "full_selection" if args.limit is None else "prefix_smoke_test",
        "selected_questions_before_limit": selected_total,
        "selected_subjects": sorted(selected_subjects) if selected_subjects else "all",
        "model": loaded.metadata,
    }
    write_json_atomic(args.output, report)
    if args.predictions_output is not None:
        jsonl = "".join(
            json.dumps(prediction, ensure_ascii=False) + "\n"
            for prediction in predictions
        )
        write_text_atomic(args.predictions_output, jsonl)
        print(f"wrote {args.predictions_output}")
    print(
        f"accuracy={float(summary['accuracy']):.4f} "
        f"macro_subject_accuracy={float(summary['macro_subject_accuracy']):.4f} "
        f"correct={summary['correct']}/{summary['total']}"
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
