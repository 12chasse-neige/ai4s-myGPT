"""Reusable language-model, MMLU, and reporting evaluation helpers."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Callable, Mapping, Sequence

import torch
import torch.nn.functional as F

from .checkpoint import load_checkpoint
from .config import ExperimentConfig
from .instruction import format_alpaca_prompt
from .model import GPT
from .tokenizer import BPETokenizer
from .trainer import select_device


CHOICE_LABELS = ("A", "B", "C", "D")


@dataclass(frozen=True)
class LoadedModel:
    """A checkpoint reconstructed for read-only inference."""

    model: GPT
    tokenizer: BPETokenizer
    config: ExperimentConfig
    device: torch.device
    metadata: dict[str, object]


def load_model_for_evaluation(
    checkpoint_path: str | Path, requested_device: str = "auto"
) -> LoadedModel:
    """Load a BPE checkpoint and reconstruct its model on the selected device."""
    path = Path(checkpoint_path)
    device = select_device(requested_device)
    checkpoint = load_checkpoint(path, "cpu")
    config = ExperimentConfig.from_dict(checkpoint["config"])
    tokenizer = BPETokenizer.from_state_dict(checkpoint["tokenizer"])
    model = GPT(config.model)
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    metadata: dict[str, object] = {
        "checkpoint": str(path.resolve()),
        "training_stage": checkpoint.get("training_stage", "pretrain"),
        "step": checkpoint.get("step"),
        "best_validation_loss": checkpoint.get("best_val_loss"),
        "parameters": model.num_parameters(),
        "vocab_size": tokenizer.vocab_size,
        "block_size": config.model.block_size,
        "device": str(device),
    }
    return LoadedModel(model, tokenizer, config, device, metadata)


def _perplexity(average_nll: float) -> float:
    try:
        return math.exp(average_nll)
    except OverflowError:
        return math.inf


@torch.no_grad()
def evaluate_token_ids(
    model: GPT,
    token_ids: Sequence[int],
    device: torch.device,
    *,
    stride: int | None = None,
    batch_size: int = 1,
    progress_every: int = 0,
    progress: Callable[[str], None] = print,
) -> dict[str, int | float]:
    """Compute token-weighted next-token NLL over one complete token sequence.

    Windows overlap so each target after the first is counted exactly once while
    retaining left context from the preceding window.
    """
    if len(token_ids) < 2:
        raise ValueError("language-model evaluation needs at least two tokens")
    block_size = model.config.block_size
    if stride is None:
        stride = max(1, block_size // 2)
    if not 1 <= stride <= block_size:
        raise ValueError(f"stride must be between 1 and block_size ({block_size})")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if progress_every < 0:
        raise ValueError("progress_every cannot be negative")

    model.eval()
    nll_sum = 0.0
    prediction_count = 0
    window_count = 0
    forward_passes = 0
    total_predictions = len(token_ids) - 1
    pending_inputs: list[torch.Tensor] = []
    pending_targets: list[torch.Tensor] = []

    def score_pending_windows() -> tuple[float, int]:
        nonlocal forward_passes
        max_length = max(inputs.numel() for inputs in pending_inputs)
        input_batch = torch.zeros(
            (len(pending_inputs), max_length), dtype=torch.long, device=device
        )
        target_batch = torch.full(
            (len(pending_targets), max_length),
            -100,
            dtype=torch.long,
            device=device,
        )
        for row, (inputs, targets) in enumerate(
            zip(pending_inputs, pending_targets)
        ):
            input_batch[row, : inputs.numel()] = inputs
            target_batch[row, : targets.numel()] = targets
        logits, _ = model(input_batch)
        loss_sum = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            target_batch.reshape(-1),
            ignore_index=-100,
            reduction="sum",
        ).item()
        counted = int((target_batch != -100).sum().item())
        forward_passes += 1
        pending_inputs.clear()
        pending_targets.clear()
        return loss_sum, counted

    for target_start in range(1, len(token_ids), stride):
        target_end = min(target_start + stride, len(token_ids))
        context_start = max(0, target_end - 1 - block_size)
        inputs = torch.tensor(
            token_ids[context_start : target_end - 1],
            dtype=torch.long,
            device=device,
        )
        targets = torch.tensor(
            token_ids[context_start + 1 : target_end],
            dtype=torch.long,
            device=device,
        )
        ignored_targets = max(0, target_start - (context_start + 1))
        if ignored_targets:
            targets[:ignored_targets] = -100
        pending_inputs.append(inputs)
        pending_targets.append(targets)
        window_count += 1
        if len(pending_inputs) == batch_size:
            batch_nll, batch_predictions = score_pending_windows()
            nll_sum += batch_nll
            prediction_count += batch_predictions
        if progress_every and window_count % progress_every == 0:
            progress(
                f"windows={window_count:,} "
                f"predictions={prediction_count:,}/{total_predictions:,}"
            )

    if pending_inputs:
        batch_nll, batch_predictions = score_pending_windows()
        nll_sum += batch_nll
        prediction_count += batch_predictions

    if prediction_count != total_predictions:
        raise AssertionError(
            f"counted {prediction_count} predictions, expected {total_predictions}"
        )
    average_nll = nll_sum / prediction_count
    return {
        "tokens": len(token_ids),
        "predictions": prediction_count,
        "windows": window_count,
        "forward_passes": forward_passes,
        "stride": stride,
        "batch_size": batch_size,
        "average_nll": average_nll,
        "perplexity": _perplexity(average_nll),
        "bits_per_token": average_nll / math.log(2.0),
    }


def validate_mmlu_record(record: object, index: int = 0) -> dict[str, object]:
    """Validate one locally exported MMLU record."""
    if not isinstance(record, Mapping):
        raise ValueError(f"MMLU record {index} must be a JSON object")
    question = record.get("question")
    subject = record.get("subject")
    choices = record.get("choices")
    answer = record.get("answer")
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"MMLU record {index} has no non-empty question")
    if not isinstance(subject, str) or not subject.strip():
        raise ValueError(f"MMLU record {index} has no non-empty subject")
    if (
        not isinstance(choices, list)
        or len(choices) != 4
        or any(not isinstance(choice, str) or not choice.strip() for choice in choices)
    ):
        raise ValueError(f"MMLU record {index} must have four non-empty choices")
    if isinstance(answer, bool) or not isinstance(answer, int) or not 0 <= answer < 4:
        raise ValueError(f"MMLU record {index} answer must be an integer from 0 to 3")
    return {
        "question": question,
        "subject": subject,
        "choices": list(choices),
        "answer": answer,
    }


def load_mmlu_records(path: str | Path) -> list[dict[str, object]]:
    """Load and validate the JSON array produced by ``prepare_data.py --mmlu``."""
    source = Path(path)
    try:
        records = json.loads(source.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"MMLU data not found: {source}. Run `python scripts/prepare_data.py "
            "--mmlu` first."
        ) from error
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid MMLU JSON in {source}: {error}") from error
    if not isinstance(records, list) or not records:
        raise ValueError("MMLU data must be a non-empty JSON list")
    return [validate_mmlu_record(record, index) for index, record in enumerate(records)]


def format_mmlu_example(record: Mapping[str, object], *, include_answer: bool) -> str:
    """Format one question using the conventional MMLU letter-answer prompt."""
    choices = record["choices"]
    assert isinstance(choices, list)
    lines = [str(record["question"]).strip()]
    lines.extend(
        f"{label}. {str(choice).strip()}"
        for label, choice in zip(CHOICE_LABELS, choices)
    )
    answer = "Answer:"
    if include_answer:
        answer_index = int(record["answer"])
        answer += f" {CHOICE_LABELS[answer_index]}"
    lines.append(answer)
    return "\n".join(lines) + "\n\n"


def build_mmlu_prompt(
    record: Mapping[str, object], demonstrations: Sequence[Mapping[str, object]] = ()
) -> str:
    """Build a subject-conditioned zero- or few-shot MMLU prompt."""
    subject = str(record["subject"]).replace("_", " ")
    header = (
        "The following are multiple choice questions (with answers) about "
        f"{subject}.\n\n"
    )
    shots = "".join(
        format_mmlu_example(example, include_answer=True)
        for example in demonstrations
    )
    return header + shots + format_mmlu_example(record, include_answer=False).rstrip()


def fit_mmlu_prompt(
    record: Mapping[str, object],
    demonstrations: Sequence[Mapping[str, object]],
    tokenizer: BPETokenizer,
    block_size: int,
    *,
    prompt_mode: str = "completion",
) -> tuple[str, int, bool]:
    """Drop trailing demonstrations until a prompt fits the model context."""
    if prompt_mode not in {"completion", "instruction"}:
        raise ValueError("prompt_mode must be 'completion' or 'instruction'")

    def format_prompt(examples: Sequence[Mapping[str, object]]) -> str:
        prompt = build_mmlu_prompt(record, examples)
        if prompt_mode == "instruction":
            return format_alpaca_prompt(prompt)
        return prompt

    retained = list(demonstrations)
    prompt = format_prompt(retained)
    while retained and len(tokenizer.encode(prompt)) > block_size:
        retained.pop()
        prompt = format_prompt(retained)
    truncated = len(tokenizer.encode(prompt)) > block_size
    return prompt, len(retained), truncated


@torch.no_grad()
def score_continuations(
    model: GPT,
    prompt_ids: Sequence[int],
    continuation_ids: Sequence[Sequence[int]],
    device: torch.device,
) -> list[float]:
    """Return conditional log-likelihood sums for candidate continuations."""
    if not prompt_ids:
        raise ValueError("the scoring prompt cannot be empty")
    if not continuation_ids or any(not candidate for candidate in continuation_ids):
        raise ValueError("candidate continuations cannot be empty")

    model.eval()
    block_size = model.config.block_size
    prompt_context = list(prompt_ids[-block_size:])
    inputs = torch.tensor(prompt_context, dtype=torch.long, device=device).unsqueeze(0)
    logits, _ = model(inputs)
    first_log_probs = F.log_softmax(logits[0, -1], dim=-1)

    scores: list[float] = []
    for candidate in continuation_ids:
        score = first_log_probs[int(candidate[0])].item()
        context = list(prompt_ids) + [int(candidate[0])]
        for token_id in candidate[1:]:
            inputs = torch.tensor(
                context[-block_size:], dtype=torch.long, device=device
            ).unsqueeze(0)
            logits, _ = model(inputs)
            log_probs = F.log_softmax(logits[0, -1], dim=-1)
            score += log_probs[int(token_id)].item()
            context.append(int(token_id))
        scores.append(score)
    return scores


def _subject_summary(
    predictions: Sequence[Mapping[str, object]],
) -> tuple[dict[str, dict[str, int | float]], float]:
    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for prediction in predictions:
        subject = str(prediction["subject"])
        counts[subject][0] += 1
        counts[subject][1] += int(bool(prediction["correct"]))
    subjects = {
        subject: {
            "total": total,
            "correct": correct,
            "accuracy": correct / total,
        }
        for subject, (total, correct) in sorted(counts.items())
    }
    macro_accuracy = sum(
        float(metrics["accuracy"]) for metrics in subjects.values()
    ) / len(subjects)
    return subjects, macro_accuracy


def evaluate_mmlu_records(
    model: GPT,
    tokenizer: BPETokenizer,
    records: Sequence[Mapping[str, object]],
    device: torch.device,
    *,
    demonstrations_by_subject: Mapping[str, Sequence[Mapping[str, object]]] | None = None,
    shots: int = 0,
    prompt_mode: str = "completion",
    batch_size: int = 1,
    progress_every: int = 0,
    progress: Callable[[str], None] = print,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Evaluate MMLU by choosing the most likely answer label continuation."""
    if not records:
        raise ValueError("MMLU evaluation records cannot be empty")
    if shots < 0:
        raise ValueError("shots cannot be negative")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if progress_every < 0:
        raise ValueError("progress_every cannot be negative")
    demonstrations_by_subject = demonstrations_by_subject or {}
    if prompt_mode not in {"completion", "instruction"}:
        raise ValueError("prompt_mode must be 'completion' or 'instruction'")
    candidate_texts = (
        [f" {label}" for label in CHOICE_LABELS]
        if prompt_mode == "completion"
        else list(CHOICE_LABELS)
    )
    candidate_ids = [tokenizer.encode(candidate) for candidate in candidate_texts]
    if any(not candidate for candidate in candidate_ids):
        raise ValueError("the tokenizer produced an empty MMLU answer label")

    prepared: list[tuple[int, Mapping[str, object], list[int], int, bool]] = []
    for index, record in enumerate(records):
        subject = str(record["subject"])
        available = list(demonstrations_by_subject.get(subject, ()))
        if shots and len(available) < shots:
            raise ValueError(
                f"subject {subject!r} has {len(available)} few-shot examples; "
                f"{shots} requested"
            )
        prompt, used_shots, truncated = fit_mmlu_prompt(
            record,
            available[:shots],
            tokenizer,
            model.config.block_size,
            prompt_mode=prompt_mode,
        )
        prompt_ids = tokenizer.encode(prompt)
        prepared.append((index, record, prompt_ids, used_shots, truncated))

    predictions: list[dict[str, object]] = []
    effective_shots: dict[int, int] = defaultdict(int)
    truncated_prompts = 0
    forward_passes = 0
    single_token_answers = all(len(candidate) == 1 for candidate in candidate_ids)

    for batch_start in range(0, len(prepared), batch_size):
        batch = prepared[batch_start : batch_start + batch_size]
        if single_token_answers:
            contexts = [
                prompt_ids[-model.config.block_size :]
                for _, _, prompt_ids, _, _ in batch
            ]
            lengths = [len(context) for context in contexts]
            max_length = max(lengths)
            input_batch = torch.full(
                (len(batch), max_length),
                tokenizer.pad_id,
                dtype=torch.long,
                device=device,
            )
            for row, context in enumerate(contexts):
                input_batch[row, : len(context)] = torch.tensor(
                    context, dtype=torch.long, device=device
                )
            logits, _ = model(input_batch)
            rows = torch.arange(len(batch), device=device)
            positions = torch.tensor(lengths, device=device) - 1
            next_log_probs = F.log_softmax(logits[rows, positions], dim=-1)
            answer_ids = torch.tensor(
                [candidate[0] for candidate in candidate_ids],
                dtype=torch.long,
                device=device,
            )
            batch_scores = next_log_probs[:, answer_ids].detach().cpu().tolist()
            forward_passes += 1
        else:
            batch_scores = [
                score_continuations(model, prompt_ids, candidate_ids, device)
                for _, _, prompt_ids, _, _ in batch
            ]
            forward_passes += len(batch)

        for (index, record, prompt_ids, used_shots, truncated), scores in zip(
            batch, batch_scores
        ):
            subject = str(record["subject"])
            predicted = max(range(len(scores)), key=scores.__getitem__)
            gold = int(record["answer"])
            prediction = {
                "index": index,
                "subject": subject,
                "question": record["question"],
                "choices": record["choices"],
                "gold": gold,
                "gold_label": CHOICE_LABELS[gold],
                "predicted": predicted,
                "predicted_label": CHOICE_LABELS[predicted],
                "correct": predicted == gold,
                "choice_log_likelihoods": {
                    label: score for label, score in zip(CHOICE_LABELS, scores)
                },
                "prompt_tokens": len(prompt_ids),
                "effective_shots": used_shots,
                "prompt_left_truncated": truncated,
            }
            predictions.append(prediction)
            effective_shots[used_shots] += 1
            truncated_prompts += int(truncated)
        completed = len(predictions)
        if progress_every and (
            completed % progress_every < len(batch) or completed == len(prepared)
        ):
            correct_so_far = sum(int(bool(item["correct"])) for item in predictions)
            progress(
                f"questions={completed:,}/{len(records):,} "
                f"accuracy={correct_so_far / completed:.4f}"
            )

    correct = sum(int(bool(prediction["correct"])) for prediction in predictions)
    subjects, macro_accuracy = _subject_summary(predictions)
    summary: dict[str, object] = {
        "benchmark": "mmlu",
        "scoring": "answer_label_conditional_log_likelihood",
        "prompt_mode": prompt_mode,
        "requested_shots": shots,
        "batch_size": batch_size,
        "forward_passes": forward_passes,
        "answer_continuations": candidate_texts,
        "answer_continuation_token_ids": candidate_ids,
        "total": len(predictions),
        "correct": correct,
        "accuracy": correct / len(predictions),
        "macro_subject_accuracy": macro_accuracy,
        "subject_count": len(subjects),
        "subjects": subjects,
        "effective_shots": {
            str(count): examples for count, examples in sorted(effective_shots.items())
        },
        "prompt_left_truncated": truncated_prompts,
    }
    return summary, predictions


def group_mmlu_by_subject(
    records: Sequence[Mapping[str, object]],
) -> dict[str, list[Mapping[str, object]]]:
    """Group MMLU examples while retaining their source order."""
    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for record in records:
        grouped[str(record["subject"])].append(record)
    return dict(grouped)


def write_text_atomic(path: str | Path, text: str) -> Path:
    """Atomically write a UTF-8 text report."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="\n",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(text)
        os.replace(temporary, destination)
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise
    return destination


def write_json_atomic(path: str | Path, value: object) -> Path:
    """Atomically write a human-readable JSON report."""
    return write_text_atomic(
        path, json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    )
