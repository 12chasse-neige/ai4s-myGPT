#!/usr/bin/env python3
"""Generate reproducible qualitative completion or instruction-following samples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

import torch

from mygpt.evaluation import (
    load_model_for_evaluation,
    write_json_atomic,
    write_text_atomic,
)
from mygpt.generation import generate_tokens
from mygpt.instruction import fit_alpaca_prompt


DEFAULT_OUTPUT = Path("outputs/evaluation/generation_samples.md")
COMPLETION_PROMPTS = [
    {"id": "story", "prompt": "Once upon a time, a small robot found a seed"},
    {
        "id": "explanation",
        "prompt": "The reason the sky looks blue during the day is",
    },
    {
        "id": "dialogue",
        "prompt": (
            'Maya said, "I am nervous about my presentation tomorrow."\n'
            'Her friend replied, "'
        ),
    },
]
INSTRUCTION_PROMPTS = [
    {
        "id": "story",
        "instruction": (
            "Write a short children's story about a robot that learns to garden."
        ),
        "input": "",
    },
    {
        "id": "explanation",
        "instruction": "Explain why the sky appears blue to a curious ten-year-old.",
        "input": "",
    },
    {
        "id": "conversation",
        "instruction": "Respond helpfully and politely to the user.",
        "input": (
            "User: I am nervous about my first presentation tomorrow. "
            "What should I do?"
        ),
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--checkpoint", type=Path, required=True, help="BPE checkpoint to sample"
    )
    parser.add_argument(
        "--mode",
        choices=("auto", "completion", "instruction"),
        default="auto",
        help="prompt style; auto selects instruction mode only for SFT checkpoints",
    )
    parser.add_argument(
        "--prompts",
        type=Path,
        help="optional JSON prompt list; built-in story/explanation/dialogue prompts otherwise",
    )
    parser.add_argument(
        "--samples-per-prompt",
        type=int,
        default=1,
        help="independent samples generated for every prompt",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=160, help="maximum generated tokens"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.8, help="sampling temperature"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=40,
        help="sample from the top-k candidates; zero disables top-k filtering",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="base seed for reproducible samples"
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "mps", "cuda"),
        default="auto",
        help="device used for generation",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT, help="Markdown report path"
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        help="JSON report path; defaults to the Markdown path with a .json suffix",
    )
    return parser.parse_args()


def load_prompts(path: Path | None, mode: str) -> list[dict[str, str]]:
    """Load and validate custom prompts, or return the built-in qualitative set."""
    if path is None:
        defaults = COMPLETION_PROMPTS if mode == "completion" else INSTRUCTION_PROMPTS
        return [dict(prompt) for prompt in defaults]
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid prompts JSON in {path}: {error}") from error
    if not isinstance(value, list) or not value:
        raise ValueError("prompts JSON must be a non-empty list")

    prompts: list[dict[str, str]] = []
    for index, item in enumerate(value):
        if mode == "completion" and isinstance(item, str) and item:
            prompts.append({"id": f"prompt-{index + 1}", "prompt": item})
            continue
        if not isinstance(item, Mapping):
            raise ValueError(f"prompt {index} must be a JSON object")
        identifier = item.get("id", f"prompt-{index + 1}")
        if not isinstance(identifier, str) or not identifier:
            raise ValueError(f"prompt {index} has an invalid id")
        required = "prompt" if mode == "completion" else "instruction"
        content = item.get(required)
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"prompt {index} requires non-empty string {required!r}")
        prompt = {"id": identifier, required: content}
        if mode == "instruction":
            input_text = item.get("input", "")
            if not isinstance(input_text, str):
                raise ValueError(f"prompt {index} input must be a string")
            prompt["input"] = input_text
        prompts.append(prompt)
    return prompts


def _generate_from_tokens(
    loaded,
    prompt_tokens: list[int],
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    seed: int,
) -> str:
    torch.manual_seed(seed)
    tokens = torch.tensor(
        [prompt_tokens], dtype=torch.long, device=loaded.device
    )
    generated = generate_tokens(
        loaded.model,
        tokens,
        max_new_tokens,
        temperature,
        top_k,
        loaded.tokenizer.eos_id,
    )
    response_ids = generated[0, len(prompt_tokens) :].tolist()
    return loaded.tokenizer.decode(response_ids, skip_special_tokens=True).strip()


def generate_samples(
    loaded,
    prompts: list[dict[str, str]],
    mode: str,
    *,
    samples_per_prompt: int,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    seed: int,
) -> list[dict[str, object]]:
    """Generate every requested qualitative sample without reloading the model."""
    results: list[dict[str, object]] = []
    sample_index = 0
    for prompt in prompts:
        if mode == "completion":
            display_prompt = prompt["prompt"]
            model_prompt = display_prompt
        else:
            display_prompt = prompt["instruction"]
            fitted = fit_alpaca_prompt(
                prompt["instruction"],
                prompt.get("input", ""),
                loaded.config.model.block_size,
                tokenizer=loaded.tokenizer,
                reserve_tokens=0,
            )
            if fitted is None:
                raise ValueError(
                    f"instruction prompt {prompt['id']!r} does not fit the context"
                )
            model_prompt, input_truncated = fitted

        prompt_tokens = loaded.tokenizer.encode(model_prompt)
        for repeat in range(samples_per_prompt):
            sample_seed = seed + sample_index
            output = _generate_from_tokens(
                loaded,
                prompt_tokens,
                max_new_tokens,
                temperature,
                top_k,
                sample_seed,
            )
            result: dict[str, object] = {
                "id": prompt["id"],
                "sample": repeat + 1,
                "seed": sample_seed,
                "prompt": display_prompt,
                "input": prompt.get("input", ""),
                "prompt_tokens": len(prompt_tokens),
                "output": output,
            }
            if mode == "instruction":
                result["input_truncated"] = input_truncated
            results.append(result)
            sample_index += 1
            print(
                f"generated prompt={prompt['id']} sample={repeat + 1} "
                f"seed={sample_seed}",
                flush=True,
            )
    return results


def _fenced(text: str) -> str:
    longest = 0
    index = 0
    while index < len(text):
        if text[index] != "`":
            index += 1
            continue
        end = index
        while end < len(text) and text[end] == "`":
            end += 1
        longest = max(longest, end - index)
        index = end
    fence = "`" * max(3, longest + 1)
    return f"{fence}text\n{text}\n{fence}"


def render_markdown(report: Mapping[str, object]) -> str:
    """Render samples as a paper-friendly qualitative appendix."""
    model = report["model"]
    assert isinstance(model, Mapping)
    lines = [
        "# Qualitative generation samples",
        "",
        f"- Checkpoint: `{model['checkpoint']}`",
        f"- Training stage: `{model['training_stage']}`",
        f"- Mode: `{report['mode']}`",
        f"- Temperature: `{report['temperature']}`",
        f"- Top-k: `{report['top_k']}`",
        f"- Maximum new tokens: `{report['max_new_tokens']}`",
        "",
        (
            "These examples are qualitative observations, not a quantitative "
            "benchmark. Report the sampling settings alongside any excerpts."
        ),
        "",
    ]
    samples = report["samples"]
    assert isinstance(samples, list)
    for item in samples:
        assert isinstance(item, Mapping)
        lines.extend(
            [
                f"## {item['id']} (sample {item['sample']}, seed {item['seed']})",
                "",
                "Prompt:",
                "",
                _fenced(str(item["prompt"])),
                "",
            ]
        )
        if item.get("input"):
            lines.extend(["Input:", "", _fenced(str(item["input"])), ""])
        lines.extend(["Model output:", "", _fenced(str(item["output"])), ""])
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = parse_args()
    if args.samples_per_prompt <= 0:
        raise ValueError("samples_per_prompt must be positive")
    if args.max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    if args.temperature <= 0:
        raise ValueError("temperature must be positive")
    if args.top_k < 0:
        raise ValueError("top_k cannot be negative")

    loaded = load_model_for_evaluation(args.checkpoint, args.device)
    mode = args.mode
    if mode == "auto":
        mode = (
            "instruction"
            if loaded.metadata["training_stage"] == "sft"
            else "completion"
        )
    prompts = load_prompts(args.prompts, mode)
    top_k = None if args.top_k == 0 else args.top_k
    print(
        f"device={loaded.device} parameters={loaded.model.num_parameters():,} "
        f"mode={mode} prompts={len(prompts)}",
        flush=True,
    )
    samples = generate_samples(
        loaded,
        prompts,
        mode,
        samples_per_prompt=args.samples_per_prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=top_k,
        seed=args.seed,
    )
    report = {
        "evaluation": "qualitative_generation",
        "mode": mode,
        "temperature": args.temperature,
        "top_k": top_k,
        "max_new_tokens": args.max_new_tokens,
        "model": loaded.metadata,
        "samples": samples,
    }
    json_output = args.json_output or args.output.with_suffix(".json")
    write_text_atomic(args.output, render_markdown(report))
    write_json_atomic(json_output, report)
    print(f"wrote {args.output}")
    print(f"wrote {json_output}")


if __name__ == "__main__":
    main()
