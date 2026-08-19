import json
import math
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

import torch

from mygpt.checkpoint import save_checkpoint
from mygpt.config import ExperimentConfig, ModelConfig
from mygpt.evaluation import (
    build_mmlu_prompt,
    evaluate_mmlu_records,
    evaluate_token_ids,
    fit_mmlu_prompt,
    load_mmlu_records,
)
from mygpt.model import GPT
from mygpt.tokenizer import BPETokenizer


ROOT = Path(__file__).parents[1]


class UniformModel:
    def __init__(self, vocab_size: int = 7, block_size: int = 4) -> None:
        self.config = type(
            "Config", (), {"vocab_size": vocab_size, "block_size": block_size}
        )()

    def eval(self) -> None:
        pass

    def __call__(self, tokens: torch.Tensor):
        return (
            torch.zeros(
                tokens.size(0),
                tokens.size(1),
                self.config.vocab_size,
                device=tokens.device,
            ),
            None,
        )


class PreferredTokenModel(UniformModel):
    def __init__(self, vocab_size: int, preferred_token: int) -> None:
        super().__init__(vocab_size=vocab_size, block_size=128)
        self.preferred_token = preferred_token

    def __call__(self, tokens: torch.Tensor):
        logits, _ = super().__call__(tokens)
        logits[:, :, self.preferred_token] = 10.0
        return logits, None


def mmlu_record(answer: int, subject: str = "test_subject") -> dict[str, object]:
    return {
        "question": "Which option should be selected?",
        "subject": subject,
        "choices": ["alpha", "beta", "gamma", "delta"],
        "answer": answer,
    }


class EvaluationUnitTest(unittest.TestCase):
    def test_language_model_metrics_count_every_target_once(self) -> None:
        model = UniformModel(vocab_size=7, block_size=4)
        metrics = evaluate_token_ids(
            model,
            [0, 1, 2, 3, 4, 5, 6],
            torch.device("cpu"),
            stride=2,
            batch_size=3,
        )
        self.assertEqual(metrics["predictions"], 6)
        self.assertAlmostEqual(float(metrics["average_nll"]), math.log(7), places=6)
        self.assertAlmostEqual(float(metrics["perplexity"]), 7.0, places=5)

    def test_mmlu_prompt_and_likelihood_accuracy(self) -> None:
        tokenizer = BPETokenizer.train_from_iterator(
            [
                build_mmlu_prompt(mmlu_record(0)),
                " Answer: A Answer: B Answer: C Answer: D",
            ],
            vocab_size=512,
            min_frequency=1,
        )
        candidates = [tokenizer.encode(f" {label}") for label in "ABCD"]
        self.assertTrue(all(len(candidate) == 1 for candidate in candidates))
        model = PreferredTokenModel(tokenizer.vocab_size, candidates[0][0])
        summary, predictions = evaluate_mmlu_records(
            model,
            tokenizer,
            [mmlu_record(0), mmlu_record(1)],
            torch.device("cpu"),
            batch_size=2,
        )
        self.assertEqual([item["predicted"] for item in predictions], [0, 0])
        self.assertEqual(summary["correct"], 1)
        self.assertEqual(summary["accuracy"], 0.5)
        self.assertEqual(summary["subject_count"], 1)

    def test_mmlu_loader_rejects_malformed_records(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "mmlu.json"
            path.write_text(
                json.dumps([{**mmlu_record(0), "choices": ["only one"]}]),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "four non-empty choices"):
                load_mmlu_records(path)

    def test_sft_mmlu_prompt_uses_alpaca_template_and_unspaced_labels(self) -> None:
        tokenizer = BPETokenizer.train_from_iterator(
            [
                build_mmlu_prompt(mmlu_record(0)),
                "Below is an instruction. ### Instruction: ### Response: A B C D",
            ],
            vocab_size=512,
            min_frequency=1,
        )
        prompt, shots, truncated = fit_mmlu_prompt(
            mmlu_record(0),
            [],
            tokenizer,
            256,
            prompt_mode="instruction",
        )
        self.assertIn("### Instruction:", prompt)
        self.assertTrue(prompt.endswith("### Response:\n"))
        self.assertEqual(shots, 0)
        self.assertFalse(truncated)

        candidates = [tokenizer.encode(label) for label in "ABCD"]
        model = PreferredTokenModel(tokenizer.vocab_size, candidates[0][0])
        summary, _ = evaluate_mmlu_records(
            model,
            tokenizer,
            [mmlu_record(0)],
            torch.device("cpu"),
            prompt_mode="instruction",
        )
        self.assertEqual(summary["prompt_mode"], "instruction")
        self.assertEqual(summary["answer_continuations"], list("ABCD"))


class EvaluationCliTest(unittest.TestCase):
    def run_script(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(ROOT / "src")
        try:
            return subprocess.run(
                [sys.executable, *arguments],
                cwd=ROOT,
                env=environment,
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as error:
            self.fail(
                f"command failed: {' '.join(arguments)}\n"
                f"stdout:\n{error.stdout}\nstderr:\n{error.stderr}"
            )

    def test_all_evaluation_entrypoints_write_reports(self) -> None:
        with TemporaryDirectory() as directory:
            temporary = Path(directory)
            training_text = (
                "Once upon a time a robot planted a seed. Answer: A B C D.\n<eos>\n"
                * 20
            )
            tokenizer = BPETokenizer.train_from_iterator(
                [training_text], vocab_size=512, min_frequency=1
            )
            config = ExperimentConfig(
                model=ModelConfig(
                    vocab_size=tokenizer.vocab_size,
                    block_size=32,
                    n_layer=1,
                    n_head=1,
                    n_embd=8,
                    dropout=0.0,
                )
            )
            model = GPT(config.model)
            checkpoint = temporary / "checkpoint.pt"
            save_checkpoint(
                checkpoint,
                {
                    "model": model.state_dict(),
                    "config": config.to_dict(),
                    "tokenizer": tokenizer.state_dict(),
                    "step": 0,
                    "best_val_loss": None,
                },
            )

            wikitext = temporary / "wikitext.txt"
            wikitext.write_text(training_text, encoding="utf-8")
            wikitext_output = temporary / "wikitext.json"
            self.run_script(
                "scripts/evaluate_wikitext.py",
                "--checkpoint",
                str(checkpoint),
                "--data",
                str(wikitext),
                "--max-tokens",
                "20",
                "--stride",
                "8",
                "--progress-every",
                "0",
                "--device",
                "cpu",
                "--output",
                str(wikitext_output),
            )
            self.assertEqual(
                json.loads(wikitext_output.read_text())["metrics"]["predictions"],
                19,
            )

            mmlu = temporary / "mmlu.json"
            mmlu.write_text(json.dumps([mmlu_record(0)]), encoding="utf-8")
            mmlu_output = temporary / "mmlu-report.json"
            self.run_script(
                "scripts/evaluate_mmlu.py",
                "--checkpoint",
                str(checkpoint),
                "--data",
                str(mmlu),
                "--limit",
                "1",
                "--progress-every",
                "0",
                "--device",
                "cpu",
                "--output",
                str(mmlu_output),
            )
            self.assertEqual(json.loads(mmlu_output.read_text())["total"], 1)

            prompts = temporary / "prompts.json"
            prompts.write_text(json.dumps(["Once upon a time"]), encoding="utf-8")
            samples_output = temporary / "samples.md"
            self.run_script(
                "scripts/generate_samples.py",
                "--checkpoint",
                str(checkpoint),
                "--prompts",
                str(prompts),
                "--max-new-tokens",
                "2",
                "--device",
                "cpu",
                "--output",
                str(samples_output),
            )
            self.assertIn("Model output:", samples_output.read_text())
            self.assertTrue(samples_output.with_suffix(".json").is_file())


if __name__ == "__main__":
    unittest.main()
