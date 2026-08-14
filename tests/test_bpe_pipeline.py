import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

import yaml

from mygpt.checkpoint import load_checkpoint


ROOT = Path(__file__).parents[1]


class BPEPipelineTest(unittest.TestCase):
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

    def test_fresh_pretraining_and_sft_use_saved_bpe_tokenizer(self) -> None:
        with TemporaryDirectory() as directory:
            temporary = Path(directory)
            corpus = temporary / "stories.txt"
            tokenizer_path = temporary / "tokenizer.json"
            pretrain_output = temporary / "pretrain"
            corpus.write_text(
                (
                    "Once upon a time, a little cat played with a red ball.\n"
                    "The cat was happy and went home.\n<eos>\n"
                )
                * 100,
                encoding="utf-8",
            )
            pretrain_config = temporary / "pretrain.yaml"
            pretrain_config.write_text(
                yaml.safe_dump(
                    {
                        "model": {
                            "block_size": 256,
                            "n_layer": 1,
                            "n_head": 1,
                            "n_embd": 8,
                            "dropout": 0.0,
                            "bias": True,
                        },
                        "data": {
                            "path": str(corpus),
                            "train_fraction": 0.8,
                            "batch_size": 2,
                            "num_workers": 0,
                        },
                        "tokenizer": {
                            "type": "bpe",
                            "path": str(tokenizer_path),
                            "vocab_size": 512,
                            "min_frequency": 1,
                        },
                        "training": {
                            "max_steps": 1,
                            "warmup_steps": 0,
                            "eval_interval": 1,
                            "eval_batches": 1,
                            "log_interval": 1,
                            "output_dir": str(pretrain_output),
                            "device": "cpu",
                        },
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )

            result = self.run_script(
                "scripts/train.py", "--config", str(pretrain_config)
            )
            self.assertIn("trained BPE tokenizer", result.stdout)
            self.assertTrue(tokenizer_path.is_file())
            pretrained = load_checkpoint(pretrain_output / "best.pt")
            self.assertEqual(pretrained["tokenizer"]["type"], "bpe")
            pretrained_vocab_size = pretrained["config"]["model"]["vocab_size"]

            records = [
                {
                    "instruction": f"Say hello number {index}.",
                    "input": "",
                    "output": f"Hello {index}.",
                }
                for index in range(8)
            ]
            alpaca_path = temporary / "alpaca.json"
            alpaca_path.write_text(json.dumps(records), encoding="utf-8")
            sft_output = temporary / "sft"
            sft_config = temporary / "sft.yaml"
            sft_config.write_text(
                yaml.safe_dump(
                    {
                        "pretrained_checkpoint": str(pretrain_output / "best.pt"),
                        "data": {
                            "path": str(alpaca_path),
                            "train_fraction": 0.75,
                            "batch_size": 2,
                            "num_workers": 0,
                            "max_seq_length": 256,
                            "max_records": 8,
                        },
                        "training": {
                            "max_steps": 1,
                            "warmup_steps": 0,
                            "eval_interval": 1,
                            "eval_batches": 1,
                            "log_interval": 1,
                            "output_dir": str(sft_output),
                            "device": "cpu",
                        },
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )

            self.run_script(
                "scripts/instruction_tune.py", "--config", str(sft_config)
            )
            tuned = load_checkpoint(sft_output / "best.pt")
            self.assertEqual(tuned["tokenizer"]["type"], "bpe")
            self.assertEqual(tuned["config"]["tokenizer"]["type"], "bpe")
            self.assertEqual(tuned["added_tokens"], [])
            self.assertEqual(
                tuned["config"]["model"]["vocab_size"], pretrained_vocab_size
            )


if __name__ == "__main__":
    unittest.main()
