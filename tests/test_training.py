from pathlib import Path
import tempfile
import unittest

import torch

from mygpt.checkpoint import load_checkpoint
from mygpt.config import DataConfig, ExperimentConfig, ModelConfig, TrainingConfig
from mygpt.data import build_dataloaders
from mygpt.model import GPT
from mygpt.tokenizer import BPETokenizer
from mygpt.trainer import train


class TrainingTest(unittest.TestCase):
    def test_one_step_training_and_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            text = "a small training example.\n<eos>\n" * 30
            tokenizer = BPETokenizer.train_from_iterator([text])
            config = ExperimentConfig(
                model=ModelConfig(
                    vocab_size=tokenizer.vocab_size,
                    block_size=8,
                    n_layer=1,
                    n_head=1,
                    n_embd=8,
                    dropout=0.0,
                ),
                data=DataConfig(train_fraction=0.8, batch_size=4),
                training=TrainingConfig(
                    max_steps=1,
                    warmup_steps=0,
                    eval_interval=1,
                    eval_batches=1,
                    log_interval=1,
                    output_dir=str(tmp_path),
                    device="cpu",
                ),
            )
            train_loader, val_loader = build_dataloaders(
                text,
                tokenizer,
                config.model.block_size,
                config.data,
                config.training.seed,
            )
            history = train(
                GPT(config.model), train_loader, val_loader, tokenizer, config
            )
            checkpoint = load_checkpoint(tmp_path / "last.pt")
            self.assertEqual(len(history), 1)
            self.assertEqual(checkpoint["step"], 1)
            self.assertTrue(torch.isfinite(torch.tensor(history[0]["val_loss"])))

    def test_stage_metadata_and_optimizer_resume_are_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            text = "instruction tuning example.\n<eos>\n" * 30
            tokenizer = BPETokenizer.train_from_iterator([text])
            config = ExperimentConfig(
                model=ModelConfig(
                    vocab_size=tokenizer.vocab_size,
                    block_size=8,
                    n_layer=1,
                    n_head=1,
                    n_embd=8,
                    dropout=0.0,
                ),
                data=DataConfig(train_fraction=0.8, batch_size=4),
                training=TrainingConfig(
                    max_steps=1,
                    warmup_steps=0,
                    eval_interval=1,
                    eval_batches=1,
                    log_interval=1,
                    output_dir=str(tmp_path),
                    device="cpu",
                ),
            )
            train_loader, val_loader = build_dataloaders(
                text,
                tokenizer,
                config.model.block_size,
                config.data,
                config.training.seed,
            )
            metadata = {"training_stage": "sft", "source_step": 10}
            train(
                GPT(config.model),
                train_loader,
                val_loader,
                tokenizer,
                config,
                checkpoint_metadata=metadata,
            )
            first = load_checkpoint(tmp_path / "last.pt")
            config.training.max_steps = 2
            resumed_model = GPT(config.model)
            resumed_model.load_state_dict(first["model"])
            train(
                resumed_model,
                train_loader,
                val_loader,
                tokenizer,
                config,
                start_step=1,
                optimizer_state=first["optimizer"],
                best_val_loss=first["best_val_loss"],
                history=first["history"],
                checkpoint_metadata=metadata,
            )
            resumed = load_checkpoint(tmp_path / "last.pt")
            self.assertEqual(resumed["step"], 2)
            self.assertEqual(len(resumed["history"]), 2)
            self.assertEqual(resumed["training_stage"], "sft")
            self.assertEqual(resumed["source_step"], 10)
