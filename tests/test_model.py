import unittest
import torch

from mygpt.config import ModelConfig
from mygpt.model import GPT


def tiny_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=17, block_size=8, n_layer=2, n_head=2, n_embd=16, dropout=0.0
    )


class ModelTest(unittest.TestCase):
    def test_forward_shape_and_loss(self) -> None:
        model = GPT(tiny_config())
        tokens = torch.randint(0, 17, (3, 8))
        logits, loss = model(tokens, tokens)
        self.assertEqual(logits.shape, (3, 8, 17))
        self.assertIsNotNone(loss)
        self.assertTrue(torch.isfinite(loss))

    def test_cannot_exceed_context_window(self) -> None:
        model = GPT(tiny_config())
        with self.assertRaisesRegex(ValueError, "exceeds block_size"):
            model(torch.zeros((1, 9), dtype=torch.long))

    def test_weight_tying(self) -> None:
        model = GPT(tiny_config())
        self.assertEqual(
            model.lm_head.weight.data_ptr(), model.token_embedding.weight.data_ptr()
        )
