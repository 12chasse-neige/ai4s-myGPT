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

    def test_resize_token_embeddings_preserves_pretrained_rows(self) -> None:
        model = GPT(tiny_config())
        old_weight = model.token_embedding.weight.detach().clone()
        model.resize_token_embeddings(23)
        self.assertEqual(model.config.vocab_size, 23)
        self.assertTrue(torch.equal(model.token_embedding.weight[:17], old_weight))
        self.assertEqual(
            model.lm_head.weight.data_ptr(), model.token_embedding.weight.data_ptr()
        )

    def test_response_mask_is_ignored_by_loss(self) -> None:
        model = GPT(tiny_config())
        tokens = torch.randint(0, 17, (1, 8))
        targets = torch.full_like(tokens, -100)
        targets[0, -1] = 3
        _, loss = model(tokens, targets)
        assert loss is not None
        self.assertTrue(torch.isfinite(loss))
