import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from mygpt.data import CharacterTokenizer, TokenDataset, load_text


class DataTest(unittest.TestCase):
    def test_tokenizer_round_trip(self) -> None:
        text = "abc cab"
        tokenizer = CharacterTokenizer.from_text(text)
        self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)
        self.assertEqual(tokenizer.decode(tokenizer.encode("?")), "�")

    def test_dataset_is_shifted_by_one_token(self) -> None:
        dataset = TokenDataset([1, 2, 3, 4, 5], block_size=3)
        inputs, targets = dataset[0]
        self.assertTrue(torch.equal(inputs, torch.tensor([1, 2, 3])))
        self.assertTrue(torch.equal(targets, torch.tensor([2, 3, 4])))

    def test_missing_default_corpus_has_preparation_hint(self) -> None:
        with TemporaryDirectory() as directory:
            missing = Path(directory) / "tinystories.txt"
            with self.assertRaisesRegex(FileNotFoundError, "prepare_data.py"):
                load_text(missing)
