import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch

from mygpt.data import TokenDataset, load_text, prepare_token_cache
from mygpt.tokenizer import BPETokenizer


class DataTest(unittest.TestCase):
    def test_bpe_tokenizer_round_trip_and_special_tokens(self) -> None:
        text = "abc cab élan <eos>"
        tokenizer = BPETokenizer.train_from_iterator([text])
        self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)
        self.assertIn(tokenizer.eos_id, tokenizer.encode(text))
        restored = BPETokenizer.from_state_dict(tokenizer.state_dict())
        self.assertEqual(restored.encode(text), tokenizer.encode(text))
        self.assertEqual(
            restored.decode(restored.encode(text), skip_special_tokens=True),
            "abc cab élan ",
        )

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

    def test_chunked_token_cache_matches_full_bpe_encoding(self) -> None:
        text = (
            "Élan found a red ball.\n<eos>\n\n"
            "Once upon a time, a cat went home.\n<eos>\n"
        ) * 20
        with TemporaryDirectory() as directory:
            corpus = Path(directory) / "stories.txt"
            corpus.write_text(text, encoding="utf-8")
            tokenizer = BPETokenizer.train_from_iterator(
                [text], vocab_size=512, min_frequency=1
            )
            with (
                patch("mygpt.data.TOKENIZE_CHUNK_BYTES", 17),
                patch("mygpt.data.TOKENIZE_PROGRESS_BYTES", 10**9),
            ):
                cache, metadata = prepare_token_cache(corpus, tokenizer)

            mapped = torch.from_file(
                str(cache),
                shared=False,
                size=int(metadata["token_count"]),
                dtype=torch.uint16,
            )
            self.assertEqual(mapped.long().tolist(), tokenizer.encode(text))

    def test_token_cache_is_reused_and_invalidated_with_the_corpus(self) -> None:
        text = "A little cat went home.\n<eos>\n" * 20
        with TemporaryDirectory() as directory:
            corpus = Path(directory) / "stories.txt"
            corpus.write_text(text, encoding="utf-8")
            tokenizer = BPETokenizer.train_from_iterator(
                [text], vocab_size=512, min_frequency=1
            )
            cache, first = prepare_token_cache(corpus, tokenizer)

            with patch(
                "mygpt.data._encode_chunks",
                side_effect=AssertionError("valid cache should be reused"),
            ):
                reused_cache, reused = prepare_token_cache(corpus, tokenizer)
            self.assertEqual(reused_cache, cache)
            self.assertEqual(reused, first)

            corpus.write_text(text + "A dog arrived.\n<eos>\n", encoding="utf-8")
            _, updated = prepare_token_cache(corpus, tokenizer)
            self.assertNotEqual(updated["source_size"], first["source_size"])
            self.assertGreater(updated["token_count"], first["token_count"])
