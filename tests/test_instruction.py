import unittest

import torch

from mygpt.data import CharacterTokenizer
from mygpt.generation import generate_tokens
from mygpt.instruction import (
    IGNORE_INDEX,
    InstructionDataset,
    collate_instruction_batch,
    encode_alpaca_record,
    fit_alpaca_prompt,
    format_alpaca_prompt,
    instruction_characters,
    split_alpaca_records,
)


def record(instruction: str, input_text: str, output: str) -> dict[str, str]:
    return {"instruction": instruction, "input": input_text, "output": output}


def instruction_tokenizer(records: list[dict[str, str]]) -> CharacterTokenizer:
    tokenizer = CharacterTokenizer.from_text("pretraining text\n")
    return tokenizer.extended(
        instruction_characters(records), add_instruction_tokens=True
    )


class InstructionDataTest(unittest.TestCase):
    def test_prompt_templates_handle_optional_input(self) -> None:
        without_input = format_alpaca_prompt("Say hello.")
        with_input = format_alpaca_prompt("Summarize.", "Some context.")
        self.assertIn("### Instruction:\nSay hello.", without_input)
        self.assertNotIn("### Input:", without_input)
        self.assertIn("### Input:\nSome context.", with_input)
        self.assertTrue(with_input.endswith("### Response:\n"))

    def test_split_is_deterministic_and_has_no_overlap(self) -> None:
        records = [record(f"task {index}", "", "answer") for index in range(20)]
        first = split_alpaca_records(records, 0.75, seed=7, max_records=12)
        second = split_alpaca_records(records, 0.75, seed=7, max_records=12)
        self.assertEqual(first, second)
        train, validation = first
        self.assertEqual((len(train), len(validation)), (9, 3))
        self.assertTrue(
            {item["instruction"] for item in train}.isdisjoint(
                item["instruction"] for item in validation
            )
        )

    def test_tokenizer_expansion_preserves_ids_and_adds_special_tokens(self) -> None:
        tokenizer = CharacterTokenizer.from_text("abc")
        old_ids = dict(tokenizer.stoi)
        expanded = tokenizer.extended(["c", "é"], add_instruction_tokens=True)
        self.assertEqual({token: expanded.stoi[token] for token in old_ids}, old_ids)
        self.assertEqual(expanded.decode(expanded.encode("é")), "é")
        self.assertNotEqual(expanded.eos_id, expanded.pad_id)

    def test_response_only_targets_are_shifted_and_end_with_eos(self) -> None:
        example = record("Add.", "2 + 2", "4")
        tokenizer = instruction_tokenizer([example])
        encoded = encode_alpaca_record(example, tokenizer, max_length=512)
        assert encoded is not None
        prompt_length = len(tokenizer.encode(format_alpaca_prompt("Add.", "2 + 2")))
        self.assertTrue(torch.all(encoded.targets[: prompt_length - 1] == IGNORE_INDEX))
        self.assertEqual(encoded.targets[prompt_length - 1].item(), tokenizer.stoi["4"])
        self.assertEqual(encoded.targets[-1].item(), tokenizer.eos_id)
        self.assertEqual(len(encoded.tokens), len(encoded.targets))

    def test_input_then_response_are_truncated_to_checkpoint_length(self) -> None:
        example = record("Answer briefly.", "context " * 100, "response " * 100)
        tokenizer = instruction_tokenizer([example])
        minimal_length = len(format_alpaca_prompt(example["instruction"]))
        max_length = minimal_length + 5
        encoded = encode_alpaca_record(example, tokenizer, max_length)
        assert encoded is not None
        self.assertTrue(encoded.input_truncated)
        self.assertTrue(encoded.response_truncated)
        self.assertEqual(len(encoded.tokens), max_length)

    def test_empty_responses_and_oversized_instructions_are_skipped(self) -> None:
        records = [
            record("valid", "", "answer"),
            record("empty", "", ""),
            record("x" * 1000, "", "answer"),
        ]
        tokenizer = instruction_tokenizer(records)
        dataset = InstructionDataset(records, tokenizer, max_length=180)
        self.assertEqual(len(dataset), 1)
        self.assertEqual(dataset.stats.empty_responses, 1)
        self.assertEqual(dataset.stats.prompts_too_long, 1)

    def test_dynamic_padding_masks_padding_targets(self) -> None:
        records = [record("short", "", "a"), record("long", "", "answer")]
        tokenizer = instruction_tokenizer(records)
        dataset = InstructionDataset(records, tokenizer, max_length=512)
        batch = collate_instruction_batch(
            [dataset[0], dataset[1]], pad_id=tokenizer.pad_id
        )
        tokens, targets = batch
        self.assertEqual(tokens.shape, targets.shape)
        short_length = len(dataset[0][0])
        self.assertTrue(torch.all(tokens[0, short_length:] == tokenizer.pad_id))
        self.assertTrue(torch.all(targets[0, short_length:] == IGNORE_INDEX))

    def test_prompt_that_cannot_fit_is_rejected(self) -> None:
        self.assertIsNone(fit_alpaca_prompt("x" * 1000, "", max_length=100))


class AlwaysEosModel:
    def __init__(self, vocab_size: int, eos_id: int) -> None:
        self.config = type("Config", (), {"block_size": 8})()
        self.vocab_size = vocab_size
        self.eos_id = eos_id

    def eval(self) -> None:
        pass

    def __call__(self, tokens: torch.Tensor):
        logits = torch.full(
            (tokens.size(0), tokens.size(1), self.vocab_size), -1000.0
        )
        logits[:, -1, self.eos_id] = 1000.0
        return logits, None


class InstructionGenerationTest(unittest.TestCase):
    def test_generation_stops_after_eos(self) -> None:
        model = AlwaysEosModel(vocab_size=5, eos_id=4)
        prompt = torch.tensor([[1, 2]], dtype=torch.long)
        generated = generate_tokens(
            model, prompt, max_new_tokens=10, top_k=None, eos_token_id=4
        )
        self.assertEqual(generated.tolist(), [[1, 2, 4]])


if __name__ == "__main__":
    unittest.main()
