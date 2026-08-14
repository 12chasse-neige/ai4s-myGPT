from io import BytesIO
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "prepare_data.py"
SPEC = importlib.util.spec_from_file_location("prepare_data", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
prepare_data = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare_data)


class PrepareDataTest(unittest.TestCase):
    def test_tinystories_is_the_default_source(self) -> None:
        with patch("sys.argv", ["prepare_data.py"]):
            args = prepare_data.parse_args()
        self.assertEqual(args.source, "tinystories")

    def test_source_flags_select_each_supported_dataset(self) -> None:
        expected_sources = {
            "--tinystories": "tinystories",
            "--stanford-alpaca": "stanford-alpaca",
            "--wikitext": "wikitext",
            "--mmlu": "mmlu",
        }
        for flag, expected in expected_sources.items():
            with self.subTest(flag=flag), patch("sys.argv", ["prepare_data.py", flag]):
                self.assertEqual(prepare_data.parse_args().source, expected)

    def test_write_stories_normalizes_and_separates_records(self) -> None:
        with TemporaryDirectory() as directory:
            output = Path(directory) / "tinystories.txt"
            count, characters = prepare_data.write_stories(
                [" First story.\r\n", "", "Second story.  "], output
            )
            expected = "First story.\n<eos>\nSecond story.\n<eos>\n"
            self.assertEqual(output.read_text(encoding="utf-8"), expected)
            self.assertEqual(count, 2)
            self.assertEqual(characters, len(expected))

    def test_empty_story_stream_is_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            output = Path(directory) / "empty.txt"
            output.write_text("keep me", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "empty"):
                prepare_data.write_stories([" \n"], output)
            self.assertEqual(output.read_text(encoding="utf-8"), "keep me")

    def test_load_stanford_alpaca_validates_expected_fields(self) -> None:
        records = [
            {
                "instruction": "Summarize the text.",
                "input": "A short passage.",
                "output": "A summary.",
            },
            {
                "instruction": "Say hello.",
                "input": "",
                "output": "",
            },
        ]
        with TemporaryDirectory() as directory:
            path = Path(directory) / "stanford_alpaca.json"
            path.write_text(json.dumps(records), encoding="utf-8")
            self.assertEqual(prepare_data.load_stanford_alpaca(path), records)

    def test_invalid_stanford_alpaca_does_not_replace_existing_file(self) -> None:
        invalid_data = json.dumps([{"instruction": "Missing fields"}]).encode()
        with TemporaryDirectory() as directory:
            output = Path(directory) / "stanford_alpaca.json"
            output.write_text("keep me", encoding="utf-8")
            with patch.object(
                prepare_data, "urlopen", return_value=BytesIO(invalid_data)
            ):
                with self.assertRaisesRegex(ValueError, "input"):
                    prepare_data.fetch_stanford_alpaca(output)
            self.assertEqual(output.read_text(encoding="utf-8"), "keep me")

    def test_fetch_stanford_alpaca_writes_valid_json_atomically(self) -> None:
        records = [
            {
                "instruction": "Add two numbers.",
                "input": "2 and 3",
                "output": "5",
            }
        ]
        payload = json.dumps(records).encode()
        with TemporaryDirectory() as directory:
            output = Path(directory) / "data" / "stanford_alpaca.json"
            with patch.object(prepare_data, "urlopen", return_value=BytesIO(payload)):
                count, byte_count = prepare_data.fetch_stanford_alpaca(output)
            self.assertEqual(count, 1)
            self.assertEqual(byte_count, len(payload))
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), records)

    def test_fetch_wikitext_preserves_raw_line_structure(self) -> None:
        records = [
            {"text": ""},
            {"text": " = Article = \r\n"},
            {"text": ""},
            {"text": " Article body. \n"},
        ]
        with TemporaryDirectory() as directory:
            output = Path(directory) / "wikitext.txt"
            with patch.object(
                prepare_data, "load_huggingface_split", return_value=records
            ) as load_split:
                count, characters = prepare_data.fetch_wikitext(output)
            expected = "\n = Article = \n\n Article body. \n"
            self.assertEqual(output.read_text(encoding="utf-8"), expected)
            self.assertEqual(count, 4)
            self.assertEqual(characters, len(expected))
            load_split.assert_called_once_with(
                prepare_data.DEFAULT_WIKITEXT_DATASET,
                prepare_data.DEFAULT_WIKITEXT_SPLIT,
                prepare_data.DEFAULT_WIKITEXT_CONFIG,
            )

    def test_fetch_mmlu_writes_valid_structured_json(self) -> None:
        records = [
            {
                "question": "What is 2 + 3?",
                "subject": "elementary_mathematics",
                "choices": ["4", "5", "6", "7"],
                "answer": 1,
            }
        ]
        with TemporaryDirectory() as directory:
            output = Path(directory) / "mmlu.json"
            with patch.object(
                prepare_data, "load_huggingface_split", return_value=records
            ) as load_split:
                count, byte_count = prepare_data.fetch_mmlu(output)
            self.assertEqual(count, 1)
            self.assertEqual(byte_count, output.stat().st_size)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), records)
            load_split.assert_called_once_with(
                prepare_data.DEFAULT_MMLU_DATASET,
                prepare_data.DEFAULT_MMLU_SPLIT,
                prepare_data.DEFAULT_MMLU_CONFIG,
            )

    def test_invalid_mmlu_does_not_replace_existing_file(self) -> None:
        records = [
            {
                "question": "Broken question",
                "subject": "test",
                "choices": ["A", "B"],
                "answer": 0,
            }
        ]
        with TemporaryDirectory() as directory:
            output = Path(directory) / "mmlu.json"
            output.write_text("keep me", encoding="utf-8")
            with patch.object(
                prepare_data, "load_huggingface_split", return_value=records
            ):
                with self.assertRaisesRegex(ValueError, "four string choices"):
                    prepare_data.fetch_mmlu(output)
            self.assertEqual(output.read_text(encoding="utf-8"), "keep me")


if __name__ == "__main__":
    unittest.main()
