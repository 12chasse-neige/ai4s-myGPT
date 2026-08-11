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
        self.assertTrue(args.tinystories)

    def test_source_flags_select_tinystories_or_stanford_alpaca(self) -> None:
        with patch("sys.argv", ["prepare_data.py", "--tinystories"]):
            tinystories_args = prepare_data.parse_args()
        with patch("sys.argv", ["prepare_data.py", "--stanford-alpaca"]):
            alpaca_args = prepare_data.parse_args()
        self.assertTrue(tinystories_args.tinystories)
        self.assertFalse(alpaca_args.tinystories)

    def test_write_stories_normalizes_and_separates_records(self) -> None:
        with TemporaryDirectory() as directory:
            output = Path(directory) / "tinystories.txt"
            count, characters = prepare_data.write_stories(
                [" First story.\r\n", "", "Second story.  "], output
            )
            expected = "First story.\n\nSecond story.\n"
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


if __name__ == "__main__":
    unittest.main()
