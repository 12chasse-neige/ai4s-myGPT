import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "prepare_data.py"
SPEC = importlib.util.spec_from_file_location("prepare_data", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
prepare_data = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare_data)


class PrepareDataTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
