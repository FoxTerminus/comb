from pathlib import Path
import tempfile
import unittest
from unittest import mock

from training.artifact_io import (
    append_jsonl_fsync,
    read_resumable_jsonl,
    write_text_atomic,
)


class AtomicArtifactTest(unittest.TestCase):
    def test_replaces_complete_text_without_leaving_temporary_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "result.json"
            output.write_text("old")
            write_text_atomic(output, "new\n")
            self.assertEqual(output.read_text(), "new\n")
            self.assertEqual(list(root.glob("*.tmp")), [])
            self.assertEqual(list(root.glob(".*.tmp")), [])

    def test_failed_replace_preserves_prior_artifact_and_cleans_temporary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "result.json"
            output.write_text("old")
            with mock.patch(
                "training.artifact_io.os.replace",
                side_effect=OSError("simulated interruption"),
            ), self.assertRaises(OSError):
                write_text_atomic(output, "incomplete")
            self.assertEqual(output.read_text(), "old")
            self.assertEqual(list(root.glob(".*.tmp")), [])

    def test_jsonl_append_is_durable_and_round_trips(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.jsonl"
            append_jsonl_fsync(output, {"index": 0, "text": "你好"})
            append_jsonl_fsync(output, {"index": 1, "text": "ok"})
            self.assertEqual(
                read_resumable_jsonl(output),
                [
                    {"index": 0, "text": "你好"},
                    {"index": 1, "text": "ok"},
                ],
            )
            self.assertTrue(output.read_bytes().endswith(b"\n"))

    def test_jsonl_truncated_tail_is_archived_then_removed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "result.jsonl"
            output.write_bytes(b'{"index": 0}\n{"ind')
            self.assertEqual(read_resumable_jsonl(output), [{"index": 0}])
            self.assertEqual(output.read_bytes(), b'{"index": 0}\n')
            archives = list(root.glob("result.jsonl.truncated_tail_*"))
            self.assertEqual(len(archives), 1)
            self.assertEqual(archives[0].read_bytes(), b'{"ind')

    def test_complete_jsonl_tail_without_newline_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.jsonl"
            output.write_bytes(b'{"index": 0}')
            self.assertEqual(read_resumable_jsonl(output), [{"index": 0}])
            self.assertEqual(output.read_bytes(), b'{"index": 0}\n')


if __name__ == "__main__":
    unittest.main()
