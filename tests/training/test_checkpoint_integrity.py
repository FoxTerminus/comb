from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import zipfile

from training.checkpoint_integrity import valid_torch_zip


class CheckpointIntegrityTest(unittest.TestCase):
    def test_requires_readable_pytorch_pickle_member(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            valid = root / "valid.pt"
            with zipfile.ZipFile(valid, "w") as archive:
                archive.writestr("archive/data.pkl", b"pickle")
                archive.writestr("archive/data/0", b"tensor")
            self.assertTrue(valid_torch_zip(valid))

            missing_pickle = root / "missing.pt"
            with zipfile.ZipFile(missing_pickle, "w") as archive:
                archive.writestr("archive/data/0", b"tensor")
            self.assertFalse(valid_torch_zip(missing_pickle))

            truncated = root / "truncated.pt"
            truncated.write_bytes(valid.read_bytes()[:-16])
            self.assertFalse(valid_torch_zip(truncated))
