"""Lightweight integrity checks for PyTorch ZIP checkpoints."""

from __future__ import annotations

from pathlib import Path
import zipfile


def valid_torch_zip(path: Path) -> bool:
    """Validate the ZIP footer/central directory and required pickle member.

    This deliberately does not stream tensor payloads.  It is cheap enough for
    the minute-level watchdog while still rejecting the usual interrupted
    ``torch.save`` artifact, whose central directory was never committed.
    """

    try:
        if not path.is_file() or path.stat().st_size == 0 or not zipfile.is_zipfile(path):
            return False
        with zipfile.ZipFile(path) as archive:
            members = archive.infolist()
            return bool(members) and any(
                member.filename == "data.pkl"
                or member.filename.endswith("/data.pkl")
                for member in members
            )
    except (OSError, zipfile.BadZipFile, EOFError):
        return False
