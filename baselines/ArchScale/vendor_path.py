"""Utilities for importing the vendored Microsoft ArchScale implementation."""

from __future__ import annotations

import sys
from pathlib import Path


ARCHSCALE_VENDOR_ROOT = Path(__file__).resolve().parent / "vendor" / "archscale"


def ensure_archscale_on_path() -> Path:
    """Prepend the vendored ArchScale source root to ``sys.path``."""

    vendor = str(ARCHSCALE_VENDOR_ROOT)
    if vendor not in sys.path:
        sys.path.insert(0, vendor)
    return ARCHSCALE_VENDOR_ROOT
