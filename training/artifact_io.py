"""Crash-safe writes for small reproduction artifacts."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
import hashlib
import json
from typing import Any


def write_bytes_atomic(path: Path | str, payload: bytes) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        directory = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_text_atomic(path: Path | str, text: str) -> None:
    write_bytes_atomic(path, text.encode())


def append_jsonl_fsync(path: Path | str, payload: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, ensure_ascii=False) + "\n").encode()
    descriptor = os.open(
        destination,
        os.O_WRONLY | os.O_CREAT | os.O_APPEND,
        0o644,
    )
    try:
        view = memoryview(encoded)
        while view:
            written = os.write(descriptor, view)
            if written < 1:
                raise OSError("JSONL append made no progress")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def read_resumable_jsonl(path: Path | str) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.exists():
        return []
    content = source.read_bytes()
    if content and not content.endswith(b"\n"):
        cutoff = content.rfind(b"\n") + 1
        prefix, tail = content[:cutoff], content[cutoff:]
        try:
            json.loads(tail)
        except (json.JSONDecodeError, UnicodeDecodeError):
            digest = hashlib.sha256(tail).hexdigest()[:16]
            archive = source.with_name(
                f"{source.name}.truncated_tail_{digest}"
            )
            if archive.exists():
                if archive.read_bytes() != tail:
                    raise RuntimeError(f"truncated-tail archive conflict: {archive}")
            else:
                write_bytes_atomic(archive, tail)
            write_bytes_atomic(source, prefix)
            content = prefix
        else:
            content += b"\n"
            write_bytes_atomic(source, content)

    records = []
    for line_number, line in enumerate(content.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise RuntimeError(
                f"malformed interior JSONL row {line_number}: {source}"
            ) from error
        if not isinstance(record, dict):
            raise RuntimeError(
                f"JSONL row {line_number} is not an object: {source}"
            )
        records.append(record)
    return records
