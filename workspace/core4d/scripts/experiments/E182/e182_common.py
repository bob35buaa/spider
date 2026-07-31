#!/usr/bin/env python3
"""Shared deterministic I/O helpers for E182."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[5]


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest for a file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    """Return the SHA-256 digest for bytes."""
    return hashlib.sha256(data).hexdigest()


def repo_path(value: str | Path) -> Path:
    """Resolve a repository-relative path."""
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def relative_to_repo(path: Path) -> str:
    """Return a stable repository-relative path when possible."""
    absolute = path.absolute()
    try:
        return absolute.relative_to(REPO_ROOT.absolute()).as_posix()
    except ValueError:
        return str(absolute)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    """Write deterministic JSON atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_tsv(
    path: Path,
    rows: Sequence[dict[str, str]],
    fields: Sequence[str],
) -> None:
    """Write a deterministic TSV atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(fields),
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def read_tsv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Read a TSV and return its header and rows."""
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if reader.fieldnames is None:
            raise RuntimeError(f"TSV has no header: {path}")
        return list(reader.fieldnames), list(reader)


def immutable_copy(source: Path, destination: Path) -> dict[str, Any]:
    """Copy one file without permitting a frozen destination to change."""
    if not source.is_file():
        raise FileNotFoundError(source)
    source_sha = sha256_file(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if not destination.is_file():
            raise RuntimeError(f"snapshot destination is not a file: {destination}")
        destination_sha = sha256_file(destination)
        if destination_sha != source_sha:
            raise RuntimeError(
                f"immutable snapshot mismatch: {destination_sha} != {source_sha}: "
                f"{destination}"
            )
    else:
        shutil.copy2(source, destination)
    return {
        "source": relative_to_repo(source),
        "snapshot": relative_to_repo(destination),
        "sha256": source_sha,
        "size_bytes": source.stat().st_size,
    }


def immutable_copy_tree(
    source_root: Path, destination_root: Path
) -> list[dict[str, Any]]:
    """Copy every file in a tree using the immutable-copy contract."""
    if not source_root.is_dir():
        raise FileNotFoundError(source_root)
    entries: list[dict[str, Any]] = []
    for source in sorted(path for path in source_root.rglob("*") if path.is_file()):
        relative = source.relative_to(source_root)
        entry = immutable_copy(source, destination_root / relative)
        entry["tree_relative_path"] = relative.as_posix()
        entries.append(entry)
    return entries


def inventory_digest(entries: Iterable[dict[str, Any]]) -> str:
    """Hash a file inventory independent of absolute workspace location."""
    canonical = [
        {
            "path": entry.get("tree_relative_path", entry["snapshot"]),
            "sha256": entry["sha256"],
            "size_bytes": entry["size_bytes"],
        }
        for entry in entries
    ]
    return sha256_bytes(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
    )
