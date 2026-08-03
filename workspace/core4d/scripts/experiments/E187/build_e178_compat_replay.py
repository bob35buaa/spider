#!/usr/bin/env python3
"""Build the three-row E187 compatibility replay authority and execution manifest."""

from __future__ import annotations

import argparse
import csv
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from freeze_authority import E178_MANIFEST, REPO_ROOT, _validate_authority

REPRESENTATIVE_CASES = (
    "bucket003_20231018_003_p1",
    "bucket004_20231002_021_p1",
    "bucket007_20231020_055_p1",
)
DEFAULT_ROOT = REPO_ROOT / "workspace/core4d/results/E187/s0_environment/e178_compat"
SOURCE_OUTPUT_FIELDS = (
    "result_npz",
    "outdir_npz",
    "config_act",
    "video",
    "log",
)
ADDED_FIELDS = tuple(f"source_e178_{field}" for field in SOURCE_OUTPUT_FIELDS) + (
    "source_e178_variant",
    "compat_device",
)


def _read(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if reader.fieldnames is None:
            raise RuntimeError(f"missing TSV header: {path}")
        return list(reader.fieldnames), list(reader)


def display_path(path: Path) -> str:
    """Use repository-relative paths in production and absolute temporary paths."""
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def _tsv_bytes(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> bytes:
    from io import StringIO

    stream = StringIO(newline="")
    writer = csv.DictWriter(
        stream, delimiter="\t", fieldnames=list(fields), lineterminator="\n"
    )
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def _write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _write_immutable(path: Path, payload: bytes) -> None:
    if path.exists():
        if not path.is_file() or path.read_bytes() != payload:
            raise RuntimeError(f"immutable E187 compat authority mismatch: {path}")
        return
    _write_atomic(path, payload)


def authority_rows(root: Path = DEFAULT_ROOT) -> tuple[list[str], list[dict[str, str]]]:
    """Return the three source-frozen rows with independent E187 outputs."""
    _validate_authority()
    source_fields, source_rows = _read(E178_MANIFEST)
    by_case = {row["case_id"]: row for row in source_rows}
    rows: list[dict[str, str]] = []
    for case_id in REPRESENTATIVE_CASES:
        source = by_case[case_id]
        row = dict(source)
        variant = f"E187_e178_compat_{case_id}"
        outdir = root / "cem" / f"{variant}_outdir"
        for field in SOURCE_OUTPUT_FIELDS:
            row[f"source_e178_{field}"] = source[field]
        row["source_e178_variant"] = source["variant"]
        row.update(
            {
                "variant": variant,
                "result_npz": display_path(root / "cem" / f"{variant}.npz"),
                "outdir_npz": display_path(outdir / "trajectory_mjwp_act.npz"),
                "config_act": display_path(outdir / "config_act.yaml"),
                "video": display_path(root / "render" / f"{variant}.mp4"),
                "log": f"logs/E187/s0/e178_compat/{variant}.log",
                "assigned_gpu": "local-0",
                "gpu_id": "0",
                "compat_device": "local_rtx5090",
                "status": "not_run",
                "failure_mode": "",
                "blocker_detail": "",
                "updated_at": "",
                "cem_samples": "1024",
                "cem_opt_steps": "32",
                "cem_seed": "0",
                "execution_mode": "compatibility",
            }
        )
        rows.append(row)
    fields = source_fields + [
        field for field in ADDED_FIELDS if field not in source_fields
    ]
    return fields, rows


def build(root: Path = DEFAULT_ROOT) -> dict[str, Any]:
    """Create the immutable authority and resumable mutable execution manifest."""
    fields, rows = authority_rows(root)
    authority_path = root / "authority_manifest.tsv"
    execution_path = root / "execution_manifest.tsv"
    payload = _tsv_bytes(rows, fields)
    _write_immutable(authority_path, payload)
    if not execution_path.exists():
        _write_atomic(execution_path, payload)
    else:
        execution_fields, execution_rows = _read(execution_path)
        if execution_fields != fields or [
            row["case_id"] for row in execution_rows
        ] != list(REPRESENTATIVE_CASES):
            raise RuntimeError("E187 compat execution manifest authority changed")
        frozen_fields = [
            field
            for field in fields
            if field
            not in {"status", "failure_mode", "blocker_detail", "updated_at", "gpu_id"}
        ]
        for authority, execution in zip(rows, execution_rows, strict=True):
            for field in frozen_fields:
                if authority[field] != execution[field]:
                    raise RuntimeError(
                        f"{authority['case_id']}: compat scientific field changed: {field}"
                    )
    return {
        "status": "PASS",
        "rows": len(rows),
        "authority_manifest": display_path(authority_path),
        "execution_manifest": display_path(execution_path),
    }


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    print(build(args.root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
