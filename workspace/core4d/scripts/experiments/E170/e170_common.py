#!/usr/bin/env python3
"""Shared contracts and filesystem helpers for E170."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[5]
RESULTS = REPO / "workspace/core4d/results/E170"
SCRIPT_DIR = REPO / "workspace/core4d/scripts/experiments/E170"
OVERRIDE_DIR = REPO / "examples/config/override"
E168_REVIEWED = RESULTS.parent / "E168/s6_downstream/cem/eval/box021_all28_reviewed/evaluated_manifest_snapshot.tsv"
E168_PRODUCTION = RESULTS.parent / "E168/s6_downstream/cem/manifests/cem_production_manifest.tsv"
E168_METRICS = RESULTS.parent / "E168/s6_downstream/cem/eval/box021_all28_reviewed/e168_case_metrics.tsv"
E169_FULL = RESULTS.parent / "E169/manifests/cem_full_manifest.tsv"

REUSE_CASES = {
    "box021_20231018_032_p1",
    "box021_20231018_033_p1",
    "box021_20231020_020_p2",
    "box021_20231020_023_p2",
}

GPU_QUEUES = {
    "0": (
        "box021_20231011_034_p2", "box021_20231011_037_p1",
        "box021_20231018_031_p2", "box021_20231020_019_p2",
        "box021_20231018_030_p1", "box021_20231020_020_p1",
    ),
    "1": (
        "box021_20231011_036_p2", "box021_20231011_038_p1",
        "box021_20231020_023_p1", "box021_20231018_033_p2",
        "box021_20231018_035_p2", "box021_20231018_028_p1",
    ),
    "2": (
        "box021_20231011_034_p1", "box021_20231011_037_p2",
        "box021_20231020_022_p1", "box021_20231020_019_p1",
        "box021_20231018_030_p2", "box021_20231018_029_p1",
    ),
    "3": (
        "box021_20231011_036_p1", "box021_20231011_038_p2",
        "box021_20231020_022_p2", "box021_20231018_034_p2",
        "box021_20231018_028_p2", "box021_20231018_032_p2",
    ),
}
CASE_GPU = {case: gpu for gpu, cases in GPU_QUEUES.items() for case in cases}

LOWER_BODY_GEOMS = (
    "left_hip_collision", "right_hip_collision", "left_thigh_collision",
    "right_thigh_collision", "left_shin_collision", "right_shin_collision",
    "left_linkage_brace_collision", "right_linkage_brace_collision",
    "lf0", "lf1", "lf2", "lf3", "rf0", "rf1", "rf2", "rf3",
)
SCENE_NAME = "scene_act_E170_lowerbody_physics"


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO / path


def rel(value: str | Path) -> str:
    path = Path(value)
    try:
        return str(path.resolve().relative_to(REPO.resolve()))
    except (OSError, ValueError):
        return str(value)


def sha256(value: str | Path) -> str:
    digest = hashlib.sha256()
    with repo_path(value).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_tsv(value: str | Path) -> list[dict[str, str]]:
    with repo_path(value).open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def write_tsv(value: str | Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    output = repo_path(value)
    output.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
        for row in rows:
            for field in row:
                if field not in fields:
                    fields.append(field)
    fd, temporary = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        Path(temporary).replace(output)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


def write_json(value: str | Path, payload: Any) -> None:
    output = repo_path(value)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def safe_id(value: str) -> str:
    return "".join(char if char.isalnum() or char == "_" else "_" for char in value)
