#!/usr/bin/env python3
"""Shared constants and filesystem helpers for E169."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[5]
RESULTS = REPO / "workspace/core4d/results/E169"
SCRIPT_DIR = REPO / "workspace/core4d/scripts/experiments/E169"
OVERRIDE_DIR = REPO / "examples/config/override"
E168_MANIFEST = (
    REPO
    / "workspace/core4d/results/E168/s6_downstream/cem/manifests/cem_production_manifest.tsv"
)

CASES = (
    "box021_20231018_033_p1",
    "box021_20231018_032_p1",
    "box021_20231020_020_p2",
    "box021_20231020_023_p2",
)
CASE_GPU = {case_id: index for index, case_id in enumerate(CASES)}
CONTROL_CASE = "box021_20231020_023_p2"

CELL_BITS = {
    "B0": (False, False, False),
    "P": (True, False, False),
    "R": (False, True, False),
    "G": (False, False, True),
    "PR": (True, True, False),
    "PG": (True, False, True),
    "RG": (False, True, True),
    "PRG": (True, True, True),
}
FULL_CELLS = tuple(cell for cell in CELL_BITS if cell != "B0")

LOWER_BODY_GEOMS = (
    "left_hip_collision",
    "right_hip_collision",
    "left_thigh_collision",
    "right_thigh_collision",
    "left_shin_collision",
    "right_shin_collision",
    "left_linkage_brace_collision",
    "right_linkage_brace_collision",
    "lf0",
    "lf1",
    "lf2",
    "lf3",
    "rf0",
    "rf1",
    "rf2",
    "rf3",
)

NO_PHYSICS_SCENE_NAME = "scene_act_E168_rubber_hull"
PHYSICS_SCENE_NAME = "scene_act_E169_lowerbody_physics"


def repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO / path


def rel(path: str | Path) -> str:
    value = Path(path)
    try:
        return str(value.resolve().relative_to(REPO.resolve()))
    except (OSError, ValueError):
        return str(path)


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with repo_path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_tsv(path: str | Path) -> list[dict[str, str]]:
    with repo_path(path).open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def write_tsv(
    path: str | Path,
    rows: list[dict[str, Any]],
    fields: list[str] | None = None,
) -> None:
    output = repo_path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
        for row in rows:
            for field in row:
                if field not in fields:
                    fields.append(field)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=fields, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def e168_rows() -> dict[str, dict[str, str]]:
    selected = {row["case_id"]: row for row in read_tsv(E168_MANIFEST) if row["case_id"] in CASES}
    missing = sorted(set(CASES) - set(selected))
    if missing:
        raise RuntimeError(f"E168 manifest is missing E169 cases: {missing}")
    return selected


def safe_id(value: str) -> str:
    return "".join(char if char.isalnum() or char == "_" else "_" for char in value)
