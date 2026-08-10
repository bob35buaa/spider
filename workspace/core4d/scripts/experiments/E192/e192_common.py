#!/usr/bin/env python3
"""Frozen contracts and IO helpers for E192 hand-gate A0/A2."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[5]
SCRIPT_DIR = REPO / "workspace/core4d/scripts/experiments/E192"
RESULTS = REPO / "workspace/core4d/results/E192"

METHOD_ID = "E192_E167A_PRG_handGate_A2_r1"
HAND_COLLISION_VARIANT_ID = "rubber_hull"
SOURCE_MANIFEST = {
    "box004": (
        "E172",
        REPO / "workspace/core4d/results/E172/s6_downstream/manifests/cem_full_manifest.tsv",
    ),
    "box024": (
        "E173",
        REPO / "workspace/core4d/results/E173/s6_downstream/manifests/cem_full_manifest.tsv",
    ),
}

CASES = {
    "box024": [
        "box024_20231011_026_p1", "box024_20231011_026_p2",
        "box024_20231011_027_p1", "box024_20231011_027_p2",
        "box024_20231011_028_p1", "box024_20231011_028_p2",
        "box024_20231011_030_p1",
        "box024_20231011_031_p1", "box024_20231011_031_p2",
    ],
    "box004": [
        "box004_20231003_2_082_p1", "box004_20231003_2_082_p2",
        "box004_20231003_2_083_p1", "box004_20231003_2_083_p2",
        "box004_20231003_2_086_p1", "box004_20231003_2_086_p2",
    ],
}
SENTINEL_CASES = [
    "box024_20231011_026_p1",
    "box004_20231003_2_082_p1",
    "box004_20231003_2_086_p2",
]
CANARY_CASES = [
    "box024_20231011_026_p1",
    "box024_20231011_027_p2",
    "box004_20231003_2_082_p1",
]

A0_GATE = {
    "cem_hand_gate_min_sdf_m": -0.010,
    "cem_hand_gate_max_violation_pct": 0.10,
    "cem_hand_gate_hard_floor_m": -0.020,
}
A2_GATE = {
    "cem_hand_gate_min_sdf_m": -0.010,
    "cem_hand_gate_max_violation_pct": 0.05,
    "cem_hand_gate_hard_floor_m": -0.015,
}
FULL_SAMPLES, FULL_OPT_STEPS = 1024, 32
CANARY_SAMPLES, CANARY_OPT_STEPS = 64, 4
CEM_SEED = 0
KP_POS, KP_ROT = 500.0, 50.0

WORKERS = [
    {"worker_id": "local-gpu0", "host": "local", "gpu_id": "0"},
    {"worker_id": "a100-gpu4", "host": "a100", "gpu_id": "4"},
    {"worker_id": "a100-gpu5", "host": "a100", "gpu_id": "5"},
    {"worker_id": "a100-gpu6", "host": "a100", "gpu_id": "6"},
    {"worker_id": "a100-gpu7", "host": "a100", "gpu_id": "7"},
]
# E192 uses local GPU0 plus remote A100 GPU4/5 throughout the corrected A0,
# A2 canary, and A2 Full sequence.
THREE_CARD_WORKERS = WORKERS[:3]
STAGE_WORKERS = {
    "baseline_sentinel": THREE_CARD_WORKERS,
    "canary": THREE_CARD_WORKERS,
    "full": THREE_CARD_WORKERS,
}

GATE_DIAGNOSTIC_KEYS = (
    "cem_hand_gate_selected_min_sdf_m",
    "cem_hand_gate_selected_mean_sdf_m",
    "cem_hand_gate_selected_p05_sdf_m",
    "cem_hand_gate_valid_frac",
    "cem_hand_gate_selected_valid_frac",
    "cem_gate_valid_frac",
    "cem_gate_fallback_used",
    "cem_leg_gate_valid_frac",
    "cem_leg_gate_selected_valid_frac",
    "cem_leg_gate_fallback_used",
    "cem_leg_gate_min_sdf_min_m",
    "cem_leg_gate_min_sdf_p05_m",
    "cem_leg_gate_violation_pct_mean",
    "cem_leg_gate_selected_all_valid",
    "cem_posture_gate_valid_frac",
    "cem_posture_gate_selected_valid_frac",
    "cem_posture_gate_fallback_used",
)


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def all_case_ids() -> list[str]:
    return [case for key in ("box024", "box004") for case in CASES[key]]


def object_key_of(case_id: str) -> str:
    for key, cases in CASES.items():
        if case_id in cases:
            return key
    raise KeyError(case_id)


def repo_path(value: str | Path) -> Path:
    path = Path(str(value))
    if not path.is_absolute():
        return REPO / path
    if path.exists():
        return path
    parts = path.parts
    for marker in ("example_datasets", "workspace", "logs", "examples"):
        if marker in parts:
            return REPO.joinpath(*parts[parts.index(marker):])
    return path


def rel(value: str | Path) -> str:
    path = repo_path(value)
    try:
        return path.relative_to(REPO).as_posix()
    except ValueError:
        return Path(str(value)).as_posix()


def sha256(value: str | Path) -> str:
    path = repo_path(value)
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_tsv(value: str | Path) -> list[dict[str, str]]:
    with repo_path(value).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def read_with_fields(value: str | Path) -> tuple[list[dict[str, str]], list[str]]:
    with repo_path(value).open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        return list(reader), list(reader.fieldnames or [])


def serial(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return value


def write_tsv(value: str | Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path = repo_path(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=fields, delimiter="\t", lineterminator="\n",
            extrasaction="ignore",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({field: serial(row.get(field, "")) for field in fields})


def write_json(value: str | Path, payload: Any) -> None:
    path = repo_path(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_source_rows() -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for object_key, (exp_id, manifest) in SOURCE_MANIFEST.items():
        rows = {row["case_id"]: row for row in read_tsv(manifest)}
        for case_id in CASES[object_key]:
            if case_id not in rows:
                raise SystemExit(f"{case_id} missing from {manifest}")
            out[case_id] = {**rows[case_id], "_source_exp": exp_id}
    if set(out) != set(all_case_ids()):
        raise SystemExit("E192 source case set drift")
    return out


def a2_overrides() -> str:
    return " ".join(f"{key}={value:.6f}" for key, value in A2_GATE.items())


def worker_for(stage: str, ordinal: int) -> dict[str, str]:
    workers = STAGE_WORKERS[stage]
    return workers[(ordinal - 1) % len(workers)]
