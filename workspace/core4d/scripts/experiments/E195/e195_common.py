#!/usr/bin/env python3
"""Frozen contracts and lightweight IO helpers for E195 hand-gate A3."""

from __future__ import annotations

import csv
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[5]
SCRIPT_DIR = REPO / "workspace/core4d/scripts/experiments/E195"
RESULTS = REPO / "workspace/core4d/results/E195"
E192_RESULTS = REPO / "workspace/core4d/results/E192"
E192_MANIFEST = E192_RESULTS / "s6_downstream/manifests/cem_full_manifest.tsv"

METHOD_ID = "E195_E192A2_PRG_handGate_A3_r1"
HAND_COLLISION_VARIANT_ID = "rubber_hull"

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

E192_GATE = {
    "cem_hand_gate_min_sdf_m": -0.010,
    "cem_hand_gate_max_violation_pct": 0.05,
    "cem_hand_gate_hard_floor_m": -0.015,
}
E195_GATE = {
    "cem_hand_gate_min_sdf_m": -0.008,
    "cem_hand_gate_max_violation_pct": 0.05,
    "cem_hand_gate_hard_floor_m": -0.012,
}
FULL_SAMPLES, FULL_OPT_STEPS, CEM_SEED = 1024, 32, 0
KP_POS, KP_ROT = 500.0, 50.0

WORKERS = {
    "local-gpu0": {"worker_id": "local-gpu0", "host": "local", "gpu_id": "0"},
    "ada-gpu0": {"worker_id": "ada-gpu0", "host": "ada6000", "gpu_id": "0"},
    "ada-gpu1": {"worker_id": "ada-gpu1", "host": "ada6000", "gpu_id": "1"},
}
CASE_WORKER = {
    "box024_20231011_026_p1": "ada-gpu0",
    "box024_20231011_026_p2": "ada-gpu1",
    "box024_20231011_027_p1": "local-gpu0",
    "box024_20231011_027_p2": "local-gpu0",
    "box024_20231011_028_p1": "local-gpu0",
    "box024_20231011_028_p2": "local-gpu0",
    "box024_20231011_030_p1": "ada-gpu0",
    "box024_20231011_031_p1": "ada-gpu1",
    "box024_20231011_031_p2": "local-gpu0",
    "box004_20231003_2_082_p1": "ada-gpu0",
    "box004_20231003_2_082_p2": "ada-gpu1",
    "box004_20231003_2_083_p1": "local-gpu0",
    "box004_20231003_2_083_p2": "ada-gpu1",
    "box004_20231003_2_086_p1": "ada-gpu0",
    "box004_20231003_2_086_p2": "local-gpu0",
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


def load_e192_rows() -> dict[str, dict[str, str]]:
    rows = {row["case_id"]: row for row in read_tsv(E192_MANIFEST)}
    expected = set(all_case_ids())
    if set(rows) != expected:
        raise SystemExit(f"E192 Full case set differs: got={len(rows)} expected=15")
    for case_id, row in rows.items():
        if row.get("status") != "run_complete_pending_eval":
            raise SystemExit(f"E192 baseline is not complete: {case_id}:{row.get('status')}")
    return rows


def e195_overrides() -> str:
    return " ".join(f"{key}={value:.6f}" for key, value in E195_GATE.items())


def worker_for(case_id: str) -> dict[str, str]:
    return dict(WORKERS[CASE_WORKER[case_id]])

