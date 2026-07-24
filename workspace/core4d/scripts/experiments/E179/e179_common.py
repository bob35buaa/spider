#!/usr/bin/env python3
"""Shared, frozen contracts for E179.

E179 compares E167A z-only body tracking without the E170 PRG additions
against the exact 16 box023 Full-CEM rows from E173.  This module is the
single source of truth for paths, hashes, queue assignments, CEM budgets and
the mandatory twelve-gate scoring adapter.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[5]
RESULTS = REPO / "workspace/core4d/results/E179"
E173_RESULTS = REPO / "workspace/core4d/results/E173"
SCRIPT_DIR = REPO / "workspace/core4d/scripts/experiments/E179"
OVERRIDE_DIR = REPO / "examples/config/override"
TASK_ROOT = (
    REPO
    / "example_datasets/processed/core4d/unitree_g1/humanoid_object"
)

E173_FULL_MANIFEST = (
    E173_RESULTS / "s6_downstream/manifests/cem_full_manifest.tsv"
)
E173_METRICS = (
    E173_RESULTS / "s6_downstream/eval/full/e173_case_metrics.tsv"
)
E173_PIPELINE_AUTHORITY = E173_RESULTS / "registries/pipeline_authority.tsv"
E167A_PROFILE = (
    REPO
    / "workspace/core4d/results/E168/s0_environment/"
    "e167a_profile/e167a_zonly_profile.json"
)

EXPECTED_E173_MANIFEST_SHA256 = (
    "2128deb8403a0efddb7578a46b2ef466811f91ba183d955b0230776de2ae889d"
)
EXPECTED_E167A_PROFILE_SHA256 = (
    "666c302dcf549e517ff02b50c139551cf98635b7b951872284d6a39766548c17"
)
EXPECTED_RAW_BOX023_ROWS = 46
EXPECTED_PAIRED_ROWS = 16
EXPECTED_VARIANTS = {"omnirt_v1": 15, "omnirt_v2": 1}

SCENE_NAME = "scene_act_E179_rubberHull"
HAND_COLLISION_VARIANT = "rubber_hull"
SPIDER_METHOD_ID = "E167A_zOnlyBody"
SOURCE_CONFIG_ID = "E167A_zOnlyBody_no_PRG"
E179_METHOD_ID = "E179_E167A_noPRG_box023_r1"
SCORING_CONTRACT_ID = "core4d-e179-12gate-tracking-v1"

CEM_SEED = 0
CEM_FULL_SAMPLES = 1024
CEM_FULL_OPT_STEPS = 32
CEM_CANARY_SAMPLES = 64
CEM_CANARY_OPT_STEPS = 4

CANARY_CASES = (
    "box023_20231008_045_p1",
    "box023_20231011_019_p2",
    "box023_20231011_018_p2",
)

WORKER_QUEUES = {
    "local-gpu0": (
        "box023_20231008_045_p1",
        "box023_20231011_018_p2",
        "box023_20231011_019_p2",
        "box023_20231008_046_p1",
    ),
    "a100-gpu2": (
        "box023_20231008_045_p2",
        "box023_20231020_040_p1",
        "box023_20231020_042_p2",
    ),
    "a100-gpu3": (
        "box023_20231011_018_p1",
        "box023_20231020_041_p2",
        "box023_20231020_039_p2",
    ),
    "a100-gpu6": (
        "box023_20231011_021_p1",
        "box023_20231020_041_p1",
        "box023_20231020_042_p1",
    ),
    "a100-gpu7": (
        "box023_20231011_021_p2",
        "box023_20231020_040_p2",
        "box023_20231008_046_p2",
    ),
}
EXPECTED_A100_GPUS = ("2", "3", "6", "7")

PHYSICS_GATES = (
    "fall",
    "body_z",
    "contact",
    "release",
    "hand_penetration",
    "lower_body",
)
TRACKING_GATES = (
    "root_pos",
    "root_ori",
    "hand_pos",
    "hand_ori",
    "object_pos",
    "object_ori",
)
ALL_GATES = PHYSICS_GATES + TRACKING_GATES
GATE_THRESHOLDS = {
    "body_z_err_p95_m_max": 0.20,
    "hand_object_physics_contact_in_mask_frac_min": 0.50,
    "hand_object_release_false_contact_3mm_frac_max": 0.30,
    "hand_object_physics_penetration_3mm_frame_frac_max": 0.30,
    "leg_penetration_frac_max": 0.10,
    "track_root_pos_err_cm_mean_max": 20.0,
    "track_root_ori_err_deg_mean_max": 20.0,
    "track_eef_pos_err_cm_mean_max": 20.0,
    "track_eef_ori_err_deg_mean_max": 20.0,
    "track_obj_pos_err_cm_mean_max": 20.0,
    "track_obj_ori_err_deg_mean_max": 10.0,
}

PRG_CONFIG_FIELDS = (
    "leg_object_penalty_scale",
    "leg_object_penalty_margin_m",
    "leg_object_penalty_geom_names",
    "leg_object_penalty_geom_ids",
    "leg_object_penalty_gate_source",
    "leg_object_penalty_start_eval_time",
    "leg_object_penalty_end_eval_time",
    "cem_leg_gate_enabled",
    "cem_leg_gate_geom_names",
    "cem_leg_gate_geom_ids",
    "cem_leg_gate_min_sdf_m",
    "cem_leg_gate_max_violation_pct",
    "cem_leg_gate_hard_floor_m",
    "cem_leg_gate_min_valid_frac",
    "cem_leg_gate_fallback",
)
PRG_DIAGNOSTIC_PREFIXES = (
    "cem_leg_gate_",
    "sample_leg_gate_",
    "leg_object_penalty_",
)


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def repo_path(value: str | Path) -> Path:
    """Localize a repo path, including historical absolute checkout paths."""

    path = Path(value)
    if not path.is_absolute():
        return REPO / path
    if path.exists():
        return path
    parts = path.parts
    for marker in ("example_datasets", "workspace", "logs", "examples"):
        if marker in parts:
            return REPO.joinpath(*parts[parts.index(marker) :])
    return path


def rel(value: str | Path) -> str:
    path = Path(value)
    try:
        return str(path.absolute().relative_to(REPO.absolute()))
    except ValueError:
        pass
    try:
        return str(path.resolve().relative_to(REPO.resolve()))
    except (OSError, ValueError):
        return str(value)


def sha256(value: str | Path) -> str:
    path = repo_path(value)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_tsv(value: str | Path) -> list[dict[str, str]]:
    with repo_path(value).open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def write_tsv(
    value: str | Path,
    rows: list[dict[str, Any]],
    fields: list[str] | None = None,
) -> None:
    output = repo_path(value)
    output.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=fields,
                delimiter="\t",
                lineterminator="\n",
            )
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        key: serial(row.get(key, ""))
                        for key in fields
                    }
                )
        Path(temporary).replace(output)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


def write_json(value: str | Path, payload: Any) -> None:
    output = repo_path(value)
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(
                payload,
                stream,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            stream.write("\n")
        Path(temporary).replace(output)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


def serial(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return value


def boolish(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def finite(value: Any, default: float) -> float:
    try:
        output = float(value)
    except (TypeError, ValueError):
        return default
    return output if math.isfinite(output) else default


def apply_12gate_scoring(item: dict[str, Any]) -> dict[str, Any]:
    """Apply the frozen E179 scoring adapter in-place and return ``item``."""

    release_applicable = boolish(item.get("release_gate_applicable", True))
    gates = {
        "fall": not boolish(item.get("fall_flag")),
        "body_z": finite(
            item.get("body_z_err_p95_m"), math.inf
        )
        <= GATE_THRESHOLDS["body_z_err_p95_m_max"],
        "contact": finite(
            item.get("hand_object_physics_contact_in_mask_frac"),
            -math.inf,
        )
        >= GATE_THRESHOLDS[
            "hand_object_physics_contact_in_mask_frac_min"
        ],
        "release": (not release_applicable)
        or finite(
            item.get(
                "hand_object_release_false_contact_3mm_frac"
            ),
            math.inf,
        )
        <= GATE_THRESHOLDS[
            "hand_object_release_false_contact_3mm_frac_max"
        ],
        "hand_penetration": finite(
            item.get(
                "hand_object_physics_penetration_3mm_frame_frac"
            ),
            math.inf,
        )
        <= GATE_THRESHOLDS[
            "hand_object_physics_penetration_3mm_frame_frac_max"
        ],
        "lower_body": finite(
            item.get("leg_penetration_frac"), math.inf
        )
        <= GATE_THRESHOLDS["leg_penetration_frac_max"],
        "root_pos": finite(
            item.get("track_root_pos_err_cm_mean"), math.inf
        )
        <= GATE_THRESHOLDS["track_root_pos_err_cm_mean_max"],
        "root_ori": finite(
            item.get("track_root_ori_err_deg_mean"), math.inf
        )
        <= GATE_THRESHOLDS["track_root_ori_err_deg_mean_max"],
        "hand_pos": finite(
            item.get("track_eef_pos_err_cm_mean"), math.inf
        )
        <= GATE_THRESHOLDS["track_eef_pos_err_cm_mean_max"],
        "hand_ori": finite(
            item.get("track_eef_ori_err_deg_mean"), math.inf
        )
        <= GATE_THRESHOLDS["track_eef_ori_err_deg_mean_max"],
        "object_pos": finite(
            item.get("track_obj_pos_err_cm_mean"), math.inf
        )
        <= GATE_THRESHOLDS["track_obj_pos_err_cm_mean_max"],
        "object_ori": finite(
            item.get("track_obj_ori_err_deg_mean"), math.inf
        )
        <= GATE_THRESHOLDS["track_obj_ori_err_deg_mean_max"],
    }
    for name, passed in gates.items():
        item[f"{name}_gate_pass"] = passed
    legacy_pass = all(gates[name] for name in PHYSICS_GATES)
    pass_12gate = all(gates[name] for name in ALL_GATES)
    item["legacy_physics6_pass"] = legacy_pass
    item["numeric_release_pass"] = pass_12gate
    item["numeric_release_pass_12gate"] = pass_12gate
    item["numeric_failure_modes"] = ",".join(
        name for name in ALL_GATES if not gates[name]
    )
    item["scoring_contract_id"] = SCORING_CONTRACT_ID
    item["tracking_gates_enabled"] = True
    return item


def worker_for_case(case_id: str) -> str:
    matches = [
        worker
        for worker, cases in WORKER_QUEUES.items()
        if case_id in cases
    ]
    if len(matches) != 1:
        raise ValueError(
            f"case must appear in exactly one worker queue: "
            f"{case_id} -> {matches}"
        )
    return matches[0]


def validate_queue_contract(case_ids: set[str]) -> None:
    queued = [
        case_id
        for cases in WORKER_QUEUES.values()
        for case_id in cases
    ]
    if len(queued) != EXPECTED_PAIRED_ROWS:
        raise ValueError(f"queue row count drift: {len(queued)}")
    if len(set(queued)) != EXPECTED_PAIRED_ROWS:
        raise ValueError("duplicate case in worker queues")
    if set(queued) != case_ids:
        raise ValueError(
            f"queue set mismatch: missing={sorted(case_ids-set(queued))} "
            f"extra={sorted(set(queued)-case_ids)}"
        )
    if len(WORKER_QUEUES["local-gpu0"]) != 4:
        raise ValueError("local queue must contain four rows")
    if any(
        len(WORKER_QUEUES[f"a100-gpu{gpu}"]) != 3
        for gpu in EXPECTED_A100_GPUS
    ):
        raise ValueError("each A100 queue must contain three rows")
