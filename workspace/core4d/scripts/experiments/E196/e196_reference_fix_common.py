#!/usr/bin/env python3
"""Frozen contracts and I/O helpers for E196 reference metadata repair."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[5]
SCRIPT_DIR = REPO / "workspace/core4d/scripts/experiments/E196"
RESULTS = REPO / "workspace/core4d/results/E196"
TASK_ROOT = REPO / "example_datasets/processed/core4d/unitree_g1/humanoid_object"
E194_RESULTS = REPO / "workspace/core4d/results/E194"
E194_AUDIT = E194_RESULTS / (
    "s6_downstream/eval/full_g1_expansion/"
    "e194_g1_object_orientation_reference_conversion_audit.tsv"
)
E194_MANIFEST = E194_RESULTS / "s6_downstream/manifests/g1_expansion_full_manifest.tsv"

CASE_SET_SHA256 = "b7255fbb0bc67dde9fb8fd0c19cd5b2285a3aee8e74ddc0b8de4f392b2941dac"
OBJECT_COUNTS = {"box001": 21, "box023": 8, "box021": 0}
N_CASES = 29
SCENE_NAME = "scene_act_E194_G1_expansion_rubberHull_PRG_gravcomp"
SIDECAR_FILE = f"{SCENE_NAME}.xml"
KP_POS = 500.0
KP_ROT = 50.0
GRAVCOMP = 1.0
FULL_SAMPLES = 1024
FULL_OPT_STEPS = 32
CEM_SEED = 0
REFERENCE_CONTRACT_VERSION = "E196-v1-fail-closed-compiled-axis"
MODEL_ARRAYS = (
    "body_mass", "body_inertia", "body_pos", "body_quat", "body_ipos", "body_iquat",
    "body_gravcomp", "geom_type", "geom_size", "geom_pos", "geom_quat",
    "geom_friction", "geom_condim", "geom_contype", "geom_conaffinity",
    "geom_solref", "geom_solimp", "geom_margin", "geom_gap", "jnt_type",
    "jnt_axis", "jnt_range", "dof_damping", "dof_armature", "pair_geom1",
    "pair_geom2", "pair_solref", "pair_margin", "pair_gap", "pair_dim",
    "actuator_gainprm", "actuator_biasprm", "actuator_trnid",
)

WORKERS = ("local-gpu0", "ada-gpu0", "ada-gpu1")
WORKER_GPU = {"local-gpu0": "0", "ada-gpu0": "0", "ada-gpu1": "1"}
WAVE0 = {
    "local-gpu0": ("box001_20231003_2_041_p1",),
    "ada-gpu0": ("box001_20231020_014_p1",),
    "ada-gpu1": ("box001_20231020_014_p2",),
}
REMAINING = {
    "local-gpu0": (
        "box001_20231003_1_039_p1",
        "box001_20231003_1_041_p1",
        "box001_20231003_2_037_p1",
        "box001_20231003_2_039_p1",
        "box001_20231023_107_p2",
        "box001_20231023_109_p2",
        "box023_20231008_046_p1",
        "box023_20231011_021_p1",
        "box023_20231020_042_p1",
    ),
    "ada-gpu0": (
        "box001_20231003_1_040_p1",
        "box001_20231003_1_042_p1",
        "box001_20231003_2_037_p2",
        "box001_20231020_010_p1",
        "box001_20231023_108_p1",
        "box001_20231023_110_p1",
        "box023_20231008_046_p2",
        "box023_20231020_040_p1",
        "box023_20231020_042_p2",
    ),
    "ada-gpu1": (
        "box001_20231003_1_040_p2",
        "box001_20231003_1_042_p2",
        "box001_20231003_2_038_p1",
        "box001_20231020_011_p1",
        "box001_20231023_108_p2",
        "box001_20231023_110_p2",
        "box023_20231011_019_p2",
        "box023_20231020_040_p2",
    ),
}

MANIFEST_DIR = RESULTS / "s6_downstream/manifests"
FULL_MANIFEST = MANIFEST_DIR / "reference_fix_full_manifest.tsv"
WAVE0_MANIFEST = MANIFEST_DIR / "reference_fix_wave0_manifest.tsv"
REMAINING_MANIFEST = MANIFEST_DIR / "reference_fix_remaining_manifest.tsv"
CASE_AUTHORITY = MANIFEST_DIR / "reference_fix_case_authority.tsv"
SNAPSHOT_ROOT = RESULTS / "scene_snapshot/reference_fix"
PREFLIGHT_ROOT = RESULTS / "s6_downstream/evidence/reference_fix/preflight"

FIELDS = [
    "ordinal", "case_id", "object_key", "retarget_variant_id", "arm", "wave",
    "worker", "execution_profile", "assigned_gpu", "gpu_id", "target_task",
    "target_scene", "trajectory", "trajectory_sha256", "contact_mask",
    "contact_mask_sha256", "override_id", "override_path", "override_sha256",
    "scene_act", "scene_name", "effective_scene_sha256", "scene_act_meta_path",
    "scene_act_meta_sha256", "meta_repair_action", "resolved_euler_convention",
    "compiled_xml_axis_sequence", "reference_contract_version",
    "compiled_physical_sha256",
    "axis_target_vs_raw_ori_err_deg_max", "axis_target_vs_raw_pos_err_cm_max",
    "kp_pos", "kp_rot", "gravcomp", "cem_samples", "cem_opt_steps", "cem_seed",
    "extra_overrides", "e194_variant", "e194_result_npz", "e194_outdir_npz",
    "e194_config_act", "e194_config_sha256", "e194_video", "e194_log",
    "variant", "result_npz", "outdir_npz", "config_act", "video", "log",
    "status", "failure_mode", "execution_mode", "updated_at",
]


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def repo_path(value: str | Path) -> Path:
    path = Path(str(value))
    if path.is_file() or path.is_dir():
        return path
    text = str(value)
    for marker in ("example_datasets/", "workspace/", "logs/"):
        if marker in text:
            return REPO / (marker + text.split(marker, 1)[1])
    return path if path.is_absolute() else REPO / path


def rel(value: str | Path) -> str:
    path = repo_path(value)
    try:
        return path.resolve().relative_to(REPO.resolve()).as_posix()
    except ValueError:
        return Path(str(value)).as_posix()


def require_file(value: str | Path, label: str) -> Path:
    path = repo_path(value)
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"missing {label}: {value}")
    return path


def sha256(value: str | Path) -> str:
    path = require_file(value, "sha256 input")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bytes_sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def case_set_sha256(case_ids: list[str] | tuple[str, ...] | set[str]) -> str:
    payload = ("\n".join(sorted(case_ids)) + "\n").encode()
    return bytes_sha256(payload)


def compiled_physical_sha256(model: Any) -> str:
    """Hash the exact compiled physical arrays frozen by E196."""

    import numpy as np

    digest = hashlib.sha256()
    for name in MODEL_ARRAYS:
        array = np.ascontiguousarray(np.asarray(getattr(model, name)))
        header = json.dumps(
            {"name": name, "dtype": str(array.dtype), "shape": array.shape},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        digest.update(array.tobytes())
    return digest.hexdigest()


def read_tsv(value: str | Path) -> list[dict[str, str]]:
    with require_file(value, "tsv").open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def read_with_fields(value: str | Path) -> tuple[list[dict[str, str]], list[str]]:
    with require_file(value, "tsv").open(encoding="utf-8", newline="") as stream:
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


def write_tsv(
    value: str | Path,
    rows: list[dict[str, Any]],
    fields: list[str] | None = None,
) -> None:
    path = repo_path(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=fields,
            delimiter="\t",
            lineterminator="\n",
            extrasaction="ignore",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({key: serial(row.get(key, "")) for key in fields})


def write_json(value: str | Path, payload: Any) -> None:
    path = repo_path(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def planned_case_ids() -> tuple[str, ...]:
    return tuple(
        case_id
        for worker in WORKERS
        for case_id in (*WAVE0[worker], *REMAINING[worker])
    )


def assignment(case_id: str) -> tuple[str, str]:
    for worker in WORKERS:
        if case_id in WAVE0[worker]:
            return worker, "wave0"
        if case_id in REMAINING[worker]:
            return worker, "remaining"
    raise KeyError(f"case is outside frozen E196 queues: {case_id}")


def validate_frozen_queues(case_ids: set[str] | None = None) -> None:
    planned = planned_case_ids()
    if len(planned) != N_CASES or len(set(planned)) != N_CASES:
        raise ValueError("E196 queue cardinality/uniqueness contract failed")
    if case_set_sha256(set(planned)) != CASE_SET_SHA256:
        raise ValueError("E196 queue case-set SHA contract failed")
    counts = Counter(case_id.split("_", 1)[0] for case_id in planned)
    if counts != Counter({"box001": 21, "box023": 8}):
        raise ValueError(f"E196 object counts differ: {dict(counts)}")
    totals = Counter()
    for worker in WORKERS:
        totals[worker] = len(WAVE0[worker]) + len(REMAINING[worker])
        seen_box023 = False
        for case_id in (*WAVE0[worker], *REMAINING[worker]):
            if case_id.startswith("box023_"):
                seen_box023 = True
            elif seen_box023:
                raise ValueError(f"{worker} is not box001-first")
    if totals != Counter({"local-gpu0": 10, "ada-gpu0": 10, "ada-gpu1": 9}):
        raise ValueError(f"E196 worker totals differ: {dict(totals)}")
    if case_ids is not None and set(planned) != case_ids:
        raise ValueError("E196 queue set differs from audit authority")


validate_frozen_queues()
