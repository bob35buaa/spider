#!/usr/bin/env python3
"""Evaluate E187's E178 legacy replays without importing an old evaluator."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import yaml
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO / "workspace/core4d/scripts"))

from eval.core.core_metrics import EvalConfig, evaluate_sequence, npz_qpos  # noqa: E402

RESULT_ROOT = REPO / "workspace/core4d/results/E187/s0_environment/e178_compat"
DEFAULT_MANIFEST = RESULT_ROOT / "execution_manifest.tsv"
DEFAULT_BASELINE = (
    REPO / "workspace/core4d/results/E178/s6_downstream/eval/full/e178_case_metrics.tsv"
)
DEFAULT_OUT_DIR = RESULT_ROOT / "eval_contract_audit_v2"
MONITORED_BODY_NAMES = (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
)
GATE_THRESHOLDS = {
    "body_z": 0.20,
    "contact": 0.50,
    "release": 0.30,
    "hand_penetration": 0.30,
    "lower_body": 0.10,
    "root_pos": 20.0,
    "root_ori": 20.0,
    "hand_pos": 20.0,
    "hand_ori": 20.0,
    "object_pos": 20.0,
    "object_ori": 10.0,
}
SEMANTIC_TOLERANCES = {
    "track_root_pos_err_cm_mean": 0.5,
    "track_eef_pos_err_cm_mean": 0.5,
    "track_obj_pos_err_cm_mean": 0.5,
    "track_root_ori_err_deg_mean": 0.5,
    "track_eef_ori_err_deg_mean": 0.5,
    "track_obj_ori_err_deg_mean": 0.5,
    "hand_object_physics_contact_in_mask_frac": 0.02,
    "hand_object_physics_penetration_3mm_frame_frac": 0.01,
}
ALLOWED_ADDED_CONFIG = {
    "object_distance_backend": "legacy_box",
    "object_distance_error_bound_m": 0.0,
    "object_distance_expected_asset_sha256": "",
    "object_distance_manifest": "",
    "object_distance_object_body_id": 31,
    "query_tape_enabled": False,
    "query_tape_max_chunks": 0,
    "query_tape_output_dir": "",
    "query_tape_record_geometry_state": False,
    "query_tape_record_start_sim_step": 0,
    "query_tape_run_id": "",
    "query_tape_stop_after_chunks": 0,
    "surface_band_continuation_far_scale_m": 0.05,
    "surface_band_continuation_far_weight": 0.25,
    "surface_band_continuation_near_scale_m": 0.015,
    "surface_band_continuation_near_weight": 0.75,
    "surface_band_continuation_smooth_delta_m": 0.001,
}
ALLOWED_CHANGED_CONFIG = {"output_dir", "video_output_path"}
EXACT_ARRAY_ATOL = 1e-5
COMPATIBILITY_CONTRACT = {
    "bucket003_20231018_003_p1": {
        "mode": "cross_gpu_semantic",
        "historical_device": "NVIDIA_A100_SXM4_80GB",
        "assigned_gpu": "remote-0",
        "gpu_id": "0",
        "compat_device": "remote_rtx6000_ada_gpu0",
    },
    "bucket004_20231002_021_p1": {
        "mode": "cross_gpu_semantic",
        "historical_device": "NVIDIA_A100_SXM4_80GB",
        "assigned_gpu": "remote-1",
        "gpu_id": "1",
        "compat_device": "remote_rtx6000_ada_gpu1",
    },
    "bucket007_20231020_055_p1": {
        "mode": "same_device_golden",
        "historical_device": "NVIDIA_GeForce_RTX_5090",
        "assigned_gpu": "local-0",
        "gpu_id": "0",
        "compat_device": "local_rtx5090",
    },
}
QUERY_TAPE_MANIFEST_FIELDS = (
    "source_e178_query_tape_manifest",
    "query_tape_manifest",
)
SAME_DEVICE_QUERY_KEYS = (
    "qpos",
    "rewards",
    "selected_indices",
    "sample_body_gate_valid_mask",
    "sample_hand_gate_valid_mask",
    "sample_leg_gate_valid_mask",
    "sample_posture_valid_mask",
    "sample_gate_valid_mask",
)


def repo_path(value: str | Path) -> Path:
    """Resolve repository-relative and workspace-symlinked paths."""
    path = Path(value)
    if path.exists():
        return path.resolve()
    text = str(value)
    for marker in ("example_datasets/", "workspace/", "logs/"):
        if marker in text:
            return REPO / (marker + text.split(marker, 1)[1])
    return path if path.is_absolute() else REPO / path


def read_tsv(path: Path) -> list[dict[str, str]]:
    """Read one TSV table."""
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def serial(value: Any) -> Any:
    """Normalize TSV/JSON scalar output."""
    if isinstance(value, (np.bool_, bool)):
        return str(bool(value)).lower()
    if isinstance(value, (np.floating, float)):
        return "" if not math.isfinite(float(value)) else float(value)
    if isinstance(value, np.integer):
        return int(value)
    return value


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write rows with stable first-seen field ordering."""
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=fields, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({key: serial(row.get(key, "")) for key in fields})


def finite(value: Any, default: float = math.nan) -> float:
    """Convert one value to a finite float or a caller-provided sentinel."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def values_equal(first: Any, second: Any) -> bool:
    """Compare YAML values while treating two NaNs as equal."""
    if (
        isinstance(first, float)
        and isinstance(second, float)
        and math.isnan(first)
        and math.isnan(second)
    ):
        return True
    return first == second


def config_whitelist_diff(row: dict[str, str]) -> dict[str, Any]:
    """Require only default-only additions and isolated output-path changes."""
    old_path = repo_path(row["source_e178_config_act"])
    new_path = repo_path(row["config_act"])
    old = yaml.safe_load(old_path.read_text(encoding="utf-8"))
    new = yaml.safe_load(new_path.read_text(encoding="utf-8"))
    added = {key: new[key] for key in sorted(set(new) - set(old))}
    removed = {key: old[key] for key in sorted(set(old) - set(new))}
    changed = {
        key: {"old": old[key], "new": new[key]}
        for key in sorted(set(old) & set(new))
        if not values_equal(old[key], new[key])
    }
    unexpected_added = {
        key: value
        for key, value in added.items()
        if key not in ALLOWED_ADDED_CONFIG
        or not values_equal(value, ALLOWED_ADDED_CONFIG[key])
    }
    unexpected_changed = {
        key: value
        for key, value in changed.items()
        if key not in ALLOWED_CHANGED_CONFIG
    }
    return {
        "pass": not removed and not unexpected_added and not unexpected_changed,
        "old_key_count": len(old),
        "new_key_count": len(new),
        "added": added,
        "removed": removed,
        "changed": changed,
        "unexpected_added": unexpected_added,
        "unexpected_changed": unexpected_changed,
    }


def npz_exact_diagnostic(row: dict[str, str]) -> dict[str, Any]:
    """Diagnose all common trajectory arrays without treating them as Gate S0."""
    old_path = repo_path(row["source_e178_result_npz"])
    new_path = repo_path(row["result_npz"])
    comparisons: list[dict[str, Any]] = []
    missing: list[str] = []
    shape_mismatch: list[str] = []
    with (
        np.load(old_path, allow_pickle=True) as old,
        np.load(new_path, allow_pickle=True) as new,
    ):
        missing = sorted(set(old.files) - set(new.files))
        for key in sorted(set(old.files) & set(new.files)):
            first = np.asarray(old[key])
            second = np.asarray(new[key])
            if first.shape != second.shape:
                shape_mismatch.append(key)
                continue
            if first.dtype.kind not in "biufc" or second.dtype.kind not in "biufc":
                continue
            if first.dtype.kind == "b" or second.dtype.kind == "b":
                max_abs = 0.0 if np.array_equal(first, second) else math.inf
            else:
                delta = np.abs(first.astype(np.float64) - second.astype(np.float64))
                max_abs = float(np.nanmax(delta)) if delta.size else 0.0
            comparisons.append(
                {
                    "key": key,
                    "shape": "x".join(map(str, first.shape)),
                    "max_abs": max_abs,
                    "pass": math.isfinite(max_abs) and max_abs <= EXACT_ARRAY_ATOL,
                }
            )
    failures = [item["key"] for item in comparisons if not item["pass"]]
    return {
        "pass": not missing and not shape_mismatch and not failures,
        "compared_arrays": len(comparisons),
        "missing": missing,
        "shape_mismatch": shape_mismatch,
        "failed_arrays": failures,
        "max_abs": max((item["max_abs"] for item in comparisons), default=math.nan),
        "comparisons": comparisons,
    }


def manifest_contract_diff(row: dict[str, str]) -> dict[str, Any]:
    """Check the frozen case-to-device route against historical provenance."""
    expected = COMPATIBILITY_CONTRACT.get(row["case_id"])
    if expected is None:
        return {
            "pass": False,
            "mode": "unknown",
            "historical_device": "unknown",
            "mismatches": {"case_id": {"expected": "representative case"}},
        }
    mismatches = {
        key: {"expected": expected[key], "actual": row.get(key, "")}
        for key in ("assigned_gpu", "gpu_id", "compat_device")
        if row.get(key, "") != expected[key]
    }
    return {
        "pass": not mismatches,
        "mode": expected["mode"],
        "historical_device": expected["historical_device"],
        "expected_replay_device": expected["compat_device"],
        "mismatches": mismatches,
    }


def _query_chunk_path(manifest_path: Path, value: str) -> Path:
    """Resolve a query chunk path relative to its manifest when necessary."""
    path = Path(value)
    if path.is_absolute():
        return repo_path(path)
    candidate = manifest_path.parent / path
    return candidate if candidate.exists() else repo_path(path)


def same_device_query_tape_diff(row: dict[str, str]) -> dict[str, Any]:
    """Apply the preregistered query/reward/valid/selected same-device gate."""
    missing_fields = [
        field for field in QUERY_TAPE_MANIFEST_FIELDS if not row.get(field)
    ]
    if missing_fields:
        return {
            "evidence_complete": False,
            "pass": False,
            "missing_manifest_fields": missing_fields,
            "missing_arrays": [],
            "failed_arrays": [],
            "compared_chunks": 0,
            "max_abs": math.nan,
        }

    manifest_paths = [repo_path(row[field]) for field in QUERY_TAPE_MANIFEST_FIELDS]
    missing_manifests = [str(path) for path in manifest_paths if not path.is_file()]
    if missing_manifests:
        return {
            "evidence_complete": False,
            "pass": False,
            "missing_manifest_fields": [],
            "missing_manifests": missing_manifests,
            "missing_arrays": [],
            "failed_arrays": [],
            "compared_chunks": 0,
            "max_abs": math.nan,
        }

    manifests = [
        json.loads(path.read_text(encoding="utf-8")) for path in manifest_paths
    ]
    manifest_errors: list[str] = []
    for label, payload in zip(("historical", "replay"), manifests, strict=True):
        if payload.get("status") != "COMPLETE":
            manifest_errors.append(f"{label}:status={payload.get('status')}")
        if not payload.get("chunks"):
            manifest_errors.append(f"{label}:no_chunks")
    old_chunks = manifests[0].get("chunks", [])
    new_chunks = manifests[1].get("chunks", [])
    if len(old_chunks) != len(new_chunks):
        manifest_errors.append(f"chunk_count:{len(old_chunks)}!={len(new_chunks)}")
    if manifest_errors:
        return {
            "evidence_complete": False,
            "pass": False,
            "manifest_errors": manifest_errors,
            "missing_manifest_fields": [],
            "missing_arrays": [],
            "failed_arrays": [],
            "compared_chunks": 0,
            "max_abs": math.nan,
        }

    missing_arrays: list[str] = []
    failed_arrays: list[str] = []
    max_abs = 0.0
    for chunk_index, (old_entry, new_entry) in enumerate(
        zip(old_chunks, new_chunks, strict=True)
    ):
        old_path = _query_chunk_path(manifest_paths[0], old_entry["path"])
        new_path = _query_chunk_path(manifest_paths[1], new_entry["path"])
        if not old_path.is_file() or not new_path.is_file():
            missing_arrays.append(f"chunk{chunk_index}:file")
            continue
        with (
            np.load(old_path, allow_pickle=False) as old,
            np.load(new_path, allow_pickle=False) as new,
        ):
            for key in SAME_DEVICE_QUERY_KEYS:
                label = f"chunk{chunk_index}:{key}"
                if key not in old.files or key not in new.files:
                    missing_arrays.append(label)
                    continue
                first = np.asarray(old[key])
                second = np.asarray(new[key])
                if first.shape != second.shape:
                    failed_arrays.append(f"{label}:shape")
                    continue
                if key in {"qpos", "rewards"}:
                    delta = np.abs(first.astype(np.float64) - second.astype(np.float64))
                    difference = float(np.nanmax(delta)) if delta.size else 0.0
                    max_abs = max(max_abs, difference)
                    if not math.isfinite(difference) or difference > EXACT_ARRAY_ATOL:
                        failed_arrays.append(label)
                elif not np.array_equal(first, second):
                    failed_arrays.append(label)
    evidence_complete = not missing_arrays
    return {
        "evidence_complete": evidence_complete,
        "pass": evidence_complete and not failed_arrays,
        "missing_manifest_fields": [],
        "missing_arrays": missing_arrays,
        "failed_arrays": failed_arrays,
        "compared_chunks": len(old_chunks),
        "max_abs": max_abs,
    }


def compatibility_gate(
    *,
    mode: str,
    manifest_contract_pass: bool,
    config_pass: bool,
    decision_match: bool,
    same_device_golden_pass: bool,
    semantic_pass: bool,
) -> bool:
    """Route each row through exactly its preregistered compatibility gate."""
    common = manifest_contract_pass and config_pass and decision_match
    if mode == "same_device_golden":
        return common and same_device_golden_pass
    if mode == "cross_gpu_semantic":
        return common and semantic_pass
    return False


def person_idx(row: dict[str, str]) -> int:
    """Resolve the CORE4D actor index."""
    return 0 if row["case_id"].lower().endswith("_p1") else 1


def reference_qpos(trajectory: Path, scene_xml: Path) -> np.ndarray:
    """Convert the kinematic freejoint reference to the scene's Euler object qpos."""
    with np.load(trajectory, allow_pickle=True) as payload:
        qpos = np.asarray(payload["qpos"], dtype=np.float64)
    if qpos.ndim == 3:
        qpos = qpos[:, 0, :]
    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    if qpos.ndim != 2:
        raise ValueError(f"invalid reference qpos shape {qpos.shape}")
    if qpos.shape[1] == model.nq:
        return qpos.copy()
    nq_robot = model.nq - 6
    if qpos.shape[1] < nq_robot + 7:
        raise ValueError(f"cannot convert reference qpos={qpos.shape} to nq={model.nq}")
    object_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    body_quat = model.body_quat[object_id]
    body_rotation = Rotation.from_quat(
        [body_quat[1], body_quat[2], body_quat[3], body_quat[0]]
    )
    object_quat = qpos[:, nq_robot + 3 : nq_robot + 7]
    object_xyzw = np.column_stack(
        [object_quat[:, 1], object_quat[:, 2], object_quat[:, 3], object_quat[:, 0]]
    )
    convention = "XYZ"
    meta = scene_xml.with_name("scene_act_meta.json")
    if meta.is_file():
        convention = str(
            json.loads(meta.read_text(encoding="utf-8")).get("euler_convention", "XYZ")
        )
    converted = np.zeros((len(qpos), model.nq), dtype=np.float64)
    converted[:, :nq_robot] = qpos[:, :nq_robot]
    converted[:, nq_robot : nq_robot + 3] = body_rotation.inv().apply(
        qpos[:, nq_robot : nq_robot + 3] - model.body_pos[object_id][np.newaxis, :]
    )
    converted[:, nq_robot + 3 : nq_robot + 6] = (
        body_rotation.inv() * Rotation.from_quat(object_xyzw)
    ).as_euler(convention)
    return converted


def body_positions(
    model: mujoco.MjModel, qpos: np.ndarray, body_ids: list[int]
) -> np.ndarray:
    """Run CPU FK and return selected body positions."""
    data = mujoco.MjData(model)
    output = np.zeros((len(qpos), len(body_ids), 3), dtype=np.float64)
    for frame, values in enumerate(qpos):
        data.qpos[:] = values
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        output[frame] = data.xpos[body_ids]
    return output


def body_z_p95(qpos_path: Path, scene_xml: Path, trajectory: Path) -> float:
    """Return the frozen four-body absolute Z-error p95."""
    sim_qpos, _ = npz_qpos(qpos_path)
    ref_qpos = reference_qpos(trajectory, scene_xml)
    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    body_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        for name in MONITORED_BODY_NAMES
    ]
    if any(body_id < 0 for body_id in body_ids):
        raise ValueError("scene missing a monitored body")
    frames = min(len(sim_qpos), len(ref_qpos))
    difference = np.abs(
        body_positions(model, sim_qpos[:frames], body_ids)[..., 2]
        - body_positions(model, ref_qpos[:frames], body_ids)[..., 2]
    )
    return float(np.percentile(difference, 95))


def release_applicable(contact_mask: Path, actor: int, frame_count: int) -> bool:
    """Return whether the frozen 3cm mask has a post-contact release window."""
    with np.load(contact_mask, allow_pickle=True) as payload:
        if "spider_contact_mask_3cm" not in payload.files:
            return False
        mask = np.asarray(payload["spider_contact_mask_3cm"])
    frames = min(frame_count, len(mask))
    active = np.any(mask[:frames, actor, :].astype(bool), axis=1)
    return bool(active.any() and int(np.flatnonzero(active)[-1]) < frames - 1)


def apply_gates(item: dict[str, Any]) -> None:
    """Apply the frozen E178 six physics plus six tracking gates."""
    applicable = bool(item["release_gate_applicable"])
    gates = {
        "fall": not bool(item["fall_flag"]),
        "body_z": finite(item["body_z_err_p95_m"], math.inf)
        <= GATE_THRESHOLDS["body_z"],
        "contact": finite(item["hand_object_physics_contact_in_mask_frac"], -math.inf)
        >= GATE_THRESHOLDS["contact"],
        "release": (not applicable)
        or finite(item["hand_object_release_false_contact_3mm_frac"], math.inf)
        <= GATE_THRESHOLDS["release"],
        "hand_penetration": finite(
            item["hand_object_physics_penetration_3mm_frame_frac"], math.inf
        )
        <= GATE_THRESHOLDS["hand_penetration"],
        "lower_body": finite(item["leg_penetration_frac"], math.inf)
        <= GATE_THRESHOLDS["lower_body"],
        "root_pos": finite(item["track_root_pos_err_cm_mean"], math.inf)
        <= GATE_THRESHOLDS["root_pos"],
        "root_ori": finite(item["track_root_ori_err_deg_mean"], math.inf)
        <= GATE_THRESHOLDS["root_ori"],
        "hand_pos": finite(item["track_eef_pos_err_cm_mean"], math.inf)
        <= GATE_THRESHOLDS["hand_pos"],
        "hand_ori": finite(item["track_eef_ori_err_deg_mean"], math.inf)
        <= GATE_THRESHOLDS["hand_ori"],
        "object_pos": finite(item["track_obj_pos_err_cm_mean"], math.inf)
        <= GATE_THRESHOLDS["object_pos"],
        "object_ori": finite(item["track_obj_ori_err_deg_mean"], math.inf)
        <= GATE_THRESHOLDS["object_ori"],
    }
    for gate, passed in gates.items():
        item[f"{gate}_gate_pass"] = bool(passed)
    item["numeric_release_pass"] = all(gates.values())
    item["numeric_failure_modes"] = ",".join(
        gate for gate, passed in gates.items() if not passed
    )


def evaluate_row(row: dict[str, str], baseline: dict[str, str]) -> dict[str, Any]:
    """Evaluate one completed replay and compare it with frozen E178 metrics."""
    qpos_path = repo_path(row["outdir_npz"])
    scene = repo_path(row["scene_act"])
    trajectory = repo_path(row["trajectory"])
    contact_mask = repo_path(row["contact_mask"])
    actor = person_idx(row)
    item = evaluate_sequence(
        row=row,
        method="E178_legacy_replay_under_E187_code",
        hand_collision_variant_id=row["hand_collision_variant_id"],
        qpos_path=qpos_path,
        scene_xml=scene,
        config=EvalConfig(),
        kin_ref_path=trajectory,
        contact_mask_path=contact_mask,
        person_idx=actor,
    )
    sim_qpos, _ = npz_qpos(qpos_path)
    item["body_z_err_p95_m"] = body_z_p95(qpos_path, scene, trajectory)
    item["release_gate_applicable"] = release_applicable(
        contact_mask, actor, len(sim_qpos)
    )
    apply_gates(item)
    config_diff = config_whitelist_diff(row)
    contract = manifest_contract_diff(row)
    exact = npz_exact_diagnostic(row)
    same_device = (
        same_device_query_tape_diff(row)
        if contract["mode"] == "same_device_golden"
        else {
            "evidence_complete": False,
            "pass": False,
            "missing_manifest_fields": [],
            "missing_arrays": [],
            "failed_arrays": [],
            "compared_chunks": 0,
            "max_abs": math.nan,
        }
    )
    item["compatibility_mode"] = contract["mode"]
    item["historical_device"] = contract["historical_device"]
    item["expected_replay_device"] = contract.get("expected_replay_device", "")
    item["manifest_device_contract_pass"] = contract["pass"]
    item["manifest_device_contract_mismatches"] = ",".join(contract["mismatches"])
    item["config_whitelist_pass"] = config_diff["pass"]
    item["config_added_keys"] = ",".join(config_diff["added"])
    item["config_unexpected_added"] = ",".join(config_diff["unexpected_added"])
    item["config_unexpected_changed"] = ",".join(config_diff["unexpected_changed"])
    item["all_common_array_diagnostic_pass"] = exact["pass"]
    item["all_common_array_diagnostic_count"] = exact["compared_arrays"]
    item["all_common_array_diagnostic_max_abs"] = exact["max_abs"]
    item["all_common_array_diagnostic_failures"] = ",".join(exact["failed_arrays"])
    item["same_device_evidence_complete"] = same_device["evidence_complete"]
    item["same_device_golden_pass"] = same_device["pass"]
    item["same_device_query_chunks"] = same_device["compared_chunks"]
    item["same_device_query_max_abs"] = same_device["max_abs"]
    item["same_device_missing_manifest_fields"] = ",".join(
        same_device.get("missing_manifest_fields", [])
    )
    item["same_device_missing_arrays"] = ",".join(same_device.get("missing_arrays", []))
    item["same_device_failed_arrays"] = ",".join(same_device.get("failed_arrays", []))
    item["baseline_numeric_release_pass"] = baseline["numeric_release_pass"]
    item["numeric_decision_match"] = (
        str(item["numeric_release_pass"]).lower()
        == str(baseline["numeric_release_pass"]).lower()
    )
    semantic_pass = True
    for metric, tolerance in SEMANTIC_TOLERANCES.items():
        current = finite(item.get(metric))
        historical = finite(baseline.get(metric))
        delta = abs(current - historical)
        passed = math.isfinite(delta) and delta <= tolerance
        item[f"baseline_{metric}"] = historical
        item[f"abs_delta_{metric}"] = delta
        item[f"semantic_{metric}_pass"] = passed
        semantic_pass = semantic_pass and passed
    item["semantic_tolerance_pass"] = semantic_pass
    item["compat_row_pass"] = compatibility_gate(
        mode=contract["mode"],
        manifest_contract_pass=contract["pass"],
        config_pass=config_diff["pass"],
        decision_match=item["numeric_decision_match"],
        same_device_golden_pass=same_device["pass"],
        semantic_pass=semantic_pass,
    )
    item["case_id"] = row["case_id"]
    item["variant"] = row["variant"]
    return item


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--require-all", action="store_true")
    args = parser.parse_args()
    rows = read_tsv(repo_path(args.manifest))
    baseline = {row["case_id"]: row for row in read_tsv(repo_path(args.baseline))}
    contract_audit = [
        {"case_id": row["case_id"], **manifest_contract_diff(row)} for row in rows
    ]
    ready: list[dict[str, str]] = []
    not_ready: list[dict[str, Any]] = []
    for row in rows:
        missing = [
            key
            for key in ("result_npz", "outdir_npz", "config_act")
            if not repo_path(row[key]).is_file()
        ]
        if row["status"] != "run_complete_pending_eval":
            missing.append(f"status:{row['status']}")
        if row["case_id"] not in baseline:
            missing.append("baseline")
        if missing:
            not_ready.append({"case_id": row["case_id"], "reasons": ",".join(missing)})
        else:
            ready.append(row)

    metrics: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for row in ready:
        try:
            metrics.append(evaluate_row(row, baseline[row["case_id"]]))
        except Exception as error:  # noqa: BLE001
            errors.append(
                {
                    "case_id": row["case_id"],
                    "error": f"{type(error).__name__}: {error}",
                }
            )
    summary = {
        "experiment_id": "E187",
        "stage": "S0_E178_compatibility",
        "expected_rows": 3,
        "manifest_rows": len(rows),
        "evaluated": len(metrics),
        "not_ready": len(not_ready),
        "errors": len(errors),
        "manifest_device_contract_pass": all(item["pass"] for item in contract_audit),
        "manifest_device_contract_failures": [
            item["case_id"] for item in contract_audit if not item["pass"]
        ],
        "compat_pass": sum(bool(row["compat_row_pass"]) for row in metrics),
        "status": (
            "PASS"
            if len(metrics) == 3
            and not not_ready
            and not errors
            and all(row["compat_row_pass"] for row in metrics)
            else "INCOMPLETE_OR_FAIL"
        ),
        "failure_counts": dict(
            Counter(
                key
                for row in metrics
                for key in (
                    "config" if not row["config_whitelist_pass"] else "",
                    "device_contract"
                    if not row["manifest_device_contract_pass"]
                    else "",
                    "numeric" if not row["numeric_decision_match"] else "",
                    "same_device_evidence"
                    if row["compatibility_mode"] == "same_device_golden"
                    and not row["same_device_golden_pass"]
                    else "",
                    "semantic"
                    if row["compatibility_mode"] == "cross_gpu_semantic"
                    and not row["semantic_tolerance_pass"]
                    else "",
                )
                if key
            )
        ),
        "evidence_boundary": (
            "Gate S0 is routed by device: same-device requires paired query-tape qpos, "
            "rewards, valid masks, and selected indices; cross-GPU requires semantic "
            "tolerance. Historical E178 artifacts have no query-tape manifest or direct "
            "selected-index array, so same-device evidence fails closed. All-common-array "
            "trajectory comparison remains diagnostic only."
        ),
    }
    out_dir = repo_path(args.out_dir)
    write_tsv(out_dir / "e187_e178_compat_metrics.tsv", metrics)
    write_tsv(out_dir / "not_ready.tsv", not_ready)
    write_tsv(out_dir / "errors.tsv", errors)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if errors:
        return 1
    if args.require_all and summary["status"] != "PASS":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
