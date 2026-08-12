#!/usr/bin/env python3
"""Build and snapshot the exact 29-row E196 corrected-reference manifest."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e196_reference_fix_common as C  # noqa: E402

from spider.simulators.scene_act_reference import (  # noqa: E402
    object_hinge_axis_sequence,
    resolve_scene_act_reference,
)


def raw_reference(path: Path, case_id: str) -> np.ndarray:
    with np.load(path, allow_pickle=True) as archive:
        qpos = np.asarray(archive["qpos"], dtype=np.float64)
    if qpos.ndim == 3:
        person = 0 if case_id.endswith("_p1") else 1
        qpos = qpos[:, min(person, qpos.shape[1] - 1), :]
    if qpos.ndim != 2:
        raise ValueError(f"unsupported reference shape {qpos.shape}: {path}")
    return qpos


def convert_reference(
    reference: np.ndarray,
    model: mujoco.MjModel,
    body_id: int,
    convention: str,
) -> np.ndarray:
    nq_robot = int(model.nq) - 6
    body_quat = np.asarray(model.body_quat[body_id], dtype=np.float64)
    body_rot = Rotation.from_quat(np.r_[body_quat[1:], body_quat[0]])
    object_pos = reference[:, nq_robot : nq_robot + 3]
    object_quat = reference[:, nq_robot + 3 : nq_robot + 7]
    object_rot = Rotation.from_quat(np.c_[object_quat[:, 1:], object_quat[:, 0]])
    converted = np.zeros((len(reference), int(model.nq)), dtype=np.float64)
    converted[:, :nq_robot] = reference[:, :nq_robot]
    converted[:, nq_robot : nq_robot + 3] = body_rot.inv().apply(
        object_pos - np.asarray(model.body_pos[body_id])[None, :]
    )
    converted[:, nq_robot + 3 : nq_robot + 6] = (
        body_rot.inv() * object_rot
    ).as_euler(convention)
    return converted


def world_pose(
    model: mujoco.MjModel, body_id: int, qpos: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    data = mujoco.MjData(model)
    positions = np.empty((len(qpos), 3), dtype=np.float64)
    quaternions = np.empty((len(qpos), 4), dtype=np.float64)
    for index, frame in enumerate(qpos):
        data.qpos[:] = frame
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        positions[index] = data.xpos[body_id]
        quaternions[index] = data.xquat[body_id]
    return positions, quaternions


def quaternion_angle_deg(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = left / np.linalg.norm(left, axis=1, keepdims=True)
    right = right / np.linalg.norm(right, axis=1, keepdims=True)
    dots = np.abs(np.sum(left * right, axis=1))
    return np.degrees(2.0 * np.arccos(np.clip(dots, -1.0, 1.0)))


def expected_meta_bytes(convention: str) -> bytes:
    return (json.dumps({"euler_convention": convention}, sort_keys=True) + "\n").encode()


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        Path(temporary).replace(path)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


def repair_and_audit_meta(
    scene: Path,
    trajectory: Path,
    case_id: str,
    *,
    apply: bool,
    previous_action: str = "",
) -> dict[str, Any]:
    model = mujoco.MjModel.from_xml_path(str(scene))
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    if body_id < 0:
        raise ValueError(f"scene has no object body: {scene}")
    convention = object_hinge_axis_sequence(model, body_id)
    meta_path = scene.with_name("scene_act_meta.json")
    existed = meta_path.is_file() and meta_path.stat().st_size > 0
    action = (
        previous_action
        if previous_action in {"generated_from_compiled_axes", "preserved_existing"}
        else "preserved_existing"
    )
    if existed:
        resolve_scene_act_reference(scene, model, emit_log=False)
    else:
        action = "generated_from_compiled_axes"
        if not apply:
            raise RuntimeError(f"metadata repair requires --apply: {case_id}")
        atomic_write(meta_path, expected_meta_bytes(convention))
        model = mujoco.MjModel.from_xml_path(str(scene))
        resolved = resolve_scene_act_reference(scene, model, emit_log=False)
        body_id = resolved.object_body_id
    resolved = resolve_scene_act_reference(scene, model, emit_log=False)
    reference = raw_reference(trajectory, case_id)
    converted = convert_reference(reference, model, body_id, resolved.convention)
    position, quaternion = world_pose(model, body_id, converted)
    nq_robot = int(model.nq) - 6
    raw_pos = reference[:, nq_robot : nq_robot + 3]
    raw_quat = reference[:, nq_robot + 3 : nq_robot + 7]
    return {
        "scene_act_meta_path": C.rel(meta_path),
        "scene_act_meta_sha256": C.sha256(meta_path),
        "meta_existed_before": existed,
        "meta_repair_action": action,
        "resolved_euler_convention": resolved.convention,
        "compiled_xml_axis_sequence": resolved.xml_axis_sequence,
        "compiled_physical_sha256": C.compiled_physical_sha256(model),
        "axis_target_vs_raw_ori_err_deg_max": float(
            np.max(quaternion_angle_deg(quaternion, raw_quat))
        ),
        "axis_target_vs_raw_pos_err_cm_max": float(
            np.max(np.linalg.norm(position - raw_pos, axis=1)) * 100.0
        ),
    }


def artifact_paths(case_id: str) -> dict[str, str]:
    variant = f"E196_{case_id}_G1_reference_fix"
    root = "workspace/core4d/results/E196/s6_downstream/cem/full_reference_fix"
    return {
        "variant": variant,
        "result_npz": f"{root}/{variant}.npz",
        "outdir_npz": f"{root}/{variant}_outdir/trajectory_mjwp_act.npz",
        "config_act": f"{root}/{variant}_outdir/config_act.yaml",
        "video": f"workspace/core4d/results/E196/s6_downstream/render/full_reference_fix/{variant}.mp4",
        "log": f"logs/E196/cem/full_reference_fix/{variant}.log",
    }


def load_authorities() -> tuple[list[dict[str, str]], dict[str, dict[str, str]]]:
    audit = C.read_tsv(C.E194_AUDIT)
    mismatch = [
        row
        for row in audit
        if row.get("runtime_convention_matches_xml_axes", "").lower() == "false"
    ]
    case_ids = {row["case_id"] for row in mismatch}
    C.validate_frozen_queues(case_ids)
    if len(mismatch) != C.N_CASES or C.case_set_sha256(case_ids) != C.CASE_SET_SHA256:
        raise ValueError("E194 mismatch authority drift")
    e194_rows = {row["case_id"]: row for row in C.read_tsv(C.E194_MANIFEST)}
    if not case_ids <= e194_rows.keys():
        raise ValueError(f"E194 manifest lacks cases: {sorted(case_ids - e194_rows.keys())}")
    return sorted(mismatch, key=lambda row: row["case_id"]), e194_rows


def make_row(
    audit: dict[str, str],
    source: dict[str, str],
    meta: dict[str, Any],
) -> dict[str, Any]:
    case_id = audit["case_id"]
    worker, wave = C.assignment(case_id)
    e194_config = C.repo_path(source["config_act"])
    row: dict[str, Any] = {
        "case_id": case_id,
        "object_key": source["object_key"],
        "retarget_variant_id": source["retarget_variant_id"],
        "arm": "G1_reference_corrected",
        "wave": wave,
        "worker": worker,
        "execution_profile": worker,
        "assigned_gpu": C.WORKER_GPU[worker],
        "gpu_id": "",
        "target_task": source["target_task"],
        "target_scene": source["target_scene"],
        "trajectory": source["trajectory"],
        "trajectory_sha256": source["trajectory_sha256"],
        "contact_mask": source["contact_mask"],
        "contact_mask_sha256": source["contact_mask_sha256"],
        "override_id": source["override_id"],
        "override_path": source["override_path"],
        "override_sha256": source["override_sha256"],
        "scene_act": source["scene_act"],
        "scene_name": source["scene_name"],
        "effective_scene_sha256": source["effective_scene_sha256"],
        "reference_contract_version": C.REFERENCE_CONTRACT_VERSION,
        "kp_pos": C.KP_POS,
        "kp_rot": C.KP_ROT,
        "gravcomp": C.GRAVCOMP,
        "cem_samples": C.FULL_SAMPLES,
        "cem_opt_steps": C.FULL_OPT_STEPS,
        "cem_seed": C.CEM_SEED,
        "extra_overrides": source["extra_overrides"],
        "e194_variant": source["variant"],
        "e194_result_npz": source["result_npz"],
        "e194_outdir_npz": source["outdir_npz"],
        "e194_config_act": source["config_act"],
        "e194_config_sha256": C.sha256(e194_config),
        "e194_video": source["video"],
        "e194_log": source["log"],
        "status": "READY_FOR_FULL",
        "failure_mode": "",
        "execution_mode": wave,
        "updated_at": C.now(),
    }
    row.update(meta)
    row.update(artifact_paths(case_id))
    return row


def snapshot_row(row: dict[str, Any], git_head: str, dirty: bool) -> dict[str, Any]:
    case_id = str(row["case_id"])
    destination = C.SNAPSHOT_ROOT / case_id
    destination.mkdir(parents=True, exist_ok=True)
    scene = C.repo_path(row["scene_act"])
    meta = C.repo_path(row["scene_act_meta_path"])
    scene_copy = destination / scene.name
    meta_copy = destination / meta.name
    shutil.copy2(scene, scene_copy)
    shutil.copy2(meta, meta_copy)
    return {
        "case_id": case_id,
        "git_head": git_head,
        "git_dirty": dirty,
        "source_scene": C.rel(scene),
        "snapshot_scene": C.rel(scene_copy),
        "scene_sha256": C.sha256(scene_copy),
        "source_meta": C.rel(meta),
        "snapshot_meta": C.rel(meta_copy),
        "meta_sha256": C.sha256(meta_copy),
        "meta_repair_action": row["meta_repair_action"],
    }


def build(*, apply: bool, snapshot: bool) -> int:
    mismatch, e194_rows = load_authorities()
    previous_actions = {}
    if C.FULL_MANIFEST.is_file():
        previous_actions = {
            row["case_id"]: row.get("meta_repair_action", "")
            for row in C.read_tsv(C.FULL_MANIFEST)
        }
    rows: list[dict[str, Any]] = []
    authority_rows: list[dict[str, Any]] = []
    for audit in mismatch:
        source = e194_rows[audit["case_id"]]
        for label, field, sha_field in (
            ("scene", "scene_act", "effective_scene_sha256"),
            ("trajectory", "trajectory", "trajectory_sha256"),
            ("contact", "contact_mask", "contact_mask_sha256"),
            ("override", "override_path", "override_sha256"),
        ):
            actual = C.sha256(source[field])
            if actual != source[sha_field]:
                raise ValueError(
                    f"{audit['case_id']} {label} SHA drift: {actual} != {source[sha_field]}"
                )
        meta = repair_and_audit_meta(
            C.repo_path(source["scene_act"]),
            C.repo_path(source["trajectory"]),
            audit["case_id"],
            apply=apply,
            previous_action=previous_actions.get(audit["case_id"], ""),
        )
        row = make_row(audit, source, meta)
        rows.append(row)
        authority_rows.append(
            {
                "case_id": audit["case_id"],
                "object_key": audit["object_key"],
                "e194_worker": audit["worker"],
                "e194_runtime_euler_convention": audit["runtime_euler_convention"],
                "compiled_xml_axis_sequence": audit["xml_hinge_axis_sequence"],
                "e196_worker": row["worker"],
                "wave": row["wave"],
                "scene_act": row["scene_act"],
                "scene_sha256": row["effective_scene_sha256"],
                "scene_act_meta_path": row["scene_act_meta_path"],
                "scene_act_meta_sha256": row["scene_act_meta_sha256"],
                "meta_repair_action": row["meta_repair_action"],
            }
        )
    order = {case_id: index for index, case_id in enumerate(C.planned_case_ids())}
    rows.sort(key=lambda row: order[str(row["case_id"])])
    for ordinal, row in enumerate(rows, 1):
        row["ordinal"] = ordinal
    wave0 = [row for row in rows if row["wave"] == "wave0"]
    remaining = [row for row in rows if row["wave"] == "remaining"]
    if len(wave0) != 3 or len(remaining) != 26:
        raise ValueError("E196 wave cardinality contract failed")
    C.write_tsv(C.FULL_MANIFEST, rows, C.FIELDS)
    C.write_tsv(C.WAVE0_MANIFEST, wave0, C.FIELDS)
    C.write_tsv(C.REMAINING_MANIFEST, remaining, C.FIELDS)
    C.write_tsv(C.CASE_AUTHORITY, authority_rows)
    snapshot_rows: list[dict[str, Any]] = []
    git_head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=C.REPO, text=True
    ).strip()
    dirty = bool(
        subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=C.REPO, text=True
        ).strip()
    )
    if snapshot:
        for row in rows:
            snapshot_rows.append(snapshot_row(row, git_head, dirty))
        C.write_tsv(C.SNAPSHOT_ROOT / "manifest.tsv", snapshot_rows)
    summary = {
        "created_at": C.now(),
        "apply": apply,
        "snapshot": snapshot,
        "case_set_sha256": C.case_set_sha256({row["case_id"] for row in rows}),
        "rows": len(rows),
        "objects": dict(Counter(row["object_key"] for row in rows)),
        "workers": dict(Counter(row["worker"] for row in rows)),
        "waves": dict(Counter(row["wave"] for row in rows)),
        "meta_actions": dict(Counter(row["meta_repair_action"] for row in rows)),
        "axis_target_vs_raw_ori_err_deg_max": max(
            float(row["axis_target_vs_raw_ori_err_deg_max"]) for row in rows
        ),
        "axis_target_vs_raw_pos_err_cm_max": max(
            float(row["axis_target_vs_raw_pos_err_cm_max"]) for row in rows
        ),
        "reference_contract_version": C.REFERENCE_CONTRACT_VERSION,
        "git_head": git_head,
        "git_dirty": dirty,
    }
    C.write_json(C.MANIFEST_DIR / "reference_fix_build_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--snapshot", action="store_true")
    args = parser.parse_args()
    return build(apply=args.apply, snapshot=args.snapshot)


if __name__ == "__main__":
    raise SystemExit(main())
