#!/usr/bin/env python3
"""Restore E174 Stage2b task inputs needed by E175.

The E174 Full authority contains 39 rows, but 21 task directories retained
only experiment sidecars.  E174 still preserves the source template and the
trimmed OmniRetarget output for every row.  This script replays only the
deterministic SPIDER Stage2b adapter; it never reruns OmniRetarget and never
overwrites a complete task.

Execution is deliberately two-phase:

1. ``--equivalence-canary`` rebuilds one of the 18 surviving tasks in an
   isolated sibling directory and requires exact array equivalence.
2. ``--restore-missing`` requires the passing canary report, builds each
   missing task transactionally in an isolated sibling directory, validates
   it, and copies only the five canonical Stage2b artifacts into the existing
   task directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import mujoco
import numpy as np
import torch
import yaml
from scipy.spatial.transform import Rotation as Rotation


REPO = Path(__file__).resolve().parents[5]
TASK_ROOT = (
    REPO
    / "example_datasets/processed/core4d/unitree_g1/humanoid_object"
)
E174_ROOT = REPO / "workspace/core4d/results/E174"
E175_ROOT = REPO / "workspace/core4d/results/E175"
RESTORE_ROOT = E175_ROOT / "stage2b_restore"
E174_FULL = (
    E174_ROOT / "s6_downstream/manifests/cem_full_manifest.tsv"
)
STAGE2B_MANIFESTS = (
    E174_ROOT
    / "s3_retarget/omnirt_v1/ref_fk/"
    "stage2b_manifest_omnirt_v1_ref_fk.tsv",
    E174_ROOT
    / "s3_retarget/omnirt_v2/ref_fk/"
    "stage2b_manifest_omnirt_v2_ref_fk.tsv",
)
CREATE_SCENE = (
    REPO / "workspace/core4d/data_preprocess/"
    "create_spider_scene_from_template.py"
)
PROCESS_CORE4D = REPO / "spider/process_datasets/core4d.py"
VERIFY_CASE = (
    REPO / "workspace/core4d/data_preprocess/verify_processed_case.py"
)
CANARY_REPORT = RESTORE_ROOT / "equivalence/equivalence_report.json"
ARRAY_KEYS = ("qpos", "qvel", "ctrl", "contact", "contact_pos")
PRIMARY_ARTIFACTS = (
    "scene.xml",
    "scene_act.xml",
    "scene_act_meta.json",
    "task_info.json",
    "0/trajectory_kinematic.npz",
)
EXPECTED_DISTRIBUTION = {
    "bucket003": 9,
    "bucket004": 4,
    "bucket007": 14,
    "bucket009": 1,
    "bucket010": 2,
    "desk007": 9,
}


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
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
        )
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rel(path: Path) -> str:
    return str(path.absolute().relative_to(REPO.absolute()))


def legacy_to_local(raw: str) -> Path:
    """Map E174 machine-specific paths to this checkout."""

    path = Path(raw)
    if not path.is_absolute():
        return REPO / path
    parts = path.parts
    if "example_datasets" in parts:
        index = parts.index("example_datasets")
        return REPO.joinpath(*parts[index:])
    e174_indices = [i for i, part in enumerate(parts) if part == "E174"]
    if e174_indices:
        index = e174_indices[-1]
        return E174_ROOT.joinpath(*parts[index + 1 :])
    if path.exists():
        return path
    raise ValueError(f"cannot localize legacy E174 path: {raw}")


def authority() -> list[dict[str, Any]]:
    full_rows = read_tsv(E174_FULL)
    if len(full_rows) != 39:
        raise ValueError(f"E174 Full authority must have 39 rows: {len(full_rows)}")
    case_ids = [row["case_id"] for row in full_rows]
    if len(set(case_ids)) != 39:
        raise ValueError("E174 Full authority contains duplicate case_id")
    distribution = Counter(row["object_key"] for row in full_rows)
    if dict(distribution) != EXPECTED_DISTRIBUTION:
        raise ValueError(f"E174 Full distribution drift: {dict(distribution)}")

    stage_rows: list[dict[str, str]] = []
    for manifest in STAGE2B_MANIFESTS:
        stage_rows.extend(read_tsv(manifest))
    stage_index = {
        (row["case_id"], row["retarget_variant_id"]): row
        for row in stage_rows
        if row.get("stage2b_status") == "pass"
    }

    records: list[dict[str, Any]] = []
    for full in full_rows:
        key = (full["case_id"], full["retarget_variant_id"])
        if key not in stage_index:
            raise KeyError(f"passing Stage2b row missing: {key}")
        stage = stage_index[key]
        source_scene = legacy_to_local(stage["source_scene_xml"])
        trimmed = legacy_to_local(stage["trimmed_npz"])
        task_dir = TASK_ROOT / full["target_task"]
        e174_scene_act = REPO / full["scene_act"]
        config_act = REPO / full["config_act"]
        outdir_npz = REPO / full["outdir_npz"]
        for required in (
            source_scene,
            trimmed,
            e174_scene_act,
            config_act,
            outdir_npz,
        ):
            if not required.is_file():
                raise FileNotFoundError(required)
        records.append(
            {
                "full": full,
                "stage": stage,
                "case_id": full["case_id"],
                "target_task": full["target_task"],
                "task_dir": task_dir,
                "source_scene": source_scene,
                "trimmed": trimmed,
                "e174_scene_act": e174_scene_act,
                "config_act": config_act,
                "outdir_npz": outdir_npz,
            }
        )
    return records


def run_command(command: list[str], log_path: Path) -> None:
    completed = subprocess.run(
        command,
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as stream:
        stream.write("$ " + " ".join(command) + "\n")
        stream.write(completed.stdout)
        if not completed.stdout.endswith("\n"):
            stream.write("\n")
    if completed.returncode != 0:
        raise RuntimeError(
            f"Stage2b adapter command failed ({completed.returncode}); "
            f"see {log_path}"
        )


def replay_adapter(
    record: dict[str, Any], temporary_task: str, log_path: Path
) -> Path:
    stage = record["stage"]
    common = [
        sys.executable,
        str(CREATE_SCENE),
        "--source-scene",
        str(record["source_scene"]),
        "--task",
        temporary_task,
        "--qpos",
        str(record["trimmed"]),
        "--data-id",
        str(stage.get("data_id", "0") or "0"),
        "--date",
        stage["date"],
        "--seq",
        stage["seq"],
        "--person",
        stage["person"],
        "--object-name",
        stage["object_name"],
        "--object-model-rel",
        stage["object_model_rel"],
    ]
    run_command(common, log_path)
    run_command(
        [
            sys.executable,
            str(PROCESS_CORE4D),
            "--source-npz",
            str(record["trimmed"]),
            "--task",
            temporary_task,
            "--data-id",
            str(stage.get("data_id", "0") or "0"),
            "--no-show-viewer",
            "--no-save-video",
        ],
        log_path,
    )
    run_command(common + ["--generate-scene-act"], log_path)
    return TASK_ROOT / temporary_task


def load_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as archive:
        missing = [key for key in ARRAY_KEYS if key not in archive.files]
        if missing:
            raise ValueError(f"{path} missing arrays: {missing}")
        return {key: np.asarray(archive[key]) for key in ARRAY_KEYS}


def compare_arrays(
    left: dict[str, np.ndarray], right: dict[str, np.ndarray]
) -> tuple[bool, dict[str, Any]]:
    details: dict[str, Any] = {}
    passed = True
    for key in ARRAY_KEYS:
        lhs, rhs = left[key], right[key]
        shape_equal = lhs.shape == rhs.shape
        exact = bool(shape_equal and np.array_equal(lhs, rhs))
        max_abs = (
            float(np.max(np.abs(lhs.astype(np.float64) - rhs.astype(np.float64))))
            if shape_equal and lhs.size
            else float("inf")
        )
        details[key] = {
            "left_shape": list(lhs.shape),
            "right_shape": list(rhs.shape),
            "exact_equal": exact,
            "max_abs": max_abs,
        }
        passed &= exact
    return passed, details


def runner_reference(
    trajectory: Path, config_path: Path, model_path: Path
) -> dict[str, np.ndarray]:
    """Reproduce load_data + run_mjwp freejoint-to-scene_act conversion."""

    from spider.io import load_data

    config_values = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config_values["device"] = "cpu"
    config = SimpleNamespace(**config_values)
    qpos, qvel, ctrl, contact, contact_pos = load_data(
        config, str(trajectory)
    )

    if (
        (config.object_pd_override or config.contact_guidance)
        and qpos.shape[1] > config.nq
    ):
        model = mujoco.MjModel.from_xml_path(str(model_path))
        object_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "object"
        )
        nq_robot = int(config.nq) - 6
        obj_pos_world = qpos[
            :, nq_robot : nq_robot + 3
        ].detach().cpu().numpy()
        obj_quat_wxyz = qpos[
            :, nq_robot + 3 : nq_robot + 7
        ].detach().cpu().numpy()

        body_pos = model.body_pos[object_id]
        body_quat_wxyz = model.body_quat[object_id]
        body_rotation = Rotation.from_quat(
            [
                body_quat_wxyz[1],
                body_quat_wxyz[2],
                body_quat_wxyz[3],
                body_quat_wxyz[0],
            ]
        )
        obj_slide = body_rotation.inv().apply(
            obj_pos_world - body_pos[np.newaxis, :]
        )
        meta_path = model_path.parent / "scene_act_meta.json"
        euler_convention = (
            json.loads(meta_path.read_text(encoding="utf-8"))[
                "euler_convention"
            ]
            if meta_path.is_file()
            else "XYZ"
        )
        world_rotation = Rotation.from_quat(
            np.column_stack(
                [
                    obj_quat_wxyz[:, 1],
                    obj_quat_wxyz[:, 2],
                    obj_quat_wxyz[:, 3],
                    obj_quat_wxyz[:, 0],
                ]
            )
        )
        obj_euler = (
            body_rotation.inv() * world_rotation
        ).as_euler(euler_convention)

        qpos_new = torch.zeros(
            (qpos.shape[0], int(config.nq)), dtype=qpos.dtype
        )
        qpos_new[:, :nq_robot] = qpos[:, :nq_robot]
        qpos_new[:, nq_robot : nq_robot + 3] = torch.from_numpy(
            obj_slide.astype(np.float32)
        )
        qpos_new[:, nq_robot + 3 : nq_robot + 6] = torch.from_numpy(
            obj_euler.astype(np.float32)
        )
        qvel_new = torch.zeros(
            (qvel.shape[0], int(config.nv)), dtype=qvel.dtype
        )
        qvel_new[:, : int(config.nv)] = qvel[:, : int(config.nv)]
        ctrl_new = torch.zeros(
            (ctrl.shape[0], int(config.nu)), dtype=ctrl.dtype
        )
        copied_ctrl = min(ctrl.shape[1], int(config.nu))
        ctrl_new[:, :copied_ctrl] = ctrl[:, :copied_ctrl]
        object_ctrl_start = int(config.nu) - 6
        ctrl_new[
            :, object_ctrl_start : object_ctrl_start + 3
        ] = torch.from_numpy(obj_slide.astype(np.float32))
        ctrl_new[
            :, object_ctrl_start + 3 : object_ctrl_start + 6
        ] = torch.from_numpy(obj_euler.astype(np.float32))
        qpos, qvel, ctrl = qpos_new, qvel_new, ctrl_new

    return {
        "qpos": qpos.detach().cpu().numpy(),
        "qvel": qvel.detach().cpu().numpy(),
        "ctrl": ctrl.detach().cpu().numpy(),
        "contact": contact.detach().cpu().numpy(),
        "contact_pos": contact_pos.detach().cpu().numpy(),
    }


def outdir_rollout_contract(
    outdir_npz: Path, config_path: Path
) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    with np.load(outdir_npz, allow_pickle=True) as archive:
        qpos_shape = list(archive["qpos"].shape)
    ctrl_steps = int(config["ctrl_steps"])
    max_sim_steps = int(config["max_sim_steps"])
    is_rollout_substeps = bool(
        len(qpos_shape) == 3
        and qpos_shape[1] == ctrl_steps
        and qpos_shape[0] * qpos_shape[1] == max_sim_steps
    )
    return {
        "qpos_shape": qpos_shape,
        "ctrl_steps": ctrl_steps,
        "max_sim_steps": max_sim_steps,
        "is_rollout_substeps": is_rollout_substeps,
        "contains_independent_ref_channel": False,
        "note": (
            "axis 1 is ctrl-step rollout, not sim/ref; "
            "qpos[:,1,:] cannot be used as reference authority"
        ),
    }


def compile_contract(task_dir: Path) -> dict[str, Any]:
    scene = mujoco.MjModel.from_xml_path(str(task_dir / "scene.xml"))
    scene_act = mujoco.MjModel.from_xml_path(
        str(task_dir / "scene_act.xml")
    )
    result = {
        "scene_nq": int(scene.nq),
        "scene_nv": int(scene.nv),
        "scene_nu": int(scene.nu),
        "scene_act_nq": int(scene_act.nq),
        "scene_act_nv": int(scene_act.nv),
        "scene_act_nu": int(scene_act.nu),
    }
    result["pass"] = bool(
        (scene.nq, scene.nv, scene.nu) == (43, 41, 29)
        and (scene_act.nq, scene_act.nv, scene_act.nu)
        == (42, 41, 35)
    )
    return result


def task_completeness(task_dir: Path) -> dict[str, bool]:
    return {
        relative: (task_dir / relative).is_file()
        for relative in PRIMARY_ARTIFACTS
    }


def audit_task(record: dict[str, Any]) -> dict[str, Any]:
    task_dir = record["task_dir"]
    completeness = task_completeness(task_dir)
    if not all(completeness.values()):
        raise FileNotFoundError(
            f"incomplete restored task {record['case_id']}: {completeness}"
        )
    trimmed = np.load(record["trimmed"], allow_pickle=True)["qpos"]
    arrays = load_arrays(task_dir / "0/trajectory_kinematic.npz")
    qpos_exact = bool(np.array_equal(trimmed, arrays["qpos"]))
    trajectory_sha = sha256(task_dir / "0/trajectory_kinematic.npz")
    authority_sha_exact = bool(
        trajectory_sha == record["full"]["trajectory_sha256"]
    )
    compile_result = compile_contract(task_dir)
    reference = runner_reference(
        task_dir / "0/trajectory_kinematic.npz",
        record["config_act"],
        record["e174_scene_act"],
    )
    reference_shapes = {
        key: list(value.shape) for key, value in reference.items()
    }
    config = yaml.safe_load(
        record["config_act"].read_text(encoding="utf-8")
    )
    dimension_pass = bool(
        reference["qpos"].shape[1] == int(config["nq"])
        and reference["qvel"].shape[1] == int(config["nv"])
        and reference["ctrl"].shape[1] == int(config["nu"])
        and reference["qpos"].shape[0] > int(config["max_sim_steps"])
    )
    return {
        "completeness": completeness,
        "trimmed_qpos_exact": qpos_exact,
        "trajectory_sha256": trajectory_sha,
        "trajectory_authority_sha_exact": authority_sha_exact,
        "compile": compile_result,
        "runner_reference_shapes": reference_shapes,
        "runner_dimension_pass": dimension_pass,
        "pass": bool(
            all(completeness.values())
            and qpos_exact
            and authority_sha_exact
            and compile_result["pass"]
            and dimension_pass
        ),
    }


def equivalence_canary(
    records: list[dict[str, Any]], case_id: str | None
) -> dict[str, Any]:
    candidates = [
        record
        for record in records
        if all(task_completeness(record["task_dir"]).values())
    ]
    if case_id:
        candidates = [
            record for record in candidates if record["case_id"] == case_id
        ]
    if not candidates:
        raise ValueError("no complete E174 task available for equivalence canary")
    record = candidates[0]
    log_path = RESTORE_ROOT / "equivalence/adapter.log"

    with tempfile.TemporaryDirectory(
        prefix="__e175_stage2b_equivalence_", dir=TASK_ROOT
    ) as temporary:
        temporary_task = Path(temporary).name
        generated_dir = replay_adapter(
            record, temporary_task, log_path
        )
        original_trajectory = (
            record["task_dir"] / "0/trajectory_kinematic.npz"
        )
        generated_trajectory = (
            generated_dir / "0/trajectory_kinematic.npz"
        )
        array_pass, array_details = compare_arrays(
            load_arrays(original_trajectory),
            load_arrays(generated_trajectory),
        )
        reference_pass, reference_details = compare_arrays(
            runner_reference(
                original_trajectory,
                record["config_act"],
                record["e174_scene_act"],
            ),
            runner_reference(
                generated_trajectory,
                record["config_act"],
                record["e174_scene_act"],
            ),
        )
        generated_compile = compile_contract(generated_dir)
        generated_scene_xml = ET.tostring(
            ET.parse(generated_dir / "scene.xml").getroot()
        )
        original_scene_xml = ET.tostring(
            ET.parse(record["task_dir"] / "scene.xml").getroot()
        )
        scene_exact = generated_scene_xml == original_scene_xml
        report = {
            "created_at": now(),
            "case_id": record["case_id"],
            "target_task": record["target_task"],
            "source_scene": rel(record["source_scene"]),
            "trimmed": rel(record["trimmed"]),
            "original_trajectory_sha256": sha256(original_trajectory),
            "generated_trajectory_sha256": sha256(generated_trajectory),
            "npz_bytes_equal": (
                sha256(original_trajectory)
                == sha256(generated_trajectory)
            ),
            "arrays_exact_pass": array_pass,
            "array_comparison": array_details,
            "runner_reference_exact_pass": reference_pass,
            "runner_reference_comparison": reference_details,
            "scene_xml_semantic_exact": scene_exact,
            "generated_compile": generated_compile,
            "e174_outdir_contract": outdir_rollout_contract(
                record["outdir_npz"], record["config_act"]
            ),
            "adapter_log": rel(log_path),
        }
        report["status"] = (
            "pass"
            if (
                array_pass
                and reference_pass
                and scene_exact
                and generated_compile["pass"]
                and report["e174_outdir_contract"][
                    "is_rollout_substeps"
                ]
            )
            else "fail"
        )
    write_json(CANARY_REPORT, report)
    if report["status"] != "pass":
        raise RuntimeError(
            f"Stage2b equivalence canary failed: {CANARY_REPORT}"
        )
    return report


def commit_from_staging(
    staged_dir: Path, task_dir: Path, final_task: str
) -> None:
    task_dir.mkdir(parents=True, exist_ok=True)
    before = task_completeness(task_dir)
    if any(before.values()):
        raise FileExistsError(
            f"refusing partial/overwrite restore for {task_dir}: {before}"
        )
    for relative in PRIMARY_ARTIFACTS:
        source = staged_dir / relative
        if not source.is_file():
            raise FileNotFoundError(source)
    (task_dir / "0").mkdir(parents=True, exist_ok=False)
    for relative in PRIMARY_ARTIFACTS:
        source = staged_dir / relative
        destination = task_dir / relative
        if relative == "task_info.json":
            metadata = json.loads(source.read_text(encoding="utf-8"))
            metadata["task"] = final_task
            destination.write_text(
                json.dumps(metadata, indent=2) + "\n",
                encoding="utf-8",
            )
        else:
            shutil.copy2(source, destination)


def backup_and_replace_from_staging(
    staged_dir: Path,
    task_dir: Path,
    final_task: str,
    case_id: str,
) -> Path:
    """Back up stale primary artifacts, then atomically replace each file."""

    before = task_completeness(task_dir)
    if not all(before.values()):
        raise FileNotFoundError(
            f"authority repair requires a complete stale task: "
            f"{case_id} {before}"
        )
    backup_dir = RESTORE_ROOT / "preexisting_backup" / case_id
    if backup_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite preexisting backup: {backup_dir}"
        )
    backup_rows = []
    for relative in PRIMARY_ARTIFACTS:
        source = task_dir / relative
        destination = backup_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        backup_rows.append(
            {
                "case_id": case_id,
                "artifact": relative,
                "original_path": rel(source),
                "backup_path": rel(destination),
                "sha256": sha256(destination),
            }
        )
    write_tsv(backup_dir / "backup_manifest.tsv", backup_rows)
    write_json(
        backup_dir / "backup_manifest.json",
        {
            "created_at": now(),
            "case_id": case_id,
            "reason": (
                "preexisting task trajectory did not match the frozen "
                "E174 Full manifest SHA"
            ),
            "artifacts": backup_rows,
        },
    )

    for relative in PRIMARY_ARTIFACTS:
        source = staged_dir / relative
        destination = task_dir / relative
        temporary = destination.with_name(
            destination.name + ".e175-authority-repair.tmp"
        )
        if temporary.exists():
            temporary.unlink()
        if relative == "task_info.json":
            metadata = json.loads(source.read_text(encoding="utf-8"))
            metadata["task"] = final_task
            temporary.write_text(
                json.dumps(metadata, indent=2) + "\n",
                encoding="utf-8",
            )
        else:
            shutil.copy2(source, temporary)
        temporary.replace(destination)
    return backup_dir


def write_final_restore_reports(
    records: list[dict[str, Any]],
    action_by_case: dict[str, str],
    *,
    canary_status: str,
) -> dict[str, Any]:
    manifest_rows: list[dict[str, Any]] = []
    for record in records:
        audit = audit_task(record)
        trajectory = record["task_dir"] / "0/trajectory_kinematic.npz"
        manifest_rows.append(
            {
                "case_id": record["case_id"],
                "object_key": record["full"]["object_key"],
                "retarget_variant_id": record["full"][
                    "retarget_variant_id"
                ],
                "target_task": record["target_task"],
                "source_scene": rel(record["source_scene"]),
                "source_scene_sha256": sha256(record["source_scene"]),
                "trimmed_npz": rel(record["trimmed"]),
                "trimmed_npz_sha256": sha256(record["trimmed"]),
                "task_dir": rel(record["task_dir"]),
                "trajectory": rel(trajectory),
                "trajectory_sha256": sha256(trajectory),
                "e174_authority_trajectory_sha256": record["full"][
                    "trajectory_sha256"
                ],
                "trajectory_authority_sha_exact": audit[
                    "trajectory_authority_sha_exact"
                ],
                "scene_sha256": sha256(record["task_dir"] / "scene.xml"),
                "scene_act_sha256": sha256(
                    record["task_dir"] / "scene_act.xml"
                ),
                "restore_action": action_by_case.get(
                    record["case_id"], "preserved_authority_match"
                ),
                "trimmed_qpos_exact": audit["trimmed_qpos_exact"],
                "scene_compile_pass": audit["compile"]["pass"],
                "runner_dimension_pass": audit[
                    "runner_dimension_pass"
                ],
                "audit_status": "pass" if audit["pass"] else "fail",
                "verified_at": now(),
            }
        )
    failures = [
        row["case_id"]
        for row in manifest_rows
        if row["audit_status"] != "pass"
    ]
    manifest_path = RESTORE_ROOT / "stage2b_restore_manifest.tsv"
    write_tsv(manifest_path, manifest_rows)
    write_json(
        RESTORE_ROOT / "stage2b_restore_manifest.json", manifest_rows
    )
    action_counts = Counter(
        row["restore_action"] for row in manifest_rows
    )
    summary = {
        "created_at": now(),
        "source_authority": rel(E174_FULL),
        "authority_rows": len(records),
        "canary_report": rel(CANARY_REPORT),
        "canary_status": canary_status,
        "action_counts": dict(action_counts),
        "authority_sha_exact": sum(
            bool(row["trajectory_authority_sha_exact"])
            for row in manifest_rows
        ),
        "audit_pass": sum(
            row["audit_status"] == "pass" for row in manifest_rows
        ),
        "audit_failures": failures,
        "manifest": rel(manifest_path),
        "status": "pass" if not failures else "fail",
    }
    write_json(RESTORE_ROOT / "stage2b_restore_summary.json", summary)
    if summary["status"] != "pass":
        raise RuntimeError(f"Stage2b restore audit failed: {summary}")
    return summary


def restore_missing(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not CANARY_REPORT.is_file():
        raise FileNotFoundError(
            f"run --equivalence-canary first: {CANARY_REPORT}"
        )
    canary = json.loads(CANARY_REPORT.read_text(encoding="utf-8"))
    if canary.get("status") != "pass":
        raise ValueError("equivalence canary is not passing")

    initial_complete = {
        record["case_id"]: all(
            task_completeness(record["task_dir"]).values()
        )
        for record in records
    }
    initial_missing = [
        record for record in records if not initial_complete[record["case_id"]]
    ]
    if len(initial_missing) != 21:
        raise ValueError(
            f"expected exactly 21 missing tasks before restore, "
            f"got {len(initial_missing)}"
        )
    for record in initial_missing:
        state = task_completeness(record["task_dir"])
        if any(state.values()):
            raise ValueError(
                f"partial primary Stage2b state requires manual audit: "
                f"{record['case_id']} {state}"
            )

    restored: set[str] = set()
    for record in initial_missing:
        case_id = record["case_id"]
        log_path = RESTORE_ROOT / "logs" / f"{case_id}.log"
        with tempfile.TemporaryDirectory(
            prefix=f"__e175_restore_{case_id}_", dir=TASK_ROOT
        ) as temporary:
            temporary_task = Path(temporary).name
            staged_dir = replay_adapter(
                record, temporary_task, log_path
            )
            staged_arrays = load_arrays(
                staged_dir / "0/trajectory_kinematic.npz"
            )
            trimmed_qpos = np.load(
                record["trimmed"], allow_pickle=True
            )["qpos"]
            if not np.array_equal(trimmed_qpos, staged_arrays["qpos"]):
                raise RuntimeError(
                    f"staged qpos differs from trimmed: {case_id}"
                )
            staged_trajectory = (
                staged_dir / "0/trajectory_kinematic.npz"
            )
            if (
                sha256(staged_trajectory)
                != record["full"]["trajectory_sha256"]
            ):
                raise RuntimeError(
                    f"staged trajectory SHA differs from E174 authority: "
                    f"{case_id}"
                )
            if not compile_contract(staged_dir)["pass"]:
                raise RuntimeError(
                    f"staged scene compile contract failed: {case_id}"
                )
            runner = runner_reference(
                staged_dir / "0/trajectory_kinematic.npz",
                record["config_act"],
                record["e174_scene_act"],
            )
            config = yaml.safe_load(
                record["config_act"].read_text(encoding="utf-8")
            )
            if (
                runner["qpos"].shape[1] != int(config["nq"])
                or runner["qvel"].shape[1] != int(config["nv"])
                or runner["ctrl"].shape[1] != int(config["nu"])
                or runner["qpos"].shape[0]
                <= int(config["max_sim_steps"])
            ):
                raise RuntimeError(
                    f"runner reference contract failed: {case_id}"
                )
            commit_from_staging(
                staged_dir, record["task_dir"], record["target_task"]
            )
        verify_path = RESTORE_ROOT / "verify" / f"{case_id}.json"
        run_command(
            [
                sys.executable,
                str(VERIFY_CASE),
                "--task",
                record["target_task"],
                "--source-scene",
                str(record["source_scene"]),
                "--trimmed",
                str(record["trimmed"]),
                "--out",
                str(verify_path),
            ],
            log_path,
        )
        restored.add(case_id)

    return write_final_restore_reports(
        records,
        {
            record["case_id"]: (
                "restored_from_e174_trimmed"
                if record["case_id"] in restored
                else "preserved_authority_match"
            )
            for record in records
        },
        canary_status=canary["status"],
    )


def repair_authority_drift(
    records: list[dict[str, Any]]
) -> dict[str, Any]:
    if not CANARY_REPORT.is_file():
        raise FileNotFoundError(
            f"run --equivalence-canary first: {CANARY_REPORT}"
        )
    canary = json.loads(CANARY_REPORT.read_text(encoding="utf-8"))
    if canary.get("status") != "pass":
        raise ValueError("equivalence canary is not passing")

    drift = []
    for record in records:
        state = task_completeness(record["task_dir"])
        if not all(state.values()):
            raise FileNotFoundError(
                f"repair requires 39 complete tasks: "
                f"{record['case_id']} {state}"
            )
        current_sha = sha256(
            record["task_dir"] / "0/trajectory_kinematic.npz"
        )
        if current_sha != record["full"]["trajectory_sha256"]:
            drift.append(record)
    if len(drift) > 11:
        raise ValueError(
            f"unexpected authority drift count >11: {len(drift)}"
        )

    previous_actions: dict[str, str] = {}
    previous_manifest = RESTORE_ROOT / "stage2b_restore_manifest.tsv"
    if previous_manifest.is_file():
        previous_actions = {
            row["case_id"]: row.get("restore_action", "")
            for row in read_tsv(previous_manifest)
        }

    repaired: set[str] = set()
    for record in drift:
        case_id = record["case_id"]
        log_path = (
            RESTORE_ROOT / "logs" / f"{case_id}_authority_repair.log"
        )
        with tempfile.TemporaryDirectory(
            prefix=f"__e175_authority_repair_{case_id}_",
            dir=TASK_ROOT,
        ) as temporary:
            temporary_task = Path(temporary).name
            staged_dir = replay_adapter(
                record, temporary_task, log_path
            )
            staged_trajectory = (
                staged_dir / "0/trajectory_kinematic.npz"
            )
            staged_sha = sha256(staged_trajectory)
            expected_sha = record["full"]["trajectory_sha256"]
            if staged_sha != expected_sha:
                raise RuntimeError(
                    f"authority repair SHA mismatch before replacement: "
                    f"{case_id} generated={staged_sha} "
                    f"expected={expected_sha}"
                )
            staged_arrays = load_arrays(staged_trajectory)
            trimmed_qpos = np.load(
                record["trimmed"], allow_pickle=True
            )["qpos"]
            if not np.array_equal(trimmed_qpos, staged_arrays["qpos"]):
                raise RuntimeError(
                    f"authority repair qpos differs from trimmed: {case_id}"
                )
            if not compile_contract(staged_dir)["pass"]:
                raise RuntimeError(
                    f"authority repair compile failed: {case_id}"
                )
            backup_and_replace_from_staging(
                staged_dir,
                record["task_dir"],
                record["target_task"],
                case_id,
            )
        verify_path = (
            RESTORE_ROOT / "verify" / f"{case_id}_authority_repair.json"
        )
        run_command(
            [
                sys.executable,
                str(VERIFY_CASE),
                "--task",
                record["target_task"],
                "--source-scene",
                str(record["source_scene"]),
                "--trimmed",
                str(record["trimmed"]),
                "--out",
                str(verify_path),
            ],
            log_path,
        )
        repaired.add(case_id)

    actions = {}
    for record in records:
        case_id = record["case_id"]
        if case_id in repaired:
            actions[case_id] = "repaired_e174_authority_drift"
        elif (
            previous_actions.get(case_id)
            == "restored_from_e174_trimmed"
        ):
            actions[case_id] = "restored_from_e174_trimmed"
        else:
            actions[case_id] = "preserved_authority_match"
    summary = write_final_restore_reports(
        records, actions, canary_status=canary["status"]
    )
    summary["authority_drift_detected"] = len(drift)
    summary["authority_drift_repaired"] = len(repaired)
    summary["backup_root"] = rel(
        RESTORE_ROOT / "preexisting_backup"
    )
    write_json(RESTORE_ROOT / "stage2b_restore_summary.json", summary)
    return summary


def inspect(records: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for record in records:
        state = task_completeness(record["task_dir"])
        rows.append(
            {
                "case_id": record["case_id"],
                "target_task": record["target_task"],
                "retarget_variant_id": record["full"][
                    "retarget_variant_id"
                ],
                "source_scene_exists": record["source_scene"].is_file(),
                "trimmed_exists": record["trimmed"].is_file(),
                **state,
                "complete": all(state.values()),
            }
        )
    payload = {
        "created_at": now(),
        "authority_rows": len(rows),
        "complete": sum(row["complete"] for row in rows),
        "missing_or_partial": sum(not row["complete"] for row in rows),
        "rows": rows,
    }
    write_json(RESTORE_ROOT / "availability_before_restore.json", payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inspect", action="store_true")
    parser.add_argument("--equivalence-canary", action="store_true")
    parser.add_argument("--canary-case")
    parser.add_argument("--restore-missing", action="store_true")
    parser.add_argument("--repair-authority-drift", action="store_true")
    args = parser.parse_args()
    if sum(
        int(flag)
        for flag in (
            args.inspect,
            args.equivalence_canary,
            args.restore_missing,
            args.repair_authority_drift,
        )
    ) != 1:
        parser.error(
            "select exactly one of --inspect, --equivalence-canary, "
            "--restore-missing, --repair-authority-drift"
        )
    records = authority()
    if args.inspect:
        payload = inspect(records)
    elif args.equivalence_canary:
        payload = equivalence_canary(records, args.canary_case)
    elif args.restore_missing:
        payload = restore_missing(records)
    else:
        payload = repair_authority_drift(records)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
