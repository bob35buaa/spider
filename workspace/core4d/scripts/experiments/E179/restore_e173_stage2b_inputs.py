#!/usr/bin/env python3
"""Restore the exact 16 E173 Stage2b tasks needed by E179.

The task directories retained only the tracked E173 PRG sidecars.  E173 still
preserves the source template, trimmed OmniRetarget output, pristine
``scene_act.xml`` snapshot and the exact final trajectory SHA for every row.
This script replays only the deterministic SPIDER Stage2b adapter.

Safety contract:

* ``--authority-canary`` first rebuilds one row in an isolated temporary task
  and requires the generated trajectory SHA to equal the E173 Full authority
  and the generated ``scene_act.xml`` to be semantically identical to the
  pristine E173 snapshot.
* ``--restore-missing`` requires that passing report and refuses partial or
  overwrite restoration.  Existing E173 sidecars are left untouched.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e179_common as C  # noqa: E402


CREATE_SCENE = (
    C.REPO
    / "workspace/core4d/data_preprocess/"
    "create_spider_scene_from_template.py"
)
PROCESS_CORE4D = C.REPO / "spider/process_datasets/core4d.py"
VERIFY_CASE = (
    C.REPO / "workspace/core4d/data_preprocess/verify_processed_case.py"
)
RESTORE_ROOT = C.RESULTS / "input_authority/stage2b_restore"
CANARY_REPORT = RESTORE_ROOT / "authority_canary.json"
PRIMARY_ARTIFACTS = (
    "scene.xml",
    "scene_act.xml",
    "scene_act_meta.json",
    "task_info.json",
    "0/trajectory_kinematic.npz",
)
ARRAY_KEYS = ("qpos", "qvel", "ctrl", "contact", "contact_pos")


def authority() -> list[dict[str, Any]]:
    path = C.RESULTS / "input_authority/input_authority.tsv"
    if not path.is_file():
        raise FileNotFoundError(
            f"build paired authority first: {path}"
        )
    rows = C.read_tsv(path)
    if len(rows) != C.EXPECTED_PAIRED_ROWS:
        raise ValueError(f"authority rows must be 16: {len(rows)}")
    if len({row["case_id"] for row in rows}) != len(rows):
        raise ValueError("duplicate authority case_id")
    for row in rows:
        for field in (
            "source_scene",
            "trimmed_npz",
            "pristine_scene_snapshot",
        ):
            if not C.repo_path(row[field]).is_file():
                raise FileNotFoundError(
                    f"{row['case_id']} missing {field}: {row[field]}"
                )
    return rows


def task_dir(row: dict[str, str]) -> Path:
    return C.TASK_ROOT / row["target_task"]


def completeness(path: Path) -> dict[str, bool]:
    return {
        relative: (path / relative).is_file()
        for relative in PRIMARY_ARTIFACTS
    }


def semantic_xml(path: Path) -> bytes:
    return ET.tostring(ET.parse(path).getroot())


def load_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as archive:
        missing = [key for key in ARRAY_KEYS if key not in archive.files]
        if missing:
            raise ValueError(f"{path} missing arrays: {missing}")
        return {key: np.asarray(archive[key]) for key in ARRAY_KEYS}


def compile_contract(path: Path) -> dict[str, Any]:
    scene = mujoco.MjModel.from_xml_path(str(path / "scene.xml"))
    scene_act = mujoco.MjModel.from_xml_path(
        str(path / "scene_act.xml")
    )
    payload = {
        "scene_nq": int(scene.nq),
        "scene_nv": int(scene.nv),
        "scene_nu": int(scene.nu),
        "scene_act_nq": int(scene_act.nq),
        "scene_act_nv": int(scene_act.nv),
        "scene_act_nu": int(scene_act.nu),
    }
    payload["pass"] = bool(
        (scene.nq, scene.nv, scene.nu) == (43, 41, 29)
        and (scene_act.nq, scene_act.nv, scene_act.nu)
        == (42, 41, 35)
    )
    return payload


def run_command(command: list[str], log_path: Path) -> None:
    completed = subprocess.run(
        command,
        cwd=C.REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as stream:
        stream.write("$ " + " ".join(command) + "\n")
        stream.write(completed.stdout)
        if completed.stdout and not completed.stdout.endswith("\n"):
            stream.write("\n")
    if completed.returncode != 0:
        raise RuntimeError(
            f"Stage2b adapter failed ({completed.returncode}); "
            f"see {log_path}"
        )


def replay_adapter(
    row: dict[str, str], temporary_task: str, log_path: Path
) -> Path:
    common = [
        sys.executable,
        str(CREATE_SCENE),
        "--source-scene",
        str(C.repo_path(row["source_scene"])),
        "--task",
        temporary_task,
        "--qpos",
        str(C.repo_path(row["trimmed_npz"])),
        "--data-id",
        row.get("data_id", "0") or "0",
        "--date",
        row["date"],
        "--seq",
        row["seq"],
        "--person",
        row["person"],
        "--object-name",
        row["object_name"],
        "--object-model-rel",
        row["object_model_rel"],
    ]
    run_command(common, log_path)
    run_command(
        [
            sys.executable,
            str(PROCESS_CORE4D),
            "--source-npz",
            str(C.repo_path(row["trimmed_npz"])),
            "--task",
            temporary_task,
            "--data-id",
            row.get("data_id", "0") or "0",
            "--no-show-viewer",
            "--no-save-video",
        ],
        log_path,
    )
    run_command(common + ["--generate-scene-act"], log_path)
    return C.TASK_ROOT / temporary_task


def audit_generated(
    row: dict[str, str], generated: Path
) -> dict[str, Any]:
    trajectory = generated / "0/trajectory_kinematic.npz"
    expected_sha = row["trajectory_sha256"]
    actual_sha = C.sha256(trajectory)
    trimmed_qpos = np.load(
        C.repo_path(row["trimmed_npz"]), allow_pickle=True
    )["qpos"]
    arrays = load_arrays(trajectory)
    compile_result = compile_contract(generated)
    scene_act_semantic_exact = bool(
        semantic_xml(generated / "scene_act.xml")
        == semantic_xml(C.repo_path(row["pristine_scene_snapshot"]))
    )
    payload = {
        "trajectory_sha256": actual_sha,
        "trajectory_expected_sha256": expected_sha,
        "trajectory_authority_sha_exact": actual_sha == expected_sha,
        "trimmed_qpos_exact": bool(
            np.array_equal(trimmed_qpos, arrays["qpos"])
        ),
        "scene_act_semantic_exact": scene_act_semantic_exact,
        "compile": compile_result,
    }
    payload["pass"] = bool(
        payload["trajectory_authority_sha_exact"]
        and payload["trimmed_qpos_exact"]
        and payload["scene_act_semantic_exact"]
        and compile_result["pass"]
    )
    return payload


def inspect(rows: list[dict[str, str]]) -> dict[str, Any]:
    details = []
    for row in rows:
        state = completeness(task_dir(row))
        details.append(
            {
                "case_id": row["case_id"],
                "target_task": row["target_task"],
                **state,
                "complete": all(state.values()),
                "partial": any(state.values()) and not all(state.values()),
            }
        )
    payload = {
        "created_at": C.now(),
        "authority_rows": len(rows),
        "complete": sum(row["complete"] for row in details),
        "partial": sum(row["partial"] for row in details),
        "missing": sum(
            not row["complete"] and not row["partial"]
            for row in details
        ),
        "rows": details,
    }
    C.write_json(RESTORE_ROOT / "availability.json", payload)
    return payload


def authority_canary(
    rows: list[dict[str, str]], case_id: str
) -> dict[str, Any]:
    matches = [row for row in rows if row["case_id"] == case_id]
    if len(matches) != 1:
        raise KeyError(f"canary case not in authority: {case_id}")
    row = matches[0]
    log_path = RESTORE_ROOT / "authority_canary.log"
    with tempfile.TemporaryDirectory(
        prefix="__e179_stage2b_canary_", dir=C.TASK_ROOT
    ) as temporary:
        generated = replay_adapter(
            row, Path(temporary).name, log_path
        )
        audit = audit_generated(row, generated)
    payload = {
        "created_at": C.now(),
        "case_id": case_id,
        "target_task": row["target_task"],
        "source_scene": row["source_scene"],
        "trimmed_npz": row["trimmed_npz"],
        "pristine_scene_snapshot": row[
            "pristine_scene_snapshot"
        ],
        "adapter_log": C.rel(log_path),
        **audit,
        "status": "pass" if audit["pass"] else "fail",
    }
    C.write_json(CANARY_REPORT, payload)
    if payload["status"] != "pass":
        raise RuntimeError(
            f"E179 Stage2b authority canary failed: {CANARY_REPORT}"
        )
    return payload


def commit_from_staging(
    staged: Path, destination: Path, final_task: str
) -> None:
    state = completeness(destination)
    if any(state.values()):
        raise FileExistsError(
            f"refusing partial/overwrite restore: "
            f"{destination} {state}"
        )
    for relative in PRIMARY_ARTIFACTS:
        if not (staged / relative).is_file():
            raise FileNotFoundError(staged / relative)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "0").mkdir(parents=True, exist_ok=False)
    for relative in PRIMARY_ARTIFACTS:
        source = staged / relative
        target = destination / relative
        if relative == "task_info.json":
            metadata = json.loads(source.read_text(encoding="utf-8"))
            metadata["task"] = final_task
            target.write_text(
                json.dumps(metadata, indent=2) + "\n",
                encoding="utf-8",
            )
        else:
            shutil.copy2(source, target)


def audit_restored(
    rows: list[dict[str, str]],
    actions: dict[str, str],
) -> dict[str, Any]:
    audit_rows = []
    for row in rows:
        destination = task_dir(row)
        state = completeness(destination)
        if not all(state.values()):
            raise FileNotFoundError(
                f"incomplete restored task: {row['case_id']} {state}"
            )
        audit = audit_generated(row, destination)
        audit_rows.append(
            {
                "case_id": row["case_id"],
                "retarget_variant_id": row[
                    "retarget_variant_id"
                ],
                "target_task": row["target_task"],
                "task_dir": C.rel(destination),
                "trajectory": C.rel(
                    destination / "0/trajectory_kinematic.npz"
                ),
                "trajectory_sha256": audit["trajectory_sha256"],
                "trajectory_expected_sha256": row[
                    "trajectory_sha256"
                ],
                "trajectory_authority_sha_exact": audit[
                    "trajectory_authority_sha_exact"
                ],
                "trimmed_qpos_exact": audit["trimmed_qpos_exact"],
                "scene_act_semantic_exact": audit[
                    "scene_act_semantic_exact"
                ],
                "scene_compile_pass": audit["compile"]["pass"],
                "restore_action": actions.get(
                    row["case_id"], "preserved_authority_match"
                ),
                "audit_status": "pass" if audit["pass"] else "fail",
                "verified_at": C.now(),
            }
        )
    C.write_tsv(
        RESTORE_ROOT / "stage2b_restore_manifest.tsv", audit_rows
    )
    C.write_json(
        RESTORE_ROOT / "stage2b_restore_manifest.json", audit_rows
    )
    failures = [
        row["case_id"]
        for row in audit_rows
        if row["audit_status"] != "pass"
    ]
    summary = {
        "created_at": C.now(),
        "authority_rows": len(rows),
        "canary_report": C.rel(CANARY_REPORT),
        "canary_status": "pass",
        "action_counts": dict(
            Counter(row["restore_action"] for row in audit_rows)
        ),
        "trajectory_authority_sha_exact": sum(
            C.boolish(row["trajectory_authority_sha_exact"])
            for row in audit_rows
        ),
        "audit_pass": sum(
            row["audit_status"] == "pass" for row in audit_rows
        ),
        "audit_failures": failures,
        "status": "pass" if not failures else "fail",
    }
    C.write_json(
        RESTORE_ROOT / "stage2b_restore_summary.json", summary
    )
    if failures:
        raise RuntimeError(f"restore audit failed: {failures}")
    return summary


def restore_missing(rows: list[dict[str, str]]) -> dict[str, Any]:
    if not CANARY_REPORT.is_file():
        raise FileNotFoundError(
            f"run --authority-canary first: {CANARY_REPORT}"
        )
    canary = json.loads(CANARY_REPORT.read_text(encoding="utf-8"))
    if canary.get("status") != "pass":
        raise ValueError("authority canary is not passing")
    actions: dict[str, str] = {}
    for row in rows:
        destination = task_dir(row)
        state = completeness(destination)
        if all(state.values()):
            audit = audit_generated(row, destination)
            if not audit["pass"]:
                raise ValueError(
                    f"existing task differs from authority: "
                    f"{row['case_id']}"
                )
            actions[row["case_id"]] = "preserved_authority_match"
            continue
        if any(state.values()):
            raise ValueError(
                f"partial primary task requires manual audit: "
                f"{row['case_id']} {state}"
            )
        log_path = RESTORE_ROOT / "logs" / f"{row['case_id']}.log"
        with tempfile.TemporaryDirectory(
            prefix=f"__e179_restore_{row['case_id']}_",
            dir=C.TASK_ROOT,
        ) as temporary:
            staged = replay_adapter(
                row, Path(temporary).name, log_path
            )
            audit = audit_generated(row, staged)
            if not audit["pass"]:
                raise RuntimeError(
                    f"staged authority audit failed: "
                    f"{row['case_id']} {audit}"
                )
            commit_from_staging(
                staged, destination, row["target_task"]
            )
        verify_path = (
            RESTORE_ROOT / "verify" / f"{row['case_id']}.json"
        )
        run_command(
            [
                sys.executable,
                str(VERIFY_CASE),
                "--task",
                row["target_task"],
                "--source-scene",
                str(C.repo_path(row["source_scene"])),
                "--trimmed",
                str(C.repo_path(row["trimmed_npz"])),
                "--out",
                str(verify_path),
            ],
            log_path,
        )
        actions[row["case_id"]] = "restored_from_e173_trimmed"
    return audit_restored(rows, actions)


def main() -> int:
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--inspect", action="store_true")
    modes.add_argument("--authority-canary", action="store_true")
    modes.add_argument("--restore-missing", action="store_true")
    parser.add_argument(
        "--canary-case",
        default="box023_20231008_045_p1",
    )
    args = parser.parse_args()
    rows = authority()
    if args.inspect:
        payload = inspect(rows)
    elif args.authority_canary:
        payload = authority_canary(rows, args.canary_case)
    else:
        payload = restore_missing(rows)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
