#!/usr/bin/env python3
"""Run one E195 Full manifest shard with required input/config/output validation."""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e195_common as C  # noqa: E402


def write_atomic(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=fields, delimiter="\t", lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(
                {key: C.serial(row.get(key, "")) for key in fields} for row in rows
            )
        Path(tmp).replace(path)
    except Exception:
        Path(tmp).unlink(missing_ok=True)
        raise


def output_paths(row: dict[str, str]) -> dict[str, Path]:
    return {
        "result_npz": C.repo_path(row["result_npz"]),
        "outdir_npz": C.repo_path(row["outdir_npz"]),
        "config_act": C.repo_path(row["config_act"]),
    }


def validate_inputs(row: dict[str, str]) -> list[str]:
    failures = []
    for key in ("override_path", "scene_act", "trajectory", "contact_mask", "target_scene"):
        if not C.repo_path(row[key]).is_file():
            failures.append(f"missing:{key}")
    return failures


def validate_runtime_outputs(row: dict[str, str]) -> list[str]:
    failures: list[str] = []
    paths = output_paths(row)
    for key, path in paths.items():
        if not path.is_file():
            failures.append(f"missing_artifact:{key}")
    if failures:
        return failures
    with np.load(paths["outdir_npz"], allow_pickle=True) as archive:
        if "qpos" not in archive.files:
            failures.append("npz_missing:qpos")
        elif not np.isfinite(np.asarray(archive["qpos"], dtype=np.float64)).all():
            failures.append("npz_nonfinite:qpos")
        for key in C.GATE_DIAGNOSTIC_KEYS:
            if key not in archive.files:
                failures.append(f"npz_missing:{key}")
    config = yaml.safe_load(paths["config_act"].read_text(encoding="utf-8"))
    for key, expected in C.E195_GATE.items():
        if not np.isclose(float(config.get(key, np.nan)), expected):
            failures.append(f"config:{key}={config.get(key)}!={expected}")
    if float(config.get("init_pos_actuator_gain", -1)) != C.KP_POS:
        failures.append("config:kp_pos")
    if float(config.get("init_rot_actuator_gain", -1)) != C.KP_ROT:
        failures.append("config:kp_rot")
    if float(config.get("leg_object_penalty_scale", 0) or 0) != 2.0:
        failures.append("config:prg_off")
    if config.get("scene_name") != row["scene_name"]:
        failures.append("config:scene_name")
    model = mujoco.MjModel.from_xml_path(str(C.repo_path(row["scene_act"])))
    object_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    if object_id < 0 or float(model.body_gravcomp[object_id]) != 0.0:
        failures.append("model:gravcomp_not_zero")
    return failures


def build_command(row: dict[str, str], python_bin: str) -> list[str]:
    command = [
        python_bin, "-u", "examples/run_mjwp.py",
        f"+override={row['override_id']}",
        f"task={row['target_task']}",
        "+use_torch_compile=false",
        "save_video=false",
        "video_camera=auto",
        f"seed={int(row['cem_seed'])}",
        f"num_samples={int(row['cem_samples'])}",
        f"max_num_iterations={int(row['cem_opt_steps'])}",
        f"output_dir={Path(row['outdir_npz']).parent.as_posix()}",
        f"video_output_path={row['video']}",
    ]
    command.extend(row["extra_overrides"].split())
    return command


def run_one(
    row: dict[str, str], *, rows: list[dict[str, str]], fields: list[str],
    manifest: Path, python_bin: str, gpu_id: str, dry_run: bool,
) -> bool:
    failures = validate_inputs(row)
    if failures:
        row.update(status="failed_preflight", failure_mode=";".join(failures), updated_at=C.now())
        write_atomic(manifest, rows, fields)
        print(f"[failed-preflight] {row['variant']}: {row['failure_mode']}", file=sys.stderr)
        return False
    command = build_command(row, python_bin)
    if dry_run:
        print(" ".join(command))
        return True
    paths = output_paths(row)
    log_path = C.repo_path(row["log"])
    for path in (*paths.values(), log_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    row.update(status="running", failure_mode="", gpu_id=gpu_id, updated_at=C.now())
    write_atomic(manifest, rows, fields)
    env = os.environ.copy()
    env.pop("MUJOCO_GL", None)
    env.update(CUDA_VISIBLE_DEVICES=gpu_id, PYTHONUNBUFFERED="1")
    with log_path.open("w", encoding="utf-8") as log:
        log.write("# E195 CEM\n# command=" + " ".join(command) + "\n\n")
        log.flush()
        completed = subprocess.run(
            command, cwd=C.REPO, env=env, stdout=log,
            stderr=subprocess.STDOUT, check=False,
        )
    if completed.returncode:
        row.update(status="failed", failure_mode=f"run_mjwp_exit_{completed.returncode}", updated_at=C.now())
        write_atomic(manifest, rows, fields)
        return False
    if paths["outdir_npz"].is_file():
        shutil.copy2(paths["outdir_npz"], paths["result_npz"])
    failures = validate_runtime_outputs(row)
    row.update(
        status="run_complete_pending_eval" if not failures else "failed_validation",
        failure_mode=";".join(failures), updated_at=C.now(),
    )
    write_atomic(manifest, rows, fields)
    print(f"[done] {row['variant']} status={row['status']}")
    return not failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("full",), default="full")
    parser.add_argument("--manifest-tsv", type=Path, required=True)
    parser.add_argument("--python-bin", default=".venv/bin/python")
    parser.add_argument("--gpu-id", required=True)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    manifest = C.repo_path(args.manifest_tsv)
    rows, fields = C.read_with_fields(manifest)
    eligible = {"READY_FOR_RUN", "failed", "failed_preflight", "failed_validation"}
    indices = [
        index for index, row in enumerate(rows)
        if row.get("status", "") in eligible or (args.all and args.dry_run)
    ]
    ok = True
    for index in indices:
        ok = run_one(
            rows[index], rows=rows, fields=fields, manifest=manifest,
            python_bin=args.python_bin, gpu_id=args.gpu_id, dry_run=args.dry_run,
        ) and ok
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

