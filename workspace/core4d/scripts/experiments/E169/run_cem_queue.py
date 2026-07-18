#!/usr/bin/env python3
"""Run E169 CEM rows and update a shard manifest atomically."""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml


REPO = Path(__file__).resolve().parents[5]


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO / path


def rel(path: str | Path) -> str:
    value = Path(path)
    try:
        return str(value.resolve().relative_to(REPO.resolve()))
    except (OSError, ValueError):
        return str(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_tsv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        return list(reader), list(reader.fieldnames or [])


def write_tsv_atomic(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=fields, delimiter="\t", lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(rows)
        Path(temporary).replace(path)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


def boolish(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def validate_inputs(row: dict[str, str]) -> list[str]:
    failures: list[str] = []
    checks = {
        "override_path": (row["override_path"], row["override_sha256"]),
        "trajectory": (row["trajectory"], row["trajectory_sha256"]),
        "scene_act": (row["scene_act"], row["effective_scene_sha256"]),
        "contact_mask": (row["contact_mask"], row["contact_mask_sha256"]),
    }
    for key, (raw_path, expected_sha) in checks.items():
        path = repo_path(raw_path)
        if not path.is_file():
            failures.append(f"missing:{key}")
        elif sha256(path) != expected_sha:
            failures.append(f"sha_mismatch:{key}")
    if not repo_path(row["target_scene"]).is_file():
        failures.append("missing:target_scene")
    return failures


def output_paths(row: dict[str, str]) -> dict[str, Path]:
    return {
        "result_npz": repo_path(row["result_npz"]),
        "outdir_npz": repo_path(row["outdir_npz"]),
        "config_act": repo_path(row["config_act"]),
    }


def output_complete(row: dict[str, str]) -> tuple[bool, list[str]]:
    missing = [key for key, path in output_paths(row).items() if not path.is_file()]
    return not missing, missing


def validate_runtime_outputs(row: dict[str, str]) -> list[str]:
    failures: list[str] = []
    paths = output_paths(row)
    with np.load(paths["outdir_npz"], allow_pickle=True) as data:
        if "qpos" not in data:
            failures.append("npz_missing:qpos")
        elif not np.isfinite(np.asarray(data["qpos"], dtype=np.float64)).all():
            failures.append("npz_nonfinite:qpos")
        if boolish(row["g_enabled"]):
            required = {
                "cem_leg_gate_valid_frac",
                "cem_leg_gate_selected_valid_frac",
                "cem_leg_gate_fallback_used",
                "sample_leg_gate_min_sdf_min",
                "sample_leg_gate_violation_pct_mean",
            }
            missing = sorted(required - set(data.files))
            failures.extend(f"npz_missing:{key}" for key in missing)
        if boolish(row["r_enabled"]) and "leg_object_penalty_mean" not in data:
            failures.append("npz_missing:leg_object_penalty_mean")

    config = yaml.safe_load(paths["config_act"].read_text(encoding="utf-8"))
    p_enabled = boolish(row["p_enabled"])
    r_enabled = boolish(row["r_enabled"])
    g_enabled = boolish(row["g_enabled"])
    expected_scene = (
        "scene_act_E169_lowerbody_physics" if p_enabled else "scene_act_E168_rubber_hull"
    )
    checks = {
        "scene_name": expected_scene,
        "leg_object_penalty_scale": 2.0 if r_enabled else 0.0,
        "cem_leg_gate_enabled": g_enabled,
        "cem_leg_gate_min_sdf_m": 0.005,
        "cem_leg_gate_max_violation_pct": 0.02,
        "cem_leg_gate_hard_floor_m": -0.005,
    }
    for key, expected in checks.items():
        if config.get(key) != expected:
            failures.append(f"config_mismatch:{key}")
    if r_enabled and len(config.get("leg_object_penalty_geom_ids", [])) != 16:
        failures.append("config_mismatch:leg_object_penalty_geom_ids")
    if g_enabled and len(config.get("cem_leg_gate_geom_ids", [])) != 16:
        failures.append("config_mismatch:cem_leg_gate_geom_ids")
    model_path = str(config.get("model_path", ""))
    if not model_path.endswith(f"/{expected_scene}.xml"):
        failures.append("config_mismatch:model_path")
    return failures


def build_command(
    row: dict[str, str],
    *,
    python_bin: str,
    mode: str,
    num_samples: int | None,
    max_num_iterations: int | None,
) -> list[str]:
    command = [
        python_bin,
        "-u",
        "examples/run_mjwp.py",
        f"+override={row['override_id']}",
        f"task={row['target_task']}",
        "+use_torch_compile=false",
        "save_video=false",
        "video_camera=auto",
        f"output_dir={rel(repo_path(row['outdir_npz']).parent)}",
        f"video_output_path={rel(row['video'])}",
    ]
    if mode == "canary":
        command.extend(
            [
                f"num_samples={num_samples or 64}",
                f"max_num_iterations={max_num_iterations or 4}",
            ]
        )
    else:
        if num_samples is not None:
            command.append(f"num_samples={num_samples}")
        if max_num_iterations is not None:
            command.append(f"max_num_iterations={max_num_iterations}")
    return command


def run_one(
    row: dict[str, str],
    *,
    rows: list[dict[str, str]],
    fields: list[str],
    manifest: Path,
    mode: str,
    python_bin: str,
    gpu_id: str,
    num_samples: int | None,
    max_num_iterations: int | None,
    force: bool,
    dry_run: bool,
) -> bool:
    complete, _ = output_complete(row)
    if complete and not force:
        failures = validate_runtime_outputs(row)
        row["status"] = "run_complete_pending_eval" if not failures else "failed_validation"
        row["failure_mode"] = ";".join(failures)
        row["updated_at"] = now()
        write_tsv_atomic(manifest, rows, fields)
        print(f"[skip-complete] {row['variant']} validation={row['status']}")
        return not failures

    failures = validate_inputs(row)
    if failures:
        row["status"] = "failed_preflight"
        row["failure_mode"] = ";".join(failures)
        row["updated_at"] = now()
        write_tsv_atomic(manifest, rows, fields)
        print(f"[failed-preflight] {row['variant']}: {row['failure_mode']}", file=sys.stderr)
        return False

    command = build_command(
        row,
        python_bin=python_bin,
        mode=mode,
        num_samples=num_samples,
        max_num_iterations=max_num_iterations,
    )
    if dry_run:
        print(" ".join(command))
        return True

    paths = output_paths(row)
    log_path = repo_path(row["log"])
    for path in (*paths.values(), log_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    row["status"] = "running"
    row["failure_mode"] = ""
    row["gpu_id"] = gpu_id
    row["updated_at"] = now()
    write_tsv_atomic(manifest, rows, fields)

    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": gpu_id,
            "MUJOCO_GL": environment.get("MUJOCO_GL", "egl"),
            "PYTHONUNBUFFERED": "1",
        }
    )
    print(f"[start] {row['variant']} case={row['case_id']} cell={row['cell_id']} gpu={gpu_id}")
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"# E169 CEM {mode}\n# started_at={now()}\n")
        log.write("# command=" + " ".join(command) + "\n\n")
        log.flush()
        process = subprocess.run(
            command, cwd=REPO, env=environment, stdout=log, stderr=subprocess.STDOUT
        )
    if process.returncode != 0:
        row["status"] = "failed"
        row["failure_mode"] = f"run_mjwp_exit_{process.returncode}"
        row["updated_at"] = now()
        write_tsv_atomic(manifest, rows, fields)
        print(f"[failed] {row['variant']} exit={process.returncode}", file=sys.stderr)
        return False
    if not paths["outdir_npz"].is_file():
        row["status"] = "failed"
        row["failure_mode"] = "missing_artifact:outdir_npz"
        row["updated_at"] = now()
        write_tsv_atomic(manifest, rows, fields)
        return False
    shutil.copy2(paths["outdir_npz"], paths["result_npz"])
    complete, missing = output_complete(row)
    failures = [] if complete else [f"missing_artifact:{key}" for key in missing]
    if not failures:
        failures = validate_runtime_outputs(row)
    row["status"] = "run_complete_pending_eval" if not failures else "failed_validation"
    row["failure_mode"] = ";".join(failures)
    row["updated_at"] = now()
    write_tsv_atomic(manifest, rows, fields)
    print(f"[done] {row['variant']} status={row['status']}")
    return not failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("canary", "full"), required=True)
    parser.add_argument("--manifest-tsv", type=Path, required=True)
    parser.add_argument("--python-bin", default=".venv/bin/python")
    parser.add_argument("--gpu-id", required=True)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--num-samples", type=int)
    parser.add_argument("--max-num-iterations", type=int)
    args = parser.parse_args()
    manifest = repo_path(args.manifest_tsv)
    rows, fields = read_tsv(manifest)
    expected_mode = "canary" if args.mode == "canary" else "production"
    indices = [
        index
        for index, row in enumerate(rows)
        if row["execution_mode"] == expected_mode
        and (args.all or row["status"] in {"", "not_run", "failed", "failed_validation"})
    ]
    if not indices:
        print(f"[no-rows] {manifest}")
        return 0
    success = True
    for index in indices:
        success = run_one(
            rows[index],
            rows=rows,
            fields=fields,
            manifest=manifest,
            mode=args.mode,
            python_bin=args.python_bin,
            gpu_id=args.gpu_id,
            num_samples=args.num_samples,
            max_num_iterations=args.max_num_iterations,
            force=args.force,
            dry_run=args.dry_run,
        ) and success
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
