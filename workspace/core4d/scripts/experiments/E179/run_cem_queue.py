#!/usr/bin/env python3
"""Run one E179 no-PRG CEM queue or merge the five worker shards."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e179_common as C  # noqa: E402


def read_with_fields(
    path: Path,
) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        return list(reader), list(reader.fieldnames or [])


def write_atomic(
    path: Path, rows: list[dict[str, Any]], fields: list[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
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
            writer.writerows(
                {
                    key: C.serial(row.get(key, ""))
                    for key in fields
                }
                for row in rows
            )
        Path(temporary).replace(path)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


def output_paths(row: dict[str, str]) -> dict[str, Path]:
    return {
        "result_npz": C.repo_path(row["result_npz"]),
        "outdir_npz": C.repo_path(row["outdir_npz"]),
        "config_act": C.repo_path(row["config_act"]),
    }


def validate_inputs(row: dict[str, str]) -> list[str]:
    failures = []
    for key, path_key, sha_key in (
        ("override", "override_path", "override_sha256"),
        ("trajectory", "trajectory", "trajectory_sha256"),
        ("scene", "scene_act", "effective_scene_sha256"),
        ("contact_mask", "contact_mask", "contact_mask_sha256"),
    ):
        path = C.repo_path(row[path_key])
        if not path.is_file():
            failures.append(f"missing:{key}")
        elif C.sha256(path) != row[sha_key]:
            failures.append(f"sha_mismatch:{key}")
    if not C.repo_path(row["target_scene"]).is_file():
        failures.append("missing:target_scene")
    if any(
        C.boolish(row.get(flag))
        for flag in ("p_enabled", "r_enabled", "g_enabled")
    ):
        failures.append("manifest_prg_flag_enabled")
    if row.get("scene_name") != C.SCENE_NAME:
        failures.append("scene_name_mismatch")
    return failures


def profile_failures(config: dict[str, Any]) -> list[str]:
    profile_doc = json.loads(
        C.E167A_PROFILE.read_text(encoding="utf-8")
    )
    expected = dict(profile_doc["profile"])
    expected["cem_smooth_enabled"] = False
    failures = []
    for key, value in expected.items():
        actual = config.get(
            key, False if key == "cem_smooth_enabled" else None
        )
        if actual != value:
            failures.append(f"e167a_profile_mismatch:{key}")
    return failures


def validate_runtime_outputs(row: dict[str, str]) -> list[str]:
    failures = []
    paths = output_paths(row)
    for key, path in paths.items():
        if not path.is_file():
            failures.append(f"missing_artifact:{key}")
    if failures:
        return failures
    with np.load(paths["outdir_npz"], allow_pickle=True) as archive:
        if "qpos" not in archive.files:
            failures.append("npz_missing:qpos")
        elif not np.isfinite(
            np.asarray(archive["qpos"], dtype=np.float64)
        ).all():
            failures.append("npz_nonfinite:qpos")
        diagnostic_leaks = sorted(
            key
            for key in archive.files
            if key.startswith(C.PRG_DIAGNOSTIC_PREFIXES)
        )
        failures.extend(
            f"prg_diagnostic_leak:{key}"
            for key in diagnostic_leaks
        )
    config = yaml.safe_load(
        paths["config_act"].read_text(encoding="utf-8")
    )
    if config.get("scene_name") != C.SCENE_NAME:
        failures.append("config_mismatch:scene_name")
    model_path = str(config.get("model_path", ""))
    if not model_path.endswith(f"/{C.SCENE_NAME}.xml"):
        failures.append("config_mismatch:model_path")
    failures.extend(profile_failures(config))

    # Runtime config serialization may include inactive default fields.  Any
    # active lower-body PRG state is forbidden.
    if float(config.get("leg_object_penalty_scale", 0.0) or 0.0) != 0.0:
        failures.append("prg_active:leg_object_penalty_scale")
    if config.get("leg_object_penalty_geom_names", []):
        failures.append("prg_active:leg_object_penalty_geom_names")
    if C.boolish(config.get("cem_leg_gate_enabled", False)):
        failures.append("prg_active:cem_leg_gate_enabled")
    if config.get("cem_leg_gate_geom_names", []):
        failures.append("prg_active:cem_leg_gate_geom_names")
    return failures


def output_complete(row: dict[str, str]) -> bool:
    return all(path.is_file() for path in output_paths(row).values())


def build_command(
    row: dict[str, str], *, python_bin: str
) -> list[str]:
    samples = int(row["cem_samples"])
    iterations = int(row["cem_opt_steps"])
    seed = int(row["cem_seed"])
    return [
        python_bin,
        "-u",
        "examples/run_mjwp.py",
        f"+override={row['override_id']}",
        f"task={row['target_task']}",
        "+use_torch_compile=false",
        "save_video=false",
        "video_camera=auto",
        f"seed={seed}",
        f"num_samples={samples}",
        f"max_num_iterations={iterations}",
        f"output_dir={C.rel(C.repo_path(row['outdir_npz']).parent)}",
        f"video_output_path={row['video']}",
    ]


def run_one(
    row: dict[str, str],
    *,
    rows: list[dict[str, str]],
    fields: list[str],
    manifest: Path,
    python_bin: str,
    gpu_id: str,
    force: bool,
    dry_run: bool,
) -> bool:
    if output_complete(row) and not force:
        failures = validate_runtime_outputs(row)
        row["status"] = (
            "run_complete_pending_eval"
            if not failures
            else "failed_validation"
        )
        row["failure_mode"] = ";".join(failures)
        row["updated_at"] = C.now()
        write_atomic(manifest, rows, fields)
        print(
            f"[skip-complete] {row['variant']} "
            f"status={row['status']}"
        )
        return not failures

    failures = validate_inputs(row)
    if failures:
        row["status"] = "failed_preflight"
        row["failure_mode"] = ";".join(failures)
        row["updated_at"] = C.now()
        write_atomic(manifest, rows, fields)
        print(
            f"[failed-preflight] {row['variant']}: "
            f"{row['failure_mode']}",
            file=sys.stderr,
        )
        return False
    command = build_command(row, python_bin=python_bin)
    if dry_run:
        print(" ".join(command))
        return True

    paths = output_paths(row)
    log_path = C.repo_path(row["log"])
    for path in (*paths.values(), log_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    row["status"] = "running"
    row["failure_mode"] = ""
    row["gpu_id"] = gpu_id
    row["updated_at"] = C.now()
    write_atomic(manifest, rows, fields)
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": gpu_id,
            "MUJOCO_GL": environment.get("MUJOCO_GL", "egl"),
            "PYTHONUNBUFFERED": "1",
        }
    )
    print(
        f"[start] {row['variant']} case={row['case_id']} "
        f"gpu={gpu_id}",
        flush=True,
    )
    with log_path.open("w", encoding="utf-8") as log:
        log.write(
            f"# E179 CEM {row['execution_mode']}\n"
            f"# started_at={C.now()}\n"
            "# command=" + " ".join(command) + "\n\n"
        )
        log.flush()
        completed = subprocess.run(
            command,
            cwd=C.REPO,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode != 0:
        row["status"] = "failed"
        row["failure_mode"] = (
            f"run_mjwp_exit_{completed.returncode}"
        )
        row["updated_at"] = C.now()
        write_atomic(manifest, rows, fields)
        print(
            f"[failed] {row['variant']} "
            f"exit={completed.returncode}",
            file=sys.stderr,
        )
        return False
    if paths["outdir_npz"].is_file():
        shutil.copy2(paths["outdir_npz"], paths["result_npz"])
    failures = validate_runtime_outputs(row)
    row["status"] = (
        "run_complete_pending_eval"
        if not failures
        else "failed_validation"
    )
    row["failure_mode"] = ";".join(failures)
    row["updated_at"] = C.now()
    write_atomic(manifest, rows, fields)
    print(f"[done] {row['variant']} status={row['status']}")
    return not failures


def merge_worker_shards() -> int:
    manifest_root = C.RESULTS / "s6_downstream/manifests"
    canonical = manifest_root / "cem_full_manifest.tsv"
    rows, fields = read_with_fields(canonical)
    by_case = {row["case_id"]: row for row in rows}
    shard_rows = []
    for worker in C.WORKER_QUEUES:
        shard = (
            manifest_root
            / f"cem_full_{worker.replace('-', '_')}.tsv"
        )
        current, current_fields = read_with_fields(shard)
        if current_fields != fields:
            raise ValueError(f"shard field drift: {shard}")
        expected = set(C.WORKER_QUEUES[worker])
        if {row["case_id"] for row in current} != expected:
            raise ValueError(f"shard set drift: {worker}")
        shard_rows.extend(current)
    if len(shard_rows) != C.EXPECTED_PAIRED_ROWS:
        raise ValueError("worker shard union must contain 16 rows")
    for shard_row in shard_rows:
        case_id = shard_row["case_id"]
        canonical_row = by_case[case_id]
        for key in (
            "trajectory_sha256",
            "contact_mask_sha256",
            "effective_scene_sha256",
            "override_sha256",
            "assigned_worker",
        ):
            if canonical_row[key] != shard_row[key]:
                raise ValueError(
                    f"static field drift {case_id}:{key}"
                )
        canonical_row.update(shard_row)
    write_atomic(canonical, rows, fields)
    summary = {
        "merged_at": C.now(),
        "rows": len(rows),
        "status_counts": dict(
            __import__("collections").Counter(
                row["status"] for row in rows
            )
        ),
        "status": "pass",
    }
    C.write_json(
        manifest_root / "cem_full_merge_summary.json", summary
    )
    print(json.dumps(summary, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--merge-worker-shards", action="store_true"
    )
    parser.add_argument("--mode", choices=("canary", "full"))
    parser.add_argument("--manifest-tsv", type=Path)
    parser.add_argument("--python-bin", default=".venv/bin/python")
    parser.add_argument("--gpu-id")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.merge_worker_shards:
        return merge_worker_shards()
    if not args.mode or not args.manifest_tsv or args.gpu_id is None:
        parser.error(
            "--mode, --manifest-tsv and --gpu-id are required"
        )
    manifest = C.repo_path(args.manifest_tsv)
    rows, fields = read_with_fields(manifest)
    expected_mode = (
        "canary" if args.mode == "canary" else "production"
    )
    eligible_statuses = {
        "",
        "not_run",
        "READY_FOR_CANARY",
        "READY_FOR_FULL",
        "failed",
        "failed_validation",
        "failed_preflight",
    }
    indices = [
        index
        for index, row in enumerate(rows)
        if row["execution_mode"] == expected_mode
        and (args.all or row["status"] in eligible_statuses)
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
            python_bin=args.python_bin,
            gpu_id=args.gpu_id,
            force=args.force,
            dry_run=args.dry_run,
        ) and success
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
