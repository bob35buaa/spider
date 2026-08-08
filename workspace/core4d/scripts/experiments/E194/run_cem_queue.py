#!/usr/bin/env python3
"""Run one E194 gravity-comp CEM queue (single local GPU shard).

PRG is ON for every arm (this is the production default the 2x2 sits on top of).
Each arm differs from the A0 baseline ONLY by the extra run_mjwp CLI tokens in
the row's `extra_overrides` column:

    G1  scene_name=scene_act_E194_rubberHull_PRG_gravcomp
    G2  init_pos_actuator_gain=2500
    G3  scene_name=scene_act_E194_rubberHull_PRG_gravcomp init_pos_actuator_gain=2500

After each run the landed config_act.yaml is validated to confirm the arm's
mechanism actually took effect (kp, rot untouched, scene_name, PRG still on).
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e194_common as C  # noqa: E402


def write_atomic(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        import csv
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n")
            writer.writeheader()
            writer.writerows({k: C.serial(r.get(k, "")) for k in fields} for r in rows)
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


def output_complete(row: dict[str, str]) -> bool:
    return all(p.is_file() for p in output_paths(row).values())


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
        elif row.get(sha_key) and C.sha256(path) != row[sha_key]:
            failures.append(f"sha_mismatch:{key}")
    if not C.repo_path(row["target_scene"]).is_file():
        failures.append("missing:target_scene")
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
        elif not np.isfinite(np.asarray(archive["qpos"], dtype=np.float64)).all():
            failures.append("npz_nonfinite:qpos")
    config = yaml.safe_load(paths["config_act"].read_text(encoding="utf-8"))
    spec = C.ARMS[row["arm"]]
    exp_scene = C.GRAVCOMP_SCENE_NAME if spec["gravcomp"] else row["scene_name"]
    if config.get("scene_name") != exp_scene:
        failures.append(f"config_scene_name:{config.get('scene_name')}!={exp_scene}")
    if float(config.get("init_pos_actuator_gain", -1)) != float(spec["kp"]):
        failures.append(f"config_init_pos:{config.get('init_pos_actuator_gain')}!={spec['kp']}")
    if float(config.get("init_rot_actuator_gain", -1)) != C.ROT_GAIN:
        failures.append(f"config_init_rot:{config.get('init_rot_actuator_gain')}!={C.ROT_GAIN}")
    if float(config.get("leg_object_penalty_scale", 0.0) or 0.0) != 2.0:
        failures.append("prg_off:leg_object_penalty_scale")
    model_path = str(config.get("model_path", ""))
    if not model_path.endswith(f"/{exp_scene}.xml"):
        failures.append(f"config_model_path:{model_path}")
    return failures


def build_command(row: dict[str, str], *, python_bin: str) -> list[str]:
    cmd = [
        python_bin, "-u", "examples/run_mjwp.py",
        f"+override={row['override_id']}",
        f"task={row['target_task']}",
        "+use_torch_compile=false",
        "save_video=false",
        "video_camera=auto",
        f"seed={int(row['cem_seed'])}",
        f"num_samples={int(row['cem_samples'])}",
        f"max_num_iterations={int(row['cem_opt_steps'])}",
        # outdir_npz is stored repo-relative; run_mjwp runs with cwd=REPO. Do NOT
        # round-trip through repo_path().resolve() -- results/ is a symlink to
        # another mount, so resolve() escapes REPO and rel() would throw.
        f"output_dir={Path(row['outdir_npz']).parent.as_posix()}",
        f"video_output_path={row['video']}",
    ]
    extra = row.get("extra_overrides", "").split()
    cmd.extend(extra)  # scene_name=... and/or init_pos_actuator_gain=2500
    return cmd


def copy_with_retry(src: Path, dst: Path, *, attempts: int = 3, delay_s: float = 5.0) -> None:
    last: OSError | None = None
    for attempt in range(1, attempts + 1):
        try:
            shutil.copy2(src, dst)
            return
        except OSError as exc:
            last = exc
            print(f"[retry] copy2 {src}->{dst} {attempt}/{attempts}: {exc}", file=sys.stderr)
            if attempt < attempts:
                time.sleep(delay_s)
    assert last is not None
    raise last


def run_one(row, *, rows, fields, manifest, python_bin, gpu_id, force, dry_run) -> bool:
    if output_complete(row) and not force:
        failures = validate_runtime_outputs(row)
        row["status"] = "run_complete_pending_eval" if not failures else "failed_validation"
        row["failure_mode"] = ";".join(failures)
        row["updated_at"] = C.now()
        write_atomic(manifest, rows, fields)
        print(f"[skip-complete] {row['variant']} status={row['status']}")
        return not failures

    failures = validate_inputs(row)
    if failures:
        row["status"] = "failed_preflight"
        row["failure_mode"] = ";".join(failures)
        row["updated_at"] = C.now()
        write_atomic(manifest, rows, fields)
        print(f"[failed-preflight] {row['variant']}: {row['failure_mode']}", file=sys.stderr)
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

    env = os.environ.copy()
    env.pop("MUJOCO_GL", None)  # see run_E194_local_8gpu.sh for why
    env.update({"CUDA_VISIBLE_DEVICES": gpu_id, "PYTHONUNBUFFERED": "1"})
    print(f"[start] {row['variant']} case={row['case_id']} arm={row['arm']} gpu={gpu_id}", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"# E194 CEM {row['execution_mode']} arm={row['arm']}\n"
                  f"# started_at={C.now()}\n# command=" + " ".join(command) + "\n\n")
        log.flush()
        completed = subprocess.run(command, cwd=C.REPO, env=env, stdout=log,
                                   stderr=subprocess.STDOUT, check=False)
    if completed.returncode != 0:
        row["status"] = "failed"
        row["failure_mode"] = f"run_mjwp_exit_{completed.returncode}"
        row["updated_at"] = C.now()
        write_atomic(manifest, rows, fields)
        print(f"[failed] {row['variant']} exit={completed.returncode}", file=sys.stderr)
        return False
    try:
        if paths["outdir_npz"].is_file():
            copy_with_retry(paths["outdir_npz"], paths["result_npz"])
        failures = validate_runtime_outputs(row)
    except OSError as exc:
        row["status"] = "failed_postprocess"
        row["failure_mode"] = f"postprocess_error:{type(exc).__name__}:{exc}"
        row["updated_at"] = C.now()
        write_atomic(manifest, rows, fields)
        print(f"[failed-postprocess] {row['variant']}: {exc}", file=sys.stderr)
        return False
    row["status"] = "run_complete_pending_eval" if not failures else "failed_validation"
    row["failure_mode"] = ";".join(failures)
    row["updated_at"] = C.now()
    write_atomic(manifest, rows, fields)
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
    args = parser.parse_args()

    manifest = C.repo_path(args.manifest_tsv)
    rows, fields = C.read_with_fields(manifest)
    expected_mode = "canary" if args.mode == "canary" else "production"
    eligible = {"", "not_run", "READY_FOR_CANARY", "READY_FOR_FULL",
                "failed", "failed_validation", "failed_preflight", "failed_postprocess"}
    indices = [i for i, r in enumerate(rows)
               if r["execution_mode"] == expected_mode and (args.all or r["status"] in eligible)]
    if not indices:
        print(f"[no-rows] {manifest}")
        return 0
    ok = True
    for i in indices:
        try:
            row_ok = run_one(rows[i], rows=rows, fields=fields, manifest=manifest,
                             python_bin=args.python_bin, gpu_id=args.gpu_id,
                             force=args.force, dry_run=args.dry_run)
        except Exception as exc:  # noqa: BLE001 - one bad row must not strand the queue
            r = rows[i]
            r["status"] = "failed_postprocess"
            r["failure_mode"] = f"unhandled:{type(exc).__name__}:{exc}"
            r["updated_at"] = C.now()
            write_atomic(manifest, rows, fields)
            print(f"[failed-unhandled] {r['variant']}: {exc}", file=sys.stderr)
            row_ok = False
        ok = row_ok and ok
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
