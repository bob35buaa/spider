#!/usr/bin/env python3
"""Resume-safe single-GPU queue runner for E196 corrected-reference Full."""

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
import e196_reference_fix_common as C  # noqa: E402

from spider.simulators.scene_act_reference import resolve_scene_act_reference  # noqa: E402


def write_atomic(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
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
                extrasaction="ignore",
            )
            writer.writeheader()
            writer.writerows(
                {key: C.serial(row.get(key, "")) for key in fields} for row in rows
            )
        Path(temporary).replace(path)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


def outputs(row: dict[str, str]) -> dict[str, Path]:
    return {
        key: C.repo_path(row[key])
        for key in ("result_npz", "outdir_npz", "config_act")
    }


def input_failures(row: dict[str, str]) -> list[str]:
    failures: list[str] = []
    for label, field, sha_field in (
        ("override", "override_path", "override_sha256"),
        ("trajectory", "trajectory", "trajectory_sha256"),
        ("scene", "scene_act", "effective_scene_sha256"),
        ("meta", "scene_act_meta_path", "scene_act_meta_sha256"),
        ("contact", "contact_mask", "contact_mask_sha256"),
    ):
        path = C.repo_path(row[field])
        if not path.is_file() or path.stat().st_size == 0:
            failures.append(f"missing:{label}")
        elif C.sha256(path) != row[sha_field]:
            failures.append(f"sha:{label}")
    if failures:
        return failures
    model = mujoco.MjModel.from_xml_path(str(C.repo_path(row["scene_act"])))
    try:
        resolved = resolve_scene_act_reference(
            C.repo_path(row["scene_act"]), model, emit_log=False
        )
    except (FileNotFoundError, ValueError) as exc:
        failures.append(f"reference_contract:{exc}")
        return failures
    if (
        resolved.convention != row["resolved_euler_convention"]
        or resolved.meta_sha256 != row["scene_act_meta_sha256"]
    ):
        failures.append("reference_manifest_parity")
    return failures


def output_failures(row: dict[str, str]) -> list[str]:
    paths = outputs(row)
    failures = [f"missing:{key}" for key, path in paths.items() if not path.is_file()]
    if failures:
        return failures
    with np.load(paths["outdir_npz"], allow_pickle=True) as archive:
        if "qpos" not in archive.files:
            failures.append("npz_missing:qpos")
        elif not np.isfinite(np.asarray(archive["qpos"], dtype=np.float64)).all():
            failures.append("npz_nonfinite:qpos")
    config = yaml.safe_load(paths["config_act"].read_text(encoding="utf-8"))
    checks = {
        "scene_name": (config.get("scene_name"), C.SCENE_NAME),
        "kp_pos": (float(config.get("init_pos_actuator_gain", -1)), C.KP_POS),
        "kp_rot": (float(config.get("init_rot_actuator_gain", -1)), C.KP_ROT),
        "samples": (int(config.get("num_samples", -1)), C.FULL_SAMPLES),
        "iterations": (int(config.get("max_num_iterations", -1)), C.FULL_OPT_STEPS),
        "seed": (int(config.get("seed", -1)), C.CEM_SEED),
        "prg": (float(config.get("leg_object_penalty_scale", 0) or 0), 2.0),
    }
    for name, (actual, expected) in checks.items():
        if actual != expected:
            failures.append(f"config:{name}:{actual}!={expected}")
    return failures


def command(row: dict[str, str], python_bin: str) -> list[str]:
    return [
        python_bin,
        "-u",
        "examples/run_mjwp.py",
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
        *row["extra_overrides"].split(),
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-tsv", type=Path, required=True)
    parser.add_argument("--gpu-id", required=True)
    parser.add_argument("--python-bin", default=".venv/bin/python")
    parser.add_argument("--worker")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    manifest = C.repo_path(args.manifest_tsv)
    rows, fields = C.read_with_fields(manifest)
    eligible = {
        "", "READY_FOR_FULL", "failed", "failed_preflight", "failed_validation",
        "failed_postprocess",
    }
    indices = [
        index
        for index, row in enumerate(rows)
        if (not args.worker or row["worker"] == args.worker)
        and (args.force or row["status"] in eligible)
    ]
    ok = True
    for index in indices:
        row = rows[index]
        output = outputs(row)
        if all(path.is_file() for path in output.values()) and not args.force:
            failures = output_failures(row)
            row["status"] = (
                "run_complete_pending_eval" if not failures else "failed_validation"
            )
            row["failure_mode"] = ";".join(failures)
            row["updated_at"] = C.now()
            write_atomic(manifest, rows, fields)
            ok = ok and not failures
            print(f"[skip] {row['variant']} {row['status']}")
            continue
        failures = input_failures(row)
        if failures:
            row["status"] = "failed_preflight"
            row["failure_mode"] = ";".join(failures)
            row["updated_at"] = C.now()
            write_atomic(manifest, rows, fields)
            print(
                f"[preflight-fail] {row['variant']}: {row['failure_mode']}",
                file=sys.stderr,
            )
            ok = False
            continue
        cmd = command(row, args.python_bin)
        if args.dry_run:
            print(" ".join(cmd))
            continue
        for path in (*output.values(), C.repo_path(row["log"])):
            path.parent.mkdir(parents=True, exist_ok=True)
        row.update(
            {
                "status": "running",
                "failure_mode": "",
                "gpu_id": args.gpu_id,
                "updated_at": C.now(),
            }
        )
        write_atomic(manifest, rows, fields)
        env = os.environ.copy()
        env.pop("MUJOCO_GL", None)
        env.update({"CUDA_VISIBLE_DEVICES": args.gpu_id, "PYTHONUNBUFFERED": "1"})
        log = C.repo_path(row["log"])
        print(
            f"[start] {row['variant']} worker={row['worker']} gpu={args.gpu_id}",
            flush=True,
        )
        with log.open("w", encoding="utf-8") as stream:
            stream.write(
                f"# E196 corrected-reference {row['execution_mode']}\n"
                f"# started_at={C.now()}\n# command={' '.join(cmd)}\n\n"
            )
            stream.flush()
            done = subprocess.run(
                cmd,
                cwd=C.REPO,
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if done.returncode:
            row["status"] = "failed"
            row["failure_mode"] = f"run_mjwp_exit_{done.returncode}"
            ok = False
        else:
            try:
                shutil.copy2(output["outdir_npz"], output["result_npz"])
                failures = output_failures(row)
                row["status"] = (
                    "run_complete_pending_eval"
                    if not failures
                    else "failed_validation"
                )
                row["failure_mode"] = ";".join(failures)
                ok = ok and not failures
            except Exception as exc:  # noqa: BLE001
                row["status"] = "failed_postprocess"
                row["failure_mode"] = f"{type(exc).__name__}:{exc}"
                ok = False
        row["updated_at"] = C.now()
        write_atomic(manifest, rows, fields)
        print(f"[done] {row['variant']} {row['status']}")
    if not indices:
        print(f"[no-rows] {manifest} worker={args.worker or '*'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
