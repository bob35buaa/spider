#!/usr/bin/env python3
"""Run one E178 legacy replay into E187's isolated compatibility root."""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from build_e178_compat_replay import (
    DEFAULT_ROOT,
    REPRESENTATIVE_CASES,
    build,
    display_path,
)
from freeze_authority import REPO_ROOT, sha256_file


def now() -> str:
    """Return a timezone-aware status timestamp."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def repo_path(value: str | Path) -> Path:
    """Resolve a repository-relative artifact path."""
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _read(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if reader.fieldnames is None:
            raise RuntimeError(f"missing TSV header: {path}")
        return list(reader.fieldnames), list(reader)


def _write(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream, delimiter="\t", fieldnames=fields, lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def build_command(row: dict[str, str], *, python_bin: str) -> list[str]:
    """Build the explicit frozen legacy replay command."""
    output_dir = display_path(repo_path(row["outdir_npz"]).parent)
    return [
        python_bin,
        "-u",
        "examples/run_mjwp.py",
        f"+override={row['override_id']}",
        f"task={row['target_task']}",
        "+use_torch_compile=false",
        "save_video=false",
        "video_camera=auto",
        f"output_dir={output_dir}",
        f"video_output_path={row['video']}",
        "num_samples=1024",
        "max_num_iterations=32",
        "seed=0",
    ]


def validate_inputs(row: dict[str, str]) -> None:
    """Require exact E178 override/scene/trajectory/mask authority."""
    checks = {
        "override": (row["override_path"], row["override_sha256"]),
        "scene": (row["scene_act"], row["effective_scene_sha256"]),
        "trajectory": (row["trajectory"], row["trajectory_sha256"]),
        "contact_mask": (row["contact_mask"], row["contact_mask_sha256"]),
    }
    for kind, (value, expected) in checks.items():
        path = repo_path(value)
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"{row['case_id']}: {kind} missing or SHA changed")
    if (row["cem_samples"], row["cem_opt_steps"], row["cem_seed"]) != (
        "1024",
        "32",
        "0",
    ):
        raise RuntimeError(f"{row['case_id']}: CEM budget changed")


def outputs_complete(row: dict[str, str]) -> bool:
    """Return whether all required compatibility artifacts are finite/readable."""
    result = repo_path(row["result_npz"])
    outdir = repo_path(row["outdir_npz"])
    config = repo_path(row["config_act"])
    if not result.is_file() or not outdir.is_file() or not config.is_file():
        return False
    if sha256_file(result) != sha256_file(outdir):
        return False
    with np.load(result, allow_pickle=True) as data:
        return "qpos" in data and np.isfinite(np.asarray(data["qpos"])).all()


def run(case_id: str, *, root: Path, python_bin: str, gpu_id: str) -> bool:
    """Execute or validate one resumable compatibility replay."""
    build(root)
    manifest = root / "execution_manifest.tsv"
    fields, rows = _read(manifest)
    matches = [row for row in rows if row["case_id"] == case_id]
    if len(matches) != 1:
        raise RuntimeError(f"expected one compat row for {case_id}")
    row = matches[0]
    validate_inputs(row)
    if outputs_complete(row):
        row["status"] = "run_complete_pending_eval"
        row["failure_mode"] = ""
        row["updated_at"] = now()
        _write(manifest, fields, rows)
        print(f"[skip-complete] {case_id}")
        return True
    if row["status"] == "running":
        raise RuntimeError(f"refusing to overlap running compat row: {case_id}")

    command = build_command(row, python_bin=python_bin)
    result = repo_path(row["result_npz"])
    outdir = repo_path(row["outdir_npz"])
    config = repo_path(row["config_act"])
    log = repo_path(row["log"])
    for path in (result, outdir, config, log):
        path.parent.mkdir(parents=True, exist_ok=True)
    row["status"] = "running"
    row["failure_mode"] = ""
    row["gpu_id"] = gpu_id
    row["updated_at"] = now()
    _write(manifest, fields, rows)

    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": gpu_id,
            "MUJOCO_GL": environment.get("MUJOCO_GL", "egl"),
            "PYTHONUNBUFFERED": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    with log.open("w", encoding="utf-8") as stream:
        stream.write(f"# E187 E178-compat replay\n# started_at={now()}\n")
        stream.write("# command=" + " ".join(command) + "\n\n")
        stream.flush()
        process = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
    if process.returncode == 0 and outdir.is_file():
        shutil.copy2(outdir, result)
    success = process.returncode == 0 and outputs_complete(row)
    row["status"] = "run_complete_pending_eval" if success else "failed"
    row["failure_mode"] = (
        "" if success else f"run_or_validation_exit_{process.returncode}"
    )
    row["updated_at"] = now()
    _write(manifest, fields, rows)
    print(f"[done] {case_id} status={row['status']}")
    return success


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("case_id", choices=REPRESENTATIVE_CASES)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--python-bin", default=".venv/bin/python")
    parser.add_argument("--gpu-id", default="0")
    parser.add_argument("--print-command", action="store_true")
    args = parser.parse_args()
    build(args.root)
    if args.print_command:
        _, rows = _read(args.root / "execution_manifest.tsv")
        row = next(row for row in rows if row["case_id"] == args.case_id)
        print(" ".join(build_command(row, python_bin=args.python_bin)))
        return 0
    return (
        0
        if run(
            args.case_id,
            root=args.root,
            python_bin=args.python_bin,
            gpu_id=args.gpu_id,
        )
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
