#!/usr/bin/env python3
"""Run E168 CEM rows from a manifest and update row status atomically."""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[5]
DEFAULT_MANIFEST_DIR = REPO / "workspace/core4d/results/E168/s6_downstream/cem/manifests"


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO / path


def rel(path: str | Path) -> str:
    p = Path(path)
    try:
        return str(p.absolute().relative_to(REPO.absolute()))
    except ValueError:
        pass
    for symlink_root in (REPO / "workspace/core4d/results",):
        if not symlink_root.exists():
            continue
        try:
            suffix = p.resolve().relative_to(symlink_root.resolve())
            return str(symlink_root.relative_to(REPO) / suffix)
        except ValueError:
            pass
    try:
        return str(p.resolve().relative_to(REPO.resolve()))
    except Exception:
        return str(path)


def read_tsv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = list(reader)
        return rows, list(reader.fieldnames or [])


def write_tsv_atomic(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t", lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        Path(tmp_name).replace(path)
    except Exception:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def manifest_for_mode(mode: str) -> Path:
    name = "cem_canary_manifest.tsv" if mode == "canary" else "cem_production_manifest.tsv"
    return DEFAULT_MANIFEST_DIR / name


def output_complete(row: dict[str, str], *, require_video: bool = True) -> tuple[bool, list[str]]:
    required = {
        "result_npz": repo_path(row["result_npz"]),
        "outdir_npz": repo_path(row["outdir_npz"]),
        "config_act": repo_path(row["config_act"]),
    }
    if require_video:
        required["video"] = repo_path(row["video"])
    missing = [key for key, path in required.items() if not path.is_file()]
    return not missing, missing


def validate_inputs(row: dict[str, str]) -> list[str]:
    checks = {
        "override_path": repo_path(row["override_path"]),
        "target_scene": repo_path(row["target_scene"]),
        "trajectory": repo_path(row["trajectory"]),
        "scene_act": repo_path(row["scene_act"]),
        "contact_mask": repo_path(row["contact_mask"]),
    }
    return [key for key, path in checks.items() if not path.is_file()]


def select_rows(
    rows: list[dict[str, str]],
    *,
    mode: str,
    variants: set[str],
    cases: set[str],
    only_pending: bool,
    limit: int | None,
) -> list[int]:
    selected: list[int] = []
    for idx, row in enumerate(rows):
        if row.get("execution_mode") != mode:
            continue
        if variants and row.get("variant") not in variants:
            continue
        if cases and row.get("case_id") not in cases:
            continue
        if only_pending:
            complete, _ = output_complete(row)
            if complete:
                continue
            if row.get("status") not in {"", "not_run", "queued", "running", "failed"}:
                continue
        selected.append(idx)
        if limit is not None and len(selected) >= limit:
            break
    return selected


def build_command(
    row: dict[str, str],
    *,
    python_bin: str,
    mode: str,
    num_samples: int | None,
    max_num_iterations: int | None,
    save_video: bool | None,
) -> list[str]:
    override_id = row["override_id"] or Path(row["override_path"]).stem
    cmd = [
        python_bin,
        "-u",
        "examples/run_mjwp.py",
        f"+override={override_id}",
        f"task={row['target_task']}",
        "+use_torch_compile=false",
        "video_camera=auto",
        f"output_dir={rel(repo_path(row['outdir_npz']).parent)}",
        f"video_output_path={rel(row['video'])}",
    ]
    if save_video is not None:
        cmd.append(f"save_video={'true' if save_video else 'false'}")
    if mode == "canary":
        cmd.extend(
            [
                f"num_samples={num_samples if num_samples is not None else 64}",
                f"max_num_iterations={max_num_iterations if max_num_iterations is not None else 4}",
            ]
        )
    else:
        if num_samples is not None:
            cmd.append(f"num_samples={num_samples}")
        if max_num_iterations is not None:
            cmd.append(f"max_num_iterations={max_num_iterations}")
    return cmd


def run_one(
    row: dict[str, str],
    *,
    fields: list[str],
    rows: list[dict[str, str]],
    manifest: Path,
    mode: str,
    python_bin: str,
    gpu_id: str,
    num_samples: int | None,
    max_num_iterations: int | None,
    require_video: bool,
    save_video: bool | None,
    force: bool,
    dry_run: bool,
) -> bool:
    complete, missing_outputs = output_complete(row, require_video=require_video)
    if complete and not force:
        row["status"] = "run_complete_pending_eval"
        row["failure_mode"] = ""
        row["updated_at"] = now()
        write_tsv_atomic(manifest, rows, fields)
        print(f"[skip-complete] {row['variant']}")
        return True

    missing_inputs = validate_inputs(row)
    if missing_inputs:
        row["status"] = "failed_preflight"
        row["failure_mode"] = "missing_input:" + ",".join(missing_inputs)
        row["updated_at"] = now()
        write_tsv_atomic(manifest, rows, fields)
        print(f"[failed-preflight] {row['variant']} missing {missing_inputs}", file=sys.stderr)
        return False

    cmd = build_command(
        row,
        python_bin=python_bin,
        mode=mode,
        num_samples=num_samples,
        max_num_iterations=max_num_iterations,
        save_video=save_video,
    )
    if dry_run:
        print(" ".join(cmd))
        return True

    result_npz = repo_path(row["result_npz"])
    outdir_npz = repo_path(row["outdir_npz"])
    log_path = repo_path(row["log"])
    outdir_npz.parent.mkdir(parents=True, exist_ok=True)
    result_npz.parent.mkdir(parents=True, exist_ok=True)
    repo_path(row["video"]).parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    row["status"] = "running"
    row["failure_mode"] = ""
    row["gpu_id"] = gpu_id
    row["updated_at"] = now()
    write_tsv_atomic(manifest, rows, fields)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_id
    env["MUJOCO_GL"] = env.get("MUJOCO_GL", "egl")
    env["PYTHONUNBUFFERED"] = "1"

    print(f"[start] {row['variant']} case={row['case_id']} gpu={gpu_id} log={rel(log_path)}")
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(f"# E168 CEM {mode} run\n# started_at={now()}\n")
        log_file.write("# command=" + " ".join(cmd) + "\n\n")
        log_file.flush()
        proc = subprocess.run(cmd, cwd=REPO, env=env, stdout=log_file, stderr=subprocess.STDOUT)

    if proc.returncode != 0:
        row["status"] = "failed"
        row["failure_mode"] = f"run_mjwp_exit_{proc.returncode}"
        row["updated_at"] = now()
        write_tsv_atomic(manifest, rows, fields)
        print(f"[failed] {row['variant']} exit={proc.returncode}", file=sys.stderr)
        return False

    if not outdir_npz.is_file():
        row["status"] = "failed"
        row["failure_mode"] = "missing_artifact:outdir_npz"
        row["updated_at"] = now()
        write_tsv_atomic(manifest, rows, fields)
        print(f"[failed] {row['variant']} missing outdir npz", file=sys.stderr)
        return False

    shutil.copy2(outdir_npz, result_npz)
    complete, missing_outputs = output_complete(row, require_video=require_video)
    if not complete:
        row["status"] = "failed"
        row["failure_mode"] = "missing_artifact:" + ",".join(missing_outputs)
        row["updated_at"] = now()
        write_tsv_atomic(manifest, rows, fields)
        print(f"[failed] {row['variant']} missing outputs {missing_outputs}", file=sys.stderr)
        return False

    row["status"] = "run_complete_pending_eval"
    row["failure_mode"] = ""
    row["updated_at"] = now()
    write_tsv_atomic(manifest, rows, fields)
    print(f"[done] {row['variant']}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["canary", "production"], required=True)
    parser.add_argument("--manifest-tsv", type=Path)
    parser.add_argument("--python-bin", default=".venv/bin/python")
    parser.add_argument("--gpu-id", default=os.environ.get("LOCAL_GPU", "0"))
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--variant", action="append", default=[])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--all", action="store_true", help="Run rows regardless of status if outputs are incomplete.")
    parser.add_argument("--force", action="store_true", help="Rerun even when all expected outputs already exist.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--num-samples", type=int)
    parser.add_argument("--max-num-iterations", type=int)
    parser.add_argument("--save-video", choices=["true", "false"])
    parser.add_argument(
        "--allow-missing-video",
        action="store_true",
        help="Treat NPZ/config outputs as complete even when video rendering is disabled.",
    )
    args = parser.parse_args()
    save_video = None
    if args.save_video is not None:
        save_video = args.save_video == "true"

    manifest = (args.manifest_tsv or manifest_for_mode(args.mode)).expanduser()
    if not manifest.is_absolute():
        manifest = REPO / manifest
    rows, fields = read_tsv(manifest)
    selected = select_rows(
        rows,
        mode="canary" if args.mode == "canary" else "production",
        variants=set(args.variant),
        cases=set(args.case_id),
        only_pending=not args.all,
        limit=args.limit,
    )
    if not selected:
        print(f"[no-rows] mode={args.mode} manifest={rel(manifest)}")
        return 0

    ok = True
    for idx in selected:
        ok = (
            run_one(
                rows[idx],
                fields=fields,
                rows=rows,
                manifest=manifest,
                mode=args.mode,
                python_bin=args.python_bin,
                gpu_id=args.gpu_id,
                num_samples=args.num_samples,
                max_num_iterations=args.max_num_iterations,
                require_video=not args.allow_missing_video,
                save_video=save_video,
                force=args.force,
                dry_run=args.dry_run,
            )
            and ok
        )
        if not ok and not args.all:
            break
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
