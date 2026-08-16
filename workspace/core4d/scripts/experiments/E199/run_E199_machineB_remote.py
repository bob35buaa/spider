#!/usr/bin/env python3
"""E199 full-scale machineB REMOTE queue -- self-contained 8-GPU CEM scheduler.

Standalone by design: it must run on a second machine that has a working SPIDER
repo + venv (examples/run_mjwp.py, mujoco_warp, hydra) but does NOT have the E199
python package. It therefore imports only stdlib + numpy + yaml, and reads its
manifest as plain dict rows -- no e199_common / e173_common import chain.

Dispatches manifest rows across the given GPUs (one run per GPU) only when free
memory >= --per-gpu-mem-mib; never preempts foreign jobs; resume-safe (skips rows
whose outputs already exist and validate). Writes status back into the manifest.

Run from the repo root, e.g.:
  .venv/bin/python run_E199_machineB_remote.py \
    --manifest workspace/core4d/results/E199/s6_downstream/manifests/e199_fullscale_machineB_manifest.tsv \
    --gpus 0,1,2,3,4,5,6,7
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import yaml

# Must be launched from the remote repo root; all manifest paths are repo-relative.
REPO = Path.cwd()

TIER_RANK = {"P0": 0, "P1": 1, "P2": 2}
ELIGIBLE = {"", "READY_FOR_FULL", "failed", "failed_preflight", "failed_validation", "failed_postprocess"}


def repo_path(value: str) -> Path:
    p = Path(str(value))
    return p if p.is_absolute() else REPO / p


def read_tsv(path: Path) -> tuple[list[dict], list[str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        rows = list(reader)
        return rows, list(reader.fieldnames or [])


def write_tsv(path: Path, rows: list[dict], fields: list[str]) -> None:
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    with os.fdopen(fd, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, delimiter="\t", lineterminator="\n", extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    Path(tmp).replace(path)


def gpu_free_mib(gpu: str) -> int:
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits", "-i", gpu],
                         text=True, capture_output=True, check=False)
    try:
        return int(out.stdout.strip().splitlines()[0])
    except (ValueError, IndexError):
        return -1


def out_paths(row: dict) -> dict[str, Path]:
    return {k: repo_path(row[k]) for k in ("result_npz", "outdir_npz", "config_act")}


def input_failures(row: dict) -> list[str]:
    fails = []
    for label, field in (("scene", "scene_act"), ("trajectory", "trajectory"),
                         ("override", "override_path"), ("contact", "contact_mask"),
                         ("target_scene", "target_scene")):
        if not repo_path(row[field]).is_file():
            fails.append(f"missing:{label}")
    return fails


def output_failures(row: dict) -> list[str]:
    out = out_paths(row)
    fails = [f"missing:{k}" for k, p in out.items() if not p.is_file()]
    if fails:
        return fails
    with np.load(out["outdir_npz"], allow_pickle=True) as arch:
        if "qpos" not in arch.files:
            fails.append("npz_missing:qpos")
        elif not np.isfinite(np.asarray(arch["qpos"], dtype=np.float64)).all():
            fails.append("npz_nonfinite:qpos")
    cfg = yaml.safe_load(out["config_act"].read_text(encoding="utf-8"))
    if cfg.get("scene_name") != row["scene_name"]:
        fails.append(f"scene_name:{cfg.get('scene_name')}")
    return fails


def command(row: dict, python_bin: str) -> list[str]:
    return [python_bin, "-u", "examples/run_mjwp.py", f"+override={row['override_id']}",
            f"task={row['target_task']}", "+use_torch_compile=false", "save_video=false",
            "video_camera=auto", f"seed={int(row['cem_seed'])}", f"num_samples={int(row['cem_samples'])}",
            f"max_num_iterations={int(row['cem_opt_steps'])}",
            f"output_dir={Path(row['outdir_npz']).parent.as_posix()}",
            f"video_output_path={row['video']}"] + row.get("extra_overrides", "").split()


def launch(row: dict, gpu: str, python_bin: str):
    out = out_paths(row)
    for p in (*out.values(), repo_path(row["log"])):
        p.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.pop("MUJOCO_GL", None)
    env.update({"CUDA_VISIBLE_DEVICES": gpu, "PYTHONUNBUFFERED": "1"})
    stream = repo_path(row["log"]).open("w", encoding="utf-8")
    cmd = command(row, python_bin)
    stream.write(f"# E199 machineB remote {row['case_id']} {row['aug_variant']} gpu={gpu}\n# {' '.join(cmd)}\n\n")
    stream.flush()
    return subprocess.Popen(cmd, cwd=REPO, env=env, stdout=stream, stderr=subprocess.STDOUT), stream


def finalize(row: dict, rc: int) -> None:
    if rc:
        row["status"], row["failure_mode"] = "failed", f"run_mjwp_exit_{rc}"
        return
    out = out_paths(row)
    try:
        shutil.copy2(out["outdir_npz"], out["result_npz"])
        fails = output_failures(row)
        row["status"] = "run_complete_pending_eval" if not fails else "failed_validation"
        row["failure_mode"] = ";".join(fails)
    except Exception as exc:  # noqa: BLE001
        row["status"], row["failure_mode"] = "failed_postprocess", f"{type(exc).__name__}:{exc}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--per-gpu-mem-mib", type=int, default=5000)
    ap.add_argument("--max-per-gpu", type=int, default=1)
    ap.add_argument("--python-bin", default=".venv/bin/python")
    ap.add_argument("--poll-interval", type=float, default=15.0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    manifest = repo_path(str(args.manifest))
    rows, fields = read_tsv(manifest)
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    order = sorted(range(len(rows)), key=lambda i: (TIER_RANK.get(rows[i]["tier"], 9),
                                                    rows[i]["object_key"], rows[i]["aug_variant"]))

    for i in order:
        row = rows[i]
        if row["status"] in ELIGIBLE and all(p.is_file() for p in out_paths(row).values()):
            fails = output_failures(row)
            row["status"] = "run_complete_pending_eval" if not fails else "failed_validation"
            row["failure_mode"] = ";".join(fails)
    write_tsv(manifest, rows, fields)

    if args.dry_run:
        pending = [rows[i] for i in order if rows[i]["status"] in ELIGIBLE]
        print(f"[dry-run] {len(pending)} pending across gpus={gpus}")
        for r in pending[:8]:
            print("  " + " ".join(command(r, args.python_bin)))
        return 0

    running: dict[str, list] = {g: [] for g in gpus}
    dispatched: set[int] = set()
    while True:
        for g in gpus:
            still = []
            for idx, proc, stream in running[g]:
                if proc.poll() is None:
                    still.append((idx, proc, stream))
                    continue
                stream.close()
                finalize(rows[idx], proc.returncode)
                write_tsv(manifest, rows, fields)
                print(f"[done] {rows[idx]['case_id']} {rows[idx]['aug_variant']} gpu={g} -> {rows[idx]['status']}", flush=True)
            running[g] = still

        for i in [i for i in order if i not in dispatched and rows[i]["status"] in ELIGIBLE]:
            placed = False
            for g in gpus:
                if len(running[g]) >= args.max_per_gpu or gpu_free_mib(g) < args.per_gpu_mem_mib:
                    continue
                fails = input_failures(rows[i])
                if fails:
                    rows[i].update({"status": "failed_preflight", "failure_mode": ";".join(fails)})
                    write_tsv(manifest, rows, fields)
                    dispatched.add(i)
                    print(f"[preflight-fail] {rows[i]['case_id']} {rows[i]['aug_variant']}: {fails}", file=sys.stderr)
                    placed = True
                    break
                proc, stream = launch(rows[i], g, args.python_bin)
                rows[i].update({"status": "running", "failure_mode": "", "gpu_id": g})
                write_tsv(manifest, rows, fields)
                running[g].append((i, proc, stream))
                dispatched.add(i)
                placed = True
                print(f"[start] {rows[i]['object_key']} {rows[i]['case_id']} {rows[i]['aug_variant']} gpu={g}", flush=True)
                break
            if not placed:
                break

        remaining = [i for i in order if rows[i]["status"] in ELIGIBLE and i not in dispatched]
        if not any(running[g] for g in gpus) and not remaining:
            break
        time.sleep(args.poll_interval)

    done = sum(1 for i in order if rows[i]["status"] == "run_complete_pending_eval")
    bad = [rows[i]["case_id"] + ":" + rows[i]["aug_variant"] for i in order
           if rows[i]["status"] != "run_complete_pending_eval"]
    print(f"[queue-complete] {done}/{len(order)} ok; problems: {bad}")
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
