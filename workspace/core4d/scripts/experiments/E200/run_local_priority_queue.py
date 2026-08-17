#!/usr/bin/env python3
"""Local 8-GPU priority scheduler for one E200 arm's augmentation full-CEM queue.

Dispatches a single arm's manifest rows across the given GPUs, one run per GPU by
default. Coexists with other jobs: a GPU is used only when its free memory is
>= --per-gpu-mem-mib; never kills/preempts foreign processes. Resume-safe (writes
the manifest after every row with a JuiceFS-EIO-safe replace).

Shared filesystem, two machines: launch this once per arm (noprg on one machine,
prg_g1a2 on the other). The two arms use SEPARATE manifests + arm-tagged output
dirs, so the two machines never contend on the same TSV or clobber each other's
rollouts. Within one arm, run exactly one queue instance.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e200_common as C  # noqa: E402

ELIGIBLE = {"", "READY_FOR_FULL", "failed", "failed_preflight", "failed_validation", "failed_postprocess"}


def gpu_free_mib(gpu: str) -> int:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits", "-i", gpu],
        text=True, capture_output=True, check=False)
    try:
        return int(out.stdout.strip().splitlines()[0])
    except (ValueError, IndexError):
        return -1


def out_paths(row: dict[str, str]) -> dict[str, Path]:
    return {k: C.repo_path(row[k]) for k in ("result_npz", "outdir_npz", "config_act")}


def input_failures(row: dict[str, str]) -> list[str]:
    fails: list[str] = []
    for label, field, sha_field in (("scene", "scene_act", "effective_scene_sha256"),
                                    ("trajectory", "trajectory", "trajectory_sha256"),
                                    ("override", "override_path", "override_sha256"),
                                    ("contact", "contact_mask", "contact_mask_sha256")):
        path = C.repo_path(row[field])
        if not path.is_file():
            fails.append(f"missing:{label}")
        elif row[sha_field] and C.sha256(path) != row[sha_field]:
            fails.append(f"sha:{label}")
    if not C.repo_path(row["target_scene"]).is_file():
        fails.append("missing:target_scene")
    return fails


def output_failures(row: dict[str, str]) -> list[str]:
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


def command(row: dict[str, str], python_bin: str,
            samples: int | None = None, opt_steps: int | None = None) -> list[str]:
    n_samples = samples if samples else int(row["cem_samples"])
    n_steps = opt_steps if opt_steps else int(row["cem_opt_steps"])
    cmd = [python_bin, "-u", "examples/run_mjwp.py", f"+override={row['override_id']}",
           f"task={row['target_task']}", "+use_torch_compile=false", "save_video=false",
           "video_camera=auto", f"seed={int(row['cem_seed'])}", f"num_samples={n_samples}",
           f"max_num_iterations={n_steps}",
           f"output_dir={Path(row['outdir_npz']).parent.as_posix()}",
           f"video_output_path={row['video']}"]
    extra = row.get("extra_overrides", "").split()
    return cmd + extra


def launch(row: dict[str, str], gpu: str, python_bin: str,
           samples: int | None = None, opt_steps: int | None = None):
    out = out_paths(row)
    for path in (*out.values(), C.repo_path(row["log"])):
        path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.pop("MUJOCO_GL", None)
    env.update({"CUDA_VISIBLE_DEVICES": gpu, "PYTHONUNBUFFERED": "1"})
    log = C.repo_path(row["log"])
    stream = log.open("w", encoding="utf-8")
    cmd = command(row, python_bin, samples, opt_steps)
    stream.write(f"# E200 priority queue {row['tier']} {row['experiment']} {row['arm']}\n"
                 f"# case={row['case_id']} aug={row['aug_variant']} gpu={gpu} started_at={C.now()}\n"
                 f"# command={' '.join(cmd)}\n\n")
    stream.flush()
    proc = subprocess.Popen(cmd, cwd=C.REPO, env=env, stdout=stream, stderr=subprocess.STDOUT)
    return proc, stream


def finalize(row: dict[str, str], returncode: int) -> None:
    if returncode:
        row["status"] = "failed"
        row["failure_mode"] = f"run_mjwp_exit_{returncode}"
        return
    out = out_paths(row)
    try:
        shutil.copy2(out["outdir_npz"], out["result_npz"])
        fails = output_failures(row)
        row["status"] = "run_complete_pending_eval" if not fails else "failed_validation"
        row["failure_mode"] = ";".join(fails)
    except Exception as exc:  # noqa: BLE001
        row["status"] = "failed_postprocess"
        row["failure_mode"] = f"{type(exc).__name__}:{exc}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=C.ARMS, help="convenience: sets --manifest to this arm's manifest")
    ap.add_argument("--manifest", type=Path, default=None)
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--per-gpu-mem-mib", type=int, default=5000)
    ap.add_argument("--max-per-gpu", type=int, default=1)
    ap.add_argument("--python-bin", default=".venv/bin/python")
    ap.add_argument("--poll-interval", type=float, default=15.0)
    ap.add_argument("--limit", type=int, default=0, help="smoke: dispatch at most N pending rows then exit")
    ap.add_argument("--samples-override", type=int, default=0, help="smoke: override num_samples (0=manifest)")
    ap.add_argument("--opt-steps-override", type=int, default=0, help="smoke: override max_num_iterations (0=manifest)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    smk_s = args.samples_override or None
    smk_i = args.opt_steps_override or None

    if args.manifest is None:
        if not args.arm:
            ap.error("pass --arm or --manifest")
        args.manifest = C.manifest_path(args.arm)

    manifest = C.repo_path(args.manifest)
    rows, fields = C.read_with_fields(manifest)
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    order = sorted(range(len(rows)),
                   key=lambda i: (C.TIER_RANK[rows[i]["tier"]], rows[i]["object_key"], rows[i]["aug_variant"]))

    # resume: mark already-complete rows
    for i in order:
        row = rows[i]
        if row["status"] not in ELIGIBLE:
            continue
        if all(p.is_file() for p in out_paths(row).values()):
            fails = output_failures(row)
            row["status"] = "run_complete_pending_eval" if not fails else "failed_validation"
            row["failure_mode"] = ";".join(fails)
            row["updated_at"] = C.now()
    C.write_tsv(manifest, rows, fields)

    eligible_order = [i for i in order if rows[i]["status"] in ELIGIBLE]
    if args.limit:
        eligible_order = eligible_order[:args.limit]
    allow = set(eligible_order)

    if args.dry_run:
        print(f"[dry-run] {len(eligible_order)} pending across gpus={gpus} mem>={args.per_gpu_mem_mib}MiB")
        for i in eligible_order[:12]:
            r = rows[i]
            print(f"  {r['tier']} {r['object_key']} {r['aug_variant']} :: "
                  f"{' '.join(command(r, args.python_bin, smk_s, smk_i))}")
        return 0

    running: dict[str, list[tuple[int, Any, Any]]] = {g: [] for g in gpus}
    dispatched: set[int] = set()

    while True:
        for g in gpus:
            still = []
            for idx, proc, stream in running[g]:
                rc = proc.poll()
                if rc is None:
                    still.append((idx, proc, stream))
                    continue
                stream.close()
                finalize(rows[idx], rc)
                rows[idx]["updated_at"] = C.now()
                C.write_tsv(manifest, rows, fields)
                print(f"[done] {rows[idx]['tier']} {rows[idx]['case_id']} {rows[idx]['aug_variant']} "
                      f"gpu={g} -> {rows[idx]['status']}", flush=True)
            running[g] = still

        pending = [i for i in eligible_order if i not in dispatched and rows[i]["status"] in ELIGIBLE]
        made_progress = False
        for i in list(pending):
            placed = False
            for g in gpus:
                if len(running[g]) >= args.max_per_gpu:
                    continue
                free = gpu_free_mib(g)
                if free < args.per_gpu_mem_mib:
                    continue
                row = rows[i]
                fails = input_failures(row)
                if fails:
                    row.update({"status": "failed_preflight", "failure_mode": ";".join(fails),
                                "updated_at": C.now()})
                    C.write_tsv(manifest, rows, fields)
                    dispatched.add(i)
                    print(f"[preflight-fail] {row['case_id']} {row['aug_variant']}: {row['failure_mode']}",
                          file=sys.stderr, flush=True)
                    placed = True
                    break
                proc, stream = launch(row, g, args.python_bin, smk_s, smk_i)
                row.update({"status": "running", "failure_mode": "", "gpu_id": g, "updated_at": C.now()})
                C.write_tsv(manifest, rows, fields)
                running[g].append((i, proc, stream))
                dispatched.add(i)
                made_progress = True
                placed = True
                print(f"[start] {row['tier']} {row['object_key']} {row['aug_variant']} gpu={g} (free={free}MiB)",
                      flush=True)
                break
            if not placed:
                break

        any_running = any(running[g] for g in gpus)
        remaining = [i for i in eligible_order if rows[i]["status"] in ELIGIBLE and i not in dispatched]
        if not any_running and not remaining and not made_progress:
            break
        time.sleep(args.poll_interval)

    done = sum(1 for i in allow if rows[i]["status"] == "run_complete_pending_eval")
    bad = [f"{rows[i]['case_id']}:{rows[i]['aug_variant']}" for i in allow
           if rows[i]["status"] != "run_complete_pending_eval"]
    print(f"[queue-complete] {done}/{len(allow)} ok; problem rows: {bad}")
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
