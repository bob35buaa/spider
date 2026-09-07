#!/usr/bin/env python3
"""E213 Phase A step 4: dispatch one shard's selected-arm aug CEM across local GPUs.

Modeled on ``run_e208_cem.py`` (E206's admission-grade preflight + E199's GPU
pool + a per-task timeout + a flock single-instance lock), specialised for E213:

* Reads ONE shard manifest (``--shard A|B|C|D``); four 8-GPU machines share this
  /mnt and the runner rewrites the whole file on every status change, so each
  machine must own a disjoint file.  merge_shards.py reconciles them.
* Every row runs its own selected-arm gravcomp scene (``scene_name`` differs per
  case), so ``output_failures`` checks each run's ``config_act.scene_name``
  against the manifest row -- the single check that catches a scene mis-wiring.
* ``task=`` is NOT passed on the command line: the E213 override inherits the
  E208 PRG override, which sets ``task``; passing it again would create a second
  authority.
* Records ``host`` so a four-machine run can attribute every row to a machine.

Usage:
    .venv/bin/python .../E213/run_source_cem.py --shard A --gpus 0,1,2,3,4,5,6,7
    ... --dry-run
    E213_FORCE=1 ...        # re-run rows whose outputs already exist
"""

from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

import e213_common as C  # noqa: E402

ELIGIBLE = {"", "READY_FOR_FULL", "failed", "failed_preflight", "failed_validation",
            "failed_postprocess", "cem_timeout"}
DONE_STATUS = "cem_ok"
HOST = socket.gethostname()


def gpu_free_mib(gpu: str) -> int:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits", "-i", gpu],
        text=True, capture_output=True, check=False)
    try:
        return int(out.stdout.strip().splitlines()[0])
    except (ValueError, IndexError):
        return -1


def input_failures(row: dict[str, str]) -> list[str]:
    fails: list[str] = []
    for label, field, sha_field in (
        ("scene", "selected_scene_act", "selected_scene_sha256"),
        ("trajectory", "trajectory", "trajectory_sha256"),
        ("override", "override_path", "override_sha256"),
        ("contact", "contact_mask", "contact_mask_sha256"),
    ):
        path = C.repo_path(row[field])
        if not path.is_file():
            fails.append(f"missing:{label}")
        elif row[sha_field] and C.sha256(path) != row[sha_field]:
            fails.append(f"sha:{label}")
    if not C.repo_path(row["target_scene"]).is_file():
        fails.append("missing:target_scene")
    return fails


def output_failures(row: dict[str, str]) -> list[str]:
    npz = C.repo_path(row["outdir_npz"])
    cfg_path = C.repo_path(row["config_act"])
    fails: list[str] = []
    if not npz.is_file():
        fails.append("missing:outdir_npz")
    if not cfg_path.is_file():
        fails.append("missing:config_act")
    if fails:
        return fails
    with np.load(npz, allow_pickle=True) as arch:
        if "qpos" not in arch.files:
            fails.append("npz_missing:qpos")
        elif not np.isfinite(np.asarray(arch["qpos"], dtype=np.float64)).all():
            fails.append("npz_nonfinite:qpos")
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if cfg.get("scene_name") != row["scene_name"]:
        fails.append(f"scene_name:{cfg.get('scene_name')}!={row['scene_name']}")
    return fails


def command(row: dict[str, str], python_bin: str) -> list[str]:
    out_dir = C.repo_path(row["outdir_npz"]).parent
    return [
        python_bin, "-u", "examples/run_mjwp.py",
        f"+override={row['override_id']}",
        "save_video=false", "video_camera=auto", "+use_torch_compile=false",
        f"seed={int(row['cem_seed'])}",
        f"num_samples={int(row['cem_samples'])}",
        f"max_num_iterations={int(row['cem_opt_steps'])}",
        f"output_dir={out_dir}",
    ]


def cem_env(gpu: str) -> dict[str, str]:
    env = dict(os.environ)
    env.setdefault("MUJOCO_GL", os.environ.get("E213_MUJOCO_GL", "disable"))
    env["TORCHDYNAMO_DISABLE"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["PYTHONUNBUFFERED"] = "1"
    return env


def launch(row: dict[str, str], gpu: str, python_bin: str):
    for key in ("outdir_npz", "config_act", "log"):
        C.repo_path(row[key]).parent.mkdir(parents=True, exist_ok=True)
    stream = C.repo_path(row["log"]).open("w", encoding="utf-8")
    cmd = command(row, python_bin)
    stream.write(f"# {C.EXP_ID} {C.RUN_ID} ordinal={row['ordinal']} shard={row['shard']} "
                 f"host={HOST} case={row['case_id']} aug={row['aug_variant']} arm={row['arm']}\n"
                 f"# gpu={gpu} started_at={C.now()}\n# command={' '.join(cmd)}\n\n")
    stream.flush()
    proc = subprocess.Popen(cmd, cwd=C.REPO, env=cem_env(gpu), stdout=stream,
                            stderr=subprocess.STDOUT)
    return proc, stream


def kill_timed_out(row: dict[str, str], proc: subprocess.Popen) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)
    npz = C.repo_path(row["outdir_npz"])
    if npz.is_file():
        npz.unlink()


def finalize(row: dict[str, str], returncode: int) -> None:
    if returncode:
        row["status"] = "failed"
        row["failure_mode"] = f"run_mjwp_exit_{returncode}"
        return
    fails = output_failures(row)
    row["status"] = DONE_STATUS if not fails else "failed_validation"
    row["failure_mode"] = ";".join(fails)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", required=True, choices=C.SHARDS)
    ap.add_argument("--gpus", default=C.GPU_DEFAULT)
    ap.add_argument("--per-gpu-mem-mib", type=int, default=C.PER_GPU_MEM_MIB)
    ap.add_argument("--max-per-gpu", type=int, default=1)
    ap.add_argument("--per-task-timeout-min", type=float, default=C.PER_TASK_TIMEOUT_MIN)
    ap.add_argument("--python-bin", default=str(C.SPIDER_PYTHON_BIN))
    ap.add_argument("--poll-interval", type=float, default=15.0)
    ap.add_argument("--cases", default="")
    ap.add_argument("--variants", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    manifest = C.manifest_path(args.shard)
    if not manifest.is_file():
        raise SystemExit(f"missing shard manifest {C.rel(manifest)}; run build_manifest.py first")
    rows, fields = C.read_with_fields(manifest)

    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    case_filter = {c for c in args.cases.split(",") if c}
    var_filter = {v for v in args.variants.split(",") if v}
    order = [i for i in sorted(range(len(rows)), key=lambda i: int(rows[i]["ordinal"]))
             if (not case_filter or rows[i]["case_id"] in case_filter)
             and (not var_filter or rows[i]["aug_variant"] in var_filter)]
    if args.limit:
        order = order[: args.limit]

    force = os.environ.get("E213_FORCE", "0") == "1"
    if not force:
        touched = False
        for i in order:
            row = rows[i]
            if row["status"] not in ELIGIBLE:
                continue
            if C.repo_path(row["outdir_npz"]).is_file() and C.repo_path(row["config_act"]).is_file():
                fails = output_failures(row)
                row["status"] = DONE_STATUS if not fails else "failed_validation"
                row["failure_mode"] = ";".join(fails)
                row["host"] = row.get("host") or HOST
                row["updated_at"] = C.now()
                touched = True
        if touched:
            C.write_tsv(manifest, rows, fields)

    pending = [i for i in order if rows[i]["status"] in ELIGIBLE]
    print(f"{C.EXP_ID} {C.RUN_ID} shard {args.shard} @ {HOST}: {len(pending)} pending of "
          f"{len(order)} selected ({len(rows)} in shard), gpus={gpus}")

    if args.dry_run:
        for i in pending[:12]:
            r = rows[i]
            print(f"  [{r['ordinal']:>3}] {r['object_key']:9s} {r['arm']:4s} {r['aug_variant']:7s} "
                  f"{r['case_id']:32s} scene={r['scene_name']}")
        if len(pending) > 12:
            print(f"  ... and {len(pending) - 12} more")
        return 0

    timeout_s = args.per_task_timeout_min * 60.0
    running: dict[str, list[tuple[int, Any, Any, float]]] = {g: [] for g in gpus}
    dispatched: set[int] = set()
    t_start = time.monotonic()

    while True:
        for g in gpus:
            still = []
            for idx, proc, stream, t0 in running[g]:
                rc = proc.poll()
                if rc is None:
                    if time.monotonic() - t0 > timeout_s:
                        kill_timed_out(rows[idx], proc)
                        stream.close()
                        rows[idx].update({"status": "cem_timeout",
                                          "failure_mode": f"exceeded_{args.per_task_timeout_min}min",
                                          "wall_min": round((time.monotonic() - t0) / 60.0, 2),
                                          "host": HOST, "updated_at": C.now()})
                        C.write_tsv(manifest, rows, fields)
                        print(f"[timeout] {rows[idx]['case_id']} {rows[idx]['aug_variant']} gpu={g}",
                              file=sys.stderr, flush=True)
                        continue
                    still.append((idx, proc, stream, t0))
                    continue
                stream.close()
                finalize(rows[idx], rc)
                rows[idx].update({"wall_min": round((time.monotonic() - t0) / 60.0, 2),
                                  "host": HOST, "updated_at": C.now()})
                C.write_tsv(manifest, rows, fields)
                n_done = sum(1 for i in order if rows[i]["status"] == DONE_STATUS)
                print(f"[done {n_done}/{len(order)}] {rows[idx]['case_id']} {rows[idx]['aug_variant']} "
                      f"{rows[idx]['arm']} gpu={g} -> {rows[idx]['status']} wall={rows[idx]['wall_min']}min",
                      flush=True)
            running[g] = still

        todo = [i for i in order if i not in dispatched and rows[i]["status"] in ELIGIBLE]
        made_progress = False
        for i in list(todo):
            placed = False
            for g in gpus:
                if len(running[g]) >= args.max_per_gpu:
                    continue
                if gpu_free_mib(g) < args.per_gpu_mem_mib:
                    continue
                row = rows[i]
                fails = input_failures(row)
                if fails:
                    row.update({"status": "failed_preflight", "failure_mode": ";".join(fails),
                                "host": HOST, "updated_at": C.now()})
                    C.write_tsv(manifest, rows, fields)
                    dispatched.add(i)
                    print(f"[preflight-fail] {row['case_id']} {row['aug_variant']}: {row['failure_mode']}",
                          file=sys.stderr, flush=True)
                    placed = True
                    break
                proc, stream = launch(row, g, args.python_bin)
                row.update({"status": "running", "failure_mode": "", "gpu_id": g,
                            "host": HOST, "updated_at": C.now()})
                C.write_tsv(manifest, rows, fields)
                running[g].append((i, proc, stream, time.monotonic()))
                dispatched.add(i)
                made_progress = placed = True
                print(f"[start {len(dispatched)}/{len(order)}] [{row['ordinal']}] {row['object_key']} "
                      f"{row['arm']} {row['aug_variant']} {row['case_id']} gpu={g}", flush=True)
                break
            if not placed:
                break

        any_running = any(running[g] for g in gpus)
        remaining = [i for i in order if rows[i]["status"] in ELIGIBLE and i not in dispatched]
        if not any_running and not remaining and not made_progress:
            break
        time.sleep(args.poll_interval)

    ok = [i for i in order if rows[i]["status"] == DONE_STATUS]
    bad = [f"{rows[i]['case_id']}:{rows[i]['aug_variant']}={rows[i]['status']}"
           for i in order if rows[i]["status"] != DONE_STATUS]
    walls = [float(rows[i]["wall_min"]) for i in ok if rows[i].get("wall_min")]
    print(f"\n[shard {args.shard} complete] {len(ok)}/{len(order)} {DONE_STATUS} in "
          f"{(time.monotonic() - t_start) / 3600:.2f} h")
    if walls:
        print(f"  wall_min: median {sorted(walls)[len(walls) // 2]:.1f}, max {max(walls):.1f}")
    if bad:
        print(f"  problem rows ({len(bad)}): {bad[:10]}")
    return 0 if not bad else 1


if __name__ == "__main__":
    lock = C.RESULTS / f".locks/run_source_cem.{socket.gethostname()}.lock"
    with C.SingleInstance(lock):
        raise SystemExit(main())
