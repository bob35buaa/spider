#!/usr/bin/env python3
"""E214: dispatch the 200-row ablation CEM manifest across local GPUs.

Cloned from E213/run_source_cem.py (E199 GPU pool + per-task timeout + flock
single-instance lock) but single-machine (no A/B/C/D shard): one manifest, one
runner, 8 GPUs round-robin.  The command uses ``load_config_path`` (load each
case's baseline config_act.yaml) + the ablation toggle tokens + output_dir
redirected to the E214 tree -- so nothing under any baseline experiment is
touched.

Usage:
    .venv/bin/python .../E214/run_ablation_cem.py --gpus 0,1,2,3,4,5,6,7
    ... --dry-run
    ... --cases <id,...> --ablations A1_contactHDMI_only,... --limit N
    E214_FORCE=1 ...   # re-run rows whose outputs already exist
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
import e214_common as C  # noqa: E402

ELIGIBLE = {"", "READY", "failed", "failed_preflight", "failed_validation", "cem_timeout"}
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
    cfg = C.repo_path(row["baseline_config_act"])
    if not cfg.is_file():
        fails.append("missing:baseline_config_act")
    elif row["baseline_config_act_sha256"] and C.sha256(cfg) != row["baseline_config_act_sha256"]:
        fails.append("sha:baseline_config_act")
    for label, field in (("model", "run_model_path"),
                         ("data", "run_data_path"),
                         ("contact", "run_contact_mask_path")):
        if row.get(field) and not C.repo_path(row[field]).is_file():
            fails.append(f"missing:{label}")
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
    # verify the ablation toggles actually landed in the saved config
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    for tok in row["toggles"].split():
        key, val = tok.split("=", 1)
        got = cfg.get(key)
        want = val.lower()
        got_s = str(got).lower()
        if want in ("true", "false"):
            if got_s != want:
                fails.append(f"toggle:{key}={got}!={val}")
        else:
            try:
                if abs(float(got) - float(val)) > 1e-9:
                    fails.append(f"toggle:{key}={got}!={val}")
            except (TypeError, ValueError):
                fails.append(f"toggle:{key}={got}!={val}")
    return fails


def command(row: dict[str, str], python_bin: str) -> list[str]:
    out_dir = C.repo_path(row["outdir_npz"]).parent
    cfg_abs = C.repo_path(row["baseline_config_act"])
    # Load the baseline config, then override on top: (a) run plumbing,
    # (b) local-resolved baseline inputs (config's own paths point at another
    # machine), (c) the ablation toggle.  hydra_token() adds '+' for keys absent
    # from default.yaml so Hydra accepts them.
    overrides: dict[str, Any] = {
        "load_config_path": str(cfg_abs),
        "output_dir": str(out_dir),
        "save_video": False,
        "save_config": True,
        "use_torch_compile": False,
        "model_path": str(C.repo_path(row["run_model_path"])),
        "data_path": str(C.repo_path(row["run_data_path"])),
    }
    if row.get("run_contact_mask_path"):
        overrides["contact_hdmi_mask_path"] = str(C.repo_path(row["run_contact_mask_path"]))
    cmd = [python_bin, "-u", "examples/run_mjwp.py"]
    cmd += [C.hydra_token(k, v) for k, v in overrides.items()]
    for tok in row["toggles"].split():
        key, val = tok.split("=", 1)
        cmd.append(C.hydra_token(key, val))
    return cmd


def cem_env(gpu: str) -> dict[str, str]:
    env = dict(os.environ)
    env.setdefault("MUJOCO_GL", os.environ.get("E214_MUJOCO_GL", "disable"))
    env["TORCHDYNAMO_DISABLE"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["PYTHONUNBUFFERED"] = "1"
    return env


def launch(row: dict[str, str], gpu: str, python_bin: str):
    for key in ("outdir_npz", "config_act", "log"):
        C.repo_path(row[key]).parent.mkdir(parents=True, exist_ok=True)
    stream = C.repo_path(row["log"]).open("w", encoding="utf-8")
    cmd = command(row, python_bin)
    stream.write(f"# {C.EXP_ID} {C.RUN_ID} ordinal={row['ordinal']} host={HOST} "
                 f"case={row['case_id']} ablation={row['ablation']} src={row['source_exp']}\n"
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
    ap.add_argument("--gpus", default=C.GPU_DEFAULT)
    ap.add_argument("--per-gpu-mem-mib", type=int, default=C.PER_GPU_MEM_MIB)
    ap.add_argument("--max-per-gpu", type=int, default=1)
    ap.add_argument("--per-task-timeout-min", type=float, default=C.PER_TASK_TIMEOUT_MIN)
    ap.add_argument("--python-bin", default=str(C.SPIDER_PYTHON_BIN))
    ap.add_argument("--poll-interval", type=float, default=15.0)
    ap.add_argument("--cases", default="")
    ap.add_argument("--ablations", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    manifest = C.MANIFEST
    if not manifest.is_file():
        raise SystemExit(f"missing manifest {C.rel(manifest)}; run build_manifest.py first")
    rows, fields = C.read_with_fields(manifest)

    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    case_filter = {c for c in args.cases.split(",") if c}
    abl_filter = {a for a in args.ablations.split(",") if a}
    order = [i for i in sorted(range(len(rows)), key=lambda i: int(rows[i]["ordinal"]))
             if (not case_filter or rows[i]["case_id"] in case_filter)
             and (not abl_filter or rows[i]["ablation"] in abl_filter)]
    if args.limit:
        order = order[: args.limit]

    force = os.environ.get("E214_FORCE", "0") == "1"
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
    print(f"{C.EXP_ID} {C.RUN_ID} @ {HOST}: {len(pending)} pending of {len(order)} selected "
          f"({len(rows)} total), gpus={gpus}")

    if args.dry_run:
        for i in pending[:12]:
            r = rows[i]
            print(f"  [{r['ordinal']:>3}] {r['object_key']:9s} {r['ablation']:22s} {r['case_id']:30s}")
            print(f"        cmd: {' '.join(command(r, args.python_bin))}")
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
                        print(f"[timeout] {rows[idx]['case_id']} {rows[idx]['ablation']} gpu={g}",
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
                print(f"[done {n_done}/{len(order)}] {rows[idx]['case_id']} {rows[idx]['ablation']} "
                      f"gpu={g} -> {rows[idx]['status']} wall={rows[idx]['wall_min']}min", flush=True)
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
                    print(f"[preflight-fail] {row['case_id']} {row['ablation']}: {row['failure_mode']}",
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
                      f"{row['ablation']} {row['case_id']} gpu={g}", flush=True)
                break
            if not placed:
                break

        any_running = any(running[g] for g in gpus)
        remaining = [i for i in order if rows[i]["status"] in ELIGIBLE and i not in dispatched]
        if not any_running and not remaining and not made_progress:
            break
        time.sleep(args.poll_interval)

    ok = [i for i in order if rows[i]["status"] == DONE_STATUS]
    bad = [f"{rows[i]['case_id']}:{rows[i]['ablation']}={rows[i]['status']}"
           for i in order if rows[i]["status"] != DONE_STATUS]
    walls = [float(rows[i]["wall_min"]) for i in ok if rows[i].get("wall_min")]
    print(f"\n[E214 complete] {len(ok)}/{len(order)} {DONE_STATUS} in "
          f"{(time.monotonic() - t_start) / 3600:.2f} h")
    if walls:
        print(f"  wall_min: median {sorted(walls)[len(walls) // 2]:.1f}, max {max(walls):.1f}")
    if bad:
        print(f"  problem rows ({len(bad)}): {bad[:12]}")
    return 0 if not bad else 1


if __name__ == "__main__":
    lock = C.LOCKS_DIR / f"run_ablation_cem.{socket.gethostname()}.lock"
    with C.SingleInstance(lock):
        raise SystemExit(main())
