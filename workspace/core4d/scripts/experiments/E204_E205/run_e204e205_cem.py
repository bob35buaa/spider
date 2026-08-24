#!/usr/bin/env python3
"""E204 (noPRG) + E205 (G1A2) full CEM driver: 27 cases x 2 arms = 54 runs.

Mirrors the E203 driver (8-GPU pool, one slot = one GPU, resume-safe skip-already-
done), but dispatches the two arms over the SAME 27 E178 bucket tasks. Because both
arms share a task dir, each job writes to a DISTINCT per-arm output_dir so the
rollouts never collide; skip-already-done keys on that per-arm rollout npz.

Prereqs (run once, in this order, on the same machine):
  1. build_arm_scenes.py   -> writes scene_act_E204/E205 into each task dir
  2. build_overrides.py    -> writes + audits the 27x2 override YAMLs

Usage:
    .venv/bin/python .../run_e204e205_cem.py --gpus 0,1,2,3,4,5,6,7          # full 54
    ... --arms noprg_e204                 # one arm only
    ... --limit 1 --num-samples 64 --max-iterations 4   # smoke (canary budget)
    ... --dry-run                         # print commands, run nothing
Env: E204E205_MUJOCO_GL (default 'disable' = headless, no GL backend import; set egl/osmesa to render), E204E205_TORCH_COMPILE (0/1),
     E204E205_FORCE=1 (ignore skip-already-done).
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import e204e205_common as C

REPO = C.REPO
# Use the venv launcher path UNRESOLVED: launching via .venv/bin/python activates the
# venv (pyvenv.cfg); .resolve() would deref the symlink to the system python and lose it.
PY = str(REPO / ".venv/bin/python")


def cem_env(gpu: str | None) -> dict:
    env = dict(os.environ)
    env.setdefault("MUJOCO_GL", os.environ.get("E204E205_MUJOCO_GL", "disable"))
    if os.environ.get("E204E205_TORCH_COMPILE", "0") != "1":
        env["TORCHDYNAMO_DISABLE"] = "1"
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu
    env["PYTHONUNBUFFERED"] = "1"
    return env


def prepare_one(arm: str, case_id: str, *, num_samples: int, max_iters: int,
                seed: int, stage: str) -> dict:
    r = {"arm": arm, "case_id": case_id, "status": "", "note": ""}
    override = C.override_path(arm, case_id)
    scene = C.arm_scene_path(arm, case_id)
    if not override.is_file():
        r["status"] = "skip_missing_override"
        r["note"] = "run build_overrides.py"
        return r
    if not scene.is_file():
        r["status"] = "skip_missing_scene"
        r["note"] = "run build_arm_scenes.py"
        return r
    out_dir = C.arm_out_dir(arm, case_id, stage)
    if C.result_npz(arm, case_id, stage).is_file() and os.environ.get("E204E205_FORCE", "0") != "1":
        r["status"] = "skip_already_done"
        return r
    # run_mjwp creates output_dir itself (os.makedirs); don't litter empty dirs on dry-run.
    r["cmd"] = [
        PY, "-u", "examples/run_mjwp.py",
        f"+override={C.override_id(arm, case_id)}",
        "save_video=false", "video_camera=auto",
        # freeze (plan234): no torch.compile. Also avoids triton JIT needing Python.h
        # (python3.12-dev absent on some boxes). '+' append: key is a dataclass default,
        # not in the composed Hydra struct (same as E179's +use_torch_compile=false).
        "+use_torch_compile=false",
        f"seed={seed}", f"num_samples={num_samples}",
        f"max_num_iterations={max_iters}",
        f"output_dir={out_dir}",
    ]
    r["output_dir"] = str(out_dir.relative_to(REPO))
    r["status"] = "ready"
    return r


def run_parallel(jobs: list[tuple[str, str]], *, gpus: list[str], max_per_gpu: int,
                 num_samples: int, max_iters: int, seed: int, log_dir: Path,
                 dry_run: bool, stage: str) -> list[dict]:
    log_dir.mkdir(parents=True, exist_ok=True)
    prepared, results = [], []
    for arm, case_id in jobs:
        r = prepare_one(arm, case_id, num_samples=num_samples, max_iters=max_iters,
                        seed=seed, stage=stage)
        if r["status"] == "ready":
            prepared.append(r)
        else:
            print(f"   prep {arm}/{case_id} -> {r['status']} {r.get('note','')}", flush=True)
            results.append(r)
    if dry_run:
        for r in prepared:
            r["command"] = " ".join(r["cmd"])
            r["status"] = "dry_run"
            r.pop("cmd", None)
            print(f"   DRY {r['arm']}/{r['case_id']}: {r['command']}", flush=True)
            results.append(r)
        return results
    if not gpus:
        gpus = [None]  # serial on default device
    slots = [(g, s) for g in gpus for s in range(max_per_gpu)]
    running: dict = {}
    pending = list(prepared)
    while pending or running:
        for slot in slots:
            if not pending:
                break
            if slot in running:
                continue
            r = pending.pop(0)
            gpu = slot[0]
            tag = f"{C.ARM_EXP[r['arm']]}_{r['case_id']}"
            fh = open(log_dir / f"{tag}.log", "w")
            fh.write(f"# gpu={gpu} command={' '.join(r['cmd'])}\n\n")
            fh.flush()
            proc = subprocess.Popen(r["cmd"], cwd=REPO, stdout=fh,
                                    stderr=subprocess.STDOUT, env=cem_env(gpu))
            r["gpu"] = gpu
            running[slot] = (proc, r, fh)
            print(f"   launch {tag} on gpu {gpu} ({len(pending)} pending)", flush=True)
        done = [(slot, v) for slot, v in running.items() if v[0].poll() is not None]
        for slot, (proc, r, fh) in done:
            fh.close()
            r["status"] = "cem_ok" if proc.returncode == 0 else f"cem_fail_rc{proc.returncode}"
            r["log"] = str((log_dir / f"{C.ARM_EXP[r['arm']]}_{r['case_id']}.log").relative_to(REPO))
            r.pop("cmd", None)
            print(f"   done {C.ARM_EXP[r['arm']]}_{r['case_id']} -> {r['status']}", flush=True)
            results.append(r)
            del running[slot]
        if running and not done:
            time.sleep(15)
    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default=",".join(C.ARMS),
                    help="comma-separated: noprg_e204,g1a2_e205")
    ap.add_argument("--cases", default="", help="comma-separated case_ids (default: all 27)")
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--max-per-gpu", type=int, default=1)
    ap.add_argument("--num-samples", type=int, default=C.FULL_SAMPLES)
    ap.add_argument("--max-iterations", type=int, default=C.FULL_OPT_STEPS)
    ap.add_argument("--seed", type=int, default=C.CEM_SEED)
    ap.add_argument("--limit", type=int, default=0, help="first N cases per arm (smoke)")
    ap.add_argument("--stage", default="full", choices=["full", "smoke"],
                    help="output subdir; smoke keeps canary rollouts from shadowing full")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="logs+summary dir (default: results/E204/s6_downstream/cem/<stage>/_driver)")
    args = ap.parse_args()

    if args.out_dir is None:
        args.out_dir = REPO / f"workspace/core4d/results/E204/s6_downstream/cem/{args.stage}/_driver"
    elif not args.out_dir.is_absolute():
        args.out_dir = REPO / args.out_dir
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    for a in arms:
        if a not in C.ARMS:
            print(f"unknown arm {a}; valid: {C.ARMS}", file=sys.stderr)
            return 2
    sources = C.load_sources()
    case_ids = [c.strip() for c in args.cases.split(",") if c.strip()] or \
        [s["case_id"] for s in sources]
    if args.limit:
        case_ids = case_ids[: args.limit]
    jobs = [(arm, cid) for arm in arms for cid in case_ids]
    print(f"E204/E205 CEM: {len(arms)} arm(s) x {len(case_ids)} case(s) = {len(jobs)} jobs "
          f"| samples={args.num_samples} iters={args.max_iterations} seed={args.seed}", flush=True)

    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    results = run_parallel(jobs, gpus=gpus, max_per_gpu=args.max_per_gpu,
                           num_samples=args.num_samples, max_iters=args.max_iterations,
                           seed=args.seed, log_dir=args.out_dir, dry_run=args.dry_run,
                           stage=args.stage)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = args.out_dir / "e204e205_cem_summary.json"
    summary.write_text(json.dumps(results, indent=2), encoding="utf-8")
    counts = collections.Counter(r["status"] for r in results)
    print("STATUS COUNTS:", dict(counts))
    print("summary ->", summary.relative_to(REPO))
    # non-zero exit if any hard failure (so the launcher surfaces it)
    return 1 if any(str(r["status"]).startswith("cem_fail") for r in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
