#!/usr/bin/env python3
"""E208 P6: dispatch the frozen 105-run CEM queue across the local GPUs.

E206's runner has the admission hard gate and the free-slot GPU pool but writes
its case set from a stage2b manifest and indexes by ``arm``, neither of which can
express 105 (case, aug variant) rows.  E199's queue runner has the row model but
no admission gate and no per-task timeout.  This is E206's gate plus E199's queue
plus the three things both incidents on 2026-09-05 argued for:

* a ``flock`` single-instance lock -- this runner rewrites the whole manifest on
  every status change, so a second instance silently reverts the first's rows
* a per-task timeout (plan238 R14) at 180 min, just above E206's measured
  per-task bound of 177.2 min, so it catches pathological runs without killing
  healthy long ones.  On timeout the half-written ``trajectory_mjwp_act.npz`` is
  unlinked before the row is marked, otherwise the next resume would mistake it
  for a completed run
* a frozen-set check before dispatch: the ``(case_id, variant)`` set must still
  hash to what ``freeze.json`` recorded (C7d).  The runner may only ever write
  status/gpu_id/wall_min/failure_mode/updated_at -- never add or drop a row.

``task=`` is deliberately NOT passed on the command line.  V5e audited the
composed override and proved ``task`` is the single key distinguishing it from
E206's PRG config; passing it again would create a second authority that V5e does
not check.  Preflight instead asserts the override file's own ``task:`` line
matches the manifest row.

Usage:
    .venv/bin/python .../E208/run_e208_cem.py --dry-run
    .venv/bin/python .../E208/run_e208_cem.py --gpus 0,1,2,3,4,5,6,7
    ... --cases desk023_20231030_019_p1 --variants trans0     # smoke subset
    E208_FORCE=1 ...                                          # ignore existing outputs
"""

from __future__ import annotations

import argparse
import json
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

import e208_common as C  # noqa: E402

LOCK = C.RESULTS / ".locks/run_e208_cem.lock"

# A row in any of these may be dispatched.  `cem_timeout` is included so a
# timed-out run is retried on the next invocation rather than silently dropped;
# `cem_ok` and `reused_e206` never are.
ELIGIBLE = {
    "", "READY_FOR_FULL", "failed", "failed_preflight", "failed_validation",
    "failed_postprocess", "cem_timeout",
}
DONE_STATUS = "cem_ok"


def admission() -> dict[str, Any]:
    if not C.ADMISSION_JSON.is_file():
        raise SystemExit(
            f"missing {C.rel(C.ADMISSION_JSON)} -- run recheck_admission.py first; the "
            "queue must be admitted and logged BEFORE any GPU work is dispatched"
        )
    payload = json.loads(C.ADMISSION_JSON.read_text(encoding="utf-8"))
    for gate in ("A1_single_task", "A2_queue"):
        if payload.get(gate, {}).get("verdict") != "pass":
            raise SystemExit(f"admission {gate} did not pass; apply the A3 ladder first")
    return payload


def check_frozen(rows: list[dict[str, str]]) -> str:
    """C7(d) up front: the queue must still be the set that was frozen."""
    if not C.FREEZE_JSON.is_file():
        raise SystemExit(
            f"missing {C.rel(C.FREEZE_JSON)} -- freeze the manifest "
            "(build_aug_manifest.py --freeze) before dispatching"
        )
    frozen = json.loads(C.FREEZE_JSON.read_text(encoding="utf-8"))
    got = C.sha256_text("\n".join(sorted(f"{r['case_id']}|{r['aug_variant']}" for r in rows)))
    if got != frozen["case_variant_set_sha256"]:
        raise SystemExit(
            "the manifest's (case_id, variant) set no longer matches freeze.json\n"
            f"  frozen {frozen['case_variant_set_sha256']}\n  actual {got}\n"
            "Rows may only change status, never appear or disappear. Re-freeze "
            "deliberately if the queue really did change."
        )
    return got


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
    for label, field, sha_field in (
        ("scene", "scene_act", "effective_scene_sha256"),
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

    # The override is the single authority for `task` (see module docstring).
    override = C.repo_path(row["override_path"])
    if override.is_file():
        base_id = None
        for line in override.read_text(encoding="utf-8").splitlines():
            if line.startswith("- core4d_"):
                base_id = line[2:].strip()
                break
        base_yaml = C.OVERRIDE_DIR / f"{base_id}.yaml" if base_id else None
        if base_yaml is None or not base_yaml.is_file():
            fails.append("missing:base_override")
        else:
            task = next(
                (ln.split(":", 1)[1].strip()
                 for ln in base_yaml.read_text(encoding="utf-8").splitlines()
                 if ln.startswith("task:")),
                None,
            )
            if task != row["target_task"]:
                fails.append(f"task_mismatch:{task}!={row['target_task']}")
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


def command(row: dict[str, str], python_bin: str) -> list[str]:
    out_dir = C.repo_path(row["outdir_npz"]).parent
    cmd = [
        python_bin, "-u", "examples/run_mjwp.py",
        f"+override={row['override_id']}",
        "save_video=false", "video_camera=auto",
        "+use_torch_compile=false",
        f"seed={int(row['cem_seed'])}",
        f"num_samples={int(row['cem_samples'])}",
        f"max_num_iterations={int(row['cem_opt_steps'])}",
        f"output_dir={out_dir}",
    ]
    return cmd + row.get("extra_overrides", "").split()


def cem_env(gpu: str) -> dict[str, str]:
    env = dict(os.environ)
    env.setdefault("MUJOCO_GL", os.environ.get("E208_MUJOCO_GL", "disable"))
    env["TORCHDYNAMO_DISABLE"] = "1"      # budget is frozen at use_torch_compile=false
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["PYTHONUNBUFFERED"] = "1"
    return env


def launch(row: dict[str, str], gpu: str, python_bin: str):
    for path in (*out_paths(row).values(), C.repo_path(row["log"])):
        path.parent.mkdir(parents=True, exist_ok=True)
    stream = C.repo_path(row["log"]).open("w", encoding="utf-8")
    cmd = command(row, python_bin)
    stream.write(
        f"# {C.EXP_ID} {C.RUN_ID} ordinal={row['ordinal']} case={row['case_id']} "
        f"aug={row['aug_variant']} band={row['offset_band']} "
        f"retarget={row['effective_retarget_variant']}\n"
        f"# gpu={gpu} started_at={C.now()}\n# command={' '.join(cmd)}\n\n")
    stream.flush()
    proc = subprocess.Popen(cmd, cwd=C.REPO, env=cem_env(gpu),
                            stdout=stream, stderr=subprocess.STDOUT)
    return proc, stream


def kill_timed_out(row: dict[str, str], proc: subprocess.Popen) -> None:
    """SIGTERM, then SIGKILL, then remove the partial rollout.

    Leaving the half-written npz behind would make the next resume classify this
    run as already done -- the failure mode a timeout is supposed to surface.
    """
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)
    for key in ("outdir_npz", "result_npz"):
        path = C.repo_path(row[key])
        if path.is_file():
            path.unlink()


def finalize(row: dict[str, str], returncode: int) -> None:
    if returncode:
        row["status"] = "failed"
        row["failure_mode"] = f"run_mjwp_exit_{returncode}"
        return
    try:
        out = out_paths(row)
        shutil.copy2(out["outdir_npz"], out["result_npz"])
        fails = output_failures(row)
        row["status"] = DONE_STATUS if not fails else "failed_validation"
        row["failure_mode"] = ";".join(fails)
    except Exception as exc:  # noqa: BLE001
        row["status"] = "failed_postprocess"
        row["failure_mode"] = f"{type(exc).__name__}:{exc}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=C.PRIORITY_MANIFEST)
    ap.add_argument("--gpus", default=C.GPU_DEFAULT)
    ap.add_argument("--per-gpu-mem-mib", type=int, default=C.PER_GPU_MEM_MIB)
    ap.add_argument("--max-per-gpu", type=int, default=1)
    ap.add_argument("--per-task-timeout-min", type=float, default=C.PER_TASK_TIMEOUT_MIN)
    ap.add_argument("--python-bin", default=str(C.SPIDER_PYTHON_BIN))
    ap.add_argument("--poll-interval", type=float, default=15.0)
    ap.add_argument("--cases", default="", help="comma-separated case_id filter")
    ap.add_argument("--variants", default="", help="comma-separated aug variant filter")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    adm = admission()
    manifest = C.repo_path(args.manifest)
    rows, fields = C.read_with_fields(manifest)
    set_sha = check_frozen(rows)

    budget = C.frozen_budget()
    bad_budget = [
        f"{r['case_id']}/{r['aug_variant']}" for r in rows
        if (int(r["cem_samples"]), int(r["cem_opt_steps"]), int(r["cem_seed"]))
        != (budget["num_samples"], budget["max_num_iterations"], budget["seed"])
    ]
    if bad_budget:
        raise SystemExit(f"manifest rows disagree with the frozen budget: {bad_budget[:5]}")

    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    case_filter = {c for c in args.cases.split(",") if c}
    var_filter = {v for v in args.variants.split(",") if v}

    order = [
        i for i in sorted(range(len(rows)), key=lambda i: int(rows[i]["ordinal"]))
        if (not case_filter or rows[i]["case_id"] in case_filter)
        and (not var_filter or rows[i]["aug_variant"] in var_filter)
    ]
    if args.limit:
        order = order[: args.limit]

    force = os.environ.get("E208_FORCE", "0") == "1"
    if not force:
        touched = False
        for i in order:
            row = rows[i]
            if row["status"] not in ELIGIBLE:
                continue
            if all(p.is_file() for p in out_paths(row).values()):
                fails = output_failures(row)
                row["status"] = DONE_STATUS if not fails else "failed_validation"
                row["failure_mode"] = ";".join(fails)
                row["updated_at"] = C.now()
                touched = True
        if touched:
            C.write_tsv(manifest, rows, fields)

    pending = [i for i in order if rows[i]["status"] in ELIGIBLE]
    print(f"{C.EXP_ID} {C.RUN_ID}: {len(pending)} pending of {len(order)} selected "
          f"({len(rows)} in manifest), gpus={gpus}, "
          f"budget={budget['num_samples']}x{budget['max_num_iterations']} seed {budget['seed']}")
    print(f"  admission A2: {adm['A2_queue']['queue_runs']} runs, "
          f"optimistic {adm['A2_queue']['optimistic_hours']} h, "
          f"bound {adm['A2_queue']['projected_hours']} h")
    print(f"  frozen set sha {set_sha[:12]} ok; per-task timeout "
          f"{args.per_task_timeout_min} min")

    if args.dry_run:
        for i in pending[:10]:
            r = rows[i]
            print(f"  [{r['ordinal']:>3}] {r['object_key']:9s} {r['aug_variant']:7s} "
                  f"{r['case_id']:32s} band={r['offset_band']:8s}")
            print(f"        {' '.join(command(r, args.python_bin))}")
        if len(pending) > 10:
            print(f"  ... and {len(pending) - 10} more")
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
                        rows[idx].update({
                            "status": "cem_timeout",
                            "failure_mode": f"exceeded_{args.per_task_timeout_min}min",
                            "wall_min": round((time.monotonic() - t0) / 60.0, 2),
                            "updated_at": C.now(),
                        })
                        C.write_tsv(manifest, rows, fields)
                        print(f"[timeout] {rows[idx]['case_id']} {rows[idx]['aug_variant']} "
                              f"gpu={g}", file=sys.stderr, flush=True)
                        continue
                    still.append((idx, proc, stream, t0))
                    continue
                stream.close()
                finalize(rows[idx], rc)
                rows[idx]["wall_min"] = round((time.monotonic() - t0) / 60.0, 2)
                rows[idx]["updated_at"] = C.now()
                C.write_tsv(manifest, rows, fields)
                n_done = sum(1 for i in order if rows[i]["status"] == DONE_STATUS)
                print(f"[done {n_done}/{len(order)}] {rows[idx]['case_id']} "
                      f"{rows[idx]['aug_variant']} gpu={g} -> {rows[idx]['status']} "
                      f"wall={rows[idx]['wall_min']}min", flush=True)
            running[g] = still

        todo = [i for i in order if i not in dispatched and rows[i]["status"] in ELIGIBLE]
        made_progress = False
        for i in list(todo):
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
                    print(f"[preflight-fail] {row['case_id']} {row['aug_variant']}: "
                          f"{row['failure_mode']}", file=sys.stderr, flush=True)
                    placed = True
                    break
                proc, stream = launch(row, g, args.python_bin)
                row.update({"status": "running", "failure_mode": "", "gpu_id": g,
                            "updated_at": C.now()})
                C.write_tsv(manifest, rows, fields)
                running[g].append((i, proc, stream, time.monotonic()))
                dispatched.add(i)
                made_progress = placed = True
                print(f"[start {len(dispatched)}/{len(order)}] [{row['ordinal']}] "
                      f"{row['object_key']} {row['aug_variant']} {row['case_id']} "
                      f"gpu={g} (free={free}MiB)", flush=True)
                break
            if not placed:
                break

        any_running = any(running[g] for g in gpus)
        remaining = [i for i in order if rows[i]["status"] in ELIGIBLE and i not in dispatched]
        if not any_running and not remaining and not made_progress:
            break
        time.sleep(args.poll_interval)

    check_frozen(rows)
    ok = [i for i in order if rows[i]["status"] == DONE_STATUS]
    bad = [f"{rows[i]['case_id']}:{rows[i]['aug_variant']}={rows[i]['status']}"
           for i in order if rows[i]["status"] != DONE_STATUS]
    walls = [float(rows[i]["wall_min"]) for i in ok if rows[i].get("wall_min")]
    print(f"\n[queue-complete] {len(ok)}/{len(order)} {DONE_STATUS} in "
          f"{(time.monotonic() - t_start) / 3600:.2f} h")
    if walls:
        print(f"  wall_min: median {sorted(walls)[len(walls) // 2]:.1f}, max {max(walls):.1f}")
    if bad:
        print(f"  problem rows ({len(bad)}): {bad[:10]}")
    return 0 if not bad else 1


if __name__ == "__main__":
    with C.SingleInstance(LOCK):
        raise SystemExit(main())
