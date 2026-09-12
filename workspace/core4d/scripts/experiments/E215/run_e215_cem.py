#!/usr/bin/env python3
"""E215 P5: local GPU priority scheduler for the rot-aug full-CEM queue.

Modelled on E200's ``run_local_priority_queue.py`` (one run per GPU, free-slot
placement, resume-safe manifest rewrite) plus the two safeguards E208 argued for:

* a ``flock`` single-instance lock -- the runner rewrites the whole manifest on
  every status change, so a second instance would silently revert the first's rows
* a per-task timeout (default 180 min); on timeout the partial rollout is removed
  before the row is marked, so the next resume does not mistake it for complete
* a frozen-set check (C7d) when freeze.json exists: the ``(case_id, variant)`` set
  must still hash to what was frozen; the runner may only write
  status/gpu_id/wall_min/failure_mode/updated_at.

``task=`` IS passed on the CLI (matching E200/E213), and equals the aug base
yaml's ``task:`` -- preflight/input_failures do not need to cross-check it here
because the override's defaults chain is still the authority; the CLI value is
byte-identical.

Usage:
    .venv/bin/python .../E215/run_e215_cem.py --dry-run
    .venv/bin/python .../E215/run_e215_cem.py --gpus 0,1,2,3,4,5,6,7
    ... --cases box021_20231011_034_p1 --variants rot0    # smoke subset
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

import e215_common as C  # noqa: E402

ELIGIBLE = {"", "READY_FOR_FULL", "failed", "failed_preflight", "failed_validation",
            "failed_postprocess", "cem_timeout"}
DONE_STATUS = "run_complete_pending_eval"


def check_frozen(rows: list[dict[str, str]], *, allow_subset: bool = False) -> str | None:
    """Full manifest: (case,variant) set must hash to freeze.json exactly.

    Shard manifest (``allow_subset``): its rows must be a SUBSET of the frozen
    set (read from the frozen manifest), so a shard cannot smuggle in a row that
    was never frozen -- but three machines can each run a disjoint third.
    """
    if not C.FREEZE_JSON.is_file():
        return None
    frozen = json.loads(C.FREEZE_JSON.read_text(encoding="utf-8"))
    pairs = sorted(f"{r['case_id']}|{r['aug_variant']}" for r in rows)
    got = C.sha256_text("\n".join(pairs))
    if got == frozen["case_variant_set_sha256"]:
        return got
    if allow_subset and C.FROZEN_MANIFEST.is_file():
        full = {f"{r['case_id']}|{r['aug_variant']}" for r in C.read_tsv(C.FROZEN_MANIFEST)}
        extra = sorted(set(pairs) - full)
        if extra:
            raise SystemExit(f"shard has rows not in the frozen set: {extra[:5]}")
        return f"subset:{got[:12]}"
    raise SystemExit(
        "manifest (case_id, variant) set no longer matches freeze.json\n"
        f"  frozen {frozen['case_variant_set_sha256']}\n  actual {got}\n"
        "Rows may only change status. Re-freeze deliberately if the queue really changed.\n"
        "(For a shard manifest pass --manifest <shard>; subset check applies automatically.)")


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


def command(row: dict[str, str], python_bin: str) -> list[str]:
    out_dir = C.repo_path(row["outdir_npz"]).parent
    cmd = [python_bin, "-u", "examples/run_mjwp.py", f"+override={row['override_id']}",
           f"task={row['target_task']}", "+use_torch_compile=false", "save_video=false",
           "video_camera=auto", f"seed={int(row['cem_seed'])}",
           f"num_samples={int(row['cem_samples'])}", f"max_num_iterations={int(row['cem_opt_steps'])}",
           f"output_dir={out_dir.as_posix()}", f"video_output_path={row['video']}"]
    return cmd + row.get("extra_overrides", "").split()


def launch(row: dict[str, str], gpu: str, python_bin: str):
    for path in (*out_paths(row).values(), C.repo_path(row["log"])):
        path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.pop("MUJOCO_GL", None)
    env.update({"CUDA_VISIBLE_DEVICES": gpu, "PYTHONUNBUFFERED": "1", "TORCHDYNAMO_DISABLE": "1"})
    stream = C.repo_path(row["log"]).open("w", encoding="utf-8")
    cmd = command(row, python_bin)
    stream.write(f"# {C.EXP_ID} {C.RUN_ID} {row['tier']} {row['arm_group']} "
                 f"case={row['case_id']} aug={row['aug_variant']} gpu={gpu} started_at={C.now()}\n"
                 f"# command={' '.join(cmd)}\n\n")
    stream.flush()
    proc = subprocess.Popen(cmd, cwd=C.REPO, env=env, stdout=stream, stderr=subprocess.STDOUT)
    return proc, stream


def kill_timed_out(row: dict[str, str], proc: subprocess.Popen) -> None:
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
    ap.add_argument("--cases", default="")
    ap.add_argument("--variants", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    manifest = C.repo_path(args.manifest)
    if not args.dry_run:
        # per-manifest lock: three machines on the shared volume each run a
        # different shard and must NOT share one lock (flock is per-file).
        lock = C.RESULTS / ".locks" / f"{manifest.stem}.lock"
        C.SingleInstance(lock).__enter__()  # released on process exit
    rows, fields = C.read_with_fields(manifest)
    # a shard manifest (anything other than the canonical priority manifest) is
    # validated as a subset of the frozen set, and gets its OWN lock so three
    # machines on the shared volume never contend on one lock or one TSV.
    is_shard = manifest.resolve() != C.PRIORITY_MANIFEST.resolve()
    set_sha = check_frozen(rows, allow_subset=is_shard)

    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    case_filter = {c for c in args.cases.split(",") if c}
    var_filter = {v for v in args.variants.split(",") if v}
    order = [i for i in sorted(range(len(rows)),
                               key=lambda i: (C.TIER_RANK[rows[i]["tier"]], int(rows[i]["ordinal"])))
             if (not case_filter or rows[i]["case_id"] in case_filter)
             and (not var_filter or rows[i]["aug_variant"] in var_filter)]

    # resume: mark already-complete rows
    for i in order:
        row = rows[i]
        if row["status"] in ELIGIBLE and all(p.is_file() for p in out_paths(row).values()):
            fails = output_failures(row)
            row["status"] = DONE_STATUS if not fails else "failed_validation"
            row["failure_mode"] = ";".join(fails)
            row["updated_at"] = C.now()
    C.write_tsv(manifest, rows, fields)

    eligible_order = [i for i in order if rows[i]["status"] in ELIGIBLE]
    if args.limit:
        eligible_order = eligible_order[:args.limit]
    allow = set(eligible_order)

    print(f"{C.EXP_ID} {C.RUN_ID}: {len(eligible_order)} pending of {len(order)} selected, gpus={gpus}, "
          f"frozen_set={'-' if set_sha is None else set_sha[:12]}, "
          f"timeout={args.per_task_timeout_min}min")

    if args.dry_run:
        for i in eligible_order[:12]:
            r = rows[i]
            print(f"  {r['tier']} {r['object_key']:9s} {r['aug_variant']:5s} {r['case_id']:32s}")
            print(f"     {' '.join(command(r, args.python_bin))}")
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
                                          "updated_at": C.now()})
                        C.write_tsv(manifest, rows, fields)
                        print(f"[timeout] {rows[idx]['case_id']} {rows[idx]['aug_variant']} gpu={g}",
                              file=sys.stderr, flush=True)
                        continue
                    still.append((idx, proc, stream, t0))
                    continue
                stream.close()
                finalize(rows[idx], rc)
                rows[idx]["wall_min"] = round((time.monotonic() - t0) / 60.0, 2)
                rows[idx]["updated_at"] = C.now()
                C.write_tsv(manifest, rows, fields)
                n_done = sum(1 for i in allow if rows[i]["status"] == DONE_STATUS)
                print(f"[done {n_done}/{len(allow)}] {rows[idx]['case_id']} {rows[idx]['aug_variant']} "
                      f"gpu={g} -> {rows[idx]['status']} wall={rows[idx]['wall_min']}min", flush=True)
            running[g] = still

        todo = [i for i in eligible_order if i not in dispatched and rows[i]["status"] in ELIGIBLE]
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
                                "updated_at": C.now()})
                    C.write_tsv(manifest, rows, fields)
                    dispatched.add(i)
                    print(f"[preflight-fail] {row['case_id']} {row['aug_variant']}: {row['failure_mode']}",
                          file=sys.stderr, flush=True)
                    placed = True
                    break
                proc, stream = launch(row, g, args.python_bin)
                row.update({"status": "running", "failure_mode": "", "gpu_id": g, "updated_at": C.now()})
                C.write_tsv(manifest, rows, fields)
                running[g].append((i, proc, stream, time.monotonic()))
                dispatched.add(i)
                made_progress = placed = True
                print(f"[start {len(dispatched)}/{len(allow)}] {row['tier']} {row['object_key']} "
                      f"{row['aug_variant']} {row['case_id']} gpu={g}", flush=True)
                break
            if not placed:
                break

        any_running = any(running[g] for g in gpus)
        remaining = [i for i in eligible_order if rows[i]["status"] in ELIGIBLE and i not in dispatched]
        if not any_running and not remaining and not made_progress:
            break
        time.sleep(args.poll_interval)

    check_frozen(rows)
    ok = [i for i in allow if rows[i]["status"] == DONE_STATUS]
    bad = [f"{rows[i]['case_id']}:{rows[i]['aug_variant']}={rows[i]['status']}"
           for i in allow if rows[i]["status"] != DONE_STATUS]
    walls = [float(rows[i]["wall_min"]) for i in ok if rows[i].get("wall_min")]
    print(f"\n[queue-complete] {len(ok)}/{len(allow)} {DONE_STATUS} in "
          f"{(time.monotonic() - t_start) / 3600:.2f} h")
    if walls:
        print(f"  wall_min: median {sorted(walls)[len(walls) // 2]:.1f}, max {max(walls):.1f}")
    if bad:
        print(f"  problem rows ({len(bad)}): {bad[:10]}")
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
