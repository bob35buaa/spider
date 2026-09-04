#!/usr/bin/env python3
"""E206: render MP4s for whatever CEM runs have finished so far.

The queue is long (130 runs, ~11 h), so this is written to be run repeatedly
while it is still going: it scans the per-arm output dirs for a completed
`trajectory_mjwp_act.npz`, skips anything already rendered, and never touches
the manifest. No manifest is needed at all -- the row `render_row` wants is
reconstructed from the output dir plus the run's own `config_act.yaml`.

MUST run with MUJOCO_GL=osmesa: this host has no NVIDIA EGL, and more
importantly all 8 GPUs are busy with the CEM queue -- rendering is CPU work and
has no business competing for them.

Renders sim (left) vs ref-FK reference (right) via the shared E168 `render_row`.

Usage:
    MUJOCO_GL=osmesa .venv/bin/python .../render_cem_results.py
    ... --arms prg --objects desk007 --limit 6 --max-frames 200
    ... --pairs-only        # only cases where BOTH arms finished (A/B comparison)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import e206_common as C  # noqa: E402

sys.path.insert(0, str(C.REPO / "workspace/core4d/scripts/experiments/E168"))

ROLLOUT = "trajectory_mjwp_act.npz"


def target_task_by_case() -> dict[str, str]:
    out: dict[str, str] = {}
    for name in ("omnirt_v1", "omnirt_v2"):
        m = C.S3_DIR / f"{name}/ref_fk/stage2b_manifest_{name}_ref_fk.tsv"
        if not m.is_file():
            continue
        for row in C.read_tsv(m):
            if row.get("stage2b_status") == "pass":
                out[row["case_id"]] = row.get("target_task") or ""
    return out


def finished_runs(stage: str) -> list[dict[str, str]]:
    """One row per finished (arm, case) run, in the shape render_row expects."""
    tasks = target_task_by_case()
    rows: list[dict[str, str]] = []
    for case_id, task in sorted(tasks.items()):
        for arm in C.ARMS:
            out_dir = C.arm_out_dir(arm, case_id, stage)
            npz = out_dir / ROLLOUT
            cfg = out_dir / "config_act.yaml"
            if not (npz.is_file() and cfg.is_file()):
                continue
            rows.append({
                "case_id": case_id,
                "arm": arm,
                "object_key": case_id.split("_")[0],
                "variant": f"E206_{case_id}_{arm}",
                "outdir_npz": str(npz.relative_to(C.REPO)),
                "config_act": str(cfg.relative_to(C.REPO)),
                "scene_act": str(
                    (C.PROCESSED_ROOT / task / f"{C.SCENE_BY_ARM[arm]}.xml")
                    .relative_to(C.REPO)),
                "trajectory": str(
                    (C.PROCESSED_ROOT / task / "0/trajectory_kinematic.npz")
                    .relative_to(C.REPO)),
            })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="full", choices=["full", "smoke"])
    ap.add_argument("--arms", default=",".join(C.ARMS))
    ap.add_argument("--objects", default="", help="comma-separated object_keys")
    ap.add_argument("--cases", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--pairs-only", action="store_true",
                    help="only cases where BOTH arms finished (A/B comparison)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    if os.environ.get("MUJOCO_GL") != "osmesa":
        print("[warn] MUJOCO_GL != osmesa; this host has no EGL and the GPUs are "
              "busy with CEM. Re-run with MUJOCO_GL=osmesa if rendering fails.",
              file=sys.stderr, flush=True)

    from render_a100_cem_videos import render_row  # noqa: E402

    rows = finished_runs(args.stage)
    if args.pairs_only:
        per_case: dict[str, int] = {}
        for r in rows:
            per_case[r["case_id"]] = per_case.get(r["case_id"], 0) + 1
        rows = [r for r in rows if per_case[r["case_id"]] == len(C.ARMS)]
    arms = {a.strip() for a in args.arms.split(",") if a.strip()}
    rows = [r for r in rows if r["arm"] in arms]
    if args.objects:
        want = {o.strip() for o in args.objects.split(",") if o.strip()}
        rows = [r for r in rows if r["object_key"] in want]
    if args.cases:
        want = {c.strip() for c in args.cases.split(",") if c.strip()}
        rows = [r for r in rows if r["case_id"] in want]
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit("no finished CEM runs match the filters")

    out_dir = args.out_dir or (C.S6_DIR / "render" / args.stage)
    out_dir.mkdir(parents=True, exist_ok=True)
    rendered, reused, failed = [], [], []
    for i, row in enumerate(rows, 1):
        mp4 = out_dir / f"{row['variant']}.mp4"
        if mp4.is_file() and not args.overwrite:
            reused.append(row["variant"])
            print(f"[{i:03d}/{len(rows)}] reuse   {row['variant']}", flush=True)
            continue
        try:
            render_row(row, out_path=mp4, max_frames=args.max_frames)
            rendered.append(row["variant"])
            print(f"[{i:03d}/{len(rows)}] rendered {row['variant']} -> "
                  f"{mp4.relative_to(C.REPO)}", flush=True)
        except Exception as exc:  # noqa: BLE001 - one bad case must not stop the batch
            failed.append({"variant": row["variant"],
                           "error": f"{type(exc).__name__}: {exc}"})
            print(f"[{i:03d}/{len(rows)}] FAILED  {row['variant']}: {exc}",
                  file=sys.stderr, flush=True)

    payload: dict[str, Any] = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "stage": args.stage,
        "candidates": len(rows),
        "rendered": rendered, "reused": reused, "failed": failed,
    }
    (out_dir / "render_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nrendered={len(rendered)} reused={len(reused)} failed={len(failed)}"
          f"  -> {out_dir.relative_to(C.REPO)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
