#!/usr/bin/env python3
"""E208 P7: render MP4s for whatever aug CEM runs have finished so far.

Designed to be run repeatedly while the 105-run queue is still going: it takes
the finished rows from the frozen manifest, skips anything already rendered, and
**never writes the manifest** -- the CEM runner owns that file and rewrites it on
every status change.

Unlike E206's version this reads the manifest rather than reconstructing rows
from output dirs: the manifest already carries every path (rollout, config_act,
scene, trajectory) and is the frozen authority, so reconstructing them would be a
second, unaudited derivation of the same thing.

The reference pane is the **aug** trajectory, not orig -- that is what the run
was asked to track, and comparing a rot variant against the orig reference would
render as a 45 deg error that is not an error.

MUST run with ``MUJOCO_GL=osmesa``: this host has no NVIDIA EGL, and all 8 GPUs
are busy with the CEM queue -- rendering is CPU work and has no business
competing for them.

Usage:
    MUJOCO_GL=osmesa .venv/bin/python .../E208/render_aug_results.py
    ... --variants trans0,rot0 --objects desk007 --limit 6 --max-frames 200
    ... --watch 300     # re-scan every 5 min until the queue drains
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import e208_common as C  # noqa: E402

sys.path.insert(0, str(C.REPO / "workspace/core4d/scripts/experiments/E168"))

DONE_STATUS = "cem_ok"


def finished_rows(args: argparse.Namespace) -> list[dict[str, str]]:
    rows = [r for r in C.read_tsv(C.PRIORITY_MANIFEST) if r.get("status") == DONE_STATUS]
    if args.objects:
        want = {o.strip() for o in args.objects.split(",") if o.strip()}
        rows = [r for r in rows if r["object_key"] in want]
    if args.variants:
        want = {v.strip() for v in args.variants.split(",") if v.strip()}
        rows = [r for r in rows if r["aug_variant"] in want]
    if args.cases:
        want = {c.strip() for c in args.cases.split(",") if c.strip()}
        rows = [r for r in rows if r["case_id"] in want]
    out = []
    for r in rows:
        if not C.repo_path(r["outdir_npz"]).is_file():
            continue
        out.append(r)
    out.sort(key=lambda r: int(r["ordinal"]))
    return out[: args.limit] if args.limit else out


def render_pass(args: argparse.Namespace, render_row) -> dict[str, Any]:
    rows = finished_rows(args)
    rendered, reused, failed = [], [], []
    for i, row in enumerate(rows, 1):
        mp4 = C.repo_path(row["video"])
        tag = row["variant"]
        if mp4.is_file() and not args.overwrite:
            reused.append(tag)
            continue
        mp4.parent.mkdir(parents=True, exist_ok=True)
        try:
            render_row(row, out_path=mp4, max_frames=args.max_frames)
            rendered.append(tag)
            print(f"[{i:03d}/{len(rows)}] rendered {tag} -> {C.rel(mp4)}", flush=True)
        except Exception as exc:  # noqa: BLE001 - one bad case must not stop the batch
            failed.append({"variant": tag, "error": f"{type(exc).__name__}: {exc}"})
            print(f"[{i:03d}/{len(rows)}] FAILED  {tag}: {exc}", file=sys.stderr, flush=True)
    return {"candidates": len(rows), "rendered": rendered, "reused": reused, "failed": failed}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="full", choices=["full", "smoke"])
    ap.add_argument("--variants", default="", help="comma-separated aug variants")
    ap.add_argument("--objects", default="", help="comma-separated object_keys")
    ap.add_argument("--cases", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--watch", type=float, default=0.0,
                    help="seconds between re-scans; 0 = single pass")
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    if os.environ.get("MUJOCO_GL") != "osmesa":
        print("[warn] MUJOCO_GL != osmesa; this host has no EGL and the GPUs are busy "
              "with the CEM queue. Re-run with MUJOCO_GL=osmesa if rendering fails.",
              file=sys.stderr, flush=True)

    from render_a100_cem_videos import render_row  # noqa: E402

    out_dir = args.out_dir or (C.S6_DIR / "render" / args.stage)
    out_dir.mkdir(parents=True, exist_ok=True)
    total = {"rendered": [], "reused": [], "failed": []}

    while True:
        res = render_pass(args, render_row)
        for key in ("rendered", "reused", "failed"):
            total[key].extend(res[key])
        # `candidates` is post-filter/post-limit, so it cannot stand in for queue
        # progress -- with --limit it would stall the watch loop forever.
        manifest = C.read_tsv(C.PRIORITY_MANIFEST)
        expected = len(manifest)
        finished = sum(1 for r in manifest if r.get("status") == DONE_STATUS)
        print(f"[pass] queue {finished}/{expected} finished; this pass considered "
              f"{res['candidates']}: rendered={len(res['rendered'])} "
              f"reused={len(res['reused'])} failed={len(res['failed'])}", flush=True)
        if not args.watch or (finished >= expected and not res["rendered"]):
            break
        time.sleep(args.watch)

    payload: dict[str, Any] = {
        "experiment": C.EXP_ID, "run_id": C.RUN_ID, "stage": args.stage,
        "generated_at": C.now(),
        "reference_pane": "aug trajectory (not orig) -- it is what the run tracked",
        "rendered": total["rendered"], "reused": sorted(set(total["reused"])),
        "failed": total["failed"],
    }
    (out_dir / "render_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nrendered={len(total['rendered'])} reused={len(set(total['reused']))} "
          f"failed={len(total['failed'])}  -> {C.rel(out_dir)}")
    return 1 if total["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
