#!/usr/bin/env python3
"""E211: render the Stage A rollouts to mp4 (sim left / reference-FK right).

Only the two middle arms are rendered by default. The g-curve endpoints already
have mp4s and re-rendering them would spend ~45 CPU-minutes reproducing identical
files:

  g=0.0 -> results/E206/s6_downstream/render/full/E206_<case>_prg.mp4
  g=1.0 -> results/E209/s6_downstream/render/full/E209_<case>_G1.mp4

**Read this before comparing two mp4s.** `_auto_video_camera`
(spider/viewers/__init__.py:262-291) recomputes lookat and radius every frame
from the union bounding box of sim and ref bodies. Different sim -> different
camera. E209 F6 and E210 F3 both established that the SAME reference trajectory
renders as visibly different poses (crouched vs upright) across two arms' videos.
So: judging "which arm deviates more from the reference" is only valid WITHIN a
single video (sim vs ref share that video's camera). Across videos, absolute
pose, on-screen position and apparent size are all artefacts.

For cross-arm comparison use the viser replay instead
(`E211/viser_replay_arms.py`) -- it puts every arm in one scene on one shared
timeline under one camera, which is the comparison mp4 cannot give.

Rendering is CPU work and does not touch the GPUs; it reads each run's own
`config_act.yaml`. MUJOCO_GL must be osmesa: on this host egl raises EGLError and
glfw has no `_mjr_context` (measured in E207 P8), and the renderer defaults to
egl, so the export is load-bearing.

Usage:
    MUJOCO_GL=osmesa .venv/bin/python \
      workspace/core4d/scripts/experiments/E211/render_cem_results.py
    ... --arms G06,G08 --cases <id> --overwrite --max-frames 200
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[5]
for _d in ("E211", "E168"):
    sys.path.insert(0, str(REPO / "workspace/core4d/scripts/experiments" / _d))

import e211_common as C  # noqa: E402

#: Keys render_row needs off each manifest row.
NEEDED = ("variant", "outdir_npz", "config_act", "scene_act", "trajectory", "case_id")

#: The endpoints are not re-rendered; these are where their mp4s already live.
ENDPOINT_RENDERS = {
    "prg": "workspace/core4d/results/E206/s6_downstream/render/full/E206_{case}_prg.mp4",
    "g1": "workspace/core4d/results/E209/s6_downstream/render/full/E209_{case}_G1.mp4",
}


def manifest_rows(stage: str) -> list[dict[str, str]]:
    path = C.manifest_path(stage)
    if not path.is_file():
        raise SystemExit(f"missing manifest: {path} (run build_manifest.py)")
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    missing = [k for k in NEEDED if rows and k not in rows[0]]
    if missing:
        raise SystemExit(f"manifest lacks renderer columns {missing}: {path}")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="full", choices=["full", "smoke"])
    ap.add_argument("--arms", default="G06,G08", help="E211 arms to render")
    ap.add_argument("--cases", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    if os.environ.get("MUJOCO_GL") != "osmesa":
        raise SystemExit(
            "MUJOCO_GL must be 'osmesa' on this host (egl -> EGLError, glfw -> no "
            "_mjr_context; verified in E207 P8). Re-run with MUJOCO_GL=osmesa."
        )

    from render_a100_cem_videos import render_row  # noqa: E402

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    unknown = [a for a in arms if a not in C.ARMS]
    if unknown:
        raise SystemExit(f"unknown E211 arm(s) {unknown}; known: {sorted(C.ARMS)}")

    rows = manifest_rows(args.stage)
    # Only rows whose rollout actually finished -- lets this run alongside CEM.
    rows = [r for r in rows if (REPO / r["outdir_npz"]).is_file()]
    rows = [r for r in rows if r["arm"] in arms]
    if args.cases:
        want = {c.strip() for c in args.cases.split(",") if c.strip()}
        rows = [r for r in rows if r["case_id"] in want]
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit("no finished E211 rollouts match the filters")

    out_dir = args.out_dir or (C.S6_DIR / "render" / args.stage)
    out_dir.mkdir(parents=True, exist_ok=True)
    rendered, reused, failed = [], [], []
    for i, row in enumerate(rows, 1):
        mp4 = out_dir / f"{row['variant']}.mp4"
        if mp4.is_file() and not args.overwrite:
            reused.append(row["variant"])
            print(f"[{i:03d}/{len(rows)}] reuse    {row['variant']}", flush=True)
            continue
        try:
            render_row(row, out_path=mp4, max_frames=args.max_frames)
            rendered.append(row["variant"])
            print(f"[{i:03d}/{len(rows)}] rendered {row['variant']} -> "
                  f"{mp4.relative_to(REPO)}", flush=True)
        except Exception as exc:  # noqa: BLE001 - one bad case must not stop the batch
            failed.append({"variant": row["variant"], "error": f"{type(exc).__name__}: {exc}"})
            print(f"[{i:03d}/{len(rows)}] FAILED   {row['variant']}: {exc}",
                  file=sys.stderr, flush=True)

    # Record where the endpoints' mp4s are, and whether they are actually there,
    # so the four-way A/B set is either complete or visibly incomplete.
    endpoints: dict[str, dict[str, bool]] = {}
    for arm, tmpl in ENDPOINT_RENDERS.items():
        endpoints[arm] = {
            case: (REPO / tmpl.format(case=case)).is_file() for case in C.CASES
        }

    payload: dict[str, Any] = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "stage": args.stage,
        "arms": arms,
        "candidates": len(rows),
        "rendered": rendered,
        "reused": reused,
        "failed": failed,
        "endpoint_renders_present": endpoints,
        "camera_caveat": (
            "auto camera is recomputed per frame from the sim-union bbox; cross-video "
            "pose comparison is INVALID (E209 F6 / E210 F3). Compare sim vs ref within "
            "one video, or use E211/viser_replay_arms.py for a shared-camera A/B."
        ),
    }
    (out_dir / "e211_render_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nrendered={len(rendered)} reused={len(reused)} failed={len(failed)} -> "
          f"{out_dir.relative_to(REPO)}")
    for arm, present in endpoints.items():
        missing = [c for c, ok in present.items() if not ok]
        print(f"  endpoint {arm}: {sum(present.values())}/{len(present)} mp4 present"
              + (f"  MISSING {missing}" if missing else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
