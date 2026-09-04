#!/usr/bin/env python3
"""E209 P7: render the 22 G1 rollouts to mp4 (sim left / reference-FK right).

Only the G1 arm is rendered: E206 already produced all 22 PRG mp4s under
``results/E206/s6_downstream/render/full/E206_<case>_prg.mp4``, and re-rendering
them would cost ~45 CPU-minutes to reproduce identical files. The review player
pairs the two directories, which is what makes E209's manual pass a true A/B --
E207 could only offer live-qpos for its comparison arms.

Rendering is CPU work and does not touch the GPUs, so it can run while CEM is
still going; it reads each run's own ``config_act.yaml``.

MUJOCO_GL must be osmesa. Measured on this host during E207 P8: ``egl`` raises
EGLError and ``glfw`` has no ``_mjr_context``. The renderer itself defaults to
egl, so the export below is load-bearing, not decoration.

Usage:
    MUJOCO_GL=osmesa .venv/bin/python \
      workspace/core4d/scripts/experiments/E209/render_cem_results.py
    ... --stage smoke --cases <id> --overwrite --max-frames 200
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
for _d in ("E209", "E168"):
    sys.path.insert(0, str(REPO / "workspace/core4d/scripts/experiments" / _d))

import e209_common as C  # noqa: E402

#: Keys render_row needs off each manifest row.
NEEDED = ("variant", "outdir_npz", "config_act", "scene_act", "trajectory", "case_id")


def manifest_rows(stage: str) -> list[dict[str, str]]:
    path = C.MANIFEST if stage == "full" else C.MANIFEST.with_name("e209_g1_smoke_manifest.tsv")
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
    ap.add_argument("--cases", default="")
    ap.add_argument("--objects", default="")
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

    rows = manifest_rows(args.stage)
    # Only rows whose rollout actually finished -- lets this run alongside CEM.
    rows = [r for r in rows if (REPO / r["outdir_npz"]).is_file()]
    if args.objects:
        want = {o.strip() for o in args.objects.split(",") if o.strip()}
        rows = [r for r in rows if r["object_key"] in want]
    if args.cases:
        want = {c.strip() for c in args.cases.split(",") if c.strip()}
        rows = [r for r in rows if r["case_id"] in want]
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit("no finished E209 rollouts match the filters")

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

    payload: dict[str, Any] = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "stage": args.stage,
        "candidates": len(rows),
        "rendered": rendered,
        "reused": reused,
        "failed": failed,
        "note": "G1 arm only; the 22 PRG mp4s are reused from E206's render dir",
    }
    (out_dir / "e209_render_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nrendered={len(rendered)} reused={len(reused)} failed={len(failed)} -> "
          f"{out_dir.relative_to(REPO)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
