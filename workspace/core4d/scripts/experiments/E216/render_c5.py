#!/usr/bin/env python3
"""E216 C5: render sim-vs-aug-ref MP4s for the visual gate (osmesa, CPU-only).

Reuses E168's render_row (sim pane vs the 0.1m AUG reference trajectory, same
video, same camera -- the only valid comparison per E210 F3).  Reads the E216
priority manifest for finished rows.  MUST run with MUJOCO_GL=osmesa: rendering
is CPU work and the GPUs are held by the CEM queue / keepalive.

Default selection covers each arm group's representative case so the visual gate
sees healthy behaviour across arms; --all renders every finished row.

Usage:
    MUJOCO_GL=osmesa .venv/bin/python .../E216/render_c5.py
    ... --cases a,b --variants trans0 --max-frames 200
    MUJOCO_GL=osmesa .venv/bin/python .../E216/render_c5.py --all
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "E168"))

import e216_common as C  # noqa: E402
from render_a100_cem_videos import render_row  # noqa: E402

# one representative (case, variant) per arm group for the visual gate sample.
DEFAULT_SELECT = [
    ("box021_20231011_034_p1", "trans0", "box_prg"),
    ("box023_20231008_046_p1", "trans0", "box_noprg"),
    ("box001_20231003_1_040_p2", "trans0", "box_prg_g1a2"),
    ("box004_20231003_2_082_p1", "trans0", "box_prg_g1a2"),
    ("box024_20231011_026_p1", "trans0", "box_prg_g1a2"),
    ("bucket003_20231018_001_p1", "trans0", "bucket_prg"),
    ("bucket007_20231003_2_021_p1", "trans0", "bucket_prg"),
]
DONE = {"run_complete_pending_eval", "cem_ok"}


def done_rows() -> dict[tuple[str, str], dict[str, str]]:
    if not C.PRIORITY_MANIFEST.is_file():
        return {}
    return {(r["case_id"], r["aug_variant"]): r
            for r in C.read_tsv(C.PRIORITY_MANIFEST) if r.get("status") in DONE}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="")
    ap.add_argument("--variants", default="")
    ap.add_argument("--all", action="store_true", help="render EVERY finished row, not the gate sample")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=C.S6_DIR / "render/c5")
    args = ap.parse_args()

    if os.environ.get("MUJOCO_GL") != "osmesa":
        print("[warn] MUJOCO_GL != osmesa; GPUs held by CEM/keepalive. Re-run with MUJOCO_GL=osmesa.",
              file=sys.stderr)

    rows_by_key = done_rows()
    select = DEFAULT_SELECT
    if args.all:
        select = [(cid, v, "all") for (cid, v) in sorted(rows_by_key)]
    elif args.cases or args.variants:
        cf = {c for c in args.cases.split(",") if c}
        vf = {v for v in args.variants.split(",") if v}
        select = [(cid, v, "") for (cid, v) in rows_by_key
                  if (not cf or cid in cf) and (not vf or v in vf)]

    out_dir = C.repo_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rendered, failed, missing = [], [], []
    for i, (cid, variant, tag) in enumerate(select, 1):
        row = rows_by_key.get((cid, variant))
        if row is None:
            missing.append(f"{cid}/{variant}")
            continue
        mp4 = out_dir / f"{C.aug_variant_id(cid, variant)}_{variant}.mp4"
        if mp4.is_file() and not args.overwrite:
            print(f"[{i:02d}] reuse {mp4.name}", flush=True)
            rendered.append(str(mp4))
            continue
        try:
            render_row(row, out_path=mp4, max_frames=args.max_frames)
            rendered.append(str(mp4))
            print(f"[{i:02d}] {tag:14s} {cid}/{variant} -> {C.rel(mp4)}", flush=True)
        except Exception as exc:  # noqa: BLE001
            failed.append({"key": f"{cid}/{variant}", "error": f"{type(exc).__name__}: {exc}"})
            print(f"[{i:02d}] FAILED {cid}/{variant}: {exc}", file=sys.stderr, flush=True)

    print(f"\nrendered {len(rendered)}/{len(select)} -> {C.rel(out_dir)}")
    if missing:
        print(f"  missing (no done row): {missing}")
    for f in failed:
        print(f"  FAILED {f['key']}: {f['error']}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
