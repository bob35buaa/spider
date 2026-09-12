#!/usr/bin/env python3
"""E215 C6: render sim-vs-aug-ref MP4s for the visual gate (osmesa, CPU-only).

Reuses E168's render_row (sim pane vs the AUG reference trajectory, same video,
same camera -- the only comparison that is valid per E210 F3).  Reads the 3 shard
manifests for the finished rows.  MUST run with MUJOCO_GL=osmesa: rendering is CPU
work and the GPUs are held by the keepalive script.

Default selection covers each arm group's clean case + the 3 new-fall cases +
the worst degenerate-yaw case, so the visual gate sees both healthy and suspect
behaviour.

Usage:
    MUJOCO_GL=osmesa .venv/bin/python .../E215/render_c6.py
    ... --cases a,b --variants rot0 --max-frames 200
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "E168"))

import e215_common as C  # noqa: E402
from render_a100_cem_videos import render_row  # noqa: E402

# (case_id, variant, tag) -- tag documents why the case is in the gate.
DEFAULT_SELECT = [
    ("box021_20231011_034_p1", "rot0", "box_prg clean"),
    ("box001_20231003_1_040_p2", "rot0", "box_prg_g1a2 clean(v2 orig)"),
    ("box004_20231003_2_083_p2", "rot0", "box_prg_g1a2 NEW-FALL"),
    ("box004_20231003_2_083_p2", "rot1", "box_prg_g1a2 NEW-FALL"),
    ("box023_20231008_045_p1", "rot0", "box_noprg clean"),
    ("box023_20231011_021_p2", "rot1", "box_noprg NEW-FALL"),
    ("bucket003_20231018_001_p1", "rot0", "bucket_prg clean"),
    ("bucket003_20231020_068_p1", "rot0", "bucket_prg DEGENERATE-yaw 6.9deg"),
    ("bucket007_20231023_073_p1", "rot0", "bucket_prg_gravcomp clean"),
]


def all_done_rows() -> dict[tuple[str, str], dict[str, str]]:
    import string
    out: dict[tuple[str, str], dict[str, str]] = {}
    done = {"run_complete_pending_eval", "cem_ok"}
    for s in string.ascii_uppercase[:3]:
        f = C.MANIFEST_DIR / f"e215_priority_manifest.shard{s}.tsv"
        if not f.is_file():
            continue
        for r in C.read_tsv(f):
            if r.get("status") in done:
                out[(r["case_id"], r["aug_variant"])] = r
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="")
    ap.add_argument("--variants", default="")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=C.S6_DIR / "render/c6")
    args = ap.parse_args()

    if os.environ.get("MUJOCO_GL") != "osmesa":
        print("[warn] MUJOCO_GL != osmesa; GPUs are held by the keepalive script. "
              "Re-run with MUJOCO_GL=osmesa.", file=sys.stderr)

    rows_by_key = all_done_rows()
    select = DEFAULT_SELECT
    if args.cases or args.variants:
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
            print(f"[{i:02d}] {tag:32s} {cid}/{variant} -> {C.rel(mp4)}", flush=True)
        except Exception as exc:  # noqa: BLE001
            failed.append({"key": f"{cid}/{variant}", "error": f"{type(exc).__name__}: {exc}"})
            print(f"[{i:02d}] FAILED {cid}/{variant}: {exc}", file=sys.stderr, flush=True)

    print(f"\nrendered {len(rendered)}/{len(select)} -> {C.rel(out_dir)}")
    if missing:
        print(f"  missing (no done row): {missing}")
    if failed:
        for f in failed:
            print(f"  FAILED {f['key']}: {f['error']}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
