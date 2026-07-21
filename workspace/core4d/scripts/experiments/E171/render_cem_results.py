#!/usr/bin/env python3
"""E171: render Full CEM result MP4s for the mandatory visual review (plan 8.3).

Reuses the generic E168 render_row (reads outdir_npz + config_act per manifest
row) over the E171 cem_full_manifest completed rows. Keyframe sheets for the
review are extracted from these MP4s with the video-frames skill.
Run with MUJOCO_GL=osmesa (this host has no nvidia EGL).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e171_common as C
sys.path.insert(0, str(C.REPO / "workspace/core4d/scripts/experiments/E168"))
from render_a100_cem_videos import render_row  # noqa: E402

COMPLETE_STATUSES = {"run_complete_pending_eval", "run_complete", "eval_complete"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path,
                        default=C.RESULTS / "s6_downstream/manifests/cem_full_manifest.tsv")
    parser.add_argument("--out-dir", type=Path, default=C.RESULTS / "s6_downstream/render/full")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--cases", nargs="*", default=[])
    args = parser.parse_args()

    rows = C.read_tsv(args.manifest)
    out_dir = C.repo_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rendered, skipped, failed = [], [], []
    for row in rows:
        if args.cases and row["case_id"] not in args.cases:
            continue
        if row.get("status") not in COMPLETE_STATUSES:
            skipped.append(row["case_id"]); continue
        output = out_dir / f"{row['variant']}_full.mp4"
        if output.is_file() and not args.overwrite:
            rendered.append(row["case_id"]); continue
        try:
            render_row(row, out_path=output, max_frames=args.max_frames)
            rendered.append(row["case_id"])
            print(f"[rendered] {row['case_id']} -> {C.rel(output)}", flush=True)
        except Exception as exc:
            failed.append({"case_id": row["case_id"], "error": f"{type(exc).__name__}: {exc}"})
            print(f"[failed] {row['case_id']}: {exc}", file=sys.stderr, flush=True)
    C.write_json(out_dir / "render_summary.json",
                 {"created_at": C.now(), "rendered": rendered, "skipped": skipped, "failed": failed})
    print(f"rendered={len(rendered)} skipped={len(skipped)} failed={len(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
