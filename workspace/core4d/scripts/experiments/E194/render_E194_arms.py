#!/usr/bin/env python3
"""E194: render self MP4s for completed G1/G2/G3 gravcomp rollouts.

Reuses the generic E168 render_row (reads outdir_npz + config_act per row) over
the E194 cem_full_manifest completed rows. For G1/G3 the landed config_act points
at the gravcomp sidecar, so the object is rendered against the exact model it was
optimized on. Run with MUJOCO_GL=osmesa (this host has no nvidia EGL).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e194_common as C  # noqa: E402
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
    parser.add_argument("--variants", nargs="*", default=[])
    args = parser.parse_args()

    rows = C.read_tsv(args.manifest)
    out_dir = C.repo_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rendered, skipped, failed = [], [], []
    for row in rows:
        if args.variants and row["variant"] not in args.variants:
            continue
        # accept a row whose rollout npz exists even if status wasn't merged
        if row.get("status") not in COMPLETE_STATUSES and not C.repo_path(row["outdir_npz"]).is_file():
            skipped.append(row["variant"]); continue
        output = out_dir / f"{row['variant']}_full.mp4"
        if output.is_file() and not args.overwrite:
            rendered.append(row["variant"]); continue
        try:
            render_row(row, out_path=output, max_frames=args.max_frames)
            rendered.append(row["variant"])
            print(f"[rendered] {row['variant']} -> {C.rel(output)}", flush=True)
        except Exception as exc:  # noqa: BLE001
            failed.append({"variant": row["variant"], "error": f"{type(exc).__name__}: {exc}"})
            print(f"[failed] {row['variant']}: {exc}", file=sys.stderr, flush=True)
    C.write_json(out_dir / "render_summary.json",
                 {"created_at": C.now(), "rendered": rendered, "skipped": skipped, "failed": failed})
    print(f"rendered={len(rendered)} skipped={len(skipped)} failed={len(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
