#!/usr/bin/env python3
"""E214c: render a focused representative set (full baseline vs A5/A6) to mp4 for the
mandatory visual evaluation (sim left / reference-FK right).

Reuses E168's render_row (replays a rollout npz in the scene from its config_act).
Ablation rows use their own E214c config_act (local model_path).  The "full" baseline
reuses the SAME case's A5 config_act (which carries the correct local scene +
reference; the gate thresholds in it are irrelevant to playback) but swaps in the
baseline cem_npz -- so full and ablations render in an identical scene/reference for
a valid A/B.

Usage (headless):
    MUJOCO_GL=egl .venv/bin/python .../E214c/render_E214c.py \
      --cases box021_20231011_034_p1,box001_20231020_014_p1 \
      --series full,A5_uniform_hand,A6_uniform_leg
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "E168"))
import e214c_common as C  # noqa: E402
from render_a100_cem_videos import render_row  # noqa: E402

OUT_DIR = C.RESULTS / "render"
DONOR_ABLATION = "A5_uniform_hand"  # any E214c arm; supplies local scene+ref for "full"


def rows_for(cases: list[str], series: list[str]) -> list[dict]:
    man, _ = C.read_with_fields(C.MANIFEST)
    by = {(r["case_id"], r["ablation"]): r for r in man}
    out: list[dict] = []
    for case in cases:
        donor = by.get((case, DONOR_ABLATION))
        if donor is None:
            continue
        scene_act = donor["run_model_path"]
        trajectory = donor["run_data_path"]
        for s in series:
            if s == "full":
                bp = C.baseline_paths(case)
                if not bp.get("cem_npz"):
                    continue
                out.append({"variant": f"E214c_{case}__full",
                            "outdir_npz": C.rel(bp["cem_npz"]),
                            "config_act": donor["config_act"],
                            "scene_act": scene_act, "trajectory": trajectory})
            else:
                r = by.get((case, s))
                if r and C.repo_path(r["outdir_npz"]).is_file():
                    out.append({"variant": f"E214c_{case}__{s}",
                                "outdir_npz": r["outdir_npz"], "config_act": r["config_act"],
                                "scene_act": r["run_model_path"], "trajectory": r["run_data_path"]})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True)
    ap.add_argument("--series", default="full,A5_uniform_hand,A6_uniform_leg")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    cases = [c for c in args.cases.split(",") if c]
    series = [s for s in args.series.split(",") if s]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = rows_for(cases, series)
    print(f"rendering {len(rows)} videos -> {C.rel(OUT_DIR)}")
    ok = 0
    for r in rows:
        mp4 = OUT_DIR / f"{r['variant']}.mp4"
        if mp4.is_file() and not args.overwrite:
            print(f"  skip {r['variant']} (exists)"); ok += 1; continue
        try:
            render_row(r, out_path=mp4, max_frames=args.max_frames)
            ok += 1
            print(f"  ok   {r['variant']}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"  FAIL {r['variant']}: {exc}", flush=True)
    print(f"done {ok}/{len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
