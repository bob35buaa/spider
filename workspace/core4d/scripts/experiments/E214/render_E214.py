#!/usr/bin/env python3
"""E214: render a focused representative set (full baseline vs ablations) to mp4
for the mandatory visual evaluation (sim left / reference-FK right).

Reuses E168's render_row (replays a rollout npz in the scene from its config_act).
Ablation rows use their own E214 config_act (local model_path). The "full"
baseline reuses the SAME case's A1 config_act (which carries the correct local
scene + reference) but swaps in the baseline cem_npz -- so full and ablations
render in an identical scene/reference for a valid A/B.

Usage (headless):
    MUJOCO_GL=egl .venv/bin/python .../E214/render_E214.py \
      --cases box021_20231011_034_p1,box001_20231020_014_p1 \
      --series full,A2_surfaceBand_only,A3_softPenalty_only,A4_hardGate_only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "E168"))
import e214_common as C  # noqa: E402
from render_a100_cem_videos import render_row  # noqa: E402

OUT_DIR = C.RESULTS / "render"


def rows_for(cases: list[str], series: list[str]) -> list[dict]:
    man, _ = C.read_with_fields(C.MANIFEST)
    by = {(r["case_id"], r["ablation"]): r for r in man}
    out: list[dict] = []
    for case in cases:
        a1 = by.get((case, "A1_contactHDMI_only"))
        if a1 is None:
            continue
        for s in series:
            # scene_act + trajectory (local-resolved) shared by full and ablations of this case
            scene_act = a1["run_model_path"]
            trajectory = a1["run_data_path"]
            if s == "full":
                bp = C.baseline_paths(case)
                if not bp.get("cem_npz"):
                    continue
                # reuse A1 config_act (local paths/scene) but the baseline rollout npz
                out.append({"variant": f"E214_{case}__full",
                            "outdir_npz": C.rel(bp["cem_npz"]),
                            "config_act": a1["config_act"],
                            "scene_act": scene_act, "trajectory": trajectory})
            else:
                r = by.get((case, s))
                if r and C.repo_path(r["outdir_npz"]).is_file():
                    out.append({"variant": f"E214_{case}__{s}",
                                "outdir_npz": r["outdir_npz"], "config_act": r["config_act"],
                                "scene_act": r["run_model_path"], "trajectory": r["run_data_path"]})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True)
    ap.add_argument("--series", default="full,A2_surfaceBand_only,A3_softPenalty_only,A4_hardGate_only")
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
