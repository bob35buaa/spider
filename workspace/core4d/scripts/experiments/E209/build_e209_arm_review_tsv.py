#!/usr/bin/env python3
"""Build the review-player arm_sweep TSV for the E209 PRG-vs-G1 compare.

Turns the already-scored `e209_two_arm_rollout.tsv` into the review_index
CaseRecord schema: 2 rows per case (PRG / G1), each pointing at that arm's CEM
rollout + scene for live-qpos playback. **No re-scoring** -- metrics come from
the eval TSV; GATE_MAP / METRIC_COLUMNS / gate_pass are imported from E206's
builder (rule 13) so E206ARM and E209ARM are read on one scale.

Both arms carry an mp4: PRG reuses E206's existing renders, G1 is rendered by
E209/render_cem_results.py. That makes this the first arm-sweep review set where
the comparison is genuinely side-by-side video rather than live-qpos only --
which matters because the risk gravcomp introduces (an object that floats /
is "ghost-carried" because it is weightless) is a *visual* failure no gate sees.

Registered as exp "E209ARM" in review_index.py; launched via
    bash workspace/core4d/scripts/eval/wrappers/review_player.sh E209ARM

NOTE the filename: E206 already owns `build_arm_review_tsv.py`, and this module
puts the E209 dir on sys.path, so a same-named file here would shadow it and
import itself instead.

Usage:
    .venv/bin/python .../E209/build_e209_arm_review_tsv.py
"""

from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[5]
for _d in ("E201", "E206", "E209"):
    sys.path.insert(0, str(REPO / "workspace/core4d/scripts/experiments" / _d))

import build_arm_review_tsv as B206  # noqa: E402  (GATE_MAP / METRIC_COLUMNS / gate_pass)
import e209_common as C  # noqa: E402

EVAL_DIR = C.S6_DIR / "eval/two_arm"
ROLLOUT_TSV = EVAL_DIR / "e209_two_arm_rollout.tsv"
OUT_TSV = EVAL_DIR / "e209_arm_case_metrics.tsv"
E209_RENDER_DIR = C.S6_DIR / "render/full"
E206_RENDER_DIR = REPO / "workspace/core4d/results/E206/s6_downstream/render/full"

ARM_LABEL = {"prg": "PRG", "g1": "G1"}


def paths_for(arm: str, row: dict[str, str]) -> tuple[str, str, str, str, str]:
    """(outdir_npz, scene_xml, config_act, trajectory, video) for one arm/case."""
    case_id = row["case_id"]
    if arm == "g1":
        npz = C.result_npz(case_id)
        cfg = C.config_act(case_id)
        scene = C.scene_path(row)
        mp4 = E209_RENDER_DIR / f"{C.EXP}_{case_id}_{C.ARM_TAG}.mp4"
    else:
        npz = C.baseline_npz(case_id)
        cfg = C.baseline_out_dir(case_id) / "config_act.yaml"
        scene = C.base_scene_path(row)
        mp4 = E206_RENDER_DIR / f"E206_{case_id}_prg.mp4"
    traj = C.kinematic_npz(row)
    rp = lambda p: str(Path(p).relative_to(REPO))  # noqa: E731
    return (
        rp(npz) if npz.is_file() else "",
        rp(scene),
        rp(cfg) if cfg.is_file() else "",
        rp(traj),
        rp(mp4) if mp4.is_file() else "",
    )


def main() -> int:
    if not ROLLOUT_TSV.is_file():
        print(f"missing {ROLLOUT_TSV}; run eval_E209_g1_gravcomp.py first", file=sys.stderr)
        return 2

    by_case = {r["case_id"]: r for r in C.sources()}
    with ROLLOUT_TSV.open(encoding="utf-8") as fh:
        rollout = list(csv.DictReader(fh, delimiter="\t"))

    cols = (["arm", "case_id", "variant", "object_key", "retarget_variant_id",
             "numeric_release_pass", "numeric_failure_modes", "status",
             "outdir_npz", "scene_xml", "config_act", "trajectory", "video"]
            + list(B206.GATE_MAP) + list(B206.METRIC_COLUMNS))

    out_rows = []
    for r in rollout:
        arm, cid = r["arm"], r["case_id"]
        source = by_case[cid]
        npz, scene, cfg, traj, mp4 = paths_for(arm, source)
        task = C.target_task(source)
        rec = {
            "arm": ARM_LABEL[arm], "case_id": cid, "variant": "orig",
            "object_key": r["object_key"],
            "retarget_variant_id": "omnirt_v2" if "omnirt_v2" in task else "omnirt_v1",
            # headline flag = the funnel's narrow caliber, same as E206ARM
            "numeric_release_pass": r.get("narrow_pass", ""),
            "numeric_failure_modes": r.get("narrow_failed", ""),
            "status": f"{ARM_LABEL[arm]}_FULL_COMPLETE",
            "outdir_npz": npz, "scene_xml": scene, "config_act": cfg,
            "trajectory": traj, "video": mp4,
        }
        for gf, (field, op, thr) in B206.GATE_MAP.items():
            rec[gf] = B206.gate_pass(field, op, thr, r)
        for m in B206.METRIC_COLUMNS:
            rec[m] = r.get(m, "")
        out_rows.append(rec)

    OUT_TSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_TSV.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, delimiter="\t")
        w.writeheader()
        w.writerows(out_rows)

    playable = sum(1 for r in out_rows if r["outdir_npz"])
    with_video = sum(1 for r in out_rows if r["video"])
    print(f"[done] {OUT_TSV.relative_to(REPO)}  rows={len(out_rows)} "
          f"playable={playable} with_video={with_video}")
    print("  per-arm:", dict(Counter(r["arm"] for r in out_rows)))
    if with_video < len(out_rows):
        missing = [f"{r['arm']}/{r['case_id']}" for r in out_rows if not r["video"]]
        print(f"  [warn] {len(missing)} rows have no mp4 (A/B review degraded): {missing[:6]}")
    return 0 if playable == len(out_rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
