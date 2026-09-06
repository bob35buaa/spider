#!/usr/bin/env python3
"""Build the review-player arm_sweep TSV for the E207 four-arm compare.

Turns the already-scored four_arm_rollout.tsv into the review_index.CaseRecord
schema: 4 rows per case (arm in {PRG, noPRG, G1A2, G1only}), each pointing at that
arm's CEM rollout + scene for live-qpos playback. NO re-scoring -- metrics come
from the rollout TSV; the gate map, metric columns and gate_pass helper are
imported from E204_E205/build_arm_review_tsv.py (rule 13), so all four arms are
flagged by the same thresholds as the existing three-arm set.

Registered as exp "E207ARM" in review_index.py; launched via
    bash workspace/core4d/scripts/eval/wrappers/review_player.sh E207ARM

Usage:
    .venv/bin/python workspace/core4d/scripts/experiments/E207/build_e207_arm_review_tsv.py

NOTE the filename: E204_E205 already owns `build_arm_review_tsv.py`, and this
module puts the E207 dir on sys.path, so a same-named file here would shadow it
and import itself instead.
"""

from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[5]
for _d in ("E201", "E204_E205", "E207"):
    sys.path.insert(0, str(REPO / "workspace/core4d/scripts/experiments" / _d))

import build_arm_review_tsv as B205  # noqa: E402  (GATE_MAP / METRIC_COLUMNS / gate_pass)
import e204e205_common as C205  # noqa: E402
import e207_common as C207  # noqa: E402

EVAL_DIR = C207.RESULTS / "s6_downstream/eval/four_arm"
ROLLOUT_TSV = EVAL_DIR / "four_arm_rollout.tsv"
OUT_TSV = EVAL_DIR / "e207_arm_case_metrics.tsv"


def paths_for(arm: str, case_id: str, source: dict) -> tuple[str, str, str, str, str]:
    """(outdir_npz, scene_xml, config_act, trajectory, video) for one arm/case."""
    if arm == "G1only":
        npz = C207.result_npz(case_id, "full")
        scene = C207.scene_path(case_id)
        video = C207.RESULTS / f"s6_downstream/render/full/E207_{case_id}_G1only.mp4"
        _t, trajectory, _m = C205.base.local_authorities(source)
        rel = (lambda p: str(p.relative_to(REPO)))
        cfg = C207.config_act(case_id, "full")
        return (rel(npz) if npz.is_file() else "", rel(scene),
                rel(cfg) if cfg.is_file() else "", rel(trajectory),
                rel(video) if video.is_file() else "")
    npz, scene, cfg, trajectory = B205.paths_for(arm, case_id, source)
    # E178 rendered all 27 of its bucket cases, so the PRG rows can carry an mp4;
    # E204/E205 ran save_video=false and have none, so those stay live-qpos only.
    video = ""
    if arm == "PRG":
        hits = sorted((REPO / "workspace/core4d/results/E178/s6_downstream/render/full")
                      .glob(f"*{case_id}*.mp4"))
        if hits:
            video = str(hits[0].relative_to(REPO))
    return npz, scene, cfg, trajectory, video


def main() -> int:
    if not ROLLOUT_TSV.is_file():
        print(f"missing {ROLLOUT_TSV}; run eval_E207_g1only.py first", file=sys.stderr)
        return 2
    src_by_case = {s["case_id"]: s for s in C207.sources()}
    with ROLLOUT_TSV.open(encoding="utf-8") as stream:
        rollout = list(csv.DictReader(stream, delimiter="\t"))

    cols = (["arm", "case_id", "variant", "object_key", "retarget_variant_id",
             "numeric_release_pass", "numeric_failure_modes", "status",
             "outdir_npz", "scene_xml", "config_act", "trajectory", "video"]
            + list(B205.GATE_MAP) + list(B205.METRIC_COLUMNS))

    out_rows = []
    for row in rollout:
        arm, case_id = row["arm"], row["case_id"]
        source = src_by_case.get(case_id)
        if source is None:
            continue
        outdir_npz, scene_xml, config_act, trajectory, video = paths_for(arm, case_id, source)
        task = C207.task_of(case_id)
        record = {
            "arm": arm, "case_id": case_id, "variant": "orig",
            "object_key": row["object_key"],
            "retarget_variant_id": "omnirt_v2" if "omnirt_v2" in task else "omnirt_v1",
            "numeric_release_pass": row.get("narrow_pass", ""),
            "numeric_failure_modes": row.get("narrow_failed", ""),
            "status": f"{arm}_FULL_COMPLETE",
            "outdir_npz": outdir_npz, "scene_xml": scene_xml,
            "config_act": config_act, "trajectory": trajectory, "video": video,
        }
        for gate, (field, op, thr) in B205.GATE_MAP.items():
            record[gate] = B205.gate_pass(field, op, thr, row)
        for metric in B205.METRIC_COLUMNS:
            record[metric] = row.get(metric, "")
        out_rows.append(record)

    OUT_TSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_TSV.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=cols, delimiter="\t")
        writer.writeheader()
        writer.writerows(out_rows)
    playable = sum(1 for r in out_rows if r["outdir_npz"] and (REPO / r["outdir_npz"]).is_file())
    videos = sum(1 for r in out_rows if r["video"])
    print(f"[done] wrote {OUT_TSV.relative_to(REPO)} ({len(out_rows)} rows, "
          f"{playable} playable, {videos} with mp4)")
    print("  per-arm:", dict(Counter(r["arm"] for r in out_rows)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
