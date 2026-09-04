#!/usr/bin/env python3
"""E206 C6: the mandatory visual-review coverage set, and the empty verdict sheet.

plan236 C6 / rule 5 fix the coverage deliberately, not by convenience sampling:

  * every object x both arms   -- so no object ships unlooked-at
  * EVERY L3_auto row          -- L3 is auto-accept, so reward hacking here is
                                  the most expensive kind of mistake
  * EVERY single-gate L1 row   -- a rollout that fails exactly one gate is where
                                  "the number is wrong, not the motion" hides

Emits `user_manual_review.tsv` with USE / DO_NOT_USE left blank for a human, plus
a `numeric_vs_visual_conflicts` column that the reviewer fills when the verdict
disagrees with the gates -- plan236 requires those be listed one by one.

Usage:
    .venv/bin/python .../build_review_coverage.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import e206_common as C  # noqa: E402

EVAL_DIR = C.S6_DIR / "eval/two_arm"
ROLLOUT = EVAL_DIR / "e206_two_arm_rollout.tsv"
RENDER_DIR = C.S6_DIR / "render/full"
OUT = C.S6_DIR / "review"


def main() -> int:
    rows = C.read_tsv(ROLLOUT)
    if not rows:
        raise SystemExit(f"missing or empty {ROLLOUT}")

    reasons: dict[tuple[str, str], set[str]] = {}

    def mark(r: dict[str, str], why: str) -> None:
        reasons.setdefault((r["case_id"], r["arm"]), set()).add(why)

    # 1. every object x both arms -- pick the worst-layer row per (object, arm)
    order = {"L1_reject": 0, "L2_review": 1, "L3_auto": 2}
    per: dict[tuple[str, str], dict[str, str]] = {}
    for r in rows:
        key = (r["object_key"], r["arm"])
        cur = per.get(key)
        if cur is None or order[r["layer"]] < order[cur["layer"]]:
            per[key] = r
    for r in per.values():
        mark(r, "object_x_arm")

    # 2. every L3_auto
    for r in rows:
        if r["layer"] == "L3_auto":
            mark(r, "L3_auto")

    # 3. every single-gate L1
    for r in rows:
        if r["layer"] != "L1_reject":
            continue
        failed = [f for f in
                  (r.get("hard_failed", "") + "," + r.get("wide_failed", "")).split(",")
                  if f]
        if len(set(failed)) == 1:
            mark(r, f"single_gate_L1:{failed[0]}")

    by = {(r["case_id"], r["arm"]): r for r in rows}
    out_rows: list[dict[str, Any]] = []
    for (case_id, arm), why in sorted(reasons.items()):
        r = by[(case_id, arm)]
        mp4 = RENDER_DIR / f"E206_{case_id}_{arm}.mp4"
        out_rows.append({
            "case_id": case_id, "arm": arm, "object_key": r["object_key"],
            "coverage_reason": ";".join(sorted(why)),
            "layer": r["layer"],
            "gate12_pass": r["gate12_pass"],
            "narrow_failed": r.get("narrow_failed", ""),
            "wide_failed": r.get("wide_failed", ""),
            "hard_failed": r.get("hard_failed", ""),
            "leg_penetration_frac": r.get("leg_penetration_frac", ""),
            "hand_object_physics_contact_in_mask_frac":
                r.get("hand_object_physics_contact_in_mask_frac", ""),
            "video": str(mp4.relative_to(C.REPO)) if mp4.is_file() else "",
            # --- filled by the human reviewer ---
            "visual_verdict": "", "failure_mode": "",
            "numeric_vs_visual_conflict": "", "reviewer": "", "reviewed_at": "",
            "notes": "",
        })

    OUT.mkdir(parents=True, exist_ok=True)
    C.write_tsv(OUT / "user_manual_review.tsv", out_rows)
    payload = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "total_rows_scored": len(rows),
        "coverage_rows": len(out_rows),
        "coverage_frac": round(len(out_rows) / len(rows), 3),
        "by_reason": dict(Counter(
            w for v in reasons.values() for w in
            {x.split(":")[0] for x in v})),
        "by_layer": dict(Counter(r["layer"] for r in out_rows)),
        "by_object": dict(Counter(r["object_key"] for r in out_rows)),
        "missing_video": [r["case_id"] for r in out_rows if not r["video"]],
        "player": "bash workspace/core4d/scripts/eval/wrappers/review_player.sh E206ARM",
    }
    (OUT / "user_manual_review_coverage.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
