#!/usr/bin/env python3
"""E206 C5a: did fixing the F7 pair bug actually improve contact on desk007?

plan236's headline scientific claim. E174 ran desk007 with a 41-box voxel proxy
but only **2** robot<->object pairs, so 40 of 41 collision boxes were invisible
to the robot (F7). E206 runs the same 9 desk007 cases, same 3cm contact mask,
with a 12-box hand-placed proxy and 2N/18N pairs. The bar:

    mean(hand_object_physics_contact_in_mask_frac) improves by >= +0.15 absolute

E174's arm is PRG, so E206-PRG is the like-for-like row; E206-noPRG is reported
alongside but is NOT the C5a comparison (different leg-constraint condition).

Paired by case_id -- these are literally the same 9 clips, so the per-case delta
is meaningful and is reported with mean / std / worst, not just the mean.
"""

from __future__ import annotations

import json
import math
import statistics
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import e206_common as C  # noqa: E402

E174_METRICS = (C.WS / "results/E174/s6_downstream/eval/full/e174_case_metrics.tsv")
E206_ROLLOUT = C.S6_DIR / "eval/two_arm/e206_two_arm_rollout.tsv"
FIELD = "hand_object_physics_contact_in_mask_frac"
BAR = 0.15


def f(v: Any) -> float:
    try:
        x = float(v)
        return x if math.isfinite(x) else math.nan
    except (TypeError, ValueError):
        return math.nan


def describe(vals: list[float]) -> dict[str, Any]:
    vals = [v for v in vals if math.isfinite(v)]
    if not vals:
        return {"n": 0}
    return {"n": len(vals), "mean": round(statistics.mean(vals), 4),
            "std": round(statistics.pstdev(vals), 4) if len(vals) > 1 else 0.0,
            "worst": round(min(vals), 4), "best": round(max(vals), 4)}


def main() -> int:
    e174 = {r["case_id"]: r for r in C.read_tsv(E174_METRICS)
            if r.get("object_key") == "desk007"}
    e206 = {(r["arm"], r["case_id"]): r for r in C.read_tsv(E206_ROLLOUT)
            if r["object_key"] == "desk007"}
    cases = sorted(set(e174) & {c for _a, c in e206})
    if not cases:
        raise SystemExit("no overlapping desk007 cases")

    rows = []
    print(f"{'case':30s} {'E174 PRG':>9s} {'E206 PRG':>9s} {'delta':>7s} "
          f"{'E206 noPRG':>11s} {'legpen174':>10s} {'legpen206':>10s}")
    for cid in cases:
        b = f(e174[cid][FIELD])
        p = f(e206[("prg", cid)][FIELD])
        n = f(e206[("noprg", cid)][FIELD])
        lb = f(e174[cid].get("leg_penetration_frac"))
        lp = f(e206[("prg", cid)].get("leg_penetration_frac"))
        rows.append({"case_id": cid, "e174_prg": b, "e206_prg": p,
                     "delta_prg": p - b, "e206_noprg": n,
                     "e174_leg_pen": lb, "e206_prg_leg_pen": lp})
        print(f"{cid:30s} {b:9.4f} {p:9.4f} {p-b:+7.4f} {n:11.4f} "
              f"{lb:10.4f} {lp:10.4f}")

    base = describe([r["e174_prg"] for r in rows])
    prg = describe([r["e206_prg"] for r in rows])
    noprg = describe([r["e206_noprg"] for r in rows])
    delta = describe([r["delta_prg"] for r in rows])
    improved = sum(1 for r in rows if r["delta_prg"] > 0)
    verdict = "pass" if (prg["mean"] - base["mean"]) >= BAR else "fail"

    payload: dict[str, Any] = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "claim": "C5a", "field": FIELD, "bar_absolute_improvement": BAR,
        "cases": len(rows), "object": "desk007", "mask": "3cm",
        "e174_prg_baseline": base, "e206_prg": prg, "e206_noprg": noprg,
        "paired_delta_prg_minus_e174": delta,
        "cases_improved": improved, "cases_regressed": len(rows) - improved,
        "mean_improvement": round(prg["mean"] - base["mean"], 4),
        "verdict": verdict,
        "leg_pen": {"e174": describe([r["e174_leg_pen"] for r in rows]),
                    "e206_prg": describe([r["e206_prg_leg_pen"] for r in rows])},
    }
    out = C.S6_DIR / "eval/two_arm"
    out.mkdir(parents=True, exist_ok=True)
    C.write_tsv(out / "c5a_vs_e174_desk007.tsv", rows)
    (out / "c5a_vs_e174_desk007.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("\n" + json.dumps({k: v for k, v in payload.items()
                             if k not in ("generated_at",)},
                            ensure_ascii=False, indent=2))
    return 0 if verdict == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
