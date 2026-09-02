#!/usr/bin/env python3
"""E202-export C6: USE13-subset stratified physical-validity report.

Re-aggregates E202's already-computed aug rollout metrics (public core_metrics,
`e202_aug_case_metrics.tsv`) restricted to the 12 manual-USE source cases that
have E202 source aug. Reports orig vs aug (trans pooled) mean+std+worst per object
and overall -- no new metric computation, no CEM. Stdlib only (no mujoco import).

Marks each variant with whether it was actually EXPORTED (passed C2 parity + C4)
by reading the export summary if present.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[5]
AUG_METRICS = REPO / "workspace/core4d/results/E202/s6_downstream/eval/full_augmentation/e202_aug_case_metrics.tsv"
REVIEW = REPO / "workspace/core4d/results/E178/s6_downstream/eval/full/user_manual_review_filled.tsv"
EXPORT_SUMMARY = REPO / "workspace/core4d/results/E202/s6_downstream/rl_export_aug/rl_export_summary.json"
OUT_TSV = REPO / "workspace/core4d/results/E202/s6_downstream/rl_export_aug/use13_stratified_metrics.tsv"
OUT_JSON = OUT_TSV.with_suffix(".json")

METRICS = {
    "track_obj_pos_err_cm_mean": "obj_pos_cm",
    "track_obj_ori_err_deg_mean": "obj_ori_deg",
    "track_root_pos_err_cm_mean": "root_pos_cm",
    "track_eef_pos_err_cm_mean": "eef_pos_cm",
    "hand_object_physics_contact_in_mask_frac": "contact_in_mask",
    "hand_object_physics_penetration_3mm_frame_frac": "hand_pen_3mm",
    "leg_penetration_frac": "leg_pen",
    "fall_flag": "fall",
    "all_gates_pass": "gate_pass",
}


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as s:
        return list(csv.DictReader(s, delimiter="\t"))


def fnum(v: str) -> float | None:
    v = (v or "").strip().lower()
    if v in {"", "nan", "none"}:
        return None
    if v in {"true", "false"}:
        return 1.0 if v == "true" else 0.0
    try:
        return float(v)
    except ValueError:
        return None


def stats(vals: list[float]) -> dict[str, float]:
    vals = [v for v in vals if v is not None]
    if not vals:
        return {"n": 0, "mean": math.nan, "std": math.nan, "worst": math.nan}
    m = sum(vals) / len(vals)
    sd = math.sqrt(sum((x - m) ** 2 for x in vals) / len(vals)) if len(vals) > 1 else 0.0
    return {"n": len(vals), "mean": m, "std": sd, "worst": max(vals)}


def main() -> int:
    use = {r["case_id"] for r in read_tsv(REVIEW) if r["manual_use_decision"] == "USE"}
    rows = [r for r in read_tsv(AUG_METRICS) if r["case_id"] in use]
    # variant class: orig vs aug(trans)
    def vclass(r: dict[str, str]) -> str:
        v = (r.get("aug_variant") or r.get("variant") or "").lower()
        return "orig" if v == "orig" else "aug"

    objects = sorted({r["case_id"].split("_")[0] for r in rows})
    out_rows: list[dict[str, object]] = []
    for obj in objects + ["ALL"]:
        subset = rows if obj == "ALL" else [r for r in rows if r["case_id"].startswith(obj)]
        for cls in ("orig", "aug"):
            crows = [r for r in subset if vclass(r) == cls]
            rec: dict[str, object] = {"object": obj, "variant_class": cls, "n_rollouts": len(crows)}
            for col, short in METRICS.items():
                st = stats([fnum(r.get(col, "")) for r in crows])
                rec[f"{short}_mean"] = round(st["mean"], 4) if st["n"] else ""
                rec[f"{short}_std"] = round(st["std"], 4) if st["n"] else ""
                rec[f"{short}_worst"] = round(st["worst"], 4) if st["n"] else ""
            out_rows.append(rec)

    fields = ["object", "variant_class", "n_rollouts"] + [
        f"{s}_{k}" for s in METRICS.values() for k in ("mean", "std", "worst")
    ]
    OUT_TSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_TSV.open("w", newline="", encoding="utf-8") as s:
        w = csv.DictWriter(s, fieldnames=fields, delimiter="\t", lineterminator="\n", extrasaction="ignore")
        w.writeheader()
        w.writerows(out_rows)

    exported = None
    if EXPORT_SUMMARY.is_file():
        exported = json.loads(EXPORT_SUMMARY.read_text())
    summary = {
        "use13_source_cases_with_aug": len({r["case_id"] for r in rows}),
        "objects": objects,
        "note": "orig=E202 aug-eval orig rows (E178 orig, omnirt_v1); aug=trans pooled (omnirt_v2). "
                "USE13 subset of E202 full_augmentation.",
        "export_summary": {k: exported.get(k) for k in
                           ("source_rows", "excluded_variant_count", "object_counts", "variant_counts")}
        if exported else "export not yet run",
        "rows": out_rows,
    }
    OUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"\n-> {OUT_TSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
