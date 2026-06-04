#!/usr/bin/env python3
"""Select the clean contact benchmark from object-trajectory metrics.

Inclusion criteria (data-driven, agreed thresholds):
  - object total rotation < 45 deg   (clean carry, not a re-orient maneuver)
  - object lift height    > 0.30 m   (genuine lift-and-carry, not bend/handover)
  - GT mocap contact frac >= 0.30    (the human actually contacts the box -> a real
                                       contact task; guards against handover/no-contact)

Handover ("传递") cases are excluded explicitly (different action type).
Emits: clean_benchmark.tsv (all 24 cases with decision+reason) and
       clean_benchmark_manifest.tsv (included cases only, with paths for re-running).
"""
from __future__ import annotations

import csv
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
METRICS = REPO / "workspace/exp_diagnostic_v3/results/trajectory_analysis/object_trajectory_metrics.tsv"
VARIANTS = REPO / "workspace/core4d/scripts/E143/variants.tsv"
OUT_DIR = REPO / "workspace/exp_diagnostic_v3/results/benchmark"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ROT_MAX_DEG = 45.0
LIFT_MIN_M = 0.30
GT_CONTACT_MIN = 0.30

# Tier-1 core = passes thresholds AND no annotation caveat.
# Tier-2 conditional = passes thresholds but has a caveat that needs handling before
# it can be a clean A/B baseline (walk-up / ref-contact-init, near-threshold rotation,
# mixed ref leg-contact). Keyed by case -> caveat.
TIER2_CAVEAT = {
    "box026_039_p1": "walk-up + single-hand + p1 pre-lift jitter (deep-purple); verify ref contact-init",
    "box026_039_p2": "walk-up + ref initial hand-object contact mis-solved (deep-purple); needs ref repair",
    "box026_139_p1": "rotation 34deg + annotation 'ref leg contact / box rotates'; mixed",
    "box026_20231023_139_p2": "rotation 45deg at boundary (max 69deg); was mislabeled ~180",
}

# explicit non-target action types (user annotation: handover/pass)
HANDOVER = {"box026_141_p1", "box026_141_p2"}

VARIANT_FIELDS = [
    "ordinal", "variant", "e109_case_id", "case_id", "object_key",
    "target_variant_id", "source_task", "derived_task", "person_idx", "split",
    "ablation", "mask_path", "mask_kind", "source_mask_path", "baseline_npz_path",
    "baseline_run_id", "omni_qpos_path", "omni_scene_xml", "spider_scene_xml",
    "override", "run_status", "reuse_source_exp", "reuse_npz_path",
    "reuse_video_path", "reuse_outdir_npz_path",
]


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def load_variants():
    out = {}
    for line in VARIANTS.open():
        if line.startswith("#") or not line.strip():
            continue
        v = line.rstrip("\n").split("\t")
        v += [""] * (len(VARIANT_FIELDS) - len(v))
        r = dict(zip(VARIANT_FIELDS, v))
        out[r["case_id"]] = r
    return out


def decide(r):
    """Return (decision, reason) for a metrics row."""
    rot = f(r["ref_rot_total_deg"]) or f(r["object_rotation_total_deg"])
    lift = f(r["object_z_lift_m"])
    gtc = f(r["gt_contact_frac"])
    cid = r["case_id"]
    if rot is None or lift is None:
        return "EXCLUDE", f"no_object_pose({r['status']})"
    if cid in HANDOVER:
        return "EXCLUDE", "handover_action"
    if rot >= ROT_MAX_DEG:
        return "EXCLUDE", f"rotation_{rot:.0f}deg>=45"
    if lift <= LIFT_MIN_M:
        return "EXCLUDE", f"lift_{lift:.2f}m<=0.30"
    if gtc is not None and gtc < GT_CONTACT_MIN:
        return "EXCLUDE", f"gt_contact_{gtc:.2f}<0.30"
    if cid in TIER2_CAVEAT:
        return "TIER2", TIER2_CAVEAT[cid]
    return "TIER1", "clean_carry"


def main():
    rows = list(csv.DictReader(METRICS.open(), delimiter="\t"))
    variants = load_variants()

    decisions = []
    for r in rows:
        dec, reason = decide(r)
        rot = f(r["ref_rot_total_deg"]) or f(r["object_rotation_total_deg"])
        o5 = f(r["e109_omni_hand_near_5cm_frac"])
        s5 = f(r["e109_spider_hand_near_5cm_frac"])
        decisions.append({
            "case_id": r["case_id"],
            "object_key": r["object_key"],
            "decision": dec,
            "reason": reason,
            "object_rotation_deg": "" if rot is None else f"{rot:.1f}",
            "object_lift_m": r["object_z_lift_m"],
            "lift_onset_frac": r["lift_onset_frac"],
            "gt_contact_frac": r["gt_contact_frac"],
            "omni_5cm": "" if o5 is None else f"{o5:.4f}",
            "spider_5cm": "" if s5 is None else f"{s5:.4f}",
            "spider_minus_omni_5cm": "" if (o5 is None or s5 is None) else f"{s5 - o5:+.4f}",
            "object_jitter_accel_mean": r["object_jitter_accel_mean"],
        })

    rank = {"TIER1": 0, "TIER2": 1, "EXCLUDE": 2}
    decisions.sort(key=lambda d: (rank[d["decision"]], d["object_key"], d["case_id"]))

    cols = list(decisions[0].keys())
    with (OUT_DIR / "clean_benchmark.tsv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(decisions)

    included = [d for d in decisions if d["decision"] in ("TIER1", "TIER2")]
    man_cols = [
        "case_id", "tier", "object_key", "object_rotation_deg", "object_lift_m",
        "gt_contact_frac", "omni_5cm", "spider_5cm", "spider_minus_omni_5cm",
        "caveat", "run_status", "spider_scene_xml", "baseline_npz_path",
        "omni_qpos_path", "mask_path", "override",
    ]
    with (OUT_DIR / "clean_benchmark_manifest.tsv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=man_cols, delimiter="\t", lineterminator="\n")
        w.writeheader()
        for d in included:
            v = variants.get(d["case_id"], {})
            w.writerow({
                "case_id": d["case_id"], "tier": d["decision"],
                "object_key": d["object_key"],
                "object_rotation_deg": d["object_rotation_deg"],
                "object_lift_m": d["object_lift_m"],
                "gt_contact_frac": d["gt_contact_frac"],
                "omni_5cm": d["omni_5cm"], "spider_5cm": d["spider_5cm"],
                "spider_minus_omni_5cm": d["spider_minus_omni_5cm"],
                "caveat": "" if d["decision"] == "TIER1" else d["reason"],
                "run_status": v.get("run_status", ""),
                "spider_scene_xml": v.get("spider_scene_xml", ""),
                "baseline_npz_path": v.get("baseline_npz_path", ""),
                "omni_qpos_path": v.get("omni_qpos_path", ""),
                "mask_path": v.get("mask_path", ""),
                "override": v.get("override", ""),
            })

    from collections import Counter
    n1 = sum(1 for d in included if d["decision"] == "TIER1")
    n2 = sum(1 for d in included if d["decision"] == "TIER2")
    print(f"TIER1 {n1} + TIER2 {n2} = {len(included)} / {len(decisions)}")
    for d in included:
        print(f"  [{d['decision']}] {d['case_id']:24s} rot={d['object_rotation_deg']:>5}deg "
              f"lift={d['object_lift_m'][:5]} gtC={d['gt_contact_frac'][:4]} "
              f"d5cm={d['spider_minus_omni_5cm']}"
              + ("" if d["decision"] == "TIER1" else f"  <- {d['reason']}"))
    print("EXCLUDED reasons:")
    for reason, c in Counter(d["reason"] for d in decisions if d["decision"] == "EXCLUDE").most_common():
        print(f"  - {reason}: {c}")


if __name__ == "__main__":
    main()
