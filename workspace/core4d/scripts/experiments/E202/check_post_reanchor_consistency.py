#!/usr/bin/env python3
"""E202-export step 4: post-reanchor two-person consistency check (stricter than E200).

The Holosoma exporter re-anchors partner hands into the SOURCE object frame, so the
raw per-person perturbation divergence does NOT reach the final motion. This verifies
that directly on the EXPORTED motions: during the grasp/contact window, the (re-anchored)
partner hands must sit near the SOURCE object surface -- i.e. the two agents agree on
one object. Baseline = the `original` (unperturbed) variant, which shares the same
re-anchor path.

Reads the Holosoma export manifests (object_mismatch_*/partner_move_* diagnostics) and
the exported NPZs (partner_hand_pos_w in the source-object world frame + object pose +
object_contact mask). Reports per (object, variant) grasp-window partner-hand-to-object
surface distance (mean/worst), and passes if aug grasp-window distance is within
GRASP_TOL_M and not materially worse than the original variant.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[5]
HS_OUT = Path("/mnt/ali-sh-1/usr/xiayibo/work_dir/embodied/holosoma/workspace/v3/data/E202_use13_aug_partner_rl")
OUT_TSV = REPO / "workspace/core4d/results/E202/s6_downstream/rl_export_aug/post_reanchor_consistency.tsv"
HALF_EXTENTS = {
    "bucket003": (0.270393755, 0.381407245, 0.23287703),
    "bucket004": (0.161566625, 0.231058755, 0.15230013),
    "bucket007": (0.273275865, 0.2869532, 0.284911265),
}
GRASP_TOL_M = 0.08           # partner hand within 8cm of source-object surface during contact
AUG_VS_ORIG_SLACK_M = 0.05   # aug grasp-window distance may exceed orig by at most this

FIELDS = [
    "object", "variant_class", "n_motions",
    "object_mismatch_max_m", "partner_move_max_m",
    "grasp_hand_surface_mean_m", "grasp_hand_surface_worst_m",
    "status", "note",
]


def quat_conj(q):  # (...,4) wxyz
    out = q.copy(); out[..., 1:] *= -1.0; return out


def quat_apply(q, v):  # rotate v by q (wxyz)
    w = q[..., :1]; xyz = q[..., 1:]
    t = 2.0 * np.cross(xyz, v)
    return v + w * t + np.cross(xyz, t)


def surface_dist_obb(hand_w, obj_pos, obj_quat, he):
    """Distance from each hand point to the object OBB surface (0 if on surface)."""
    he = np.asarray(he)
    local = quat_apply(quat_conj(obj_quat)[:, None, :], hand_w - obj_pos[:, None, :])  # (T,2,3)
    outside = np.clip(np.abs(local) - he, 0.0, None)
    return np.linalg.norm(outside, axis=-1)  # (T,2)


def variant_class(case_id: str) -> str:
    # exported motions keep the source aug unit id: ..._pN__transK ; orig has no __
    return "aug" if "__trans" in case_id else "orig"


def process_object(obj: str) -> list[dict]:
    man = HS_OUT / obj / "manifest.tsv"
    if not man.is_file():
        return [{"object": obj, "variant_class": "-", "status": "NO_MANIFEST", "note": str(man)}]
    with man.open(newline="", encoding="utf-8") as s:
        rows = list(csv.DictReader(s, delimiter="\t"))
    he = HALF_EXTENTS[obj]
    agg = defaultdict(lambda: {"mm": [], "mv": [], "gs": [], "n": 0})
    for row in rows:
        if row.get("decision") != "export_pass":
            continue
        cls = variant_class(row.get("case_id", ""))
        a = agg[cls]; a["n"] += 1
        for k in ("object_mismatch_max_m",):
            if row.get(k):
                a["mm"].append(float(row[k]))
        for k in ("partner_move_max_l_m", "partner_move_max_r_m"):
            if row.get(k):
                a["mv"].append(float(row[k]))
        p = Path(row.get("export_with_partner_path", ""))
        if not p.is_file():
            continue
        with np.load(p, allow_pickle=False) as z:
            if not all(k in z.files for k in ("partner_hand_pos_w", "object_pos_w", "object_quat_w")):
                continue
            hand = np.asarray(z["partner_hand_pos_w"], float)
            opos = np.asarray(z["object_pos_w"], float)
            oquat = np.asarray(z["object_quat_w"], float)
            d = surface_dist_obb(hand, opos, oquat, he)  # (T,2)
            if "object_contact" in z.files:
                mask = np.asarray(z["object_contact"], float) > 0.5  # (T,2)
                sel = d[mask] if mask.any() else d.reshape(-1)
            else:
                sel = d.reshape(-1)
            if sel.size:
                a["gs"].append(float(np.mean(sel)))
                a.setdefault("gsworst", []).append(float(np.max(sel)))
    out = []
    orig_mean = None
    for cls in ("orig", "aug"):
        if cls not in agg:
            continue
        a = agg[cls]
        gmean = float(np.mean(a["gs"])) if a["gs"] else float("nan")
        gworst = float(np.max(a.get("gsworst", []))) if a.get("gsworst") else float("nan")
        if cls == "orig":
            orig_mean = gmean
        out.append({
            "object": obj, "variant_class": cls, "n_motions": a["n"],
            "object_mismatch_max_m": round(max(a["mm"]), 4) if a["mm"] else "",
            "partner_move_max_m": round(max(a["mv"]), 4) if a["mv"] else "",
            "grasp_hand_surface_mean_m": round(gmean, 4),
            "grasp_hand_surface_worst_m": round(gworst, 4),
            "status": "", "note": "",
        })
    # judge aug rows
    for rec in out:
        if rec["variant_class"] != "aug":
            rec["status"] = "BASELINE"
            continue
        gm = rec["grasp_hand_surface_mean_m"]
        fails = []
        if not (gm == gm) or gm > GRASP_TOL_M:
            fails.append(f"grasp_surface>{GRASP_TOL_M}m")
        has_orig = orig_mean is not None and orig_mean == orig_mean
        if has_orig and gm == gm and gm > orig_mean + AUG_VS_ORIG_SLACK_M:
            fails.append("worse_than_orig_baseline")
        rec["status"] = "PASS" if not fails else "FAIL"
        rec["note"] = "; ".join(fails) or (
            f"orig_baseline_mean={orig_mean}" if has_orig
            else "no orig baseline in aug export (orig set in log264); judged vs abs GRASP_TOL")
    return out


def main() -> int:
    argparse.ArgumentParser().parse_args()
    all_rows: list[dict] = []
    for obj in ("bucket003", "bucket004", "bucket007"):
        all_rows.extend(process_object(obj))
    OUT_TSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_TSV.open("w", newline="", encoding="utf-8") as s:
        w = csv.DictWriter(s, fieldnames=FIELDS, delimiter="\t", lineterminator="\n", extrasaction="ignore")
        w.writeheader(); w.writerows(all_rows)
    print(json.dumps(all_rows, ensure_ascii=False, indent=2))
    aug = [r for r in all_rows if r["variant_class"] == "aug"]
    n_fail = sum(r["status"] == "FAIL" for r in aug)
    print(f"\n[post-reanchor] aug rows {len(aug)}, FAIL={n_fail} -> {OUT_TSV}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
