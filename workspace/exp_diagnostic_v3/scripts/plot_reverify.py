#!/usr/bin/env python3
"""Re-verification plots for diagnostic v3.

Generates:
  1. rotation bar chart (GT object rotation per case, with user's 180-claims marked)
  2. lift-height bar chart
  3. scatter: Spider-Omni 5cm gap vs case quality (rotation, lift) -> shows gap is
     near-constant offset, NOT driven by quality.
  4. scatter: absolute 5cm contact (Omni & Spider) vs GT contact -> shows quality
     drives achievable contact for both methods together.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[3]
TSV = REPO / "workspace/exp_diagnostic_v3/results/trajectory_analysis/object_trajectory_metrics.tsv"
OUT = REPO / "workspace/exp_diagnostic_v3/results/figures"
OUT.mkdir(parents=True, exist_ok=True)

CAT = {
    "box021_035_p1": "green", "box021_035_p2": "green", "box021_029_p2": "green",
    "box004_082_p1": "green", "box004_083_p1": "green", "box004_083_p2": "green",
    "box023_person2": "green", "box026_139_p1": "mix",
    "box026_133_p1": "rot", "box026_133_p2": "rot",
    "box026_20231020_139_p2": "rot", "box026_20231023_139_p2": "rot",
    "box026_135_p1": "lift", "box026_135_p2": "lift", "box026_137_p1": "lift",
    "box026_134_p2": "lift", "box026_039_p2": "walkup", "box026_134_p1": "walkup",
    "box026_138_p2": "walkup", "box026_141_p1": "gray", "box026_141_p2": "gray",
}
CAT_COLOR = {
    "green": "#3a9c3a", "mix": "#9acd32", "rot": "#9b7fc7", "lift": "#e0c040",
    "walkup": "#6a4fa0", "gray": "#999999",
}


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def load():
    rows = list(csv.DictReader(TSV.open(), delimiter="\t"))
    return [r for r in rows if r["case_id"] in CAT]


def short(cid):
    return cid.replace("box026_", "26_").replace("box021_", "21_").replace(
        "box004_", "04_").replace("box023_", "23_").replace("20231020_", "").replace(
        "20231023_", "")


def main():
    rows = load()
    rows.sort(key=lambda r: (CAT[r["case_id"]], -(f(r["ref_rot_total_deg"]) if not np.isnan(f(r["ref_rot_total_deg"])) else f(r["object_rotation_total_deg"]))))

    cids = [short(r["case_id"]) for r in rows]
    cols = [CAT_COLOR[CAT[r["case_id"]]] for r in rows]
    rot = [f(r["ref_rot_total_deg"]) if not np.isnan(f(r["ref_rot_total_deg"])) else f(r["object_rotation_total_deg"]) for r in rows]
    lift = [f(r["object_z_lift_m"]) for r in rows]

    # 1+2 rotation & lift bars
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 9))
    ax1.bar(range(len(cids)), rot, color=cols)
    ax1.axhline(180, ls="--", c="r", lw=1, label="180 deg (user claim)")
    ax1.axhline(30, ls=":", c="gray", lw=1, label="30 deg clean threshold")
    ax1.set_ylabel("GT object total rotation (deg)")
    ax1.set_title("Object rotation per case (GT-locked reference) - verifies '180 rotation' claims")
    ax1.set_xticks(range(len(cids)))
    ax1.set_xticklabels(cids, rotation=90, fontsize=7)
    ax1.legend()
    ax2.bar(range(len(cids)), lift, color=cols)
    ax2.axhline(0.05, ls=":", c="gray", lw=1, label="0.05 m lift threshold")
    ax2.set_ylabel("object lift height (m)")
    ax2.set_title("Object lift per case - verifies 'insufficient lift / never lifted' claims")
    ax2.set_xticks(range(len(cids)))
    ax2.set_xticklabels(cids, rotation=90, fontsize=7)
    ax2.legend()
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in CAT_COLOR.values()]
    ax1.legend(handles + ax1.get_legend().legend_handles,
               list(CAT_COLOR.keys()) + ["180 deg", "30 deg"], ncol=4, fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "reverify_rotation_lift.png", dpi=110)
    plt.close(fig)

    # 3 gap vs quality
    o5 = np.array([f(r["e109_omni_hand_near_5cm_frac"]) for r in rows])
    s5 = np.array([f(r["e109_spider_hand_near_5cm_frac"]) for r in rows])
    gtc = np.array([f(r["gt_contact_frac"]) for r in rows])
    d5 = s5 - o5
    rot_a = np.array(rot)
    lift_a = np.array(lift)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].scatter(rot_a, d5 * 100, c=cols, s=60, edgecolor="k")
    axes[0].axhline(np.nanmean(d5) * 100, ls="--", c="b",
                    label=f"mean {np.nanmean(d5)*100:+.1f}pp")
    axes[0].set_xlabel("GT object rotation (deg)")
    axes[0].set_ylabel("Spider - Omni  5cm contact gap (pp)")
    axes[0].set_title(f"Gap vs rotation  (r={np.corrcoef(rot_a, d5)[0,1]:+.2f}) - near-constant offset")
    axes[0].legend()
    axes[1].scatter(lift_a, d5 * 100, c=cols, s=60, edgecolor="k")
    axes[1].axhline(np.nanmean(d5) * 100, ls="--", c="b")
    axes[1].set_xlabel("object lift height (m)")
    axes[1].set_ylabel("Spider - Omni  5cm contact gap (pp)")
    m = ~(np.isnan(lift_a) | np.isnan(d5))
    axes[1].set_title(f"Gap vs lift  (r={np.corrcoef(lift_a[m], d5[m])[0,1]:+.2f})")
    fig.tight_layout()
    fig.savefig(OUT / "reverify_gap_vs_quality.png", dpi=110)
    plt.close(fig)

    # 4 absolute contact vs GT
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(gtc, o5, c="#c0392b", s=55, edgecolor="k", label="OmniRetarget 5cm")
    ax.scatter(gtc, s5, c="#2471a3", s=55, edgecolor="k", marker="^", label="Spider 5cm")
    for i in range(len(rows)):
        ax.plot([gtc[i], gtc[i]], [o5[i], s5[i]], c="gray", lw=0.5, alpha=0.5)
    mm = ~(np.isnan(gtc) | np.isnan(o5))
    ax.set_xlabel("GT mocap contact frac (data quality proxy)")
    ax.set_ylabel("5cm near-contact frac")
    ax.set_title(f"Achievable contact tracks data quality\nOmni_5cm vs GT r={np.corrcoef(gtc[mm], o5[mm])[0,1]:+.2f}; both methods move together")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "reverify_contact_vs_gt.png", dpi=110)
    plt.close(fig)
    print("wrote 3 figures to", OUT)


if __name__ == "__main__":
    main()
