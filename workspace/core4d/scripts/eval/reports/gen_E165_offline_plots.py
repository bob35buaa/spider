#!/usr/bin/env python3
"""E165 Phase0 可视化: A(box004/box021 距离 vs 标签 vs recall) / C(box023 手↔髋) / E1(三标量)."""
from __future__ import annotations
import csv, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "/mnt/public/usr/yancilin/work_dir/embodied"
OUT = f"{BASE}/spider/workspace/core4d/results/E165"
PROBE = f"{BASE}/SUGAR-private/outputs/core4d_e163_threecase_contact_probe"


def read_perframe(case):
    p = f"{OUT}/A_box004_contact_audit/{case}_per_frame.csv"
    rows = list(csv.DictReader(open(p)))
    fr = np.array([int(r["frame"]) for r in rows])
    lab = np.array([int(r["label"]) for r in rows])
    md = np.array([float(r["min_true"]) for r in rows])
    return fr, lab, md


def read_probe(case_sub):
    rows = list(csv.DictReader(open(f"{PROBE}/{case_sub}/isaac_contact_framewise.csv")))
    sc = np.array([r["source_contact"] == "True" for r in rows])
    ie = np.array([(r["isaac_left_contact"] == "True") or (r["isaac_right_contact"] == "True") for r in rows])
    net = np.array([max(float(r["left_net_force_n"]), float(r["right_net_force_n"])) for r in rows])
    return sc, ie, net


# ---- Plot A: box004 vs box021 distance + label band + isaac recall ----
fig, axs = plt.subplots(2, 1, figsize=(11, 7), sharex=False)
for ax, case, sub in [(axs[0], "box004", "box004_r161"), (axs[1], "box021", "box021_r160")]:
    fr, lab, md = read_perframe(case)
    sc, ie, net = read_probe(sub)
    n = min(len(fr), len(sc))
    ax.plot(fr[:n], md[:n] * 100, "b-", lw=1.3, label="wrist↔box surface dist (cm)")
    ax.axhline(8.0, color="k", ls="--", lw=0.8, label="proxy radius 8cm")
    ax.fill_between(fr[:n], 0, 30, where=lab[:n].astype(bool), color="orange", alpha=0.15, label="contact LABEL")
    ax.fill_between(fr[:n], 0, 30, where=ie[:n], color="green", alpha=0.30, label="Isaac filtered contact")
    rec = ie[sc[:n]].mean() if sc[:n].sum() else 0
    ax.set_title(f"{case}: labeled-frame Isaac recall = {rec:.2f}  (dist median≈{np.median(md[lab.astype(bool)])*100:.1f}cm)")
    ax.set_ylabel("dist (cm)"); ax.set_ylim(0, 30); ax.legend(fontsize=7, loc="upper right")
axs[1].set_xlabel("frame")
fig.suptitle("E165-A: box004 标签虚高=近而未触 (label 有/Isaac 接触≈无); box021 对照 label≈Isaac 接触")
fig.tight_layout(); fig.savefig(f"{OUT}/A_box004_contact_audit/A_distance_label_recall.png", dpi=120)
print("wrote A_distance_label_recall.png")

# ---- Plot C: box023 hand-hip spider vs omni ----
trace = json.load(open(f"{OUT}/C_box023_penetration/box023_trace.json"))
fig, ax = plt.subplots(figsize=(8, 4.5))
for key, c, m in [("spider", "C0", "o"), ("omni", "C1", "s")]:
    L = trace[key]["f0_10_L_hand_L_hip"]; R = trace[key]["f0_10_R_hand_R_hip"]
    x = np.arange(len(L))
    ax.plot(x, np.array(L) * 100, c + "-" + m, label=f"{key} L_hand↔L_hip")
    ax.plot(x, np.array(R) * 100, c + "--" + m, alpha=0.7, label=f"{key} R_hand↔R_hip")
ax.axhline(10.0, color="k", ls=":", label="10cm 判定线")
ax.set_xlabel("frame (0-9)"); ax.set_ylabel("hand↔same-side hip (cm)")
ax.set_title(f"E165-C: box023 手贴髋 — {trace['CLAIM_C_C_verdict'].split('(')[0].strip()}")
ax.legend(fontsize=7); fig.tight_layout()
fig.savefig(f"{OUT}/C_box023_penetration/C_hand_hip_spider_vs_omni.png", dpi=120)
print("wrote C_hand_hip_spider_vs_omni.png")

# ---- Plot E1: 3 scalars bar ----
sc = json.load(open(f"{OUT}/E1_onrails_probe/onrails_probe_scalars.json"))["cases"]
cases = ["box021", "box004", "box023"]
fig, axs = plt.subplots(1, 3, figsize=(13, 4))
metrics = [("filtered_contact_recall_either", "filtered recall", 1.0),
           ("phantom_force_rate", "phantom force rate", 1.0),
           ("max_init_net_force", "max init net force (N)", None)]
for ax, (k, title, ymax) in zip(axs, metrics):
    vals = [sc[c][k] for c in cases]
    stg = [sc[c]["staggered_success"] for c in cases]
    bars = ax.bar(cases, vals, color=["C2", "C1", "C3"])
    for b, s in zip(bars, stg):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"stag={s}", ha="center", va="bottom", fontsize=8)
    ax.set_title(title)
    if ymax: ax.set_ylim(0, ymax)
fig.suptitle("E165-E1: 三正交标量 (单标量不预测下游; 联合分病: box021清洁/box004虚高/box023自碰撞)")
fig.tight_layout(); fig.savefig(f"{OUT}/E1_onrails_probe/E1_three_scalars.png", dpi=120)
print("wrote E1_three_scalars.png")
