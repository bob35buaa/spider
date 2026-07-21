#!/usr/bin/env python3
"""E165-C: box023 初始穿透(手贴髋自碰撞)溯源 (训练-free, 离线).

Claim C-C: box023 frame0 自碰撞是继承自源/姿态, 还是 spider CEM 独有?
判据: 对 spider 与 omni 两条 handoff 比 frame0-10 的 min(同侧 wrist↔hip)。
  - 两者都 < 0.10m  -> 继承(两条 pipeline 共有, 源/姿态问题)
  - 仅 spider < 0.10 -> CEM 引入

口径: wrist_yaw_link (idx 10/13) ↔ hip_roll_link (idx 1/4), 同侧配对。robot_50hz body_pos_w 世界系。
附带: 同帧 wrist 高度 z、wrist↔box 表面距离, 复现文档"手在髋高、远离箱"的描述。
"""
from __future__ import annotations
import argparse, json, os, pickle
import numpy as np

BASE = "/mnt/public/usr/yancilin/work_dir/embodied"
L_HIP, R_HIP, L_WRIST, R_WRIST = 1, 4, 10, 13


def obj_aabb(path):
    vs = []
    with open(path) as f:
        for ln in f:
            if ln.startswith("v "):
                p = ln.split(); vs.append([float(p[1]), float(p[2]), float(p[3])])
    v = np.asarray(vs); return v.min(0), v.max(0)


def load_case(folder):
    d = f"{BASE}/SUGAR-private/data/{folder}/data_000"
    robot = np.load(f"{d}/robot_50hz.npz")
    obj = pickle.load(open(f"{d}/obj_motion_global_50hz.pkl", "rb"))
    return robot["body_pos_w"], obj["obj_trans"], obj["obj_rot"]


def hand_to_obb_dist(wrist_w, obj_trans, obj_rot, mn, mx):
    rel = wrist_w - obj_trans[:, None, :]
    rel_l = np.einsum("tji,thj->thi", obj_rot, rel)
    cl = np.clip(rel_l, mn[None, None, :], mx[None, None, :])
    return np.linalg.norm(rel_l - cl, axis=-1)


def analyze(folder, mesh, nframes=10):
    body, obj_trans, obj_rot = load_case(folder)
    T = min(len(body), len(obj_trans))
    body, obj_trans, obj_rot = body[:T], obj_trans[:T], obj_rot[:T]
    lw, rw = body[:, L_WRIST], body[:, R_WRIST]
    lh, rh = body[:, L_HIP], body[:, R_HIP]
    d_lhand_lhip = np.linalg.norm(lw - lh, axis=1)
    d_rhand_rhip = np.linalg.norm(rw - rh, axis=1)
    min_same_side = np.minimum(d_lhand_lhip, d_rhand_rhip)
    mn, mx = obj_aabb(mesh)
    wrist = np.stack([lw, rw], 1)
    dist_box = hand_to_obb_dist(wrist, obj_trans, obj_rot, mn, mx).min(1)
    k = min(nframes, T)
    return dict(
        folder=folder, frames=int(T),
        f0_10_min_hand_hip=float(min_same_side[:k].min()),
        f0_10_mean_hand_hip=float(min_same_side[:k].mean()),
        f0_L_hand_L_hip=float(d_lhand_lhip[0]), f0_R_hand_R_hip=float(d_rhand_rhip[0]),
        f0_10_L_hand_L_hip=[round(float(x), 4) for x in d_lhand_lhip[:k]],
        f0_10_R_hand_R_hip=[round(float(x), 4) for x in d_rhand_rhip[:k]],
        f0_10_wrist_z=[round(float(x), 4) for x in wrist[:k, :, 2].min(1)],
        f0_10_min_hand_box=[round(float(x), 4) for x in dist_box[:k]],
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spider", default="Core4D_E163N_Box023_R158")
    ap.add_argument("--omni", default="Core4D_E163N_Box023_R158_SamePersonOmniRT")
    ap.add_argument("--mesh", default=f"{BASE}/spider/example_datasets/processed/core4d/assets/objects/box023/box023_m.obj")
    ap.add_argument("--threshold", type=float, default=0.10)
    ap.add_argument("--out", default=f"{BASE}/spider/workspace/core4d/results/E165/C_box023_penetration")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    sp = analyze(args.spider, args.mesh)
    om = analyze(args.omni, args.mesh)
    sp_close = sp["f0_10_min_hand_hip"] < args.threshold
    om_close = om["f0_10_min_hand_hip"] < args.threshold
    if sp_close and om_close:
        verdict = "INHERITED (both pipelines hands-at-hip -> 源/姿态问题, 非 CEM 独有)"
    elif sp_close and not om_close:
        verdict = "CEM_INTRODUCED (仅 spider 手贴髋)"
    elif not sp_close:
        verdict = "NOT_REPRODUCED (spider frame0-10 手不贴髋, 重新核对自碰撞假设)"
    else:
        verdict = "OMNI_ONLY (仅 omni 贴髋)"

    out = dict(threshold=args.threshold, spider=sp, omni=om,
               spider_close=bool(sp_close), omni_close=bool(om_close),
               CLAIM_C_C_verdict=verdict)
    with open(f"{args.out}/box023_trace.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    print(f"[E165-C] wrote {args.out}/box023_trace.json")


if __name__ == "__main__":
    main()
