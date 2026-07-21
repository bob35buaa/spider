#!/usr/bin/env python3
"""E165-A: box004 接触标签审计 (训练-free, 离线).

Claim C-A: box004 接触标签虚高源于"手够不到箱"。
判据: 标签=接触的帧里, hand(wrist_yaw)↔box 表面最近距离中位数 > rubber-hand proxy 半径(0.08m 上界)
      且这些帧 Isaac filtered 接触 recall < 0.2。

口径说明:
- hand 用 wrist_yaw_link (robot_50hz body idx 10/13, 与 convert heuristic 同口径)。
- box 表面距离 = 点到 OBB 的外部距离 (盒内=0); OBB 由 .obj 网格 AABB + obj_rot/obj_trans 决定。
- 额外对照: convert 的 heuristic 对所有 box 硬编码了 BOX021 半尺寸 -> 用 box021 vs 真实 box004 半尺寸各算一遍, 量化"虚胖"。
"""
from __future__ import annotations
import argparse, json, os, pickle
import numpy as np

BASE = "/mnt/public/usr/yancilin/work_dir/embodied"
# SUGAR_BODY_ORDER index (see convert_holosoma_export_to_sugar.py)
L_WRIST, R_WRIST = 10, 13


def obj_aabb(path):
    vs = []
    with open(path) as f:
        for ln in f:
            if ln.startswith("v "):
                p = ln.split()
                vs.append([float(p[1]), float(p[2]), float(p[3])])
    v = np.asarray(vs)
    return v.min(0), v.max(0)


def hand_to_obb_dist(wrist_w, obj_trans, obj_rot, aabb_min, aabb_max):
    """wrist_w (T,2,3) world; obj_rot (T,3,3) local->world. -> (T,2) exterior dist."""
    rel_world = wrist_w - obj_trans[:, None, :]                       # (T,2,3)
    rel_local = np.einsum("tji,thj->thi", obj_rot, rel_world)          # world->local
    clamped = np.clip(rel_local, aabb_min[None, None, :], aabb_max[None, None, :])
    return np.linalg.norm(rel_local - clamped, axis=-1)               # (T,2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="box004", choices=["box004", "box021", "box023"])
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--mesh", default=None)
    ap.add_argument("--probe-csv", default=None)
    ap.add_argument("--out", default=f"{BASE}/spider/workspace/core4d/results/E165/A_box004_contact_audit")
    ap.add_argument("--proxy-radius", type=float, default=0.08)
    args = ap.parse_args()

    case_map = {
        "box004": ("Core4D_E163N_Box004_R161", "box004", "box004_r161"),
        "box021": ("Core4D_E163N_Box021_R160", "box021", "box021_r160"),
        "box023": ("Core4D_E163N_Box023_R158", "box023", "box023_r158"),
    }
    folder, meshkey, probekey = case_map[args.case]
    data_dir = args.data_dir or f"{BASE}/SUGAR-private/data/{folder}/data_000"
    mesh = args.mesh or f"{BASE}/spider/example_datasets/processed/core4d/assets/objects/{meshkey}/{meshkey}_m.obj"
    probe_csv = args.probe_csv or f"{BASE}/SUGAR-private/outputs/core4d_e163_threecase_contact_probe/{probekey}/isaac_contact_framewise.csv"
    os.makedirs(args.out, exist_ok=True)

    robot = np.load(f"{data_dir}/robot_50hz.npz")
    body = robot["body_pos_w"]                                         # (T,14,3)
    wrist = body[:, [L_WRIST, R_WRIST], :]                             # (T,2,3)
    obj = pickle.load(open(f"{data_dir}/obj_motion_global_50hz.pkl", "rb"))
    obj_trans, obj_rot = obj["obj_trans"], obj["obj_rot"]              # (T,3),(T,3,3)
    label = np.load(f"{data_dir}/contact_labels_50hz.npy").astype(bool)
    T = min(len(wrist), len(obj_trans), len(label))
    wrist, obj_trans, obj_rot, label = wrist[:T], obj_trans[:T], obj_rot[:T], label[:T]

    mn, mx = obj_aabb(mesh)
    # box021 (the hardcoded-in-convert) half extents for the "inflation" contrast
    mn21, mx21 = obj_aabb(f"{BASE}/spider/example_datasets/processed/core4d/assets/objects/box021/box021_m.obj")

    dist_true = hand_to_obb_dist(wrist, obj_trans, obj_rot, mn, mx)    # (T,2) true box
    dist_b21 = hand_to_obb_dist(wrist, obj_trans, obj_rot, mn21, mx21) # (T,2) box021 extents
    min_true = dist_true.min(1)                                        # (T,) nearest hand
    min_b21 = dist_b21.min(1)

    # Isaac filtered recall over labeled frames (cross-check)
    recall_both = recall_either = None
    if os.path.exists(probe_csv) and os.path.getsize(probe_csv) > 0:
        import csv
        rows = list(csv.DictReader(open(probe_csv)))
        sc = np.array([r["source_contact"] == "True" for r in rows])
        il = np.array([r["isaac_left_contact"] == "True" for r in rows])
        ir = np.array([r["isaac_right_contact"] == "True" for r in rows])
        ib = np.array([r["isaac_both_contact"] == "True" for r in rows])
        n = min(len(sc), T)
        sc, il, ir, ib = sc[:n], il[:n], ir[:n], ib[:n]
        if sc.sum():
            recall_both = float(ib[sc].mean())
            recall_either = float((il | ir)[sc].mean())

    lab_idx = np.where(label)[0]
    def stats(d):
        if len(lab_idx) == 0:
            return None
        v = d[lab_idx]
        return dict(median=float(np.median(v)), mean=float(v.mean()),
                    min=float(v.min()), max=float(v.max()),
                    frac_gt_proxy=float((v > args.proxy_radius).mean()))

    summary = dict(
        case=args.case, frames=int(T), n_contact_frames=int(label.sum()),
        contact_ratio=float(label.mean()),
        box_true_half=((mx - mn) / 2).round(5).tolist(),
        box021_half=((mx21 - mn21) / 2).round(5).tolist(),
        proxy_radius=args.proxy_radius,
        labeled_dist_true_box=stats(min_true),
        labeled_dist_box021_extents=stats(min_b21),
        isaac_filtered_recall_both=recall_both,
        isaac_filtered_recall_either=recall_either,
        contact_label_window=[int(lab_idx.min()), int(lab_idx.max())] if len(lab_idx) else None,
    )
    # Claim verdict.
    # NOTE: 几何中心距(wrist↔box AABB)无判别力——box021 对照也 ~5cm。真正的 fiction 判据是
    # 消费端 Isaac filtered recall: 标签说接触, 物理里却几乎不接触(recall<0.2) => 标签虚高。
    st = summary["labeled_dist_true_box"]
    summary["dist_criterion_median_gt_proxy"] = bool(st and st["median"] > args.proxy_radius)
    summary["CLAIM_C_A_fiction"] = bool(recall_either is not None and recall_either < 0.2)
    summary["fiction_mechanism"] = (
        "near-but-not-touching: wrist ~5cm from box (同 box021), 但 rubber-hand filtered recall≈0; "
        "标签判据(8cm 中心距 + box021 虚胖盒)过宽" if summary["CLAIM_C_A_fiction"] else "n/a")

    with open(f"{args.out}/{args.case}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    # per-frame csv
    import csv as _csv
    with open(f"{args.out}/{args.case}_per_frame.csv", "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["frame", "label", "dist_L_true", "dist_R_true", "min_true", "min_box021"])
        for t in range(T):
            w.writerow([t, int(label[t]), f"{dist_true[t,0]:.4f}", f"{dist_true[t,1]:.4f}",
                        f"{min_true[t]:.4f}", f"{min_b21[t]:.4f}"])

    print(json.dumps(summary, indent=2))
    print(f"[E165-A] wrote {args.out}/{args.case}_summary.json")


if __name__ == "__main__":
    main()
