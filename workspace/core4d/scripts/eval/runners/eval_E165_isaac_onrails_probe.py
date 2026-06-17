#!/usr/bin/env python3
"""E165-E1: Isaac on-rails 运动学接触探针 -> handoff preflight 三标量 (训练-free, 离线).

物体保持 on-rails(按 ref 驱动), 只量接触几何 (杠杆1, 不释放物体/不考虑 free-joint z)。
复用已有 probe 输出: isaac_contact_framewise.csv。

三个互不合成的标量:
- filtered_contact_recall : 源接触帧里 Isaac filtered(Obj) 命中率 -> 抓 box004 型"标签虚高"
- phantom_force_rate      : 非源接触帧里 net 力 >阈值 占比     -> 抓 box023 型"穿透/打错物体"
- max_init_net_force      : 前 init_frames 帧的 max net 力     -> 抓 box023 型"初始穿透"
排序对照下游 staggered 完成率 (box021 0.67 / box004 0.48 / box023 0.00)。
"""
from __future__ import annotations
import argparse, csv, json, os
import numpy as np

BASE = "/mnt/public/usr/yancilin/work_dir/embodied"
STAGGERED = {"box021": 0.67, "box004": 0.48, "box023": 0.00}


def load_csv(path):
    rows = list(csv.DictReader(open(path)))
    g = lambda k: np.array([r[k] for r in rows])
    b = lambda k: g(k) == "True"
    f = lambda k: g(k).astype(float)
    return dict(
        source=b("source_contact"),
        l=b("isaac_left_contact"), r=b("isaac_right_contact"), both=b("isaac_both_contact"),
        lnet=f("left_net_force_n"), rnet=f("right_net_force_n"),
        lf=f("left_force_n"), rf=f("right_force_n"),
    )


def probe_scalars(d, net_thresh=0.1, init_frames=10):
    sc = d["source"]
    net = np.maximum(d["lnet"], d["rnet"])
    recall_both = float(d["both"][sc].mean()) if sc.sum() else None
    recall_either = float((d["l"] | d["r"])[sc].mean()) if sc.sum() else None
    noncontact = ~sc
    phantom = float((net[noncontact] > net_thresh).mean()) if noncontact.sum() else 0.0
    k = min(init_frames, len(net))
    return dict(
        n_frames=int(len(sc)), n_source_contact=int(sc.sum()),
        filtered_contact_recall_both=recall_both,
        filtered_contact_recall_either=recall_either,
        phantom_force_rate=phantom,
        max_init_net_force=float(net[:k].max()),
        max_net_force=float(net.max()),
        mean_filtered_force_contact=float(np.maximum(d["lf"], d["rf"])[sc].mean()) if sc.sum() else None,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe-root", default=f"{BASE}/SUGAR-private/outputs/core4d_e163_threecase_contact_probe")
    ap.add_argument("--out", default=f"{BASE}/spider/workspace/core4d/results/E165/E1_onrails_probe")
    ap.add_argument("--net-thresh", type=float, default=0.1)
    ap.add_argument("--init-frames", type=int, default=10)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    cases = {"box021": "box021_r160", "box004": "box004_r161", "box023": "box023_r158"}
    res = {}
    for case, sub in cases.items():
        p = f"{args.probe_root}/{sub}/isaac_contact_framewise.csv"
        if not (os.path.exists(p) and os.path.getsize(p) > 0):
            res[case] = {"error": f"missing {p}"}; continue
        s = probe_scalars(load_csv(p), args.net_thresh, args.init_frames)
        s["staggered_success"] = STAGGERED.get(case)
        # joint diagnosis from the 3 orthogonal scalars (single scalar does NOT predict downstream)
        recall = s["filtered_contact_recall_either"] or 0
        if s["max_init_net_force"] > 1000:
            s["mode"] = "init_penetration/self-collision (box023型)"
        elif recall < 0.2:
            s["mode"] = "label_fiction/near-but-not-touching (box004型)"
        else:
            s["mode"] = "clean_transfer (box021型)"
        res[case] = s

    # consistency check: does recall ranking align with staggered, does init-net flag box023
    valid = {c: v for c, v in res.items() if "error" not in v}
    order_recall = sorted(valid, key=lambda c: valid[c]["filtered_contact_recall_either"] or 0, reverse=True)
    order_stag = sorted(valid, key=lambda c: valid[c]["staggered_success"] or 0, reverse=True)
    out = dict(
        cases=res,
        modes={c: valid[c]["mode"] for c in valid},
        recall_order=order_recall, staggered_order=order_stag,
        # 单标量(recall)不预测下游顺序 —— 这是预期: box023 recall 高却 0/64。
        # 结论是"三标量联合分病", 不是"某一标量排序=下游排序"。
        note_single_scalar_insufficient=order_recall != order_stag,
        box023_init_net_flag=bool(valid.get("box023", {}).get("max_init_net_force", 0) > 1000),
        box021_box004_init_net_clean=bool(
            (valid.get("box021", {}).get("max_init_net_force", 1e9) < 100) and
            (valid.get("box004", {}).get("max_init_net_force", 1e9) < 100)),
        three_scalar_modes_distinct=len(set(valid[c]["mode"] for c in valid)) == 3,
    )
    with open(f"{args.out}/onrails_probe_scalars.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    print(f"[E165-E1] wrote {args.out}/onrails_probe_scalars.json")


if __name__ == "__main__":
    main()
