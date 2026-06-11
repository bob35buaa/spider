"""E098 anchor refit demo: 在 6 个典型 box021 D003 case 上跑「老 xy-only vs 新全 3D」对比。

读 trajectory_kinematic.npz 的 ``contact_pos``（在 world frame）+ ``qpos``，
把每帧每只手的 contact_pos 投到 object local frame，分别用：

1. **OLD** 算法 (B1 bug)：xy-only argmax，永远不返回 ±z
2. **NEW** 算法 (B1 fixed)：全 3D argmax，可返回 ±z

逐 case 输出主面投票表 + JSON。下游 render_anchor_compare.py 读 JSON 出 mp4。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
import face_utils as fu  # noqa: E402

CASES = [
    "d003_box021_20231018_029_p2",
    "d003_box021_20231011_035_p2",
    "d003_box021_20231020_019_p1",
    "d003_box021_20231018_030_p1",
    "d003_box021_20231020_020_p2",
    "d003_box021_20231018_028_p2",
]


def _old_face_label(point: np.ndarray, half: np.ndarray) -> str:
    """B1 修复前的实现：xy-only argmax，仅 ±x/±y。"""
    xy_norm = np.abs(point[:2]) / np.clip(half[:2], 1e-6, None)
    axis = int(np.argmax(xy_norm))
    sign = "+" if point[axis] >= 0.0 else "-"
    return f"{sign}{'xy'[axis]}"


def _quat_apply_inv(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """world → local，q 为 [w,x,y,z]。"""
    out = np.zeros(3)
    qinv = q.copy()
    qinv[1:] *= -1
    mujoco.mju_rotVecQuat(out, v, qinv)
    return out


def process_case(case: str, base_dir: Path) -> dict:
    case_dir = base_dir / case
    scene = case_dir / "scene.xml"
    traj_path = case_dir / "0" / "trajectory_kinematic.npz"
    if not scene.is_file() or not traj_path.is_file():
        return {"case": case, "status": "missing", "scene": str(scene), "traj": str(traj_path)}

    model = mujoco.MjModel.from_xml_path(str(scene))
    geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "object_collision")
    half = model.geom_size[geom, :3].astype(np.float64)
    obj_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    obj_jnt = model.body_jntadr[obj_body]
    obj_qadr = int(model.jnt_qposadr[obj_jnt])

    data = mujoco.MjData(model)
    traj = np.load(traj_path, allow_pickle=True)
    qpos = traj["qpos"]
    contact_pos = traj["contact_pos"]  # (T, 2, 3) FK palm site
    contact = traj["contact"] if "contact" in traj else None  # (T, 2)

    points_local: list[list[np.ndarray]] = [[], []]  # [L, R]
    old_labels: list[list[str]] = [[], []]
    new_labels: list[list[str]] = [[], []]

    T = qpos.shape[0]
    for i in range(T):
        data.qpos[:] = qpos[i]
        mujoco.mj_forward(model, data)
        obj_pos = data.qpos[obj_qadr : obj_qadr + 3].astype(np.float64)
        obj_quat = data.qpos[obj_qadr + 3 : obj_qadr + 7].astype(np.float64)
        for hand in (0, 1):
            if contact is not None and not bool(contact[i, hand]):
                continue
            world = np.asarray(contact_pos[i, hand], dtype=np.float64)
            if not np.all(np.isfinite(world)):
                continue
            local = _quat_apply_inv(obj_quat, world - obj_pos)
            if not np.all(np.isfinite(local)):
                continue
            points_local[hand].append(local)
            old_labels[hand].append(_old_face_label(local, half))
            new_labels[hand].append(fu.face_label(local, half))

    result = {
        "case": case,
        "status": "ok",
        "scene": str(scene),
        "T": int(T),
        "half": [float(x) for x in half],
    }
    for hand_idx, hand_name in enumerate(["L", "R"]):
        pts = np.asarray(points_local[hand_idx]) if points_local[hand_idx] else np.zeros((0, 3))
        old_lbls = old_labels[hand_idx]
        new_lbls = new_labels[hand_idx]
        # 统计 old vs new 主面
        from collections import Counter

        old_counts = Counter(old_lbls)
        new_counts = Counter(new_lbls)
        old_top = old_counts.most_common(1)[0] if old_counts else ("", 0)
        new_top = new_counts.most_common(1)[0] if new_counts else ("", 0)
        result[hand_name] = {
            "n_contact_frames": len(old_lbls),
            "old_top_face": old_top[0],
            "old_top_count": old_top[1],
            "new_top_face": new_top[0],
            "new_top_count": new_top[1],
            "old_counts": dict(old_counts),
            "new_counts": dict(new_counts),
            "points_local": pts.tolist(),
            "differs": old_top[0] != new_top[0],
        }
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--base",
        type=Path,
        default=Path("example_datasets/processed/core4d/unitree_g1/humanoid_object"),
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("workspace/core4d/results/E098/anchor_refit"),
    )
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict] = []
    for case in CASES:
        print(f"[REFIT] {case}")
        res = process_case(case, args.base)
        out_json = args.out / f"{case}.json"
        with out_json.open("w") as f:
            json.dump(res, f, indent=2)
        if res["status"] != "ok":
            print(f"  SKIP ({res['status']})")
            summary_rows.append({"case": case, "status": res["status"]})
            continue
        for hand in ("L", "R"):
            h = res[hand]
            differs = "DIFFER" if h["differs"] else "same"
            summary_rows.append(
                {
                    "case": case,
                    "hand": hand,
                    "n_contact": h["n_contact_frames"],
                    "old_top": f"{h['old_top_face']}({h['old_top_count']})",
                    "new_top": f"{h['new_top_face']}({h['new_top_count']})",
                    "delta": differs,
                }
            )
            print(
                f"  {hand}: n={h['n_contact_frames']}  "
                f"old_top={h['old_top_face']}({h['old_top_count']})  "
                f"new_top={h['new_top_face']}({h['new_top_count']})  {differs}"
            )

    tsv = args.out / "summary.tsv"
    with tsv.open("w") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["case", "hand", "n_contact", "old_top", "new_top", "delta", "status"],
            delimiter="\t",
        )
        w.writeheader()
        for r in summary_rows:
            w.writerow({k: r.get(k, "") for k in w.fieldnames})
    print(f"\nsummary -> {tsv}")


if __name__ == "__main__":
    main()
