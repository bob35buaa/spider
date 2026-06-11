"""E099 quat 普查：扫全 20 case obj quat 偏角 vs identity (0,0,0,1)。

输出：
- quat_mean_deg: 全程平均偏角（degrees from identity quat about *some* axis）
- quat_max_deg: 全程最大偏角
- disable_world_up: True if quat_mean_deg > 30°（v2 §3 box021 D003 box quat 90° 旋转的典型阈值）

quat 来自 spider 仓库 trajectory_kinematic.npz 的 qpos（freejoint quat = qpos[obj_qadr+3 : obj_qadr+7]）。
为了与 spider 仓库一致，**不读 raw CORE4D obj poses**，直接读 processed 后的 trajectory_kinematic.npz。
这样可以发现 OmniRetarget 处理后的 quat（保留原始旋转）是否需要绕过 world-up 投影。
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
from case_to_raw import parse_case  # noqa: E402

IDENTITY = np.array([1.0, 0.0, 0.0, 0.0])  # wxyz


def quat_angle_deg(q: np.ndarray) -> float:
    """与 identity 的最短旋转角（degrees）。"""
    w = float(np.clip(abs(q[0]), 0.0, 1.0))
    return float(np.degrees(2.0 * np.arccos(w)))


def audit_case(case: str, base: Path) -> dict:
    """对一个 case 跑 quat audit."""
    spec = parse_case(case)
    case_dir = base / case
    scene_xml = case_dir / "scene.xml"
    traj = case_dir / "0" / "trajectory_kinematic.npz"
    if not scene_xml.is_file() or not traj.is_file():
        return {"case": case, "status": "missing_traj", "scene": str(scene_xml), "traj": str(traj)}

    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    obj_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    if obj_body < 0:
        return {"case": case, "status": "no_object_body"}
    obj_jnt = model.body_jntadr[obj_body]
    obj_qadr = int(model.jnt_qposadr[obj_jnt])

    data = np.load(traj, allow_pickle=True)
    qpos = data["qpos"]  # (T, nq)
    T = qpos.shape[0]
    quats = qpos[:, obj_qadr + 3 : obj_qadr + 7]  # (T, 4) wxyz

    angles = np.array([quat_angle_deg(quats[t]) for t in range(T)])
    mean_deg = float(angles.mean())
    max_deg = float(angles.max())

    return {
        "case": case,
        "status": "ok",
        "T": int(T),
        "obj_name": spec.obj_name,
        "quat_mean_deg": mean_deg,
        "quat_max_deg": max_deg,
        "disable_world_up": mean_deg > 30.0,
        "first_quat": [float(x) for x in quats[0]],
        "last_quat": [float(x) for x in quats[-1]],
    }


def read_manifest(path: Path) -> list[str]:
    cases = []
    with path.open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            cases.append(row["case_name"])
    return cases


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--manifest",
        type=Path,
        default=Path("workspace/core4d/scripts/E098/historical_case_manifest.tsv"),
    )
    ap.add_argument(
        "--base",
        type=Path,
        default=Path("example_datasets/processed/core4d/unitree_g1/humanoid_object"),
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("workspace/core4d/results/E099/quat_audit.tsv"),
    )
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for case in read_manifest(args.manifest):
        print(f"[QUAT] {case}", flush=True)
        try:
            res = audit_case(case, args.base)
        except Exception as e:
            print(f"  EXCEPTION: {e}")
            res = {"case": case, "status": "exception", "error": str(e)}
        if res["status"] != "ok":
            print(f"  [SKIP] {res['status']}")
            rows.append({
                "case": case, "status": res["status"],
                "obj": "", "T": "",
                "quat_mean_deg": "", "quat_max_deg": "",
                "disable_world_up": "", "first_quat": "", "last_quat": "",
            })
            continue
        rows.append({
            "case": case, "status": "ok",
            "obj": res["obj_name"], "T": res["T"],
            "quat_mean_deg": f"{res['quat_mean_deg']:.2f}",
            "quat_max_deg": f"{res['quat_max_deg']:.2f}",
            "disable_world_up": str(res["disable_world_up"]),
            "first_quat": json.dumps(res["first_quat"]),
            "last_quat": json.dumps(res["last_quat"]),
        })
        print(f"  mean={res['quat_mean_deg']:6.2f}°  max={res['quat_max_deg']:6.2f}°"
              f"  disable_world_up={res['disable_world_up']}")

    with args.out.open("w") as f:
        w = csv.DictWriter(
            f, fieldnames=["case", "status", "obj", "T", "quat_mean_deg",
                           "quat_max_deg", "disable_world_up", "first_quat", "last_quat"],
            delimiter="\t",
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nsummary -> {args.out}")


if __name__ == "__main__":
    main()
