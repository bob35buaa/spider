"""E099 palm vote 全 case 扩展（复用 E098 anchor_refit_demo 逻辑，但跑全 historical case）。

从 trajectory_kinematic.npz 读 contact_pos (FK palm)，投到 obj local frame，按 E098
face_utils.face_label 投票。输出 palm_face_stats.tsv 用于与 fingertip_face_stats.tsv 对比。
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import mujoco
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR.parent / "E098"))
sys.path.insert(0, str(THIS_DIR))

import face_utils as fu  # noqa: E402
from case_to_raw import parse_case  # noqa: E402


def palm_vote_case(case: str, scene_xml: Path, traj: Path) -> dict:
    spec = parse_case(case)
    if not scene_xml.is_file() or not traj.is_file():
        return {"case": case, "status": "missing_traj"}

    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "object_collision")
    half = model.geom_size[gid, :3].astype(np.float64)
    obj_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    obj_jnt = model.body_jntadr[obj_body]
    obj_qadr = int(model.jnt_qposadr[obj_jnt])

    data = np.load(traj, allow_pickle=True)
    qpos = data["qpos"]
    contact_pos = data.get("contact_pos", None)
    contact = data.get("contact", None)
    if contact_pos is None:
        return {"case": case, "status": "no_contact_pos"}

    mj_data = mujoco.MjData(model)
    labels_L: list[str] = []
    labels_R: list[str] = []
    T = qpos.shape[0]
    for t in range(T):
        mj_data.qpos[:] = qpos[t]
        mujoco.mj_forward(model, mj_data)
        obj_pos = mj_data.qpos[obj_qadr:obj_qadr+3].astype(np.float64)
        obj_quat = mj_data.qpos[obj_qadr+3:obj_qadr+7].astype(np.float64)
        for hand_idx, labels in [(0, labels_L), (1, labels_R)]:
            if contact is not None and not bool(contact[t, hand_idx]):
                continue
            w = np.asarray(contact_pos[t, hand_idx], dtype=np.float64)
            if not np.all(np.isfinite(w)):
                continue
            out = np.zeros(3)
            qinv = obj_quat.copy(); qinv[1:] *= -1
            mujoco.mju_rotVecQuat(out, w - obj_pos, qinv)
            labels.append(fu.face_label(out, half))

    def topvote(labels: list[str]) -> tuple[str, float, int]:
        if not labels:
            return "", 0.0, 0
        c = Counter(labels)
        top, n = c.most_common(1)[0]
        return top, n / len(labels), len(labels)

    L = topvote(labels_L)
    R = topvote(labels_R)
    return {
        "case": case, "status": "ok", "T": int(T),
        "obj": spec.obj_name,
        "L_face": L[0], "L_frac": L[1], "L_count": L[2],
        "R_face": R[0], "R_frac": R[1], "R_count": R[2],
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
        default=Path("workspace/core4d/results/E099/palm_face_stats.tsv"),
    )
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in read_manifest(args.manifest):
        scene = args.base / case / "scene.xml"
        traj = args.base / case / "0" / "trajectory_kinematic.npz"
        print(f"[PALM] {case}", flush=True)
        try:
            r = palm_vote_case(case, scene, traj)
        except Exception as e:
            print(f"  EXCEPTION: {e}")
            r = {"case": case, "status": "exception"}
        if r["status"] != "ok":
            print(f"  [SKIP] {r['status']}")
            rows.append({"case": case, "status": r["status"],
                         "obj": "", "T": "",
                         "L_face": "", "L_frac": "", "L_count": "",
                         "R_face": "", "R_frac": "", "R_count": ""})
            continue
        rows.append({
            "case": case, "status": "ok",
            "obj": r["obj"], "T": r["T"],
            "L_face": r["L_face"], "L_frac": f"{r['L_frac']:.3f}", "L_count": r["L_count"],
            "R_face": r["R_face"], "R_frac": f"{r['R_frac']:.3f}", "R_count": r["R_count"],
        })
        print(f"  L: {r['L_face']:>4} {r['L_frac']*100:5.1f}% n={r['L_count']}"
              f"  R: {r['R_face']:>4} {r['R_frac']*100:5.1f}% n={r['R_count']}")

    with args.out.open("w") as f:
        w = csv.DictWriter(
            f, fieldnames=["case", "status", "obj", "T",
                           "L_face", "L_frac", "L_count",
                           "R_face", "R_frac", "R_count"],
            delimiter="\t",
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nsummary -> {args.out}")


if __name__ == "__main__":
    main()
