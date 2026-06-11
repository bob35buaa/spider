"""E100 build_fingertip_aware_target.py：用 E099 fingertip vote face 重做 contact target。

逻辑（每帧每只手）：
1. 从 spider 仓库 trajectory_kinematic.npz 读 obj pose (qpos) + palm world 位置 (contact_pos)
2. palm_local = R^T (palm_world - obj_pos)
3. 若 E099 fingertip vote 给出 vote_face：
   - 把 vote_face 那个轴 (axis) 的坐标 clip 到 ± half[axis]（即贴 face）
   - in-plane 两轴保留 palm 当前 in-plane 投影，clip 到 ±(half - 1cm) 内（防溢出 box）
4. 若 vote_face 空（IK 过拟合 case，R no_contact）：
   - target = palm_local 不动（等效 ref_fk）

输出 NPZ：key=`spider_contact_target_object_local`，shape (T, 2, 3) in obj local。
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
sys.path.insert(0, str(THIS_DIR.parent / "E098"))
sys.path.insert(0, str(THIS_DIR.parent / "E099"))

# 面内 clip 边距（避免 target 跑到 face 外）
IN_PLANE_MARGIN = 0.01  # 1 cm


def palm_local_per_frame(scene_xml: Path, traj: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """读 spider 仓库 trajectory_kinematic.npz，返回 (T, 2, 3) palm_local + (T,) half + (T,2) active."""
    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "object_collision")
    half = model.geom_size[gid, :3].astype(np.float64)
    obj_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    obj_jnt = model.body_jntadr[obj_body]
    obj_qadr = int(model.jnt_qposadr[obj_jnt])

    d = np.load(traj, allow_pickle=True)
    qpos = d["qpos"]
    contact_pos = d.get("contact_pos", None)
    contact = d.get("contact", None)
    T = qpos.shape[0]
    palm_local = np.zeros((T, 2, 3), dtype=np.float64)
    active = np.zeros((T, 2), dtype=bool)
    palm_world_track = np.zeros((T, 2, 3), dtype=np.float64)

    data = mujoco.MjData(model)
    for t in range(T):
        data.qpos[:] = qpos[t]
        mujoco.mj_forward(model, data)
        obj_pos = data.qpos[obj_qadr:obj_qadr+3].astype(np.float64)
        obj_quat = data.qpos[obj_qadr+3:obj_qadr+7].astype(np.float64)
        for h in (0, 1):
            if contact_pos is None:
                continue
            w = np.asarray(contact_pos[t, h], dtype=np.float64)
            if not np.all(np.isfinite(w)):
                continue
            palm_world_track[t, h] = w
            out = np.zeros(3)
            qinv = obj_quat.copy(); qinv[1:] *= -1
            mujoco.mju_rotVecQuat(out, w - obj_pos, qinv)
            palm_local[t, h] = out
            if contact is not None:
                active[t, h] = bool(contact[t, h])
            else:
                active[t, h] = True
    return palm_local, half, active, palm_world_track


def project_to_face(palm_local: np.ndarray, vote_face: str, half: np.ndarray) -> np.ndarray:
    """把 palm_local 投到 vote_face：vote 轴 = ±half；in-plane 两轴 clip 到 ±(half - margin)."""
    if not vote_face:
        return palm_local.copy()
    axis = "xyz".index(vote_face[1])
    sign = 1.0 if vote_face[0] == "+" else -1.0
    out = palm_local.copy()
    out[axis] = sign * half[axis]
    others = [i for i in (0, 1, 2) if i != axis]
    for o in others:
        lim = max(0.0, half[o] - IN_PLANE_MARGIN)
        out[o] = float(np.clip(palm_local[o], -lim, lim))
    return out


def _palm_top_vote(palm_local: np.ndarray, half: np.ndarray, active: np.ndarray) -> str:
    """palm-based face vote (与 E098 face_utils.face_label 一致)，作 baseline."""
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "E098"))
    import face_utils as fu  # noqa: E402
    from collections import Counter
    labels = []
    for t in range(palm_local.shape[0]):
        if not active[t]:
            continue
        p = palm_local[t]
        if not np.all(np.isfinite(p)):
            continue
        labels.append(fu.face_label(p, half))
    if not labels:
        return ""
    return Counter(labels).most_common(1)[0][0]


def build_target(case: str, scene_xml: Path, traj: Path, vote: dict) -> dict:
    """对一个 case 跑 target 生成 (fingertip-based)，同时算 palm-based 对照."""
    palm_local, half, active, palm_world = palm_local_per_frame(scene_xml, traj)
    T = palm_local.shape[0]
    target = np.zeros_like(palm_local)
    target_palm = np.zeros_like(palm_local)
    L_face = vote.get("L", {}).get("vote_face", "")
    R_face = vote.get("R", {}).get("vote_face", "")
    # palm-based vote face per hand (baseline for 守门 reverse check)
    L_palm_face = _palm_top_vote(palm_local[:, 0], half, active[:, 0])
    R_palm_face = _palm_top_vote(palm_local[:, 1], half, active[:, 1])

    for t in range(T):
        target[t, 0] = project_to_face(palm_local[t, 0], L_face, half)
        target[t, 1] = project_to_face(palm_local[t, 1], R_face, half)
        target_palm[t, 0] = project_to_face(palm_local[t, 0], L_palm_face, half)
        target_palm[t, 1] = project_to_face(palm_local[t, 1], R_palm_face, half)

    # 统计
    # delta vs palm position (反映 target 与 reward source palm 的距离)
    delta_palm_L = target[:, 0] - palm_local[:, 0]
    delta_palm_R = target[:, 1] - palm_local[:, 1]
    # delta vs palm-based target (反映 face 切换的影响 — 守门看这个)
    delta_swap_L = target[:, 0] - target_palm[:, 0]
    delta_swap_R = target[:, 1] - target_palm[:, 1]
    face_changed_L = (L_face != L_palm_face)
    face_changed_R = (R_face != R_palm_face)
    summary = {
        "case": case,
        "T": int(T),
        "half": [float(x) for x in half],
        "vote_L": L_face, "vote_R": R_face,
        "palm_vote_L": L_palm_face, "palm_vote_R": R_palm_face,
        "face_changed_L": face_changed_L, "face_changed_R": face_changed_R,
        "delta_palm_mean_L_m": float(np.linalg.norm(delta_palm_L, axis=1).mean()),
        "delta_palm_max_L_m": float(np.linalg.norm(delta_palm_L, axis=1).max()),
        "delta_palm_mean_R_m": float(np.linalg.norm(delta_palm_R, axis=1).mean()),
        "delta_palm_max_R_m": float(np.linalg.norm(delta_palm_R, axis=1).max()),
        "delta_swap_mean_L_m": float(np.linalg.norm(delta_swap_L, axis=1).mean()),
        "delta_swap_max_L_m": float(np.linalg.norm(delta_swap_L, axis=1).max()),
        "delta_swap_mean_R_m": float(np.linalg.norm(delta_swap_R, axis=1).mean()),
        "delta_swap_max_R_m": float(np.linalg.norm(delta_swap_R, axis=1).max()),
        "active_L_frac": float(active[:, 0].mean()),
        "active_R_frac": float(active[:, 1].mean()),
    }
    return {
        "spider_contact_target_object_local": target.astype(np.float32),
        "palm_local_record": palm_local.astype(np.float32),
        "active": active,
        "summary": summary,
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
        "--fingertip-dir",
        type=Path,
        default=Path("workspace/core4d/results/E099/fingertip_vote_per_case"),
    )
    ap.add_argument(
        "--scene-base",
        type=Path,
        default=Path("example_datasets/processed/core4d/unitree_g1/humanoid_object"),
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("workspace/core4d/results/E100/fingertip_targets"),
    )
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    for case in read_manifest(args.manifest):
        vote_path = args.fingertip_dir / f"{case}.json"
        scene = args.scene_base / case / "scene.xml"
        traj = args.scene_base / case / "0" / "trajectory_kinematic.npz"
        empty_cols = {k: "" for k in [
            "T", "vote_L", "vote_R", "palm_vote_L", "palm_vote_R",
            "face_changed_L", "face_changed_R",
            "delta_palm_mean_L_m", "delta_palm_max_L_m",
            "delta_palm_mean_R_m", "delta_palm_max_R_m",
            "delta_swap_mean_L_m", "delta_swap_max_L_m",
            "delta_swap_mean_R_m", "delta_swap_max_R_m",
            "active_L_frac", "active_R_frac",
        ]}
        if not vote_path.is_file() or not scene.is_file() or not traj.is_file():
            print(f"[SKIP] {case} (missing inputs)")
            summary_rows.append({"case": case, "status": "missing_inputs", **empty_cols})
            continue
        vote = json.loads(vote_path.read_text())
        if vote.get("status") != "ok":
            print(f"[SKIP] {case} (vote status={vote.get('status')})")
            summary_rows.append({"case": case, "status": f"vote_{vote.get('status')}", **empty_cols})
            continue
        print(f"[BUILD] {case}", flush=True)
        try:
            res = build_target(case, scene, traj, vote)
        except Exception as e:
            print(f"  EXCEPTION: {e}")
            summary_rows.append({"case": case, "status": "exception", **empty_cols})
            continue
        case_out = args.out / case
        case_out.mkdir(parents=True, exist_ok=True)
        np.savez(
            case_out / "spider_contact_target_object_local.npz",
            spider_contact_target_object_local=res["spider_contact_target_object_local"],
            palm_local_record=res["palm_local_record"],
            active=res["active"],
        )
        with (case_out / "summary.json").open("w") as f:
            json.dump(res["summary"], f, indent=2)
        s = res["summary"]
        summary_rows.append({
            "case": case, "status": "ok",
            "T": s["T"], "vote_L": s["vote_L"], "vote_R": s["vote_R"],
            "palm_vote_L": s["palm_vote_L"], "palm_vote_R": s["palm_vote_R"],
            "face_changed_L": str(s["face_changed_L"]),
            "face_changed_R": str(s["face_changed_R"]),
            "delta_palm_mean_L_m": f"{s['delta_palm_mean_L_m']:.4f}",
            "delta_palm_max_L_m": f"{s['delta_palm_max_L_m']:.4f}",
            "delta_palm_mean_R_m": f"{s['delta_palm_mean_R_m']:.4f}",
            "delta_palm_max_R_m": f"{s['delta_palm_max_R_m']:.4f}",
            "delta_swap_mean_L_m": f"{s['delta_swap_mean_L_m']:.4f}",
            "delta_swap_max_L_m": f"{s['delta_swap_max_L_m']:.4f}",
            "delta_swap_mean_R_m": f"{s['delta_swap_mean_R_m']:.4f}",
            "delta_swap_max_R_m": f"{s['delta_swap_max_R_m']:.4f}",
            "active_L_frac": f"{s['active_L_frac']:.3f}",
            "active_R_frac": f"{s['active_R_frac']:.3f}",
        })
        print(f"  finger L={s['vote_L']!r}({'≠' if s['face_changed_L'] else '='}palm {s['palm_vote_L']!r}) "
              f"R={s['vote_R']!r}({'≠' if s['face_changed_R'] else '='}palm {s['palm_vote_R']!r}) "
              f"swap ΔL/R mean={s['delta_swap_mean_L_m']*100:.1f}/{s['delta_swap_mean_R_m']*100:.1f}cm")

    tsv = args.out.parent / "target_gap_summary.tsv"
    tsv.parent.mkdir(parents=True, exist_ok=True)
    with tsv.open("w") as f:
        w = csv.DictWriter(
            f, fieldnames=["case", "status", "T", "vote_L", "vote_R",
                           "palm_vote_L", "palm_vote_R",
                           "face_changed_L", "face_changed_R",
                           "delta_palm_mean_L_m", "delta_palm_max_L_m",
                           "delta_palm_mean_R_m", "delta_palm_max_R_m",
                           "delta_swap_mean_L_m", "delta_swap_max_L_m",
                           "delta_swap_mean_R_m", "delta_swap_max_R_m",
                           "active_L_frac", "active_R_frac"],
            delimiter="\t",
        )
        w.writeheader()
        for r in summary_rows:
            w.writerow(r)
    print(f"\nsummary -> {tsv}")


if __name__ == "__main__":
    main()
