"""E099 fingertip face vote helper：raw CORE4D 10 指尖 → obj local → face vote。

核心步骤：
1. 读 raw CORE4D person*_poses.npz: joints (T, 127, 3) Y-up
2. 提 L/R 5 指尖: L 27/30/33/36/39, R 42/45/48/51/54
3. 读 smooth_objposes.npy: (T, 4, 4) Y-up
4. 同步 Y-up → Z-up: x'=x, y'=-z, z'=y（与 convert_core4d_to_omniretarget 一致）
5. 每帧每只手把 5 指尖投到 obj local frame，与 E098 face_utils.face_label 一致
6. 多数面投票

注意：本 helper 不依赖 OmniRetarget 输出（不读 trajectory_kinematic.npz），
即可在 raw mocap 层面给出"如果当时启用了 --include_fingertip_centers 应该是哪个面"的答案。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(THIS_DIR.parent / "E098"))

import face_utils as fu  # noqa: E402
from case_to_raw import parse_case  # noqa: E402

LEFT_FINGERTIP_INDICES = [27, 30, 33, 36, 39]
RIGHT_FINGERTIP_INDICES = [42, 45, 48, 51, 54]

# CORE4D Y-up → Z-up 转换（与 convert_core4d_to_omniretarget.yup_to_zup 一致）
def yup_to_zup_points(p: np.ndarray) -> np.ndarray:
    out = np.empty_like(p)
    out[..., 0] = p[..., 0]
    out[..., 1] = -p[..., 2]
    out[..., 2] = p[..., 1]
    return out


def yup_to_zup_T(T: np.ndarray) -> np.ndarray:
    """(N, 4, 4) Y-up → Z-up，与 holosoma 实现一致。"""
    R_conv = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)
    out = np.empty_like(T)
    for i in range(T.shape[0]):
        R = T[i, :3, :3]
        t = T[i, :3, 3]
        out[i, :3, :3] = R_conv @ R
        out[i, :3, 3] = R_conv @ t
        out[i, 3, :] = [0, 0, 0, 1]
    return out


def world_to_local_points(points_world: np.ndarray, T_world_obj: np.ndarray) -> np.ndarray:
    """(N, K, 3) points + (N, 4, 4) world←obj → (N, K, 3) in obj local frame.

    p_local = R^T @ (p_world - t)
    """
    N, K, _ = points_world.shape
    out = np.empty_like(points_world)
    for i in range(N):
        R = T_world_obj[i, :3, :3]
        t = T_world_obj[i, :3, 3]
        rel = points_world[i] - t  # (K, 3)
        out[i] = rel @ R  # (K, 3) = rel @ R = R^T @ rel.T then transposed back
    return out


def get_box_half_extents(obj_name: str) -> np.ndarray:
    """从 spider 仓库 scene metadata 拿 box 半轴长度。fallback: 从 case task_info.json 推。"""
    # 直接 hardcode 已知 box 半轴（来自 manifest task_info.json 多次校对）
    KNOWN = {
        "Box021": (0.1596, 0.2089, 0.2647),  # 来自 box021_person1/task_info.json (E018b)
        "Box023": (0.1531, 0.1568, 0.1196),  # 待校对
        "Box025": (0.135, 0.135, 0.135),
        "Box022": (0.20, 0.25, 0.25),  # 估值
        "Box026": (0.31449, 0.19725, 0.23449),  # 来自 box026_person2/task_info.json
        "Box004": (0.13, 0.265, 0.305),  # 来自 e091_box004_* task_info.json
    }
    if obj_name in KNOWN:
        return np.array(KNOWN[obj_name], dtype=np.float64)
    raise ValueError(f"unknown obj_name: {obj_name}")


def load_box_half_from_scene(scene_xml: Path, geom_name: str = "object_collision") -> np.ndarray:
    """从 MuJoCo scene.xml 读 box 半轴。优先于 hardcode。"""
    import mujoco
    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
    if gid < 0:
        raise ValueError(f"no geom '{geom_name}' in {scene_xml}")
    return model.geom_size[gid, :3].astype(np.float64)


def vote_case(case: str, scene_xml: Path | None = None, contact_thresh: float = 0.02) -> dict:
    """对一个 case 跑 fingertip face vote。

    Args:
        case: case name
        scene_xml: 可选；若提供则从 scene.xml 读 half（最准），否则用 hardcode KNOWN
        contact_thresh: 指尖到 box 表面距离阈值（米），≤ 阈值才计入投票

    Returns:
        {case, status, hand_L/R: {vote_face: ..., vote_frac, total_frames, contact_frames, counts}}
    """
    spec = parse_case(case)
    if spec.note.startswith("SKIP"):
        return {"case": case, "status": "skip", "reason": spec.note}
    if not spec.raw_ok():
        return {"case": case, "status": "missing_raw",
                "person_npz": str(spec.person_npz), "obj_poses": str(spec.obj_poses_npy)}

    # 1. 读 raw mocap
    data = np.load(spec.person_npz, allow_pickle=True)["arr_0"].item()
    joints_all = data["joints"]  # (T, 127, 3) Y-up
    T = joints_all.shape[0]
    L_tips_yup = joints_all[:, LEFT_FINGERTIP_INDICES, :]  # (T, 5, 3)
    R_tips_yup = joints_all[:, RIGHT_FINGERTIP_INDICES, :]
    # Y-up → Z-up
    L_tips_zup = yup_to_zup_points(L_tips_yup)
    R_tips_zup = yup_to_zup_points(R_tips_yup)

    # 2. 读 obj pose
    T_obj_yup = np.load(spec.obj_poses_npy)  # (T_obj, 4, 4) Y-up
    T_obj_zup = yup_to_zup_T(T_obj_yup)
    # 帧数应一致（CORE4D aligned）
    if T_obj_zup.shape[0] != T:
        print(f"  [WARN] T mismatch joints={T} obj={T_obj_zup.shape[0]}, truncating to min")
        T = min(T, T_obj_zup.shape[0])
        L_tips_zup = L_tips_zup[:T]
        R_tips_zup = R_tips_zup[:T]
        T_obj_zup = T_obj_zup[:T]

    # 3. world → obj local
    L_local = world_to_local_points(L_tips_zup, T_obj_zup)  # (T, 5, 3)
    R_local = world_to_local_points(R_tips_zup, T_obj_zup)

    # 4. 拿 box half (优先 scene.xml)
    if scene_xml is not None and scene_xml.is_file():
        half = load_box_half_from_scene(scene_xml)
        half_src = "scene.xml"
    else:
        half = get_box_half_extents(spec.obj_name)
        half_src = "hardcode"

    # 5. per-frame per-hand face vote
    def vote_hand(local_pts: np.ndarray, name: str) -> dict:
        # local_pts (T, 5, 3)
        # 每帧投票面：5 个指尖各算 face_label，多数面胜，若多数面在表面 contact_thresh 内则计入有效帧
        labels_per_frame: list[str] = []
        contact_frames = 0
        for t in range(T):
            tip_labels = []
            for k in range(5):
                p = local_pts[t, k]
                if not np.all(np.isfinite(p)):
                    continue
                tip_labels.append(fu.face_label(p, half))
            if not tip_labels:
                continue
            most = Counter(tip_labels).most_common(1)[0][0]
            # 检查多数面对应的指尖到 face 的距离
            axis = "xyz".index(most[1])
            sign = 1.0 if most[0] == "+" else -1.0
            face_coord = sign * half[axis]
            # 取所有投给 most 的指尖
            min_d = np.inf
            for k in range(5):
                p = local_pts[t, k]
                if not np.all(np.isfinite(p)):
                    continue
                if fu.face_label(p, half) != most:
                    continue
                d = abs(p[axis] - face_coord)
                # 检查 p 是否在 face 外侧延伸（contact_thresh），且 in-plane bbox 内
                others = [i for i in (0, 1, 2) if i != axis]
                in_plane = (
                    abs(p[others[0]]) <= half[others[0]] + 0.05
                    and abs(p[others[1]]) <= half[others[1]] + 0.05
                )
                if in_plane and d <= contact_thresh:
                    min_d = min(min_d, d)
            if min_d < np.inf:
                contact_frames += 1
                labels_per_frame.append(most)
            else:
                labels_per_frame.append(most + "_far")  # 远离任何面，标记 _far

        counts_close = Counter(l for l in labels_per_frame if not l.endswith("_far"))
        counts_far = Counter(l[:-4] for l in labels_per_frame if l.endswith("_far"))
        total_close = sum(counts_close.values())
        if total_close == 0:
            top_face = ""
            top_frac = 0.0
        else:
            top_face, top_count = counts_close.most_common(1)[0]
            top_frac = top_count / total_close
        return {
            "hand": name,
            "total_frames": T,
            "contact_frames": contact_frames,
            "vote_face": top_face,
            "vote_frac": top_frac,
            "counts_close": dict(counts_close),
            "counts_far": dict(counts_far),
        }

    res_L = vote_hand(L_local, "L")
    res_R = vote_hand(R_local, "R")

    return {
        "case": case,
        "status": "ok",
        "T": T,
        "half": [float(x) for x in half],
        "half_source": half_src,
        "obj_name": spec.obj_name,
        "date_seq_person": f"{spec.date}/{spec.seq}/{spec.person}",
        "L": res_L,
        "R": res_R,
    }


def read_manifest(path: Path) -> list[str]:
    """读 historical_case_manifest.tsv 第一列。"""
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
        "--scene-base",
        type=Path,
        default=Path("example_datasets/processed/core4d/unitree_g1/humanoid_object"),
        help="若存在 {case}/scene.xml 则用它的 half；否则用 hardcode",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("workspace/core4d/results/E099/fingertip_face_stats.tsv"),
    )
    ap.add_argument(
        "--json-dir",
        type=Path,
        default=Path("workspace/core4d/results/E099/fingertip_vote_per_case"),
    )
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.json_dir.mkdir(parents=True, exist_ok=True)

    cases = read_manifest(args.manifest)
    rows = []
    for case in cases:
        scene = args.scene_base / case / "scene.xml"
        scene_arg = scene if scene.is_file() else None
        print(f"[VOTE] {case}", flush=True)
        try:
            res = vote_case(case, scene_xml=scene_arg)
        except Exception as e:
            print(f"  EXCEPTION: {e}")
            res = {"case": case, "status": "exception", "error": str(e)}
        json_path = args.json_dir / f"{case}.json"
        with json_path.open("w") as f:
            json.dump(res, f, indent=2, default=str)
        if res["status"] != "ok":
            print(f"  [SKIP] {res['status']}: {res.get('reason') or res.get('error') or ''}")
            rows.append({
                "case": case, "status": res["status"],
                "L_vote": "", "L_frac": "", "L_contact": "",
                "R_vote": "", "R_frac": "", "R_contact": "",
                "T": "", "obj": "", "date_seq_person": "",
            })
            continue
        L, R = res["L"], res["R"]
        rows.append({
            "case": case, "status": "ok",
            "L_vote": L["vote_face"], "L_frac": f"{L['vote_frac']:.3f}",
            "L_contact": L["contact_frames"],
            "R_vote": R["vote_face"], "R_frac": f"{R['vote_frac']:.3f}",
            "R_contact": R["contact_frames"],
            "T": res["T"], "obj": res["obj_name"],
            "date_seq_person": res["date_seq_person"],
        })
        print(
            f"  T={res['T']:3d}  L: {L['vote_face']:>4} {L['vote_frac']*100:5.1f}% (n_contact={L['contact_frames']})"
            f"  R: {R['vote_face']:>4} {R['vote_frac']*100:5.1f}% (n_contact={R['contact_frames']})"
        )

    with args.out.open("w") as f:
        w = csv.DictWriter(
            f, fieldnames=["case", "status", "L_vote", "L_frac", "L_contact",
                           "R_vote", "R_frac", "R_contact", "T", "obj", "date_seq_person"],
            delimiter="\t",
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nsummary -> {args.out}")
    print(f"per-case json -> {args.json_dir}")


if __name__ == "__main__":
    main()
