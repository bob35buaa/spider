"""E098 pelvis + lie-on-box CEM gate replay.

新 gate（exp_diagnostic_v2 Part 4 + 整体计划 Stage 0 C5/C6）：
- 三个独立信号：
  - **pelvis_low**：episode 最后 10% 帧 pelvis_z 平均 < 0.40 m（瞬时蹲下不算，"未站起来" 才算）
  - **pelvis_tilt_fall**：max pelvis_tilt > 90°（真摔倒）OR 最后 10% 帧 mean tilt > 55°（最终仍前倾）
  - **lie_on_box**：≥ 30% 帧的 torso_z - box_top_z < 0.10 m（持续贴箱，不是瞬时擦边）
- 单帧 tilt 60-70° 在正常 squat-lift 是正常的，不会触发。

本脚本不改 SPIDER CEM 在线流程，而是离线 replay：吃一个 qpos 序列 + scene.xml，
逐帧 mj_forward 取 pelvis / torso / object 状态，输出 per-case gate verdict。

用途：
1. back-test 历史 rollout（如 E094 C2 趴箱、E090 S1 pelvis 塌、E096b WORK）
2. E101 full CEM 后批量分类失败模式

输出 TSV schema:
  case, T, pelvis_min_z, pelvis_tilt_max_deg, lie_on_box_frac,
  gate_pelvis_low, gate_pelvis_tilt, gate_lie_on_box, gate_overall_fail

使用:
  .venv/bin/python workspace/core4d/scripts/E098/replay_gate.py \\
      --qpos <path>.npz --scene <path>.xml --case <name> --out <results_dir>

或批量:
  .venv/bin/python workspace/core4d/scripts/E098/replay_gate.py --batch <tsv>
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import mujoco
import numpy as np

PELVIS_BODY = "pelvis"
TORSO_BODY = "torso_link"
OBJECT_BODY = "object"
OBJECT_GEOM = "object_collision"

PELVIS_END_Z_THRESH = 0.55  # m: episode 最后 10% 帧 pelvis_z 均值低于此 → 没站起来
PELVIS_TILT_END_THRESH_DEG = 75.0  # episode 最后 10% 帧 mean tilt > 75° → 倒伏/趴
LIE_ON_BOX_GAP_THRESH = 0.10  # m
LIE_ON_BOX_FRAC_THRESH = 0.30  # ≥30% 帧贴箱 → 触发
END_FRAC = 0.10  # 取最后 10% 帧

# 备注：删除原 pelvis_tilt_max 单帧阈值——正常 squat-lift 在弯腰那帧可达
# ~100°，没意义。pelvis_tilt_end 与 pelvis_end_z 联合已能区分 work/fall。


def _body_id(model, name: str) -> int:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if bid < 0:
        raise ValueError(f"body {name!r} not found in scene")
    return bid


def _geom_size(model, name: str) -> np.ndarray:
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
    if gid < 0:
        raise ValueError(f"geom {name!r} not found")
    return model.geom_size[gid, :3].astype(np.float64)


def _pelvis_tilt_deg(quat_wxyz: np.ndarray) -> float:
    """Body up vector (local +Z) 与世界 +Z 的夹角，度。

    不受 yaw 影响：直立 = 0°；横躺 = 90°；倒立 = 180°。
    用 R[2,2] = 1 - 2(x²+y²) = body_up.dot(world_up)。
    """
    w, x, y, z = float(quat_wxyz[0]), float(quat_wxyz[1]), float(quat_wxyz[2]), float(quat_wxyz[3])
    cos_tilt = 1.0 - 2.0 * (x * x + y * y)
    cos_tilt = max(-1.0, min(1.0, cos_tilt))
    return float(np.degrees(np.arccos(cos_tilt)))


def _rotated_box_top_z(obj_pos_world: np.ndarray, obj_quat_wxyz: np.ndarray, half: np.ndarray) -> float:
    """8 个角点在 world frame 下的 max z（box 物理 top）。"""
    # 8 corners in local frame
    signs = np.array(
        [
            [s0, s1, s2]
            for s0 in (-1, 1)
            for s1 in (-1, 1)
            for s2 in (-1, 1)
        ],
        dtype=np.float64,
    )
    corners_local = signs * half[None, :]
    # rotate by quat: world = q * local * q^-1; 用 mujoco.mju_rotVecQuat
    corners_world_z: list[float] = []
    for c in corners_local:
        out = np.zeros(3)
        mujoco.mju_rotVecQuat(out, c, obj_quat_wxyz)
        corners_world_z.append(float(out[2] + obj_pos_world[2]))
    return max(corners_world_z)


def evaluate(qpos: np.ndarray, scene_xml: Path) -> dict:
    """对 qpos 序列跑 gate。返回 dict (含 per-frame array 与汇总)。"""
    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    data = mujoco.MjData(model)
    pelvis_id = _body_id(model, PELVIS_BODY)
    torso_id = _body_id(model, TORSO_BODY)
    object_id = _body_id(model, OBJECT_BODY)
    obj_half = _geom_size(model, OBJECT_GEOM)

    T = qpos.shape[0]
    assert qpos.shape[1] == model.nq, f"qpos ncol={qpos.shape[1]} != model.nq={model.nq}"

    pelvis_z = np.zeros(T)
    pelvis_tilt = np.zeros(T)
    torso_z = np.zeros(T)
    box_top_z = np.zeros(T)

    for i in range(T):
        data.qpos[:] = qpos[i]
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        pelvis_z[i] = float(data.xpos[pelvis_id, 2])
        pelvis_tilt[i] = _pelvis_tilt_deg(data.xquat[pelvis_id])
        torso_z[i] = float(data.xpos[torso_id, 2])
        box_top_z[i] = _rotated_box_top_z(
            data.xpos[object_id], data.xquat[object_id], obj_half
        )

    gap_torso_box = torso_z - box_top_z  # >0 means torso above box top

    n_end = max(1, int(T * END_FRAC))
    pelvis_min_z = float(pelvis_z.min())
    pelvis_end_z = float(pelvis_z[-n_end:].mean())
    pelvis_tilt_max_deg = float(np.max(np.abs(pelvis_tilt)))
    pelvis_tilt_end_deg = float(np.mean(np.abs(pelvis_tilt[-n_end:])))
    lie_frac = float(np.mean(gap_torso_box < LIE_ON_BOX_GAP_THRESH))

    gate_pelvis_low = pelvis_end_z < PELVIS_END_Z_THRESH
    gate_pelvis_tilt = pelvis_tilt_end_deg > PELVIS_TILT_END_THRESH_DEG
    gate_lie_on_box = lie_frac >= LIE_ON_BOX_FRAC_THRESH

    return {
        "T": T,
        "pelvis_min_z": pelvis_min_z,
        "pelvis_end_z": pelvis_end_z,
        "pelvis_tilt_max_deg": pelvis_tilt_max_deg,
        "pelvis_tilt_end_deg": pelvis_tilt_end_deg,
        "lie_on_box_frac": lie_frac,
        "gate_pelvis_low": bool(gate_pelvis_low),
        "gate_pelvis_tilt": bool(gate_pelvis_tilt),
        "gate_lie_on_box": bool(gate_lie_on_box),
        "gate_overall_fail": bool(gate_pelvis_low or gate_pelvis_tilt or gate_lie_on_box),
        "per_frame": {
            "pelvis_z": pelvis_z.tolist(),
            "pelvis_tilt_deg": pelvis_tilt.tolist(),
            "torso_z": torso_z.tolist(),
            "box_top_z": box_top_z.tolist(),
            "gap_torso_box": gap_torso_box.tolist(),
        },
    }


def load_qpos(qpos_path: Path, channel: str = "sim") -> np.ndarray:
    """从 .npz / .npy 加载 qpos (T, nq)。

    支持 SPIDER 既有约定：
    - kinematic ref npz ``qpos`` shape=(T, nq)
    - CEM rollout npz ``qpos`` shape=(T, 2, nq)，[:,0,:]=sim、[:,1,:]=ref；
      ``channel="sim"`` 取 sim（默认）、``"ref"`` 取 ref
    """
    if qpos_path.suffix == ".npy":
        arr = np.load(qpos_path)
    else:
        data = np.load(qpos_path, allow_pickle=True)
        if "qpos" in data:
            arr = data["qpos"]
        elif "qpos_seq" in data:
            arr = data["qpos_seq"]
        else:
            raise ValueError(f"{qpos_path} no qpos/qpos_seq key; keys={list(data.keys())}")
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim == 3 and arr.shape[1] == 2:
        idx = 0 if channel == "sim" else 1
        arr = arr[:, idx, :]
    if arr.ndim != 2:
        raise ValueError(f"qpos must be (T,nq) or (T,2,nq), got {arr.shape}")
    return arr


def run_single(case: str, qpos_path: Path, scene: Path, out_dir: Path) -> dict:
    qpos = load_qpos(qpos_path)
    result = evaluate(qpos, scene)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{case}_gate.json"
    with json_path.open("w") as f:
        json.dump(
            {
                "case": case,
                "qpos_path": str(qpos_path),
                "scene_xml": str(scene),
                **{k: v for k, v in result.items() if k != "per_frame"},
            },
            f,
            indent=2,
        )
    # 单独存 per-frame npz
    pf_path = out_dir / f"{case}_per_frame.npz"
    np.savez(pf_path, **{k: np.asarray(v) for k, v in result["per_frame"].items()})
    return {
        "case": case,
        **{k: v for k, v in result.items() if k != "per_frame"},
        "json": str(json_path),
        "per_frame_npz": str(pf_path),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qpos", type=Path)
    ap.add_argument("--scene", type=Path)
    ap.add_argument("--case", type=str)
    ap.add_argument("--out", type=Path, default=Path("workspace/core4d/results/E098/gate_replay"))
    ap.add_argument("--batch", type=Path, help="TSV with columns: case, qpos_path, scene_xml")
    args = ap.parse_args()

    summary_rows: list[dict] = []
    if args.batch is not None:
        with args.batch.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                case = row["case"]
                qpos = Path(row["qpos_path"])
                scene = Path(row["scene_xml"])
                if not qpos.is_file() or not scene.is_file():
                    print(f"[SKIP] {case}: missing files")
                    summary_rows.append(
                        {
                            "case": case,
                            "T": -1,
                            "pelvis_min_z": float("nan"),
                            "pelvis_end_z": float("nan"),
                            "pelvis_tilt_max_deg": float("nan"),
                            "pelvis_tilt_end_deg": float("nan"),
                            "lie_on_box_frac": float("nan"),
                            "gate_pelvis_low": "skipped",
                            "gate_pelvis_tilt": "skipped",
                            "gate_lie_on_box": "skipped",
                            "gate_overall_fail": "skipped",
                            "json": "",
                            "per_frame_npz": "",
                        }
                    )
                    continue
                print(f"[RUN] {case}")
                summary_rows.append(run_single(case, qpos, scene, args.out))
    elif args.qpos is not None and args.scene is not None and args.case is not None:
        summary_rows.append(run_single(args.case, args.qpos, args.scene, args.out))
    else:
        ap.error("either --batch or all of --qpos/--scene/--case must be supplied")

    # 写汇总 TSV
    summary_tsv = args.out / "summary.tsv"
    args.out.mkdir(parents=True, exist_ok=True)
    fields = [
        "case", "T",
        "pelvis_min_z", "pelvis_end_z",
        "pelvis_tilt_max_deg", "pelvis_tilt_end_deg",
        "lie_on_box_frac",
        "gate_pelvis_low", "gate_pelvis_tilt", "gate_lie_on_box", "gate_overall_fail",
        "json", "per_frame_npz",
    ]
    with summary_tsv.open("w", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        w.writeheader()
        for r in summary_rows:
            w.writerow({k: r.get(k, "") for k in fields})
    print(f"summary -> {summary_tsv}")


if __name__ == "__main__":
    main()
