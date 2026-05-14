#!/usr/bin/env python3
"""E070: compare MuJoCo CPU and MJWarp ref-control commit dynamics.

This script intentionally does not run CEM. It reproduces the initial
setup_env() state and then commits ctrl_ref substeps through both MuJoCo CPU and
MJWarp, with and without the object actuator gain restoration used by
examples/run_mjwp.py during the commit phase.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import torch
import warp as wp
from scipy.spatial.transform import Rotation as R

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEBUG_DIR = ROOT / "workspace/core4d/scripts/debug"
if str(DEBUG_DIR) not in sys.path:
    sys.path.insert(0, str(DEBUG_DIR))

from diagnose_E068_init_drift import (  # noqa: E402
    _build_config,
    _convert_scene_act_ref,
    _normalize_angle_deg,
)
from spider.io import load_data  # noqa: E402
from spider.simulators.mjwp import (  # noqa: E402
    get_qpos,
    get_qvel,
    load_env_params,
    setup_env,
    setup_mj_model,
    step_env,
)


def _quat_wxyz_to_yaw_deg(q: np.ndarray) -> float:
    q = np.asarray(q, dtype=np.float64)
    return float(R.from_quat([q[1], q[2], q[3], q[0]]).as_euler("xyz", degrees=True)[2])


def _quat_angle_deg(q: np.ndarray, q_ref: np.ndarray) -> float:
    q = np.asarray(q, dtype=np.float64)
    q_ref = np.asarray(q_ref, dtype=np.float64)
    dot = abs(float(np.dot(q / np.linalg.norm(q), q_ref / np.linalg.norm(q_ref))))
    dot = min(1.0, max(-1.0, dot))
    return math.degrees(2.0 * math.acos(dot))


def _body_id(model: mujoco.MjModel, candidates: list[str]) -> int:
    for name in candidates:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid >= 0:
            return bid
    raise ValueError(f"None of these bodies exist: {candidates}")


def _commit_gain_vectors(config) -> tuple[np.ndarray, np.ndarray]:
    actuator_names = config.object_actuator_names
    if not actuator_names:
        model = mujoco.MjModel.from_xml_path(config.model_path)
        actuator_names = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, int(aid))
            for aid in config.object_actuator_ids
        ]
    kp = np.array(
        [
            config.init_rot_actuator_gain if "_rot_" in (name or "") else config.init_pos_actuator_gain
            for name in actuator_names
        ],
        dtype=np.float32,
    )
    kd = np.array(
        [
            config.init_rot_actuator_bias if "_rot_" in (name or "") else config.init_pos_actuator_bias
            for name in actuator_names
        ],
        dtype=np.float32,
    )
    return kp, kd


def _apply_cpu_object_gains(model: mujoco.MjModel, config, enabled: bool) -> None:
    if not config.object_actuator_ids:
        return
    if enabled:
        kp, kd = _commit_gain_vectors(config)
    else:
        kp = np.zeros((len(config.object_actuator_ids),), dtype=np.float32)
        kd = np.zeros((len(config.object_actuator_ids),), dtype=np.float32)
    ids = np.asarray(config.object_actuator_ids, dtype=int)
    model.actuator_gainprm[ids, 0] = kp
    model.actuator_biasprm[ids, 1] = -kd


def _safe_first_world_norm(env, name: str) -> float:
    if not hasattr(env.data_wp, name):
        return float("nan")
    try:
        arr = wp.to_torch(getattr(env.data_wp, name))[0].detach().cpu().numpy()
        return float(np.linalg.norm(arr))
    except Exception:
        return float("nan")


def _safe_first_world_scalar(env, name: str) -> float:
    if not hasattr(env.data_wp, name):
        return float("nan")
    try:
        arr = wp.to_torch(getattr(env.data_wp, name)).detach().cpu().numpy()
        return float(np.asarray(arr).reshape(-1)[0])
    except Exception:
        return float("nan")


def _measure(
    *,
    path: str,
    phase: str,
    step_idx: int,
    model: mujoco.MjModel,
    qpos: np.ndarray,
    qvel: np.ndarray,
    ctrl: np.ndarray,
    qpos_ref: np.ndarray,
    qvel_ref: np.ndarray,
    ctrl_ref: np.ndarray,
    e069_qpos: np.ndarray | None = None,
    cpu_data: mujoco.MjData | None = None,
    env=None,
) -> dict[str, Any]:
    ref_idx = min(max(step_idx + 1, 0), len(qpos_ref) - 1)
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    data.qvel[:] = qvel
    data.ctrl[:] = ctrl
    mujoco.mj_forward(model, data)

    left_foot = _body_id(model, ["left_ankle_roll_link", "left_foot_link"])
    right_foot = _body_id(model, ["right_ankle_roll_link", "right_foot_link"])
    yaw = _quat_wxyz_to_yaw_deg(qpos[3:7])
    yaw_ref0 = _quat_wxyz_to_yaw_deg(qpos_ref[0, 3:7])
    yaw_ref_idx = _quat_wxyz_to_yaw_deg(qpos_ref[ref_idx, 3:7])
    row: dict[str, Any] = {
        "path": path,
        "phase": phase,
        "step_idx": step_idx,
        "ref_idx": ref_idx,
        "time": float(cpu_data.time if cpu_data is not None else data.time),
        "yaw_deg": yaw,
        "yaw_err_vs_ref0_deg": _normalize_angle_deg(yaw - yaw_ref0),
        "yaw_err_vs_ref_idx_deg": _normalize_angle_deg(yaw - yaw_ref_idx),
        "quat_err_vs_ref0_deg": _quat_angle_deg(qpos[3:7], qpos_ref[0, 3:7]),
        "quat_err_vs_ref_idx_deg": _quat_angle_deg(qpos[3:7], qpos_ref[ref_idx, 3:7]),
        "qpos_max_abs_diff_vs_ref_idx": float(np.max(np.abs(qpos - qpos_ref[ref_idx]))),
        "qvel_max_abs_diff_vs_ref_idx": float(np.max(np.abs(qvel - qvel_ref[ref_idx]))),
        "qvel_norm": float(np.linalg.norm(qvel)),
        "ctrl_max_abs_diff_vs_ref_step": float(np.max(np.abs(ctrl - ctrl_ref[min(step_idx, len(ctrl_ref) - 1)]))),
        "left_foot_z": float(data.xpos[left_foot, 2]),
        "right_foot_z": float(data.xpos[right_foot, 2]),
        "cpu_ncon": float(cpu_data.ncon if cpu_data is not None else data.ncon),
        "cpu_qfrc_actuator_norm": float(np.linalg.norm(cpu_data.qfrc_actuator) if cpu_data is not None else np.linalg.norm(data.qfrc_actuator)),
    }
    if e069_qpos is not None and 0 <= step_idx < len(e069_qpos):
        row["qpos_max_abs_diff_vs_e069"] = float(np.max(np.abs(qpos - e069_qpos[step_idx])))
        row["yaw_err_vs_e069_deg"] = _normalize_angle_deg(
            yaw - _quat_wxyz_to_yaw_deg(e069_qpos[step_idx, 3:7])
        )
    else:
        row["qpos_max_abs_diff_vs_e069"] = float("nan")
        row["yaw_err_vs_e069_deg"] = float("nan")

    if env is not None:
        row["wp_time"] = _safe_first_world_scalar(env, "time")
        row["wp_qfrc_actuator_norm"] = _safe_first_world_norm(env, "qfrc_actuator")
        row["wp_ncon"] = _safe_first_world_scalar(env, "ncon")
    else:
        row["wp_time"] = float("nan")
        row["wp_qfrc_actuator_norm"] = float("nan")
        row["wp_ncon"] = float("nan")
    return row


def _load_refs(config, *, use_qpos_as_robot_ctrl: bool):
    qpos_ref, qvel_ref, ctrl_ref, contact, contact_pos = load_data(config, config.data_path)
    if (
        use_qpos_as_robot_ctrl
        and config.contact_guidance
        and ctrl_ref.shape[1] != config.nu
        and qpos_ref.shape[1] >= config.nu
    ):
        ctrl_ref = qpos_ref[:, : config.nu]
    qpos_ref, qvel_ref, ctrl_ref = _convert_scene_act_ref(config, qpos_ref, qvel_ref, ctrl_ref)
    return qpos_ref, qvel_ref, ctrl_ref, contact, contact_pos


def _load_e069_trace(path: Path) -> np.ndarray | None:
    if not path.is_file():
        return None
    data = np.load(path)
    qpos = data["qpos"]
    return qpos.reshape(-1, qpos.shape[-1])


def _run_cpu_path(
    *,
    ctrl_label: str,
    config,
    qpos_ref_np: np.ndarray,
    qvel_ref_np: np.ndarray,
    ctrl_ref_np: np.ndarray,
    steps: int,
    restore_object_gains: bool,
    e069_qpos: np.ndarray | None,
) -> list[dict[str, Any]]:
    model = setup_mj_model(config)
    _apply_cpu_object_gains(model, config, enabled=False)
    data = mujoco.MjData(model)
    data.qpos[:] = qpos_ref_np[0]
    data.qvel[:] = qvel_ref_np[0]
    data.ctrl[:] = ctrl_ref_np[0]
    mujoco.mj_step(model, data)

    rows = [
        _measure(
            path=f"{ctrl_label}_cpu_{'restored' if restore_object_gains else 'zero'}_gains",
            phase="after_setup_env_equivalent",
            step_idx=-1,
            model=model,
            qpos=data.qpos.copy(),
            qvel=data.qvel.copy(),
            ctrl=data.ctrl.copy(),
            qpos_ref=qpos_ref_np,
            qvel_ref=qvel_ref_np,
            ctrl_ref=ctrl_ref_np,
            e069_qpos=e069_qpos,
            cpu_data=data,
        )
    ]

    _apply_cpu_object_gains(model, config, enabled=restore_object_gains)
    for i in range(steps):
        data.ctrl[:] = ctrl_ref_np[i]
        mujoco.mj_step(model, data)
        rows.append(
            _measure(
                path=f"{ctrl_label}_cpu_{'restored' if restore_object_gains else 'zero'}_gains",
                phase="commit_ref_ctrl",
                step_idx=i,
                model=model,
                qpos=data.qpos.copy(),
                qvel=data.qvel.copy(),
                ctrl=data.ctrl.copy(),
                qpos_ref=qpos_ref_np,
                qvel_ref=qvel_ref_np,
                ctrl_ref=ctrl_ref_np,
                e069_qpos=e069_qpos,
                cpu_data=data,
            )
        )
    return rows


def _run_mjwarp_path(
    *,
    ctrl_label: str,
    config,
    ref_data,
    qpos_ref_np: np.ndarray,
    qvel_ref_np: np.ndarray,
    ctrl_ref_np: np.ndarray,
    steps: int,
    restore_object_gains: bool,
    e069_qpos: np.ndarray | None,
) -> list[dict[str, Any]]:
    env = setup_env(config, ref_data)
    model = env.model_cpu
    qpos = get_qpos(config, env)[0].detach().cpu().numpy()
    qvel = get_qvel(config, env)[0].detach().cpu().numpy()
    ctrl = wp.to_torch(env.data_wp.ctrl)[0].detach().cpu().numpy()

    rows = [
        _measure(
            path=f"{ctrl_label}_mjwarp_{'restored' if restore_object_gains else 'zero'}_gains",
            phase="after_setup_env",
            step_idx=-1,
            model=model,
            qpos=qpos,
            qvel=qvel,
            ctrl=ctrl,
            qpos_ref=qpos_ref_np,
            qvel_ref=qvel_ref_np,
            ctrl_ref=ctrl_ref_np,
            e069_qpos=e069_qpos,
            env=env,
        )
    ]

    if restore_object_gains and config.object_actuator_ids:
        kp, kd = _commit_gain_vectors(config)
        load_env_params(config, env, {"kp": kp, "kd": kd})

    for i in range(steps):
        ctrl_t = torch.from_numpy(ctrl_ref_np[i]).to(device=config.device, dtype=torch.float32)
        step_env(config, env, ctrl_t)
        qpos = get_qpos(config, env)[0].detach().cpu().numpy()
        qvel = get_qvel(config, env)[0].detach().cpu().numpy()
        ctrl = wp.to_torch(env.data_wp.ctrl)[0].detach().cpu().numpy()
        rows.append(
            _measure(
                path=f"{ctrl_label}_mjwarp_{'restored' if restore_object_gains else 'zero'}_gains",
                phase="commit_ref_ctrl",
                step_idx=i,
                model=model,
                qpos=qpos,
                qvel=qvel,
                ctrl=ctrl,
                qpos_ref=qpos_ref_np,
                qvel_ref=qvel_ref_np,
                ctrl_ref=ctrl_ref_np,
                e069_qpos=e069_qpos,
                env=env,
            )
        )
    return rows


def _write_outputs(rows: list[dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "ref_control_parity.json"
    csv_path = out_dir / "ref_control_parity.csv"
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--override", default="core4d_e069w02_box023")
    parser.add_argument("--task", default="box023_person1")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--out-dir", default="workspace/core4d/results/E070")
    parser.add_argument("--e069-npz", default="workspace/core4d/results/E069/E069W02_box023.npz")
    args = parser.parse_args()

    config = _build_config(args.override, args.task, args.device, args.num_samples)
    e069_qpos = _load_e069_trace(ROOT / args.e069_npz)

    rows: list[dict[str, Any]] = []
    ctrl_variants = [
        ("qpos_ctrl", True),
        ("orig_ctrl", False),
    ]
    for ctrl_label, use_qpos_as_robot_ctrl in ctrl_variants:
        qpos_ref, qvel_ref, ctrl_ref, contact, contact_pos = _load_refs(
            config, use_qpos_as_robot_ctrl=use_qpos_as_robot_ctrl
        )
        ref_data = (qpos_ref, qvel_ref, ctrl_ref, contact, contact_pos)
        qpos_ref_np = qpos_ref.detach().cpu().numpy()
        qvel_ref_np = qvel_ref.detach().cpu().numpy()
        ctrl_ref_np = ctrl_ref.detach().cpu().numpy()

        for restore_object_gains in (False, True):
            rows.extend(
                _run_cpu_path(
                    ctrl_label=ctrl_label,
                    config=config,
                    qpos_ref_np=qpos_ref_np,
                    qvel_ref_np=qvel_ref_np,
                    ctrl_ref_np=ctrl_ref_np,
                    steps=args.steps,
                    restore_object_gains=restore_object_gains,
                    e069_qpos=e069_qpos,
                )
            )
            rows.extend(
                _run_mjwarp_path(
                    ctrl_label=ctrl_label,
                    config=config,
                    ref_data=ref_data,
                    qpos_ref_np=qpos_ref_np,
                    qvel_ref_np=qvel_ref_np,
                    ctrl_ref_np=ctrl_ref_np,
                    steps=args.steps,
                    restore_object_gains=restore_object_gains,
                    e069_qpos=e069_qpos,
                )
            )

    print("path,phase,step,ref_idx,yaw_err_ref_idx,feet_z,qvel_norm,qfrc_act,wp_qfrc,e069_qpos_diff")
    for row in rows:
        print(
            f"{row['path']},{row['phase']},{row['step_idx']},{row['ref_idx']},"
            f"{row['yaw_err_vs_ref_idx_deg']:.3f},"
            f"({row['left_foot_z']:.3f},{row['right_foot_z']:.3f}),"
            f"{row['qvel_norm']:.3f},{row['cpu_qfrc_actuator_norm']:.3f},"
            f"{row['wp_qfrc_actuator_norm']:.3f},"
            f"{row['qpos_max_abs_diff_vs_e069']:.5f}"
        )

    _write_outputs(rows, ROOT / args.out_dir)


if __name__ == "__main__":
    main()
