#!/usr/bin/env python3
"""E068: diagnose MJWP initial-state drift for CORE4D box023.

This script compares CPU MuJoCo and MJWarp initialization paths without running
the full CEM loop. It is intentionally narrow and experiment-local.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any

import mujoco
import mujoco_warp as mjwarp
import numpy as np
import torch
import warp as wp
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation as R

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spider.config import Config, process_config  # noqa: E402
from spider.io import load_data  # noqa: E402
from spider.simulators.mjwp import setup_env  # noqa: E402


OVERRIDE_DIR = ROOT / "examples/config/override"


def _normalize_angle_deg(x: float) -> float:
    return (x + 180.0) % 360.0 - 180.0


def _quat_wxyz_to_euler_xyz_deg(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    return R.from_quat([q[1], q[2], q[3], q[0]]).as_euler("xyz", degrees=True)


def _quat_angle_deg(q: np.ndarray, q_ref: np.ndarray) -> float:
    q = np.asarray(q, dtype=np.float64)
    q_ref = np.asarray(q_ref, dtype=np.float64)
    dot = abs(float(np.dot(q / np.linalg.norm(q), q_ref / np.linalg.norm(q_ref))))
    dot = min(1.0, max(-1.0, dot))
    return math.degrees(2.0 * math.acos(dot))


def _load_override_chain(name: str, seen: set[str] | None = None) -> dict[str, Any]:
    """Load examples/config/override/<name>.yaml, resolving simple defaults chains."""
    if seen is None:
        seen = set()
    if name in seen:
        raise ValueError(f"Cyclic config defaults: {name}")
    seen.add(name)

    path = OVERRIDE_DIR / f"{name}.yaml"
    if not path.is_file():
        raise FileNotFoundError(path)
    raw = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(raw, dict):
        raise ValueError(f"Config did not resolve to a mapping: {path}")

    merged: dict[str, Any] = {}
    defaults = raw.pop("defaults", []) or []
    for item in defaults:
        if item == "_self_":
            continue
        if isinstance(item, str):
            parent = item
        elif isinstance(item, dict):
            # Only local override defaults are used in core4d experiment yamls.
            parent = next(iter(item.values()))
        else:
            continue
        if parent in {"_self_", None}:
            continue
        merged.update(_load_override_chain(str(parent), seen))

    raw.pop("hydra", None)
    merged.update(raw)
    return merged


def _build_config(override: str, task: str, device: str, num_samples: int) -> Config:
    cfg = Config()
    override_dict = _load_override_chain(override)
    allowed = {f.name for f in fields(Config)}
    for key, value in override_dict.items():
        if key in allowed:
            setattr(cfg, key, value)
    cfg.task = task
    cfg.device = device
    cfg.num_samples = num_samples
    cfg.viewer = "none"
    cfg.save_video = False
    cfg.save_info = False
    cfg.save_metrics = False
    cfg.save_config = False
    cfg.use_torch_compile = False
    return process_config(cfg)


def _convert_scene_act_ref(config: Config, qpos_ref, qvel_ref, ctrl_ref):
    """Mirror run_mjwp.py's freejoint -> scene_act conversion for contact guidance."""
    if not ((config.object_pd_override or config.contact_guidance) and qpos_ref.shape[1] > config.nq):
        return qpos_ref, qvel_ref, ctrl_ref

    model = mujoco.MjModel.from_xml_path(config.model_path)
    nq_model = config.nq
    nq_robot = nq_model - 6
    obj_pos_world = qpos_ref[:, nq_robot : nq_robot + 3].detach().cpu().numpy()
    obj_quat_wxyz = qpos_ref[:, nq_robot + 3 : nq_robot + 7].detach().cpu().numpy()

    obj_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    body_pos = model.body_pos[obj_body_id]
    body_quat_wxyz = model.body_quat[obj_body_id]
    r_body = R.from_quat(
        [body_quat_wxyz[1], body_quat_wxyz[2], body_quat_wxyz[3], body_quat_wxyz[0]]
    )
    obj_slide_pos = r_body.inv().apply(obj_pos_world - body_pos[np.newaxis, :])

    meta_path = Path(config.model_path).with_name("scene_act_meta.json")
    if meta_path.is_file():
        euler_conv = json.loads(meta_path.read_text())["euler_convention"]
    else:
        euler_conv = "XYZ"
    obj_quat_xyzw = np.column_stack(
        [obj_quat_wxyz[:, 1], obj_quat_wxyz[:, 2], obj_quat_wxyz[:, 3], obj_quat_wxyz[:, 0]]
    )
    obj_euler = (r_body.inv() * R.from_quat(obj_quat_xyzw)).as_euler(euler_conv)

    qpos_new = torch.zeros((qpos_ref.shape[0], nq_model), device=qpos_ref.device, dtype=qpos_ref.dtype)
    qpos_new[:, :nq_robot] = qpos_ref[:, :nq_robot]
    qpos_new[:, nq_robot : nq_robot + 3] = torch.from_numpy(obj_slide_pos.astype(np.float32)).to(qpos_ref.device)
    qpos_new[:, nq_robot + 3 : nq_robot + 6] = torch.from_numpy(obj_euler.astype(np.float32)).to(qpos_ref.device)

    qvel_new = torch.zeros((qvel_ref.shape[0], config.nv), device=qvel_ref.device, dtype=qvel_ref.dtype)
    qvel_new[:, : min(qvel_ref.shape[1], config.nv)] = qvel_ref[:, : config.nv]

    ctrl_new = torch.zeros((ctrl_ref.shape[0], config.nu), device=ctrl_ref.device, dtype=ctrl_ref.dtype)
    ctrl_new[:, : min(ctrl_ref.shape[1], config.nu)] = ctrl_ref[:, : min(ctrl_ref.shape[1], config.nu)]
    obj_act_start = config.nu - 6
    ctrl_new[:, obj_act_start : obj_act_start + 3] = torch.from_numpy(obj_slide_pos.astype(np.float32)).to(ctrl_ref.device)
    ctrl_new[:, obj_act_start + 3 : obj_act_start + 6] = torch.from_numpy(obj_euler.astype(np.float32)).to(ctrl_ref.device)
    return qpos_new, qvel_new, ctrl_new


def _body_id(model: mujoco.MjModel, candidates: list[str]) -> int:
    for name in candidates:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid >= 0:
            return bid
    raise ValueError(f"None of these bodies exist: {candidates}")


def _measure_state(
    label: str,
    model: mujoco.MjModel,
    qpos: np.ndarray,
    qvel: np.ndarray | None,
    qpos_ref0: np.ndarray,
) -> dict[str, Any]:
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    if qvel is not None:
        data.qvel[:] = qvel
    mujoco.mj_forward(model, data)

    pelvis_id = _body_id(model, ["pelvis", "torso_link"])
    left_foot_id = _body_id(model, ["left_ankle_roll_link", "left_foot_link"])
    right_foot_id = _body_id(model, ["right_ankle_roll_link", "right_foot_link"])
    obj_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")

    q_ref = qpos_ref0[3:7]
    q_cur = np.asarray(qpos[3:7], dtype=np.float64)
    e_ref = _quat_wxyz_to_euler_xyz_deg(q_ref)
    e_cur = _quat_wxyz_to_euler_xyz_deg(q_cur)
    yaw_err = _normalize_angle_deg(float(e_cur[2] - e_ref[2]))
    qpos_diff = np.asarray(qpos) - np.asarray(qpos_ref0)

    out: dict[str, Any] = {
        "label": label,
        "pelvis_pos": np.asarray(qpos[:3], dtype=float).tolist(),
        "pelvis_quat_wxyz": q_cur.tolist(),
        "pelvis_euler_xyz_deg": e_cur.tolist(),
        "ref_yaw_deg": float(e_ref[2]),
        "yaw_err_deg": yaw_err,
        "quat_angle_err_deg": _quat_angle_deg(q_cur, q_ref),
        "qpos_max_abs_diff": float(np.max(np.abs(qpos_diff))),
        "qpos_l2_diff": float(np.linalg.norm(qpos_diff)),
        "left_foot_z": float(data.xpos[left_foot_id, 2]),
        "right_foot_z": float(data.xpos[right_foot_id, 2]),
    }
    if obj_id >= 0:
        out["object_pos"] = data.xpos[obj_id].astype(float).tolist()
        out["object_quat_wxyz"] = data.xquat[obj_id].astype(float).tolist()
    return out


def _write_outputs(rows: list[dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "init_drift_diagnosis.json"
    csv_path = out_dir / "init_drift_diagnosis.csv"
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    scalar_keys = [
        "label",
        "yaw_err_deg",
        "quat_angle_err_deg",
        "qpos_max_abs_diff",
        "qpos_l2_diff",
        "left_foot_z",
        "right_foot_z",
        "ref_yaw_deg",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=scalar_keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in scalar_keys})
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")


def _try_manual_put_data(
    config: Config,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    qpos_ref0: np.ndarray,
) -> dict[str, Any]:
    """Inspect mjwarp.put_data without CUDA graph capture."""
    dev = str(config.device)
    wp.set_device(dev)
    with wp.ScopedDevice(dev):
        mjwarp.put_model(model)
        data_wp = mjwarp.put_data(
            model,
            data,
            nworld=int(config.num_samples),
            nconmax=int(config.nconmax_per_env),
            njmax=int(config.njmax_per_env),
        )
    qpos = wp.to_torch(data_wp.qpos)[0].detach().cpu().numpy()
    qvel = wp.to_torch(data_wp.qvel)[0].detach().cpu().numpy()
    return _measure_state("warp_manual_put_data_from_cpu_step", model, qpos, qvel, qpos_ref0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--override", default="core4d_e062_box023")
    parser.add_argument("--task", default="box023_person1")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--out-dir", default="workspace/core4d/results/E068")
    args = parser.parse_args()

    config = _build_config(args.override, args.task, args.device, args.num_samples)
    qpos_ref, qvel_ref, ctrl_ref, contact, contact_pos = load_data(config, config.data_path)
    qpos_ref, qvel_ref, ctrl_ref = _convert_scene_act_ref(config, qpos_ref, qvel_ref, ctrl_ref)
    ref_data = (qpos_ref, qvel_ref, ctrl_ref, contact, contact_pos)

    model = mujoco.MjModel.from_xml_path(config.model_path)
    qpos0 = qpos_ref[0].detach().cpu().numpy()
    qvel0 = qvel_ref[0].detach().cpu().numpy()
    ctrl0 = ctrl_ref[0].detach().cpu().numpy()

    rows: list[dict[str, Any]] = []
    rows.append(_measure_state("ref_qpos0", model, qpos0, qvel0, qpos0))

    data_forward = mujoco.MjData(model)
    data_forward.qpos[:] = qpos0
    data_forward.qvel[:] = qvel0
    data_forward.ctrl[:] = ctrl0
    mujoco.mj_forward(model, data_forward)
    rows.append(_measure_state("cpu_forward", model, data_forward.qpos.copy(), data_forward.qvel.copy(), qpos0))

    data_step = mujoco.MjData(model)
    data_step.qpos[:] = qpos0
    data_step.qvel[:] = qvel0
    data_step.ctrl[:] = ctrl0
    mujoco.mj_step(model, data_step)
    rows.append(_measure_state("cpu_after_one_mj_step", model, data_step.qpos.copy(), data_step.qvel.copy(), qpos0))

    try:
        rows.append(_try_manual_put_data(config, model, data_step, qpos0))
    except Exception as exc:
        rows.append(
            {
                "label": "warp_manual_put_data_from_cpu_step",
                "error": repr(exc),
                "yaw_err_deg": float("nan"),
                "quat_angle_err_deg": float("nan"),
                "qpos_max_abs_diff": float("nan"),
                "qpos_l2_diff": float("nan"),
                "left_foot_z": float("nan"),
                "right_foot_z": float("nan"),
                "ref_yaw_deg": float("nan"),
            }
        )

    try:
        env = setup_env(config, ref_data)
        warp_qpos = wp.to_torch(env.data_wp.qpos)[0].detach().cpu().numpy()
        warp_qvel = wp.to_torch(env.data_wp.qvel)[0].detach().cpu().numpy()
        rows.append(_measure_state("warp_after_current_setup_env", model, warp_qpos, warp_qvel, qpos0))
    except Exception as exc:
        rows.append(
            {
                "label": "warp_after_current_setup_env",
                "error": repr(exc),
                "yaw_err_deg": float("nan"),
                "quat_angle_err_deg": float("nan"),
                "qpos_max_abs_diff": float("nan"),
                "qpos_l2_diff": float("nan"),
                "left_foot_z": float("nan"),
                "right_foot_z": float("nan"),
                "ref_yaw_deg": float("nan"),
            }
        )

    for row in rows:
        print(
            f"{row['label']:36s} yaw_err={row['yaw_err_deg']:8.3f} deg "
            f"quat_err={row['quat_angle_err_deg']:8.3f} deg "
            f"qpos_max={row['qpos_max_abs_diff']:8.5f} "
            f"feet_z=({row['left_foot_z']:.3f},{row['right_foot_z']:.3f})"
            + (f" error={row['error']}" if "error" in row else "")
        )

    _write_outputs(rows, ROOT / args.out_dir)


if __name__ == "__main__":
    main()
