#!/usr/bin/env python3
"""No-training MuJoCo sanity for E029 dynamic D6 support load path."""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio
import mujoco
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import e029_common as common  # noqa: E402


MANIFEST = common.E029_RESULTS / "d6/manifest.tsv"
OUT_DIR = common.E029_RESULTS / "d6/sanity"
REPRESENTATIVE_SOURCE = "E028_d003_box021_20231018_029_p2_canonical_t02"


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def _row_for_mode(rows: list[dict[str, str]], mode: str, subset: str) -> list[dict[str, str]]:
    if mode in {"direct-object-wrench-control", "mocap-weld-current"}:
        selected = common.candidate_rows(manifest_path=common.MANIFEST, candidates_path=common.CANDIDATES)
        if subset == "representative":
            selected = [row for row in selected if row["variant"] == REPRESENTATIVE_SOURCE]
        elif subset != "candidates":
            raise ValueError(f"Unknown subset={subset}")
        if not selected:
            raise ValueError(f"No E028 rows selected for mode={mode} subset={subset}")
        return selected

    profiles = {
        "d6-compliant-support": "d6_compliant",
        "d6-locked-support": "d6_locked",
        "free-compliant-support": "free_compliant",
        "free-locked-support": "free_locked",
        "connect-compliant-support": "connect_compliant",
        "multi-triad-support": "multi_triad",
        "multi-cross-support": "multi_cross",
        "multi-cross-stiff-support": "multi_cross_stiff",
        "multi-cross-ultra-support": "multi_cross_ultra",
    }
    if mode not in profiles:
        raise ValueError(f"Unsupported mode for this phase: {mode}")
    profile = profiles[mode]
    selected = [row for row in rows if row["profile"] == profile]
    if subset == "representative":
        selected = [row for row in selected if row["source_e028_variant"] == REPRESENTATIVE_SOURCE]
    elif subset == "candidates":
        pass
    else:
        raise ValueError(f"Unknown subset={subset}")
    if not selected:
        raise ValueError(f"No rows selected for mode={mode} subset={subset}")
    return selected


def _wrap_pi(x: np.ndarray) -> np.ndarray:
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def _clamp_vec(vec: np.ndarray, limit: float) -> tuple[np.ndarray, bool]:
    if limit <= 0:
        return vec, False
    norm = float(np.linalg.norm(vec))
    if norm <= limit:
        return vec, False
    return vec / max(norm, 1e-8) * limit, True


def _quat_angle_deg(q0: np.ndarray, q1: np.ndarray) -> float:
    q0 = q0 / max(float(np.linalg.norm(q0)), 1e-8)
    q1 = q1 / max(float(np.linalg.norm(q1)), 1e-8)
    dot = min(1.0, max(-1.0, abs(float(np.dot(q0, q1)))))
    return float(np.degrees(2.0 * np.arccos(dot)))


def _quat_normalize(q: np.ndarray) -> np.ndarray:
    return q / max(float(np.linalg.norm(q)), 1e-8)


def _quat_conj(q: np.ndarray) -> np.ndarray:
    return np.asarray([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.asarray(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=np.float64,
    )


def _quat_error_rotvec(target: np.ndarray, current: np.ndarray) -> np.ndarray:
    err = _quat_mul(_quat_normalize(target), _quat_conj(_quat_normalize(current)))
    if err[0] < 0.0:
        err = -err
    xyz = err[1:4]
    sin_half = float(np.linalg.norm(xyz))
    if sin_half < 1e-8:
        return np.zeros(3, dtype=np.float64)
    angle = 2.0 * math.atan2(sin_half, max(float(err[0]), 1e-8))
    return xyz / sin_half * angle


def _joint_addrs(model: mujoco.MjModel, body_name: str) -> tuple[int, int]:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if bid < 0:
        raise ValueError(f"missing body {body_name}")
    jadr = int(model.body_jntadr[bid])
    if jadr < 0:
        raise ValueError(f"body {body_name} has no joint")
    return int(model.jnt_qposadr[jadr]), int(model.jnt_dofadr[jadr])


def _support_joint_info(model: mujoco.MjModel, body_name: str) -> tuple[int, int, int, bool]:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if bid < 0:
        raise ValueError(f"missing body {body_name}")
    jadr = int(model.body_jntadr[bid])
    jnum = int(model.body_jntnum[bid])
    if jadr < 0:
        raise ValueError(f"body {body_name} has no joint")
    if jnum == 1 and int(model.jnt_type[jadr]) == int(mujoco.mjtJoint.mjJNT_FREE):
        return int(model.jnt_qposadr[jadr]), int(model.jnt_dofadr[jadr]), 7, True
    if jnum == 6:
        qaddrs = [int(model.jnt_qposadr[jadr + i]) for i in range(jnum)]
        daddrs = [int(model.jnt_dofadr[jadr + i]) for i in range(jnum)]
        if qaddrs != list(range(qaddrs[0], qaddrs[0] + 6)):
            raise ValueError(f"support dynamic qpos addresses are not contiguous: {qaddrs}")
        if daddrs != list(range(daddrs[0], daddrs[0] + 6)):
            raise ValueError(f"support dynamic dof addresses are not contiguous: {daddrs}")
        return qaddrs[0], daddrs[0], 6, False
    raise ValueError(f"{body_name} requires a freejoint or 6 scalar joints, got {jnum}")


def _body_pos(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> np.ndarray:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    return data.xpos[bid].copy()


def _support_point(model: mujoco.MjModel, data: mujoco.MjData, point_local: np.ndarray) -> np.ndarray:
    obj_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    obj_quat = data.xquat[obj_bid].copy()
    return data.xpos[obj_bid].copy() + common.quat_apply(obj_quat, point_local)


def _render_video(model: mujoco.MjModel, qpos_frames: list[np.ndarray], out_mp4: Path, width: int, height: int) -> None:
    if not qpos_frames:
        return
    model.vis.global_.offwidth = width
    model.vis.global_.offheight = height
    renderer = mujoco.Renderer(model, height=height, width=width)
    data = mujoco.MjData(model)
    frames: list[np.ndarray] = []
    step = max(1, len(qpos_frames) // 120)
    for qpos in qpos_frames[::step]:
        data.qpos[: model.nq] = qpos
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, "track")
        frames.append(renderer.render())
    renderer.close()
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(out_mp4), frames, fps=25, codec="libx264", quality=8)


def _smoothstep(n: int) -> np.ndarray:
    if n <= 1:
        return np.ones((n,), dtype=np.float64)
    x = np.linspace(0.0, 1.0, n, dtype=np.float64)
    return x * x * (3.0 - 2.0 * x)


def _finite_velocity(values: np.ndarray, dt: float) -> np.ndarray:
    vel = np.zeros_like(values)
    if len(values) > 1:
        vel[1:] = (values[1:] - values[:-1]) / max(dt, 1e-8)
        vel[0] = vel[1]
    return vel


def _finite_freejoint_velocity(qpos: np.ndarray, dt: float) -> np.ndarray:
    vel = np.zeros((len(qpos), 6), dtype=np.float64)
    vel[:, :3] = _finite_velocity(qpos[:, :3], dt)
    if len(qpos) > 1:
        for i in range(1, len(qpos)):
            vel[i, 3:6] = _quat_error_rotvec(qpos[i, 3:7], qpos[i - 1, 3:7]) / max(dt, 1e-8)
        vel[0, 3:6] = vel[1, 3:6]
    return vel


def _target_tracks(
    qpos_ref: np.ndarray,
    qvel_ref: np.ndarray,
    *,
    support_qadr: int,
    support_dadr: int,
    support_qwidth: int,
    support_is_freejoint: bool,
    object_qadr: int,
    dt: float,
    target_mode: str,
    target_lowpass_tau: float,
    target_ramp_time: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if target_mode == "raw_ref":
        support_target = qpos_ref[:, support_qadr : support_qadr + support_qwidth].copy()
        object_target = qpos_ref[:, object_qadr : object_qadr + 3].copy()
        if not support_is_freejoint:
            support_target[:, 3:6] = np.unwrap(support_target[:, 3:6], axis=0)
    elif target_mode == "smooth_final":
        s = _smoothstep(len(qpos_ref))[:, None]
        support_raw = qpos_ref[:, support_qadr : support_qadr + support_qwidth].copy()
        object_raw = qpos_ref[:, object_qadr : object_qadr + 3].copy()
        if support_is_freejoint:
            raise ValueError("smooth_final target is not implemented for freejoint support")
        support_raw[:, 3:6] = np.unwrap(support_raw[:, 3:6], axis=0)
        support_target = support_raw[:1] + s * (support_raw[-1:] - support_raw[:1])
        object_target = object_raw[:1] + s * (object_raw[-1:] - object_raw[:1])
    else:
        raise ValueError(f"Unknown target_mode={target_mode}")

    if target_lowpass_tau > 0.0 and len(support_target) > 1:
        if support_is_freejoint:
            raise ValueError("target low-pass is not implemented for freejoint support")
        alpha = dt / (target_lowpass_tau + dt)
        support_filtered = support_target.copy()
        object_filtered = object_target.copy()
        for i in range(1, len(support_target)):
            support_filtered[i] = support_filtered[i - 1] + alpha * (support_target[i] - support_filtered[i - 1])
            object_filtered[i] = object_filtered[i - 1] + alpha * (object_target[i] - object_filtered[i - 1])
        support_target = support_filtered
        object_target = object_filtered

    if target_ramp_time > 0.0 and len(support_target) > 1:
        if support_is_freejoint:
            raise ValueError("target ramp is not implemented for freejoint support")
        ramp = np.clip(np.arange(len(support_target), dtype=np.float64) * dt / target_ramp_time, 0.0, 1.0)
        ramp = (ramp * ramp * (3.0 - 2.0 * ramp))[:, None]
        support_target = support_target[:1] + ramp * (support_target - support_target[:1])
        object_target = object_target[:1] + ramp * (object_target - object_target[:1])

    if support_is_freejoint:
        if qvel_ref.shape[1] >= support_dadr + 6:
            support_vel = qvel_ref[:, support_dadr : support_dadr + 6].copy()
        else:
            support_vel = _finite_freejoint_velocity(support_target, dt)
    else:
        support_vel = _finite_velocity(support_target, dt)
    return support_target, support_vel, object_target


def run_row(
    row: dict[str, str],
    *,
    out_dir: Path,
    render_video: bool,
    max_frames: int | None,
    target_mode: str,
    target_lowpass_tau: float = 0.0,
    target_ramp_time: float = 0.0,
    force_clamp_override: float | None = None,
    torque_clamp_override: float | None = None,
    pos_kp_override: float | None = None,
    pos_kd_override: float | None = None,
    rot_kp_override: float | None = None,
    rot_kd_override: float | None = None,
    file_tag: str = "",
) -> dict[str, str]:
    scene = common.BASE / row["derived_task"] / f"{row['scene_name']}.xml"
    ref_path = common.REPO / row["data_relpath"]
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    ref = np.load(ref_path, allow_pickle=True)
    qpos_ref = ref["qpos"].astype(np.float64)
    qvel_ref = ref["qvel"].astype(np.float64)
    ctrl_ref = ref["ctrl"].astype(np.float64) if "ctrl" in ref.files else np.zeros((len(qpos_ref), model.nu))
    if max_frames is not None:
        qpos_ref = qpos_ref[:max_frames]
        qvel_ref = qvel_ref[:max_frames]
        ctrl_ref = ctrl_ref[:max_frames]
    support_qadr, support_dadr, support_qwidth, support_is_freejoint = _support_joint_info(
        model, "support_dynamic_anchor"
    )
    object_qadr, object_dadr = _joint_addrs(model, "object")
    dt = float(model.opt.timestep)
    support_target, support_target_vel, object_target_pos = _target_tracks(
        qpos_ref,
        qvel_ref,
        support_qadr=support_qadr,
        support_dadr=support_dadr,
        support_qwidth=support_qwidth,
        support_is_freejoint=support_is_freejoint,
        object_qadr=object_qadr,
        dt=dt,
        target_mode=target_mode,
        target_lowpass_tau=target_lowpass_tau,
        target_ramp_time=target_ramp_time,
    )
    point_local = np.asarray(
        [
            float(row["support_proxy_point_local_x"]),
            float(row["support_proxy_point_local_y"]),
            float(row["support_proxy_point_local_z"]),
        ],
        dtype=np.float64,
    )

    data.qpos[: model.nq] = qpos_ref[0]
    data.qvel[: model.nv] = qvel_ref[0]
    data.ctrl[: model.nu] = ctrl_ref[0, : model.nu]
    mujoco.mj_forward(model, data)

    locked_upper_bound = row["profile"] in {"d6_locked", "free_locked"}
    kp = float(row["support_dynamic_pos_kp"])
    if pos_kp_override is not None:
        kp = float(pos_kp_override)
    if locked_upper_bound:
        kp = max(kp, 8000.0)
    kd = float(row["support_dynamic_pos_kd"])
    if kd < 0:
        kd = 2.0 * math.sqrt(max(float(row["support_dynamic_mass"]), 1e-6) * max(kp, 0.0))
    if pos_kd_override is not None:
        kd = float(pos_kd_override)
    rkp = float(row["support_dynamic_rot_kp"])
    if rot_kp_override is not None:
        rkp = float(rot_kp_override)
    if locked_upper_bound:
        rkp = max(rkp, 200.0)
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "support_dynamic_anchor")
    avg_inertia = float(np.mean(model.body_inertia[body_id]))
    rkd = float(row["support_dynamic_rot_kd"])
    if rkd < 0:
        rkd = 2.0 * math.sqrt(max(avg_inertia, 1e-8) * max(rkp, 0.0))
    if rot_kd_override is not None:
        rkd = float(rot_kd_override)
    # The locked profile is an upper-bound load-path sanity: it still acts
    # through support-body generalized force, but intentionally removes the
    # force clamp so clamp saturation is not mistaken for a D6 connection issue.
    force_limit = 0.0 if locked_upper_bound else float(row["support_dynamic_force_clamp"])
    torque_limit = 0.0 if locked_upper_bound else float(row["support_dynamic_torque_clamp"])
    if force_clamp_override is not None and not locked_upper_bound:
        force_limit = float(force_clamp_override)
    if torque_clamp_override is not None and not locked_upper_bound:
        torque_limit = float(torque_clamp_override)

    rows: list[dict[str, str]] = []
    qpos_frames: list[np.ndarray] = []
    sat_count = 0
    torque_sat_count = 0
    nan_count = 0
    for i in range(len(qpos_ref) - 1):
        idx = min(i, len(qpos_ref) - 1)
        data.qpos[:36] = qpos_ref[idx, :36]
        data.qvel[:35] = qvel_ref[idx, :35]
        data.ctrl[: model.nu] = ctrl_ref[idx, : model.nu]
        data.qfrc_applied[:] = 0.0
        target_q = support_target[idx]
        target_v = support_target_vel[idx]
        cur_q = data.qpos[support_qadr : support_qadr + support_qwidth]
        cur_v = data.qvel[support_dadr : support_dadr + 6]
        pos_force = kp * (target_q[:3] - cur_q[:3]) + kd * (target_v[:3] - cur_v[:3])
        if support_is_freejoint:
            rot_err = _quat_error_rotvec(target_q[3:7], cur_q[3:7])
        else:
            rot_err = _wrap_pi(target_q[3:6] - cur_q[3:6])
        rot_torque = rkp * rot_err + rkd * (target_v[3:6] - cur_v[3:6])
        pos_force = np.nan_to_num(pos_force)
        rot_torque = np.nan_to_num(rot_torque)
        pos_force, saturated = _clamp_vec(pos_force, force_limit)
        rot_torque, torque_saturated = _clamp_vec(rot_torque, torque_limit)
        sat_count += int(saturated)
        torque_sat_count += int(torque_saturated)
        if support_is_freejoint:
            qfrc = np.zeros(model.nv, dtype=np.float64)
            mujoco.mj_applyFT(model, data, pos_force, rot_torque, data.xpos[body_id], body_id, qfrc)
            data.qfrc_applied[:] += qfrc
        else:
            data.qfrc_applied[support_dadr : support_dadr + 3] = pos_force
            data.qfrc_applied[support_dadr + 3 : support_dadr + 6] = rot_torque
        mujoco.mj_step(model, data)
        unstable_state = (
            (not np.all(np.isfinite(data.qpos)))
            or (not np.all(np.isfinite(data.qvel)))
            or (not np.all(np.isfinite(data.qacc)))
            or float(np.max(np.abs(data.qpos))) > 1e5
            or float(np.max(np.abs(data.qvel))) > 1e5
            or float(np.max(np.abs(data.qacc))) > 1e8
        )
        if unstable_state:
            nan_count += 1
        mujoco.mj_forward(model, data)
        support_pos = _body_pos(model, data, "support_dynamic_anchor")
        object_support = _support_point(model, data, point_local)
        obj_ref_pos = object_target_pos[idx]
        obj_ref_quat = qpos_ref[idx, object_qadr + 3 : object_qadr + 7]
        obj_pos = data.qpos[object_qadr : object_qadr + 3].copy()
        obj_quat = data.qpos[object_qadr + 3 : object_qadr + 7].copy()
        rows.append(
            {
                "frame": str(i),
                "time": f"{data.time:.8g}",
                "support_target_err_m": f"{float(np.linalg.norm(target_q[:3] - data.qpos[support_qadr:support_qadr+3])):.8g}",
                "support_object_drift_m": f"{float(np.linalg.norm(support_pos - object_support)):.8g}",
                "object_pos_err_m": f"{float(np.linalg.norm(obj_pos - obj_ref_pos)):.8g}",
                "object_quat_err_deg": f"{_quat_angle_deg(obj_quat, obj_ref_quat):.8g}",
                "support_force_norm": f"{float(np.linalg.norm(pos_force)):.8g}",
                "support_torque_norm": f"{float(np.linalg.norm(rot_torque)):.8g}",
                "force_saturated": str(int(saturated)),
                "torque_saturated": str(int(torque_saturated)),
            }
        )
        if render_video and i % 2 == 0:
            qpos_frames.append(data.qpos.copy())

    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{file_tag}" if file_tag else ""
    ts_path = out_dir / f"{row['variant']}{suffix}_timeseries.csv"
    with ts_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    if render_video:
        _render_video(model, qpos_frames, out_dir / f"{row['variant']}{suffix}_sanity.mp4", 960, 544)

    def arr(key: str) -> np.ndarray:
        return np.asarray([float(item[key]) for item in rows], dtype=np.float64)

    support_drift = arr("support_object_drift_m")
    obj_err = arr("object_pos_err_m")
    target_err = arr("support_target_err_m")
    force_sat_frac = sat_count / max(1, len(rows))
    torque_sat_frac = torque_sat_count / max(1, len(rows))
    pass_gate = (
        nan_count == 0
        and float(np.mean(support_drift)) <= 0.05
        and float(np.max(support_drift)) <= 0.12
        and (locked_upper_bound or force_sat_frac <= 0.15)
        and (locked_upper_bound or torque_sat_frac <= 0.15)
        and float(np.mean(obj_err)) <= 0.12
    )
    return {
        "variant": row["variant"],
        "source_e028_variant": row["source_e028_variant"],
        "profile": row["profile"],
        "support_joint_layout": "freejoint" if support_is_freejoint else "scalar6",
        "target_mode": target_mode,
        "target_lowpass_tau": f"{target_lowpass_tau:.8g}",
        "target_ramp_time": f"{target_ramp_time:.8g}",
        "pos_kp": f"{kp:.8g}",
        "pos_kd": f"{kd:.8g}",
        "rot_kp": f"{rkp:.8g}",
        "rot_kd": f"{rkd:.8g}",
        "force_clamp": f"{force_limit:.8g}",
        "torque_clamp": f"{torque_limit:.8g}",
        "frames": str(len(rows)),
        "support_drift_mean_m": f"{float(np.mean(support_drift)):.8g}",
        "support_drift_max_m": f"{float(np.max(support_drift)):.8g}",
        "support_target_err_mean_m": f"{float(np.mean(target_err)):.8g}",
        "support_target_err_max_m": f"{float(np.max(target_err)):.8g}",
        "object_pos_err_mean_m": f"{float(np.mean(obj_err)):.8g}",
        "object_pos_err_max_m": f"{float(np.max(obj_err)):.8g}",
        "force_saturation_frac": f"{force_sat_frac:.8g}",
        "torque_saturation_frac": f"{torque_sat_frac:.8g}",
        "nan_count": str(nan_count),
        "pass_gate": "true" if pass_gate else "false",
        "timeseries_path": common.rel_or_abs(ts_path),
        "video_path": common.rel_or_abs(out_dir / f"{row['variant']}{suffix}_sanity.mp4") if render_video else "",
    }


def run_mocap_weld_row(
    row: dict[str, str],
    *,
    out_dir: Path,
    render_video: bool,
    max_frames: int | None,
    file_tag: str = "",
) -> dict[str, str]:
    scene = common.BASE / row["derived_task"] / f"{row['scene_name']}.xml"
    ref_path = Path(row["trajectory_path"])
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    ref = np.load(ref_path, allow_pickle=True)
    qpos_ref = ref["qpos"].astype(np.float64)
    qvel_ref = ref["qvel"].astype(np.float64)
    ctrl_ref = ref["ctrl"].astype(np.float64) if "ctrl" in ref.files else np.zeros((len(qpos_ref), model.nu))
    if max_frames is not None:
        qpos_ref = qpos_ref[:max_frames]
        qvel_ref = qvel_ref[:max_frames]
        ctrl_ref = ctrl_ref[:max_frames]
    object_qadr, _object_dadr = _joint_addrs(model, "object")
    support_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "support_weld_anchor")
    if support_bid < 0:
        raise ValueError(f"{scene} missing support_weld_anchor")
    mocap_id = int(model.body_mocapid[support_bid])
    if mocap_id < 0:
        raise ValueError(f"{scene} support_weld_anchor is not mocap")
    point_local = common.anchor_local(row)

    data.qpos[: model.nq] = qpos_ref[0, : model.nq]
    data.qvel[: model.nv] = qvel_ref[0, : model.nv]
    data.ctrl[: model.nu] = ctrl_ref[0, : model.nu]
    mujoco.mj_forward(model, data)

    rows: list[dict[str, str]] = []
    qpos_frames: list[np.ndarray] = []
    for i in range(len(qpos_ref) - 1):
        idx = min(i, len(qpos_ref) - 1)
        data.qpos[:36] = qpos_ref[idx, :36]
        data.qvel[:35] = qvel_ref[idx, :35]
        data.ctrl[: model.nu] = ctrl_ref[idx, : model.nu]
        obj_ref_pos = qpos_ref[idx, object_qadr : object_qadr + 3]
        obj_ref_quat = qpos_ref[idx, object_qadr + 3 : object_qadr + 7]
        mocap_target = obj_ref_pos + common.quat_apply(obj_ref_quat, point_local)
        data.mocap_pos[mocap_id] = mocap_target
        data.mocap_quat[mocap_id] = obj_ref_quat
        mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        support_pos = data.xpos[support_bid].copy()
        object_support = _support_point(model, data, point_local)
        obj_pos = data.qpos[object_qadr : object_qadr + 3].copy()
        obj_quat = data.qpos[object_qadr + 3 : object_qadr + 7].copy()
        rows.append(
            {
                "frame": str(i),
                "time": f"{data.time:.8g}",
                "support_target_err_m": f"{float(np.linalg.norm(support_pos - mocap_target)):.8g}",
                "support_object_drift_m": f"{float(np.linalg.norm(support_pos - object_support)):.8g}",
                "object_pos_err_m": f"{float(np.linalg.norm(obj_pos - obj_ref_pos)):.8g}",
                "object_quat_err_deg": f"{_quat_angle_deg(obj_quat, obj_ref_quat):.8g}",
                "support_force_norm": "0",
                "support_torque_norm": "0",
                "force_saturated": "0",
                "torque_saturated": "0",
            }
        )
        if render_video and i % 2 == 0:
            qpos_frames.append(data.qpos.copy())

    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{file_tag}" if file_tag else ""
    ts_path = out_dir / f"{row['variant']}_mocap_weld_current{suffix}_timeseries.csv"
    with ts_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    video_path = out_dir / f"{row['variant']}_mocap_weld_current{suffix}_sanity.mp4"
    if render_video:
        _render_video(model, qpos_frames, video_path, 960, 544)
    support_drift = np.asarray([float(item["support_object_drift_m"]) for item in rows], dtype=np.float64)
    obj_err = np.asarray([float(item["object_pos_err_m"]) for item in rows], dtype=np.float64)
    target_err = np.asarray([float(item["support_target_err_m"]) for item in rows], dtype=np.float64)
    return {
        "variant": f"{row['variant']}_mocap_weld_current",
        "source_e028_variant": row["variant"],
        "profile": "mocap_weld_current",
        "support_joint_layout": "mocap",
        "target_mode": "raw_ref",
        "target_lowpass_tau": "0",
        "target_ramp_time": "0",
        "pos_kp": "0",
        "pos_kd": "0",
        "rot_kp": "0",
        "rot_kd": "0",
        "force_clamp": "0",
        "torque_clamp": "0",
        "frames": str(len(rows)),
        "support_drift_mean_m": f"{float(np.mean(support_drift)):.8g}",
        "support_drift_max_m": f"{float(np.max(support_drift)):.8g}",
        "support_target_err_mean_m": f"{float(np.mean(target_err)):.8g}",
        "support_target_err_max_m": f"{float(np.max(target_err)):.8g}",
        "object_pos_err_mean_m": f"{float(np.mean(obj_err)):.8g}",
        "object_pos_err_max_m": f"{float(np.max(obj_err)):.8g}",
        "force_saturation_frac": "0",
        "torque_saturation_frac": "0",
        "nan_count": "0",
        "pass_gate": "false",
        "timeseries_path": common.rel_or_abs(ts_path),
        "video_path": common.rel_or_abs(video_path) if render_video else "",
    }


def run_direct_object_wrench_row(
    row: dict[str, str],
    *,
    out_dir: Path,
    render_video: bool,
    max_frames: int | None,
    pos_kp_override: float | None = None,
    pos_kd_override: float | None = None,
    rot_kp_override: float | None = None,
    rot_kd_override: float | None = None,
    force_clamp_override: float | None = None,
    torque_clamp_override: float | None = None,
    file_tag: str = "",
) -> dict[str, str]:
    scene = common.BASE / row["source_task"] / "scene.xml"
    ref_path = Path(row["trajectory_path"])
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    ref = np.load(ref_path, allow_pickle=True)
    qpos_ref = ref["qpos"].astype(np.float64)
    qvel_ref = ref["qvel"].astype(np.float64)
    ctrl_ref = ref["ctrl"].astype(np.float64) if "ctrl" in ref.files else np.zeros((len(qpos_ref), model.nu))
    if max_frames is not None:
        qpos_ref = qpos_ref[:max_frames]
        qvel_ref = qvel_ref[:max_frames]
        ctrl_ref = ctrl_ref[:max_frames]
    object_qadr, object_dadr = _joint_addrs(model, "object")
    object_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    point_local = common.anchor_local(row)

    kp = 1000.0 if pos_kp_override is None else float(pos_kp_override)
    kd = 80.0 if pos_kd_override is None else float(pos_kd_override)
    rkp = 80.0 if rot_kp_override is None else float(rot_kp_override)
    rkd = 8.0 if rot_kd_override is None else float(rot_kd_override)
    force_limit = 0.0 if force_clamp_override is None else float(force_clamp_override)
    torque_limit = 0.0 if torque_clamp_override is None else float(torque_clamp_override)

    data.qpos[: model.nq] = qpos_ref[0, : model.nq]
    data.qvel[: model.nv] = qvel_ref[0, : model.nv]
    data.ctrl[: model.nu] = ctrl_ref[0, : model.nu]
    mujoco.mj_forward(model, data)

    rows: list[dict[str, str]] = []
    qpos_frames: list[np.ndarray] = []
    sat_count = 0
    torque_sat_count = 0
    nan_count = 0
    for i in range(len(qpos_ref) - 1):
        idx = min(i, len(qpos_ref) - 1)
        data.qpos[:36] = qpos_ref[idx, :36]
        data.qvel[:35] = qvel_ref[idx, :35]
        data.ctrl[: model.nu] = ctrl_ref[idx, : model.nu]
        data.qfrc_applied[:] = 0.0
        obj_ref_pos = qpos_ref[idx, object_qadr : object_qadr + 3]
        obj_ref_quat = qpos_ref[idx, object_qadr + 3 : object_qadr + 7]
        obj_ref_vel = qvel_ref[idx, object_dadr : object_dadr + 6]
        obj_pos = data.qpos[object_qadr : object_qadr + 3].copy()
        obj_quat = data.qpos[object_qadr + 3 : object_qadr + 7].copy()
        obj_vel = data.qvel[object_dadr : object_dadr + 6].copy()
        pos_force = kp * (obj_ref_pos - obj_pos) + kd * (obj_ref_vel[:3] - obj_vel[:3])
        rot_torque = rkp * _quat_error_rotvec(obj_ref_quat, obj_quat) + rkd * (obj_ref_vel[3:6] - obj_vel[3:6])
        pos_force = np.nan_to_num(pos_force)
        rot_torque = np.nan_to_num(rot_torque)
        pos_force, saturated = _clamp_vec(pos_force, force_limit)
        rot_torque, torque_saturated = _clamp_vec(rot_torque, torque_limit)
        sat_count += int(saturated)
        torque_sat_count += int(torque_saturated)
        qfrc = np.zeros(model.nv, dtype=np.float64)
        mujoco.mj_applyFT(model, data, pos_force, rot_torque, data.xpos[object_bid], object_bid, qfrc)
        data.qfrc_applied[:] += qfrc
        mujoco.mj_step(model, data)
        unstable_state = (
            (not np.all(np.isfinite(data.qpos)))
            or (not np.all(np.isfinite(data.qvel)))
            or (not np.all(np.isfinite(data.qacc)))
            or float(np.max(np.abs(data.qpos))) > 1e5
            or float(np.max(np.abs(data.qvel))) > 1e5
            or float(np.max(np.abs(data.qacc))) > 1e8
        )
        if unstable_state:
            nan_count += 1
        mujoco.mj_forward(model, data)
        obj_pos = data.qpos[object_qadr : object_qadr + 3].copy()
        obj_quat = data.qpos[object_qadr + 3 : object_qadr + 7].copy()
        object_support = _support_point(model, data, point_local)
        target_support = obj_ref_pos + common.quat_apply(obj_ref_quat, point_local)
        rows.append(
            {
                "frame": str(i),
                "time": f"{data.time:.8g}",
                "support_target_err_m": f"{float(np.linalg.norm(object_support - target_support)):.8g}",
                "support_object_drift_m": "0",
                "object_pos_err_m": f"{float(np.linalg.norm(obj_pos - obj_ref_pos)):.8g}",
                "object_quat_err_deg": f"{_quat_angle_deg(obj_quat, obj_ref_quat):.8g}",
                "support_force_norm": f"{float(np.linalg.norm(pos_force)):.8g}",
                "support_torque_norm": f"{float(np.linalg.norm(rot_torque)):.8g}",
                "force_saturated": str(int(saturated)),
                "torque_saturated": str(int(torque_saturated)),
            }
        )
        if render_video and i % 2 == 0:
            qpos_frames.append(data.qpos.copy())

    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{file_tag}" if file_tag else ""
    ts_path = out_dir / f"{row['variant']}_direct_object_wrench{suffix}_timeseries.csv"
    with ts_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    video_path = out_dir / f"{row['variant']}_direct_object_wrench{suffix}_sanity.mp4"
    if render_video:
        _render_video(model, qpos_frames, video_path, 960, 544)

    support_err = np.asarray([float(item["support_target_err_m"]) for item in rows], dtype=np.float64)
    obj_err = np.asarray([float(item["object_pos_err_m"]) for item in rows], dtype=np.float64)
    force_sat_frac = sat_count / max(1, len(rows))
    torque_sat_frac = torque_sat_count / max(1, len(rows))
    pass_gate = nan_count == 0 and float(np.mean(obj_err)) <= 0.12 and float(np.mean(support_err)) <= 0.08
    return {
        "variant": f"{row['variant']}_direct_object_wrench",
        "source_e028_variant": row["variant"],
        "profile": "direct_object_wrench",
        "support_joint_layout": "none_object_wrench",
        "target_mode": "raw_ref",
        "target_lowpass_tau": "0",
        "target_ramp_time": "0",
        "pos_kp": f"{kp:.8g}",
        "pos_kd": f"{kd:.8g}",
        "rot_kp": f"{rkp:.8g}",
        "rot_kd": f"{rkd:.8g}",
        "force_clamp": f"{force_limit:.8g}",
        "torque_clamp": f"{torque_limit:.8g}",
        "frames": str(len(rows)),
        "support_drift_mean_m": "0",
        "support_drift_max_m": "0",
        "support_target_err_mean_m": f"{float(np.mean(support_err)):.8g}",
        "support_target_err_max_m": f"{float(np.max(support_err)):.8g}",
        "object_pos_err_mean_m": f"{float(np.mean(obj_err)):.8g}",
        "object_pos_err_max_m": f"{float(np.max(obj_err)):.8g}",
        "force_saturation_frac": f"{force_sat_frac:.8g}",
        "torque_saturation_frac": f"{torque_sat_frac:.8g}",
        "nan_count": str(nan_count),
        "pass_gate": "true" if pass_gate else "false",
        "timeseries_path": common.rel_or_abs(ts_path),
        "video_path": common.rel_or_abs(video_path) if render_video else "",
    }


def _write_summary(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    md = path.with_suffix(".md")
    lines = [
        "# E029 D6 Support Load-Path Sanity",
        "",
        f"- Rows: {len(rows)}",
        f"- Pass: {sum(row['pass_gate'] == 'true' for row in rows)}/{len(rows)}",
        f"- CSV: `{common.rel_or_abs(path)}`",
        "",
        "| Variant | target | clamp | drift mean/max | object mean/max | force sat | torque sat | pass |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['variant']}`",
                    f"{row.get('target_mode', '')} tau={row.get('target_lowpass_tau', '')} ramp={row.get('target_ramp_time', '')}",
                    row.get("force_clamp", ""),
                    f"{row['support_drift_mean_m']}/{row['support_drift_max_m']}",
                    f"{row['object_pos_err_mean_m']}/{row['object_pos_err_max_m']}",
                    row["force_saturation_frac"],
                    row["torque_saturation_frac"],
                    row["pass_gate"],
                ]
            )
            + " |"
        )
    lines.append("")
    md.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--subset", choices=["representative", "candidates"], default="representative")
    parser.add_argument(
        "--mode",
        choices=[
            "direct-object-wrench-control",
            "mocap-weld-current",
            "d6-compliant-support",
            "d6-locked-support",
            "free-compliant-support",
            "free-locked-support",
            "connect-compliant-support",
            "multi-triad-support",
            "multi-cross-support",
            "multi-cross-stiff-support",
            "multi-cross-ultra-support",
        ],
        default="d6-compliant-support",
    )
    parser.add_argument("--target-mode", choices=["raw_ref", "smooth_final"], default="raw_ref")
    parser.add_argument("--target-lowpass-tau", type=float, default=0.0)
    parser.add_argument("--target-ramp-time", type=float, default=0.0)
    parser.add_argument("--force-clamp-override", type=float, default=None)
    parser.add_argument("--torque-clamp-override", type=float, default=None)
    parser.add_argument("--pos-kp-override", type=float, default=None)
    parser.add_argument("--pos-kd-override", type=float, default=None)
    parser.add_argument("--rot-kp-override", type=float, default=None)
    parser.add_argument("--rot-kd-override", type=float, default=None)
    parser.add_argument("--tag", default="")
    parser.add_argument("--render-video", action="store_true")
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()
    rows = _row_for_mode(read_manifest(args.manifest), args.mode, args.subset)
    if args.mode == "mocap-weld-current":
        summary = [
            run_mocap_weld_row(
                row,
                out_dir=args.out_dir,
                render_video=args.render_video,
                max_frames=args.max_frames,
                file_tag=args.tag,
            )
            for row in rows
        ]
    elif args.mode == "direct-object-wrench-control":
        summary = [
            run_direct_object_wrench_row(
                row,
                out_dir=args.out_dir,
                render_video=args.render_video,
                max_frames=args.max_frames,
                force_clamp_override=args.force_clamp_override,
                torque_clamp_override=args.torque_clamp_override,
                pos_kp_override=args.pos_kp_override,
                pos_kd_override=args.pos_kd_override,
                rot_kp_override=args.rot_kp_override,
                rot_kd_override=args.rot_kd_override,
                file_tag=args.tag,
            )
            for row in rows
        ]
    else:
        summary = [
            run_row(
                row,
                out_dir=args.out_dir,
                render_video=args.render_video,
                max_frames=args.max_frames,
                target_mode=args.target_mode,
                target_lowpass_tau=args.target_lowpass_tau,
                target_ramp_time=args.target_ramp_time,
                force_clamp_override=args.force_clamp_override,
                torque_clamp_override=args.torque_clamp_override,
                pos_kp_override=args.pos_kp_override,
                pos_kd_override=args.pos_kd_override,
                rot_kp_override=args.rot_kp_override,
                rot_kd_override=args.rot_kd_override,
                file_tag=args.tag,
            )
            for row in rows
        ]
    tag = f"_{args.tag}" if args.tag else ""
    summary_path = args.out_dir / f"{args.subset}_{args.mode}_{args.target_mode}{tag}_summary.csv"
    _write_summary(summary, summary_path)
    print(f"Wrote {common.rel_or_abs(summary_path)}")
    print(f"Wrote {common.rel_or_abs(summary_path.with_suffix('.md'))}")


if __name__ == "__main__":
    main()
