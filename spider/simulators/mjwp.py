# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Simulator for sampling with MuJoCo Warp (mjwarp).

This module provides a minimal MJWP backend that matches the sampling API used by
the generic optimizer pipeline. It intentionally keeps the implementation simple
and robust (no DR groups here; see legacy script for advanced features).
"""

from __future__ import annotations

from dataclasses import dataclass

import loguru
import mujoco
import mujoco_warp as mjwarp
import numpy as np
import torch
import warp as wp

# NOTE: this is a hacky solution to make sure domain randomization works for contact margin. Otherwise, it will create a surrogate memory for all worlds and we cannot override each individual world's contact parameters.
# mjwarp._src.io.MAX_WORLDS = 1024
from spider.config import Config
from spider.math import quat_sub

# Initialize Warp once per process
try:
    wp.init()
except RuntimeError:
    # Already initialized
    pass


@dataclass
class MJWPEnv:
    model_cpu: mujoco.MjModel
    data_cpu: mujoco.MjData
    # Unified data sink always reflecting last step's state
    model_wp: mjwarp.Model
    data_wp: mjwarp.Data
    data_wp_prev: mjwarp.Data
    graph: wp.ScopedCapture.Graph
    # Device alias used for Warp allocations/launches (e.g., "cuda:1" or "cpu")
    device: str
    num_worlds: int


def _compile_step(
    model_wp: mjwarp.Model, data_wp: mjwarp.Data, decimation: int = 1
) -> wp.ScopedCapture.Graph:
    """Warm up and capture a CUDA graph that runs `decimation` × mjwarp.step."""

    def _step_once():
        for _ in range(decimation):
            mjwarp.step(model_wp, data_wp)

    # Capture
    with wp.ScopedCapture() as capture:
        _step_once()
    wp.synchronize()
    return capture.graph


# TODO: define update environment parameter kernel functions, combine them compile step, also add parameter to be modified into MJWPEnv

# --
# Key functions
# --


def setup_mj_model(config: Config) -> mujoco.MjModel:
    model_cpu = mujoco.MjModel.from_xml_path(config.model_path)
    # Path Y: physics_dt for Holosoma alignment (decimation handled in graph capture)
    if config.physics_dt > 0:
        model_cpu.opt.timestep = float(config.physics_dt)
    else:
        model_cpu.opt.timestep = float(config.sim_dt)
    if config.embodiment_type in ["left", "right", "bimanual"]:
        # setup for hand
        model_cpu.opt.iterations = 20
        model_cpu.opt.ls_iterations = 50
        model_cpu.opt.o_solref = [0.02, 1.0]
        model_cpu.opt.o_solimp = [
            0.0,
            0.95,
            0.03,
            0.5,
            2,
        ]  # softer contact for sim2real
        model_cpu.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    elif config.embodiment_type in ["humanoid", "humanoid_object", "dual_humanoid_object"]:
        # setup for humanoid
        model_cpu.opt.iterations = 5
        model_cpu.opt.ls_iterations = 10
        model_cpu.opt.o_solref = [0.02, 1.0]
        model_cpu.opt.o_solimp = [
            0.9,
            0.95,
            0.001,
            0.5,
            2,
        ]  # softer contact for sim2real
        model_cpu.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    # Path Y: override PD gains with Holosoma values
    if getattr(config, "apply_holosoma_pd", False):
        from spider.mujoco_utils import apply_holosoma_g1_pd
        n = apply_holosoma_g1_pd(model_cpu, verbose=False)
        loguru.logger.info(f"Applied Holosoma G1 PD to {n} actuators")
    return model_cpu


def setup_env(config: Config, ref_data: tuple[torch.Tensor, ...]) -> MJWPEnv:
    """Setup and reset the environment backed by MJWP.
    Returns an MJWPEnv with captured graph.
    """
    qpos_ref, qvel_ref, ctrl_ref, contact_ref, contact_pos_ref = ref_data
    qpos_init = qpos_ref[0]

    # CPU model/data
    model_cpu = setup_mj_model(config)
    data_cpu = mujoco.MjData(model_cpu)
    # Seed initial state
    arrs = (qpos_init, qvel_ref[0], ctrl_ref[0])
    data_cpu.qpos[:] = arrs[0].detach().cpu().numpy()
    data_cpu.qvel[:] = arrs[1].detach().cpu().numpy()
    data_cpu.ctrl[:] = arrs[2].detach().cpu().numpy()
    mujoco.mj_step(model_cpu, data_cpu)

    # Move to Warp (batched worlds)
    # Set Warp default device to match config to ensure kernels/modules load on it
    wp.set_device(str(config.device))
    # Build default model/data/graph on the configured device
    dev = str(config.device)
    with wp.ScopedDevice(dev):
        default_model_wp = mjwarp.put_model(model_cpu)
        # pair_margin_override_np = (
        #     np.zeros((int(config.num_samples), model_cpu.npair)).astype(np.float32)
        #     + 0.01
        # )
        # pair_margin_override_wp = wp.from_numpy(
        #     pair_margin_override_np, dtype=wp.float32, device=dev
        # )
        # default_model_wp.pair_margin = pair_margin_override_wp

        default_data_wp = mjwarp.put_data(
            model_cpu,
            data_cpu,
            nworld=int(config.num_samples),
            nconmax=int(config.nconmax_per_env),
            njmax=int(config.njmax_per_env),
        )
        data_wp_prev = mjwarp.put_data(
            model_cpu,
            data_cpu,
            nworld=int(config.num_samples),
            nconmax=int(config.nconmax_per_env),
            njmax=int(config.njmax_per_env),
        )
        default_graph = _compile_step(default_model_wp, default_data_wp, decimation=config.sim_decimation)

    # Initialize env; default active is main
    env = MJWPEnv(
        model_cpu=model_cpu,
        data_cpu=data_cpu,
        model_wp=default_model_wp,
        data_wp=default_data_wp,
        data_wp_prev=data_wp_prev,
        graph=default_graph,
        device=dev,
        num_worlds=int(config.num_samples),
    )

    # Load mocap partner trajectory if configured
    if config.mocap_partner_trajectory:
        _load_mocap_partner(config, env)

    return env


def _weight_diff_qpos(config: Config) -> torch.Tensor:
    w = torch.ones(config.nv, device=config.device)
    if config.embodiment_type == "bimanual":
        half_dof = int(config.nu // 2)
        w[:3] = config.base_pos_rew_scale
        w[3:6] = config.base_rot_rew_scale
        w[6:half_dof] = config.joint_rew_scale
        w[half_dof : half_dof + 3] = config.base_pos_rew_scale
        w[half_dof + 3 : half_dof + 6] = config.base_rot_rew_scale
        w[half_dof + 6 : config.nu] = config.joint_rew_scale
        # object: weights live in nv-space (6-dim per freejoint), regardless of nq_obj
        w[-12:-9] = config.pos_rew_scale
        w[-9:-6] = config.rot_rew_scale
        w[-6:-3] = config.pos_rew_scale
        w[-3:] = config.rot_rew_scale
    elif config.embodiment_type in ["right", "left"]:
        w[:3] = config.base_pos_rew_scale
        w[3:6] = config.base_rot_rew_scale
        w[6 : config.nu] = config.joint_rew_scale
        w[-6:-3] = config.pos_rew_scale
        w[-3:] = config.rot_rew_scale
    elif config.embodiment_type in ["humanoid"]:  # humanoid robot
        # robot pos and rot
        w[:3] = config.pos_rew_scale
        w[3:6] = config.rot_rew_scale
        # robot joint
        w[6:] = config.joint_rew_scale
    elif config.embodiment_type in ["humanoid_object"]:
        # robot pos and rot
        w[:3] = config.base_pos_rew_scale
        w[3:6] = config.base_rot_rew_scale
        # robot joint
        w[6:-6] = config.joint_rew_scale
        # object pos and rot
        w[-6:-3] = config.pos_rew_scale
        w[-3:] = config.rot_rew_scale
    elif config.embodiment_type == "dual_humanoid_object":
        # Two robots + one shared object
        # nv layout: robot1_base(3)+rot(3)+joints(29) + robot2_base(3)+rot(3)+joints(29) + obj_pos(3)+rot(3)
        nv_robot = (config.nv - 6) // 2  # 35 per robot
        # robot1
        w[:3] = config.base_pos_rew_scale
        w[3:6] = config.base_rot_rew_scale
        w[6:nv_robot] = config.joint_rew_scale
        # robot2
        w[nv_robot:nv_robot + 3] = config.base_pos_rew_scale
        w[nv_robot + 3:nv_robot + 6] = config.base_rot_rew_scale
        w[nv_robot + 6:2 * nv_robot] = config.joint_rew_scale
        # object
        w[-6:-3] = config.pos_rew_scale
        w[-3:] = config.rot_rew_scale
    else:
        raise ValueError(f"Invalid embodiment_type: {config.embodiment_type}")
    return w


def _diff_qpos(
    config: Config, qpos_sim: torch.Tensor, qpos_ref: torch.Tensor
) -> torch.Tensor:
    """Compute the difference between qpos_sim and qpos_ref
    TODO: replace with mujoco built-in function, not sure how to call warp internal function yet.
    """
    batch_size = qpos_sim.shape[0]
    qpos_diff = torch.zeros((batch_size, config.nv), device=config.device)
    if config.embodiment_type == "bimanual":
        if config.nq_obj == 12:
            qpos_diff[:, :-12] = qpos_sim[:, :-12] - qpos_ref[:, :-12]
            qpos_diff[:, -12:-9] = qpos_sim[:, -12:-9] - qpos_ref[:, -12:-9]
            qpos_diff[:, -9:-6] = qpos_sim[:, -9:-6] - qpos_ref[:, -9:-6]
            qpos_diff[:, -6:-3] = qpos_sim[:, -6:-3] - qpos_ref[:, -6:-3]
            qpos_diff[:, -3:] = qpos_sim[:, -3:] - qpos_ref[:, -3:]
            return qpos_diff
        # joint
        qpos_diff[:, :-12] = qpos_sim[:, :-14] - qpos_ref[:, :-14]
        # position
        qpos_diff[:, -12:-9] = qpos_sim[:, -14:-11] - qpos_ref[:, -14:-11]
        qpos_diff[:, -6:-3] = qpos_sim[:, -7:-4] - qpos_ref[:, -7:-4]
        # rotation
        qpos_diff[:, -9:-6] = quat_sub(qpos_sim[:, -11:-7], qpos_ref[:, -11:-7])
        qpos_diff[:, -3:] = quat_sub(qpos_sim[:, -4:], qpos_ref[:, -4:])
    elif config.embodiment_type in ["right", "left"]:
        if config.nq_obj == 6:
            qpos_diff[:, :-6] = qpos_sim[:, :-6] - qpos_ref[:, :-6]
            qpos_diff[:, -6:-3] = qpos_sim[:, -6:-3] - qpos_ref[:, -6:-3]
            qpos_diff[:, -3:] = qpos_sim[:, -3:] - qpos_ref[:, -3:]
            return qpos_diff
        # joint
        qpos_diff[:, :-6] = qpos_sim[:, :-7] - qpos_ref[:, :-7]
        # position
        qpos_diff[:, -6:-3] = qpos_sim[:, -7:-4] - qpos_ref[:, -7:-4]
        # rotation
        qpos_diff[:, -3:] = quat_sub(qpos_sim[:, -4:], qpos_ref[:, -4:])
    elif config.embodiment_type in ["humanoid"]:
        # joint
        qpos_diff[:, 6:] = qpos_sim[:, 7:] - qpos_ref[:, 7:]
        # position
        qpos_diff[:, :3] = qpos_sim[:, :3] - qpos_ref[:, :3]
        # rotation
        qpos_diff[:, 3:6] = quat_sub(qpos_sim[:, 3:7], qpos_ref[:, 3:7])
    elif config.embodiment_type in ["humanoid_object"]:
        nq_obj = config.nq_obj  # 7 (freejoint) or 6 (contact_guidance)
        qpos_humanoid = qpos_sim[:, :-nq_obj]
        qpos_object = qpos_sim[:, -nq_obj:]
        qpos_ref_humanoid = qpos_ref[:, :-nq_obj]
        qpos_ref_object = qpos_ref[:, -nq_obj:]
        # position
        qpos_diff[:, :3] = qpos_humanoid[:, :3] - qpos_ref_humanoid[:, :3]
        # rotation
        qpos_diff[:, 3:6] = quat_sub(qpos_humanoid[:, 3:7], qpos_ref_humanoid[:, 3:7])
        # joint
        qpos_diff[:, 6:-6] = qpos_humanoid[:, 7:] - qpos_ref_humanoid[:, 7:]
        # object
        if nq_obj == 7:
            # freejoint: pos(3) + quat(4)
            qpos_diff[:, -6:-3] = qpos_object[:, :3] - qpos_ref_object[:, :3]
            qpos_diff[:, -3:] = quat_sub(qpos_object[:, 3:7], qpos_ref_object[:, 3:7])
        else:
            # contact_guidance: pos(3) + rpy(3), all direct subtraction
            qpos_diff[:, -6:-3] = qpos_object[:, :3] - qpos_ref_object[:, :3]
            qpos_diff[:, -3:] = qpos_object[:, 3:6] - qpos_ref_object[:, 3:6]
    elif config.embodiment_type == "dual_humanoid_object":
        nq_obj = config.nq_obj  # 7 (freejoint) or 6 (contact_guidance)
        nq_robot = (config.nq - nq_obj) // 2  # 36 per robot
        nv_robot = (config.nv - 6) // 2  # 35 per robot
        # robot1: nq[0:nq_robot], robot2: nq[nq_robot:2*nq_robot], obj: nq[-nq_obj:]
        r1 = qpos_sim[:, :nq_robot]
        r2 = qpos_sim[:, nq_robot:2 * nq_robot]
        obj = qpos_sim[:, -nq_obj:]
        r1_ref = qpos_ref[:, :nq_robot]
        r2_ref = qpos_ref[:, nq_robot:2 * nq_robot]
        obj_ref = qpos_ref[:, -nq_obj:]
        # robot1: base pos/rot/joints
        qpos_diff[:, :3] = r1[:, :3] - r1_ref[:, :3]
        qpos_diff[:, 3:6] = quat_sub(r1[:, 3:7], r1_ref[:, 3:7])
        qpos_diff[:, 6:nv_robot] = r1[:, 7:] - r1_ref[:, 7:]
        # robot2: base pos/rot/joints
        qpos_diff[:, nv_robot:nv_robot + 3] = r2[:, :3] - r2_ref[:, :3]
        qpos_diff[:, nv_robot + 3:nv_robot + 6] = quat_sub(r2[:, 3:7], r2_ref[:, 3:7])
        qpos_diff[:, nv_robot + 6:2 * nv_robot] = r2[:, 7:] - r2_ref[:, 7:]
        # object
        if nq_obj == 7:
            qpos_diff[:, -6:-3] = obj[:, :3] - obj_ref[:, :3]
            qpos_diff[:, -3:] = quat_sub(obj[:, 3:7], obj_ref[:, 3:7])
        else:
            qpos_diff[:, -6:-3] = obj[:, :3] - obj_ref[:, :3]
            qpos_diff[:, -3:] = obj[:, 3:6] - obj_ref[:, 3:6]
    else:
        raise ValueError(f"Invalid embodiment_type: {config.embodiment_type}")
    return qpos_diff


# ---------------------------------------------------------------------------
# E035: Local-frame tracking helpers (ported from HDMI)
# ---------------------------------------------------------------------------

def _lf_yaw_quat(q: torch.Tensor) -> torch.Tensor:
    """Extract yaw-only rotation from quaternion. q: (..., 4) wxyz."""
    w, x, y, z = q.unbind(-1)
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return torch.stack(
        [torch.cos(yaw / 2), torch.zeros_like(yaw), torch.zeros_like(yaw), torch.sin(yaw / 2)],
        dim=-1,
    )


def _lf_quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    """Quaternion conjugate. q: (..., 4) wxyz."""
    return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)


def _lf_quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Hamilton product. q1, q2: (..., 4) wxyz."""
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dim=-1)


def _lf_quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vector v by quaternion q. q: (..., 4) wxyz, v: (..., 3)."""
    t = 2.0 * torch.cross(q[..., 1:], v, dim=-1)
    return v + q[..., :1] * t + torch.cross(q[..., 1:], t, dim=-1)


def _lf_quat_apply_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vector v by inverse of quaternion q."""
    return _lf_quat_apply(_lf_quat_conjugate(q), v)


def _lf_axis_angle_from_quat(q: torch.Tensor) -> torch.Tensor:
    """Convert quaternion to axis-angle. q: (..., 4) wxyz -> (..., 3)."""
    sin_half = torch.norm(q[..., 1:], dim=-1, keepdim=True).clamp(min=1e-8)
    cos_half = q[..., :1]
    angle = 2.0 * torch.atan2(sin_half, cos_half)
    axis = q[..., 1:] / sin_half
    return axis * angle


def _local_pos_tracking(
    xpos_batch: torch.Tensor,
    xquat_batch: torch.Tensor,
    body_ids: list[int],
    root_id: int,
    ref_body_pos: torch.Tensor,
    ref_root_pos: torch.Tensor,
    ref_root_quat: torch.Tensor,
    sigma: float,
) -> torch.Tensor:
    """Position tracking in root-yaw-relative frame. Returns (N,).
    ref_body_pos: (B, 3), ref_root_pos: (3,), ref_root_quat: (4,)
    """
    N = xpos_batch.shape[0]
    body_pos = xpos_batch[:, body_ids, :]  # (N, B, 3)
    root_pos = xpos_batch[:, root_id, :]  # (N, 3)
    root_quat = xquat_batch[:, root_id, :]  # (N, 4)
    B = len(body_ids)

    root_pos_xy = root_pos.clone()
    root_pos_xy[..., 2] = 0.0
    root_quat_yaw = _lf_yaw_quat(root_quat)  # (N, 4)

    ref_root_xy = ref_root_pos.clone()
    ref_root_xy[2] = 0.0
    ref_root_quat_yaw = _lf_yaw_quat(ref_root_quat.unsqueeze(0)).squeeze(0)  # (4,)

    # Expand for batch and body dims
    rp = root_pos_xy.unsqueeze(1).expand(-1, B, -1)  # (N, B, 3)
    rq = root_quat_yaw.unsqueeze(1).expand(-1, B, -1)  # (N, B, 4)
    ref_rp = ref_root_xy.unsqueeze(0).unsqueeze(0).expand(N, B, -1)  # (N, B, 3)
    ref_rq = ref_root_quat_yaw.unsqueeze(0).unsqueeze(0).expand(N, B, -1)  # (N, B, 4)

    body_local = _lf_quat_apply_inverse(rq, body_pos - rp)  # (N, B, 3)
    ref_body_pos_exp = ref_body_pos.unsqueeze(0).expand(N, -1, -1)  # (N, B, 3)
    ref_local = _lf_quat_apply_inverse(ref_rq, ref_body_pos_exp - ref_rp)  # (N, B, 3)

    error = (ref_local - body_local).norm(dim=-1).clamp_min(0.0)  # (N, B)
    return torch.exp(-error.mean(dim=1) / sigma)


def _local_ori_tracking(
    xquat_batch: torch.Tensor,
    body_ids: list[int],
    root_id: int,
    ref_body_quat: torch.Tensor,
    ref_root_quat: torch.Tensor,
    sigma: float,
) -> torch.Tensor:
    """Orientation tracking in root-yaw-relative frame. Returns (N,).
    ref_body_quat: (B, 4), ref_root_quat: (4,)
    """
    N = xquat_batch.shape[0]
    B = len(body_ids)
    body_quat = xquat_batch[:, body_ids, :]  # (N, B, 4)
    root_quat = xquat_batch[:, root_id, :]  # (N, 4)

    root_yaw = _lf_yaw_quat(root_quat)  # (N, 4)
    ref_root_yaw = _lf_yaw_quat(ref_root_quat.unsqueeze(0)).squeeze(0)  # (4,)

    rq = root_yaw.unsqueeze(1).expand(-1, B, -1)  # (N, B, 4)
    ref_rq = ref_root_yaw.unsqueeze(0).unsqueeze(0).expand(N, B, -1)  # (N, B, 4)

    body_local = _lf_quat_mul(_lf_quat_conjugate(rq), body_quat)  # (N, B, 4)
    ref_body_quat_exp = ref_body_quat.unsqueeze(0).expand(N, -1, -1)  # (N, B, 4)
    ref_local = _lf_quat_mul(_lf_quat_conjugate(ref_rq), ref_body_quat_exp)  # (N, B, 4)

    diff = _lf_quat_mul(_lf_quat_conjugate(ref_local), body_local)  # (N, B, 4)
    error = _lf_axis_angle_from_quat(diff).norm(dim=-1).clamp_min(0.0)  # (N, B)
    return torch.exp(-error.mean(dim=1) / sigma)


def get_reward(
    config: Config,
    env: MJWPEnv,
    ref: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    """Non-terminal step reward for MJWP batched worlds.
    ref is a tuple: (qpos_ref, qvel_ref, ctrl_ref, contact_ref, contact_pos_ref,
                     body_xpos_ref) where body_xpos_ref is (K, 3) per timestep
    Returns (N,)

    TODO: move reward computation to task-specific module
    """
    # Unpack with backward compatibility (5-tuple legacy, 6-tuple E018, 7-tuple E034, 8-tuple E035, 9-tuple E040)
    approach_mask_val = 1.0
    body_xquat_ref = None
    contact_target_dynamic = None
    if len(ref) == 5:
        qpos_ref, qvel_ref, ctrl_ref, contact_ref, contact_pos_ref = ref
        body_xpos_ref = None
    elif len(ref) == 6:
        qpos_ref, qvel_ref, ctrl_ref, contact_ref, contact_pos_ref, body_xpos_ref = ref
    elif len(ref) == 7:
        qpos_ref, qvel_ref, ctrl_ref, contact_ref, contact_pos_ref, body_xpos_ref, approach_mask_val = ref
    elif len(ref) == 8:
        qpos_ref, qvel_ref, ctrl_ref, contact_ref, contact_pos_ref, body_xpos_ref, approach_mask_val, body_xquat_ref = ref
    else:
        qpos_ref, qvel_ref, ctrl_ref, contact_ref, contact_pos_ref, body_xpos_ref, approach_mask_val, body_xquat_ref, contact_target_dynamic = ref
    qpos_sim = wp.to_torch(env.data_wp.qpos)
    qvel_sim = wp.to_torch(env.data_wp.qvel)
    N = qpos_sim.shape[0]

    # weighted qpos tracking
    qpos_diff = _diff_qpos(
        config, qpos_sim, qpos_ref.unsqueeze(0).repeat(N, 1)
    )
    qpos_weight = _weight_diff_qpos(config)
    delta_qpos = qpos_diff * qpos_weight
    qpos_dist = torch.norm(delta_qpos, p=2, dim=1)
    qvel_dist = torch.norm(qvel_sim - qvel_ref, p=2, dim=1)

    qpos_rew = (
        config.qpos_reward_scale * torch.exp(-qpos_dist / config.qpos_reward_sigma)
        if config.use_bounded_qpos_reward
        else -qpos_dist * 1.0
    )
    qvel_rew = -config.vel_rew_scale * qvel_dist * 1.0

    # E035: local-frame body tracking (replaces qpos_rew when enabled)
    local_frame_rew = torch.zeros(N, device=config.device)
    if config.use_local_frame_reward and body_xpos_ref is not None and body_xquat_ref is not None:
        xpos_sim = wp.to_torch(env.data_wp.xpos)  # (N, nbody, 3)
        xquat_sim = wp.to_torch(env.data_wp.xquat)  # (N, nbody, 4) wxyz

        root_id = 1  # pelvis
        ref_root_pos = body_xpos_ref[root_id]  # (3,)
        ref_root_quat = body_xquat_ref[root_id]  # (4,)

        upper_ids = config.local_frame_upper_ids
        lower_ids = config.local_frame_lower_ids

        upper_pos_rew = _local_pos_tracking(
            xpos_sim, xquat_sim, upper_ids, root_id,
            body_xpos_ref[upper_ids], ref_root_pos, ref_root_quat,
            config.local_frame_pos_sigma,
        )
        upper_ori_rew = _local_ori_tracking(
            xquat_sim, upper_ids, root_id,
            body_xquat_ref[upper_ids], ref_root_quat,
            config.local_frame_ori_sigma,
        )
        lower_pos_rew = _local_pos_tracking(
            xpos_sim, xquat_sim, lower_ids, root_id,
            body_xpos_ref[lower_ids], ref_root_pos, ref_root_quat,
            config.local_frame_pos_sigma,
        )
        lower_ori_rew = _local_ori_tracking(
            xquat_sim, lower_ids, root_id,
            body_xquat_ref[lower_ids], ref_root_quat,
            config.local_frame_ori_sigma,
        )

        # Root global tracking
        root_pos_err = (xpos_sim[:, root_id] - ref_root_pos.unsqueeze(0)).norm(dim=-1)
        root_pos_rew = torch.exp(-root_pos_err / config.local_frame_root_sigma)

        root_quat_sim = xquat_sim[:, root_id]  # (N, 4)
        root_quat_ref = ref_root_quat.unsqueeze(0).expand(N, -1)
        root_diff = _lf_quat_mul(_lf_quat_conjugate(root_quat_ref), root_quat_sim)
        root_ori_err = _lf_axis_angle_from_quat(root_diff).norm(dim=-1)
        root_ori_rew = torch.exp(-root_ori_err / config.local_frame_root_sigma)

        # Joint tracking from qpos (joint angles only, not base)
        if config.embodiment_type == "humanoid_object":
            jt_sim = qpos_sim[:, 7:-7] if qpos_sim.shape[1] > 14 else qpos_sim[:, 7:]
            jt_ref = qpos_ref[7:-7] if qpos_ref.shape[0] > 14 else qpos_ref[7:]
            jt_err = (jt_sim - jt_ref.unsqueeze(0)).abs().mean(dim=1)
        else:
            jt_err = torch.zeros(N, device=config.device)
        joint_rew = torch.exp(-jt_err / config.local_frame_joint_sigma)

        W = config.local_frame_w_track
        local_frame_rew = W * (
            upper_pos_rew + upper_ori_rew
            + lower_pos_rew + lower_ori_rew
            + root_pos_rew + root_ori_rew
            + joint_rew
        )  # max = W * 7

        # Replace qpos_rew with local_frame_rew
        qpos_rew = local_frame_rew

    # contact reward
    if config.contact_rew_scale > 0.0 and len(config.contact_site_ids) > 0:
        site_xpos_torch = wp.to_torch(env.data_wp.site_xpos)
        contact_pos = site_xpos_torch[:, config.contact_site_ids]
        contact_dist = torch.norm(contact_pos - contact_pos_ref, p=2, dim=-1)
        contact_dist_masked = contact_dist * contact_ref.unsqueeze(0)
        contact_rew = -contact_dist_masked.sum(dim=1)
    else:
        contact_rew = 0.0

    # E018: task-space body world-position tracking (DynaRetarget Table II)
    task_body_rew = torch.zeros(N, device=config.device)
    if (
        config.task_body_rew_scale > 0.0
        and config.task_body_ids
        and body_xpos_ref is not None
        and body_xpos_ref.shape[0] > 0
    ):
        xpos_sim = wp.to_torch(env.data_wp.xpos)  # (N, nbody, 3)
        body_pos_sim = xpos_sim[:, config.task_body_ids]  # (N, K, 3)
        body_weights = torch.tensor(
            config.task_body_weights, device=config.device, dtype=body_pos_sim.dtype
        )  # (K,)
        # body_xpos_ref is (K, 3) for this timestep
        err = ((body_pos_sim - body_xpos_ref.unsqueeze(0)) ** 2).sum(dim=-1)  # (N, K)
        task_body_rew = -config.task_body_rew_scale * (err * body_weights).sum(dim=1)

    # E018: separate object position/orientation tracking with high weight
    task_obj_rew = torch.zeros(N, device=config.device)
    if (
        config.task_obj_pos_rew_scale > 0.0 or config.task_obj_rot_rew_scale > 0.0
    ) and config.embodiment_type in [
        "humanoid_object",
        "dual_humanoid_object",
        "bimanual",
        "right",
        "left",
    ]:
        nq_obj = config.nq_obj
        if nq_obj == 7:
            obj_pos_sim = qpos_sim[:, -7:-4]
            obj_pos_ref = qpos_ref[-7:-4].unsqueeze(0)
            pos_err = ((obj_pos_sim - obj_pos_ref) ** 2).sum(dim=-1)
            task_obj_rew = task_obj_rew - config.task_obj_pos_rew_scale * pos_err
            if config.task_obj_rot_rew_scale > 0.0:
                obj_quat_sim = qpos_sim[:, -4:]
                obj_quat_ref = qpos_ref[-4:].unsqueeze(0).repeat(N, 1)
                rot_err = (quat_sub(obj_quat_sim, obj_quat_ref) ** 2).sum(dim=-1)
                task_obj_rew = task_obj_rew - config.task_obj_rot_rew_scale * rot_err
        elif nq_obj == 6:
            obj_pos_sim = qpos_sim[:, -6:-3]
            obj_pos_ref = qpos_ref[-6:-3].unsqueeze(0)
            pos_err = ((obj_pos_sim - obj_pos_ref) ** 2).sum(dim=-1)
            task_obj_rew = task_obj_rew - config.task_obj_pos_rew_scale * pos_err
            if config.task_obj_rot_rew_scale > 0.0:
                obj_euler_sim = qpos_sim[:, -3:]
                obj_euler_ref = qpos_ref[-3:].unsqueeze(0)
                rot_err = ((obj_euler_sim - obj_euler_ref) ** 2).sum(dim=-1)
                task_obj_rew = task_obj_rew - config.task_obj_rot_rew_scale * rot_err

    # E018: interaction reward (Harmanoid Eq.15) — match relative offsets
    # between pairs of bodies in task_body_ids
    interact_rew = torch.zeros(N, device=config.device)
    if (
        config.interact_rew_scale > 0.0
        and config.interact_pairs
        and config.task_body_ids
        and body_xpos_ref is not None
        and body_xpos_ref.shape[0] > 0
    ):
        xpos_sim = wp.to_torch(env.data_wp.xpos)
        body_pos_sim = xpos_sim[:, config.task_body_ids]  # (N, K, 3)
        pair_err_total = torch.zeros(N, device=config.device)
        for ia, ib in config.interact_pairs:
            delta_sim = body_pos_sim[:, ia] - body_pos_sim[:, ib]  # (N, 3)
            delta_ref = body_xpos_ref[ia] - body_xpos_ref[ib]  # (3,)
            pair_err_total = (
                pair_err_total + ((delta_sim - delta_ref.unsqueeze(0)) ** 2).sum(dim=-1)
            )
        interact_rew = config.interact_rew_scale * torch.exp(
            -config.interact_sigma * pair_err_total
        )

    # E025: hand approach reward — exp decay of hand-to-object-surface distance
    hand_approach_rew = torch.zeros(N, device=config.device)
    if config.hand_approach_rew_scale > 0.0 and config.hand_approach_body_ids:
        obj_body_id = mujoco.mj_name2id(
            env.model_cpu, mujoco.mjtObj.mjOBJ_BODY, "object"
        )
        if obj_body_id != -1 and config.hand_approach_obj_half_extents:
            xpos_sim = wp.to_torch(env.data_wp.xpos)  # (N, nbody, 3)
            hand_pos = xpos_sim[:, config.hand_approach_body_ids]  # (N, K_hand, 3)
            obj_pos = xpos_sim[:, obj_body_id:obj_body_id + 1]  # (N, 1, 3)
            # Surface distance: clamp(|delta| - half_ext, min=0) then norm
            half_ext = torch.tensor(
                config.hand_approach_obj_half_extents,
                device=config.device,
                dtype=hand_pos.dtype,
            )
            delta = torch.abs(hand_pos - obj_pos)  # (N, K_hand, 3)
            surface_dist = torch.clamp(delta - half_ext, min=0.0)  # (N, K_hand, 3)
            # Min distance over hands (reward best hand)
            dist_per_hand = surface_dist.norm(dim=-1)  # (N, K_hand)
            min_dist = dist_per_hand.min(dim=1).values  # (N,)
            hand_approach_rew = approach_mask_val * config.hand_approach_rew_scale * torch.exp(
                -config.hand_approach_sigma * min_dist
            )

    # E037: contact mask-gated reward — HDMI-style proximity with mask gate
    contact_mask_rew = torch.zeros(N, device=config.device)
    if config.contact_mask_rew_scale > 0.0 and config.hand_approach_body_ids:
        obj_body_id = mujoco.mj_name2id(
            env.model_cpu, mujoco.mjtObj.mjOBJ_BODY, "object"
        )
        if obj_body_id != -1 and config.hand_approach_obj_half_extents:
            xpos_sim = wp.to_torch(env.data_wp.xpos)  # (N, nbody, 3)
            hand_pos = xpos_sim[:, config.hand_approach_body_ids]  # (N, K, 3)
            obj_pos = xpos_sim[:, obj_body_id : obj_body_id + 1]  # (N, 1, 3)
            half_ext = torch.tensor(
                config.hand_approach_obj_half_extents,
                device=config.device,
                dtype=hand_pos.dtype,
            )
            delta = torch.abs(hand_pos - obj_pos)
            surface_dist = torch.clamp(delta - half_ext, min=0.0)
            dist_per_hand = surface_dist.norm(dim=-1)  # (N, K)
            min_dist = dist_per_hand.min(dim=1).values  # (N,)
            # mask=1 → exp proximity reward; mask=0 → baseline (CEM ignores)
            gain = config.contact_mask_rew_scale
            mask = approach_mask_val  # scalar or (N,) from ref[6], per-timestep
            proximity = gain * torch.exp(-min_dist / config.contact_mask_rew_sigma)
            baseline = config.contact_mask_rew_baseline
            contact_mask_rew = mask * proximity + (1.0 - mask) * baseline

    # E039: HDMI-aligned contact — predefined target points + per-EEF + mask gate
    # E040: dynamic per-frame target support
    contact_hdmi_rew = torch.zeros(N, device=config.device)
    if config.contact_hdmi_gain > 0.0 and (config.contact_hdmi_target_left or contact_target_dynamic is not None):
        obj_body_id = mujoco.mj_name2id(
            env.model_cpu, mujoco.mjtObj.mjOBJ_BODY, "object"
        )
        if obj_body_id != -1 and config.hand_approach_body_ids:
            xpos_sim = wp.to_torch(env.data_wp.xpos)  # (N, nbody, 3)
            xquat_sim = wp.to_torch(env.data_wp.xquat)  # (N, nbody, 4) wxyz
            obj_pos = xpos_sim[:, obj_body_id]  # (N, 3)
            obj_quat = xquat_sim[:, obj_body_id]  # (N, 4) wxyz

            eef_bids = config.hand_approach_body_ids  # [left_wrist, right_wrist]
            eef_offset = torch.tensor(config.contact_hdmi_eef_offset, device=config.device, dtype=obj_pos.dtype)

            # E040: choose between dynamic per-frame targets and fixed targets
            if contact_target_dynamic is not None:
                # contact_target_dynamic is (n_eef, 3) for this timestep (already indexed by t)
                targets = [contact_target_dynamic[ei] for ei in range(len(eef_bids))]
            else:
                targets = [
                    torch.tensor(config.contact_hdmi_target_left, device=config.device, dtype=obj_pos.dtype),
                    torch.tensor(config.contact_hdmi_target_right, device=config.device, dtype=obj_pos.dtype),
                ]

            per_eef_rew = []
            for ei, (bid, target_off) in enumerate(zip(eef_bids, targets)):
                # Target in world = obj_pos + quat_apply(obj_quat, target_offset)
                target_world = obj_pos + _lf_quat_apply(obj_quat, target_off.unsqueeze(0).expand(N, -1))
                # EEF contact point = eef_pos + quat_apply(eef_quat, eef_offset)
                eef_pos = xpos_sim[:, bid]  # (N, 3)
                eef_quat = xquat_sim[:, bid]  # (N, 4)
                contact_point = eef_pos + _lf_quat_apply(eef_quat, eef_offset.unsqueeze(0).expand(N, -1))
                # Distance and exp reward
                dist = (target_world - contact_point).norm(dim=-1)  # (N,)
                pos_rew = torch.exp(-dist / config.contact_hdmi_sigma)
                per_eef_rew.append(pos_rew)

            # Stack per-EEF rewards: (N, 2)
            rew_stack = torch.stack(per_eef_rew, dim=1)
            mask = approach_mask_val
            gain = config.contact_hdmi_gain
            # HDMI formula: mask=1 → gain*pos_rew, mask=0 → 1.0
            contact_hdmi_rew = (rew_stack * mask * gain + (1.0 - mask)).mean(dim=1)

    reward = qpos_rew + qvel_rew + contact_rew + task_body_rew + task_obj_rew + interact_rew + hand_approach_rew + contact_mask_rew + contact_hdmi_rew

    # E034: stability penalty — penalize when pelvis z drops below threshold
    stability_penalty = torch.zeros(N, device=config.device)
    if config.stability_penalty_scale > 0.0:
        xpos_sim = wp.to_torch(env.data_wp.xpos)  # (N, nbody, 3)
        pelvis_z = xpos_sim[:, 1, 2]  # body 1 is typically pelvis/torso
        below = torch.clamp(config.stability_penalty_threshold - pelvis_z, min=0.0)
        stability_penalty = -config.stability_penalty_scale * below
        reward = reward + stability_penalty

    info = {
        "qpos_dist": qpos_dist,
        "qvel_dist": qvel_dist,
        "qpos_rew": qpos_rew,
        "qvel_rew": qvel_rew,
        "task_body_rew": task_body_rew,
        "task_obj_rew": task_obj_rew,
        "interact_rew": interact_rew,
        "hand_approach_rew": hand_approach_rew,
    }
    return reward, info


def get_terminal_reward(
    config: Config,
    env: MJWPEnv,
    ref_slice: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    """Terminal reward focusing on object tracking."""
    # return config.terminal_rew_scale * get_reward(config, env, ref_slice)
    # qpos_ref, qvel_ref, ctrl_ref, contact_ref, _ = ref_slice
    # qpos_sim = wp.to_torch(env.data_wp.qpos)
    # qpos_weight = torch.zeros(qpos_sim.shape[1], device=config.device)
    # if config.embodiment_type == "bimanual":
    #     qpos_weight[-14:-11] = config.pos_rew_scale
    #     qpos_weight[-11:-7] = config.rot_rew_scale
    #     qpos_weight[-7:-4] = config.pos_rew_scale
    #     qpos_weight[-4:] = config.rot_rew_scale
    # elif config.embodiment_type in ["right", "left"]:
    #     qpos_weight[-7:-4] = config.pos_rew_scale
    #     qpos_weight[-4:] = config.rot_rew_scale
    # elif config.embodiment_type in ["CMU", "DanceDB"]:
    #     qpos_weight[:3] = config.pos_rew_scale
    #     qpos_weight[3:7] = config.rot_rew_scale
    # else:
    #     raise ValueError(f"Invalid embodiment_type: {config.embodiment_type}")
    # delta_qpos = (qpos_sim - qpos_ref) * qpos_weight
    # cost_object = config.terminal_rew_scale * torch.sum(delta_qpos**2, dim=1)

    rew, info = get_reward(config, env, ref_slice)
    terminal_rew = config.terminal_rew_scale * rew
    return terminal_rew, info


def get_terminate(
    config: Config, env: MJWPEnv, ref_slice: tuple[torch.Tensor, ...]
) -> torch.Tensor:
    # compute object position and orientation error, compare to thereshold
    qpos_sim = wp.to_torch(env.data_wp.qpos)
    # Tolerate both legacy 5-tuple and E018 6-tuple (with body_xpos_ref).
    qpos_ref = ref_slice[0]
    qvel_ref = ref_slice[1]
    ctrl_ref = ref_slice[2]
    contact_ref = ref_slice[3]
    _contact_pos_ref = ref_slice[4]
    if config.embodiment_type == "bimanual":
        if config.nq_obj == 12:
            right_obj_pos = qpos_sim[:, -12:-9]
            right_obj_pos_ref = qpos_ref[-12:-9].unsqueeze(0)
            right_obj_pos_error = torch.norm(
                right_obj_pos - right_obj_pos_ref, p=2, dim=1
            )
            right_obj_rot = qpos_sim[:, -9:-6]
            right_obj_rot_ref = qpos_ref[-9:-6].unsqueeze(0)
            right_obj_rot_error = torch.norm(
                right_obj_rot - right_obj_rot_ref, p=2, dim=1
            )
            left_obj_pos = qpos_sim[:, -6:-3]
            left_obj_pos_ref = qpos_ref[-6:-3].unsqueeze(0)
            left_obj_pos_error = torch.norm(left_obj_pos - left_obj_pos_ref, p=2, dim=1)
            left_obj_rot = qpos_sim[:, -3:]
            left_obj_rot_ref = qpos_ref[-3:].unsqueeze(0)
            left_obj_rot_error = torch.norm(left_obj_rot - left_obj_rot_ref, p=2, dim=1)
            if torch.all(right_obj_pos_ref.abs() < 1e-4):
                right_obj_pos_error *= 0.0
                right_obj_rot_error *= 0.0
            if torch.all(left_obj_pos_ref.abs() < 1e-4):
                left_obj_pos_error *= 0.0
                left_obj_rot_error *= 0.0
            terminate = (
                (left_obj_pos_error > config.object_pos_threshold)
                | (right_obj_pos_error > config.object_pos_threshold)
                | (left_obj_rot_error > config.object_rot_threshold)
                | (right_obj_rot_error > config.object_rot_threshold)
            )
            return terminate
        left_obj_pos = qpos_sim[:, -14:-11]
        left_obj_pos_ref = qpos_ref[-14:-11].unsqueeze(0)
        left_obj_pos_error = torch.norm(left_obj_pos - left_obj_pos_ref, p=2, dim=1)
        left_obj_quat = qpos_sim[:, -11:-7]
        left_obj_quat_ref = qpos_ref[-11:-7].unsqueeze(0)
        left_obj_quat_error = torch.norm(
            quat_sub(left_obj_quat, left_obj_quat_ref.repeat(qpos_sim.shape[0], 1)),
            p=2,
            dim=1,
        )
        right_obj_pos = qpos_sim[:, -7:-4]
        right_obj_pos_ref = qpos_ref[-7:-4].unsqueeze(0)
        right_obj_pos_error = torch.norm(right_obj_pos - right_obj_pos_ref, p=2, dim=1)
        right_obj_quat = qpos_sim[:, -4:]
        right_obj_quat_ref = qpos_ref[-4:].unsqueeze(0)
        right_obj_quat_error = torch.norm(
            quat_sub(right_obj_quat, right_obj_quat_ref.repeat(qpos_sim.shape[0], 1)),
            p=2,
            dim=1,
        )
        # special case: only have left object
        if torch.all(right_obj_pos_ref.abs() < 1e-4):
            right_obj_pos_error *= 0.0
            right_obj_quat_error *= 0.0
        # special case: only have right object
        if torch.all(left_obj_pos_ref.abs() < 1e-4):
            left_obj_pos_error *= 0.0
            left_obj_quat_error *= 0.0
        terminate = (
            (left_obj_pos_error > config.object_pos_threshold)
            | (right_obj_pos_error > config.object_pos_threshold)
            | (left_obj_quat_error > config.object_rot_threshold)
            | (right_obj_quat_error > config.object_rot_threshold)
        )
    elif config.embodiment_type in ["right", "left"]:
        if config.nq_obj == 6:
            obj_pos = qpos_sim[:, -6:-3]
            obj_pos_ref = qpos_ref[-6:-3].unsqueeze(0)
            obj_pos_error = torch.norm(obj_pos - obj_pos_ref, p=2, dim=1)
            obj_rot = qpos_sim[:, -3:]
            obj_rot_ref = qpos_ref[-3:].unsqueeze(0)
            obj_rot_error = torch.norm(obj_rot - obj_rot_ref, p=2, dim=1)
            terminate = (obj_pos_error > config.object_pos_threshold) | (
                obj_rot_error > config.object_rot_threshold
            )
            return terminate
        obj_pos = qpos_sim[:, -7:-4]
        obj_pos_ref = qpos_ref[-7:-4].unsqueeze(0)
        obj_pos_error = torch.norm(obj_pos - obj_pos_ref, p=2, dim=1)
        obj_quat = qpos_sim[:, -4:]
        obj_quat_ref = qpos_ref[-4:].unsqueeze(0)
        obj_quat_error = torch.norm(
            quat_sub(obj_quat, obj_quat_ref.repeat(qpos_sim.shape[0], 1)), p=2, dim=1
        )
        terminate = (obj_pos_error > config.object_pos_threshold) | (
            obj_quat_error > config.object_rot_threshold
        )
    elif config.embodiment_type in ["humanoid", "humanoid_object"]:
        base_pos = qpos_sim[:, :3]
        base_pos_ref = qpos_ref[:3].unsqueeze(0)
        base_pos_error = torch.norm(base_pos - base_pos_ref, p=2, dim=1)
        base_quat = qpos_sim[:, 3:7]
        base_quat_ref = qpos_ref[3:7].unsqueeze(0)
        base_quat_error = torch.norm(
            quat_sub(base_quat, base_quat_ref.repeat(qpos_sim.shape[0], 1)), p=2, dim=1
        )
        terminate = (base_pos_error > config.base_pos_threshold) | (
            base_quat_error > config.base_rot_threshold
        )
    elif config.embodiment_type == "dual_humanoid_object":
        nq_robot = (config.nq - config.nq_obj) // 2  # 36 per robot
        N = qpos_sim.shape[0]
        # robot1 base
        r1_pos_err = torch.norm(qpos_sim[:, :3] - qpos_ref[:3].unsqueeze(0), p=2, dim=1)
        r1_rot_err = torch.norm(
            quat_sub(qpos_sim[:, 3:7], qpos_ref[3:7].unsqueeze(0).expand(N, -1)),
            p=2, dim=1,
        )
        # robot2 base
        r2_pos_err = torch.norm(
            qpos_sim[:, nq_robot:nq_robot + 3] - qpos_ref[nq_robot:nq_robot + 3].unsqueeze(0),
            p=2, dim=1,
        )
        r2_rot_err = torch.norm(
            quat_sub(
                qpos_sim[:, nq_robot + 3:nq_robot + 7],
                qpos_ref[nq_robot + 3:nq_robot + 7].unsqueeze(0).expand(N, -1),
            ),
            p=2, dim=1,
        )
        terminate = (
            (r1_pos_err > config.base_pos_threshold)
            | (r1_rot_err > config.base_rot_threshold)
            | (r2_pos_err > config.base_pos_threshold)
            | (r2_rot_err > config.base_rot_threshold)
        )
    else:
        raise ValueError(f"Invalid embodiment_type: {config.embodiment_type}")
    return terminate


def get_qpos(config: Config, env: MJWPEnv) -> torch.Tensor:
    return wp.to_torch(env.data_wp.qpos)


def set_qpos(config: Config, env: MJWPEnv, qpos: torch.Tensor):
    qpos = qpos.to(config.device)
    if qpos.dim() == 1:
        qpos = qpos.unsqueeze(0).repeat(env.num_worlds, 1)
    wp.copy(env.data_wp.qpos, wp.from_torch(qpos))
    # reset velocities/time as well for consistency
    zero_qvel = torch.zeros((env.num_worlds, env.model_cpu.nv), device=config.device)
    wp.copy(env.data_wp.qvel, wp.from_torch(zero_qvel))
    wp.copy(
        env.data_wp.time,
        wp.from_torch(
            torch.zeros(env.num_worlds, dtype=torch.float32, device=config.device)
        ),
    )


def get_qvel(config: Config, env: MJWPEnv) -> torch.Tensor:
    return wp.to_torch(env.data_wp.qvel)


def compute_contact_point_delta(
    contact_mask_step: torch.Tensor,
    contact_pos_ref_step: torch.Tensor,
    site_xpos: torch.Tensor,
    hand_contact_site_ids: list[int | None],
    contact_indices: list[int],
) -> torch.Tensor | None:
    """Compute mean contact position delta for a hand (current - reference).

    Args:
        contact_mask_step: (N_contact,) mask for active contacts.
        contact_pos_ref_step: (N_contact, 3) reference contact positions.
        site_xpos: (N_site, 3) current site positions for the active world.
        hand_contact_site_ids: list mapping contact indices to site ids (None if missing).
        contact_indices: indices for the hand contacts to aggregate.
    """
    current_positions = []
    reference_positions = []
    for idx in contact_indices:
        if idx >= len(hand_contact_site_ids) or idx >= contact_pos_ref_step.shape[0]:
            continue
        sid = hand_contact_site_ids[idx]
        if sid is None or contact_mask_step[idx] <= 0.5:
            continue
        current_positions.append(site_xpos[sid])
        reference_positions.append(contact_pos_ref_step[idx])

    if not current_positions:
        return None

    current_mean = torch.stack(current_positions, dim=0).mean(dim=0)
    reference_mean = torch.stack(reference_positions, dim=0).mean(dim=0)
    return current_mean - reference_mean


def get_trace(config: Config, env: MJWPEnv) -> torch.Tensor:
    """Return per-world trace points used for visualization. Minimal default returns
    an empty trace set of shape (N, 0, 3) when not configured.
    """
    site_xpos = wp.to_torch(env.data_wp.site_xpos)  # (N, nsite, 3)
    return site_xpos[:, config.trace_site_ids, :]


def save_state(env: MJWPEnv):
    """Clone the essential set of Warp arrays to restore later.
    Includes core state variables and key derived quantities.
    """
    _copy_state(env.data_wp, env.data_wp_prev)
    return env
    # qpos = wp.clone(env.data_wp.qpos)
    # qvel = wp.clone(env.data_wp.qvel)
    # qacc = wp.clone(env.data_wp.qacc)
    # time_arr = wp.clone(env.data_wp.time)
    # ctrl = wp.clone(env.data_wp.ctrl) if hasattr(env.data_wp, "ctrl") else None
    # act = wp.clone(env.data_wp.act) if hasattr(env.data_wp, "act") else None
    # act_dot = wp.clone(env.data_wp.act_dot) if hasattr(env.data_wp, "act_dot") else None
    # site_xpos = wp.clone(env.data_wp.site_xpos)
    # site_xmat = wp.clone(env.data_wp.site_xmat)
    # mocap_pos = (
    #     wp.clone(env.data_wp.mocap_pos) if hasattr(env.data_wp, "mocap_pos") else None
    # )
    # mocap_quat = (
    #     wp.clone(env.data_wp.mocap_quat) if hasattr(env.data_wp, "mocap_quat") else None
    # )
    # energy = wp.clone(env.data_wp.energy) if hasattr(env.data_wp, "energy") else None
    # return (
    #     qpos,
    #     qvel,
    #     qacc,
    #     time_arr,
    #     ctrl,
    #     act,
    #     act_dot,
    #     site_xpos,
    #     site_xmat,
    #     mocap_pos,
    #     mocap_quat,
    #     energy,
    # )


def load_state(env: MJWPEnv, state):
    _copy_state(env.data_wp_prev, env.data_wp)
    return env
    # (
    #     qpos,
    #     qvel,
    #     qacc,
    #     time_arr,
    #     ctrl,
    #     act,
    #     act_dot,
    #     site_xpos,
    #     site_xmat,
    #     mocap_pos,
    #     mocap_quat,
    #     energy,
    # ) = state
    # wp.copy(env.data_wp.qpos, qpos)
    # wp.copy(env.data_wp.qvel, qvel)
    # wp.copy(env.data_wp.qacc, qacc)
    # wp.copy(env.data_wp.time, time_arr)
    # if ctrl is not None and hasattr(env.data_wp, "ctrl"):
    #     wp.copy(env.data_wp.ctrl, ctrl)
    # if act is not None and hasattr(env.data_wp, "act"):
    #     wp.copy(env.data_wp.act, act)
    # if act_dot is not None and hasattr(env.data_wp, "act_dot"):
    #     wp.copy(env.data_wp.act_dot, act_dot)
    # if mocap_pos is not None and hasattr(env.data_wp, "mocap_pos"):
    #     wp.copy(env.data_wp.mocap_pos, mocap_pos)
    # if mocap_quat is not None and hasattr(env.data_wp, "mocap_quat"):
    #     wp.copy(env.data_wp.mocap_quat, mocap_quat)
    # if energy is not None and hasattr(env.data_wp, "energy"):
    #     wp.copy(env.data_wp.energy, energy)
    # if site_xpos is not None and hasattr(env.data_wp, "site_xpos"):
    #     wp.copy(env.data_wp.site_xpos, site_xpos)
    # if site_xmat is not None and hasattr(env.data_wp, "site_xmat"):
    #     wp.copy(env.data_wp.site_xmat, site_xmat)
    # return env


def apply_perturbation(config: Config, env: MJWPEnv):
    # get object id
    right_obj_id = mujoco.mj_name2id(
        env.model_cpu, mujoco.mjtObj.mjOBJ_BODY, "right_object"
    )
    left_obj_id = mujoco.mj_name2id(
        env.model_cpu, mujoco.mjtObj.mjOBJ_BODY, "left_object"
    )
    xfrc_applied = wp.to_torch(env.data_wp.xfrc_applied)
    if right_obj_id != -1:
        xfrc_applied[:, right_obj_id, :3] = config.perturb_force
        xfrc_applied[:, right_obj_id, 3:] = config.perturb_torque
    if left_obj_id != -1:
        xfrc_applied[:, left_obj_id, :3] = config.perturb_force
        xfrc_applied[:, left_obj_id, 3:] = config.perturb_torque
    wp.copy(env.data_wp.xfrc_applied, wp.from_torch(xfrc_applied))
    return env


def _apply_object_pd_override(config: Config, env: MJWPEnv):
    """E027b: Override object actuator ctrl to PD-track ref trajectory.

    For scene_act.xml with 6 object position actuators (3 slide + 3 hinge),
    sets ctrl = ref_target so the actuator's built-in PD drives the object.
    Called after CEM ctrl is written, effectively overriding CEM's object dims.
    Adds gravity compensation offset to z-target (mg/kp) for zero steady-state error.
    """
    time_arr = wp.to_torch(env.data_wp.time)
    t = time_arr[0].item()
    dt = 1.0 / 30.0
    T = env.object_pd_ref_pos.shape[0]
    idx = min(int(t / dt), T - 1)

    # Get ref pos (3) and euler (3) directly
    ref_pos = env.object_pd_ref_pos[idx]  # (3,)
    ref_euler = env.object_pd_ref_euler[idx]  # (3,) xyz euler

    # Gravity compensation offset for z: target += mg/kp
    grav_comp = env.object_mass * 9.81 / config.object_pd_kp_pos

    # Object actuator target = [pos_x, pos_y, pos_z + grav_comp, rot_x, rot_y, rot_z]
    obj_target = torch.tensor(
        [ref_pos[0].item(), ref_pos[1].item(), ref_pos[2].item() + grav_comp,
         ref_euler[0].item(), ref_euler[1].item(), ref_euler[2].item()],
        dtype=torch.float32, device=config.device,
    )

    # Write to ctrl for object actuator channels (last 6 of nu)
    ctrl = wp.to_torch(env.data_wp.ctrl)
    obj_act_start = ctrl.shape[1] - 6
    ctrl[:, obj_act_start:] = obj_target.unsqueeze(0)
    wp.copy(env.data_wp.ctrl, wp.from_torch(ctrl))


def _apply_partner_force(config: Config, env: MJWPEnv):
    """Apply external force on the object body to simulate partner support.

    Models the human partner holding one side of the object, providing:
    1. Gravity compensation: upward force = partner_force_scale * object_weight
    2. (Optional) Spring: pull toward reference position with partner_force_spring_kp

    This enables single-robot retargeting of cooperative carrying tasks.
    """
    obj_body_id = mujoco.mj_name2id(
        env.model_cpu, mujoco.mjtObj.mjOBJ_BODY, "object"
    )
    if obj_body_id == -1:
        return

    xfrc_applied = wp.to_torch(env.data_wp.xfrc_applied)

    # Gravity compensation: upward force on object
    obj_mass = env.model_cpu.body_mass[obj_body_id]
    gravity_z = -env.model_cpu.opt.gravity[2]  # positive (9.81)
    upward_force = config.partner_force_scale * obj_mass * gravity_z
    xfrc_applied[:, obj_body_id, 2] = upward_force

    # Optional: damped spring toward reference position
    if config.partner_force_spring_kp > 0 and hasattr(env, "partner_force_ref_pos"):
        # Get current object position from qpos
        qpos = wp.to_torch(env.data_wp.qpos)
        qvel = wp.to_torch(env.data_wp.qvel)
        # Object freejoint position: find the qpos address
        obj_jnt_id = env.model_cpu.body_jntadr[obj_body_id]
        obj_qadr = env.model_cpu.jnt_qposadr[obj_jnt_id]
        obj_vadr = env.model_cpu.jnt_dofadr[obj_jnt_id]
        obj_pos_sim = qpos[:, obj_qadr:obj_qadr + 3]  # (N, 3)
        obj_vel_sim = qvel[:, obj_vadr:obj_vadr + 3]  # (N, 3)

        # Get reference pos for current time
        time_arr = wp.to_torch(env.data_wp.time)
        t = time_arr[0].item()
        dt = 1.0 / 30.0  # ref fps
        T = env.partner_force_ref_pos.shape[0]
        idx = min(int(t / dt), T - 1)
        ref_pos = env.partner_force_ref_pos[idx]  # (3,) on GPU

        # Ramp-up: linearly increase spring over first 0.5s to avoid initial jolt
        ramp = min(t / 0.5, 1.0)

        # Damped spring: F = kp*(ref - pos) - kd*vel
        kp = config.partner_force_spring_kp * ramp
        if config.partner_force_spring_kd < 0:
            kd = 2.0 * (obj_mass * config.partner_force_spring_kp) ** 0.5
        else:
            kd = config.partner_force_spring_kd
        spring_force = kp * (ref_pos.unsqueeze(0) - obj_pos_sim) - kd * obj_vel_sim
        xfrc_applied[:, obj_body_id, :3] += spring_force

        # E030: Orientation control via xfrc_applied torque
        if config.partner_force_spring_kp_rot > 0 and hasattr(env, "partner_force_ref_quat"):
            from spider.math import quat_sub

            # Object quaternion from qpos: freejoint stores (w,x,y,z)
            obj_quat_sim = qpos[:, obj_qadr + 3:obj_qadr + 7]  # (N, 4) wxyz
            obj_angvel_sim = qvel[:, obj_vadr + 3:obj_vadr + 6]  # (N, 3)

            # Normalize quaternion (can become unnormalized in unstable rollouts)
            quat_norm = obj_quat_sim.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            obj_quat_sim = obj_quat_sim / quat_norm

            # Reference quaternion for current time
            ref_quat = env.partner_force_ref_quat[idx]  # (4,) wxyz on GPU

            # axis-angle error in world frame: quat_sub(ref, cur)
            aa_err = quat_sub(
                ref_quat.unsqueeze(0).expand(obj_quat_sim.shape[0], -1),
                obj_quat_sim,
            )  # (N, 3)

            # Replace NaN with zero (from degenerate quaternions)
            aa_err = torch.nan_to_num(aa_err, nan=0.0)

            # Clamp axis-angle magnitude
            aa_mag = aa_err.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            clamp_val = config.partner_force_rot_clamp
            aa_err = torch.where(
                aa_mag > clamp_val,
                aa_err / aa_mag * clamp_val,
                aa_err,
            )

            # Clamp angular velocity too
            obj_angvel_sim = torch.nan_to_num(obj_angvel_sim, nan=0.0)
            obj_angvel_sim = obj_angvel_sim.clamp(-10.0, 10.0)

            # PD torque with ramp
            kp_rot = config.partner_force_spring_kp_rot * ramp
            if config.partner_force_spring_kd_rot < 0:
                avg_inertia = float(np.mean(env.model_cpu.body_inertia[obj_body_id]))
                kd_rot = 2.0 * (avg_inertia * config.partner_force_spring_kp_rot) ** 0.5
            else:
                kd_rot = config.partner_force_spring_kd_rot
            torque = kp_rot * aa_err - kd_rot * obj_angvel_sim

            # Clamp total torque magnitude to prevent instability
            torque_mag = torque.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            max_torque = 5.0  # Nm — conservative limit
            torque = torch.where(
                torque_mag > max_torque,
                torque / torque_mag * max_torque,
                torque,
            )
            xfrc_applied[:, obj_body_id, 3:6] += torque

    wp.copy(env.data_wp.xfrc_applied, wp.from_torch(xfrc_applied))


def _update_object_weld_target(config: Config, env: MJWPEnv):
    """Update the mocap 'object_target' body to track the reference trajectory.

    Used with scene_weld.xml: a soft weld equality constraint pulls the freejoint
    object toward this mocap body. MuJoCo's solver handles pos+orient coupling.
    """
    if not hasattr(env, "_weld_mocap_id"):
        # Find the mocap body index for "object_target"
        body_id = mujoco.mj_name2id(
            env.model_cpu, mujoco.mjtObj.mjOBJ_BODY, "object_target"
        )
        if body_id == -1:
            env._weld_mocap_id = -1
            return
        # mocap body index (0-based among mocap bodies)
        # In MuJoCo, body_mocapid maps body_id → mocap_id
        env._weld_mocap_id = env.model_cpu.body_mocapid[body_id]

    if env._weld_mocap_id < 0:
        return

    if not hasattr(env, "partner_force_ref_pos"):
        return

    # Get reference for current time
    time_arr = wp.to_torch(env.data_wp.time)
    t = time_arr[0].item()
    dt = 1.0 / 30.0
    T = env.partner_force_ref_pos.shape[0]
    idx = min(int(t / dt), T - 1)

    ref_pos = env.partner_force_ref_pos[idx]  # (3,)
    ref_quat = env.partner_force_ref_quat[idx]  # (4,) wxyz

    # Write to mocap body (shared-memory in-place)
    mocap_pos_all = wp.to_torch(env.data_wp.mocap_pos)  # (N, nmocap, 3)
    mocap_quat_all = wp.to_torch(env.data_wp.mocap_quat)  # (N, nmocap, 4)
    mid = env._weld_mocap_id
    N = mocap_pos_all.shape[0]
    mocap_pos_all[:, mid] = ref_pos.unsqueeze(0).expand(N, -1)
    mocap_quat_all[:, mid] = ref_quat.unsqueeze(0).expand(N, -1)


def _update_mocap_partner(env: MJWPEnv):
    """Update mocap body positions from partner trajectory based on current sim time.

    Uses wp.to_torch for shared-memory in-place writes (wp.copy fails after
    CUDA graph capture because data_wp.mocap_pos.ptr becomes None).
    """
    # Get current time from data
    time_arr = wp.to_torch(env.data_wp.time)  # (N,)
    t = time_arr[0].item()  # all worlds share same time

    # Map sim time to trajectory frame index
    dt = env.mocap_partner_dt
    T = env.mocap_partner_pos.shape[0]
    idx = min(int(t / dt), T - 1)

    # Get position and quaternion for this frame: (2, 3) and (2, 4)
    pos = env.mocap_partner_pos[idx]  # (2, 3) on GPU
    quat = env.mocap_partner_quat[idx]  # (2, 4) on GPU

    # Write via shared-memory torch view (in-place, no wp.copy needed)
    mocap_pos_all = wp.to_torch(env.data_wp.mocap_pos)  # (N, nmocap, 3)
    if mocap_pos_all.shape[1] >= 2:
        mocap_quat_all = wp.to_torch(env.data_wp.mocap_quat)  # (N, nmocap, 4)
        N = mocap_pos_all.shape[0]
        mocap_pos_all[:, :2] = pos.unsqueeze(0).expand(N, -1, -1)
        mocap_quat_all[:, :2] = quat.unsqueeze(0).expand(N, -1, -1)


def _load_mocap_partner(config: Config, env: MJWPEnv):
    """Load partner trajectory data and attach to env for runtime updates."""
    import os

    path = config.mocap_partner_trajectory
    if not os.path.isabs(path):
        # Resolve relative to data directory (same dir as trajectory_kinematic.npz)
        data_dir = os.path.dirname(config.data_path)
        path = os.path.join(data_dir, path)

    data = np.load(path)
    partner_pos = data["partner_pos"]  # (T, 2, 3)
    partner_quat = data["partner_quat"]  # (T, 2, 4) wxyz format

    # Store as GPU tensors on env
    env.mocap_partner_pos = torch.from_numpy(partner_pos).float().to(config.device)
    env.mocap_partner_quat = torch.from_numpy(partner_quat).float().to(config.device)
    env.mocap_partner_dt = config.ref_dt  # partner trajectory is at ref framerate

    loguru.logger.info(
        f"Loaded mocap partner trajectory: {partner_pos.shape[0]} frames @ {1/config.ref_dt:.0f}fps"
    )


def step_env(config: Config, env: MJWPEnv, ctrl_mujoco: torch.Tensor):
    """Step all worlds with provided MuJoCo-format controls of shape (N, nu)."""
    if ctrl_mujoco.dim() == 1:
        ctrl_mujoco = ctrl_mujoco.unsqueeze(0).repeat(env.num_worlds, 1)
    # Ensure we operate on the correct CUDA context/device
    with wp.ScopedDevice(env.device):
        # apply perturbation
        env = apply_perturbation(config, env)
        # E024: apply partner force on object (simulating human partner support)
        if config.partner_force_scale > 0:
            _apply_partner_force(config, env)
        # E030: update weld target mocap body (for scene_weld.xml)
        if config.scene_name == "scene_weld":
            _update_object_weld_target(config, env)
        # step control
        wp.copy(env.data_wp.ctrl, wp.from_torch(ctrl_mujoco.to(torch.float32)))
        # E027b: object PD override — set object actuator ctrl to track ref
        if config.object_pd_override and hasattr(env, "object_pd_ref_pos"):
            _apply_object_pd_override(config, env)
        # Update partner mocap positions within rollout (E013: intra-rollout update)
        if (
            config.mocap_partner_intra_step
            and hasattr(env, "mocap_partner_pos")
            and env.mocap_partner_pos is not None
        ):
            _update_mocap_partner(env)
        wp.capture_launch(env.graph)
        # E029: kinematic object override AFTER physics step
        if hasattr(env, "partner_force_ref_pos") and config.partner_force_spring_kp < 0:
            qpos = wp.to_torch(env.data_wp.qpos)
            time_arr = wp.to_torch(env.data_wp.time)
            t = time_arr[0].item()
            dt = 1.0 / 30.0
            T = env.partner_force_ref_pos.shape[0]
            idx = min(int(t / dt), T - 1)
            ref_pos = env.partner_force_ref_pos[idx]
            qpos[:, 36:39] = ref_pos.unsqueeze(0)
            if hasattr(env, "partner_force_ref_quat"):
                ref_quat = env.partner_force_ref_quat[idx]
                qpos[:, 39:43] = ref_quat.unsqueeze(0)
            wp.copy(env.data_wp.qpos, wp.from_torch(qpos))


def save_env_params(config: Config, env: MJWPEnv):
    """Save the current simulation parameters."""
    # Only record which group is active; parameters are embedded in separate models
    # TODO: explicitly read pair_margin and xy_offset from env.data_wp
    # currently we choose this solution since pair_margin has a huge virtual dimension,
    # convert it to torch would lead to OOM
    pair_margin = 0.0
    xy_offset = 0.0
    return {"pair_margin": pair_margin, "xy_offset": xy_offset}


def load_env_params(config: Config, env: MJWPEnv, env_param: dict):
    """Load the simulation parameters.

    Parameters to be updated:
    - pair_margin
    - xy_offset of the object
    """
    # update model parameters (pair_margin)
    if "pair_margin" in env_param:
        pair_margin_single_np = np.full(
            shape=(config.npair,), fill_value=env_param["pair_margin"], dtype=np.float32
        )

        # 2. Copy this small array to the GPU
        pair_margin_override_wp = wp.from_numpy(
            pair_margin_single_np, dtype=wp.float32, device=config.device
        )

        # 3. Apply the stride trick to broadcast it
        # This makes Warp treat the single instance as if it were num_samples copies
        # without allocating any new memory.
        pair_margin_override_wp.strides = (0,) + pair_margin_override_wp.strides
        pair_margin_override_wp.shape = (
            config.num_samples,
        ) + pair_margin_override_wp.shape
        pair_margin_override_wp.ndim += 1
        wp.copy(env.model_wp.pair_margin, pair_margin_override_wp)

    # update object position (NOTE: currently, xy_offset is only one scalar, which means we only update in the diagonal direction)
    if "xy_offset" in env_param:
        qpos_override_th = wp.to_torch(env.data_wp.qpos)
        # TODO: make object pos detection automatic
        if config.embodiment_type == "bimanual":
            qpos_override_th[:, -14:-12] = (
                qpos_override_th[:, -14:-12] + env_param["xy_offset"]
            )
            qpos_override_th[:, -12:-10] = (
                qpos_override_th[:, -12:-10] + env_param["xy_offset"]
            )
        elif config.embodiment_type in ["right", "left"]:
            qpos_override_th[:, -7:-5] = (
                qpos_override_th[:, -7:-5] + env_param["xy_offset"]
            )
        elif config.embodiment_type in ["humanoid_object", "dual_humanoid_object"]:
            nq_obj = config.nq_obj  # 7 (freejoint) or 6 (contact_guidance)
            qpos_override_th[:, -nq_obj:-nq_obj + 2] = (
                qpos_override_th[:, -nq_obj:-nq_obj + 2] + env_param["xy_offset"]
            )

        wp.copy(env.data_wp.qpos, wp.from_torch(qpos_override_th))

    # update object actuator gains
    if "kp" in env_param or "kd" in env_param:
        actuator_ids = config.object_actuator_ids
        if not actuator_ids:
            loguru.logger.warning(
                "Object actuator ids are empty; skipping kp/kd updates."
            )
        else:
            kp = env_param.get("kp")
            kd = env_param.get("kd")
            if kp is None or kd is None:
                loguru.logger.warning(
                    "Both kp and kd are required to update actuator gains; skipping."
                )
            else:
                kp_np = np.asarray(kp, dtype=np.float32)
                kd_np = np.asarray(kd, dtype=np.float32)
                if kp_np.ndim == 0:
                    kp_np = np.full((len(actuator_ids),), kp_np, dtype=np.float32)
                if kd_np.ndim == 0:
                    kd_np = np.full((len(actuator_ids),), kd_np, dtype=np.float32)
                if kp_np.shape[0] != len(actuator_ids) or kd_np.shape[0] != len(
                    actuator_ids
                ):
                    raise ValueError(
                        "kp/kd size mismatch for object actuators: "
                        f"kp={kp_np.shape}, kd={kd_np.shape}, "
                        f"expected={len(actuator_ids)}"
                    )

                # Update CPU model (used for viewer and as source of truth)
                env.model_cpu.actuator_gainprm[actuator_ids, 0] = kp_np
                env.model_cpu.actuator_biasprm[actuator_ids, 1] = -kd_np

                # Propagate to MJWarp model if available
                if hasattr(env.model_wp, "actuator_gainprm") and hasattr(
                    env.model_wp, "actuator_biasprm"
                ):
                    gain_full = np.array(
                        env.model_cpu.actuator_gainprm, dtype=np.float32
                    )
                    bias_full = np.array(
                        env.model_cpu.actuator_biasprm, dtype=np.float32
                    )
                    wp.copy(
                        env.model_wp.actuator_gainprm,
                        wp.from_numpy(
                            gain_full, dtype=wp.float32, device=config.device
                        ),
                    )
                    wp.copy(
                        env.model_wp.actuator_biasprm,
                        wp.from_numpy(
                            bias_full, dtype=wp.float32, device=config.device
                        ),
                    )
                else:
                    loguru.logger.warning(
                        "MJWarp model has no actuator_gainprm/biasprm; updated CPU model only."
                    )

    return env


def _broadcast_state(data_wp, num_worlds: int):
    """Broadcast state from first world/env to all worlds/envs.

    This is a generic function that can be used by both MJWP and HDMI simulators.

    Args:
        data_wp: MuJoCo Warp data object (mjwarp.Data or wrapped version)
        num_worlds: Number of parallel worlds/environments
    """
    # Core state variables - always try these first
    qpos0 = wp.to_torch(data_wp.qpos)[:1]
    qvel0 = wp.to_torch(data_wp.qvel)[:1]
    time0 = wp.to_torch(data_wp.time)[:1]
    ctrl0 = wp.to_torch(data_wp.ctrl)[:1]

    # Handle time specially as it might be 1D
    if time0.dim() == 1:
        time_repeated = time0.repeat(num_worlds)
    else:
        time_repeated = time0.repeat(num_worlds, 1)

    wp.copy(data_wp.qpos, wp.from_torch(qpos0.repeat(num_worlds, 1)))
    wp.copy(data_wp.qvel, wp.from_torch(qvel0.repeat(num_worlds, 1)))
    wp.copy(data_wp.time, wp.from_torch(time_repeated))
    wp.copy(data_wp.ctrl, wp.from_torch(ctrl0.repeat(num_worlds, 1)))

    # Additional core state variables
    qacc0 = wp.to_torch(data_wp.qacc)[:1]
    wp.copy(data_wp.qacc, wp.from_torch(qacc0.repeat(num_worlds, 1)))

    act0 = wp.to_torch(data_wp.act)[:1]
    wp.copy(data_wp.act, wp.from_torch(act0.repeat(num_worlds, 1)))

    act_dot0 = wp.to_torch(data_wp.act_dot)[:1]
    wp.copy(data_wp.act_dot, wp.from_torch(act_dot0.repeat(num_worlds, 1)))

    # Forces and applied forces
    qfrc_applied0 = wp.to_torch(data_wp.qfrc_applied)[:1]
    wp.copy(data_wp.qfrc_applied, wp.from_torch(qfrc_applied0.repeat(num_worlds, 1)))

    xfrc_applied0 = wp.to_torch(data_wp.xfrc_applied)[:1]
    wp.copy(data_wp.xfrc_applied, wp.from_torch(xfrc_applied0.repeat(num_worlds, 1, 1)))

    # Mocap data
    mocap_pos0 = wp.to_torch(data_wp.mocap_pos)[:1]
    wp.copy(data_wp.mocap_pos, wp.from_torch(mocap_pos0.repeat(num_worlds, 1, 1)))

    mocap_quat0 = wp.to_torch(data_wp.mocap_quat)[:1]
    wp.copy(data_wp.mocap_quat, wp.from_torch(mocap_quat0.repeat(num_worlds, 1, 1)))

    # Spatial transformations
    xpos0 = wp.to_torch(data_wp.xpos)[:1]
    wp.copy(data_wp.xpos, wp.from_torch(xpos0.repeat(num_worlds, 1, 1)))

    xquat0 = wp.to_torch(data_wp.xquat)[:1]
    wp.copy(data_wp.xquat, wp.from_torch(xquat0.repeat(num_worlds, 1, 1)))

    xmat0 = wp.to_torch(data_wp.xmat)[:1]
    wp.copy(data_wp.xmat, wp.from_torch(xmat0.repeat(num_worlds, 1, 1, 1)))

    # Geometry positions
    geom_xpos0 = wp.to_torch(data_wp.geom_xpos)[:1]
    wp.copy(data_wp.geom_xpos, wp.from_torch(geom_xpos0.repeat(num_worlds, 1, 1)))

    geom_xmat0 = wp.to_torch(data_wp.geom_xmat)[:1]
    wp.copy(data_wp.geom_xmat, wp.from_torch(geom_xmat0.repeat(num_worlds, 1, 1, 1)))

    # Site positions
    site_xpos0 = wp.to_torch(data_wp.site_xpos)[:1]
    wp.copy(data_wp.site_xpos, wp.from_torch(site_xpos0.repeat(num_worlds, 1, 1)))


def sync_env(config: Config, env: MJWPEnv, mj_data: mujoco.MjData):
    """Broadcast the state from first env to all envs

    This function synchronizes states from the first environment to all environments.
    Uses safe copying with buffer size validation to avoid mismatches.
    """
    # Update mocap partner positions before broadcasting
    if hasattr(env, "mocap_partner_pos") and env.mocap_partner_pos is not None:
        t = mj_data.time
        dt = env.mocap_partner_dt
        T = env.mocap_partner_pos.shape[0]
        idx = min(int(t / dt), T - 1)
        pos = env.mocap_partner_pos[idx]  # (2, 3) GPU tensor
        quat = env.mocap_partner_quat[idx]  # (2, 4) GPU tensor
        # Write to world 0 of data_wp, _broadcast_state will copy to all worlds
        mocap_pos_all = wp.to_torch(env.data_wp.mocap_pos)  # (N, nmocap, 3)
        if mocap_pos_all.shape[1] > 0:
            mocap_quat_all = wp.to_torch(env.data_wp.mocap_quat)  # (N, nmocap, 4)
            mocap_pos_all[0] = pos
            mocap_quat_all[0] = quat

    _broadcast_state(env.data_wp, env.num_worlds)


def sync_env_mujoco(config: Config, env: MJWPEnv, mj_data: mujoco.MjData):
    """Sync state from mj_data to env.data_wp"""
    # Define field mappings with their data and target shapes
    fields = [
        # Core state variables
        ("qpos", mj_data.qpos, (env.data_wp.nworld, -1)),
        ("qvel", mj_data.qvel, (env.data_wp.nworld, -1)),
        ("qacc", mj_data.qacc, (env.data_wp.nworld, -1)),
        ("time", np.array([mj_data.time], dtype=np.float32), (env.data_wp.nworld, 1)),
        ("ctrl", mj_data.ctrl, (env.data_wp.nworld, -1)),
        ("act", mj_data.act, (env.data_wp.nworld, -1)),
        ("act_dot", mj_data.act_dot, (env.data_wp.nworld, -1)),
        ("qacc_warmstart", mj_data.qacc_warmstart, (env.data_wp.nworld, -1)),
        # Forces
        ("qfrc_applied", mj_data.qfrc_applied, (env.data_wp.nworld, -1)),
        ("xfrc_applied", mj_data.xfrc_applied, (env.data_wp.nworld, -1, -1)),
        # Energy (2D: kinetic + potential)
        ("energy", mj_data.energy, (env.data_wp.nworld, 2)),
        # Mocap data
        ("mocap_pos", mj_data.mocap_pos, (env.data_wp.nworld, -1, 3)),
        ("mocap_quat", mj_data.mocap_quat, (env.data_wp.nworld, -1, 4)),
        # Spatial transformations
        ("xpos", mj_data.xpos, (env.data_wp.nworld, -1, 3)),
        ("xquat", mj_data.xquat, (env.data_wp.nworld, -1, 4)),
        ("xmat", mj_data.xmat, (env.data_wp.nworld, -1, 9)),
        ("xipos", mj_data.xipos, (env.data_wp.nworld, -1, 3)),
        ("ximat", mj_data.ximat, (env.data_wp.nworld, -1, 9)),
        # Geometry positions
        ("geom_xpos", mj_data.geom_xpos, (env.data_wp.nworld, -1, 3)),
        ("geom_xmat", mj_data.geom_xmat, (env.data_wp.nworld, -1, 9)),
        ("site_xpos", mj_data.site_xpos, (env.data_wp.nworld, -1, 3)),
        ("site_xmat", mj_data.site_xmat, (env.data_wp.nworld, -1, 9)),
        # Body dynamics (spatial vectors)
        ("cacc", mj_data.cacc, (env.data_wp.nworld, -1, 6)),
        ("cfrc_int", mj_data.cfrc_int, (env.data_wp.nworld, -1, 6)),
        ("cfrc_ext", mj_data.cfrc_ext, (env.data_wp.nworld, -1, 6)),
        # Sensor data
        ("sensordata", mj_data.sensordata, (env.data_wp.nworld, -1)),
        # Actuator data
        ("actuator_length", mj_data.actuator_length, (env.data_wp.nworld, -1)),
        ("actuator_velocity", mj_data.actuator_velocity, (env.data_wp.nworld, -1)),
        ("actuator_force", mj_data.actuator_force, (env.data_wp.nworld, -1)),
        # Tendon data
        ("ten_length", mj_data.ten_length, (env.data_wp.nworld, -1)),
        ("ten_velocity", mj_data.ten_velocity, (env.data_wp.nworld, -1)),
    ]

    # Contact struct fields - these need special handling
    contact_fields = [
        ("dist", "contact.dist"),
        ("pos", "contact.pos"),
        ("frame", "contact.frame"),
        ("includemargin", "contact.includemargin"),
        ("friction", "contact.friction"),
        ("solref", "contact.solref"),
        ("solreffriction", "contact.solreffriction"),
        ("solimp", "contact.solimp"),
        ("dim", "contact.dim"),
        ("geom", "contact.geom"),
        ("efc_address", "contact.efc_address"),
        ("worldid", "contact.worldid"),
    ]

    # Constraint (efc) fields - these are direct fields on mj_data, not nested in a struct
    efc_fields = [
        ("efc_type", "efc.type"),
        ("efc_id", "efc.id"),
        ("efc_J", "efc.J"),
        ("efc_pos", "efc.pos"),
        ("efc_margin", "efc.margin"),
        ("efc_D", "efc.D"),
        ("efc_vel", "efc.vel"),
        ("efc_aref", "efc.aref"),
        ("efc_frictionloss", "efc.frictionloss"),
        ("efc_force", "efc.force"),
    ]

    # Copy data to all environments
    for field_name, source_data, target_shape in fields:
        # Skip if field doesn't exist in either source or destination
        if not hasattr(mj_data, field_name) or not hasattr(env.data_wp, field_name):
            continue

        source_data_np = np.array(source_data, dtype=np.float32)
        tensor = torch.from_numpy(source_data_np).to(config.device)

        # Handle scalar time field
        if tensor.dim() == 0:
            tensor = tensor.unsqueeze(0)

        # Reshape tensor to match target shape
        if len(target_shape) == 2:
            if tensor.dim() == 1:
                tensor = tensor.unsqueeze(0).repeat(target_shape[0], 1)
            elif tensor.dim() == 2:
                tensor = tensor.unsqueeze(0).repeat(target_shape[0], 1, 1).squeeze(-1)
        elif len(target_shape) == 3:
            if tensor.dim() == 1:
                # For 1D data that needs to be 3D
                tensor = (
                    tensor.unsqueeze(0)
                    .unsqueeze(-1)
                    .repeat(target_shape[0], 1, target_shape[2])
                )
            elif tensor.dim() == 2:
                tensor = tensor.unsqueeze(0).repeat(target_shape[0], 1, 1)
            elif tensor.dim() == 3:
                tensor = tensor.unsqueeze(0).repeat(target_shape[0], 1, 1, 1).squeeze(1)
        elif len(target_shape) == 4:
            tensor = tensor.unsqueeze(0).repeat(target_shape[0], 1, 1, 1)

        wp.copy(getattr(env.data_wp, field_name), wp.from_torch(tensor))

    # Handle contact struct fields
    for mj_field, wp_field in contact_fields:
        if hasattr(mj_data.contact, mj_field):
            source_data = getattr(mj_data.contact, mj_field)
            source_data_np = np.array(source_data, dtype=np.float32)
            tensor = torch.from_numpy(source_data_np).to(config.device)

            # Handle scalar fields
            if tensor.dim() == 0:
                tensor = tensor.unsqueeze(0)

            # Reshape for batched environments
            if tensor.dim() == 1:
                tensor = tensor.unsqueeze(0).repeat(env.data_wp.nworld, 1)
            elif tensor.dim() == 2:
                tensor = tensor.unsqueeze(0).repeat(env.data_wp.nworld, 1, 1)
            elif tensor.dim() == 3:
                tensor = tensor.unsqueeze(0).repeat(env.data_wp.nworld, 1, 1, 1)

            # Get the destination field using nested attribute access
            dst_obj = env.data_wp
            for attr in wp_field.split("."):
                dst_obj = getattr(dst_obj, attr)
            wp.copy(dst_obj, wp.from_torch(tensor))

    # Handle efc fields - these are direct fields on mj_data
    for mj_field, wp_field in efc_fields:
        if hasattr(mj_data, mj_field):
            source_data = getattr(mj_data, mj_field)
            source_data_np = np.array(source_data, dtype=np.float32)
            tensor = torch.from_numpy(source_data_np).to(config.device)

            # Handle scalar fields
            if tensor.dim() == 0:
                tensor = tensor.unsqueeze(0)

            # Reshape for batched environments
            if tensor.dim() == 1:
                tensor = tensor.unsqueeze(0).repeat(env.data_wp.nworld, 1)
            elif tensor.dim() == 2:
                tensor = tensor.unsqueeze(0).repeat(env.data_wp.nworld, 1, 1)
            elif tensor.dim() == 3:
                tensor = tensor.unsqueeze(0).repeat(env.data_wp.nworld, 1, 1, 1)

            # Get the destination field using nested attribute access
            dst_obj = env.data_wp
            for attr in wp_field.split("."):
                dst_obj = getattr(dst_obj, attr)
            wp.copy(dst_obj, wp.from_torch(tensor))

    return env


def copy_sample_state(
    config: Config, env: MJWPEnv, src_indices: torch.Tensor, dst_indices: torch.Tensor
):
    """Copy simulation state from source samples to destination samples.

    Args:
        config: Config
        env: MJWPEnv environment
        src_indices: Tensor of shape (n,) containing source sample indices
        dst_indices: Tensor of shape (n,) containing destination sample indices
    """
    # Convert to numpy for indexing
    src_idx = src_indices.cpu().numpy()
    dst_idx = dst_indices.cpu().numpy()

    # Get all state data as torch tensors
    qpos = wp.to_torch(env.data_wp.qpos)
    qvel = wp.to_torch(env.data_wp.qvel)
    qacc = wp.to_torch(env.data_wp.qacc)
    time_arr = wp.to_torch(env.data_wp.time)
    ctrl = wp.to_torch(env.data_wp.ctrl)
    act = wp.to_torch(env.data_wp.act)
    act_dot = wp.to_torch(env.data_wp.act_dot)
    qacc_warmstart = wp.to_torch(env.data_wp.qacc_warmstart)
    qfrc_applied = wp.to_torch(env.data_wp.qfrc_applied)
    xfrc_applied = wp.to_torch(env.data_wp.xfrc_applied)
    energy = wp.to_torch(env.data_wp.energy)
    mocap_pos = wp.to_torch(env.data_wp.mocap_pos)
    mocap_quat = wp.to_torch(env.data_wp.mocap_quat)
    xpos = wp.to_torch(env.data_wp.xpos)
    xquat = wp.to_torch(env.data_wp.xquat)
    xmat = wp.to_torch(env.data_wp.xmat)
    xipos = wp.to_torch(env.data_wp.xipos)
    ximat = wp.to_torch(env.data_wp.ximat)
    geom_xpos = wp.to_torch(env.data_wp.geom_xpos)
    geom_xmat = wp.to_torch(env.data_wp.geom_xmat)
    site_xpos = wp.to_torch(env.data_wp.site_xpos)
    site_xmat = wp.to_torch(env.data_wp.site_xmat)
    cacc = wp.to_torch(env.data_wp.cacc)
    cfrc_int = wp.to_torch(env.data_wp.cfrc_int)
    cfrc_ext = wp.to_torch(env.data_wp.cfrc_ext)
    sensordata = wp.to_torch(env.data_wp.sensordata)
    actuator_length = wp.to_torch(env.data_wp.actuator_length)
    actuator_velocity = wp.to_torch(env.data_wp.actuator_velocity)
    actuator_force = wp.to_torch(env.data_wp.actuator_force)
    ten_length = wp.to_torch(env.data_wp.ten_length)
    ten_velocity = wp.to_torch(env.data_wp.ten_velocity)

    # Copy from src to dst
    qpos[dst_idx] = qpos[src_idx]
    qvel[dst_idx] = qvel[src_idx]
    qacc[dst_idx] = qacc[src_idx]
    time_arr[dst_idx] = time_arr[src_idx]
    ctrl[dst_idx] = ctrl[src_idx]
    act[dst_idx] = act[src_idx]
    act_dot[dst_idx] = act_dot[src_idx]
    qacc_warmstart[dst_idx] = qacc_warmstart[src_idx]
    qfrc_applied[dst_idx] = qfrc_applied[src_idx]
    xfrc_applied[dst_idx] = xfrc_applied[src_idx]
    energy[dst_idx] = energy[src_idx]
    mocap_pos[dst_idx] = mocap_pos[src_idx]
    mocap_quat[dst_idx] = mocap_quat[src_idx]
    xpos[dst_idx] = xpos[src_idx]
    xquat[dst_idx] = xquat[src_idx]
    xmat[dst_idx] = xmat[src_idx]
    xipos[dst_idx] = xipos[src_idx]
    ximat[dst_idx] = ximat[src_idx]
    geom_xpos[dst_idx] = geom_xpos[src_idx]
    geom_xmat[dst_idx] = geom_xmat[src_idx]
    site_xpos[dst_idx] = site_xpos[src_idx]
    site_xmat[dst_idx] = site_xmat[src_idx]
    cacc[dst_idx] = cacc[src_idx]
    cfrc_int[dst_idx] = cfrc_int[src_idx]
    cfrc_ext[dst_idx] = cfrc_ext[src_idx]
    sensordata[dst_idx] = sensordata[src_idx]
    actuator_length[dst_idx] = actuator_length[src_idx]
    actuator_velocity[dst_idx] = actuator_velocity[src_idx]
    actuator_force[dst_idx] = actuator_force[src_idx]
    ten_length[dst_idx] = ten_length[src_idx]
    ten_velocity[dst_idx] = ten_velocity[src_idx]

    # Copy back to warp arrays
    wp.copy(env.data_wp.qpos, wp.from_torch(qpos))
    wp.copy(env.data_wp.qvel, wp.from_torch(qvel))
    wp.copy(env.data_wp.qacc, wp.from_torch(qacc))
    wp.copy(env.data_wp.time, wp.from_torch(time_arr))
    wp.copy(env.data_wp.ctrl, wp.from_torch(ctrl))
    wp.copy(env.data_wp.act, wp.from_torch(act))
    wp.copy(env.data_wp.act_dot, wp.from_torch(act_dot))
    wp.copy(env.data_wp.qacc_warmstart, wp.from_torch(qacc_warmstart))
    wp.copy(env.data_wp.qfrc_applied, wp.from_torch(qfrc_applied))
    wp.copy(env.data_wp.xfrc_applied, wp.from_torch(xfrc_applied))
    wp.copy(env.data_wp.energy, wp.from_torch(energy))
    wp.copy(env.data_wp.mocap_pos, wp.from_torch(mocap_pos))
    wp.copy(env.data_wp.mocap_quat, wp.from_torch(mocap_quat))
    wp.copy(env.data_wp.xpos, wp.from_torch(xpos))
    wp.copy(env.data_wp.xquat, wp.from_torch(xquat))
    wp.copy(env.data_wp.xmat, wp.from_torch(xmat))
    wp.copy(env.data_wp.xipos, wp.from_torch(xipos))
    wp.copy(env.data_wp.ximat, wp.from_torch(ximat))
    wp.copy(env.data_wp.geom_xpos, wp.from_torch(geom_xpos))
    wp.copy(env.data_wp.geom_xmat, wp.from_torch(geom_xmat))
    wp.copy(env.data_wp.site_xpos, wp.from_torch(site_xpos))
    wp.copy(env.data_wp.site_xmat, wp.from_torch(site_xmat))
    wp.copy(env.data_wp.cacc, wp.from_torch(cacc))
    wp.copy(env.data_wp.cfrc_int, wp.from_torch(cfrc_int))
    wp.copy(env.data_wp.cfrc_ext, wp.from_torch(cfrc_ext))
    wp.copy(env.data_wp.sensordata, wp.from_torch(sensordata))
    wp.copy(env.data_wp.actuator_length, wp.from_torch(actuator_length))
    wp.copy(env.data_wp.actuator_velocity, wp.from_torch(actuator_velocity))
    wp.copy(env.data_wp.actuator_force, wp.from_torch(actuator_force))
    wp.copy(env.data_wp.ten_length, wp.from_torch(ten_length))
    wp.copy(env.data_wp.ten_velocity, wp.from_torch(ten_velocity))


def _copy_state(src: mjwarp.Data, dst: mjwarp.Data):
    """Copy the state from src to dst

    TODO: this function is a temporary solution for domain randomization. A better way should be defining a new warp kernel to update simulation parameter accordingly.

    Args:
        src: mjwarp.Data
            the source data to be copied from
        dst: mjwarp.Data
            the destination data to be copied to
    """
    # Core state variables
    wp.copy(dst.qpos, src.qpos)
    wp.copy(dst.qvel, src.qvel)
    wp.copy(dst.qacc, src.qacc)
    wp.copy(dst.time, src.time)
    wp.copy(dst.ctrl, src.ctrl)
    wp.copy(dst.act, src.act)
    wp.copy(dst.act_dot, src.act_dot)
    wp.copy(dst.qacc_warmstart, src.qacc_warmstart)

    # Forces and applied forces
    wp.copy(dst.qfrc_applied, src.qfrc_applied)
    wp.copy(dst.xfrc_applied, src.xfrc_applied)

    # Energy tracking
    wp.copy(dst.energy, src.energy)

    # Mocap data
    wp.copy(dst.mocap_pos, src.mocap_pos)
    wp.copy(dst.mocap_quat, src.mocap_quat)

    # Spatial transformations
    wp.copy(dst.xpos, src.xpos)
    wp.copy(dst.xquat, src.xquat)
    wp.copy(dst.xmat, src.xmat)
    wp.copy(dst.xipos, src.xipos)
    wp.copy(dst.ximat, src.ximat)

    # Geometry positions
    wp.copy(dst.geom_xpos, src.geom_xpos)
    wp.copy(dst.geom_xmat, src.geom_xmat)
    wp.copy(dst.site_xpos, src.site_xpos)
    wp.copy(dst.site_xmat, src.site_xmat)

    # Camera and lighting (if present)
    if hasattr(src, "cam_xpos") and hasattr(dst, "cam_xpos"):
        wp.copy(dst.cam_xpos, src.cam_xpos)
        wp.copy(dst.cam_xmat, src.cam_xmat)
    if hasattr(src, "light_xpos") and hasattr(dst, "light_xpos"):
        wp.copy(dst.light_xpos, src.light_xpos)
        wp.copy(dst.light_xdir, src.light_xdir)

    # Body dynamics
    wp.copy(dst.cacc, src.cacc)
    wp.copy(dst.cfrc_int, src.cfrc_int)
    wp.copy(dst.cfrc_ext, src.cfrc_ext)

    # Sensor data
    wp.copy(dst.sensordata, src.sensordata)

    # Actuator data
    wp.copy(dst.actuator_length, src.actuator_length)
    wp.copy(dst.actuator_velocity, src.actuator_velocity)
    wp.copy(dst.actuator_force, src.actuator_force)

    # Tendon data
    wp.copy(dst.ten_length, src.ten_length)
    wp.copy(dst.ten_velocity, src.ten_velocity)

    # Contact struct - copy all fields
    wp.copy(dst.contact.dist, src.contact.dist)
    wp.copy(dst.contact.pos, src.contact.pos)
    wp.copy(dst.contact.frame, src.contact.frame)
    wp.copy(dst.contact.includemargin, src.contact.includemargin)
    wp.copy(dst.contact.friction, src.contact.friction)
    wp.copy(dst.contact.solref, src.contact.solref)
    wp.copy(dst.contact.solreffriction, src.contact.solreffriction)
    wp.copy(dst.contact.solimp, src.contact.solimp)
    wp.copy(dst.contact.dim, src.contact.dim)
    wp.copy(dst.contact.geom, src.contact.geom)
    wp.copy(dst.contact.efc_address, src.contact.efc_address)
    wp.copy(dst.contact.worldid, src.contact.worldid)

    # Constraint (efc) struct - copy all fields
    wp.copy(dst.efc.type, src.efc.type)
    wp.copy(dst.efc.id, src.efc.id)
    wp.copy(dst.efc.J, src.efc.J)
    wp.copy(dst.efc.pos, src.efc.pos)
    wp.copy(dst.efc.margin, src.efc.margin)
    wp.copy(dst.efc.D, src.efc.D)
    wp.copy(dst.efc.vel, src.efc.vel)
    wp.copy(dst.efc.aref, src.efc.aref)
    wp.copy(dst.efc.frictionloss, src.efc.frictionloss)
    wp.copy(dst.efc.force, src.efc.force)
    # Note: The workspace fields (Jaref, Ma, grad, etc.) are typically not needed for state transfer
    # as they are recomputed during solving, but include if needed:
    # wp.copy(dst.efc.Jaref, src.efc.Jaref)
    # wp.copy(dst.efc.Ma, src.efc.Ma)
    # wp.copy(dst.efc.grad, src.efc.grad)
    # ... (other workspace fields)
    #
    return dst
