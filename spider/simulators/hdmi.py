# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Simulator for sampling with HDMI using MuJoCo Warp GPU batch stepping.

Architecture (R005 — aligned with MJWP):
- HDMI env created with num_envs=1 (for reference data / reward config extraction)
- MuJoCo Warp GPU environment from mjlab scene.xml (batched worlds)
- ctrl = joint position targets (radians), PD via XML affine actuators
- Reward reference precomputed from motion dataset (GPU tensors)
- Reward computed on GPU from warp xpos/xquat (zero-copy)
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import loguru
import mujoco
import mujoco_warp as mjwarp
import numpy as np
import torch
import warp as wp
from omegaconf import OmegaConf

from spider.config import Config

# Initialize Warp once per process
try:
    wp.init()
except RuntimeError:
    pass


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class HDMIEnv:
    """HDMI environment backed by MuJoCo Warp GPU."""

    # MuJoCo Warp GPU environment (for MPC batch simulation)
    model_cpu: mujoco.MjModel
    data_cpu: mujoco.MjData
    model_wp: mjwarp.Model
    data_wp: mjwarp.Data  # nworld=N
    data_wp_prev: mjwarp.Data  # save/load buffer
    graph: wp.ScopedCapture.Graph  # CUDA graph
    device: str
    num_worlds: int

    # HDMI env (for reference data extraction)
    hdmi_env: object = None  # SimpleEnv — kept alive for command_manager access

    # Reward configuration
    _rcfg: dict = field(default_factory=dict)
    _precomputed_ref: dict = field(default_factory=dict)
    _ref_step: int = 0

    # Cached attributes from HDMI env (for get_reference compatibility)
    action_manager: object = None
    command_manager: object = None
    max_episode_length: int = 0
    action_spec_shape: tuple = ()


# ---------------------------------------------------------------------------
# Quaternion math utilities (wxyz convention)
# ---------------------------------------------------------------------------


def _quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    """Quaternion conjugate. q: (..., 4) wxyz."""
    return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)


def _quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Hamilton product of two quaternions. q1, q2: (..., 4) wxyz."""
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dim=-1,
    )


def _quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vector v by quaternion q. q: (..., 4) wxyz, v: (..., 3)."""
    t = 2.0 * torch.cross(q[..., 1:], v, dim=-1)
    return v + q[..., :1] * t + torch.cross(q[..., 1:], t, dim=-1)


def _quat_apply_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vector v by inverse of quaternion q."""
    return _quat_apply(_quat_conjugate(q), v)


def _yaw_quat(q: torch.Tensor) -> torch.Tensor:
    """Extract yaw-only rotation from quaternion. q: (..., 4) wxyz."""
    w, x, y, z = q.unbind(-1)
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return torch.stack(
        [
            torch.cos(yaw / 2),
            torch.zeros_like(yaw),
            torch.zeros_like(yaw),
            torch.sin(yaw / 2),
        ],
        dim=-1,
    )


def _axis_angle_from_quat(q: torch.Tensor) -> torch.Tensor:
    """Convert quaternion to axis-angle. q: (..., 4) wxyz -> (..., 3)."""
    sin_half = torch.norm(q[..., 1:], dim=-1, keepdim=True).clamp(min=1e-8)
    cos_half = q[..., :1]
    angle = 2.0 * torch.atan2(sin_half, cos_half)
    axis = q[..., 1:] / sin_half
    return axis * angle


# ---------------------------------------------------------------------------
# Name resolution
# ---------------------------------------------------------------------------


def _resolve_names(patterns: list[str], name_list: list[str]):
    """Resolve regex patterns against a list of names.

    Returns (indices, names) sorted by name for consistency.
    """
    matched = []
    seen = set()
    for pattern in patterns:
        regex = re.compile(f"^{pattern}$")
        for i, name in enumerate(name_list):
            if name and regex.match(name) and i not in seen:
                matched.append((i, name))
                seen.add(i)
    matched.sort(key=lambda x: x[1])
    return [m[0] for m in matched], [m[1] for m in matched]


def _find_in_scene(
    model: mujoco.MjModel, obj_type: int, name: str
) -> int:
    """Find a named entity in the scene model, trying common prefixes.

    Scene XML uses robot/ and suitcase/ prefixes on all entities.
    """
    mid = mujoco.mj_name2id(model, obj_type, name)
    if mid >= 0:
        return mid
    for prefix in ("robot/", "suitcase/"):
        mid = mujoco.mj_name2id(model, obj_type, f"{prefix}{name}")
        if mid >= 0:
            return mid
    # Try stripping prefix if name already has one
    if "/" in name:
        stripped = name.split("/", 1)[1]
        mid = mujoco.mj_name2id(model, obj_type, stripped)
        if mid >= 0:
            return mid
    return -1


# ---------------------------------------------------------------------------
# CUDA graph compilation (from mjwp.py)
# ---------------------------------------------------------------------------


def _compile_step(
    model_wp: mjwarp.Model, data_wp: mjwarp.Data, decimation: int = 1
) -> wp.ScopedCapture.Graph:
    """Capture a CUDA graph that runs decimation × mjwarp.step."""

    def _step_n():
        for _ in range(decimation):
            mjwarp.step(model_wp, data_wp)

    with wp.ScopedCapture() as capture:
        _step_n()
    wp.synchronize()
    return capture.graph


# ---------------------------------------------------------------------------
# State copy helpers (from mjwp.py)
# ---------------------------------------------------------------------------


def _copy_state(src: mjwarp.Data, dst: mjwarp.Data):
    """Copy full simulation state from src to dst (GPU bulk copy)."""
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

    # Contact struct
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

    # Constraint (efc) struct
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

    return dst


def _broadcast_state(data_wp, num_worlds: int):
    """Broadcast state from first world to all worlds (from mjwp.py)."""
    qpos0 = wp.to_torch(data_wp.qpos)[:1]
    qvel0 = wp.to_torch(data_wp.qvel)[:1]
    time0 = wp.to_torch(data_wp.time)[:1]
    ctrl0 = wp.to_torch(data_wp.ctrl)[:1]

    if time0.dim() == 1:
        time_repeated = time0.repeat(num_worlds)
    else:
        time_repeated = time0.repeat(num_worlds, 1)

    wp.copy(data_wp.qpos, wp.from_torch(qpos0.repeat(num_worlds, 1)))
    wp.copy(data_wp.qvel, wp.from_torch(qvel0.repeat(num_worlds, 1)))
    wp.copy(data_wp.time, wp.from_torch(time_repeated))
    wp.copy(data_wp.ctrl, wp.from_torch(ctrl0.repeat(num_worlds, 1)))

    qacc0 = wp.to_torch(data_wp.qacc)[:1]
    wp.copy(data_wp.qacc, wp.from_torch(qacc0.repeat(num_worlds, 1)))

    act0 = wp.to_torch(data_wp.act)[:1]
    wp.copy(data_wp.act, wp.from_torch(act0.repeat(num_worlds, 1)))

    act_dot0 = wp.to_torch(data_wp.act_dot)[:1]
    wp.copy(data_wp.act_dot, wp.from_torch(act_dot0.repeat(num_worlds, 1)))

    qfrc_applied0 = wp.to_torch(data_wp.qfrc_applied)[:1]
    wp.copy(data_wp.qfrc_applied, wp.from_torch(qfrc_applied0.repeat(num_worlds, 1)))

    xfrc_applied0 = wp.to_torch(data_wp.xfrc_applied)[:1]
    wp.copy(
        data_wp.xfrc_applied,
        wp.from_torch(xfrc_applied0.repeat(num_worlds, 1, 1)),
    )

    mocap_pos0 = wp.to_torch(data_wp.mocap_pos)[:1]
    wp.copy(data_wp.mocap_pos, wp.from_torch(mocap_pos0.repeat(num_worlds, 1, 1)))

    mocap_quat0 = wp.to_torch(data_wp.mocap_quat)[:1]
    wp.copy(data_wp.mocap_quat, wp.from_torch(mocap_quat0.repeat(num_worlds, 1, 1)))

    xpos0 = wp.to_torch(data_wp.xpos)[:1]
    wp.copy(data_wp.xpos, wp.from_torch(xpos0.repeat(num_worlds, 1, 1)))

    xquat0 = wp.to_torch(data_wp.xquat)[:1]
    wp.copy(data_wp.xquat, wp.from_torch(xquat0.repeat(num_worlds, 1, 1)))

    xmat0 = wp.to_torch(data_wp.xmat)[:1]
    wp.copy(data_wp.xmat, wp.from_torch(xmat0.repeat(num_worlds, 1, 1, 1)))

    geom_xpos0 = wp.to_torch(data_wp.geom_xpos)[:1]
    wp.copy(data_wp.geom_xpos, wp.from_torch(geom_xpos0.repeat(num_worlds, 1, 1)))

    geom_xmat0 = wp.to_torch(data_wp.geom_xmat)[:1]
    wp.copy(data_wp.geom_xmat, wp.from_torch(geom_xmat0.repeat(num_worlds, 1, 1, 1)))

    site_xpos0 = wp.to_torch(data_wp.site_xpos)[:1]
    wp.copy(data_wp.site_xpos, wp.from_torch(site_xpos0.repeat(num_worlds, 1, 1)))


# ---------------------------------------------------------------------------
# Key functions
# ---------------------------------------------------------------------------


def _create_hdmi_env(config: Config):
    """Create HDMI SimpleEnv (num_envs=1) for reference data extraction."""
    import active_adaptation

    active_adaptation.set_backend("mujoco")
    from active_adaptation.envs import SimpleEnv

    hdmi_dir = os.path.dirname(os.path.dirname(active_adaptation.__file__))

    base_cfg = OmegaConf.load(
        os.path.join(hdmi_dir, "cfg/task/base/hdmi-base.yaml")
    )
    task_cfg = OmegaConf.load(
        os.path.join(hdmi_dir, f"cfg/task/G1/hdmi/{config.task}.yaml")
    )
    cfg = OmegaConf.merge(base_cfg, task_cfg)

    cfg.num_envs = 1
    cfg.viewer.headless = True  # Always headless — we use our own viewer
    cfg.viewer.env_spacing = 0.0
    OmegaConf.set_struct(cfg, False)

    # Minimal observation groups
    for key in list(cfg.observation.keys()):
        if key not in ["command", "policy", "priv"] and not key.endswith("_"):
            cfg.observation.pop(key)

    # Disable randomizations
    for rng_key in ("pose_range", "velocity_range", "object_pose_range"):
        if rng_key in cfg.command:
            for k in cfg.command[rng_key]:
                cfg.command[rng_key][k] = [0.0, 0.0]
    if "init_joint_pos_noise" in cfg.command:
        cfg.command.init_joint_pos_noise = 0.0
    if "init_joint_vel_noise" in cfg.command:
        cfg.command.init_joint_vel_noise = 0.0
    cfg.command.sample_motion = False
    cfg.command.reset_range = None

    if "action" in cfg:
        cfg.action.min_delay = 0
        cfg.action.max_delay = 0
        cfg.action.alpha = [1.0, 1.0]

    cfg.randomization = {}

    # Keep only tracking reward groups
    filtered_rewards = {}
    for group_name, group_params in cfg.reward.items():
        if "tracking" not in group_name.lower():
            continue
        filtered_rewards[group_name] = group_params
        for key in list(group_params.keys()):
            if "vel" in key.lower():
                del filtered_rewards[group_name][key]
    cfg.reward = filtered_rewards

    env = SimpleEnv(cfg)
    env.eval()
    env.reset()
    return env


def setup_env(config: Config, ref_data: tuple[torch.Tensor, ...]) -> HDMIEnv:
    """Setup HDMI env + MuJoCo Warp GPU environment.

    1. Creates HDMI SimpleEnv (num_envs=1) for reference data extraction
    2. Loads mjlab scene.xml and creates MuJoCo Warp GPU batched environment
    """
    del ref_data  # HDMI has built-in reference

    # 1. Create HDMI env for reference data
    hdmi_env = _create_hdmi_env(config)

    N = int(config.num_samples)
    device = str(config.device)

    # 2. Load mjlab scene.xml
    scene_xml_path = str(
        Path("example_datasets/processed/hdmi")
        / config.robot_type
        / config.embodiment_type
        / config.task
        / "scene"
        / "mjlab scene.xml"
    )
    if not Path(scene_xml_path).exists():
        raise FileNotFoundError(f"Scene XML not found: {scene_xml_path}")

    model_cpu = mujoco.MjModel.from_xml_path(scene_xml_path)
    # Use small physics timestep with decimation for stability (matches HDMI)
    physics_dt = 0.002
    decimation = int(round(config.sim_dt / physics_dt))
    model_cpu.opt.timestep = physics_dt
    model_cpu.opt.iterations = 5
    model_cpu.opt.ls_iterations = 10
    model_cpu.opt.o_solref = [0.02, 1.0]
    model_cpu.opt.o_solimp = [0.9, 0.95, 0.001, 0.5, 2]
    model_cpu.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST

    loguru.logger.info(
        f"Scene XML: nq={model_cpu.nq}, nv={model_cpu.nv}, nu={model_cpu.nu}, "
        f"nbody={model_cpu.nbody}, physics_dt={physics_dt}, decimation={decimation}"
    )

    # 3. Override actuator gains with HDMI's PD gains (scene XML gains are too weak)
    robot = hdmi_env.scene["robot"]
    hdmi_stiffness = robot._data.joint_stiffness[0].cpu().numpy()
    hdmi_damping = robot._data.joint_damping[0].cpu().numpy()
    hdmi_joint_names = [
        mujoco.mj_id2name(hdmi_env.sim.mj_model, mujoco.mjtObj.mjOBJ_JOINT, ji)
        for ji in range(hdmi_env.sim.mj_model.njnt)
        if hdmi_env.sim.mj_model.jnt_type[ji] == 3  # hinge only
    ]
    for ai in range(model_cpu.nu):
        act_name = mujoco.mj_id2name(model_cpu, mujoco.mjtObj.mjOBJ_ACTUATOR, ai)
        # Actuator name = "robot/{joint_name}" → strip prefix
        joint_name = act_name.split("/", 1)[1] if "/" in act_name else act_name
        if joint_name in hdmi_joint_names:
            idx = hdmi_joint_names.index(joint_name)
            kp = float(hdmi_stiffness[idx])
            kd = float(hdmi_damping[idx])
            # affine actuator: gainprm[0]=Kp, biasprm=[0, -Kp, -Kd]
            model_cpu.actuator_gainprm[ai, 0] = kp
            model_cpu.actuator_biasprm[ai, 1] = -kp
            model_cpu.actuator_biasprm[ai, 2] = -kd

    loguru.logger.info("Actuator gains overridden with HDMI PD gains")

    # 4. Initialize CPU data with initial pose from HDMI env
    data_cpu = mujoco.MjData(model_cpu)

    # Get initial qpos from HDMI env and map to scene model
    hdmi_mj_data = hdmi_env.sim.mj_data
    hdmi_mj_model = hdmi_env.sim.mj_model

    # Copy freejoint positions from MOTION DATA (HDMI mj_data has default pose)
    cmd = hdmi_env.command_manager
    motion_data_init = cmd.dataset.get_slice(cmd.motion_ids, 0, steps=1)
    root_body_idx = cmd.root_body_idx_motion
    obj_body_idx = cmd.object_body_id_motion

    freejoint_init = {
        "pelvis": {
            "pos": motion_data_init.body_pos_w[0, 0, root_body_idx].numpy(),
            "quat": motion_data_init.body_quat_w[0, 0, root_body_idx].numpy(),
        },
        "suitcase": {
            "pos": motion_data_init.body_pos_w[0, 0, obj_body_idx].numpy(),
            "quat": motion_data_init.body_quat_w[0, 0, obj_body_idx].numpy(),
        },
    }

    for ji in range(model_cpu.njnt):
        if model_cpu.jnt_type[ji] != 0:  # freejoint only
            continue
        scene_bid = model_cpu.jnt_bodyid[ji]
        scene_bname = mujoco.mj_id2name(
            model_cpu, mujoco.mjtObj.mjOBJ_BODY, scene_bid
        )
        bare_name = scene_bname.split("/")[-1] if scene_bname else ""
        if bare_name not in freejoint_init:
            continue
        dst_qadr = model_cpu.jnt_qposadr[ji]
        init = freejoint_init[bare_name]
        data_cpu.qpos[dst_qadr : dst_qadr + 3] = init["pos"]
        data_cpu.qpos[dst_qadr + 3 : dst_qadr + 7] = init["quat"]
        loguru.logger.info(
            f"Freejoint init from motion: {scene_bname} "
            f"pos={init['pos'].tolist()}"
        )

    # Also set hinge joints from motion data initial frame
    motion_joint_names = cmd.dataset.joint_names
    joint_pos_init = motion_data_init.joint_pos[0, 0]  # (njoint,)
    for mi, jname in enumerate(motion_joint_names):
        mj_jid = _find_in_scene(model_cpu, mujoco.mjtObj.mjOBJ_JOINT, jname)
        if mj_jid < 0:
            continue
        if model_cpu.jnt_type[mj_jid] != 3:  # hinge only
            continue
        qadr = model_cpu.jnt_qposadr[mj_jid]
        data_cpu.qpos[qadr] = float(joint_pos_init[mi])

    mujoco.mj_step(model_cpu, data_cpu)

    # 5. Create MuJoCo Warp GPU environment
    wp.set_device(device)
    with wp.ScopedDevice(device):
        model_wp = mjwarp.put_model(model_cpu)
        data_wp = mjwarp.put_data(
            model_cpu,
            data_cpu,
            nworld=N,
            nconmax=int(config.nconmax_per_env),
            njmax=int(config.njmax_per_env),
        )
        data_wp_prev = mjwarp.put_data(
            model_cpu,
            data_cpu,
            nworld=N,
            nconmax=int(config.nconmax_per_env),
            njmax=int(config.njmax_per_env),
        )
        graph = _compile_step(model_wp, data_wp, decimation=decimation)

    loguru.logger.info(
        f"MuJoCo Warp GPU env created: N={N}, device={device}"
    )

    # 6. Build reward configuration
    cmd = hdmi_env.command_manager
    mj_model = model_cpu
    tracking_names = list(cmd.tracking_keypoint_names)

    def _body_reward_ids(patterns):
        idx, names = _resolve_names(patterns, tracking_names)
        mj_ids = [
            _find_in_scene(mj_model, mujoco.mjtObj.mjOBJ_BODY, n)
            for n in names
        ]
        return (
            torch.tensor(idx, dtype=torch.long, device=device),
            torch.tensor(mj_ids, dtype=torch.long, device=device),
        )

    upper_motion_idx, upper_mj_ids = _body_reward_ids(
        [".*_shoulder_pitch_link", ".*_elbow_link", ".*_wrist_yaw_link"]
    )
    lower_motion_idx, lower_mj_ids = _body_reward_ids(
        [".*_hip_pitch_link", ".*_knee_link", ".*_ankle_roll_link"]
    )
    root_motion_idx = tracking_names.index("pelvis")
    root_mj_id = _find_in_scene(mj_model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")

    object_entity = cmd.object
    # Find object body in scene model by name
    hdmi_obj_body_id = object_entity._body_id
    hdmi_obj_body_name = mujoco.mj_id2name(
        hdmi_mj_model, mujoco.mjtObj.mjOBJ_BODY, hdmi_obj_body_id
    )
    object_mj_id = _find_in_scene(
        mj_model, mujoco.mjtObj.mjOBJ_BODY, hdmi_obj_body_name
    )
    if object_mj_id < 0:
        # Fallback: scan for suitcase-like body
        for bi in range(mj_model.nbody):
            bname = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, bi)
            if bname and "suitcase" in bname.lower():
                object_mj_id = bi
                break
    loguru.logger.info(f"Object body: {hdmi_obj_body_name} -> mj_id={object_mj_id}")

    eef_names = ["left_wrist_yaw_link", "right_wrist_yaw_link"]
    eef_mj_ids = torch.tensor(
        [
            _find_in_scene(mj_model, mujoco.mjtObj.mjOBJ_BODY, n)
            for n in eef_names
        ],
        dtype=torch.long,
        device=device,
    )

    tracking_joint_names = list(cmd.tracking_joint_names)
    joint_patterns = [
        "waist_.*_joint",
        ".*_hip_.*_joint",
        ".*_knee_joint",
        ".*_shoulder_.*_joint",
        ".*_elbow_joint",
    ]
    jt_motion_idx, jt_names = _resolve_names(joint_patterns, tracking_joint_names)
    jt_qpos_addrs = []
    for jname in jt_names:
        jid = _find_in_scene(mj_model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        jt_qpos_addrs.append(mj_model.jnt_qposadr[jid])

    contact_target_offset = cmd.contact_target_pos_offset[0].to(device)
    contact_eef_offset = cmd.contact_eef_pos_offset[0].to(device)

    rcfg = {
        "upper_motion_idx": upper_motion_idx,
        "upper_mj_ids": upper_mj_ids,
        "upper_sigma_pos": 0.5,
        "upper_sigma_ori": 1.0,
        "lower_motion_idx": lower_motion_idx,
        "lower_mj_ids": lower_mj_ids,
        "lower_sigma_pos": 0.5,
        "lower_sigma_ori": 1.0,
        "root_motion_idx": root_motion_idx,
        "root_mj_id": root_mj_id,
        "root_sigma_pos": 0.5,
        "root_sigma_ori": 0.5,
        "object_mj_id": object_mj_id,
        "object_sigma_pos": 0.5,
        "object_sigma_ori": 0.5,
        "eef_mj_ids": eef_mj_ids,
        "eef_pos_sigma": 0.3,
        "eef_frc_sigma": 40.0,
        "eef_frc_thres": 10.0,
        "eef_gain": 5.0,
        "contact_target_offset": contact_target_offset,
        "contact_eef_offset": contact_eef_offset,
        "jt_motion_idx": torch.tensor(jt_motion_idx, dtype=torch.long, device=device),
        "jt_qpos_addrs": torch.tensor(jt_qpos_addrs, dtype=torch.long, device=device),
        "jt_sigma": 0.25,
    }

    return HDMIEnv(
        model_cpu=model_cpu,
        data_cpu=data_cpu,
        model_wp=model_wp,
        data_wp=data_wp,
        data_wp_prev=data_wp_prev,
        graph=graph,
        device=device,
        num_worlds=N,
        hdmi_env=hdmi_env,
        _rcfg=rcfg,
        action_manager=hdmi_env.action_manager,
        command_manager=hdmi_env.command_manager,
        max_episode_length=hdmi_env.max_episode_length,
        action_spec_shape=hdmi_env.action_spec.shape,
    )


# ---------------------------------------------------------------------------
# Precompute all reference data
# ---------------------------------------------------------------------------


def precompute_reward_reference(config: Config, env: HDMIEnv):
    """Precompute all reward reference data for the full trajectory."""
    cmd = env.command_manager
    device = env.device

    total_steps = config.max_sim_steps + config.horizon_steps + config.ctrl_steps + 10

    motion_data = cmd.dataset.get_slice(cmd.motion_ids, 0, steps=total_steps)

    tracking_body_idx = cmd.tracking_body_indices_motion
    tracking_joint_idx = cmd.tracking_joint_indices_motion
    root_body_idx = cmd.root_body_idx_motion
    obj_body_idx = cmd.object_body_id_motion

    ref = {}
    ref["body_pos"] = motion_data.body_pos_w[0, :, tracking_body_idx, :].to(device)
    ref["body_quat"] = motion_data.body_quat_w[0, :, tracking_body_idx, :].to(device)
    ref["root_pos"] = motion_data.body_pos_w[0, :, root_body_idx, :].to(device)
    ref["root_quat"] = motion_data.body_quat_w[0, :, root_body_idx, :].to(device)
    ref["joint_pos"] = motion_data.joint_pos[0, :, tracking_joint_idx].to(device)
    ref["object_pos"] = motion_data.body_pos_w[0, :, obj_body_idx, :].to(device)
    ref["object_quat"] = motion_data.body_quat_w[0, :, obj_body_idx, :].to(device)

    motion_start = int(cmd.motion_starts[0])
    motion_end = int(cmd.dataset.ends[cmd.motion_ids[0]])
    global_idx = (motion_start + torch.arange(total_steps)).clamp_max_(motion_end - 1)
    ref["object_contact"] = cmd._object_contact[global_idx].float().to(device)

    env._precomputed_ref = ref
    env._ref_step = 0


# ---------------------------------------------------------------------------
# State save / load / sync
# ---------------------------------------------------------------------------


def save_state(env: HDMIEnv):
    """Save full GPU state via _copy_state + ref step counter."""
    _copy_state(env.data_wp, env.data_wp_prev)
    return {"ref_step": env._ref_step}


def load_state(env: HDMIEnv, state):
    """Restore full GPU state via _copy_state + ref step counter."""
    _copy_state(env.data_wp_prev, env.data_wp)
    env._ref_step = state["ref_step"]
    return env


def sync_env(config: Config, env: HDMIEnv):
    """Broadcast world 0 state to all worlds."""
    _broadcast_state(env.data_wp, env.num_worlds)
    return env


# ---------------------------------------------------------------------------
# Step
# ---------------------------------------------------------------------------


def step_env(config: Config, env: HDMIEnv, ctrl: torch.Tensor):
    """Step all N worlds with joint position targets (radians).

    ctrl: (N, nu) joint position targets — written directly to MuJoCo ctrl.
    PD control is handled by the XML affine actuators on the GPU.
    """
    if ctrl.dim() == 1:
        ctrl = ctrl.unsqueeze(0).repeat(env.num_worlds, 1)
    with wp.ScopedDevice(env.device):
        wp.copy(env.data_wp.ctrl, wp.from_torch(ctrl.to(torch.float32)))
        wp.capture_launch(env.graph)
    env._ref_step += 1
    return env


# ---------------------------------------------------------------------------
# Reward computation helpers
# ---------------------------------------------------------------------------


def _pos_tracking_local(
    xpos_batch: torch.Tensor,
    xquat_batch: torch.Tensor,
    body_mj_ids: torch.Tensor,
    root_mj_id: int,
    ref_body_pos: torch.Tensor,
    ref_root_pos: torch.Tensor,
    ref_root_quat: torch.Tensor,
    sigma: float,
) -> torch.Tensor:
    """Position tracking in root-yaw-relative frame. Returns (N,)."""
    body_pos = xpos_batch[:, body_mj_ids, :]
    root_pos = xpos_batch[:, root_mj_id, :]
    root_quat = xquat_batch[:, root_mj_id, :]

    root_pos_xy = root_pos.clone()
    root_pos_xy[..., 2] = 0.0
    root_quat_yaw = _yaw_quat(root_quat)

    ref_root_xy = ref_root_pos.clone()
    ref_root_xy[..., 2] = 0.0
    ref_root_quat_yaw = _yaw_quat(ref_root_quat)

    B = body_pos.shape[1]
    rp = root_pos_xy.unsqueeze(1).expand(-1, B, -1)
    rq = root_quat_yaw.unsqueeze(1).expand(-1, B, -1)
    ref_rp = ref_root_xy.unsqueeze(1).expand(-1, B, -1)
    ref_rq = ref_root_quat_yaw.unsqueeze(1).expand(-1, B, -1)

    body_local = _quat_apply_inverse(rq, body_pos - rp)
    ref_local = _quat_apply_inverse(ref_rq, ref_body_pos - ref_rp)

    error = (ref_local - body_local).norm(dim=-1).clamp_min(0.0)
    return torch.exp(-error.mean(dim=1) / sigma)


def _ori_tracking_local(
    xquat_batch: torch.Tensor,
    body_mj_ids: torch.Tensor,
    root_mj_id: int,
    ref_body_quat: torch.Tensor,
    ref_root_quat: torch.Tensor,
    sigma: float,
) -> torch.Tensor:
    """Orientation tracking in root-yaw-relative frame. Returns (N,)."""
    body_quat = xquat_batch[:, body_mj_ids, :]
    root_quat = xquat_batch[:, root_mj_id, :]

    root_yaw = _yaw_quat(root_quat)
    ref_root_yaw = _yaw_quat(ref_root_quat)

    B = body_quat.shape[1]
    rq = root_yaw.unsqueeze(1).expand(-1, B, -1)
    ref_rq = ref_root_yaw.unsqueeze(1).expand(-1, B, -1)

    body_local = _quat_mul(_quat_conjugate(rq), body_quat)
    ref_local = _quat_mul(_quat_conjugate(ref_rq), ref_body_quat)

    diff = _quat_mul(_quat_conjugate(ref_local), body_local)
    error = _axis_angle_from_quat(diff).norm(dim=-1).clamp_min(0.0)
    return torch.exp(-error.mean(dim=1) / sigma)


def _pos_tracking_global(
    xpos_batch: torch.Tensor,
    body_mj_ids: torch.Tensor | int,
    ref_body_pos: torch.Tensor,
    sigma: float,
) -> torch.Tensor:
    """Global position tracking. Returns (N,)."""
    if isinstance(body_mj_ids, int):
        pos = xpos_batch[:, body_mj_ids, :].unsqueeze(1)
    else:
        pos = xpos_batch[:, body_mj_ids, :]
    if ref_body_pos.dim() == 2:
        ref_body_pos = ref_body_pos.unsqueeze(1)
    error = (ref_body_pos - pos).norm(dim=-1).clamp_min(0.0)
    return torch.exp(-error.mean(dim=1) / sigma)


def _ori_tracking_global(
    xquat_batch: torch.Tensor,
    body_mj_ids: torch.Tensor | int,
    ref_body_quat: torch.Tensor,
    sigma: float,
) -> torch.Tensor:
    """Global orientation tracking. Returns (N,)."""
    if isinstance(body_mj_ids, int):
        quat = xquat_batch[:, body_mj_ids, :].unsqueeze(1)
    else:
        quat = xquat_batch[:, body_mj_ids, :]
    if ref_body_quat.dim() == 2:
        ref_body_quat = ref_body_quat.unsqueeze(1)
    diff = _quat_mul(_quat_conjugate(ref_body_quat), quat)
    error = _axis_angle_from_quat(diff).norm(dim=-1).clamp_min(0.0)
    return torch.exp(-error.mean(dim=1) / sigma)


# ---------------------------------------------------------------------------
# Reward computation
# ---------------------------------------------------------------------------


def get_reward(
    config: Config,
    env: HDMIEnv,
    ref: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, dict]:
    """Compute HDMI rewards from precomputed reference data.

    Reads xpos/xquat directly from MuJoCo Warp GPU data (zero-copy).
    """
    del ref
    rc = env._rcfg
    device = env.device
    pref = env._precomputed_ref

    T = pref["body_pos"].shape[0]
    t = min(env._ref_step + 1, T - 1)
    t_contact = min(env._ref_step + 2, T - 1)

    # Read xpos/xquat directly from Warp (GPU, zero-copy)
    xpos = wp.to_torch(env.data_wp.xpos)  # (N, nbody, 3)
    xquat = wp.to_torch(env.data_wp.xquat)  # (N, nbody, 4)

    # Precomputed reference
    ref_body_pos = pref["body_pos"][t].unsqueeze(0)
    ref_body_quat = pref["body_quat"][t].unsqueeze(0)
    ref_root_pos = pref["root_pos"][t].unsqueeze(0)
    ref_root_quat = pref["root_quat"][t].unsqueeze(0)

    root_mj_id = rc["root_mj_id"]
    W_TRACK = 0.5

    rew_upper_pos = _pos_tracking_local(
        xpos, xquat, rc["upper_mj_ids"], root_mj_id,
        ref_body_pos[:, rc["upper_motion_idx"]],
        ref_root_pos, ref_root_quat, rc["upper_sigma_pos"],
    )
    rew_upper_ori = _ori_tracking_local(
        xquat, rc["upper_mj_ids"], root_mj_id,
        ref_body_quat[:, rc["upper_motion_idx"]],
        ref_root_quat, rc["upper_sigma_ori"],
    )
    rew_lower_pos = _pos_tracking_local(
        xpos, xquat, rc["lower_mj_ids"], root_mj_id,
        ref_body_pos[:, rc["lower_motion_idx"]],
        ref_root_pos, ref_root_quat, rc["lower_sigma_pos"],
    )
    rew_lower_ori = _ori_tracking_local(
        xquat, rc["lower_mj_ids"], root_mj_id,
        ref_body_quat[:, rc["lower_motion_idx"]],
        ref_root_quat, rc["lower_sigma_ori"],
    )
    ref_root_pos_sel = ref_body_pos[:, rc["root_motion_idx"]]
    rew_root_pos = _pos_tracking_global(
        xpos, root_mj_id, ref_root_pos_sel, rc["root_sigma_pos"],
    )
    ref_root_quat_sel = ref_body_quat[:, rc["root_motion_idx"]]
    rew_root_ori = _ori_tracking_global(
        xquat, root_mj_id, ref_root_quat_sel, rc["root_sigma_ori"],
    )

    # Joint tracking — read qpos from Warp
    ref_jt = pref["joint_pos"][t, rc["jt_motion_idx"]].unsqueeze(0)
    qpos_batch = wp.to_torch(env.data_wp.qpos)  # (N, nq) — GPU, zero-copy
    cur_jt = qpos_batch[:, rc["jt_qpos_addrs"]]
    jt_error = (ref_jt - cur_jt).abs().clamp_min(0.0)
    rew_joint = torch.exp(-jt_error.mean(dim=1) / rc["jt_sigma"])

    tracking = W_TRACK * (
        rew_upper_pos + rew_upper_ori
        + rew_lower_pos + rew_lower_ori
        + rew_root_pos + rew_root_ori
        + rew_joint
    )

    # Object tracking
    ref_obj_pos = pref["object_pos"][t].unsqueeze(0)
    obj_pos = xpos[:, rc["object_mj_id"]]
    obj_pos_error = (ref_obj_pos - obj_pos).norm(dim=-1)
    rew_obj_pos = torch.exp(-obj_pos_error / rc["object_sigma_pos"])

    ref_obj_quat = pref["object_quat"][t].unsqueeze(0)
    obj_quat = xquat[:, rc["object_mj_id"]]
    obj_diff = _quat_mul(_quat_conjugate(ref_obj_quat), obj_quat)
    obj_ori_error = _axis_angle_from_quat(obj_diff).norm(dim=-1)
    rew_obj_ori = torch.exp(-obj_ori_error / rc["object_sigma_ori"])

    # EEF contact
    eef_pos = xpos[:, rc["eef_mj_ids"]]
    eef_quat = xquat[:, rc["eef_mj_ids"]]
    obj_pos_e = xpos[:, rc["object_mj_id"]].unsqueeze(1)
    obj_quat_e = xquat[:, rc["object_mj_id"]].unsqueeze(1)

    target_pos = obj_pos_e + _quat_apply(
        obj_quat_e.expand_as(eef_quat), rc["contact_target_offset"].unsqueeze(0)
    )
    contact_eef = eef_pos + _quat_apply(
        eef_quat, rc["contact_eef_offset"].unsqueeze(0)
    )

    eef_dist = (target_pos - contact_eef).norm(dim=-1).clamp_min(0.0)
    pos_rew = torch.exp(-eef_dist / rc["eef_pos_sigma"])

    force_factor = np.exp(-rc["eef_frc_thres"] / rc["eef_frc_sigma"])
    in_range = pref["object_contact"][t_contact].unsqueeze(0)
    gain = rc["eef_gain"]
    contact_rew = pos_rew * force_factor
    rew_contact = (
        contact_rew * in_range * gain + 1.0 - in_range
    ).mean(dim=-1)

    object_tracking = rew_obj_pos + rew_obj_ori + rew_contact
    total = tracking + object_tracking

    return total, {"tracking": tracking, "object_tracking": object_tracking}


# ---------------------------------------------------------------------------
# Terminate / terminal reward
# ---------------------------------------------------------------------------


def get_terminate(
    config: Config, env: HDMIEnv, ref_slice: tuple[torch.Tensor, ...]
) -> torch.Tensor:
    """No termination for HDMI."""
    return torch.zeros(env.num_worlds, device=env.device)


def get_terminal_reward(
    config: Config,
    env: HDMIEnv,
    ref_slice: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, dict]:
    """Terminal reward = scaled get_reward."""
    rew, info = get_reward(config, env, ref_slice)
    return config.terminal_rew_scale * rew, info


# ---------------------------------------------------------------------------
# Trace (visualization)
# ---------------------------------------------------------------------------


def get_trace(config: Config, env: HDMIEnv) -> torch.Tensor:
    """Get trace points (hand + foot + object). Returns (N, 5, 3)."""
    rc = env._rcfg
    xpos = wp.to_torch(env.data_wp.xpos)  # (N, nbody, 3)

    hand_pos = xpos[:, rc["eef_mj_ids"]]

    mj_model = env.model_cpu
    foot_names = ["left_ankle_roll_link", "right_ankle_roll_link"]
    foot_ids = torch.tensor(
        [
            _find_in_scene(mj_model, mujoco.mjtObj.mjOBJ_BODY, n)
            for n in foot_names
        ],
        dtype=torch.long,
        device=env.device,
    )
    foot_pos = xpos[:, foot_ids]
    obj_pos = xpos[:, rc["object_mj_id"]].unsqueeze(1)

    return torch.cat([hand_pos, foot_pos, obj_pos], dim=1)


# ---------------------------------------------------------------------------
# Reference extraction
# ---------------------------------------------------------------------------


def get_reference(
    config: Config, env: HDMIEnv, scene_model=None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract full reference motion from HDMI command manager.

    ctrl_ref = joint position targets (radians) for the scene XML's affine actuators.
    """
    action_manager = env.action_manager
    command_manager = env.command_manager
    mj_model = scene_model if scene_model is not None else env.model_cpu

    motion_data = command_manager.dataset.get_slice(
        command_manager.motion_ids, 0, steps=config.max_sim_steps
    )

    # --- ctrl_ref: joint position targets for scene XML actuators ---
    # Map actuator names to motion dataset joint names
    ctrl_ref_list = []
    for ai in range(mj_model.nu):
        act_name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, ai)
        # Actuator names are "robot/{joint_name}" — strip prefix for motion lookup
        if act_name and "/" in act_name:
            joint_name = act_name.split("/", 1)[1]
        else:
            joint_name = act_name

        # Find in motion dataset
        try:
            mi = command_manager.dataset.joint_names.index(joint_name)
            ctrl_ref_list.append(motion_data.joint_pos[0, :, mi])
        except ValueError:
            loguru.logger.warning(
                f"Actuator '{act_name}' joint '{joint_name}' not in motion dataset, "
                f"using zeros"
            )
            ctrl_ref_list.append(torch.zeros(motion_data.joint_pos.shape[1]))

    ctrl_ref = torch.stack(ctrl_ref_list, dim=-1)  # (T, nu)

    T = motion_data.joint_pos.shape[1]
    nq = mj_model.nq
    nv = mj_model.nv

    # --- Build qpos_ref in scene model's layout ---
    qpos_ref = torch.zeros(T, nq)
    qvel_ref = torch.zeros(T, nv)

    # Robot root body (pelvis) — find freejoint
    root_body_idx = command_manager.root_body_idx_motion
    root_pos = motion_data.body_pos_w[0, :, root_body_idx, :]
    root_quat = motion_data.body_quat_w[0, :, root_body_idx, :]
    root_lin_vel = motion_data.body_lin_vel_w[0, :, root_body_idx, :]
    root_ang_vel = motion_data.body_ang_vel_w[0, :, root_body_idx, :]

    pelvis_jnt_id = -1
    for candidate in ["pelvis_root", "robot/floating_base_joint"]:
        pelvis_jnt_id = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_JOINT, candidate
        )
        if pelvis_jnt_id >= 0:
            break
    if pelvis_jnt_id < 0:
        for ji in range(mj_model.njnt):
            if mj_model.jnt_type[ji] == 0:
                bname = mujoco.mj_id2name(
                    mj_model, mujoco.mjtObj.mjOBJ_BODY, mj_model.jnt_bodyid[ji]
                )
                if bname and "pelvis" in bname.lower():
                    pelvis_jnt_id = ji
                    break

    pelvis_qadr = mj_model.jnt_qposadr[pelvis_jnt_id]
    pelvis_vadr = mj_model.jnt_dofadr[pelvis_jnt_id]
    qpos_ref[:, pelvis_qadr:pelvis_qadr + 3] = root_pos
    qpos_ref[:, pelvis_qadr + 3:pelvis_qadr + 7] = root_quat
    qvel_ref[:, pelvis_vadr:pelvis_vadr + 3] = root_lin_vel
    qvel_ref[:, pelvis_vadr + 3:pelvis_vadr + 6] = root_ang_vel

    # Hinge joints
    motion_joint_names = command_manager.dataset.joint_names
    robot_joint_pos = motion_data.joint_pos[0]
    robot_joint_vel = motion_data.joint_vel[0]

    for mi, jname in enumerate(motion_joint_names):
        mj_jid = _find_in_scene(mj_model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        if mj_jid < 0:
            continue
        qadr = mj_model.jnt_qposadr[mj_jid]
        vadr = mj_model.jnt_dofadr[mj_jid]
        qpos_ref[:, qadr] = robot_joint_pos[:, mi]
        qvel_ref[:, vadr] = robot_joint_vel[:, mi]

    # Object freejoint
    if hasattr(command_manager, "object_body_id_motion"):
        obj_idx = command_manager.object_body_id_motion
        obj_pos = motion_data.body_pos_w[0, :, obj_idx, :]
        obj_quat = motion_data.body_quat_w[0, :, obj_idx, :]
        obj_lin_vel = motion_data.body_lin_vel_w[0, :, obj_idx, :]
        obj_ang_vel = motion_data.body_ang_vel_w[0, :, obj_idx, :]

        obj_jnt_id = -1
        obj_name = getattr(command_manager, "object_asset_name", "suitcase")
        for candidate in [
            f"{obj_name}_root",
            "suitcase_root",
            f"suitcase/{obj_name}",
            f"suitcase/{obj_name}_root",
            f"suitcase/suitcase_root",
        ]:
            obj_jnt_id = mujoco.mj_name2id(
                mj_model, mujoco.mjtObj.mjOBJ_JOINT, candidate
            )
            if obj_jnt_id >= 0:
                break
        if obj_jnt_id < 0:
            # Fallback: find freejoint on suitcase-like body
            for ji in range(mj_model.njnt):
                if mj_model.jnt_type[ji] == 0:
                    bname = mujoco.mj_id2name(
                        mj_model,
                        mujoco.mjtObj.mjOBJ_BODY,
                        mj_model.jnt_bodyid[ji],
                    )
                    if bname and "suitcase" in bname.lower():
                        obj_jnt_id = ji
                        break

        if obj_jnt_id >= 0:
            obj_qadr = mj_model.jnt_qposadr[obj_jnt_id]
            obj_vadr = mj_model.jnt_dofadr[obj_jnt_id]
            qpos_ref[:, obj_qadr:obj_qadr + 3] = obj_pos
            qpos_ref[:, obj_qadr + 3:obj_qadr + 7] = obj_quat
            qvel_ref[:, obj_vadr:obj_vadr + 3] = obj_lin_vel
            qvel_ref[:, obj_vadr + 3:obj_vadr + 6] = obj_ang_vel

    # Pad with repeated last frame
    pad = config.horizon_steps + config.ctrl_steps
    qpos_ref = torch.cat([qpos_ref, qpos_ref[-1:].repeat(pad, 1)], dim=0)
    qvel_ref = torch.cat([qvel_ref, qvel_ref[-1:].repeat(pad, 1) * 0.0], dim=0)
    ctrl_ref = torch.cat([ctrl_ref, ctrl_ref[-1:].repeat(pad, 1)], dim=0)

    assert qpos_ref.shape[-1] == nq, (
        f"qpos shape mismatch: {qpos_ref.shape[-1]} vs {nq}"
    )
    assert qvel_ref.shape[-1] == nv, (
        f"qvel shape mismatch: {qvel_ref.shape[-1]} vs {nv}"
    )
    assert ctrl_ref.shape[-1] == mj_model.nu, (
        f"ctrl shape mismatch: {ctrl_ref.shape[-1]} vs {mj_model.nu}"
    )

    return qpos_ref, qvel_ref, ctrl_ref


# ---------------------------------------------------------------------------
# Domain randomization (no-op for HDMI)
# ---------------------------------------------------------------------------


def save_env_params(config: Config, env: HDMIEnv):
    """No domain randomization for HDMI."""
    return {}


def load_env_params(config: Config, env: HDMIEnv, env_param: dict):
    """No domain randomization for HDMI."""
    return env


# ---------------------------------------------------------------------------
# Sample state copy (for CEM resampling)
# ---------------------------------------------------------------------------


def copy_sample_state(
    config: Config, env: HDMIEnv, src_indices: torch.Tensor, dst_indices: torch.Tensor
):
    """Copy simulation state from source to destination sample indices (GPU)."""
    src_idx = src_indices.cpu().numpy()
    dst_idx = dst_indices.cpu().numpy()

    # Get all state data as torch tensors (zero-copy from Warp)
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

    # Copy from src to dst (in-place GPU tensor ops)
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

    # Write back
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
