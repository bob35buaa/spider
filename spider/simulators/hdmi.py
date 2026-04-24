# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Simulator for sampling with HDMI based on MuJoCo Warp batch stepping.

Reference: https://github.com/LeCAR-Lab/HDMI

This module provides humanoid whole-body retargeting support with SPIDER.
Uses MuJoCo Warp for GPU-batched parallel rollouts instead of HDMI's single-env step.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import mujoco
import mujoco_warp as mjwarp
import numpy as np
import torch
import warp as wp
from omegaconf import OmegaConf

from spider.config import Config
from spider.math import quat_sub
from spider.simulators.mjwp import _broadcast_state, _copy_state

# --
# Data structures
# --


@dataclass
class HDMIWarpEnv:
    """Holds Warp batch simulation state for HDMI."""

    # HDMI environment (for initialization, command manager, action manager)
    hdmi_env: object
    # CPU MuJoCo model/data (for viewer)
    mj_model: mujoco.MjModel
    mj_data: mujoco.MjData
    # Warp batch simulation
    model_wp: mjwarp.Model
    data_wp: mjwarp.Data
    data_wp_prev: mjwarp.Data
    graph: object  # wp.ScopedCapture.Graph
    # Configuration
    device: str
    num_worlds: int
    decimation: int
    physics_dt: float
    # PD control parameters (all in isaac joint order)
    joint_stiffness: torch.Tensor  # (nu_action,)
    joint_damping: torch.Tensor  # (nu_action,)
    action_scaling: torch.Tensor  # (nu_action,)
    default_joint_pos: torch.Tensor  # (1, num_all_joints)
    action_joint_ids: list  # indices into all joints for action joints
    # Joint address mappings (mujoco qpos/qvel addresses in isaac order)
    joint_qposadr_read: np.ndarray
    joint_qveladr_read: np.ndarray
    # Ctrl mapping: for each isaac joint i, _jnt_mjc2isaac[i] = mjc actuator index
    jnt_mjc2isaac: list
    # Environment state tracking
    episode_length_buf: torch.Tensor
    timestamp: int


def _compile_step(
    model_wp: mjwarp.Model, data_wp: mjwarp.Data
) -> object:
    """Compile a CUDA graph for a single mjwarp.step."""

    def _step_once():
        mjwarp.step(model_wp, data_wp)

    with wp.ScopedCapture() as capture:
        _step_once()
    wp.synchronize()
    return capture.graph


# --
# Key functions
# --


def setup_env(config: Config, ref_data: tuple[torch.Tensor, ...]) -> HDMIWarpEnv:
    """Setup and reset the environment backed by HDMI + MuJoCo Warp batch stepping.

    Creates the HDMI environment for initialization and config,
    then sets up Warp batch model/data/graph for parallel rollouts.
    """
    del ref_data  # not used

    # Import HDMI dependencies
    import active_adaptation

    active_adaptation.set_backend("mujoco")
    from active_adaptation.envs.locomotion import SimpleEnv

    # Use active_adaptation module path to find HDMI directory
    hdmi_dir = os.path.dirname(os.path.dirname(active_adaptation.__file__))

    # Load base config
    base_config_path = os.path.join(hdmi_dir, "cfg/task/base/hdmi-base.yaml")
    base_cfg = OmegaConf.load(base_config_path)

    # Load task-specific config (e.g., move_suitcase)
    task_config_path = os.path.join(
        hdmi_dir, f"cfg/task/G1/hdmi/{config.task}.yaml"
    )
    print(f"task_config_path: {task_config_path}")
    task_cfg = OmegaConf.load(task_config_path)

    # Merge configurations (task overrides base)
    cfg = OmegaConf.merge(base_cfg, task_cfg)

    # Override with SPIDER config parameters
    cfg.num_envs = 1
    cfg.viewer.headless = "mjlab" not in config.viewer.lower()
    cfg.viewer.env_spacing = 0.0

    OmegaConf.set_struct(cfg, False)

    # Remove observation groups not needed for sampling
    for obs_group_key in list(cfg.observation.keys()):
        if obs_group_key not in [
            "command",
            "policy",
            "priv",
        ] and not obs_group_key.endswith("_"):
            cfg.observation.pop(obs_group_key)

    # Remove randomizations in command manager
    if "pose_range" in cfg.command:
        for key in cfg.command.pose_range:
            cfg.command.pose_range[key] = [0.0, 0.0]
    if "velocity_range" in cfg.command:
        for key in cfg.command.velocity_range:
            cfg.command.velocity_range[key] = [0.0, 0.0]
    if "object_pose_range" in cfg.command:
        for key in cfg.command.object_pose_range:
            cfg.command.object_pose_range[key] = [0.0, 0.0]
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

    # Filter to tracking rewards only
    filtered_rewards = {}
    for group_name, group_params in cfg.reward.items():
        if "tracking" in group_name.lower():
            filtered_rewards[group_name] = group_params
            for key in list(group_params.keys()):
                if "vel" in key.lower():
                    del filtered_rewards[group_name][key]
    cfg.reward = filtered_rewards

    # Create HDMI environment (single-env for initialization)
    hdmi_env = SimpleEnv(cfg)
    hdmi_env.eval()
    hdmi_env.reset()

    # Extract PD control parameters from HDMI's articulation
    robot = hdmi_env.scene.articulations["robot"]
    action_mgr = hdmi_env.action_manager

    joint_stiffness = robot.data.joint_stiffness[0].clone()  # (num_all_joints,)
    joint_damping = robot.data.joint_damping[0].clone()  # (num_all_joints,)
    action_scaling = action_mgr.action_scaling.clone()  # (nu_action,)
    default_joint_pos = action_mgr.default_joint_pos.clone()  # (1, num_all_joints)
    action_joint_ids = list(action_mgr.joint_ids)

    # Get joint address mappings
    joint_qposadr_read = robot.joint_qposadr_read.copy()
    joint_qveladr_read = robot.joint_qveladr_read.copy()
    jnt_mjc2isaac = list(robot._jnt_mjc2isaac)

    # Get simulation parameters
    mj_model = hdmi_env.sim.mj_model
    mj_data = hdmi_env.sim.mj_data
    decimation = hdmi_env.decimation
    physics_dt = hdmi_env.physics_dt

    # Set config fields that process_config normally sets for mjwp
    config.nq = mj_model.nq
    config.nv = mj_model.nv

    # Create Warp batch model/data/graph
    nconmax = int(config.nconmax_per_env)
    njmax = int(config.njmax_per_env)
    num_worlds = int(config.num_samples)
    dev = str(config.device)

    wp.set_device(dev)
    with wp.ScopedDevice(dev):
        model_wp = mjwarp.put_model(mj_model)
        data_wp = mjwarp.put_data(
            mj_model, mj_data,
            nworld=num_worlds,
            nconmax=nconmax,
            njmax=njmax,
        )
        data_wp_prev = mjwarp.put_data(
            mj_model, mj_data,
            nworld=num_worlds,
            nconmax=nconmax,
            njmax=njmax,
        )
        graph = _compile_step(model_wp, data_wp)

    # Move PD parameters to device
    joint_stiffness = joint_stiffness.to(config.device)
    joint_damping = joint_damping.to(config.device)
    action_scaling = action_scaling.to(config.device)
    default_joint_pos = default_joint_pos.to(config.device)

    env = HDMIWarpEnv(
        hdmi_env=hdmi_env,
        mj_model=mj_model,
        mj_data=mj_data,
        model_wp=model_wp,
        data_wp=data_wp,
        data_wp_prev=data_wp_prev,
        graph=graph,
        device=dev,
        num_worlds=num_worlds,
        decimation=decimation,
        physics_dt=physics_dt,
        joint_stiffness=joint_stiffness,
        joint_damping=joint_damping,
        action_scaling=action_scaling,
        default_joint_pos=default_joint_pos,
        action_joint_ids=action_joint_ids,
        joint_qposadr_read=joint_qposadr_read,
        joint_qveladr_read=joint_qveladr_read,
        jnt_mjc2isaac=jnt_mjc2isaac,
        episode_length_buf=hdmi_env.episode_length_buf.clone(),
        timestamp=0,
    )

    return env


def save_state(env: HDMIWarpEnv):
    """Save the Warp simulation state."""
    _copy_state(env.data_wp, env.data_wp_prev)
    state = {
        "data_wp_prev": env.data_wp_prev,
        "episode_length_buf": env.episode_length_buf.clone(),
        "timestamp": env.timestamp,
    }
    # Save command manager state
    if hasattr(env.hdmi_env.command_manager, "t"):
        state["command_t"] = env.hdmi_env.command_manager.t.clone()
    return state


def load_state(env: HDMIWarpEnv, state):
    """Load the Warp simulation state from backup."""
    _copy_state(state["data_wp_prev"], env.data_wp)
    env.episode_length_buf[:] = state["episode_length_buf"]
    env.timestamp = state["timestamp"]
    if "command_t" in state:
        env.hdmi_env.command_manager.t[:] = state["command_t"]
    return env


def step_env(config: Config, env: HDMIWarpEnv, ctrl: torch.Tensor):
    """Step all worlds with provided controls of shape (N, nu_action).

    Converts actions to PD torques and steps via Warp graph.
    """
    if ctrl.dim() == 1:
        ctrl = ctrl.unsqueeze(0).repeat(env.num_worlds, 1)

    # Convert actions to joint position targets
    # action * scaling + default_pos at action joint indices
    joint_pos_target_all = env.default_joint_pos.repeat(env.num_worlds, 1)  # (N, num_joints)
    joint_pos_target_all[:, env.action_joint_ids] += ctrl * env.action_scaling

    with wp.ScopedDevice(env.device):
        for _substep in range(env.decimation):
            # Read current joint state from wp_data
            qpos = wp.to_torch(env.data_wp.qpos)  # (N, nq)
            qvel = wp.to_torch(env.data_wp.qvel)  # (N, nv)

            # Extract ALL joint positions/velocities in isaac order
            current_jpos = qpos[:, env.joint_qposadr_read]  # (N, num_all_joints)
            current_jvel = qvel[:, env.joint_qveladr_read]  # (N, num_all_joints)

            # Compute PD torques for ALL joints
            pos_error = joint_pos_target_all - current_jpos
            vel_error = -current_jvel  # target vel = 0

            torque = env.joint_stiffness * pos_error + env.joint_damping * vel_error

            # Map torques to mujoco ctrl order (isaac → mjc)
            mj_ctrl = torch.zeros(
                env.num_worlds, env.mj_model.nu,
                device=config.device, dtype=torch.float32,
            )
            mj_ctrl[:, env.jnt_mjc2isaac] = torque.float()

            # Write ctrl and step
            wp.copy(env.data_wp.ctrl, wp.from_torch(mj_ctrl))
            wp.capture_launch(env.graph)

    env.episode_length_buf.add_(1)
    env.timestamp += 1
    return env


def _update_viewer(env: HDMIWarpEnv):
    """Update mjlab viewer from wp_data state."""
    hdmi_env = env.hdmi_env
    if not hdmi_env._viewer_enabled or hdmi_env.viewer is None:
        return

    # Copy first world state back to CPU mj_data for viewer
    qpos = wp.to_torch(env.data_wp.qpos)[0].detach().cpu().numpy()
    qvel = wp.to_torch(env.data_wp.qvel)[0].detach().cpu().numpy()
    env.mj_data.qpos[:] = qpos
    env.mj_data.qvel[:] = qvel
    mujoco.mj_forward(env.mj_model, env.mj_data)

    hdmi_env.viewer.user_scn.ngeom = 0
    if getattr(hdmi_env.viewer, "is_running", lambda: True)():
        hdmi_env.viewer.sync(state_only=True)


def _diff_qpos(
    config: Config, qpos_sim: torch.Tensor, qpos_ref: torch.Tensor
) -> torch.Tensor:
    """Compute qpos difference handling quaternions properly for humanoid_object."""
    batch_size = qpos_sim.shape[0]
    qpos_diff = torch.zeros((batch_size, config.nv), device=config.device)

    if config.embodiment_type == "humanoid_object":
        nq_obj = config.nq_obj  # 7 (freejoint) or 6 (contact_guidance)
        qpos_humanoid = qpos_sim[:, :-nq_obj]
        qpos_ref_humanoid = qpos_ref[:, :-nq_obj]
        qpos_object = qpos_sim[:, -nq_obj:]
        qpos_ref_object = qpos_ref[:, -nq_obj:]
        # humanoid position
        qpos_diff[:, :3] = qpos_humanoid[:, :3] - qpos_ref_humanoid[:, :3]
        # humanoid rotation (quaternion)
        qpos_diff[:, 3:6] = quat_sub(
            qpos_humanoid[:, 3:7], qpos_ref_humanoid[:, 3:7]
        )
        # humanoid joints
        qpos_diff[:, 6:-6] = qpos_humanoid[:, 7:] - qpos_ref_humanoid[:, 7:]
        # object
        if nq_obj == 7:
            qpos_diff[:, -6:-3] = qpos_object[:, :3] - qpos_ref_object[:, :3]
            qpos_diff[:, -3:] = quat_sub(
                qpos_object[:, 3:7], qpos_ref_object[:, 3:7]
            )
        else:
            qpos_diff[:, -6:-3] = qpos_object[:, :3] - qpos_ref_object[:, :3]
            qpos_diff[:, -3:] = qpos_object[:, 3:6] - qpos_ref_object[:, 3:6]
    elif config.embodiment_type == "humanoid":
        qpos_diff[:, 6:] = qpos_sim[:, 7:] - qpos_ref[:, 7:]
        qpos_diff[:, :3] = qpos_sim[:, :3] - qpos_ref[:, :3]
        qpos_diff[:, 3:6] = quat_sub(qpos_sim[:, 3:7], qpos_ref[:, 3:7])
    else:
        raise ValueError(f"Unsupported embodiment_type: {config.embodiment_type}")

    return qpos_diff


def _weight_diff_qpos(config: Config) -> torch.Tensor:
    """Per-DOF weights for humanoid_object tracking."""
    w = torch.ones(config.nv, device=config.device)
    if config.embodiment_type == "humanoid_object":
        w[:3] = config.base_pos_rew_scale
        w[3:6] = config.base_rot_rew_scale
        w[6:-6] = config.joint_rew_scale
        w[-6:-3] = config.pos_rew_scale
        w[-3:] = config.rot_rew_scale
    elif config.embodiment_type == "humanoid":
        w[:3] = config.pos_rew_scale
        w[3:6] = config.rot_rew_scale
        w[6:] = config.joint_rew_scale
    return w


def get_reward(
    config: Config,
    env: HDMIWarpEnv,
    ref: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, dict]:
    """Compute tracking reward from Warp batched state.

    Uses qpos/qvel tracking similar to mjwp.py get_reward.
    ref is a tuple: (qpos_ref, qvel_ref, ctrl_ref, ...) but we only
    use first element as placeholder - actual ref comes from get_reference.
    """
    # Read batched state from Warp
    qpos_sim = wp.to_torch(env.data_wp.qpos)  # (N, nq)
    qvel_sim = wp.to_torch(env.data_wp.qvel)  # (N, nv)

    # Get reference from command manager's current timestep
    # The ref_slice from the optimizer is based on the placeholder ref_data
    # We need the actual reference qpos/qvel stored on the env
    qpos_ref = env._current_qpos_ref  # (nq,) set by run_hdmi before optimize
    qvel_ref = env._current_qvel_ref  # (nv,)

    # Broadcast reference to batch
    qpos_ref_batch = qpos_ref.unsqueeze(0).repeat(qpos_sim.shape[0], 1)
    qvel_ref_batch = qvel_ref.unsqueeze(0).repeat(qvel_sim.shape[0], 1)

    # Weighted qpos tracking
    qpos_diff = _diff_qpos(config, qpos_sim, qpos_ref_batch)
    qpos_weight = _weight_diff_qpos(config)
    delta_qpos = qpos_diff * qpos_weight
    qpos_dist = torch.norm(delta_qpos, p=2, dim=1)
    qvel_dist = torch.norm(qvel_sim - qvel_ref_batch, p=2, dim=1)

    qpos_rew = -qpos_dist
    qvel_rew = -config.vel_rew_scale * qvel_dist

    reward = qpos_rew + qvel_rew

    info = {
        "qpos_dist": qpos_dist,
        "qvel_dist": qvel_dist,
        "qpos_rew": qpos_rew,
        "qvel_rew": qvel_rew,
    }
    return reward, info


def get_terminate(
    config: Config, env: HDMIWarpEnv, ref_slice: tuple[torch.Tensor, ...]
) -> torch.Tensor:
    return torch.zeros(env.num_worlds, device=env.device)


def get_terminal_reward(
    config: Config,
    env: HDMIWarpEnv,
    ref_slice: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, dict]:
    """Terminal reward."""
    rew, info = get_reward(config, env, ref_slice)
    return config.terminal_rew_scale * rew, info


def get_trace(config: Config, env: HDMIWarpEnv) -> torch.Tensor:
    """Get trace information for visualization from Warp batched state.

    Returns (N, num_trace_points, 3) shaped tensor with body positions.
    """
    xpos = wp.to_torch(env.data_wp.xpos)  # (N, nbody, 3)

    robot = env.hdmi_env.scene.articulations["robot"]

    # Get hand positions (wrist yaw links)
    hand_ids = robot.find_bodies(".*_wrist_yaw_link")[0]
    # Map isaac body ids to mujoco body addresses
    hand_body_adrs = robot.body_adrs_read[hand_ids]
    hand_pos = xpos[:, hand_body_adrs, :]  # (N, 2, 3)

    # Get foot positions (ankle roll links)
    foot_ids = robot.find_bodies(".*_ankle_roll_link")[0]
    foot_body_adrs = robot.body_adrs_read[foot_ids]
    foot_pos = xpos[:, foot_body_adrs, :]  # (N, 2, 3)

    trace_points = [hand_pos, foot_pos]

    # Get object position if available (look for rigid objects in the scene)
    # The object is typically the second free body in the model
    # For humanoid_object, the object qpos is at the end
    if config.nq_obj > 0:
        qpos = wp.to_torch(env.data_wp.qpos)  # (N, nq)
        obj_pos = qpos[:, -config.nq_obj : -config.nq_obj + 3].unsqueeze(1)  # (N, 1, 3)
        trace_points.append(obj_pos)

    trace = torch.cat(trace_points, dim=1)
    return trace


def save_env_params(config: Config, env: HDMIWarpEnv):
    """Save environment parameters (no domain randomization for HDMI)."""
    return {}


def load_env_params(config: Config, env: HDMIWarpEnv, env_param: dict):
    """Load environment parameters (no-op for HDMI)."""
    return env


def copy_sample_state(
    config: Config,
    env: HDMIWarpEnv,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
):
    """Copy simulation state from source samples to destination samples."""
    src_idx = src_indices.cpu().numpy()
    dst_idx = dst_indices.cpu().numpy()

    # Get state as torch tensors
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

    # Copy back to warp
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


def sync_env(config: Config, env: HDMIWarpEnv):
    """Broadcast state from first world to all worlds, and sync to CPU mj_data."""
    _broadcast_state(env.data_wp, env.num_worlds)

    # Copy first world state back to CPU mj_data (for viewer and run_hdmi.py)
    qpos = wp.to_torch(env.data_wp.qpos)[0].detach().cpu().numpy()
    qvel = wp.to_torch(env.data_wp.qvel)[0].detach().cpu().numpy()
    env.mj_data.qpos[:] = qpos
    env.mj_data.qvel[:] = qvel
    mujoco.mj_forward(env.mj_model, env.mj_data)

    # Update HDMI env's articulation data from mj_data (for command manager)
    env.hdmi_env.scene.update(env.physics_dt)

    return env


def get_reference(
    config: Config, env: HDMIWarpEnv,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Get full reference motion data from HDMI command manager.

    Returns tuple of:
        - qpos_ref: (max_sim_steps, nq)
        - qvel_ref: (max_sim_steps, nv)
        - ctrl_ref: (max_sim_steps, nu_action)
    """
    hdmi_env = env.hdmi_env
    action_manager = hdmi_env.action_manager
    command_manager = hdmi_env.command_manager

    # Get full motion data slice
    motion_data = command_manager.dataset.get_slice(
        command_manager.motion_ids, 0, steps=config.max_sim_steps
    )

    # Get action joint indices in motion data
    action_indices_motion = [
        command_manager.dataset.joint_names.index(joint_name)
        for joint_name in action_manager.joint_names
    ]

    # Get reference joint positions for control
    ref_joint_pos = motion_data.joint_pos[0, :, action_indices_motion]

    # Convert to actions using action manager's normalization
    default_joint_pos = action_manager.default_joint_pos[0, action_manager.joint_ids]
    action_scaling = action_manager.action_scaling
    ctrl_ref = (ref_joint_pos - default_joint_pos) / action_scaling

    # Extract reference qpos and qvel for all bodies
    robot_joint_pos = motion_data.joint_pos[0]
    robot_joint_vel = motion_data.joint_vel[0]

    root_body_idx = command_manager.root_body_idx_motion
    root_pos = motion_data.body_pos_w[0, :, root_body_idx, :]
    root_quat = motion_data.body_quat_w[0, :, root_body_idx, :]
    root_lin_vel = motion_data.body_lin_vel_w[0, :, root_body_idx, :]
    root_ang_vel = motion_data.body_ang_vel_w[0, :, root_body_idx, :]

    qpos_ref = torch.cat([root_pos, root_quat, robot_joint_pos], dim=-1)
    qvel_ref = torch.cat([root_lin_vel, root_ang_vel, robot_joint_vel], dim=-1)

    # Add object states if available
    if hasattr(command_manager, "object_body_id_motion"):
        object_body_idx = command_manager.object_body_id_motion
        object_pos = motion_data.body_pos_w[0, :, object_body_idx, :]
        object_quat = motion_data.body_quat_w[0, :, object_body_idx, :]
        object_lin_vel = motion_data.body_lin_vel_w[0, :, object_body_idx, :]
        object_ang_vel = motion_data.body_ang_vel_w[0, :, object_body_idx, :]

        qpos_ref = torch.cat([qpos_ref, object_pos, object_quat], dim=-1)
        qvel_ref = torch.cat([qvel_ref, object_lin_vel, object_ang_vel], dim=-1)

        if (
            hasattr(command_manager, "object_joint_idx_motion")
            and command_manager.object_joint_idx_motion is not None
        ):
            object_joint_pos = motion_data.joint_pos[
                0, :, command_manager.object_joint_idx_motion
            ].unsqueeze(-1)
            object_joint_vel = motion_data.joint_vel[
                0, :, command_manager.object_joint_idx_motion
            ].unsqueeze(-1)
            qpos_ref = torch.cat([qpos_ref, object_joint_pos], dim=-1)
            qvel_ref = torch.cat([qvel_ref, object_joint_vel], dim=-1)

    # Pad last frames to avoid overflow
    pad_len = config.horizon_steps + config.ctrl_steps
    last_qpos = qpos_ref[-1:].repeat(pad_len, 1)
    qpos_ref = torch.cat([qpos_ref, last_qpos], dim=0)
    last_qvel = qvel_ref[-1:].repeat(pad_len, 1) * 0.0
    qvel_ref = torch.cat([qvel_ref, last_qvel], dim=0)
    last_ctrl = ctrl_ref[-1:].repeat(pad_len, 1)
    ctrl_ref = torch.cat([ctrl_ref, last_ctrl], dim=0)

    # Verify shapes
    assert qpos_ref.shape[-1] == env.mj_model.nq, (
        f"nq_ref: {qpos_ref.shape[-1]}, nq_env: {env.mj_model.nq}"
    )
    assert qvel_ref.shape[-1] == env.mj_model.nv, (
        f"nv_ref: {qvel_ref.shape[-1]}, nv_env: {env.mj_model.nv}"
    )
    assert ctrl_ref.shape[-1] == config.nu, (
        f"nu_ref: {ctrl_ref.shape[-1]}, nu_env: {config.nu}"
    )

    return qpos_ref, qvel_ref, ctrl_ref
