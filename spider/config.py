# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Define the configuration for the optimizer.

Author: Chaoyi Pan
Date: 2025-08-10
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field, fields

import loguru
import mujoco
import numpy as np
import torch
from omegaconf import OmegaConf

import spider
from spider.io import get_processed_data_dir


@dataclass
class Config:
    # === TASK CONFIGURATION ===
    robot_type: str = "xhand"  # "inspire", "allegro", "g1"
    embodiment_type: str = "bimanual"  # "left", "right", "bimanual", "CMU"
    task: str = "pick_spoon_bowl"
    seed: int = 0

    # === DATASET CONFIGURATION ===
    dataset_dir: str = f"{spider.ROOT}/../example_datasets"
    dataset_name: str = "oakink"
    data_id: int = 0
    model_path: str = ""
    data_path: str = ""
    scene_name: str = ""  # override scene XML basename (e.g. "scene_mocap_partner" → scene_mocap_partner.xml)
    # Optional config loader (used by CLI runner)
    load_config_path: str = ""

    # === SIMULATOR CONFIGURATION ===
    simulator: str = "mjwp"  # "isaac" | "mujoco" | "mjwp" | "mjwp_cons" | "mjwp_eq" | "mjwp_cons_eq" | "kinematic"
    device: str = "cuda:0"
    # Simulation timing
    sim_dt: float = 0.01  # simulation timestep
    physics_dt: float = -1.0  # if > 0, run physics at this dt with decimation = sim_dt/physics_dt (Path Y, Holosoma alignment)
    sim_decimation: int = 1  # auto-set from physics_dt; do not set manually
    ctrl_dt: float = 0.4  # control timestep
    ref_dt: float = 0.02  # reference data timestep
    render_dt: float = 0.02  # rendering timestep
    horizon: float = 1.6  # planning horizon
    knot_dt: float = 0.4  # knot point spacing
    max_sim_steps: int = -1  # maximum simulation steps (-1 for unlimited)
    # Simulation constraints
    nconmax_per_env: int = 100  # max contacts per environment
    njmax_per_env: int = 350  # max joints per environment
    # Simulation annealing
    num_dyn: int = (
        1  # number of environments for annealing, used for virtual contact constraint
    )
    # Domain randomization
    num_dr: int = (
        1  # number of domain randomization groups, used for domain randomization
    )
    pair_margin_range: tuple[float, float] = (-0.005, 0.005)
    xy_offset_range: tuple[float, float] = (-0.005, 0.005)
    perturb_force: float = 0.0
    perturb_torque: float = 0.0
    # E024: partner force — external upward force on object simulating human partner support
    partner_force_scale: float = (
        0.0  # fraction of object gravity to apply as upward force (0.5 = 50%)
    )
    partner_force_spring_kp: float = (
        0.0  # spring stiffness pulling object toward ref pos (0 = pure gravity comp)
    )
    partner_force_spring_kd: float = (
        -1.0
    )  # damping (-1 = auto critical damping = 2*sqrt(m*kp))
    # E030: orientation spring for freejoint object
    partner_force_spring_kp_rot: float = 0.0  # rotation spring stiffness (0 = disabled)
    partner_force_spring_kd_rot: float = (
        -1.0
    )  # rotation damping (-1 = auto critical damping)
    partner_force_rot_clamp: float = (
        0.5  # max axis-angle magnitude (rad) to prevent large torques
    )
    partner_force_ref_dt: float = (
        -1.0  # reference dt for partner force indexing; <=0 uses config.ref_dt
    )
    partner_force_point_local: list[float] = field(
        default_factory=list
    )  # optional object-local support point; empty means apply force at COM
    partner_force_points_local: list[list[float]] = field(
        default_factory=list
    )  # optional object-local support points; overrides partner_force_point_local when non-empty
    partner_force_force_clamp: float = (
        0.0  # max external force norm in N; <=0 disables clamp
    )
    partner_force_torque_clamp: float = (
        0.0  # max torque norm in Nm for support-point wrench; <=0 disables clamp
    )
    # E006: COLA-style virtual support-body proxy. This keeps the object true
    # freejoint and applies a connector wrench from an independent proxy target.
    support_proxy_enabled: bool = False
    support_proxy_mode: str = (
        "wrench"  # "wrench" | "mocap_pad" | "wrench_pad"
    )
    support_proxy_mocap_body_name: str = "support_proxy_pad"
    support_proxy_mocap_quat_mode: str = (
        "identity"  # "identity" | "object_ref"; used by E014 weld anchor
    )
    support_proxy_point_local: list[float] = field(
        default_factory=list
    )  # object-local support site; required when enabled
    support_proxy_gravity_scale: float = (
        0.5  # fraction of object weight supplied by support proxy
    )
    support_proxy_connector_kp: float = 0.0
    support_proxy_connector_kd: float = (
        -1.0  # -1 = critical damping with object mass and connector_kp
    )
    support_proxy_xy_velocity_scale: float = (
        1.0  # scale reference support-site XY velocity before integration
    )
    support_proxy_max_xy_speed: float = (
        0.0  # m/s clip for proxy horizontal command; <=0 disables clip
    )
    support_proxy_height_tau: float = (
        0.0  # low-pass time constant for proxy height; <=0 tracks ref height
    )
    support_proxy_ref_dt: float = (
        -1.0
    )  # <=0 uses config.sim_dt because qpos_ref is already interpolated
    support_proxy_force_clamp: float = 0.0
    support_proxy_torque_clamp: float = 0.0
    # E015: dynamic support body driven by generalized PD forces while still
    # coupled to the object via a soft equality/weld constraint.
    support_dynamic_body_name: str = "support_dynamic_anchor"
    support_dynamic_mass: float = 2.0
    support_dynamic_pos_kp: float = 0.0
    support_dynamic_pos_kd: float = -1.0
    support_dynamic_rot_kp: float = 0.0
    support_dynamic_rot_kd: float = -1.0
    support_dynamic_force_clamp: float = 0.0
    support_dynamic_torque_clamp: float = 0.0
    # E027b: object PD override — object actuators track ref directly, CEM only optimizes robot
    object_pd_override: bool = False  # enable object actuator PD override in step_env
    object_pd_kp_pos: float = 2000.0  # position actuator gain (strong tracking)
    object_pd_kp_rot: float = 2000.0  # rotation actuator gain
    # E013: true-freejoint object oracle. Unlike object_pd_override, this does
    # not require scene_act object actuators; it writes the object freejoint
    # state from the interpolated reference before and after each physics step.
    object_kinematic_override: bool = False
    object_kinematic_ref_dt: float = -1.0  # <=0 uses sim_dt
    object_kinematic_set_qvel: bool = True
    # E025: hand approach reward — guides hands toward object surface
    hand_approach_rew_scale: float = (
        0.0  # weight of hand-to-object-surface distance reward
    )
    hand_approach_sigma: float = 5.0  # steepness of exponential decay
    hand_approach_body_names: list[str] = field(
        default_factory=lambda: ["left_wrist_yaw_link", "right_wrist_yaw_link"]
    )
    hand_approach_body_ids: list[int] = field(
        default_factory=list
    )  # resolved at runtime
    hand_approach_obj_half_extents: list[float] = field(
        default_factory=list
    )  # resolved at runtime from geom
    hand_approach_contact_threshold: float = (
        0.3  # ref hand-obj dist below this activates hand_approach (m)
    )
    # E037: contact mask-gated reward (HDMI-style)
    contact_mask_rew_scale: float = 0.0  # gain; 0.0 = disabled
    contact_mask_rew_sigma: float = 0.3  # exp kernel bandwidth (meters), HDMI default
    contact_mask_rew_baseline: float = (
        0.0  # reward on non-contact frames (HDMI uses 1.0)
    )
    # E039: HDMI-aligned contact with predefined target points
    contact_hdmi_gain: float = 0.0  # 0=disabled; HDMI uses 5.0
    contact_hdmi_sigma: float = 0.3  # exp kernel bandwidth (same as HDMI eef_pos_sigma)
    contact_hdmi_target_left: list[float] = field(
        default_factory=list
    )  # [x,y,z] in obj local frame
    contact_hdmi_target_right: list[float] = field(
        default_factory=list
    )  # [x,y,z] in obj local frame
    contact_hdmi_eef_offset: list[float] = field(
        default_factory=lambda: [0.05, 0.0, 0.0]
    )  # wrist→palm
    contact_hdmi_threshold: float = (
        0.30  # mask threshold: activate when hand-target < this (m)
    )
    contact_hdmi_mask_source: str = (
        "rotated_sdf"  # "rotated_sdf" | "core4d_3cm"
    )
    contact_hdmi_mask_path: str = ""
    contact_hdmi_mask_person_idx: int = 0
    contact_hdmi_mask_time_axis: str = "auto"  # "auto" | "spider" | "eval"
    contact_hdmi_mask_carry_union: bool = False  # E155: carry task L/R union
    contact_hdmi_mask_ramp_frames: int = 0  # E155: linear ramp frames at boundary, 0=off
    # E040: dynamic per-frame contact target (from ref FK)
    contact_hdmi_dynamic_target: bool = (
        False  # True=use per-frame ref-derived target instead of fixed
    )
    contact_hdmi_target_source: str = (
        "ref_fk"  # "ref_fk" | "external"; external loads object-local targets from npz
    )
    contact_hdmi_target_path: str = ""
    contact_hdmi_target_time_axis: str = "auto"  # "auto" | "spider" | "eval" | "raw"
    contact_hdmi_target_uses_eef_offset: bool = (
        False  # True=derive dynamic target from ref wrist+eef_offset, not wrist origin
    )
    # E041: orientation reward — palm must face object surface
    contact_hdmi_ori_weight: float = 0.0  # 0=disabled; >0 = enable orientation term
    contact_hdmi_ori_mode: str = "multiply"  # "multiply" | "additive" | "near_field"
    contact_hdmi_palm_normal_left: list[float] = field(
        default_factory=lambda: [0.0, -1.0, 0.0]
    )
    contact_hdmi_palm_normal_right: list[float] = field(
        default_factory=lambda: [0.0, 1.0, 0.0]
    )
    # E074A: trust-region guard against robot actuator ctrl drifting too far
    # from the reference controls during the post-contact hold window.
    ctrl_ref_guard_scale: float = 0.0
    ctrl_ref_guard_robot_only: bool = True
    ctrl_ref_guard_sigma: float = 0.25
    ctrl_ref_guard_start_eval_time: float = 1.8
    ctrl_ref_guard_end_eval_time: float = 3.0
    # E074C: proximity surrogate for maintaining hand-object contact in frames
    # where the reference still indicates a hold/contact phase.
    hold_contact_rew_scale: float = 0.0
    hold_contact_sigma: float = 0.05
    hold_contact_start_eval_time: float = 1.8
    hold_contact_end_eval_time: float = 3.0
    hold_contact_require_ref_contact: bool = True
    use_bounded_qpos_reward: bool = (
        False  # use exp(-dist/σ) instead of -dist for qpos reward
    )
    qpos_reward_sigma: float = 2.0  # sigma for bounded qpos reward
    qpos_reward_scale: float = 5.0  # scale for bounded qpos reward
    stability_penalty_scale: float = 0.0  # penalty when pelvis z < threshold
    stability_penalty_threshold: float = 0.55  # pelvis z threshold (m)
    # E025: training-time robot/object penetration penalties. These are
    # disabled by default; eval-only penetration metrics remain unchanged.
    robot_object_penalty_scale: float = 0.0
    robot_object_penalty_margin_m: float = 0.0
    robot_object_penalty_deep_threshold_m: float = 0.02
    robot_object_penalty_geom_names: list[str] = field(
        default_factory=lambda: ["lh", "rh"]
    )
    robot_object_penalty_geom_ids: list[int] = field(default_factory=list)
    leg_object_penalty_scale: float = 0.0
    leg_object_penalty_margin_m: float = 0.02
    leg_object_penalty_geom_names: list[str] = field(default_factory=list)
    leg_object_penalty_geom_ids: list[int] = field(default_factory=list)
    # E117: gate lower-body/object penalty so it can be enabled after the
    # hand-object contact objective is active/satisfied instead of competing
    # with contact acquisition throughout the whole rollout.
    # Values: "always" | "contact_mask" | "time_window" |
    # "contact_mask_time_window" | "hand_target" |
    # "contact_mask_and_hand_target".
    leg_object_penalty_gate_source: str = "always"
    leg_object_penalty_start_eval_time: float = 0.0
    leg_object_penalty_end_eval_time: float = 999.0
    leg_object_penalty_hand_target_threshold_m: float = 0.06
    # E084: direct hand-floor and object support/lift shaping for unstable
    # box-lift cases. Disabled by default.
    hand_floor_penalty_scale: float = 0.0
    hand_floor_penalty_margin_m: float = 0.03
    hand_floor_penalty_geom_names: list[str] = field(
        default_factory=lambda: ["lh", "rh"]
    )
    hand_floor_penalty_geom_ids: list[int] = field(default_factory=list)
    hand_object_deep_penalty_scale: float = 0.0
    hand_object_deep_penalty_threshold_m: float = 0.01
    hand_object_deep_penalty_geom_names: list[str] = field(
        default_factory=lambda: ["lh", "rh"]
    )
    hand_object_deep_penalty_geom_ids: list[int] = field(default_factory=list)
    object_lift_rew_scale: float = 0.0
    object_lift_sigma: float = 0.05
    object_floor_penalty_scale: float = 0.0
    object_floor_margin_m: float = 0.02
    # E088: CEM-level hard safety gate. Disabled by default; when enabled,
    # samples that penetrate the object with upper-body geoms are excluded from
    # elite selection before the control distribution is updated.
    cem_safety_gate_enabled: bool = False
    cem_safety_gate_mode: str = "elite_filter"
    cem_safety_gate_geom_names: list[str] = field(
        default_factory=lambda: [
            "head_collision",
            "torso_collision",
            "pelvis_collision",
            "left_shoulder_yaw_collision",
            "right_shoulder_yaw_collision",
            "left_elbow_yaw_collision",
            "right_elbow_yaw_collision",
        ]
    )
    cem_safety_gate_geom_ids: list[int] = field(default_factory=list)
    cem_safety_gate_min_sdf_m: float = -0.005
    cem_safety_gate_max_violation_pct: float = 0.0
    # E153: absolute single-frame hard floor, decoupled from min_sdf_m. NaN (default)
    # => floor = min_sdf_m (legacy: any frame below min_sdf_m kills the sample, so
    # max_violation_pct is inert). Set deeper than min_sdf_m to activate
    # max_violation_pct (allow a few frames in [hard_floor, min_sdf_m), reject only
    # frames below hard_floor). Default NaN keeps E088-E152 numerically unchanged.
    cem_safety_gate_hard_floor_m: float = float("nan")
    cem_safety_gate_min_valid_frac: float = 0.02
    cem_safety_gate_fallback: str = "least_violation"
    # E152: independent hand/object hard gate. Hands need a looser threshold
    # than body safety geoms because light hand-object contact is intentional.
    cem_hand_gate_enabled: bool = False
    cem_hand_gate_geom_names: list[str] = field(default_factory=list)
    cem_hand_gate_geom_ids: list[int] = field(default_factory=list)
    cem_hand_gate_min_sdf_m: float = -0.005
    cem_hand_gate_max_violation_pct: float = 0.05
    # E153: see cem_safety_gate_hard_floor_m. NaN (default) => floor = min_sdf_m
    # (legacy, max_violation_pct inert). Set deeper (e.g. -0.020) to activate.
    cem_hand_gate_hard_floor_m: float = float("nan")
    # E088: absolute object bottom clearance shaping. This uses world-frame
    # object_collision bottom height instead of relative-to-reference bottom.
    object_clearance_rew_scale: float = 0.0
    object_clearance_penalty_scale: float = 0.0
    object_clearance_floor_z: float = 0.0
    object_clearance_min_m: float = 0.04
    object_clearance_max_m: float = 0.18
    object_clearance_sigma: float = 0.04
    object_clearance_above_weight: float = 0.25
    object_clearance_gate_source: str = "contact_mask"
    object_clearance_start_eval_time: float = 0.0
    object_clearance_end_eval_time: float = 999.0
    # E118: soft carry-state corridor. This rewards coherent hand contact,
    # object clearance/orientation, pelvis height, and lower-body clearance as
    # one coupled state instead of independent scalar terms.
    carry_corridor_rew_scale: float = 0.0
    carry_corridor_gate_source: str = "contact_mask"  # contact_mask | time_window | contact_mask_time_window
    carry_corridor_start_eval_time: float = 0.0
    carry_corridor_end_eval_time: float = 999.0
    carry_corridor_hand_target_threshold_m: float = 0.02
    carry_corridor_hand_sigma: float = 0.04
    carry_corridor_clearance_min_m: float = 0.04
    carry_corridor_clearance_max_m: float = 0.20
    carry_corridor_clearance_sigma: float = 0.05
    carry_corridor_pelvis_min_m: float = 0.60
    carry_corridor_pelvis_sigma: float = 0.06
    carry_corridor_rot_sigma: float = 0.40
    carry_corridor_leg_margin_m: float = 0.02
    carry_corridor_leg_sigma: float = 0.04
    carry_corridor_leg_geom_names: list[str] = field(default_factory=list)
    carry_corridor_leg_geom_ids: list[int] = field(default_factory=list)
    # E120: object support decomposition. Reward true hand-supported
    # near-zero contact while penalizing non-hand body/object support shortcuts.
    hand_support_rew_scale: float = 0.0
    hand_support_sigma: float = 0.015
    hand_support_margin_m: float = 0.01
    hand_support_gate_source: str = "contact_mask"  # always | contact_mask | time_window | contact_mask_time_window
    hand_support_start_eval_time: float = 0.0
    hand_support_end_eval_time: float = 999.0
    hand_support_geom_names: list[str] = field(
        default_factory=lambda: ["lh", "rh"]
    )
    hand_support_geom_ids: list[int] = field(default_factory=list)
    hand_support_decay_frac: float = 0.0  # E155-C: tail decay fraction, 0=off
    hand_support_neutral_baseline: float = 0.0  # E155-D: gate=0 neutral value, 0=current
    nonhand_support_penalty_scale: float = 0.0
    nonhand_support_penalty_margin_m: float = 0.02
    nonhand_support_penalty_gate_source: str = "contact_mask"
    nonhand_support_penalty_start_eval_time: float = 0.0
    nonhand_support_penalty_end_eval_time: float = 999.0
    nonhand_support_penalty_geom_names: list[str] = field(default_factory=list)
    nonhand_support_penalty_geom_ids: list[int] = field(default_factory=list)
    # E121: terminal carry-state semantic gate. When hard/hard_soft, terminal
    # violation is folded into the existing CEM elite safety gate.
    terminal_carry_gate_enabled: bool = False
    terminal_carry_gate_mode: str = "soft"  # soft | hard | hard_soft
    terminal_carry_gate_soft_scale: float = 5.0
    terminal_carry_gate_pelvis_min_m: float = 0.60
    terminal_carry_gate_obj_rot_max_rad: float = 0.55
    terminal_carry_gate_nonhand_margin_m: float = 0.02
    terminal_carry_gate_hand_near_margin_m: float = 0.02
    terminal_carry_gate_hand_min_near_frac: float = 0.5
    # E035: local-frame body tracking (HDMI-style)
    use_local_frame_reward: bool = False
    local_frame_upper_ids: list[int] = field(
        default_factory=lambda: list(range(14, 31))
    )  # waist→wrists
    local_frame_lower_ids: list[int] = field(
        default_factory=lambda: list(range(2, 14))
    )  # hips→ankles
    local_frame_pos_sigma: float = 0.5
    local_frame_ori_sigma: float = 1.0
    local_frame_root_sigma: float = 0.5
    local_frame_joint_sigma: float = 0.25
    local_frame_w_track: float = 0.5
    # E044: extra weight for wrist bodies in local-frame tracking
    local_frame_wrist_ids: list[int] = field(
        default_factory=lambda: [23, 30]
    )  # left/right wrist_yaw_link
    local_frame_wrist_weight: float = 1.0  # 1.0 = no extra weight
    contact_guidance: bool = False
    euler_convention: str = "XYZ"  # Intrinsic euler convention for object hinge joints
    use_scene_act: str = ""  # Path to scene_act.xml (bypass _make_contact_guidance_model)
    object_pos_actuator_names: list[str] = field(
        default_factory=lambda: [
            "right_object_pos_x",
            "right_object_pos_y",
            "right_object_pos_z",
            "left_object_pos_x",
            "left_object_pos_y",
            "left_object_pos_z",
        ]
    )
    object_rot_actuator_names: list[str] = field(
        default_factory=lambda: [
            "right_object_rot_x",
            "right_object_rot_y",
            "right_object_rot_z",
            "left_object_rot_x",
            "left_object_rot_y",
            "left_object_rot_z",
        ]
    )
    object_action_dims: int = 0
    object_actuator_ids: list[int] = field(default_factory=list)
    object_actuator_names: list[str] = field(default_factory=list)
    init_pos_actuator_gain: float = 10.0
    init_pos_actuator_bias: float = 10.0
    init_rot_actuator_gain: float = 0.1
    init_rot_actuator_bias: float = 0.1
    guidance_decay_ratio: float = 0.5
    residual_gain_ratio: float = (
        0.0  # if > 0, last CEM iteration keeps this fraction of decayed gains
    )
    # Mocap partner trajectory (E011)
    mocap_partner_trajectory: str = (
        ""  # path to NPZ with partner_pos (T,2,3) and partner_quat (T,2,4)
    )
    mocap_partner_intra_step: bool = (
        True  # update partner mocap within rollout steps (not just MPC steps)
    )
    gibbs_sampling: bool = False

    # === SBTO (Sampling-Based Trajectory Optimization, DynaRetarget) ===
    use_sbto: bool = (
        False  # if True, use incremental full-horizon optimization instead of MPC
    )
    sbto_sigma_min: float = 0.05  # convergence threshold: max(noise_scale) < this
    sbto_max_iter_per_knot: int = 50  # max optimization iterations per knot increment
    sbto_knot_dt: float = 0.25  # knot spacing for SBTO (independent of MPC knot_dt)
    # DynaRetarget paper alignment (Table I)
    sbto_elite_fraction: float = 0.03  # ρ_e: fraction of samples used for update
    sbto_mean_momentum: float = 0.95  # α_μ: EWMA momentum on mean (0=no momentum)
    sbto_cov_momentum: float = 0.2  # α_Σ: EWMA momentum on covariance

    # === Path Y: Holosoma physics alignment ===
    apply_holosoma_pd: bool = (
        False  # override actuator gains using Holosoma G1 PD config (Isaac order)
    )
    apply_wrist_dof_damping: bool = (
        False  # add dof_damping to wrist joints (HDMI R013 fix for underdamped wrists)
    )
    wrist_dof_damping: float = 5.0  # critically damped for Kp≈15, inertia≈0.43
    use_local_contact_reward: bool = (
        False  # add HDMI-style local-frame contact offset reward
    )
    local_contact_sigma: float = 0.3
    local_contact_rew_scale: float = 5.0

    # === OPTIMIZER CONFIGURATION ===
    # Sampling parameters
    num_samples: int = 2048
    temperature: float = 0.3
    max_num_iterations: int = 16
    improvement_threshold: float = 0.01
    improvement_check_steps: int = 1
    warmup_steps: int = (
        0  # skip CEM optimization for first N ctrl steps (use ref ctrl directly)
    )
    # Termination parameters
    terminate_resample: bool = False
    object_pos_threshold: float = 0.1
    object_rot_threshold: float = 0.3
    max_revert_forward_attempts: int = 3
    max_revert_depth: int = 3
    base_pos_threshold: float = 0.5
    base_rot_threshold: float = 0.4
    # Compilation
    use_torch_compile: bool = True  # use torch.compile for acceleration
    # Noise scheduling
    first_ctrl_noise_scale: float = 0.5
    last_ctrl_noise_scale: float = 1.0
    final_noise_scale: float = 0.1
    exploit_ratio: float = 0.01
    exploit_noise_scale: float = 0.01
    # Noise scaling by component
    joint_noise_scale: float = 0.15
    pos_noise_scale: float = 0.03
    rot_noise_scale: float = 0.03
    # E042: zero noise for specific joints (like HDMI wrist freeze)
    zero_noise_joint_keywords: list[str] = field(
        default_factory=list
    )  # e.g. ["wrist_roll", "wrist_pitch", "wrist_yaw"]
    # Reward mode
    use_rl_reward: bool = False  # use dexmachina RL training reward formulation
    # Reward scaling
    base_pos_rew_scale: float = 1.0
    base_rot_rew_scale: float = 0.3
    joint_rew_scale: float = 0.003
    pos_rew_scale: float = 1.0
    rot_rew_scale: float = 0.3
    vel_rew_scale: float = 0.0001
    terminal_rew_scale: float = 1.0
    contact_rew_scale: float = 0.0
    num_resamples: int = 0
    resample_ratio: float = 0.2

    # === TASK-SPACE REWARDS (E018, DynaRetarget/Harmanoid inspired) ===
    # Body world-position tracking (uses data_wp.xpos), DynaRetarget Table II
    task_body_rew_scale: float = 0.0
    task_body_names: list[str] = field(default_factory=list)
    task_body_weights: list[float] = field(default_factory=list)
    # Resolved at runtime from task_body_names
    task_body_ids: list[int] = field(default_factory=list)
    # Separate object pos/rot tracking weights (DynaRetarget uses obj_pos=40)
    task_obj_pos_rew_scale: float = 0.0
    task_obj_rot_rew_scale: float = 0.0
    # E065: switch task_obj_rew form from unbounded L2 (default, original) to
    # HDMI-style saturating exp(-err/sigma). When True, reward becomes
    #   scale * exp(-||obj_err||/pos_sigma)  ∈ [0, scale]
    # instead of  -scale * sum(err²)  (unbounded). Diagnosis log 82 §10.2.
    task_obj_use_exp: bool = False
    task_obj_pos_sigma: float = 0.5  # HDMI default
    task_obj_rot_sigma: float = 0.5
    # Interaction reward (Harmanoid Eq.15): match relative offsets between two robots' bodies
    interact_rew_scale: float = 0.0
    interact_sigma: float = 1.0
    # List of [body_idx_in_task_body_ids_a, body_idx_in_task_body_ids_b] pairs
    interact_pairs: list[list[int]] = field(default_factory=list)

    # === VISUALIZATION CONFIGURATION ===
    show_viewer: bool = True
    viewer: str = "mujoco"  # "mujoco" | "rerun" | "viser" | "isaac"
    wait_on_finish: bool = True  # block after optimization to keep viewer alive
    rerun_spawn: bool = False
    save_video: bool = True
    video_output_path: str = ""  # custom video output path; empty = default (output_dir/visualization_mjwp_act.mp4)
    video_camera: str = "front"  # named MuJoCo camera; if missing or "auto", use full-body free camera
    video_auto_camera_min_distance: float = 3.8
    video_auto_camera_distance_scale: float = 3.0
    video_auto_camera_azimuth: float = 135.0
    video_auto_camera_elevation: float = -22.0
    warmstart_qpos_path: str = ""  # path to .npz containing qpos_snap + snap_mask; if set, replaces qpos_ref slices in intent window (E058+ Path B-CEM)
    warmstart_update_ctrl_from_qpos: bool = False
    save_info: bool = True
    save_rerun: bool = False
    save_metrics: bool = True
    save_config: bool = True

    # === TRACE RECORDING ===
    trace_dt: float = 1 / 50.0
    num_trace_uniform_samples: int = 4
    num_trace_topk_samples: int = 2
    trace_site_ids: list = field(default_factory=list)

    # === CONTACT GUIDANCE (DERIVED) ===
    contact_order: list = field(default_factory=list)
    hand_contact_site_ids: list = field(default_factory=list)
    right_contact_indices: list = field(default_factory=list)
    left_contact_indices: list = field(default_factory=list)
    right_pos_ctrl_ids: list = field(default_factory=list)
    left_pos_ctrl_ids: list = field(default_factory=list)
    contact_len: int = 0

    # === AUTOMATICALLY SET PROPERTIES ===
    # Computed timesteps
    horizon_steps: int = -1
    knot_steps: int = -1
    ref_steps: int = -1
    ctrl_steps: int = -1
    # Model dimensions
    nq_obj: int = -1  # object DOF
    nq: int = -1  # total position DOF
    nv: int = -1  # total velocity DOF
    nu: int = -1  # total control DOF
    npair: int = -1  # total pair DOF
    # Computed tensors
    noise_scale: torch.Tensor = field(default_factory=lambda: torch.ones(1))
    beta_traj: float = -1.0
    # Runtime state
    env_params_list: list = field(default_factory=list)
    viewer_body_entity_and_ids: list = field(default_factory=list)
    output_dir: str = ""


def resolve_object_actuator_ids(
    model: mujoco.MjModel,
    desired_names: list[str],
    object_action_dims: int,
) -> tuple[list[int], list[str]]:
    """Resolve object actuator ids by name, with a fallback to last N actuators."""
    resolved_ids: list[int] = []
    resolved_names: list[str] = []
    for name in desired_names:
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if aid != -1:
            resolved_ids.append(int(aid))
            resolved_names.append(name)
    if resolved_ids:
        if len(resolved_ids) != len(desired_names):
            loguru.logger.info(
                "Resolved {} / {} object actuators by name.",
                len(resolved_ids),
                len(desired_names),
            )
        return resolved_ids, resolved_names

    obj_start = max(model.nu - max(object_action_dims, 0), 0)
    fallback_ids = list(range(obj_start, model.nu))
    fallback_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, int(aid))
        for aid in fallback_ids
    ]
    loguru.logger.warning(
        "No named object actuators found; falling back to last {} actuators.",
        len(fallback_ids),
    )
    return fallback_ids, fallback_names


def load_config_yaml(path: str) -> dict:
    """Load a config YAML into a plain dict, resolving OmegaConf types."""
    if not path:
        return {}
    abs_path = os.path.abspath(os.path.expanduser(path))
    if not os.path.isfile(abs_path):
        raise FileNotFoundError(f"Config file not found: {abs_path}")
    cfg = OmegaConf.load(abs_path)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(cfg_dict, dict):
        raise ValueError(f"Config at {abs_path} did not resolve to a mapping.")
    cfg_dict.pop("hydra", None)
    return cfg_dict


def filter_config_fields(config_dict: dict) -> dict:
    """Filter config dict keys to those defined in Config."""
    allowed = {field.name for field in fields(Config)}
    return {key: value for key, value in config_dict.items() if key in allowed}


def build_hand_contact_site_ids(
    mj_model: mujoco.MjModel, embodiment_type: str
) -> tuple[list[tuple[str, str]], list[int | None]]:
    contact_order = []
    if embodiment_type in ["bimanual", "right"]:
        contact_order.extend(
            [
                ("right", "thumb"),
                ("right", "index"),
                ("right", "middle"),
                ("right", "ring"),
                ("right", "pinky"),
            ]
        )
    if embodiment_type in ["bimanual", "left"]:
        contact_order.extend(
            [
                ("left", "thumb"),
                ("left", "index"),
                ("left", "middle"),
                ("left", "ring"),
                ("left", "pinky"),
            ]
        )
    if embodiment_type == "humanoid_object":
        # Humanoid hands use wrist tracking sites (no finger tips)
        # Use "hand" as finger placeholder — matches "track_hand_right" / "track_hand_left"
        contact_order.extend(
            [
                ("right", "hand"),
                ("left", "hand"),
            ]
        )

    site_ids: list[int | None] = [None] * len(contact_order)
    for sid in range(mj_model.nsite):
        name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_SITE, sid)
        if name is None:
            continue
        name_l = name.lower()
        if "track" not in name_l or "hand" not in name_l:
            continue
        for idx, (side, finger) in enumerate(contact_order):
            if side in name_l and finger in name_l:
                if site_ids[idx] is None:
                    site_ids[idx] = sid
                break

    missing = [i for i, sid in enumerate(site_ids) if sid is None]
    if missing:
        loguru.logger.warning(
            "Missing {} hand contact sites for guidance; indices: {}",
            len(missing),
            missing,
        )
    return contact_order, site_ids


def get_object_pos_ctrl_indices(config: Config) -> tuple[list[int], list[int]]:
    right_ids: list[int] = []
    left_ids: list[int] = []
    if config.object_actuator_ids and config.object_actuator_names:
        for aid, name in zip(
            config.object_actuator_ids, config.object_actuator_names, strict=False
        ):
            name_l = (name or "").lower()
            if "_pos_" not in name_l:
                continue
            if "right" in name_l:
                right_ids.append(int(aid))
            elif "left" in name_l:
                left_ids.append(int(aid))
            elif config.embodiment_type == "humanoid_object":
                # Single object: assign pos actuators to both right and left
                right_ids.append(int(aid))
                left_ids.append(int(aid))

    if right_ids or left_ids:
        return right_ids, left_ids

    obj_dims = int(config.object_action_dims) if config.object_action_dims > 0 else 0
    if obj_dims == 0:
        obj_dims = 12 if config.embodiment_type == "bimanual" else 6
    start = max(int(config.nu) - obj_dims, 0)
    if config.embodiment_type == "bimanual" and obj_dims >= 12:
        right_ids = list(range(start, start + 3))
        left_ids = list(range(start + 3, start + 6))
    elif config.embodiment_type == "humanoid_object":
        # Single object: same 3 pos actuators for both hands
        right_ids = list(range(start, start + 3))
        left_ids = list(range(start, start + 3))
    else:
        if config.embodiment_type == "right":
            right_ids = list(range(start, start + 3))
        elif config.embodiment_type == "left":
            left_ids = list(range(start, start + 3))
        else:
            right_ids = list(range(start, start + 3))
    return right_ids, left_ids


def get_noise_scale(config: Config) -> torch.Tensor:
    """Get the noise scale for sampling.

    Args:
        config: Config

    Returns:
        Noise scale, shape (num_samples, knot_steps, nu)
    """
    noise_scale = torch.logspace(
        start=torch.log10(torch.tensor(config.first_ctrl_noise_scale)),
        end=torch.log10(torch.tensor(config.last_ctrl_noise_scale)),
        steps=int(round(config.horizon / config.knot_dt)),
        device=config.device,
        base=10,
    )[None, :, None]  # Shape: (1, num_knot_steps, 1)
    noise_scale = noise_scale.repeat(1, 1, config.nu)
    if config.embodiment_type in ["bimanual", "right", "left"]:
        object_action_dims = max(int(config.object_action_dims), 0)
        robot_nu = max(int(config.nu - object_action_dims), 0)
        noise_scale[:, :, :3] *= config.pos_noise_scale
        noise_scale[:, :, 3:6] *= config.rot_noise_scale
        if config.embodiment_type == "bimanual":
            half_dof = robot_nu // 2
            noise_scale[:, :, 6:half_dof] *= config.joint_noise_scale
            noise_scale[:, :, half_dof : half_dof + 3] *= config.pos_noise_scale
            noise_scale[:, :, half_dof + 3 : half_dof + 6] *= config.rot_noise_scale
            noise_scale[:, :, half_dof + 6 : robot_nu] *= config.joint_noise_scale
        elif config.embodiment_type in ["right", "left"]:
            noise_scale[:, :, 6:robot_nu] *= config.joint_noise_scale
    else:
        noise_scale *= config.joint_noise_scale
    if config.contact_guidance and config.object_actuator_ids:
        object_ids = torch.as_tensor(
            config.object_actuator_ids, device=config.device, dtype=torch.long
        )
        noise_scale[:, :, object_ids] *= 0.0
    # E042: zero noise for keyword-matched joints (e.g. wrist freeze)
    if config.zero_noise_joint_keywords and hasattr(config, "_model_cpu_for_noise"):
        import mujoco as _mj

        _model = config._model_cpu_for_noise
        for ai in range(_model.nu):
            aname = _mj.mj_id2name(_model, _mj.mjtObj.mjOBJ_ACTUATOR, ai)
            if aname and any(kw in aname for kw in config.zero_noise_joint_keywords):
                noise_scale[:, :, ai] *= 0.0
    # repeat to match num_samples; same samples used across DR groups
    noise_scale = noise_scale.repeat(config.num_samples, 1, 1)
    # set first sample to 0
    noise_scale[0] *= 0.0
    # set last few samples to exploit_noise_scale
    num_exploit_samples = int(config.num_samples * config.exploit_ratio)
    noise_scale[-num_exploit_samples:] *= config.exploit_noise_scale
    return noise_scale


def compute_steps(config: Config):
    # make sure every dt can be divided by sim_dt
    config.horizon_steps = int(np.round(config.horizon / config.sim_dt))
    config.knot_steps = int(np.round(config.knot_dt / config.sim_dt))
    config.ref_steps = int(np.round(config.ref_dt / config.sim_dt))
    config.ctrl_steps = int(np.round(config.ctrl_dt / config.sim_dt))
    assert np.isclose(
        config.horizon - config.horizon_steps * config.sim_dt, 0, atol=1e-5
    ), "horizon must be divisible by sim_dt"
    assert np.isclose(
        config.ctrl_dt - config.ctrl_steps * config.sim_dt, 0, atol=1e-5
    ), "ctrl_dt must be divisible by sim_dt"
    assert np.isclose(
        config.knot_dt - config.knot_steps * config.sim_dt, 0, atol=1e-5
    ), "knot_dt must be divisible by sim_dt"
    return config


def compute_noise_schedule(config: Config) -> Config:
    config.noise_scale = get_noise_scale(config)
    if config.max_num_iterations > 0:
        config.beta_traj = config.final_noise_scale ** (1 / config.max_num_iterations)
    else:
        config.beta_traj = 1.0
    return config


def process_config(config: Config):
    """Process the configuration to fill in the missing fields."""
    config = compute_steps(config)
    # Path Y: physics_dt + decimation
    if config.physics_dt > 0:
        decimation = int(round(config.sim_dt / config.physics_dt))
        assert abs(config.sim_dt - decimation * config.physics_dt) < 1e-5, (
            f"sim_dt ({config.sim_dt}) must be integer multiple of physics_dt ({config.physics_dt})"
        )
        config.sim_decimation = decimation
    else:
        config.sim_decimation = 1
    trace_steps_tmp = int(np.round(config.trace_dt / config.sim_dt))
    assert np.isclose(
        config.trace_dt - trace_steps_tmp * config.sim_dt, 0, atol=1e-3
    ), "trace_dt must be divisible by sim_dt"

    # Set object DOF based on hand type
    if config.contact_guidance:
        config.nq_obj = {
            "bimanual": 12,
            "right": 6,
            "left": 6,
            "humanoid_object": 6,
            "dual_humanoid_object": 6,
        }.get(config.embodiment_type, 0)
    else:
        config.nq_obj = {
            "bimanual": 14,
            "right": 7,
            "left": 7,
            "humanoid_object": 7,
            "dual_humanoid_object": 7,
        }.get(config.embodiment_type, 0)

    # E027b: object_pd_override uses scene_act (6DOF object) regardless of contact_guidance
    if config.object_pd_override and config.embodiment_type == "humanoid_object":
        config.nq_obj = 6

    # resolve processed directories for this trial
    dataset_dir_abs = os.path.abspath(config.dataset_dir)
    processed_dir_robot = get_processed_data_dir(
        dataset_dir=dataset_dir_abs,
        dataset_name=config.dataset_name,
        robot_type=config.robot_type,
        embodiment_type=config.embodiment_type,
        task=config.task,
        data_id=config.data_id,
    )
    # model and data within processed directory (scene_eq.xml support for annealing over equality constraints)
    if config.scene_name:
        scene_xml = f"{config.scene_name}.xml"
    elif config.contact_guidance:
        scene_xml = "scene_act.xml"
    else:
        scene_xml = "scene.xml" if config.num_dyn == 1 else "scene_eq.xml"
    config.model_path = f"{processed_dir_robot}/../{scene_xml}"
    # default to MJWP retargeted trajectory if available (skip if data_path already set)
    if not config.data_path:
        if config.embodiment_type == "dual_humanoid_object":
            config.data_path = f"{processed_dir_robot}/trajectory_kinematic_dual.npz"
        else:
            # Always load freejoint data; scene_act conversion happens in run_mjwp.py
            config.data_path = f"{processed_dir_robot}/trajectory_kinematic.npz"

    # get model data
    if config.simulator == "mjwp":
        model = mujoco.MjModel.from_xml_path(config.model_path)
        config.nq = model.nq
        config.nv = model.nv
        config.nu = model.nu
        config.npair = model.npair
        if config.contact_guidance:
            if config.object_action_dims <= 0:
                config.object_action_dims = (
                    12 if config.embodiment_type == "bimanual" else 6
                )
            desired_names = (
                config.object_pos_actuator_names + config.object_rot_actuator_names
            )
            object_ids, object_names = resolve_object_actuator_ids(
                model,
                desired_names,
                config.object_action_dims,
            )
            config.object_actuator_ids = object_ids
            config.object_actuator_names = object_names
            config.right_pos_ctrl_ids, config.left_pos_ctrl_ids = (
                get_object_pos_ctrl_indices(config)
            )
            config.contact_order, config.hand_contact_site_ids = (
                build_hand_contact_site_ids(model, config.embodiment_type)
            )
            if config.embodiment_type == "humanoid_object":
                # Humanoid: use all hand contacts (no finger distinction)
                config.right_contact_indices = [
                    idx
                    for idx, (side, finger) in enumerate(config.contact_order)
                    if side == "right"
                ]
                config.left_contact_indices = [
                    idx
                    for idx, (side, finger) in enumerate(config.contact_order)
                    if side == "left"
                ]
            else:
                config.right_contact_indices = [
                    idx
                    for idx, (side, finger) in enumerate(config.contact_order)
                    if (side == "right") and (finger in ["thumb"])
                ]
                config.left_contact_indices = [
                    idx
                    for idx, (side, finger) in enumerate(config.contact_order)
                    if side == "left" and (finger in ["thumb"])
                ]

    # get noise scale
    if config.zero_noise_joint_keywords:
        config._model_cpu_for_noise = model
    config = compute_noise_schedule(config)

    # Resolve task_body_names → task_body_ids from model
    if config.task_body_names and config.simulator == "mjwp":
        resolved_ids = []
        for name in config.task_body_names:
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid == -1:
                loguru.logger.warning(
                    "task_body_names: body '{}' not found in model, skipping.", name
                )
            else:
                resolved_ids.append(bid)
        config.task_body_ids = resolved_ids
        if len(config.task_body_weights) != len(config.task_body_ids):
            loguru.logger.warning(
                "task_body_weights length ({}) != task_body_ids length ({}); using uniform weights.",
                len(config.task_body_weights),
                len(config.task_body_ids),
            )
            config.task_body_weights = [1.0] * len(config.task_body_ids)
        loguru.logger.info(
            "Task-space body tracking: {} bodies resolved.",
            len(config.task_body_ids),
        )

    # Resolve hand_approach_body_ids and object half-extents for E025/E039
    if (
        config.hand_approach_rew_scale > 0.0
        or config.contact_mask_rew_scale > 0.0
        or config.contact_hdmi_gain > 0.0
        or config.robot_object_penalty_scale > 0.0
        or config.leg_object_penalty_scale > 0.0
        or config.hand_floor_penalty_scale > 0.0
        or config.object_lift_rew_scale > 0.0
        or config.object_floor_penalty_scale > 0.0
        or config.cem_safety_gate_enabled
        or config.cem_hand_gate_enabled
        or config.object_clearance_rew_scale > 0.0
        or config.object_clearance_penalty_scale > 0.0
        or config.carry_corridor_rew_scale > 0.0
        or config.hand_support_rew_scale > 0.0
        or config.nonhand_support_penalty_scale > 0.0
        or config.terminal_carry_gate_enabled
    ) and config.simulator == "mjwp":
        resolved_ids = []
        for name in config.hand_approach_body_names:
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid != -1:
                resolved_ids.append(bid)
            else:
                loguru.logger.warning(
                    "hand_approach_body_names: body '{}' not found.", name
                )
        config.hand_approach_body_ids = resolved_ids
        # Resolve object half-extents from collision geom
        obj_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
        if obj_body_id != -1:
            for g in range(model.ngeom):
                if model.geom_bodyid[g] == obj_body_id:
                    gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g)
                    if gname and "collision" in gname:
                        config.hand_approach_obj_half_extents = [
                            float(x) for x in model.geom_size[g]
                        ]
                        break
            if not config.hand_approach_obj_half_extents:
                # Fallback: use first object geom size
                for g in range(model.ngeom):
                    if model.geom_bodyid[g] == obj_body_id:
                        config.hand_approach_obj_half_extents = [
                            float(x) for x in model.geom_size[g]
                        ]
                        break
        loguru.logger.info(
            "Hand approach: {} bodies, obj half_ext={}",
            len(config.hand_approach_body_ids),
            config.hand_approach_obj_half_extents,
        )

    if config.simulator == "mjwp":
        if config.robot_object_penalty_scale > 0.0:
            geom_ids = []
            for name in config.robot_object_penalty_geom_names:
                gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
                if gid != -1:
                    geom_ids.append(gid)
                else:
                    loguru.logger.warning(
                        "robot_object_penalty_geom_names: geom '{}' not found.", name
                    )
            config.robot_object_penalty_geom_ids = geom_ids
            loguru.logger.info(
                "Robot/object penalty: {} geoms resolved.", len(geom_ids)
            )
        if config.leg_object_penalty_scale > 0.0:
            geom_ids = []
            for name in config.leg_object_penalty_geom_names:
                gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
                if gid != -1:
                    geom_ids.append(gid)
                else:
                    loguru.logger.warning(
                        "leg_object_penalty_geom_names: geom '{}' not found.", name
                    )
            config.leg_object_penalty_geom_ids = geom_ids
            loguru.logger.info(
                "Leg/object penalty: {} geoms resolved.", len(geom_ids)
            )
        if config.carry_corridor_rew_scale > 0.0:
            geom_ids = []
            for name in config.carry_corridor_leg_geom_names:
                gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
                if gid != -1:
                    geom_ids.append(gid)
                else:
                    loguru.logger.warning(
                        "carry_corridor_leg_geom_names: geom '{}' not found.", name
                    )
            config.carry_corridor_leg_geom_ids = geom_ids
            loguru.logger.info(
                "Carry corridor leg geoms: {} resolved.", len(geom_ids)
            )
        if config.hand_floor_penalty_scale > 0.0:
            geom_ids = []
            for name in config.hand_floor_penalty_geom_names:
                gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
                if gid != -1:
                    geom_ids.append(gid)
                else:
                    loguru.logger.warning(
                        "hand_floor_penalty_geom_names: geom '{}' not found.", name
                    )
            config.hand_floor_penalty_geom_ids = geom_ids
            loguru.logger.info(
                "Hand/floor penalty: {} geoms resolved.", len(geom_ids)
            )
        if config.hand_object_deep_penalty_scale > 0.0:
            geom_ids = []
            for name in config.hand_object_deep_penalty_geom_names:
                gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
                if gid != -1:
                    geom_ids.append(gid)
                else:
                    loguru.logger.warning(
                        "hand_object_deep_penalty_geom_names: geom '{}' not found.",
                        name,
                    )
            config.hand_object_deep_penalty_geom_ids = geom_ids
            loguru.logger.info(
                "Hand/object deep penalty: {} geoms resolved.", len(geom_ids)
            )
        if config.hand_support_rew_scale > 0.0 or config.terminal_carry_gate_enabled:
            geom_ids = []
            for name in config.hand_support_geom_names:
                gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
                if gid != -1:
                    geom_ids.append(gid)
                else:
                    loguru.logger.warning(
                        "hand_support_geom_names: geom '{}' not found.", name
                    )
            config.hand_support_geom_ids = geom_ids
            loguru.logger.info("Hand support geoms: {} resolved.", len(geom_ids))
        if (
            config.nonhand_support_penalty_scale > 0.0
            or config.terminal_carry_gate_enabled
        ):
            geom_ids = []
            for name in config.nonhand_support_penalty_geom_names:
                gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
                if gid != -1:
                    geom_ids.append(gid)
                else:
                    loguru.logger.warning(
                        "nonhand_support_penalty_geom_names: geom '{}' not found.",
                        name,
                    )
            config.nonhand_support_penalty_geom_ids = geom_ids
            loguru.logger.info(
                "Non-hand support penalty geoms: {} resolved.", len(geom_ids)
            )
        if config.cem_safety_gate_enabled:
            geom_ids = []
            for name in config.cem_safety_gate_geom_names:
                gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
                if gid != -1:
                    geom_ids.append(gid)
                else:
                    loguru.logger.warning(
                        "cem_safety_gate_geom_names: geom '{}' not found.", name
                    )
            config.cem_safety_gate_geom_ids = geom_ids
            loguru.logger.info(
                "CEM safety gate: {} geoms resolved.", len(geom_ids)
            )
        if config.cem_hand_gate_enabled:
            geom_ids = []
            for name in config.cem_hand_gate_geom_names:
                gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
                if gid != -1:
                    geom_ids.append(gid)
                else:
                    loguru.logger.warning(
                        "cem_hand_gate_geom_names: geom '{}' not found.", name
                    )
            config.cem_hand_gate_geom_ids = geom_ids
            loguru.logger.info("CEM hand gate: {} geoms resolved.", len(geom_ids))

    # output dir: write artifacts alongside the trial unless explicitly overridden
    if not config.output_dir:
        config.output_dir = processed_dir_robot
    os.makedirs(config.output_dir, exist_ok=True)

    # read task info
    task_info_path = f"{processed_dir_robot}/../task_info.json"
    try:
        with open(task_info_path, encoding="utf-8") as f:
            task_info = json.load(f)
    except FileNotFoundError:
        loguru.logger.warning(
            f"task_info.json not found at {task_info_path}, using default values"
        )
        task_info = {}
    if "ref_dt" in task_info:
        config.ref_dt = task_info["ref_dt"]
        loguru.logger.info(f"overriding ref_dt: {config.ref_dt} from task_info.json")

    # override contact site ids
    if config.contact_rew_scale > 0.0:
        if "contact_site_ids" in task_info:
            config.contact_site_ids = task_info["contact_site_ids"]
            loguru.logger.info(
                f"overriding contact_site_ids: {config.contact_site_ids} from task_info.json"
            )
        else:
            raise ValueError(
                "contact_site_ids not found in task_info.json while contact_rew_scale > 0.0"
            )

    # set seed
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    random.seed(config.seed)

    return config
