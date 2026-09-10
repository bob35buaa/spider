# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""A standalone script to run DIAL MPC with Mujoco + Warp

Up to now, domain randomization is not supported. Will add it later.

Author: Chaoyi Pan
Date: 2025-08-11
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import fields
from pathlib import Path

import hydra
import imageio
import loguru
import mujoco
import numpy as np
import torch
import warp as wp
from omegaconf import DictConfig, OmegaConf

from spider.config import (
    Config,
    filter_config_fields,
    load_config_yaml,
    process_config,
)
from spider.interp import get_slice
from spider.io import load_data
from spider.optimizers.sampling import (
    make_optimize_fn,
    make_optimize_once_fn,
    make_rollout_fn,
)
from spider.query_tape import cem_query_tape_chunk_count
from spider.postprocess.get_success_rate import compute_object_tracking_error
from spider.simulators.mjwp import (
    compute_contact_point_delta,
    copy_sample_state,
    get_geometry_state,
    get_partner_force_state,
    get_qpos,
    get_qvel,
    get_reward,
    get_support_proxy_state,
    get_terminal_reward,
    get_terminate,
    get_trace,
    load_env_params,
    load_state,
    save_env_params,
    save_state,
    setup_env,
    setup_mj_model,  # mjwp specific
    step_env,
    sync_env,
)
from spider.simulators.scene_act_reference import resolve_scene_act_reference
from spider.viewers import (
    log_frame,
    render_image,
    setup_renderer,
    setup_viewer,
    update_viewer,
)

_CONFIG_SKIP_FIELDS = {
    "noise_scale",
    "env_params_list",
    "viewer_body_entity_and_ids",
}


def _parse_override_tokens(tokens: list[str]) -> dict:
    allowed = {field.name for field in fields(Config)}
    override_dict: dict = {}
    for item in tokens:
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        key = key.lstrip("+")
        if key not in allowed:
            continue
        parsed = OmegaConf.to_container(
            OmegaConf.from_dotlist([f"{key}={value}"]), resolve=True
        )
        if isinstance(parsed, dict) and key in parsed:
            override_dict[key] = parsed[key]
    return override_dict


def _extract_cli_overrides(cfg: DictConfig) -> dict:
    """Extract CLI overrides so they can be applied on top of a loaded config."""
    overrides = OmegaConf.select(cfg, "hydra.overrides.task") or []
    override_dict = _parse_override_tokens(overrides)
    if override_dict:
        return override_dict
    return _parse_override_tokens(sys.argv[1:])


def _aggregate_info_list(info_list: list[dict]) -> dict:
    info_aggregated = {}
    keys = []
    seen = set()
    for info in info_list:
        for key in info:
            if key not in seen:
                seen.add(key)
                keys.append(key)

    def to_numpy(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        return value

    for k in keys:
        sample = next((to_numpy(info[k]) for info in info_list if k in info), None)
        if sample is None:
            continue
        filler = np.zeros_like(sample)
        values = [to_numpy(info[k]) if k in info else filler for info in info_list]
        try:
            info_aggregated[k] = np.stack(values, axis=0)
        except ValueError as exc:
            shapes = [np.shape(v) for v in values]
            unique_shapes = sorted({str(shape) for shape in shapes})
            loguru.logger.warning(
                "Skipping info key '{}' because shapes vary across ticks: {} ({})",
                k,
                unique_shapes,
                exc,
            )
    return info_aggregated


def _assert_object_actuator_gains_zero(
    env, config: Config, stage: str, atol: float = 1e-4
) -> None:
    if not config.contact_guidance or not config.object_actuator_ids:
        return
    actuator_ids = np.asarray(config.object_actuator_ids, dtype=int)
    if not hasattr(env, "model_wp") or not hasattr(env.model_wp, "actuator_gainprm"):
        raise AssertionError("MJWarp model does not expose actuator_gainprm.")
    gainprm = wp.to_torch(env.model_wp.actuator_gainprm).detach().cpu().numpy()
    biasprm = wp.to_torch(env.model_wp.actuator_biasprm).detach().cpu().numpy()
    if gainprm.ndim == 3:
        gainprm = gainprm[0]
    if biasprm.ndim == 3:
        biasprm = biasprm[0]
    kp = gainprm[actuator_ids, 0]
    kd = -biasprm[actuator_ids, 1]
    assert np.allclose(kp, 0.0, atol=atol) or config.residual_gain_ratio > 0, (
        f"Object actuator Kp not near zero at {stage}: max={np.max(np.abs(kp))}"
    )
    assert np.allclose(kd, 0.0, atol=atol) or config.residual_gain_ratio > 0, (
        f"Object actuator Kd not near zero at {stage}: max={np.max(np.abs(kd))}"
    )


def _normalize_yaml_value(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return list(value)
    return value


def _save_config_yaml(config: Config) -> None:
    if not config.save_config:
        return
    config_dict = {}
    for field in fields(config):
        if field.name in _CONFIG_SKIP_FIELDS:
            continue
        config_dict[field.name] = _normalize_yaml_value(getattr(config, field.name))
    output_path = (
        Path(config.output_dir)
        / f"config{'_act' if config.contact_guidance else ''}.yaml"
    )
    OmegaConf.save(config=OmegaConf.create(config_dict), f=str(output_path))
    loguru.logger.info(f"Saved config to {output_path}")


def _get_bimanual_hand_indices(config: Config) -> tuple[list[int], list[int]]:
    robot_nu = int(config.nu)
    if config.contact_guidance:
        obj_dims = (
            int(config.object_action_dims) if config.object_action_dims > 0 else 12
        )
        robot_nu = max(robot_nu - obj_dims, 0)
    half = robot_nu // 2
    right_ids = list(range(0, half))
    left_ids = list(range(half, robot_nu))
    return right_ids, left_ids


def _apply_noise_mask(
    base_noise_scale: torch.Tensor, zero_indices: list[int]
) -> torch.Tensor:
    noise_scale = base_noise_scale.clone()
    if zero_indices:
        idx = torch.as_tensor(
            zero_indices, device=base_noise_scale.device, dtype=torch.long
        )
        noise_scale[:, :, idx] *= 0.0
    return noise_scale


def run_sbto(
    config,
    env,
    ref_data,
    mj_model,
    mj_data,
    mj_data_ref,
    qpos_ref,
    qvel_ref,
    ctrl_ref,
    renderer,
    images,
):
    """SBTO: Sampling-Based Trajectory Optimization (DynaRetarget Algorithm 2).

    Incrementally grows the optimization horizon from knot 0 to the full trajectory,
    warm-starting each increment from the previous solution.
    """
    from spider.config import compute_noise_schedule
    from spider.interp import get_slice
    from spider.simulators.mjwp import (
        step_env,
        save_state,
        load_state,
        get_reward,
        get_terminal_reward,
        get_terminate,
        get_trace,
        save_env_params,
        load_env_params,
        copy_sample_state,
        sync_env,
        get_qpos,
        get_qvel,
    )
    from spider.optimizers.sampling import (
        make_rollout_fn,
        make_optimize_once_fn,
        make_optimize_fn,
    )
    from spider.viewers import render_image

    total_steps = config.max_sim_steps
    sbto_knot_steps = int(np.round(config.sbto_knot_dt / config.sim_dt))
    total_knots = total_steps // sbto_knot_steps
    loguru.logger.info(
        "SBTO: total_steps={}, knot_dt={}, total_knots={}",
        total_steps,
        config.sbto_knot_dt,
        total_knots,
    )

    # Build optimizer
    rollout = make_rollout_fn(
        step_env,
        save_state,
        load_state,
        get_reward,
        get_terminal_reward,
        get_terminate,
        get_trace,
        save_env_params,
        load_env_params,
        copy_sample_state,
    )
    optimize_once = make_optimize_once_fn(rollout)
    optimize = make_optimize_fn(optimize_once)

    # Initialize controls from reference
    full_ctrls = ctrl_ref[:total_steps].clone()

    # Save original config for restore
    orig_horizon = config.horizon
    orig_horizon_steps = config.horizon_steps
    orig_knot_dt = config.knot_dt
    orig_knot_steps = config.knot_steps

    config.knot_dt = config.sbto_knot_dt
    config.knot_steps = sbto_knot_steps

    # Gibbs for dual humanoid
    gibbs_enabled = (
        config.gibbs_sampling and config.embodiment_type == "dual_humanoid_object"
    )
    if gibbs_enabled:
        half_nu = config.nu // 2
        robot1_ids = list(range(0, half_nu))
        robot2_ids = list(range(half_nu, config.nu))

    t_start = time.perf_counter()
    for k in range(1, total_knots):
        active_steps = min((k + 1) * sbto_knot_steps, total_steps)
        active_ctrls = full_ctrls[:active_steps]

        # Update config for growing horizon
        config.horizon = active_steps * config.sim_dt
        config.horizon_steps = active_steps
        config = compute_noise_schedule(config)

        # Reference slice (offset +1 for lookahead)
        end_idx = min(active_steps + 1, ref_data[0].shape[0])
        ref_slice = get_slice(ref_data, 1, end_idx)
        # Pad if ref is shorter than active horizon
        if ref_slice[0].shape[0] < active_steps:
            pad_len = active_steps - ref_slice[0].shape[0]
            ref_slice = tuple(
                torch.cat([s, s[-1:].repeat(pad_len, *([1] * (s.ndim - 1)))], dim=0)
                for s in ref_slice
            )

        # Optimize this horizon increment using optimize_once (single CEM iterations)
        # DynaRetarget Algorithm 2: inner loop converges when max(diag(Σ)) < σ_min
        # Σ is updated via EWMA from elite sample statistics each iteration
        current_noise = config.noise_scale.clone()  # (N, knot_steps, nu)
        alpha_cov = config.sbto_cov_momentum  # α_Σ = 0.2 (paper)

        for iteration in range(config.sbto_max_iter_per_knot):
            # SBTO sample_params: pass elite_fraction, mean_momentum, request elite_std
            sample_params = {
                "global_noise_scale": 1.0,
                "elite_fraction": config.sbto_elite_fraction,
                "mean_momentum": config.sbto_mean_momentum,
                "return_elite_std": True,
            }

            if gibbs_enabled:
                base_ns = current_noise.clone()
                config.noise_scale = _apply_noise_mask(base_ns, robot2_ids)
                active_ctrls, terminate, info = optimize_once(
                    config,
                    env,
                    active_ctrls,
                    ref_slice,
                    config.env_params_list[
                        min(iteration, len(config.env_params_list) - 1)
                    ],
                    sample_params,
                )
                config.noise_scale = _apply_noise_mask(base_ns, robot1_ids)
                active_ctrls, terminate, info = optimize_once(
                    config,
                    env,
                    active_ctrls,
                    ref_slice,
                    config.env_params_list[
                        min(iteration, len(config.env_params_list) - 1)
                    ],
                    sample_params,
                )
                config.noise_scale = base_ns
            else:
                active_ctrls, terminate, info = optimize_once(
                    config,
                    env,
                    active_ctrls,
                    ref_slice,
                    config.env_params_list[
                        min(iteration, len(config.env_params_list) - 1)
                    ],
                    sample_params,
                )

            # Sigma EWMA: Σ_new = α_Σ · Σ_old + (1-α_Σ) · Σ_elite
            elite_std = info.get("elite_std", None)
            if elite_std is not None:
                # elite_std shape: (H, nu) — expand to match noise_scale (N, knot_steps, nu)
                # Map horizon steps back to knot steps by subsampling
                knot_step_size = max(1, elite_std.shape[0] // current_noise.shape[1])
                elite_knot_std = elite_std[::knot_step_size][: current_noise.shape[1]]
                if elite_knot_std.shape[0] < current_noise.shape[1]:
                    # Pad with last value
                    pad = elite_knot_std[-1:].expand(
                        current_noise.shape[1] - elite_knot_std.shape[0], -1
                    )
                    elite_knot_std = torch.cat([elite_knot_std, pad], dim=0)
                # EWMA update (broadcast across samples dimension)
                current_noise_mean = current_noise.mean(dim=0)  # (knot_steps, nu)
                new_noise_mean = (
                    alpha_cov * current_noise_mean + (1.0 - alpha_cov) * elite_knot_std
                )
                # Scale all samples proportionally
                scale = new_noise_mean / (current_noise_mean + 1e-8)
                current_noise = current_noise * scale.unsqueeze(0)
                config.noise_scale = current_noise

            # Convergence criterion: max(noise_scale) < σ_min (DynaRetarget paper)
            max_sigma = current_noise.abs().max().item()
            if max_sigma < config.sbto_sigma_min:
                break

        full_ctrls[:active_steps] = active_ctrls
        elapsed = time.perf_counter() - t_start
        rew_val = info.get("rew_max", 0.0)
        print(
            f"SBTO: knot {k}/{total_knots}, h={active_steps * config.sim_dt:.2f}s, iter={iteration + 1}, max_σ={max_sigma:.4f}, rew={rew_val:.3f}, t={elapsed:.0f}s"
        )

    # Restore config
    config.horizon = orig_horizon
    config.horizon_steps = orig_horizon_steps
    config.knot_dt = orig_knot_dt
    config.knot_steps = orig_knot_steps
    config = compute_noise_schedule(config)

    # Execute optimized trajectory: collect qpos for saving + render video
    mj_data.qpos[:] = qpos_ref[0].detach().cpu().numpy()
    mj_data.qvel[:] = qvel_ref[0].detach().cpu().numpy()
    mj_data.time = 0.0
    sync_env(config, env, mj_data)

    step_info = {
        "qpos": [],
        "qvel": [],
        "time": [],
        "ctrl": [],
        "qpos_ref": [],
        "qvel_ref": [],
        "time_ref": [],
        "ctrl_ref": [],
    }
    for step_idx in range(total_steps):
        ctrl_step = full_ctrls[step_idx]
        step_env(config, env, ctrl_step)
        mj_data.qpos[:] = get_qpos(config, env)[0].detach().cpu().numpy()
        mj_data.qvel[:] = get_qvel(config, env)[0].detach().cpu().numpy()
        mj_data.ctrl[:] = ctrl_step.detach().cpu().numpy()
        mj_data.time += config.sim_dt
        ref_idx = min(step_idx, qpos_ref.shape[0] - 1)
        step_info["qpos"].append(mj_data.qpos.copy())
        step_info["qvel"].append(mj_data.qvel.copy())
        ref_ctrl = ctrl_ref[min(step_idx, ctrl_ref.shape[0] - 1)].detach().cpu().numpy()
        step_info["time"].append(mj_data.time)
        step_info["ctrl"].append(mj_data.ctrl.copy())
        step_info["qpos_ref"].append(qpos_ref[ref_idx].detach().cpu().numpy())
        step_info["qvel_ref"].append(qvel_ref[ref_idx].detach().cpu().numpy())
        step_info["time_ref"].append(mj_data.time)
        step_info["ctrl_ref"].append(ref_ctrl)
        if config.save_video and renderer is not None:
            if step_idx % int(np.round(config.render_dt / config.sim_dt)) == 0:
                mj_data_ref.qpos[:] = qpos_ref[ref_idx].detach().cpu().numpy()
                image = render_image(config, renderer, mj_model, mj_data, mj_data_ref)
                images.append(image)
    for kk in step_info:
        step_info[kk] = np.stack(step_info[kk], axis=0)

    t_end = time.perf_counter()
    print(f"SBTO total: {t_end - t_start:.1f}s")
    return [step_info]


def main(config: Config):
    """Run the SPIDER using MuJoCo Warp backend"""
    # process config, set defaults and derived fields
    config = process_config(config)
    if config.contact_guidance and config.improvement_threshold > 0.0:
        loguru.logger.warning(
            "contact_guidance requires improvement_threshold <= 0; overriding to 0.0."
        )
        config.improvement_threshold = 0.0

    # load reference data (already interpolated and extended)
    qpos_ref, qvel_ref, ctrl_ref, contact, contact_pos = load_data(
        config, config.data_path
    )

    # E058: optional warmstart — replace qpos_ref slices in intent window with
    # snap-projected qpos (e.g. from spider/preprocess/hand_snap_ik.py). The
    # snap file is at source ref_dt; we re-interpolate to match qpos_ref length.
    if config.warmstart_qpos_path:
        from spider.interp import interp as _interp

        ws = np.load(config.warmstart_qpos_path)
        qpos_snap_src = (
            torch.from_numpy(ws["qpos_snap"]).to(qpos_ref.device).to(qpos_ref.dtype)
        )
        snap_mask_src = torch.from_numpy(ws["snap_mask"].astype(np.float32)).to(
            qpos_ref.device
        )
        if config.ref_dt > config.sim_dt:
            qpos_snap_i = _interp(qpos_snap_src.unsqueeze(0), config.ref_steps).squeeze(
                0
            )
            # nearest-neighbor mask upsample by repeat (avoid spider.interp align_corners bug)
            snap_mask_i = snap_mask_src.repeat_interleave(config.ref_steps)
        else:
            ds = int(config.sim_dt / config.ref_dt)
            qpos_snap_i = qpos_snap_src[::ds]
            snap_mask_i = snap_mask_src[::ds]
        # pad with last frame to match qpos_ref length (matches load_data trailing repeat)
        n_pad = qpos_ref.shape[0] - qpos_snap_i.shape[0]
        if n_pad > 0:
            qpos_snap_i = torch.cat(
                [qpos_snap_i, qpos_snap_i[-1:].repeat(n_pad, 1)], dim=0
            )
            snap_mask_i = torch.cat(
                [snap_mask_i, torch.zeros(n_pad, device=snap_mask_i.device)], dim=0
            )
        elif n_pad < 0:
            qpos_snap_i = qpos_snap_i[: qpos_ref.shape[0]]
            snap_mask_i = snap_mask_i[: qpos_ref.shape[0]]
        assert qpos_snap_i.shape == qpos_ref.shape, (
            f"warmstart shape mismatch after interp: {qpos_snap_i.shape} vs {qpos_ref.shape}"
        )
        snap_mask_b = snap_mask_i > 0.5
        qpos_ref = torch.where(snap_mask_b.unsqueeze(-1), qpos_snap_i, qpos_ref)
        if config.warmstart_update_ctrl_from_qpos:
            if config.embodiment_type != "humanoid_object":
                raise ValueError(
                    "warmstart_update_ctrl_from_qpos currently supports humanoid_object only"
                )
            robot_ctrl_dim = min(
                ctrl_ref.shape[1],
                max(int(config.nu) - max(int(config.object_action_dims), 0), 0),
                max(qpos_ref.shape[1] - 7, 0),
            )
            if robot_ctrl_dim <= 0:
                raise ValueError(
                    "warmstart_update_ctrl_from_qpos could not infer robot control dimension"
                )
            ctrl_snap = qpos_snap_i[:, 7 : 7 + robot_ctrl_dim]
            ctrl_ref[:, :robot_ctrl_dim] = torch.where(
                snap_mask_b.unsqueeze(-1),
                ctrl_snap,
                ctrl_ref[:, :robot_ctrl_dim],
            )
        n_replaced = int(snap_mask_b.sum().item())
        loguru.logger.info(
            f"[E058 warmstart] {config.warmstart_qpos_path}: replaced {n_replaced}/{qpos_ref.shape[0]} frames of qpos_ref"
        )

    if config.contact_guidance and ctrl_ref.shape[1] != config.nu:
        loguru.logger.info(
            "Preserving raw ctrl reference for contact guidance (ctrl dims: {} -> {}); "
            "scene_act conversion will pad object controls when applicable.",
            ctrl_ref.shape[1],
            config.nu,
        )
    if config.contact_guidance and torch.all(contact <= 0):
        raise ValueError("contact_guidance is enabled, but contact mask is all zeros.")
    # hack: start from step 500
    # qpos_ref = qpos_ref[500:]
    # qvel_ref = qvel_ref[500:]
    # ctrl_ref = ctrl_ref[500:]
    # contact = contact[500:]
    # contact_pos = contact_pos[500:]
    ref_data = (qpos_ref, qvel_ref, ctrl_ref, contact, contact_pos)
    # E027b: convert freejoint ref (nq=43) to scene_act format (nq=42) by quat→euler
    if (config.object_pd_override or config.contact_guidance) and qpos_ref.shape[
        1
    ] > config.nq:
        from scipy.spatial.transform import Rotation as R
        import mujoco as _mj

        nq_model = config.nq  # 42 for scene_act
        nq_robot = nq_model - 6  # 36
        # Extract object pos(3) + quat(4) from end of freejoint ref
        obj_pos_world = qpos_ref[:, nq_robot : nq_robot + 3].detach().cpu().numpy()
        obj_quat_wxyz = qpos_ref[:, nq_robot + 3 : nq_robot + 7].detach().cpu().numpy()
        # Get object body_pos from scene_act model (slide joints are relative to this)
        _m_act = _mj.MjModel.from_xml_path(config.model_path)
        _reference_contract = resolve_scene_act_reference(config.model_path, _m_act)
        _obj_body_id = _reference_contract.object_body_id
        body_pos = _m_act.body_pos[_obj_body_id]
        # Slide position = R_body^-1 * (world_pos - body_pos)
        # Slide joints operate in the body frame, not world frame
        body_quat_wxyz_pos = _m_act.body_quat[_obj_body_id]
        body_quat_xyzw_pos = [
            body_quat_wxyz_pos[1],
            body_quat_wxyz_pos[2],
            body_quat_wxyz_pos[3],
            body_quat_wxyz_pos[0],
        ]
        R_body_pos = R.from_quat(body_quat_xyzw_pos)
        world_offset = obj_pos_world - body_pos[np.newaxis, :]
        obj_slide_pos = R_body_pos.inv().apply(world_offset)
        # The convention is inferred from compiled hinge order and must match metadata.
        euler_conv = _reference_contract.convention
        # Get body_quat for relative rotation: R_joint = R_body^-1 * R_world
        body_quat_wxyz = _m_act.body_quat[_obj_body_id]
        body_quat_xyzw = [
            body_quat_wxyz[1],
            body_quat_wxyz[2],
            body_quat_wxyz[3],
            body_quat_wxyz[0],
        ]
        R_body = R.from_quat(body_quat_xyzw)
        # Convert world quat to relative euler
        obj_quat_xyzw = np.column_stack(
            [
                obj_quat_wxyz[:, 1],
                obj_quat_wxyz[:, 2],
                obj_quat_wxyz[:, 3],
                obj_quat_wxyz[:, 0],
            ]
        )
        R_world = R.from_quat(obj_quat_xyzw)
        R_joint = R_body.inv() * R_world
        obj_euler = R_joint.as_euler(euler_conv)
        # Build new qpos: robot(36) + obj_slide(3) + obj_euler(3) = 42
        qpos_ref_new = torch.zeros(
            (qpos_ref.shape[0], nq_model), device=qpos_ref.device, dtype=qpos_ref.dtype
        )
        qpos_ref_new[:, :nq_robot] = qpos_ref[:, :nq_robot]
        qpos_ref_new[:, nq_robot : nq_robot + 3] = torch.from_numpy(
            obj_slide_pos.astype(np.float32)
        ).to(qpos_ref.device)
        qpos_ref_new[:, nq_robot + 3 : nq_robot + 6] = torch.from_numpy(
            obj_euler.astype(np.float32)
        ).to(qpos_ref.device)
        # Also adapt qvel and ctrl
        nv_model = config.nv  # 41
        qvel_ref_new = torch.zeros(
            (qvel_ref.shape[0], nv_model), device=qvel_ref.device, dtype=qvel_ref.dtype
        )
        qvel_ref_new[:, : min(qvel_ref.shape[1], nv_model)] = qvel_ref[:, :nv_model]
        ctrl_ref_new = torch.zeros(
            (ctrl_ref.shape[0], config.nu), device=ctrl_ref.device, dtype=ctrl_ref.dtype
        )
        ctrl_ref_new[:, : min(ctrl_ref.shape[1], config.nu)] = ctrl_ref[
            :, : min(ctrl_ref.shape[1], config.nu)
        ]
        # Fix: set object actuator ctrl channels to converted slide_pos + euler (body-frame)
        # Object actuators are the last 6 of nu (ids 29-34 for G1)
        obj_act_start = config.nu - 6
        ctrl_ref_new[:, obj_act_start : obj_act_start + 3] = torch.from_numpy(
            obj_slide_pos.astype(np.float32)
        ).to(ctrl_ref.device)
        ctrl_ref_new[:, obj_act_start + 3 : obj_act_start + 6] = torch.from_numpy(
            obj_euler.astype(np.float32)
        ).to(ctrl_ref.device)
        loguru.logger.info(
            "E027b: converted ref nq {} → {} (quat→{} euler, body_pos={})",
            qpos_ref.shape[1],
            nq_model,
            euler_conv,
            body_pos.tolist(),
        )
        qpos_ref = qpos_ref_new
        qvel_ref = qvel_ref_new
        ctrl_ref = ctrl_ref_new
        ref_data = (qpos_ref, qvel_ref, ctrl_ref, contact, contact_pos)

    ref_dims = {
        "qpos_ref": (int(qpos_ref.shape[1]), int(config.nq)),
        "qvel_ref": (int(qvel_ref.shape[1]), int(config.nv)),
        "ctrl_ref": (int(ctrl_ref.shape[1]), int(config.nu)),
    }
    bad_ref_dims = {name: dims for name, dims in ref_dims.items() if dims[0] != dims[1]}
    if bad_ref_dims:
        detail = ", ".join(
            f"{name}={got} expected {expected}"
            for name, (got, expected) in bad_ref_dims.items()
        )
        raise ValueError(
            "Reference/model dimension mismatch after preprocessing: "
            f"{detail}. This usually means ctrl_ref is not actuator-space."
        )
    config.max_sim_steps = (
        config.max_sim_steps
        if config.max_sim_steps > 0
        else qpos_ref.shape[0] - config.horizon_steps - config.ctrl_steps
    )

    # setup env with initial state from first sim qpos
    env = setup_env(config, ref_data)

    # E013: freejoint object oracle. This is intentionally separate from
    # object_pd_override, which only works for scene_act object actuators.
    object_kinematic_enabled = bool(config.object_kinematic_override) or (
        config.partner_force_spring_kp < 0
    )
    if object_kinematic_enabled and config.embodiment_type in ["humanoid_object"]:
        if int(config.nq_obj) != 7:
            raise ValueError(
                "object_kinematic_override requires true-freejoint object "
                f"with nq_obj=7, got nq_obj={config.nq_obj}."
            )
        env.object_kinematic_ref_qpos = qpos_ref[:, -7:].to(
            config.device, dtype=torch.float32
        )
        env.object_kinematic_ref_qvel = qvel_ref[:, -6:].to(
            config.device, dtype=torch.float32
        )
        env.object_kinematic_ref_dt = (
            float(config.object_kinematic_ref_dt)
            if config.object_kinematic_ref_dt > 0
            else float(config.sim_dt)
        )
        loguru.logger.info(
            "Object kinematic oracle: ref_qpos shape={}, ref_qvel shape={}, ref_dt={}",
            tuple(env.object_kinematic_ref_qpos.shape),
            tuple(env.object_kinematic_ref_qvel.shape),
            env.object_kinematic_ref_dt,
        )

    # E025/E026/E028/E030: precompute partner force reference object positions + quaternions
    if (
        config.partner_force_spring_kp > 0 or config.scene_name == "scene_weld"
    ) and config.embodiment_type in ["humanoid_object", "dual_humanoid_object"]:
        # Object freejoint: last 7 dof in qpos [nq-7:nq] = [pos(3), quat(4)]
        nq_obj = 7
        obj_pos_ref_np = (
            qpos_ref[:, -nq_obj : -nq_obj + 3].detach().cpu().numpy()
        )  # (T, 3)
        obj_quat_ref_np = (
            qpos_ref[:, -nq_obj + 3 :].detach().cpu().numpy()
        )  # (T, 4) wxyz
        env.partner_force_ref_pos = torch.tensor(
            obj_pos_ref_np, device=config.device, dtype=torch.float32
        )
        env.partner_force_ref_quat = torch.tensor(
            obj_quat_ref_np, device=config.device, dtype=torch.float32
        )
        env.partner_force_ref_dt = (
            float(config.partner_force_ref_dt)
            if config.partner_force_ref_dt > 0
            else float(config.ref_dt)
        )
        loguru.logger.info(
            "Partner force spring: ref_pos shape={}, ref_quat shape={}, ref_dt={}",
            tuple(env.partner_force_ref_pos.shape),
            tuple(env.partner_force_ref_quat.shape),
            env.partner_force_ref_dt,
        )

    # E027b: object PD override — precompute ref pos/euler for scene_act object actuators
    if config.object_pd_override and config.embodiment_type in ["humanoid_object"]:
        # After E027b conversion, qpos_ref is 42-dim: robot(36) + obj(6: px,py,pz,rx,ry,rz)
        nq_robot = config.nq - 6
        obj_pos_ref_np = qpos_ref[:, nq_robot : nq_robot + 3].detach().cpu().numpy()
        obj_euler_ref_np = (
            qpos_ref[:, nq_robot + 3 : nq_robot + 6].detach().cpu().numpy()
        )
        env.object_pd_ref_pos = torch.tensor(
            obj_pos_ref_np, device=config.device, dtype=torch.float32
        )
        env.object_pd_ref_euler = torch.tensor(
            obj_euler_ref_np, device=config.device, dtype=torch.float32
        )
        # Set actuator gains on model (last 6 actuators = object)
        obj_act_ids = list(range(env.model_cpu.nu - 6, env.model_cpu.nu))
        kp_pos = config.object_pd_kp_pos
        kp_rot = config.object_pd_kp_rot
        for i, aid in enumerate(obj_act_ids):
            kp = kp_pos if i < 3 else kp_rot
            env.model_cpu.actuator_gainprm[aid, 0] = kp
            env.model_cpu.actuator_biasprm[aid, 1] = -kp  # position actuator bias
            env.model_cpu.actuator_biasprm[aid, 2] = 0  # no velocity bias
        # Store object mass for gravity compensation in _apply_object_pd_override
        obj_body_id = mujoco.mj_name2id(
            env.model_cpu, mujoco.mjtObj.mjOBJ_BODY, "object"
        )
        env.object_mass = env.model_cpu.body_mass[obj_body_id]
        # Propagate to Warp model
        gain_full = np.array(env.model_cpu.actuator_gainprm, dtype=np.float32)
        bias_full = np.array(env.model_cpu.actuator_biasprm, dtype=np.float32)
        wp.copy(
            env.model_wp.actuator_gainprm,
            wp.from_numpy(gain_full, dtype=wp.float32, device=config.device),
        )
        wp.copy(
            env.model_wp.actuator_biasprm,
            wp.from_numpy(bias_full, dtype=wp.float32, device=config.device),
        )
        loguru.logger.info(
            "Object PD override: kp_pos={}, kp_rot={}, ref shape={}",
            kp_pos,
            kp_rot,
            tuple(env.object_pd_ref_pos.shape),
        )

    # setup mujoco (for viewer only)
    mj_model = setup_mj_model(config)
    mj_data = mujoco.MjData(mj_model)
    mj_data_ref = mujoco.MjData(mj_model)

    # E018: precompute reference body world positions for task-space tracking
    if config.task_body_ids:
        T_body = qpos_ref.shape[0]
        body_xpos_ref_np = np.zeros(
            (T_body, len(config.task_body_ids), 3), dtype=np.float32
        )
        for t in range(T_body):
            mj_data_ref.qpos[:] = qpos_ref[t].detach().cpu().numpy()
            mujoco.mj_kinematics(mj_model, mj_data_ref)
            for k, bid in enumerate(config.task_body_ids):
                body_xpos_ref_np[t, k] = mj_data_ref.xpos[bid]
        body_xpos_ref = torch.tensor(body_xpos_ref_np, device=config.device)
        loguru.logger.info(
            "Precomputed body_xpos_ref: shape={}", tuple(body_xpos_ref.shape)
        )
    else:
        body_xpos_ref = torch.zeros(
            (qpos_ref.shape[0], 0, 3), device=config.device, dtype=torch.float32
        )

    # E034: precompute hand-object contact mask for hand_approach gating
    approach_mask_t = None
    if config.hand_approach_body_ids and config.hand_approach_contact_threshold < float(
        "inf"
    ):
        obj_body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "object")
        if obj_body_id != -1 and config.hand_approach_obj_half_extents:
            half_ext = np.array(config.hand_approach_obj_half_extents)
            T_mask = qpos_ref.shape[0]
            approach_mask_np = np.zeros(T_mask, dtype=np.float32)
            for t in range(T_mask):
                mj_data_ref.qpos[:] = qpos_ref[t].detach().cpu().numpy()
                mujoco.mj_kinematics(mj_model, mj_data_ref)
                obj_pos = mj_data_ref.xpos[obj_body_id]
                for hid in config.hand_approach_body_ids:
                    hand_pos = mj_data_ref.xpos[hid]
                    delta = np.abs(hand_pos - obj_pos)
                    surf_dist = np.linalg.norm(np.maximum(delta - half_ext, 0))
                    if surf_dist < config.hand_approach_contact_threshold:
                        approach_mask_np[t] = 1.0
                        break
            approach_mask_t = torch.tensor(approach_mask_np, device=config.device)
            active_pct = approach_mask_np.mean() * 100
            loguru.logger.info(
                "E034 approach_mask: {:.1f}% frames active (threshold={:.2f}m)",
                active_pct,
                config.hand_approach_contact_threshold,
            )

    # E035/E165-D: precompute full body xpos + xquat for local-frame tracking
    # and peak-margin CEM rerank diagnostics.
    body_xquat_ref_t = None
    body_xpos_full_ref_t = None
    if config.use_local_frame_reward or config.cem_peak_margin_enabled:
        T_full = qpos_ref.shape[0]
        nbody = mj_model.nbody
        body_xpos_full_np = np.zeros((T_full, nbody, 3), dtype=np.float32)
        body_xquat_full_np = np.zeros((T_full, nbody, 4), dtype=np.float32)
        for t in range(T_full):
            mj_data_ref.qpos[:] = qpos_ref[t].detach().cpu().numpy()
            mujoco.mj_kinematics(mj_model, mj_data_ref)
            body_xpos_full_np[t] = mj_data_ref.xpos[:nbody]
            body_xquat_full_np[t] = mj_data_ref.xquat[:nbody]
        body_xpos_full_ref_t = torch.tensor(body_xpos_full_np, device=config.device)
        body_xquat_ref_t = torch.tensor(body_xquat_full_np, device=config.device)
        loguru.logger.info(
            "E035 precomputed body FK: xpos={}, xquat={}",
            tuple(body_xpos_full_ref_t.shape),
            tuple(body_xquat_ref_t.shape),
        )
        # Override body_xpos_ref with full version for local-frame
        body_xpos_ref = body_xpos_full_ref_t

    def _resize_contact_mask(mask_np: np.ndarray, target_len: int) -> np.ndarray:
        """Nearest-neighbor resize along time, preserving per-EEF columns."""
        if mask_np.shape[0] == target_len:
            return mask_np.astype(np.float32)
        if mask_np.shape[0] <= 0:
            raise ValueError("contact mask has zero frames")
        idx = np.round(np.linspace(0, mask_np.shape[0] - 1, target_len)).astype(
            np.int64
        )
        return mask_np[idx].astype(np.float32)

    def _apply_mask_ramp(mask_np: np.ndarray, ramp_frames: int) -> np.ndarray:
        """E155: linear ramp-down at the last contact boundary, centered on the edge.

        The midpoint of the ramp (mask=0.5) is placed at the original boundary B
        (last frame where mask > 0.5 before ramp).

        Args:
            mask_np: (T, n_eef) float32, after union both columns are identical.
            ramp_frames: total ramp width in sim frames.
        Returns:
            (T, n_eef) float32 with ramp applied.
        """
        mask_np = mask_np.copy()
        # Find last frame with any contact (boundary B)
        any_active = mask_np.max(axis=1)
        active_idx = np.where(any_active > 0.5)[0]
        if len(active_idx) == 0:
            return mask_np
        B = int(active_idx[-1])
        half = ramp_frames // 2
        ramp_start = max(B - half, 0)
        ramp_end = min(B + half + 1, mask_np.shape[0])
        ramp_len = ramp_end - ramp_start
        if ramp_len <= 1:
            return mask_np
        ramp_values = np.linspace(1.0, 0.0, ramp_len).astype(np.float32)
        # Ensure before ramp stays 1, after ramp stays 0
        mask_np[:ramp_start] = np.where(
            mask_np[:ramp_start] > 0.5, 1.0, mask_np[:ramp_start]
        )
        mask_np[ramp_start:ramp_end] = ramp_values[:, None]
        mask_np[ramp_end:] = 0.0
        return mask_np

    # E039b/E078: precompute per-EEF contact mask.
    # E214: the contact-mask precompute historically only ran when the HDMI
    # coarse term was active (contact_hdmi_gain>0), but surface_band's
    # contact-mask gate reads the same mask. The E214 "surface_band only"
    # ablation sets contact_hdmi_gain=0 while keeping surface_band on, so also
    # trigger the precompute when surface_band needs a contact-mask gate. Guarded
    # so every gain>0 run (baseline / A1 / A3 / A4) is byte-identical.
    _surface_band_needs_mask = (
        (config.surface_band_rew_scale > 0.0 or config.surface_band_penalty_scale > 0.0)
        and config.surface_band_gate_source in (
            "contact_mask", "contact_mask_strict_current", "contact_mask_time_window",
        )
    )
    if (config.contact_hdmi_gain > 0.0 or _surface_band_needs_mask) and config.hand_approach_body_ids:
        obj_body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "object")
        if config.contact_hdmi_mask_source == "core4d_3cm":
            if not config.contact_hdmi_mask_path:
                raise ValueError(
                    "contact_hdmi_mask_source=core4d_3cm requires contact_hdmi_mask_path"
                )
            mask_data = np.load(config.contact_hdmi_mask_path, allow_pickle=True)
            target_len = qpos_ref.shape[0]
            axis = config.contact_hdmi_mask_time_axis
            if axis == "auto":
                if (
                    "spider_contact_mask_3cm" in mask_data
                    and mask_data["spider_contact_mask_3cm"].shape[0] == target_len
                ):
                    axis = "spider"
                elif (
                    "eval_contact_mask_3cm" in mask_data
                    and mask_data["eval_contact_mask_3cm"].shape[0] == target_len
                ):
                    axis = "eval"
                else:
                    # Prefer eval for MJWP because load_data normally upsamples 30Hz refs to 50Hz.
                    axis = "eval" if "eval_contact_mask_3cm" in mask_data else "spider"
            key = f"{axis}_contact_mask_3cm"
            if key not in mask_data:
                raise KeyError(f"{config.contact_hdmi_mask_path} missing {key}")
            raw_mask = mask_data[key]
            person_idx = int(config.contact_hdmi_mask_person_idx)
            if raw_mask.ndim != 3 or raw_mask.shape[1] <= person_idx:
                raise ValueError(
                    f"{key} expected shape (T, person, hand), got {raw_mask.shape}, person_idx={person_idx}"
                )
            per_eef_mask_np = raw_mask[:, person_idx, :].astype(np.float32)
            if per_eef_mask_np.shape[1] != len(config.hand_approach_body_ids):
                raise ValueError(
                    f"{key} hand dim {per_eef_mask_np.shape[1]} != hand bodies {len(config.hand_approach_body_ids)}"
                )
            original_len = per_eef_mask_np.shape[0]
            per_eef_mask_np = _resize_contact_mask(per_eef_mask_np, target_len)
            # E155: carry union (max L/R) + boundary ramp
            if config.contact_hdmi_mask_carry_union:
                union = per_eef_mask_np.max(axis=1, keepdims=True)
                per_eef_mask_np = np.broadcast_to(union, per_eef_mask_np.shape).copy()
            if config.contact_hdmi_mask_ramp_frames > 0:
                per_eef_mask_np = _apply_mask_ramp(
                    per_eef_mask_np, config.contact_hdmi_mask_ramp_frames
                )
            approach_mask_t = torch.tensor(per_eef_mask_np, device=config.device)
            active_pct = per_eef_mask_np.mean(axis=0) * 100
            loguru.logger.info(
                "E078 core4d_3cm per-EEF mask: source={} key={} person_idx={} len {}→{} active L/R={:.1f}%/{:.1f}%",
                config.contact_hdmi_mask_path,
                key,
                person_idx,
                original_len,
                target_len,
                active_pct[0],
                active_pct[1],
            )
        elif obj_body_id != -1 and config.hand_approach_obj_half_extents:
            half_ext = np.array(config.hand_approach_obj_half_extents)
            T_mask = qpos_ref.shape[0]
            threshold = config.contact_hdmi_threshold
            per_eef_mask_np = np.zeros(
                (T_mask, len(config.hand_approach_body_ids)), dtype=np.float32
            )
            for t in range(T_mask):
                mj_data_ref.qpos[:] = qpos_ref[t].detach().cpu().numpy()
                mujoco.mj_forward(mj_model, mj_data_ref)
                obj_pos = mj_data_ref.xpos[obj_body_id]
                obj_mat = mj_data_ref.xmat[obj_body_id].reshape(3, 3)
                for ei, hid in enumerate(config.hand_approach_body_ids):
                    hand_pos = mj_data_ref.xpos[hid]
                    # Correct rotated SDF: transform to object local frame
                    local = obj_mat.T @ (hand_pos - obj_pos)
                    clamped = np.clip(local, -half_ext, half_ext)
                    surf_dist = np.linalg.norm(local - clamped)
                    if surf_dist < threshold:
                        per_eef_mask_np[t, ei] = 1.0
            # Override approach_mask with corrected version
            approach_mask_t = torch.tensor(per_eef_mask_np, device=config.device)
            active_pct = per_eef_mask_np.mean(axis=0) * 100
            loguru.logger.info(
                "E039b rotated-SDF per-EEF mask: L/R={:.1f}%/{:.1f}% frames active (threshold={:.2f}m)",
                active_pct[0],
                active_pct[1],
                threshold,
            )

    # E040: precompute per-frame contact target from ref FK (hand pos in object local frame)
    contact_target_per_frame = None
    if config.contact_hdmi_dynamic_target and config.hand_approach_body_ids:
        obj_body_id_e040 = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_BODY, "object"
        )
        if obj_body_id_e040 != -1:
            T = qpos_ref.shape[0]
            n_eef = len(config.hand_approach_body_ids)
            if config.contact_hdmi_target_source == "external":
                if not config.contact_hdmi_target_path:
                    raise ValueError(
                        "contact_hdmi_target_source=external requires contact_hdmi_target_path"
                    )
                target_path = Path(config.contact_hdmi_target_path)
                if not target_path.is_absolute():
                    target_path = Path.cwd() / target_path
                target_data = np.load(target_path, allow_pickle=True)
                axis = config.contact_hdmi_target_time_axis
                if axis == "auto":
                    if (
                        "spider_contact_target_object_local" in target_data
                        and target_data["spider_contact_target_object_local"].shape[0]
                        == T
                    ):
                        axis = "spider"
                    elif (
                        "eval_contact_target_object_local" in target_data
                        and target_data["eval_contact_target_object_local"].shape[0]
                        == T
                    ):
                        axis = "eval"
                    else:
                        axis = (
                            "eval"
                            if "eval_contact_target_object_local" in target_data
                            else "spider"
                        )
                key = f"{axis}_contact_target_object_local"
                if key not in target_data:
                    raise KeyError(f"{target_path} missing {key}")
                target_np = target_data[key].astype(np.float32)
                if target_np.ndim != 3 or target_np.shape[1:] != (n_eef, 3):
                    raise ValueError(
                        f"{key} expected shape (T,{n_eef},3), got {target_np.shape}"
                    )
                original_len = target_np.shape[0]
                target_np = _resize_contact_mask(target_np, T)
                loguru.logger.info(
                    "E085 external contact target: source={} key={} len {}→{}",
                    target_path,
                    key,
                    original_len,
                    T,
                )
            elif config.contact_hdmi_target_source == "ref_fk":
                target_np = np.zeros((T, n_eef, 3), dtype=np.float32)
                eef_offset_np = np.asarray(
                    config.contact_hdmi_eef_offset, dtype=np.float32
                )
                if config.contact_hdmi_target_uses_eef_offset:
                    from scipy.spatial.transform import Rotation as _R_e073
                for t in range(T):
                    mj_data_ref.qpos[:] = qpos_ref[t].detach().cpu().numpy()
                    mujoco.mj_forward(mj_model, mj_data_ref)
                    obj_pos = mj_data_ref.xpos[obj_body_id_e040]
                    obj_mat = mj_data_ref.xmat[obj_body_id_e040].reshape(3, 3)
                    for ei, hid in enumerate(config.hand_approach_body_ids):
                        hand_pos = mj_data_ref.xpos[hid]
                        # Hand/contact point in object local frame. Historical
                        # dynamic targets used the wrist body origin; E073 can
                        # switch to the same wrist+eef_offset point used by reward.
                        if config.contact_hdmi_target_uses_eef_offset:
                            hand_quat = mj_data_ref.xquat[hid]
                            hand_rot = _R_e073.from_quat(
                                [
                                    hand_quat[1],
                                    hand_quat[2],
                                    hand_quat[3],
                                    hand_quat[0],
                                ]
                            )
                            contact_delta = hand_rot.apply(eef_offset_np)
                            hand_pos = hand_pos + contact_delta
                        target_np[t, ei] = obj_mat.T @ (hand_pos - obj_pos)
            else:
                raise ValueError(
                    f"Unsupported contact_hdmi_target_source={config.contact_hdmi_target_source}"
                )
            contact_target_per_frame = torch.tensor(target_np, device=config.device)
            loguru.logger.info(
                "E040 dynamic target: shape={}, source={}, uses_eef_offset={}",
                tuple(contact_target_per_frame.shape),
                config.contact_hdmi_target_source,
                config.contact_hdmi_target_uses_eef_offset,
            )

    if approach_mask_t is not None:
        if body_xquat_ref_t is not None:
            if contact_target_per_frame is not None:
                ref_data = (
                    qpos_ref,
                    qvel_ref,
                    ctrl_ref,
                    contact,
                    contact_pos,
                    body_xpos_ref,
                    approach_mask_t,
                    body_xquat_ref_t,
                    contact_target_per_frame,
                )
            else:
                ref_data = (
                    qpos_ref,
                    qvel_ref,
                    ctrl_ref,
                    contact,
                    contact_pos,
                    body_xpos_ref,
                    approach_mask_t,
                    body_xquat_ref_t,
                )
        else:
            ref_data = (
                qpos_ref,
                qvel_ref,
                ctrl_ref,
                contact,
                contact_pos,
                body_xpos_ref,
                approach_mask_t,
            )
    else:
        ref_data = (qpos_ref, qvel_ref, ctrl_ref, contact, contact_pos, body_xpos_ref)
    mj_data.qpos[:] = qpos_ref[0].detach().cpu().numpy()
    mj_data.qvel[:] = qvel_ref[0].detach().cpu().numpy()
    mj_data.ctrl[:] = ctrl_ref[0].detach().cpu().numpy()
    mujoco.mj_step(mj_model, mj_data)
    mj_data.time = 0.0
    _assert_object_actuator_gains_zero(env, config, "start")
    images = []
    object_trace_site_ids = []
    robot_trace_site_ids = []
    for sid in range(mj_model.nsite):
        name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_SITE, sid)
        if name is not None:
            if name.startswith("trace"):
                if "object" in name:
                    object_trace_site_ids.append(sid)
                else:
                    robot_trace_site_ids.append(sid)
    config.trace_site_ids = object_trace_site_ids + robot_trace_site_ids
    contact_guidance_enabled = (
        config.contact_guidance and len(config.object_actuator_ids) > 0
    )
    if config.contact_guidance and not contact_guidance_enabled:
        loguru.logger.warning(
            "contact_guidance is enabled but no object actuators were resolved."
        )
    contact_offset = 0
    if contact_guidance_enabled:
        config.contact_len = int(
            min(contact.shape[1], contact_pos.shape[1], len(config.contact_order))
        )
        if (
            config.contact_len != len(config.contact_order)
            or config.contact_len != contact.shape[1]
        ):
            loguru.logger.warning(
                "Contact length mismatch (mask={}, pos={}, expected={}); truncating to {}.",
                contact.shape[1],
                contact_pos.shape[1],
                len(config.contact_order),
                config.contact_len,
            )
        config.contact_order = config.contact_order[: config.contact_len]
        config.hand_contact_site_ids = config.hand_contact_site_ids[
            : config.contact_len
        ]
        contact_offset = max(contact.shape[1] - config.contact_len, 0)

    # setup env params
    env_params_list = []
    if config.num_dr == 0:
        xy_offset_list = [0.0]
        pair_margin_list = [0.0]
    else:
        xy_offset_list = np.linspace(
            config.xy_offset_range[0], config.xy_offset_range[1], config.num_dr
        )
        pair_margin_list = np.linspace(
            config.pair_margin_range[0], config.pair_margin_range[1], config.num_dr
        )
    kp_schedule = []
    kd_schedule = []
    if contact_guidance_enabled and config.max_num_iterations > 0:
        actuator_names = config.object_actuator_names
        if not actuator_names:
            actuator_names = [
                mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, int(aid))
                for aid in config.object_actuator_ids
            ]
        base_kp = np.array(
            [
                (
                    config.init_rot_actuator_gain
                    if ("_rot_" in (name or ""))
                    else config.init_pos_actuator_gain
                )
                for name in actuator_names
            ],
            dtype=np.float32,
        )
        base_kd = np.array(
            [
                (
                    config.init_rot_actuator_bias
                    if ("_rot_" in (name or ""))
                    else config.init_pos_actuator_bias
                )
                for name in actuator_names
            ],
            dtype=np.float32,
        )
        for i in range(config.max_num_iterations):
            decay = float(config.guidance_decay_ratio) ** i
            kp_i = base_kp * decay
            kd_i = base_kd * decay
            if i == config.max_num_iterations - 1:
                if config.residual_gain_ratio > 0:
                    # Keep a fraction of the decayed gains on the last iteration
                    kp_i = base_kp * config.residual_gain_ratio
                    kd_i = base_kd * config.residual_gain_ratio
                else:
                    kp_i = np.zeros_like(base_kp, dtype=np.float32)
                    kd_i = np.zeros_like(base_kd, dtype=np.float32)
            kp_schedule.append(kp_i)
            kd_schedule.append(kd_i)

    for i in range(config.max_num_iterations):
        env_params = []
        for j in range(config.num_dr):
            params = {
                "xy_offset": xy_offset_list[j],
                "pair_margin": pair_margin_list[j],
            }
            if contact_guidance_enabled and kp_schedule:
                params["kp"] = kp_schedule[i]
                params["kd"] = kd_schedule[i]
            env_params.append(params)
        env_params_list.append(env_params)
    config.env_params_list = env_params_list
    _save_config_yaml(config)

    # setup viewer and renderer
    run_viewer = setup_viewer(config, mj_model, mj_data)
    renderer = setup_renderer(config, mj_model)

    # ─── SBTO mode ──────────────────────────────────────────────────────────
    if config.use_sbto:
        info_list = run_sbto(
            config,
            env,
            ref_data,
            mj_model,
            mj_data,
            mj_data_ref,
            qpos_ref,
            qvel_ref,
            ctrl_ref,
            renderer,
            images,
        )
        # Jump directly to save section (shared with MPC)
    else:
        # ─── Standard MPC mode ──────────────────────────────────────────────

        # setup optimizer
        rollout = make_rollout_fn(
            step_env,
            save_state,
            load_state,
            get_reward,
            get_terminal_reward,
            get_terminate,
            get_trace,
            save_env_params,
            load_env_params,
            copy_sample_state,
            get_qpos,
            get_geometry_state,
        )
        optimize_once = make_optimize_once_fn(rollout)
        optimize = make_optimize_fn(optimize_once)
        base_noise_scale = config.noise_scale.clone()
        gibbs_enabled = config.gibbs_sampling and config.embodiment_type in [
            "bimanual",
            "dual_humanoid_object",
        ]
        if config.gibbs_sampling and not gibbs_enabled:
            loguru.logger.warning(
                "gibbs_sampling is enabled but embodiment_type is {}, disabling.",
                config.embodiment_type,
            )
        if gibbs_enabled:
            if config.embodiment_type == "bimanual":
                right_ids, left_ids = _get_bimanual_hand_indices(config)
                right_only_zero = left_ids
                left_only_zero = right_ids
            elif config.embodiment_type == "dual_humanoid_object":
                # Split by robot: R1 = first half of nu, R2 = second half
                half_nu = config.nu // 2
                robot1_ids = list(range(0, half_nu))
                robot2_ids = list(range(half_nu, config.nu))
                right_only_zero = robot2_ids  # Zero R2 noise → optimize R1
                left_only_zero = robot1_ids  # Zero R1 noise → optimize R2

        # initial controls
        ctrls = ctrl_ref[: config.horizon_steps]
        # buffers for saving info and trajectory
        info_list = []

        # run viewer + control loop
        t_start = time.perf_counter()
        with run_viewer() as viewer:
            while viewer.is_running():
                t0 = time.perf_counter()

                # optimize using future reference window at control-rate (+1 lookahead)
                sim_step = int(np.round(mj_data.time / config.sim_dt))
                ref_slice = get_slice(
                    ref_data, sim_step + 1, sim_step + config.horizon_steps + 1
                )
                config._query_tape_current_sim_step = sim_step
                ctrls_for_opt = ctrls
                # E027d: always reset object actuator ctrl to ref qpos for contact_guidance
                if contact_guidance_enabled and config.object_actuator_ids:
                    ref_ctrl_window = ctrl_ref[sim_step : sim_step + ctrls.shape[0]]
                    if ref_ctrl_window.shape[0] == ctrls.shape[0]:
                        ctrls_for_opt = ctrls_for_opt.clone()
                        obj_ids = config.object_actuator_ids
                        ctrls_for_opt[:, obj_ids] = ref_ctrl_window[:, obj_ids]
                if contact_guidance_enabled and config.contact_len > 0:
                    contact_mask_step = contact[sim_step][
                        contact_offset : contact_offset + config.contact_len
                    ]
                    contact_pos_ref_step = contact_pos[sim_step]
                    site_xpos = wp.to_torch(env.data_wp.site_xpos)[0]

                    right_delta = compute_contact_point_delta(
                        contact_mask_step,
                        contact_pos_ref_step,
                        site_xpos,
                        config.hand_contact_site_ids,
                        config.right_contact_indices,
                    )
                    left_delta = compute_contact_point_delta(
                        contact_mask_step,
                        contact_pos_ref_step,
                        site_xpos,
                        config.hand_contact_site_ids,
                        config.left_contact_indices,
                    )
                    if (
                        right_delta is not None
                        and config.right_pos_ctrl_ids
                        and sim_step + ctrls.shape[0] <= ctrl_ref.shape[0]
                    ):
                        ctrls_for_opt = ctrls_for_opt.clone()
                        ref_ctrl_slice = ctrl_ref[sim_step : sim_step + ctrls.shape[0]]
                        ctrls_for_opt[:, config.right_pos_ctrl_ids] = ref_ctrl_slice[
                            :, config.right_pos_ctrl_ids
                        ] + torch.clip(right_delta, -0.01, 0.01)
                    if (
                        left_delta is not None
                        and config.left_pos_ctrl_ids
                        and sim_step + ctrls.shape[0] <= ctrl_ref.shape[0]
                    ):
                        if ctrls_for_opt is ctrls:
                            ctrls_for_opt = ctrls_for_opt.clone()
                            ref_ctrl_slice = ctrl_ref[
                                sim_step : sim_step + ctrls.shape[0]
                            ]
                        ctrls_for_opt[:, config.left_pos_ctrl_ids] = ref_ctrl_slice[
                            :, config.left_pos_ctrl_ids
                        ] + torch.clip(left_delta, -0.01, 0.01)
                if gibbs_enabled:
                    config.noise_scale = _apply_noise_mask(
                        base_noise_scale, right_only_zero
                    )
                    ctrls, infos = optimize(config, env, ctrls_for_opt, ref_slice)
                    config.noise_scale = _apply_noise_mask(
                        base_noise_scale, left_only_zero
                    )
                    ctrls, infos = optimize(config, env, ctrls, ref_slice)
                    config.noise_scale = base_noise_scale
                else:
                    config.noise_scale = base_noise_scale
                    # Warmup: skip CEM for first N ctrl steps, use ref ctrl directly
                    warmup_ctrl_steps = (
                        int(config.warmup_steps / config.ctrl_dt)
                        if config.warmup_steps > 0
                        else 0
                    )
                    ctrl_step_idx = (
                        sim_step // config.ctrl_steps_int
                        if hasattr(config, "ctrl_steps_int")
                        else sim_step
                        // max(1, int(np.round(config.ctrl_dt / config.sim_dt)))
                    )
                    if warmup_ctrl_steps > 0 and ctrl_step_idx < warmup_ctrl_steps:
                        # During warmup: use ref ctrl, no CEM
                        ctrls = ctrl_ref[sim_step : sim_step + config.horizon_steps]
                        if ctrls.shape[0] < config.horizon_steps:
                            ctrls = torch.cat(
                                [
                                    ctrls,
                                    ctrls[-1:].repeat(
                                        config.horizon_steps - ctrls.shape[0], 1
                                    ),
                                ],
                                dim=0,
                            )
                        infos = {"opt_steps": np.array([0]), "improvement": 0.0}
                    else:
                        ctrls, infos = optimize(config, env, ctrls_for_opt, ref_slice)

                # Compute trace_ref from reference qpos over the horizon
                if len(config.trace_site_ids) > 0:
                    trace_ref = []
                    qpos_ref_horizon = ref_slice[0]
                    for h in range(config.horizon_steps):
                        mj_data_ref.qpos[:] = qpos_ref_horizon[h].detach().cpu().numpy()
                        mujoco.mj_kinematics(mj_model, mj_data_ref)
                        site_xpos = np.array(
                            [
                                mj_data_ref.site_xpos[sid]
                                for sid in config.trace_site_ids
                            ]
                        )
                        trace_ref.append(site_xpos)
                    # (H, K, 3) -> (1, 1, H, K, 3) to match trace_sample shape
                    trace_ref_np = np.stack(trace_ref, axis=0)[None, None, :, :, :]
                    infos["trace_ref"] = trace_ref_np

                # E027d2: restore object actuator gains before commit step
                # After optimize(), the last CEM iteration may have set gains to 0
                # (residual_gain_ratio=0). Restore initial gains so PD actuator
                # drives the object during the commit phase.
                if contact_guidance_enabled and config.object_actuator_ids:
                    _commit_params = {
                        "kp": np.array(
                            [
                                (
                                    config.init_rot_actuator_gain
                                    if ("_rot_" in (n or ""))
                                    else config.init_pos_actuator_gain
                                )
                                for n in (config.object_actuator_names or [])
                            ],
                            dtype=np.float32,
                        ),
                        "kd": np.array(
                            [
                                (
                                    config.init_rot_actuator_bias
                                    if ("_rot_" in (n or ""))
                                    else config.init_pos_actuator_bias
                                )
                                for n in (config.object_actuator_names or [])
                            ],
                            dtype=np.float32,
                        ),
                    }
                    load_env_params(config, env, _commit_params)

                # step environment for ctrl_steps
                step_info = {"qpos": [], "qvel": [], "time": [], "ctrl": []}
                if config.support_proxy_enabled:
                    step_info.update(
                        {
                            "support_proxy_force": [],
                            "support_proxy_torque": [],
                            "support_proxy_pos": [],
                            "support_proxy_vel": [],
                            "support_point_pos": [],
                            "support_point_vel": [],
                            "support_proxy_ref_idx": [],
                        }
                    )
                partner_force_enabled = (
                    config.partner_force_scale > 0
                    or config.partner_force_spring_kp > 0
                    or config.partner_force_spring_kp_rot > 0
                )
                if partner_force_enabled:
                    step_info.update(
                        {
                            "partner_force_force": [],
                            "partner_force_torque": [],
                        }
                    )
                for i in range(config.ctrl_steps):
                    ctrl_step = ctrls[i]

                    # option 1: use mujoco step
                    # mj_data.ctrl[:] = ctrls[i].detach().cpu().numpy()
                    # mujoco.mj_step(mj_model, mj_data)
                    # option 2: use warp step
                    step_env(config, env, ctrl_step)
                    mj_data.qpos[:] = get_qpos(config, env)[0].detach().cpu().numpy()
                    mj_data.qvel[:] = get_qvel(config, env)[0].detach().cpu().numpy()
                    mj_data.ctrl[:] = ctrl_step.detach().cpu().numpy()
                    mj_data.time += config.sim_dt
                    if config.save_video and renderer is not None:
                        if i % int(np.round(config.render_dt / config.sim_dt)) == 0:
                            mj_data_ref.qpos[:] = (
                                qpos_ref[sim_step + i].detach().cpu().numpy()
                            )
                            image = render_image(
                                config, renderer, mj_model, mj_data, mj_data_ref
                            )
                            images.append(image)
                    if "rerun" in config.viewer or "viser" in config.viewer:
                        mj_data_ref.qpos[:] = (
                            qpos_ref[sim_step + i].detach().cpu().numpy()
                        )
                        mujoco.mj_kinematics(mj_model, mj_data_ref)
                        log_frame(
                            mj_data,
                            sim_time=mj_data.time,
                            viewer_body_entity_and_ids=config.viewer_body_entity_and_ids,
                            data_ref=mj_data_ref,
                        )
                    step_info["qpos"].append(mj_data.qpos.copy())
                    step_info["qvel"].append(mj_data.qvel.copy())
                    step_info["time"].append(mj_data.time)
                    step_info["ctrl"].append(mj_data.ctrl.copy())
                    if config.support_proxy_enabled:
                        support_state = get_support_proxy_state(config, env)
                        for key, value in support_state.items():
                            step_info[key].append(value.copy())
                    if partner_force_enabled:
                        partner_state = get_partner_force_state(config, env)
                        for key, value in partner_state.items():
                            step_info[key].append(value.copy())
                for k in step_info:
                    step_info[k] = np.stack(step_info[k], axis=0)
                infos.update(step_info)
                # sync env state
                sync_env(config, env, mj_data)

                # receding horizon update
                sim_step = int(np.round(mj_data.time / config.sim_dt))
                prev_ctrl = ctrls[config.ctrl_steps :]
                new_ctrl = ctrl_ref[
                    sim_step + prev_ctrl.shape[0] : sim_step
                    + prev_ctrl.shape[0]
                    + config.ctrl_steps
                ]
                ctrls = torch.cat([prev_ctrl, new_ctrl], dim=0)

                # sync viewer state and render
                mj_data.qpos[:] = get_qpos(config, env)[0].detach().cpu().numpy()
                mj_data.qvel[:] = get_qvel(config, env)[0].detach().cpu().numpy()
                mj_data_ref.qpos[:] = qpos_ref[sim_step].detach().cpu().numpy()
                update_viewer(config, viewer, mj_model, mj_data, mj_data_ref, infos)

                # progress
                t1 = time.perf_counter()
                rtr = config.ctrl_dt / (t1 - t0)
                print(
                    f"Realtime rate: {rtr:.2f}, plan time: {t1 - t0:.4f}s, sim_steps: {sim_step}/{config.max_sim_steps}, opt_steps: {infos['opt_steps'][0]}",
                    end="\r",
                )

                # record info/trajectory at control tick
                # rule out "trace"
                info_list.append(
                    {k: v for k, v in infos.items() if k != "trace_sample"}
                )

                stop_after_chunks = int(config.query_tape_stop_after_chunks)
                if stop_after_chunks < 0:
                    raise ValueError(
                        "query_tape_stop_after_chunks must be non-negative"
                    )
                if (
                    config.query_tape_enabled
                    and stop_after_chunks > 0
                    and cem_query_tape_chunk_count(config) >= stop_after_chunks
                ):
                    break

                if sim_step >= config.max_sim_steps:
                    break

            t_end = time.perf_counter()
            print(f"Total time: {t_end - t_start:.4f}s")

    # save retargeted trajectory
    if config.save_info and len(info_list) > 0:
        info_aggregated = _aggregate_info_list(info_list)
        np.savez(
            f"{config.output_dir}/trajectory_mjwp{'_act' if config.contact_guidance else ''}.npz",
            **info_aggregated,
        )
        loguru.logger.info(
            f"Saved info to {config.output_dir}/trajectory_mjwp{'_act' if config.contact_guidance else ''}.npz"
        )

    # save video
    if config.save_video and len(images) > 0:
        if config.video_output_path:
            video_path = config.video_output_path
        else:
            video_path = f"{config.output_dir}/visualization_mjwp{'_act' if config.contact_guidance else ''}.mp4"
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        imageio.mimsave(
            video_path,
            images,
            fps=int(1 / config.render_dt),
        )
        loguru.logger.info(f"Saved video to {video_path}")

    errors = None
    if info_list:
        qpos_traj = np.concatenate([info["qpos"] for info in info_list], axis=0)
        qpos_ref_np = qpos_ref[: qpos_traj.shape[0]].detach().cpu().numpy()
        data_type = "mjwp_act" if config.contact_guidance else "mjwp"
        errors = compute_object_tracking_error(
            qpos_traj, qpos_ref_np, config.embodiment_type, data_type
        )
        loguru.logger.info(
            "Final object tracking error: pos={:.4f}, quat={:.4f}",
            errors["obj_pos_err"],
            errors["obj_quat_err"],
        )

    _assert_object_actuator_gains_zero(env, config, "end")

    if "viser" in config.viewer and config.wait_on_finish:
        loguru.logger.info(
            "Optimization complete! Keeping Viser server alive. Press Ctrl+C to exit."
        )
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass

    return errors


@hydra.main(version_base=None, config_path="config", config_name="default")
def run_main(cfg: DictConfig) -> None:
    """Entry point for Hydra configuration runner."""
    # Convert DictConfig to Config dataclass, handling special fields
    config_dict = dict(cfg)

    # Optionally load a saved config YAML and merge; CLI overrides take priority.
    load_config_path = config_dict.get("load_config_path", "")
    if load_config_path:
        loaded_config = load_config_yaml(load_config_path)
        cli_overrides = _extract_cli_overrides(cfg)
        config_dict = {**loaded_config, **cli_overrides}
    else:
        config_dict = filter_config_fields(config_dict)

    # Handle special conversions
    if "noise_scale" in config_dict and config_dict["noise_scale"] is None:
        config_dict.pop("noise_scale")  # Let the default factory handle it

    # Convert lists to tuples where needed
    if "pair_margin_range" in config_dict:
        config_dict["pair_margin_range"] = tuple(config_dict["pair_margin_range"])
    if "xy_offset_range" in config_dict:
        config_dict["xy_offset_range"] = tuple(config_dict["xy_offset_range"])

    config = Config(**config_dict)
    main(config)


if __name__ == "__main__":
    run_main()
