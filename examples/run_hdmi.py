# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Retargeting for humand-object interaction with HDMI simulator.

Author: Chaoyi Pan
Date: 2025-10-18
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import hydra
import imageio
import loguru
import mujoco
import numpy as np
import torch
import warp as wp
from omegaconf import DictConfig

from spider.config import Config, process_config
from spider.interp import get_slice
from spider.optimizers.sampling import (
    make_optimize_fn,
    make_optimize_once_fn,
    make_rollout_fn,
)
from spider.simulators.hdmi import (
    copy_sample_state,
    get_reference,
    get_reward,
    get_terminal_reward,
    get_terminate,
    get_trace,
    load_env_params,
    load_state,
    precompute_reward_reference,
    save_env_params,
    save_state,
    setup_env,
    step_env,
    sync_env,
)
from spider.viewers import (
    log_frame,
    render_image,
    setup_renderer,
    setup_viewer,
    update_viewer,
)


def main(config: Config):
    """Run the SPIDER using HDMI backend."""
    # Setup env (ref_data set to None since environment has built-in reference)
    env = setup_env(config, None)
    if config.max_sim_steps == -1:
        config.max_sim_steps = env.max_episode_length
        loguru.logger.info(f"Max simulation steps set to {config.max_sim_steps}")

    # nu = number of actuators in scene XML (joint position targets)
    config.nu = env.model_cpu.nu

    # Process config, set defaults and derived fields
    user_output_dir = config.output_dir
    config = process_config(config)
    # Restore user output_dir (process_config overwrites it with HF data path)
    if user_output_dir and user_output_dir != "outputs/hdmi":
        config.output_dir = str(Path(user_output_dir).resolve())
        os.makedirs(config.output_dir, exist_ok=True)
        loguru.logger.info(f"Output dir: {config.output_dir}")

    # Create placeholder reference data for compatibility
    ref_data = (
        torch.zeros(
            config.max_sim_steps + config.horizon_steps + config.ctrl_steps,
            config.nu,
            device=config.device,
        ),
    )

    # Setup env params — with guidance gain decay if contact_guidance is on
    contact_guidance_enabled = getattr(config, "contact_guidance", False)
    env_params_list = []
    if contact_guidance_enabled:
        # Resolve object actuator IDs
        obj_act_ids = []
        for ai in range(env.model_cpu.nu):
            aname = mujoco.mj_id2name(
                env.model_cpu, mujoco.mjtObj.mjOBJ_ACTUATOR, ai
            )
            if aname and aname.startswith("object_"):
                obj_act_ids.append(ai)
        config.object_actuator_ids = obj_act_ids
        loguru.logger.info(f"Contact guidance: {len(obj_act_ids)} object actuators")

        # Zero noise on object actuator dims
        if hasattr(config, "noise_scale") and config.noise_scale is not None:
            for aid in obj_act_ids:
                config.noise_scale[:, :, aid] *= 0.0

        # Zero noise on wrist joints (not in HF 23-DOF action space)
        if hasattr(config, "noise_scale") and config.noise_scale is not None:
            wrist_keywords = ["wrist_roll", "wrist_pitch", "wrist_yaw"]
            for ai in range(env.model_cpu.nu):
                aname = mujoco.mj_id2name(
                    env.model_cpu, mujoco.mjtObj.mjOBJ_ACTUATOR, ai
                )
                if aname and any(w in aname for w in wrist_keywords):
                    config.noise_scale[:, :, ai] *= 0.0
                    loguru.logger.info(f"Zeroed noise for wrist actuator {ai}: {aname}")

        decay = getattr(config, "guidance_decay_ratio", 0.8)
        pos_kp = getattr(config, "init_pos_actuator_gain", 10.0)
        pos_kd = getattr(config, "init_pos_actuator_bias", 10.0)
        rot_kp = getattr(config, "init_rot_actuator_gain", 0.3)
        rot_kd = getattr(config, "init_rot_actuator_bias", 0.3)
        for i in range(config.max_num_iterations):
            scale = decay ** i
            if i == config.max_num_iterations - 1:
                scale = 0.0  # Release on last iteration
            env_params = [{
                "kp": pos_kp * scale,
                "kd": pos_kd * scale,
            }] * config.num_dr
            env_params_list.append(env_params)
    else:
        config.object_actuator_ids = []
        for _ in range(config.max_num_iterations):
            env_params = [{}] * config.num_dr
            env_params_list.append(env_params)
    config.env_params_list = env_params_list

    # Get reference data (states and controls)
    qpos_ref, qvel_ref, ctrl_ref = get_reference(config, env)
    # Precompute all reward reference data on GPU (eliminates CPU sync in MPC)
    precompute_reward_reference(config, env)
    # Move to device for MPC
    qpos_ref = qpos_ref.to(config.device)
    qvel_ref = qvel_ref.to(config.device)
    ctrl_ref = ctrl_ref.to(config.device)
    np.savez(
        f"{config.output_dir}/trajectory_kinematic.npz",
        qpos=qpos_ref.detach().cpu().numpy(),
        qvel=qvel_ref.detach().cpu().numpy(),
        ctrl=ctrl_ref.detach().cpu().numpy(),
    )

    # Setup mujoco model and data for rendering
    # Sim rendering uses env.model_cpu (may have contact guidance modifications)
    mj_model = env.model_cpu
    scene_xml_path = str(
        Path("example_datasets/processed/hdmi")
        / config.robot_type
        / config.embodiment_type
        / config.task
        / "scene"
        / "mjlab scene.xml"
    )
    config.model_path = scene_xml_path
    mj_data = mujoco.MjData(mj_model)

    # Ref rendering uses ORIGINAL scene XML (freejoint, nq=43) for accurate suitcase pose
    mj_model_ref = mujoco.MjModel.from_xml_path(scene_xml_path)
    mj_data_ref = mujoco.MjData(mj_model_ref)
    # Build qpos_ref in freejoint layout for ref rendering
    qpos_ref_render, _, _ = get_reference(config, env, scene_model=mj_model_ref)
    qpos_ref_render = qpos_ref_render.to(config.device)

    # Adjust tracking camera on BOTH models
    for m in [mj_model, mj_model_ref]:
        cam_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "robot/tracking")
        if cam_id >= 0:
            m.cam_pos[cam_id] = [3.0, 0.5, 1.0]
            m.cam_quat[cam_id] = [0.60, 0.60, 0.36, 0.36]

    # Initialize mj_data with current env state (read from Warp)
    qpos_wp = wp.to_torch(env.data_wp.qpos)[0].detach().cpu().numpy()
    qvel_wp = wp.to_torch(env.data_wp.qvel)[0].detach().cpu().numpy()
    mj_data.qpos[:] = qpos_wp
    mj_data.qvel[:] = qvel_wp
    mujoco.mj_step(mj_model, mj_data)
    mj_data.time = 0.0

    # Initialize reference mj_data (uses freejoint model)
    mj_data_ref.qpos[:] = qpos_ref_render[0].detach().cpu().numpy()
    mj_data_ref.qvel[:] = 0  # ref rendering doesn't need velocities
    mujoco.mj_step(mj_model_ref, mj_data_ref)

    # Setup for video rendering
    images = []

    # Setup viewer and renderer
    run_viewer = setup_viewer(config, mj_model, mj_data)
    renderer = setup_renderer(config, mj_model)
    renderer_ref = setup_renderer(config, mj_model_ref) if config.save_video else None

    # Setup optimizer
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

    # Initial controls - first horizon_steps from reference
    ctrls = ctrl_ref[: config.horizon_steps]

    # Buffers for saving info and trajectory
    info_list = []

    # Run viewer + control loop
    t_start = time.perf_counter()
    t_render_total = 0.0
    sim_step = 0
    from tqdm import tqdm
    pbar = tqdm(total=config.max_sim_steps, desc="HDMI retarget", unit="step")
    with run_viewer() as viewer:
        while viewer.is_running():
            t0 = time.perf_counter()

            # Optimize using future reference window at control-rate (+1 lookahead)
            ref_slice = get_slice(
                ref_data, sim_step + 1, sim_step + config.horizon_steps + 1
            )
            if config.max_num_iterations > 0:
                ctrls, infos = optimize(config, env, ctrls, ref_slice)
            else:
                infos = {"opt_steps": [0], "improvement": 0.0}
            infos["sim_step"] = sim_step

            # Step environment for ctrl_steps
            step_info = {"qpos": [], "qvel": [], "time": [], "ctrl": [], "ctrl_ref": []}
            for i in range(config.ctrl_steps):
                ctrl = ctrls[i]
                ctrl_repeat = ctrl.unsqueeze(0).repeat(
                    env.num_worlds, 1
                )
                step_env(config, env, ctrl_repeat)
                pbar.update(1)

                # Update mj_data with current state (read from Warp world 0)
                qpos_wp = wp.to_torch(env.data_wp.qpos)[0].detach().cpu().numpy()
                qvel_wp = wp.to_torch(env.data_wp.qvel)[0].detach().cpu().numpy()
                mj_data.qpos[:] = qpos_wp
                mj_data.qvel[:] = qvel_wp
                mj_data.time = (sim_step + 1) * config.sim_dt
                mujoco.mj_forward(mj_model, mj_data)

                # Render video if enabled
                should_render = (
                    config.save_video
                    and renderer is not None
                    and i % int(np.round(config.render_dt / config.sim_dt)) == 0
                )
                if should_render:
                    t_render_start = time.perf_counter()
                    ref_idx = min(sim_step, qpos_ref_render.shape[0] - 1)
                    mj_data_ref.qpos[:] = qpos_ref_render[ref_idx].detach().cpu().numpy()
                    mujoco.mj_forward(mj_model_ref, mj_data_ref)
                    # Render sim and ref side-by-side
                    import cv2
                    options = mujoco.MjvOption()
                    mujoco.mjv_defaultOption(options)
                    # Sim frame
                    mujoco.mj_forward(mj_model, mj_data)
                    try:
                        renderer.update_scene(mj_data, "front", options)
                    except Exception:
                        renderer.update_scene(mj_data, 0, options)
                    sim_img = renderer.render()
                    cv2.putText(sim_img, "sim", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 128, 128), 2)
                    # Ref frame (separate model/renderer)
                    try:
                        renderer_ref.update_scene(mj_data_ref, "front", options)
                    except Exception:
                        renderer_ref.update_scene(mj_data_ref, 0, options)
                    ref_img = renderer_ref.render()
                    cv2.putText(ref_img, "ref", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 128, 128), 2)
                    images.append(np.concatenate([ref_img, sim_img], axis=1))
                    t_render_total += time.perf_counter() - t_render_start
                if "rerun" in config.viewer or "viser" in config.viewer:
                    log_frame(
                        mj_data,
                        sim_time=mj_data.time,
                        viewer_body_entity_and_ids=config.viewer_body_entity_and_ids,
                    )

                # Record state info
                step_info["qpos"].append(mj_data.qpos.copy())
                step_info["qvel"].append(mj_data.qvel.copy())
                step_info["time"].append(mj_data.time)
                step_info["ctrl"].append(ctrl.detach().cpu().numpy())
                step_info["ctrl_ref"].append(ctrl_ref[sim_step].detach().cpu().numpy())

                sim_step += 1

            # Stack step info
            for k in step_info:
                step_info[k] = np.stack(step_info[k], axis=0)
            infos.update(step_info)

            # Sync env state (broadcast from first env to all)
            env = sync_env(config, env)

            # Receding horizon update
            prev_ctrl = ctrls[config.ctrl_steps :]
            new_ctrl = ctrl_ref[
                sim_step + prev_ctrl.shape[0] : sim_step
                + prev_ctrl.shape[0]
                + config.ctrl_steps
            ]
            ctrls = torch.cat([prev_ctrl, new_ctrl], dim=0)

            # Sync viewer state and render
            qpos_wp = wp.to_torch(env.data_wp.qpos)[0].detach().cpu().numpy()
            qvel_wp = wp.to_torch(env.data_wp.qvel)[0].detach().cpu().numpy()
            mj_data.qpos[:] = qpos_wp
            mj_data.qvel[:] = qvel_wp
            mujoco.mj_forward(mj_model, mj_data)
            # Update reference state (uses freejoint model)
            ref_idx = min(sim_step, qpos_ref_render.shape[0] - 1)
            mj_data_ref.qpos[:] = qpos_ref_render[ref_idx].detach().cpu().numpy()
            mujoco.mj_forward(mj_model_ref, mj_data_ref)
            update_viewer(config, viewer, mj_model, mj_data, mj_data_ref, infos)

            # Progress
            t1 = time.perf_counter()
            rtr = config.ctrl_dt / (t1 - t0)
            pbar.set_postfix(rtr=f"{rtr:.2f}", plan=f"{t1-t0:.2f}s", opt=infos['opt_steps'][0])

            # Record info/trajectory at control tick
            info_list.append({k: v for k, v in infos.items() if k != "trace_sample"})

            if sim_step >= config.max_sim_steps:
                break

        t_end = time.perf_counter()
        pbar.close()
        t_total = t_end - t_start
        render_pct = t_render_total / t_total * 100 if t_total > 0 else 0
        print(f"\nTotal time: {t_total:.1f}s (render: {t_render_total:.1f}s = {render_pct:.1f}%)")

    # Save retargeted trajectory
    if config.save_info and len(info_list) > 0:
        info_aggregated = {}
        for k in info_list[0]:
            info_aggregated[k] = np.stack([info[k] for info in info_list], axis=0)
        np.savez(f"{config.output_dir}/trajectory_hdmi.npz", **info_aggregated)
        loguru.logger.info(f"Saved info to {config.output_dir}/trajectory_hdmi.npz")

    # Save video
    if config.save_video and len(images) > 0:
        video_path = f"{config.output_dir}/visualization_hdmi.mp4"
        imageio.mimsave(
            video_path,
            images,
            fps=int(1 / config.render_dt),
        )
        loguru.logger.info(f"Saved video to {video_path}")

    return


@hydra.main(version_base=None, config_path="config", config_name="hdmi")
def run_main(cfg: DictConfig) -> None:
    """Main entry point for HDMI retargeting."""
    # Convert DictConfig to Config dataclass, handling special fields
    config_dict = dict(cfg)

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
