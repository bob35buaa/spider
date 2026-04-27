"""Quick end-to-end MPC test bypassing hydra."""
import os
import sys
import time

os.environ["HYDRA_FULL_ERROR"] = "1"

from hydra import compose, initialize

with initialize(version_base=None, config_path="examples/config"):
    cfg = compose(
        config_name="hdmi",
        overrides=[
            "task=move_suitcase",
            "+data_id=1",
            "viewer=none",
            "save_video=false",
            "max_sim_steps=-1",
            "num_samples=1024",
            "joint_noise_scale=0.2",
            "knot_dt=0.2",
            "ctrl_dt=0.04",
            "horizon=0.8",
            "use_torch_compile=false",
            "max_num_iterations=32",
        ],
    )

from spider.config import Config, process_config

cd = dict(cfg)
if "noise_scale" in cd and cd["noise_scale"] is None:
    cd.pop("noise_scale")
if "pair_margin_range" in cd:
    cd["pair_margin_range"] = tuple(cd["pair_margin_range"])
if "xy_offset_range" in cd:
    cd["xy_offset_range"] = tuple(cd["xy_offset_range"])

config = Config(**cd)
config.output_dir = "workspace/hdmi_reproduce/results/R004"
os.makedirs(config.output_dir, exist_ok=True)

import numpy as np
import torch

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

print("Setting up env...", flush=True)
t_setup = time.perf_counter()
env = setup_env(config, None)
if config.max_sim_steps == -1:
    config.max_sim_steps = env.max_episode_length
config.nu = env.action_spec.shape[-1]
config = process_config(config)
# Override output_dir AFTER process_config (which sets it to HF data dir)
config.output_dir = "workspace/hdmi_reproduce/results/R004"
os.makedirs(config.output_dir, exist_ok=True)
print(f"Setup: {time.perf_counter()-t_setup:.1f}s, output={config.output_dir}", flush=True)

# Reference + precompute
print("Getting reference + precompute...", flush=True)
# Use mjlab scene model for qpos layout (matches HF format for rendering)
scene_xml = "example_datasets/processed/hdmi/unitree_g1/humanoid_object/move_suitcase/scene/mjlab scene.xml"
import mujoco as _mj
scene_model = _mj.MjModel.from_xml_path(scene_xml) if os.path.exists(scene_xml) else None
qpos_ref, qvel_ref, ctrl_ref = get_reference(config, env, scene_model=scene_model)
precompute_reward_reference(config, env)
qpos_ref = qpos_ref.to(config.device)
qvel_ref = qvel_ref.to(config.device)
ctrl_ref = ctrl_ref.to(config.device)

np.savez(
    f"{config.output_dir}/trajectory_kinematic.npz",
    qpos=qpos_ref.detach().cpu().numpy(),
    qvel=qvel_ref.detach().cpu().numpy(),
    ctrl=ctrl_ref.detach().cpu().numpy(),
)

# Placeholder ref_data
ref_data = (
    torch.zeros(
        config.max_sim_steps + config.horizon_steps + config.ctrl_steps,
        config.nu,
        device=config.device,
    ),
)

# Env params
env_params_list = []
for _ in range(config.max_num_iterations):
    env_params_list.append([{}] * config.num_dr)
config.env_params_list = env_params_list

# Optimizer
rollout = make_rollout_fn(
    step_env, save_state, load_state, get_reward, get_terminal_reward,
    get_terminate, get_trace, save_env_params, load_env_params, copy_sample_state,
)
optimize_once = make_optimize_once_fn(rollout)
optimize = make_optimize_fn(optimize_once)

ctrls = ctrl_ref[: config.horizon_steps]
info_list = []

# MPC loop
t_start = time.perf_counter()
sim_step = 0
while sim_step < config.max_sim_steps:
    t0 = time.perf_counter()

    ref_slice = get_slice(ref_data, sim_step + 1, sim_step + config.horizon_steps + 1)
    ctrls, infos = optimize(config, env, ctrls, ref_slice)
    infos["sim_step"] = sim_step

    # Real steps
    step_info = {"qpos": [], "qvel": [], "time": [], "ctrl": [], "ctrl_ref": []}
    for i in range(config.ctrl_steps):
        ctrl = ctrls[i].unsqueeze(0).repeat(int(config.num_samples), 1)
        step_env(config, env, ctrl)

        d0 = env._mj_datas[0]
        step_info["qpos"].append(d0.qpos.copy())
        step_info["qvel"].append(d0.qvel.copy())
        step_info["time"].append(d0.time)
        step_info["ctrl"].append(ctrls[i].detach().cpu().numpy())
        step_info["ctrl_ref"].append(ctrl_ref[sim_step].detach().cpu().numpy())
        sim_step += 1

    for k in step_info:
        step_info[k] = np.stack(step_info[k], axis=0)
    infos.update(step_info)

    env = sync_env(config, env)

    # Receding horizon
    prev_ctrl = ctrls[config.ctrl_steps:]
    new_ctrl = ctrl_ref[sim_step + prev_ctrl.shape[0]: sim_step + prev_ctrl.shape[0] + config.ctrl_steps]
    ctrls = torch.cat([prev_ctrl, new_ctrl], dim=0)

    t1 = time.perf_counter()
    rtr = config.ctrl_dt / (t1 - t0)
    opt_steps = infos["opt_steps"]
    if hasattr(opt_steps, "__len__"):
        opt_steps = opt_steps[0]
    rew_mean = float(infos["rew_mean"]) if np.ndim(infos["rew_mean"]) == 0 else float(infos["rew_mean"].mean())
    print(
        f"[{sim_step:3d}/{config.max_sim_steps}] "
        f"plan={t1-t0:.2f}s RTR={rtr:.2f} "
        f"rew={rew_mean:.3f} opt={opt_steps}",
        flush=True,
    )

    info_list.append({k: v for k, v in infos.items() if k != "trace_sample"})

t_end = time.perf_counter()
print(f"\nTotal: {t_end - t_start:.2f}s for {sim_step} steps", flush=True)

# Save trajectory
if len(info_list) > 0:
    info_agg = {}
    for k in info_list[0]:
        info_agg[k] = np.stack([info[k] for info in info_list], axis=0)
    np.savez(f"{config.output_dir}/trajectory_hdmi.npz", **info_agg)
    print(f"Saved to {config.output_dir}/trajectory_hdmi.npz", flush=True)

print("DONE", flush=True)
