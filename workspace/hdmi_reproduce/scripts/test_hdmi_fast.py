"""Minimal test: verify hdmi.py MuJoCo Warp GPU backend."""
import sys
import time

import torch
import warp as wp

# Minimal config via hydra
sys.argv = [
    "test",
    "task=move_suitcase",
    "+data_id=0",
    "viewer=none",
    "save_video=false",
    "max_sim_steps=20",
    "num_samples=64",
    "joint_noise_scale=0.2",
    "knot_dt=0.2",
    "ctrl_dt=0.04",
    "horizon=0.8",
]

from examples.run_hdmi import run_main

# Hijack main to run our quick test instead
from spider.config import Config, process_config
from spider.simulators.hdmi import (
    get_reference,
    get_reward,
    load_state,
    precompute_reward_reference,
    save_state,
    setup_env,
    step_env,
    sync_env,
)


def quick_test():
    """Quick functional + perf test."""
    from hydra import compose, initialize

    with initialize(version_base=None, config_path="examples/config"):
        cfg = compose(
            config_name="hdmi",
            overrides=[
                "task=move_suitcase",
                "+data_id=0",
                "viewer=none",
                "save_video=false",
                "max_sim_steps=20",
                "num_samples=64",
                "joint_noise_scale=0.2",
                "knot_dt=0.2",
                "ctrl_dt=0.04",
                "horizon=0.8",
            ],
        )

    config_dict = dict(cfg)
    if "noise_scale" in config_dict and config_dict["noise_scale"] is None:
        config_dict.pop("noise_scale")
    if "pair_margin_range" in config_dict:
        config_dict["pair_margin_range"] = tuple(config_dict["pair_margin_range"])
    if "xy_offset_range" in config_dict:
        config_dict["xy_offset_range"] = tuple(config_dict["xy_offset_range"])

    config = Config(**config_dict)

    # Setup
    print("Setting up env...")
    env = setup_env(config, None)
    if config.max_sim_steps == -1:
        config.max_sim_steps = env.max_episode_length
    config.nu = env.model_cpu.nu  # scene XML actuator count
    config = process_config(config)
    print(f"nu={config.nu}, num_worlds={env.num_worlds}, device={env.device}")
    print(f"  nq={env.model_cpu.nq}, nv={env.model_cpu.nv}, nbody={env.model_cpu.nbody}")

    # Get reference + precompute
    print("Getting reference...")
    qpos_ref, qvel_ref, ctrl_ref = get_reference(config, env)
    print(f"  ctrl_ref shape: {ctrl_ref.shape}, range: [{ctrl_ref.min():.3f}, {ctrl_ref.max():.3f}]")
    print("Precomputing reward reference...")
    precompute_reward_reference(config, env)
    ctrl_ref = ctrl_ref.to(config.device)

    print(f"Precomputed ref shapes: body_pos={env._precomputed_ref['body_pos'].shape}")
    print(f"  ref_step={env._ref_step}")

    # Test step + reward at initial state
    ctrl0 = ctrl_ref[0].unsqueeze(0).expand(env.num_worlds, -1)
    step_env(config, env, ctrl0)
    print(f"After step_env: ref_step={env._ref_step}")

    rew, info = get_reward(config, env, None)
    print(f"rew={rew.mean():.4f}, tracking={info['tracking'].mean():.4f}, "
          f"obj_track={info['object_tracking'].mean():.4f}")

    # Test save/load
    state = save_state(env)
    step_env(config, env, ctrl0)  # step again
    print(f"After 2nd step: ref_step={env._ref_step}")
    load_state(env, state)
    print(f"After load: ref_step={env._ref_step}")

    # Sync
    sync_env(config, env)
    print("sync OK")

    # Performance test: 10 steps of step_env + get_reward
    print("\nPerformance test: 10 iterations of step_env + get_reward...")
    wp.synchronize()
    t0 = time.perf_counter()
    for i in range(10):
        step_env(config, env, ctrl0)
        rew, info = get_reward(config, env, None)
    wp.synchronize()
    t1 = time.perf_counter()
    avg = (t1 - t0) / 10
    print(f"Total: {t1-t0:.3f}s, avg per step: {avg*1000:.1f}ms")
    print(f"Final ref_step={env._ref_step}")
    print(f"rew={rew.mean():.4f}, tracking={info['tracking'].mean():.4f}, "
          f"obj_track={info['object_tracking'].mean():.4f}")

    # Open-loop test: run ctrl_ref for 10 steps
    print("\nOpen-loop test (10 steps with ref ctrl)...")
    sync_env(config, env)
    env._ref_step = 0
    for i in range(10):
        ctrl_i = ctrl_ref[i].unsqueeze(0).expand(env.num_worlds, -1)
        step_env(config, env, ctrl_i)
        rew, info = get_reward(config, env, None)
        if i == 0 or i == 9:
            print(f"  step {i+1}: rew={rew.mean():.4f}, "
                  f"tracking={info['tracking'].mean():.4f}, "
                  f"obj_track={info['object_tracking'].mean():.4f}")

    print("\nALL PASSED")


if __name__ == "__main__":
    quick_test()
