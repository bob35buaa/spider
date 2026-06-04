#!/usr/bin/env python3
"""Holosoma replay startup probe without debug marker drawing.

This mirrors holosoma.replay but disables the debug drawing callback after the
environment is created. It is intended for E128 startup validation only; it does
not run PPO or create checkpoints.
"""

from __future__ import annotations

import os

import tyro

from holosoma.config_types.env import get_tyro_env_config
from holosoma.config_types.experiment import ExperimentConfig
from holosoma.config_values.experiment import AnnotatedExperimentConfig
from holosoma.utils.eval_utils import init_sim_imports
from holosoma.utils.helpers import get_class
from holosoma.utils.sim_utils import close_simulation_app
from holosoma.utils.tyro_utils import TYRO_CONIFG


def replay_probe(tyro_config: ExperimentConfig) -> None:
    simulation_app = init_sim_imports(tyro_config)

    import torch

    from holosoma.utils.common import seeding

    seeding(42, torch_deterministic=False)

    env_target = tyro_config.env_class
    tyro_env_config = get_tyro_env_config(tyro_config)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = get_class(env_target)(tyro_env_config, device=device)

    # Holosoma replay.py unconditionally calls _draw_debug_vis(), which can fail
    # in headless startup checks when MotionCommand visualization markers are
    # not created. The loader/startup contract does not require marker drawing.
    setattr(env, "_draw_debug_vis", lambda: None)

    motion_command = env.command_manager.get_state("motion_command")
    print(
        "E128_PROBE_LOADED "
        f"has_object={motion_command.motion.has_object} "
        f"has_partner={motion_command.motion.has_partner} "
        f"has_object_contact={motion_command.motion.has_object_contact} "
        f"time_step_total={motion_command.motion.time_step_total}",
        flush=True,
    )

    done = False
    max_steps = int(os.environ.get("E128_MAX_STEPS", "0") or "0")
    steps = 0
    while not done:
        env.simulator.sim.step()
        done = env.step_visualize_motion(None)  # type: ignore[attr-defined]
        steps += 1
        if max_steps and steps >= max_steps:
            break

    print(f"E128_PROBE_DONE steps={steps} done={done}", flush=True)
    close_simulation_app(simulation_app)


def main() -> None:
    tyro_cfg = tyro.cli(AnnotatedExperimentConfig, config=TYRO_CONIFG)
    replay_probe(tyro_cfg)


if __name__ == "__main__":
    main()
