#!/usr/bin/env python3
"""Probe Holosoma motion_command.ref_object_contact under bounded env stepping."""

from __future__ import annotations

import json
import os
from typing import Any

import tyro

from holosoma.config_types.env import get_tyro_env_config
from holosoma.config_types.experiment import ExperimentConfig
from holosoma.config_values.experiment import AnnotatedExperimentConfig
from holosoma.utils.eval_utils import init_sim_imports
from holosoma.utils.helpers import get_class
from holosoma.utils.sim_utils import close_simulation_app
from holosoma.utils.tyro_utils import TYRO_CONIFG


def _contact_stats(contact: Any, prefix: str) -> dict[str, Any]:
    """Return small bool-contact stats for tensors shaped [..., 2]."""
    import torch

    tensor = contact.detach().bool().cpu() if isinstance(contact, torch.Tensor) else torch.as_tensor(contact).bool()
    if tensor.numel() == 0:
        return {
            f"{prefix}_shape": "x".join(str(x) for x in tensor.shape),
            f"{prefix}_samples": 0,
            f"{prefix}_contact_total": 0,
            f"{prefix}_left_active": 0,
            f"{prefix}_right_active": 0,
            f"{prefix}_both_active": 0,
            f"{prefix}_either_active": 0,
            f"{prefix}_positive": False,
            f"{prefix}_both_positive": False,
        }
    if tensor.shape[-1] != 2:
        raise ValueError(f"{prefix} contact tensor must have final dim 2, got {tuple(tensor.shape)}")
    flat = tensor.reshape(-1, 2)
    left = flat[:, 0]
    right = flat[:, 1]
    both = left & right
    either = left | right
    samples = int(flat.shape[0])
    return {
        f"{prefix}_shape": "x".join(str(x) for x in tensor.shape),
        f"{prefix}_samples": samples,
        f"{prefix}_contact_total": int(flat.sum().item()),
        f"{prefix}_left_active": int(left.sum().item()),
        f"{prefix}_right_active": int(right.sum().item()),
        f"{prefix}_both_active": int(both.sum().item()),
        f"{prefix}_either_active": int(either.sum().item()),
        f"{prefix}_left_active_frac": float(left.float().mean().item()) if samples else 0.0,
        f"{prefix}_right_active_frac": float(right.float().mean().item()) if samples else 0.0,
        f"{prefix}_both_active_frac": float(both.float().mean().item()) if samples else 0.0,
        f"{prefix}_either_active_frac": float(either.float().mean().item()) if samples else 0.0,
        f"{prefix}_positive": bool(either.any().item()),
        f"{prefix}_both_positive": bool(both.any().item()),
    }


def replay_probe(tyro_config: ExperimentConfig) -> None:
    simulation_app = init_sim_imports(tyro_config)

    import torch

    from holosoma.utils.common import seeding

    if os.environ.get("E133_DISABLE_REPLAY_SLEEP", "1") != "0":
        try:
            import holosoma.envs.wbt.wbt_manager as wbt_manager_module

            wbt_manager_module.time.sleep = lambda _dt: None
        except Exception as exc:  # pragma: no cover - diagnostic only
            print(f"E133_SLEEP_PATCH_SKIPPED reason={type(exc).__name__}:{exc}", flush=True)

    seeding(42, torch_deterministic=False)

    env_target = tyro_config.env_class
    tyro_env_config = get_tyro_env_config(tyro_config)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = get_class(env_target)(tyro_env_config, device=device)

    setattr(env, "_draw_debug_vis", lambda: None)

    motion_command = env.command_manager.get_state("motion_command")
    full_contact = motion_command.motion.object_contact
    full_stats = _contact_stats(full_contact, "motion")
    loaded = {
        "has_object": bool(motion_command.motion.has_object),
        "has_partner": bool(motion_command.motion.has_partner),
        "has_object_contact": bool(motion_command.motion.has_object_contact),
        "time_step_total": int(motion_command.motion.time_step_total),
    }
    print(
        "E133_PROBE_LOADED "
        f"has_object={loaded['has_object']} "
        f"has_partner={loaded['has_partner']} "
        f"has_object_contact={loaded['has_object_contact']} "
        f"time_step_total={loaded['time_step_total']} "
        f"motion_contact_total={full_stats['motion_contact_total']} "
        f"motion_shape={full_stats['motion_shape']}",
        flush=True,
    )

    done = False
    max_steps = int(os.environ.get("E133_MAX_STEPS", "250") or "250")
    steps = 0
    ref_samples: list[torch.Tensor] = []
    observed_time_steps: list[int] = []
    while not done:
        env.simulator.sim.step()
        done = env.step_visualize_motion(None)  # type: ignore[attr-defined]
        ref_samples.append(motion_command.ref_object_contact.detach().bool().cpu())
        observed_time_steps.extend(int(x) for x in motion_command.time_steps.detach().cpu().reshape(-1).tolist())
        steps += 1
        if max_steps and steps >= max_steps:
            break

    if ref_samples:
        ref_contact = torch.stack(ref_samples, dim=0)
        ref_stats = _contact_stats(ref_contact, "ref")
    else:
        ref_stats = _contact_stats(torch.zeros((0, 2), dtype=torch.bool), "ref")

    summary = {
        **loaded,
        **full_stats,
        **ref_stats,
        "steps": steps,
        "done": bool(done),
        "observed_time_step_min": min(observed_time_steps) if observed_time_steps else None,
        "observed_time_step_max": max(observed_time_steps) if observed_time_steps else None,
        "training_launched": False,
        "cem_launched": False,
        "remote_jobs_launched": False,
        "semantic_ref_mask_ready": False,
        "rl_ready": False,
    }
    print("E133_PROBE_SUMMARY " + json.dumps(summary, sort_keys=True), flush=True)
    print(f"E133_PROBE_DONE steps={steps} done={done}", flush=True)
    close_simulation_app(simulation_app)


def main() -> None:
    tyro_cfg = tyro.cli(AnnotatedExperimentConfig, config=TYRO_CONIFG)
    replay_probe(tyro_cfg)


if __name__ == "__main__":
    main()
