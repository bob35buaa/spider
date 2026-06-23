# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Define functions to get noise schedule for the optimizer.

Convention:
- All info should be numpy array.

Author: Chaoyi Pan
Date: 2025-08-10
"""

from __future__ import annotations

import loguru
import numpy as np
import torch
import torch.nn.functional as F

from spider.config import Config
from spider.interp import interp


def _cem_any_gate_enabled(config: Config) -> bool:
    return (
        config.cem_safety_gate_enabled
        or config.cem_hand_gate_enabled
        or config.cem_posture_gate_enabled
        or config.cem_peak_margin_enabled
    )


def _cem_min_valid_frac(config: Config) -> float:
    vals = []
    if config.cem_safety_gate_enabled or config.cem_hand_gate_enabled:
        vals.append(float(config.cem_safety_gate_min_valid_frac))
    if config.cem_posture_gate_enabled:
        vals.append(float(config.cem_posture_gate_min_valid_frac))
    if config.cem_peak_margin_enabled:
        vals.append(float(config.cem_peak_margin_min_valid_frac))
    return max(vals) if vals else float(config.cem_safety_gate_min_valid_frac)


def _compute_sample_smooth_info(
    config: Config, info_combined: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor] | None:
    """Compute E166 sample-level body-trajectory smoothness penalties."""
    if (
        not config.cem_smooth_enabled
        or (
            float(config.cem_smooth_accel_weight) <= 0.0
            and float(config.cem_smooth_jerk_weight) <= 0.0
        )
        or "cem_smooth_body_pos" not in info_combined
    ):
        return None

    pos = info_combined["cem_smooth_body_pos"]
    if pos.ndim != 4 or pos.shape[0] < 3:
        return None
    axis = str(getattr(config, "cem_smooth_axis", "xyz")).lower()
    if axis == "z":
        pos = pos[..., 2:3]
    elif axis != "xyz":
        raise ValueError(f"unsupported cem_smooth_axis={config.cem_smooth_axis!r}")
    dt = max(float(config.sim_dt), 1e-8)
    penalty = torch.zeros(pos.shape[1], device=pos.device, dtype=pos.dtype)
    out: dict[str, torch.Tensor] = {}

    accel = (pos[2:] - 2.0 * pos[1:-1] + pos[:-2]) / (dt * dt)
    accel_norm = accel.norm(dim=-1).mean(dim=-1)
    accel_p95 = torch.quantile(accel_norm, 0.95, dim=0)
    out["sample_smooth_accel_p95"] = accel_p95
    if float(config.cem_smooth_accel_weight) > 0.0:
        penalty = penalty + float(config.cem_smooth_accel_weight) * accel_p95

    if pos.shape[0] >= 4:
        jerk = (pos[3:] - 3.0 * pos[2:-1] + 3.0 * pos[1:-2] - pos[:-3]) / (
            dt * dt * dt
        )
        jerk_norm = jerk.norm(dim=-1).mean(dim=-1)
        jerk_p95 = torch.quantile(jerk_norm, 0.95, dim=0)
    else:
        jerk_p95 = torch.zeros(pos.shape[1], device=pos.device, dtype=pos.dtype)
    out["sample_smooth_jerk_p95"] = jerk_p95
    if float(config.cem_smooth_jerk_weight) > 0.0:
        penalty = penalty + float(config.cem_smooth_jerk_weight) * jerk_p95

    out["sample_smooth_penalty"] = penalty
    return out


def _sample_p95_over_time_body(values: torch.Tensor) -> torch.Tensor:
    """Return per-sample p95 for a tensor shaped (time, sample, body)."""
    flat = values.permute(1, 0, 2).reshape(values.shape[1], -1)
    return torch.quantile(flat, 0.95, dim=1)


def _compute_sample_e167_z_info(
    config: Config, info_combined: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor] | None:
    """Compute E167 Holosoma-style z-only body/ground penalties."""
    body_active = (
        config.e167_body_z_enabled
        and float(config.e167_body_z_weight) > 0.0
        and "e167_body_z_pos" in info_combined
        and "e167_body_z_ref_pos" in info_combined
    )
    ground_active = (
        config.e167_ground_z_enabled
        and float(config.e167_ground_z_weight) > 0.0
        and "e167_ground_z_pos" in info_combined
        and "e167_ground_z_ref_pos" in info_combined
    )
    if not body_active and not ground_active:
        return None

    template = (
        info_combined["e167_body_z_pos"]
        if body_active
        else info_combined["e167_ground_z_pos"]
    )
    if template.ndim != 4:
        return None
    penalty = torch.zeros(template.shape[1], device=template.device, dtype=template.dtype)
    out: dict[str, torch.Tensor] = {}

    if body_active:
        pos = info_combined["e167_body_z_pos"]
        ref = info_combined["e167_body_z_ref_pos"]
        if pos.ndim == 4 and ref.shape == pos.shape:
            z_err = (pos[..., 2] - ref[..., 2]).abs()
            over = torch.clamp(z_err - float(config.e167_body_z_threshold_m), min=0.0)
            out["sample_e167_body_z_err_mean"] = z_err.mean(dim=(0, 2))
            out["sample_e167_body_z_err_p95"] = _sample_p95_over_time_body(z_err)
            out["sample_e167_body_z_err_peak"] = z_err.amax(dim=(0, 2))
            out["sample_e167_body_z_over_frac"] = (
                z_err > float(config.e167_body_z_threshold_m)
            ).to(z_err.dtype).mean(dim=(0, 2))
            out["sample_e167_body_z_over_mean"] = over.mean(dim=(0, 2))
            penalty = penalty + float(config.e167_body_z_weight) * out[
                "sample_e167_body_z_err_mean"
            ]

    if ground_active:
        pos = info_combined["e167_ground_z_pos"]
        ref = info_combined["e167_ground_z_ref_pos"]
        if pos.ndim == 4 and ref.shape == pos.shape:
            grounded = ref[..., 2] <= float(config.e167_ground_contact_height_m)
            z_dev = (pos[..., 2] - ref[..., 2]).abs()
            grounded_z = torch.where(grounded, z_dev, torch.zeros_like(z_dev))
            denom = grounded.to(pos.dtype).sum(dim=(0, 2)).clamp_min(1.0)
            ground_mean = grounded_z.sum(dim=(0, 2)) / denom
            out["sample_e167_ground_z_dev_mean"] = ground_mean
            out["sample_e167_ground_z_dev_peak"] = grounded_z.amax(dim=(0, 2))
            out["sample_e167_ground_z_frame_frac"] = grounded.to(pos.dtype).mean(
                dim=(0, 2)
            )
            penalty = penalty + float(config.e167_ground_z_weight) * ground_mean

    out["sample_e167_z_penalty"] = penalty
    return out


def _compute_sample_foot_info(
    config: Config, info_combined: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor] | None:
    """Compute E166 foot-slip and foot-ground sample penalties."""
    slip_active = config.foot_slip_enabled and float(config.foot_slip_weight) > 0.0
    ground_active = config.foot_ground_enabled and float(config.foot_ground_weight) > 0.0
    if (
        (not slip_active and not ground_active)
        or "foot_body_pos" not in info_combined
        or "foot_body_ref_pos" not in info_combined
    ):
        return None

    pos = info_combined["foot_body_pos"]
    ref = info_combined["foot_body_ref_pos"]
    if pos.ndim != 4 or ref.shape != pos.shape or pos.shape[0] < 2:
        return None

    grounded = ref[..., 2] <= float(config.foot_slip_contact_height_m)
    penalty = torch.zeros(pos.shape[1], device=pos.device, dtype=pos.dtype)
    out: dict[str, torch.Tensor] = {}
    dt = max(float(config.sim_dt), 1e-8)

    if slip_active:
        xy_speed = (pos[1:, :, :, :2] - pos[:-1, :, :, :2]).norm(dim=-1) / dt
        grounded_pair = grounded[1:] & grounded[:-1]
        slip_values = torch.where(grounded_pair, xy_speed, torch.zeros_like(xy_speed))
        denom = grounded_pair.to(pos.dtype).sum(dim=(0, 2)).clamp_min(1.0)
        slip_mean = slip_values.sum(dim=(0, 2)) / denom
        slip_peak = slip_values.amax(dim=(0, 2))
        out["sample_foot_slip_speed_mean"] = slip_mean
        out["sample_foot_slip_speed_peak"] = slip_peak
        penalty = penalty + float(config.foot_slip_weight) * slip_mean

    if ground_active:
        z_dev = (pos[..., 2] - ref[..., 2]).abs()
        grounded_z = torch.where(grounded, z_dev, torch.zeros_like(z_dev))
        denom = grounded.to(pos.dtype).sum(dim=(0, 2)).clamp_min(1.0)
        ground_mean = grounded_z.sum(dim=(0, 2)) / denom
        ground_peak = grounded_z.amax(dim=(0, 2))
        out["sample_foot_ground_dev_mean"] = ground_mean
        out["sample_foot_ground_dev_peak"] = ground_peak
        penalty = penalty + float(config.foot_ground_weight) * ground_mean

    out["sample_foot_penalty"] = penalty
    return out


def _sample_ctrls_impl(
    config, ctrls: torch.Tensor, sample_params: dict | None = None
) -> torch.Tensor:
    """Sample control actions from the control signal (implementation).

    Args:
        config: Config
        ctrls: Control actions, shape (horizon_steps, nu)
        sample_params: Optional dict with sampling parameters (e.g., global_noise_scale)

    Returns:
        Control actions, shape (num_samples, horizon_steps, nu)
    """
    # decode sample_params
    global_noise_scale = sample_params.get("global_noise_scale", 1.0)
    # sample knot with shape (num_samples, num_knots, nu)
    knot_samples = (
        torch.randn_like(config.noise_scale, device=config.device)
        * config.noise_scale
        * global_noise_scale
    )
    # interp to horizon_steps
    delta_ctrl_samples = interp(knot_samples, config.knot_steps)
    # add to ctrls
    ctrls_samples = ctrls + delta_ctrl_samples
    return ctrls_samples


# Compiled version (torch.compile requires PyTorch 2.0+)
if hasattr(torch, "compile"):
    _sample_ctrls_compiled = torch.compile(_sample_ctrls_impl)
else:
    _sample_ctrls_compiled = _sample_ctrls_impl


def sample_ctrls(
    config, ctrls: torch.Tensor, sample_params: dict | None = None
) -> torch.Tensor:
    """Sample control actions from the control signal.

    Args:
        config: Config
        ctrls: Control actions, shape (horizon_steps, nu)
        sample_params: Optional dict with sampling parameters (e.g., global_noise_scale)

    Returns:
        Control actions, shape (num_samples, horizon_steps, nu)
    """
    if config.use_torch_compile:
        return _sample_ctrls_compiled(config, ctrls, sample_params)
    else:
        return _sample_ctrls_impl(config, ctrls, sample_params)


def _compute_sample_gate_info(
    config: Config, info_combined: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor] | None:
    """Build sample-level gate masks with independent body/hand thresholds."""
    if not _cem_any_gate_enabled(config):
        return None

    gate_masks: list[torch.Tensor] = []
    sample_gate_min_sdf = None
    sample_gate_violation_pct = None
    sample_gate_violation_depth_mean = None
    out: dict[str, torch.Tensor] = {}

    def add_gate(
        source_prefix: str,
        output_prefix: str,
        min_sdf_m: float,
        max_violation_pct: float,
        hard_floor_m: float = float("nan"),
    ) -> None:
        nonlocal sample_gate_min_sdf
        nonlocal sample_gate_violation_pct
        nonlocal sample_gate_violation_depth_mean
        min_key = f"{source_prefix}_min_sdf"
        violation_key = f"{source_prefix}_violation"
        depth_key = f"{source_prefix}_violation_depth"
        if (
            min_key not in info_combined
            or violation_key not in info_combined
            or depth_key not in info_combined
        ):
            return
        min_sdf = info_combined[min_key].min(dim=0).values
        violation_pct = info_combined[violation_key].mean(dim=0)
        violation_depth_mean = info_combined[depth_key].mean(dim=0)
        # E153: hard floor decoupled from the per-frame violation threshold. NaN
        # (hard_floor_m != hard_floor_m) => floor = min_sdf_m (legacy behavior, where
        # the floor subsumes max_violation_pct). A deeper floor lets max_violation_pct
        # tolerate a few frames in [floor, min_sdf_m) while still rejecting any frame
        # below the absolute floor.
        floor = min_sdf_m if hard_floor_m != hard_floor_m else hard_floor_m
        valid_mask = (min_sdf >= floor) & (
            violation_pct <= max_violation_pct
        )
        gate_masks.append(valid_mask)
        sample_gate_min_sdf = (
            min_sdf
            if sample_gate_min_sdf is None
            else torch.minimum(sample_gate_min_sdf, min_sdf)
        )
        sample_gate_violation_pct = (
            violation_pct
            if sample_gate_violation_pct is None
            else torch.maximum(sample_gate_violation_pct, violation_pct)
        )
        sample_gate_violation_depth_mean = (
            violation_depth_mean
            if sample_gate_violation_depth_mean is None
            else torch.maximum(
                sample_gate_violation_depth_mean, violation_depth_mean
            )
        )
        out[f"{output_prefix}_min_sdf"] = min_sdf
        out[f"{output_prefix}_violation_pct"] = violation_pct
        out[f"{output_prefix}_violation_depth_mean"] = violation_depth_mean
        out[f"{output_prefix}_valid_mask"] = valid_mask

    if config.cem_safety_gate_enabled:
        source_prefix = (
            "cem_body_gate"
            if "cem_body_gate_min_sdf" in info_combined
            else "cem_gate"
        )
        add_gate(
            source_prefix,
            "sample_body_gate",
            config.cem_safety_gate_min_sdf_m,
            config.cem_safety_gate_max_violation_pct,
            config.cem_safety_gate_hard_floor_m,
        )
    if config.cem_hand_gate_enabled:
        add_gate(
            "cem_hand_gate",
            "sample_hand_gate",
            config.cem_hand_gate_min_sdf_m,
            config.cem_hand_gate_max_violation_pct,
            config.cem_hand_gate_hard_floor_m,
        )
    if config.cem_posture_gate_enabled and {
        "cem_posture_z_err",
        "cem_posture_z_drop",
    }.issubset(info_combined):
        z_err = info_combined["cem_posture_z_err"]
        z_drop = info_combined["cem_posture_z_drop"]
        terminal_frac = float(config.cem_posture_gate_terminal_frac)
        terminal_steps = max(1, int(np.ceil(z_err.shape[0] * terminal_frac)))
        mean_z_err = z_err.mean(dim=0)
        terminal_z_err = z_err[-terminal_steps:].mean(dim=0)
        max_z_drop = z_drop.max(dim=0).values
        valid_mask = (
            (mean_z_err <= config.cem_posture_gate_mean_z_err_m)
            & (terminal_z_err <= config.cem_posture_gate_terminal_z_err_m)
            & (max_z_drop <= config.cem_posture_gate_max_z_drop_m)
        )
        violation = (
            torch.clamp(mean_z_err - config.cem_posture_gate_mean_z_err_m, min=0.0)
            / 0.05
            + torch.clamp(
                terminal_z_err - config.cem_posture_gate_terminal_z_err_m,
                min=0.0,
            )
            / 0.05
            + torch.clamp(max_z_drop - config.cem_posture_gate_max_z_drop_m, min=0.0)
            / 0.05
        )
        gate_masks.append(valid_mask)
        sample_gate_min_sdf = (
            -violation
            if sample_gate_min_sdf is None
            else torch.minimum(sample_gate_min_sdf, -violation)
        )
        sample_gate_violation_pct = (
            violation
            if sample_gate_violation_pct is None
            else torch.maximum(sample_gate_violation_pct, violation)
        )
        sample_gate_violation_depth_mean = (
            violation
            if sample_gate_violation_depth_mean is None
            else torch.maximum(sample_gate_violation_depth_mean, violation)
        )
        out.update(
            {
                "sample_posture_mean_z_err": mean_z_err,
                "sample_posture_terminal_z_err": terminal_z_err,
                "sample_posture_max_z_drop": max_z_drop,
                "sample_posture_violation": violation,
                "sample_posture_valid_mask": valid_mask,
            }
        )

    if config.cem_peak_margin_enabled and {
        "cem_peak_margin_ee_body_err",
        "cem_peak_margin_anchor_pos_err",
    }.issubset(info_combined):
        ee_peak = info_combined["cem_peak_margin_ee_body_err"].max(dim=0).values
        anchor_peak = info_combined["cem_peak_margin_anchor_pos_err"].max(dim=0).values
        ee_margin = float(config.cem_peak_margin_ee_threshold_m) - ee_peak
        anchor_margin = float(config.cem_peak_margin_anchor_threshold_m) - anchor_peak
        buffer_m = max(float(config.cem_peak_margin_buffer_m), 1e-6)
        ee_violation = torch.clamp(buffer_m - ee_margin, min=0.0) / buffer_m
        anchor_violation = torch.clamp(buffer_m - anchor_margin, min=0.0) / buffer_m
        posture_violation = out.get(
            "sample_posture_violation",
            torch.zeros_like(ee_violation),
        )
        violation = (
            float(config.cem_peak_margin_w_ee) * ee_violation
            + float(config.cem_peak_margin_w_anchor) * anchor_violation
            + float(config.cem_peak_margin_w_posture) * posture_violation
        )
        valid_mask = (
            (ee_peak <= float(config.cem_peak_margin_ee_threshold_m))
            & (anchor_peak <= float(config.cem_peak_margin_anchor_threshold_m))
        )
        gate_masks.append(valid_mask)
        min_margin = torch.minimum(ee_margin, anchor_margin)
        sample_gate_min_sdf = (
            min_margin
            if sample_gate_min_sdf is None
            else torch.minimum(sample_gate_min_sdf, min_margin)
        )
        sample_gate_violation_pct = (
            violation
            if sample_gate_violation_pct is None
            else torch.maximum(sample_gate_violation_pct, violation)
        )
        sample_gate_violation_depth_mean = (
            violation
            if sample_gate_violation_depth_mean is None
            else torch.maximum(sample_gate_violation_depth_mean, violation)
        )
        out.update(
            {
                "sample_peak_margin_ee_peak": ee_peak,
                "sample_peak_margin_anchor_peak": anchor_peak,
                "sample_peak_margin_ee_margin": ee_margin,
                "sample_peak_margin_anchor_margin": anchor_margin,
                "sample_peak_margin_ee_violation": ee_violation,
                "sample_peak_margin_anchor_violation": anchor_violation,
                "sample_peak_margin_violation": violation,
                "sample_peak_margin_valid_mask": valid_mask,
            }
        )

    if not gate_masks:
        return None
    sample_gate_valid_mask = gate_masks[0]
    for mask in gate_masks[1:]:
        sample_gate_valid_mask = sample_gate_valid_mask & mask
    out.update(
        {
            "sample_gate_min_sdf": sample_gate_min_sdf,
            "sample_gate_violation_pct": sample_gate_violation_pct,
            "sample_gate_violation_depth_mean": sample_gate_violation_depth_mean,
            "sample_gate_valid_mask": sample_gate_valid_mask,
        }
    )
    return out


def make_rollout_fn(
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
):
    def rollout(
        config: Config,
        env,
        ctrls: torch.Tensor,
        ref_slice: tuple[torch.Tensor, ...],
        env_param: dict,
    ) -> torch.Tensor:
        """Rollout the control actions to get reward

        Args:
            config: Config
            env: Environment
            ctrls: Control actions, shape (horizon_steps, nu)
            ref_slice: Reference slice, shape (nq, nv, nu, ncon, ncon_pos)

        Returns:
            Reward, shape (num_samples,)
            Info: dict, including trace (N, H, n_trace, 3)
        """
        # save initial state
        init_state = save_state(env)

        # save initial env params (active group pointer)
        init_env_param = save_env_params(config, env)

        # select rollout graph/data pointers for this rollout
        env = load_env_params(config, env, env_param)

        # rollout to get reward
        N, H = ctrls.shape[:2]
        trace_list = []
        cum_rew = torch.zeros(N, device=config.device)
        info_list = []
        for t in range(H):
            # step the environment
            step_env(config, env, ctrls[:, t])  # (N, nu)
            # get reward
            ref = [r[t] for r in ref_slice]
            rew, info = (
                get_reward(config, env, ref)
                if t < H - 1
                else get_terminal_reward(config, env, ref)
            )
            cum_rew += rew
            # get trace
            trace = get_trace(config, env)
            trace_list.append(trace)
            info_list.append(info)
            # Resampling: replace bad samples with good samples periodically
            terminate = get_terminate(config, env, ref)
            if (
                config.terminate_resample
                and t < H - 1
                and terminate.any()
                and (not terminate.all())
            ):
                # if terminate.all():
                #     # if all terminate, replace low reward samples with high reward samples
                #     # top indices: top 50% samples
                #     good_indices = torch.topk(
                #         cum_rew, k=int(0.5 * config.num_samples), largest=True
                #     ).indices
                #     bad_indices = torch.topk(
                #         cum_rew, k=int(0.5 * config.num_samples), largest=False
                #     ).indices
                # else:
                # bad indices is terminate
                bad_indices = torch.nonzero(terminate).squeeze(-1)
                good_indices = torch.nonzero(~terminate).squeeze(-1)
                # make sure good indices shape is the same as bad indices
                if good_indices.shape[0] > bad_indices.shape[0]:
                    good_indices = good_indices[: bad_indices.shape[0]]
                elif good_indices.shape[0] < bad_indices.shape[0]:
                    random_idx = torch.randint(
                        0, good_indices.shape[0], (bad_indices.shape[0],)
                    )
                    good_indices = good_indices[random_idx]

                # Replace bad sample simulation state with good sample simulation state
                copy_sample_state(config, env, good_indices, bad_indices)

                # Replace bad samples control with good samples (for initial control and current timestep only)
                ctrls[bad_indices, :t] = ctrls[good_indices, :t]

                # Replace bad samples cumulative reward with good samples reward
                cum_rew[bad_indices] = cum_rew[good_indices]

        info_combined = {
            k: torch.stack([info[k] for info in info_list], axis=0)
            for k in info_list[0].keys()
        }
        mean_info = {k: v.mean(axis=0) for k, v in info_combined.items()}
        mean_rew = cum_rew / H

        # reset all envs back to initial state
        env = load_state(env, init_state)

        # reset env params
        env = load_env_params(config, env, init_env_param)

        # get info
        trace_list = torch.stack(trace_list, dim=1)
        info = {
            "trace": trace_list,  # (N, H, n_trace, 3)
            **mean_info,
        }
        gate_info = _compute_sample_gate_info(config, info_combined)
        if gate_info is not None:
            info.update(gate_info)
        smooth_info = _compute_sample_smooth_info(config, info_combined)
        if smooth_info is not None:
            info.update(smooth_info)
        e167_z_info = _compute_sample_e167_z_info(config, info_combined)
        if e167_z_info is not None:
            info.update(e167_z_info)
        foot_info = _compute_sample_foot_info(config, info_combined)
        if foot_info is not None:
            info.update(foot_info)
        return ctrls, mean_rew, terminate, info

    return rollout


def _compute_weights_impl(
    rews: torch.Tensor,
    num_samples: int,
    temperature: float,
    elite_fraction: float = 0.1,
) -> torch.Tensor:
    """Compute softmax weights from rewards (implementation).

    Args:
        rews: Rewards, shape (num_samples,)
        num_samples: Number of samples
        temperature: Temperature for softmax
        elite_fraction: Fraction of top samples to use (default 0.1 = 10%)

    Returns:
        Weights, shape (num_samples,)
    """
    # Handle NaNs and compute weights over N samples
    nan_mask = torch.isnan(rews) | torch.isinf(rews)
    rews_min = (
        rews[~nan_mask].min()
        if (~nan_mask).any()
        else torch.tensor(-1000.0, device=rews.device)
    )
    rews = torch.where(nan_mask, rews_min, rews)

    # Select top elite_fraction samples for softmax weighting
    top_k = max(1, int(elite_fraction * num_samples))
    top_indices = torch.topk(rews, k=top_k, largest=True).indices

    # Initialize weights as zeros and compute softmax only for top samples
    weights = torch.zeros_like(rews)
    top_rews = rews[top_indices]
    top_rews_normalized = (top_rews - top_rews.mean()) / (top_rews.std() + 1e-2)
    top_weights = F.softmax(top_rews_normalized / temperature, dim=0)
    weights[top_indices] = top_weights

    return weights, nan_mask


def _compute_weights_with_gate_impl(
    rews: torch.Tensor,
    num_samples: int,
    temperature: float,
    elite_fraction: float,
    valid_mask: torch.Tensor,
    violation_pct: torch.Tensor,
    violation_depth: torch.Tensor,
    min_valid_frac: float,
    fallback_score: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
    """Compute elite weights after applying a sample-level hard safety gate."""
    nan_mask = torch.isnan(rews) | torch.isinf(rews)
    rews_min = (
        rews[~nan_mask].min()
        if (~nan_mask).any()
        else torch.tensor(-1000.0, device=rews.device)
    )
    rews_clean = torch.where(nan_mask, rews_min, rews)

    top_k = max(1, int(elite_fraction * num_samples))
    min_valid_count = max(1, int(np.ceil(float(min_valid_frac) * num_samples)))
    valid_mask = valid_mask.to(device=rews.device, dtype=torch.bool) & (~nan_mask)
    valid_count = int(valid_mask.sum().item())
    fallback_used = valid_count < min_valid_count

    if fallback_used:
        if fallback_score is None:
            rew_std = rews_clean.std(unbiased=False).clamp(min=1e-6)
            rew_norm = (rews_clean - rews_clean.mean()) / rew_std
            fallback_score = (
                -violation_depth.to(rews.device)
                - violation_pct.to(rews.device)
                + 1e-3 * rew_norm
            )
        else:
            fallback_score = fallback_score.to(rews.device)
        fallback_score = torch.where(
            nan_mask,
            torch.full_like(fallback_score, -float("inf")),
            fallback_score,
        )
        k = min(top_k, num_samples)
        top_indices = torch.topk(fallback_score, k=k, largest=True).indices
    else:
        candidate_indices = torch.nonzero(valid_mask).squeeze(-1)
        k = min(top_k, int(candidate_indices.shape[0]))
        candidate_rews = rews_clean[candidate_indices]
        rel_top = torch.topk(candidate_rews, k=k, largest=True).indices
        top_indices = candidate_indices[rel_top]

    weights = torch.zeros_like(rews_clean)
    top_rews = rews_clean[top_indices]
    top_rews_normalized = (
        (top_rews - top_rews.mean()) / (top_rews.std(unbiased=False) + 1e-2)
    )
    top_weights = F.softmax(top_rews_normalized / temperature, dim=0)
    weights[top_indices] = top_weights
    return weights, nan_mask, top_indices, fallback_used


# Compiled version (torch.compile requires PyTorch 2.0+)
if hasattr(torch, "compile"):
    _compute_weights_compiled = torch.compile(_compute_weights_impl)
else:
    _compute_weights_compiled = _compute_weights_impl


def make_optimize_once_fn(
    rollout,
):
    def optimize_once(
        config: Config,
        env,
        ctrls: torch.Tensor,
        ref_slice: tuple[torch.Tensor, ...],
        env_params: list[dict] = [{}],
        sample_params: dict | None = None,
    ) -> torch.Tensor:
        """Single step optimization of the policy parameters using DIAL MPC, no annealing is involved

        Args:
            config: Config
            graph: Warp graph
            model_wp: Warp model
            data_wp: Warp data
            ctrls: Control actions, shape (horizon_steps, num_actions)

        Returns:
            Control actions, shape (horizon_steps, num_actions)
        """
        # sample ctrls
        ctrls_samples = sample_ctrls(
            config, ctrls, sample_params
        )  # (num_samples, horizon_steps, num_actions)

        # rollout
        # domain randomization: pick the minimum reward across all DR parameter sets
        min_rew = torch.full((config.num_samples,), float("inf"), device=config.device)
        combined_gate_valid_mask = None
        combined_gate_min_sdf = None
        combined_gate_violation_pct = None
        combined_gate_violation_depth_mean = None
        combined_smooth_penalty = None
        combined_e167_z_penalty = None
        combined_foot_penalty = None
        for env_param in env_params:
            ctrls_samples, rews, terminate, rollout_info = rollout(
                config,
                env,
                ctrls_samples,
                ref_slice,
                env_param,
            )
            min_rew = torch.minimum(min_rew, rews)
            if (
                _cem_any_gate_enabled(config)
                and "sample_gate_valid_mask" in rollout_info
            ):
                valid_mask = rollout_info["sample_gate_valid_mask"]
                min_sdf = rollout_info["sample_gate_min_sdf"]
                violation_pct = rollout_info["sample_gate_violation_pct"]
                violation_depth = rollout_info["sample_gate_violation_depth_mean"]
                combined_gate_valid_mask = (
                    valid_mask
                    if combined_gate_valid_mask is None
                    else (combined_gate_valid_mask & valid_mask)
                )
                combined_gate_min_sdf = (
                    min_sdf
                    if combined_gate_min_sdf is None
                    else torch.minimum(combined_gate_min_sdf, min_sdf)
                )
                combined_gate_violation_pct = (
                    violation_pct
                    if combined_gate_violation_pct is None
                    else torch.maximum(combined_gate_violation_pct, violation_pct)
                )
                combined_gate_violation_depth_mean = (
                    violation_depth
                    if combined_gate_violation_depth_mean is None
                    else torch.maximum(
                        combined_gate_violation_depth_mean, violation_depth
                    )
                )
            if config.cem_smooth_enabled and "sample_smooth_penalty" in rollout_info:
                smooth_penalty = rollout_info["sample_smooth_penalty"]
                combined_smooth_penalty = (
                    smooth_penalty
                    if combined_smooth_penalty is None
                    else torch.maximum(combined_smooth_penalty, smooth_penalty)
                )
            if (
                (config.e167_body_z_enabled or config.e167_ground_z_enabled)
                and "sample_e167_z_penalty" in rollout_info
            ):
                e167_z_penalty = rollout_info["sample_e167_z_penalty"]
                combined_e167_z_penalty = (
                    e167_z_penalty
                    if combined_e167_z_penalty is None
                    else torch.maximum(combined_e167_z_penalty, e167_z_penalty)
                )
            if (
                (config.foot_slip_enabled or config.foot_ground_enabled)
                and "sample_foot_penalty" in rollout_info
            ):
                foot_penalty = rollout_info["sample_foot_penalty"]
                combined_foot_penalty = (
                    foot_penalty
                    if combined_foot_penalty is None
                    else torch.maximum(combined_foot_penalty, foot_penalty)
                )
        # Use worst-case rewards across DR parameter sets
        rews = min_rew
        if combined_smooth_penalty is not None:
            rollout_info["sample_smooth_penalty"] = combined_smooth_penalty
            rews = rews - combined_smooth_penalty
        if combined_e167_z_penalty is not None:
            rollout_info["sample_e167_z_penalty"] = combined_e167_z_penalty
            rews = rews - combined_e167_z_penalty
        if combined_foot_penalty is not None:
            rollout_info["sample_foot_penalty"] = combined_foot_penalty
            rews = rews - combined_foot_penalty
        if (
            _cem_any_gate_enabled(config)
            and combined_gate_valid_mask is not None
        ):
            rollout_info["sample_gate_valid_mask"] = combined_gate_valid_mask
            rollout_info["sample_gate_min_sdf"] = combined_gate_min_sdf
            rollout_info["sample_gate_violation_pct"] = combined_gate_violation_pct
            rollout_info["sample_gate_violation_depth_mean"] = (
                combined_gate_violation_depth_mean
            )

        # resample based on terminate condition

        # Compute weights using compiled or non-compiled version
        elite_fraction = (
            sample_params.get("elite_fraction", 0.1) if sample_params else 0.1
        )
        gate_enabled = (
            _cem_any_gate_enabled(config)
            and "sample_gate_valid_mask" in rollout_info
        )
        selected_indices = None
        gate_fallback_used = False
        if gate_enabled:
            fallback_score = None
            if (
                config.cem_peak_margin_enabled
                and "sample_peak_margin_violation" in rollout_info
            ):
                fallback_score = rews - (
                    float(config.cem_peak_margin_lambda)
                    * rollout_info["sample_peak_margin_violation"].to(rews.device)
                )
            if (
                fallback_score is None
                and
                config.cem_posture_gate_enabled
                and "sample_posture_violation" in rollout_info
            ):
                fallback_score = rews - (
                    float(config.cem_posture_gate_fallback_lambda)
                    * rollout_info["sample_posture_violation"].to(rews.device)
                )
            weights, nan_mask, selected_indices, gate_fallback_used = (
                _compute_weights_with_gate_impl(
                    rews,
                    config.num_samples,
                    config.temperature,
                    elite_fraction,
                    rollout_info["sample_gate_valid_mask"],
                    rollout_info["sample_gate_violation_pct"],
                    rollout_info["sample_gate_violation_depth_mean"],
                    _cem_min_valid_frac(config),
                    fallback_score,
                )
            )
        elif config.use_torch_compile and elite_fraction == 0.1:
            weights, nan_mask = _compute_weights_compiled(
                rews, config.num_samples, config.temperature
            )
        else:
            weights, nan_mask = _compute_weights_impl(
                rews, config.num_samples, config.temperature, elite_fraction
            )

        if nan_mask.any():
            loguru.logger.warning(
                f"NaNs or infs in rews: {nan_mask.sum()}/{config.num_samples}"
            )

        ctrls_mean = (weights[:, None, None] * ctrls_samples).sum(dim=0)

        # SBTO: apply mean EWMA (DynaRetarget α_μ)
        mean_momentum = (
            sample_params.get("mean_momentum", 0.0) if sample_params else 0.0
        )
        if mean_momentum > 0.0:
            ctrls_mean = mean_momentum * ctrls + (1.0 - mean_momentum) * ctrls_mean

        # SBTO: compute elite sample std for Sigma EWMA (returned via info)
        elite_std = None
        if sample_params and sample_params.get("return_elite_std", False):
            top_k = max(1, int(elite_fraction * config.num_samples))
            top_indices = (
                selected_indices
                if selected_indices is not None
                else torch.topk(rews, k=top_k, largest=True).indices
            )
            elite_ctrls = ctrls_samples[top_indices]  # (top_k, H, nu)
            elite_std = elite_ctrls.std(dim=0, unbiased=False)  # (H, nu)

        # down sample traces by selecting topk and uniform samples for visualization
        n_uni = max(0, min(config.num_trace_uniform_samples, config.num_samples))
        n_topk = max(0, min(config.num_trace_topk_samples, config.num_samples))
        idx_uni = (
            torch.linspace(
                0,
                config.num_samples - 1,
                steps=n_uni,
                dtype=torch.long,
                device=config.device,
            )
            if n_uni > 0
            else torch.tensor([], dtype=torch.long, device=config.device)
        )
        idx_top = (
            selected_indices[:n_topk]
            if selected_indices is not None and n_topk > 0
            else torch.topk(rews, k=n_topk, largest=True).indices
            if n_topk > 0
            else torch.tensor([], dtype=torch.long, device=config.device)
        )
        sel_idx = torch.cat([idx_uni, idx_top], dim=0).long()

        # compute info
        info = {}
        for k, v in rollout_info.items():
            if k not in ["trace", "trace_sample"]:
                if isinstance(v, torch.Tensor):
                    v = v.cpu().numpy()
                if v.ndim == 1:
                    k_max = k + "_max"
                    k_min = k + "_min"
                    k_median = k + "_median"
                    k_mean = k + "_mean"
                    info[k_max] = v.max()
                    info[k_min] = v.min()
                    info[k_median] = np.median(v)
                    info[k_mean] = v.mean()
        # log reward
        rews_np = rews.cpu().numpy()
        improvement = rews_np.max() - rews_np[0]
        info["improvement"] = improvement
        info["rew_max"] = rews_np.max()
        info["rew_min"] = rews_np.min()
        info["rew_median"] = np.median(rews_np)
        info["rew_mean"] = rews_np.mean()
        if gate_enabled:
            valid_mask = rollout_info["sample_gate_valid_mask"]
            info["cem_gate_valid_frac"] = valid_mask.float().mean().item()
            info["cem_gate_fallback_used"] = float(gate_fallback_used)
            info["cem_gate_selected_valid_frac"] = (
                valid_mask[selected_indices].float().mean().item()
                if selected_indices is not None and selected_indices.numel() > 0
                else 0.0
            )
            if "sample_hand_gate_valid_mask" in rollout_info:
                hand_mask = rollout_info["sample_hand_gate_valid_mask"]
                info["cem_hand_gate_valid_frac"] = hand_mask.float().mean().item()
                info["cem_hand_gate_selected_valid_frac"] = (
                    hand_mask[selected_indices].float().mean().item()
                    if selected_indices is not None and selected_indices.numel() > 0
                    else 0.0
                )
            if "sample_body_gate_valid_mask" in rollout_info:
                body_mask = rollout_info["sample_body_gate_valid_mask"]
                info["cem_body_gate_valid_frac"] = body_mask.float().mean().item()
                info["cem_body_gate_selected_valid_frac"] = (
                    body_mask[selected_indices].float().mean().item()
                    if selected_indices is not None and selected_indices.numel() > 0
                    else 0.0
                )
            if "sample_posture_valid_mask" in rollout_info:
                posture_mask = rollout_info["sample_posture_valid_mask"]
                info["cem_posture_gate_valid_frac"] = (
                    posture_mask.float().mean().item()
                )
                info["cem_posture_gate_selected_valid_frac"] = (
                    posture_mask[selected_indices].float().mean().item()
                    if selected_indices is not None and selected_indices.numel() > 0
                    else 0.0
                )
                info["cem_posture_gate_fallback_used"] = float(gate_fallback_used)
            if "sample_peak_margin_valid_mask" in rollout_info:
                peak_mask = rollout_info["sample_peak_margin_valid_mask"]
                info["cem_peak_margin_valid_frac"] = (
                    peak_mask.float().mean().item()
                )
                info["cem_peak_margin_selected_valid_frac"] = (
                    peak_mask[selected_indices].float().mean().item()
                    if selected_indices is not None and selected_indices.numel() > 0
                    else 0.0
                )
                info["cem_peak_margin_fallback_used"] = float(gate_fallback_used)

        # Downsample and store trace site positions for selected sample trajectories
        if "trace" in rollout_info:
            info["trace_sample"] = (
                rollout_info["trace"][sel_idx].cpu().numpy()
            )  # (M, H, n_trace, 3)
            info["trace_cost"] = -rews[sel_idx].cpu().numpy()

        # SBTO: attach elite_std to info for Sigma EWMA
        if elite_std is not None:
            info["elite_std"] = elite_std  # (H, nu) tensor, stays on GPU

        return ctrls_mean, terminate, info

    return optimize_once


def make_optimize_fn(
    optimize_once,
):
    def optimize(
        config: Config,
        env,
        ctrls: torch.Tensor,
        ref_slice: tuple[torch.Tensor, ...],
    ):
        """Full optimization loop at certain time step. Mainly involves simulation parameter annealing and sampling parameter annealing"""
        infos = []

        # schedule sampling parameters
        sample_params_list = []
        for i in range(config.max_num_iterations):
            sample_params = {
                "global_noise_scale": config.beta_traj**i,
            }
            sample_params_list.append(sample_params)

        # optimize
        improvement_history = []
        for i in range(config.max_num_iterations):
            ctrls, terminate, info = optimize_once(
                config,
                env,
                ctrls,
                ref_slice,
                config.env_params_list[i],
                sample_params_list[i],
            )
            infos.append(info)
            improvement_history.append(info["improvement"])

            # early stopping: check if last n steps all have improvement below threshold
            terminate_all = terminate.all()
            terminate_early_stopping = terminate_all and config.terminate_resample
            if (
                len(improvement_history) >= config.improvement_check_steps
                and not terminate_early_stopping
            ):
                recent_improvements = improvement_history[
                    -config.improvement_check_steps :
                ]
                if all(
                    imp < config.improvement_threshold for imp in recent_improvements
                ):
                    break

        # TODO: think about a better logic
        # append zeros to infos to make sure the length is the same as max_num_iterations
        fake_info = {}
        for k, v in infos[0].items():
            fake_info[k] = np.zeros_like(v)
        for _ in range(config.max_num_iterations - len(infos)):
            infos.append(fake_info)
        info_aggregated = {}
        for k in infos[0].keys():
            info_aggregated[k] = np.stack([info[k] for info in infos], axis=0)
        info_aggregated["opt_steps"] = np.array([i + 1])
        return ctrls, info_aggregated

    return optimize
