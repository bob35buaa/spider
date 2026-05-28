#!/usr/bin/env python3
"""Offline reward-term breakdown for saved MJWP rollouts.

This is a CPU replay approximation of the final rollout reward terms. It uses
the same config/ref/mask/target inputs as run_mjwp.py, but evaluates only the
saved trajectory states rather than all CEM samples.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import mujoco
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R


REPO = Path(__file__).resolve().parents[4]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
DEBUG_DIR = REPO / "workspace/core4d/scripts/debug"
EVAL_DIR = REPO / "workspace/core4d/scripts/eval"
for path in (DEBUG_DIR, EVAL_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from diagnose_E068_init_drift import _build_config, _convert_scene_act_ref  # noqa: E402
import eval_E072 as e072  # noqa: E402
from spider.io import load_data  # noqa: E402
from spider.simulators.mjwp import (  # noqa: E402
    _diff_qpos,
    _lf_axis_angle_from_quat,
    _lf_quat_apply,
    _lf_quat_conjugate,
    _lf_quat_mul,
    _local_ori_tracking,
    _local_pos_tracking,
    _weight_diff_qpos,
    quat_sub,
)


OUT_ROOT = REPO / "workspace/core4d/results/E087/reward_breakdown"
TERM_KEYS = [
    "qpos_rew",
    "qvel_rew",
    "local_upper_pos",
    "local_upper_ori",
    "local_lower_pos",
    "local_lower_ori",
    "local_root_pos",
    "local_root_ori",
    "local_joint",
    "task_body_rew",
    "task_obj_rew",
    "contact_hdmi_rew",
    "ctrl_ref_guard_rew",
    "robot_object_penalty",
    "hand_floor_penalty",
    "hand_object_deep_penalty",
    "object_lift_rew",
    "object_floor_penalty",
    "stability_penalty",
    "total_reward",
]


def _as_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO / p


def _resize_time(arr: np.ndarray, target_len: int) -> np.ndarray:
    if arr.shape[0] == target_len:
        return arr.astype(np.float32)
    idx = np.round(np.linspace(0, arr.shape[0] - 1, target_len)).astype(np.int64)
    return arr[idx].astype(np.float32)


def _load_refs(config):
    qpos_ref, qvel_ref, ctrl_ref, contact, contact_pos = load_data(config, config.data_path)
    qpos_ref, qvel_ref, ctrl_ref = _convert_scene_act_ref(
        config, qpos_ref, qvel_ref, ctrl_ref
    )
    return (
        qpos_ref.detach().cpu().numpy(),
        qvel_ref.detach().cpu().numpy(),
        ctrl_ref.detach().cpu().numpy(),
        contact.detach().cpu().numpy(),
        contact_pos.detach().cpu().numpy(),
    )


def _precompute_fk(model: mujoco.MjModel, qpos: np.ndarray) -> dict[str, np.ndarray]:
    data = mujoco.MjData(model)
    T = len(qpos)
    out = {
        "xpos": np.zeros((T, model.nbody, 3), dtype=np.float32),
        "xquat": np.zeros((T, model.nbody, 4), dtype=np.float32),
        "geom_xpos": np.zeros((T, model.ngeom, 3), dtype=np.float32),
        "geom_xmat": np.zeros((T, model.ngeom, 3, 3), dtype=np.float32),
        "site_xpos": np.zeros((T, model.nsite, 3), dtype=np.float32),
    }
    for t, q in enumerate(qpos):
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        out["xpos"][t] = data.xpos
        out["xquat"][t] = data.xquat
        out["geom_xpos"][t] = data.geom_xpos
        out["geom_xmat"][t] = data.geom_xmat.reshape(model.ngeom, 3, 3)
        out["site_xpos"][t] = data.site_xpos
    return out


def _contact_mask(config, target_len: int) -> tuple[np.ndarray | None, str]:
    if not (config.contact_hdmi_gain > 0.0 and config.hand_approach_body_ids):
        return None, ""
    if config.contact_hdmi_mask_source != "core4d_3cm":
        return None, ""
    path = _as_path(config.contact_hdmi_mask_path)
    data = np.load(path, allow_pickle=True)
    axis = config.contact_hdmi_mask_time_axis
    if axis == "auto":
        if (
            "spider_contact_mask_3cm" in data
            and data["spider_contact_mask_3cm"].shape[0] == target_len
        ):
            axis = "spider"
        elif (
            "eval_contact_mask_3cm" in data
            and data["eval_contact_mask_3cm"].shape[0] == target_len
        ):
            axis = "eval"
        else:
            axis = "eval" if "eval_contact_mask_3cm" in data else "spider"
    key = f"{axis}_contact_mask_3cm"
    raw = data[key][:, int(config.contact_hdmi_mask_person_idx), :]
    return _resize_time(raw, target_len), key


def _contact_target(config, target_len: int, model: mujoco.MjModel, qpos_ref: np.ndarray) -> tuple[np.ndarray | None, str]:
    if not (config.contact_hdmi_dynamic_target and config.hand_approach_body_ids):
        return None, ""
    if config.contact_hdmi_target_source == "external":
        path = _as_path(config.contact_hdmi_target_path)
        data = np.load(path, allow_pickle=True)
        axis = config.contact_hdmi_target_time_axis
        if axis == "auto":
            if (
                "spider_contact_target_object_local" in data
                and data["spider_contact_target_object_local"].shape[0] == target_len
            ):
                axis = "spider"
            elif (
                "eval_contact_target_object_local" in data
                and data["eval_contact_target_object_local"].shape[0] == target_len
            ):
                axis = "eval"
            else:
                axis = "eval" if "eval_contact_target_object_local" in data else "spider"
        key = f"{axis}_contact_target_object_local"
        return _resize_time(data[key], target_len), key

    obj_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    data = mujoco.MjData(model)
    out = np.zeros((target_len, len(config.hand_approach_body_ids), 3), dtype=np.float32)
    eef_offset = np.asarray(config.contact_hdmi_eef_offset, dtype=np.float32)
    for t in range(target_len):
        data.qpos[:] = qpos_ref[t]
        mujoco.mj_forward(model, data)
        obj_pos = data.xpos[obj_bid]
        obj_mat = data.xmat[obj_bid].reshape(3, 3)
        for ei, bid in enumerate(config.hand_approach_body_ids):
            hand_pos = data.xpos[bid].copy()
            if config.contact_hdmi_target_uses_eef_offset:
                q = data.xquat[bid]
                hand_pos += R.from_quat([q[1], q[2], q[3], q[0]]).apply(eef_offset)
            out[t, ei] = obj_mat.T @ (hand_pos - obj_pos)
    return out, "ref_fk"


def _geom_box_sdf_min(
    model: mujoco.MjModel,
    fk: dict[str, np.ndarray],
    geom_ids: list[int],
    object_geom_id: int,
    half_ext: np.ndarray,
) -> np.ndarray:
    if not geom_ids:
        return np.zeros(len(fk["geom_xpos"]), dtype=np.float32)
    centers = fk["geom_xpos"][:, geom_ids]
    mats = fk["geom_xmat"][:, geom_ids]
    axes = mats[:, :, :, 2]
    radii = np.asarray([float(model.geom_size[gid, 0]) for gid in geom_ids], dtype=np.float32)
    half_lens = np.asarray(
        [
            float(model.geom_size[gid, 1])
            if int(model.geom_type[gid]) == int(mujoco.mjtGeom.mjGEOM_CAPSULE)
            else 0.0
            for gid in geom_ids
        ],
        dtype=np.float32,
    )
    sample_scalars = np.stack([-half_lens, np.zeros_like(half_lens), half_lens], axis=1)
    points = centers[:, :, None, :] + axes[:, :, None, :] * sample_scalars[None, :, :, None]
    obj_pos = fk["geom_xpos"][:, object_geom_id]
    obj_mat = fk["geom_xmat"][:, object_geom_id]
    delta = points - obj_pos[:, None, None, :]
    local = np.einsum("tji,tgpj->tgpi", obj_mat, delta)
    q = np.abs(local) - half_ext.reshape(1, 1, 1, 3)
    outside = np.linalg.norm(np.maximum(q, 0.0), axis=-1)
    inside = np.minimum(np.max(q, axis=-1), 0.0)
    sdf_points = outside + inside
    sdf_geom = np.min(sdf_points, axis=2) - radii.reshape(1, -1)
    return np.min(sdf_geom, axis=1)


def _case_window(mask: np.ndarray | None, T: int) -> tuple[int, int]:
    if mask is None:
        return 0, T - 1
    active = np.where(mask.max(axis=1) > 0.5)[0]
    if active.size == 0:
        return 0, T - 1
    return max(0, int(active[0]) - 10), min(T - 1, int(active[-1]) + 10)


def compute_breakdown(variant: str, override: str, task: str, npz_path: Path) -> tuple[list[dict[str, float]], dict[str, object]]:
    config = _build_config(override, task, "cpu", 4)
    model = mujoco.MjModel.from_xml_path(config.model_path)
    qpos_ref, qvel_ref, ctrl_ref, _contact, contact_pos = _load_refs(config)
    data = np.load(npz_path, allow_pickle=True)
    qpos = e072.flatten_time_major(data["qpos"])
    qvel = e072.flatten_time_major(data["qvel"])
    ctrl = e072.flatten_time_major(data["ctrl"])
    times = e072.flatten_time_major(data["time"])
    T = min(len(qpos), len(qvel), len(ctrl), len(qpos_ref), len(qvel_ref), len(ctrl_ref))
    qpos = qpos[:T]
    qvel = qvel[:T]
    ctrl = ctrl[:T]
    times = times[:T]
    qpos_ref = qpos_ref[:T]
    qvel_ref = qvel_ref[:T]
    ctrl_ref = ctrl_ref[:T]

    sim = _precompute_fk(model, qpos)
    ref_fk = _precompute_fk(model, qpos_ref)
    mask, mask_key = _contact_mask(config, T)
    target, target_key = _contact_target(config, T, model, qpos_ref)
    start, end = _case_window(mask, T)

    device = torch.device("cpu")
    qpos_t = torch.tensor(qpos, dtype=torch.float32, device=device)
    qpos_ref_t = torch.tensor(qpos_ref, dtype=torch.float32, device=device)
    qvel_t = torch.tensor(qvel, dtype=torch.float32, device=device)
    qvel_ref_t = torch.tensor(qvel_ref, dtype=torch.float32, device=device)
    config.device = "cpu"

    qpos_diff = _diff_qpos(config, qpos_t, qpos_ref_t)
    qpos_weight = _weight_diff_qpos(config)
    qpos_dist = torch.norm(qpos_diff * qpos_weight, p=2, dim=1)
    qvel_dist = torch.norm(qvel_t - qvel_ref_t, p=2, dim=1)
    qpos_rew = (
        config.qpos_reward_scale * torch.exp(-qpos_dist / config.qpos_reward_sigma)
        if config.use_bounded_qpos_reward
        else -qpos_dist
    )
    qvel_rew = -config.vel_rew_scale * qvel_dist

    local_terms = {k: torch.zeros(T, dtype=torch.float32) for k in [
        "local_upper_pos", "local_upper_ori", "local_lower_pos", "local_lower_ori",
        "local_root_pos", "local_root_ori", "local_joint",
    ]}
    if config.use_local_frame_reward:
        xpos_sim = torch.tensor(sim["xpos"], dtype=torch.float32)
        xquat_sim = torch.tensor(sim["xquat"], dtype=torch.float32)
        body_xpos_ref = torch.tensor(ref_fk["xpos"], dtype=torch.float32)
        body_xquat_ref = torch.tensor(ref_fk["xquat"], dtype=torch.float32)
        root_id = 1
        rows = []
        for t in range(T):
            ref_root_pos = body_xpos_ref[t, root_id]
            ref_root_quat = body_xquat_ref[t, root_id]
            upper = config.local_frame_upper_ids
            lower = config.local_frame_lower_ids
            local_terms["local_upper_pos"][t] = _local_pos_tracking(
                xpos_sim[t:t+1], xquat_sim[t:t+1], upper, root_id,
                body_xpos_ref[t, upper], ref_root_pos, ref_root_quat,
                config.local_frame_pos_sigma,
            )[0]
            local_terms["local_upper_ori"][t] = _local_ori_tracking(
                xquat_sim[t:t+1], upper, root_id, body_xquat_ref[t, upper],
                ref_root_quat, config.local_frame_ori_sigma,
            )[0]
            local_terms["local_lower_pos"][t] = _local_pos_tracking(
                xpos_sim[t:t+1], xquat_sim[t:t+1], lower, root_id,
                body_xpos_ref[t, lower], ref_root_pos, ref_root_quat,
                config.local_frame_pos_sigma,
            )[0]
            local_terms["local_lower_ori"][t] = _local_ori_tracking(
                xquat_sim[t:t+1], lower, root_id, body_xquat_ref[t, lower],
                ref_root_quat, config.local_frame_ori_sigma,
            )[0]
            root_pos_err = torch.norm(xpos_sim[t, root_id] - ref_root_pos)
            local_terms["local_root_pos"][t] = torch.exp(-root_pos_err / config.local_frame_root_sigma)
            root_diff = _lf_quat_mul(
                _lf_quat_conjugate(ref_root_quat.unsqueeze(0)),
                xquat_sim[t, root_id].unsqueeze(0),
            )
            root_ori_err = _lf_axis_angle_from_quat(root_diff).norm(dim=-1)[0]
            local_terms["local_root_ori"][t] = torch.exp(-root_ori_err / config.local_frame_root_sigma)
            obj_start = -int(config.nq_obj) if int(config.nq_obj) > 0 else None
            jt_err = (qpos_t[t, 7:obj_start] - qpos_ref_t[t, 7:obj_start]).abs().mean()
            local_terms["local_joint"][t] = torch.exp(-jt_err / config.local_frame_joint_sigma)
        qpos_rew = config.local_frame_w_track * sum(local_terms.values())

    task_body_rew = torch.zeros(T)
    if config.task_body_rew_scale > 0.0 and config.task_body_ids:
        body_pos_sim = torch.tensor(sim["xpos"][:, config.task_body_ids], dtype=torch.float32)
        task_body_xpos_ref = torch.tensor(ref_fk["xpos"][:, config.task_body_ids], dtype=torch.float32)
        weights = torch.tensor(config.task_body_weights, dtype=torch.float32)
        err = ((body_pos_sim - task_body_xpos_ref) ** 2).sum(dim=-1)
        task_body_rew = -config.task_body_rew_scale * (err * weights).sum(dim=1)

    task_obj_rew = torch.zeros(T)
    if config.task_obj_pos_rew_scale > 0.0 or config.task_obj_rot_rew_scale > 0.0:
        nq_obj = int(config.nq_obj)
        if nq_obj == 6:
            pos_err_norm = torch.norm(qpos_t[:, -6:-3] - qpos_ref_t[:, -6:-3], dim=-1)
            rot_err_norm = torch.norm(qpos_t[:, -3:] - qpos_ref_t[:, -3:], dim=-1)
        else:
            pos_err_norm = torch.norm(qpos_t[:, -7:-4] - qpos_ref_t[:, -7:-4], dim=-1)
            rot_err_norm = torch.norm(quat_sub(qpos_t[:, -4:], qpos_ref_t[:, -4:]), dim=-1)
        if config.task_obj_pos_rew_scale > 0.0:
            if config.task_obj_use_exp:
                task_obj_rew += config.task_obj_pos_rew_scale * torch.exp(-pos_err_norm / config.task_obj_pos_sigma)
            else:
                task_obj_rew -= config.task_obj_pos_rew_scale * pos_err_norm.square()
        if config.task_obj_rot_rew_scale > 0.0:
            if config.task_obj_use_exp:
                task_obj_rew += config.task_obj_rot_rew_scale * torch.exp(-rot_err_norm / config.task_obj_rot_sigma)
            else:
                task_obj_rew -= config.task_obj_rot_rew_scale * rot_err_norm.square()

    contact_hdmi_rew = torch.zeros(T)
    if config.contact_hdmi_gain > 0.0 and target is not None:
        obj_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
        eef_offset = torch.tensor(config.contact_hdmi_eef_offset, dtype=torch.float32)
        mask_arr = mask if mask is not None else np.ones((T, len(config.hand_approach_body_ids)), dtype=np.float32)
        obj_pos = torch.tensor(sim["xpos"][:, obj_bid], dtype=torch.float32)
        obj_quat = torch.tensor(sim["xquat"][:, obj_bid], dtype=torch.float32)
        target_t = torch.tensor(target, dtype=torch.float32)
        per = []
        for ei, bid in enumerate(config.hand_approach_body_ids):
            target_world = obj_pos + _lf_quat_apply(obj_quat, target_t[:, ei])
            eef_pos = torch.tensor(sim["xpos"][:, bid], dtype=torch.float32)
            eef_quat = torch.tensor(sim["xquat"][:, bid], dtype=torch.float32)
            contact_point = eef_pos + _lf_quat_apply(eef_quat, eef_offset.expand(T, -1))
            dist = torch.norm(target_world - contact_point, dim=-1)
            per.append(torch.exp(-dist / config.contact_hdmi_sigma))
        rew_stack = torch.stack(per, dim=1)
        mask_t = torch.tensor(mask_arr, dtype=torch.float32)
        contact_hdmi_rew = (rew_stack * mask_t * config.contact_hdmi_gain + (1.0 - mask_t)).mean(dim=1)

    ctrl_ref_guard_rew = torch.zeros(T)
    if config.ctrl_ref_guard_scale > 0.0:
        ctrl_dim = min(ctrl.shape[1], ctrl_ref.shape[1])
        guard_dim = ctrl_dim
        if config.ctrl_ref_guard_robot_only:
            obj_dims = int(config.object_action_dims) if config.object_action_dims > 0 else (6 if config.contact_guidance and ctrl_dim > 29 else 0)
            guard_dim = max(ctrl_dim - obj_dims, 0)
        if guard_dim > 0:
            diff = torch.tensor(ctrl[:, :guard_dim] - ctrl_ref[:, :guard_dim], dtype=torch.float32)
            abs_scaled = torch.abs(diff) / max(float(config.ctrl_ref_guard_sigma), 1e-6)
            huber = torch.where(abs_scaled <= 1.0, 0.5 * abs_scaled.square(), abs_scaled - 0.5)
            gate = ((times >= config.ctrl_ref_guard_start_eval_time) & (times <= config.ctrl_ref_guard_end_eval_time)).astype(np.float32)
            ctrl_ref_guard_rew = -config.ctrl_ref_guard_scale * huber.mean(dim=1) * torch.tensor(gate)

    robot_object_penalty = np.zeros(T, dtype=np.float32)
    hand_object_deep_penalty = np.zeros(T, dtype=np.float32)
    object_lift_rew = np.zeros(T, dtype=np.float32)
    object_floor_penalty = np.zeros(T, dtype=np.float32)
    object_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "object_collision")
    if object_gid >= 0 and config.hand_approach_obj_half_extents:
        half_ext = np.asarray(config.hand_approach_obj_half_extents, dtype=np.float32)
        if config.robot_object_penalty_scale > 0.0 and config.robot_object_penalty_geom_ids:
            sdf = _geom_box_sdf_min(model, sim, config.robot_object_penalty_geom_ids, object_gid, half_ext)
            deep_limit = config.robot_object_penalty_margin_m - config.robot_object_penalty_deep_threshold_m
            robot_object_penalty = -config.robot_object_penalty_scale * np.maximum(deep_limit - sdf, 0.0)
        if config.hand_object_deep_penalty_scale > 0.0 and config.hand_object_deep_penalty_geom_ids:
            sdf = _geom_box_sdf_min(model, sim, config.hand_object_deep_penalty_geom_ids, object_gid, half_ext)
            hinge = np.maximum(-config.hand_object_deep_penalty_threshold_m - sdf, 0.0)
            hand_object_deep_penalty = -config.hand_object_deep_penalty_scale * hinge
        if config.object_lift_rew_scale > 0.0 or config.object_floor_penalty_scale > 0.0:
            obj_half_z = float(config.hand_approach_obj_half_extents[2])
            obj_bottom = sim["geom_xpos"][:, object_gid, 2] - obj_half_z
            ref_obj_z = qpos_ref[:, -4] if int(config.nq_obj) == 6 else qpos_ref[:, -5]
            ref_bottom = ref_obj_z - obj_half_z
            if config.object_lift_rew_scale > 0.0:
                object_lift_rew = config.object_lift_rew_scale * np.exp(-np.abs(obj_bottom - ref_bottom) / max(config.object_lift_sigma, 1e-6))
            if config.object_floor_penalty_scale > 0.0:
                floor_hinge = np.maximum((ref_bottom - config.object_floor_margin_m) - obj_bottom, 0.0)
                object_floor_penalty = -config.object_floor_penalty_scale * floor_hinge

    hand_floor_penalty = np.zeros(T, dtype=np.float32)
    if config.hand_floor_penalty_scale > 0.0 and config.hand_floor_penalty_geom_ids:
        radii = np.asarray([float(model.geom_size[gid, 0]) for gid in config.hand_floor_penalty_geom_ids], dtype=np.float32)
        clearance = sim["geom_xpos"][:, config.hand_floor_penalty_geom_ids, 2] - radii.reshape(1, -1)
        hinge = np.maximum(config.hand_floor_penalty_margin_m - clearance, 0.0)
        hand_floor_penalty = -config.hand_floor_penalty_scale * hinge.sum(axis=1)

    stability_penalty = np.zeros(T, dtype=np.float32)
    if config.stability_penalty_scale > 0.0:
        pelvis_z = sim["xpos"][:, 1, 2]
        below = np.maximum(config.stability_penalty_threshold - pelvis_z, 0.0)
        stability_penalty = -config.stability_penalty_scale * below

    arrays = {
        "qpos_rew": qpos_rew.detach().numpy(),
        "qvel_rew": qvel_rew.detach().numpy(),
        **{k: v.detach().numpy() for k, v in local_terms.items()},
        "task_body_rew": task_body_rew.detach().numpy(),
        "task_obj_rew": task_obj_rew.detach().numpy(),
        "contact_hdmi_rew": contact_hdmi_rew.detach().numpy(),
        "ctrl_ref_guard_rew": ctrl_ref_guard_rew.detach().numpy(),
        "robot_object_penalty": robot_object_penalty,
        "hand_floor_penalty": hand_floor_penalty,
        "hand_object_deep_penalty": hand_object_deep_penalty,
        "object_lift_rew": object_lift_rew,
        "object_floor_penalty": object_floor_penalty,
        "stability_penalty": stability_penalty,
    }
    total = np.zeros(T, dtype=np.float32)
    for key, value in arrays.items():
        if key.startswith("local_"):
            continue
        total += value.astype(np.float32)
    arrays["total_reward"] = total

    rows = []
    for t in range(T):
        row = {"variant": variant, "frame": t, "time_s": float(times[t])}
        for key in TERM_KEYS:
            row[key] = float(arrays.get(key, np.zeros(T))[t])
        rows.append(row)

    def stats_for(s: int, e: int) -> dict[str, object]:
        out: dict[str, object] = {"start_frame": s, "end_frame": e, "num_frames": e - s + 1}
        sl = slice(s, e + 1)
        for key in TERM_KEYS:
            vals = np.asarray(arrays.get(key, np.zeros(T))[sl], dtype=np.float64)
            out[f"{key}_mean"] = float(vals.mean())
            out[f"{key}_sum"] = float(vals.sum())
            out[f"{key}_min"] = float(vals.min())
            out[f"{key}_max"] = float(vals.max())
        return out

    summary = {
        "variant": variant,
        "override": override,
        "task": task,
        "npz": str(npz_path.relative_to(REPO) if npz_path.is_relative_to(REPO) else npz_path),
        "T": T,
        "mask_key": mask_key,
        "target_key": target_key,
        "case_window": {"start": start, "end": end},
        "full": stats_for(0, T - 1),
        "case_window_stats": stats_for(start, end),
        "config_scales": {
            "contact_hdmi_gain": config.contact_hdmi_gain,
            "task_obj_pos_rew_scale": config.task_obj_pos_rew_scale,
            "task_obj_rot_rew_scale": config.task_obj_rot_rew_scale,
            "robot_object_penalty_scale": config.robot_object_penalty_scale,
            "hand_floor_penalty_scale": config.hand_floor_penalty_scale,
            "hand_object_deep_penalty_scale": config.hand_object_deep_penalty_scale,
            "object_lift_rew_scale": config.object_lift_rew_scale,
            "object_floor_penalty_scale": config.object_floor_penalty_scale,
            "stability_penalty_scale": config.stability_penalty_scale,
        },
    }
    return rows, summary


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    keys = ["variant", "frame", "time_s", *TERM_KEYS]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True)
    parser.add_argument("--override", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--npz", required=True)
    parser.add_argument("--out-dir", default=str(OUT_ROOT))
    args = parser.parse_args()

    out_dir = _as_path(args.out_dir) / args.variant
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, summary = compute_breakdown(
        args.variant,
        args.override,
        args.task,
        _as_path(args.npz),
    )
    csv_path = out_dir / "reward_terms.csv"
    summary_path = out_dir / "summary.json"
    _write_csv(csv_path, rows)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {csv_path}")
    print(f"Wrote {summary_path}")
    cw = summary["case_window_stats"]
    print(
        json.dumps(
            {
                "variant": args.variant,
                "case_window": summary["case_window"],
                "qpos_mean": cw["qpos_rew_mean"],
                "contact_mean": cw["contact_hdmi_rew_mean"],
                "task_obj_mean": cw["task_obj_rew_mean"],
                "robot_obj_pen_mean": cw["robot_object_penalty_mean"],
                "hand_deep_pen_mean": cw["hand_object_deep_penalty_mean"],
                "total_mean": cw["total_reward_mean"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

