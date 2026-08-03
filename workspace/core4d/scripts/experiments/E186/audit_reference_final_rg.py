#!/usr/bin/env python3
"""Audit E186 exact-C versus production grid-SDF on reference/E178-final tapes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import torch
import yaml
from scipy.spatial.transform import Rotation
from scipy.stats import spearmanr

from spider.simulators.mjwp_object_distance import (
    GridObjectDistanceRuntime,
    sample_robot_geoms,
)

HERE = Path(__file__).absolute().parent
REPO = HERE.parents[4]
sys.path.insert(0, str(HERE))
import bake_canonical_grid_sdf as exact  # noqa: E402

RESULTS = REPO / "workspace/core4d/results/E186"
SCENE_MANIFEST = RESULTS / "s2_compound_physics/compound_scene_manifest.tsv"
COLLIDER_LOCK = RESULTS / "s0_environment/collider_lock.json"
E178_MANIFEST = (
    REPO / "workspace/core4d/results/E178/s6_downstream/manifests/"
    "semantic_bucket_full_manifest.tsv"
)
OUTPUT = RESULTS / "s3_prg_audit/reference_final"


def now() -> str:
    """Return an ISO timestamp."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def repo_path(value: str | Path) -> Path:
    """Resolve a repository-relative path."""
    path = Path(value)
    return path if path.is_absolute() else REPO / path


def rel(path: Path) -> str:
    """Serialize a lexical repository-relative path."""
    absolute = path.absolute()
    try:
        return str(absolute.relative_to(REPO.absolute()))
    except ValueError:
        return str(absolute)


def sha256(path: Path) -> str:
    """Return a streaming SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_tsv(path: Path) -> list[dict[str, str]]:
    """Read a tab-separated manifest."""
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a stable tab-separated manifest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=fields, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    """Write deterministic JSON evidence."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def person_index(case_id: str) -> int:
    """Map the case suffix to the frozen person axis."""
    return 0 if case_id.endswith("_p1") else 1


def reference_qpos(trajectory: Path, scene: Path) -> np.ndarray:
    """Convert the freejoint reference to the E178/E186 scene-act layout."""
    qpos = np.asarray(np.load(trajectory, allow_pickle=True)["qpos"], dtype=np.float64)
    if qpos.ndim == 3:
        qpos = qpos[:, 0]
    model = mujoco.MjModel.from_xml_path(str(scene))
    if qpos.shape[1] == model.nq:
        return qpos
    nq_robot = model.nq - 6
    object_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    convention = "XYZ"
    meta = scene.with_name("scene_act_meta.json")
    if meta.is_file():
        convention = str(
            json.loads(meta.read_text(encoding="utf-8")).get(
                "euler_convention", convention
            )
        )
    body_quat = model.body_quat[object_id]
    body_rotation = Rotation.from_quat(
        [body_quat[1], body_quat[2], body_quat[3], body_quat[0]]
    )
    object_pos = qpos[:, nq_robot : nq_robot + 3]
    object_quat = qpos[:, nq_robot + 3 : nq_robot + 7]
    object_xyzw = np.column_stack(
        (object_quat[:, 1], object_quat[:, 2], object_quat[:, 3], object_quat[:, 0])
    )
    converted = np.zeros((len(qpos), model.nq), dtype=np.float64)
    converted[:, :nq_robot] = qpos[:, :nq_robot]
    converted[:, nq_robot : nq_robot + 3] = body_rotation.inv().apply(
        object_pos - model.body_pos[object_id][None]
    )
    converted[:, nq_robot + 3 : nq_robot + 6] = (
        body_rotation.inv() * Rotation.from_quat(object_xyzw)
    ).as_euler(convention)
    return converted


def final_qpos(path: Path) -> np.ndarray:
    """Load E178-final using the established world0 output convention."""
    qpos = np.asarray(np.load(path, allow_pickle=True)["qpos"], dtype=np.float64)
    return qpos[:, 0] if qpos.ndim == 3 else qpos


def forward_transforms(
    model: mujoco.MjModel, qpos: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate geometry and body transforms for every tape frame."""
    data = mujoco.MjData(model)
    geom_xpos = np.empty((len(qpos), model.ngeom, 3), dtype=np.float32)
    geom_xmat = np.empty((len(qpos), model.ngeom, 3, 3), dtype=np.float32)
    body_xpos = np.empty((len(qpos), model.nbody, 3), dtype=np.float32)
    body_xmat = np.empty((len(qpos), model.nbody, 3, 3), dtype=np.float32)
    for frame, values in enumerate(qpos):
        data.qpos[:] = values
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        geom_xpos[frame] = data.geom_xpos
        geom_xmat[frame] = data.geom_xmat.reshape(model.ngeom, 3, 3)
        body_xpos[frame] = data.xpos
        body_xmat[frame] = data.xmat.reshape(model.nbody, 3, 3)
    return geom_xpos, geom_xmat, body_xpos, body_xmat


def geom_ids(model: mujoco.MjModel, names: list[str]) -> list[int]:
    """Resolve required robot geom names."""
    ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name) for name in names]
    if any(gid < 0 for gid in ids):
        raise ValueError(f"missing geom names: {names}")
    return ids


def exact_per_geom(
    runtime: GridObjectDistanceRuntime,
    model: mujoco.MjModel,
    ordered_ids: list[int],
    geom_xpos: torch.Tensor,
    geom_xmat: torch.Tensor,
    body_xpos: torch.Tensor,
    body_xmat: torch.Tensor,
    exact_scene: Any,
    volumes: list[dict[str, np.ndarray]],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Evaluate exact-C and grid-domain provenance on production samples."""
    sampled = sample_robot_geoms(
        model, ordered_ids, geom_xpos=geom_xpos, geom_xmat=geom_xmat
    )
    exact_columns = []
    grid_columns = []
    all_inside_columns = []
    exact_winner_inside_columns = []
    grid_winner_inside_columns = []
    inside_errors: list[np.ndarray] = []
    outside_errors: list[np.ndarray] = []
    sample_count = 0
    inside_sample_count = 0
    for gid in ordered_ids:
        sample = sampled[gid]
        local_tensor = runtime.world_to_object(
            sample.points_world, body_xpos, body_xmat
        )
        local = local_tensor.numpy()
        exact_samples = exact._signed_distance(exact_scene, volumes, local)
        grid_samples = runtime.grid.query(local_tensor).numpy()
        in_bounds = runtime.grid.in_bounds_mask(local_tensor).numpy()
        exact_adjusted = exact_samples - sample.radius_m
        grid_adjusted = grid_samples - sample.radius_m
        exact_columns.append(exact_adjusted.min(axis=1))
        grid_columns.append(grid_adjusted.min(axis=1))
        all_inside_columns.append(in_bounds.all(axis=1))
        exact_winner = np.argmin(exact_adjusted, axis=1)
        grid_winner = np.argmin(grid_adjusted, axis=1)
        frame = np.arange(len(exact_adjusted))
        exact_winner_inside_columns.append(in_bounds[frame, exact_winner])
        grid_winner_inside_columns.append(in_bounds[frame, grid_winner])
        error = np.abs(grid_samples - exact_samples)
        inside_errors.append(error[in_bounds])
        outside_errors.append(error[~in_bounds])
        sample_count += int(in_bounds.size)
        inside_sample_count += int(in_bounds.sum())

    def concatenate(values: list[np.ndarray]) -> np.ndarray:
        nonempty = [value.reshape(-1) for value in values if value.size]
        return np.concatenate(nonempty) if nonempty else np.empty(0, dtype=np.float64)

    return np.stack(exact_columns, axis=1), {
        "grid_per_geom": np.stack(grid_columns, axis=1),
        "all_samples_inside": np.stack(all_inside_columns, axis=1),
        "exact_winner_inside": np.stack(exact_winner_inside_columns, axis=1),
        "grid_winner_inside": np.stack(grid_winner_inside_columns, axis=1),
        "inside_errors": concatenate(inside_errors),
        "outside_errors": concatenate(outside_errors),
        "sample_count": sample_count,
        "inside_sample_count": inside_sample_count,
    }


def group_min(
    per_geom: np.ndarray, ordered_ids: list[int], group: list[int]
) -> np.ndarray:
    """Reduce per-geom distance over one configured group."""
    columns = {gid: index for index, gid in enumerate(ordered_ids)}
    return per_geom[:, [columns[gid] for gid in group]].min(axis=1)


def group_winner_flag(
    per_geom: np.ndarray,
    flags: np.ndarray,
    ordered_ids: list[int],
    group: list[int],
) -> np.ndarray:
    """Return a per-frame flag from the minimum-distance geom in one group."""
    columns = {gid: index for index, gid in enumerate(ordered_ids)}
    selected_columns = [columns[gid] for gid in group]
    winner = np.argmin(per_geom[:, selected_columns], axis=1)
    return flags[np.arange(len(per_geom)), np.asarray(selected_columns)[winner]]


def quantile_or_zero(values: np.ndarray, q: float) -> float:
    """Return a finite quantile for possibly empty diagnostic slices."""
    return float(np.quantile(values, q)) if values.size else 0.0


def reward_components(
    distances: dict[str, np.ndarray],
    config: dict[str, Any],
    mask: np.ndarray,
) -> dict[str, np.ndarray]:
    """Recompute every active E178 object-distance reward component."""
    robot_limit = float(config["robot_object_penalty_margin_m"]) - float(
        config["robot_object_penalty_deep_threshold_m"]
    )
    robot = -float(config["robot_object_penalty_scale"]) * np.maximum(
        robot_limit - distances["body"], 0.0
    )
    leg = -float(config["leg_object_penalty_scale"]) * np.maximum(
        float(config["leg_object_penalty_margin_m"]) - distances["leg"], 0.0
    )
    hand = distances["hand"]
    in_band = (hand >= float(config["surface_band_min_sdf_m"])) & (
        hand <= float(config["surface_band_width_m"])
    )
    score = np.exp(-np.abs(hand) / float(config["surface_band_sigma"]))
    surface = float(config["surface_band_rew_scale"]) * score * in_band * mask
    decay_frac = float(config["surface_band_decay_frac"])
    if decay_frac > 0.0:
        decay_start = (1.0 - decay_frac) * len(surface)
        frame = np.arange(len(surface), dtype=np.float64)
        factor = np.clip(
            (len(surface) - frame) / max(len(surface) - decay_start, 1.0), 0, 1
        )
        surface *= factor
    return {
        "robot_object_penalty": robot,
        "leg_object_penalty": leg,
        "surface_band_rew": surface,
        "affected_total": robot + leg + surface,
    }


def reward_error_bounds(config: dict[str, Any], epsilon: float) -> dict[str, float]:
    """Propagate the SDF error budget through active reward components."""
    robot = float(config["robot_object_penalty_scale"]) * epsilon
    leg = float(config["leg_object_penalty_scale"]) * epsilon
    surface_scale = float(config["surface_band_rew_scale"])
    sigma = float(config["surface_band_sigma"])
    boundaries = (
        float(config["surface_band_min_sdf_m"]),
        float(config["surface_band_width_m"]),
    )
    boundary_jump = surface_scale * max(
        np.exp(-abs(boundary) / sigma) for boundary in boundaries
    )
    surface = min(surface_scale, boundary_jump + surface_scale * epsilon / sigma)
    return {
        "robot_object_penalty": robot,
        "leg_object_penalty": leg,
        "surface_band_rew": surface,
        "affected_total": robot + leg + surface,
    }


def rank_correlation(exact_values: np.ndarray, grid_values: np.ndarray) -> float:
    """Return stable Spearman correlation, including constant equal arrays."""
    if np.allclose(exact_values, grid_values, atol=1e-12, rtol=0.0):
        return 1.0
    value = float(spearmanr(exact_values, grid_values).statistic)
    return value if np.isfinite(value) else 0.0


def gate_arrays(
    distances: dict[str, np.ndarray], config: dict[str, Any]
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Return per-axis validity, combined validity, and least-violation score."""
    thresholds = {
        "body": float(config["cem_safety_gate_min_sdf_m"]),
        "hand": float(config["cem_hand_gate_min_sdf_m"]),
        "leg": float(config["cem_leg_gate_min_sdf_m"]),
    }
    axes = {key: distances[key] >= threshold for key, threshold in thresholds.items()}
    combined = axes["body"] & axes["hand"] & axes["leg"]
    violation = sum(
        np.maximum(thresholds[key] - distances[key], 0.0) for key in thresholds
    )
    return axes, combined, violation


def short_rollout(model: mujoco.MjModel, initial_qpos: np.ndarray) -> dict[str, Any]:
    """Run a bounded CPU dynamics check for finite, non-explosive state."""
    data = mujoco.MjData(model)
    data.qpos[:] = initial_qpos
    data.qvel[:] = 0.0
    if model.nu:
        data.ctrl[:] = 0.0
    maximum_qacc = 0.0
    for _ in range(5):
        mujoco.mj_step(model, data)
        maximum_qacc = max(maximum_qacc, float(np.max(np.abs(data.qacc))))
    finite = bool(
        np.isfinite(data.qpos).all()
        and np.isfinite(data.qvel).all()
        and np.isfinite(data.qacc).all()
    )
    return {
        "short_rollout_finite": finite,
        "short_rollout_max_abs_qacc": maximum_qacc,
        "short_rollout_status": "PASS" if finite and maximum_qacc < 1e8 else "FAIL",
    }


def audit_tape(
    case: dict[str, str],
    source: str,
    qpos: np.ndarray,
    config: dict[str, Any],
    runtime: GridObjectDistanceRuntime,
    exact_scene: Any,
    volumes: list[dict[str, np.ndarray]],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Audit one case/source tape and return persisted metrics plus arrays."""
    model = mujoco.MjModel.from_xml_path(str(repo_path(case["scene_act"])))
    transforms = forward_transforms(model, qpos)
    tensors = [torch.from_numpy(value) for value in transforms]
    groups = {
        "body": geom_ids(model, list(config["cem_safety_gate_geom_names"])),
        "hand": geom_ids(model, list(config["cem_hand_gate_geom_names"])),
        "leg": geom_ids(model, list(config["cem_leg_gate_geom_names"])),
    }
    ordered_ids = list(dict.fromkeys(gid for group in groups.values() for gid in group))
    started = time.perf_counter()
    grid_per_geom = runtime.per_geom_sdf(
        model,
        ordered_ids,
        geom_xpos=tensors[0],
        geom_xmat=tensors[1],
        body_xpos=tensors[2],
        body_xmat=tensors[3],
    ).numpy()
    grid_seconds = time.perf_counter() - started
    started = time.perf_counter()
    exact_values, domain = exact_per_geom(
        runtime,
        model,
        ordered_ids,
        tensors[0],
        tensors[1],
        tensors[2],
        tensors[3],
        exact_scene,
        volumes,
    )
    exact_seconds = time.perf_counter() - started
    if not np.allclose(grid_per_geom, domain["grid_per_geom"], atol=2e-6, rtol=0.0):
        raise RuntimeError("diagnostic grid samples disagree with production query")
    epsilon = runtime.epsilon_grid_m
    exact_dist = {
        key: group_min(exact_values, ordered_ids, group)
        for key, group in groups.items()
    }
    grid_dist = {
        key: group_min(grid_per_geom, ordered_ids, group)
        for key, group in groups.items()
    }
    grid_winner_inside = {
        key: group_winner_flag(
            grid_per_geom,
            domain["grid_winner_inside"],
            ordered_ids,
            group,
        )
        for key, group in groups.items()
    }
    lower_dist = {key: values - epsilon for key, values in grid_dist.items()}
    mask_payload = np.load(repo_path(case["contact_mask"]), allow_pickle=True)
    mask = np.asarray(mask_payload["spider_contact_mask_3cm"])[
        :, person_index(case["case_id"]), :
    ].any(axis=1)
    mask = mask[: len(qpos)].astype(np.float64)
    exact_reward = reward_components(exact_dist, config, mask)
    grid_reward = reward_components(grid_dist, config, mask)
    exact_axes, exact_combined, exact_violation = gate_arrays(exact_dist, config)
    grid_axes, grid_combined, grid_violation = gate_arrays(lower_dist, config)
    false_accept = {
        key: int((grid_axes[key] & ~exact_axes[key]).sum()) for key in groups
    }
    errors = {key: np.abs(grid_reward[key] - exact_reward[key]) for key in exact_reward}
    component_bounds = reward_error_bounds(config, epsilon)
    reward_support = {
        key: (exact_reward[key] != 0.0) | (grid_reward[key] != 0.0)
        for key in ("robot_object_penalty", "leg_object_penalty", "surface_band_rew")
    }
    row: dict[str, Any] = {
        "case_id": case["case_id"],
        "object_key": case["object_key"],
        "source": source,
        "frame_count": len(qpos),
        "query_geom_count": len(ordered_ids),
        "finite": bool(
            np.isfinite(grid_per_geom).all() and np.isfinite(exact_values).all()
        ),
        "epsilon_grid_m": epsilon,
        "distance_abs_error_p99_m": float(
            np.quantile(np.abs(grid_per_geom - exact_values), 0.99)
        ),
        "distance_abs_error_max_m": float(np.max(np.abs(grid_per_geom - exact_values))),
        "query_sample_count": domain["sample_count"],
        "inside_grid_sample_count": domain["inside_sample_count"],
        "outside_grid_sample_count": (
            domain["sample_count"] - domain["inside_sample_count"]
        ),
        "inside_grid_sample_abs_error_p99_m": quantile_or_zero(
            domain["inside_errors"], 0.99
        ),
        "inside_grid_sample_abs_error_max_m": quantile_or_zero(
            domain["inside_errors"], 1.0
        ),
        "outside_grid_sample_abs_error_p99_m": quantile_or_zero(
            domain["outside_errors"], 0.99
        ),
        "outside_grid_sample_abs_error_max_m": quantile_or_zero(
            domain["outside_errors"], 1.0
        ),
        "body_false_safe_accept": false_accept["body"],
        "hand_false_safe_accept": false_accept["hand"],
        "leg_false_safe_accept": false_accept["leg"],
        "combined_false_safe_accept": int((grid_combined & ~exact_combined).sum()),
        "combined_false_reject": int((~grid_combined & exact_combined).sum()),
        "exact_combined_valid_count": int(exact_combined.sum()),
        "grid_combined_valid_count": int(grid_combined.sum()),
        "affected_total_spearman": rank_correlation(
            exact_reward["affected_total"], grid_reward["affected_total"]
        ),
        "grid_query_seconds_cpu": grid_seconds,
        "exact_query_seconds_cpu": exact_seconds,
        "grid_query_points_per_second": int(
            sum(
                len(sample.points_world.reshape(-1, 3))
                for sample in sample_robot_geoms(
                    model, ordered_ids, geom_xpos=tensors[0], geom_xmat=tensors[1]
                ).values()
            )
            / max(grid_seconds, 1e-9)
        ),
    }
    for component, error in errors.items():
        row[f"{component}_abs_error_p99"] = float(np.quantile(error, 0.99))
        row[f"{component}_abs_error_max"] = float(np.max(error))
        row[f"{component}_analytic_error_bound"] = component_bounds[component]
        row[f"{component}_p99_within_bound"] = bool(
            row[f"{component}_abs_error_p99"] <= component_bounds[component] + 1e-9
        )
    for axis in groups:
        row[f"{axis}_grid_winner_outside_count"] = int(
            (~grid_winner_inside[axis]).sum()
        )
    for component, support in reward_support.items():
        exact_active = exact_reward[component] != 0.0
        grid_active = grid_reward[component] != 0.0
        support_error = errors[component][support]
        row[f"{component}_support_count"] = int(support.sum())
        row[f"{component}_phantom_active_count"] = int(
            (grid_active & ~exact_active).sum()
        )
        row[f"{component}_missed_active_count"] = int(
            (~grid_active & exact_active).sum()
        )
        row[f"{component}_support_abs_error_p99"] = quantile_or_zero(
            support_error, 0.99
        )
        row[f"{component}_support_abs_error_max"] = quantile_or_zero(support_error, 1.0)
    row["reward_component_bounds_pass"] = bool(
        all(
            row[f"{component}_p99_within_bound"]
            for component in (
                "robot_object_penalty",
                "leg_object_penalty",
                "surface_band_rew",
                "affected_total",
            )
        )
    )
    row["status"] = (
        "PASS"
        if row["finite"]
        and sum(false_accept.values()) == 0
        and row["combined_false_safe_accept"] == 0
        and row["reward_component_bounds_pass"]
        and row["affected_total_spearman"] >= 0.999
        else "FAIL"
    )
    arrays = {
        "exact_reward": exact_reward["affected_total"],
        "grid_reward": grid_reward["affected_total"],
        "exact_combined": exact_combined,
        "grid_combined": grid_combined,
        "exact_violation": exact_violation,
        "grid_violation": grid_violation,
    }
    return row, arrays


def selection_metrics(
    reference: dict[str, np.ndarray], final: dict[str, np.ndarray]
) -> dict[str, Any]:
    """Compare exact/grid selection over paired reference/final candidates."""
    frames = min(len(reference["exact_reward"]), len(final["exact_reward"]))
    exact_reward = np.stack(
        (reference["exact_reward"][:frames], final["exact_reward"][:frames]), axis=1
    )
    grid_reward = np.stack(
        (reference["grid_reward"][:frames], final["grid_reward"][:frames]), axis=1
    )
    exact_valid = np.stack(
        (reference["exact_combined"][:frames], final["exact_combined"][:frames]), axis=1
    )
    grid_valid = np.stack(
        (reference["grid_combined"][:frames], final["grid_combined"][:frames]), axis=1
    )
    exact_violation = np.stack(
        (reference["exact_violation"][:frames], final["exact_violation"][:frames]),
        axis=1,
    )
    grid_violation = np.stack(
        (reference["grid_violation"][:frames], final["grid_violation"][:frames]), axis=1
    )

    def select(
        reward: np.ndarray, valid: np.ndarray, violation: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        selected = np.empty(frames, dtype=np.int64)
        fallback = ~valid.any(axis=1)
        for frame in range(frames):
            if fallback[frame]:
                selected[frame] = int(np.argmin(violation[frame]))
            else:
                score = np.where(valid[frame], reward[frame], -np.inf)
                selected[frame] = int(np.argmax(score))
        return selected, fallback

    exact_selected, exact_fallback = select(exact_reward, exact_valid, exact_violation)
    grid_selected, grid_fallback = select(grid_reward, grid_valid, grid_violation)
    index = np.arange(frames)
    nonfallback = ~grid_fallback
    selected_exact_valid = exact_valid[index, grid_selected]
    return {
        "paired_frame_count": frames,
        "selected_index_match_count": int((exact_selected == grid_selected).sum()),
        "selected_index_match_frac": float((exact_selected == grid_selected).mean()),
        "grid_fallback_count": int(grid_fallback.sum()),
        "exact_fallback_count": int(exact_fallback.sum()),
        "selected_valid_recompute_count": int(nonfallback.sum()),
        "selected_valid_recompute_pass_count": int(
            (selected_exact_valid & nonfallback).sum()
        ),
        "selected_valid_recompute_frac": float(
            (selected_exact_valid & nonfallback).sum() / max(nonfallback.sum(), 1)
        ),
    }


def parse_args() -> argparse.Namespace:
    """Parse formal or bounded diagnostic scope."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--diagnostic", action="store_true")
    parser.add_argument(
        "--grid-root",
        type=Path,
        default=None,
        help="Diagnostic-only grid root; formal scope consumes the S2 manifest.",
    )
    return parser.parse_args()


def main() -> int:
    """Run the formal 22-case reference/E178-final R/G audit."""
    args = parse_args()
    cases = read_tsv(SCENE_MANIFEST)
    if args.case_id:
        requested = set(args.case_id)
        cases = [case for case in cases if case["case_id"] in requested]
        missing = requested - {case["case_id"] for case in cases}
        if missing:
            raise ValueError(f"unknown case ids: {sorted(missing)}")
    if args.diagnostic and not args.case_id:
        raise ValueError("--diagnostic requires at least one --case-id")
    if args.grid_root is not None and not args.diagnostic:
        raise ValueError("--grid-root is allowed only for bounded diagnostics")
    e178 = {row["case_id"]: row for row in read_tsv(E178_MANIFEST)}
    lock = json.loads(COLLIDER_LOCK.read_text(encoding="utf-8"))
    authorities: dict[str, tuple[Any, list[dict[str, np.ndarray]]]] = {}
    tape_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    for case in cases:
        object_key = case["object_key"]
        if object_key not in authorities:
            parts = exact._load_parts(lock["objects"][object_key])
            union = exact.build_exact_union_mesh(parts)
            authorities[object_key] = (
                exact._scene(union),
                exact._convex_union_halfspaces(parts),
            )
        exact_scene, volumes = authorities[object_key]
        grid_manifest = (
            repo_path(args.grid_root) / object_key / "manifest.json"
            if args.grid_root is not None
            else repo_path(case["object_distance_manifest"])
        )
        grid_payload = json.loads(grid_manifest.read_text(encoding="utf-8"))
        runtime = GridObjectDistanceRuntime.load(
            str(grid_manifest),
            expected_candidate_asset_sha256=case["collider_asset_sha256"],
            expected_error_bound_m=float(grid_payload["validation"]["epsilon_grid_m"]),
            object_body_id=mujoco.mj_name2id(
                mujoco.MjModel.from_xml_path(str(repo_path(case["scene_act"]))),
                mujoco.mjtObj.mjOBJ_BODY,
                "object",
            ),
        )
        source = e178[case["case_id"]]
        config = yaml.safe_load(
            repo_path(source["config_act"]).read_text(encoding="utf-8")
        )
        tapes = {
            "reference": reference_qpos(
                repo_path(case["trajectory"]), repo_path(case["scene_act"])
            ),
            "e178_final": final_qpos(repo_path(source["outdir_npz"])),
        }
        arrays: dict[str, dict[str, np.ndarray]] = {}
        for source_name, qpos in tapes.items():
            row, values = audit_tape(
                case,
                source_name,
                qpos,
                config,
                runtime,
                exact_scene,
                volumes,
            )
            tape_rows.append(row)
            arrays[source_name] = values
        selection = selection_metrics(arrays["reference"], arrays["e178_final"])
        rollout_model = mujoco.MjModel.from_xml_path(str(repo_path(case["scene_act"])))
        rollout = short_rollout(rollout_model, tapes["reference"][0])
        case_status = (
            "PASS"
            if all(row["status"] == "PASS" for row in tape_rows[-2:])
            and selection["selected_valid_recompute_frac"] == 1.0
            and rollout["short_rollout_status"] == "PASS"
            else "FAIL"
        )
        case_rows.append(
            {
                "case_id": case["case_id"],
                "object_key": object_key,
                **selection,
                **rollout,
                "status": case_status,
            }
        )
        print(
            f"{case_status} {case['case_id']} "
            f"false_safe={sum(tape_rows[-1][k] + tape_rows[-2][k] for k in ['body_false_safe_accept', 'hand_false_safe_accept', 'leg_false_safe_accept'])} "
            f"rho={min(tape_rows[-1]['affected_total_spearman'], tape_rows[-2]['affected_total_spearman']):.6f}"
        )
    output = args.output if args.output.is_absolute() else REPO / args.output
    write_tsv(output / "tape_metrics.tsv", tape_rows)
    write_tsv(output / "case_metrics.tsv", case_rows)
    aggregate = {
        "schema": "e186_reference_final_rg_audit_v1",
        "experiment_id": "E186",
        "generated_at": now(),
        "case_count": len(case_rows),
        "tape_count": len(tape_rows),
        "status_counts": dict(Counter(row["status"] for row in case_rows)),
        "finite_tapes": sum(bool(row["finite"]) for row in tape_rows),
        "false_safe_accept_total": sum(
            int(row[axis])
            for row in tape_rows
            for axis in (
                "body_false_safe_accept",
                "hand_false_safe_accept",
                "leg_false_safe_accept",
                "combined_false_safe_accept",
            )
        ),
        "minimum_affected_total_spearman": min(
            float(row["affected_total_spearman"]) for row in tape_rows
        ),
        "selected_valid_recompute_frac": float(
            sum(row["selected_valid_recompute_pass_count"] for row in case_rows)
            / max(sum(row["selected_valid_recompute_count"] for row in case_rows), 1)
        ),
        "short_rollout_pass": sum(
            row["short_rollout_status"] == "PASS" for row in case_rows
        ),
        "scope": "DIAGNOSTIC" if args.diagnostic else "FORMAL",
        "grid_root": rel(repo_path(args.grid_root))
        if args.grid_root
        else "S2_MANIFEST",
        "tape_metrics_sha256": sha256(output / "tape_metrics.tsv"),
        "case_metrics_sha256": sha256(output / "case_metrics.tsv"),
    }
    aggregate["status"] = (
        "PASS"
        if aggregate["status_counts"] == {"PASS": 22}
        and aggregate["finite_tapes"] == 44
        and aggregate["false_safe_accept_total"] == 0
        and aggregate["minimum_affected_total_spearman"] >= 0.999
        and aggregate["selected_valid_recompute_frac"] == 1.0
        and aggregate["short_rollout_pass"] == 22
        else "FAIL"
    )
    write_json(output / "aggregate.json", aggregate)
    print(json.dumps(aggregate, indent=2, sort_keys=True))
    return 0 if args.diagnostic or aggregate["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
