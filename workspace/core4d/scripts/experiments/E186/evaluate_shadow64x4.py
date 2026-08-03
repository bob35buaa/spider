#!/usr/bin/env python3
"""Evaluate exact-C versus production grid on bounded E186 CEM shadows."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import torch
import yaml

from spider.simulators.mjwp_object_distance import GridObjectDistanceRuntime

HERE = Path(__file__).absolute().parent
REPO = HERE.parents[4]
sys.path.insert(0, str(HERE))
import audit_reference_final_rg as audit  # noqa: E402
import bake_canonical_grid_sdf as exact  # noqa: E402

RESULTS = REPO / "workspace/core4d/results/E186"
SCENE_MANIFEST = RESULTS / "s2_compound_physics/compound_scene_manifest.tsv"
COLLIDER_LOCK = RESULTS / "s0_environment/collider_lock.json"
SHADOW_ROOT = RESULTS / "s3_prg_audit/shadow64x4_v7"


def read_rows(path: Path) -> list[dict[str, str]]:
    """Read a tab-separated manifest."""
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def component_traces(
    distances: dict[str, np.ndarray],
    config: dict[str, Any],
    surface_gate: np.ndarray,
    surface_decay: np.ndarray,
) -> dict[str, np.ndarray]:
    """Recompute the three active object-distance reward traces."""
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
    if config["surface_band_score_mode"] != "symmetric_abs":
        raise ValueError("E186 shadow expects symmetric_abs surface score")
    surface = (
        float(config["surface_band_rew_scale"])
        * np.exp(-np.abs(hand) / float(config["surface_band_sigma"]))
        * in_band
        * surface_gate
        * surface_decay
    )
    return {
        "robot_object_penalty": robot,
        "leg_object_penalty": leg,
        "surface_band_rew": surface,
    }


def gate_summary(
    values: np.ndarray,
    *,
    threshold: float,
    max_violation_pct: float,
    hard_floor: float,
) -> dict[str, np.ndarray]:
    """Mirror one optimizer geometry-gate reduction over horizon time."""
    violation = values < threshold
    depth = np.maximum(threshold - values, 0.0)
    minimum = values.min(axis=1)
    violation_pct = violation.mean(axis=1)
    depth_mean = depth.mean(axis=1)
    floor = threshold if np.isnan(hard_floor) else hard_floor
    return {
        "min_sdf": minimum,
        "violation_pct": violation_pct,
        "violation_depth_mean": depth_mean,
        "valid": (minimum >= floor) & (violation_pct <= max_violation_pct),
    }


def geometry_gates(
    distances: dict[str, np.ndarray], config: dict[str, Any]
) -> dict[str, dict[str, np.ndarray]]:
    """Compute body, hand, and leg optimizer gate summaries."""
    return {
        "body": gate_summary(
            distances["body"],
            threshold=float(config["cem_safety_gate_min_sdf_m"]),
            max_violation_pct=float(config["cem_safety_gate_max_violation_pct"]),
            hard_floor=float(config["cem_safety_gate_hard_floor_m"]),
        ),
        "hand": gate_summary(
            distances["hand"],
            threshold=float(config["cem_hand_gate_min_sdf_m"]),
            max_violation_pct=float(config["cem_hand_gate_max_violation_pct"]),
            hard_floor=float(config["cem_hand_gate_hard_floor_m"]),
        ),
        "leg": gate_summary(
            distances["leg"],
            threshold=float(config["cem_leg_gate_min_sdf_m"]),
            max_violation_pct=float(config["cem_leg_gate_max_violation_pct"]),
            hard_floor=float(config["cem_leg_gate_hard_floor_m"]),
        ),
    }


def combine_gates(
    geometry: dict[str, dict[str, np.ndarray]],
    posture_valid: np.ndarray,
    posture_violation: np.ndarray,
) -> dict[str, np.ndarray]:
    """Mirror optimizer max/min combination with the frozen posture gate."""
    axes = list(geometry.values())
    return {
        "valid": np.logical_and.reduce([axis["valid"] for axis in axes])
        & posture_valid,
        "min_sdf": np.minimum.reduce(
            [axis["min_sdf"] for axis in axes] + [-posture_violation]
        ),
        "violation_pct": np.maximum.reduce(
            [axis["violation_pct"] for axis in axes] + [posture_violation]
        ),
        "violation_depth_mean": np.maximum.reduce(
            [axis["violation_depth_mean"] for axis in axes] + [posture_violation]
        ),
    }


def select_elites(
    reward: np.ndarray,
    gate: dict[str, np.ndarray],
    posture_violation: np.ndarray,
    *,
    top_k: int,
    min_valid_frac: float,
    fallback_lambda: float,
) -> tuple[np.ndarray, bool]:
    """Mirror the CEM hard-gate elite index selection."""
    minimum_valid = max(1, int(np.ceil(min_valid_frac * len(reward))))
    valid_indices = np.flatnonzero(gate["valid"])
    if len(valid_indices) >= minimum_valid:
        ordered = valid_indices[np.argsort(reward[valid_indices], kind="stable")[::-1]]
        return ordered[:top_k], False
    fallback = reward - fallback_lambda * posture_violation
    return np.argsort(fallback, kind="stable")[::-1][:top_k], True


def evaluate(case_id: str, *, shadow_root: Path = SHADOW_ROOT) -> dict[str, Any]:
    """Evaluate one completed single-chunk representative shadow."""
    cases = {row["case_id"]: row for row in read_rows(SCENE_MANIFEST)}
    if case_id not in cases:
        raise ValueError(f"case is outside frozen keep22: {case_id}")
    case = cases[case_id]
    run_manifest_path = shadow_root / "manifests" / f"{case_id}.json"
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    if run_manifest.get("status") != "PASS":
        raise RuntimeError("shadow run is not PASS")
    chunk_path = shadow_root / "raw_chunks" / case_id / "chunk_000000.npz"
    config_path = REPO / run_manifest["config"]["path"]
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    model = mujoco.MjModel.from_xml_path(str(REPO / case["scene_act"]))
    object_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    runtime = GridObjectDistanceRuntime.load(
        str(REPO / case["object_distance_manifest"]),
        expected_candidate_asset_sha256=case["collider_asset_sha256"],
        expected_error_bound_m=float(case["object_distance_error_bound_m"]),
        object_body_id=object_body_id,
    )
    lock = json.loads(COLLIDER_LOCK.read_text(encoding="utf-8"))
    parts = exact._load_parts(lock["objects"][case["object_key"]])
    exact_scene = exact._scene(exact.build_exact_union_mesh(parts))
    volumes = exact._convex_union_halfspaces(parts)

    with np.load(chunk_path, allow_pickle=False) as chunk:
        required_geometry = {
            "geometry_geom_xpos",
            "geometry_geom_xmat",
            "geometry_body_xpos",
            "geometry_body_xmat",
        }
        if not required_geometry <= set(chunk.files):
            raise RuntimeError("shadow chunk lacks reward-aligned MJWarp transforms")
        qpos = np.asarray(chunk["geometry_qpos"], dtype=np.float64)
        samples, horizon, nq = qpos.shape
        if horizon != 48 or nq != model.nq:
            raise RuntimeError(f"unexpected shadow qpos shape: {qpos.shape}")
        transforms = (
            np.asarray(chunk["geometry_geom_xpos"]),
            np.asarray(chunk["geometry_geom_xmat"]),
            np.asarray(chunk["geometry_body_xpos"]),
            np.asarray(chunk["geometry_body_xmat"]),
        )
        groups = {
            "body": audit.geom_ids(model, list(config["cem_safety_gate_geom_names"])),
            "hand": audit.geom_ids(model, list(config["cem_hand_gate_geom_names"])),
            "leg": audit.geom_ids(model, list(config["cem_leg_gate_geom_names"])),
        }
        ordered_ids = list(
            dict.fromkeys(gid for group in groups.values() for gid in group)
        )
        grid_batches = []
        exact_batches = []
        for sample_start in range(0, samples, 64):
            sample_end = min(sample_start + 64, samples)
            tensors = [
                torch.from_numpy(
                    value[sample_start:sample_end].reshape(
                        (sample_end - sample_start) * horizon, *value.shape[2:]
                    )
                )
                for value in transforms
            ]
            grid_batch = runtime.per_geom_sdf(
                model,
                ordered_ids,
                geom_xpos=tensors[0],
                geom_xmat=tensors[1],
                body_xpos=tensors[2],
                body_xmat=tensors[3],
            ).numpy()
            exact_batch, domain = audit.exact_per_geom(
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
            if not np.allclose(grid_batch, domain["grid_per_geom"], atol=2e-6, rtol=0):
                raise RuntimeError(
                    "shadow diagnostic grid disagrees with production query"
                )
            grid_batches.append(grid_batch)
            exact_batches.append(exact_batch)
        grid_per_geom = np.concatenate(grid_batches, axis=0)
        exact_per_geom = np.concatenate(exact_batches, axis=0)
        grid_dist = {
            key: audit.group_min(grid_per_geom, ordered_ids, group).reshape(
                samples, horizon
            )
            for key, group in groups.items()
        }
        exact_dist = {
            key: audit.group_min(exact_per_geom, ordered_ids, group).reshape(
                samples, horizon
            )
            for key, group in groups.items()
        }
        surface_gate = np.asarray(chunk["reward_trace_surface_band_gate"], dtype=float)
        surface_decay = np.asarray(
            chunk["reward_trace_surface_band_decay_factor"], dtype=float
        )
        grid_components = component_traces(
            grid_dist, config, surface_gate, surface_decay
        )
        exact_components = component_traces(
            exact_dist, config, surface_gate, surface_decay
        )
        recorded_components = {
            key: np.asarray(chunk[f"reward_trace_{key}"], dtype=float)
            for key in grid_components
        }
        component_reproduction = {
            key: float(np.max(np.abs(grid_components[key] - recorded_components[key])))
            for key in grid_components
        }
        grid_geometry = sum(recorded_components.values()).mean(axis=1)
        exact_geometry = sum(exact_components.values()).mean(axis=1)
        raw_grid_total = np.asarray(chunk["rewards"], dtype=float)
        exact_total = raw_grid_total - grid_geometry + exact_geometry

        epsilon = runtime.epsilon_grid_m
        grid_lower = {key: value - epsilon for key, value in grid_dist.items()}
        grid_geometry_gate = geometry_gates(grid_lower, config)
        exact_geometry_gate = geometry_gates(exact_dist, config)
        posture_valid = np.asarray(chunk["sample_posture_valid_mask"], dtype=bool)
        posture_violation = np.asarray(chunk["sample_posture_violation"], dtype=float)
        grid_gate = combine_gates(grid_geometry_gate, posture_valid, posture_violation)
        exact_gate = combine_gates(
            exact_geometry_gate, posture_valid, posture_violation
        )
        recorded_valid = np.asarray(chunk["sample_gate_valid_mask"], dtype=bool)
        top_k = len(chunk["selected_indices"])
        min_valid_frac = max(
            float(config["cem_safety_gate_min_valid_frac"]),
            float(config["cem_leg_gate_min_valid_frac"]),
            float(config["cem_posture_gate_min_valid_frac"]),
        )
        selection_args = {
            "top_k": top_k,
            "min_valid_frac": min_valid_frac,
            "fallback_lambda": float(config["cem_posture_gate_fallback_lambda"]),
        }
        grid_selected, grid_fallback = select_elites(
            raw_grid_total,
            grid_gate,
            posture_violation,
            **selection_args,
        )
        exact_selected, exact_fallback = select_elites(
            exact_total,
            exact_gate,
            posture_violation,
            **selection_args,
        )
        recorded_selected = np.asarray(chunk["selected_indices"], dtype=np.int64)

    overlap = len(set(grid_selected.tolist()) & set(exact_selected.tolist()))
    active = (np.abs(grid_geometry) > 1e-12) | (np.abs(exact_geometry) > 1e-12)
    component_active = {
        key: int(
            (
                (np.abs(recorded_components[key]).max(axis=1) > 1e-12)
                | (np.abs(exact_components[key]).max(axis=1) > 1e-12)
            ).sum()
        )
        for key in grid_components
    }
    payload = {
        "schema": "e186_shadow_exact_eval_v4",
        "experiment_id": "E186",
        "stage": "S3_shadow64x4",
        "case_id": case_id,
        "object_key": case["object_key"],
        "sample_count": samples,
        "horizon": horizon,
        "finite": bool(
            np.isfinite(grid_per_geom).all()
            and np.isfinite(exact_per_geom).all()
            and np.isfinite(exact_total).all()
        ),
        "grid_component_trace_max_abs_error": component_reproduction,
        "grid_component_trace_reproduction_pass": bool(
            max(component_reproduction.values()) <= 2e-5
        ),
        "geometry_active_candidate_count": int(active.sum()),
        "geometry_component_active_candidate_count": component_active,
        "surface_gate_active_frame_count": int((surface_gate > 0.0).sum()),
        "geometry_delta_abs_p99": float(
            np.quantile(np.abs(exact_geometry - grid_geometry), 0.99)
        ),
        "total_reward_spearman": audit.rank_correlation(raw_grid_total, exact_total),
        "grid_gate_reproduction_pass": bool(
            np.array_equal(grid_gate["valid"], recorded_valid)
        ),
        "combined_false_safe_accept": int(
            (grid_gate["valid"] & ~exact_gate["valid"]).sum()
        ),
        "combined_false_reject": int((~grid_gate["valid"] & exact_gate["valid"]).sum()),
        "grid_valid_count": int(grid_gate["valid"].sum()),
        "exact_valid_count": int(exact_gate["valid"].sum()),
        "posture_valid_count": int(posture_valid.sum()),
        "recorded_valid_count": int(recorded_valid.sum()),
        "grid_axis_valid_count": {
            key: int(value["valid"].sum()) for key, value in grid_geometry_gate.items()
        },
        "exact_axis_valid_count": {
            key: int(value["valid"].sum()) for key, value in exact_geometry_gate.items()
        },
        "grid_axis_min_sdf_min_m": {
            key: float(value["min_sdf"].min())
            for key, value in grid_geometry_gate.items()
        },
        "exact_axis_min_sdf_min_m": {
            key: float(value["min_sdf"].min())
            for key, value in exact_geometry_gate.items()
        },
        "recorded_selected_reproduction_pass": bool(
            np.array_equal(grid_selected, recorded_selected)
        ),
        "grid_fallback": grid_fallback,
        "exact_fallback": exact_fallback,
        "selected_index0_match": bool(grid_selected[0] == exact_selected[0]),
        "topk_overlap_frac": overlap / top_k,
        "exact_selected_valid_frac": float(exact_gate["valid"][exact_selected].mean()),
    }
    payload["status"] = (
        "PASS"
        if payload["finite"]
        and payload["grid_component_trace_reproduction_pass"]
        and payload["geometry_active_candidate_count"] > 0
        and payload["total_reward_spearman"] >= 0.999
        and payload["grid_gate_reproduction_pass"]
        and payload["combined_false_safe_accept"] == 0
        and payload["recorded_selected_reproduction_pass"]
        and payload["exact_selected_valid_frac"] == 1.0
        else "FAIL"
    )
    output = shadow_root / "eval" / f"{case_id}.json"
    audit.write_json(output, payload)
    return payload


def parse_args() -> argparse.Namespace:
    """Parse one representative case evaluation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--shadow-root", type=Path, default=SHADOW_ROOT)
    return parser.parse_args()


def main() -> int:
    """Evaluate a completed bounded shadow."""
    args = parse_args()
    shadow_root = args.shadow_root
    if not shadow_root.is_absolute():
        shadow_root = REPO / shadow_root
    payload = evaluate(args.case_id, shadow_root=shadow_root)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
