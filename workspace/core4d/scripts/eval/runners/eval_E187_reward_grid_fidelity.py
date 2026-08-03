#!/usr/bin/env python3
"""Evaluate E187 continuation reward fidelity on formal reward-aligned tapes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO / "workspace/core4d/scripts"))

from eval.core.canonical_distance_query import (  # noqa: E402
    build_exact_scene,
    build_exact_union_mesh,
    convex_union_halfspaces,
    exact_per_geom,
    group_min,
    load_frozen_parts,
    resolve_geom_ids,
)
from eval.core.distance_fidelity import (  # noqa: E402
    combine_gates,
    conservative_grid_distances,
    continuation_score,
    elite_fidelity_metrics,
    geometry_gates,
    object_reward_components,
    rank_correlation,
    reconstruct_continuation_totals,
    select_elites,
)

from spider.geometry.grid_sdf import sha256_file  # noqa: E402
from spider.simulators.mjwp_object_distance import (  # noqa: E402
    GridObjectDistanceRuntime,
)

RESULTS_E186 = REPO / "workspace/core4d/results/E186"
RESULTS_E187 = REPO / "workspace/core4d/results/E187"
SCENE_MANIFEST = RESULTS_E186 / "s2_compound_physics/compound_scene_manifest.tsv"
COLLIDER_LOCK = RESULTS_E186 / "s0_environment/collider_lock.json"
EXPECTED_COLLIDER_LOCK_SHA256 = (
    "6a20df7c2b47d0d5a5ea104bdd18ab129753db7c3a14e5f82aa0ce93a1a66065"
)
TAPE_ROOT = RESULTS_E186 / "s3_prg_audit/fullbudget_fidelity_v1"
DEFAULT_OUTPUT_DIR = RESULTS_E187 / "s2_canonical_grid_sdf/formal_fidelity_v2/eval"
BUCKET007_2P5_MANIFEST = (
    RESULTS_E187 / "s2_canonical_grid_sdf/candidates/bucket007_2p5mm/manifest.json"
)
FORMAL_CASES = {
    "bucket003_20231018_003_p1": {"record_step": 16, "tape_root": TAPE_ROOT},
    "bucket004_20231002_021_p1": {
        "record_step": 12,
        "tape_root": RESULTS_E187 / "s2_canonical_grid_sdf/bucket004_formal_tape",
    },
    "bucket007_20231020_055_p1": {"record_step": 22, "tape_root": TAPE_ROOT},
}
REQUIRED_CHUNK_KEYS = {
    "rewards",
    "selected_indices",
    "geometry_geom_xpos",
    "geometry_geom_xmat",
    "geometry_body_xpos",
    "geometry_body_xmat",
    "reward_trace_robot_object_penalty",
    "reward_trace_leg_object_penalty",
    "reward_trace_surface_band_rew",
    "reward_trace_surface_band_gate",
    "reward_trace_surface_band_decay_factor",
    "sample_posture_valid_mask",
    "sample_posture_violation",
    "sample_gate_valid_mask",
}


def read_tsv(path: Path) -> list[dict[str, str]]:
    """Read a tab-separated authority manifest."""
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def relative(path: Path) -> str:
    """Render a repository-relative artifact path when possible."""
    absolute = path.absolute()
    try:
        return absolute.relative_to(REPO.absolute()).as_posix()
    except ValueError:
        return str(absolute)


def canonical_digest(value: Any) -> str:
    """Hash a JSON-compatible payload with canonical separators."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def write_immutable_json(path: Path, payload: dict[str, Any]) -> None:
    """Create one immutable JSON result or verify byte-identical reruns."""
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    if path.exists():
        if path.read_bytes() != encoded:
            raise RuntimeError(f"immutable fidelity result mismatch: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def authority(
    case_id: str, grid_manifest_override: Path | None = None
) -> dict[str, Any]:
    """Load and verify all immutable inputs for one formal tape."""
    if case_id not in FORMAL_CASES:
        raise ValueError(f"case is not an available E187 formal tape: {case_id}")
    rows = {row["case_id"]: row for row in read_tsv(SCENE_MANIFEST)}
    if case_id not in rows:
        raise RuntimeError(
            f"case is absent from frozen keep22 scene manifest: {case_id}"
        )
    row = rows[case_id]
    tape_root = FORMAL_CASES[case_id]["tape_root"]
    run_manifest_path = tape_root / "manifests" / f"{case_id}.json"
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    if run_manifest.get("status") != "PASS" or run_manifest.get("case_id") != case_id:
        raise RuntimeError("formal run manifest is not a PASS authority")
    chunk_manifest_path = tape_root / "raw_chunks" / case_id / "chunk_manifest.json"
    chunk_manifest = json.loads(chunk_manifest_path.read_text(encoding="utf-8"))
    expected_step = FORMAL_CASES[case_id]["record_step"]
    expected_budget = {"samples": 1024, "iterations": 32, "seed": 0}
    provenance = chunk_manifest.get("provenance", {})
    if (
        chunk_manifest.get("status") != "COMPLETE"
        or chunk_manifest.get("chunk_count") != 1
        or provenance.get("budget") != expected_budget
        or provenance.get("record_start_sim_step") != expected_step
        or not provenance.get("geometry_state_recorded")
    ):
        raise RuntimeError("formal chunk manifest violates the frozen tape contract")
    if sha256_file(chunk_manifest_path) != run_manifest["chunk_manifest"]["sha256"]:
        raise RuntimeError("formal chunk manifest SHA changed")
    chunk_path = tape_root / "raw_chunks" / case_id / "chunk_000000.npz"
    chunk_entry = chunk_manifest["chunks"][0]
    if (
        chunk_entry.get("qpos_shape") != [1024, 48, 42]
        or chunk_entry.get("reward_shape") != [1024]
        or chunk_path.stat().st_size != int(chunk_entry["size_bytes"])
        or sha256_file(chunk_path) != chunk_entry["sha256"]
        or chunk_manifest["content_sha256"] != run_manifest["chunk_content_sha256"]
    ):
        raise RuntimeError("formal chunk payload changed")
    config_path = REPO / run_manifest["config"]["path"]
    if sha256_file(config_path) != run_manifest["config"]["sha256"]:
        raise RuntimeError("formal config SHA changed")
    scene_path = REPO / row["scene_act"]
    if sha256_file(scene_path) != row["effective_scene_sha256"]:
        raise RuntimeError("compound scene SHA changed")
    if sha256_file(COLLIDER_LOCK) != EXPECTED_COLLIDER_LOCK_SHA256:
        raise RuntimeError("collider lock SHA changed")
    collider_lock = json.loads(COLLIDER_LOCK.read_text(encoding="utf-8"))
    collider = collider_lock["objects"][row["object_key"]]
    if collider["candidate_asset_sha256"] != row["collider_asset_sha256"]:
        raise RuntimeError("scene/collider authority mismatch")
    grid_manifest_path = REPO / row["object_distance_manifest"]
    if grid_manifest_override is not None:
        override = grid_manifest_override.absolute()
        expected = BUCKET007_2P5_MANIFEST.absolute()
        if case_id != "bucket007_20231020_055_p1" or override != expected:
            raise RuntimeError(
                "grid override is outside the unlocked bucket007 2.5mm route"
            )
        grid_manifest_path = override
    elif sha256_file(grid_manifest_path) != row["object_distance_manifest_sha256"]:
        raise RuntimeError("grid manifest SHA changed")
    grid_manifest = json.loads(grid_manifest_path.read_text(encoding="utf-8"))
    if (
        grid_manifest["source"]["candidate_asset_sha256"]
        != row["collider_asset_sha256"]
    ):
        raise RuntimeError("grid/collider authority mismatch")
    return {
        "row": row,
        "run_manifest": run_manifest,
        "run_manifest_path": run_manifest_path,
        "chunk_manifest": chunk_manifest,
        "chunk_manifest_path": chunk_manifest_path,
        "chunk_path": chunk_path,
        "config_path": config_path,
        "scene_path": scene_path,
        "collider": collider,
        "grid_manifest": grid_manifest,
        "grid_manifest_path": grid_manifest_path,
    }


def evaluate(
    case_id: str,
    *,
    batch_size: int = 64,
    grid_manifest_override: Path | None = None,
) -> dict[str, Any]:
    """Evaluate one formal E186 tape under the frozen E187 continuation."""
    auth = authority(case_id, grid_manifest_override)
    row = auth["row"]
    config = yaml.safe_load(auth["config_path"].read_text(encoding="utf-8"))
    if config["surface_band_score_mode"] != "symmetric_abs":
        raise RuntimeError("formal source tape does not use the frozen legacy score")
    model = mujoco.MjModel.from_xml_path(str(auth["scene_path"]))
    object_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    runtime = GridObjectDistanceRuntime.load(
        str(auth["grid_manifest_path"]),
        expected_candidate_asset_sha256=row["collider_asset_sha256"],
        expected_error_bound_m=float(
            auth["grid_manifest"]["validation"]["epsilon_grid_m"]
        ),
        object_body_id=object_body_id,
    )
    parts = load_frozen_parts(auth["collider"], REPO)
    exact_scene = build_exact_scene(build_exact_union_mesh(parts))
    volumes = convex_union_halfspaces(parts)

    with np.load(auth["chunk_path"], allow_pickle=False) as chunk:
        missing = REQUIRED_CHUNK_KEYS - set(chunk.files)
        if missing:
            raise RuntimeError(f"formal tape lacks keys: {sorted(missing)}")
        transforms = tuple(
            np.asarray(chunk[key])
            for key in (
                "geometry_geom_xpos",
                "geometry_geom_xmat",
                "geometry_body_xpos",
                "geometry_body_xmat",
            )
        )
        samples, horizon = transforms[0].shape[:2]
        if (samples, horizon) != (1024, 48):
            raise RuntimeError(
                f"unexpected formal transform shape: {transforms[0].shape}"
            )
        groups = {
            "body": resolve_geom_ids(model, config["cem_safety_gate_geom_names"]),
            "hand": resolve_geom_ids(model, config["cem_hand_gate_geom_names"]),
            "leg": resolve_geom_ids(model, config["cem_leg_gate_geom_names"]),
        }
        ordered_ids = list(
            dict.fromkeys(gid for group in groups.values() for gid in group)
        )
        grid_batches: list[np.ndarray] = []
        exact_batches: list[np.ndarray] = []
        for sample_start in range(0, samples, batch_size):
            sample_end = min(sample_start + batch_size, samples)
            flattened = [
                torch.from_numpy(
                    value[sample_start:sample_end].reshape(
                        (sample_end - sample_start) * horizon, *value.shape[2:]
                    )
                )
                for value in transforms
            ]
            production_grid = runtime.per_geom_sdf(
                model,
                ordered_ids,
                geom_xpos=flattened[0],
                geom_xmat=flattened[1],
                body_xpos=flattened[2],
                body_xmat=flattened[3],
            ).numpy()
            exact_batch, diagnostic_grid = exact_per_geom(
                runtime,
                model,
                ordered_ids,
                *flattened,
                exact_scene,
                volumes,
            )
            if not np.allclose(production_grid, diagnostic_grid, atol=2e-6, rtol=0.0):
                raise RuntimeError("diagnostic grid disagrees with production query")
            grid_batches.append(production_grid)
            exact_batches.append(exact_batch)
        grid_per_geom = np.concatenate(grid_batches, axis=0)
        exact_per_geom_values = np.concatenate(exact_batches, axis=0)
        grid_distances = {
            key: group_min(grid_per_geom, ordered_ids, group).reshape(samples, horizon)
            for key, group in groups.items()
        }
        exact_distances = {
            key: group_min(exact_per_geom_values, ordered_ids, group).reshape(
                samples, horizon
            )
            for key, group in groups.items()
        }
        temporal_gate = np.asarray(chunk["reward_trace_surface_band_gate"])
        temporal_decay = np.asarray(chunk["reward_trace_surface_band_decay_factor"])
        grid_components = object_reward_components(
            grid_distances, config, temporal_gate, temporal_decay
        )
        exact_components = object_reward_components(
            exact_distances, config, temporal_gate, temporal_decay
        )
        old_components = {
            key: np.asarray(chunk[f"reward_trace_{key}"])
            for key in (
                "robot_object_penalty",
                "leg_object_penalty",
                "surface_band_rew",
            )
        }
        totals = reconstruct_continuation_totals(
            np.asarray(chunk["rewards"]),
            old_components,
            grid_components,
            exact_components,
        )
        epsilon = runtime.epsilon_grid_m
        grid_geometry = geometry_gates(
            conservative_grid_distances(grid_distances, epsilon), config
        )
        exact_geometry = geometry_gates(exact_distances, config)
        posture_valid = np.asarray(chunk["sample_posture_valid_mask"], dtype=bool)
        posture_violation = np.asarray(chunk["sample_posture_violation"], dtype=float)
        recorded_old_gate = np.asarray(chunk["sample_gate_valid_mask"], dtype=bool)
        grid_gate = combine_gates(grid_geometry, posture_valid, posture_violation)
        exact_gate = combine_gates(exact_geometry, posture_valid, posture_violation)
        top_k = len(chunk["selected_indices"])
        min_valid_frac = max(
            float(config["cem_safety_gate_min_valid_frac"]),
            float(config["cem_leg_gate_min_valid_frac"]),
            float(config["cem_posture_gate_min_valid_frac"]),
        )
        selection = {
            "top_k": top_k,
            "min_valid_frac": min_valid_frac,
            "fallback_lambda": float(config["cem_posture_gate_fallback_lambda"]),
        }
        grid_selected, grid_fallback = select_elites(
            totals["grid_total"], grid_gate, posture_violation, **selection
        )
        exact_selected, exact_fallback = select_elites(
            totals["exact_total"], exact_gate, posture_violation, **selection
        )
        elite = elite_fidelity_metrics(
            totals["grid_total"],
            totals["exact_total"],
            grid_selected,
            exact_selected,
            exact_gate["valid"],
        )

    grid_surface = grid_components["surface_band_rew"].mean(axis=1)
    exact_surface = exact_components["surface_band_rew"].mean(axis=1)
    grid_score_trace = continuation_score(grid_distances["hand"])
    grid_score_peak = grid_score_trace.max(axis=1)
    grid_score_mean = grid_score_trace.mean(axis=1)
    surface_p99 = float(np.quantile(np.abs(grid_surface - exact_surface), 0.99))
    spread = float(np.quantile(grid_surface, 0.95) - np.quantile(grid_surface, 0.05))
    false_safe = int((grid_gate["valid"] & ~exact_gate["valid"]).sum())
    static = auth["grid_manifest"]["validation"]
    static_gates = static["gates"]
    grid_static_pass = bool(
        static.get("status") == "PASS"
        and static.get("validation_point_count") == 1_000_000
        and static_gates.get("finite")
        and static_gates.get("cpu_cuda_max_le_1e5")
        and static_gates.get("minkowski_reward_support_covered")
        and static["minkowski_support_coverage"]["minimum_actual_padding_m"] >= 0.110
    )
    finite = bool(
        all(np.isfinite(value).all() for value in grid_distances.values())
        and all(np.isfinite(value).all() for value in exact_distances.values())
        and all(np.isfinite(value).all() for value in totals.values())
    )
    gates = {
        "finite": finite,
        "grid_static_validation": grid_static_pass,
        "false_safe_zero": false_safe == 0,
        "grid_selected_exact_valid_100pct": elite["grid_selected_exact_valid_frac"]
        == 1.0,
        "surface_component_p99_le_0p05": surface_p99 <= 0.05,
        "total_reward_spearman_ge_0p99": rank_correlation(
            totals["grid_total"], totals["exact_total"]
        )
        >= 0.99,
        "elite_topk_overlap_ge_0p90": elite["topk_overlap_frac"] >= 0.90,
        "selected0_exact_regret_le_0p005": elite["grid_selected0_exact_regret_frac"]
        <= 0.005,
        "selected0_exact_top1pct": elite["grid_selected0_exact_top1pct"],
    }
    if row["object_key"] == "bucket003":
        gates["bucket003_peak_score_gt_0p05_ge_50pct"] = (
            float((grid_score_peak > 0.05).mean()) >= 0.50
        )
        gates["bucket003_surface_spread_ge_0p01"] = spread >= 0.01
    payload = {
        "schema": "e187_continuation_formal_fidelity_v2",
        "experiment_id": "E187",
        "stage": "A1_S1_S2_FORMAL_FIDELITY",
        "gate0_technical_status": "FAIL",
        "progression_authority": "USER_WAIVED",
        "case_id": case_id,
        "object_key": row["object_key"],
        "sample_count": samples,
        "horizon": horizon,
        "elite_count": top_k,
        "grid": {
            "manifest": relative(auth["grid_manifest_path"]),
            "manifest_sha256": sha256_file(auth["grid_manifest_path"]),
            "payload_sha256": auth["grid_manifest"]["grid"]["sha256"],
            "voxel_size_m": auth["grid_manifest"]["grid"]["voxel_size_m"],
            "epsilon_grid_m": epsilon,
            "minimum_actual_padding_m": static["minkowski_support_coverage"][
                "minimum_actual_padding_m"
            ],
            "cpu_cuda_max_abs_m": static["cpu_cuda_parity"]["max_abs_error_m"],
            "validation_point_count": static["validation_point_count"],
        },
        "authority": {
            "scene": relative(auth["scene_path"]),
            "scene_sha256": sha256_file(auth["scene_path"]),
            "config": relative(auth["config_path"]),
            "config_sha256": sha256_file(auth["config_path"]),
            "chunk": relative(auth["chunk_path"]),
            "chunk_sha256": auth["chunk_manifest"]["chunks"][0]["sha256"],
            "chunk_content_sha256": auth["chunk_manifest"]["content_sha256"],
            "collider_lock_sha256": EXPECTED_COLLIDER_LOCK_SHA256,
            "collider_asset_sha256": row["collider_asset_sha256"],
        },
        "metrics": {
            "finite": finite,
            "false_safe_accept_count": false_safe,
            "false_reject_count": int(
                (~grid_gate["valid"] & exact_gate["valid"]).sum()
            ),
            "grid_valid_count": int(grid_gate["valid"].sum()),
            "exact_valid_count": int(exact_gate["valid"].sum()),
            "surface_component_grid_exact_abs_p99": surface_p99,
            "total_reward_spearman": rank_correlation(
                totals["grid_total"], totals["exact_total"]
            ),
            "grid_continuation_peak_score_gt_0p05_frac": float(
                (grid_score_peak > 0.05).mean()
            ),
            "grid_continuation_mean_score_gt_0p05_frac_diagnostic": float(
                (grid_score_mean > 0.05).mean()
            ),
            "grid_surface_component_p95_minus_p05": spread,
            "grid_surface_component_min": float(grid_surface.min()),
            "grid_surface_component_max": float(grid_surface.max()),
            "grid_fallback": grid_fallback,
            "exact_fallback": exact_fallback,
            **elite,
        },
        "diagnostics": {
            "base_total_min": float(totals["base_total"].min()),
            "base_total_max": float(totals["base_total"].max()),
            "old_object_total_min": float(totals["old_object_total"].min()),
            "old_object_total_max": float(totals["old_object_total"].max()),
            "grid_object_total_min": float(totals["grid_object_total"].min()),
            "grid_object_total_max": float(totals["grid_object_total"].max()),
            "exact_object_total_min": float(totals["exact_object_total"].min()),
            "exact_object_total_max": float(totals["exact_object_total"].max()),
            "grid_axis_valid_count": {
                key: int(value["valid"].sum()) for key, value in grid_geometry.items()
            },
            "exact_axis_valid_count": {
                key: int(value["valid"].sum()) for key, value in exact_geometry.items()
            },
            "recorded_old_gate_reproduction": bool(
                np.array_equal(grid_gate["valid"], recorded_old_gate)
            ),
        },
        "gates": gates,
        "fidelity_status": "PASS" if all(gates.values()) else "FAIL",
        "a1_status": "PENDING_BUCKET004_AND_EFFICIENCY",
    }
    payload["scientific_payload_sha256"] = canonical_digest(
        {key: value for key, value in payload.items() if key != "diagnostics"}
    )
    return payload


def parse_args() -> argparse.Namespace:
    """Parse formal fidelity evaluation arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-id", action="append", choices=sorted(FORMAL_CASES))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--grid-manifest", type=Path)
    parser.add_argument("--preflight", action="store_true")
    return parser.parse_args()


def main() -> int:
    """Evaluate requested formal tapes or verify their immutable authority."""
    args = parse_args()
    case_ids = args.case_id or list(FORMAL_CASES)
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    grid_manifest = args.grid_manifest
    if grid_manifest is not None and not grid_manifest.is_absolute():
        grid_manifest = REPO / grid_manifest
    if grid_manifest is not None and case_ids != ["bucket007_20231020_055_p1"]:
        raise ValueError(
            "the 2.5mm grid override requires only the formal bucket007 case"
        )
    if args.preflight:
        for case_id in case_ids:
            auth = authority(case_id, grid_manifest)
            print(
                f"PREFLIGHT PASS {case_id} chunk={auth['chunk_manifest']['content_sha256']}"
            )
        return 0
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = REPO / output_dir
    statuses = []
    for case_id in case_ids:
        payload = evaluate(
            case_id,
            batch_size=args.batch_size,
            grid_manifest_override=grid_manifest,
        )
        output = output_dir / f"{case_id}.json"
        write_immutable_json(output, payload)
        statuses.append(payload["fidelity_status"])
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if all(status == "PASS" for status in statuses) else 1


if __name__ == "__main__":
    raise SystemExit(main())
