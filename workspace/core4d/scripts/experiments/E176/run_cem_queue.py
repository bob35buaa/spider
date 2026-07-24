#!/usr/bin/env python3
"""E176 queue runner with production multi-geom runtime hard gates."""

from __future__ import annotations

import importlib.util
from collections import Counter
from pathlib import Path
from typing import Any

import mujoco
import yaml


RUNNER = Path(__file__).resolve().parents[1] / "E169/run_cem_queue.py"
SPEC = importlib.util.spec_from_file_location(
    "core4d_e176_cem_queue_runner", RUNNER
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load queue runner: {RUNNER}")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

ROBOT_OBJECT_GEOMS = (
    "lh",
    "rh",
    "left_hip_collision",
    "right_hip_collision",
    "left_thigh_collision",
    "right_thigh_collision",
    "left_shin_collision",
    "right_shin_collision",
    "left_linkage_brace_collision",
    "right_linkage_brace_collision",
    "lf0",
    "lf1",
    "lf2",
    "lf3",
    "rf0",
    "rf1",
    "rf2",
    "rf3",
)
MAX_OBJECT_GEOMS = 9
MAX_ROBOT_OBJECT_PAIRS = len(ROBOT_OBJECT_GEOMS) * MAX_OBJECT_GEOMS
BASE_VALIDATE_RUNTIME_OUTPUTS = MODULE.validate_runtime_outputs


def expected_object_names(row: dict[str, str]) -> list[str]:
    names = [
        value
        for value in row.get(
            "object_collision_geom_names", ""
        ).split(",")
        if value
    ]
    expected_count = int(row["object_geom_count"])
    if not 1 <= expected_count <= MAX_OBJECT_GEOMS:
        raise ValueError(
            f"E176 object geom count must be 1..{MAX_OBJECT_GEOMS}, "
            f"got {expected_count}"
        )
    if len(names) != expected_count or len(names) != len(set(names)):
        raise ValueError(
            f"manifest object names/count mismatch: "
            f"names={len(names)} count={expected_count}"
        )
    if not names or names[0] != "object_collision":
        raise ValueError(f"invalid object geom names: {names[:3]}")
    return names


def compiled_pair_contract(
    model: mujoco.MjModel,
    object_names: list[str],
) -> dict[str, Any]:
    robot_ids = {
        name: mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, name
        )
        for name in ROBOT_OBJECT_GEOMS
    }
    object_ids = {
        name: mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, name
        )
        for name in object_names
    }
    if any(gid < 0 for gid in robot_ids.values()):
        raise ValueError("compiled model missing E176 robot object geom")
    if any(gid < 0 for gid in object_ids.values()):
        raise ValueError("compiled model missing E176 object geom")
    robot_by_id = {gid: name for name, gid in robot_ids.items()}
    object_by_id = {gid: name for name, gid in object_ids.items()}
    observed = []
    for pair_id in range(model.npair):
        first = int(model.pair_geom1[pair_id])
        second = int(model.pair_geom2[pair_id])
        if first in robot_by_id and second in object_by_id:
            observed.append(
                (robot_by_id[first], object_by_id[second])
            )
        elif second in robot_by_id and first in object_by_id:
            observed.append(
                (robot_by_id[second], object_by_id[first])
            )
    expected = {
        (robot_name, object_name)
        for robot_name in ROBOT_OBJECT_GEOMS
        for object_name in object_names
    }
    counts = Counter(observed)
    return {
        "observed_count": len(observed),
        "expected_count": len(expected),
        "missing": expected - set(counts),
        "extras": set(counts) - expected,
        "duplicates": {
            pair for pair, count in counts.items() if count != 1
        },
        "object_ids": [object_ids[name] for name in object_names],
    }


def validate_runtime_outputs(row: dict[str, str]) -> list[str]:
    failures = list(BASE_VALIDATE_RUNTIME_OUTPUTS(row))
    paths = MODULE.output_paths(row)
    if (
        paths["result_npz"].is_file()
        and paths["outdir_npz"].is_file()
        and MODULE.sha256(paths["result_npz"])
        != MODULE.sha256(paths["outdir_npz"])
    ):
        failures.append("artifact_mismatch:root_vs_outdir_npz")

    config = yaml.safe_load(
        paths["config_act"].read_text(encoding="utf-8")
    )
    expected_mode = row.get("object_collision_sdf_mode", "union")
    if config.get("object_collision_sdf_mode") != expected_mode:
        failures.append("config_mismatch:object_collision_sdf_mode")
    expected_batch_groups = str(
        row.get("object_collision_sdf_batch_groups", "true")
    ).strip().lower() in {"1", "true", "yes"}
    if (
        config.get("object_collision_sdf_batch_groups")
        is not expected_batch_groups
    ):
        failures.append(
            "config_mismatch:object_collision_sdf_batch_groups"
        )
    try:
        object_names = expected_object_names(row)
    except (KeyError, TypeError, ValueError) as exc:
        failures.append(f"manifest_invalid:object_geoms:{exc}")
        return failures

    resolved_names = list(
        config.get("object_collision_geom_names", [])
    )
    resolved_ids = [
        int(value)
        for value in config.get("object_collision_geom_ids", [])
    ]
    if resolved_names != object_names:
        failures.append("config_mismatch:object_collision_geom_names")
    if len(resolved_ids) != len(object_names):
        failures.append("config_mismatch:object_collision_geom_ids")

    scene_path = MODULE.repo_path(row["scene_act"])
    model_path = Path(str(config.get("model_path", "")))
    if not model_path.is_file():
        model_path = scene_path
    if not model_path.is_file():
        failures.append("missing:runtime_model_path")
        return failures
    if MODULE.sha256(model_path) != row["effective_scene_sha256"]:
        failures.append("sha_mismatch:runtime_model_path")

    try:
        model = mujoco.MjModel.from_xml_path(str(model_path))
        contract = compiled_pair_contract(model, object_names)
        if resolved_ids != contract["object_ids"]:
            failures.append(
                "config_mismatch:object_collision_geom_ids_order"
            )
        expected_pairs = int(row["expected_robot_object_pair_count"])
        manifest_pairs = int(row["compiled_robot_object_pair_count"])
        if expected_pairs > MAX_ROBOT_OBJECT_PAIRS:
            failures.append("manifest_invalid:robot_object_pair_limit")
        if (
            contract["expected_count"] != expected_pairs
            or contract["observed_count"] != manifest_pairs
            or manifest_pairs != expected_pairs
        ):
            failures.append("compiled_mismatch:robot_object_pair_count")
        if (
            contract["missing"]
            or contract["extras"]
            or contract["duplicates"]
        ):
            failures.append("compiled_mismatch:robot_object_pair_matrix")
        for geom_id in contract["object_ids"]:
            if int(model.geom_type[geom_id]) != int(
                mujoco.mjtGeom.mjGEOM_BOX
            ):
                failures.append("compiled_mismatch:nonbox_object_geom")
                break
    except Exception as exc:  # noqa: BLE001
        failures.append(
            f"compiled_contract_error:{type(exc).__name__}:{exc}"
        )
    return failures


MODULE.validate_runtime_outputs = validate_runtime_outputs


if __name__ == "__main__":
    raise SystemExit(MODULE.main())
