#!/usr/bin/env python3
"""Audit E187 keep22 production config, physics, and recorder-off contracts."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import mujoco
import mujoco_warp as mjwarp
import numpy as np
import warp as wp

from spider.config import Config

REPO = Path(__file__).resolve().parents[5]
RESULTS = REPO / "workspace/core4d/results/E187/s3_prg_audit"
PRODUCTION_OVERRIDES = RESULTS / "production_overrides.json"
SCENE_MANIFEST = (
    REPO
    / "workspace/core4d/results/E186/s2_compound_physics/compound_scene_manifest.tsv"
)
COLLIDER_LOCK = REPO / "workspace/core4d/results/E186/s0_environment/collider_lock.json"
OUTPUT = RESULTS / "production_integration.json"
EXPECTED_CASES_BY_OBJECT = {"bucket003": 5, "bucket004": 4, "bucket007": 13}
HAND_GEOMS = ("lh", "rh")
LOWER_BODY_GEOMS = (
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
ROBOT_OBJECT_GEOMS = HAND_GEOMS + LOWER_BODY_GEOMS


def repo_path(value: str | Path) -> Path:
    """Resolve one repository-relative authority path."""
    path = Path(value)
    return path if path.is_absolute() else REPO / path


def relative(path: Path) -> str:
    """Serialize one repository-relative path."""
    return path.absolute().relative_to(REPO.absolute()).as_posix()


def sha256(path: Path) -> str:
    """Return a streaming SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_tsv(path: Path) -> list[dict[str, str]]:
    """Read a tab-separated authority manifest."""
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def write_immutable(path: Path, payload: bytes) -> None:
    """Create one deterministic evidence artifact or verify byte identity."""
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError(f"immutable E187 integration artifact mismatch: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def object_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    """Count authority rows by frozen object key."""
    return dict(Counter(str(row["object_key"]) for row in rows))


def load_authority() -> tuple[dict[str, Any], list[dict[str, str]], dict[str, Any]]:
    """Load and cross-check the 22 override, scene, grid, and collider authorities."""
    overrides = json.loads(PRODUCTION_OVERRIDES.read_text(encoding="utf-8"))
    scenes = read_tsv(SCENE_MANIFEST)
    lock_payload = json.loads(COLLIDER_LOCK.read_text(encoding="utf-8"))
    lock = {**lock_payload, "file_sha256": sha256(COLLIDER_LOCK)}
    if (
        overrides.get("status") != "PASS"
        or overrides.get("row_count") != 22
        or len(overrides.get("rows", [])) != 22
        or len(scenes) != 22
        or lock.get("status") != "FROZEN"
    ):
        raise RuntimeError("E187 production authority is not frozen keep22")
    if object_counts(overrides["rows"]) != EXPECTED_CASES_BY_OBJECT:
        raise RuntimeError("E187 override object counts changed")
    if object_counts(scenes) != EXPECTED_CASES_BY_OBJECT:
        raise RuntimeError("E186 scene object counts changed")
    if overrides["reward_grid_lock"]["sha256"] != sha256(
        repo_path(overrides["reward_grid_lock"]["path"])
    ):
        raise RuntimeError("E187 reward/grid lock SHA changed")
    scene_by_case = {row["case_id"]: row for row in scenes}
    if [row["case_id"] for row in overrides["rows"]] != [
        row["case_id"] for row in scenes
    ]:
        raise RuntimeError("E187 override order differs from keep22")
    for override in overrides["rows"]:
        scene = scene_by_case[override["case_id"]]
        if (
            override["object_key"] != scene["object_key"]
            or override["scene_sha256"] != scene["effective_scene_sha256"]
            or override["source_override_sha256"] != scene["override_sha256"]
            or sha256(repo_path(override["override_path"]))
            != override["override_sha256"]
        ):
            raise RuntimeError(f"{override['case_id']}: override/scene authority drift")
    return overrides, scenes, lock


def recorder_off_contract() -> dict[str, Any]:
    """Verify query-tape recording is default-off and absent from A2 outputs."""
    config = Config()
    contract = {
        "query_tape_enabled": bool(config.query_tape_enabled),
        "query_tape_record_geometry_state": bool(
            config.query_tape_record_geometry_state
        ),
        "raw_chunks_exists": (RESULTS / "raw_chunks").exists(),
    }
    contract["status"] = (
        "PASS"
        if not contract["query_tape_enabled"]
        and not contract["query_tape_record_geometry_state"]
        and not contract["raw_chunks_exists"]
        else "FAIL"
    )
    return contract


def name_id(model: mujoco.MjModel, kind: mujoco.mjtObj, name: str) -> int:
    """Resolve a required MuJoCo name."""
    identifier = mujoco.mj_name2id(model, kind, name)
    if identifier < 0:
        raise RuntimeError(f"compiled model is missing {name}")
    return identifier


def compile_contract(
    scene: dict[str, str], override: dict[str, Any], collider: dict[str, Any]
) -> dict[str, Any]:
    """Compile one CPU/MJWarp model and verify P plus exact 18×K pairs."""
    case_id = scene["case_id"]
    source_path = repo_path(scene["source_e178_scene_act"])
    effective_path = repo_path(scene["scene_act"])
    if (
        sha256(source_path) != scene["source_e178_scene_sha256"]
        or sha256(effective_path) != scene["effective_scene_sha256"]
        or sha256(repo_path(scene["trajectory"])) != scene["trajectory_sha256"]
        or sha256(repo_path(scene["contact_mask"])) != scene["contact_mask_sha256"]
    ):
        raise RuntimeError(f"{case_id}: scene/input SHA changed")

    grid_path = repo_path(override["grid_manifest"])
    if sha256(grid_path) != override["grid_manifest_sha256"]:
        raise RuntimeError(f"{case_id}: selected grid manifest SHA changed")
    grid = json.loads(grid_path.read_text(encoding="utf-8"))
    grid_payload = repo_path(grid["grid"]["path"])
    if (
        grid.get("status") != "GRID_FROZEN"
        or grid["validation"]["status"] != "PASS"
        or sha256(grid_payload) != grid["grid"]["sha256"]
        or grid["source"]["candidate_asset_sha256"]
        != collider["candidate_asset_sha256"]
        or grid["source"]["ordered_parts_sha256"] != collider["ordered_parts_sha256"]
        or float(grid["validation"]["epsilon_grid_m"])
        != float(override["epsilon_grid_m"])
    ):
        raise RuntimeError(f"{case_id}: selected grid/collider contract changed")

    source = mujoco.MjModel.from_xml_path(str(source_path))
    model = mujoco.MjModel.from_xml_path(str(effective_path))
    object_body = name_id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    source_object_body = name_id(source, mujoco.mjtObj.mjOBJ_BODY, "object")
    np.testing.assert_array_equal(
        model.body_mass[object_body], source.body_mass[source_object_body]
    )
    np.testing.assert_array_equal(
        model.body_inertia[object_body], source.body_inertia[source_object_body]
    )
    hull_count = int(collider["actual_hulls"])
    object_ids = [
        name_id(model, mujoco.mjtObj.mjOBJ_GEOM, f"object_collision_{index:03d}")
        for index in range(hull_count)
    ]
    source_primary = name_id(source, mujoco.mjtObj.mjOBJ_GEOM, "object_collision")
    for geom_id in object_ids:
        if int(model.geom_type[geom_id]) != int(mujoco.mjtGeom.mjGEOM_MESH):
            raise RuntimeError(f"{case_id}: compound geom is not a mesh")
        np.testing.assert_array_equal(
            model.geom_friction[geom_id], source.geom_friction[source_primary]
        )
        if int(model.geom_condim[geom_id]) != int(source.geom_condim[source_primary]):
            raise RuntimeError(f"{case_id}: object condim changed")

    robot_ids = {
        name: name_id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        for name in ROBOT_OBJECT_GEOMS
    }
    expected_pairs = {
        (robot_id, object_id)
        for robot_id in robot_ids.values()
        for object_id in object_ids
    }
    observed_pairs: list[tuple[int, int]] = []
    for pair_id in range(model.npair):
        first = int(model.pair_geom1[pair_id])
        second = int(model.pair_geom2[pair_id])
        normalized = (second, first) if first in object_ids else (first, second)
        if normalized in expected_pairs:
            observed_pairs.append(normalized)
    if Counter(observed_pairs) != Counter(dict.fromkeys(expected_pairs, 1)):
        raise RuntimeError(f"{case_id}: robot-object pair matrix is not exact 18×K")

    model_wp = mjwarp.put_model(model)
    contract = {
        "case_id": case_id,
        "object_key": scene["object_key"],
        "object_geom_count": len(object_ids),
        "robot_object_pair_count": len(observed_pairs),
        "expected_robot_object_pair_count": len(expected_pairs),
        "cpu_ngeom": int(model.ngeom),
        "cpu_nmesh": int(model.nmesh),
        "cpu_npair": int(model.npair),
        "warp_ngeom": int(model_wp.ngeom),
        "warp_nmesh": int(model_wp.nmesh),
        "warp_npair": int(model_wp.npair),
        "object_mass": float(model.body_mass[object_body]),
        "object_inertia": [float(value) for value in model.body_inertia[object_body]],
        "grid_manifest_sha256": override["grid_manifest_sha256"],
        "collider_asset_sha256": collider["candidate_asset_sha256"],
        "status": "PASS",
    }
    del model_wp
    gc.collect()
    wp.synchronize()
    return contract


def build(*, compile_models: bool) -> dict[str, Any]:
    """Build deterministic E187 A2 production-integration evidence."""
    overrides, scenes, lock = load_authority()
    scene_by_case = {row["case_id"]: row for row in scenes}
    rows = []
    if compile_models:
        for override in overrides["rows"]:
            case_id = override["case_id"]
            row = compile_contract(
                scene_by_case[case_id],
                override,
                lock["objects"][override["object_key"]],
            )
            rows.append(row)
            print(
                f"PASS {case_id} pairs={row['robot_object_pair_count']} "
                f"warp={row['warp_ngeom']}/{row['warp_nmesh']}/{row['warp_npair']}"
            )
    recorder = recorder_off_contract()
    payload = {
        "schema": "e187_production_integration_v1",
        "experiment_id": "E187",
        "stage": "A2_S3_PRODUCTION_INTEGRATION",
        "gate0_technical_status": "FAIL",
        "progression_authority": "USER_WAIVED",
        "production_overrides": {
            "path": relative(PRODUCTION_OVERRIDES),
            "sha256": sha256(PRODUCTION_OVERRIDES),
        },
        "scene_manifest": {
            "path": relative(SCENE_MANIFEST),
            "sha256": sha256(SCENE_MANIFEST),
        },
        "collider_lock": {
            "path": relative(COLLIDER_LOCK),
            "sha256": lock["file_sha256"],
        },
        "row_count": len(rows),
        "expected_row_count": 22,
        "object_counts": EXPECTED_CASES_BY_OBJECT,
        "recorder_off": recorder,
        "rows": rows,
        "status": (
            "PASS"
            if len(rows) == 22
            and all(row["status"] == "PASS" for row in rows)
            and recorder["status"] == "PASS"
            else "PREFLIGHT_PASS"
            if not compile_models and recorder["status"] == "PASS"
            else "FAIL"
        ),
    }
    return payload


def parse_args() -> argparse.Namespace:
    """Parse preflight or formal compile mode."""
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("preflight", "run"))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    return parser.parse_args()


def main() -> int:
    """Run the production integration preflight or formal audit."""
    args = parse_args()
    payload = build(compile_models=args.mode == "run")
    if args.mode == "run":
        output = args.output if args.output.is_absolute() else REPO / args.output
        encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
        write_immutable(output, encoded)
    print(json.dumps(payload, indent=2, sort_keys=True))
    expected = "PASS" if args.mode == "run" else "PREFLIGHT_PASS"
    return 0 if payload["status"] == expected else 1


if __name__ == "__main__":
    raise SystemExit(main())
