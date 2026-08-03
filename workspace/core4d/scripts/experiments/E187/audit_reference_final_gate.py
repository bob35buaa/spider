#!/usr/bin/env python3
"""Audit E187 conservative G on all 44 reference/E178-final query tapes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
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
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
CORE4D_SCRIPTS = REPO / "workspace/core4d/scripts"
sys.path.insert(0, str(CORE4D_SCRIPTS))

from eval.core.canonical_distance_query import (  # noqa: E402
    build_exact_scene,
    build_exact_union_mesh,
    convex_union_halfspaces,
    exact_per_geom,
    group_min,
    load_frozen_parts,
    resolve_geom_ids,
)

from spider.simulators.mjwp_object_distance import (  # noqa: E402
    GridObjectDistanceRuntime,
)

RESULTS = REPO / "workspace/core4d/results/E187/s3_prg_audit"
PRODUCTION_OVERRIDES = RESULTS / "production_overrides.json"
SCENE_MANIFEST = (
    REPO
    / "workspace/core4d/results/E186/s2_compound_physics/compound_scene_manifest.tsv"
)
COLLIDER_LOCK = REPO / "workspace/core4d/results/E186/s0_environment/collider_lock.json"
E178_MANIFEST = (
    REPO
    / "workspace/core4d/results/E178/s6_downstream/manifests/semantic_bucket_full_manifest.tsv"
)
OUTPUT = RESULTS / "reference_final_gate"


def repo_path(value: str | Path) -> Path:
    """Resolve one repository-relative path."""
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
    """Read a tab-separated manifest."""
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def encode_tsv(rows: list[dict[str, Any]]) -> bytes:
    """Encode rows as a deterministic tab-separated artifact."""
    fields: list[str] = []
    for row in rows:
        fields.extend(key for key in row if key not in fields)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream, fieldnames=fields, delimiter="\t", lineterminator="\n"
    )
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def write_immutable(path: Path, payload: bytes) -> None:
    """Create one evidence artifact or verify byte identity."""
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError(f"immutable E187 gate artifact mismatch: {path}")
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


def reference_qpos(trajectory: Path, scene: Path) -> np.ndarray:
    """Convert a freejoint reference to the frozen scene-act layout."""
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
    """Load E178-final using the frozen world0 convention."""
    qpos = np.asarray(np.load(path, allow_pickle=True)["qpos"], dtype=np.float64)
    return qpos[:, 0] if qpos.ndim == 3 else qpos


def forward_transforms(
    model: mujoco.MjModel, qpos: np.ndarray
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluate deterministic CPU geometry/body transforms for a qpos tape."""
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
    return tuple(
        torch.from_numpy(value)
        for value in (geom_xpos, geom_xmat, body_xpos, body_xmat)
    )


def false_safe_counts(
    exact: dict[str, np.ndarray],
    grid: dict[str, np.ndarray],
    *,
    epsilon_grid_m: float,
    thresholds: dict[str, float],
) -> dict[str, int]:
    """Count conservative-grid accepts rejected by exact-C."""
    keys = {"body", "hand", "leg"}
    if set(exact) != keys or set(grid) != keys or set(thresholds) != keys:
        raise ValueError("G axes must be exactly body/hand/leg")
    exact_axes = {
        key: np.asarray(exact[key]) >= thresholds[key] for key in sorted(keys)
    }
    grid_axes = {
        key: np.asarray(grid[key]) - epsilon_grid_m >= thresholds[key]
        for key in sorted(keys)
    }
    exact_combined = np.logical_and.reduce(list(exact_axes.values()))
    grid_combined = np.logical_and.reduce(list(grid_axes.values()))
    return {
        **{key: int((grid_axes[key] & ~exact_axes[key]).sum()) for key in sorted(keys)},
        "combined": int((grid_combined & ~exact_combined).sum()),
    }


def audit_tape(
    case: dict[str, str],
    source_name: str,
    qpos: np.ndarray,
    config: dict[str, Any],
    runtime: GridObjectDistanceRuntime,
    exact_scene: Any,
    volumes: list[dict[str, np.ndarray]],
    input_path: Path,
) -> dict[str, Any]:
    """Audit one reference/final query tape against conservative G."""
    model = mujoco.MjModel.from_xml_path(str(repo_path(case["scene_act"])))
    transforms = forward_transforms(model, qpos)
    groups = {
        "body": resolve_geom_ids(model, config["cem_safety_gate_geom_names"]),
        "hand": resolve_geom_ids(model, config["cem_hand_gate_geom_names"]),
        "leg": resolve_geom_ids(model, config["cem_leg_gate_geom_names"]),
    }
    ordered_ids = list(dict.fromkeys(gid for group in groups.values() for gid in group))
    grid_per_geom = runtime.per_geom_sdf(
        model,
        ordered_ids,
        geom_xpos=transforms[0],
        geom_xmat=transforms[1],
        body_xpos=transforms[2],
        body_xmat=transforms[3],
    ).numpy()
    exact_values, diagnostic_grid = exact_per_geom(
        runtime,
        model,
        ordered_ids,
        transforms[0],
        transforms[1],
        transforms[2],
        transforms[3],
        exact_scene,
        volumes,
    )
    if not np.allclose(grid_per_geom, diagnostic_grid, atol=2e-6, rtol=0.0):
        raise RuntimeError(
            f"{case['case_id']}: production and diagnostic grid disagree"
        )
    exact = {
        key: group_min(exact_values, ordered_ids, group)
        for key, group in groups.items()
    }
    grid = {
        key: group_min(grid_per_geom, ordered_ids, group)
        for key, group in groups.items()
    }
    thresholds = {
        "body": float(config["cem_safety_gate_min_sdf_m"]),
        "hand": float(config["cem_hand_gate_min_sdf_m"]),
        "leg": float(config["cem_leg_gate_min_sdf_m"]),
    }
    false_safe = false_safe_counts(
        exact,
        grid,
        epsilon_grid_m=runtime.epsilon_grid_m,
        thresholds=thresholds,
    )
    finite = bool(np.isfinite(exact_values).all() and np.isfinite(grid_per_geom).all())
    return {
        "case_id": case["case_id"],
        "object_key": case["object_key"],
        "source": source_name,
        "input_path": relative(input_path),
        "input_sha256": sha256(input_path),
        "frame_count": len(qpos),
        "query_geom_count": len(ordered_ids),
        "epsilon_grid_m": runtime.epsilon_grid_m,
        "finite": finite,
        "distance_abs_error_p99_m": float(
            np.quantile(np.abs(grid_per_geom - exact_values), 0.99)
        ),
        "distance_abs_error_max_m": float(np.max(np.abs(grid_per_geom - exact_values))),
        "body_false_safe_accept": false_safe["body"],
        "hand_false_safe_accept": false_safe["hand"],
        "leg_false_safe_accept": false_safe["leg"],
        "combined_false_safe_accept": false_safe["combined"],
        "status": "PASS" if finite and sum(false_safe.values()) == 0 else "FAIL",
    }


def build(*, run_queries: bool) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Build preflight or formal 44-tape conservative-G evidence."""
    overrides = json.loads(PRODUCTION_OVERRIDES.read_text(encoding="utf-8"))
    scenes = read_tsv(SCENE_MANIFEST)
    e178 = {row["case_id"]: row for row in read_tsv(E178_MANIFEST)}
    collider_lock = json.loads(COLLIDER_LOCK.read_text(encoding="utf-8"))
    if (
        overrides.get("status") != "PASS"
        or len(overrides.get("rows", [])) != 22
        or len(scenes) != 22
        or collider_lock.get("status") != "FROZEN"
    ):
        raise RuntimeError("E187 reference/final authority is not frozen")
    if [row["case_id"] for row in overrides["rows"]] != [
        row["case_id"] for row in scenes
    ]:
        raise RuntimeError("override/scene authority order changed")
    override_by_case = {row["case_id"]: row for row in overrides["rows"]}
    authorities: dict[str, tuple[Any, list[dict[str, np.ndarray]]]] = {}
    rows: list[dict[str, Any]] = []
    if run_queries:
        for case in scenes:
            case_id = case["case_id"]
            override = override_by_case[case_id]
            source = e178[case_id]
            if (
                source["override_sha256"] != case["source_e178_override_sha256"]
                or sha256(repo_path(source["override_path"]))
                != source["override_sha256"]
                or sha256(repo_path(case["scene_act"]))
                != case["effective_scene_sha256"]
                or sha256(repo_path(override["grid_manifest"]))
                != override["grid_manifest_sha256"]
            ):
                raise RuntimeError(f"{case_id}: query input authority changed")
            object_key = case["object_key"]
            if object_key not in authorities:
                parts = load_frozen_parts(collider_lock["objects"][object_key], REPO)
                exact_mesh = build_exact_union_mesh(parts)
                authorities[object_key] = (
                    build_exact_scene(exact_mesh),
                    convex_union_halfspaces(parts),
                )
            exact_scene, volumes = authorities[object_key]
            model = mujoco.MjModel.from_xml_path(str(repo_path(case["scene_act"])))
            object_body_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, "object"
            )
            runtime = GridObjectDistanceRuntime.load(
                str(repo_path(override["grid_manifest"])),
                expected_candidate_asset_sha256=case["collider_asset_sha256"],
                expected_error_bound_m=float(override["epsilon_grid_m"]),
                object_body_id=object_body_id,
            )
            config_path = repo_path(source["config_act"])
            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            tape_inputs = {
                "reference": (
                    repo_path(case["trajectory"]),
                    reference_qpos(
                        repo_path(case["trajectory"]), repo_path(case["scene_act"])
                    ),
                ),
                "e178_final": (
                    repo_path(source["outdir_npz"]),
                    final_qpos(repo_path(source["outdir_npz"])),
                ),
            }
            for source_name, (input_path, qpos) in tape_inputs.items():
                row = audit_tape(
                    case,
                    source_name,
                    qpos,
                    config,
                    runtime,
                    exact_scene,
                    volumes,
                    input_path,
                )
                rows.append(row)
                print(
                    f"{row['status']} {case_id} {source_name} "
                    f"false_safe={sum(row[key] for key in ('body_false_safe_accept', 'hand_false_safe_accept', 'leg_false_safe_accept', 'combined_false_safe_accept'))}"
                )
    false_safe_total = sum(
        int(row[key])
        for row in rows
        for key in (
            "body_false_safe_accept",
            "hand_false_safe_accept",
            "leg_false_safe_accept",
            "combined_false_safe_accept",
        )
    )
    payload = {
        "schema": "e187_reference_final_gate_v1",
        "experiment_id": "E187",
        "stage": "A2_S3_REFERENCE_FINAL_G",
        "gate0_technical_status": "FAIL",
        "progression_authority": "USER_WAIVED",
        "production_overrides_sha256": sha256(PRODUCTION_OVERRIDES),
        "scene_manifest_sha256": sha256(SCENE_MANIFEST),
        "collider_lock_sha256": sha256(COLLIDER_LOCK),
        "tape_count": len(rows),
        "expected_tape_count": 44,
        "finite_tape_count": sum(bool(row["finite"]) for row in rows),
        "false_safe_accept_total": false_safe_total,
        "status": (
            "PASS"
            if len(rows) == 44
            and all(row["status"] == "PASS" for row in rows)
            and false_safe_total == 0
            else "PREFLIGHT_PASS"
            if not run_queries
            else "FAIL"
        ),
    }
    return payload, rows


def parse_args() -> argparse.Namespace:
    """Parse preflight or formal query mode."""
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("preflight", "run"))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    return parser.parse_args()


def main() -> int:
    """Run the 44-tape conservative-G preflight or formal audit."""
    args = parse_args()
    payload, rows = build(run_queries=args.mode == "run")
    if args.mode == "run":
        output = args.output if args.output.is_absolute() else REPO / args.output
        metrics_path = output / "tape_metrics.tsv"
        write_immutable(metrics_path, encode_tsv(rows))
        frozen = {
            **payload,
            "tape_metrics": {
                "path": relative(metrics_path),
                "sha256": sha256(metrics_path),
            },
        }
        aggregate_path = output / "aggregate.json"
        write_immutable(
            aggregate_path,
            (json.dumps(frozen, indent=2, sort_keys=True) + "\n").encode(),
        )
        payload = frozen
    print(json.dumps(payload, indent=2, sort_keys=True))
    expected = "PASS" if args.mode == "run" else "PREFLIGHT_PASS"
    return 0 if payload["status"] == expected else 1


if __name__ == "__main__":
    raise SystemExit(main())
