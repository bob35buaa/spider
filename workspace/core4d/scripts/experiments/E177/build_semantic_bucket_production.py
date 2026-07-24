#!/usr/bin/env python3
"""Build the E177 27-case semantic bucket proxy authority."""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
import time
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any

import yaml
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
E176_DIR = HERE.parent / "E176"
E175_DIR = HERE.parent / "E175"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(E176_DIR))
sys.path.insert(0, str(E175_DIR))

import build_lowgeom_production as e176  # noqa: E402
from semantic_bucket_proxy import (  # noqa: E402
    EXPECTED_BOXES_BY_OBJECT,
    build_semantic_boxes,
    fidelity_metrics,
    proxy_xml,
    union_fidelity_metrics,
)


base = e176.base
EXPERIMENT_ID = "E177"
RESULTS = REPO / "workspace/core4d/results/E177"
E174_MANIFEST = e176.E174_MANIFEST
OVERRIDE_DIR = e176.OVERRIDE_DIR
PROXY_TAG = "semanticBucketProxy"
SCENE_NAME = "scene_act_E177_semanticBucketProxy"
METHOD_ID = "E177_semantic_bucket_five_step_no_lid_union_r2"
CELL_ID = "semantic_bucket_proxy"
FIVE_BODY_PROXY_VARIANT = "five_body_steps_no_lid"
CANARY_CASES = (
    "bucket003_20231018_001_p1",
    "bucket004_20231002_021_p1",
    "bucket007_20231020_055_p1",
)
EXPECTED_OBJECT_COUNTS = {
    "bucket003": 9,
    "bucket004": 4,
    "bucket007": 14,
}
MAX_OBJECT_GEOMS = max(EXPECTED_BOXES_BY_OBJECT.values())
MAX_PAIR_COUNT = len(base.ROBOT_OBJECT_GEOMS) * MAX_OBJECT_GEOMS


def replace_proxy(
    object_body: ET.Element,
    generated_xml: str,
) -> list[str]:
    return base.replace_bucket_proxy(object_body, generated_xml)


def replace_robot_object_pairs(
    root: ET.Element,
    object_names: list[str],
) -> int:
    contact = root.find("contact")
    if contact is None:
        contact = ET.SubElement(root, "contact")
    for pair in list(contact.findall("pair")):
        if base.is_robot_object_pair(pair):
            contact.remove(pair)

    for object_index, object_name in enumerate(object_names):
        for hand in base.HAND_GEOMS:
            ET.SubElement(
                contact,
                "pair",
                {
                    "name": (
                        f"{EXPERIMENT_ID}_{hand}_obj{object_index:03d}"
                    ),
                    "geom1": hand,
                    "geom2": object_name,
                    "solref": "0.008 1",
                    "friction": "2 1",
                    "condim": "4",
                },
            )
        for body_geom in base.LOWER_BODY_GEOMS:
            ET.SubElement(
                contact,
                "pair",
                {
                    "name": (
                        f"{EXPERIMENT_ID}_{body_geom}_obj"
                        f"{object_index:03d}"
                    ),
                    "geom1": body_geom,
                    "geom2": object_name,
                    "solref": "0.008 1",
                    "margin": "0",
                    "gap": "0",
                    "condim": "1",
                },
            )
    return len(base.ROBOT_OBJECT_GEOMS) * len(object_names)


def write_override(e174_row: dict[str, str]) -> Path:
    override_id = (
        f"core4d_{EXPERIMENT_ID}_{e174_row['case_id']}_{PROXY_TAG}"
    )
    output = OVERRIDE_DIR / f"{override_id}.yaml"
    payload = {
        "defaults": [e174_row["override_id"], "_self_"],
        "scene_name": SCENE_NAME,
        "object_collision_sdf_mode": "union",
        "object_collision_sdf_batch_groups": True,
    }
    header = (
        "# @package _global_\n"
        f"# Auto-generated for {EXPERIMENT_ID} semantic bucket proxy.\n"
    )
    output.write_text(
        header + yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )
    return output


def audit_override(
    e174_override_id: str,
    e177_override: Path,
) -> tuple[bool, str]:
    config_dir = (REPO / "examples/config").resolve()
    with initialize_config_dir(
        version_base=None,
        config_dir=str(config_dir),
    ):
        baseline = OmegaConf.to_container(
            compose(
                config_name="default",
                overrides=[f"+override={e174_override_id}"],
            ),
            resolve=True,
        )
        candidate = OmegaConf.to_container(
            compose(
                config_name="default",
                overrides=[f"+override={e177_override.stem}"],
            ),
            resolve=True,
        )
    assert isinstance(baseline, dict) and isinstance(candidate, dict)
    allowed = {
        "scene_name",
        "object_collision_sdf_mode",
        "object_collision_sdf_batch_groups",
    }
    failures = [
        f"non_axis_drift:{key}"
        for key in sorted(set(baseline) | set(candidate))
        if key not in allowed and baseline.get(key) != candidate.get(key)
    ]
    if candidate.get("scene_name") != SCENE_NAME:
        failures.append("axis_mismatch:scene_name")
    if candidate.get("object_collision_sdf_mode") != "union":
        failures.append("axis_mismatch:object_collision_sdf_mode")
    if candidate.get("object_collision_sdf_batch_groups") is not True:
        failures.append(
            "axis_mismatch:object_collision_sdf_batch_groups"
        )
    return not failures, ";".join(failures)


def artifact_paths(case_id: str, stage: str) -> dict[str, Any]:
    is_canary = stage == "canary"
    suffix = "canary" if is_canary else "full"
    variant = f"{EXPERIMENT_ID}_{case_id}_{PROXY_TAG}_{suffix}"
    root = (
        f"workspace/core4d/results/{EXPERIMENT_ID}/"
        f"s6_downstream/cem/{stage}"
    )
    return {
        "variant": variant,
        "result_npz": f"{root}/{variant}.npz",
        "outdir_npz": (
            f"{root}/{variant}_outdir/trajectory_mjwp_act.npz"
        ),
        "config_act": f"{root}/{variant}_outdir/config_act.yaml",
        "video": (
            f"workspace/core4d/results/{EXPERIMENT_ID}/"
            "s6_downstream/render/"
            f"{stage}/{variant}.mp4"
        ),
        "log": f"logs/{EXPERIMENT_ID}/cem/{stage}/{variant}.log",
        "cem_samples": 64 if is_canary else 1024,
        "cem_opt_steps": 4 if is_canary else 32,
        "cem_seed": 0,
        "execution_mode": "canary" if is_canary else "production",
        "status": "READY_FOR_CANARY" if is_canary else "READY_FOR_FULL",
    }


def build_scene(
    source: dict[str, str],
    proxy_cache: dict[
        tuple[str, str],
        tuple[str, dict[str, Any], dict[str, float]],
    ],
    trajectory: Path,
    *,
    overwrite: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    case_id = source["case_id"]
    object_key = source["object_key"]
    if object_key not in EXPECTED_BOXES_BY_OBJECT:
        raise ValueError(f"unexpected E177 object: {object_key}")

    source_scene = base.repo_path(source["scene_act"])
    source_root = ET.parse(source_scene).getroot()
    root = copy.deepcopy(source_root)
    object_body = base.find_object_body(root)
    mesh = base.resolve_object_mesh(source_scene)
    cache_key = (object_key, base.sha256(mesh))
    if cache_key not in proxy_cache:
        boxes, metadata = build_semantic_boxes(mesh, object_key)
        generated_xml, _ = proxy_xml(boxes)
        metrics = fidelity_metrics(mesh, boxes)
        metrics.update(union_fidelity_metrics(mesh, boxes))
        if metrics["mesh_to_proxy_p90_m"] > 0.08:
            raise AssertionError(
                f"{object_key} mesh-to-proxy p90 exceeds 8cm: {metrics}"
            )
        if object_key != "bucket004" and (
            metrics["union_mesh_to_proxy_p90_m"] > 0.04
            or metrics["union_proxy_to_mesh_p90_m"] > 0.04
        ):
            raise AssertionError(
                f"{object_key} union fidelity exceeds 4cm: {metrics}"
            )
        proxy_cache[cache_key] = (generated_xml, metadata, metrics)
    generated_xml, metadata, metrics = proxy_cache[cache_key]

    object_names = replace_proxy(object_body, generated_xml)
    expected_count = EXPECTED_BOXES_BY_OBJECT[object_key]
    if len(object_names) != expected_count:
        raise AssertionError(
            f"{object_key} count={len(object_names)} expected={expected_count}"
        )
    xml_pair_count = replace_robot_object_pairs(root, object_names)
    if xml_pair_count != len(base.ROBOT_OBJECT_GEOMS) * expected_count:
        raise AssertionError("pair count is not exact 18×N")
    if xml_pair_count > MAX_PAIR_COUNT:
        raise AssertionError(
            f"{object_key} pair count={xml_pair_count}>{MAX_PAIR_COUNT}"
        )
    if base.stripped_signature(source_root) != base.stripped_signature(root):
        raise AssertionError("scene drift outside proxy/pair axes")

    output = source_scene.with_name(f"{SCENE_NAME}.xml")
    tree = ET.ElementTree(root)
    if output.exists() and not overwrite:
        existing = ET.parse(output).getroot()
        if (
            base.semantic_xml_signature(existing)
            != base.semantic_xml_signature(root)
        ):
            raise FileExistsError(f"different E177 scene exists: {output}")
    else:
        ET.indent(tree, space="  ")
        tree.write(output, encoding="utf-8", xml_declaration=True)

    compiled = base.compiled_contract(output)
    if compiled["object_geom_count"] != expected_count:
        raise AssertionError("XML/compiled geom count drift")
    if compiled["compiled_robot_object_pair_count"] != xml_pair_count:
        raise AssertionError("XML/compiled pair count drift")

    converted_reference = base.runner_reference(
        trajectory,
        base.repo_path(source["config_act"]),
        output,
    )["qpos"]
    diagnostic = base.reference_first5_diagnostic(
        compiled["model"],
        compiled["object_ids"],
        converted_reference,
    )

    snapshot = (
        RESULTS / "scene_snapshot/semantic_bucket_proxy" / case_id
    )
    snapshot.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_scene, snapshot / source_scene.name)
    shutil.copy2(output, snapshot / output.name)

    scene = {
        "case_id": case_id,
        "object_key": object_key,
        "proxy_variant": (
            "solid_mesh_aabb_1box"
            if object_key == "bucket004"
            else FIVE_BODY_PROXY_VARIANT
        ),
        "collision_policy": metadata["collision_policy"],
        "mesh_path": base.rel(mesh),
        "mesh_sha256": base.sha256(mesh),
        "source_scene_act": base.rel(source_scene),
        "source_scene_sha256": base.sha256(source_scene),
        "scene_act": base.rel(output),
        "scene_name": SCENE_NAME,
        "effective_scene_sha256": base.sha256(output),
        "object_geom_count": compiled["object_geom_count"],
        "object_collision_geom_names": ",".join(compiled["object_names"]),
        "compiled_robot_object_pair_count": compiled[
            "compiled_robot_object_pair_count"
        ],
        "expected_robot_object_pair_count": compiled[
            "expected_robot_object_pair_count"
        ],
        "scene_compile_seconds": compiled["compile_seconds"],
        "semantic_proxy_metadata_json": json.dumps(
            metadata,
            sort_keys=True,
        ),
        **metrics,
        **diagnostic,
    }
    geom_rows = [
        {
            "case_id": case_id,
            "object_key": object_key,
            "proxy_variant": scene["proxy_variant"],
            "scene_act": base.rel(output),
            **row,
        }
        for row in compiled["geom_rows"]
    ]
    return scene, geom_rows


def manifest_row(
    source: dict[str, str],
    scene: dict[str, Any],
    override: Path,
    target_scene: Path,
    trajectory: Path,
    contact_mask: Path,
    stage: str,
    assigned_gpu: int,
) -> dict[str, Any]:
    row: dict[str, Any] = dict(source)
    row.update(
        {
            "source_e174_status": source.get("status", ""),
            "source_e174_scene_act": source["scene_act"],
            "source_e174_scene_sha256": source[
                "effective_scene_sha256"
            ],
            "source_e174_override_id": source["override_id"],
            "source_e174_override_sha256": source["override_sha256"],
            "source_e174_result_npz": source["result_npz"],
            "source_e174_outdir_npz": source["outdir_npz"],
            "source_e174_config_act": source["config_act"],
            "spider_method_id": METHOD_ID,
            "cell_id": CELL_ID,
            "proxy_variant": scene["proxy_variant"],
            "collision_policy": scene["collision_policy"],
            "physics_pair_mode": "union",
            "object_collision_sdf_mode": "union",
            "object_collision_sdf_batch_groups": True,
            "object_geom_count": scene["object_geom_count"],
            "object_collision_geom_names": scene[
                "object_collision_geom_names"
            ],
            "compiled_robot_object_pair_count": scene[
                "compiled_robot_object_pair_count"
            ],
            "expected_robot_object_pair_count": scene[
                "expected_robot_object_pair_count"
            ],
            "reference_first5_union_min_distance_m": scene[
                "reference_first5_union_min_distance_m"
            ],
            "reference_first5_lowerbody_penetration_frac": scene[
                "reference_first5_lowerbody_penetration_frac"
            ],
            "reference_first5_any_penetration_frame_frac": scene[
                "reference_first5_any_penetration_frame_frac"
            ],
            "mesh_to_proxy_p90_m": scene["mesh_to_proxy_p90_m"],
            "proxy_to_mesh_p90_m": scene["proxy_to_mesh_p90_m"],
            "union_mesh_to_proxy_p90_m": scene[
                "union_mesh_to_proxy_p90_m"
            ],
            "union_proxy_to_mesh_p90_m": scene[
                "union_proxy_to_mesh_p90_m"
            ],
            "target_scene": base.rel(target_scene),
            "trajectory": base.rel(trajectory),
            "contact_mask": base.rel(contact_mask),
            "assigned_gpu": assigned_gpu,
            "gpu_id": "",
            "override_id": override.stem,
            "override_path": base.rel(override),
            "override_sha256": base.sha256(override),
            "scene_act": scene["scene_act"],
            "scene_name": SCENE_NAME,
            "base_scene_sha256": scene["source_scene_sha256"],
            "effective_scene_sha256": scene["effective_scene_sha256"],
            "trajectory_sha256": base.sha256(trajectory),
            "contact_mask_sha256": base.sha256(contact_mask),
            "failure_mode": "",
            "blocker_detail": "",
            "updated_at": base.now(),
        }
    )
    row.update(artifact_paths(source["case_id"], stage))
    return row


def build(*, overwrite: bool, require_review: bool) -> dict[str, Any]:
    all_sources = base.read_tsv(E174_MANIFEST)
    sources = [
        row
        for row in all_sources
        if row["object_key"] in EXPECTED_OBJECT_COUNTS
    ]
    case_ids = [row["case_id"] for row in sources]
    distribution = Counter(row["object_key"] for row in sources)
    if len(sources) != 27 or len(case_ids) != len(set(case_ids)):
        raise ValueError(
            f"{EXPERIMENT_ID} authority must be 27 unique rows, "
            f"got {len(sources)}"
        )
    if dict(distribution) != EXPECTED_OBJECT_COUNTS:
        raise ValueError(
            f"{EXPERIMENT_ID} object distribution drift: "
            f"{dict(distribution)}"
        )
    if not set(CANARY_CASES).issubset(case_ids):
        raise ValueError(
            f"{EXPERIMENT_ID} canary case missing from authority"
        )

    proxy_cache: dict[
        tuple[str, str],
        tuple[str, dict[str, Any], dict[str, float]],
    ] = {}
    scenes: dict[str, dict[str, Any]] = {}
    scene_audit: list[dict[str, Any]] = []
    geom_manifest: list[dict[str, Any]] = []
    full_rows: list[dict[str, Any]] = []
    canary_rows: list[dict[str, Any]] = []

    for ordinal, source in enumerate(sources, start=1):
        target_scene, trajectory, contact_mask = base.local_authorities(source)
        scene, geom_rows = build_scene(
            source,
            proxy_cache,
            trajectory,
            overwrite=overwrite,
        )
        scenes[source["case_id"]] = scene
        geom_manifest.extend(geom_rows)
        override = write_override(source)
        override_ok, override_detail = audit_override(
            source["override_id"],
            override,
        )
        if not override_ok:
            raise AssertionError(
                f"{source['case_id']} override parity: {override_detail}"
            )
        scene_audit.append(
            {
                **scene,
                "override_id": override.stem,
                "override_sha256": base.sha256(override),
                "override_parity": "pass",
                "status": "pass",
            }
        )
        full_rows.append(
            manifest_row(
                source,
                scene,
                override,
                target_scene,
                trajectory,
                contact_mask,
                "full",
                (ordinal - 1) % 4,
            )
        )

    source_by_case = {row["case_id"]: row for row in sources}
    for ordinal, case_id in enumerate(CANARY_CASES):
        source = source_by_case[case_id]
        scene = scenes[case_id]
        override = OVERRIDE_DIR / (
            f"core4d_{EXPERIMENT_ID}_{case_id}_{PROXY_TAG}.yaml"
        )
        target_scene, trajectory, contact_mask = base.local_authorities(source)
        canary_rows.append(
            manifest_row(
                source,
                scene,
                override,
                target_scene,
                trajectory,
                contact_mask,
                "canary",
                ordinal,
            )
        )

    manifest_root = RESULTS / "s6_downstream/manifests"
    base.write_tsv(
        manifest_root / "semantic_bucket_full_manifest.tsv",
        full_rows,
    )
    base.write_tsv(
        manifest_root / "semantic_bucket_canary_manifest.tsv",
        canary_rows,
    )
    snapshot_root = RESULTS / "scene_snapshot/semantic_bucket_proxy"
    base.write_tsv(snapshot_root / "scene_audit.tsv", scene_audit)
    base.write_tsv(
        snapshot_root / "scene_variant_manifest.tsv",
        geom_manifest,
    )

    review_path = snapshot_root / "proxy_visual_review.tsv"
    previous_reviews = (
        {
            row["object_key"]: row
            for row in base.read_tsv(review_path)
        }
        if review_path.is_file()
        else {}
    )
    review_rows = []
    for object_key in EXPECTED_BOXES_BY_OBJECT:
        exemplar = next(
            row for row in scene_audit if row["object_key"] == object_key
        )
        previous = previous_reviews.get(object_key, {})
        review_rows.append(
            {
                "object_key": object_key,
                "source_scene_task": Path(exemplar["scene_act"]).parent.name,
                "object_category": "bucket",
                "template_status": "manual_review_required",
                "recommended_action": "review_mesh_collision_overlay",
                "proxy_variant": exemplar["proxy_variant"],
                "proxy_policy": exemplar["collision_policy"],
                "mesh_path": exemplar["mesh_path"],
                "scene_act": exemplar["scene_act"],
                "scene_xml": exemplar["scene_act"],
                "object_geom_count": exemplar["object_geom_count"],
                "review_decision": previous.get(
                    "review_decision",
                    "PENDING_CODEX_REVIEW",
                ),
                "reviewer": previous.get("reviewer", ""),
                "reviewed_at": previous.get("reviewed_at", ""),
                "evidence_path": previous.get("evidence_path", ""),
                "notes": previous.get("notes", ""),
            }
        )
    base.write_tsv(review_path, review_rows)
    pending = [
        row["object_key"]
        for row in review_rows
        if row["review_decision"] != "approve_clean"
    ]
    if require_review and pending:
        raise ValueError(f"proxy visual review pending: {pending}")

    by_object = {}
    for object_key in EXPECTED_BOXES_BY_OBJECT:
        exemplar = next(
            row for row in scene_audit if row["object_key"] == object_key
        )
        by_object[object_key] = {
            "object_geom_count": int(exemplar["object_geom_count"]),
            "pair_count": int(
                exemplar["compiled_robot_object_pair_count"]
            ),
            "mesh_to_proxy_p90_m": float(
                exemplar["mesh_to_proxy_p90_m"]
            ),
            "proxy_to_mesh_p90_m": float(
                exemplar["proxy_to_mesh_p90_m"]
            ),
            "union_mesh_to_proxy_p90_m": float(
                exemplar["union_mesh_to_proxy_p90_m"]
            ),
            "union_proxy_to_mesh_p90_m": float(
                exemplar["union_proxy_to_mesh_p90_m"]
            ),
        }
    summary = {
        "created_at": base.now(),
        "experiment_id": EXPERIMENT_ID,
        "source_manifest": base.rel(E174_MANIFEST),
        "source_rows": len(all_sources),
        "full_rows": len(full_rows),
        "canary_rows": len(canary_rows),
        "case_set_equal": set(case_ids)
        == {row["case_id"] for row in full_rows},
        "object_distribution": dict(distribution),
        "max_object_geoms": max(
            int(row["object_geom_count"]) for row in scene_audit
        ),
        "max_pair_count": max(
            int(row["compiled_robot_object_pair_count"])
            for row in scene_audit
        ),
        "scene_audit_pass": sum(
            row["status"] == "pass" for row in scene_audit
        ),
        "unique_proxy_meshes": len(proxy_cache),
        "objects": by_object,
        "review_pending": pending,
        "status": "pass",
    }
    base.write_json(
        manifest_root / "semantic_bucket_manifest_summary.json",
        summary,
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--require-review", action="store_true")
    args = parser.parse_args()
    started = time.perf_counter()
    summary = build(
        overwrite=args.overwrite,
        require_review=args.require_review,
    )
    summary["wall_seconds"] = time.perf_counter() - started
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
