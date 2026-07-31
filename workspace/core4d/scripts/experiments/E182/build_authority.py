#!/usr/bin/env python3
"""Build the immutable E182 authority and input snapshot from E178/E181."""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any

from e182_common import (
    REPO_ROOT,
    atomic_json,
    atomic_tsv,
    immutable_copy,
    immutable_copy_tree,
    inventory_digest,
    read_tsv,
    relative_to_repo,
    repo_path,
    sha256_file,
)

DEFAULT_SOURCE = (
    REPO_ROOT
    / "workspace/core4d/results/E178/s6_downstream/manifests"
    / "semantic_bucket_full_manifest.tsv"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "workspace/core4d/results/E182/authority"
DEFAULT_SNAPSHOT_ROOT = REPO_ROOT / "workspace/core4d/results/E182/scene_snapshot"
DEFAULT_AUTHORITY_PATH = (
    REPO_ROOT / "workspace/core4d/results/E182/s0_environment/authority_manifest.json"
)
E178_SCENE_SNAPSHOT_ROOT = (
    REPO_ROOT / "workspace/core4d/results/E178/scene_snapshot/semantic_bucket_proxy"
)
E181_ORACLE_ROOT = REPO_ROOT / "workspace/core4d/results/E181/s1_oracle"
E181_COACD_ROOT = REPO_ROOT / "workspace/core4d/results/E181/s2_coacd"
EXPECTED_SOURCE_SHA256 = (
    "de9a3d165301318049da208aad52b1f5bc4d3ed671631f0097958734a0f022a8"
)
DEV_CASE_IDS = (
    "bucket003_20231018_001_p1",
    "bucket004_20231002_021_p1",
    "bucket007_20231020_055_p1",
)
EXPECTED_OBJECT_COUNTS = {
    "bucket003": 9,
    "bucket004": 4,
    "bucket007": 14,
}
AUTHORITY_FIELDS = (
    "ordinal",
    "case_id",
    "object_key",
    "date",
    "seq",
    "person",
    "retarget_variant_id",
    "selected_retarget_variant_id",
    "rescue_of",
    "target_variant_id",
    "hand_collision_variant_id",
    "contact_mask_label",
    "target_task",
    "target_scene",
    "trajectory",
    "contact_mask",
    "base_scene_sha256",
    "trajectory_sha256",
    "contact_mask_sha256",
    "cem_samples",
    "cem_opt_steps",
    "cem_seed",
)
PROJECTION_FIELDS = (
    "authority_row_index",
    *AUTHORITY_FIELDS,
    "source_e178_scene_act",
    "source_e178_scene_sha256",
    "source_e174_scene_act",
    "source_e174_scene_sha256",
    "source_e178_manifest_sha256",
    "selection_role",
    "e182_status",
)
REQUIRED_SOURCE_FIELDS = (
    *AUTHORITY_FIELDS,
    "scene_act",
    "effective_scene_sha256",
    "source_e174_scene_act",
    "source_e174_scene_sha256",
)


def resolve_object_mesh(scene_path: Path, object_key: str) -> Path:
    """Resolve the visual object mesh from a MuJoCo scene."""
    root = ET.parse(scene_path).getroot()
    compiler = root.find("./compiler")
    mesh_dir = compiler.attrib.get("meshdir", "") if compiler is not None else ""
    matches = [
        element
        for element in root.findall("./asset/mesh")
        if element.attrib.get("name") == object_key
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"{scene_path}: expected one mesh named {object_key}, found {len(matches)}"
        )
    mesh_file = matches[0].attrib.get("file")
    scale = [float(value) for value in matches[0].attrib.get("scale", "1 1 1").split()]
    if not mesh_file or scale != [1.0, 1.0, 1.0]:
        raise RuntimeError(f"{scene_path}: invalid mesh file/scale for {object_key}")
    return (scene_path.parent / mesh_dir / mesh_file).resolve()


def validate_file(
    checks: list[dict[str, Any]],
    *,
    case_id: str,
    field: str,
    path: Path,
    expected_sha256: str,
) -> None:
    """Validate one frozen authority input."""
    if not path.is_file():
        raise FileNotFoundError(f"{case_id}: missing {field}: {path}")
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise RuntimeError(
            f"{case_id}: {field} SHA mismatch {actual} != {expected_sha256}"
        )
    checks.append(
        {
            "case_id": case_id,
            "field": field,
            "path": relative_to_repo(path),
            "sha256": actual,
            "size_bytes": path.stat().st_size,
        }
    )


def validate_source(rows: list[dict[str, str]]) -> dict[str, Any]:
    """Validate budgets, files, historical snapshots, and object meshes."""
    checks: list[dict[str, Any]] = []
    object_meshes: dict[str, dict[str, Any]] = {}
    for row in rows:
        case_id = row["case_id"]
        if (row["cem_samples"], row["cem_opt_steps"], row["cem_seed"]) != (
            "1024",
            "32",
            "0",
        ):
            raise RuntimeError(f"{case_id}: E178 CEM budget is not 1024x32 seed0")
        target_scene = repo_path(row["target_scene"])
        if not target_scene.is_file():
            raise FileNotFoundError(f"{case_id}: missing target scene: {target_scene}")
        validate_file(
            checks,
            case_id=case_id,
            field="trajectory",
            path=repo_path(row["trajectory"]),
            expected_sha256=row["trajectory_sha256"],
        )
        validate_file(
            checks,
            case_id=case_id,
            field="contact_mask",
            path=repo_path(row["contact_mask"]),
            expected_sha256=row["contact_mask_sha256"],
        )
        source_e174 = repo_path(row["source_e174_scene_act"])
        if row["source_e174_scene_sha256"] != row["base_scene_sha256"]:
            raise RuntimeError(f"{case_id}: source/base scene SHA fields disagree")
        validate_file(
            checks,
            case_id=case_id,
            field="source_e174_scene_act",
            path=source_e174,
            expected_sha256=row["base_scene_sha256"],
        )
        source_e178 = repo_path(row["scene_act"])
        validate_file(
            checks,
            case_id=case_id,
            field="source_e178_scene_act",
            path=source_e178,
            expected_sha256=row["effective_scene_sha256"],
        )
        historical_root = E178_SCENE_SNAPSHOT_ROOT / case_id
        validate_file(
            checks,
            case_id=case_id,
            field="snapshot_source_e174_scene_act",
            path=historical_root / source_e174.name,
            expected_sha256=row["base_scene_sha256"],
        )
        validate_file(
            checks,
            case_id=case_id,
            field="snapshot_source_e178_scene_act",
            path=historical_root / source_e178.name,
            expected_sha256=row["effective_scene_sha256"],
        )
        mesh_path = resolve_object_mesh(source_e178, row["object_key"])
        if not mesh_path.is_file():
            raise FileNotFoundError(mesh_path)
        mesh_entry = {
            "path": relative_to_repo(mesh_path),
            "sha256": sha256_file(mesh_path),
            "size_bytes": mesh_path.stat().st_size,
        }
        previous = object_meshes.setdefault(row["object_key"], mesh_entry)
        if previous != mesh_entry:
            raise RuntimeError(f"{row['object_key']}: inconsistent mesh across cases")
    return {"input_checks": checks, "object_meshes": object_meshes}


def project_rows(
    rows: list[dict[str, str]], source_sha256: str
) -> list[dict[str, str]]:
    """Project E178 rows into the immutable E182 schema."""
    dev_cases = set(DEV_CASE_IDS)
    return [
        {
            "authority_row_index": str(index),
            **{field: row[field] for field in AUTHORITY_FIELDS},
            "source_e178_scene_act": row["scene_act"],
            "source_e178_scene_sha256": row["effective_scene_sha256"],
            "source_e174_scene_act": row["source_e174_scene_act"],
            "source_e174_scene_sha256": row["source_e174_scene_sha256"],
            "source_e178_manifest_sha256": source_sha256,
            "selection_role": (
                "selection_eligible"
                if row["case_id"] in dev_cases
                else "evaluation_only_after_freeze"
            ),
            "e182_status": "authority_frozen",
        }
        for index, row in enumerate(rows, start=1)
    ]


def snapshot_inputs(rows: list[dict[str, str]], snapshot_root: Path) -> dict[str, Any]:
    """Create immutable copies of E178 inputs and E181 geometry assets."""
    case_entries: list[dict[str, Any]] = []
    for row in rows:
        case_root = snapshot_root / "authority_cases" / row["case_id"]
        sources = (
            (
                repo_path(row["source_e174_scene_act"]),
                case_root / "source_e174_scene_act.xml",
            ),
            (repo_path(row["scene_act"]), case_root / "source_e178_scene_act.xml"),
            (
                repo_path(row["trajectory"]),
                case_root / f"trajectory{Path(row['trajectory']).suffix}",
            ),
            (
                repo_path(row["contact_mask"]),
                case_root / f"contact_mask{Path(row['contact_mask']).suffix}",
            ),
        )
        for source, destination in sources:
            entry = immutable_copy(source, destination)
            entry["case_id"] = row["case_id"]
            case_entries.append(entry)

    oracle_entries = immutable_copy_tree(
        E181_ORACLE_ROOT, snapshot_root / "e181_oracle"
    )
    coacd_entries = immutable_copy_tree(
        E181_COACD_ROOT,
        snapshot_root / "e181_coacd_candidates",
    )
    payload = {
        "experiment_id": "E182",
        "status": "PASS",
        "authority_case_files": {
            "count": len(case_entries),
            "digest": inventory_digest(case_entries),
            "entries": case_entries,
        },
        "e181_oracle": {
            "count": len(oracle_entries),
            "digest": inventory_digest(oracle_entries),
            "entries": oracle_entries,
        },
        "e181_coacd_candidates": {
            "count": len(coacd_entries),
            "digest": inventory_digest(coacd_entries),
            "candidate_manifests": sum(
                entry["tree_relative_path"].endswith("/manifest.json")
                for entry in coacd_entries
            ),
            "part_objs": sum(
                "/parts/part_" in entry["tree_relative_path"]
                and entry["tree_relative_path"].endswith(".obj")
                for entry in coacd_entries
            ),
            "entries": coacd_entries,
        },
    }
    manifest_path = snapshot_root / "snapshot_manifest.json"
    atomic_json(manifest_path, payload)
    return {
        "path": relative_to_repo(manifest_path),
        "sha256": sha256_file(manifest_path),
        "authority_case_file_count": len(case_entries),
        "oracle_file_count": len(oracle_entries),
        "coacd_file_count": len(coacd_entries),
        "candidate_manifests": payload["e181_coacd_candidates"]["candidate_manifests"],
        "part_objs": payload["e181_coacd_candidates"]["part_objs"],
    }


def build_authority(
    *,
    source_path: Path = DEFAULT_SOURCE,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    snapshot_root: Path = DEFAULT_SNAPSHOT_ROOT,
    expected_sha256: str = EXPECTED_SOURCE_SHA256,
    authority_path: Path | None = None,
    copy_snapshots: bool = True,
) -> dict[str, Any]:
    """Validate E178 and emit E182 full/dev/heldout authority plus snapshots."""
    source_path = source_path.absolute()
    output_root = output_root.absolute()
    snapshot_root = snapshot_root.absolute()
    if authority_path is None:
        authority_path = (
            DEFAULT_AUTHORITY_PATH
            if output_root == DEFAULT_OUTPUT_ROOT.absolute()
            else output_root / "authority_manifest.json"
        )
    source_sha256 = sha256_file(source_path)
    if source_sha256 != expected_sha256:
        raise RuntimeError(
            f"E178 source manifest SHA mismatch: {source_sha256} != {expected_sha256}"
        )
    fields, source_rows = read_tsv(source_path)
    missing = sorted(set(REQUIRED_SOURCE_FIELDS) - set(fields))
    if missing:
        raise RuntimeError(f"E178 source missing fields: {missing}")
    if len(source_rows) != 27:
        raise RuntimeError(f"expected 27 rows, found {len(source_rows)}")
    case_ids = [row["case_id"] for row in source_rows]
    if len(set(case_ids)) != 27:
        raise RuntimeError("E178 source contains duplicate case IDs")
    counts = dict(Counter(row["object_key"] for row in source_rows))
    if counts != EXPECTED_OBJECT_COUNTS:
        raise RuntimeError(f"object distribution mismatch: {counts}")
    if not set(DEV_CASE_IDS).issubset(case_ids):
        raise RuntimeError("dev3 is not a subset of E178 full27")

    input_evidence = validate_source(source_rows)
    projected = project_rows(source_rows, source_sha256)
    dev_order = {case_id: index for index, case_id in enumerate(DEV_CASE_IDS)}
    dev_rows = sorted(
        (row for row in projected if row["case_id"] in dev_order),
        key=lambda row: dev_order[row["case_id"]],
    )
    heldout_rows = [row for row in projected if row["case_id"] not in dev_order]
    if len(dev_rows) != 3 or len(heldout_rows) != 24:
        raise RuntimeError("invalid E182 dev/heldout split")

    split_rows = {"full27": projected, "dev3": dev_rows, "heldout24": heldout_rows}
    outputs: dict[str, dict[str, Any]] = {}
    for split, rows in split_rows.items():
        output_path = output_root / split / "manifest.tsv"
        atomic_tsv(output_path, rows, PROJECTION_FIELDS)
        outputs[split] = {
            "path": relative_to_repo(output_path),
            "sha256": sha256_file(output_path),
            "rows": len(rows),
        }
    snapshot = snapshot_inputs(source_rows, snapshot_root) if copy_snapshots else None
    manifest = {
        "experiment_id": "E182",
        "gate": "S0_authority",
        "status": "PASS",
        "source_manifest": relative_to_repo(source_path),
        "source_manifest_sha256": source_sha256,
        "case_count": 27,
        "case_ids_ordered": case_ids,
        "object_counts": counts,
        "dev_case_ids_ordered": list(DEV_CASE_IDS),
        "heldout_case_ids_ordered": [row["case_id"] for row in heldout_rows],
        "dev_heldout_overlap": [],
        "cem_contract": {"samples": 1024, "opt_steps": 32, "seed": 0},
        "heldout_policy": {
            "pre_freeze": "selection_forbidden",
            "post_freeze": "evaluation_only",
            "full_execution": "required",
        },
        "input_check_count": len(input_evidence["input_checks"]),
        "object_meshes": input_evidence["object_meshes"],
        "outputs": outputs,
        "snapshot": snapshot,
        "input_checks": input_evidence["input_checks"],
    }
    atomic_json(authority_path.absolute(), manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--snapshot-root", type=Path, default=DEFAULT_SNAPSHOT_ROOT)
    parser.add_argument("--expected-sha256", default=EXPECTED_SOURCE_SHA256)
    parser.add_argument("--no-copy-snapshots", action="store_true")
    return parser.parse_args()


def main() -> int:
    """Build and report the E182 authority."""
    args = parse_args()
    manifest = build_authority(
        source_path=args.source,
        output_root=args.output_root,
        snapshot_root=args.snapshot_root,
        expected_sha256=args.expected_sha256,
        copy_snapshots=not args.no_copy_snapshots,
    )
    snapshot = manifest["snapshot"] or {}
    print(
        "E182_AUTHORITY=PASS "
        f"rows={manifest['case_count']} dev=3 heldout=24 "
        f"source_sha={manifest['source_manifest_sha256']} "
        f"snapshot_files={snapshot.get('authority_case_file_count', 0)} "
        f"candidates={snapshot.get('candidate_manifests', 0)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
