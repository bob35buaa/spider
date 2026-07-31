#!/usr/bin/env python3
"""Build the immutable E181 authority projection from E178 Full."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_SOURCE = (
    REPO_ROOT
    / "workspace/core4d/results/E178/s6_downstream/manifests"
    / "semantic_bucket_full_manifest.tsv"
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "workspace/core4d/results/E181/s6_downstream/manifests"
)
DEFAULT_AUTHORITY_PATH = (
    REPO_ROOT / "workspace/core4d/results/E181/s0_environment/authority_manifest.json"
)
E178_SCENE_SNAPSHOT_ROOT = (
    REPO_ROOT / "workspace/core4d/results/E178/scene_snapshot/semantic_bucket_proxy"
)
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
    "e181_status",
)
REQUIRED_SOURCE_FIELDS = (
    *AUTHORITY_FIELDS,
    "scene_act",
    "effective_scene_sha256",
    "source_e174_scene_act",
    "source_e174_scene_sha256",
)


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest for a file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repo_path(value: str) -> Path:
    """Resolve a manifest path relative to the repository."""
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def relative_to_repo(path: Path) -> str:
    """Return a stable repository-relative path when possible."""
    logical_path = path.absolute()
    try:
        return logical_path.relative_to(REPO_ROOT.absolute()).as_posix()
    except ValueError:
        return str(path.resolve())


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    """Write deterministic JSON atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_tsv(path: Path, rows: list[dict[str, str]]) -> None:
    """Write an authority TSV atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(PROJECTION_FIELDS),
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def load_source(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Load the E178 source manifest."""
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if reader.fieldnames is None:
            raise RuntimeError(f"manifest has no header: {path}")
        return list(reader.fieldnames), list(reader)


def resolve_object_mesh(scene_path: Path, object_key: str) -> tuple[Path, list[float]]:
    """Resolve the visual object mesh and scale from a source scene."""
    root = ET.parse(scene_path).getroot()
    compiler = root.find("./compiler")
    mesh_dir = compiler.attrib.get("meshdir", "") if compiler is not None else ""
    candidates = [
        element
        for element in root.findall("./asset/mesh")
        if element.attrib.get("name") == object_key
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            f"{scene_path}: expected one asset mesh named {object_key}, "
            f"found {len(candidates)}"
        )
    element = candidates[0]
    mesh_file = element.attrib.get("file")
    if not mesh_file:
        raise RuntimeError(f"{scene_path}: object mesh has no file attribute")
    scale = [float(value) for value in element.attrib.get("scale", "1 1 1").split()]
    if scale != [1.0, 1.0, 1.0]:
        raise RuntimeError(f"{scene_path}: non-unit object scale {scale}")
    return (scene_path.parent / mesh_dir / mesh_file).resolve(), scale


def validate_inputs(rows: list[dict[str, str]]) -> dict[str, Any]:
    """Validate all frozen source inputs and discover object meshes."""
    checks: list[dict[str, str]] = []
    object_meshes: dict[str, dict[str, Any]] = {}

    def validate_file(
        case_id: str,
        label: str,
        path: Path,
        expected_sha: str,
    ) -> None:
        if not path.is_file():
            raise FileNotFoundError(f"{case_id}: missing {label}: {path}")
        actual_sha = sha256_file(path)
        if actual_sha != expected_sha:
            raise RuntimeError(
                f"{case_id}: {label} SHA mismatch {actual_sha} != {expected_sha}"
            )
        checks.append(
            {
                "case_id": case_id,
                "field": label,
                "path": relative_to_repo(path),
                "sha256": actual_sha,
            }
        )

    for row in rows:
        case_id = row["case_id"]
        target_scene = repo_path(row["target_scene"])
        if not target_scene.is_file():
            raise FileNotFoundError(f"{case_id}: missing target_scene: {target_scene}")
        validate_file(
            case_id,
            "trajectory",
            repo_path(row["trajectory"]),
            row["trajectory_sha256"],
        )
        validate_file(
            case_id,
            "contact_mask",
            repo_path(row["contact_mask"]),
            row["contact_mask_sha256"],
        )

        source_e174_scene = repo_path(row["source_e174_scene_act"])
        if row["source_e174_scene_sha256"] != row["base_scene_sha256"]:
            raise RuntimeError(f"{case_id}: base/source E174 SHA fields disagree")
        validate_file(
            case_id,
            "source_e174_scene_act",
            source_e174_scene,
            row["base_scene_sha256"],
        )
        source_e178_scene = repo_path(row["scene_act"])
        validate_file(
            case_id,
            "source_e178_scene_act",
            source_e178_scene,
            row["effective_scene_sha256"],
        )

        snapshot_root = E178_SCENE_SNAPSHOT_ROOT / case_id
        validate_file(
            case_id,
            "snapshot_source_e174_scene_act",
            snapshot_root / source_e174_scene.name,
            row["base_scene_sha256"],
        )
        validate_file(
            case_id,
            "snapshot_source_e178_scene_act",
            snapshot_root / source_e178_scene.name,
            row["effective_scene_sha256"],
        )

        object_key = row["object_key"]
        mesh_path, scale = resolve_object_mesh(source_e178_scene, object_key)
        if not mesh_path.is_file():
            raise FileNotFoundError(f"{case_id}: missing object mesh: {mesh_path}")
        mesh_entry = {
            "path": relative_to_repo(mesh_path),
            "sha256": sha256_file(mesh_path),
            "scale": scale,
        }
        previous = object_meshes.setdefault(object_key, mesh_entry)
        if previous != mesh_entry:
            raise RuntimeError(
                f"{object_key}: inconsistent visual mesh across authority rows"
            )
    return {
        "input_checks": checks,
        "object_meshes": object_meshes,
    }


def project_rows(
    source_rows: list[dict[str, str]], source_sha256: str
) -> list[dict[str, str]]:
    """Project source rows into the immutable E181 authority schema."""
    projected: list[dict[str, str]] = []
    for row_index, source_row in enumerate(source_rows, start=1):
        projected_row = {
            "authority_row_index": str(row_index),
            **{field: source_row[field] for field in AUTHORITY_FIELDS},
            "source_e178_scene_act": source_row["scene_act"],
            "source_e178_scene_sha256": source_row["effective_scene_sha256"],
            "source_e174_scene_act": source_row["source_e174_scene_act"],
            "source_e174_scene_sha256": source_row["source_e174_scene_sha256"],
            "source_e178_manifest_sha256": source_sha256,
            "e181_status": "authority_frozen",
        }
        projected.append(projected_row)
    return projected


def build_authority(
    source_path: Path = DEFAULT_SOURCE,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    expected_sha256: str = EXPECTED_SOURCE_SHA256,
    authority_path: Path | None = None,
) -> dict[str, Any]:
    """Validate E178 and emit E181 full/dev/heldout authority files."""
    source_path = source_path.absolute()
    output_root = output_root.absolute()
    if authority_path is None:
        authority_path = (
            DEFAULT_AUTHORITY_PATH
            if output_root == DEFAULT_OUTPUT_ROOT.absolute()
            else output_root / "authority_manifest.json"
        )
    authority_path = authority_path.absolute()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    source_sha256 = sha256_file(source_path)
    if source_sha256 != expected_sha256:
        raise RuntimeError(
            f"E178 source manifest SHA mismatch: {source_sha256} != {expected_sha256}"
        )

    source_fields, source_rows = load_source(source_path)
    missing_fields = sorted(set(REQUIRED_SOURCE_FIELDS) - set(source_fields))
    if missing_fields:
        raise RuntimeError(f"E178 source missing authority fields: {missing_fields}")
    if len(source_rows) != 27:
        raise RuntimeError(f"expected 27 E178 rows, found {len(source_rows)}")

    case_ids = [row["case_id"] for row in source_rows]
    if len(case_ids) != len(set(case_ids)):
        raise RuntimeError("E178 source contains duplicate case_id values")
    object_counts = dict(Counter(row["object_key"] for row in source_rows))
    if object_counts != EXPECTED_OBJECT_COUNTS:
        raise RuntimeError(
            f"unexpected object distribution {object_counts} "
            f"!= {EXPECTED_OBJECT_COUNTS}"
        )
    if not set(DEV_CASE_IDS).issubset(case_ids):
        raise RuntimeError("one or more dev3 cases are absent from E178 Full")

    input_evidence = validate_inputs(source_rows)
    projected = project_rows(source_rows, source_sha256)
    dev_lookup = {case_id: index for index, case_id in enumerate(DEV_CASE_IDS)}
    dev_rows = sorted(
        (row for row in projected if row["case_id"] in dev_lookup),
        key=lambda row: dev_lookup[row["case_id"]],
    )
    heldout_rows = [row for row in projected if row["case_id"] not in set(DEV_CASE_IDS)]
    if len(dev_rows) != 3 or len(heldout_rows) != 24:
        raise RuntimeError(
            f"invalid dev/heldout split: {len(dev_rows)}/{len(heldout_rows)}"
        )

    full_path = output_root / "full27.tsv"
    dev_path = output_root / "dev3.tsv"
    heldout_path = output_root / "heldout24.tsv"
    atomic_tsv(full_path, projected)
    atomic_tsv(dev_path, dev_rows)
    atomic_tsv(heldout_path, heldout_rows)
    source_sha_path = output_root / "e178_source_manifest.sha256"
    source_sha_path.write_text(
        f"{source_sha256}  {relative_to_repo(source_path)}\n",
        encoding="utf-8",
    )

    manifest = {
        "experiment_id": "E181",
        "gate": "S0_authority",
        "status": "PASS",
        "source_manifest": relative_to_repo(source_path),
        "source_manifest_sha256": source_sha256,
        "authority_fields": list(AUTHORITY_FIELDS),
        "projection_fields": list(PROJECTION_FIELDS),
        "case_count": len(projected),
        "case_ids_ordered": case_ids,
        "object_counts": object_counts,
        "dev_case_ids_ordered": list(DEV_CASE_IDS),
        "heldout_case_ids_ordered": [row["case_id"] for row in heldout_rows],
        "dev_heldout_overlap": sorted(
            set(DEV_CASE_IDS) & {row["case_id"] for row in heldout_rows}
        ),
        "input_check_count": len(input_evidence["input_checks"]),
        "object_meshes": input_evidence["object_meshes"],
        "outputs": {
            "full27": {
                "path": relative_to_repo(full_path),
                "sha256": sha256_file(full_path),
                "rows": 27,
            },
            "dev3": {
                "path": relative_to_repo(dev_path),
                "sha256": sha256_file(dev_path),
                "rows": 3,
            },
            "heldout24": {
                "path": relative_to_repo(heldout_path),
                "sha256": sha256_file(heldout_path),
                "rows": 24,
            },
        },
        "input_checks": input_evidence["input_checks"],
    }
    atomic_json(authority_path, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--expected-sha256",
        default=EXPECTED_SOURCE_SHA256,
    )
    return parser.parse_args()


def main() -> int:
    """Build authority and print a compact summary."""
    args = parse_args()
    manifest = build_authority(
        source_path=args.source,
        output_root=args.output_root,
        expected_sha256=args.expected_sha256,
    )
    print(
        "E181_AUTHORITY=PASS "
        f"rows={manifest['case_count']} "
        f"dev={len(manifest['dev_case_ids_ordered'])} "
        f"heldout={len(manifest['heldout_case_ids_ordered'])} "
        f"source_sha={manifest['source_manifest_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
