#!/usr/bin/env python3
"""Import the reviewed E145 Bucket004 template evidence into E168."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET


REPO = Path(__file__).resolve().parents[5]
DEFAULT_SOURCE = (
    REPO
    / "workspace/core4d/results/E145/full_nonbox_to_rl_ready"
    / "s2_templates/nonbox_template_review.tsv"
)
DEFAULT_OUT_DIR = (
    REPO
    / "workspace/core4d/results/E168/imported_snapshots"
    / "E145_bucket004_template_review"
)
REFERENCE_SCENE_ROOT = (
    REPO
    / "workspace/core4d/results/E144/E144_full_nonbox_raw_contact"
    / "s2_templates_pipeline_policy_check/scene_root"
)
EVIDENCE_FIELDS = [
    "proxy_scene_xml",
    "evidence_video",
    "evidence_sheet",
    "mesh_collision_evidence_video",
    "mesh_collision_evidence_sheet",
    "orbit_evidence_video",
    "orbit_evidence_sheet",
]


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO / path


def physics_fingerprint(path: Path) -> tuple[dict[str, Any], str]:
    root = ET.parse(path).getroot()
    body = root.find(".//body[@name='object']")
    if body is None:
        raise SystemExit(f"object body missing from template: {path}")
    inertial = body.find("inertial")
    if inertial is None:
        raise SystemExit(f"object inertial missing from template: {path}")
    geoms = []
    for geom in body.findall("geom"):
        name = geom.get("name", "")
        if not name.startswith("object_collision"):
            continue
        geoms.append(
            {
                key: geom.get(key, "")
                for key in (
                    "name",
                    "type",
                    "pos",
                    "quat",
                    "size",
                    "friction",
                    "condim",
                    "contype",
                    "conaffinity",
                )
            }
        )
    geoms.sort(key=lambda geom: geom["name"])
    value = {
        "inertial": {
            key: inertial.get(key, "")
            for key in ("pos", "quat", "mass", "diaginertia")
        },
        "collision_geoms": geoms,
    }
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return value, hashlib.sha256(encoded).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-review-tsv", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    source = args.source_review_tsv.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    with source.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        fields = list(reader.fieldnames or [])
        rows = [
            row
            for row in reader
            if row.get("object_key") == "bucket004"
            and row.get("person") in {"person1", "person2"}
        ]
    rows.sort(key=lambda row: row["person"])
    if [row["person"] for row in rows] != ["person1", "person2"]:
        raise SystemExit(f"expected Bucket004 person1/person2 reviews, got {len(rows)} rows")
    if any(row.get("review_decision") != "approve_clean" for row in rows):
        raise SystemExit("Bucket004 imported template review is not approve_clean")

    artifacts: list[dict[str, Any]] = []
    missing: list[str] = []
    fingerprints: list[dict[str, Any]] = []
    for row in rows:
        for field in EVIDENCE_FIELDS:
            value = row.get(field, "")
            if not value:
                continue
            path = resolve_path(value)
            exists = path.is_file() and path.stat().st_size > 0
            if not exists:
                missing.append(f"{row['person']}:{field}:{value}")
            artifacts.append(
                {
                    "person": row["person"],
                    "field": field,
                    "path": value,
                    "resolved_path": str(path),
                    "exists": exists,
                    "size_bytes": path.stat().st_size if exists else 0,
                    "sha256": sha256(path) if exists else "",
                }
            )
        current_scene = resolve_path(row["proxy_scene_xml"])
        reference_scene = REFERENCE_SCENE_ROOT / row["source_scene_task"] / "scene.xml"
        if current_scene.is_file() and reference_scene.is_file():
            current_value, current_sha = physics_fingerprint(current_scene)
            reference_value, reference_sha = physics_fingerprint(reference_scene)
            if current_value != reference_value:
                raise SystemExit(
                    f"Bucket004 physics fingerprint mismatch for {row['person']}: "
                    f"current={current_sha} reference={reference_sha}"
                )
            fingerprints.append(
                {
                    "person": row["person"],
                    "current_scene": str(current_scene),
                    "current_scene_sha256": sha256(current_scene),
                    "reference_scene": str(reference_scene),
                    "reference_scene_sha256": sha256(reference_scene),
                    "physics_fingerprint_sha256": current_sha,
                    "status": "pass",
                }
            )
            snapshot_scene = (
                REPO
                / "workspace/core4d/results/E168/scene_snapshot"
                / row["source_scene_task"]
                / "scene.xml"
            )
            snapshot_scene.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(current_scene, snapshot_scene)
            if sha256(snapshot_scene) != sha256(current_scene):
                raise SystemExit(f"scene snapshot checksum mismatch: {snapshot_scene}")
            fingerprints[-1]["e168_scene_snapshot"] = str(snapshot_scene)
            fingerprints[-1]["e168_scene_snapshot_sha256"] = sha256(snapshot_scene)
    if missing:
        raise SystemExit("Bucket004 review evidence missing: " + ", ".join(missing))

    out_dir.mkdir(parents=True, exist_ok=True)
    snapshot_tsv = out_dir / "nonbox_template_review.tsv"
    with snapshot_tsv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "snapshot_id": "E168_import_E145_bucket004_template_review",
        "created_at": now(),
        "status": "pass",
        "source_exp_id": "E145",
        "imported_by_exp_id": "E168",
        "source_review_tsv": str(source),
        "source_review_sha256": sha256(source),
        "snapshot_review_tsv": str(snapshot_tsv),
        "snapshot_review_sha256": sha256(snapshot_tsv),
        "rows": len(rows),
        "review_decisions": sorted({row["review_decision"] for row in rows}),
        "physics_fingerprints": fingerprints,
        "artifacts": artifacts,
    }
    (out_dir / "import_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
