#!/usr/bin/env python3
"""Record the explicit E181 Gate A oracle visual review."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from build_authority import REPO_ROOT, relative_to_repo, sha256_file
from build_oracle_meshes import atomic_json

DEFAULT_OUTPUT_ROOT = REPO_ROOT / "workspace/core4d/results/E181/s1_oracle"
DEFAULT_REVIEWER = "codex_local_visual_review_20260731"
REVIEW_SOURCE = "E181_component_overlay_and_principal_axis_cross_section_review"
APPROVED_FINDINGS = {
    "bucket003": (
        "Raw and cleaned XY/XZ/YZ projections are pointwise consistent; no component "
        "was removed. All three center cross-sections are continuous and closed, with "
        "no outlier or cleanup artifact."
    ),
    "bucket004": (
        "The single 10-face numerical fragment is localized near the outer surface. "
        "The cleaned watertight main component has no visible hole or deformation; "
        "all center cross-sections remain continuous and closed."
    ),
    "bucket007": (
        "The 4-face and 2-face numerical fragments project near the outer surface. "
        "The cleaned bucket body and rim remain complete in all views; all center "
        "cross-sections are continuous and free of cleanup artifacts."
    ),
}


def now() -> str:
    """Return an auditable local timestamp."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object."""
    return json.loads(path.read_text(encoding="utf-8"))


def build_review(output_root: Path, reviewer: str) -> dict[str, Any]:
    """Validate evidence and persist the 3/3 visual approval."""
    summary_path = output_root / "gate_a_summary.json"
    review_path = output_root / "visual_review.json"
    summary = load_json(summary_path)
    if (
        summary.get("status") == "PASS"
        and summary.get("visual_review", {}).get("status") == "APPROVED"
        and review_path.is_file()
    ):
        return load_json(review_path)
    if summary.get("status") != "PASS_NONVISUAL":
        raise RuntimeError("Gate A summary must be PASS_NONVISUAL before review")
    if summary.get("visual_review") != "PENDING":
        raise RuntimeError("Gate A visual review must be PENDING before approval")

    reviewed_at = now()
    object_reviews: dict[str, Any] = {}
    manifests: dict[str, tuple[Path, dict[str, Any]]] = {}
    for object_key, findings in APPROVED_FINDINGS.items():
        manifest_path = output_root / object_key / "oracle_manifest.json"
        manifest = load_json(manifest_path)
        if manifest.get("status") != "PASS_NONVISUAL":
            raise RuntimeError(f"{object_key}: nonvisual Gate A is not PASS")
        if manifest.get("visual_review") != "PENDING":
            raise RuntimeError(f"{object_key}: visual review is not PENDING")
        evidence = {}
        for evidence_key, logical_path in manifest["visual_evidence"].items():
            path = REPO_ROOT / logical_path
            if not path.is_file():
                raise FileNotFoundError(path)
            evidence[evidence_key] = {
                "path": logical_path,
                "sha256": sha256_file(path),
            }
        object_reviews[object_key] = {
            "status": "APPROVED",
            "findings": findings,
            "oracle_manifest_pre_review_sha256": sha256_file(manifest_path),
            "evidence": evidence,
        }
        manifests[object_key] = (manifest_path, manifest)

    review = {
        "experiment_id": "E181",
        "stage": "S1_oracle_visual_review",
        "status": "APPROVED",
        "approved_objects": len(object_reviews),
        "required_objects": len(APPROVED_FINDINGS),
        "reviewer": reviewer,
        "review_source": REVIEW_SOURCE,
        "reviewed_at": reviewed_at,
        "objects": object_reviews,
    }
    atomic_json(review_path, review)
    review_sha256 = sha256_file(review_path)
    review_pointer = {
        "status": "APPROVED",
        "reviewer": reviewer,
        "review_source": REVIEW_SOURCE,
        "reviewed_at": reviewed_at,
        "review_record": relative_to_repo(review_path),
        "review_record_sha256": review_sha256,
    }
    for object_key, (manifest_path, manifest) in manifests.items():
        manifest["nonvisual_status"] = "PASS_NONVISUAL"
        manifest["status"] = "PASS"
        manifest["visual_review"] = {
            **review_pointer,
            "findings": object_reviews[object_key]["findings"],
        }
        atomic_json(manifest_path, manifest)

    summary["nonvisual_status"] = "PASS_NONVISUAL"
    summary["status"] = "PASS"
    summary["visual_review"] = {
        **review_pointer,
        "approved_objects": len(object_reviews),
        "required_objects": len(APPROVED_FINDINGS),
    }
    summary["object_status"] = dict.fromkeys(APPROVED_FINDINGS, "PASS")
    atomic_json(summary_path, summary)
    return review


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--reviewer", default=DEFAULT_REVIEWER)
    args = parser.parse_args()
    review = build_review(args.output_root, args.reviewer)
    print(
        "E181_GATE_A=PASS "
        f"approved={review['approved_objects']}/{review['required_objects']} "
        f"reviewer={review['reviewer']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
