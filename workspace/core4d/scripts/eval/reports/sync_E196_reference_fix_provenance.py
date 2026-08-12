#!/usr/bin/env python3
"""Mirror lightweight E196 evidence from the external results symlink into Git."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[5]
SOURCE = ROOT / "workspace/core4d/results/E196"
DESTINATION = ROOT / "workspace/core4d/report/E196/provenance"

MANIFEST_NAMES = (
    "reference_fix_build_summary.json",
    "reference_fix_case_authority.tsv",
    "reference_fix_full_manifest.tsv",
    "reference_fix_remaining_manifest.tsv",
    "reference_fix_wave0_manifest.tsv",
)
EVAL_NAMES = (
    "e196_reference_fix_case_metrics.tsv",
    "e196_reference_fix_by_case.tsv",
    "e196_reference_integrity_audit.tsv",
    "e196_reference_fix_eval_errors.tsv",
    "e196_reference_fix_eval_summary.json",
    "e196_reference_fix_by_object.tsv",
    "e196_reference_fix_gate_migrations.tsv",
    "e196_reference_fix_summary.json",
    "E196_reference_fix_comparison.xlsx",
    "E196_reference_metadata_integrity_fix_report.md",
)
VISUAL_NAMES = (
    "three_arm_video_manifest.tsv",
    "render_summary.json",
    "render_failures.tsv",
    "visual_review_summary.json",
    "visual_review_notes.md",
)
LIGHTWEIGHT_SUFFIXES = {".json", ".tsv", ".md", ".xlsx", ".xml"}
MAX_FILE_BYTES = 32 * 1024 * 1024


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_checked(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    if source.suffix.lower() not in LIGHTWEIGHT_SUFFIXES:
        raise ValueError(f"non-lightweight suffix rejected: {source}")
    if source.stat().st_size > MAX_FILE_BYTES:
        raise ValueError(f"file exceeds {MAX_FILE_BYTES} bytes: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--require-all", action="store_true")
    args = parser.parse_args()

    snapshot_root = SOURCE / "scene_snapshot/reference_fix"
    manifest_root = SOURCE / "s6_downstream/manifests"
    eval_root = SOURCE / "s6_downstream/eval/full_reference_fix"
    render_root = SOURCE / "s6_downstream/render/full_reference_fix"
    evidence_root = SOURCE / "s6_downstream/evidence/reference_fix"
    snapshot_files = sorted(path for path in snapshot_root.rglob("*") if path.is_file())
    if args.require_all and len(snapshot_files) != 59:
        raise SystemExit(f"expected 59 snapshot files, found {len(snapshot_files)}")

    required = [manifest_root / name for name in MANIFEST_NAMES]
    required += [eval_root / name for name in EVAL_NAMES]
    required_visual = [render_root / name for name in VISUAL_NAMES]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit("missing required provenance files:\n" + "\n".join(missing))
    missing_visual = [str(path) for path in required_visual if not path.is_file()]
    if missing_visual:
        raise SystemExit("missing required visual provenance files:\n" + "\n".join(missing_visual))

    destination_parent = DESTINATION.parent
    staging = destination_parent / f".provenance.tmp-{os.getpid()}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        for source in snapshot_files:
            copy_checked(source, staging / "scene_snapshot/reference_fix" / source.relative_to(snapshot_root))
        for source in required[: len(MANIFEST_NAMES)]:
            copy_checked(source, staging / "manifests" / source.name)
        for source in required[len(MANIFEST_NAMES) :]:
            copy_checked(source, staging / "eval" / source.name)
        for source in required_visual:
            copy_checked(source, staging / "visual" / source.name)
        evidence_files = sorted(
            path for path in evidence_root.rglob("*")
            if path.is_file() and path.suffix.lower() in LIGHTWEIGHT_SUFFIXES
        )
        for source in evidence_files:
            copy_checked(source, staging / "evidence" / source.relative_to(evidence_root))

        metadata = {
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "canonical_source": "workspace/core4d/results/E196",
            "snapshot_files": len(snapshot_files),
            "manifest_files": len(MANIFEST_NAMES),
            "eval_files": len(EVAL_NAMES),
            "visual_files": len(VISUAL_NAMES),
            "evidence_files": len(evidence_files),
        }
        (staging / "provenance_manifest.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        mirrored = sorted(path for path in staging.rglob("*") if path.is_file())
        checksums = [f"{sha256(path)}  {path.relative_to(staging).as_posix()}" for path in mirrored]
        (staging / "SHA256SUMS").write_text("\n".join(checksums) + "\n", encoding="utf-8")

        backup: Path | None = None
        if DESTINATION.exists():
            if not args.replace:
                raise SystemExit(f"destination exists; rerun with --replace: {DESTINATION}")
            marker = DESTINATION / "provenance_manifest.json"
            try:
                owned = json.loads(marker.read_text(encoding="utf-8"))
            except (FileNotFoundError, json.JSONDecodeError) as exc:
                raise SystemExit(f"refusing to replace unowned provenance directory: {DESTINATION}") from exc
            if owned.get("canonical_source") != "workspace/core4d/results/E196":
                raise SystemExit(f"refusing to replace provenance with foreign source: {DESTINATION}")
            backup = destination_parent / f".provenance.backup-{os.getpid()}"
            if backup.exists():
                raise SystemExit(f"backup path already exists: {backup}")
            DESTINATION.rename(backup)
        try:
            staging.rename(DESTINATION)
        except BaseException:
            if backup is not None and backup.exists() and not DESTINATION.exists():
                backup.rename(DESTINATION)
            raise
        if backup is not None:
            shutil.rmtree(backup)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise

    print(json.dumps(metadata, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
