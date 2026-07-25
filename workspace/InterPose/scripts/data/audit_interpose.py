#!/usr/bin/env python3
"""Audit the released InterPose motion package for human-object-human data."""

from __future__ import annotations

import argparse
import hashlib
import logging
from datetime import datetime
from pathlib import Path

from interpose_audit_core import (
    LOGGER,
    PairClassification,
    ParsedSequenceId,
    ParsedTextRecord,
    SequenceEvidence,
    SequenceRecord,
    _git_commit,
    classify_pair,
    extract_labels,
    parse_sequence_id,
    parse_text_record,
    scan_release,
)
from interpose_audit_output import write_outputs
from interpose_audit_summary import aggregate_records
from interpose_vocab import SOURCE_DIRS

__all__ = [
    "PairClassification",
    "ParsedSequenceId",
    "ParsedTextRecord",
    "SOURCE_DIRS",
    "SequenceEvidence",
    "SequenceRecord",
    "aggregate_records",
    "classify_pair",
    "extract_labels",
    "main",
    "parse_sequence_id",
    "parse_text_record",
    "scan_release",
]

AUDITOR_MODULES = (
    "audit_interpose.py",
    "interpose_audit_core.py",
    "interpose_audit_output.py",
    "interpose_audit_summary.py",
    "interpose_vocab.py",
)


def _validate_inputs(args: argparse.Namespace) -> None:
    """Fail fast on missing or unsafe read/write boundaries."""
    if not args.data_root.is_dir():
        raise FileNotFoundError(f"InterPose data root not found: {args.data_root}")
    for source in SOURCE_DIRS:
        if not (args.data_root / source).is_dir():
            raise FileNotFoundError(f"Missing source directory: {source}")
    if not (args.data_root / "texts").is_dir():
        raise FileNotFoundError(f"Missing text directory: {args.data_root / 'texts'}")
    if not args.official_code_root.is_dir():
        raise FileNotFoundError(
            f"Official code root not found: {args.official_code_root}"
        )
    if not args.collection_code_root.is_dir():
        raise FileNotFoundError(
            f"Collection code root not found: {args.collection_code_root}"
        )
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be at least 1")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--official-code-root", type=Path, required=True)
    parser.add_argument("--collection-code-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def _auditor_sha256() -> str:
    """Hash every source module that defines the audit result."""
    source_root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for module_name in AUDITOR_MODULES:
        digest.update(module_name.encode("utf-8"))
        digest.update((source_root / module_name).read_bytes())
    return digest.hexdigest()


def _build_provenance(
    args: argparse.Namespace,
    started_at: str,
) -> dict[str, object]:
    """Build source, code, and classification provenance."""
    return {
        "started_at": started_at,
        "completed_at": datetime.now().astimezone().isoformat(),
        "data_root": str(args.data_root.resolve()),
        "data_root_mtime": datetime.fromtimestamp(args.data_root.stat().st_mtime)
        .astimezone()
        .isoformat(),
        "official_code_root": str(args.official_code_root.resolve()),
        "official_code_commit": _git_commit(args.official_code_root),
        "collection_code_root": str(args.collection_code_root.resolve()),
        "collection_code_commit": _git_commit(args.collection_code_root),
        "auditor_sha256": _auditor_sha256(),
        "auditor_modules": list(AUDITOR_MODULES),
        "workers": args.workers,
        "limit": args.limit,
        "classification_contract": {
            "confirmed_target": "requires synchronized two-person and object trajectories",
            "strict_text_candidate": "same clip group, shared object label, relation cue",
            "shared_object_candidate": "same clip group and shared object label",
        },
    }


def main() -> int:
    """Run the complete release audit."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    args = parse_args()
    _validate_inputs(args)
    started_at = datetime.now().astimezone().isoformat()
    records = scan_release(
        data_root=args.data_root,
        collection_root=args.collection_code_root,
        workers=args.workers,
        limit=args.limit,
    )
    summary, group_rows, pair_rows = aggregate_records(records)
    provenance = _build_provenance(args, started_at)
    write_outputs(
        output_root=args.output_root,
        records=records,
        summary=summary,
        group_rows=group_rows,
        pair_rows=pair_rows,
        provenance=provenance,
    )
    LOGGER.info(
        "Audit complete: sequences=%d multi_groups=%d shared_pairs=%d",
        len(records),
        len(group_rows),
        len(pair_rows),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
