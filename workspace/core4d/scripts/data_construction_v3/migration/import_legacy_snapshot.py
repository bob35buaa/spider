#!/usr/bin/env python3
"""Import an explicit legacy/manual state TSV into a v3 snapshot."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
import sys

SCRIPT_ROOT = next(parent for parent in Path(__file__).resolve().parents if parent.name == "data_construction_v3")
for _path in (SCRIPT_ROOT / "lib", SCRIPT_ROOT / "state", SCRIPT_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
from typing import Any

from common import SCHEMA_VERSION, read_tsv, sha256_file, timestamp, write_json, write_tsv
from update_case_state_registry import FIELDS, current_decision_from_registry, normalize_row


VALID_SOURCE_TYPES = ("legacy_import", "manual_seed")


def needs_evidence(row: dict[str, str]) -> bool:
    for field in [
        "raw_inventory_status",
        "raw_contact_3cm_status",
        "raw_contact_5cm_status",
        "template_status",
        "stage2b_status",
        "target_gate_status",
        "visual_qc_status",
        "cem_status",
        "rl_status",
    ]:
        value = row.get(field, "")
        if value and value != "not_run":
            return True
    return bool(row.get("current_decision") and row.get("current_decision") != "not_run")


def build_rows(
    rows: list[dict[str, str]],
    *,
    source_type: str,
    source_ref: str,
    snapshot_dir: Path,
    default_evidence_root: str,
    note_prefix: str,
) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for row in rows:
        norm = normalize_row(row, source_type=source_type, source_ref=source_ref)
        if not norm.get("case_id"):
            continue
        if not norm.get("evidence_root"):
            norm["evidence_root"] = default_evidence_root or str(snapshot_dir)
        if note_prefix:
            existing = norm.get("notes", "")
            norm["notes"] = note_prefix + (f";{existing}" if existing else "")
        norm["updated_at"] = timestamp()
        norm["schema_version"] = SCHEMA_VERSION
        norm["current_decision"] = current_decision_from_registry(norm)
        out.append(norm)
    out.sort(key=lambda row: (row.get("case_id", ""), row.get("retarget_variant_id", ""), row.get("target_variant_id", "")))
    return out


def validate_rows(rows: list[dict[str, str]], require_existing_evidence: bool) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    keys: set[tuple[str, str, str]] = set()
    for row in rows:
        key = (
            row.get("case_id", ""),
            row.get("retarget_variant_id", "shared") or "shared",
            row.get("target_variant_id", "ref_fk") or "ref_fk",
        )
        if key in keys:
            errors.append(f"duplicate key: {key}")
        keys.add(key)
        if needs_evidence(row):
            evidence = row.get("evidence_root", "")
            if not evidence:
                errors.append(f"{row.get('case_id')} has non-not_run state but empty evidence_root")
            elif require_existing_evidence and not Path(evidence).expanduser().exists():
                errors.append(f"{row.get('case_id')} evidence_root missing: {evidence}")
            elif not Path(evidence).expanduser().exists():
                warnings.append(f"{row.get('case_id')} evidence_root currently missing: {evidence}")
    return errors, warnings


def markdown_summary(manifest: dict[str, Any], rows: list[dict[str, str]]) -> str:
    lines = [
        "# Legacy/manual snapshot import",
        "",
        f"- snapshot_id: `{manifest['snapshot_id']}`",
        f"- source_type: `{manifest['source_type']}`",
        f"- original_input: `{manifest['original_input']}`",
        f"- rows: `{manifest['rows']}`",
        f"- status: `{manifest['status']}`",
        "",
        "## current decision counts",
        "",
        "| decision | count |",
        "|---|---:|",
    ]
    for key, count in manifest["current_decision_counts"].items():
        lines.append(f"| `{key}` | {count} |")
    if manifest["known_risks"]:
        lines.extend(["", "## known risks", ""])
        for risk in manifest["known_risks"]:
            lines.append(f"- {risk}")
    if manifest["errors"]:
        lines.extend(["", "## errors", ""])
        for item in manifest["errors"]:
            lines.append(f"- `{item}`")
    if manifest["warnings"]:
        lines.extend(["", "## warnings", ""])
        for item in manifest["warnings"]:
            lines.append(f"- `{item}`")
    lines.extend(["", "## rows", "", "| case | variant | target | decision | evidence |", "|---|---|---|---|---|"])
    for row in rows[:100]:
        lines.append(
            f"| `{row['case_id']}` | `{row['retarget_variant_id']}` | `{row['target_variant_id']}` | `{row['current_decision']}` | `{row['evidence_root']}` |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-tsv", type=Path, required=True, help="v3-compatible case-state TSV or manual seed TSV")
    parser.add_argument("--out-dir", type=Path, required=True, help="snapshot output directory")
    parser.add_argument("--snapshot-id", default="")
    parser.add_argument("--source-type", choices=VALID_SOURCE_TYPES, default="legacy_import")
    parser.add_argument("--source-ref", default="")
    parser.add_argument("--legacy-root", type=Path, default=None)
    parser.add_argument("--default-evidence-root", default="")
    parser.add_argument("--known-risk", action="append", default=[])
    parser.add_argument("--note-prefix", default="")
    parser.add_argument("--require-existing-evidence", action="store_true")
    args = parser.parse_args()

    original_input = args.input_tsv.expanduser().resolve()
    if not original_input.is_file():
        raise SystemExit(f"missing --input-tsv: {original_input}")
    snapshot_id = args.snapshot_id or f"import_{timestamp().replace(':', '').replace('-', '').replace('+', '_')}"
    out_dir = (args.out_dir.expanduser().resolve() / snapshot_id).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    source_ref = args.source_ref or str(args.legacy_root.expanduser().resolve() if args.legacy_root else original_input)

    snapshot_input = out_dir / original_input.name
    shutil.copy2(original_input, snapshot_input)
    raw_rows = read_tsv(snapshot_input)
    imported_rows = build_rows(
        raw_rows,
        source_type=args.source_type,
        source_ref=source_ref,
        snapshot_dir=out_dir,
        default_evidence_root=args.default_evidence_root,
        note_prefix=args.note_prefix,
    )
    errors, warnings = validate_rows(imported_rows, args.require_existing_evidence)
    write_tsv(out_dir / "imported_case_state_registry.tsv", imported_rows, FIELDS)
    write_json(out_dir / "imported_case_state_registry.json", imported_rows)
    manifest = {
        "stage": "import_legacy_snapshot",
        "snapshot_id": snapshot_id,
        "created_at": timestamp(),
        "schema_version": SCHEMA_VERSION,
        "status": "pass" if not errors else "fail",
        "source_type": args.source_type,
        "source_ref": source_ref,
        "legacy_root": str(args.legacy_root.expanduser().resolve()) if args.legacy_root else "",
        "original_input": str(original_input),
        "snapshot_input": str(snapshot_input),
        "input_sha256": sha256_file(snapshot_input),
        "imported_registry_tsv": str(out_dir / "imported_case_state_registry.tsv"),
        "rows": len(imported_rows),
        "current_decision_counts": dict(Counter(row["current_decision"] for row in imported_rows)),
        "source_ref_counts": dict(Counter(row["source_ref"] for row in imported_rows)),
        "known_risks": args.known_risk,
        "errors": errors,
        "warnings": warnings,
    }
    write_json(out_dir / "import_manifest.json", manifest)
    (out_dir / "import_manifest.md").write_text(markdown_summary(manifest, imported_rows), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
