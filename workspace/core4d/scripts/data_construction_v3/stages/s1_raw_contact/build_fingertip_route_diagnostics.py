#!/usr/bin/env python3
"""Build the v3 E099-E101 route diagnostic manifest for fingertip-aware targets."""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path
import sys

SCRIPT_ROOT = next(parent for parent in Path(__file__).resolve().parents if parent.name == "data_construction_v3")
for _path in (SCRIPT_ROOT / "lib", SCRIPT_ROOT / "state", SCRIPT_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
from typing import Any

from common import SCHEMA_VERSION, read_tsv, sha256_file, timestamp, write_json, write_tsv


FIELDS = [
    "case_id",
    "target_variant_id",
    "route_diagnostic_status",
    "route_diagnostic_ref",
    "fingertip_vote_status",
    "palm_vote_status",
    "quat_audit_status",
    "target_active_mask_status",
    "e101_route_evidence_status",
    "fingertip_vote_L",
    "fingertip_vote_R",
    "palm_vote_L",
    "palm_vote_R",
    "face_changed_l",
    "face_changed_r",
    "disable_world_up",
    "target_npz",
    "target_npz_sha256",
    "active_L_frac",
    "active_R_frac",
    "target_gap_status",
    "e101_evidence_ref",
    "diagnostic_notes",
    "schema_version",
    "updated_at",
]


def normalize_case_id(value: str) -> str:
    text = value.strip().lower()
    text = re.sub(r"^(e\d+|d\d+)_", "", text)
    text = re.sub(r"_(e\d+|d\d+|e\d+[a-z]*)_(dyn|clean|ref_fk|adaptive|fingertip).*$", "", text)
    text = re.sub(r"_(e\d+|d\d+|e\d+[a-z]*)$", "", text)
    text = text.replace("_person1", "_p1").replace("_person2", "_p2")
    return text


def index_by_case(rows: list[dict[str, str]], case_field: str = "case") -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for row in rows:
        case = row.get(case_field, "")
        if not case:
            continue
        out[normalize_case_id(case)] = row
    return out


def bool_text(value: str) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "pass", "ok"}


def status_from_ok(value: str) -> str:
    if value == "ok":
        return "pass"
    if not value:
        return "missing"
    if value.startswith("missing"):
        return "missing"
    return "reject"


def active_mask_status(row: dict[str, str] | None, min_active_frac: float) -> str:
    if not row:
        return "missing"
    if row.get("status") != "ok":
        return "reject"
    try:
        active_l = float(row.get("active_L_frac", "nan"))
        active_r = float(row.get("active_R_frac", "nan"))
    except ValueError:
        return "reject"
    if active_l >= min_active_frac and active_r >= min_active_frac:
        return "pass"
    return "reject"


def build_e101_index(rows: list[dict[str, str]]) -> tuple[bool, dict[str, list[dict[str, str]]]]:
    guard_pass = False
    negatives: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        role = row.get("role", "")
        status = row.get("status", "")
        gate_pass = bool_text(row.get("gate_pass", ""))
        norm = normalize_case_id(row.get("task", ""))
        if "box004_guard" in role and status == "WORK" and gate_pass:
            guard_pass = True
        if status != "WORK" or not gate_pass:
            negatives.setdefault(norm, []).append(row)
    return guard_pass, negatives


def e101_status(case_id: str, guard_pass: bool, negatives: dict[str, list[dict[str, str]]]) -> tuple[str, str]:
    norm = normalize_case_id(case_id)
    if norm in negatives:
        labels = sorted({row.get("visual_classification", "") or row.get("status", "") for row in negatives[norm]})
        return "reject", "case_current_negative:" + ",".join(labels)
    if guard_pass:
        return "pass", "route_guard_pass;case_not_current_negative"
    return "missing", "missing_box004_guard_pass_evidence"


def read_optional_tsv(path: Path | None) -> list[dict[str, str]]:
    if not path or not path.is_file():
        return []
    return read_tsv(path)


def route_status(parts: list[str]) -> str:
    if all(part == "pass" for part in parts):
        return "pass"
    if any(part == "missing" for part in parts):
        return "missing"
    return "reject"


def build_rows(
    input_rows: list[dict[str, str]],
    fingertip_index: dict[str, dict[str, str]],
    palm_index: dict[str, dict[str, str]],
    quat_index: dict[str, dict[str, str]],
    target_index: dict[str, dict[str, str]],
    guard_pass: bool,
    e101_negatives: dict[str, list[dict[str, str]]],
    target_dir: Path,
    min_active_frac: float,
    evidence_ref: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source in input_rows:
        case_id = source.get("case_id") or source.get("case") or source.get("case_name")
        if not case_id:
            continue
        norm = normalize_case_id(case_id)
        fingertip = fingertip_index.get(norm)
        palm = palm_index.get(norm)
        quat = quat_index.get(norm)
        target = target_index.get(norm)
        fingertip_status = status_from_ok(fingertip.get("status", "") if fingertip else "")
        palm_status = status_from_ok(palm.get("status", "") if palm else "")
        quat_status = status_from_ok(quat.get("status", "") if quat else "")
        target_status = active_mask_status(target, min_active_frac)
        e101, e101_note = e101_status(case_id, guard_pass, e101_negatives)

        target_npz = target_dir / (target.get("case", case_id) if target else case_id) / "spider_contact_target_object_local.npz"
        target_npz_str = str(target_npz) if target_npz.is_file() else ""
        target_sha = sha256_file(target_npz) if target_npz.is_file() else ""
        parts = [fingertip_status, palm_status, quat_status, target_status, e101]
        overall = route_status(parts)
        notes: list[str] = []
        if overall != "pass":
            for name, status in [
                ("fingertip_vote", fingertip_status),
                ("palm_vote", palm_status),
                ("quat_audit", quat_status),
                ("target_active_mask", target_status),
                ("e101_route_evidence", e101),
            ]:
                if status != "pass":
                    notes.append(f"{name}={status}")
        notes.append(e101_note)

        rows.append(
            {
                "case_id": case_id,
                "target_variant_id": "fingertip_aware",
                "route_diagnostic_status": overall,
                "route_diagnostic_ref": evidence_ref,
                "fingertip_vote_status": fingertip_status,
                "palm_vote_status": palm_status,
                "quat_audit_status": quat_status,
                "target_active_mask_status": target_status,
                "e101_route_evidence_status": e101,
                "fingertip_vote_L": fingertip.get("L_vote", "") if fingertip else "",
                "fingertip_vote_R": fingertip.get("R_vote", "") if fingertip else "",
                "palm_vote_L": palm.get("L_face", target.get("palm_vote_L", "") if target else "") if palm else (target.get("palm_vote_L", "") if target else ""),
                "palm_vote_R": palm.get("R_face", target.get("palm_vote_R", "") if target else "") if palm else (target.get("palm_vote_R", "") if target else ""),
                "face_changed_l": target.get("face_changed_L", "") if target else "",
                "face_changed_r": target.get("face_changed_R", "") if target else "",
                "disable_world_up": quat.get("disable_world_up", "") if quat else "",
                "target_npz": target_npz_str,
                "target_npz_sha256": target_sha,
                "active_L_frac": target.get("active_L_frac", "") if target else "",
                "active_R_frac": target.get("active_R_frac", "") if target else "",
                "target_gap_status": target.get("status", "") if target else "missing",
                "e101_evidence_ref": "E101/cem_outcome_matrix",
                "diagnostic_notes": ";".join(notes),
                "schema_version": SCHEMA_VERSION,
                "updated_at": timestamp(),
            }
        )
    rows.sort(key=lambda row: (row["route_diagnostic_status"] != "pass", row["case_id"]))
    return rows


def markdown_summary(summary: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Fingertip-aware route diagnostic summary",
        "",
        f"- rows: `{summary['rows']}`",
        f"- pass rows: `{summary['pass_rows']}`",
        "- 说明：这是 `target_variant_id=fingertip_aware` 的 E099-E101 route contract manifest；默认 `ref_fk` 不需要该 manifest。",
        "",
        "## route diagnostic status",
        "",
        "| status | count |",
        "|---|---:|",
    ]
    for key, count in summary["route_diagnostic_counts"].items():
        lines.append(f"| `{key}` | {count} |")
    lines.extend(["", "## non-pass rows", "", "| case | status | notes |", "|---|---|---|"])
    for row in rows:
        if row["route_diagnostic_status"] != "pass":
            lines.append(f"| `{row['case_id']}` | `{row['route_diagnostic_status']}` | `{row['diagnostic_notes']}` |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-tsv", type=Path, required=True, help="case list, usually selected raw_contact_pass_<label>.tsv")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--fingertip-face-stats", type=Path, default=Path("workspace/core4d/results/E099/fingertip_face_stats.tsv"))
    parser.add_argument("--palm-face-stats", type=Path, default=Path("workspace/core4d/results/E099/palm_face_stats.tsv"))
    parser.add_argument("--quat-audit", type=Path, default=Path("workspace/core4d/results/E099/quat_audit.tsv"))
    parser.add_argument("--target-gap-summary", type=Path, default=Path("workspace/core4d/results/E100/target_gap_summary.tsv"))
    parser.add_argument("--target-dir", type=Path, default=Path("workspace/core4d/results/E100/fingertip_targets"))
    parser.add_argument("--e101-outcome-matrix", type=Path, default=Path("workspace/core4d/results/E101/cem_outcome_matrix.tsv"))
    parser.add_argument("--min-active-frac", type=float, default=0.01)
    args = parser.parse_args()

    input_rows = read_tsv(args.input_tsv)
    fingertip_index = index_by_case(read_optional_tsv(args.fingertip_face_stats))
    palm_index = index_by_case(read_optional_tsv(args.palm_face_stats))
    quat_index = index_by_case(read_optional_tsv(args.quat_audit))
    target_index = index_by_case(read_optional_tsv(args.target_gap_summary))
    guard_pass, e101_negatives = build_e101_index(read_optional_tsv(args.e101_outcome_matrix))

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    evidence_ref = str(out_dir / "fingertip_route_diagnostics.tsv")
    rows = build_rows(
        input_rows=input_rows,
        fingertip_index=fingertip_index,
        palm_index=palm_index,
        quat_index=quat_index,
        target_index=target_index,
        guard_pass=guard_pass,
        e101_negatives=e101_negatives,
        target_dir=args.target_dir,
        min_active_frac=args.min_active_frac,
        evidence_ref=evidence_ref,
    )
    write_tsv(out_dir / "fingertip_route_diagnostics.tsv", rows, FIELDS)
    write_json(out_dir / "fingertip_route_diagnostics.json", rows)
    summary = {
        "stage": "S1_fingertip_route_diagnostics",
        "created_at": timestamp(),
        "schema_version": SCHEMA_VERSION,
        "input_tsv": str(args.input_tsv),
        "rows": len(rows),
        "pass_rows": sum(row["route_diagnostic_status"] == "pass" for row in rows),
        "route_diagnostic_counts": dict(Counter(row["route_diagnostic_status"] for row in rows)),
        "component_counts": {
            "fingertip_vote_status": dict(Counter(row["fingertip_vote_status"] for row in rows)),
            "palm_vote_status": dict(Counter(row["palm_vote_status"] for row in rows)),
            "quat_audit_status": dict(Counter(row["quat_audit_status"] for row in rows)),
            "target_active_mask_status": dict(Counter(row["target_active_mask_status"] for row in rows)),
            "e101_route_evidence_status": dict(Counter(row["e101_route_evidence_status"] for row in rows)),
        },
        "out_dir": str(out_dir),
    }
    write_json(out_dir / "fingertip_route_diagnostics_summary.json", summary)
    (out_dir / "fingertip_route_diagnostics_summary.md").write_text(markdown_summary(summary, rows), encoding="utf-8")
    print(f"wrote {out_dir / 'fingertip_route_diagnostics.tsv'} rows={len(rows)} pass={summary['pass_rows']}")


if __name__ == "__main__":
    main()
