#!/usr/bin/env python3
"""Build S6 downstream CEM/RL evidence manifests without changing data gates."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

SCRIPT_ROOT = next(parent for parent in Path(__file__).resolve().parents if parent.name == "data_construction_v3")
for _path in (SCRIPT_ROOT / "lib", SCRIPT_ROOT / "state", SCRIPT_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
from typing import Any

from common import SCHEMA_VERSION, read_tsv, timestamp, write_json, write_tsv


FIELDS = [
    "case_id",
    "object_key",
    "object_name",
    "date",
    "seq",
    "person",
    "person_idx",
    "retarget_variant_id",
    "target_variant_id",
    "candidate_decision",
    "handoff_decision",
    "cem_status",
    "rl_status",
    "downstream_decision",
    "downstream_failure_mode",
    "downstream_notes",
    "cem_run_id",
    "cem_result_npz",
    "cem_video",
    "cem_metrics_ref",
    "rl_run_id",
    "rl_checkpoint",
    "rl_metrics_ref",
    "rl_video",
    "downstream_evidence_root",
    "source_ref",
    "schema_version",
    "updated_at",
]


STATUS_PASS = {"pass", "passed", "success", "succeeded", "work", "working", "ok", "true", "1"}
STATUS_FAIL = {"fail", "failed", "reject", "rejected", "bad", "false", "0"}
STATUS_PENDING = {"", "not_run", "pending", "review", "unknown", "na", "n/a", "none"}


def key(row: dict[str, str]) -> tuple[str, str, str]:
    return (
        row.get("case_id", ""),
        row.get("retarget_variant_id", "shared") or "shared",
        row.get("target_variant_id", "ref_fk") or "ref_fk",
    )


def normalize_status(value: str) -> str:
    text = str(value or "").strip().lower()
    if text in STATUS_PASS:
        return "pass"
    if text in STATUS_FAIL:
        return "fail"
    if text in STATUS_PENDING:
        return "not_run"
    return text


def pick(*values: str) -> str:
    for value in values:
        if value not in ("", None):
            return str(value)
    return ""


def downstream_decision(cem_status: str, rl_status: str, failure_mode: str, notes: str) -> str:
    cem = normalize_status(cem_status)
    rl = normalize_status(rl_status)
    if rl == "pass":
        return "DOWNSTREAM_RL_PASS"
    if cem == "pass" and rl in {"", "not_run"}:
        return "DOWNSTREAM_CEM_PASS"
    if cem in {"", "not_run"} and rl in {"", "not_run"}:
        return "DOWNSTREAM_NOT_RUN"

    text = f"{failure_mode} {notes}".lower()
    if any(token in text for token in ("posture", "upright", "pelvis", "lie_on_box", "low_hip", "tilt")):
        return "DOWNSTREAM_POSTURE_FAIL"
    if any(token in text for token in ("motion", "binding", "transport", "object", "carry", "slip")):
        return "DOWNSTREAM_MOTION_BINDING_FAIL"
    if rl == "fail":
        return "DOWNSTREAM_RL_FAIL"
    if cem == "fail":
        return "DOWNSTREAM_CEM_FAIL"
    return "DOWNSTREAM_REVIEW"


def build_rows(
    handoff_rows: list[dict[str, str]],
    evidence_rows: list[dict[str, str]],
    evidence_root: str,
    source_ref: str,
) -> list[dict[str, Any]]:
    base_by_key = {key(row): row for row in handoff_rows if row.get("case_id")}
    evidence_by_key = {key(row): row for row in evidence_rows if row.get("case_id")}
    keys = sorted(set(base_by_key) | set(evidence_by_key))
    out: list[dict[str, Any]] = []
    for row_key in keys:
        base = base_by_key.get(row_key, {})
        ev = evidence_by_key.get(row_key, {})
        cem_status = normalize_status(pick(ev.get("cem_status", ""), base.get("cem_status", "")))
        rl_status = normalize_status(pick(ev.get("rl_status", ""), base.get("rl_status", "")))
        failure_mode = pick(ev.get("downstream_failure_mode", ""), ev.get("failure_mode", ""))
        notes = pick(ev.get("downstream_notes", ""), ev.get("notes", ""))
        row = {
            "case_id": pick(ev.get("case_id", ""), base.get("case_id", "")),
            "object_key": pick(ev.get("object_key", ""), base.get("object_key", "")),
            "object_name": pick(ev.get("object_name", ""), base.get("object_name", "")),
            "date": pick(ev.get("date", ""), base.get("date", "")),
            "seq": pick(ev.get("seq", ""), base.get("seq", "")),
            "person": pick(ev.get("person", ""), base.get("person", "")),
            "person_idx": pick(ev.get("person_idx", ""), base.get("person_idx", "")),
            "retarget_variant_id": pick(ev.get("retarget_variant_id", ""), base.get("retarget_variant_id", "shared")),
            "target_variant_id": pick(ev.get("target_variant_id", ""), base.get("target_variant_id", "ref_fk")),
            "candidate_decision": base.get("candidate_decision", ""),
            "handoff_decision": base.get("handoff_decision", ""),
            "cem_status": cem_status,
            "rl_status": rl_status,
            "downstream_decision": downstream_decision(cem_status, rl_status, failure_mode, notes),
            "downstream_failure_mode": failure_mode,
            "downstream_notes": notes,
            "cem_run_id": ev.get("cem_run_id", ""),
            "cem_result_npz": ev.get("cem_result_npz", ev.get("result_npz", "")),
            "cem_video": ev.get("cem_video", ev.get("video_path", "")),
            "cem_metrics_ref": ev.get("cem_metrics_ref", ev.get("metrics_ref", "")),
            "rl_run_id": ev.get("rl_run_id", ""),
            "rl_checkpoint": ev.get("rl_checkpoint", ""),
            "rl_metrics_ref": ev.get("rl_metrics_ref", ""),
            "rl_video": ev.get("rl_video", ""),
            "downstream_evidence_root": evidence_root or ev.get("downstream_evidence_root", ""),
            "source_ref": source_ref,
            "schema_version": SCHEMA_VERSION,
            "updated_at": timestamp(),
        }
        out.append(row)
    return out


def markdown_summary(summary: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    lines = [
        "# S6 downstream evidence summary",
        "",
        f"- rows: `{summary['rows']}`",
        f"- out_dir: `{summary['out_dir']}`",
        "",
        "## downstream decisions",
        "",
        "| decision | count |",
        "|---|---:|",
    ]
    for decision, count in summary["downstream_decision_counts"].items():
        lines.append(f"| `{decision}` | {count} |")
    lines.extend(
        [
            "",
            "说明：S6 只记录 CEM/RL 下游证据，不反向改变 raw contact、template、Stage2b、target gate 或 visual QC 的数据构建判定。",
            "",
            "## failure rows",
            "",
            "| case | variant | target | decision | failure |",
            "|---|---|---|---|---|",
        ]
    )
    for row in rows:
        if str(row["downstream_decision"]).endswith("_FAIL") or row["downstream_decision"] in {"DOWNSTREAM_POSTURE_FAIL", "DOWNSTREAM_MOTION_BINDING_FAIL"}:
            lines.append(
                f"| `{row['case_id']}` | `{row['retarget_variant_id']}` | `{row['target_variant_id']}` | `{row['downstream_decision']}` | `{row['downstream_failure_mode']}` |"
            )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--handoff-manifest-tsv", type=Path, default=None)
    parser.add_argument("--evidence-tsv", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--evidence-root", default="")
    parser.add_argument("--source-ref", default="S6_downstream_evidence")
    args = parser.parse_args()

    handoff_rows = read_tsv(args.handoff_manifest_tsv) if args.handoff_manifest_tsv and args.handoff_manifest_tsv.is_file() else []
    evidence_rows = read_tsv(args.evidence_tsv) if args.evidence_tsv and args.evidence_tsv.is_file() else []
    rows = build_rows(handoff_rows, evidence_rows, args.evidence_root, args.source_ref)
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    write_tsv(out_dir / "downstream_evidence_manifest.tsv", rows, FIELDS)
    write_json(out_dir / "downstream_evidence_manifest.json", rows)

    summary = {
        "stage": "S6_downstream_evidence",
        "created_at": timestamp(),
        "schema_version": SCHEMA_VERSION,
        "rows": len(rows),
        "downstream_decision_counts": dict(Counter(row["downstream_decision"] for row in rows)),
        "cem_status_counts": dict(Counter(row["cem_status"] for row in rows)),
        "rl_status_counts": dict(Counter(row["rl_status"] for row in rows)),
        "out_dir": str(out_dir),
    }
    write_json(out_dir / "downstream_evidence_summary.json", summary)
    (out_dir / "downstream_evidence_summary.md").write_text(markdown_summary(summary, rows), encoding="utf-8")
    print(f"wrote {out_dir / 'downstream_evidence_manifest.tsv'} rows={len(rows)}")


if __name__ == "__main__":
    main()
