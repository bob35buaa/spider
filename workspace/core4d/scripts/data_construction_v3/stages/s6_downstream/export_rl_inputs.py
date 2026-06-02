#!/usr/bin/env python3
"""Join S5 handoff rows with S6 CEM evidence for RL motion export."""

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

from common import SCHEMA_VERSION, find_spider_repo, read_tsv, timestamp, write_json, write_tsv


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
    "handoff_decision",
    "candidate_decision",
    "target_gate_status",
    "visual_qc_status",
    "target_scene",
    "trajectory",
    "scene_act",
    "contact_mask",
    "stage2b_target_task",
    "stage2b_result_root",
    "stage2b_manifest_ref",
    "raw_contact_threshold_label",
    "cem_status",
    "cem_run_id",
    "cem_result_npz",
    "cem_video",
    "cem_metrics_ref",
    "downstream_decision",
    "downstream_failure_mode",
    "downstream_notes",
    "rl_export_decision",
    "skip_reason",
    "scene_act_exists",
    "trajectory_exists",
    "contact_mask_exists",
    "cem_result_exists",
    "source_handoff_manifest",
    "source_cem_evidence",
    "schema_version",
    "updated_at",
]

HANDOFF_READY = {"HANDOFF_READY", "HANDOFF_REVIEW_VISUAL_QC"}
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


def resolve_existing(path_text: str, repo: Path) -> bool:
    if not path_text:
        return False
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = repo / path
    return path.exists() and path.stat().st_size > 0


def decide(row: dict[str, str], repo: Path) -> tuple[str, str]:
    if row["handoff_decision"] not in HANDOFF_READY:
        return "SKIP_NOT_HANDOFF_READY", row["handoff_decision"] or "empty_handoff_decision"
    if row["target_gate_status"] != "pass":
        return "SKIP_TARGET_GATE_NOT_PASS", row["target_gate_status"] or "empty_target_gate_status"
    if row["visual_qc_status"] != "pass":
        return "SKIP_VISUAL_QC_NOT_PASS", row["visual_qc_status"] or "empty_visual_qc_status"
    if row["cem_status"] == "pass":
        missing = []
        for field in ("scene_act", "trajectory", "cem_result_npz"):
            if not resolve_existing(row[field], repo):
                missing.append(field)
        if missing:
            return "BLOCKED_MISSING_REQUIRED_FILE", ",".join(missing)
        return "RL_EXPORT_READY", ""
    if row["cem_status"] == "fail":
        return "SKIP_CEM_FAIL", row["downstream_failure_mode"] or row["downstream_decision"] or "cem_fail"
    return "WAIT_CEM_NOT_RUN", row["cem_status"] or "not_run"


def build_rows(
    handoff_rows: list[dict[str, str]],
    cem_rows: list[dict[str, str]],
    repo: Path,
    source_handoff_manifest: str,
    source_cem_evidence: str,
) -> list[dict[str, Any]]:
    cem_by_key = {key(row): row for row in cem_rows if row.get("case_id")}
    rows: list[dict[str, Any]] = []
    for handoff in handoff_rows:
        if not handoff.get("case_id"):
            continue
        cem = cem_by_key.get(key(handoff), {})
        cem_status = normalize_status(pick(cem.get("cem_status", ""), handoff.get("cem_status", "")))
        row: dict[str, Any] = {
            "case_id": handoff.get("case_id", ""),
            "object_key": handoff.get("object_key", ""),
            "object_name": handoff.get("object_name", ""),
            "date": handoff.get("date", ""),
            "seq": handoff.get("seq", ""),
            "person": handoff.get("person", ""),
            "person_idx": handoff.get("person_idx", ""),
            "retarget_variant_id": handoff.get("retarget_variant_id", "shared") or "shared",
            "target_variant_id": handoff.get("target_variant_id", "ref_fk") or "ref_fk",
            "handoff_decision": handoff.get("handoff_decision", ""),
            "candidate_decision": handoff.get("candidate_decision", ""),
            "target_gate_status": handoff.get("target_gate_status", ""),
            "visual_qc_status": handoff.get("visual_qc_status", ""),
            "target_scene": handoff.get("target_scene", ""),
            "trajectory": handoff.get("trajectory", ""),
            "scene_act": handoff.get("scene_act", ""),
            "contact_mask": handoff.get("contact_mask", ""),
            "stage2b_target_task": handoff.get("stage2b_target_task", ""),
            "stage2b_result_root": handoff.get("stage2b_result_root", ""),
            "stage2b_manifest_ref": handoff.get("stage2b_manifest_ref", ""),
            "raw_contact_threshold_label": handoff.get("raw_contact_threshold_label", ""),
            "cem_status": cem_status,
            "cem_run_id": cem.get("cem_run_id", ""),
            "cem_result_npz": cem.get("cem_result_npz", cem.get("result_npz", "")),
            "cem_video": cem.get("cem_video", cem.get("video_path", "")),
            "cem_metrics_ref": cem.get("cem_metrics_ref", cem.get("metrics_ref", "")),
            "downstream_decision": cem.get("downstream_decision", ""),
            "downstream_failure_mode": cem.get("downstream_failure_mode", cem.get("failure_mode", "")),
            "downstream_notes": cem.get("downstream_notes", cem.get("notes", "")),
            "source_handoff_manifest": source_handoff_manifest,
            "source_cem_evidence": source_cem_evidence,
            "schema_version": SCHEMA_VERSION,
            "updated_at": timestamp(),
        }
        row["scene_act_exists"] = str(resolve_existing(row["scene_act"], repo))
        row["trajectory_exists"] = str(resolve_existing(row["trajectory"], repo))
        row["contact_mask_exists"] = str(resolve_existing(row["contact_mask"], repo))
        row["cem_result_exists"] = str(resolve_existing(row["cem_result_npz"], repo))
        row["rl_export_decision"], row["skip_reason"] = decide(row, repo)
        rows.append(row)
    return rows


def markdown_summary(summary: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    lines = [
        "# S6 RL export input summary",
        "",
        f"- rows: `{summary['rows']}`",
        f"- ready rows: `{summary['rl_export_decision_counts'].get('RL_EXPORT_READY', 0)}`",
        f"- out_dir: `{summary['out_dir']}`",
        "",
        "## decisions",
        "",
        "| decision | count |",
        "|---|---:|",
    ]
    for decision, count in summary["rl_export_decision_counts"].items():
        lines.append(f"| `{decision}` | {count} |")
    lines.extend(["", "## rows", "", "| case | variant | target | CEM | RL export | reason |", "|---|---|---|---|---|---|"])
    for row in rows:
        lines.append(
            f"| `{row['case_id']}` | `{row['retarget_variant_id']}` | `{row['target_variant_id']}` | "
            f"`{row['cem_status']}` | `{row['rl_export_decision']}` | `{row['skip_reason']}` |"
        )
    lines.extend(
        [
            "",
            "说明：本表是 RL 导出的唯一数据输入索引。它由 S5 handoff 与 S6 CEM evidence join 生成，不回写 S5。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--handoff-manifest-tsv", type=Path, required=True)
    parser.add_argument("--cem-evidence-tsv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--spider-repo", type=Path, default=None)
    args = parser.parse_args()

    repo = (args.spider_repo or find_spider_repo()).expanduser().resolve()
    handoff_path = args.handoff_manifest_tsv.expanduser()
    cem_path = args.cem_evidence_tsv.expanduser()
    handoff_rows = read_tsv(handoff_path)
    cem_rows = read_tsv(cem_path)
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = build_rows(handoff_rows, cem_rows, repo, str(handoff_path), str(cem_path))
    write_tsv(out_dir / "rl_export_input.tsv", rows, FIELDS)
    write_json(out_dir / "rl_export_input.json", rows)

    summary = {
        "stage": "S6_rl_export_input",
        "created_at": timestamp(),
        "schema_version": SCHEMA_VERSION,
        "rows": len(rows),
        "rl_export_decision_counts": dict(Counter(row["rl_export_decision"] for row in rows)),
        "cem_status_counts": dict(Counter(row["cem_status"] for row in rows)),
        "source_handoff_manifest": str(handoff_path),
        "source_cem_evidence": str(cem_path),
        "out_dir": str(out_dir),
    }
    write_json(out_dir / "rl_export_summary.json", summary)
    (out_dir / "rl_export_summary.md").write_text(markdown_summary(summary, rows), encoding="utf-8")
    print(f"wrote {out_dir / 'rl_export_input.tsv'} rows={len(rows)}")


if __name__ == "__main__":
    main()
