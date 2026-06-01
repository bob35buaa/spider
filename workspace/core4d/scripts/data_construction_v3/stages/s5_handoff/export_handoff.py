#!/usr/bin/env python3
"""Export S5 candidate bank and handoff manifest from v3 registries."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
import sys

SCRIPT_ROOT = next(parent for parent in Path(__file__).resolve().parents if parent.name == "data_construction_v3")
for _path in (SCRIPT_ROOT / "lib", SCRIPT_ROOT / "state", SCRIPT_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
from typing import Any

from common import SCHEMA_VERSION, json_dumps, read_tsv, timestamp, write_json, write_tsv


def key(row: dict[str, str]) -> tuple[str, str, str]:
    return (
        row.get("case_id", ""),
        row.get("retarget_variant_id", "shared") or "shared",
        row.get("target_variant_id", "ref_fk") or "ref_fk",
    )


def read_many_tsv(paths: list[Path] | None) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths or []:
        if path.is_file():
            rows.extend(read_tsv(path))
    return rows


def index_by_key(rows: list[dict[str, str]]) -> dict[tuple[str, str, str], dict[str, str]]:
    return {key(row): row for row in rows if row.get("case_id")}


def classify_candidate(row: dict[str, str]) -> tuple[str, str]:
    raw3 = row.get("raw_contact_3cm_status", "not_run")
    raw5 = row.get("raw_contact_5cm_status", "not_run")
    template = row.get("template_status", "not_run")
    stage2b = row.get("stage2b_status", "not_run")
    target_gate = row.get("target_gate_status", "not_run")
    visual = row.get("visual_qc_status", "not_run")
    retarget_variant = row.get("retarget_variant_id", "shared")
    target_variant = row.get("target_variant_id", "ref_fk")
    route_diag = row.get("route_diagnostic_status", "")

    if row.get("raw_inventory_status") == "invalid_raw":
        return "REJECT_RAW_INVENTORY", "raw inventory invalid"
    if raw3 == "reject" and raw5 == "reject":
        return "REJECT_RAW_CONTACT", "3cm and 5cm raw contact rejected"
    if target_variant == "fingertip_aware" and route_diag != "pass":
        return "REJECT_FINGERTIP_ROUTE_CONTRACT", "fingertip-aware route requires pass on E099-E101 diagnostics"
    if target_variant == "fingertip_aware" and row.get("fingertip_vote_status") in {"reject", "missing"}:
        return "REJECT_RAW_FINGERTIP_MISMATCH", "fingertip-aware route missing/rejects fingertip vote"
    if template == "backlog":
        return "REJECT_TEMPLATE_BACKLOG", "source template missing and must be built"
    if template in {"audit_fail", "manual_review_required"}:
        return "REJECT_TEMPLATE_AUDIT", f"template status is {template}"
    if stage2b in {"omniretarget_infeasible", "preprocess_fail"}:
        return "REJECT_OMNIRETARGET", f"Stage2b status is {stage2b}"
    if stage2b in {"variant_requires_checkout", "variant_unresolved", "variant_unavailable"}:
        return "VARIANT_PENDING", f"retarget variant requires adapter/confirmation: {stage2b}"
    if target_gate == "reject":
        return "REJECT_TARGET_GATE", "target gate rejected"
    if visual == "reject":
        return "REJECT_VISUAL_QC", "visual QC rejected"
    if target_gate == "pass" and visual == "pass":
        return "PASS", "machine gate and visual QC pass"
    if target_gate == "pass":
        return "REVIEW", "machine target gate pass; visual QC pending or review"
    if retarget_variant not in {"", "shared"} and stage2b == "pass":
        return "REVIEW", "Stage2b pass; target gate not complete"
    if retarget_variant not in {"", "shared"} and template in {"clean", "clean_reviewed"} and (raw3 in {"pass", "review"} or raw5 in {"pass", "review"}):
        return "STAGE2B_READY", "ready to run Stage2b for this variant"
    if raw3 in {"pass", "review"} or raw5 in {"pass", "review"}:
        return "RAW_CONTACT_READY", "raw contact ready; template/variant stages pending"
    if row.get("raw_inventory_status") == "present":
        return "INVENTORY_READY", "raw inventory present; raw contact pending"
    return "NOT_RUN", "no actionable evidence yet"


def handoff_decision(candidate_decision: str, row: dict[str, str]) -> str:
    if candidate_decision == "PASS":
        return "HANDOFF_READY"
    if candidate_decision == "REVIEW" and row.get("target_gate_status") == "pass":
        return "HANDOFF_REVIEW_VISUAL_QC"
    if candidate_decision == "STAGE2B_READY":
        return "HANDOFF_PENDING_STAGE2B"
    if candidate_decision == "RAW_CONTACT_READY":
        return "HANDOFF_PENDING_TEMPLATE_OR_VARIANT"
    if candidate_decision == "VARIANT_PENDING":
        return "HANDOFF_PENDING_TEMPLATE_OR_VARIANT"
    if candidate_decision.startswith("REJECT_"):
        return "NO_HANDOFF_REJECTED"
    return "NO_HANDOFF_PENDING"


def build_rows(
    registry_rows: list[dict[str, str]],
    stage2b_index: dict[tuple[str, str, str], dict[str, str]],
    gate_index: dict[tuple[str, str, str], dict[str, str]],
    include_rejected_in_handoff: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    candidate_rows: list[dict[str, Any]] = []
    handoff_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []
    for row in registry_rows:
        decision, reason = classify_candidate(row)
        k = key(row)
        s3 = stage2b_index.get(k, {})
        s4 = gate_index.get(k, {})
        handoff = handoff_decision(decision, row)
        candidate = {
            "case_id": row.get("case_id", ""),
            "object_key": row.get("object_key", ""),
            "object_name": row.get("object_name", ""),
            "date": row.get("date", ""),
            "seq": row.get("seq", ""),
            "person": row.get("person", ""),
            "person_idx": row.get("person_idx", ""),
            "retarget_variant_id": row.get("retarget_variant_id", ""),
            "target_variant_id": row.get("target_variant_id", ""),
            "candidate_decision": decision,
            "decision_reason": reason,
            "handoff_decision": handoff,
            "raw_inventory_status": row.get("raw_inventory_status", ""),
            "raw_contact_3cm_status": row.get("raw_contact_3cm_status", ""),
            "raw_contact_5cm_status": row.get("raw_contact_5cm_status", ""),
            "template_status": row.get("template_status", ""),
            "stage2b_status": row.get("stage2b_status", ""),
            "target_gate_status": row.get("target_gate_status", ""),
            "visual_qc_status": row.get("visual_qc_status", ""),
            "cem_status": row.get("cem_status", ""),
            "rl_status": row.get("rl_status", ""),
            "downstream_decision": row.get("downstream_decision", ""),
            "downstream_failure_mode": row.get("downstream_failure_mode", ""),
            "downstream_evidence_root": row.get("downstream_evidence_root", ""),
            "cem_run_id": row.get("cem_run_id", ""),
            "cem_result_npz": row.get("cem_result_npz", ""),
            "cem_video": row.get("cem_video", ""),
            "cem_metrics_ref": row.get("cem_metrics_ref", ""),
            "rl_run_id": row.get("rl_run_id", ""),
            "rl_checkpoint": row.get("rl_checkpoint", ""),
            "rl_metrics_ref": row.get("rl_metrics_ref", ""),
            "rl_video": row.get("rl_video", ""),
            "current_decision": row.get("current_decision", ""),
            "route_diagnostic_status": row.get("route_diagnostic_status", ""),
            "route_diagnostic_ref": row.get("route_diagnostic_ref", ""),
            "fingertip_vote_status": row.get("fingertip_vote_status", ""),
            "palm_vote_status": row.get("palm_vote_status", ""),
            "quat_audit_status": row.get("quat_audit_status", ""),
            "target_active_mask_status": row.get("target_active_mask_status", ""),
            "target_npz": row.get("target_npz", ""),
            "target_npz_sha256": row.get("target_npz_sha256", ""),
            "target_gap_status": row.get("target_gap_status", ""),
            "active_L_frac": row.get("active_L_frac", ""),
            "active_R_frac": row.get("active_R_frac", ""),
            "e101_route_evidence_status": row.get("e101_route_evidence_status", ""),
            "diagnostic_contracts": row.get("diagnostic_contracts", ""),
            "evidence_root": row.get("evidence_root", ""),
            "notes": row.get("notes", ""),
            "schema_version": SCHEMA_VERSION,
            "updated_at": timestamp(),
        }
        candidate_rows.append(candidate)

        handoff_row = {
            **candidate,
            "source_scene": s3.get("source_scene_xml", ""),
            "target_scene": s4.get("target_scene", ""),
            "trajectory": s4.get("trajectory", ""),
            "scene_act": s4.get("scene_act", ""),
            "contact_mask": s4.get("contact_mask_npz", ""),
            "raw_contact_threshold_label": s3.get("raw_contact_threshold_label", ""),
            "stage2b_target_task": s3.get("target_task", s4.get("target_task", "")),
            "stage2b_result_root": s3.get("result_root", ""),
            "stage2b_manifest_ref": s3.get("source_ref", ""),
            "target_gate_failure_mode": s4.get("failure_mode", ""),
            "target_gate_notes": s4.get("decision_notes", ""),
            "contact_pos_source": s4.get("contact_pos_source", ""),
            "pelvis_end_z": s4.get("pelvis_end_z", ""),
            "pelvis_tilt_end": s4.get("pelvis_tilt_end", ""),
        }
        if handoff.startswith("HANDOFF"):
            handoff_rows.append(handoff_row)
        elif handoff == "NO_HANDOFF_REJECTED":
            rejected_rows.append(handoff_row)
            if include_rejected_in_handoff:
                handoff_rows.append(handoff_row)
    order = {
        "PASS": 0,
        "REVIEW": 1,
        "STAGE2B_READY": 2,
        "RAW_CONTACT_READY": 3,
        "INVENTORY_READY": 4,
        "REJECT_RAW_CONTACT": 5,
        "REJECT_TEMPLATE_BACKLOG": 6,
        "REJECT_TEMPLATE_AUDIT": 7,
        "REJECT_OMNIRETARGET": 8,
        "REJECT_FINGERTIP_ROUTE_CONTRACT": 9,
        "REJECT_TARGET_GATE": 10,
        "REJECT_VISUAL_QC": 11,
    }
    candidate_rows.sort(key=lambda r: (order.get(str(r["candidate_decision"]), 99), r["case_id"], r["retarget_variant_id"], r["target_variant_id"]))
    handoff_rows.sort(key=lambda r: (r["handoff_decision"], r["case_id"], r["retarget_variant_id"], r["target_variant_id"]))
    rejected_rows.sort(key=lambda r: (r["candidate_decision"], r["case_id"], r["retarget_variant_id"], r["target_variant_id"]))
    return candidate_rows, handoff_rows, rejected_rows


def summarize(
    candidate_rows: list[dict[str, Any]],
    handoff_rows: list[dict[str, Any]],
    rejected_rows: list[dict[str, Any]],
    out_dir: Path,
) -> dict[str, Any]:
    return {
        "stage": "S5_candidate_bank_handoff",
        "created_at": timestamp(),
        "schema_version": SCHEMA_VERSION,
        "candidate_rows": len(candidate_rows),
        "handoff_rows": len(handoff_rows),
        "rejected_rows": len(rejected_rows),
        "candidate_decision_counts": dict(Counter(row["candidate_decision"] for row in candidate_rows)),
        "handoff_decision_counts": dict(Counter(row["handoff_decision"] for row in handoff_rows)),
        "rejected_decision_counts": dict(Counter(row["candidate_decision"] for row in rejected_rows)),
        "variant_counts": dict(Counter(row["retarget_variant_id"] for row in candidate_rows)),
        "out_dir": str(out_dir),
    }


def markdown_summary(summary: dict[str, Any], candidate_rows: list[dict[str, Any]], handoff_rows: list[dict[str, Any]]) -> str:
    lines = [
        "# S5 candidate bank / handoff summary",
        "",
        f"- candidate rows: `{summary['candidate_rows']}`",
        f"- handoff rows: `{summary['handoff_rows']}`",
        f"- rejected rows: `{summary['rejected_rows']}`",
        "",
        "## candidate decisions",
        "",
        "| decision | count |",
        "|---|---:|",
    ]
    for key, count in summary["candidate_decision_counts"].items():
        lines.append(f"| `{key}` | {count} |")
    lines.extend(["", "## handoff decisions", "", "| decision | count |", "|---|---:|"])
    for key, count in summary["handoff_decision_counts"].items():
        lines.append(f"| `{key}` | {count} |")
    lines.extend(["", "## handoff rows", "", "| handoff | case | variant | target | reason |", "|---|---|---|---|---|"])
    for row in handoff_rows[:50]:
        lines.append(
            f"| `{row['handoff_decision']}` | `{row['case_id']}` | `{row['retarget_variant_id']}` | `{row['target_variant_id']}` | `{row['decision_reason']}` |"
        )
    lines.extend(
        [
            "",
            "说明：S5 不把 CEM/RL 成败反向定义为数据构建成败；`cem_status` 和 `rl_status` 只作为 downstream evidence。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-state-registry", type=Path, required=True)
    parser.add_argument("--stage2b-manifest-tsv", type=Path, action="append", default=[])
    parser.add_argument("--target-gate-manifest-tsv", type=Path, action="append", default=[])
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--include-rejected-in-handoff", action="store_true")
    args = parser.parse_args()

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    registry_rows = read_tsv(args.case_state_registry)
    stage2b_index = index_by_key(read_many_tsv(args.stage2b_manifest_tsv))
    gate_index = index_by_key(read_many_tsv(args.target_gate_manifest_tsv))
    candidate_rows, handoff_rows, rejected_rows = build_rows(
        registry_rows,
        stage2b_index,
        gate_index,
        include_rejected_in_handoff=args.include_rejected_in_handoff,
    )
    candidate_fields = list(candidate_rows[0].keys()) if candidate_rows else []
    handoff_fields = list(handoff_rows[0].keys()) if handoff_rows else []
    rejected_fields = list(rejected_rows[0].keys()) if rejected_rows else []
    write_tsv(out_dir / "candidate_bank.tsv", candidate_rows, candidate_fields)
    write_json(out_dir / "candidate_bank.json", candidate_rows)
    write_tsv(out_dir / "handoff_manifest.tsv", handoff_rows, handoff_fields)
    write_json(out_dir / "handoff_manifest.json", handoff_rows)
    write_tsv(out_dir / "rejected_manifest.tsv", rejected_rows, rejected_fields)
    write_json(out_dir / "rejected_manifest.json", rejected_rows)
    summary = summarize(candidate_rows, handoff_rows, rejected_rows, out_dir)
    write_json(out_dir / "handoff_summary.json", summary)
    (out_dir / "handoff_summary.md").write_text(markdown_summary(summary, candidate_rows, handoff_rows), encoding="utf-8")
    print(json_dumps(summary))


if __name__ == "__main__":
    main()
