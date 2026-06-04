#!/usr/bin/env python3
"""Initialize or merge Core4D v3 case-state registry rows."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

SCRIPT_ROOT = next(parent for parent in Path(__file__).resolve().parents if parent.name == "data_construction_v3")
for _path in (SCRIPT_ROOT / "lib", SCRIPT_ROOT / "state", SCRIPT_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from common import SCHEMA_VERSION, read_tsv, timestamp, write_json, write_tsv


FIELDS = [
    "case_id",
    "object_key",
    "object_name",
    "date",
    "seq",
    "person",
    "person_idx",
    "raw_inventory_status",
    "raw_contact_3cm_status",
    "raw_contact_5cm_status",
    "template_status",
    "retarget_variant_id",
    "stage2b_status",
    "target_variant_id",
    "target_gate_status",
    "visual_qc_status",
    "cem_status",
    "rl_status",
    "downstream_decision",
    "downstream_failure_mode",
    "downstream_evidence_root",
    "cem_run_id",
    "cem_result_npz",
    "cem_video",
    "cem_metrics_ref",
    "rl_run_id",
    "rl_checkpoint",
    "rl_metrics_ref",
    "rl_video",
    "current_decision",
    "route_diagnostic_status",
    "route_diagnostic_ref",
    "contact_mask_npz",
    "contact_mask_status",
    "contact_mask_label",
    "contact_mask_person_idx",
    "contact_mask_time_axis",
    "contact_mask_3cm_npz",
    "contact_mask_5cm_npz",
    "raw_contact_artifact_npz",
    "raw_contact_3cm_artifact_npz",
    "raw_contact_5cm_artifact_npz",
    "raw_contact_time_axis",
    "raw_to_trimmed_mapping_status",
    "left_active_frac",
    "right_active_frac",
    "both_active_frac",
    "left_longest_run_frac",
    "right_longest_run_frac",
    "both_longest_run_frac",
    "contact_target_status",
    "contact_target_npz",
    "contact_target_source",
    "contact_target_frame",
    "contact_target_time_axis",
    "contact_route_diagnostic_ref",
    "fingertip_vote_status",
    "palm_vote_status",
    "quat_audit_status",
    "face_changed_l",
    "face_changed_r",
    "disable_world_up",
    "target_active_mask_status",
    "target_npz",
    "target_npz_sha256",
    "target_gap_status",
    "active_L_frac",
    "active_R_frac",
    "e101_route_evidence_status",
    "diagnostic_contracts",
    "evidence_root",
    "source_type",
    "source_ref",
    "schema_version",
    "updated_at",
    "notes",
]


DEFAULT_ROW = {
    "raw_inventory_status": "not_seen",
    "raw_contact_3cm_status": "not_run",
    "raw_contact_5cm_status": "not_run",
    "template_status": "not_run",
    "retarget_variant_id": "shared",
    "stage2b_status": "not_run",
    "target_variant_id": "ref_fk",
    "target_gate_status": "not_run",
    "visual_qc_status": "not_run",
    "cem_status": "not_run",
    "rl_status": "not_run",
    "current_decision": "not_run",
    "diagnostic_contracts": "E098_global",
    "source_type": "v3_run",
    "schema_version": SCHEMA_VERSION,
}

TEMPLATE_CLEAN_STATUSES = {"clean", "clean_reviewed"}


def row_key(row: dict[str, str]) -> tuple[str, str, str]:
    return (
        row.get("case_id", ""),
        row.get("retarget_variant_id", "shared") or "shared",
        row.get("target_variant_id", "ref_fk") or "ref_fk",
    )


def normalize_row(row: dict[str, str], source_type: str | None = None, source_ref: str | None = None) -> dict[str, str]:
    out = {field: "" for field in FIELDS}
    out.update(DEFAULT_ROW)
    for field in FIELDS:
        if field in row and row[field] not in (None, ""):
            out[field] = str(row[field])
    if source_type:
        out["source_type"] = source_type
    if source_ref:
        out["source_ref"] = source_ref
    out["schema_version"] = out.get("schema_version") or SCHEMA_VERSION
    out["updated_at"] = out.get("updated_at") or timestamp()
    if out.get("target_variant_id") == "fingertip_aware" and "E099" not in out.get("diagnostic_contracts", ""):
        out["diagnostic_contracts"] = "E098_global+E099_E100_E101_fingertip_aware"
    return out


def status_from_contact_decision(decision: str) -> str:
    if decision == "raw_contact_pass":
        return "pass"
    if decision == "raw_contact_review":
        return "review"
    if decision in {"raw_contact_fail", "raw_contact_error"}:
        return "reject"
    return "not_run"


def current_decision_from_registry(row: dict[str, str]) -> str:
    if row.get("raw_inventory_status") == "invalid_raw":
        return "REJECT_RAW_INVENTORY"
    if row.get("raw_contact_3cm_status") == "reject" and row.get("raw_contact_5cm_status") == "reject":
        return "REJECT_RAW_CONTACT"
    if row.get("template_status") == "backlog":
        return "REJECT_TEMPLATE_BACKLOG"
    if row.get("template_status") in {"audit_fail", "manual_review_required"}:
        return "REJECT_TEMPLATE_AUDIT"
    if row.get("target_variant_id") == "fingertip_aware" and row.get("route_diagnostic_status") not in {"pass"}:
        return "REJECT_FINGERTIP_ROUTE_CONTRACT"
    if row.get("visual_qc_status") == "reject":
        return "REJECT_VISUAL_QC"
    if row.get("stage2b_status") == "pass":
        if row.get("target_gate_status") == "pass":
            if row.get("visual_qc_status") == "pass":
                return "VISUAL_QC_PASS"
            return "TARGET_GATE_PASS"
        if row.get("target_gate_status") == "reject":
            return "REJECT_TARGET_GATE"
        return "STAGE2B_PASS"
    if row.get("stage2b_status") in {"omniretarget_infeasible", "preprocess_fail"}:
        return "REJECT_OMNIRETARGET"
    if row.get("stage2b_status") in {"variant_requires_checkout", "variant_unresolved", "variant_unavailable"}:
        return "PENDING_VARIANT_ADAPTER"
    if (
        row.get("retarget_variant_id") not in {"", "shared"}
        and row.get("stage2b_status") == "not_run"
        and row.get("template_status") in TEMPLATE_CLEAN_STATUSES
        and (row.get("raw_contact_3cm_status") in {"pass", "review"} or row.get("raw_contact_5cm_status") in {"pass", "review"})
    ):
        return "STAGE2B_READY"
    if row.get("raw_contact_3cm_status") in {"pass", "review"} or row.get("raw_contact_5cm_status") in {"pass", "review"}:
        return "RAW_CONTACT_READY"
    if row.get("raw_inventory_status") == "present":
        return "INVENTORY_READY"
    return row.get("current_decision", "not_run") or "not_run"


def registry_rows_from_inventory(rows: list[dict[str, str]], evidence_root: str, source_ref: str | None) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for row in rows:
        group = row.get("decision_group", "")
        template_status = "clean" if row.get("source_scene_template_status") == "clean_or_existing" else "backlog"
        reg = {
            "case_id": row.get("case_id", ""),
            "object_key": row.get("object_key", ""),
            "object_name": row.get("object_name", ""),
            "date": row.get("date", ""),
            "seq": row.get("seq", ""),
            "person": row.get("person", ""),
            "person_idx": row.get("person_idx", ""),
            "raw_inventory_status": "present" if group != "抛弃/跳过" else "invalid_raw",
            "raw_contact_3cm_status": "not_run",
            "raw_contact_5cm_status": "not_run",
            "template_status": template_status,
            "retarget_variant_id": "shared",
            "target_variant_id": "ref_fk",
            "current_decision": "INVENTORY_READY" if group != "抛弃/跳过" else "REJECT_RAW_INVENTORY",
            "evidence_root": evidence_root,
            "source_type": "v3_run",
            "source_ref": source_ref or "S1_inventory",
            "notes": row.get("hard_reject_reason") or row.get("review_flags", ""),
        }
        out.append(normalize_row(reg))
    return out


def registry_rows_from_visual_qc_manifest(
    rows: list[dict[str, str]],
    existing_rows: list[dict[str, str]],
    evidence_root: str,
    source_ref: str | None,
) -> list[dict[str, str]]:
    existing_by_key = {
        row_key(normalize_row(row)): normalize_row(row)
        for row in existing_rows
        if row.get("case_id")
    }
    out: list[dict[str, str]] = []
    for row in rows:
        key = (
            row.get("case_id", ""),
            row.get("retarget_variant_id", "shared") or "shared",
            row.get("target_variant_id", "ref_fk") or "ref_fk",
        )
        base = existing_by_key.get(key, {})
        reg = {
            **base,
            "case_id": row.get("case_id", ""),
            "object_key": row.get("object_key", base.get("object_key", "")),
            "object_name": row.get("object_name", base.get("object_name", "")),
            "date": row.get("date", base.get("date", "")),
            "seq": row.get("seq", base.get("seq", "")),
            "person": row.get("person", base.get("person", "")),
            "person_idx": row.get("person_idx", base.get("person_idx", "")),
            "retarget_variant_id": row.get("retarget_variant_id", base.get("retarget_variant_id", "")),
            "target_variant_id": row.get("target_variant_id", base.get("target_variant_id", "ref_fk")),
            "target_gate_status": row.get("target_gate_status", base.get("target_gate_status", "not_run")) or "not_run",
            "visual_qc_status": row.get("visual_qc_status", base.get("visual_qc_status", "not_run")) or "not_run",
            "evidence_root": evidence_root,
            "source_type": "v3_run",
            "source_ref": source_ref or "S4_visual_qc_manifest",
            "notes": row.get("review_notes", base.get("notes", "")),
        }
        norm = normalize_row(reg)
        norm["current_decision"] = current_decision_from_registry(norm)
        out.append(norm)
    return out


def registry_rows_from_downstream_evidence(
    rows: list[dict[str, str]],
    existing_rows: list[dict[str, str]],
    evidence_root: str,
    source_ref: str | None,
) -> list[dict[str, str]]:
    existing_by_key = {
        row_key(normalize_row(row)): normalize_row(row)
        for row in existing_rows
        if row.get("case_id")
    }
    out: list[dict[str, str]] = []
    for row in rows:
        key = (
            row.get("case_id", ""),
            row.get("retarget_variant_id", "shared") or "shared",
            row.get("target_variant_id", "ref_fk") or "ref_fk",
        )
        base = existing_by_key.get(key, {})
        reg = {
            **base,
            "case_id": row.get("case_id", ""),
            "object_key": row.get("object_key", base.get("object_key", "")),
            "object_name": row.get("object_name", base.get("object_name", "")),
            "date": row.get("date", base.get("date", "")),
            "seq": row.get("seq", base.get("seq", "")),
            "person": row.get("person", base.get("person", "")),
            "person_idx": row.get("person_idx", base.get("person_idx", "")),
            "retarget_variant_id": row.get("retarget_variant_id", base.get("retarget_variant_id", "")),
            "target_variant_id": row.get("target_variant_id", base.get("target_variant_id", "ref_fk")),
            "cem_status": row.get("cem_status", base.get("cem_status", "not_run")) or "not_run",
            "rl_status": row.get("rl_status", base.get("rl_status", "not_run")) or "not_run",
            "downstream_decision": row.get("downstream_decision", ""),
            "downstream_failure_mode": row.get("downstream_failure_mode", ""),
            "downstream_evidence_root": evidence_root or row.get("downstream_evidence_root", ""),
            "cem_run_id": row.get("cem_run_id", ""),
            "cem_result_npz": row.get("cem_result_npz", ""),
            "cem_video": row.get("cem_video", ""),
            "cem_metrics_ref": row.get("cem_metrics_ref", ""),
            "rl_run_id": row.get("rl_run_id", ""),
            "rl_checkpoint": row.get("rl_checkpoint", ""),
            "rl_metrics_ref": row.get("rl_metrics_ref", ""),
            "rl_video": row.get("rl_video", ""),
            "evidence_root": base.get("evidence_root", evidence_root),
            "source_type": "v3_run",
            "source_ref": source_ref or row.get("source_ref", "S6_downstream_evidence"),
            "notes": row.get("downstream_notes", base.get("notes", "")),
        }
        norm = normalize_row(reg)
        norm["current_decision"] = current_decision_from_registry(norm)
        out.append(norm)
    return out


def registry_rows_from_raw_contact(
    rows: list[dict[str, str]],
    evidence_root: str,
    source_ref: str | None,
    contact_label: str | None,
) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for row in rows:
        label = contact_label or row.get("contact_threshold_label", "")
        contact_field = f"raw_contact_{label}_status"
        if contact_field not in FIELDS:
            raise ValueError(f"unsupported contact label {label}; expected 3cm or 5cm")
        reg = {
            "case_id": row.get("case_id", ""),
            "object_key": row.get("object_key", ""),
            "object_name": row.get("object_name", ""),
            "date": row.get("date", ""),
            "seq": row.get("seq", ""),
            "person": row.get("person", ""),
            "person_idx": row.get("person_idx", ""),
            "raw_inventory_status": "present",
            "template_status": "clean" if row.get("source_scene_template_status") == "clean_or_existing" else "backlog",
            "retarget_variant_id": "shared",
            "target_variant_id": "ref_fk",
            contact_field: status_from_contact_decision(row.get("raw_contact_decision", "")),
            "contact_mask_npz": row.get("contact_mask_npz", ""),
            "contact_mask_status": "raw_mask_available" if row.get("contact_mask_npz") else "",
            "contact_mask_label": row.get("contact_label", label),
            "contact_mask_person_idx": row.get("contact_person_idx", row.get("person_idx", "")),
            "contact_mask_time_axis": "raw_sequence",
            f"contact_mask_{label}_npz": row.get("contact_mask_npz", ""),
            "raw_contact_artifact_npz": row.get("contact_mask_npz", ""),
            f"raw_contact_{label}_artifact_npz": row.get("contact_mask_npz", ""),
            "raw_contact_time_axis": "raw_sequence",
            "raw_to_trimmed_mapping_status": row.get("raw_to_trimmed_mapping_status", ""),
            "left_active_frac": row.get("left_active_frac", row.get("target_left_active_frac", "")),
            "right_active_frac": row.get("right_active_frac", row.get("target_right_active_frac", "")),
            "both_active_frac": row.get("both_active_frac", row.get("target_both_active_frac", "")),
            "left_longest_run_frac": row.get("left_longest_run_frac", ""),
            "right_longest_run_frac": row.get("right_longest_run_frac", ""),
            "both_longest_run_frac": row.get("both_longest_run_frac", row.get("target_both_longest_run_active_frac", "")),
            "contact_target_status": row.get("contact_target_status", ""),
            "evidence_root": evidence_root,
            "source_type": "v3_run",
            "source_ref": source_ref or f"S1_raw_contact_{label}",
            "notes": row.get("raw_contact_notes", ""),
        }
        norm = normalize_row(reg)
        norm["current_decision"] = current_decision_from_registry(norm)
        out.append(norm)
    return out


def registry_rows_from_template_backlog(
    template_rows: list[dict[str, str]],
    existing_rows: list[dict[str, str]],
    evidence_root: str,
    source_ref: str | None,
) -> list[dict[str, str]]:
    by_task = {row.get("source_scene_task", ""): row for row in template_rows}
    incoming: list[dict[str, str]] = []
    for existing in existing_rows:
        norm = normalize_row(existing)
        object_key = norm.get("object_key", "")
        person = norm.get("person", "")
        if not object_key or person not in {"person1", "person2"}:
            continue
        task = f"{object_key}_{person}"
        template = by_task.get(task)
        if not template:
            continue
        reg = {
            **norm,
            "template_status": template.get("template_status", norm.get("template_status", "not_run")),
            "evidence_root": evidence_root or norm.get("evidence_root", ""),
            "source_type": "v3_run",
            "source_ref": source_ref or "S2_template_backlog_audit",
            "notes": template.get("recommended_action", "") + ("," + template.get("notes", "") if template.get("notes") else ""),
        }
        reg["current_decision"] = current_decision_from_registry(reg)
        incoming.append(normalize_row(reg))
    return incoming


def review_template_status(decision: str) -> str:
    if decision == "approve_clean":
        return "clean_reviewed"
    if decision == "reject":
        return "audit_fail"
    return "manual_review_required"


def registry_rows_from_template_review(
    review_rows: list[dict[str, str]],
    existing_rows: list[dict[str, str]],
    evidence_root: str,
    source_ref: str | None,
) -> list[dict[str, str]]:
    by_task = {row.get("source_scene_task", ""): row for row in review_rows}
    incoming: list[dict[str, str]] = []
    for existing in existing_rows:
        norm = normalize_row(existing)
        object_key = norm.get("object_key", "")
        person = norm.get("person", "")
        if not object_key or person not in {"person1", "person2"}:
            continue
        task = f"{object_key}_{person}"
        review = by_task.get(task)
        if not review:
            continue
        decision = review.get("review_decision", "")
        status = review_template_status(decision)
        notes = [
            f"nonbox_template_review={decision or 'missing_decision'}",
            f"reviewer={review.get('reviewer', '')}",
            review.get("review_notes", ""),
            f"collision_policy={review.get('approved_collision_policy', '')}",
            f"mass_policy={review.get('approved_mass_policy', '')}",
        ]
        reg = {
            **norm,
            "template_status": status,
            "evidence_root": evidence_root or norm.get("evidence_root", ""),
            "source_type": "v3_run",
            "source_ref": source_ref or "S2_nonbox_template_review",
            "notes": ",".join(part for part in notes if part),
        }
        reg["current_decision"] = current_decision_from_registry(reg)
        incoming.append(normalize_row(reg))
    return incoming


def registry_rows_from_stage2b_manifest(
    rows: list[dict[str, str]],
    existing_rows: list[dict[str, str]],
    evidence_root: str,
    source_ref: str | None,
) -> list[dict[str, str]]:
    shared_by_case = {
        row.get("case_id", ""): normalize_row(row)
        for row in existing_rows
        if row.get("case_id") and (row.get("retarget_variant_id") or "shared") == "shared"
    }
    out: list[dict[str, str]] = []
    for row in rows:
        shared = shared_by_case.get(row.get("case_id", ""), {})
        decision = row.get("stage2b_decision", "")
        status = row.get("stage2b_status", "not_run") or "not_run"
        if decision == "stage2b_ready":
            status = "not_run"
        elif "infeasible" in decision:
            status = "omniretarget_infeasible"
        elif decision == "stage2b_variant_requires_solver_checkout":
            status = "variant_requires_checkout"
        elif decision == "stage2b_variant_unresolved":
            status = "variant_unresolved"
        elif decision == "stage2b_variant_unavailable":
            status = "variant_unavailable"
        elif decision.startswith("stage2b_template_"):
            status = "not_run"
        elif decision not in {"", "stage2b_ready"} and status == "not_run":
            status = "preprocess_fail" if "fail" in decision or "error" in decision else "not_run"
        reg = {
            "case_id": row.get("case_id", ""),
            "object_key": row.get("object_key", ""),
            "object_name": row.get("object_name", ""),
            "date": row.get("date", ""),
            "seq": row.get("seq", ""),
            "person": row.get("person", ""),
            "person_idx": row.get("person_idx", ""),
            "raw_inventory_status": "present",
            "raw_contact_3cm_status": shared.get("raw_contact_3cm_status", "not_run"),
            "raw_contact_5cm_status": shared.get("raw_contact_5cm_status", "not_run"),
            "template_status": row.get("source_template_status", "") or shared.get("template_status", "not_run"),
            "retarget_variant_id": row.get("retarget_variant_id", ""),
            "target_variant_id": row.get("target_variant_id", "ref_fk"),
            "stage2b_status": status,
            "diagnostic_contracts": (
                "E098_global+E099_E100_E101_fingertip_aware"
                if row.get("target_variant_id") == "fingertip_aware"
                else "E098_global"
            ),
            "route_diagnostic_status": row.get("route_diagnostic_status", ""),
            "route_diagnostic_ref": row.get("route_diagnostic_ref", ""),
            "contact_mask_npz": row.get("contact_mask_npz", ""),
            "contact_mask_status": row.get("contact_mask_status", shared.get("contact_mask_status", "")),
            "contact_mask_label": row.get("contact_mask_label", row.get("raw_contact_threshold_label", "")),
            "contact_mask_person_idx": row.get("contact_mask_person_idx", row.get("person_idx", "")),
            "contact_mask_time_axis": row.get("contact_mask_time_axis", ""),
            "contact_mask_3cm_npz": shared.get("contact_mask_3cm_npz", ""),
            "contact_mask_5cm_npz": shared.get("contact_mask_5cm_npz", ""),
            "raw_contact_artifact_npz": row.get("raw_contact_artifact_npz", shared.get("raw_contact_artifact_npz", "")),
            "raw_contact_3cm_artifact_npz": shared.get("raw_contact_3cm_artifact_npz", ""),
            "raw_contact_5cm_artifact_npz": shared.get("raw_contact_5cm_artifact_npz", ""),
            "raw_contact_time_axis": row.get("raw_contact_time_axis", shared.get("raw_contact_time_axis", "")),
            "raw_to_trimmed_mapping_status": row.get("raw_to_trimmed_mapping_status", shared.get("raw_to_trimmed_mapping_status", "")),
            "left_active_frac": row.get("left_active_frac", shared.get("left_active_frac", "")),
            "right_active_frac": row.get("right_active_frac", shared.get("right_active_frac", "")),
            "both_active_frac": row.get("both_active_frac", shared.get("both_active_frac", "")),
            "left_longest_run_frac": row.get("left_longest_run_frac", shared.get("left_longest_run_frac", "")),
            "right_longest_run_frac": row.get("right_longest_run_frac", shared.get("right_longest_run_frac", "")),
            "both_longest_run_frac": row.get("both_longest_run_frac", shared.get("both_longest_run_frac", "")),
            "contact_target_status": row.get("contact_target_status", shared.get("contact_target_status", "")),
            "contact_target_npz": row.get("contact_target_npz", row.get("target_npz", "")),
            "contact_target_source": row.get("contact_target_source", ""),
            "contact_target_frame": row.get("contact_target_frame", ""),
            "contact_target_time_axis": row.get("contact_target_time_axis", ""),
            "contact_route_diagnostic_ref": row.get("contact_route_diagnostic_ref", row.get("route_diagnostic_ref", "")),
            "fingertip_vote_status": row.get("fingertip_vote_status", ""),
            "palm_vote_status": row.get("palm_vote_status", ""),
            "quat_audit_status": row.get("quat_audit_status", ""),
            "face_changed_l": row.get("face_changed_l", ""),
            "face_changed_r": row.get("face_changed_r", ""),
            "disable_world_up": row.get("disable_world_up", ""),
            "target_active_mask_status": row.get("target_active_mask_status", ""),
            "target_npz": row.get("target_npz", ""),
            "target_npz_sha256": row.get("target_npz_sha256", ""),
            "target_gap_status": row.get("target_gap_status", ""),
            "active_L_frac": row.get("active_L_frac", ""),
            "active_R_frac": row.get("active_R_frac", ""),
            "e101_route_evidence_status": row.get("e101_route_evidence_status", ""),
            "evidence_root": evidence_root or row.get("result_root", ""),
            "source_type": "v3_run",
            "source_ref": source_ref or "S3_stage2b_manifest",
            "notes": row.get("stage2b_decision", "") + ("," + row.get("decision_notes", "") if row.get("decision_notes") else ""),
        }
        norm = normalize_row(reg)
        norm["current_decision"] = current_decision_from_registry(norm)
        out.append(norm)
    return out


def registry_rows_from_target_gate_manifest(
    rows: list[dict[str, str]],
    existing_rows: list[dict[str, str]],
    evidence_root: str,
    source_ref: str | None,
) -> list[dict[str, str]]:
    existing_by_key = {
        row_key(normalize_row(row)): normalize_row(row)
        for row in existing_rows
        if row.get("case_id")
    }
    out: list[dict[str, str]] = []
    for row in rows:
        key = (
            row.get("case_id", ""),
            row.get("retarget_variant_id", "shared") or "shared",
            row.get("target_variant_id", "ref_fk") or "ref_fk",
        )
        base = existing_by_key.get(key, {})
        gate_status = row.get("target_gate_status", "not_run") or "not_run"
        stage2b_status = base.get("stage2b_status", "not_run")
        if gate_status in {"pass", "reject", "review"}:
            stage2b_status = "pass"
        reg = {
            **base,
            "case_id": row.get("case_id", ""),
            "object_key": row.get("object_key", ""),
            "object_name": row.get("object_name", ""),
            "date": row.get("date", ""),
            "seq": row.get("seq", ""),
            "person": row.get("person", ""),
            "person_idx": row.get("person_idx", ""),
            "retarget_variant_id": row.get("retarget_variant_id", ""),
            "target_variant_id": row.get("target_variant_id", "ref_fk"),
            "stage2b_status": stage2b_status,
            "target_gate_status": gate_status,
            "visual_qc_status": row.get("visual_qc_status", base.get("visual_qc_status", "not_run")) or "not_run",
            "diagnostic_contracts": (
                "E098_global+E099_E100_E101_fingertip_aware"
                if row.get("target_variant_id") == "fingertip_aware"
                else "E098_global"
            ),
            "route_diagnostic_status": base.get("route_diagnostic_status", ""),
            "route_diagnostic_ref": base.get("route_diagnostic_ref", ""),
            "contact_mask_npz": row.get("contact_mask_npz", base.get("contact_mask_npz", "")),
            "contact_mask_status": row.get("contact_mask_status", base.get("contact_mask_status", "")),
            "contact_mask_label": row.get("contact_mask_label", base.get("contact_mask_label", "")),
            "contact_mask_person_idx": row.get("contact_mask_person_idx", base.get("contact_mask_person_idx", "")),
            "contact_mask_time_axis": row.get("contact_mask_time_axis", base.get("contact_mask_time_axis", "")),
            "contact_mask_3cm_npz": base.get("contact_mask_3cm_npz", ""),
            "contact_mask_5cm_npz": base.get("contact_mask_5cm_npz", ""),
            "raw_contact_artifact_npz": row.get("raw_contact_artifact_npz", base.get("raw_contact_artifact_npz", "")),
            "raw_contact_3cm_artifact_npz": base.get("raw_contact_3cm_artifact_npz", ""),
            "raw_contact_5cm_artifact_npz": base.get("raw_contact_5cm_artifact_npz", ""),
            "raw_contact_time_axis": row.get("raw_contact_time_axis", base.get("raw_contact_time_axis", "")),
            "raw_to_trimmed_mapping_status": row.get("raw_to_trimmed_mapping_status", base.get("raw_to_trimmed_mapping_status", "")),
            "left_active_frac": base.get("left_active_frac", ""),
            "right_active_frac": base.get("right_active_frac", ""),
            "both_active_frac": base.get("both_active_frac", ""),
            "left_longest_run_frac": base.get("left_longest_run_frac", ""),
            "right_longest_run_frac": base.get("right_longest_run_frac", ""),
            "both_longest_run_frac": base.get("both_longest_run_frac", ""),
            "contact_target_status": base.get("contact_target_status", ""),
            "contact_target_npz": row.get("contact_target_npz", base.get("contact_target_npz", "")),
            "contact_target_source": row.get("contact_target_source", base.get("contact_target_source", "")),
            "contact_target_frame": row.get("contact_target_frame", base.get("contact_target_frame", "")),
            "contact_target_time_axis": row.get("contact_target_time_axis", base.get("contact_target_time_axis", "")),
            "contact_route_diagnostic_ref": row.get("contact_route_diagnostic_ref", base.get("contact_route_diagnostic_ref", "")),
            "fingertip_vote_status": base.get("fingertip_vote_status", ""),
            "palm_vote_status": base.get("palm_vote_status", ""),
            "quat_audit_status": base.get("quat_audit_status", ""),
            "face_changed_l": base.get("face_changed_l", ""),
            "face_changed_r": base.get("face_changed_r", ""),
            "disable_world_up": base.get("disable_world_up", ""),
            "target_active_mask_status": base.get("target_active_mask_status", ""),
            "target_npz": base.get("target_npz", ""),
            "target_npz_sha256": base.get("target_npz_sha256", ""),
            "target_gap_status": base.get("target_gap_status", ""),
            "active_L_frac": base.get("active_L_frac", ""),
            "active_R_frac": base.get("active_R_frac", ""),
            "e101_route_evidence_status": base.get("e101_route_evidence_status", ""),
            "evidence_root": evidence_root,
            "source_type": "v3_run",
            "source_ref": source_ref or "S4_target_gate_manifest",
            "notes": row.get("failure_mode", "") + ("," + row.get("decision_notes", "") if row.get("decision_notes") else ""),
        }
        norm = normalize_row(reg)
        norm["current_decision"] = current_decision_from_registry(norm)
        out.append(norm)
    return out


def merge(existing: list[dict[str, str]], incoming: list[dict[str, str]]) -> list[dict[str, str]]:
    rows = {row_key(normalize_row(row)): normalize_row(row) for row in existing if row.get("case_id")}
    non_regressive_not_run = {
        "raw_contact_3cm_status",
        "raw_contact_5cm_status",
        "stage2b_status",
        "target_gate_status",
        "visual_qc_status",
        "cem_status",
        "rl_status",
    }
    for row in incoming:
        norm = normalize_row(row)
        if not norm.get("case_id"):
            continue
        key = row_key(norm)
        if key in rows:
            merged = rows[key].copy()
            for field, value in norm.items():
                if value not in ("", None):
                    if field in non_regressive_not_run and value == "not_run" and merged.get(field) not in {"", "not_run"}:
                        continue
                    merged[field] = value
            merged["current_decision"] = current_decision_from_registry(merged)
            rows[key] = merged
        else:
            norm["current_decision"] = current_decision_from_registry(norm)
            rows[key] = norm
    return [rows[key] for key in sorted(rows)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry-dir", type=Path, required=True)
    parser.add_argument("--init-empty", action="store_true")
    parser.add_argument("--input-tsv", type=Path, default=None)
    parser.add_argument("--from-inventory-tsv", type=Path, default=None)
    parser.add_argument("--from-raw-contact-tsv", type=Path, default=None)
    parser.add_argument("--from-template-backlog-tsv", type=Path, default=None)
    parser.add_argument("--from-template-review-tsv", type=Path, default=None)
    parser.add_argument("--from-stage2b-manifest-tsv", type=Path, default=None)
    parser.add_argument("--from-target-gate-manifest-tsv", type=Path, default=None)
    parser.add_argument("--from-visual-qc-manifest-tsv", type=Path, default=None)
    parser.add_argument("--from-downstream-evidence-tsv", type=Path, default=None)
    parser.add_argument("--raw-contact-label", choices=("3cm", "5cm"), default=None)
    parser.add_argument("--evidence-root", default="")
    parser.add_argument("--source-type", default=None)
    parser.add_argument("--source-ref", default=None)
    args = parser.parse_args()

    registry_dir = args.registry_dir.resolve()
    tsv_path = registry_dir / "case_state_registry.tsv"
    existing = read_tsv(tsv_path) if tsv_path.is_file() else []
    incoming: list[dict[str, str]] = []
    if args.input_tsv:
        incoming = [normalize_row(row, args.source_type, args.source_ref) for row in read_tsv(args.input_tsv)]
    if args.from_inventory_tsv:
        incoming.extend(registry_rows_from_inventory(read_tsv(args.from_inventory_tsv), args.evidence_root, args.source_ref))
    if args.from_raw_contact_tsv:
        incoming.extend(
            registry_rows_from_raw_contact(
                read_tsv(args.from_raw_contact_tsv),
                args.evidence_root,
                args.source_ref,
                args.raw_contact_label,
            )
        )
    if args.from_template_backlog_tsv:
        incoming.extend(
            registry_rows_from_template_backlog(
                read_tsv(args.from_template_backlog_tsv),
                existing + incoming,
                args.evidence_root,
                args.source_ref,
            )
        )
    if args.from_template_review_tsv:
        incoming.extend(
            registry_rows_from_template_review(
                read_tsv(args.from_template_review_tsv),
                existing + incoming,
                args.evidence_root,
                args.source_ref,
            )
        )
    if args.from_stage2b_manifest_tsv:
        incoming.extend(
            registry_rows_from_stage2b_manifest(
                read_tsv(args.from_stage2b_manifest_tsv),
                existing + incoming,
                args.evidence_root,
                args.source_ref,
            )
        )
    if args.from_target_gate_manifest_tsv:
        incoming.extend(
            registry_rows_from_target_gate_manifest(
                read_tsv(args.from_target_gate_manifest_tsv),
                existing + incoming,
                args.evidence_root,
                args.source_ref,
            )
        )
    if args.from_visual_qc_manifest_tsv:
        incoming.extend(
            registry_rows_from_visual_qc_manifest(
                read_tsv(args.from_visual_qc_manifest_tsv),
                existing + incoming,
                args.evidence_root,
                args.source_ref,
            )
        )
    if args.from_downstream_evidence_tsv:
        incoming.extend(
            registry_rows_from_downstream_evidence(
                read_tsv(args.from_downstream_evidence_tsv),
                existing + incoming,
                args.evidence_root,
                args.source_ref,
            )
        )

    rows = merge(existing, incoming)
    registry_dir.mkdir(parents=True, exist_ok=True)
    write_tsv(tsv_path, rows, FIELDS)
    write_json(registry_dir / "case_state_registry.json", rows)

    summary_lines = [
        "# Core4D v3 case state registry summary",
        "",
        f"- updated_at: {timestamp()}",
        f"- rows: {len(rows)}",
        f"- schema_version: {SCHEMA_VERSION}",
        "",
        "说明：S3 之后状态按 `(case_id, retarget_variant_id, target_variant_id)` 区分；默认 `ref_fk` 只绑定 E098，`fingertip_aware` 绑定 E098-E101。",
        "",
    ]
    (registry_dir / "case_state_summary.md").write_text("\n".join(summary_lines), encoding="utf-8")
    print(f"wrote {tsv_path} rows={len(rows)}")


if __name__ == "__main__":
    main()
