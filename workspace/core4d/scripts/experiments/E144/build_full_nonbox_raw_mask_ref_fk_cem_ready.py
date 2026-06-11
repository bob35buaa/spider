#!/usr/bin/env python3
"""Build E144 full-nonbox accounting, template audit, CEM-ready manifest, and overrides."""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


REPO = Path(__file__).resolve().parents[4]
TASK_ROOT = REPO / "example_datasets/processed/core4d/unitree_g1/humanoid_object"
OVERRIDE_ROOT = REPO / "examples/config/override"
SCRIPT_ROOT = REPO / "workspace/core4d/scripts/E144"
DEFAULT_SOURCE_RUN_ROOT = REPO / "workspace/core4d/results/E144/E144_full_nonbox_raw_contact"
DEFAULT_RUN_ROOT = REPO / "workspace/core4d/results/E144"
VARIANTS_TSV = SCRIPT_ROOT / "variants.tsv"

BASE_OVERRIDE = "core4d_E089A_box021_person1_upperobj"

ACCOUNTING_FIELDS = [
    "case_id",
    "object_key",
    "object_name",
    "object_category",
    "date",
    "seq",
    "person",
    "person_idx",
    "size_band",
    "action_family",
    "motion_quality",
    "decision_group",
    "hard_reject_reason",
    "source_scene_template_status",
    "source_scene_xml",
    "object_mesh_rel",
    "template_creation_strategy",
    "template_release_policy",
    "phase1_bucket",
    "raw_contact_3cm_decision",
    "raw_contact_5cm_decision",
    "raw_contact_3cm_notes",
    "raw_contact_5cm_notes",
    "object_rotation_active_deg",
    "object_lift_active_m",
    "accounting_decision",
    "accounting_reason",
]

TEMPLATE_FIELDS = [
    "source_template_key",
    "object_key",
    "object_category",
    "person",
    "source_scene_template_status",
    "source_scene_xml",
    "object_mesh_rel",
    "template_creation_strategy",
    "collision_policy",
    "auto_build_allowed",
    "auto_release_allowed",
    "manual_review_required",
    "case_person_rows",
    "flow_candidate_rows",
    "raw_contact_5cm_pass_rows",
    "template_notes",
]

VARIANT_FIELDS = [
    "ordinal",
    "variant",
    "case_id",
    "object_key",
    "object_category",
    "person",
    "person_idx",
    "split",
    "ablation",
    "retarget_variant_id",
    "target_variant_id",
    "derived_task",
    "target_scene",
    "trajectory",
    "scene_act",
    "mask_path",
    "mask_kind",
    "override",
    "handoff_decision",
    "candidate_decision",
    "raw_contact_threshold_label",
    "run_status",
]


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO.resolve()))
    except ValueError:
        return str(path)


def repo_path(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else REPO / path


def safe_id(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", text).strip("_")


def read_tsv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        lines = [line for line in f if line.strip() and not line.startswith("#")]
        if not lines:
            return []
        return list(csv.DictReader(lines, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, Any]], fields: list[str], comment: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        if comment:
            f.write(f"# {comment}\n")
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def template_strategy(category: str, mesh_rel: str) -> dict[str, str]:
    if not mesh_rel:
        return {
            "strategy": "missing_mesh_accounting_only",
            "collision_policy": "",
            "auto_build_allowed": "False",
            "auto_release_allowed": "False",
            "manual_review_required": "True",
            "release_policy": "blocked_missing_mesh",
        }
    if category == "bucket":
        return {
            "strategy": "nonbox_proxy_aabb_review",
            "collision_policy": "bucket_wall_proxy_aabb",
            "auto_build_allowed": "True",
            "auto_release_allowed": "False",
            "manual_review_required": "True",
            "release_policy": "clean_reviewed_required",
        }
    if category in {"board", "stick"}:
        return {
            "strategy": "nonbox_proxy_aabb_review",
            "collision_policy": "mesh_aabb_box_proxy",
            "auto_build_allowed": "True",
            "auto_release_allowed": "False",
            "manual_review_required": "True",
            "release_policy": "clean_reviewed_required",
        }
    if category in {"desk", "chair"}:
        return {
            "strategy": "manual_complex_shape",
            "collision_policy": "manual_review_defined",
            "auto_build_allowed": "False",
            "auto_release_allowed": "False",
            "manual_review_required": "True",
            "release_policy": "manual_shape_template_required",
        }
    return {
        "strategy": "manual_unknown_shape",
        "collision_policy": "manual_review_defined",
        "auto_build_allowed": "False",
        "auto_release_allowed": "False",
        "manual_review_required": "True",
        "release_policy": "manual_identity_required",
    }


def flow_candidate(row: dict[str, str]) -> bool:
    return (
        row.get("size_band") == "target_medium_between_box023_and_box025"
        and row.get("action_family") == "main_move_obs0"
        and row.get("motion_quality") == "motion_pass"
    )


def raw_maps(source_root: Path) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    raw_root = source_root / "s1_raw_contact/raw_contact"
    rows3 = read_tsv(raw_root / "raw_contact_candidates_3cm.tsv")
    rows5 = read_tsv(raw_root / "raw_contact_candidates_5cm.tsv")
    return ({row["case_id"]: row for row in rows3}, {row["case_id"]: row for row in rows5})


def accounting_rows(inventory: list[dict[str, str]], raw3: dict[str, dict[str, str]], raw5: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for inv in inventory:
        if inv.get("object_category") == "box":
            continue
        category = inv.get("object_category", "unknown") or "unknown"
        strategy = template_strategy(category, inv.get("object_mesh_rel", ""))
        is_flow = flow_candidate(inv)
        r3 = raw3.get(inv["case_id"], {})
        r5 = raw5.get(inv["case_id"], {})
        raw5_decision = r5.get("raw_contact_decision", "")
        if not is_flow:
            phase_bucket = "inventory_accounting_only"
            decision = "NOT_FLOW_CANDIDATE"
            reason = "size_action_or_motion_filter"
        elif category in {"desk", "chair"}:
            phase_bucket = "complex_template_review"
            decision = "WAIT_COMPLEX_TEMPLATE_REVIEW"
            reason = "desk_chair_manual_complex_shape"
        elif strategy["auto_build_allowed"] == "True":
            phase_bucket = "proxy_template_review_candidate"
            if raw5_decision == "raw_contact_pass":
                decision = "WAIT_TEMPLATE_REVIEW"
                reason = "raw_contact_pass_template_review_required"
            elif raw5_decision:
                decision = "RAW_CONTACT_NOT_PASS"
                reason = raw5_decision
            else:
                decision = "WAIT_RAW_CONTACT"
                reason = "not_in_source_raw_contact_candidates"
        else:
            phase_bucket = "manual_template_review"
            decision = "WAIT_MANUAL_TEMPLATE_REVIEW"
            reason = strategy["strategy"]
        out.append(
            {
                **{field: inv.get(field, "") for field in ACCOUNTING_FIELDS},
                "template_creation_strategy": strategy["strategy"],
                "template_release_policy": strategy["release_policy"],
                "phase1_bucket": phase_bucket,
                "raw_contact_3cm_decision": r3.get("raw_contact_decision", ""),
                "raw_contact_5cm_decision": raw5_decision,
                "raw_contact_3cm_notes": r3.get("raw_contact_notes", ""),
                "raw_contact_5cm_notes": r5.get("raw_contact_notes", ""),
                "object_rotation_active_deg": r5.get("object_rotation_active_deg", r3.get("object_rotation_active_deg", "")),
                "object_lift_active_m": r5.get("object_lift_active_m", r3.get("object_lift_active_m", "")),
                "accounting_decision": decision,
                "accounting_reason": reason,
            }
        )
    out.sort(key=lambda row: (row["object_category"], row["object_key"], row["date"], row["seq"], row["person"]))
    return out


def template_audit_rows(accounting: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in accounting:
        key = (row["object_key"], row["person"])
        grouped.setdefault(key, []).append(row)
    out: list[dict[str, Any]] = []
    for (object_key, person), rows in grouped.items():
        first = rows[0]
        strategy = template_strategy(first["object_category"], first.get("object_mesh_rel", ""))
        flow_rows = [row for row in rows if row["phase1_bucket"] != "inventory_accounting_only"]
        pass_rows = [row for row in rows if row.get("raw_contact_5cm_decision") == "raw_contact_pass"]
        notes = []
        if first["object_category"] in {"desk", "chair"}:
            notes.append("irregular_shape_requires_manual_collision_design")
        if first["source_scene_template_status"] == "backlog":
            notes.append("source_scene_missing_or_backlog")
        if strategy["auto_build_allowed"] == "True":
            notes.append("proxy_can_be_generated_for_review_only")
        out.append(
            {
                "source_template_key": f"{object_key}_{person}",
                "object_key": object_key,
                "object_category": first["object_category"],
                "person": person,
                "source_scene_template_status": first["source_scene_template_status"],
                "source_scene_xml": first["source_scene_xml"],
                "object_mesh_rel": first["object_mesh_rel"],
                "template_creation_strategy": strategy["strategy"],
                "collision_policy": strategy["collision_policy"],
                "auto_build_allowed": strategy["auto_build_allowed"],
                "auto_release_allowed": strategy["auto_release_allowed"],
                "manual_review_required": strategy["manual_review_required"],
                "case_person_rows": len(rows),
                "flow_candidate_rows": len(flow_rows),
                "raw_contact_5cm_pass_rows": len(pass_rows),
                "template_notes": ",".join(notes),
            }
        )
    out.sort(key=lambda row: (row["object_category"], row["object_key"], row["person"]))
    return out


def file_ok(path_text: str) -> bool:
    return bool(path_text) and repo_path(path_text).is_file()


def validate_mask(path_text: str, person_idx: str) -> tuple[bool, str, str]:
    if not path_text:
        return False, "missing_contact_mask", ""
    path = repo_path(path_text)
    if not path.is_file():
        return False, "contact_mask_missing", ""
    try:
        data = np.load(path, allow_pickle=True)
    except Exception as exc:  # noqa: BLE001
        return False, f"contact_mask_load_error:{type(exc).__name__}", ""
    key = "eval_contact_mask_3cm" if "eval_contact_mask_3cm" in data.files else "spider_contact_mask_3cm"
    if key not in data.files and "raw_contact_mask_3cm" not in data.files:
        return False, "contact_mask_3cm_key_missing", ""
    if key in data.files:
        arr = np.asarray(data[key])
        if arr.ndim != 3 or arr.shape[2] != 2:
            return False, f"{key}_bad_shape:{arr.shape}", key
        try:
            idx = int(person_idx)
        except ValueError:
            idx = 0
        if idx >= arr.shape[1]:
            return False, f"{key}_person_idx_out_of_range:{arr.shape}", key
    return True, "", key if key in data.files else "raw_contact_mask_3cm"


def split_for_index(index: int) -> str:
    return ["local-gpu0", "remote-gpu0", "remote-gpu1"][index % 3]


def localize_mask(source_mask: str, run_root: Path, case_id: str) -> str:
    src = repo_path(source_mask)
    dst = run_root / "contact_masks" / safe_id(case_id) / "raw_contact_mask_3cm.npz"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_file() and src.resolve() != dst.resolve():
        shutil.copy2(src, dst)
    return rel(dst)


def write_override(variant: str, task: str, mask_path: str, person_idx: str) -> Path:
    path = OVERRIDE_ROOT / f"core4d_{variant}.yaml"
    content = f"""# @package _global_
# Auto-generated by workspace/core4d/scripts/E144/build_full_nonbox_raw_mask_ref_fk_cem_ready.py.
# E144 full-nonbox raw_mask_ref_fk; variant={variant}.
defaults:
  - {BASE_OVERRIDE}
  - _self_

task: {task}

contact_hdmi_target_source: ref_fk
contact_hdmi_target_path: ""
contact_hdmi_target_uses_eef_offset: true
contact_hdmi_gain: 5.0
contact_hdmi_mask_source: "core4d_3cm"
contact_hdmi_mask_path: "{mask_path}"
contact_hdmi_mask_person_idx: {person_idx}
contact_hdmi_mask_time_axis: auto

hold_contact_rew_scale: 0.0
hold_contact_sigma: 0.05
hold_contact_start_eval_time: 0.0
hold_contact_end_eval_time: 0.0
hold_contact_require_ref_contact: true

video_camera: auto
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def build_variants(
    handoff_rows: list[dict[str, str]],
    accounting_by_case: dict[str, dict[str, Any]],
    run_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    variants: list[dict[str, Any]] = []
    preflight: list[dict[str, Any]] = []
    ready_rows = [
        row
        for row in handoff_rows
        if row.get("handoff_decision") == "HANDOFF_READY"
        and row.get("target_gate_status") == "pass"
        and row.get("visual_qc_status") == "pass"
        and row.get("target_variant_id", "ref_fk") == "ref_fk"
        and row.get("retarget_variant_id") == "omnirt_v1"
    ]
    ready_rows.sort(key=lambda row: (row.get("object_key", ""), row.get("case_id", ""), row.get("person", "")))
    for idx, row in enumerate(ready_rows, start=1):
        acc = accounting_by_case.get(row["case_id"], {})
        variant = f"E144_{safe_id(row['case_id'])}_raw_mask_ref_fk"
        task = row.get("stage2b_target_task") or row.get("target_task") or row["case_id"]
        source_mask_path = row.get("contact_mask", "")
        mask_path = localize_mask(source_mask_path, run_root, row["case_id"]) if source_mask_path else ""
        override = write_override(variant, task, mask_path, row.get("person_idx", "0"))
        item = {
            "ordinal": idx,
            "variant": variant,
            "case_id": row["case_id"],
            "object_key": row.get("object_key", ""),
            "object_category": acc.get("object_category", ""),
            "person": row.get("person", ""),
            "person_idx": row.get("person_idx", ""),
            "split": split_for_index(idx - 1),
            "ablation": "raw_mask_ref_fk",
            "retarget_variant_id": row.get("retarget_variant_id", ""),
            "target_variant_id": row.get("target_variant_id", ""),
            "derived_task": task,
            "target_scene": row.get("target_scene", ""),
            "trajectory": row.get("trajectory", ""),
            "scene_act": row.get("scene_act", ""),
            "mask_path": mask_path,
            "mask_kind": "s5_contact_mask_3cm",
            "override": rel(override),
            "handoff_decision": row.get("handoff_decision", ""),
            "candidate_decision": row.get("candidate_decision", ""),
            "raw_contact_threshold_label": row.get("raw_contact_threshold_label", ""),
            "run_status": "to_run",
        }
        variants.append(item)
        mask_ok, mask_failure, mask_key = validate_mask(mask_path, row.get("person_idx", "0"))
        preflight.append(
            {
                **item,
                "target_scene_exists": file_ok(row.get("target_scene", "")),
                "trajectory_exists": file_ok(row.get("trajectory", "")),
                "scene_act_exists": file_ok(row.get("scene_act", "")),
                "source_mask_path": source_mask_path,
                "source_mask_exists": file_ok(source_mask_path),
                "contact_mask_exists": file_ok(mask_path),
                "contact_mask_valid": mask_ok,
                "contact_mask_failure": mask_failure,
                "contact_mask_key": mask_key,
                "override_exists": repo_path(item["override"]).is_file(),
                "preflight_ok": (
                    file_ok(row.get("target_scene", ""))
                    and file_ok(row.get("trajectory", ""))
                    and file_ok(row.get("scene_act", ""))
                    and mask_ok
                    and repo_path(item["override"]).is_file()
                ),
            }
        )
    return variants, preflight


def markdown_summary(summary: dict[str, Any], variants: list[dict[str, Any]]) -> str:
    lines = [
        "# E144 Full Nonbox Raw-Mask Ref-FK CEM-Ready Summary",
        "",
        f"- source run root: `{summary['source_run_root']}`",
        f"- accounting rows: `{summary['accounting_rows']}`",
        f"- template audit rows: `{summary['template_audit_rows']}`",
        f"- CEM-ready variants: `{summary['variant_rows']}`",
        "",
        "## nonbox category counts",
        "",
        "| category | count |",
        "|---|---:|",
    ]
    for key, count in summary["category_counts"].items():
        lines.append(f"| `{key}` | {count} |")
    lines.extend(["", "## template strategies", "", "| strategy | count |", "|---|---:|"])
    for key, count in summary["template_strategy_counts"].items():
        lines.append(f"| `{key}` | {count} |")
    lines.extend(["", "## CEM-ready variants", "", "| variant | split | case | object | task |", "|---|---|---|---|---|"])
    for row in variants:
        lines.append(f"| `{row['variant']}` | `{row['split']}` | `{row['case_id']}` | `{row['object_key']}` | `{row['derived_task']}` |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run-root", type=Path, default=DEFAULT_SOURCE_RUN_ROOT)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--handoff-tsv", type=Path, default=None)
    args = parser.parse_args()

    source_root = args.source_run_root if args.source_run_root.is_absolute() else REPO / args.source_run_root
    run_root = args.run_root if args.run_root.is_absolute() else REPO / args.run_root
    inventory_path = source_root / "s1_raw_contact/inventory/inventory.tsv"
    handoff_path = args.handoff_tsv or source_root / "s5_handoff/handoff_manifest.tsv"
    if not inventory_path.is_file():
        raise SystemExit(
            "E144 source inventory missing. Run Phase 1 first, for example:\n"
            "workspace/core4d/scripts/data_construction_v3/orchestration/run_pipeline.py "
            "--mode full-from-raw --run-id E144_full_nonbox_raw_contact "
            "--run-root workspace/core4d/results/E144 --queue review-medium-nonbox "
            "--stage2b-contact-label 5cm --sample-count 500 --apply-build-templates"
        )

    inventory = read_tsv(inventory_path)
    raw3, raw5 = raw_maps(source_root)
    accounting = accounting_rows(inventory, raw3, raw5)
    template_audit = template_audit_rows(accounting)
    handoff_rows = read_tsv(handoff_path)
    variants, preflight = build_variants(handoff_rows, {row["case_id"]: row for row in accounting}, run_root)

    for path in OVERRIDE_ROOT.glob("core4d_E144_*_raw_mask_ref_fk.yaml"):
        if path.name not in {Path(row["override"]).name for row in variants}:
            path.unlink()

    write_tsv(run_root / "nonbox_inventory_accounting.tsv", accounting, ACCOUNTING_FIELDS, "E144 full nonbox inventory accounting")
    write_tsv(run_root / "template_creation_audit.tsv", template_audit, TEMPLATE_FIELDS, "E144 nonbox template creation audit")
    write_tsv(run_root / "cem_ready/raw_mask_ref_fk_cem_ready.tsv", variants, VARIANT_FIELDS, "E144 raw_mask_ref_fk CEM-ready rows")
    preflight_fields = sorted({key for row in preflight for key in row}) if preflight else VARIANT_FIELDS
    write_tsv(run_root / "cem_ready/raw_mask_ref_fk_cem_ready_preflight.tsv", preflight, preflight_fields, "E144 raw_mask_ref_fk CEM-ready preflight")
    write_tsv(VARIANTS_TSV, variants, VARIANT_FIELDS, "E144 raw_mask_ref_fk full CEM variants")

    summary = {
        "source_run_root": rel(source_root),
        "run_root": rel(run_root),
        "inventory_tsv": rel(inventory_path),
        "handoff_tsv": rel(handoff_path),
        "accounting_rows": len(accounting),
        "template_audit_rows": len(template_audit),
        "variant_rows": len(variants),
        "category_counts": dict(Counter(row["object_category"] for row in accounting)),
        "phase1_bucket_counts": dict(Counter(row["phase1_bucket"] for row in accounting)),
        "template_strategy_counts": dict(Counter(row["template_creation_strategy"] for row in accounting)),
        "accounting_decision_counts": dict(Counter(row["accounting_decision"] for row in accounting)),
        "split_counts": dict(Counter(row["split"] for row in variants)),
        "preflight_ok_counts": dict(Counter(str(row.get("preflight_ok", "")) for row in preflight)),
        "variants_tsv": rel(VARIANTS_TSV),
        "cem_ready_tsv": rel(run_root / "cem_ready/raw_mask_ref_fk_cem_ready.tsv"),
    }
    write_json(run_root / "cem_ready/raw_mask_ref_fk_cem_ready_summary.json", summary)
    (run_root / "cem_ready/raw_mask_ref_fk_cem_ready_summary.md").write_text(markdown_summary(summary, variants), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
