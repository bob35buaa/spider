#!/usr/bin/env python3
"""Prepare or run S3 OmniRetarget + SPIDER preprocess by retarget variant."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import subprocess
from collections import Counter
from pathlib import Path
import sys

SCRIPT_ROOT = next(parent for parent in Path(__file__).resolve().parents if parent.name == "data_construction_v3")
for _path in (SCRIPT_ROOT / "lib", SCRIPT_ROOT / "state", SCRIPT_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
from typing import Any

from common import (
    SCHEMA_VERSION,
    find_spider_repo,
    json_dumps,
    read_tsv,
    resolve_holosoma_repo,
    timestamp,
    write_json,
    write_tsv,
)


PERSON_SHORT = {"person1": "p1", "person2": "p2"}


def safe_id(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", text).strip("_")


def read_variant(registry: Path, variant_id: str) -> dict[str, str]:
    rows = read_tsv(registry)
    for row in rows:
        if row.get("retarget_variant_id") == variant_id:
            return row
    raise KeyError(f"retarget variant not found: {variant_id}")


def variant_params(variant: dict[str, str]) -> dict[str, Any]:
    raw = variant.get("params_json", "") or "{}"
    return json.loads(raw)


def read_inventory_index(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None or not path.is_file():
        return {}
    return {row["case_id"]: row for row in read_tsv(path) if row.get("case_id")}


def read_template_index(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None or not path.is_file():
        return {}
    return {row["source_scene_task"]: row for row in read_tsv(path) if row.get("source_scene_task")}


def apply_template_reviews(
    template_index: dict[str, dict[str, str]],
    paths: list[Path] | None,
) -> dict[str, dict[str, str]]:
    out = {key: value.copy() for key, value in template_index.items()}
    for path in paths or []:
        if not path.is_file():
            continue
        for review in read_tsv(path):
            task = review.get("source_scene_task", "")
            if not task or review.get("review_decision") != "approve_clean":
                continue
            row = out.get(task, {}).copy()
            row.update(
                {
                    "source_scene_task": task,
                    "object_key": review.get("object_key", row.get("object_key", "")),
                    "object_category": review.get("object_category", row.get("object_category", "")),
                    "person": review.get("person", row.get("person", "")),
                    "scene_xml": review.get("proxy_scene_xml", row.get("scene_xml", "")),
                    "template_status": "clean_reviewed",
                    "recommended_action": "review_approved_clean_template",
                    "template_adapter": row.get("template_adapter", "nonbox_review_override"),
                    "proxy_template": row.get("proxy_template", "True"),
                    "collision_policy": review.get("approved_collision_policy", row.get("collision_policy", "")),
                    "notes": "review_approved;reviewer="
                    + review.get("reviewer", "")
                    + (";" + review.get("review_notes", "") if review.get("review_notes") else ""),
                }
            )
            out[task] = row
    return out


def read_route_diagnostic_index(paths: list[Path] | None) -> dict[tuple[str, str], dict[str, str]]:
    out: dict[tuple[str, str], dict[str, str]] = {}
    for path in paths or []:
        if not path.is_file():
            continue
        for row in read_tsv(path):
            case_id = row.get("case_id", "")
            if not case_id:
                continue
            target_variant = row.get("target_variant_id", "")
            out[(case_id, target_variant)] = row
            if not target_variant:
                out[(case_id, "")] = row
    return out


def template_task(row: dict[str, str]) -> str:
    return f"{row.get('object_key', '').lower()}_{row.get('person', '')}"


def target_route_contract(target_variant_id: str) -> str:
    if target_variant_id == "fingertip_aware":
        return "E098_global+E099_E100_E101_fingertip_aware"
    return "E098_global"


def holosoma_task_name(row: dict[str, str]) -> str:
    return f"{row.get('date', '')}-{row.get('seq', '')}-{row.get('person', '')}-{row.get('object_name', '')}_with_obj"


def route_status_pass(value: str) -> bool:
    return str(value).strip().lower() in {"pass", "ok", "true", "1"}


def fingertip_route_decision(route_diag: dict[str, str] | None) -> tuple[str, str]:
    if route_diag is None:
        return "stage2b_route_diagnostics_missing", "fingertip-aware route requires E099-E101 diagnostics before Stage2b"
    route_status = route_diag.get("route_diagnostic_status") or route_diag.get("route_gate_status")
    if route_status and not route_status_pass(route_status):
        return "stage2b_route_diagnostics_not_pass", f"route diagnostic status is {route_status}"
    required = [
        "fingertip_vote_status",
        "palm_vote_status",
        "quat_audit_status",
        "target_active_mask_status",
        "e101_route_evidence_status",
    ]
    missing = [field for field in required if not route_diag.get(field)]
    if missing:
        return "stage2b_route_diagnostics_missing", "missing fingertip-aware diagnostic fields: " + ",".join(missing)
    failed = [field for field in required if not route_status_pass(route_diag.get(field, ""))]
    if failed:
        return "stage2b_route_diagnostics_not_pass", "non-pass fingertip-aware diagnostic fields: " + ",".join(failed)
    return "stage2b_route_diagnostics_pass", "fingertip-aware route diagnostics pass"


def stage2b_decision(
    row: dict[str, str],
    template: dict[str, str] | None,
    variant: dict[str, str],
    target_variant_id: str,
    route_diag: dict[str, str] | None,
) -> tuple[str, str, int]:
    if row.get("raw_contact_decision") != "raw_contact_pass":
        return "stage2b_hold_raw_contact_not_pass", "raw contact is not pass", 0
    if template is None:
        return "stage2b_template_unknown", "template backlog/audit row missing", 0
    if template.get("template_status") not in {"clean", "clean_reviewed"}:
        return f"stage2b_template_{template.get('template_status', 'not_clean')}", "source template is not clean", 0
    if target_variant_id == "fingertip_aware":
        route_decision, route_notes = fingertip_route_decision(route_diag)
        if route_decision != "stage2b_route_diagnostics_pass":
            return route_decision, route_notes, 0
    params = variant_params(variant)
    if params.get("requires_solver_checkout"):
        return (
            "stage2b_variant_requires_solver_checkout",
            "variant is pinned to a historical solver commit and needs a dedicated checkout/adapter before execution",
            0,
        )
    if variant.get("solver_repo_path") == "REQUIRES_CONFIRMATION":
        return "stage2b_variant_unresolved", "variant exact repo/path/commit requires confirmation", 0
    if variant.get("solver_repo_path") == "REQUIRES_ORIGINAL_CHECKOUT":
        return "stage2b_variant_unavailable", "original solver git ref is not available in this Holosoma checkout", 0
    return "stage2b_ready", "ready for explicit execute", 1


def rel_to_repo(path: Path, repo: Path) -> str:
    return os.path.relpath(path.resolve(), repo.resolve())


def build_rows(
    *,
    raw_rows: list[dict[str, str]],
    inventory_index: dict[str, dict[str, str]],
    template_index: dict[str, dict[str, str]],
    route_diag_index: dict[tuple[str, str], dict[str, str]],
    variant: dict[str, str],
    target_variant_id: str,
    out_dir: Path,
    spider_repo: Path,
    holosoma_repo: Path,
    core4d_raw_root: Path,
    smplx_model_dir: str,
    python_bin: str,
    retarget_python_bin: str,
    force: bool,
    spider_dataset: str = "core4d",
    spider_source_dataset: str | None = None,
) -> tuple[list[dict[str, Any]], list[list[str]]]:
    # spider_dataset: SPIDER dataset_name for TARGET outputs (default core4d).
    # spider_source_dataset: dataset holding the source scene templates
    # (defaults to spider_dataset). E203 uses target=core4d_v2, source=core4d
    # so v2 outputs live in a parallel tree while reusing v1 templates/assets.
    if spider_source_dataset is None:
        spider_source_dataset = spider_dataset
    params = variant_params(variant)
    variant_id = variant["retarget_variant_id"]
    run_slug = safe_id(f"{variant_id}_{target_variant_id}")
    result_root = out_dir / "results" / run_slug
    result_root_rel = rel_to_repo(result_root, spider_repo)
    case_file_rel = rel_to_repo(out_dir / f"cases_stage2b_ready_{run_slug}.tsv", spider_repo)
    replace_wrist = "1" if bool(params.get("replace_wrist_with_fingertip", False)) else "0"
    enable_constraint_relaxation = "1" if bool(params.get("enable_constraint_relaxation", False)) else "0"
    enable_foot_z_constraint = "1" if bool(params.get("enable_foot_z_constraint", False)) else "0"
    foot_slide_penalty_weight = str(float(params.get("foot_slide_penalty_weight", 0.0)))
    enable_contact_preservation = "1" if bool(params.get("enable_contact_preservation", False)) else "0"
    object_penetration_tolerance_scale = str(
        float(params.get("object_penetration_tolerance_scale", 1.0))
    )
    command = [
        "env",
        f"REPO={spider_repo}",
        f"SPIDER_DATASET={spider_dataset}",
        f"SPIDER_SOURCE_DATASET={spider_source_dataset}",
        f"KEEP_GOING={os.environ.get('KEEP_GOING', '0')}",
        f"HOLOSOMA_DIR={holosoma_repo}",
        f"CORE4D_REAL_ROOT={core4d_raw_root}",
        f"SMPLX_MODEL_DIR={smplx_model_dir}",
        f"RESULT_ROOT={result_root_rel}",
        f"PYTHON_BIN={python_bin}",
        f"RETARGET_PYTHON_BIN={retarget_python_bin}",
        f"REPLACE_WRIST_WITH_FINGERTIP={replace_wrist}",
        f"RETARGET_ENABLE_CONSTRAINT_RELAXATION={enable_constraint_relaxation}",
        f"RETARGET_ENABLE_FOOT_Z_CONSTRAINT={enable_foot_z_constraint}",
        f"RETARGET_FOOT_SLIDE_PENALTY_WEIGHT={foot_slide_penalty_weight}",
        f"RETARGET_ENABLE_CONTACT_PRESERVATION={enable_contact_preservation}",
        f"RETARGET_OBJECT_PENETRATION_TOLERANCE_SCALE={object_penetration_tolerance_scale}",
        f"TARGET_VARIANT_ID={target_variant_id}",
        "bash",
        "workspace/core4d/data_preprocess/pipeline.sh",
        "--case-file",
        case_file_rel,
    ]
    if force:
        command.append("--force")

    rows: list[dict[str, Any]] = []
    pipeline_cases: list[list[str]] = []
    for row in raw_rows:
        inv = inventory_index.get(row.get("case_id", ""), {})
        object_key = row.get("object_key", "").lower()
        person = row.get("person", "")
        pshort = PERSON_SHORT.get(person, person)
        source_scene_task = f"{object_key}_{person}"
        template = template_index.get(source_scene_task)
        route_diag = route_diag_index.get((row.get("case_id", ""), target_variant_id)) or route_diag_index.get((row.get("case_id", ""), ""))
        decision, notes, enabled = stage2b_decision(row, template, variant, target_variant_id, route_diag)
        route_diagnostic_status = route_diag.get("route_diagnostic_status", "") if route_diag else ""
        if target_variant_id == "fingertip_aware" and decision == "stage2b_ready" and not route_diagnostic_status:
            route_diagnostic_status = "pass"
        elif target_variant_id == "fingertip_aware" and route_diag is None:
            route_diagnostic_status = "missing"
        object_model_rel = row.get("object_model_rel") or inv.get("object_mesh_rel", "")
        target_task = f"dcv3_{run_slug}_{object_key}_{row.get('date')}_{row.get('seq')}_{pshort}"
        mask_slug = target_task
        case_root = result_root / f"holosoma_{target_task}"
        holosoma_task = holosoma_task_name(row)
        converted_npz = case_root / "converted" / f"{holosoma_task}.npz"
        omniretarget_output_npz = case_root / "retargeted" / f"{holosoma_task}_original.npz"
        trimmed_npz = case_root / "trimmed" / f"{holosoma_task}_original.npz"
        spider_task_dir = spider_repo / f"example_datasets/processed/{spider_dataset}/unitree_g1/humanoid_object" / target_task
        spider_trajectory = spider_task_dir / "0/trajectory_kinematic.npz"
        stage2b_contact_mask_npz_expected = result_root / "contact_masks" / mask_slug / "raw_contact_mask_3cm.npz"
        verify_summary = result_root / f"{target_task}_verify_summary.json"
        contact_label = row.get("contact_label") or row.get("contact_threshold_label", "")
        contact_target_npz = route_diag.get("target_npz", "") if route_diag else ""
        if target_variant_id == "ref_fk":
            contact_target_source = "ref_fk"
            contact_target_frame = "spider_fk"
            contact_target_time_axis = "trimmed"
            contact_target_status = "ref_fk_target"
        elif contact_target_npz:
            contact_target_source = target_variant_id
            contact_target_frame = "object_local"
            contact_target_time_axis = "trimmed"
            contact_target_status = route_diag.get("target_active_mask_status", "") if route_diag else "external_target"
        else:
            contact_target_source = target_variant_id
            contact_target_frame = ""
            contact_target_time_axis = ""
            contact_target_status = "missing_external_target"
        manifest_row = {
            "stage": "S3_stage2b",
            "case_id": row.get("case_id", ""),
            "retarget_variant_id": variant_id,
            "target_variant_id": target_variant_id,
            "pipeline_enabled": enabled,
            "stage2b_decision": decision,
            "stage2b_status": "not_run",
            "failure_mode": "" if enabled else decision,
            "decision_notes": notes,
            "sequence": row.get("sequence", ""),
            "date": row.get("date", ""),
            "seq": row.get("seq", ""),
            "person": person,
            "person_idx": row.get("person_idx", ""),
            "object_name": row.get("object_name", ""),
            "object_key": object_key,
            "object_category": row.get("object_category", ""),
            "object_model_rel": object_model_rel,
            "source_scene_task": source_scene_task,
            "source_template_status": template.get("template_status", "") if template else "",
            "source_scene_xml": template.get("scene_xml", "") if template else "",
            "route_diagnostic_status": route_diagnostic_status,
            "route_diagnostic_ref": route_diag.get("route_diagnostic_ref", "") if route_diag else "",
            "contact_mask_label": contact_label,
            "contact_mask_person_idx": row.get("contact_person_idx", row.get("person_idx", "")),
            "contact_mask_npz": "",
            "contact_mask_status": "missing_trimmed_mask",
            "contact_mask_time_axis": "",
            "stage2b_contact_mask_npz_expected": str(stage2b_contact_mask_npz_expected),
            "raw_contact_artifact_npz": row.get("contact_mask_npz", ""),
            "raw_contact_time_axis": "raw_sequence",
            "raw_to_trimmed_mapping_status": row.get("raw_to_trimmed_mapping_status", ""),
            "left_active_frac": row.get("left_active_frac", row.get("target_left_active_frac", "")),
            "right_active_frac": row.get("right_active_frac", row.get("target_right_active_frac", "")),
            "both_active_frac": row.get("both_active_frac", row.get("target_both_active_frac", "")),
            "left_longest_run_frac": row.get("left_longest_run_frac", ""),
            "right_longest_run_frac": row.get("right_longest_run_frac", ""),
            "both_longest_run_frac": row.get("both_longest_run_frac", row.get("target_both_longest_run_active_frac", "")),
            "contact_target_status": contact_target_status,
            "contact_target_npz": contact_target_npz,
            "contact_target_source": contact_target_source,
            "contact_target_frame": contact_target_frame,
            "contact_target_time_axis": contact_target_time_axis,
            "contact_route_diagnostic_ref": route_diag.get("route_diagnostic_ref", "") if route_diag else "",
            "fingertip_vote_status": route_diag.get("fingertip_vote_status", "") if route_diag else "",
            "palm_vote_status": route_diag.get("palm_vote_status", "") if route_diag else "",
            "quat_audit_status": route_diag.get("quat_audit_status", "") if route_diag else "",
            "face_changed_l": route_diag.get("face_changed_l", route_diag.get("face_changed_L", "")) if route_diag else "",
            "face_changed_r": route_diag.get("face_changed_r", route_diag.get("face_changed_R", "")) if route_diag else "",
            "disable_world_up": route_diag.get("disable_world_up", "") if route_diag else "",
            "target_active_mask_status": route_diag.get("target_active_mask_status", "") if route_diag else "",
            "target_npz": route_diag.get("target_npz", "") if route_diag else "",
            "target_npz_sha256": route_diag.get("target_npz_sha256", "") if route_diag else "",
            "target_gap_status": route_diag.get("target_gap_status", "") if route_diag else "",
            "active_L_frac": route_diag.get("active_L_frac", "") if route_diag else "",
            "active_R_frac": route_diag.get("active_R_frac", "") if route_diag else "",
            "e101_route_evidence_status": route_diag.get("e101_route_evidence_status", "") if route_diag else "",
            "target_task": target_task,
            "mask_slug": mask_slug,
            "trim_start": "auto",
            "trim_frames": "auto",
            "data_id": "0",
            "raw_contact_threshold_label": row.get("contact_threshold_label", ""),
            "raw_contact_score": row.get("raw_contact_score", ""),
            "holosoma_case_root": str(case_root),
            "converted_npz": str(converted_npz),
            "omniretarget_output_npz": str(omniretarget_output_npz),
            "retargeted_npz": str(omniretarget_output_npz),
            "trimmed_npz": str(trimmed_npz),
            "spider_task_dir": str(spider_task_dir),
            "spider_trajectory": str(spider_trajectory),
            "verify_summary": str(verify_summary),
            "solver_family": variant.get("solver_family", ""),
            "solver_version": variant.get("solver_version", ""),
            "solver_repo_path": variant.get("solver_repo_path", ""),
            "solver_git_sha": variant.get("solver_git_sha", ""),
            "solver_dirty": variant.get("solver_dirty", ""),
            "converter_script": variant.get("converter_script", ""),
            "converter_git_sha": variant.get("converter_git_sha", ""),
            "replace_wrist_with_fingertip": replace_wrist,
            "python_bin": python_bin,
            "retarget_python_bin": retarget_python_bin,
            "params_json": variant.get("params_json", ""),
            "result_root": str(result_root),
            "command_line": " ".join(shlex.quote(x) for x in command),
            "schema_version": SCHEMA_VERSION,
            "updated_at": timestamp(),
        }
        rows.append(manifest_row)
        if enabled:
            pipeline_cases.append(
                [
                    "1",
                    row.get("date", ""),
                    row.get("seq", ""),
                    person,
                    row.get("object_name", ""),
                    object_model_rel,
                    source_scene_task,
                    target_task,
                    "auto",
                    "auto",
                    "0",
                    mask_slug,
                ]
            )
    rows.sort(key=lambda r: (-int(r["pipeline_enabled"]), r["stage2b_decision"], r["case_id"]))
    return rows, pipeline_cases


def write_pipeline_cases(path: Path, cases: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "# enabled",
        "date",
        "seq",
        "person",
        "object_name",
        "object_model_rel",
        "source_scene_task",
        "target_task",
        "trim_start",
        "trim_frames",
        "data_id",
        "mask_slug",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerow(header)
        writer.writerows(cases)


def write_run_script(path: Path, command: list[str], dry_run: bool) -> None:
    cmd = command + (["--dry-run"] if dry_run else [])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/usr/bin/env bash\nset -euo pipefail\ncd \"$(git rev-parse --show-toplevel)\"\n" + " ".join(shlex.quote(x) for x in cmd) + "\n", encoding="utf-8")
    path.chmod(0o755)


def refresh_executed_outputs(rows: list[dict[str, Any]]) -> int:
    required_fields = [
        "converted_npz",
        "omniretarget_output_npz",
        "trimmed_npz",
        "spider_trajectory",
        "verify_summary",
    ]
    missing_rows = 0
    for row in rows:
        if str(row.get("pipeline_enabled", "")) != "1":
            continue
        contact_mask_path = Path(str(row.get("stage2b_contact_mask_npz_expected") or row.get("contact_mask_npz", "")))
        missing = [field for field in required_fields if not Path(str(row.get(field, ""))).is_file()]
        if not contact_mask_path.is_file():
            missing.append("contact_mask_npz")
        row["updated_at"] = timestamp()
        if missing:
            missing_rows += 1
            row["stage2b_status"] = "fail"
            row["failure_mode"] = "stage2b_outputs_missing_after_execute"
            row["decision_notes"] = "missing executed outputs: " + ",".join(missing)
        else:
            row["stage2b_status"] = "pass"
            row["failure_mode"] = ""
            row["decision_notes"] = "Stage2b execute completed and expected outputs exist"
            row["contact_mask_npz"] = str(contact_mask_path)
            row["contact_mask_status"] = "trimmed_mask_available"
            row["contact_mask_time_axis"] = "trimmed_stage2b_output"
    return missing_rows


def summarize(rows: list[dict[str, Any]], variant: dict[str, str], out_dir: Path, run_script: Path, case_file: Path) -> dict[str, Any]:
    target_variant = rows[0].get("target_variant_id", "ref_fk") if rows else "ref_fk"
    return {
        "stage": "S3_stage2b",
        "created_at": timestamp(),
        "schema_version": SCHEMA_VERSION,
        "retarget_variant_id": variant["retarget_variant_id"],
        "target_variant_id": target_variant,
        "target_route_contract": target_route_contract(str(target_variant)),
        "rows": len(rows),
        "pipeline_ready": sum(int(row["pipeline_enabled"]) for row in rows),
        "decision_counts": dict(Counter(row["stage2b_decision"] for row in rows)),
        "object_counts": dict(Counter(row["object_key"] for row in rows)),
        "case_file": str(case_file),
        "run_script": str(run_script),
        "out_dir": str(out_dir),
    }


def markdown_summary(summary: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    lines = [
        "# S3 Stage2b queue summary",
        "",
        f"- retarget_variant_id: `{summary['retarget_variant_id']}`",
        f"- target_variant_id: `{summary['target_variant_id']}`",
        f"- target_route_contract: `{summary['target_route_contract']}`",
        f"- rows: `{summary['rows']}`",
        f"- pipeline_ready: `{summary['pipeline_ready']}`",
        f"- run_script: `{summary['run_script']}`",
        "",
        "## decision counts",
        "",
        "| decision | count |",
        "|---|---:|",
    ]
    for key, count in summary["decision_counts"].items():
        lines.append(f"| `{key}` | {count} |")
    lines.extend(["", "## ready cases", "", "| case | target_task | object | person |", "|---|---|---|---|"])
    for row in rows:
        if int(row["pipeline_enabled"]):
            lines.append(f"| `{row['case_id']}` | `{row['target_task']}` | `{row['object_key']}` | `{row['person']}` |")
    lines.extend(
        [
            "",
            "说明：本阶段 wrapper 可以显式调用 legacy `workspace/core4d/data_preprocess/pipeline.sh`，但必须通过 v3 manifest 记录 variant、参数、输出路径和命令。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-contact-tsv", type=Path, required=True)
    parser.add_argument("--template-backlog-tsv", type=Path, required=True)
    parser.add_argument("--template-review-tsv", type=Path, action="append", default=[])
    parser.add_argument("--retarget-variant-registry", type=Path, required=True)
    parser.add_argument("--retarget-variant-id", required=True)
    parser.add_argument("--target-variant-id", default="ref_fk")
    parser.add_argument("--route-diagnostic-tsv", type=Path, action="append", default=[])
    parser.add_argument("--inventory-tsv", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--spider-dataset", default="core4d",
                        help="SPIDER dataset_name for TARGET outputs (default core4d; E203 uses core4d_v2)")
    parser.add_argument("--spider-source-dataset", default=None,
                        help="dataset holding source scene templates (default = --spider-dataset; E203 uses core4d)")
    parser.add_argument("--spider-repo", type=Path, default=None)
    parser.add_argument("--holosoma-repo", type=Path, default=None)
    parser.add_argument("--core4d-raw-root", type=Path, default=None)
    parser.add_argument("--smplx-model-dir", default=os.environ.get("SMPLX_MODEL_DIR", ""))
    parser.add_argument("--python-bin", default=".venv/bin/python")
    parser.add_argument(
        "--retarget-python-bin",
        default=os.environ.get("RETARGET_PYTHON_BIN", ""),
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--allow-legacy-stage2b-wrapper", action="store_true")
    args = parser.parse_args()

    spider_repo = (args.spider_repo or find_spider_repo()).resolve()
    holosoma_repo = (args.holosoma_repo or resolve_holosoma_repo()).resolve()
    raw_root = args.core4d_raw_root or (Path(os.environ["CORE4D_RAW_ROOT"]) if os.environ.get("CORE4D_RAW_ROOT") else None)
    if raw_root is None:
        raise SystemExit("missing --core4d-raw-root or CORE4D_RAW_ROOT")
    if args.execute and not args.allow_legacy_stage2b_wrapper:
        raise SystemExit("--execute requires explicit --allow-legacy-stage2b-wrapper")
    if args.execute and args.target_variant_id != "ref_fk":
        raise SystemExit("--execute currently supports only target_variant_id=ref_fk; non-ref_fk target routes need a dedicated Stage2b target adapter")
    if not args.smplx_model_dir:
        raise SystemExit("missing --smplx-model-dir or SMPLX_MODEL_DIR")

    variant = read_variant(args.retarget_variant_registry, args.retarget_variant_id)
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, pipeline_cases = build_rows(
        raw_rows=read_tsv(args.raw_contact_tsv),
        inventory_index=read_inventory_index(args.inventory_tsv),
        template_index=apply_template_reviews(read_template_index(args.template_backlog_tsv), args.template_review_tsv),
        route_diag_index=read_route_diagnostic_index(args.route_diagnostic_tsv),
        variant=variant,
        target_variant_id=args.target_variant_id,
        out_dir=out_dir,
        spider_repo=spider_repo,
        holosoma_repo=holosoma_repo,
        core4d_raw_root=raw_root.expanduser().resolve(),
        smplx_model_dir=args.smplx_model_dir,
        python_bin=args.python_bin,
        retarget_python_bin=args.retarget_python_bin,
        force=args.force,
        spider_dataset=args.spider_dataset,
        spider_source_dataset=args.spider_source_dataset,
    )
    run_slug = safe_id(f"{args.retarget_variant_id}_{args.target_variant_id}")
    case_file = out_dir / f"cases_stage2b_ready_{run_slug}.tsv"
    manifest_tsv = out_dir / f"stage2b_manifest_{run_slug}.tsv"
    run_script = out_dir / f"run_stage2b_{run_slug}.sh"
    write_pipeline_cases(case_file, pipeline_cases)
    fields = list(rows[0].keys()) if rows else []
    write_tsv(manifest_tsv, rows, fields)
    write_json(out_dir / f"stage2b_manifest_{run_slug}.json", rows)

    command = shlex.split(rows[0]["command_line"]) if rows else []
    write_run_script(run_script, command, dry_run=not args.execute)
    if args.execute and pipeline_cases:
        completed = subprocess.run([str(run_script)], cwd=spider_repo, check=False)
        if completed.returncode != 0:
            return completed.returncode
        missing_rows = refresh_executed_outputs(rows)
        write_tsv(manifest_tsv, rows, fields)
        write_json(out_dir / f"stage2b_manifest_{run_slug}.json", rows)
        if missing_rows:
            summary = summarize(rows, variant, out_dir, run_script, case_file)
            write_json(out_dir / f"stage2b_summary_{run_slug}.json", summary)
            (out_dir / f"stage2b_summary_{run_slug}.md").write_text(markdown_summary(summary, rows), encoding="utf-8")
            return 1
    summary = summarize(rows, variant, out_dir, run_script, case_file)
    write_json(out_dir / f"stage2b_summary_{run_slug}.json", summary)
    (out_dir / f"stage2b_summary_{run_slug}.md").write_text(markdown_summary(summary, rows), encoding="utf-8")
    print(json_dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
