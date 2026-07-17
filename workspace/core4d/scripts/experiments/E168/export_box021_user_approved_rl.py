#!/usr/bin/env python3
"""Export all user-approved E168 Box021 CEM rows with paired OmniRetarget."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[5]
RUN_ROOT = REPO / "workspace/core4d/results/E168"
HANDOFF = RUN_ROOT / "s5_handoff/rubber_hull/handoff_manifest.tsv"
CEM_MANIFEST = RUN_ROOT / "s6_downstream/cem/manifests/cem_production_manifest.tsv"
METRICS = RUN_ROOT / "s6_downstream/cem/eval/box021_all28_reviewed/e168_case_metrics.tsv"
MANUAL_REVIEW = RUN_ROOT / "s6_downstream/cem/eval/manual_review/e168_user_visual_review.tsv"
RL_DIR = RUN_ROOT / "s6_downstream/rl_export"
EVIDENCE_DIR = RUN_ROOT / "s6_downstream/evidence/box021_user_approved"
S6_SCRIPTS = REPO / "workspace/core4d/scripts/data_construction_v3/stages/s6_downstream"
STAGE2B_MANIFESTS = [
    RUN_ROOT / "s3_retarget/omnirt_v1/ref_fk/stage2b_manifest_omnirt_v1_ref_fk.tsv",
    RUN_ROOT / "s3_retarget/omnirt_v2/ref_fk/stage2b_manifest_omnirt_v2_ref_fk.tsv",
]

HANDOFF_FIELDS = [
    "case_id", "short_case_id", "object_key", "object_name", "date", "seq",
    "person", "person_idx", "retarget_variant_id", "target_variant_id",
    "hand_collision_variant_id", "source_exp_id", "spider_method_id",
    "handoff_decision", "candidate_decision", "target_gate_status",
    "visual_qc_status", "target_scene", "trajectory", "scene_act", "contact_mask",
    "stage2b_target_task", "stage2b_result_root", "stage2b_manifest_ref",
    "raw_contact_threshold_label", "source_exp", "source_variant",
    "source_metrics_ref", "notes",
]

EVIDENCE_FIELDS = [
    "case_id", "object_key", "object_name", "date", "seq", "person", "person_idx",
    "retarget_variant_id", "target_variant_id", "hand_collision_variant_id",
    "source_exp_id", "spider_method_id", "cem_status", "rl_status",
    "downstream_failure_mode", "downstream_notes", "cem_run_id",
    "cem_result_npz", "cem_video", "cem_metrics_ref",
]

PARTNER_FIELDS = [
    "source_case_id", "source_rl_export_decision", "source_handoff_decision",
    "source_cem_status", "source_person", "source_person_idx", "partner_case_id",
    "partner_person", "partner_person_idx", "object_key", "object_name",
    "object_model_rel", "date", "seq", "retarget_variant_id", "target_variant_id",
    "generation_mode", "pipeline_enabled", "partner_status", "failure_mode",
    "decision_notes", "partner_target_task", "result_root", "holosoma_case_root",
    "converted_npz", "omniretarget_output_npz", "retargeted_npz", "trimmed_npz",
    "trim_window_json", "trim_start", "trim_end", "trim_frames", "untrimmed_frames",
    "trimmed_frames", "raw_window_start_frame", "raw_window_end_frame",
    "raw_window_num_frames", "case_file", "run_script", "command_line",
    "replace_wrist_with_fingertip", "source_rl_export_input", "schema_version",
    "updated_at", "pair_status", "partner_stage2b_status",
    "partner_stage2b_manifest_ref", "partner_verify_summary", "solver_family",
    "solver_version", "solver_repo_path", "solver_git_sha", "solver_dirty",
    "converter_script", "converter_git_sha", "params_json", "converted_sha256",
    "omniretarget_output_sha256", "retargeted_sha256", "trimmed_sha256",
    "trim_window_sha256",
]

PAIRED_PARTNER_FIELDS = [
    "pair_status", "partner_case_id", "partner_person", "partner_person_idx",
    "partner_retarget_variant_id", "partner_target_variant_id", "partner_status",
    "partner_generation_mode", "partner_converted_npz", "partner_omniretarget_output_npz",
    "partner_retargeted_npz", "partner_trimmed_npz", "partner_trim_window_json",
    "partner_trim_start", "partner_trim_end", "partner_trim_frames",
    "partner_stage2b_manifest_ref", "partner_verify_summary", "partner_solver_family",
    "partner_solver_version", "partner_solver_git_sha", "partner_converter_script",
    "partner_converter_git_sha", "partner_params_json", "partner_converted_sha256",
    "partner_retargeted_sha256", "partner_trimmed_sha256", "partner_trim_window_sha256",
    "paired_rl_export_decision",
]


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def repo_path(raw: str | Path) -> Path:
    path = Path(raw)
    if path.exists():
        return path.resolve()
    text = str(raw)
    for marker in ("example_datasets/", "workspace/", "logs/"):
        if marker in text:
            return (REPO / (marker + text.split(marker, 1)[1])).resolve()
    return path if path.is_absolute() else (REPO / path).resolve()


def rel(raw: str | Path) -> str:
    path = repo_path(raw)
    try:
        return str(path.relative_to(REPO.resolve()))
    except ValueError:
        return str(path)


def require_file(raw: str | Path, label: str) -> Path:
    path = repo_path(raw)
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"missing {label}: {raw}")
    return path


def sha256(raw: str | Path) -> str:
    path = require_file(raw, "sha256 input")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def partner_case_id(case_id: str) -> str:
    if case_id.endswith("_p1"):
        return case_id[:-3] + "_p2"
    if case_id.endswith("_p2"):
        return case_id[:-3] + "_p1"
    raise RuntimeError(f"cannot infer partner: {case_id}")


def trim_info(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "trim_start": payload.get("trim_start", ""),
        "trim_end": payload.get("trim_end", ""),
        "trim_frames": payload.get("trim_frames", ""),
        "untrimmed_frames": payload.get("untrimmed_frames", ""),
        "trimmed_frames": payload.get("trimmed_frames", ""),
        "raw_window_start_frame": payload.get("trim_start", ""),
        "raw_window_end_frame": payload.get("trim_end", ""),
        "raw_window_num_frames": payload.get("trim_frames", ""),
    }


def run(command: list[str]) -> None:
    subprocess.run([str(item) for item in command], cwd=REPO, check=True)


def source_notes(review: dict[str, str], metric: dict[str, str]) -> str:
    return (
        f"E168 Box021 user review={review['manual_use_decision']}/{review['manual_quality_label']}; "
        f"numeric_pass={metric.get('numeric_release_pass', '')}; "
        f"numeric_failures={metric.get('failure_modes', '')}; "
        f"z_p95={metric.get('body_z_err_p95_m', '')}; "
        f"raw_contact={metric.get('hand_object_physics_contact_in_mask_frac', '')}; "
        f"penetration={metric.get('hand_object_physics_penetration_3mm_frame_frac', '')}; "
        f"lower_body={metric.get('leg_penetration_frac', '')}"
    )


def build_partner_rows(
    rl_rows: list[dict[str, str]],
    handoff_by_case: dict[str, dict[str, str]],
    stage2b_by_case: dict[str, tuple[dict[str, str], Path]],
    rl_input: Path,
    partner_dir: Path,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for source in rl_rows:
        partner_id = partner_case_id(source["case_id"])
        partner_handoff = handoff_by_case[partner_id]
        stage2b, stage2b_manifest = stage2b_by_case[partner_id]
        if stage2b["retarget_variant_id"] != partner_handoff["retarget_variant_id"]:
            raise RuntimeError(f"partner variant mismatch: {partner_id}")
        converted = require_file(stage2b["converted_npz"], f"{partner_id} converted")
        omnirt = require_file(stage2b["omniretarget_output_npz"], f"{partner_id} OmniRetarget")
        retargeted = require_file(stage2b["retargeted_npz"], f"{partner_id} retargeted")
        trimmed = require_file(stage2b["trimmed_npz"], f"{partner_id} trimmed")
        trim_window = require_file(Path(stage2b["holosoma_case_root"]) / "trim_window.json", f"{partner_id} trim window")
        verify = require_file(stage2b["verify_summary"], f"{partner_id} verify summary")
        info = trim_info(trim_window)
        row: dict[str, Any] = {
            "source_case_id": source["case_id"],
            "source_rl_export_decision": source["rl_export_decision"],
            "source_handoff_decision": source["handoff_decision"],
            "source_cem_status": source["cem_status"],
            "source_person": source["person"],
            "source_person_idx": source["person_idx"],
            "partner_case_id": partner_id,
            "partner_person": partner_handoff["person"],
            "partner_person_idx": partner_handoff["person_idx"],
            "object_key": source["object_key"],
            "object_name": source["object_name"],
            "object_model_rel": stage2b.get("object_model_rel", ""),
            "date": source["date"],
            "seq": source["seq"],
            "retarget_variant_id": stage2b["retarget_variant_id"],
            "target_variant_id": stage2b["target_variant_id"],
            "generation_mode": "reuse_e168_stage2b_partner",
            "pipeline_enabled": "0",
            "partner_status": "pass",
            "failure_mode": "",
            "decision_notes": "reused complete E168 Stage2b OmniRetarget partner artifacts",
            "partner_target_task": stage2b["target_task"],
            "result_root": rel(stage2b["result_root"]),
            "holosoma_case_root": rel(stage2b["holosoma_case_root"]),
            "converted_npz": rel(converted),
            "omniretarget_output_npz": rel(omnirt),
            "retargeted_npz": rel(retargeted),
            "trimmed_npz": rel(trimmed),
            "trim_window_json": rel(trim_window),
            **info,
            "case_file": "",
            "run_script": str(partner_dir / "run_rl_partner_omnirt.sh"),
            "command_line": stage2b.get("execution_command", stage2b.get("command_line", "")),
            "replace_wrist_with_fingertip": stage2b.get("replace_wrist_with_fingertip", "0"),
            "source_rl_export_input": rel(rl_input),
            "schema_version": "core4d_data_construction_v3.0",
            "updated_at": now(),
            "pair_status": "PAIR_COMPLETE",
            "partner_stage2b_status": stage2b["stage2b_status"],
            "partner_stage2b_manifest_ref": rel(stage2b_manifest),
            "partner_verify_summary": rel(verify),
            "solver_family": stage2b.get("solver_family", ""),
            "solver_version": stage2b.get("solver_version", ""),
            "solver_repo_path": stage2b.get("solver_repo_path", ""),
            "solver_git_sha": stage2b.get("solver_git_sha", ""),
            "solver_dirty": stage2b.get("solver_dirty", ""),
            "converter_script": stage2b.get("converter_script", ""),
            "converter_git_sha": stage2b.get("converter_git_sha", ""),
            "params_json": stage2b.get("params_json", ""),
            "converted_sha256": sha256(converted),
            "omniretarget_output_sha256": sha256(omnirt),
            "retargeted_sha256": sha256(retargeted),
            "trimmed_sha256": sha256(trimmed),
            "trim_window_sha256": sha256(trim_window),
        }
        output.append(row)
    return output


def main() -> int:
    manual_rows = read_tsv(MANUAL_REVIEW)
    if len(manual_rows) != 28 or len({row["case_id"] for row in manual_rows}) != 28:
        raise SystemExit("manual review must contain 28 unique Box021 cases")
    approved = {row["case_id"] for row in manual_rows if row["manual_use_decision"] == "USE"}
    rejected = {row["case_id"] for row in manual_rows if row["manual_use_decision"] == "DO_NOT_USE"}
    if len(approved) != 13 or len(rejected) != 15:
        raise SystemExit(f"expected USE=13/DO_NOT_USE=15, got {len(approved)}/{len(rejected)}")

    handoff_rows = read_tsv(HANDOFF)
    cem_rows = read_tsv(CEM_MANIFEST)
    metric_rows = read_tsv(METRICS)
    handoff_by_case = {row["case_id"]: row for row in handoff_rows if row.get("object_key") == "box021"}
    cem_by_case = {row["case_id"]: row for row in cem_rows if row.get("object_key") == "box021"}
    metric_by_case = {row["case_id"]: row for row in metric_rows}
    review_by_case = {row["case_id"]: row for row in manual_rows}
    if set(handoff_by_case) != set(review_by_case) or set(cem_by_case) != set(review_by_case):
        raise SystemExit("Box021 handoff/CEM/manual case sets do not match")

    stage2b_by_case: dict[str, tuple[dict[str, str], Path]] = {}
    for manifest in STAGE2B_MANIFESTS:
        for row in read_tsv(manifest):
            if row.get("object_key") == "box021" and row.get("stage2b_status") == "pass":
                if row["case_id"] in stage2b_by_case:
                    raise SystemExit(f"duplicate passing Stage2b row: {row['case_id']}")
                stage2b_by_case[row["case_id"]] = (row, manifest)
    if set(stage2b_by_case) != set(review_by_case):
        raise SystemExit("passing Stage2b set does not cover all 28 Box021 cases")

    RL_DIR.mkdir(parents=True, exist_ok=True)
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    partner_dir = RL_DIR / "partner_omnirt"
    partner_dir.mkdir(parents=True, exist_ok=True)
    review_snapshot = RL_DIR / "box021_manual_review_snapshot.tsv"
    write_tsv(review_snapshot, manual_rows, list(manual_rows[0]))

    scoped_handoff: list[dict[str, Any]] = []
    evidence_input: list[dict[str, Any]] = []
    source_snapshot: list[dict[str, Any]] = []
    for case_id in sorted(approved):
        handoff = handoff_by_case[case_id]
        cem = cem_by_case[case_id]
        metric = metric_by_case[case_id]
        review = review_by_case[case_id]
        for field in ("scene_act", "trajectory", "contact_mask"):
            require_file(handoff[field], f"{case_id} {field}")
        require_file(cem["result_npz"], f"{case_id} CEM result")
        require_file(cem["video"], f"{case_id} CEM video")
        notes = source_notes(review, metric)
        scoped_handoff.append({
            **{field: handoff.get(field, "") for field in HANDOFF_FIELDS},
            "short_case_id": case_id,
            "source_exp_id": "E168",
            "spider_method_id": "E167A_zOnlyBody",
            "source_exp": "E168",
            "source_variant": cem["variant"],
            "source_metrics_ref": rel(METRICS),
            "notes": notes,
        })
        evidence_input.append({
            "case_id": case_id,
            "object_key": handoff["object_key"],
            "object_name": handoff["object_name"],
            "date": handoff["date"],
            "seq": handoff["seq"],
            "person": handoff["person"],
            "person_idx": handoff["person_idx"],
            "retarget_variant_id": handoff["retarget_variant_id"],
            "target_variant_id": handoff["target_variant_id"],
            "hand_collision_variant_id": handoff["hand_collision_variant_id"],
            "source_exp_id": "E168",
            "spider_method_id": "E167A_zOnlyBody",
            "cem_status": "pass",
            "rl_status": "not_run",
            "downstream_failure_mode": "",
            "downstream_notes": notes,
            "cem_run_id": cem["variant"],
            "cem_result_npz": cem["result_npz"],
            "cem_video": cem["video"],
            "cem_metrics_ref": rel(METRICS),
        })
        source_snapshot.append({
            "case_id": case_id,
            "manual_use_decision": review["manual_use_decision"],
            "manual_quality_label": review["manual_quality_label"],
            "numeric_release_pass": metric["numeric_release_pass"],
            "numeric_failure_modes": metric["failure_modes"],
            "cem_variant": cem["variant"],
            "retarget_variant_id": handoff["retarget_variant_id"],
            "scene_act": handoff["scene_act"],
            "trajectory": handoff["trajectory"],
            "contact_mask": handoff["contact_mask"],
            "cem_result_npz": cem["result_npz"],
            "cem_video": cem["video"],
        })

    handoff_path = RL_DIR / "box021_user_approved_handoff_manifest.tsv"
    evidence_input_path = EVIDENCE_DIR / "downstream_evidence_input.tsv"
    write_tsv(handoff_path, scoped_handoff, HANDOFF_FIELDS)
    write_tsv(evidence_input_path, evidence_input, EVIDENCE_FIELDS)
    write_tsv(RL_DIR / "box021_user_approved_source_rows.tsv", source_snapshot, list(source_snapshot[0]))

    run([
        sys.executable, S6_SCRIPTS / "record_downstream_evidence.py",
        "--handoff-manifest-tsv", handoff_path,
        "--evidence-tsv", evidence_input_path,
        "--out-dir", EVIDENCE_DIR,
        "--evidence-root", RUN_ROOT,
        "--source-ref", "E168_box021_user_visual_review_28case",
    ])
    evidence_manifest = EVIDENCE_DIR / "downstream_evidence_manifest.tsv"
    run([
        sys.executable, S6_SCRIPTS / "export_rl_inputs.py",
        "--handoff-manifest-tsv", handoff_path,
        "--cem-evidence-tsv", evidence_manifest,
        "--out-dir", RL_DIR,
        "--spider-repo", REPO,
    ])

    rl_input = RL_DIR / "rl_export_input.tsv"
    rl_rows = read_tsv(rl_input)
    if len(rl_rows) != 13 or any(row["rl_export_decision"] != "RL_EXPORT_READY" for row in rl_rows):
        raise SystemExit("standard RL export is not 13/13 ready")
    for row in rl_rows:
        for field in ("scene_act", "trajectory", "contact_mask", "cem_result_npz"):
            require_file(row[field], f"{row['case_id']} RL {field}")

    partner_rows = build_partner_rows(rl_rows, handoff_by_case, stage2b_by_case, rl_input, partner_dir)
    partner_manifest = partner_dir / "rl_partner_omnirt_manifest.tsv"
    write_tsv(partner_manifest, partner_rows, PARTNER_FIELDS)
    write_json(partner_dir / "rl_partner_omnirt_manifest.json", partner_rows)
    write_tsv(
        partner_dir / "cases_rl_partner_omnirt.tsv",
        [{"# enabled": "0", "source_case_id": row["source_case_id"], "partner_case_id": row["partner_case_id"], "generation_mode": row["generation_mode"]} for row in partner_rows],
        ["# enabled", "source_case_id", "partner_case_id", "generation_mode"],
    )
    run_script = partner_dir / "run_rl_partner_omnirt.sh"
    run_script.write_text("#!/usr/bin/env bash\nset -euo pipefail\necho 'All 13 partner OmniRetarget artifacts are reused; no execution required.'\n", encoding="utf-8")
    run_script.chmod(0o755)

    partner_by_source = {row["source_case_id"]: row for row in partner_rows}
    paired_rows: list[dict[str, Any]] = []
    for source in rl_rows:
        partner = partner_by_source[source["case_id"]]
        paired_rows.append({
            **source,
            "pair_status": partner["pair_status"],
            "partner_case_id": partner["partner_case_id"],
            "partner_person": partner["partner_person"],
            "partner_person_idx": partner["partner_person_idx"],
            "partner_retarget_variant_id": partner["retarget_variant_id"],
            "partner_target_variant_id": partner["target_variant_id"],
            "partner_status": partner["partner_status"],
            "partner_generation_mode": partner["generation_mode"],
            "partner_converted_npz": partner["converted_npz"],
            "partner_omniretarget_output_npz": partner["omniretarget_output_npz"],
            "partner_retargeted_npz": partner["retargeted_npz"],
            "partner_trimmed_npz": partner["trimmed_npz"],
            "partner_trim_window_json": partner["trim_window_json"],
            "partner_trim_start": partner["trim_start"],
            "partner_trim_end": partner["trim_end"],
            "partner_trim_frames": partner["trim_frames"],
            "partner_stage2b_manifest_ref": partner["partner_stage2b_manifest_ref"],
            "partner_verify_summary": partner["partner_verify_summary"],
            "partner_solver_family": partner["solver_family"],
            "partner_solver_version": partner["solver_version"],
            "partner_solver_git_sha": partner["solver_git_sha"],
            "partner_converter_script": partner["converter_script"],
            "partner_converter_git_sha": partner["converter_git_sha"],
            "partner_params_json": partner["params_json"],
            "partner_converted_sha256": partner["converted_sha256"],
            "partner_retargeted_sha256": partner["retargeted_sha256"],
            "partner_trimmed_sha256": partner["trimmed_sha256"],
            "partner_trim_window_sha256": partner["trim_window_sha256"],
            "paired_rl_export_decision": "RL_EXPORT_READY",
        })
    paired_fields = [*rl_rows[0].keys(), *PAIRED_PARTNER_FIELDS]
    write_tsv(RL_DIR / "paired_rl_export_input.tsv", paired_rows, paired_fields)
    write_json(RL_DIR / "paired_rl_export_input.json", paired_rows)

    partner_summary = {
        "stage": "S6_rl_partner_omnirt",
        "created_at": now(),
        "schema_version": "core4d_data_construction_v3.0",
        "rows": len(partner_rows),
        "runnable_partner_cases": 0,
        "executed": "false",
        "returncode": 0,
        "partner_status_counts": dict(Counter(row["partner_status"] for row in partner_rows)),
        "pair_status_counts": dict(Counter(row["pair_status"] for row in partner_rows)),
        "retarget_variant_counts": dict(Counter(row["retarget_variant_id"] for row in partner_rows)),
        "generation_mode_counts": dict(Counter(row["generation_mode"] for row in partner_rows)),
        "source_rl_export_input": rel(rl_input),
        "manifest_tsv": rel(partner_manifest),
        "out_dir": rel(partner_dir),
    }
    write_json(partner_dir / "rl_partner_omnirt_summary.json", partner_summary)
    partner_lines = [
        "# S6 RL partner OmniRetarget summary", "",
        f"- rows: `{len(partner_rows)}`", "- status: `13/13 pass`",
        "- generation: `reuse_e168_stage2b_partner`", "",
        "| source | partner | variant | status | trimmed |", "|---|---|---|---|---|",
    ]
    for row in partner_rows:
        partner_lines.append(f"| `{row['source_case_id']}` | `{row['partner_case_id']}` | `{row['retarget_variant_id']}` | `{row['partner_status']}` | `{row['trimmed_npz']}` |")
    (partner_dir / "rl_partner_omnirt_summary.md").write_text("\n".join(partner_lines) + "\n", encoding="utf-8")

    summary = {
        "experiment": "E168",
        "created_at": now(),
        "scope": "Box021 user-approved source-person cases",
        "manual_review_rows": 28,
        "manual_use": 13,
        "manual_do_not_use": 15,
        "source_rows": 13,
        "rl_export_decision_counts": dict(Counter(row["rl_export_decision"] for row in rl_rows)),
        "partner_rows": 13,
        "partner_status_counts": dict(Counter(row["partner_status"] for row in partner_rows)),
        "pair_status_counts": dict(Counter(row["pair_status"] for row in partner_rows)),
        "partner_retarget_variant_counts": dict(Counter(row["retarget_variant_id"] for row in partner_rows)),
        "paired_rl_export_decision_counts": dict(Counter(row["paired_rl_export_decision"] for row in paired_rows)),
        "result_root": rel(RL_DIR),
    }
    write_json(RL_DIR / "box021_paired_rl_export_summary.json", summary)
    summary_lines = [
        "# E168 Box021 paired RL export", "",
        "- manual review: `28/28`", "- approved sources: `13`", "- rejected sources: `15`",
        "- standard RL export: `13/13 RL_EXPORT_READY`", "- partner OmniRetarget: `13/13 pass`",
        "- paired export: `13/13 PAIR_COMPLETE + RL_EXPORT_READY`", "",
        "| source | partner | partner variant | pair |", "|---|---|---|---|",
    ]
    for row in paired_rows:
        summary_lines.append(f"| `{row['case_id']}` | `{row['partner_case_id']}` | `{row['partner_retarget_variant_id']}` | `{row['pair_status']}` |")
    (RL_DIR / "box021_paired_rl_export_summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
