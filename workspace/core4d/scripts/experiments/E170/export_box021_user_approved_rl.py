#!/usr/bin/env python3
"""Export E170 user-approved Box021 PRG sources with paired OmniRetarget."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[5]
RUN_ROOT = REPO / "workspace/core4d/results/E170"
E168_ROOT = REPO / "workspace/core4d/results/E168"
ANALYSIS_MANIFEST = RUN_ROOT / "s6_downstream/manifests/analysis_manifest.tsv"
METRICS = RUN_ROOT / "s6_downstream/eval/full/e170_case_metrics.tsv"
MANUAL_REVIEW = RUN_ROOT / "s6_downstream/eval/full/user_manual_review_template.tsv"
HANDOFF = E168_ROOT / "s5_handoff/rubber_hull/handoff_manifest.tsv"
RL_DIR = RUN_ROOT / "s6_downstream/rl_export"
EVIDENCE_DIR = RUN_ROOT / "s6_downstream/evidence/box021_user_approved"
S6_SCRIPTS = REPO / "workspace/core4d/scripts/data_construction_v3/stages/s6_downstream"
METHOD_ID = "E170_PRG_lowerbodyPhysics_softPenalty_candidateGate"
STAGE2B_MANIFESTS = [
    E168_ROOT / "s3_retarget/omnirt_v1/ref_fk/stage2b_manifest_omnirt_v1_ref_fk.tsv",
    E168_ROOT / "s3_retarget/omnirt_v2/ref_fk/stage2b_manifest_omnirt_v2_ref_fk.tsv",
]


def load_e168_export_helpers() -> Any:
    path = REPO / "workspace/core4d/scripts/experiments/E168/export_box021_user_approved_rl.py"
    spec = importlib.util.spec_from_file_location("e168_box021_rl_export_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load E168 export helpers: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


HELPERS = load_e168_export_helpers()
HANDOFF_FIELDS = HELPERS.HANDOFF_FIELDS
EVIDENCE_FIELDS = HELPERS.EVIDENCE_FIELDS
PARTNER_FIELDS = HELPERS.PARTNER_FIELDS
PAIRED_PARTNER_FIELDS = HELPERS.PAIRED_PARTNER_FIELDS
build_partner_rows = HELPERS.build_partner_rows
now = HELPERS.now
read_tsv = HELPERS.read_tsv
rel = HELPERS.rel
require_file = HELPERS.require_file
run = HELPERS.run
sha256 = HELPERS.sha256
write_json = HELPERS.write_json
write_tsv = HELPERS.write_tsv


SOURCE_AUDIT_FIELDS = [
    "case_id",
    "manual_use_decision",
    "manual_quality_label",
    "manual_failure_taxonomy",
    "numeric_release_pass",
    "numeric_failure_modes",
    "execution_source",
    "reused_full",
    "cem_variant",
    "retarget_variant_id",
    "target_variant_id",
    "hand_collision_variant_id",
    "source_exp_id",
    "spider_method_id",
    "scene_act",
    "scene_act_sha256",
    "trajectory",
    "trajectory_sha256",
    "contact_mask",
    "contact_mask_sha256",
    "cem_result_npz",
    "cem_result_sha256",
    "cem_video",
    "cem_video_sha256",
    "config_act",
    "config_act_sha256",
    "metrics_ref",
    "manual_review_ref",
]


def unique_by_case(rows: list[dict[str, str]], label: str) -> dict[str, dict[str, str]]:
    output = {row["case_id"]: row for row in rows if row.get("case_id")}
    if len(output) != len([row for row in rows if row.get("case_id")]):
        raise SystemExit(f"duplicate case_id in {label}")
    return output


def source_notes(review: dict[str, str], metric: dict[str, str], analysis: dict[str, str]) -> str:
    failures = metric.get("numeric_failure_modes", "")
    return (
        f"E170 final user review={review['manual_use_decision']}/{review['manual_quality_label']}; "
        f"numeric_pass={metric.get('numeric_release_pass', '')}; "
        f"numeric_warnings={failures or 'none'}; "
        f"execution_source={analysis.get('execution_source', '')}; "
        f"gate_health_pass={metric.get('leg_gate_health_pass', '')}; "
        "manual operational acceptance does not overwrite strict numeric or gate-health facts"
    )


def validate_authority(
    manual_rows: list[dict[str, str]],
    metric_by_case: dict[str, dict[str, str]],
    analysis_by_case: dict[str, dict[str, str]],
) -> tuple[set[str], set[str]]:
    if len(manual_rows) != 28 or len({row["case_id"] for row in manual_rows}) != 28:
        raise SystemExit("manual review must contain 28 unique Box021 cases")
    invalid = [
        row["case_id"]
        for row in manual_rows
        if row.get("user_manual_review_status") != "reviewed"
        or row.get("manual_use_decision") not in {"USE", "DO_NOT_USE"}
    ]
    if invalid:
        raise SystemExit(f"manual review is not final for: {invalid}")
    approved = {row["case_id"] for row in manual_rows if row["manual_use_decision"] == "USE"}
    rejected = {row["case_id"] for row in manual_rows if row["manual_use_decision"] == "DO_NOT_USE"}
    if len(approved) != 18 or len(rejected) != 10:
        raise SystemExit(f"expected USE=18/DO_NOT_USE=10, got {len(approved)}/{len(rejected)}")
    authority_cases = approved | rejected
    if set(metric_by_case) != authority_cases or set(analysis_by_case) != authority_cases:
        raise SystemExit("manual/metrics/analysis case sets do not match")
    for case_id in authority_cases:
        metric = metric_by_case[case_id]
        review = next(row for row in manual_rows if row["case_id"] == case_id)
        if metric.get("manual_use_decision") != review["manual_use_decision"]:
            raise SystemExit(f"manual decision drift between authority and metrics: {case_id}")
    return approved, rejected


def main() -> int:
    manual_rows = read_tsv(MANUAL_REVIEW)
    metric_by_case = unique_by_case(read_tsv(METRICS), "E170 metrics")
    analysis_by_case = unique_by_case(read_tsv(ANALYSIS_MANIFEST), "E170 analysis manifest")
    approved, rejected = validate_authority(manual_rows, metric_by_case, analysis_by_case)
    review_by_case = unique_by_case(manual_rows, "E170 manual review")

    handoff_by_case = {
        row["case_id"]: row
        for row in read_tsv(HANDOFF)
        if row.get("object_key") == "box021"
    }
    if set(handoff_by_case) != approved | rejected:
        raise SystemExit("E168 S5 Box021 handoff set does not cover E170 authority")

    stage2b_by_case: dict[str, tuple[dict[str, str], Path]] = {}
    for manifest in STAGE2B_MANIFESTS:
        for row in read_tsv(manifest):
            if row.get("object_key") != "box021" or row.get("stage2b_status") != "pass":
                continue
            chosen = handoff_by_case.get(row["case_id"])
            if not chosen or row.get("retarget_variant_id") != chosen.get("retarget_variant_id"):
                continue
            if row["case_id"] in stage2b_by_case:
                raise SystemExit(f"duplicate chosen passing Stage2b row: {row['case_id']}")
            stage2b_by_case[row["case_id"]] = (row, manifest)
    if set(stage2b_by_case) != approved | rejected:
        missing = sorted((approved | rejected) - set(stage2b_by_case))
        raise SystemExit(f"chosen passing Stage2b set does not cover all 28 Box021 cases: {missing}")

    RL_DIR.mkdir(parents=True, exist_ok=True)
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    partner_dir = RL_DIR / "partner_omnirt"
    partner_dir.mkdir(parents=True, exist_ok=True)
    review_snapshot = RL_DIR / "box021_manual_review_snapshot.tsv"
    write_tsv(review_snapshot, manual_rows, list(manual_rows[0]))

    scoped_handoff: list[dict[str, Any]] = []
    evidence_input: list[dict[str, Any]] = []
    source_audit: list[dict[str, Any]] = []
    for case_id in sorted(approved):
        base = handoff_by_case[case_id]
        analysis = analysis_by_case[case_id]
        metric = metric_by_case[case_id]
        review = review_by_case[case_id]
        source_paths = {
            "scene_act": require_file(analysis["scene_act"], f"{case_id} E170 scene_act"),
            "trajectory": require_file(analysis["trajectory"], f"{case_id} trajectory"),
            "contact_mask": require_file(analysis["contact_mask"], f"{case_id} contact_mask"),
            "cem_result_npz": require_file(analysis["result_npz"], f"{case_id} E170 CEM result"),
            "cem_video": require_file(analysis["video"], f"{case_id} E170 video"),
            "config_act": require_file(analysis["config_act"], f"{case_id} E170 config"),
        }
        notes = source_notes(review, metric, analysis)
        scoped_handoff.append({
            **{field: base.get(field, "") for field in HANDOFF_FIELDS},
            "short_case_id": case_id,
            "source_exp_id": "E170",
            "spider_method_id": METHOD_ID,
            "source_exp": "E170",
            "source_variant": analysis["variant"],
            "source_metrics_ref": rel(METRICS),
            "target_scene": analysis["target_scene"],
            "trajectory": rel(source_paths["trajectory"]),
            "scene_act": rel(source_paths["scene_act"]),
            "contact_mask": rel(source_paths["contact_mask"]),
            "notes": notes,
        })
        evidence_input.append({
            "case_id": case_id,
            "object_key": base["object_key"],
            "object_name": base["object_name"],
            "date": base["date"],
            "seq": base["seq"],
            "person": base["person"],
            "person_idx": base["person_idx"],
            "retarget_variant_id": base["retarget_variant_id"],
            "target_variant_id": base["target_variant_id"],
            "hand_collision_variant_id": base["hand_collision_variant_id"],
            "source_exp_id": "E170",
            "spider_method_id": METHOD_ID,
            "cem_status": "pass",
            "rl_status": "not_run",
            "downstream_failure_mode": "",
            "downstream_notes": notes,
            "cem_run_id": analysis["variant"],
            "cem_result_npz": rel(source_paths["cem_result_npz"]),
            "cem_video": rel(source_paths["cem_video"]),
            "cem_metrics_ref": rel(METRICS),
        })
        source_audit.append({
            "case_id": case_id,
            "manual_use_decision": review["manual_use_decision"],
            "manual_quality_label": review["manual_quality_label"],
            "manual_failure_taxonomy": review["manual_failure_taxonomy"],
            "numeric_release_pass": metric["numeric_release_pass"],
            "numeric_failure_modes": metric["numeric_failure_modes"],
            "execution_source": analysis["execution_source"],
            "reused_full": analysis["reused_full"],
            "cem_variant": analysis["variant"],
            "retarget_variant_id": base["retarget_variant_id"],
            "target_variant_id": base["target_variant_id"],
            "hand_collision_variant_id": base["hand_collision_variant_id"],
            "source_exp_id": "E170",
            "spider_method_id": METHOD_ID,
            **{
                field: rel(path)
                for field, path in source_paths.items()
            },
            "scene_act_sha256": sha256(source_paths["scene_act"]),
            "trajectory_sha256": sha256(source_paths["trajectory"]),
            "contact_mask_sha256": sha256(source_paths["contact_mask"]),
            "cem_result_sha256": sha256(source_paths["cem_result_npz"]),
            "cem_video_sha256": sha256(source_paths["cem_video"]),
            "config_act_sha256": sha256(source_paths["config_act"]),
            "metrics_ref": rel(METRICS),
            "manual_review_ref": rel(MANUAL_REVIEW),
        })

    handoff_path = RL_DIR / "box021_user_approved_handoff_manifest.tsv"
    evidence_input_path = EVIDENCE_DIR / "downstream_evidence_input.tsv"
    source_audit_path = RL_DIR / "box021_user_approved_source_rows.tsv"
    write_tsv(handoff_path, scoped_handoff, HANDOFF_FIELDS)
    write_tsv(evidence_input_path, evidence_input, EVIDENCE_FIELDS)
    write_tsv(source_audit_path, source_audit, SOURCE_AUDIT_FIELDS)

    run([
        sys.executable,
        S6_SCRIPTS / "record_downstream_evidence.py",
        "--handoff-manifest-tsv", handoff_path,
        "--evidence-tsv", evidence_input_path,
        "--out-dir", EVIDENCE_DIR,
        "--evidence-root", RUN_ROOT,
        "--source-ref", "E170_box021_final_user_review_28case",
    ])
    evidence_manifest = EVIDENCE_DIR / "downstream_evidence_manifest.tsv"
    run([
        sys.executable,
        S6_SCRIPTS / "export_rl_inputs.py",
        "--handoff-manifest-tsv", handoff_path,
        "--cem-evidence-tsv", evidence_manifest,
        "--out-dir", RL_DIR,
        "--spider-repo", REPO,
    ])

    rl_input = RL_DIR / "rl_export_input.tsv"
    rl_rows = read_tsv(rl_input)
    rl_cases = {row["case_id"] for row in rl_rows}
    if rl_cases != approved or len(rl_rows) != 18:
        raise SystemExit("standard RL export case set is not exactly the 18 approved cases")
    if any(row["rl_export_decision"] != "RL_EXPORT_READY" for row in rl_rows):
        raise SystemExit("standard RL export is not 18/18 ready")
    if rl_cases & rejected:
        raise SystemExit("rejected case entered standard RL export")
    for row in rl_rows:
        for field in ("scene_act", "trajectory", "contact_mask", "cem_result_npz"):
            require_file(row[field], f"{row['case_id']} RL {field}")

    partner_rows = build_partner_rows(rl_rows, handoff_by_case, stage2b_by_case, rl_input, partner_dir)
    partner_manifest = partner_dir / "rl_partner_omnirt_manifest.tsv"
    write_tsv(partner_manifest, partner_rows, PARTNER_FIELDS)
    write_json(partner_dir / "rl_partner_omnirt_manifest.json", partner_rows)
    write_tsv(
        partner_dir / "cases_rl_partner_omnirt.tsv",
        [
            {
                "# enabled": "0",
                "source_case_id": row["source_case_id"],
                "partner_case_id": row["partner_case_id"],
                "generation_mode": row["generation_mode"],
            }
            for row in partner_rows
        ],
        ["# enabled", "source_case_id", "partner_case_id", "generation_mode"],
    )
    run_script = partner_dir / "run_rl_partner_omnirt.sh"
    run_script.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\necho 'All 18 partner OmniRetarget artifacts are reused; no execution required.'\n",
        encoding="utf-8",
    )
    run_script.chmod(0o755)

    partner_by_source = {row["source_case_id"]: row for row in partner_rows}
    if set(partner_by_source) != approved:
        raise SystemExit("partner source set is not exactly the 18 approved cases")
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
    paired_path = RL_DIR / "paired_rl_export_input.tsv"
    write_tsv(paired_path, paired_rows, paired_fields)
    write_json(RL_DIR / "paired_rl_export_input.json", paired_rows)

    partner_summary = {
        "stage": "S6_rl_partner_omnirt",
        "created_at": now(),
        "schema_version": "core4d_data_construction_v3.0",
        "rows": len(partner_rows),
        "runnable_partner_cases": 0,
        "executed": False,
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
        "# S6 RL partner OmniRetarget summary",
        "",
        f"- rows: `{len(partner_rows)}`",
        "- status: `18/18 pass`",
        "- generation: `reuse_e168_stage2b_partner`",
        "",
        "| source | partner | variant | status | trimmed |",
        "|---|---|---|---|---|",
    ]
    for row in partner_rows:
        partner_lines.append(
            f"| `{row['source_case_id']}` | `{row['partner_case_id']}` | `{row['retarget_variant_id']}` | "
            f"`{row['partner_status']}` | `{row['trimmed_npz']}` |"
        )
    (partner_dir / "rl_partner_omnirt_summary.md").write_text(
        "\n".join(partner_lines) + "\n", encoding="utf-8"
    )

    audit = {
        "experiment": "E170",
        "created_at": now(),
        "status": "pass",
        "authority_rows": 28,
        "approved_source_rows": len(approved),
        "rejected_source_rows": len(rejected),
        "source_set_matches_approved": rl_cases == approved,
        "rejected_rows_in_export": sorted(rl_cases & rejected),
        "rl_ready_rows": sum(row["rl_export_decision"] == "RL_EXPORT_READY" for row in rl_rows),
        "partner_rows": len(partner_rows),
        "partner_pass_rows": sum(row["partner_status"] == "pass" for row in partner_rows),
        "pair_complete_rows": sum(row["pair_status"] == "PAIR_COMPLETE" for row in partner_rows),
        "paired_ready_rows": sum(row["paired_rl_export_decision"] == "RL_EXPORT_READY" for row in paired_rows),
        "source_execution_counts": dict(Counter(row["execution_source"] for row in source_audit)),
        "source_numeric_release_counts": dict(Counter(row["numeric_release_pass"] for row in source_audit)),
        "partner_retarget_variant_counts": dict(Counter(row["retarget_variant_id"] for row in partner_rows)),
        "source_method_id": METHOD_ID,
        "authority_sha256": sha256(MANUAL_REVIEW),
        "metrics_sha256": sha256(METRICS),
        "analysis_manifest_sha256": sha256(ANALYSIS_MANIFEST),
        "source_audit_sha256": sha256(source_audit_path),
        "rl_export_input_sha256": sha256(rl_input),
        "partner_manifest_sha256": sha256(partner_manifest),
        "paired_rl_export_input_sha256": sha256(paired_path),
    }
    write_json(RL_DIR / "box021_paired_rl_export_audit.json", audit)

    summary = {
        "experiment": "E170",
        "created_at": now(),
        "scope": "Box021 final user-approved E170 PRG source-person cases",
        "manual_review_rows": 28,
        "manual_use": 18,
        "manual_do_not_use": 10,
        "strict_release_usable": sum(row["numeric_release_pass"] == "true" for row in source_audit),
        "source_rows": len(rl_rows),
        "rl_export_decision_counts": dict(Counter(row["rl_export_decision"] for row in rl_rows)),
        "partner_rows": len(partner_rows),
        "partner_status_counts": dict(Counter(row["partner_status"] for row in partner_rows)),
        "pair_status_counts": dict(Counter(row["pair_status"] for row in partner_rows)),
        "partner_retarget_variant_counts": dict(Counter(row["retarget_variant_id"] for row in partner_rows)),
        "paired_rl_export_decision_counts": dict(Counter(row["paired_rl_export_decision"] for row in paired_rows)),
        "machine_strict_recommendation": "FAIL",
        "result_root": rel(RL_DIR),
        "audit": rel(RL_DIR / "box021_paired_rl_export_audit.json"),
    }
    write_json(RL_DIR / "box021_paired_rl_export_summary.json", summary)
    summary_lines = [
        "# E170 Box021 paired RL export",
        "",
        "- final manual review: `28/28`",
        "- approved operational sources: `18`",
        "- rejected sources: `10`",
        "- strict numeric release among approved: `12/18`",
        "- standard RL export: `18/18 RL_EXPORT_READY`",
        "- partner OmniRetarget: `18/18 pass` (E168 Stage2b reuse)",
        "- paired export: `18/18 PAIR_COMPLETE + RL_EXPORT_READY`",
        "- machine strict recommendation remains: `FAIL`",
        "",
        "| source | execution | numeric | partner | partner variant | pair |",
        "|---|---|---|---|---|---|",
    ]
    audit_by_case = {row["case_id"]: row for row in source_audit}
    for row in paired_rows:
        source = audit_by_case[row["case_id"]]
        summary_lines.append(
            f"| `{row['case_id']}` | `{source['execution_source']}` | `{source['numeric_release_pass']}` | "
            f"`{row['partner_case_id']}` | `{row['partner_retarget_variant_id']}` | `{row['pair_status']}` |"
        )
    (RL_DIR / "box021_paired_rl_export_summary.md").write_text(
        "\n".join(summary_lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
