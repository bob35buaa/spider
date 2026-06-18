#!/usr/bin/env python3
"""Export E165D peak-margin rerank rows as S6 RL handoff inputs."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[5]
E163_SCRIPT_DIR = REPO / "workspace/core4d/scripts/experiments/E163"
sys.path.insert(0, str(E163_SCRIPT_DIR))

import export_narrowSurfaceBand_rl_handoff as e163_export  # noqa: E402


RESULT_ROOT = REPO / "workspace/core4d/results/E165/peak_margin_rerank/rl_export"
E165_VARIANTS = REPO / "workspace/core4d/scripts/experiments/E165/variants.tsv"
E165_METRICS = REPO / "workspace/core4d/results/E165/peak_margin_rerank/eval/full/e165d_method_metrics.tsv"
E165_METRICS_REF = "workspace/core4d/results/E165/peak_margin_rerank/eval/full/e165d_method_metrics.tsv"
S6_ROOT = REPO / "workspace/core4d/scripts/data_construction_v3/stages/s6_downstream"
RAW_ROOT = Path("/mnt/a0ccc676-9496-49f8-a861-f8a1797dec52/mocap_data/CORE4D/CORE4D_Real")
SMPLX_DIR = Path("/mnt/a0ccc676-9496-49f8-a861-f8a1797dec52/mocap_data/smplx")
HOLOSOMA_REPO = Path("/home/ubuntu/Workspace/holosoma")

SOURCE_EXP_ID = "E165D"
SPIDER_METHOD_ID = "gateA_surfaceBandA2_postureRerankA_narrowSurfaceBandReleaseDecay_peakMargin025"
SOURCE_REF = "E165D_peakMargin025_three_case"
METHOD_LABEL = "E165D peakMargin025"
TARGET_SHORT_CASES = ["box023_person2", "box021_029_p2", "box004_083_p2"]

OBJECT_NAME = {
    "box021": "Box021",
    "box023": "Box023",
    "box004": "box004",
}

SOURCE_FIELDS = [
    "short_case_id",
    "case_id",
    "variant",
    "object_key",
    "object_name",
    "date",
    "seq",
    "person",
    "person_idx",
    "source_exp_id",
    "spider_method_id",
    "derived_task",
    "target_scene",
    "trajectory",
    "scene_act",
    "contact_mask",
    "cem_result_npz",
    "cem_outdir_npz",
    "cem_video",
    "success_tracked",
    "fall_flag",
    "track_pelvis_z_err_terminal_m",
    "raw_contact_in_mask",
    "inmask_contact_3mm",
    "phys_penetration_3mm",
    "hand_geom_penetration_2mm",
    "peak_margin_valid_frac",
    "peak_margin_selected_valid_frac",
    "peak_margin_fallback_used",
    "peak_margin_ee_peak_mean",
    "peak_margin_anchor_peak_mean",
    "peak_margin_violation_mean",
    "obj_err_mean_m",
]


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def rel(path: str | Path) -> str:
    if not path:
        return ""
    p = Path(path)
    try:
        return str(p.resolve().relative_to(REPO.resolve()))
    except Exception:
        return str(path)


def repo_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO / p


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    e163_export.write_tsv(path, rows, fields)


def write_json(path: Path, data: Any) -> None:
    e163_export.write_json(path, data)


def parse_identity(row: dict[str, str]) -> tuple[str, str, str, str]:
    date, seq, person, object_name = e163_export.parse_identity(row)
    if object_name in {"box021", "box023", "box004"}:
        object_name = OBJECT_NAME.get(object_name, object_name)
    return date, seq, person, object_name


def metric_value(row: dict[str, str], key: str) -> str:
    return row.get(key, "")


def load_sources() -> list[dict[str, str]]:
    variants = {
        row["short_case_id"]: row
        for row in read_tsv(E165_VARIANTS)
        if row.get("short_case_id") in TARGET_SHORT_CASES and row.get("method_group") == "peakMarginRerank025"
    }
    missing = [case for case in TARGET_SHORT_CASES if case not in variants]
    if missing:
        raise SystemExit(f"missing E165D variants: {missing}")

    metrics = {
        row["short_case_id"]: row
        for row in read_tsv(E165_METRICS)
        if row.get("short_case_id") in TARGET_SHORT_CASES and row.get("method") == METHOD_LABEL
    }
    missing_metrics = [case for case in TARGET_SHORT_CASES if case not in metrics]
    if missing_metrics:
        raise SystemExit(f"missing E165D metrics: {missing_metrics}")

    rows: list[dict[str, str]] = []
    for short_case in TARGET_SHORT_CASES:
        variant = variants[short_case]
        metric = metrics[short_case]
        date, seq, person, object_name = parse_identity(variant)
        rows.append(
            {
                "short_case_id": short_case,
                "case_id": variant["case_id"],
                "variant": variant["variant"],
                "object_key": variant["object_key"],
                "object_name": object_name,
                "date": date,
                "seq": seq,
                "person": person,
                "person_idx": variant["person_idx"],
                "source_exp_id": SOURCE_EXP_ID,
                "spider_method_id": SPIDER_METHOD_ID,
                "derived_task": variant["derived_task"],
                "target_scene": variant["target_scene"],
                "trajectory": variant["trajectory"],
                "scene_act": variant["rubber_scene_act"],
                "contact_mask": variant["mask_path"],
                "cem_result_npz": variant["result_npz"],
                "cem_outdir_npz": variant["outdir_npz"],
                "cem_video": variant["video"],
                "success_tracked": metric_value(metric, "success_tracked"),
                "fall_flag": metric_value(metric, "fall_flag"),
                "track_pelvis_z_err_terminal_m": metric_value(metric, "track_pelvis_z_err_terminal_m"),
                "raw_contact_in_mask": metric_value(metric, "hand_object_physics_contact_in_mask_frac"),
                "inmask_contact_3mm": metric_value(metric, "hand_object_physics_contact_3mm_in_mask_frac"),
                "phys_penetration_3mm": metric_value(metric, "hand_object_physics_penetration_3mm_frame_frac"),
                "hand_geom_penetration_2mm": metric_value(metric, "hand_geom_penetration_2mm_frac"),
                "peak_margin_valid_frac": metric_value(metric, "peak_margin_valid_frac"),
                "peak_margin_selected_valid_frac": metric_value(metric, "peak_margin_selected_valid_frac"),
                "peak_margin_fallback_used": metric_value(metric, "peak_margin_fallback_used"),
                "peak_margin_ee_peak_mean": metric_value(metric, "peak_margin_ee_peak_mean"),
                "peak_margin_anchor_peak_mean": metric_value(metric, "peak_margin_anchor_peak_mean"),
                "peak_margin_violation_mean": metric_value(metric, "peak_margin_violation_mean"),
                "obj_err_mean_m": metric_value(metric, "obj_err_mean_m"),
            }
        )
    return rows


def source_notes(row: dict[str, str]) -> str:
    return (
        "E165D peakMargin025 three-case diagnostic export; "
        f"tracked={row['success_tracked']}; fall={row['fall_flag']}; "
        f"pelvis_z_terminal={row['track_pelvis_z_err_terminal_m']}; "
        f"rawContact={row['raw_contact_in_mask']}; "
        f"inmaskC3={row['inmask_contact_3mm']}; "
        f"physPen3={row['phys_penetration_3mm']}; "
        f"geomPen2={row['hand_geom_penetration_2mm']}; "
        f"pmValid={row['peak_margin_valid_frac']}; "
        f"pmSelectedValid={row['peak_margin_selected_valid_frac']}; "
        f"pmFallback={row['peak_margin_fallback_used']}; "
        f"pmEePeak={row['peak_margin_ee_peak_mean']}; "
        f"pmAnchorPeak={row['peak_margin_anchor_peak_mean']}; "
        f"pmViolation={row['peak_margin_violation_mean']}; "
        f"objErr={row['obj_err_mean_m']}"
    )


def validate_source_rows(rows: list[dict[str, str]]) -> None:
    failures: list[str] = []
    for row in rows:
        for field in ["target_scene", "trajectory", "scene_act", "contact_mask", "cem_result_npz", "cem_outdir_npz", "cem_video"]:
            path = repo_path(row[field])
            if not path.exists() or path.stat().st_size <= 0:
                failures.append(f"{row['short_case_id']}:{field}:{row[field]}")
        if row["success_tracked"] != "true":
            failures.append(f"{row['short_case_id']}:success_tracked={row['success_tracked']}")
        if row["fall_flag"] != "false":
            failures.append(f"{row['short_case_id']}:fall_flag={row['fall_flag']}")
    if failures:
        raise SystemExit("source validation failed:\n" + "\n".join(failures))


def build_handoff_rows(source_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for source in source_rows:
        rows.append(
            {
                "case_id": source["case_id"],
                "short_case_id": source["short_case_id"],
                "object_key": source["object_key"],
                "object_name": source["object_name"],
                "date": source["date"],
                "seq": source["seq"],
                "person": source["person"],
                "person_idx": source["person_idx"],
                "retarget_variant_id": "omnirt_v1",
                "target_variant_id": "ref_fk",
                "hand_collision_variant_id": "rubber_hull",
                "source_exp_id": SOURCE_EXP_ID,
                "spider_method_id": SPIDER_METHOD_ID,
                "handoff_decision": "HANDOFF_READY",
                "candidate_decision": "E165D_PEAK_MARGIN_DIAGNOSTIC_READY",
                "target_gate_status": "pass",
                "visual_qc_status": "pass",
                "target_scene": source["target_scene"],
                "trajectory": source["trajectory"],
                "scene_act": source["scene_act"],
                "contact_mask": source["contact_mask"],
                "stage2b_target_task": source["derived_task"],
                "stage2b_result_root": "workspace/core4d/results/E165/peak_margin_rerank",
                "stage2b_manifest_ref": "workspace/core4d/scripts/experiments/E165/variants.tsv",
                "raw_contact_threshold_label": "3cm",
                "source_exp": SOURCE_EXP_ID,
                "source_variant": source["variant"],
                "source_metrics_ref": E165_METRICS_REF,
                "notes": source_notes(source),
            }
        )
    return rows


def build_evidence_rows(source_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for source in source_rows:
        rows.append(
            {
                "case_id": source["case_id"],
                "object_key": source["object_key"],
                "object_name": source["object_name"],
                "date": source["date"],
                "seq": source["seq"],
                "person": source["person"],
                "person_idx": source["person_idx"],
                "retarget_variant_id": "omnirt_v1",
                "target_variant_id": "ref_fk",
                "hand_collision_variant_id": "rubber_hull",
                "source_exp_id": SOURCE_EXP_ID,
                "spider_method_id": SPIDER_METHOD_ID,
                "cem_status": "pass",
                "rl_status": "not_run",
                "downstream_failure_mode": "",
                "downstream_notes": source_notes(source),
                "cem_run_id": source["variant"],
                "cem_result_npz": source["cem_result_npz"],
                "cem_video": source["cem_video"],
                "cem_metrics_ref": E165_METRICS_REF,
            }
        )
    return rows


def run_cmd(cmd: list[Any]) -> None:
    e163_export.run_cmd(cmd)


def validate_exports(rl_export_input: Path, partner_manifest: Path, *, require_partner_pass: bool) -> None:
    rl_rows = read_tsv(rl_export_input)
    if len(rl_rows) != len(TARGET_SHORT_CASES):
        raise SystemExit(f"expected {len(TARGET_SHORT_CASES)} RL rows, got {len(rl_rows)}")
    not_ready = [row for row in rl_rows if row.get("rl_export_decision") != "RL_EXPORT_READY"]
    if not_ready:
        raise SystemExit(f"RL export has non-ready rows: {not_ready}")
    missing_required = []
    bad_versions = []
    for row in rl_rows:
        if row.get("source_exp_id") != SOURCE_EXP_ID or row.get("spider_method_id") != SPIDER_METHOD_ID:
            bad_versions.append(f"{row.get('case_id', '')}:{row.get('source_exp_id', '')}:{row.get('spider_method_id', '')}")
        for field in ["scene_act", "trajectory", "contact_mask", "cem_result_npz"]:
            path = repo_path(row[field])
            if not path.exists() or path.stat().st_size <= 0:
                missing_required.append(f"{row['case_id']}:{field}:{row[field]}")
    if bad_versions:
        raise SystemExit("bad source_exp_id/spider_method_id:\n" + "\n".join(bad_versions))
    if missing_required:
        raise SystemExit("missing RL required files:\n" + "\n".join(missing_required))

    partner_rows = read_tsv(partner_manifest)
    if len(partner_rows) != len(TARGET_SHORT_CASES):
        raise SystemExit(f"expected {len(TARGET_SHORT_CASES)} partner rows, got {len(partner_rows)}")
    if require_partner_pass:
        failed = [row for row in partner_rows if row.get("partner_status") != "pass"]
        if failed:
            raise SystemExit(f"partner OmniRetarget failed rows: {failed}")


def summarize(source_rows: list[dict[str, str]], rl_export_input: Path, partner_manifest: Path, out_path: Path) -> dict[str, Any]:
    rl_rows = read_tsv(rl_export_input)
    partner_rows = read_tsv(partner_manifest) if partner_manifest.is_file() else []
    summary = {
        "experiment": SOURCE_EXP_ID,
        "created_at": now(),
        "source": "E165D peakMargin025 three-case",
        "result_root": rel(RESULT_ROOT),
        "source_cases": [row["short_case_id"] for row in source_rows],
        "rl_export_rows": len(rl_rows),
        "rl_export_decision_counts": dict(Counter(row.get("rl_export_decision", "") for row in rl_rows)),
        "partner_rows": len(partner_rows),
        "partner_status_counts": dict(Counter(row.get("partner_status", "") for row in partner_rows)),
        "partner_cases": [
            {
                "source_case_id": row.get("source_case_id", ""),
                "partner_case_id": row.get("partner_case_id", ""),
                "partner_status": row.get("partner_status", ""),
                "failure_mode": row.get("failure_mode", ""),
                "trimmed_npz": row.get("trimmed_npz", ""),
            }
            for row in partner_rows
        ],
    }
    write_json(out_path.with_suffix(".json"), summary)
    lines = [
        "# E165D peakMargin025 RL export summary",
        "",
        f"- created_at: `{summary['created_at']}`",
        f"- result_root: `{summary['result_root']}`",
        f"- RL export rows: `{summary['rl_export_rows']}`",
        f"- RL export decisions: `{summary['rl_export_decision_counts']}`",
        f"- partner statuses: `{summary['partner_status_counts']}`",
        "",
        "## Source cases",
        "",
        "| short case | case_id | person | object | scene_act | CEM result |",
        "|---|---|---|---|---|---|",
    ]
    for row in source_rows:
        lines.append(
            f"| `{row['short_case_id']}` | `{row['case_id']}` | `{row['person']}` | `{row['object_name']}` | "
            f"`{row['scene_act']}` | `{row['cem_result_npz']}` |"
        )
    lines.extend(["", "## Partner OmniRetarget", "", "| source | partner | status | failure | trimmed_npz |", "|---|---|---|---|---|"])
    for row in partner_rows:
        lines.append(
            f"| `{row.get('source_case_id', '')}` | `{row.get('partner_case_id', '')}` | "
            f"`{row.get('partner_status', '')}` | `{row.get('failure_mode', '')}` | `{row.get('trimmed_npz', '')}` |"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    global RESULT_ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=RESULT_ROOT)
    parser.add_argument("--core4d-raw-root", type=Path, default=RAW_ROOT)
    parser.add_argument("--smplx-model-dir", type=Path, default=SMPLX_DIR)
    parser.add_argument("--holosoma-repo", type=Path, default=HOLOSOMA_REPO)
    parser.add_argument("--python-bin", default=".venv/bin/python")
    parser.add_argument("--execute-partner", action="store_true")
    parser.add_argument("--force-partner", action="store_true")
    parser.add_argument("--allow-partner-failure", action="store_true")
    args = parser.parse_args()

    RESULT_ROOT = args.out_root.expanduser().resolve()
    source_rows = load_sources()
    validate_source_rows(source_rows)

    manifest_dir = RESULT_ROOT / "manifest"
    handoff_dir = RESULT_ROOT / "s5_handoff"
    evidence_dir = RESULT_ROOT / "s6_downstream/evidence"
    rl_export_dir = RESULT_ROOT / "s6_downstream/rl_export"
    partner_dir = rl_export_dir / "partner_omnirt"
    for path in [manifest_dir, handoff_dir, evidence_dir, rl_export_dir, partner_dir]:
        path.mkdir(parents=True, exist_ok=True)

    source_path = manifest_dir / "peakMargin025_source_rows.tsv"
    handoff_path = handoff_dir / "handoff_manifest.tsv"
    evidence_input = evidence_dir / "downstream_evidence_input.tsv"
    write_tsv(source_path, source_rows, SOURCE_FIELDS)
    write_tsv(handoff_path, build_handoff_rows(source_rows), e163_export.HANDOFF_FIELDS)
    write_tsv(evidence_input, build_evidence_rows(source_rows), e163_export.EVIDENCE_FIELDS)

    run_cmd(
        [
            sys.executable,
            S6_ROOT / "record_downstream_evidence.py",
            "--handoff-manifest-tsv",
            handoff_path,
            "--evidence-tsv",
            evidence_input,
            "--out-dir",
            evidence_dir,
            "--evidence-root",
            RESULT_ROOT,
            "--source-ref",
            SOURCE_REF,
        ]
    )
    evidence_manifest = evidence_dir / "downstream_evidence_manifest.tsv"
    run_cmd(
        [
            sys.executable,
            S6_ROOT / "export_rl_inputs.py",
            "--handoff-manifest-tsv",
            handoff_path,
            "--cem-evidence-tsv",
            evidence_manifest,
            "--out-dir",
            rl_export_dir,
            "--spider-repo",
            REPO,
        ]
    )

    e163_export.run_partner_export(args, source_rows, rl_export_dir, partner_dir)

    partner_manifest = partner_dir / "rl_partner_omnirt_manifest.tsv"
    validate_exports(
        rl_export_dir / "rl_export_input.tsv",
        partner_manifest,
        require_partner_pass=args.execute_partner and not args.allow_partner_failure,
    )
    summary = summarize(
        source_rows,
        rl_export_dir / "rl_export_input.tsv",
        partner_manifest,
        RESULT_ROOT / "summary.md",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
