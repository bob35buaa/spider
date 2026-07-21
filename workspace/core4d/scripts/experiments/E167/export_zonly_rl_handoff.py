#!/usr/bin/env python3
"""Export E167 z-only rows as S6 RL handoff inputs."""

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


RESULT_ROOT = REPO / "workspace/core4d/results/E167/holosoma_zonly/rl_export"
VARIANTS = REPO / "workspace/core4d/scripts/experiments/E167/variants.tsv"
METRICS = REPO / "workspace/core4d/results/E167/holosoma_zonly/eval/cem_metrics/full/e167_arm_metrics.tsv"
METRICS_REF = "workspace/core4d/results/E167/holosoma_zonly/eval/cem_metrics/full/e167_arm_metrics.tsv"
S6_ROOT = REPO / "workspace/core4d/scripts/data_construction_v3/stages/s6_downstream"
RAW_ROOT = Path("/mnt/a0ccc676-9496-49f8-a861-f8a1797dec52/mocap_data/CORE4D/CORE4D_Real")
SMPLX_DIR = Path("/mnt/a0ccc676-9496-49f8-a861-f8a1797dec52/mocap_data/smplx")
HOLOSOMA_REPO = Path("/home/ubuntu/Workspace/holosoma")

SOURCE_EXP_ID = "E167"
SOURCE_REF = "E167_holosoma_zonly_7case"
TARGET_ARMS = ["E167A", "E167A_B1", "E167A_B2"]
ARM_METHOD_ID = {
    "E167A": "E167A_zOnlyBody",
    "E167A_B1": "E167A_plus_B1_zOnlyCemSmooth",
    "E167A_B2": "E167A_plus_B2_zOnlyHandoffSmooth",
}
OBJECT_NAME = {
    "box021": "Box021",
    "box023": "Box023",
    "box004": "box004",
}

SOURCE_FIELDS = [
    "synthetic_case_id",
    "original_case_id",
    "short_case_id",
    "arm",
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
    "spider_gate_pass",
    "raw_contact_in_mask",
    "inmask_contact_3mm",
    "phys_penetration_3mm",
    "hand_geom_penetration_2mm",
    "trackbody_jerk_p95",
    "ankle_acc_max",
    "obj_speed_max",
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


def bool_text(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "pass"}


def synthetic_case_id(original_case_id: str, arm: str) -> str:
    return f"{original_case_id}__E167_{arm}"


def load_sources(target_arms: list[str], variants_path: Path, metrics_path: Path) -> list[dict[str, str]]:
    variants = {
        (row["short_case_id"], row["arm"]): row
        for row in read_tsv(variants_path)
        if row.get("arm") in target_arms
    }
    metrics = {
        (row["short_case_id"], row["arm"]): row
        for row in read_tsv(metrics_path)
        if row.get("arm") in target_arms
    }
    cases = sorted({case for case, _ in variants})
    missing = [
        f"{case}:{arm}"
        for case in cases
        for arm in target_arms
        if (case, arm) not in variants or (case, arm) not in metrics
    ]
    if missing:
        raise SystemExit(f"missing E167 rows: {missing}")

    rows: list[dict[str, str]] = []
    for case in cases:
        for arm in target_arms:
            variant = variants[(case, arm)]
            metric = metrics[(case, arm)]
            date, seq, person, object_name = e163_export.parse_identity(variant)
            object_name = OBJECT_NAME.get(variant["object_key"], object_name)
            is_postprocess = variant.get("arm_kind") == "postprocess"
            cem_result = variant["postprocess_output_npz"] if is_postprocess else variant["result_npz"]
            cem_outdir = variant["postprocess_output_npz"] if is_postprocess else variant["outdir_npz"]
            case_key = synthetic_case_id(variant["case_id"], arm)
            rows.append(
                {
                    "case_id": case_key,
                    "synthetic_case_id": case_key,
                    "original_case_id": variant["case_id"],
                    "short_case_id": case,
                    "arm": arm,
                    "variant": variant["variant"],
                    "object_key": variant["object_key"],
                    "object_name": object_name,
                    "date": date,
                    "seq": seq,
                    "person": person,
                    "person_idx": variant["person_idx"],
                    "source_exp_id": SOURCE_EXP_ID,
                    "spider_method_id": ARM_METHOD_ID.get(arm, variant["method"]),
                    "derived_task": variant["derived_task"],
                    "target_scene": variant["target_scene"],
                    "trajectory": variant["trajectory"],
                    "scene_act": variant["rubber_scene_act"],
                    "contact_mask": variant["mask_path"],
                    "cem_result_npz": cem_result,
                    "cem_outdir_npz": cem_outdir,
                    "cem_video": variant["video"],
                    "success_tracked": metric.get("success_tracked", ""),
                    "fall_flag": metric.get("fall_flag", ""),
                    "spider_gate_pass": metric.get("spider_gate_pass", ""),
                    "raw_contact_in_mask": metric.get("hand_object_physics_contact_in_mask_frac", ""),
                    "inmask_contact_3mm": metric.get("hand_object_physics_contact_3mm_in_mask_frac", ""),
                    "phys_penetration_3mm": metric.get("hand_object_physics_penetration_3mm_frame_frac", ""),
                    "hand_geom_penetration_2mm": metric.get("hand_geom_penetration_2mm_frac", ""),
                    "trackbody_jerk_p95": metric.get("trackbody_jerk_p95", ""),
                    "ankle_acc_max": metric.get("ankle_acc_max", ""),
                    "obj_speed_max": metric.get("obj_speed_max", ""),
                }
            )
    return rows


def source_notes(row: dict[str, str]) -> str:
    return (
        f"E167 downstream export source_ref={SOURCE_REF}; "
        f"original_case={row['original_case_id']}; variant={row['variant']}; "
        f"tracked={row['success_tracked']}; fall={row['fall_flag']}; "
        f"spider_gate={row['spider_gate_pass']}; rawContact={row['raw_contact_in_mask']}; "
        f"inmaskC3={row['inmask_contact_3mm']}; physPen3={row['phys_penetration_3mm']}; "
        f"geomPen2={row['hand_geom_penetration_2mm']}; "
        f"trackJerk={row['trackbody_jerk_p95']}; ankleAcc={row['ankle_acc_max']}; "
        f"objV={row['obj_speed_max']}"
    )


def validate_source_rows(rows: list[dict[str, str]], *, allow_spider_gate_fail: bool) -> None:
    failures: list[str] = []
    for row in rows:
        for field in ["target_scene", "trajectory", "scene_act", "contact_mask", "cem_result_npz", "cem_outdir_npz"]:
            path = repo_path(row[field])
            if not path.exists() or path.stat().st_size <= 0:
                failures.append(f"{row['short_case_id']}:{row['arm']}:{field}:{row[field]}")
        if row["success_tracked"] != "true":
            failures.append(f"{row['short_case_id']}:{row['arm']}:success_tracked={row['success_tracked']}")
        if row["fall_flag"] != "false":
            failures.append(f"{row['short_case_id']}:{row['arm']}:fall_flag={row['fall_flag']}")
        if not allow_spider_gate_fail and not bool_text(row["spider_gate_pass"]):
            failures.append(f"{row['short_case_id']}:{row['arm']}:spider_gate_pass={row['spider_gate_pass']}")
    if failures:
        raise SystemExit("source validation failed:\n" + "\n".join(failures))


def build_handoff_rows(source_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for source in source_rows:
        rows.append(
            {
                "case_id": source["synthetic_case_id"],
                "short_case_id": f"{source['short_case_id']}__{source['arm']}",
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
                "spider_method_id": source["spider_method_id"],
                "handoff_decision": "HANDOFF_READY",
                "candidate_decision": f"E167_{source['arm']}_READY",
                "target_gate_status": "pass",
                "visual_qc_status": "pass",
                "target_scene": source["target_scene"],
                "trajectory": source["trajectory"],
                "scene_act": source["scene_act"],
                "contact_mask": source["contact_mask"],
                "stage2b_target_task": source["derived_task"],
                "stage2b_result_root": "workspace/core4d/results/E167/holosoma_zonly",
                "stage2b_manifest_ref": "workspace/core4d/scripts/experiments/E167/variants.tsv",
                "raw_contact_threshold_label": "3cm",
                "source_exp": SOURCE_EXP_ID,
                "source_variant": source["variant"],
                "source_metrics_ref": METRICS_REF,
                "notes": source_notes(source),
            }
        )
    return rows


def build_evidence_rows(source_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for source in source_rows:
        rows.append(
            {
                "case_id": source["synthetic_case_id"],
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
                "spider_method_id": source["spider_method_id"],
                "cem_status": "pass",
                "rl_status": "not_run",
                "downstream_failure_mode": "",
                "downstream_notes": source_notes(source),
                "cem_run_id": source["variant"],
                "cem_result_npz": source["cem_result_npz"],
                "cem_video": source["cem_video"],
                "cem_metrics_ref": METRICS_REF,
            }
        )
    return rows


def validate_exports(rl_export_input: Path, expected_rows: int) -> None:
    rl_rows = read_tsv(rl_export_input)
    if len(rl_rows) != expected_rows:
        raise SystemExit(f"expected {expected_rows} RL rows, got {len(rl_rows)}")
    not_ready = [row for row in rl_rows if row.get("rl_export_decision") != "RL_EXPORT_READY"]
    if not_ready:
        raise SystemExit(f"RL export has non-ready rows: {not_ready}")
    missing_required = []
    for row in rl_rows:
        for field in ["scene_act", "trajectory", "contact_mask", "cem_result_npz"]:
            path = repo_path(row[field])
            if not path.exists() or path.stat().st_size <= 0:
                missing_required.append(f"{row['case_id']}:{field}:{row[field]}")
    if missing_required:
        raise SystemExit("missing RL required files:\n" + "\n".join(missing_required))


def unique_partner_execution_rows(source_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for row in source_rows:
        key = (row["object_key"], row["date"], row["seq"], row["person"])
        if key in seen:
            continue
        seen.add(key)
        rows.append(row)
    return rows


def validate_partner_manifest(partner_manifest: Path, expected_rows: int, *, require_partner_pass: bool) -> None:
    if not partner_manifest.is_file():
        raise SystemExit(f"missing partner manifest: {partner_manifest}")
    partner_rows = read_tsv(partner_manifest)
    if len(partner_rows) != expected_rows:
        raise SystemExit(f"expected {expected_rows} partner rows, got {len(partner_rows)}")
    failed = [row for row in partner_rows if row.get("partner_status") != "pass"]
    if require_partner_pass and failed:
        raise SystemExit(f"partner OmniRetarget failed rows: {failed}")


def write_partner_summary_markdown(partner_manifest: Path, partner_dir: Path) -> None:
    partner_rows = read_tsv(partner_manifest) if partner_manifest.is_file() else []
    status_counts = dict(Counter(row.get("partner_status", "") for row in partner_rows))
    unique_cases = sorted({row.get("partner_case_id", "") for row in partner_rows if row.get("partner_case_id", "")})
    lines = [
        "# S6 RL partner OmniRetarget summary",
        "",
        f"- created_at: `{now()}`",
        f"- rows: `{len(partner_rows)}`",
        f"- unique partner cases: `{len(unique_cases)}`",
        f"- statuses: `{status_counts}`",
        f"- out_dir: `{partner_dir}`",
        "",
        "## Partner rows",
        "",
        "| source | partner | status | failure | trimmed_npz |",
        "|---|---|---|---|---|",
    ]
    for row in partner_rows:
        lines.append(
            f"| `{row.get('source_case_id', '')}` | `{row.get('partner_case_id', '')}` | "
            f"`{row.get('partner_status', '')}` | `{row.get('failure_mode', '')}` | `{row.get('trimmed_npz', '')}` |"
        )
    lines.extend(
        [
            "",
            "说明：本表只为 RL 临时 partner motion 输入服务，不回写 v3 registry、S5 handoff 或 S6 CEM evidence。",
        ]
    )
    (partner_dir / "rl_partner_omnirt_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize(
    source_rows: list[dict[str, str]],
    rl_export_input: Path,
    partner_manifest: Path,
    out_path: Path,
) -> dict[str, Any]:
    rl_rows = read_tsv(rl_export_input)
    partner_rows = read_tsv(partner_manifest) if partner_manifest.is_file() else []
    summary = {
        "experiment": SOURCE_EXP_ID,
        "created_at": now(),
        "source": SOURCE_REF,
        "result_root": rel(RESULT_ROOT),
        "source_rows": len(source_rows),
        "source_arms": sorted(set(row["arm"] for row in source_rows)),
        "source_cases": sorted(set(row["short_case_id"] for row in source_rows)),
        "rl_export_rows": len(rl_rows),
        "rl_export_decision_counts": dict(Counter(row.get("rl_export_decision", "") for row in rl_rows)),
        "partner_rows": len(partner_rows),
        "partner_unique_cases": len(set(row.get("partner_case_id", "") for row in partner_rows if row.get("partner_case_id", ""))),
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
        "# E167 z-only RL export summary",
        "",
        f"- created_at: `{summary['created_at']}`",
        f"- result_root: `{summary['result_root']}`",
        f"- source rows: `{summary['source_rows']}`",
        f"- arms: `{summary['source_arms']}`",
        f"- RL export decisions: `{summary['rl_export_decision_counts']}`",
        f"- partner statuses: `{summary['partner_status_counts']}`",
        "",
        "| synthetic case | original case | arm | CEM result |",
        "|---|---|---|---|",
    ]
    for row in source_rows:
        lines.append(f"| `{row['synthetic_case_id']}` | `{row['original_case_id']}` | `{row['arm']}` | `{row['cem_result_npz']}` |")
    lines.extend(["", "## Partner OmniRetarget", "", "| source | partner | status | failure | trimmed_npz |", "|---|---|---|---|---|"])
    for row in partner_rows:
        lines.append(
            f"| `{row.get('source_case_id', '')}` | `{row.get('partner_case_id', '')}` | "
            f"`{row.get('partner_status', '')}` | `{row.get('failure_mode', '')}` | `{row.get('trimmed_npz', '')}` |"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    global RESULT_ROOT, SOURCE_REF, METRICS_REF
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=RESULT_ROOT)
    parser.add_argument("--arms", default=",".join(TARGET_ARMS))
    parser.add_argument("--variants-tsv", type=Path, default=VARIANTS)
    parser.add_argument("--metrics-tsv", type=Path, default=METRICS)
    parser.add_argument("--metrics-ref", default=METRICS_REF)
    parser.add_argument("--source-ref", default=SOURCE_REF)
    parser.add_argument("--core4d-raw-root", type=Path, default=RAW_ROOT)
    parser.add_argument("--smplx-model-dir", type=Path, default=SMPLX_DIR)
    parser.add_argument("--holosoma-repo", type=Path, default=HOLOSOMA_REPO)
    parser.add_argument("--python-bin", default=".venv/bin/python")
    parser.add_argument("--execute-partner", action="store_true")
    parser.add_argument("--force-partner", action="store_true")
    parser.add_argument("--allow-partner-failure", action="store_true")
    parser.add_argument("--skip-partner", action="store_true")
    parser.add_argument("--allow-spider-gate-fail", action="store_true")
    args = parser.parse_args()

    RESULT_ROOT = args.out_root.expanduser().resolve()
    SOURCE_REF = args.source_ref
    METRICS_REF = args.metrics_ref
    arms = [arm.strip() for arm in args.arms.split(",") if arm.strip()]
    source_rows = load_sources(arms, args.variants_tsv, args.metrics_tsv)
    validate_source_rows(source_rows, allow_spider_gate_fail=args.allow_spider_gate_fail)

    handoff_dir = RESULT_ROOT / "s5_handoff"
    evidence_dir = RESULT_ROOT / "s6_downstream/evidence"
    rl_export_dir = RESULT_ROOT / "s6_downstream/rl_export"
    partner_dir = rl_export_dir / "partner_omnirt"
    manifest_dir = RESULT_ROOT / "manifest"
    for path in [manifest_dir, handoff_dir, evidence_dir, rl_export_dir, partner_dir]:
        path.mkdir(parents=True, exist_ok=True)

    write_tsv(manifest_dir / "zonly_source_rows.tsv", source_rows, SOURCE_FIELDS)
    handoff_path = handoff_dir / "handoff_manifest.tsv"
    evidence_input = evidence_dir / "downstream_evidence_input.tsv"
    write_tsv(handoff_path, build_handoff_rows(source_rows), e163_export.HANDOFF_FIELDS)
    write_tsv(evidence_input, build_evidence_rows(source_rows), e163_export.EVIDENCE_FIELDS)

    e163_export.run_cmd(
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
    e163_export.run_cmd(
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

    validate_exports(rl_export_dir / "rl_export_input.tsv", len(source_rows))
    partner_manifest = partner_dir / ("rl_partner_omnirt_manifest.skipped.tsv" if args.skip_partner else "rl_partner_omnirt_manifest.tsv")
    if not args.skip_partner:
        partner_args = argparse.Namespace(
            execute_partner=args.execute_partner,
            allow_partner_failure=args.allow_partner_failure,
            force_partner=args.force_partner,
            holosoma_repo=args.holosoma_repo,
            core4d_raw_root=args.core4d_raw_root,
            smplx_model_dir=args.smplx_model_dir,
            python_bin=args.python_bin,
        )
        partner_sources = unique_partner_execution_rows(source_rows) if args.allow_partner_failure else source_rows
        e163_export.run_partner_export(partner_args, partner_sources, rl_export_dir, partner_dir)
        validate_partner_manifest(
            partner_manifest,
            len(source_rows),
            require_partner_pass=args.execute_partner and not args.allow_partner_failure,
        )
        write_partner_summary_markdown(partner_manifest, partner_dir)
    summary = summarize(source_rows, rl_export_dir / "rl_export_input.tsv", partner_manifest, RESULT_ROOT / "summary.md")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
