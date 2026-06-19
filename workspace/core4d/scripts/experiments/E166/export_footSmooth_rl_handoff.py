#!/usr/bin/env python3
"""Export E166 foot/smooth rows as S6 RL handoff inputs."""

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


RESULT_ROOT = REPO / "workspace/core4d/results/E166/foot_smooth_retarget/rl_export"
E166_VARIANTS = REPO / "workspace/core4d/scripts/experiments/E166/variants.tsv"
E166_METRICS = REPO / "workspace/core4d/results/E166/foot_smooth_retarget/eval/full/e166_arm_metrics.tsv"
E166_METRICS_REF = "workspace/core4d/results/E166/foot_smooth_retarget/eval/full/e166_arm_metrics.tsv"
S6_ROOT = REPO / "workspace/core4d/scripts/data_construction_v3/stages/s6_downstream"
RAW_ROOT = Path("/mnt/a0ccc676-9496-49f8-a861-f8a1797dec52/mocap_data/CORE4D/CORE4D_Real")
SMPLX_DIR = Path("/mnt/a0ccc676-9496-49f8-a861-f8a1797dec52/mocap_data/smplx")
HOLOSOMA_REPO = Path("/home/ubuntu/Workspace/holosoma")

SOURCE_EXP_ID = "E166"
SOURCE_REF = "E166_A_A_B2_postSmooth_threecase"
TARGET_SHORT_CASES = ["box021_035_p2", "box004_082_p1", "box004_083_p2"]
TARGET_ARMS = ["A", "A_B2_postSmooth"]
ARM_METHOD_ID = {
    "A": "E166_A_footConstraints",
    "A_B2_postSmooth": "E166_A_then_B2_postSmooth",
    "AplusB": "E166_AplusB_footSmoothFull",
    "B1": "E166_B1_cemSmooth",
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
    "foot_slip_max_m",
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


def synthetic_case_id(case_id: str, arm: str) -> str:
    return f"{case_id}__E166_{arm}"


def load_sources(
    target_arms: list[str],
    target_cases: list[str] | None = None,
    variants_path: Path = E166_VARIANTS,
    metrics_path: Path = E166_METRICS,
) -> list[dict[str, str]]:
    if target_cases is None:
        target_cases = TARGET_SHORT_CASES
    variants = {
        (row["short_case_id"], row["arm"]): row
        for row in read_tsv(variants_path)
        if row.get("short_case_id") in target_cases and row.get("arm") in target_arms
    }
    metrics = {
        (row["short_case_id"], row["arm"]): row
        for row in read_tsv(metrics_path)
        if row.get("short_case_id") in target_cases and row.get("arm") in target_arms
    }
    missing = [
        f"{case}:{arm}"
        for case in target_cases
        for arm in target_arms
        if (case, arm) not in variants or (case, arm) not in metrics
    ]
    if missing:
        raise SystemExit(f"missing E166 rows: {missing}")

    rows: list[dict[str, str]] = []
    for case in target_cases:
        for arm in target_arms:
            variant = variants[(case, arm)]
            metric = metrics[(case, arm)]
            date, seq, person, object_name = e163_export.parse_identity(variant)
            object_name = OBJECT_NAME.get(variant["object_key"], object_name)
            is_postprocess = variant.get("arm_kind") == "postprocess"
            cem_result = variant["postprocess_output_npz"] if is_postprocess else variant["result_npz"]
            cem_outdir = variant["postprocess_output_npz"] if is_postprocess else variant["outdir_npz"]
            rows.append(
                {
                    "synthetic_case_id": synthetic_case_id(variant["case_id"], arm),
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
                    "foot_slip_max_m": metric.get("foot_slip_max_m", ""),
                    "obj_speed_max": metric.get("obj_speed_max", ""),
                }
            )
    return rows


def source_notes(row: dict[str, str]) -> str:
    return (
        f"E166 {row['arm']} downstream export source_ref={SOURCE_REF}; "
        f"original_case={row['original_case_id']}; variant={row['variant']}; "
        f"tracked={row['success_tracked']}; fall={row['fall_flag']}; "
        f"spider_gate={row['spider_gate_pass']}; rawContact={row['raw_contact_in_mask']}; "
        f"inmaskC3={row['inmask_contact_3mm']}; physPen3={row['phys_penetration_3mm']}; "
        f"geomPen2={row['hand_geom_penetration_2mm']}; "
        f"trackJerk={row['trackbody_jerk_p95']}; ankleAcc={row['ankle_acc_max']}; "
        f"footSlip={row['foot_slip_max_m']}; objV={row['obj_speed_max']}"
    )


def validate_source_rows(rows: list[dict[str, str]], *, allow_spider_gate_fail: bool = False) -> None:
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
                "candidate_decision": f"E166_{source['arm']}_READY",
                "target_gate_status": "pass",
                "visual_qc_status": "pass",
                "target_scene": source["target_scene"],
                "trajectory": source["trajectory"],
                "scene_act": source["scene_act"],
                "contact_mask": source["contact_mask"],
                "stage2b_target_task": source["derived_task"],
                "stage2b_result_root": "workspace/core4d/results/E166/foot_smooth_retarget",
                "stage2b_manifest_ref": "workspace/core4d/scripts/experiments/E166/variants.tsv",
                "raw_contact_threshold_label": "3cm",
                "source_exp": SOURCE_EXP_ID,
                "source_variant": source["variant"],
                "source_metrics_ref": E166_METRICS_REF,
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
                "cem_metrics_ref": E166_METRICS_REF,
            }
        )
    return rows


def validate_exports(rl_export_input: Path, partner_manifest: Path, expected_rows: int, *, require_partner: bool = True) -> None:
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

    if not require_partner:
        return
    if not partner_manifest.is_file():
        return
    partner_rows = read_tsv(partner_manifest)
    if len(partner_rows) != expected_rows:
        raise SystemExit(f"expected {expected_rows} partner rows, got {len(partner_rows)}")
    failed = [row for row in partner_rows if row.get("partner_status") != "pass"]
    if failed:
        raise SystemExit(f"partner OmniRetarget failed rows: {failed}")


def summarize(source_rows: list[dict[str, str]], rl_export_input: Path, partner_manifest: Path, out_path: Path) -> dict[str, Any]:
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
        "partner_status_counts": dict(Counter(row.get("partner_status", "") for row in partner_rows)),
    }
    write_json(out_path.with_suffix(".json"), summary)
    lines = [
        "# E166 foot/smooth RL export summary",
        "",
        f"- created_at: `{summary['created_at']}`",
        f"- result_root: `{summary['result_root']}`",
        f"- source rows: `{summary['source_rows']}`",
        f"- arms: `{summary['source_arms']}`",
        f"- RL export decisions: `{summary['rl_export_decision_counts']}`",
        f"- partner statuses: `{summary['partner_status_counts']}`",
        "",
        "| synthetic case | original case | arm | scene_act | CEM result |",
        "|---|---|---|---|---|",
    ]
    for row in source_rows:
        lines.append(
            f"| `{row['synthetic_case_id']}` | `{row['original_case_id']}` | `{row['arm']}` | "
            f"`{row['scene_act']}` | `{row['cem_result_npz']}` |"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    global RESULT_ROOT, SOURCE_REF, E166_METRICS_REF
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=RESULT_ROOT)
    parser.add_argument("--arms", default="A,A_B2_postSmooth")
    parser.add_argument("--target-cases", default=",".join(TARGET_SHORT_CASES))
    parser.add_argument("--variants-tsv", type=Path, default=E166_VARIANTS)
    parser.add_argument("--metrics-tsv", type=Path, default=E166_METRICS)
    parser.add_argument("--metrics-ref", default=E166_METRICS_REF)
    parser.add_argument("--source-ref", default=SOURCE_REF)
    parser.add_argument("--core4d-raw-root", type=Path, default=RAW_ROOT)
    parser.add_argument("--smplx-model-dir", type=Path, default=SMPLX_DIR)
    parser.add_argument("--holosoma-repo", type=Path, default=HOLOSOMA_REPO)
    parser.add_argument("--python-bin", default=".venv/bin/python")
    parser.add_argument("--execute-partner", action="store_true")
    parser.add_argument("--force-partner", action="store_true")
    parser.add_argument("--skip-partner", action="store_true")
    parser.add_argument(
        "--allow-spider-gate-fail",
        action="store_true",
        help="Allow RL export when success/fall/files pass but spider_gate_pass is false; used for explicit caveat runs.",
    )
    args = parser.parse_args()

    RESULT_ROOT = args.out_root.expanduser().resolve()
    SOURCE_REF = args.source_ref
    E166_METRICS_REF = args.metrics_ref
    arms = [arm.strip() for arm in args.arms.split(",") if arm.strip()]
    target_cases = [case.strip() for case in args.target_cases.split(",") if case.strip()]
    source_rows = load_sources(arms, target_cases, args.variants_tsv, args.metrics_tsv)
    validate_source_rows(source_rows, allow_spider_gate_fail=args.allow_spider_gate_fail)

    manifest_dir = RESULT_ROOT / "manifest"
    handoff_dir = RESULT_ROOT / "s5_handoff"
    evidence_dir = RESULT_ROOT / "s6_downstream/evidence"
    rl_export_dir = RESULT_ROOT / "s6_downstream/rl_export"
    partner_dir = rl_export_dir / "partner_omnirt"
    for path in [manifest_dir, handoff_dir, evidence_dir, rl_export_dir, partner_dir]:
        path.mkdir(parents=True, exist_ok=True)

    source_path = manifest_dir / "footSmooth_source_rows.tsv"
    handoff_path = handoff_dir / "handoff_manifest.tsv"
    evidence_input = evidence_dir / "downstream_evidence_input.tsv"
    write_tsv(source_path, source_rows, SOURCE_FIELDS)
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

    partner_manifest = partner_dir / ("rl_partner_omnirt_manifest.skipped.tsv" if args.skip_partner else "rl_partner_omnirt_manifest.tsv")
    if not args.skip_partner:
        partner_args = argparse.Namespace(
            execute_partner=args.execute_partner,
            allow_partner_failure=False,
            force_partner=args.force_partner,
            holosoma_repo=args.holosoma_repo,
            core4d_raw_root=args.core4d_raw_root,
            smplx_model_dir=args.smplx_model_dir,
            python_bin=args.python_bin,
        )
        e163_export.run_partner_export(partner_args, source_rows, rl_export_dir, partner_dir)
    validate_exports(rl_export_dir / "rl_export_input.tsv", partner_manifest, len(source_rows), require_partner=not args.skip_partner)
    summary = summarize(source_rows, rl_export_dir / "rl_export_input.tsv", partner_manifest, RESULT_ROOT / "summary.md")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
