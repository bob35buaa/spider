#!/usr/bin/env python3
"""Freeze the exact E173 box023 paired authority for E179.

The script is read-only with respect to E173.  It verifies the frozen E173
Full-manifest hash, closes the 46-row box023 raw funnel to the exact 16
CEM-eligible rows, records all reusable input authorities and recomputes the
E173 baseline with the mandatory E179 twelve-gate adapter.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e179_common as C  # noqa: E402


BASELINE_METRICS = (
    "fall_flag",
    "release_gate_applicable",
    "body_z_err_p95_m",
    "hand_object_physics_contact_in_mask_frac",
    "hand_object_release_false_contact_3mm_frac",
    "hand_object_physics_penetration_3mm_frame_frac",
    "leg_penetration_frac",
    "track_root_pos_err_cm_mean",
    "track_root_ori_err_deg_mean",
    "track_eef_pos_err_cm_mean",
    "track_eef_ori_err_deg_mean",
    "track_obj_pos_err_cm_mean",
    "track_obj_ori_err_deg_mean",
)


def must_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"{label}: {path}")
    return path


def stage2b_index() -> dict[tuple[str, str], dict[str, str]]:
    rows: list[dict[str, str]] = []
    for variant in ("omnirt_v1", "omnirt_v2"):
        manifest = (
            C.E173_RESULTS
            / f"s3_retarget/{variant}/ref_fk/"
            f"stage2b_manifest_{variant}_ref_fk.tsv"
        )
        rows.extend(C.read_tsv(manifest))
    return {
        (row["case_id"], row["retarget_variant_id"]): row
        for row in rows
        if row.get("stage2b_status") == "pass"
    }


def file_record(label: str, path: Path) -> dict[str, Any]:
    return {
        "label": label,
        "path": C.rel(path),
        "exists": path.is_file(),
        "bytes": path.stat().st_size if path.is_file() else 0,
        "sha256": C.sha256(path) if path.is_file() else "",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--require-restored-tasks",
        action="store_true",
        help="also require the 16 local Stage2b task directories",
    )
    args = parser.parse_args()

    manifest_sha = C.sha256(C.E173_FULL_MANIFEST)
    if manifest_sha != C.EXPECTED_E173_MANIFEST_SHA256:
        raise ValueError(
            "E173 Full manifest hash drift: "
            f"{manifest_sha} != {C.EXPECTED_E173_MANIFEST_SHA256}"
        )
    full_rows = [
        row
        for row in C.read_tsv(C.E173_FULL_MANIFEST)
        if row.get("object_key") == "box023"
    ]
    if len(full_rows) != C.EXPECTED_PAIRED_ROWS:
        raise ValueError(f"expected 16 box023 Full rows, got {len(full_rows)}")
    case_ids = [row["case_id"] for row in full_rows]
    if len(set(case_ids)) != C.EXPECTED_PAIRED_ROWS:
        raise ValueError("duplicate case_id in E173 box023 Full rows")
    complete_statuses = {
        "run_complete_pending_eval",
        "run_complete",
        "eval_complete",
    }
    bad_status = [
        row["case_id"]
        for row in full_rows
        if row.get("status") not in complete_statuses
    ]
    if bad_status:
        raise ValueError(f"E173 rows not complete: {bad_status}")
    distribution = Counter(
        row["selected_retarget_variant_id"] for row in full_rows
    )
    if dict(distribution) != C.EXPECTED_VARIANTS:
        raise ValueError(f"retarget distribution drift: {dict(distribution)}")
    C.validate_queue_contract(set(case_ids))

    raw_rows = [
        row
        for row in C.read_tsv(C.E173_PIPELINE_AUTHORITY)
        if row.get("object_key") == "box023"
    ]
    if len(raw_rows) != C.EXPECTED_RAW_BOX023_ROWS:
        raise ValueError(
            f"E173 box023 raw authority must be 46: {len(raw_rows)}"
        )

    profile_doc = json.loads(C.E167A_PROFILE.read_text(encoding="utf-8"))
    if profile_doc.get("status") != "pass":
        raise ValueError("E167A profile is not passing")
    if profile_doc.get("spider_method_id") != C.SPIDER_METHOD_ID:
        raise ValueError("E167A profile method drift")
    if (
        profile_doc.get("profile_sha256")
        != C.EXPECTED_E167A_PROFILE_SHA256
    ):
        raise ValueError("E167A profile SHA drift")

    stage_index = stage2b_index()
    metrics_rows = {
        row["case_id"]: row
        for row in C.read_tsv(C.E173_METRICS)
        if row.get("object_key") == "box023"
    }
    if set(metrics_rows) != set(case_ids):
        raise ValueError(
            "E173 metric set differs from Full box023 authority"
        )

    authority_rows: list[dict[str, Any]] = []
    hash_rows: list[dict[str, Any]] = [
        file_record("e173_full_manifest", C.E173_FULL_MANIFEST),
        file_record("e173_metrics", C.E173_METRICS),
        file_record("e167a_profile_json", C.E167A_PROFILE),
        file_record(
            "e173_pipeline_authority", C.E173_PIPELINE_AUTHORITY
        ),
    ]
    for ordinal, full in enumerate(full_rows, 1):
        case_id = full["case_id"]
        variant = full["selected_retarget_variant_id"]
        stage = stage_index.get((case_id, variant))
        if stage is None:
            raise KeyError(f"passing Stage2b row missing: {(case_id, variant)}")

        task_dir = C.TASK_ROOT / full["target_task"]
        local_trajectory = task_dir / "0/trajectory_kinematic.npz"
        local_target_scene = task_dir / "scene.xml"
        local_scene_act = task_dir / "scene_act.xml"
        snapshot_dir = (
            C.E173_RESULTS / "scene_snapshot/cem_sidecars" / case_id
        )
        snapshot_pristine = must_file(
            snapshot_dir / "scene_act.xml",
            f"{case_id} pristine scene snapshot",
        )
        source_scene = must_file(
            C.E173_RESULTS
            / f"scene_snapshot/source_templates/box023_{stage['person']}/"
            "scene.xml",
            f"{case_id} source template snapshot",
        )
        trimmed = must_file(
            C.repo_path(stage["trimmed_npz"]),
            f"{case_id} trimmed Stage2b input",
        )
        contact_mask = must_file(
            C.repo_path(full["contact_mask"]),
            f"{case_id} 3cm contact mask",
        )
        dcv3_override = must_file(
            C.OVERRIDE_DIR
            / (
                f"core4d_dcv3_{variant}_ref_fk_{case_id}.yaml"
            ),
            f"{case_id} dcv3 override",
        )
        e173_scene = must_file(
            C.repo_path(full["scene_act"]), f"{case_id} E173 scene"
        )
        e173_result = must_file(
            C.repo_path(full["result_npz"]), f"{case_id} E173 result"
        )
        e173_outdir = must_file(
            C.repo_path(full["outdir_npz"]), f"{case_id} E173 outdir"
        )
        e173_config = must_file(
            C.repo_path(full["config_act"]), f"{case_id} E173 config"
        )
        e173_video = must_file(
            C.repo_path(full["video"]), f"{case_id} E173 video"
        )

        restored = all(
            path.is_file()
            for path in (
                local_trajectory,
                local_target_scene,
                local_scene_act,
            )
        )
        if args.require_restored_tasks and not restored:
            raise FileNotFoundError(
                f"{case_id} Stage2b task is not restored: {task_dir}"
            )
        row = {
            "ordinal": ordinal,
            "case_id": case_id,
            "object_key": "box023",
            "date": full["date"],
            "seq": full["seq"],
            "person": full["person"],
            "person_idx": stage["person_idx"],
            "object_name": stage["object_name"],
            "object_model_rel": stage["object_model_rel"],
            "data_id": stage.get("data_id", "0") or "0",
            "retarget_variant_id": variant,
            "selected_retarget_variant_id": variant,
            "rescue_of": full.get("rescue_of", ""),
            "target_variant_id": "ref_fk",
            "target_task": full["target_task"],
            "worker": C.worker_for_case(case_id),
            "canary_selected": case_id in C.CANARY_CASES,
            "source_scene": C.rel(source_scene),
            "source_scene_sha256": C.sha256(source_scene),
            "trimmed_npz": C.rel(trimmed),
            "trimmed_npz_sha256": C.sha256(trimmed),
            "target_scene": C.rel(local_target_scene),
            "trajectory": C.rel(local_trajectory),
            "trajectory_sha256": full["trajectory_sha256"],
            "base_scene_act": C.rel(local_scene_act),
            "pristine_scene_snapshot": C.rel(snapshot_pristine),
            "pristine_scene_snapshot_sha256": C.sha256(
                snapshot_pristine
            ),
            "contact_mask": C.rel(contact_mask),
            "contact_mask_sha256": C.sha256(contact_mask),
            "contact_mask_label": full.get(
                "contact_mask_label", "3cm"
            ),
            "dcv3_override_id": dcv3_override.stem,
            "dcv3_override_path": C.rel(dcv3_override),
            "dcv3_override_sha256": C.sha256(dcv3_override),
            "e173_result_npz": C.rel(e173_result),
            "e173_result_npz_sha256": C.sha256(e173_result),
            "e173_outdir_npz": C.rel(e173_outdir),
            "e173_outdir_npz_sha256": C.sha256(e173_outdir),
            "e173_config_act": C.rel(e173_config),
            "e173_config_act_sha256": C.sha256(e173_config),
            "e173_scene_act": C.rel(e173_scene),
            "e173_scene_act_sha256": C.sha256(e173_scene),
            "e173_video": C.rel(e173_video),
            "e173_video_sha256": C.sha256(e173_video),
            "stage2b_task_restored": restored,
            "input_authority_status": (
                "ready" if restored else "restore_required"
            ),
        }
        authority_rows.append(row)
        for label, path in (
            ("source_scene", source_scene),
            ("trimmed_npz", trimmed),
            ("pristine_scene_snapshot", snapshot_pristine),
            ("contact_mask", contact_mask),
            ("dcv3_override", dcv3_override),
            ("e173_result_npz", e173_result),
            ("e173_outdir_npz", e173_outdir),
            ("e173_config_act", e173_config),
            ("e173_scene_act", e173_scene),
            ("e173_video", e173_video),
        ):
            hash_rows.append(
                {
                    "case_id": case_id,
                    **file_record(label, path),
                }
            )
        hash_rows.append(
            {
                "case_id": case_id,
                "label": "trajectory_frozen_authority",
                "path": C.rel(local_trajectory),
                "exists": local_trajectory.is_file(),
                "bytes": (
                    local_trajectory.stat().st_size
                    if local_trajectory.is_file()
                    else 0
                ),
                "sha256": (
                    C.sha256(local_trajectory)
                    if local_trajectory.is_file()
                    else full["trajectory_sha256"]
                ),
                "expected_sha256": full["trajectory_sha256"],
            }
        )

    baseline_rows: list[dict[str, Any]] = []
    for authority in authority_rows:
        metric = metrics_rows[authority["case_id"]]
        item: dict[str, Any] = {
            "case_id": authority["case_id"],
            "variant": metric.get("variant", ""),
            "object_key": "box023",
            "retarget_variant_id": authority[
                "retarget_variant_id"
            ],
            "e173_historical_numeric_release_pass": metric.get(
                "numeric_release_pass", ""
            ),
            "e173_historical_numeric_failure_modes": metric.get(
                "numeric_failure_modes", ""
            ),
            "e173_metrics_source": C.rel(C.E173_METRICS),
        }
        for key in BASELINE_METRICS:
            item[key] = metric.get(key, "")
        baseline_rows.append(C.apply_12gate_scoring(item))

    legacy_pass = sum(
        bool(row["legacy_physics6_pass"]) for row in baseline_rows
    )
    pass_12gate = sum(
        bool(row["numeric_release_pass"]) for row in baseline_rows
    )
    if legacy_pass != 13 or pass_12gate != 7:
        raise ValueError(
            "E173 baseline drift: "
            f"legacy={legacy_pass}/16 twelve={pass_12gate}/16"
        )

    input_dir = C.RESULTS / "input_authority"
    C.write_tsv(input_dir / "input_authority.tsv", authority_rows)
    C.write_json(input_dir / "input_authority.json", authority_rows)
    C.write_tsv(
        input_dir / "e173_box023_baseline.tsv", baseline_rows
    )
    C.write_tsv(
        input_dir / "e173_box023_baseline_12gate.tsv",
        baseline_rows,
    )
    C.write_json(
        input_dir / "e173_box023_baseline_12gate.json",
        baseline_rows,
    )
    C.write_tsv(input_dir / "hash_audit.tsv", hash_rows)
    summary = {
        "created_at": C.now(),
        "status": "pass",
        "source_experiment": "E173",
        "source_manifest": C.rel(C.E173_FULL_MANIFEST),
        "source_manifest_sha256": manifest_sha,
        "raw_box023_authority_rows": len(raw_rows),
        "paired_expected": C.EXPECTED_PAIRED_ROWS,
        "paired_rows": len(authority_rows),
        "unique_case_ids": len(set(case_ids)),
        "selected_variant_distribution": dict(distribution),
        "worker_distribution": dict(
            Counter(row["worker"] for row in authority_rows)
        ),
        "canary_cases": list(C.CANARY_CASES),
        "stage2b_tasks_restored": sum(
            bool(row["stage2b_task_restored"])
            for row in authority_rows
        ),
        "stage2b_tasks_restore_required": sum(
            not bool(row["stage2b_task_restored"])
            for row in authority_rows
        ),
        "e167a_profile_sha256": profile_doc[
            "profile_sha256"
        ],
        "scoring_contract_id": C.SCORING_CONTRACT_ID,
        "tracking_gates_enabled": True,
        "e173_legacy_physics6_pass": legacy_pass,
        "e173_numeric_release_pass_12gate": pass_12gate,
        "gate_thresholds": C.GATE_THRESHOLDS,
        "e173_read_only": True,
    }
    C.write_json(input_dir / "hash_audit.json", {
        **summary,
        "files": hash_rows,
    })
    C.write_json(input_dir / "authority_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
