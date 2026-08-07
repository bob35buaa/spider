#!/usr/bin/env python3
"""Freeze the exact E172(box004)+E173(box024,box001) paired authority for E189.

Read-only with respect to E172/E173.  For each of the three objects this
verifies the frozen Full-manifest hash, closes the object's raw funnel to
the exact CEM-eligible rows, records all reusable input authorities and
recomputes the E172/E173 baseline with the mandatory E189 twelve-gate
adapter.  box023 is intentionally excluded (already closed by E179).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e189_common as C  # noqa: E402


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

# Historical six-gate (legacy) pass counts already published by E172/E173;
# used only as a drift sanity check on the read-only metrics join, not as a
# claim about the (still to-be-computed) twelve-gate numbers.
EXPECTED_LEGACY_PASS = {"box004": 5, "box024": 3, "box001": 10}
COMPLETE_STATUSES = {
    "run_complete_pending_eval",
    "run_complete",
    "eval_complete",
}


def must_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"{label}: {path}")
    return path


def stage2b_index(results_dir: Path) -> dict[tuple[str, str], dict[str, str]]:
    rows: list[dict[str, str]] = []
    for variant in ("omnirt_v1", "omnirt_v2"):
        manifest = (
            results_dir
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


def collect_object(
    object_key: str, hash_rows: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    source = C.SOURCES[object_key]
    results_dir: Path = source["results"]
    full_manifest: Path = source["full_manifest"]
    metrics_path: Path = source["metrics"]
    pipeline_authority: Path = source["pipeline_authority"]

    manifest_sha = C.sha256(full_manifest)
    full_rows = [
        row
        for row in C.read_tsv(full_manifest)
        if row.get("object_key") == object_key
    ]
    if len(full_rows) != source["expected_rows"]:
        raise ValueError(
            f"{object_key}: expected {source['expected_rows']} Full rows, "
            f"got {len(full_rows)}"
        )
    case_ids = [row["case_id"] for row in full_rows]
    if len(set(case_ids)) != len(case_ids):
        raise ValueError(f"{object_key}: duplicate case_id in Full rows")
    bad_status = [
        row["case_id"]
        for row in full_rows
        if row.get("status") not in COMPLETE_STATUSES
    ]
    if bad_status:
        raise ValueError(f"{object_key}: rows not complete: {bad_status}")
    distribution = Counter(
        row["selected_retarget_variant_id"] for row in full_rows
    )
    if dict(distribution) != source["expected_variants"]:
        raise ValueError(
            f"{object_key}: retarget distribution drift: "
            f"{dict(distribution)} != {source['expected_variants']}"
        )

    raw_rows = [
        row
        for row in C.read_tsv(pipeline_authority)
        if row.get("object_key") == object_key
    ]

    stage_index = stage2b_index(results_dir)
    metrics_rows = {
        row["case_id"]: row
        for row in C.read_tsv(metrics_path)
        if row.get("object_key") == object_key
    }
    if set(metrics_rows) != set(case_ids):
        raise ValueError(f"{object_key}: metric set differs from Full authority")

    authority_rows: list[dict[str, Any]] = []
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
        snapshot_dir = results_dir / "scene_snapshot/cem_sidecars" / case_id
        snapshot_pristine = must_file(
            snapshot_dir / "scene_act.xml", f"{case_id} pristine scene snapshot"
        )
        source_scene = must_file(
            results_dir
            / f"scene_snapshot/source_templates/{object_key}_{stage['person']}/"
            "scene.xml",
            f"{case_id} source template snapshot",
        )
        trimmed = must_file(
            C.repo_path(stage["trimmed_npz"]), f"{case_id} trimmed Stage2b input"
        )
        contact_mask = must_file(
            C.repo_path(full["contact_mask"]), f"{case_id} 3cm contact mask"
        )
        dcv3_override = must_file(
            C.OVERRIDE_DIR
            / f"core4d_dcv3_{variant}_ref_fk_{case_id}.yaml",
            f"{case_id} dcv3 override",
        )
        source_scene_act = must_file(
            C.repo_path(full["scene_act"]), f"{case_id} {source['experiment_id']} scene"
        )
        source_result = must_file(
            C.repo_path(full["result_npz"]), f"{case_id} {source['experiment_id']} result"
        )
        source_outdir = must_file(
            C.repo_path(full["outdir_npz"]), f"{case_id} {source['experiment_id']} outdir"
        )
        source_config = must_file(
            C.repo_path(full["config_act"]), f"{case_id} {source['experiment_id']} config"
        )
        source_video = must_file(
            C.repo_path(full["video"]), f"{case_id} {source['experiment_id']} video"
        )

        restored = all(
            path.is_file()
            for path in (local_trajectory, local_target_scene, local_scene_act)
        )
        row = {
            "ordinal": ordinal,
            "case_id": case_id,
            "object_key": object_key,
            "source_experiment_id": source["experiment_id"],
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
            "pristine_scene_snapshot_sha256": C.sha256(snapshot_pristine),
            "contact_mask": C.rel(contact_mask),
            "contact_mask_sha256": C.sha256(contact_mask),
            "contact_mask_label": full.get("contact_mask_label", "3cm"),
            "dcv3_override_id": dcv3_override.stem,
            "dcv3_override_path": C.rel(dcv3_override),
            "dcv3_override_sha256": C.sha256(dcv3_override),
            "source_result_npz": C.rel(source_result),
            "source_result_npz_sha256": C.sha256(source_result),
            "source_outdir_npz": C.rel(source_outdir),
            "source_outdir_npz_sha256": C.sha256(source_outdir),
            "source_config_act": C.rel(source_config),
            "source_config_act_sha256": C.sha256(source_config),
            "source_scene_act": C.rel(source_scene_act),
            "source_scene_act_sha256": C.sha256(source_scene_act),
            "source_video": C.rel(source_video),
            "source_video_sha256": C.sha256(source_video),
            "source_preprg_scene_name": source["preprg_scene_name"],
            "stage2b_task_restored": restored,
            "input_authority_status": "ready" if restored else "restore_required",
        }
        authority_rows.append(row)
        for label, path in (
            ("source_scene", source_scene),
            ("trimmed_npz", trimmed),
            ("pristine_scene_snapshot", snapshot_pristine),
            ("contact_mask", contact_mask),
            ("dcv3_override", dcv3_override),
            ("source_result_npz", source_result),
            ("source_outdir_npz", source_outdir),
            ("source_config_act", source_config),
            ("source_scene_act", source_scene_act),
            ("source_video", source_video),
        ):
            hash_rows.append({"case_id": case_id, **file_record(label, path)})

    baseline_rows: list[dict[str, Any]] = []
    for authority in authority_rows:
        metric = metrics_rows[authority["case_id"]]
        item: dict[str, Any] = {
            "case_id": authority["case_id"],
            "variant": metric.get("variant", ""),
            "object_key": object_key,
            "source_experiment_id": source["experiment_id"],
            "retarget_variant_id": authority["retarget_variant_id"],
            "source_historical_numeric_release_pass": metric.get(
                "numeric_release_pass", ""
            ),
            "source_historical_numeric_failure_modes": metric.get(
                "numeric_failure_modes", ""
            ),
            "source_metrics_path": C.rel(metrics_path),
        }
        for key in BASELINE_METRICS:
            item[key] = metric.get(key, "")
        baseline_rows.append(C.apply_12gate_scoring(item))

    legacy_pass = sum(bool(row["legacy_physics6_pass"]) for row in baseline_rows)
    pass_12gate = sum(bool(row["numeric_release_pass"]) for row in baseline_rows)
    if legacy_pass != EXPECTED_LEGACY_PASS[object_key]:
        raise ValueError(
            f"{object_key} legacy six-gate baseline drift: "
            f"{legacy_pass}/{len(baseline_rows)} != "
            f"{EXPECTED_LEGACY_PASS[object_key]}"
        )

    hash_rows.append(file_record(f"{object_key}_full_manifest", full_manifest))
    hash_rows.append(file_record(f"{object_key}_metrics", metrics_path))
    hash_rows.append(
        file_record(f"{object_key}_pipeline_authority", pipeline_authority)
    )

    object_summary = {
        "object_key": object_key,
        "source_experiment_id": source["experiment_id"],
        "source_manifest": C.rel(full_manifest),
        "source_manifest_sha256": manifest_sha,
        "raw_authority_rows": len(raw_rows),
        "paired_rows": len(authority_rows),
        "selected_variant_distribution": dict(distribution),
        "legacy_physics6_pass": legacy_pass,
        "numeric_release_pass_12gate": pass_12gate,
    }
    return authority_rows, baseline_rows, object_summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--require-restored-tasks",
        action="store_true",
        help="also require the 43 local Stage2b task directories",
    )
    args = parser.parse_args()

    profile_doc = json.loads(C.E167A_PROFILE.read_text(encoding="utf-8"))
    if profile_doc.get("status") != "pass":
        raise ValueError("E167A profile is not passing")
    if profile_doc.get("spider_method_id") != C.SPIDER_METHOD_ID:
        raise ValueError("E167A profile method drift")
    if profile_doc.get("profile_sha256") != C.EXPECTED_E167A_PROFILE_SHA256:
        raise ValueError("E167A profile SHA drift")

    hash_rows: list[dict[str, Any]] = [
        file_record("e167a_profile_json", C.E167A_PROFILE)
    ]
    all_authority: list[dict[str, Any]] = []
    all_baseline: list[dict[str, Any]] = []
    object_summaries: dict[str, Any] = {}
    for object_key in ("box004", "box024", "box001"):
        authority_rows, baseline_rows, object_summary = collect_object(
            object_key, hash_rows
        )
        all_authority.extend(authority_rows)
        all_baseline.extend(baseline_rows)
        object_summaries[object_key] = object_summary

    if len(all_authority) != C.EXPECTED_PAIRED_ROWS:
        raise ValueError(
            f"expected {C.EXPECTED_PAIRED_ROWS} total rows, got {len(all_authority)}"
        )
    case_ids = {row["case_id"] for row in all_authority}
    if len(case_ids) != C.EXPECTED_PAIRED_ROWS:
        raise ValueError("duplicate case_id across objects")
    if C.EXCLUDED_OBJECT in {row["object_key"] for row in all_authority}:
        raise ValueError("box023 must not appear in E189 authority")

    gpu_queues = C.compute_gpu_queues(list(case_ids))
    C.validate_queue_contract(case_ids, gpu_queues)
    for row in all_authority:
        row["worker"] = C.worker_for_case(row["case_id"], gpu_queues)

    if args.require_restored_tasks:
        not_restored = [
            row["case_id"]
            for row in all_authority
            if not C.boolish(row["stage2b_task_restored"])
        ]
        if not_restored:
            raise FileNotFoundError(f"Stage2b tasks not restored: {not_restored}")

    input_dir = C.RESULTS / "input_authority"
    C.write_tsv(input_dir / "input_authority.tsv", all_authority)
    C.write_json(input_dir / "input_authority.json", all_authority)
    C.write_tsv(input_dir / "e172_e173_baseline_12gate.tsv", all_baseline)
    C.write_json(input_dir / "e172_e173_baseline_12gate.json", all_baseline)
    C.write_tsv(input_dir / "hash_audit.tsv", hash_rows)

    canary_selected = sum(bool(row["canary_selected"]) for row in all_authority)
    summary = {
        "created_at": C.now(),
        "status": "pass",
        "excluded_object": C.EXCLUDED_OBJECT,
        "excluded_object_reference": (
            "E179 16/16 box023 paired ablation, conclusion PRG_BETTER; "
            "referenced only, not recomputed"
        ),
        "paired_expected": C.EXPECTED_PAIRED_ROWS,
        "paired_rows": len(all_authority),
        "unique_case_ids": len(case_ids),
        "object_summaries": object_summaries,
        "worker_distribution": dict(
            Counter(row["worker"] for row in all_authority)
        ),
        "gpu_queues": {worker: list(cases) for worker, cases in gpu_queues.items()},
        "canary_cases": list(C.CANARY_CASES),
        "canary_selected_rows": canary_selected,
        "stage2b_tasks_restored": sum(
            bool(row["stage2b_task_restored"]) for row in all_authority
        ),
        "stage2b_tasks_restore_required": sum(
            not bool(row["stage2b_task_restored"]) for row in all_authority
        ),
        "e167a_profile_sha256": profile_doc["profile_sha256"],
        "scoring_contract_id": C.SCORING_CONTRACT_ID,
        "tracking_gates_enabled": True,
        "gate_thresholds": C.GATE_THRESHOLDS,
        "source_read_only": True,
    }
    C.write_json(input_dir / "hash_audit.json", {**summary, "files": hash_rows})
    C.write_json(input_dir / "authority_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
