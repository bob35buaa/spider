#!/usr/bin/env python3
"""Export final E187 manual-USE cases as partner-complete RL inputs.

This is an S6-only packaging step. Human USE controls the exported source set;
numeric release results remain warnings/provenance. Opposite-person partners are
resolved from passing E174 Stage2b evidence via the canonical shared adapter.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


REPO = Path(__file__).resolve().parents[5]
RUN_ROOT = REPO / "workspace/core4d/results/E187"
E174_ROOT = REPO / "workspace/core4d/results/E174"
REVIEW = RUN_ROOT / "s6_downstream/eval/full/user_manual_review_filled.tsv"
EVAL_MANIFEST = RUN_ROOT / "s6_downstream/manifests/e187_full_evaluation_manifest.tsv"
METRICS = RUN_ROOT / "s6_downstream/eval/full/e187_case_metrics.tsv"
OUT = RUN_ROOT / "s6_downstream/rl_export"
PARTNER_OUT = OUT / "partner_omnirt"
EXPECTED_REVIEW_SHA256 = "42dd0ef48cb47dde6cd35d88db0a4e46932e89bbe123227eae2b98f5252f431b"
EXPECTED_USE = 14
EXPECTED_DO_NOT_USE = 8
METHOD_ID = "E187_canonical_distance_continuation_reward_r1"
HAND_COLLISION_ID = "object_specific_coacd_compound"
STAGE2B_MANIFESTS = [
    E174_ROOT / "s3_retarget/omnirt_v1/ref_fk/stage2b_manifest_omnirt_v1_ref_fk.tsv",
    E174_ROOT / "s3_retarget/omnirt_v2/ref_fk/stage2b_manifest_omnirt_v2_ref_fk.tsv",
]


SOURCE_FIELDS = [
    "case_id", "object_key", "object_name", "date", "seq", "person", "person_idx",
    "retarget_variant_id", "target_variant_id", "hand_collision_variant_id",
    "source_exp_id", "spider_method_id", "handoff_decision", "candidate_decision",
    "target_gate_status", "visual_qc_status", "target_scene", "trajectory", "scene_act",
    "contact_mask", "stage2b_target_task", "stage2b_result_root", "stage2b_manifest_ref",
    "raw_contact_threshold_label", "cem_status", "cem_run_id", "cem_result_npz",
    "cem_video", "cem_metrics_ref", "downstream_decision", "downstream_failure_mode",
    "downstream_notes", "rl_export_decision", "skip_reason", "scene_act_exists",
    "trajectory_exists", "contact_mask_exists", "cem_result_exists",
    "source_handoff_manifest", "source_cem_evidence", "schema_version", "updated_at",
    "manual_use_decision", "manual_quality_label", "manual_failure_taxonomy",
    "manual_review_note", "manual_reviewer", "manual_reviewed_at", "manual_review_ref",
    "manual_review_sha256", "numeric_release_pass", "numeric_failure_modes",
    "leg_gate_health_pass", "fall_gate_pass", "body_z_gate_pass", "contact_gate_pass",
    "release_gate_pass", "hand_penetration_gate_pass", "lower_body_gate_pass",
    "root_pos_gate_pass", "root_ori_gate_pass", "hand_pos_gate_pass",
    "hand_ori_gate_pass", "object_pos_gate_pass", "object_ori_gate_pass",
    "c9_technical_status", "c9_progression_authority", "execution_kind",
    "result_sha256", "scene_sha256", "trajectory_sha256", "contact_mask_sha256",
    "metrics_sha256", "evaluation_manifest_sha256",
]

ALIGNMENT_FIELDS = [
    "source_case_id", "source_person", "source_retarget_variant_id", "source_stage2b_manifest",
    "source_trim_window_json", "source_trim_start", "source_trim_end", "source_trim_frames",
    "source_trajectory_frames", "source_cem_frames", "source_contact_mask_frames",
    "partner_case_id", "partner_person", "partner_retarget_variant_id",
    "partner_stage2b_manifest", "partner_trim_window_json", "partner_trim_start",
    "partner_trim_end", "partner_trim_frames", "partner_npz_frames", "common_raw_start",
    "common_raw_end", "common_raw_frames", "source_crop_offset", "partner_crop_offset",
    "alignment_policy", "alignment_status", "alignment_failure_mode",
]


def load_partner_adapter() -> Any:
    path = REPO / "workspace/core4d/scripts/data_construction_v3/stages/s6_downstream/finalize_reused_partner_rl.py"
    spec = importlib.util.spec_from_file_location("dcv3_finalize_reused_partner_rl", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import partner adapter: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PARTNER = load_partner_adapter()


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def read_tsv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        return list(reader.fieldnames or []), list(reader)


def write_tsv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=fields, delimiter="\t", lineterminator="\n", extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def rel(value: str | Path) -> str:
    return PARTNER.path_ref(value, REPO)


def local(value: str | Path) -> Path:
    return PARTNER.local_path(value, REPO)


def required(value: str | Path, label: str) -> Path:
    return PARTNER.require_file(value, label, REPO)


def unique(rows: list[dict[str, str]], label: str) -> dict[str, dict[str, str]]:
    result = {row.get("case_id", ""): row for row in rows if row.get("case_id")}
    if len(result) != len(rows) or "" in result:
        raise SystemExit(f"{label} must have unique non-empty case_id rows")
    return result


def frame_count(path: Path, label: str, preferred_key: str = "qpos") -> int:
    try:
        with np.load(path, allow_pickle=False) as payload:
            if preferred_key in payload.files:
                array = np.asarray(payload[preferred_key])
                if array.ndim < 1:
                    raise ValueError(f"{preferred_key} is scalar")
                return int(array.shape[0])
            arrays = [np.asarray(payload[key]) for key in payload.files]
            counts = [int(array.shape[0]) for array in arrays if array.ndim >= 1]
            if not counts:
                raise ValueError("no time-axis array")
            if len(set(counts)) != 1:
                raise ValueError(f"ambiguous time axes: {sorted(set(counts))}")
            return counts[0]
    except Exception as exc:
        raise SystemExit(f"invalid {label}: {path}: {exc}") from exc


def trim_window(evidence: dict[str, str], case_id: str) -> tuple[Path, dict[str, Any]]:
    path = PARTNER.trim_window_path(evidence, case_id, REPO)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(f"invalid trim window for {case_id}: {path}: {exc}") from exc
    return path, value


def int_field(value: Any, label: str) -> int:
    try:
        return int(value)
    except Exception as exc:
        raise SystemExit(f"invalid integer {label}: {value!r}") from exc


def source_stage2b(
    source: dict[str, str], stage2b_index: dict[str, list[tuple[Path, dict[str, str]]]]
) -> tuple[Path, dict[str, str]]:
    eligible = [
        item for item in stage2b_index.get(source["case_id"], [])
        if item[1].get("retarget_variant_id") == source["retarget_variant_id"]
        and item[1].get("target_variant_id") == "ref_fk"
    ]
    if len(eligible) != 1:
        raise SystemExit(
            f"expected one passing source Stage2b row for {source['case_id']} "
            f"variant={source['retarget_variant_id']}, got {len(eligible)}"
        )
    return eligible[0]


def validate_authority(
    reviews: list[dict[str, str]], eval_rows: list[dict[str, str]], metrics: list[dict[str, str]]
) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    actual_hash = sha256(REVIEW)
    if actual_hash != EXPECTED_REVIEW_SHA256:
        raise SystemExit(f"manual authority SHA drift: {actual_hash}")
    review_by = unique(reviews, "manual review")
    eval_by = unique(eval_rows, "E187 evaluation manifest")
    metric_by = unique(metrics, "E187 metrics")
    if set(review_by) != set(eval_by) or set(review_by) != set(metric_by):
        raise SystemExit("manual/evaluation/metrics case sets differ")
    invalid = [
        case_id for case_id, row in review_by.items()
        if row.get("user_manual_review_status") != "reviewed"
        or row.get("manual_use_decision") not in {"USE", "DO_NOT_USE"}
    ]
    if invalid:
        raise SystemExit(f"manual authority is not final: {invalid}")
    counts = Counter(row["manual_use_decision"] for row in reviews)
    if counts != Counter({"USE": EXPECTED_USE, "DO_NOT_USE": EXPECTED_DO_NOT_USE}):
        raise SystemExit(f"manual counts drift: {dict(counts)}")
    return review_by, eval_by, metric_by


def make_source_row(
    case_id: str,
    review: dict[str, str],
    evaluation: dict[str, str],
    metric: dict[str, str],
    stage_row: dict[str, str],
    stage_manifest: Path,
) -> dict[str, Any]:
    if evaluation.get("target_variant_id") != "ref_fk":
        raise SystemExit(f"{case_id}: target_variant_id drift")
    if evaluation.get("spider_method_id") != METHOD_ID:
        raise SystemExit(f"{case_id}: spider_method_id drift")
    if evaluation.get("hand_collision_variant_id") != HAND_COLLISION_ID:
        raise SystemExit(f"{case_id}: hand collision variant drift")
    paths = {
        "scene_act": required(evaluation.get("scene_act", ""), f"{case_id} scene_act"),
        "trajectory": required(evaluation.get("trajectory", ""), f"{case_id} trajectory"),
        "contact_mask": required(evaluation.get("contact_mask", ""), f"{case_id} contact_mask"),
        "cem_result_npz": required(evaluation.get("result_npz", ""), f"{case_id} CEM result"),
        "cem_video": required(evaluation.get("video", ""), f"{case_id} CEM video"),
    }
    numeric_pass = metric.get("numeric_release_pass", "")
    numeric_failures = metric.get("numeric_failure_modes", "")
    notes = (
        f"manual={review['manual_use_decision']}/{review['manual_quality_label']}; "
        f"numeric_release_pass={numeric_pass}; numeric_failure_modes={numeric_failures or 'none'}; "
        "manual authority permits RL validation but does not overwrite numeric/C9 facts or claim RL success"
    )
    # Holosoma's historical asset bundle is case-sensitive for Bucket007
    # (`models/Bucket007/Bucket007.obj`); E174 rows mix lower/upper spelling.
    object_name = "Bucket007" if evaluation["object_key"] == "bucket007" else (
        stage_row.get("object_name") or evaluation["object_key"]
    )
    row: dict[str, Any] = {
        "case_id": case_id,
        "object_key": evaluation["object_key"],
        "object_name": object_name,
        "date": stage_row["date"],
        "seq": stage_row["seq"],
        "person": stage_row["person"],
        "person_idx": stage_row["person_idx"],
        "retarget_variant_id": evaluation["retarget_variant_id"],
        "target_variant_id": "ref_fk",
        "hand_collision_variant_id": HAND_COLLISION_ID,
        "source_exp_id": "E187",
        "spider_method_id": METHOD_ID,
        "handoff_decision": "HANDOFF_READY",
        "candidate_decision": "USER_APPROVED_USE",
        "target_gate_status": "pass",
        "visual_qc_status": "manual_use",
        "target_scene": stage_row.get("target_scene", ""),
        "trajectory": rel(paths["trajectory"]),
        "scene_act": rel(paths["scene_act"]),
        "contact_mask": rel(paths["contact_mask"]),
        "stage2b_target_task": stage_row.get("target_task", ""),
        "stage2b_result_root": rel(stage_row.get("result_root", "")),
        "stage2b_manifest_ref": rel(stage_manifest),
        "raw_contact_threshold_label": "3cm",
        "cem_status": "pass",
        "cem_run_id": evaluation.get("variant", ""),
        "cem_result_npz": rel(paths["cem_result_npz"]),
        "cem_video": rel(paths["cem_video"]),
        "cem_metrics_ref": rel(METRICS),
        "downstream_decision": "DOWNSTREAM_USER_APPROVED_FOR_RL_VALIDATION",
        "downstream_failure_mode": numeric_failures,
        "downstream_notes": notes,
        "rl_export_decision": "RL_EXPORT_READY",
        "skip_reason": "",
        "scene_act_exists": "True",
        "trajectory_exists": "True",
        "contact_mask_exists": "True",
        "cem_result_exists": "True",
        "source_handoff_manifest": rel(EVAL_MANIFEST),
        "source_cem_evidence": rel(METRICS),
        "schema_version": "core4d_data_construction_v3.0",
        "updated_at": now(),
        "manual_use_decision": review["manual_use_decision"],
        "manual_quality_label": review["manual_quality_label"],
        "manual_failure_taxonomy": review.get("manual_failure_taxonomy", ""),
        "manual_review_note": review.get("manual_review_note", ""),
        "manual_reviewer": review.get("manual_reviewer", ""),
        "manual_reviewed_at": review.get("manual_reviewed_at", ""),
        "manual_review_ref": rel(REVIEW),
        "manual_review_sha256": EXPECTED_REVIEW_SHA256,
        "numeric_release_pass": numeric_pass,
        "numeric_failure_modes": numeric_failures,
        "c9_technical_status": evaluation.get("c9_technical_status", ""),
        "c9_progression_authority": evaluation.get("c9_progression_authority", ""),
        "execution_kind": evaluation.get("execution_kind", ""),
        "result_sha256": sha256(paths["cem_result_npz"]),
        "scene_sha256": sha256(paths["scene_act"]),
        "trajectory_sha256": sha256(paths["trajectory"]),
        "contact_mask_sha256": sha256(paths["contact_mask"]),
        "metrics_sha256": sha256(METRICS),
        "evaluation_manifest_sha256": sha256(EVAL_MANIFEST),
    }
    for field in (
        "leg_gate_health_pass", "fall_gate_pass", "body_z_gate_pass", "contact_gate_pass",
        "release_gate_pass", "hand_penetration_gate_pass", "lower_body_gate_pass",
        "root_pos_gate_pass", "root_ori_gate_pass", "hand_pos_gate_pass", "hand_ori_gate_pass",
        "object_pos_gate_pass", "object_ori_gate_pass",
    ):
        row[field] = metric.get(field, "")
    return row


def alignment_audit(
    source: dict[str, Any], source_evidence: dict[str, str], source_provenance: Path,
    partner: dict[str, Any], partner_evidence: dict[str, str], partner_provenance: Path,
) -> dict[str, Any]:
    case_id = source["case_id"]
    partner_id = partner["partner_case_id"]
    source_trim_path, source_trim = trim_window(source_evidence, case_id)
    partner_trim_path, partner_trim = trim_window(partner_evidence, partner_id)
    ss = int_field(source_trim.get("trim_start"), f"{case_id} trim_start")
    se = int_field(source_trim.get("trim_end"), f"{case_id} trim_end")
    ps = int_field(partner_trim.get("trim_start"), f"{partner_id} trim_start")
    pe = int_field(partner_trim.get("trim_end"), f"{partner_id} trim_end")
    common_start, common_end = max(ss, ps), min(se, pe)
    common_frames = common_end - common_start
    source_trajectory_frames = frame_count(required(source["trajectory"], f"{case_id} trajectory"), f"{case_id} trajectory")
    source_cem_frames = frame_count(required(source["cem_result_npz"], f"{case_id} CEM"), f"{case_id} CEM")
    source_mask_frames = frame_count(
        required(source["contact_mask"], f"{case_id} contact mask"),
        f"{case_id} contact mask", preferred_key="spider_contact_mask_3cm",
    )
    partner_npz_frames = frame_count(
        required(partner["trimmed_npz"], f"{partner_id} trimmed NPZ"), f"{partner_id} trimmed NPZ"
    )
    source_trim_frames = int_field(source_trim.get("trim_frames", se - ss), f"{case_id} trim_frames")
    partner_trim_frames = int_field(partner_trim.get("trim_frames", pe - ps), f"{partner_id} trim_frames")
    failures: list[str] = []
    if source_trim_frames != se - ss:
        failures.append("source_trim_metadata_mismatch")
    if partner_trim_frames != pe - ps:
        failures.append("partner_trim_metadata_mismatch")
    if source_trajectory_frames != source_trim_frames:
        failures.append("source_trajectory_trim_mismatch")
    if source_cem_frames != source_trajectory_frames:
        failures.append("source_cem_trajectory_mismatch")
    if source_mask_frames != source_trajectory_frames:
        failures.append("source_contact_mask_trajectory_mismatch")
    if partner_npz_frames != partner_trim_frames:
        failures.append("partner_npz_trim_mismatch")
    if common_frames < 2:
        failures.append("no_usable_common_raw_window")
    status = "RL_EXPORT_READY" if not failures else "RL_EXPORT_BLOCKED_PARTNER_ALIGNMENT"
    return {
        "source_case_id": case_id,
        "source_person": source["person"],
        "source_retarget_variant_id": source["retarget_variant_id"],
        "source_stage2b_manifest": rel(source_provenance),
        "source_trim_window_json": rel(source_trim_path),
        "source_trim_start": ss,
        "source_trim_end": se,
        "source_trim_frames": source_trim_frames,
        "source_trajectory_frames": source_trajectory_frames,
        "source_cem_frames": source_cem_frames,
        "source_contact_mask_frames": source_mask_frames,
        "partner_case_id": partner_id,
        "partner_person": partner["partner_person"],
        "partner_retarget_variant_id": partner["partner_retarget_variant_id"],
        "partner_stage2b_manifest": rel(partner_provenance),
        "partner_trim_window_json": rel(partner_trim_path),
        "partner_trim_start": ps,
        "partner_trim_end": pe,
        "partner_trim_frames": partner_trim_frames,
        "partner_npz_frames": partner_npz_frames,
        "common_raw_start": common_start,
        "common_raw_end": common_end,
        "common_raw_frames": common_frames,
        "source_crop_offset": common_start - ss,
        "partner_crop_offset": common_start - ps,
        "alignment_policy": "common_raw_window",
        "alignment_status": status,
        "alignment_failure_mode": ",".join(failures),
    }


def main() -> int:
    _, reviews = read_tsv(REVIEW)
    _, evaluations = read_tsv(EVAL_MANIFEST)
    _, metrics = read_tsv(METRICS)
    review_by, eval_by, metric_by = validate_authority(reviews, evaluations, metrics)
    approved = sorted(case_id for case_id, row in review_by.items() if row["manual_use_decision"] == "USE")
    rejected = {case_id for case_id, row in review_by.items() if row["manual_use_decision"] == "DO_NOT_USE"}

    stage2b_index = PARTNER.load_stage2b_index(STAGE2B_MANIFESTS)
    source_rows: list[dict[str, Any]] = []
    source_stage: dict[str, tuple[Path, dict[str, str]]] = {}
    for case_id in approved:
        provenance, evidence = source_stage2b(eval_by[case_id], stage2b_index)
        source_stage[case_id] = (provenance, evidence)
        source_rows.append(
            make_source_row(case_id, review_by[case_id], eval_by[case_id], metric_by[case_id], evidence, provenance)
        )
    if {row["case_id"] for row in source_rows} != set(approved) or rejected & set(approved):
        raise SystemExit("source selection does not exactly match manual USE authority")

    staging = OUT.parent / f".{OUT.name}.staging"
    if staging.exists():
        shutil.rmtree(staging)
    partner_staging = staging / "partner_omnirt"
    partner_staging.mkdir(parents=True, exist_ok=True)
    source_path = staging / "rl_export_input.tsv"
    write_tsv(source_path, source_rows, SOURCE_FIELDS)
    write_json(staging / "rl_export_input.json", source_rows)
    shutil.copy2(REVIEW, staging / "manual_review_snapshot.tsv")

    partner_rows: list[dict[str, Any]] = []
    alignment_rows: list[dict[str, Any]] = []
    for source in source_rows:
        partner_case, partner_person, partner_person_idx = PARTNER.infer_partner(source)
        partner_provenance, partner_evidence = PARTNER.choose_partner(
            stage2b_index.get(partner_case, []), PARTNER.DEFAULT_VARIANT_PREFERENCE, partner_case
        )
        partner = PARTNER.build_partner_row(
            source,
            partner_case=partner_case,
            partner_person=partner_person,
            partner_person_idx=partner_person_idx,
            evidence=partner_evidence,
            provenance=partner_provenance,
            source_rl=source_path,
            repo=REPO,
            generation_mode="reuse_e174_stage2b_partner",
        )
        source_provenance, source_evidence = source_stage[source["case_id"]]
        audit_row = alignment_audit(
            source, source_evidence, source_provenance, partner, partner_evidence, partner_provenance
        )
        partner_rows.append(partner)
        alignment_rows.append(audit_row)

    blocked = [row for row in alignment_rows if row["alignment_status"] != "RL_EXPORT_READY"]
    write_tsv(staging / "partner_resolution_audit.tsv", alignment_rows, ALIGNMENT_FIELDS)
    if blocked:
        write_json(staging / "validation_report.json", {"status": "blocked", "blocked": blocked})
        raise SystemExit(f"RL export blocked by partner alignment: {[row['source_case_id'] for row in blocked]}")

    partner_manifest = partner_staging / "rl_partner_omnirt_manifest.tsv"
    write_tsv(partner_manifest, partner_rows, PARTNER.PARTNER_FIELDS)
    write_json(partner_staging / "rl_partner_omnirt_manifest.json", partner_rows)
    partner_hash = sha256(partner_manifest)
    paired_rows = [
        PARTNER.paired_row(
            source, partner, manifest_ref=rel(partner_manifest), manifest_hash=partner_hash, repo=REPO
        )
        for source, partner in zip(source_rows, partner_rows, strict=True)
    ]
    paired_path = staging / "paired_rl_export_input.tsv"
    write_tsv(paired_path, paired_rows, SOURCE_FIELDS + PARTNER.PAIRED_EXTRA_FIELDS)
    write_json(staging / "paired_rl_export_input.json", paired_rows)

    summary = {
        "experiment": "E187",
        "stage": "S6_manual_use_partner_rl_export",
        "created_at": now(),
        "status": "RL_EXPORT_READY",
        "manual_authority": rel(REVIEW),
        "manual_authority_sha256": EXPECTED_REVIEW_SHA256,
        "manual_counts": {"USE": EXPECTED_USE, "DO_NOT_USE": EXPECTED_DO_NOT_USE, "PENDING": 0},
        "source_rows": len(source_rows),
        "partner_rows": len(partner_rows),
        "pair_complete_rows": sum(row["pair_status"] == "PAIR_COMPLETE" for row in partner_rows),
        "paired_ready_rows": sum(row["paired_rl_export_decision"] == "RL_EXPORT_READY" for row in paired_rows),
        "numeric_release_counts": dict(Counter(row["numeric_release_pass"] for row in source_rows)),
        "manual_quality_counts": dict(Counter(row["manual_quality_label"] for row in source_rows)),
        "object_counts": dict(Counter(row["object_key"] for row in source_rows)),
        "partner_variant_counts": dict(Counter(row["partner_retarget_variant_id"] for row in partner_rows)),
        "alignment_policy_counts": dict(Counter(row["alignment_policy"] for row in alignment_rows)),
        "all_partner_artifacts_nonempty": True,
        "all_source_partner_hashes_recomputed": True,
        "c9_technical_status": "FAIL",
        "c9_progression_authority": "USER_WAIVED",
        "claim_boundary": "RL export and loader readiness only; no RL outcome claim",
    }
    write_json(staging / "rl_export_summary.json", summary)
    validation = {
        **summary,
        "checks": {
            "authority_sha_exact": True,
            "approved_source_set_exact": True,
            "rejected_zero_entry": True,
            "source_required_files_loadable": True,
            "partner_resolution_14_of_14": len(partner_rows) == EXPECTED_USE,
            "partner_alignment_14_of_14": len(alignment_rows) == EXPECTED_USE and not blocked,
            "paired_ready_14_of_14": len(paired_rows) == EXPECTED_USE,
        },
    }
    write_json(staging / "validation_report.json", validation)

    if OUT.exists():
        existing = OUT / "rl_export_summary.json"
        if existing.is_file() and json.loads(existing.read_text(encoding="utf-8")).get("manual_authority_sha256") != EXPECTED_REVIEW_SHA256:
            raise SystemExit("refusing to overwrite RL export from a different manual authority")
        shutil.rmtree(OUT)
    staging.rename(OUT)

    # Rebind the two published-manifest references that temporarily pointed at
    # the staging directory, then recompute the dependent manifest hash.
    published_source = OUT / "rl_export_input.tsv"
    published_partner = OUT / "partner_omnirt/rl_partner_omnirt_manifest.tsv"
    for row in partner_rows:
        row["source_rl_export_input"] = rel(published_source)
        row["source_rl_export_input_sha256"] = sha256(published_source)
    write_tsv(published_partner, partner_rows, PARTNER.PARTNER_FIELDS)
    write_json(OUT / "partner_omnirt/rl_partner_omnirt_manifest.json", partner_rows)
    published_partner_hash = sha256(published_partner)
    paired_rows = [
        PARTNER.paired_row(
            source,
            partner,
            manifest_ref=rel(published_partner),
            manifest_hash=published_partner_hash,
            repo=REPO,
        )
        for source, partner in zip(source_rows, partner_rows, strict=True)
    ]
    write_tsv(OUT / "paired_rl_export_input.tsv", paired_rows, SOURCE_FIELDS + PARTNER.PAIRED_EXTRA_FIELDS)
    write_json(OUT / "paired_rl_export_input.json", paired_rows)
    validation["artifact_sha256"] = {
        "rl_export_input.tsv": sha256(published_source),
        "partner_omnirt/rl_partner_omnirt_manifest.tsv": published_partner_hash,
        "paired_rl_export_input.tsv": sha256(OUT / "paired_rl_export_input.tsv"),
        "partner_resolution_audit.tsv": sha256(OUT / "partner_resolution_audit.tsv"),
        "manual_review_snapshot.tsv": sha256(OUT / "manual_review_snapshot.tsv"),
    }
    write_json(OUT / "validation_report.json", validation)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
