#!/usr/bin/env python3
"""Export the user-reviewed E173 Box023 source+partner RL input set.

This is an S6-only packaging step.  It treats CLEAN and MINOR_ACCEPTABLE
manual reviews as the operational allowlist, preserves numeric/gate-health
facts, and binds every source row to the opposite-person OmniRetarget motion
from the same raw sequence.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from export_box001_user_approved_rl import (
    AUDIT_FIELDS,
    EVIDENCE_FIELDS,
    REVIEW_FIELDS,
    local_path,
    now,
    read_tsv,
    rel,
    require_file,
    sha256,
    unique,
    write_json,
    write_tsv,
)


REPO = Path(__file__).resolve().parents[5]
RUN_ROOT = REPO / "workspace/core4d/results/E173"
METRICS = RUN_ROOT / "s6_downstream/eval/full/e173_case_metrics.tsv"
CEM_MANIFEST = RUN_ROOT / "s6_downstream/manifests/cem_full_manifest.tsv"
HANDOFF = RUN_ROOT / "s5_handoff/handoff_manifest.tsv"
REGISTRY = RUN_ROOT / "registries/case_state_registry.tsv"
RL_DIR = RUN_ROOT / "s6_downstream/rl_export/box023_user_approved"
PARTNER_DIR = RL_DIR / "partner_omnirt"
EVIDENCE_DIR = RUN_ROOT / "s6_downstream/evidence/box023_user_approved"
S6_SCRIPTS = (
    REPO / "workspace/core4d/scripts/data_construction_v3/stages/s6_downstream"
)
REGISTRY_SCRIPT = (
    REPO
    / "workspace/core4d/scripts/data_construction_v3/state"
    / "update_case_state_registry.py"
)
STAGE2B_MANIFESTS = {
    variant: RUN_ROOT
    / f"s3_retarget/{variant}/ref_fk/stage2b_manifest_{variant}_ref_fk.tsv"
    for variant in ("omnirt_v1", "omnirt_v2")
}
AUTHORITY_SOURCE = "user_message_2026-07-26_box023_manual_review"

MANUAL_GROUPS = {
    "UNUSABLE": {
        "box023_20231011_019_p2",
        "box023_20231011_018_p2",
        "box023_20231011_018_p1",
        "box023_20231020_042_p1",
        "box023_20231020_040_p1",
        "box023_20231020_039_p2",
        "box023_20231008_045_p2",
        "box023_20231008_046_p2",
        "box023_20231020_041_p2",
    },
    "MINOR_ACCEPTABLE": {
        "box023_20231020_041_p1",
        "box023_20231008_045_p1",
        "box023_20231020_042_p2",
        "box023_20231008_046_p1",
        "box023_20231020_040_p2",
    },
    "CLEAN": {
        "box023_20231011_021_p2",
        "box023_20231011_021_p1",
    },
}

PARTNER_FIELDS = [
    "source_case_id",
    "source_person",
    "source_person_idx",
    "source_rl_export_decision",
    "partner_case_id",
    "partner_person",
    "partner_person_idx",
    "object_key",
    "object_name",
    "date",
    "seq",
    "pair_status",
    "partner_status",
    "paired_rl_export_decision",
    "partner_retarget_variant_id",
    "partner_target_variant_id",
    "generation_mode",
    "stage2b_status",
    "failure_mode",
    "decision_notes",
    "partner_target_task",
    "partner_provenance_ref",
    "partner_provenance_sha256",
    "partner_params_json",
    "holosoma_case_root",
    "converted_npz",
    "converted_npz_sha256",
    "omniretarget_output_npz",
    "omniretarget_output_npz_sha256",
    "retargeted_npz",
    "retargeted_npz_sha256",
    "trimmed_npz",
    "trimmed_npz_sha256",
    "trim_window_json",
    "trim_window_json_sha256",
    "trim_start",
    "trim_end",
    "trim_frames",
    "untrimmed_frames",
    "trimmed_frames",
    "source_rl_export_input",
    "source_rl_export_input_sha256",
    "schema_version",
    "updated_at",
]

PAIRED_EXTRA_FIELDS = [
    "manual_use_decision",
    "manual_quality_label",
    "pair_status",
    "paired_rl_export_decision",
    "partner_case_id",
    "partner_person",
    "partner_person_idx",
    "partner_status",
    "partner_retarget_variant_id",
    "partner_target_variant_id",
    "partner_generation_mode",
    "partner_trimmed_npz",
    "partner_trimmed_npz_sha256",
    "partner_omniretarget_output_npz",
    "partner_omniretarget_output_npz_sha256",
    "partner_trim_window_json",
    "partner_trim_window_json_sha256",
    "partner_manifest_ref",
    "partner_manifest_sha256",
    "trajectory_sha256",
    "scene_act_sha256",
    "contact_mask_sha256",
    "cem_result_sha256",
]

PERSON_PARTNER = {"person1": ("person2", "p2", "1"), "person2": ("person1", "p1", "0")}
S1_S5_FACT_FIELDS = [
    "raw_inventory_status",
    "raw_contact_3cm_status",
    "raw_contact_5cm_status",
    "template_status",
    "stage2b_status",
    "target_gate_status",
    "visual_qc_status",
]


def manual_sets() -> tuple[list[dict[str, Any]], set[str], set[str]]:
    all_cases = set().union(*MANUAL_GROUPS.values())
    if len(all_cases) != 16:
        raise SystemExit("Box023 manual authority must contain 16 unique cases")
    approved = MANUAL_GROUPS["CLEAN"] | MANUAL_GROUPS["MINOR_ACCEPTABLE"]
    rejected = MANUAL_GROUPS["UNUSABLE"]
    if len(approved) != 7 or len(rejected) != 9 or approved & rejected:
        raise SystemExit("Box023 manual authority drift: expected USE=7/DNU=9")

    snapshot: list[dict[str, Any]] = []
    for case_id in sorted(all_cases):
        quality = next(
            label for label, cases in MANUAL_GROUPS.items() if case_id in cases
        )
        use = "DO_NOT_USE" if quality == "UNUSABLE" else "USE"
        snapshot.append(
            {
                "case_id": case_id,
                "object_key": "box023",
                "user_manual_review_status": "reviewed",
                "manual_use_decision": use,
                "manual_quality_label": quality,
                "manual_failure_taxonomy": (
                    "manual_visual_unusable"
                    if quality == "UNUSABLE"
                    else (
                        "manual_visual_minor_acceptable"
                        if quality == "MINOR_ACCEPTABLE"
                        else ""
                    )
                ),
                "manual_review_note": (
                    "用户人工审查：不可用"
                    if quality == "UNUSABLE"
                    else (
                        "用户人工审查：勉强能用"
                        if quality == "MINOR_ACCEPTABLE"
                        else "用户人工审查：可以用"
                    )
                ),
                "manual_reviewer": "human_user",
                "manual_reviewed_at": "2026-07-26",
                "authority_workbook": AUTHORITY_SOURCE,
                "authority_sheet": "",
            }
        )
    return snapshot, approved, rejected


def opposite_case(source: dict[str, str]) -> tuple[str, str, str]:
    partner_person, partner_short, partner_idx = PERSON_PARTNER[source["person"]]
    return (
        f"{source['object_key']}_{source['date']}_{source['seq']}_{partner_short}",
        partner_person,
        partner_idx,
    )


def load_trim(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def choose_partner_stage2b(
    partner_case_id: str,
    stage2b_index: dict[str, list[tuple[Path, dict[str, str]]]],
) -> tuple[Path, dict[str, str]]:
    passing = [
        item
        for item in stage2b_index.get(partner_case_id, [])
        if item[1].get("stage2b_status") == "pass"
    ]
    passing.sort(
        key=lambda item: (
            0 if item[1].get("retarget_variant_id") == "omnirt_v1" else 1,
            item[1].get("updated_at", ""),
        )
    )
    if not passing:
        raise SystemExit(f"no passing Stage2b partner: {partner_case_id}")
    return passing[0]


def build_partner(
    source: dict[str, str],
    stage2b_index: dict[str, list[tuple[Path, dict[str, str]]]],
    source_rl: Path,
) -> dict[str, Any]:
    partner_case, partner_person, partner_idx = opposite_case(source)
    provenance, evidence = choose_partner_stage2b(partner_case, stage2b_index)
    variant = evidence["retarget_variant_id"]
    artifacts = {
        field: require_file(evidence[field], f"{partner_case} {field}")
        for field in (
            "converted_npz",
            "omniretarget_output_npz",
            "retargeted_npz",
            "trimmed_npz",
        )
    }
    case_root = local_path(evidence["holosoma_case_root"])
    trim_path = require_file(case_root / "trim_window.json", f"{partner_case} trim")
    trim = load_trim(trim_path)
    try:
        params: Any = json.loads(evidence.get("params_json", "") or "{}")
        params_json = json.dumps(
            params, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
    except json.JSONDecodeError:
        params_json = evidence.get("params_json", "")

    row: dict[str, Any] = {
        "source_case_id": source["case_id"],
        "source_person": source["person"],
        "source_person_idx": source["person_idx"],
        "source_rl_export_decision": source["rl_export_decision"],
        "partner_case_id": partner_case,
        "partner_person": partner_person,
        "partner_person_idx": partner_idx,
        "object_key": "box023",
        "object_name": source["object_name"],
        "date": source["date"],
        "seq": source["seq"],
        "pair_status": "PAIR_COMPLETE",
        "partner_status": "pass",
        "paired_rl_export_decision": "RL_EXPORT_READY",
        "partner_retarget_variant_id": variant,
        "partner_target_variant_id": evidence.get("target_variant_id", "ref_fk"),
        "generation_mode": "reuse_e173_stage2b_partner",
        "stage2b_status": evidence["stage2b_status"],
        "failure_mode": "",
        "decision_notes": (
            f"reused passing E173 Stage2b partner; selected {variant} by "
            "production-v1-then-v2 preference"
        ),
        "partner_target_task": evidence.get("target_task", ""),
        "partner_provenance_ref": rel(provenance),
        "partner_provenance_sha256": sha256(provenance),
        "partner_params_json": params_json,
        "holosoma_case_root": rel(case_root),
        "trim_window_json": rel(trim_path),
        "trim_window_json_sha256": sha256(trim_path),
        "trim_start": trim.get("trim_start", ""),
        "trim_end": trim.get("trim_end", ""),
        "trim_frames": trim.get("trim_frames", ""),
        "untrimmed_frames": trim.get("untrimmed_frames", ""),
        "trimmed_frames": trim.get("trimmed_frames", ""),
        "source_rl_export_input": rel(source_rl),
        "source_rl_export_input_sha256": sha256(source_rl),
        "schema_version": "core4d_data_construction_v3.0",
        "updated_at": now(),
    }
    for field, path in artifacts.items():
        row[field] = rel(path)
        row[f"{field}_sha256"] = sha256(path)
    return row


def update_registry(
    rl_rows: list[dict[str, str]],
    evidence_manifest: Path,
) -> None:
    registry_fields, registry_rows = read_tsv(REGISTRY)
    seed_rows: list[dict[str, Any]] = []
    base_by_case: dict[str, dict[str, str]] = {}
    for source in rl_rows:
        candidates = [
            row
            for row in registry_rows
            if row.get("case_id") == source["case_id"]
            and row.get("retarget_variant_id") == source["retarget_variant_id"]
            and row.get("target_variant_id") == source["target_variant_id"]
            and row.get("hand_collision_variant_id", "sphere5cm") == "sphere5cm"
        ]
        if len(candidates) != 1:
            raise SystemExit(
                f"expected one sphere5cm registry base for {source['case_id']}, "
                f"got {len(candidates)}"
            )
        base = dict(candidates[0])
        base_by_case[source["case_id"]] = dict(base)
        base.update(
            {
                "hand_collision_variant_id": source["hand_collision_variant_id"],
                "cem_status": "not_run",
                "rl_status": "not_run",
                "downstream_decision": "",
                "downstream_failure_mode": "",
                "downstream_evidence_root": "",
                "cem_run_id": "",
                "cem_result_npz": "",
                "cem_video": "",
                "cem_metrics_ref": "",
                "source_type": "v3_run",
                "source_ref": "E173_box023_rubber_hull_S5_base",
                "updated_at": now(),
            }
        )
        seed_rows.append(base)

    seed_path = EVIDENCE_DIR / "registry_rubber_hull_base.tsv"
    write_tsv(seed_path, seed_rows, registry_fields)
    subprocess.run(
        [
            sys.executable,
            str(REGISTRY_SCRIPT),
            "--registry-dir",
            str(REGISTRY.parent),
            "--input-tsv",
            str(seed_path),
            "--evidence-root",
            str(RUN_ROOT),
            "--source-ref",
            "E173_box023_rubber_hull_S5_base",
        ],
        cwd=REPO,
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(REGISTRY_SCRIPT),
            "--registry-dir",
            str(REGISTRY.parent),
            "--from-downstream-evidence-tsv",
            str(evidence_manifest),
            "--evidence-root",
            str(EVIDENCE_DIR),
            "--source-ref",
            "E173_box023_user_review_20260726",
        ],
        cwd=REPO,
        check=True,
    )

    _, updated_rows = read_tsv(REGISTRY)
    updated = {
        row["case_id"]: row
        for row in updated_rows
        if row.get("case_id") in base_by_case
        and row.get("retarget_variant_id")
        == next(
            source["retarget_variant_id"]
            for source in rl_rows
            if source["case_id"] == row["case_id"]
        )
        and row.get("target_variant_id") == "ref_fk"
        and row.get("hand_collision_variant_id") == "rubber_hull"
    }
    if len(updated) != len(rl_rows):
        raise SystemExit("registry did not receive all Box023 rubber_hull S6 rows")
    for case_id, row in updated.items():
        if row.get("cem_status") != "pass" or row.get("rl_status") != "not_run":
            raise SystemExit(f"invalid S6 registry status: {case_id}")
        for field in S1_S5_FACT_FIELDS:
            if row.get(field) != base_by_case[case_id].get(field):
                raise SystemExit(f"S6 registry update changed {field}: {case_id}")


def main() -> int:
    snapshot, approved, rejected = manual_sets()
    review_by_case = unique(
        [{key: str(value) for key, value in row.items()} for row in snapshot],
        "Box023 user authority",
    )
    _, metrics_rows = read_tsv(METRICS)
    metric_by_case = unique(
        [row for row in metrics_rows if row.get("object_key") == "box023"],
        "E173 Box023 metrics",
    )
    _, cem_rows = read_tsv(CEM_MANIFEST)
    cem_by_case = unique(
        [row for row in cem_rows if row.get("object_key") == "box023"],
        "E173 Box023 CEM manifest",
    )
    handoff_fields, handoff_rows = read_tsv(HANDOFF)
    stage2b_index: dict[str, list[tuple[Path, dict[str, str]]]] = {}
    stage2b_by_variant: dict[str, dict[str, dict[str, str]]] = {}
    for variant, manifest in STAGE2B_MANIFESTS.items():
        _, rows = read_tsv(manifest)
        stage2b_by_variant[variant] = unique(rows, f"E173 {variant} Stage2b")
        for row in rows:
            stage2b_index.setdefault(row.get("case_id", ""), []).append(
                (manifest, row)
            )

    authority_cases = approved | rejected
    if set(metric_by_case) != authority_cases or set(cem_by_case) != authority_cases:
        raise SystemExit("Box023 authority/metrics/CEM case sets do not match")

    RL_DIR.mkdir(parents=True, exist_ok=True)
    PARTNER_DIR.mkdir(parents=True, exist_ok=True)
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    review_snapshot = RL_DIR / "box023_manual_review_snapshot.tsv"
    write_tsv(review_snapshot, snapshot, REVIEW_FIELDS)

    scoped_handoff: list[dict[str, Any]] = []
    evidence_input: list[dict[str, Any]] = []
    source_audit: list[dict[str, Any]] = []
    for case_id in sorted(approved):
        review = review_by_case[case_id]
        metric = metric_by_case[case_id]
        cem = cem_by_case[case_id]
        if cem.get("status") not in {
            "completed",
            "pass",
            "success",
            "run_complete_pending_eval",
        }:
            raise SystemExit(f"CEM is not complete for approved source: {case_id}")
        variant = cem["retarget_variant_id"]
        stage2b = stage2b_by_variant[variant].get(case_id)
        if not stage2b or stage2b.get("stage2b_status") != "pass":
            raise SystemExit(f"selected Stage2b source row is not pass: {case_id}")
        matches = [
            row
            for row in handoff_rows
            if row.get("case_id") == case_id
            and row.get("retarget_variant_id") == variant
            and row.get("target_variant_id") == cem["target_variant_id"]
            and row.get("handoff_decision") == "HANDOFF_READY"
            and row.get("target_gate_status") == "pass"
            and row.get("visual_qc_status") == "pass"
        ]
        if len(matches) != 1:
            raise SystemExit(f"expected one ready S5 handoff row for {case_id}")
        base = matches[0]
        paths = {
            "scene_act": require_file(cem["scene_act"], f"{case_id} scene_act"),
            "trajectory": require_file(metric["trajectory"], f"{case_id} trajectory"),
            "contact_mask": require_file(
                metric["contact_mask"], f"{case_id} contact_mask"
            ),
            "cem_result_npz": require_file(
                metric["result_npz"], f"{case_id} CEM result"
            ),
            "cem_video": require_file(metric["video"], f"{case_id} CEM video"),
            "config_act": require_file(metric["config_act"], f"{case_id} config"),
        }
        if (
            cem.get("effective_scene_sha256")
            and sha256(paths["scene_act"]) != cem["effective_scene_sha256"]
        ):
            raise SystemExit(f"scene_act SHA mismatch: {case_id}")
        if (
            cem.get("trajectory_sha256")
            and sha256(paths["trajectory"]) != cem["trajectory_sha256"]
        ):
            raise SystemExit(f"trajectory SHA mismatch: {case_id}")
        if (
            cem.get("contact_mask_sha256")
            and sha256(paths["contact_mask"]) != cem["contact_mask_sha256"]
        ):
            raise SystemExit(f"contact_mask SHA mismatch: {case_id}")

        notes = (
            f"E173 user review={review['manual_use_decision']}/"
            f"{review['manual_quality_label']}; numeric_pass="
            f"{metric.get('numeric_release_pass', '')}; numeric_warnings="
            f"{metric.get('numeric_failure_modes', '') or 'none'}; "
            "manual operational approval does not overwrite numeric or gate-health facts"
        )
        task_dir = paths["trajectory"].parents[1]
        scoped_handoff.append(
            {
                **base,
                "short_case_id": case_id,
                "hand_collision_variant_id": cem["hand_collision_variant_id"],
                "source_exp_id": "E173",
                "spider_method_id": cem["spider_method_id"],
                "source_exp": "E173",
                "source_variant": cem["variant"],
                "source_metrics_ref": rel(METRICS),
                "source_review_ref": rel(review_snapshot),
                "target_scene": rel(task_dir / "scene.xml"),
                "trajectory": rel(paths["trajectory"]),
                "scene_act": rel(paths["scene_act"]),
                "contact_mask": rel(paths["contact_mask"]),
                "stage2b_target_task": cem["target_task"],
                "stage2b_manifest_ref": rel(STAGE2B_MANIFESTS[variant]),
                "raw_contact_threshold_label": cem["contact_mask_label"],
                "notes": notes,
            }
        )
        evidence_input.append(
            {
                "case_id": case_id,
                "object_key": "box023",
                "object_name": base["object_name"],
                "date": base["date"],
                "seq": base["seq"],
                "person": base["person"],
                "person_idx": base["person_idx"],
                "retarget_variant_id": variant,
                "target_variant_id": cem["target_variant_id"],
                "hand_collision_variant_id": cem["hand_collision_variant_id"],
                "source_exp_id": "E173",
                "spider_method_id": cem["spider_method_id"],
                "cem_status": "pass",
                "rl_status": "not_run",
                "downstream_failure_mode": "",
                "downstream_notes": notes,
                "cem_run_id": cem["variant"],
                "cem_result_npz": rel(paths["cem_result_npz"]),
                "cem_video": rel(paths["cem_video"]),
                "cem_metrics_ref": rel(METRICS),
            }
        )
        source_audit.append(
            {
                "case_id": case_id,
                "object_key": "box023",
                "manual_use_decision": review["manual_use_decision"],
                "manual_quality_label": review["manual_quality_label"],
                "manual_failure_taxonomy": review["manual_failure_taxonomy"],
                "manual_review_note": review["manual_review_note"],
                "numeric_release_pass": metric["numeric_release_pass"],
                "numeric_failure_modes": metric["numeric_failure_modes"],
                "retarget_variant_id": variant,
                "target_variant_id": cem["target_variant_id"],
                "hand_collision_variant_id": cem["hand_collision_variant_id"],
                "source_exp_id": "E173",
                "spider_method_id": cem["spider_method_id"],
                **{field: rel(path) for field, path in paths.items()},
                "scene_act_sha256": sha256(paths["scene_act"]),
                "trajectory_sha256": sha256(paths["trajectory"]),
                "contact_mask_sha256": sha256(paths["contact_mask"]),
                "cem_result_sha256": sha256(paths["cem_result_npz"]),
                "cem_video_sha256": sha256(paths["cem_video"]),
                "config_act_sha256": sha256(paths["config_act"]),
                "metrics_ref": rel(METRICS),
                "authority_ref": rel(review_snapshot),
                "authority_sha256": sha256(review_snapshot),
            }
        )

    extra_handoff_fields = [
        "short_case_id",
        "source_exp_id",
        "spider_method_id",
        "source_exp",
        "source_variant",
        "source_metrics_ref",
        "source_review_ref",
    ]
    scoped_handoff_path = RL_DIR / "box023_user_approved_handoff_manifest.tsv"
    evidence_input_path = EVIDENCE_DIR / "downstream_evidence_input.tsv"
    source_audit_path = RL_DIR / "box023_user_approved_source_rows.tsv"
    write_tsv(
        scoped_handoff_path,
        scoped_handoff,
        [
            *handoff_fields,
            *[field for field in extra_handoff_fields if field not in handoff_fields],
        ],
    )
    write_tsv(evidence_input_path, evidence_input, EVIDENCE_FIELDS)
    write_tsv(source_audit_path, source_audit, AUDIT_FIELDS)

    subprocess.run(
        [
            sys.executable,
            str(S6_SCRIPTS / "record_downstream_evidence.py"),
            "--handoff-manifest-tsv",
            str(scoped_handoff_path),
            "--evidence-tsv",
            str(evidence_input_path),
            "--out-dir",
            str(EVIDENCE_DIR),
            "--evidence-root",
            str(RUN_ROOT),
            "--source-ref",
            "E173_box023_user_review_20260726",
        ],
        cwd=REPO,
        check=True,
    )
    evidence_manifest = EVIDENCE_DIR / "downstream_evidence_manifest.tsv"
    subprocess.run(
        [
            sys.executable,
            str(S6_SCRIPTS / "export_rl_inputs.py"),
            "--handoff-manifest-tsv",
            str(scoped_handoff_path),
            "--cem-evidence-tsv",
            str(evidence_manifest),
            "--out-dir",
            str(RL_DIR),
            "--spider-repo",
            str(REPO),
        ],
        cwd=REPO,
        check=True,
    )

    source_rl = RL_DIR / "rl_export_input.tsv"
    source_fields, rl_rows = read_tsv(source_rl)
    rl_cases = {row["case_id"] for row in rl_rows}
    if len(rl_rows) != 7 or rl_cases != approved or rl_cases & rejected:
        raise SystemExit("RL export set does not exactly match Box023 USE reviews")
    if any(
        row.get("object_key") != "box023"
        or row.get("rl_export_decision") != "RL_EXPORT_READY"
        for row in rl_rows
    ):
        raise SystemExit("Box023 source RL input contains an invalid row")
    for row in rl_rows:
        for field in ("scene_act", "trajectory", "contact_mask", "cem_result_npz"):
            require_file(row[field], f"{row['case_id']} RL {field}")

    update_registry(rl_rows, evidence_manifest)

    partner_rows = [
        build_partner(source, stage2b_index, source_rl) for source in rl_rows
    ]
    if len(partner_rows) != 7:
        raise SystemExit("Box023 partner output must contain seven rows")
    for source, partner in zip(rl_rows, partner_rows, strict=True):
        expected_case, expected_person, expected_idx = opposite_case(source)
        if (
            partner["partner_case_id"] != expected_case
            or partner["partner_person"] != expected_person
            or partner["partner_person_idx"] != expected_idx
            or partner["date"] != source["date"]
            or partner["seq"] != source["seq"]
            or partner["pair_status"] != "PAIR_COMPLETE"
            or partner["partner_status"] != "pass"
            or partner["paired_rl_export_decision"] != "RL_EXPORT_READY"
        ):
            raise SystemExit(f"invalid source/partner pairing: {source['case_id']}")

    partner_manifest = PARTNER_DIR / "rl_partner_omnirt_manifest.tsv"
    write_tsv(partner_manifest, partner_rows, PARTNER_FIELDS)
    write_json(PARTNER_DIR / "rl_partner_omnirt_manifest.json", partner_rows)
    partner_manifest_hash = sha256(partner_manifest)

    paired_rows: list[dict[str, Any]] = []
    for source, partner in zip(rl_rows, partner_rows, strict=True):
        review = review_by_case[source["case_id"]]
        paired = dict(source)
        paired.update(
            {
                "manual_use_decision": "USE",
                "manual_quality_label": review["manual_quality_label"],
                "pair_status": "PAIR_COMPLETE",
                "paired_rl_export_decision": "RL_EXPORT_READY",
                "partner_case_id": partner["partner_case_id"],
                "partner_person": partner["partner_person"],
                "partner_person_idx": partner["partner_person_idx"],
                "partner_status": partner["partner_status"],
                "partner_retarget_variant_id": partner[
                    "partner_retarget_variant_id"
                ],
                "partner_target_variant_id": partner["partner_target_variant_id"],
                "partner_generation_mode": partner["generation_mode"],
                "partner_trimmed_npz": partner["trimmed_npz"],
                "partner_trimmed_npz_sha256": partner["trimmed_npz_sha256"],
                "partner_omniretarget_output_npz": partner[
                    "omniretarget_output_npz"
                ],
                "partner_omniretarget_output_npz_sha256": partner[
                    "omniretarget_output_npz_sha256"
                ],
                "partner_trim_window_json": partner["trim_window_json"],
                "partner_trim_window_json_sha256": partner[
                    "trim_window_json_sha256"
                ],
                "partner_manifest_ref": rel(partner_manifest),
                "partner_manifest_sha256": partner_manifest_hash,
                "trajectory_sha256": sha256(
                    require_file(source["trajectory"], f"{source['case_id']} trajectory")
                ),
                "scene_act_sha256": sha256(
                    require_file(source["scene_act"], f"{source['case_id']} scene_act")
                ),
                "contact_mask_sha256": sha256(
                    require_file(
                        source["contact_mask"], f"{source['case_id']} contact mask"
                    )
                ),
                "cem_result_sha256": sha256(
                    require_file(
                        source["cem_result_npz"], f"{source['case_id']} CEM result"
                    )
                ),
            }
        )
        paired_rows.append(paired)

    paired_tsv = PARTNER_DIR / "paired_rl_export_input.tsv"
    write_tsv(paired_tsv, paired_rows, source_fields + PAIRED_EXTRA_FIELDS)
    write_json(PARTNER_DIR / "paired_rl_export_input.json", paired_rows)

    quality_counts = Counter(
        review_by_case[row["case_id"]]["manual_quality_label"] for row in rl_rows
    )
    numeric_counts = Counter(
        metric_by_case[row["case_id"]]["numeric_release_pass"] for row in rl_rows
    )
    source_variant_counts = Counter(row["retarget_variant_id"] for row in rl_rows)
    partner_variant_counts = Counter(
        row["partner_retarget_variant_id"] for row in partner_rows
    )
    audit = {
        "experiment": "E173",
        "scope": "Box023 only",
        "status": "pass",
        "authority_source": AUTHORITY_SOURCE,
        "authority_review_snapshot": rel(review_snapshot),
        "authority_review_snapshot_sha256": sha256(review_snapshot),
        "reviewed_rows": len(snapshot),
        "approved_rows": len(approved),
        "rejected_rows": len(rejected),
        "manual_quality_counts_approved": dict(quality_counts),
        "source_rows": len(rl_rows),
        "source_rl_ready_rows": sum(
            row["rl_export_decision"] == "RL_EXPORT_READY" for row in rl_rows
        ),
        "source_set_matches_approved": rl_cases == approved,
        "rejected_source_rows_in_output": len(rl_cases & rejected),
        "numeric_release_counts": dict(numeric_counts),
        "source_variant_counts": dict(source_variant_counts),
        "partner_rows": len(partner_rows),
        "paired_rl_ready_rows": sum(
            row["paired_rl_export_decision"] == "RL_EXPORT_READY"
            for row in partner_rows
        ),
        "same_sequence_opposite_person": True,
        "all_partner_artifacts_nonempty": True,
        "all_hashes_recomputed": True,
        "partner_variant_counts": dict(partner_variant_counts),
        "source_rl_export_input": rel(source_rl),
        "source_rl_export_input_sha256": sha256(source_rl),
        "partner_manifest": rel(partner_manifest),
        "partner_manifest_sha256": partner_manifest_hash,
        "paired_rl_export_input": rel(paired_tsv),
        "paired_rl_export_input_sha256": sha256(paired_tsv),
        "registry_updated_rows": len(rl_rows),
        "created_at": now(),
    }
    write_json(RL_DIR / "box023_rl_export_audit.json", audit)
    write_json(PARTNER_DIR / "paired_rl_export_audit.json", audit)
    write_json(PARTNER_DIR / "rl_partner_omnirt_summary.json", audit)

    source_summary = f"""# E173 Box023 user-approved RL export

- authority: `{AUTHORITY_SOURCE}`
- reviewed Box023 rows: `16`
- approved / rejected: `7 / 9`
- approved quality: `{dict(quality_counts)}`
- source `RL_EXPORT_READY`: `7/7`
- source variants: `{dict(source_variant_counts)}`
- numeric release among approved: `{dict(numeric_counts)}`

人工 `CLEAN` 和 `MINOR_ACCEPTABLE` 构成 operational allowlist；numeric/gate-health
事实保留在 source audit，不被人工标签覆盖。该导出不宣称 RL 已训练成功。
"""
    (RL_DIR / "box023_rl_export_summary.md").write_text(
        source_summary, encoding="utf-8"
    )
    partner_summary = f"""# E173 Box023 paired RL-ready export

- source rows: `7`
- partner rows: `7`
- paired `PAIR_COMPLETE + RL_EXPORT_READY`: `7/7`
- scope: `box023` only
- partner rule: same `(object, date, seq)`, opposite person
- partner generation: `reuse_e173_stage2b_partner=7`
- partner variants: `{dict(partner_variant_counts)}`

All partner artifacts were reused from passing E173 Stage2b rows; no new
OmniRetarget execution or rescue was needed. This S6 package does not rewrite
S1-S5 facts and does not claim RL training success.
"""
    (PARTNER_DIR / "rl_partner_omnirt_summary.md").write_text(
        partner_summary, encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
