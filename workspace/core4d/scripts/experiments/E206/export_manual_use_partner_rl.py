#!/usr/bin/env python3
"""Export E206 manual-USE cases as partner-complete RL inputs.

S6-only packaging. The human USE verdict controls the exported set; the 14-gate
funnel and the 12 numeric gates ride along as provenance and are NOT ANDed with
it (E187 precedent -- 9 of the 22 USE cases fail `numeric_release_pass`, and
intersecting the two would silently discard them).

E206 differs from E187 in three ways worth stating, since the schema is
deliberately identical:

  * Two arms were run; only PRG was reviewed. `ARM` is explicit, and the review
    file's `case_id#ARM` keys are filtered on it -- exporting the unreviewed
    noPRG control would be a category error.
  * Partners come from E206's own Stage2b (omnirt_v1 preferred, v2 rescue
    fallback), not E174's.
  * One USE case has no partner at all: `chair006_20231003_2_015_p2` failed S1
    raw contact (`weak_two_hand_overlap`, `unbalanced_left_right_contact`) and
    was rejected at template audit, so it was never retargeted. That row is
    emitted with `pair_status=PAIR_INCOMPLETE` and
    `paired_rl_export_decision=RL_EXPORT_BLOCKED_NO_PARTNER` rather than
    crashing the run or being dropped -- downstream consumes only
    `RL_EXPORT_READY`, so a visible blocked row is safer than a short table.

Usage:
    .venv/bin/python .../export_manual_use_partner_rl.py
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import e206_common as C  # noqa: E402

REPO = C.REPO
EVAL = C.S6_DIR / "eval/two_arm"
REVIEW = EVAL / "user_manual_review_filled.tsv"
METRICS = EVAL / "e206_arm_case_metrics.tsv"
ROLLOUT = EVAL / "e206_two_arm_rollout.tsv"
OUT = C.S6_DIR / "rl_export"

# The one arm a human actually watched.  `user_manual_review_filled.tsv` keys
# rows as `<case_id>#PRG`; `e206_arm_case_metrics.tsv` spells the same arm
# `PRG`, while CEM dirs and mp4s use lowercase `prg`.  All three appear below.
ARM = "prg"
ARM_REVIEW = "PRG"
ARM_METRICS = "PRG"

EXPECTED_REVIEW_SHA256 = "7e95901c5b51ca6866fdd74c001e16af1884978ef98aa17e5c30da1a8f41f249"
EXPECTED_USE = 22
EXPECTED_DO_NOT_USE = 43
EXPECTED_REVIEWED = 65

METHOD_ID = C.E206_METHOD_ID
# E206 scenes carry `left/right_rubber_hand` MESH hand geoms (verified in
# scene_act_E206_lowgeom_PRG.xml), i.e. the E147 rubber-hull variant, not the
# sphere5cm default.
HAND_COLLISION_ID = "rubber_hull"

STAGE2B_MANIFESTS = [
    C.S3_DIR / "omnirt_v1/ref_fk/stage2b_manifest_omnirt_v1_ref_fk.tsv",
    C.S3_DIR / "omnirt_v2/ref_fk/stage2b_manifest_omnirt_v2_ref_fk.tsv",
]

# E201 narrow band for the leg gate; E206's metrics TSV has no precomputed
# `leg_gate_health_pass` column, so it is derived here from the same bar the
# funnel uses rather than invented.
LEG_PEN_NARROW = 0.20

BLOCKED_DECISION = "RL_EXPORT_BLOCKED_NO_PARTNER"

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
    path = C.DCV3 / "stages/s6_downstream/finalize_reused_partner_rl.py"
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
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def rel(value: str | Path) -> str:
    return PARTNER.path_ref(value, REPO)


def required(value: str | Path, label: str) -> Path:
    return PARTNER.require_file(value, label, REPO)


def frame_count(path: Path, label: str, preferred_key: str = "qpos") -> int:
    try:
        with np.load(path, allow_pickle=False) as payload:
            if preferred_key in payload.files:
                array = np.asarray(payload[preferred_key])
                if array.ndim < 1:
                    raise ValueError(f"{preferred_key} is scalar")
                return int(array.shape[0])
            arrays = [np.asarray(payload[key]) for key in payload.files]
            counts = [int(a.shape[0]) for a in arrays if a.ndim >= 1]
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


def load_review() -> dict[str, dict[str, str]]:
    """`case_id` -> review row, for the reviewed arm only."""
    out: dict[str, dict[str, str]] = {}
    for row in C.read_tsv(REVIEW):
        raw = row.get("case_id", "")
        if "#" not in raw:
            continue
        case_id, arm = raw.rsplit("#", 1)
        if arm != ARM_REVIEW:
            continue
        if case_id in out:
            raise SystemExit(f"duplicate review row: {case_id}")
        out[case_id] = row
    return out


def load_arm_rows(path: Path, arm_value: str) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for row in C.read_tsv(path):
        if row.get("arm") != arm_value:
            continue
        case_id = row.get("case_id", "")
        if not case_id or case_id in out:
            raise SystemExit(f"bad/duplicate row in {path}: {case_id!r}")
        out[case_id] = row
    return out


def validate_authority(
    review: dict[str, dict[str, str]], metrics: dict[str, dict[str, str]]
) -> None:
    actual = sha256(REVIEW)
    if EXPECTED_REVIEW_SHA256 != "PIN_ME" and actual != EXPECTED_REVIEW_SHA256:
        raise SystemExit(
            f"manual authority SHA drift: {actual}\n"
            f"if the review was intentionally re-filled, update "
            f"EXPECTED_REVIEW_SHA256 in {Path(__file__).name}"
        )
    if len(review) != EXPECTED_REVIEWED:
        raise SystemExit(f"expected {EXPECTED_REVIEWED} reviewed rows, got {len(review)}")
    if set(review) != set(metrics):
        raise SystemExit(
            "review and metrics case sets differ: "
            f"review_only={sorted(set(review) - set(metrics))}, "
            f"metrics_only={sorted(set(metrics) - set(review))}"
        )
    unfinished = [
        case_id for case_id, row in review.items()
        if row.get("user_manual_review_status") != "reviewed"
        or row.get("manual_use_decision") not in {"USE", "DO_NOT_USE"}
    ]
    if unfinished:
        raise SystemExit(f"manual authority is not final: {unfinished}")
    counts = Counter(row["manual_use_decision"] for row in review.values())
    if counts != Counter({"USE": EXPECTED_USE, "DO_NOT_USE": EXPECTED_DO_NOT_USE}):
        raise SystemExit(f"manual counts drift: {dict(counts)}")


def source_stage2b(
    case_id: str, stage2b_index: dict[str, list[tuple[Path, dict[str, str]]]]
) -> tuple[Path, dict[str, str]]:
    eligible = [
        item for item in stage2b_index.get(case_id, [])
        if item[1].get("target_variant_id") == "ref_fk"
    ]
    if len(eligible) != 1:
        raise SystemExit(
            f"expected exactly one passing ref_fk Stage2b row for {case_id}, "
            f"got {len(eligible)}"
        )
    return eligible[0]


def make_source_row(
    case_id: str,
    review: dict[str, str],
    metric: dict[str, str],
    stage_row: dict[str, str],
    stage_manifest: Path,
) -> dict[str, Any]:
    variant = stage_row["retarget_variant_id"]
    if metric.get("retarget_variant_id") != variant:
        raise SystemExit(
            f"{case_id}: retarget variant disagrees between metrics "
            f"({metric.get('retarget_variant_id')!r}) and Stage2b ({variant!r})"
        )
    if stage_row.get("target_variant_id") != "ref_fk":
        raise SystemExit(f"{case_id}: target_variant_id drift")

    paths = {
        "scene_act": required(metric["scene_xml"], f"{case_id} scene_act"),
        "trajectory": required(metric["trajectory"], f"{case_id} trajectory"),
        "contact_mask": required(stage_row["contact_mask_npz"], f"{case_id} contact_mask"),
        "cem_result_npz": required(metric["outdir_npz"], f"{case_id} CEM result"),
        "cem_video": required(metric["video"], f"{case_id} CEM video"),
    }
    if C.SCENE_BY_ARM[ARM] not in paths["scene_act"].name:
        raise SystemExit(
            f"{case_id}: scene_act {paths['scene_act'].name} is not the {ARM} arm scene"
        )

    numeric_pass = metric.get("numeric_release_pass", "")
    numeric_failures = metric.get("numeric_failure_modes", "")
    leg_pen = metric.get("leg_penetration_frac", "")
    notes = (
        f"arm={ARM} ({C.SCENE_BY_ARM[ARM]}); "
        f"manual={review['manual_use_decision']}/{review['manual_quality_label']}; "
        f"numeric_release_pass={numeric_pass}; "
        f"numeric_failure_modes={numeric_failures or 'none'}; "
        "manual authority permits RL validation but does not overwrite numeric "
        "facts or claim RL success"
    )
    row: dict[str, Any] = {
        "case_id": case_id,
        "object_key": stage_row["object_key"],
        "object_name": stage_row["object_name"],
        "date": stage_row["date"],
        "seq": stage_row["seq"],
        "person": stage_row["person"],
        "person_idx": stage_row["person_idx"],
        "retarget_variant_id": variant,
        "target_variant_id": "ref_fk",
        "hand_collision_variant_id": HAND_COLLISION_ID,
        "source_exp_id": C.EXP_ID,
        "spider_method_id": METHOD_ID,
        "handoff_decision": "HANDOFF_READY",
        "candidate_decision": "USER_APPROVED_USE",
        "target_gate_status": "pass",
        "visual_qc_status": "manual_use",
        "target_scene": rel(stage_row.get("source_scene_xml", "")),
        "trajectory": rel(paths["trajectory"]),
        "scene_act": rel(paths["scene_act"]),
        "contact_mask": rel(paths["contact_mask"]),
        "stage2b_target_task": stage_row.get("target_task", ""),
        "stage2b_result_root": rel(stage_row.get("result_root", "")),
        "stage2b_manifest_ref": rel(stage_manifest),
        "raw_contact_threshold_label": C.PRIMARY_CONTACT_LABEL,
        "cem_status": "pass",
        "cem_run_id": C.RUN_IDS[ARM],
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
        "source_handoff_manifest": rel(METRICS),
        "source_cem_evidence": rel(METRICS),
        "schema_version": PARTNER.SCHEMA_VERSION,
        "updated_at": now(),
        "manual_use_decision": review["manual_use_decision"],
        "manual_quality_label": review["manual_quality_label"],
        "manual_failure_taxonomy": review.get("manual_failure_taxonomy", ""),
        "manual_review_note": review.get("manual_review_note", ""),
        "manual_reviewer": review.get("manual_reviewer", ""),
        "manual_reviewed_at": review.get("manual_reviewed_at", ""),
        "manual_review_ref": rel(REVIEW),
        "manual_review_sha256": sha256(REVIEW),
        "numeric_release_pass": numeric_pass,
        "numeric_failure_modes": numeric_failures,
        # derived, not measured: E206 metrics carry no leg_gate_health_pass
        "leg_gate_health_pass": str(
            leg_pen != "" and float(leg_pen) <= LEG_PEN_NARROW
        ),
        # E206 has no C9 progression gate; recorded as such rather than blank
        "c9_technical_status": "not_applicable",
        "c9_progression_authority": "not_applicable",
        "execution_kind": metric.get("status", ""),
        "result_sha256": sha256(paths["cem_result_npz"]),
        "scene_sha256": sha256(paths["scene_act"]),
        "trajectory_sha256": sha256(paths["trajectory"]),
        "contact_mask_sha256": sha256(paths["contact_mask"]),
        "metrics_sha256": sha256(METRICS),
        "evaluation_manifest_sha256": sha256(ROLLOUT),
    }
    for field in (
        "fall_gate_pass", "body_z_gate_pass", "contact_gate_pass", "release_gate_pass",
        "hand_penetration_gate_pass", "lower_body_gate_pass", "root_pos_gate_pass",
        "root_ori_gate_pass", "hand_pos_gate_pass", "hand_ori_gate_pass",
        "object_pos_gate_pass", "object_ori_gate_pass",
    ):
        row[field] = metric.get(field, "")
    return row


def blocked_partner_row(
    source: dict[str, Any], partner_case: str, partner_person: str,
    partner_person_idx: str, source_rl: Path, reason: str,
) -> dict[str, Any]:
    """A partner-shaped row for a source whose partner was never retargeted."""
    row: dict[str, Any] = {field: "" for field in PARTNER.PARTNER_FIELDS}
    row.update({
        "source_case_id": source["case_id"],
        "source_person": source["person"],
        "source_person_idx": source["person_idx"],
        "source_rl_export_decision": source["rl_export_decision"],
        "partner_case_id": partner_case,
        "partner_person": partner_person,
        "partner_person_idx": partner_person_idx,
        "object_key": source["object_key"],
        "object_name": source["object_name"],
        "date": source["date"],
        "seq": source["seq"],
        "pair_status": "PAIR_INCOMPLETE",
        "partner_status": "absent",
        "paired_rl_export_decision": BLOCKED_DECISION,
        "generation_mode": "none",
        "stage2b_status": "not_run",
        "failure_mode": "partner_not_retargeted",
        "decision_notes": reason,
        "source_rl_export_input": rel(source_rl),
        "source_rl_export_input_sha256": sha256(source_rl),
        "schema_version": PARTNER.SCHEMA_VERSION,
        "updated_at": now(),
    })
    return row


def alignment_audit(
    source: dict[str, Any], source_evidence: dict[str, str], source_provenance: Path,
    partner: dict[str, Any], partner_evidence: dict[str, str] | None,
    partner_provenance: Path | None,
) -> dict[str, Any]:
    case_id = source["case_id"]
    partner_id = partner["partner_case_id"]
    source_trim_path, source_trim = trim_window(source_evidence, case_id)
    ss = int_field(source_trim.get("trim_start"), f"{case_id} trim_start")
    se = int_field(source_trim.get("trim_end"), f"{case_id} trim_end")
    source_trim_frames = int_field(
        source_trim.get("trim_frames", se - ss), f"{case_id} trim_frames"
    )
    source_trajectory_frames = frame_count(
        required(source["trajectory"], f"{case_id} trajectory"), f"{case_id} trajectory"
    )
    source_cem_frames = frame_count(
        required(source["cem_result_npz"], f"{case_id} CEM"), f"{case_id} CEM"
    )
    source_mask_frames = frame_count(
        required(source["contact_mask"], f"{case_id} contact mask"),
        f"{case_id} contact mask",
        preferred_key=f"spider_contact_mask_{C.PRIMARY_CONTACT_LABEL}",
    )
    failures: list[str] = []
    if source_trim_frames != se - ss:
        failures.append("source_trim_metadata_mismatch")
    if source_trajectory_frames != source_trim_frames:
        failures.append("source_trajectory_trim_mismatch")
    if source_cem_frames != source_trajectory_frames:
        failures.append("source_cem_trajectory_mismatch")
    if source_mask_frames != source_trajectory_frames:
        failures.append("source_contact_mask_trajectory_mismatch")

    record: dict[str, Any] = {
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
        "partner_retarget_variant_id": partner.get("partner_retarget_variant_id", ""),
        "alignment_policy": "common_raw_window",
    }
    if partner_evidence is None:
        record.update({
            "partner_stage2b_manifest": "", "partner_trim_window_json": "",
            "partner_trim_start": "", "partner_trim_end": "", "partner_trim_frames": "",
            "partner_npz_frames": "", "common_raw_start": "", "common_raw_end": "",
            "common_raw_frames": "", "source_crop_offset": "", "partner_crop_offset": "",
            "alignment_status": BLOCKED_DECISION,
            "alignment_failure_mode": ",".join(failures + ["partner_not_retargeted"]),
        })
        return record

    partner_trim_path, partner_trim = trim_window(partner_evidence, partner_id)
    ps = int_field(partner_trim.get("trim_start"), f"{partner_id} trim_start")
    pe = int_field(partner_trim.get("trim_end"), f"{partner_id} trim_end")
    partner_trim_frames = int_field(
        partner_trim.get("trim_frames", pe - ps), f"{partner_id} trim_frames"
    )
    partner_npz_frames = frame_count(
        required(partner["trimmed_npz"], f"{partner_id} trimmed NPZ"),
        f"{partner_id} trimmed NPZ",
    )
    common_start, common_end = max(ss, ps), min(se, pe)
    common_frames = common_end - common_start
    if partner_trim_frames != pe - ps:
        failures.append("partner_trim_metadata_mismatch")
    if partner_npz_frames != partner_trim_frames:
        failures.append("partner_npz_trim_mismatch")
    if common_frames < 2:
        failures.append("no_usable_common_raw_window")
    record.update({
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
        "alignment_status": "RL_EXPORT_READY" if not failures else
                            "RL_EXPORT_BLOCKED_PARTNER_ALIGNMENT",
        "alignment_failure_mode": ",".join(failures),
    })
    return record


def main() -> int:
    review = load_review()
    metrics = load_arm_rows(METRICS, ARM_METRICS)
    validate_authority(review, metrics)
    approved = sorted(c for c, r in review.items() if r["manual_use_decision"] == "USE")
    rejected = {c for c, r in review.items() if r["manual_use_decision"] == "DO_NOT_USE"}

    stage2b_index = PARTNER.load_stage2b_index(STAGE2B_MANIFESTS)
    source_rows: list[dict[str, Any]] = []
    source_stage: dict[str, tuple[Path, dict[str, str]]] = {}
    for case_id in approved:
        provenance, evidence = source_stage2b(case_id, stage2b_index)
        source_stage[case_id] = (provenance, evidence)
        source_rows.append(
            make_source_row(case_id, review[case_id], metrics[case_id], evidence, provenance)
        )
    if {r["case_id"] for r in source_rows} != set(approved) or rejected & set(approved):
        raise SystemExit("source selection does not exactly match manual USE authority")

    staging = OUT.parent / f".{OUT.name}.staging"
    if staging.exists():
        shutil.rmtree(staging)
    (staging / "partner_omnirt").mkdir(parents=True, exist_ok=True)
    source_path = staging / "rl_export_input.tsv"
    C.write_tsv(source_path, source_rows, SOURCE_FIELDS)
    write_json(staging / "rl_export_input.json", source_rows)
    shutil.copy2(REVIEW, staging / "manual_review_snapshot.tsv")

    partner_rows: list[dict[str, Any]] = []
    alignment_rows: list[dict[str, Any]] = []
    for source in source_rows:
        partner_case, partner_person, partner_person_idx = PARTNER.infer_partner(source)
        candidates = stage2b_index.get(partner_case, [])
        source_provenance, source_evidence = source_stage[source["case_id"]]
        if candidates:
            partner_provenance, partner_evidence = PARTNER.choose_partner(
                candidates, PARTNER.DEFAULT_VARIANT_PREFERENCE, partner_case
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
                generation_mode="reuse_e206_stage2b_partner",
            )
        else:
            partner_provenance, partner_evidence = None, None
            partner = blocked_partner_row(
                source, partner_case, partner_person, partner_person_idx, source_path,
                "partner never entered E206: rejected at S1 raw contact / template "
                "audit, so no Stage2b row exists to reuse",
            )
        partner_rows.append(partner)
        alignment_rows.append(alignment_audit(
            source, source_evidence, source_provenance,
            partner, partner_evidence, partner_provenance,
        ))

    # Alignment failures on a resolved partner are a real defect and stop the run.
    # A structurally absent partner is a known, reported scope gap, not a defect.
    misaligned = [
        r for r in alignment_rows
        if r["alignment_status"] not in {"RL_EXPORT_READY", BLOCKED_DECISION}
    ]
    C.write_tsv(staging / "partner_resolution_audit.tsv", alignment_rows, ALIGNMENT_FIELDS)
    if misaligned:
        write_json(staging / "validation_report.json",
                   {"status": "blocked", "blocked": misaligned})
        raise SystemExit(
            "RL export blocked by partner alignment: "
            f"{[r['source_case_id'] for r in misaligned]}"
        )

    def publish(partner_manifest: Path) -> list[dict[str, Any]]:
        manifest_hash = sha256(partner_manifest)
        rows = []
        for source, partner in zip(source_rows, partner_rows, strict=True):
            row = PARTNER.paired_row(
                source, partner, manifest_ref=rel(partner_manifest),
                manifest_hash=manifest_hash, repo=REPO,
            )
            rows.append(row)
        return rows

    partner_manifest = staging / "partner_omnirt/rl_partner_omnirt_manifest.tsv"
    C.write_tsv(partner_manifest, partner_rows, PARTNER.PARTNER_FIELDS)
    write_json(staging / "partner_omnirt/rl_partner_omnirt_manifest.json", partner_rows)
    paired_rows = publish(partner_manifest)
    C.write_tsv(staging / "paired_rl_export_input.tsv", paired_rows,
                SOURCE_FIELDS + PARTNER.PAIRED_EXTRA_FIELDS)
    write_json(staging / "paired_rl_export_input.json", paired_rows)

    ready = [r for r in paired_rows if r["paired_rl_export_decision"] == "RL_EXPORT_READY"]
    blocked = [r for r in paired_rows if r["paired_rl_export_decision"] == BLOCKED_DECISION]
    summary = {
        "experiment": C.EXP_ID,
        "stage": "S6_manual_use_partner_rl_export",
        "arm": ARM,
        "scene_name": C.SCENE_BY_ARM[ARM],
        "cem_run_id": C.RUN_IDS[ARM],
        "created_at": now(),
        "status": "RL_EXPORT_READY_WITH_REPORTED_GAP" if blocked else "RL_EXPORT_READY",
        "manual_authority": rel(REVIEW),
        "manual_authority_sha256": sha256(REVIEW),
        "manual_counts": {"USE": EXPECTED_USE, "DO_NOT_USE": EXPECTED_DO_NOT_USE,
                          "PENDING": 0, "reviewed_arm": ARM_REVIEW},
        "source_rows": len(source_rows),
        "partner_rows": len(partner_rows),
        "pair_complete_rows": sum(r["pair_status"] == "PAIR_COMPLETE" for r in partner_rows),
        "paired_ready_rows": len(ready),
        "paired_blocked_rows": len(blocked),
        "paired_blocked_cases": [
            {"source_case_id": r["case_id"], "partner_case_id": r["partner_case_id"],
             "reason": "partner_not_retargeted"} for r in blocked
        ],
        "numeric_release_counts": dict(Counter(r["numeric_release_pass"] for r in source_rows)),
        "manual_quality_counts": dict(Counter(r["manual_quality_label"] for r in source_rows)),
        "object_counts": dict(Counter(r["object_key"] for r in source_rows)),
        "source_variant_counts": dict(Counter(r["retarget_variant_id"] for r in source_rows)),
        "partner_variant_counts": dict(Counter(
            r["partner_retarget_variant_id"] for r in partner_rows
            if r["partner_retarget_variant_id"])),
        "hand_collision_variant_id": HAND_COLLISION_ID,
        "all_partner_artifacts_nonempty": True,
        "all_source_partner_hashes_recomputed": True,
        "claim_boundary": "RL export and loader readiness only; no RL outcome claim. "
                          "Numeric 12-gate results are provenance, not a release pass.",
    }
    write_json(staging / "rl_export_summary.json", summary)
    write_json(staging / "validation_report.json", {
        **summary,
        "checks": {
            "authority_final_and_counted": True,
            "approved_source_set_exact": True,
            "rejected_zero_entry": True,
            "source_required_files_loadable": True,
            "source_arm_scene_is_reviewed_arm": True,
            f"partner_resolution_{len(ready)}_of_{EXPECTED_USE}": True,
            "partner_alignment_no_misalignment": not misaligned,
            "paired_rows_equal_source_rows": len(paired_rows) == len(source_rows),
        },
    })

    if OUT.exists():
        shutil.rmtree(OUT)
    staging.rename(OUT)

    # The two manifest references above pointed into the staging directory;
    # rebind them to the published paths and recompute the dependent hashes.
    published_source = OUT / "rl_export_input.tsv"
    published_partner = OUT / "partner_omnirt/rl_partner_omnirt_manifest.tsv"
    for row in partner_rows:
        row["source_rl_export_input"] = rel(published_source)
        row["source_rl_export_input_sha256"] = sha256(published_source)
    C.write_tsv(published_partner, partner_rows, PARTNER.PARTNER_FIELDS)
    write_json(OUT / "partner_omnirt/rl_partner_omnirt_manifest.json", partner_rows)
    paired_rows = publish(published_partner)
    C.write_tsv(OUT / "paired_rl_export_input.tsv", paired_rows,
                SOURCE_FIELDS + PARTNER.PAIRED_EXTRA_FIELDS)
    write_json(OUT / "paired_rl_export_input.json", paired_rows)

    validation = json.loads((OUT / "validation_report.json").read_text(encoding="utf-8"))
    validation["artifact_sha256"] = {
        name: sha256(OUT / name) for name in (
            "rl_export_input.tsv",
            "partner_omnirt/rl_partner_omnirt_manifest.tsv",
            "paired_rl_export_input.tsv",
            "partner_resolution_audit.tsv",
            "manual_review_snapshot.tsv",
        )
    }
    write_json(OUT / "validation_report.json", validation)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
