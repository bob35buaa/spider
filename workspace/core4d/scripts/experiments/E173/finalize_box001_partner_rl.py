#!/usr/bin/env python3
"""Finalize the E173 Box001 approved source+partner RL input set.

This is an S6-only packaging step. It reuses passing E173 Stage2b partner
motions where available and consumes the two isolated downstream partner
generations for cases that were not Stage2b-eligible in the original run.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[5]
RUN_ROOT = REPO / "workspace/core4d/results/E173"
RL_ROOT = RUN_ROOT / "s6_downstream/rl_export/box001_user_approved"
PARTNER_ROOT = RL_ROOT / "partner_omnirt"
SOURCE_RL = RL_ROOT / "rl_export_input.tsv"
REVIEW_SNAPSHOT = RL_ROOT / "box001_manual_review_snapshot.tsv"
STAGE2B = {
    variant: RUN_ROOT
    / f"s3_retarget/{variant}/ref_fk/stage2b_manifest_{variant}_ref_fk.tsv"
    for variant in ("omnirt_v1", "omnirt_v2")
}
GENERATED = {
    "box001_20231003_2_039_p2": (
        PARTNER_ROOT
        / "generated_missing_039_v2/rl_partner_omnirt_manifest.tsv"
    ),
    "box001_20231003_2_041_p2": (
        PARTNER_ROOT
        / "generated_missing_041_v1/rl_partner_omnirt_manifest.tsv"
    ),
}
V1_FAILURE_LOG = PARTNER_ROOT / "generated_missing/v1_execution.log"

PERSON_PARTNER = {"person1": "person2", "person2": "person1"}
PERSON_SHORT = {"person1": "p1", "person2": "p2"}
PERSON_INDEX = {"person1": "0", "person2": "1"}
PHASE4_V2_PARAMS = {
    "enable_constraint_relaxation": True,
    "enable_foot_z_constraint": True,
    "foot_slide_penalty_weight": 1.0,
    "enable_contact_preservation": True,
    "object_penetration_tolerance_scale": 0.8,
    "replace_wrist_with_fingertip": False,
}
V1_PARAMS = {
    "enable_constraint_relaxation": False,
    "enable_foot_z_constraint": False,
    "foot_slide_penalty_weight": 0.0,
    "enable_contact_preservation": False,
    "object_penetration_tolerance_scale": 1.0,
    "replace_wrist_with_fingertip": False,
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
    "v1_failure_status",
    "v1_failure_evidence",
    "v1_failure_evidence_sha256",
    "rescue_of",
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


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def read_tsv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        return list(reader.fieldnames or []), list(reader)


def write_tsv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=fields, delimiter="\t", extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def local_path(value: str | Path) -> Path:
    path = Path(str(value))
    if not path.is_absolute():
        return REPO / path
    if path.exists():
        return path
    text = path.as_posix()
    if "/workspace/core4d/" in text:
        return REPO / "workspace/core4d" / text.split("/workspace/core4d/", 1)[1]
    return path


def require_file(value: str | Path, label: str) -> Path:
    path = local_path(value)
    if not path.is_file() or path.stat().st_size == 0:
        raise SystemExit(f"missing or empty {label}: {path}")
    return path


def path_ref(value: str | Path) -> str:
    path = local_path(value)
    resolved = path.resolve()
    run_resolved = RUN_ROOT.resolve()
    try:
        return (Path("workspace/core4d/results/E173") / resolved.relative_to(run_resolved)).as_posix()
    except ValueError:
        pass
    try:
        return resolved.relative_to(REPO.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def partner_case_id(source: dict[str, str]) -> tuple[str, str, str]:
    partner_person = PERSON_PARTNER.get(source.get("person", ""), "")
    if not partner_person:
        raise SystemExit(f"unsupported source person: {source.get('case_id')}")
    partner_case = (
        f"{source['object_key']}_{source['date']}_{source['seq']}_"
        f"{PERSON_SHORT[partner_person]}"
    )
    return partner_case, partner_person, PERSON_INDEX[partner_person]


def load_trim(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def generated_partner(
    source: dict[str, str],
    partner_case: str,
    partner_person: str,
    partner_person_idx: str,
) -> dict[str, Any]:
    manifest = require_file(GENERATED[partner_case], f"{partner_case} generated manifest")
    _, rows = read_tsv(manifest)
    selected = [row for row in rows if row.get("partner_case_id") == partner_case]
    if len(selected) != 1 or selected[0].get("partner_status") != "pass":
        raise SystemExit(f"generated partner is not uniquely passing: {partner_case}")
    row = selected[0]
    variant = row["retarget_variant_id"]
    generation_mode = (
        "generated_s6_partner_v2_rescue"
        if variant == "omnirt_v2"
        else "generated_s6_partner_v1"
    )
    return build_partner_row(
        source,
        partner_case,
        partner_person,
        partner_person_idx,
        row,
        manifest,
        generation_mode=generation_mode,
        variant=variant,
        params=PHASE4_V2_PARAMS if variant == "omnirt_v2" else V1_PARAMS,
        target_task=row.get("partner_target_task", ""),
        stage2b_status="pass",
    )


def stage2b_partner(
    source: dict[str, str],
    partner_case: str,
    partner_person: str,
    partner_person_idx: str,
    candidates: list[tuple[Path, dict[str, str]]],
) -> dict[str, Any]:
    passing = [
        (manifest, row)
        for manifest, row in candidates
        if row.get("stage2b_status") == "pass"
    ]
    passing.sort(
        key=lambda item: (
            0 if item[1].get("retarget_variant_id") == "omnirt_v1" else 1,
            item[1].get("updated_at", ""),
        )
    )
    if not passing:
        raise SystemExit(f"no passing Stage2b partner: {partner_case}")
    manifest, row = passing[0]
    variant = row["retarget_variant_id"]
    try:
        params: Any = json.loads(row.get("params_json", "") or "{}")
    except json.JSONDecodeError:
        params = row.get("params_json", "")
    return build_partner_row(
        source,
        partner_case,
        partner_person,
        partner_person_idx,
        row,
        manifest,
        generation_mode="reuse_e173_stage2b",
        variant=variant,
        params=params,
        target_task=row.get("target_task", ""),
        stage2b_status=row.get("stage2b_status", ""),
    )


def build_partner_row(
    source: dict[str, str],
    partner_case: str,
    partner_person: str,
    partner_person_idx: str,
    evidence: dict[str, str],
    provenance: Path,
    *,
    generation_mode: str,
    variant: str,
    params: Any,
    target_task: str,
    stage2b_status: str,
) -> dict[str, Any]:
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
    trim_path_text = evidence.get("trim_window_json", "")
    trim_path = (
        require_file(trim_path_text, f"{partner_case} trim_window_json")
        if trim_path_text
        else require_file(case_root / "trim_window.json", f"{partner_case} trim_window_json")
    )
    trim = load_trim(trim_path)
    is_rescue = generation_mode == "generated_s6_partner_v2_rescue"
    failure_log = require_file(V1_FAILURE_LOG, "039_p2 fresh v1 failure log") if is_rescue else None
    if is_rescue:
        failure_text = failure_log.read_text(encoding="utf-8", errors="replace")
        if "CVXPY solve failed: infeasible" not in failure_text:
            raise SystemExit("039_p2 v1 failure log does not prove CVXPY infeasible")

    row: dict[str, Any] = {
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
        "partner_status": "pass",
        "paired_rl_export_decision": "RL_EXPORT_READY",
        "partner_retarget_variant_id": variant,
        "partner_target_variant_id": evidence.get("target_variant_id", "ref_fk"),
        "generation_mode": generation_mode,
        "stage2b_status": stage2b_status,
        "failure_mode": "",
        "decision_notes": (
            "fresh omnirt_v1 CVXPY infeasible; isolated Phase4 omnirt_v2 rescue pass"
            if is_rescue
            else (
                "generated downstream-only because original E173 3cm raw-contact gate "
                "did not run this partner"
                if generation_mode == "generated_s6_partner_v1"
                else "reused passing E173 Stage2b partner artifact"
            )
        ),
        "partner_target_task": target_task,
        "partner_provenance_ref": path_ref(provenance),
        "partner_provenance_sha256": sha256(provenance),
        "v1_failure_status": "omniretarget_infeasible" if is_rescue else "",
        "v1_failure_evidence": path_ref(failure_log) if failure_log else "",
        "v1_failure_evidence_sha256": sha256(failure_log) if failure_log else "",
        "rescue_of": "omnirt_v1" if is_rescue else "",
        "partner_params_json": json.dumps(
            params, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        if not isinstance(params, str)
        else params,
        "holosoma_case_root": path_ref(case_root),
        "trim_window_json": path_ref(trim_path),
        "trim_window_json_sha256": sha256(trim_path),
        "trim_start": trim.get("trim_start", ""),
        "trim_end": trim.get("trim_end", ""),
        "trim_frames": trim.get("trim_frames", ""),
        "untrimmed_frames": trim.get("untrimmed_frames", ""),
        "trimmed_frames": trim.get("trimmed_frames", ""),
        "source_rl_export_input": path_ref(SOURCE_RL),
        "source_rl_export_input_sha256": sha256(SOURCE_RL),
        "schema_version": "core4d_data_construction_v3.0",
        "updated_at": now(),
    }
    for field, path in artifacts.items():
        row[field] = path_ref(path)
        row[f"{field}_sha256"] = sha256(path)
    return row


def validate_sources(source_rows: list[dict[str, str]]) -> set[str]:
    if len(source_rows) != 13 or len({row["case_id"] for row in source_rows}) != 13:
        raise SystemExit("source RL input must contain exactly 13 unique rows")
    invalid = [
        row["case_id"]
        for row in source_rows
        if row.get("object_key") != "box001"
        or row.get("rl_export_decision") != "RL_EXPORT_READY"
    ]
    if invalid:
        raise SystemExit(f"source RL input contains non-ready/non-box001 rows: {invalid}")

    _, reviews = read_tsv(REVIEW_SNAPSHOT)
    approved = {
        row["case_id"]
        for row in reviews
        if row.get("object_key") == "box001"
        and row.get("manual_use_decision") == "USE"
    }
    rejected = {
        row["case_id"]
        for row in reviews
        if row.get("object_key") == "box001"
        and row.get("manual_use_decision") == "DO_NOT_USE"
    }
    source_ids = {row["case_id"] for row in source_rows}
    if source_ids != approved or source_ids & rejected:
        raise SystemExit("source RL input does not exactly match approved manual reviews")
    return rejected


def main() -> int:
    _, source_rows = read_tsv(require_file(SOURCE_RL, "source RL input"))
    rejected = validate_sources(source_rows)

    stage2b_index: dict[str, list[tuple[Path, dict[str, str]]]] = {}
    for manifest in STAGE2B.values():
        _, rows = read_tsv(require_file(manifest, "Stage2b manifest"))
        for row in rows:
            stage2b_index.setdefault(row.get("case_id", ""), []).append((manifest, row))

    partner_rows: list[dict[str, Any]] = []
    for source in source_rows:
        partner_case, partner_person, partner_person_idx = partner_case_id(source)
        if partner_case in GENERATED:
            partner = generated_partner(
                source, partner_case, partner_person, partner_person_idx
            )
        else:
            partner = stage2b_partner(
                source,
                partner_case,
                partner_person,
                partner_person_idx,
                stage2b_index.get(partner_case, []),
            )
        partner_rows.append(partner)

    partner_ids = {row["partner_case_id"] for row in partner_rows}
    source_ids = {row["source_case_id"] for row in partner_rows}
    if len(partner_rows) != 13 or len(partner_ids) != 13 or len(source_ids) != 13:
        raise SystemExit("paired output must contain 13 unique source/partner rows")
    if source_ids & rejected:
        raise SystemExit("paired output contains a manually rejected source row")
    for source, partner in zip(source_rows, partner_rows, strict=True):
        expected_case, expected_person, expected_idx = partner_case_id(source)
        if (
            partner["partner_case_id"] != expected_case
            or partner["partner_person"] != expected_person
            or partner["partner_person_idx"] != expected_idx
            or partner["date"] != source["date"]
            or partner["seq"] != source["seq"]
            or partner["object_key"] != "box001"
            or partner["partner_status"] != "pass"
            or partner["paired_rl_export_decision"] != "RL_EXPORT_READY"
        ):
            raise SystemExit(f"invalid source/partner pairing: {source['case_id']}")

    manifest_tsv = PARTNER_ROOT / "rl_partner_omnirt_manifest.tsv"
    manifest_json = PARTNER_ROOT / "rl_partner_omnirt_manifest.json"
    write_tsv(manifest_tsv, partner_rows, PARTNER_FIELDS)
    write_json(manifest_json, partner_rows)

    source_fields, _ = read_tsv(SOURCE_RL)
    paired_extra_fields = [
        "manual_use_decision",
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
    paired_rows: list[dict[str, Any]] = []
    for source, partner in zip(source_rows, partner_rows, strict=True):
        paired = dict(source)
        paired.update(
            {
                "manual_use_decision": "USE",
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
                "partner_manifest_ref": path_ref(manifest_tsv),
                "partner_manifest_sha256": "",
                "trajectory_sha256": sha256(
                    require_file(source["trajectory"], f"{source['case_id']} trajectory")
                ),
                "scene_act_sha256": sha256(
                    require_file(source["scene_act"], f"{source['case_id']} scene_act")
                ),
                "contact_mask_sha256": sha256(
                    require_file(source["contact_mask"], f"{source['case_id']} contact_mask")
                ),
                "cem_result_sha256": sha256(
                    require_file(source["cem_result_npz"], f"{source['case_id']} CEM result")
                ),
            }
        )
        paired_rows.append(paired)

    # The manifest hash is stable before paired outputs are written.
    manifest_hash = sha256(manifest_tsv)
    for row in paired_rows:
        row["partner_manifest_sha256"] = manifest_hash
    paired_tsv = PARTNER_ROOT / "paired_rl_export_input.tsv"
    paired_json = PARTNER_ROOT / "paired_rl_export_input.json"
    write_tsv(paired_tsv, paired_rows, source_fields + paired_extra_fields)
    write_json(paired_json, paired_rows)

    generation_counts = Counter(row["generation_mode"] for row in partner_rows)
    variant_counts = Counter(
        row["partner_retarget_variant_id"] for row in partner_rows
    )
    audit = {
        "authority_review_snapshot": path_ref(REVIEW_SNAPSHOT),
        "authority_review_snapshot_sha256": sha256(REVIEW_SNAPSHOT),
        "source_rl_export_input": path_ref(SOURCE_RL),
        "source_rl_export_input_sha256": sha256(SOURCE_RL),
        "source_rows": len(source_rows),
        "partner_rows": len(partner_rows),
        "box001_only": all(row["object_key"] == "box001" for row in partner_rows),
        "approved_source_set_exact": True,
        "rejected_source_rows_in_output": 0,
        "same_sequence_opposite_person": True,
        "all_partner_artifacts_nonempty": True,
        "all_hashes_recomputed": True,
        "paired_rl_ready_rows": sum(
            row["paired_rl_export_decision"] == "RL_EXPORT_READY"
            for row in partner_rows
        ),
        "generation_mode_counts": dict(generation_counts),
        "partner_variant_counts": dict(variant_counts),
        "partner_manifest": path_ref(manifest_tsv),
        "partner_manifest_sha256": manifest_hash,
        "paired_rl_export_input": path_ref(paired_tsv),
        "paired_rl_export_input_sha256": sha256(paired_tsv),
        "created_at": now(),
    }
    write_json(PARTNER_ROOT / "paired_rl_export_audit.json", audit)
    write_json(PARTNER_ROOT / "rl_partner_omnirt_summary.json", audit)

    summary_md = f"""# E173 Box001 paired RL-ready export

- source rows: `13`
- partner rows: `13`
- paired `RL_EXPORT_READY`: `13`
- scope: `box001` only
- manual authority: `{path_ref(REVIEW_SNAPSHOT)}`
- reused E173 Stage2b partners: `{generation_counts['reuse_e173_stage2b']}`
- generated downstream v1 partners: `{generation_counts['generated_s6_partner_v1']}`
- generated downstream v2 rescues: `{generation_counts['generated_s6_partner_v2_rescue']}`
- partner variants: `{dict(variant_counts)}`

The two originally absent partner artifacts were not lost files. Their original
E173 registry rows were outside the 3cm-pass Stage2b production subset. The
`041_p2` partner passed a new v1 run. The `039_p2` partner reproduced a fresh
v1 `CVXPY infeasible` failure and then passed the isolated frozen Phase4 v2
rescue.

This package prepares paired RL inputs only. It does not claim RL training
success and does not rewrite E173 S1-S5 facts.
"""
    (PARTNER_ROOT / "rl_partner_omnirt_summary.md").write_text(
        summary_md, encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
