#!/usr/bin/env python3
"""Build per-object S5 handoff + S6 evidence input rows for the E190 noPRG export.

Sources noPRG CEM evidence for the 38 approved case_ids:
  - box001/box004/box024: E189 s6_downstream/manifests/cem_full_manifest.tsv
  - box023: E179 s6_downstream/manifests/cem_full_manifest.tsv
  - box021 (9 standard cases): E168 s6_downstream/cem/manifests/cem_production_manifest.tsv
  - box021 (2 bridge cases 029_p2/035_p1): E167's own already-built rl_export_input.tsv
    (these were approved via the pre-PRG E167/E167A method and never ran under PRG at all)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from e190_common import (  # noqa: E402
    BOX021_BRIDGE_CASE_IDS,
    E167_RL_EXPORT,
    HAND_COLLISION_VARIANT_ID,
    OBJECT_RECIPES,
    RL_DIR,
    RUN_ROOT,
    SPIDER_METHOD_ID,
    load_authority,
    read_tsv,
    require_file,
    write_tsv,
)

HANDOFF_FIELDS = [
    "case_id", "short_case_id", "object_key", "object_name", "date", "seq",
    "person", "person_idx", "retarget_variant_id", "target_variant_id",
    "hand_collision_variant_id", "source_exp_id", "spider_method_id",
    "handoff_decision", "candidate_decision", "target_gate_status",
    "visual_qc_status", "target_scene", "trajectory", "scene_act", "contact_mask",
    "stage2b_target_task", "stage2b_result_root", "stage2b_manifest_ref",
    "raw_contact_threshold_label", "source_exp", "source_variant",
    "source_metrics_ref", "notes",
]

EVIDENCE_FIELDS = [
    "case_id", "object_key", "object_name", "date", "seq", "person", "person_idx",
    "retarget_variant_id", "target_variant_id", "hand_collision_variant_id",
    "source_exp_id", "spider_method_id", "cem_status", "rl_status",
    "downstream_failure_mode", "downstream_notes", "cem_run_id",
    "cem_result_npz", "cem_video", "cem_metrics_ref",
]

# config_act is not part of the standard v3 handoff/evidence schema; tracked
# separately so audit_38case_noprg_rl.py can run the per-row PRG-negative check.
CONFIG_ACT_FIELDS = ["case_id", "config_act"]


def _matches(rows, case_id, retarget_variant_id, target_variant_id):
    return [
        row for row in rows
        if row.get("case_id") == case_id
        and row.get("retarget_variant_id") == retarget_variant_id
        and row.get("target_variant_id") == target_variant_id
        and row.get("handoff_decision") == "HANDOFF_READY"
        and row.get("target_gate_status") == "pass"
        and row.get("visual_qc_status") == "pass"
    ]


def _build_row(object_key, source_exp_id, case_id, base, cem, cem_note):
    variant = cem.get("retarget_variant_id", base.get("retarget_variant_id", ""))
    target_variant = cem.get("target_variant_id", base.get("target_variant_id", ""))
    hand_collision = cem.get("hand_collision_variant_id", HAND_COLLISION_VARIANT_ID)
    notes = f"E190 noPRG counterpart of PRG-approved case; source_exp_id={source_exp_id}; {cem_note}"
    handoff = {
        **{field: base.get(field, "") for field in HANDOFF_FIELDS},
        "case_id": case_id,
        "short_case_id": case_id,
        "object_key": object_key,
        "retarget_variant_id": variant,
        "target_variant_id": target_variant,
        "hand_collision_variant_id": hand_collision,
        "source_exp_id": source_exp_id,
        "spider_method_id": SPIDER_METHOD_ID,
        "source_exp": source_exp_id,
        "source_variant": cem.get("variant", ""),
        "source_metrics_ref": "",
        "target_scene": cem.get("target_scene", ""),
        "trajectory": cem.get("trajectory", ""),
        "scene_act": cem.get("scene_act", ""),
        "contact_mask": cem.get("contact_mask", ""),
        "notes": notes,
    }
    evidence = {
        "case_id": case_id,
        "object_key": object_key,
        "object_name": base.get("object_name", ""),
        "date": base.get("date", ""),
        "seq": base.get("seq", ""),
        "person": base.get("person", ""),
        "person_idx": base.get("person_idx", ""),
        "retarget_variant_id": variant,
        "target_variant_id": target_variant,
        "hand_collision_variant_id": hand_collision,
        "source_exp_id": source_exp_id,
        "spider_method_id": SPIDER_METHOD_ID,
        "cem_status": "pass",
        "rl_status": "not_run",
        "downstream_failure_mode": "",
        "downstream_notes": notes,
        "cem_run_id": cem.get("variant", ""),
        "cem_result_npz": cem.get("result_npz", ""),
        "cem_video": cem.get("video", ""),
        "cem_metrics_ref": "",
    }
    config_row = {"case_id": case_id, "config_act": cem.get("config_act", "")}
    return handoff, evidence, config_row


def rows_from_full_manifest(object_key, source_exp_id, handoff_path, cem_path, case_ids):
    handoff_rows = [row for row in read_tsv(handoff_path) if row.get("object_key") == object_key]
    cem_rows = {
        row["case_id"]: row
        for row in read_tsv(cem_path)
        if row.get("object_key") == object_key and row.get("case_id") in case_ids
    }
    handoff_out, evidence_out, config_out = [], [], []
    for case_id in sorted(case_ids):
        cem = cem_rows.get(case_id)
        if not cem:
            raise SystemExit(f"{object_key}: no noPRG CEM manifest row for {case_id} in {cem_path}")
        matches = _matches(handoff_rows, case_id, cem["retarget_variant_id"], cem["target_variant_id"])
        if len(matches) != 1:
            raise SystemExit(
                f"{object_key}: expected exactly one ready S5 handoff row for {case_id}, got {len(matches)}"
            )
        handoff, evidence, config_row = _build_row(
            object_key, source_exp_id, case_id, matches[0], cem, f"status={cem.get('status', '')}"
        )
        handoff_out.append(handoff)
        evidence_out.append(evidence)
        config_out.append(config_row)
    return handoff_out, evidence_out, config_out


def rows_from_production_manifest(object_key, source_exp_id, handoff_path, cem_path, case_ids):
    handoff_rows = [row for row in read_tsv(handoff_path) if row.get("object_key") == object_key]
    cem_rows = {
        row["case_id"]: row
        for row in read_tsv(cem_path)
        if row.get("case_id") in case_ids
    }
    handoff_out, evidence_out, config_out = [], [], []
    for case_id in sorted(case_ids):
        cem = cem_rows.get(case_id)
        if not cem:
            raise SystemExit(f"{object_key}: no noPRG CEM manifest row for {case_id} in {cem_path}")
        matches = [
            row for row in handoff_rows
            if row.get("case_id") == case_id
            and row.get("handoff_decision") == "HANDOFF_READY"
            and row.get("target_gate_status") == "pass"
            and row.get("visual_qc_status") == "pass"
        ]
        if len(matches) != 1:
            raise SystemExit(
                f"{object_key}: expected exactly one ready S5 handoff row for {case_id}, got {len(matches)}"
            )
        handoff, evidence, config_row = _build_row(
            object_key, source_exp_id, case_id, matches[0], cem, f"status={cem.get('status', '')}"
        )
        handoff_out.append(handoff)
        evidence_out.append(evidence)
        config_out.append(config_row)
    return handoff_out, evidence_out, config_out


def rows_from_e167_bridge(case_ids):
    e167_rows = {row["case_id"]: row for row in read_tsv(E167_RL_EXPORT)}
    handoff_out, evidence_out, config_out = [], [], []
    for case_id in sorted(case_ids):
        long_id = f"d003_{case_id}__E167_E167A"
        row = e167_rows.get(long_id)
        if not row:
            raise SystemExit(f"box021 bridge case not found in E167 rl_export_input.tsv: {long_id}")
        if row.get("rl_export_decision") != "RL_EXPORT_READY":
            raise SystemExit(f"box021 bridge case is not RL_EXPORT_READY in E167: {long_id}")
        notes = (
            "E190 bridge case: never run under PRG at all (approved via pre-PRG E167/E167A method "
            "as USER_APPROVED_BRIDGE_OVERRIDE, see workspace/core4d/exp_analysis_0726.md); the noPRG "
            "row is a relabeled copy of the same E167 evidence already used on the PRG side."
        )
        handoff_out.append({
            "case_id": case_id,
            "short_case_id": case_id,
            "object_key": "box021",
            "object_name": row.get("object_name", "Box021"),
            "date": row.get("date", ""),
            "seq": row.get("seq", ""),
            "person": row.get("person", ""),
            "person_idx": row.get("person_idx", ""),
            "retarget_variant_id": row.get("retarget_variant_id", ""),
            "target_variant_id": row.get("target_variant_id", ""),
            "hand_collision_variant_id": row.get("hand_collision_variant_id", HAND_COLLISION_VARIANT_ID),
            "source_exp_id": "E167",
            "spider_method_id": SPIDER_METHOD_ID,
            "handoff_decision": row.get("handoff_decision", ""),
            "candidate_decision": row.get("candidate_decision", ""),
            "target_gate_status": row.get("target_gate_status", ""),
            "visual_qc_status": row.get("visual_qc_status", ""),
            "target_scene": row.get("target_scene", ""),
            "trajectory": row.get("trajectory", ""),
            "scene_act": row.get("scene_act", ""),
            "contact_mask": row.get("contact_mask", ""),
            "stage2b_target_task": "",
            "stage2b_result_root": "",
            "stage2b_manifest_ref": "",
            "raw_contact_threshold_label": "3cm",
            "source_exp": "E167",
            "source_variant": row.get("cem_run_id", ""),
            "source_metrics_ref": row.get("cem_metrics_ref", ""),
            "notes": notes,
        })
        evidence_out.append({
            "case_id": case_id,
            "object_key": "box021",
            "object_name": row.get("object_name", "Box021"),
            "date": row.get("date", ""),
            "seq": row.get("seq", ""),
            "person": row.get("person", ""),
            "person_idx": row.get("person_idx", ""),
            "retarget_variant_id": row.get("retarget_variant_id", ""),
            "target_variant_id": row.get("target_variant_id", ""),
            "hand_collision_variant_id": row.get("hand_collision_variant_id", HAND_COLLISION_VARIANT_ID),
            "source_exp_id": "E167",
            "spider_method_id": SPIDER_METHOD_ID,
            "cem_status": row.get("cem_status", "pass"),
            "rl_status": "not_run",
            "downstream_failure_mode": "",
            "downstream_notes": notes,
            "cem_run_id": row.get("cem_run_id", ""),
            "cem_result_npz": row.get("cem_result_npz", ""),
            "cem_video": row.get("cem_video", ""),
            "cem_metrics_ref": row.get("cem_metrics_ref", ""),
        })
        short_tag = "029_p2" if "029_p2" in case_id else "035_p1"
        config_out.append({
            "case_id": case_id,
            "config_act": (
                f"workspace/core4d/results/E167/holosoma_zonly/cem/full/"
                f"E167_box021_{short_tag}_E167A_outdir_full/config_act.yaml"
            ),
        })
    return handoff_out, evidence_out, config_out


def main() -> int:
    authority = load_authority()
    for object_key, case_ids in authority.items():
        source_exp_id, handoff_path, cem_path, flavor = OBJECT_RECIPES[object_key]
        require_file(handoff_path, f"{object_key} handoff manifest")
        require_file(cem_path, f"{object_key} cem manifest")

        if object_key == "box021":
            standard_ids = set(case_ids) - BOX021_BRIDGE_CASE_IDS
            h1, e1, c1 = rows_from_production_manifest(
                object_key, source_exp_id, handoff_path, cem_path, standard_ids
            )
            h2, e2, c2 = rows_from_e167_bridge(BOX021_BRIDGE_CASE_IDS)
            handoff_rows, evidence_rows, config_rows = h1 + h2, e1 + e2, c1 + c2
        elif flavor == "full_manifest":
            handoff_rows, evidence_rows, config_rows = rows_from_full_manifest(
                object_key, source_exp_id, handoff_path, cem_path, set(case_ids)
            )
        else:
            raise SystemExit(f"unknown recipe flavor for {object_key}: {flavor}")

        if len(handoff_rows) != len(case_ids) or len(evidence_rows) != len(case_ids):
            raise SystemExit(f"{object_key}: row count mismatch, expected {len(case_ids)}")
        if {row["case_id"] for row in handoff_rows} != set(case_ids):
            raise SystemExit(f"{object_key}: handoff case_id set does not match authority")

        scope_dir = RL_DIR / f"{object_key}_noPRG_user_approved"
        evidence_dir = RUN_ROOT / f"s6_downstream/evidence/{object_key}_noPRG_user_approved"
        write_tsv(scope_dir / f"{object_key}_noprg_handoff_manifest.tsv", handoff_rows, HANDOFF_FIELDS)
        write_tsv(evidence_dir / "downstream_evidence_input.tsv", evidence_rows, EVIDENCE_FIELDS)
        write_tsv(scope_dir / f"{object_key}_noprg_config_act.tsv", config_rows, CONFIG_ACT_FIELDS)
        print(f"{object_key}: wrote {len(handoff_rows)} handoff + evidence rows -> {scope_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
