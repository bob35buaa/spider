#!/usr/bin/env python3
"""Export the E198 G1A2-arm CEM results as E173-aligned RL inputs (+ partner).

One parametrized wrapper for box001 / box024 / box004, fixed to the G1A2 arm
(object gravcomp + A2 hand-gate). Case selection:

  box001 / box024 -> manual_use_decision == USE in the E198 review workbook
  box004          -> four user-specified cases (not in the review workbook)

It assembles the G1A2 evidence from ``e198_arm_cache.tsv`` (scene_act = gravcomp
sidecar, cem_result = E198 *_G1A2.npz), reuses the source experiment's S5 handoff
row (box001/box024 -> E173, box004 -> E172), then calls the SAME shared S6 tools
E173 used (``record_downstream_evidence.py`` -> ``export_rl_inputs.py`` ->
``finalize_reused_partner_rl.py``) so the schemas match E173 exactly. Manual USE is
an operational allowlist; numeric/gate-health facts are preserved in the source
audit and never overwritten.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "E173"))  # reuse E173 helpers + field lists
from export_box001_user_approved_rl import (  # noqa: E402
    AUDIT_FIELDS,
    EVIDENCE_FIELDS,
    REVIEW_FIELDS,
    read_tsv,
    rel,
    require_file,
    sha256,
    write_json,
    write_tsv,
)

REPO = HERE.parents[5]
RUN_ROOT = REPO / "workspace/core4d/results/E198"
ARM_CACHE = RUN_ROOT / "s6_downstream/eval/full_factorial/e198_arm_cache.tsv"
REVIEW_TSV = RUN_ROOT / "s6_downstream/eval/full_factorial/user_manual_review_filled.tsv"
S6_SCRIPTS = REPO / "workspace/core4d/scripts/data_construction_v3/stages/s6_downstream"

ARM = "G1A2"
HAND_COLLISION = "rubber_hull"  # G1A2 uses the rubber-hull gravcomp sidecar scene
TARGET_VARIANT = "ref_fk"
READY = {"HANDOFF_READY", "HANDOFF_REVIEW_VISUAL_QC"}
VARIANTS = ("omnirt_v1", "omnirt_v2")

# Per-object source experiment owning the S5 handoff + Stage2b retarget artifacts.
SOURCE_EXP = {"box001": "E173", "box024": "E173", "box004": "E172"}
BOX004_EXPLICIT = [
    "box004_20231003_2_082_p1",
    "box004_20231003_2_082_p2",
    "box004_20231003_2_083_p1",
    "box004_20231003_2_083_p2",
]

EXTRA_HANDOFF_FIELDS = [
    "short_case_id",
    "source_exp_id",
    "spider_method_id",
    "source_exp",
    "source_variant",
    "source_metrics_ref",
    "source_review_ref",
]


def source_root(object_key: str) -> Path:
    return REPO / f"workspace/core4d/results/{SOURCE_EXP[object_key]}"


def handoff_path(object_key: str) -> Path:
    return source_root(object_key) / "s5_handoff/handoff_manifest.tsv"


def stage2b_manifest(object_key: str, variant: str) -> Path:
    return (
        source_root(object_key)
        / f"s3_retarget/{variant}/ref_fk/stage2b_manifest_{variant}_ref_fk.tsv"
    )


def existing_stage2b_manifests(object_key: str) -> list[Path]:
    """Source-experiment Stage2b manifests, plus E198 supplement for box001."""
    paths = [stage2b_manifest(object_key, v) for v in VARIANTS]
    if object_key == "box001":
        paths += [
            RUN_ROOT / f"s3_retarget/{v}/ref_fk/stage2b_manifest_{v}_ref_fk.tsv"
            for v in VARIANTS
        ]
    return [p for p in paths if p.is_file()]


def select_review(object_key: str) -> tuple[list[dict[str, str]], set[str], set[str]]:
    """Return (snapshot rows, approved USE set, rejected DO_NOT_USE set)."""
    if object_key == "box004":
        snapshot = [
            {
                "case_id": case_id,
                "object_key": "box004",
                "user_manual_review_status": "reviewed",
                "manual_use_decision": "USE",
                "manual_quality_label": "USER_SELECTED",
                "manual_failure_taxonomy": "",
                "manual_review_note": "not in review tsv; user-specified for E198 G1A2 RL export",
                "manual_reviewer": "xiayibo",
                "manual_reviewed_at": "",
                "authority_workbook": "user_specified",
                "authority_sheet": "",
            }
            for case_id in BOX004_EXPLICIT
        ]
        return snapshot, set(BOX004_EXPLICIT), set()

    _, review_rows = read_tsv(REVIEW_TSV)
    scoped = [r for r in review_rows if str(r.get("case_id", "")).startswith(f"{object_key}_")]
    if not scoped:
        raise SystemExit(f"no {object_key} rows in {rel(REVIEW_TSV)}")
    snapshot = [
        {
            "case_id": r["case_id"],
            "object_key": object_key,
            "user_manual_review_status": r.get("user_manual_review_status", ""),
            "manual_use_decision": r.get("manual_use_decision", ""),
            "manual_quality_label": r.get("manual_quality_label", ""),
            "manual_failure_taxonomy": r.get("manual_failure_taxonomy", ""),
            "manual_review_note": r.get("manual_review_note", ""),
            "manual_reviewer": r.get("manual_reviewer", ""),
            "manual_reviewed_at": r.get("manual_reviewed_at", ""),
            "authority_workbook": rel(REVIEW_TSV),
            "authority_sheet": "user_manual_review_filled",
        }
        for r in scoped
    ]
    approved = {r["case_id"] for r in snapshot if r["manual_use_decision"] == "USE"}
    rejected = {r["case_id"] for r in snapshot if r["manual_use_decision"] == "DO_NOT_USE"}
    return snapshot, approved, rejected


def load_g1a2() -> dict[str, dict[str, str]]:
    _, rows = read_tsv(ARM_CACHE)
    out: dict[str, dict[str, str]] = {}
    for r in rows:
        if r.get("arm") == ARM:
            out[r["case_id"]] = r
    return out


def match_handoff(handoff_rows: list[dict[str, str]], case_id: str, variant: str) -> dict[str, str]:
    matches = [
        r
        for r in handoff_rows
        if r.get("case_id") == case_id
        and r.get("retarget_variant_id") == variant
        and r.get("target_variant_id") == TARGET_VARIANT
        and r.get("handoff_decision") in READY
        and r.get("target_gate_status") == "pass"
        and r.get("visual_qc_status") == "pass"
    ]
    if len(matches) != 1:
        raise SystemExit(
            f"expected exactly one ready S5 handoff row for {case_id} ({variant}), got {len(matches)}"
        )
    return matches[0]


def export_object(object_key: str) -> dict[str, Any]:
    snapshot, approved, rejected = select_review(object_key)
    review_by_case = {r["case_id"]: r for r in snapshot}
    g1a2 = load_g1a2()
    handoff_fields, handoff_rows = read_tsv(handoff_path(object_key))

    rl_dir = RUN_ROOT / f"s6_downstream/rl_export/{object_key}_user_approved"
    evidence_dir = RUN_ROOT / f"s6_downstream/evidence/{object_key}_user_approved"
    rl_dir.mkdir(parents=True, exist_ok=True)
    evidence_dir.mkdir(parents=True, exist_ok=True)

    review_snapshot_path = rl_dir / f"{object_key}_manual_review_snapshot.tsv"
    write_tsv(review_snapshot_path, snapshot, REVIEW_FIELDS)

    scoped_handoff: list[dict[str, Any]] = []
    evidence_input: list[dict[str, Any]] = []
    source_audit: list[dict[str, Any]] = []
    for case_id in sorted(approved):
        if case_id not in g1a2:
            raise SystemExit(f"{case_id} has no G1A2 arm row in {rel(ARM_CACHE)}")
        g = g1a2[case_id]
        variant = g["retarget_variant_id"]
        method = g.get("method", "")
        base = match_handoff(handoff_rows, case_id, variant)
        review = review_by_case[case_id]

        scene_act = require_file(g["scene_xml"], f"{case_id} scene_act")
        trajectory = require_file(g["trajectory"], f"{case_id} trajectory")
        contact_mask = require_file(g["contact_mask"], f"{case_id} contact_mask")
        cem_result = require_file(g["result_npz"], f"{case_id} cem_result_npz")
        config_act = require_file(g["config_act"], f"{case_id} config_act")
        if g.get("trajectory_sha256") and sha256(trajectory) != g["trajectory_sha256"]:
            raise SystemExit(f"trajectory SHA mismatch: {case_id}")

        task_dir = trajectory.parents[1]  # .../<target_task>/0/traj -> <target_task>
        notes = (
            f"E198 G1A2 arm (gravcomp G1 + A2 hand-gate); manual review="
            f"{review['manual_use_decision']}/{review['manual_quality_label']}; "
            f"numeric_pass={g.get('numeric_release_pass', '')}; numeric_warnings="
            f"{g.get('numeric_failure_modes', '') or 'none'}; "
            "manual operational approval does not overwrite numeric or gate-health facts"
        )
        scoped_handoff.append(
            {
                **base,
                "short_case_id": case_id,
                "retarget_variant_id": variant,
                "target_variant_id": TARGET_VARIANT,
                "hand_collision_variant_id": HAND_COLLISION,
                "source_exp_id": "E198",
                "spider_method_id": method,
                "source_exp": "E198",
                "source_variant": g.get("variant", ""),
                "source_metrics_ref": rel(ARM_CACHE),
                "source_review_ref": rel(REVIEW_TSV),
                "target_scene": rel(task_dir / "scene.xml"),
                "trajectory": rel(trajectory),
                "scene_act": rel(scene_act),
                "contact_mask": rel(contact_mask),
                "stage2b_target_task": task_dir.name,
                "stage2b_manifest_ref": rel(stage2b_manifest(object_key, variant)),
                "raw_contact_threshold_label": "3cm",
                "notes": notes,
            }
        )
        evidence_input.append(
            {
                "case_id": case_id,
                "object_key": object_key,
                "object_name": base.get("object_name", ""),
                "date": base.get("date", ""),
                "seq": base.get("seq", ""),
                "person": base.get("person", ""),
                "person_idx": base.get("person_idx", ""),
                "retarget_variant_id": variant,
                "target_variant_id": TARGET_VARIANT,
                "hand_collision_variant_id": HAND_COLLISION,
                "source_exp_id": "E198",
                "spider_method_id": method,
                "cem_status": "pass",
                "rl_status": "not_run",
                "downstream_failure_mode": "",
                "downstream_notes": notes,
                "cem_run_id": g.get("variant", ""),
                "cem_result_npz": rel(cem_result),
                "cem_video": "",
                "cem_metrics_ref": rel(ARM_CACHE),
            }
        )
        source_audit.append(
            {
                "case_id": case_id,
                "object_key": object_key,
                "manual_use_decision": review["manual_use_decision"],
                "manual_quality_label": review["manual_quality_label"],
                "manual_failure_taxonomy": review["manual_failure_taxonomy"],
                "manual_review_note": review["manual_review_note"],
                "numeric_release_pass": g.get("numeric_release_pass", ""),
                "numeric_failure_modes": g.get("numeric_failure_modes", ""),
                "retarget_variant_id": variant,
                "target_variant_id": TARGET_VARIANT,
                "hand_collision_variant_id": HAND_COLLISION,
                "source_exp_id": "E198",
                "spider_method_id": method,
                "scene_act": rel(scene_act),
                "scene_act_sha256": sha256(scene_act),
                "trajectory": rel(trajectory),
                "trajectory_sha256": sha256(trajectory),
                "contact_mask": rel(contact_mask),
                "contact_mask_sha256": sha256(contact_mask),
                "cem_result_npz": rel(cem_result),
                "cem_result_sha256": sha256(cem_result),
                "cem_video": "",
                "cem_video_sha256": "",
                "config_act": rel(config_act),
                "config_act_sha256": sha256(config_act),
                "metrics_ref": rel(ARM_CACHE),
                "authority_ref": rel(REVIEW_TSV),
                "authority_sha256": sha256(REVIEW_TSV),
            }
        )

    scoped_handoff_path = rl_dir / f"{object_key}_user_approved_handoff_manifest.tsv"
    evidence_input_path = evidence_dir / "downstream_evidence_input.tsv"
    source_audit_path = rl_dir / f"{object_key}_user_approved_source_rows.tsv"
    write_tsv(
        scoped_handoff_path,
        scoped_handoff,
        [*handoff_fields, *[f for f in EXTRA_HANDOFF_FIELDS if f not in handoff_fields]],
    )
    write_tsv(evidence_input_path, evidence_input, EVIDENCE_FIELDS)
    write_tsv(source_audit_path, source_audit, AUDIT_FIELDS)

    subprocess.run(
        [
            sys.executable,
            str(S6_SCRIPTS / "record_downstream_evidence.py"),
            "--handoff-manifest-tsv", str(scoped_handoff_path),
            "--evidence-tsv", str(evidence_input_path),
            "--out-dir", str(evidence_dir),
            "--evidence-root", str(RUN_ROOT),
            "--source-ref", f"E198_{object_key}_g1a2_user_review",
        ],
        cwd=REPO,
        check=True,
    )
    evidence_manifest = evidence_dir / "downstream_evidence_manifest.tsv"
    subprocess.run(
        [
            sys.executable,
            str(S6_SCRIPTS / "export_rl_inputs.py"),
            "--handoff-manifest-tsv", str(scoped_handoff_path),
            "--cem-evidence-tsv", str(evidence_manifest),
            "--out-dir", str(rl_dir),
            "--spider-repo", str(REPO),
        ],
        cwd=REPO,
        check=True,
    )

    _, rl_rows = read_tsv(rl_dir / "rl_export_input.tsv")
    rl_cases = {r["case_id"] for r in rl_rows}
    if rl_cases != approved:
        raise SystemExit(f"{object_key} RL export set != approved: {rl_cases ^ approved}")
    if any(r["object_key"] != object_key for r in rl_rows):
        raise SystemExit(f"non-{object_key} row entered RL export")
    if any(r["rl_export_decision"] != "RL_EXPORT_READY" for r in rl_rows):
        bad = [r["case_id"] for r in rl_rows if r["rl_export_decision"] != "RL_EXPORT_READY"]
        raise SystemExit(f"not all {object_key} rows RL_EXPORT_READY: {bad}")
    if rl_cases & rejected:
        raise SystemExit(f"DO_NOT_USE row entered RL export: {sorted(rl_cases & rejected)}")
    for r in rl_rows:
        for field in ("scene_act", "trajectory", "contact_mask", "cem_result_npz"):
            require_file(r[field], f"{r['case_id']} RL {field}")

    # Partner (reuse passing Stage2b of the opposite person); hard gate on completeness.
    partner_dir = rl_dir / "partner_omnirt"
    partner_cmd = [
        sys.executable,
        str(S6_SCRIPTS / "finalize_reused_partner_rl.py"),
        "--repo", str(REPO),
        "--experiment-id", "E198",
        "--object-key", object_key,
        "--rl-export-input-tsv", str(rl_dir / "rl_export_input.tsv"),
        "--out-dir", str(partner_dir),
        "--expected-rows", str(len(approved)),
        "--variant-preference", "omnirt_v1",
        "--variant-preference", "omnirt_v2",
    ]
    for manifest in existing_stage2b_manifests(object_key):
        partner_cmd += ["--stage2b-manifest-tsv", str(manifest)]
    if object_key != "box004":
        partner_cmd += ["--manual-review-snapshot", str(review_snapshot_path)]
    subprocess.run(partner_cmd, cwd=REPO, check=True)

    _, paired_rows = read_tsv(partner_dir / "paired_rl_export_input.tsv")
    if len(paired_rows) != len(approved):
        raise SystemExit(f"{object_key} paired rows != approved ({len(paired_rows)} vs {len(approved)})")
    if any(r.get("pair_status") != "PAIR_COMPLETE" for r in paired_rows):
        raise SystemExit(f"{object_key} has incomplete partner pairs")
    if any(r.get("paired_rl_export_decision") != "RL_EXPORT_READY" for r in paired_rows):
        raise SystemExit(f"{object_key} paired rows not all RL_EXPORT_READY")

    audit = {
        "experiment": "E198",
        "arm": ARM,
        "object_key": object_key,
        "status": "pass",
        "review_tsv": rel(REVIEW_TSV) if object_key != "box004" else "user_specified",
        "approved_rows": len(approved),
        "rejected_rows": len(rejected),
        "rl_ready_rows": len(rl_rows),
        "paired_rows": len(paired_rows),
        "source_set_matches_approved": rl_cases == approved,
        "numeric_release_counts": dict(
            Counter(r["numeric_release_pass"] for r in source_audit)
        ),
        "retarget_variant_counts": dict(
            Counter(r["retarget_variant_id"] for r in source_audit)
        ),
        "source_rows_sha256": sha256(source_audit_path),
        "rl_export_input_sha256": sha256(rl_dir / "rl_export_input.tsv"),
    }
    write_json(rl_dir / f"{object_key}_rl_export_audit.json", audit)
    summary = [
        f"# E198 {object_key} user-approved RL export (G1A2 arm)",
        "",
        f"- arm: `{ARM}` (gravcomp G1 + A2 hand-gate)",
        f"- authority: `{audit['review_tsv']}`",
        f"- approved (USE): `{len(approved)}`  rejected: `{len(rejected)}`",
        f"- RL_EXPORT_READY: `{len(rl_rows)}/{len(approved)}`",
        f"- paired PAIR_COMPLETE: `{len(paired_rows)}/{len(approved)}`",
        f"- retarget variants: `{audit['retarget_variant_counts']}`",
        f"- numeric release among approved: `{audit['numeric_release_counts']}`",
        "",
        "说明：人工 USE 是 operational allowlist；numeric/gate-health 事实保留在 source audit，不被人工标签覆盖。",
        "",
    ]
    (rl_dir / f"{object_key}_rl_export_summary.md").write_text("\n".join(summary), encoding="utf-8")
    print(json.dumps({"object_key": object_key, **{k: audit[k] for k in ("approved_rows", "rl_ready_rows", "paired_rows")}, "status": "pass"}, ensure_ascii=False))
    return audit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--object-key", required=True, choices=["box001", "box024", "box004"])
    args = parser.parse_args()
    export_object(args.object_key)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
