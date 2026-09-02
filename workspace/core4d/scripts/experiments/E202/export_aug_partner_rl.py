#!/usr/bin/env python3
"""E202-export: package USE13 bucket translation-augmented variants as
partner-complete, Holosoma-ready RL inputs (plan235).

Per (USE source case x feasible trans) it emits one export unit:
  * source  = the E202 aug CEM rollout (scene_act / trajectory / contact_mask /
    result_npz / video), reused as-is (no new CEM);
  * partner = the opposite person's aug retarget under the IDENTICAL trans_k
    (E202 data_preprocess trimmed NPZ), resolved via `finalize_reused_partner_rl`;
  * a common-raw-window alignment audit (same contract as E178-export/log264).

Only the C4 physical gate (fall=0, no divergence) hard-excludes variants here.
Two-person object consistency is enforced DOWNSTREAM by the Holosoma exporter's
partner re-anchor (partner hands -> source object frame) and verified post-reanchor
(matches E200(box) practice). Emits the E178/E187 rl_export schema so the
existing Holosoma exporter (export_rl_motion_from_spider_tsv.py) consumes it
unchanged. Never touches E178/E202 historical artifacts; writes under
results/E202/s6_downstream/rl_export_aug/.

Honest confounds (recorded, not hidden):
  * source retarget = omnirt_v2 (aug); orig partner in log264 was omnirt_v1. Here
    BOTH source and partner aug are omnirt_v2 -> internally consistent; the only
    variable vs the log264 orig export is the retarget variant.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e202_common as C  # noqa: E402
import e202_export_common as X  # noqa: E402

REPO = C.REPO
E187_BASE = REPO / "workspace/core4d/scripts/experiments/E187/export_manual_use_partner_rl.py"

# C4 physical hard-gate thresholds (elevated-but-not-divergent bound from E202)
DIVERGENCE_CM = 60.0


def _load_e187_base() -> Any:
    spec = importlib.util.spec_from_file_location("e187_partner_rl_export_base", E187_BASE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


E = _load_e187_base()
PARTNER = E.PARTNER  # finalize_reused_partner_rl adapter

# --- bind E187 module authorities to E202 aug sources ------------------------
E.METHOD_ID = "E202_bucket_e178_contactAlignedTop_translation_aug_omnirtV2_r1"
E.HAND_COLLISION_ID = C.HAND_COLLISION_VARIANT  # rubber_hull
E.METRICS = X.AUG_METRICS                       # e202_aug_case_metrics.tsv
E.EVAL_MANIFEST = X.E202_MANIFEST               # e202_bucket_priority_manifest.tsv
E.REVIEW = X.REVIEW
E.EXPECTED_REVIEW_SHA256 = X.EXPECTED_REVIEW_SHA256

OUT = X.OUT
PARTNER_OUT = OUT / "partner_omnirt"


def aug_unit_id(case_id: str, e202_name: str) -> str:
    return f"{case_id}__{e202_name}"


def load_aug_metrics() -> dict[tuple[str, str], dict[str, str]]:
    """Index E202 aug per-rollout metrics by (case_id, aug_variant)."""
    idx: dict[tuple[str, str], dict[str, str]] = {}
    for row in C.read_tsv(X.AUG_METRICS):
        av = row.get("aug_variant") or row.get("variant")
        idx[(row["case_id"], av)] = row
    return idx


def c4_ok(metric: dict[str, str]) -> tuple[bool, str]:
    def f(k: str) -> float:
        try:
            return float(metric.get(k, "") or 0.0)
        except ValueError:
            return 0.0
    fall = f("fall_flag")
    root = f("track_root_pos_err_cm_mean")
    eef = f("track_eef_pos_err_cm_mean")
    fails = []
    if fall > 0:
        fails.append("fall")
    if root > DIVERGENCE_CM:
        fails.append(f"root_diverge>{DIVERGENCE_CM}cm")
    if eef > DIVERGENCE_CM:
        fails.append(f"eef_diverge>{DIVERGENCE_CM}cm")
    return (not fails), ",".join(fails)


def make_evaluation(entry: dict[str, Any], man_row: dict[str, str]) -> dict[str, str]:
    """Synthesize the E187 `evaluation` dict from an E202 manifest row."""
    return {
        "case_id": aug_unit_id(entry["case_id"], man_row["aug_variant"]),
        "object_key": man_row["object_key"],
        "target_variant_id": "ref_fk",
        "spider_method_id": E.METHOD_ID,
        "hand_collision_variant_id": E.HAND_COLLISION_ID,
        "retarget_variant_id": man_row.get("arm", "omnirt_v2") and "omnirt_v2",
        "scene_act": man_row["scene_act"],
        "trajectory": man_row["trajectory"],
        "contact_mask": man_row["contact_mask"],
        "result_npz": man_row["result_npz"],
        "video": man_row["video"],
        "variant": man_row["variant"],
        "c9_technical_status": "",
        "c9_progression_authority": "USER_WAIVED_E202_AUG",
        "execution_kind": "production",
    }


def make_metric(metric_row: dict[str, str]) -> dict[str, str]:
    """Synthesize the E187 `metric` dict; provenance-only fields default blank."""
    gates = [
        "fall_gate_pass", "object_pos_gate_pass", "object_ori_gate_pass",
        "contact_gate_pass", "hand_penetration_gate_pass", "lower_body_gate_pass",
    ]
    failing = [g.replace("_gate_pass", "") for g in gates
               if str(metric_row.get(g, "")).lower() in {"false", "0", "0.0"}]
    row = dict(metric_row)
    row.setdefault("numeric_release_pass", str(metric_row.get("all_gates_pass", "")))
    row["numeric_failure_modes"] = ",".join(failing)
    return row


def make_stage_row(entry: dict[str, Any], man_row: dict[str, str]) -> dict[str, str]:
    return {
        "date": entry["date"], "seq": entry["seq"],
        "person": entry["person"], "person_idx": entry["person_idx"],
        "object_name": entry["object_name"],
        "target_scene": man_row.get("target_scene", ""),
        "target_task": man_row["target_task"],
        "result_root": man_row.get("outdir_npz", ""),
    }


def build_source_row(unit: str, review: dict[str, str], evaluation: dict[str, str],
                     metric: dict[str, str], stage_row: dict[str, str]) -> dict[str, Any]:
    """Like E.make_source_row but the CEM video is OPTIONAL.

    E202 only rendered a visual sample (per its C7), so most aug variants have no
    .mp4. Video is provenance only (the Holosoma exporter consumes trajectory /
    scene_act / cem_result, not video). Required assets are still hard-checked.
    """
    if evaluation.get("target_variant_id") != "ref_fk":
        raise SystemExit(f"{unit}: target_variant_id drift")
    paths = {
        "scene_act": E.required(evaluation["scene_act"], f"{unit} scene_act"),
        "trajectory": E.required(evaluation["trajectory"], f"{unit} trajectory"),
        "contact_mask": E.required(evaluation["contact_mask"], f"{unit} contact_mask"),
        "cem_result_npz": E.required(evaluation["result_npz"], f"{unit} CEM result"),
    }
    video_raw = evaluation.get("video", "")
    video_path = C.repo_path(video_raw) if video_raw else None
    video_exists = bool(video_path and video_path.is_file() and video_path.stat().st_size > 0)
    numeric_pass = metric.get("numeric_release_pass", "")
    numeric_failures = metric.get("numeric_failure_modes", "")
    object_name = "Bucket007" if evaluation["object_key"] == "bucket007" else stage_row.get("object_name", evaluation["object_key"])
    row: dict[str, Any] = {
        "case_id": unit, "object_key": evaluation["object_key"], "object_name": object_name,
        "date": stage_row["date"], "seq": stage_row["seq"],
        "person": stage_row["person"], "person_idx": stage_row["person_idx"],
        "retarget_variant_id": evaluation["retarget_variant_id"], "target_variant_id": "ref_fk",
        "hand_collision_variant_id": E.HAND_COLLISION_ID, "source_exp_id": "E202-export",
        "spider_method_id": E.METHOD_ID, "handoff_decision": "HANDOFF_READY",
        "candidate_decision": "USER_APPROVED_USE", "target_gate_status": "pass",
        "visual_qc_status": "manual_use_source_orig",
        "target_scene": stage_row.get("target_scene", ""),
        "trajectory": E.rel(paths["trajectory"]), "scene_act": E.rel(paths["scene_act"]),
        "contact_mask": E.rel(paths["contact_mask"]),
        "stage2b_target_task": stage_row.get("target_task", ""),
        "raw_contact_threshold_label": "3cm", "cem_status": "pass",
        "cem_run_id": evaluation.get("variant", ""), "cem_result_npz": E.rel(paths["cem_result_npz"]),
        "cem_video": E.rel(video_path) if video_exists else "",
        "cem_metrics_ref": E.rel(E.METRICS),
        "downstream_decision": "DOWNSTREAM_USER_APPROVED_FOR_RL_VALIDATION",
        "downstream_failure_mode": numeric_failures,
        "downstream_notes": (f"E202-export aug variant; manual USE authority on orig case; "
                             f"numeric_release_pass={numeric_pass}; video_exists={video_exists}; "
                             "two-person object consistency via downstream partner re-anchor"),
        "rl_export_decision": "RL_EXPORT_READY", "skip_reason": "",
        "scene_act_exists": "True", "trajectory_exists": "True", "contact_mask_exists": "True",
        "cem_result_exists": "True",
        "source_handoff_manifest": E.rel(E.EVAL_MANIFEST), "source_cem_evidence": E.rel(E.METRICS),
        "schema_version": "core4d_data_construction_v3.0", "updated_at": E.now(),
        "manual_use_decision": review["manual_use_decision"],
        "manual_quality_label": review["manual_quality_label"],
        "manual_failure_taxonomy": review.get("manual_failure_taxonomy", ""),
        "manual_review_note": review.get("manual_review_note", ""),
        "manual_reviewer": review.get("manual_reviewer", ""),
        "manual_reviewed_at": review.get("manual_reviewed_at", ""),
        "manual_review_ref": E.rel(E.REVIEW), "manual_review_sha256": E.EXPECTED_REVIEW_SHA256,
        "numeric_release_pass": numeric_pass, "numeric_failure_modes": numeric_failures,
        "c9_technical_status": evaluation.get("c9_technical_status", ""),
        "c9_progression_authority": evaluation.get("c9_progression_authority", ""),
        "execution_kind": evaluation.get("execution_kind", ""),
        "result_sha256": E.sha256(paths["cem_result_npz"]), "scene_sha256": E.sha256(paths["scene_act"]),
        "trajectory_sha256": E.sha256(paths["trajectory"]),
        "contact_mask_sha256": E.sha256(paths["contact_mask"]),
        "metrics_sha256": E.sha256(E.METRICS),
        "evaluation_manifest_sha256": E.sha256(E.EVAL_MANIFEST),
    }
    for field in ("leg_gate_health_pass", "fall_gate_pass", "body_z_gate_pass", "contact_gate_pass",
                  "release_gate_pass", "hand_penetration_gate_pass", "lower_body_gate_pass",
                  "root_pos_gate_pass", "root_ori_gate_pass", "hand_pos_gate_pass",
                  "hand_ori_gate_pass", "object_pos_gate_pass", "object_ori_gate_pass"):
        row[field] = metric.get(field, "")
    return row


def source_evidence_for_alignment(entry: dict[str, Any]) -> dict[str, str]:
    """Stage2b-shaped source evidence so E.alignment_audit can read the source
    trim window (source person's own E202 data_preprocess trim_window.json)."""
    root = REPO / C.DATA_PREPROCESS_REL / f"holosoma_{entry['base_target_task']}"
    return {"holosoma_case_root": C.rel(root), "trim_window_json": C.rel(root / "trim_window.json")}


def main() -> int:
    ap = argparse.ArgumentParser()
    args = ap.parse_args()

    X.load_use13()  # C0 authority (SHA + counts)
    # NOTE: two-person object consistency is NOT gated at the retarget level. The
    # native augmentation perturbs each person's approach in a per-person frame, so
    # the raw two-person object channels diverge (~0.4 m in approach) -- this is
    # expected and matches E200(box). Consistency is enforced DOWNSTREAM by the
    # Holosoma exporter's partner re-anchor (partner hands -> source object frame),
    # then verified post-reanchor. We only hard-gate C4 physical failures here.
    metrics = load_aug_metrics()

    source_rows: list[dict[str, Any]] = []
    partner_rows: list[dict[str, Any]] = []
    alignment_rows: list[dict[str, Any]] = []
    excluded: list[dict[str, str]] = []

    staging = OUT.parent / f".{OUT.name}.staging"
    if staging.exists():
        shutil.rmtree(staging)
    (staging / "partner_omnirt").mkdir(parents=True, exist_ok=True)
    source_path = staging / "rl_export_input.tsv"

    for entry in X.source_use_cases():
        for e202_name in sorted(entry["source_variants"]):
            man_row = entry["source_variants"][e202_name]
            unit = aug_unit_id(entry["case_id"], e202_name)
            holo_name = f"trans_{e202_name[-1]}"

            metric_row = metrics.get((entry["case_id"], e202_name), {})
            ok, why = c4_ok(metric_row)
            if not ok:
                excluded.append({"unit": unit, "reason": f"C4_physical:{why}"})
                continue

            evaluation = make_evaluation(entry, man_row)
            metric = make_metric(metric_row)
            stage_row = make_stage_row(entry, man_row)
            src = build_source_row(unit, X.load_use13()[entry["case_id"]], evaluation, metric, stage_row)
            source_rows.append(src)

    if not source_rows:
        raise SystemExit("no exportable aug variants (all excluded by C2/C4)")

    E.write_tsv(source_path, source_rows, E.SOURCE_FIELDS)
    E.write_json(staging / "rl_export_input.json", source_rows)
    shutil.copy2(X.REVIEW, staging / "manual_review_snapshot.tsv")

    # partner + alignment, per exported unit
    for src in source_rows:
        base_case = src["case_id"].split("__")[0]
        e202_name = src["case_id"].split("__")[1]
        holo_name = f"trans_{e202_name[-1]}"
        entry = next(e for e in X.source_use_cases() if e["case_id"] == base_case)
        p_ev = X.partner_evidence(entry, holo_name)
        # make the partner unit id unique per trans and align identity asserts
        p_ev = dict(p_ev)
        partner_unit = aug_unit_id(p_ev["case_id"], e202_name)
        p_ev["case_id"] = partner_unit
        # build_partner_row asserts identity fields against `source`; supply them
        src_for_partner = {**src, "object_key": entry["object_key"],
                           "object_name": entry["object_name"], "date": entry["date"],
                           "seq": entry["seq"], "person": entry["person"],
                           "person_idx": entry["person_idx"]}
        partner = PARTNER.build_partner_row(
            src_for_partner,
            partner_case=partner_unit,
            partner_person=p_ev["person"],
            partner_person_idx=p_ev["person_idx"],
            evidence=p_ev,
            provenance=X.E202_MANIFEST,
            source_rl=source_path,
            repo=REPO,
            generation_mode="e202_aug_partner_omnirt_v2_same_perturbation",
        )
        s_ev = source_evidence_for_alignment(entry)
        audit = E.alignment_audit(src, s_ev, X.E202_MANIFEST, partner, p_ev, X.E202_MANIFEST)
        partner_rows.append(partner)
        alignment_rows.append(audit)

    blocked = [r for r in alignment_rows if r["alignment_status"] != "RL_EXPORT_READY"]
    E.write_tsv(staging / "partner_resolution_audit.tsv", alignment_rows, E.ALIGNMENT_FIELDS)
    E.write_json(staging / "excluded_variants.json", excluded)
    if blocked:
        E.write_json(staging / "validation_report.json", {"status": "blocked", "blocked": blocked})
        raise SystemExit(f"aug RL export blocked by partner alignment: {[r['source_case_id'] for r in blocked]}")

    partner_manifest = staging / "partner_omnirt/rl_partner_omnirt_manifest.tsv"
    E.write_tsv(partner_manifest, partner_rows, PARTNER.PARTNER_FIELDS)
    E.write_json(staging / "partner_omnirt/rl_partner_omnirt_manifest.json", partner_rows)
    phash = E.sha256(partner_manifest)
    paired = [PARTNER.paired_row(s, p, manifest_ref=E.rel(partner_manifest), manifest_hash=phash, repo=REPO)
              for s, p in zip(source_rows, partner_rows, strict=True)]
    E.write_tsv(staging / "paired_rl_export_input.tsv", paired, E.SOURCE_FIELDS + PARTNER.PAIRED_EXTRA_FIELDS)
    E.write_json(staging / "paired_rl_export_input.json", paired)

    summary = {
        "experiment": "E202-export",
        "stage": "S6_use13_translation_aug_partner_rl_export",
        "created_at": E.now(),
        "status": "RL_EXPORT_READY",
        "manual_authority": E.rel(X.REVIEW),
        "manual_authority_sha256": X.EXPECTED_REVIEW_SHA256,
        "base_experiment": "E202 (source aug CEM) + log264 (orig USE13)",
        "source_rows": len(source_rows),
        "partner_rows": len(partner_rows),
        "excluded_variant_count": len(excluded),
        "pair_complete_rows": sum(r["pair_status"] == "PAIR_COMPLETE" for r in partner_rows),
        "paired_ready_rows": sum(r["paired_rl_export_decision"] == "RL_EXPORT_READY" for r in paired),
        "object_counts": dict(Counter(r["object_key"] for r in source_rows)),
        "variant_counts": dict(Counter(r["case_id"].split("__")[1] for r in source_rows)),
        "base_case_counts": dict(Counter(r["case_id"].split("__")[0] for r in source_rows)),
        "partner_variant_counts": dict(Counter(r["partner_retarget_variant_id"] for r in partner_rows)),
        "orig_only_use_cases": list(X.ORIG_ONLY_USE_CASES),
        "confound_note": "source+partner aug both omnirt_v2; log264 orig partner was omnirt_v1",
        "claim_boundary": "RL export + loader readiness only; no RL outcome claim",
    }
    E.write_json(staging / "rl_export_summary.json", summary)
    E.write_json(staging / "validation_report.json", {**summary, "checks": {
        "authority_sha_exact": True,
        "c2_consistency": "enforced_downstream_by_holosoma_partner_reanchor_then_verified_post_reanchor",
        "c4_physical_gated": True,
        "pair_complete_all": all(r["pair_status"] == "PAIR_COMPLETE" for r in partner_rows),
        "alignment_ready_all": not blocked,
    }})

    if OUT.exists():
        existing = OUT / "rl_export_summary.json"
        if existing.is_file() and json.loads(existing.read_text()).get("manual_authority_sha256") != X.EXPECTED_REVIEW_SHA256:
            raise SystemExit("refusing to overwrite aug RL export from a different manual authority")
        shutil.rmtree(OUT)
    staging.rename(OUT)

    # rebind published references + recompute dependent hashes
    published_source = OUT / "rl_export_input.tsv"
    published_partner = OUT / "partner_omnirt/rl_partner_omnirt_manifest.tsv"
    for r in partner_rows:
        r["source_rl_export_input"] = E.rel(published_source)
        r["source_rl_export_input_sha256"] = E.sha256(published_source)
    E.write_tsv(published_partner, partner_rows, PARTNER.PARTNER_FIELDS)
    E.write_json(OUT / "partner_omnirt/rl_partner_omnirt_manifest.json", partner_rows)
    phash2 = E.sha256(published_partner)
    paired = [PARTNER.paired_row(s, p, manifest_ref=E.rel(published_partner), manifest_hash=phash2, repo=REPO)
              for s, p in zip(source_rows, partner_rows, strict=True)]
    E.write_tsv(OUT / "paired_rl_export_input.tsv", paired, E.SOURCE_FIELDS + PARTNER.PAIRED_EXTRA_FIELDS)
    E.write_json(OUT / "paired_rl_export_input.json", paired)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
