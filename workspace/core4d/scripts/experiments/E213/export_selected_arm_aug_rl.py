#!/usr/bin/env python3
"""E213-export: package the user-curated selected-arm aug units as partner-paired,
dcv3 Holosoma-ready RL inputs (plan244).

Per xlsx-selected (case, variant) unit it emits one export unit:
  * source  = the E213 selected-arm aug CEM rollout (scene_act / trajectory /
    contact_mask / cem_result), reused as-is (no new CEM);
  * partner = the OPPOSITE person's kinematic aug retarget under the IDENTICAL
    variant (E213 partner_aug OR E208 source aug trimmed NPZ), resolved via the
    dcv3 partner adapter;
  * a common-raw-window alignment audit.

Two-person object consistency is enforced DOWNSTREAM by the Holosoma exporter's
partner re-anchor (partner hands -> source object frame), then verified
post-reanchor (E200/E202 practice). Only C4 physical failures (fall / divergence)
hard-exclude a unit here. Emits the dcv3 rl_export schema so the existing Holosoma
exporter consumes it unchanged. Writes only under results/E213/s6_downstream/export/.

Usage:
    .venv/bin/python workspace/core4d/scripts/experiments/E213/export_selected_arm_aug_rl.py --dry-run
    .venv/bin/python workspace/core4d/scripts/experiments/E213/export_selected_arm_aug_rl.py
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import e213_common as C  # noqa: E402
import e213_export_common as X  # noqa: E402

REPO = C.REPO
E206_EXPORTER = REPO / "workspace/core4d/scripts/experiments/E206/export_manual_use_partner_rl.py"

# C4 physical hard-gate (elevated-but-not-divergent bound; E202-export precedent).
DIVERGENCE_CM = 60.0


def _load_e206_exporter() -> Any:
    """Import E206's exporter as a LIBRARY (PARTNER adapter + alignment_audit +
    field lists + IO helpers). Its main() would re-validate E206's own review
    authority, so we import the module and call only the pieces we need (the
    E212 precedent). Import is side-effect-free (module level only sets paths)."""
    for d in ("E206", "E200", "E199"):
        p = str(REPO / "workspace/core4d/scripts/experiments" / d)
        if p not in sys.path:
            sys.path.insert(0, p)
    spec = importlib.util.spec_from_file_location("e213_e206_exporter", E206_EXPORTER)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


E = _load_e206_exporter()
PARTNER = E.PARTNER
SOURCE_FIELDS = E.SOURCE_FIELDS
ALIGNMENT_FIELDS = E.ALIGNMENT_FIELDS

METHOD_ID = "E213_selected_arm_object_aug_omnirt_r1"
HAND_COLLISION_ID = "rubber_hull"
EVAL_ROLLOUT = C.EVAL_DIR / "aug" / "e213_aug_rollout.tsv"

# Arm columns appended so the mixture is auditable from the TSV alone (E212).
EXTRA_FIELDS = [
    "aug_variant", "arm", "arm_experiment", "arm_gravcomp", "arm_scene_name",
    "arm_selection_reason", "selection_authority",
]

# The 6 gate verdicts the E213 evaluator carries (dcv3 SOURCE_FIELDS lists 13;
# the rest stay blank exactly as E202-export leaves them).
EVAL_GATE_FIELDS = [
    "fall_gate_pass", "object_pos_gate_pass", "object_ori_gate_pass",
    "contact_gate_pass", "hand_penetration_gate_pass", "lower_body_gate_pass",
]


# --- IO (repo-relative normalization; results/ is a /mnt symlink -- E212) -----
RESULTS_REAL = os.path.realpath(REPO / "workspace/core4d/results")
REPO_REAL = os.path.realpath(REPO)


def to_repo_rel(value: Any) -> Any:
    if not isinstance(value, str) or not value.startswith("/"):
        return value
    rp = os.path.realpath(value)
    if rp == RESULTS_REAL or rp.startswith(RESULTS_REAL + os.sep):
        return ("workspace/core4d/results/" + rp[len(RESULTS_REAL) + 1:]).rstrip("/")
    if rp == REPO_REAL or rp.startswith(REPO_REAL + os.sep):
        return rp[len(REPO_REAL) + 1:]
    return value


def norm(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{k: to_repo_rel(v) for k, v in row.items()} for row in rows]


def write_tsv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def unit_id(case_id: str, variant: str, arm: str) -> str:
    return f"{case_id}__aug_{variant}_{arm}"


def c4_ok(eval_row: dict[str, str]) -> tuple[bool, str]:
    def f(k: str) -> float:
        try:
            return float(eval_row.get(k, "") or 0.0)
        except ValueError:
            return 0.0
    fails = []
    if str(eval_row.get("fall_flag", "")).lower() in {"true", "1", "1.0"} or f("fall_flag") > 0:
        fails.append("fall")
    if f("track_root_pos_err_cm_mean") > DIVERGENCE_CM:
        fails.append(f"root_diverge>{DIVERGENCE_CM}cm")
    if f("track_eef_pos_err_cm_mean") > DIVERGENCE_CM:
        fails.append(f"eef_diverge>{DIVERGENCE_CM}cm")
    return (not fails), ",".join(fails)


def build_source_row(unit: str, u: dict[str, str], seed: dict[str, str],
                     man: dict[str, str], ev: dict[str, str]) -> dict[str, Any]:
    """One dcv3 source row for an aug unit (E212 mixed-arm + E202 aug fields)."""
    scene_act = man["selected_scene_act"]
    if u["arm_scene_name"] not in Path(scene_act).name:
        raise SystemExit(f"{unit}: scene_act {Path(scene_act).name!r} != arm scene {u['arm_scene_name']!r}")
    paths = {
        "scene_act": E.required(scene_act, f"{unit} scene_act"),
        "trajectory": E.required(man["trajectory"], f"{unit} trajectory"),
        "contact_mask": E.required(man["contact_mask"], f"{unit} contact_mask"),
        "cem_result_npz": E.required(man["outdir_npz"], f"{unit} cem_result"),
    }
    video_raw = man.get("video", "") or man.get("cem_video", "")
    video_path = C.repo_path(video_raw) if video_raw else None
    video_ok = bool(video_path and video_path.is_file() and video_path.stat().st_size > 0)
    numeric_pass = ev.get("numeric_release_pass", "")
    numeric_fail = ev.get("numeric_failure_modes", "")
    row: dict[str, Any] = {
        "case_id": unit, "object_key": seed["object_key"], "object_name": seed["object_name"],
        "date": seed["date"], "seq": seed["seq"], "person": seed["person"],
        "person_idx": seed["person_idx"],
        "retarget_variant_id": man.get("effective_retarget_variant", seed.get("retarget_variant_id", "")),
        "target_variant_id": X.TARGET_VARIANT,
        "hand_collision_variant_id": HAND_COLLISION_ID, "source_exp_id": "E213-export",
        "spider_method_id": METHOD_ID, "handoff_decision": "HANDOFF_READY",
        "candidate_decision": "USER_SELECTED_E213_AUG_XLSX", "target_gate_status": "pass",
        "visual_qc_status": "xlsx_selected",
        "target_scene": seed.get("target_scene", ""),
        "trajectory": E.rel(paths["trajectory"]), "scene_act": E.rel(paths["scene_act"]),
        "contact_mask": E.rel(paths["contact_mask"]),
        "stage2b_target_task": man["target_task"], "stage2b_result_root": "",
        "stage2b_manifest_ref": E.rel(X.SOURCE_MANIFEST),
        "raw_contact_threshold_label": "3cm", "cem_status": "pass", "cem_run_id": C.RUN_ID,
        "cem_result_npz": E.rel(paths["cem_result_npz"]),
        "cem_video": E.rel(video_path) if video_ok else "",
        "cem_metrics_ref": E.rel(EVAL_ROLLOUT),
        "downstream_decision": "DOWNSTREAM_USER_APPROVED_FOR_RL_VALIDATION",
        "downstream_failure_mode": numeric_fail,
        "downstream_notes": (
            f"E213-export selected-arm ({u['arm']}) aug unit; xlsx-selected; "
            f"numeric_release_pass={numeric_pass}; video_exists={video_ok}; "
            "two-person object consistency via downstream partner re-anchor"),
        "rl_export_decision": "RL_EXPORT_READY", "skip_reason": "",
        "scene_act_exists": "True", "trajectory_exists": "True",
        "contact_mask_exists": "True", "cem_result_exists": "True",
        "source_handoff_manifest": E.rel(X.SOURCE_MANIFEST), "source_cem_evidence": E.rel(EVAL_ROLLOUT),
        "schema_version": PARTNER.SCHEMA_VERSION, "updated_at": E.now(),
        # selection authority = xlsx (human curation); NOT a manual-review verdict
        "manual_use_decision": "USER_SELECTED_E213_AUG_XLSX", "manual_quality_label": "",
        "manual_failure_taxonomy": "", "manual_review_note": "",
        "manual_reviewer": "", "manual_reviewed_at": "",
        "manual_review_ref": E.rel(X.XLSX), "manual_review_sha256": X.EXPECTED_XLSX_SHA256,
        "numeric_release_pass": numeric_pass, "numeric_failure_modes": numeric_fail,
        "c9_technical_status": "", "c9_progression_authority": "USER_SELECTED_E213_AUG_XLSX",
        "execution_kind": f"E213_{u['arm']}_aug_FULL_COMPLETE",
        "result_sha256": E.sha256(paths["cem_result_npz"]), "scene_sha256": E.sha256(paths["scene_act"]),
        "trajectory_sha256": E.sha256(paths["trajectory"]),
        "contact_mask_sha256": E.sha256(paths["contact_mask"]),
        "metrics_sha256": E.sha256(EVAL_ROLLOUT), "evaluation_manifest_sha256": E.sha256(X.SOURCE_MANIFEST),
        # arm audit columns
        "aug_variant": u["variant"], "arm": u["arm"],
        "arm_experiment": seed.get("arm_experiment", ""), "arm_gravcomp": seed.get("arm_gravcomp", ""),
        "arm_scene_name": seed.get("arm_scene_name", ""),
        "arm_selection_reason": seed.get("arm_selection_reason", ""),
        "selection_authority": "E213 aug xlsx (user-curated) + E212 per-case arm",
    }
    for g in EVAL_GATE_FIELDS:
        row[g] = ev.get(g, "")
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=X.OUT)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    units = X.load_units()
    seed = X.seed_index()
    man_idx = X.source_manifest_index()
    eval_idx = {(r["case_id"], r["aug_variant"], r["arm"]): r for r in C.read_tsv(EVAL_ROLLOUT)}
    pm_idx = X.partner_manifest_index()
    e208_idx = X.e208_aug_index()
    source_ids = {u["case_id"] for u in units}

    source_rows: list[dict[str, Any]] = []
    excluded: list[dict[str, str]] = []
    unit_meta: list[dict[str, Any]] = []  # parallel to source_rows: (unit dict, man row)
    for u in units:
        key = (u["case_id"], u["variant"], u["arm"])
        man = man_idx.get(key)
        if man is None:
            raise SystemExit(f"{key}: no cem_ok row in E213 source manifest")
        ev = eval_idx.get(key, {})
        ok, why = c4_ok(ev)
        uid = unit_id(u["case_id"], u["variant"], u["arm"])
        if not ok:
            excluded.append({"unit": uid, "reason": f"C4_physical:{why}"})
            continue
        srow = build_source_row(uid, {**u, **{k: seed[u["case_id"]].get(k, "") for k in
                                               ("arm_experiment", "arm_gravcomp", "arm_scene_name")}},
                                seed[u["case_id"]], man, ev)
        source_rows.append(srow)
        unit_meta.append({"unit": u, "man": man})

    if not source_rows:
        raise SystemExit("no exportable aug units (all excluded by C4)")

    print(f"{len(source_rows)} source units ({len(excluded)} excluded):")
    for srow, meta in zip(source_rows, unit_meta, strict=True):
        u = meta["unit"]
        print(f"  {srow['case_id']:48s} {u['arm']:4s} narrow={srow['numeric_release_pass']:5s}")
    if excluded:
        for e in excluded:
            print(f"  EXCLUDED {e['unit']}: {e['reason']}")

    fields = SOURCE_FIELDS + [f for f in EXTRA_FIELDS if f not in SOURCE_FIELDS]
    if args.dry_run:
        print(f"\n[dry-run] {len(source_rows)} source rows, {len(fields)} columns; nothing written")
        return 0

    staging = args.out_dir.parent / f".{args.out_dir.name}.staging"
    if staging.exists():
        shutil.rmtree(staging)
    (staging / "partner_omnirt").mkdir(parents=True)
    source_path = staging / "rl_export_input.tsv"
    write_tsv(source_path, norm(source_rows), fields)
    (staging / "rl_export_input.json").write_text(
        json.dumps(norm(source_rows), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    shutil.copy2(X.XLSX, staging / "selection_snapshot.xlsx")

    partner_rows: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    blocked: list[str] = []
    for srow, meta in zip(source_rows, unit_meta, strict=True):
        u, man = meta["unit"], meta["man"]
        partner_case = X.flip_case(u["case_id"])
        variant_u = X.VAR_UNDERSCORE[u["variant"]]
        p_person, p_idx = X.person_of(partner_case)
        aug = X.resolve_partner_aug(partner_case, variant_u, pm_idx=pm_idx,
                                    e208_idx=e208_idx, source_ids=source_ids)
        p_ev = X.partner_evidence(srow, partner_case, aug)
        prow = PARTNER.build_partner_row(
            srow, partner_case=partner_case, partner_person=p_person,
            partner_person_idx=p_idx, evidence=p_ev,
            provenance=Path(aug["trimmed_npz"]), source_rl=source_path, repo=REPO,
            generation_mode="e213_selected_arm_aug_partner_kinematic_same_variant")
        s_ev = X.source_aug_evidence(man)
        audit = E.alignment_audit(srow, s_ev, X.SOURCE_MANIFEST, prow, p_ev, Path(aug["trimmed_npz"]))
        partner_rows.append(prow)
        audits.append(audit)

    misaligned = [a for a in audits if a["alignment_status"] not in ("RL_EXPORT_READY", E.BLOCKED_DECISION)]
    write_tsv(staging / "partner_resolution_audit.tsv", norm(audits), ALIGNMENT_FIELDS)
    (staging / "excluded_units.json").write_text(
        json.dumps(excluded, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if misaligned:
        for a in misaligned:
            print(f"  MISALIGNED {a['source_case_id']}: {a.get('alignment_failure_mode')}")
        (staging / "validation_report.json").write_text(
            json.dumps({"status": "blocked", "misaligned":
                        [a["source_case_id"] for a in misaligned]}, indent=2) + "\n", encoding="utf-8")
        raise SystemExit(f"aug RL export blocked by partner alignment: {len(misaligned)} unit(s)")

    pdir = staging / "partner_omnirt"
    partner_manifest = pdir / "rl_partner_omnirt_manifest.tsv"
    write_tsv(partner_manifest, norm(partner_rows), PARTNER.PARTNER_FIELDS)
    (pdir / "rl_partner_omnirt_manifest.json").write_text(
        json.dumps(norm(partner_rows), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    paired_fields = fields + [f for f in PARTNER.PAIRED_EXTRA_FIELDS if f not in fields]
    phash = E.sha256(partner_manifest)
    paired = [PARTNER.paired_row(s, p, manifest_ref=E.rel(partner_manifest), manifest_hash=phash, repo=REPO)
              for s, p in zip(source_rows, partner_rows, strict=True)]
    write_tsv(staging / "paired_rl_export_input.tsv", norm(paired), paired_fields)
    (staging / "paired_rl_export_input.json").write_text(
        json.dumps(norm(paired), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    ready = sum(1 for p in paired if p.get("paired_rl_export_decision") == "RL_EXPORT_READY")
    summary = {
        "experiment": "E213-export", "stage": "s6_downstream/export",
        "kind": "selected_arm_object_aug_paired", "created_at": E.now(),
        "status": "RL_EXPORT_READY" if not blocked and ready == len(paired) else "PARTIAL",
        "selection_authority": E.rel(X.XLSX), "selection_authority_sha256": X.EXPECTED_XLSX_SHA256,
        "seed_export": E.rel(X.SEED_TSV), "seed_sha256": X.EXPECTED_SEED_SHA256,
        "source_manifest": E.rel(X.SOURCE_MANIFEST),
        "source_rows": len(source_rows), "partner_rows": len(partner_rows),
        "excluded_unit_count": len(excluded), "paired_ready_rows": ready,
        "object_counts": dict(Counter(u["object_key"] for u in units)),
        "arm_counts": dict(Counter(u["arm"] for u in units)),
        "variant_counts": dict(Counter(u["variant"] for u in units)),
        "distinct_source_cases": len({u["case_id"] for u in units}),
        "partner_source_kinds": dict(Counter(
            "e213_partner_aug" if X.flip_case(m["unit"]["case_id"]) in pm_idx else "e208_source_aug"
            for m in unit_meta)),
        "partner_variant_counts": dict(Counter(p["partner_retarget_variant_id"] for p in partner_rows)),
        "consistency_note": ("two-person object consistency enforced DOWNSTREAM by Holosoma "
                             "partner re-anchor, verified post-reanchor (E200/E202)"),
        "claim_boundary": "paired RL export INPUT + loader readiness only; no RL outcome claim",
    }
    (staging / "rl_export_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if args.out_dir.exists():
        shutil.rmtree(args.out_dir)
    staging.rename(args.out_dir)

    # Re-pin partner manifest back-reference to the PUBLISHED source tsv (E212).
    published_source = args.out_dir / "rl_export_input.tsv"
    published_sha = E.sha256(published_source)
    for prow in partner_rows:
        prow["source_rl_export_input"] = E.rel(published_source)
        prow["source_rl_export_input_sha256"] = published_sha
    pdir = args.out_dir / "partner_omnirt"
    partner_manifest = pdir / "rl_partner_omnirt_manifest.tsv"
    write_tsv(partner_manifest, norm(partner_rows), PARTNER.PARTNER_FIELDS)
    (pdir / "rl_partner_omnirt_manifest.json").write_text(
        json.dumps(norm(partner_rows), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    phash2 = E.sha256(partner_manifest)
    paired = [PARTNER.paired_row(s, p, manifest_ref=E.rel(partner_manifest), manifest_hash=phash2, repo=REPO)
              for s, p in zip(source_rows, partner_rows, strict=True)]
    write_tsv(args.out_dir / "paired_rl_export_input.tsv", norm(paired), paired_fields)
    (args.out_dir / "paired_rl_export_input.json").write_text(
        json.dumps(norm(paired), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    validation = {**summary, "checks": {
        "selection_sha_exact": True, "seed_sha_exact": True,
        "per_unit_arm_scene_matches": True, "source_required_files_loadable": True,
        "partner_resolution_complete": len(partner_rows) == len(source_rows),
        "partner_alignment_no_misalignment": not misaligned,
        "paired_rows_equal_source_rows": len(paired) == len(source_rows),
    }, "artifact_sha256": {p.name: E.sha256(p) for p in sorted(args.out_dir.rglob("*.tsv"))}}
    (args.out_dir / "validation_report.json").write_text(
        json.dumps(validation, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"\nwrote {len(source_rows)} source + {len(partner_rows)} partner rows -> {E.rel(args.out_dir)}")
    print(f"  paired ready {ready}/{len(paired)}  excluded={len(excluded)}  blocked={blocked}")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
