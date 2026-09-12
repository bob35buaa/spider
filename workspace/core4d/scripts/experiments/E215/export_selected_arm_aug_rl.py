#!/usr/bin/env python3
"""E215-export: package the user-curated rot-aug units as partner-paired,
dcv3 Holosoma-ready RL inputs (plan247 RL-export).

Per xlsx-selected (case, rot variant) unit it emits one export unit:
  * source  = the E215 rot aug CEM rollout on the case's selected arm_group
    (scene_act / trajectory / contact_mask / cem_result), reused as-is (no CEM);
  * partner = the OPPOSITE person's kinematic rot retarget under the IDENTICAL
    variant, from the E215 source tree (delivered partner) or the E215
    partner_aug tree (gap partner, build_partner_aug.py);
  * a common-raw-window alignment audit.

Two-person object consistency is enforced DOWNSTREAM by the Holosoma exporter's
partner re-anchor (partner hands -> source object frame), verified post-reanchor
(E200/E202/E213 practice). A unit is EXCLUDED (not fatal) if it fails the C4
physical hard-gate, has no resolvable partner rot retarget, or its partner
window does not align -- every other unit still exports. Emits the dcv3 rl_export
schema so the existing Holosoma exporter consumes it unchanged. Writes only under
results/E215/s6_downstream/export/.

Usage:
    .venv/bin/python .../E215/export_selected_arm_aug_rl.py --dry-run
    .venv/bin/python .../E215/export_selected_arm_aug_rl.py
"""

from __future__ import annotations

import argparse
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

import e215_common as C  # noqa: E402
import e215_export_common as X  # noqa: E402

REPO = C.REPO
E206_EXPORTER = REPO / "workspace/core4d/scripts/experiments/E206/export_manual_use_partner_rl.py"

DIVERGENCE_CM = 60.0  # C4 physical hard-gate (E202/E213-export precedent)
METHOD_ID = "E215_rot_object_aug_omnirt_v2"
HAND_COLLISION_ID = "rubber_hull"
EVAL_ROLLOUT = C.EVAL_DIR / "e215_rot_rollout.tsv"

EXTRA_FIELDS = ["aug_variant", "arm_group", "base_variant", "selection_authority"]
EVAL_GATE_FIELDS = [
    "fall_gate_pass", "object_pos_gate_pass", "object_ori_gate_pass",
    "contact_gate_pass", "hand_penetration_gate_pass", "lower_body_gate_pass",
]


def _load_e206_exporter() -> Any:
    """Import E206's exporter as a LIBRARY (dcv3 PARTNER adapter + alignment_audit +
    field lists + IO helpers), exactly as E213-export does."""
    for d in ("E206", "E200", "E199"):
        p = str(REPO / "workspace/core4d/scripts/experiments" / d)
        if p not in sys.path:
            sys.path.insert(0, p)
    spec = importlib.util.spec_from_file_location("e215_e206_exporter", E206_EXPORTER)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


E = _load_e206_exporter()
PARTNER = E.PARTNER
SOURCE_FIELDS = E.SOURCE_FIELDS
ALIGNMENT_FIELDS = E.ALIGNMENT_FIELDS

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
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def unit_id(case_id: str, variant: str, arm_group: str) -> str:
    return f"{case_id}__aug_{variant}_{arm_group}"


def c4_ok(ev: dict[str, str]) -> tuple[bool, str]:
    def f(k: str) -> float:
        try:
            return float(ev.get(k, "") or 0.0)
        except ValueError:
            return 0.0
    fails = []
    if str(ev.get("fall_flag", "")).lower() in {"true", "1", "1.0"} or f("fall_flag") > 0:
        fails.append("fall")
    if f("track_root_pos_err_cm_mean") > DIVERGENCE_CM:
        fails.append(f"root_diverge>{DIVERGENCE_CM}cm")
    if f("track_eef_pos_err_cm_mean") > DIVERGENCE_CM:
        fails.append(f"eef_diverge>{DIVERGENCE_CM}cm")
    return (not fails), ",".join(fails)


def build_source_row(uid: str, u: dict[str, str], meta: dict[str, str],
                     man: dict[str, str], ev: dict[str, str]) -> dict[str, Any]:
    paths = {
        "scene_act": E.required(man["scene_act"], f"{uid} scene_act"),
        "trajectory": E.required(man["trajectory"], f"{uid} trajectory"),
        "contact_mask": E.required(man["contact_mask"], f"{uid} contact_mask"),
        "cem_result_npz": E.required(man["outdir_npz"], f"{uid} cem_result"),
    }
    video_raw = man.get("video", "")
    video_path = C.repo_path(video_raw) if video_raw else None
    video_ok = bool(video_path and video_path.is_file() and video_path.stat().st_size > 0)
    numeric_pass = ev.get("numeric_release_pass", "")
    numeric_fail = ev.get("numeric_failure_modes", "")
    row: dict[str, Any] = {
        "case_id": uid, "object_key": meta["object_key"], "object_name": meta["object_name"],
        "date": meta["date"], "seq": meta["seq"], "person": meta["person"],
        "person_idx": meta["person_idx"],
        "retarget_variant_id": meta["effective_retarget_variant"],
        "target_variant_id": X.TARGET_VARIANT,
        "hand_collision_variant_id": HAND_COLLISION_ID, "source_exp_id": "E215-export",
        "spider_method_id": METHOD_ID, "handoff_decision": "HANDOFF_READY",
        "candidate_decision": "USER_SELECTED_E215_ROT_XLSX", "target_gate_status": "pass",
        "visual_qc_status": "xlsx_selected", "target_scene": meta.get("target_scene", ""),
        "trajectory": E.rel(paths["trajectory"]), "scene_act": E.rel(paths["scene_act"]),
        "contact_mask": E.rel(paths["contact_mask"]),
        "stage2b_target_task": man["target_task"], "stage2b_result_root": "",
        "stage2b_manifest_ref": E.rel(X.SHARD_MANIFESTS[0]),
        "raw_contact_threshold_label": "3cm", "cem_status": "pass", "cem_run_id": C.RUN_ID,
        "cem_result_npz": E.rel(paths["cem_result_npz"]),
        "cem_video": E.rel(video_path) if video_ok else "",
        "cem_metrics_ref": E.rel(EVAL_ROLLOUT),
        "downstream_decision": "DOWNSTREAM_USER_APPROVED_FOR_RL_VALIDATION",
        "downstream_failure_mode": numeric_fail,
        "downstream_notes": (
            f"E215-export rot aug unit ({u['arm_group']}); xlsx-selected; "
            f"numeric_release_pass={numeric_pass}; video_exists={video_ok}; "
            "two-person object consistency via downstream partner re-anchor"),
        "rl_export_decision": "RL_EXPORT_READY", "skip_reason": "",
        "scene_act_exists": "True", "trajectory_exists": "True",
        "contact_mask_exists": "True", "cem_result_exists": "True",
        "source_handoff_manifest": E.rel(X.SHARD_MANIFESTS[0]), "source_cem_evidence": E.rel(EVAL_ROLLOUT),
        "schema_version": PARTNER.SCHEMA_VERSION, "updated_at": E.now(),
        "manual_use_decision": "USER_SELECTED_E215_ROT_XLSX", "manual_quality_label": "",
        "manual_failure_taxonomy": "", "manual_review_note": "",
        "manual_reviewer": "", "manual_reviewed_at": "",
        "manual_review_ref": E.rel(X.XLSX), "manual_review_sha256": X.EXPECTED_XLSX_SHA256,
        "numeric_release_pass": numeric_pass, "numeric_failure_modes": numeric_fail,
        "c9_technical_status": "", "c9_progression_authority": "USER_SELECTED_E215_ROT_XLSX",
        "execution_kind": f"E215_{u['arm_group']}_rot_aug_FULL_COMPLETE",
        "result_sha256": E.sha256(paths["cem_result_npz"]), "scene_sha256": E.sha256(paths["scene_act"]),
        "trajectory_sha256": E.sha256(paths["trajectory"]),
        "contact_mask_sha256": E.sha256(paths["contact_mask"]),
        "metrics_sha256": E.sha256(EVAL_ROLLOUT), "evaluation_manifest_sha256": E.sha256(X.SHARD_MANIFESTS[0]),
        # audit columns
        "aug_variant": u["variant"], "arm_group": u["arm_group"],
        "base_variant": man.get("base_variant", ""),
        "selection_authority": "E215 rot aug xlsx (user-curated)",
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
    man_idx = X.source_manifest_index()
    eval_idx = {(r["case_id"], r["aug_variant"]): r for r in C.read_tsv(EVAL_ROLLOUT)}

    source_rows: list[dict[str, Any]] = []
    unit_meta: list[dict[str, Any]] = []
    excluded: list[dict[str, str]] = []
    for u in units:
        key = (u["case_id"], u["variant"])
        man = man_idx.get(key)
        uid = unit_id(u["case_id"], u["variant"], u["arm_group"])
        if man is None:
            excluded.append({"unit": uid, "reason": "no_source_cem_row"})
            continue
        ev = eval_idx.get(key, {})
        ok, why = c4_ok(ev)
        if not ok:
            excluded.append({"unit": uid, "reason": f"C4_physical:{why}"})
            continue
        meta = X.source_meta(man)
        srow = build_source_row(uid, u, meta, man, ev)
        source_rows.append(srow)
        unit_meta.append({"unit": u, "man": man, "meta": meta})

    # phase 1: resolve partner + align each unit (minimal stub, no file writes);
    # exclude (not fatal) on unresolved/misaligned. build_partner_row is deferred
    # to phase 2 because it hashes the published source tsv (must exist first).
    audits: list[dict[str, Any]] = []
    keep_src: list[dict[str, Any]] = []
    keep_meta: list[dict[str, Any]] = []
    keep_partner: list[dict[str, Any]] = []   # resolution context for phase 2
    for srow, m in zip(source_rows, unit_meta, strict=True):
        u = m["unit"]
        uid = srow["case_id"]
        partner_case = X.flip_case(u["case_id"])
        variant_u = X.VAR_UNDERSCORE[u["variant"]]
        p_person, p_idx = X.person_of(partner_case)
        try:
            aug = X.resolve_partner_aug(partner_case, variant_u)
            p_ev = X.partner_evidence(srow, partner_case, aug)
            stub = {"partner_case_id": partner_case, "partner_person": p_person,
                    "partner_retarget_variant_id": aug["retarget_variant_id"],
                    "trimmed_npz": aug["trimmed_npz"]}
            s_ev = X.source_aug_evidence(m["man"])
            audit = E.alignment_audit(srow, s_ev, X.SHARD_MANIFESTS[0], stub, p_ev,
                                      Path(aug["trimmed_npz"]))
        except SystemExit as exc:
            excluded.append({"unit": uid, "reason": f"partner_unresolved:{exc}"})
            continue
        if audit.get("alignment_status") != "RL_EXPORT_READY":
            excluded.append({"unit": uid,
                             "reason": f"partner_misaligned:{audit.get('alignment_failure_mode')}"})
            continue
        keep_src.append(srow)
        keep_meta.append(m)
        keep_partner.append({"partner_case": partner_case, "p_person": p_person,
                             "p_idx": p_idx, "aug": aug, "p_ev": p_ev})
        audits.append(audit)
    source_rows = keep_src
    unit_meta = keep_meta

    print(f"{len(source_rows)} exportable units ({len(excluded)} excluded):")
    for srow in source_rows:
        print(f"  {srow['case_id']:52s} narrow_pass={srow['numeric_release_pass']:5s}")
    for e in excluded:
        print(f"  EXCLUDED {e['unit']}: {e['reason']}")
    if not source_rows:
        raise SystemExit("no exportable units")

    fields = SOURCE_FIELDS + [f for f in EXTRA_FIELDS if f not in SOURCE_FIELDS]
    if args.dry_run:
        print(f"\n[dry-run] {len(source_rows)} source + {len(source_rows)} partner rows, "
              f"{len(fields)} source columns; nothing written")
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

    # phase 2: build partner rows now that the source tsv exists (build_partner_row
    # hashes it as a back-reference).
    partner_rows = [
        PARTNER.build_partner_row(
            srow, partner_case=ctx["partner_case"], partner_person=ctx["p_person"],
            partner_person_idx=ctx["p_idx"], evidence=ctx["p_ev"],
            provenance=Path(ctx["aug"]["trimmed_npz"]), source_rl=source_path,
            repo=REPO, generation_mode="e215_rot_aug_partner_kinematic_same_variant")
        for srow, ctx in zip(source_rows, keep_partner, strict=True)
    ]

    write_tsv(staging / "partner_resolution_audit.tsv", norm(audits), ALIGNMENT_FIELDS)
    (staging / "excluded_units.json").write_text(
        json.dumps(excluded, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

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
        "experiment": "E215-export", "stage": "s6_downstream/export",
        "kind": "rot_object_aug_paired", "created_at": E.now(),
        "status": "RL_EXPORT_READY" if ready == len(paired) else "PARTIAL",
        "selection_authority": E.rel(X.XLSX), "selection_authority_sha256": X.EXPECTED_XLSX_SHA256,
        "source_rows": len(source_rows), "partner_rows": len(partner_rows),
        "excluded_unit_count": len(excluded), "paired_ready_rows": ready,
        "object_counts": dict(Counter(m["unit"]["object_key"] for m in unit_meta)),
        "arm_group_counts": dict(Counter(m["unit"]["arm_group"] for m in unit_meta)),
        "variant_counts": dict(Counter(m["unit"]["variant"] for m in unit_meta)),
        "distinct_source_cases": len({m["unit"]["case_id"] for m in unit_meta}),
        "partner_source_kinds": dict(Counter(
            "e215_source_retarget" if X.is_delivered_source(X.flip_case(m["unit"]["case_id"]))
            else "e215_partner_aug" for m in unit_meta)),
        "partner_variant_counts": dict(Counter(p["partner_retarget_variant_id"] for p in partner_rows)),
        "consistency_note": ("two-person object consistency enforced DOWNSTREAM by Holosoma "
                             "partner re-anchor, verified post-reanchor (E200/E202/E213)"),
        "claim_boundary": "paired RL export INPUT + loader readiness only; no RL outcome claim",
    }
    (staging / "rl_export_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if args.out_dir.exists():
        shutil.rmtree(args.out_dir)
    staging.rename(args.out_dir)

    # re-pin partner back-reference to the published source tsv (E212/E213 practice)
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
        "selection_sha_exact": True,
        "source_required_files_loadable": True,
        "partner_resolution_complete": len(partner_rows) == len(source_rows),
        "partner_alignment_no_misalignment": all(
            a["alignment_status"] == "RL_EXPORT_READY" for a in audits),
        "paired_rows_equal_source_rows": len(paired) == len(source_rows),
    }, "artifact_sha256": {p.name: E.sha256(p) for p in sorted(args.out_dir.rglob("*.tsv"))}}
    (args.out_dir / "validation_report.json").write_text(
        json.dumps(validation, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"\nwrote {len(source_rows)} source + {len(partner_rows)} partner rows -> {E.rel(args.out_dir)}")
    print(f"  paired ready {ready}/{len(paired)}  excluded={len(excluded)}")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
