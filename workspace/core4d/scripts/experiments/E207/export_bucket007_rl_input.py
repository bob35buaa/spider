#!/usr/bin/env python3
"""Build the E207 (PRG+G1) bucket007 source RL-export input, dcv3 schema.

Produces ``rl_export_input.tsv/.json`` for the 6 bucket007 cases of the E207
G1only arm, in the same 74-column ``core4d_data_construction_v3.0`` schema the
E178 export used, so the standard dcv3 partner adapter
(``data_construction_v3/stages/s6_downstream/finalize_reused_partner_rl.py``)
can consume it unchanged and attach the opposite-person OmniRetarget partner.

Selection authority (IMPORTANT -- differs from E178):
    E178 gated on a filled manual-review TSV (manual_use_decision == USE).
    **E207 has no manual review.** All 6 bucket007 cases are exported and the
    numeric outcome travels with each row as provenance (numeric_release_pass,
    numeric_failure_modes, the 12 gate flags, z_bias_cm, z_mae_cm) rather than
    as a gate. The manual_* columns are therefore blank and
    manual_use_decision is "NOT_REVIEWED" -- do not read them as an approval.
    Downstream must not interpret RL_EXPORT_READY here as "human approved".

Static per-case metadata (object_name/date/seq/person/trajectory/contact_mask/
target_scene/stage2b provenance) is seeded byte-for-byte from E178's already
validated rl_export_input row for the same case; only the arm-specific fields
are overridden (scene_act -> gravcomp sidecar, cem_* -> E207 rollout, numeric
gates -> E207 scores, sha256s recomputed).

Usage:
    .venv/bin/python workspace/core4d/scripts/experiments/E207/export_bucket007_rl_input.py
    ... --dry-run
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO / "workspace/core4d/scripts/experiments/E207"))
import e207_common as C  # noqa: E402

OBJECT_KEY = "bucket007"
ARM = "G1only"
METHOD_ID = "E207_bucket_g1only_gravcomp_r1"
SCHEMA_VERSION = "core4d_data_construction_v3.0"

E178_RL_INPUT = REPO / "workspace/core4d/results/E178/s6_downstream/rl_export/rl_export_input.tsv"
E207_MANIFEST = C.MANIFEST
FOUR_ARM = C.RESULTS / "s6_downstream/eval/four_arm/e207_arm_case_metrics.tsv"
ZDIFF = C.RESULTS / "s6_downstream/eval/four_arm/e207_object_z_diff_by_case.tsv"
OUT_DIR = C.RESULTS / "s6_downstream/rl_export"

#: gate flags carried straight from the E207 review TSV (same thresholds as E178).
GATE_FLAGS = [
    "fall_gate_pass", "body_z_gate_pass", "contact_gate_pass", "release_gate_pass",
    "hand_penetration_gate_pass", "lower_body_gate_pass", "root_pos_gate_pass",
    "root_ori_gate_pass", "hand_pos_gate_pass", "hand_ori_gate_pass",
    "object_pos_gate_pass", "object_ori_gate_pass",
]
#: appended beyond the E178 schema; extra columns are ignored by the adapter.
EXTRA_FIELDS = ["z_bias_cm", "z_mae_cm", "arm", "selection_authority",
                "prior_arm_manual_use_decision", "prior_arm_manual_quality_label",
                "prior_arm_manual_review_ref"]

NOT_REVIEWED = "NOT_REVIEWED"


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def build() -> tuple[list[dict[str, Any]], list[str], dict[str, Any]]:
    C.audit(verbose=False)
    e178 = {r["case_id"]: r for r in read_tsv(E178_RL_INPUT)}
    manifest = {r["case_id"]: r for r in read_tsv(E207_MANIFEST)}
    scores = {r["case_id"]: r for r in read_tsv(FOUR_ARM) if r["arm"] == ARM}
    zrows = {r["case_id"]: r for r in read_tsv(ZDIFF) if r["arm"] == ARM}

    cases = [c for c in C.CASES if c.startswith(OBJECT_KEY)]
    if not cases:
        raise SystemExit(f"no {OBJECT_KEY} cases in E207")
    for label, table in (("E178 rl_export_input", e178), ("E207 manifest", manifest),
                         ("four-arm scores", scores), ("z-diff", zrows)):
        missing = [c for c in cases if c not in table]
        if missing:
            raise SystemExit(f"{label} missing cases: {missing}")

    base_fields = list(next(iter(e178.values())))
    fields = base_fields + [f for f in EXTRA_FIELDS if f not in base_fields]

    rows: list[dict[str, Any]] = []
    for case_id in cases:
        row = dict(e178[case_id])  # validated dcv3 static provenance
        man, score, zrow = manifest[case_id], scores[case_id], zrows[case_id]

        scene_act = REPO / man["scene_act"]
        cem_npz = REPO / man["outdir_npz"]
        video = REPO / man["video"]
        for label, path in (("scene_act", scene_act), ("cem_result_npz", cem_npz)):
            if not path.is_file():
                raise SystemExit(f"{case_id}: missing {label}: {path}")
        # the shared inputs must still be exactly what E178 consumed
        for field in ("trajectory", "contact_mask", "target_scene"):
            if row[field] != man.get(field, row[field]) and field != "target_scene":
                raise SystemExit(f"{case_id}: {field} drifted from E178")

        row.update({
            "source_exp_id": "E207",
            "spider_method_id": METHOD_ID,
            "scene_act": man["scene_act"],
            "scene_name": man["scene_name"],
            "cem_status": "pass",
            "cem_run_id": man["variant"],
            # the outdir rollout is the primary artifact (result_npz is its copy);
            # E206 points cem_result_npz here too.
            "cem_result_npz": man["outdir_npz"],
            "cem_video": man["video"] if video.is_file() else "",
            "cem_metrics_ref": str((FOUR_ARM).relative_to(REPO)),
            "downstream_decision": "DOWNSTREAM_EXPORTED_FOR_RL_VALIDATION",
            "downstream_failure_mode": score.get("numeric_failure_modes", ""),
            "downstream_notes": (
                "E207 PRG+G1 (object gravcomp, A0 hand-gate). "
                "NO manual review exists for E207: all bucket007 cases are exported and "
                "the numeric outcome is provenance, not a gate. "
                "RL_EXPORT_READY here means assets are complete and partner-alignable, "
                "NOT that a human approved the motion."
            ),
            "rl_export_decision": "RL_EXPORT_READY",
            "skip_reason": "",
            "scene_act_exists": "True",
            "trajectory_exists": "True",
            "contact_mask_exists": "True",
            "cem_result_exists": "True",
            "source_handoff_manifest": str(E207_MANIFEST.relative_to(REPO)),
            "source_cem_evidence": str(FOUR_ARM.relative_to(REPO)),
            "schema_version": SCHEMA_VERSION,
            "updated_at": now(),
            # Manual authority: E178's USE verdict is NOT inherited as this arm's
            # approval. All 6 of these cases were reviewed USE under E178's PRG arm,
            # but nobody has reviewed the gravcomp arm -- and gravcomp visibly changes
            # the motion (log296 §5: different lift strategy, object airborne where
            # PRG braced it against the knee). So the prior verdict is recorded as
            # prior-arm provenance only.
            "manual_use_decision": NOT_REVIEWED,
            "manual_quality_label": "",
            "manual_failure_taxonomy": "",
            "manual_review_note": (
                "E207 (gravcomp arm) 未经人工审查。该 case 在 E178 PRG arm 下曾被人工判为 "
                f"USE/{e178[case_id].get('manual_quality_label', '')}，但 gravcomp 改变了动作，"
                "该结论不迁移到本 arm。"
            ),
            "manual_reviewer": "",
            "manual_reviewed_at": "",
            "manual_review_ref": "",
            "manual_review_sha256": "",
            "prior_arm_manual_use_decision": e178[case_id].get("manual_use_decision", ""),
            "prior_arm_manual_quality_label": e178[case_id].get("manual_quality_label", ""),
            "prior_arm_manual_review_ref": e178[case_id].get("manual_review_ref", ""),
            # numeric provenance (E201 14-gate narrow, same ruler as E178/E204/E205)
            "numeric_release_pass": score.get("numeric_release_pass", ""),
            "numeric_failure_modes": score.get("numeric_failure_modes", ""),
            "result_sha256": sha256(cem_npz),
            "scene_sha256": man["effective_scene_sha256"],
            "metrics_sha256": sha256(FOUR_ARM),
            "evaluation_manifest_sha256": sha256(E207_MANIFEST),
            "z_bias_cm": zrow["z_bias_cm"],
            "z_mae_cm": zrow["z_mae_cm"],
            "arm": ARM,
            "selection_authority": "E207_manifest_all_bucket007_no_manual_review",
        })
        for flag in GATE_FLAGS:
            if flag in row:
                row[flag] = score.get(flag, "")
        # not part of the 12-flag review TSV -> must not keep E178's stale value
        row["leg_gate_health_pass"] = ""
        row.pop("c9_technical_status", None) or row.update({"c9_technical_status": ""})
        row["c9_progression_authority"] = ""
        row["execution_kind"] = "production"
        rows.append(row)

    summary = {
        "experiment_id": "E207",
        "arm": ARM,
        "object_key": OBJECT_KEY,
        "schema_version": SCHEMA_VERSION,
        "spider_method_id": METHOD_ID,
        "rows": len(rows),
        "case_ids": cases,
        "selection_authority": {
            "kind": "manifest_scope",
            "detail": "all bucket007 cases of the E207 G1only arm",
            "manual_review": None,
            "note": "E178 gated on manual USE; E207 has none. Numeric outcome is "
                    "provenance only -- RL_EXPORT_READY is not a human approval.",
        },
        "numeric_release_pass_counts": dict(
            Counter(str(r["numeric_release_pass"]) for r in rows)
        ),
        "inputs": {
            str(E178_RL_INPUT.relative_to(REPO)): sha256(E178_RL_INPUT),
            str(E207_MANIFEST.relative_to(REPO)): sha256(E207_MANIFEST),
            str(FOUR_ARM.relative_to(REPO)): sha256(FOUR_ARM),
            str(ZDIFF.relative_to(REPO)): sha256(ZDIFF),
        },
        "generator": "workspace/core4d/scripts/experiments/E207/export_bucket007_rl_input.py",
        "created_at": now(),
    }
    return rows, fields, summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    rows, fields, summary = build()
    if args.dry_run:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        for row in rows:
            print(f"  {row['case_id']:30s} narrow={row['numeric_release_pass']:5s} "
                  f"fail={row['numeric_failure_modes'] or '-':22s} "
                  f"z_bias={float(row['z_bias_cm']):+.3f}")
        return 0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tsv = args.out_dir / "rl_export_input.tsv"
    with tsv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t",
                                lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    (args.out_dir / "rl_export_input.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary["outputs"] = {str(tsv.relative_to(REPO)): sha256(tsv)}
    (args.out_dir / "rl_export_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"[done] {tsv.relative_to(REPO)} ({len(rows)} rows, object={OBJECT_KEY}, arm={ARM})")
    print(f"  numeric_release_pass: {summary['numeric_release_pass_counts']}")
    print("  selection authority: manifest scope (NO manual review) — see downstream_notes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
