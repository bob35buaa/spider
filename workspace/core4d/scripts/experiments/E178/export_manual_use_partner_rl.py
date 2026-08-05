#!/usr/bin/env python3
"""Export final E178 manual-USE cases as partner-complete RL inputs.

The final filled manual-review TSV is the selection authority.  This is an
S6-only packaging step: numeric gates remain provenance, while every exported
source must have a real opposite-person E174 Stage2b partner and a valid common
raw-frame alignment.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[5]
BASE_PATH = REPO / "workspace/core4d/scripts/experiments/E187/export_manual_use_partner_rl.py"


def load_base() -> Any:
    spec = importlib.util.spec_from_file_location("e187_partner_rl_export_base", BASE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import validated export base: {BASE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


E = load_base()
RUN_ROOT = REPO / "workspace/core4d/results/E178"
REVIEW = RUN_ROOT / "s6_downstream/eval/full/user_manual_review_filled.tsv"
EVAL_MANIFEST = RUN_ROOT / "s6_downstream/eval/full/evaluated_manifest_snapshot.tsv"
METRICS = RUN_ROOT / "s6_downstream/eval/full/e178_case_metrics.tsv"
OUT = RUN_ROOT / "s6_downstream/rl_export"
EXPECTED_REVIEW_SHA256 = "d430a8ef125117027bc54e0b7bd9bbeafe5c7a0a2e2a25f17a6cea0fd57c6c9f"
EXPECTED_USE = 13
EXPECTED_DO_NOT_USE = 14
METHOD_ID = "E178_semantic_bucket_contact_aligned_top_segment_union_r1"
HAND_COLLISION_ID = "rubber_hull"

# The E187 implementation is the already validated contract implementation.
# Bind only its experiment-specific authorities; no E187 artifact is modified.
E.RUN_ROOT = RUN_ROOT
E.REVIEW = REVIEW
E.EVAL_MANIFEST = EVAL_MANIFEST
E.METRICS = METRICS
E.OUT = OUT
E.PARTNER_OUT = OUT / "partner_omnirt"
E.EXPECTED_REVIEW_SHA256 = EXPECTED_REVIEW_SHA256
E.EXPECTED_USE = EXPECTED_USE
E.EXPECTED_DO_NOT_USE = EXPECTED_DO_NOT_USE
E.METHOD_ID = METHOD_ID
E.HAND_COLLISION_ID = HAND_COLLISION_ID


def make_source_row(*args: Any, **kwargs: Any) -> dict[str, Any]:
    row = E.make_source_row(*args, **kwargs)
    row["source_exp_id"] = "E178"
    return row


def main() -> int:
    _, reviews = E.read_tsv(REVIEW)
    _, evaluations = E.read_tsv(EVAL_MANIFEST)
    _, metrics = E.read_tsv(METRICS)
    review_by, eval_by, metric_by = E.validate_authority(reviews, evaluations, metrics)
    approved = sorted(case_id for case_id, row in review_by.items() if row["manual_use_decision"] == "USE")
    rejected = {case_id for case_id, row in review_by.items() if row["manual_use_decision"] == "DO_NOT_USE"}

    stage2b_index = E.PARTNER.load_stage2b_index(E.STAGE2B_MANIFESTS)
    source_rows: list[dict[str, Any]] = []
    source_stage: dict[str, tuple[Path, dict[str, str]]] = {}
    for case_id in approved:
        provenance, evidence = E.source_stage2b(eval_by[case_id], stage2b_index)
        source_stage[case_id] = (provenance, evidence)
        source_rows.append(
            make_source_row(
                case_id,
                review_by[case_id],
                eval_by[case_id],
                metric_by[case_id],
                evidence,
                provenance,
            )
        )
    if {row["case_id"] for row in source_rows} != set(approved) or rejected & set(approved):
        raise SystemExit("source selection does not exactly match final manual USE authority")

    staging = OUT.parent / f".{OUT.name}.staging"
    if staging.exists():
        shutil.rmtree(staging)
    partner_staging = staging / "partner_omnirt"
    partner_staging.mkdir(parents=True, exist_ok=True)
    source_path = staging / "rl_export_input.tsv"
    E.write_tsv(source_path, source_rows, E.SOURCE_FIELDS)
    E.write_json(staging / "rl_export_input.json", source_rows)
    shutil.copy2(REVIEW, staging / "manual_review_snapshot.tsv")

    partner_rows: list[dict[str, Any]] = []
    alignment_rows: list[dict[str, Any]] = []
    for source in source_rows:
        partner_case, partner_person, partner_person_idx = E.PARTNER.infer_partner(source)
        partner_provenance, partner_evidence = E.PARTNER.choose_partner(
            stage2b_index.get(partner_case, []), E.PARTNER.DEFAULT_VARIANT_PREFERENCE, partner_case
        )
        partner = E.PARTNER.build_partner_row(
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
        alignment_rows.append(
            E.alignment_audit(
                source,
                source_evidence,
                source_provenance,
                partner,
                partner_evidence,
                partner_provenance,
            )
        )
        partner_rows.append(partner)

    blocked = [row for row in alignment_rows if row["alignment_status"] != "RL_EXPORT_READY"]
    E.write_tsv(staging / "partner_resolution_audit.tsv", alignment_rows, E.ALIGNMENT_FIELDS)
    if blocked:
        E.write_json(staging / "validation_report.json", {"status": "blocked", "blocked": blocked})
        raise SystemExit(
            f"RL export blocked by partner alignment: {[row['source_case_id'] for row in blocked]}"
        )

    partner_manifest = partner_staging / "rl_partner_omnirt_manifest.tsv"
    E.write_tsv(partner_manifest, partner_rows, E.PARTNER.PARTNER_FIELDS)
    E.write_json(partner_staging / "rl_partner_omnirt_manifest.json", partner_rows)
    partner_hash = E.sha256(partner_manifest)
    paired_rows = [
        E.PARTNER.paired_row(
            source,
            partner,
            manifest_ref=E.rel(partner_manifest),
            manifest_hash=partner_hash,
            repo=REPO,
        )
        for source, partner in zip(source_rows, partner_rows, strict=True)
    ]
    E.write_tsv(staging / "paired_rl_export_input.tsv", paired_rows, E.SOURCE_FIELDS + E.PARTNER.PAIRED_EXTRA_FIELDS)
    E.write_json(staging / "paired_rl_export_input.json", paired_rows)

    summary = {
        "experiment": "E178",
        "stage": "S6_manual_use_partner_rl_export",
        "created_at": E.now(),
        "status": "RL_EXPORT_READY",
        "manual_authority": E.rel(REVIEW),
        "manual_authority_sha256": EXPECTED_REVIEW_SHA256,
        "manual_counts": {"USE": EXPECTED_USE, "DO_NOT_USE": EXPECTED_DO_NOT_USE, "PENDING": 0},
        "source_rows": len(source_rows),
        "partner_rows": len(partner_rows),
        "pair_complete_rows": sum(row["pair_status"] == "PAIR_COMPLETE" for row in partner_rows),
        "paired_ready_rows": sum(row["paired_rl_export_decision"] == "RL_EXPORT_READY" for row in paired_rows),
        "numeric_release_counts": dict(Counter(row["numeric_release_pass"] for row in source_rows)),
        "manual_quality_counts": dict(Counter(row["manual_quality_label"] for row in source_rows)),
        "object_counts": dict(Counter(row["object_key"] for row in source_rows)),
        "source_retarget_variant_counts": dict(Counter(row["retarget_variant_id"] for row in source_rows)),
        "partner_variant_counts": dict(Counter(row["partner_retarget_variant_id"] for row in partner_rows)),
        "hand_collision_variant_counts": dict(Counter(row["hand_collision_variant_id"] for row in source_rows)),
        "alignment_policy_counts": dict(Counter(row["alignment_policy"] for row in alignment_rows)),
        "all_partner_artifacts_nonempty": True,
        "all_source_partner_hashes_recomputed": True,
        "claim_boundary": "RL export and loader readiness only; no RL outcome claim",
    }
    E.write_json(staging / "rl_export_summary.json", summary)
    validation = {
        **summary,
        "checks": {
            "authority_sha_exact": True,
            "approved_source_set_exact": True,
            "rejected_zero_entry": True,
            "source_required_files_loadable": True,
            "partner_resolution_13_of_13": len(partner_rows) == EXPECTED_USE,
            "partner_alignment_13_of_13": len(alignment_rows) == EXPECTED_USE and not blocked,
            "paired_ready_13_of_13": len(paired_rows) == EXPECTED_USE,
        },
    }
    E.write_json(staging / "validation_report.json", validation)

    if OUT.exists():
        existing = OUT / "rl_export_summary.json"
        if existing.is_file():
            old_authority = json.loads(existing.read_text(encoding="utf-8")).get("manual_authority_sha256")
            if old_authority != EXPECTED_REVIEW_SHA256:
                raise SystemExit("refusing to overwrite RL export from a different manual authority")
        shutil.rmtree(OUT)
    staging.rename(OUT)

    published_source = OUT / "rl_export_input.tsv"
    published_partner = OUT / "partner_omnirt/rl_partner_omnirt_manifest.tsv"
    for row in partner_rows:
        row["source_rl_export_input"] = E.rel(published_source)
        row["source_rl_export_input_sha256"] = E.sha256(published_source)
    E.write_tsv(published_partner, partner_rows, E.PARTNER.PARTNER_FIELDS)
    E.write_json(OUT / "partner_omnirt/rl_partner_omnirt_manifest.json", partner_rows)
    published_partner_hash = E.sha256(published_partner)
    paired_rows = [
        E.PARTNER.paired_row(
            source,
            partner,
            manifest_ref=E.rel(published_partner),
            manifest_hash=published_partner_hash,
            repo=REPO,
        )
        for source, partner in zip(source_rows, partner_rows, strict=True)
    ]
    E.write_tsv(OUT / "paired_rl_export_input.tsv", paired_rows, E.SOURCE_FIELDS + E.PARTNER.PAIRED_EXTRA_FIELDS)
    E.write_json(OUT / "paired_rl_export_input.json", paired_rows)
    validation["artifact_sha256"] = {
        "rl_export_input.tsv": E.sha256(published_source),
        "partner_omnirt/rl_partner_omnirt_manifest.tsv": published_partner_hash,
        "paired_rl_export_input.tsv": E.sha256(OUT / "paired_rl_export_input.tsv"),
        "partner_resolution_audit.tsv": E.sha256(OUT / "partner_resolution_audit.tsv"),
        "manual_review_snapshot.tsv": E.sha256(OUT / "manual_review_snapshot.tsv"),
    }
    E.write_json(OUT / "validation_report.json", validation)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
