#!/usr/bin/env python3
"""Contract single-source for the E202-export step (plan235).

Turns the E202 bucket translation-augmentation source rollouts into partner-paired,
Holosoma-ready RL assets for the E178 manual-USE set. The manual-review TSV
(`user_manual_review_filled.tsv`, USE=13) is the selection authority. E202 already
produced the source-side aug CEM rollouts; this layer:

  * validates the USE13 authority (SHA + counts),
  * locates each USE case's feasible trans source rollouts in the E202 manifest,
  * builds partner *evidence* rows (Stage2b-shaped) pointing at the E202
    data_preprocess aug retarget of the opposite person under the SAME trans_k,

so the vetted `finalize_reused_partner_rl` partner/alignment contract can be reused
verbatim for the aug variants.

Scope facts (verified 2026-08-28):
  * USE=13 (bucket003x5 / bucket004x1 / bucket007x7).
  * 12/13 USE cases have E202 source aug (all 3 trans, run_complete_pending_eval,
    fall=0, no divergence); `bucket007_20231023_075_p2` has 0 feasible source trans
    (CEM-scene runtime leg-bucket overlap) -> orig-only (already in log264).
  * Partner aug retarget: 10 reuse E202, 2 new build (059_p2, 073_p2).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e202_common as C  # noqa: E402

REPO = C.REPO

# --- authority ---------------------------------------------------------------
REVIEW = REPO / "workspace/core4d/results/E178/s6_downstream/eval/full/user_manual_review_filled.tsv"
EXPECTED_REVIEW_SHA256 = "d430a8ef125117027bc54e0b7bd9bbeafe5c7a0a2e2a25f17a6cea0fd57c6c9f"
EXPECTED_USE = 13
EXPECTED_DO_NOT_USE = 14

# --- E202 source authorities -------------------------------------------------
E202_MANIFEST = C.FULL_MANIFEST  # e202_bucket_priority_manifest.tsv
AUG_METRICS = REPO / "workspace/core4d/results/E202/s6_downstream/eval/full_augmentation/e202_aug_case_metrics.tsv"

# --- E202-export outputs -----------------------------------------------------
OUT = REPO / "workspace/core4d/results/E202/s6_downstream/rl_export_aug"
PARTNER_AUG_MANIFEST = OUT / "partner_aug_build_manifest.json"
PARITY_REPORT = OUT / "object_traj_parity_report.tsv"
SNAPSHOT_DIR = REPO / "workspace/core4d/results/E202/scene_snapshot_export"

# partner aug retarget contract (matches E202 upstream: omnirt_v2 / ref_fk)
PARTNER_RETARGET_VARIANT = "omnirt_v2"
PARTNER_TARGET_VARIANT = "ref_fk"

# orig-only USE case (no E202 source aug) -- exported orig lives in log264
ORIG_ONLY_USE_CASES = ("bucket007_20231023_075_p2",)

_PERSON_FROM_SUFFIX = {"p1": "person1", "p2": "person2"}
_PERSON_IDX = {"person1": "0", "person2": "1"}
_FLIP = {"p1": "p2", "p2": "p1"}


# --- authority ---------------------------------------------------------------
def load_use13() -> dict[str, dict[str, str]]:
    """Return {case_id: review_row} for USE cases; assert SHA + USE/DNU counts."""
    actual = C.sha256(REVIEW)
    if actual != EXPECTED_REVIEW_SHA256:
        raise SystemExit(f"manual authority SHA drift: {actual}")
    rows = C.read_tsv(REVIEW)
    use = {r["case_id"]: r for r in rows if r.get("manual_use_decision") == "USE"}
    dnu = [r for r in rows if r.get("manual_use_decision") == "DO_NOT_USE"]
    if len(use) != EXPECTED_USE or len(dnu) != EXPECTED_DO_NOT_USE:
        raise SystemExit(f"manual counts drift: USE={len(use)} DNU={len(dnu)}")
    return use


# --- E202 source rollout index ----------------------------------------------
def load_source_variants() -> dict[tuple[str, str], dict[str, str]]:
    """Index E202 completed source aug rollouts by (case_id, aug_variant)."""
    idx: dict[tuple[str, str], dict[str, str]] = {}
    for row in C.read_tsv(E202_MANIFEST):
        if row.get("status") != "run_complete_pending_eval":
            continue
        idx[(row["case_id"], row["aug_variant"])] = row
    return idx


def _meta_fields(case_id: str, base_task: str) -> dict[str, str]:
    person = _PERSON_FROM_SUFFIX.get(case_id.rsplit("_", 1)[-1], "")
    meta = C.load_case_meta(base_task)
    return {
        "date": meta["date"], "seq": meta["seq"],
        "person": person, "person_idx": _PERSON_IDX.get(person, ""),
        "object_name": meta["object_name"], "holosoma_task": meta["holosoma_task"],
    }


def source_use_cases() -> list[dict[str, Any]]:
    """The 12 USE cases that have >=1 E202 source aug variant (excludes orig-only).

    Each entry carries base_target_task + object_key + parsed meta and the list of
    feasible (aug_variant -> source manifest row).
    """
    use = load_use13()
    variants = load_source_variants()
    # base_target_task per case from any manifest row
    base_by_case: dict[str, str] = {}
    obj_by_case: dict[str, str] = {}
    for (case_id, _v), row in variants.items():
        base_by_case.setdefault(case_id, row["base_target_task"])
        obj_by_case.setdefault(case_id, row["object_key"])

    out: list[dict[str, Any]] = []
    for case_id in sorted(use):
        case_variants = {v: variants[(case_id, v)] for (c, v) in variants if c == case_id}
        if not case_variants:
            if case_id not in ORIG_ONLY_USE_CASES:
                raise SystemExit(f"USE case {case_id} has no source aug and is not orig-only")
            continue
        base_task = base_by_case[case_id]
        entry = {
            "case_id": case_id,
            "object_key": obj_by_case[case_id],
            "base_target_task": base_task,
            "quality_label": use[case_id].get("manual_quality_label", ""),
            "source_variants": case_variants,  # {trans0: row, ...}
            **_meta_fields(case_id, base_task),
        }
        out.append(entry)
    return out


# --- source-side path resolvers ---------------------------------------------
def source_trim_window_json(base_task: str) -> Path:
    """The source person's own E202 trim_window.json (shared across its trans)."""
    root = REPO / C.DATA_PREPROCESS_REL / f"holosoma_{base_task}"
    p = root / "trim_window.json"
    if not p.is_file():
        raise SystemExit(f"source trim_window.json missing: {p}")
    return p


# --- partner evidence (Stage2b-shaped, over E202 aug data_preprocess) --------
def partner_case_id(source_case_id: str) -> str:
    base, _, suffix = source_case_id.rpartition("_")
    return f"{base}_{_FLIP[suffix]}"


def partner_base_task(source_base_task: str) -> str:
    base, _, suffix = source_base_task.rpartition("_")
    return f"{base}_{_FLIP[suffix]}"


def partner_evidence(source: dict[str, Any], holo_name: str) -> dict[str, str]:
    """Build a passing-Stage2b-shaped evidence row for the partner aug retarget.

    holo_name in {trans_0, trans_1, trans_2}. Points at the E202 data_preprocess
    aug retarget of the opposite person under the same trans_k. Compatible with
    finalize_reused_partner_rl.build_partner_row / trim_window_path.
    """
    p_case = partner_case_id(source["case_id"])
    p_base = partner_base_task(source["base_target_task"])
    p_suffix = p_case.rsplit("_", 1)[-1]
    p_person = _PERSON_FROM_SUFFIX[p_suffix]
    root = REPO / C.DATA_PREPROCESS_REL / f"holosoma_{p_base}"
    p_meta = C.load_case_meta(p_base)
    holo = p_meta["holosoma_task"]
    trimmed = root / "trimmed" / f"{holo}_{holo_name}.npz"
    retargeted = root / "retargeted" / f"{holo}_{holo_name}.npz"
    converted = root / "converted" / f"{holo}.npz"
    if not converted.is_file():
        # convert step writes `{seq}-personN-{Obj}_with_obj.npz`; holo already is that stem
        converted = root / "converted" / f"{holo}.npz"
    return {
        "case_id": p_case,
        "object_key": source["object_key"],
        "object_name": p_meta["object_name"],
        "date": p_meta["date"],
        "seq": p_meta["seq"],
        "person": p_person,
        "person_idx": _PERSON_IDX[p_person],
        "stage2b_status": "pass",
        "retarget_variant_id": PARTNER_RETARGET_VARIANT,
        "target_variant_id": PARTNER_TARGET_VARIANT,
        "target_task": f"{p_base}__aug_{holo_name.replace('trans_', 'trans')}",
        "holosoma_case_root": C.rel(root),
        "trim_window_json": C.rel(root / "trim_window.json"),
        "converted_npz": C.rel(converted),
        "omniretarget_output_npz": C.rel(retargeted),
        "retargeted_npz": C.rel(retargeted),
        "trimmed_npz": C.rel(trimmed),
    }
