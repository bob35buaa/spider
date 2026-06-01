"""Shared constants/helpers for E105 Box026 clean-scene reruns."""

from __future__ import annotations

import csv
from pathlib import Path


REPO = Path(__file__).resolve().parents[4]
TASK_ROOT = REPO / "example_datasets/processed/core4d/unitree_g1/humanoid_object"
OVERRIDE_ROOT = REPO / "examples/config/override"
RESULTS_ROOT = REPO / "workspace/core4d/results/E105"
SCRIPTS_ROOT = REPO / "workspace/core4d/scripts/E105"
VARIANTS_TSV = SCRIPTS_ROOT / "variants.tsv"

FIELDS = [
    "route",
    "case_id",
    "variant",
    "source_task",
    "derived_task",
    "person_idx",
    "split",
    "object",
    "role",
    "note",
    "target_route",
    "old_variant",
    "old_summary_path",
    "target_npz",
    "source_variant",
]

SOURCE_CASES = {
    "039": {
        "case_id": "C2",
        "short": "box026_039_p2",
        "source_task": "e091_box026_20231018_039_p2",
        "derived_task": "e091_box026_20231018_039_p2_e105_clean",
        "person_idx": "1",
    },
    "135": {
        "case_id": "C3",
        "short": "box026_135_p2",
        "source_task": "e091_box026_20231020_135_p2",
        "derived_task": "e091_box026_20231020_135_p2_e105_clean",
        "person_idx": "1",
    },
}

OLD_SUMMARIES = {
    "E092": "workspace/core4d/results/E092/spider_dyn/full/full_eval_summary.csv",
    "E094": "workspace/core4d/results/E094/cem/full/full_eval_summary.csv",
}


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def rows() -> list[dict[str, str]]:
    c039 = SOURCE_CASES["039"]
    c135 = SOURCE_CASES["135"]
    adaptive_039 = rel(
        RESULTS_ROOT
        / "adaptive_targets/targets/box026_039_p2_e091_box026_20231018_039_p2_adaptive_support_targets.npz"
    )
    adaptive_135 = rel(
        RESULTS_ROOT
        / "adaptive_targets/targets/box026_135_p2_e091_box026_20231020_135_p2_adaptive_support_targets.npz"
    )
    fingertip_039 = rel(
        RESULTS_ROOT / "fingertip_targets/e091_box026_20231018_039_p2/spider_contact_target_object_local.npz"
    )
    fingertip_135 = rel(
        RESULTS_ROOT / "fingertip_targets/e091_box026_20231020_135_p2/spider_contact_target_object_local.npz"
    )
    return [
        {
            "route": "ref_fk_clean",
            "case_id": c039["case_id"],
            "variant": "E105R1_box026_039_p2_ref_fk_clean",
            "source_task": c039["source_task"],
            "derived_task": c039["derived_task"],
            "person_idx": c039["person_idx"],
            "split": "wave1-remote-gpu0",
            "object": "box026",
            "role": "historical_E092_ref_fk_low_support",
            "note": "Clean-scene rerun aligned to E092D2.",
            "target_route": "ref_fk",
            "old_variant": "E092D2_box026_039_p2_dyn",
            "old_summary_path": OLD_SUMMARIES["E092"],
            "target_npz": "",
            "source_variant": "",
        },
        {
            "route": "ref_fk_clean",
            "case_id": c135["case_id"],
            "variant": "E105R2_box026_135_p2_ref_fk_clean",
            "source_task": c135["source_task"],
            "derived_task": c135["derived_task"],
            "person_idx": c135["person_idx"],
            "split": "wave1-remote-gpu1",
            "object": "box026",
            "role": "historical_E092_ref_fk_inside_risk",
            "note": "Clean-scene rerun aligned to E092D3.",
            "target_route": "ref_fk",
            "old_variant": "E092D3_box026_135_p2_dyn",
            "old_summary_path": OLD_SUMMARIES["E092"],
            "target_npz": "",
            "source_variant": "",
        },
        {
            "route": "adaptive_clean",
            "case_id": c039["case_id"],
            "variant": "E105A1_box026_039_p2_adaptive_clean",
            "source_task": c039["source_task"],
            "derived_task": c039["derived_task"],
            "person_idx": c039["person_idx"],
            "split": "wave1-local-gpu0",
            "object": "box026",
            "role": "historical_E094_adaptive_near_pass",
            "note": "Clean-scene rerun aligned to E094P2.",
            "target_route": "adaptive_support",
            "old_variant": "E094P2_box026_039_p2_hbproj",
            "old_summary_path": OLD_SUMMARIES["E094"],
            "target_npz": adaptive_039,
            "source_variant": "E105R1_box026_039_p2_ref_fk_clean",
        },
        {
            "route": "adaptive_clean",
            "case_id": c135["case_id"],
            "variant": "E105A2_box026_135_p2_adaptive_clean",
            "source_task": c135["source_task"],
            "derived_task": c135["derived_task"],
            "person_idx": c135["person_idx"],
            "split": "wave2-remote-gpu0",
            "object": "box026",
            "role": "historical_E094_adaptive_inside_risk",
            "note": "Clean-scene rerun aligned to E094P3.",
            "target_route": "adaptive_support",
            "old_variant": "E094P3_box026_135_p2_hbproj",
            "old_summary_path": OLD_SUMMARIES["E094"],
            "target_npz": adaptive_135,
            "source_variant": "E105R2_box026_135_p2_ref_fk_clean",
        },
        {
            "route": "fingertip_clean",
            "case_id": c039["case_id"],
            "variant": "E105F1_box026_039_p2_fingertip_clean",
            "source_task": c039["source_task"],
            "derived_task": c039["derived_task"],
            "person_idx": c039["person_idx"],
            "split": "wave2-local-gpu0",
            "object": "box026",
            "role": "E101_style_fingertip_ablation_039",
            "note": "Secondary ablation using E100/E101 fingertip-aware target.",
            "target_route": "fingertip",
            "old_variant": "",
            "old_summary_path": "",
            "target_npz": fingertip_039,
            "source_variant": "E105A1_box026_039_p2_adaptive_clean",
        },
        {
            "route": "fingertip_clean",
            "case_id": c135["case_id"],
            "variant": "E105F2_box026_135_p2_fingertip_clean",
            "source_task": c135["source_task"],
            "derived_task": c135["derived_task"],
            "person_idx": c135["person_idx"],
            "split": "wave2-remote-gpu1",
            "object": "box026",
            "role": "E101_style_fingertip_ablation_135",
            "note": "Secondary ablation using E100/E101 fingertip-aware target.",
            "target_route": "fingertip",
            "old_variant": "",
            "old_summary_path": "",
            "target_npz": fingertip_135,
            "source_variant": "E105A2_box026_135_p2_adaptive_clean",
        },
    ]


def read_variants(path: Path = VARIANTS_TSV) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows_out = list(
            csv.DictReader(
                (line for line in f if line.strip() and not line.startswith("#")),
                delimiter="\t",
                fieldnames=FIELDS,
            )
        )
    return [
        {field: (row.get(field) or "") for field in FIELDS}
        for row in rows_out
    ]


def write_variants(path: Path = VARIANTS_TSV) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        f.write("# E105 variants generated from e105_common.py\n")
        f.write("# " + "\t".join(FIELDS) + "\n")
        for row in rows():
            f.write("\t".join(row.get(field, "") for field in FIELDS).rstrip("\t") + "\n")
