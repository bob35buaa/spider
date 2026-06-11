"""Shared constants/helpers for E107 Box021 selected-4 full CEM."""

from __future__ import annotations

import csv
from pathlib import Path


REPO = Path(__file__).resolve().parents[4]
TASK_ROOT = REPO / "example_datasets/processed/core4d/unitree_g1/humanoid_object"
OVERRIDE_ROOT = REPO / "examples/config/override"
RESULTS_ROOT = REPO / "workspace/core4d/results/E107"
SCRIPTS_ROOT = REPO / "workspace/core4d/scripts/E107"
SELECTED_JSON = RESULTS_ROOT / "selected_case_to_cem.json"
GATE_SUMMARY_TSV = RESULTS_ROOT / "box021_clean_gate_summary.tsv"
CANDIDATES_TSV = SCRIPTS_ROOT / "selected4_candidates.tsv"
VARIANTS_TSV = SCRIPTS_ROOT / "selected4_variants.tsv"

PERSON_IDX = {"person1": "0", "person2": "1"}

CANDIDATE_FIELDS = [
    "ordinal",
    "selected_id",
    "variant",
    "source_task",
    "derived_task",
    "split",
    "date",
    "seq",
    "person",
    "person_idx",
    "object_key",
    "source_scene_task",
    "qpos_shape",
    "target_both_active_frac_3cm",
    "target_both_active_frac_5cm",
    "raw_contact_proxy_path",
]

VARIANT_FIELDS = [
    "ordinal",
    "variant",
    "source_task",
    "derived_task",
    "person_idx",
    "split",
    "selected_id",
    "source_scene_task",
]


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def read_tsv(path: Path, fields: list[str] | None = None) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        lines = (line for line in f if line.strip() and not line.startswith("#"))
        if fields is None:
            return list(csv.DictReader(lines, delimiter="\t"))
        return list(csv.DictReader(lines, fieldnames=fields, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, object]], fields: list[str], comment: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        if comment:
            f.write(f"# {comment}\n")
            f.write("# " + "\t".join(fields) + "\n")
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t", lineterminator="\n")
        if not comment:
            writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def read_candidates(path: Path = CANDIDATES_TSV) -> list[dict[str, str]]:
    return read_tsv(path, CANDIDATE_FIELDS)


def read_variants(path: Path = VARIANTS_TSV) -> list[dict[str, str]]:
    return read_tsv(path, VARIANT_FIELDS)


def selected_id_to_source_task(selected_id: str) -> str:
    if not selected_id.endswith("_e107"):
        raise ValueError(f"Unexpected E107 selected id: {selected_id}")
    return selected_id.removesuffix("_e107")


def selected_id_to_clean_task(selected_id: str) -> str:
    return f"{selected_id_to_source_task(selected_id)}_e107_clean"


def split_for_ordinal(ordinal: int) -> str:
    if ordinal == 1:
        return "local-gpu0"
    if ordinal in {2, 4}:
        return "remote-gpu0"
    if ordinal == 3:
        return "remote-gpu1"
    raise ValueError(f"E107 selected-4 ordinal out of range: {ordinal}")
