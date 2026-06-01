"""Shared constants/helpers for E106 Box026 30-candidate batch."""

from __future__ import annotations

import csv
from pathlib import Path


REPO = Path(__file__).resolve().parents[4]
TASK_ROOT = REPO / "example_datasets/processed/core4d/unitree_g1/humanoid_object"
OVERRIDE_ROOT = REPO / "examples/config/override"
RESULTS_ROOT = REPO / "workspace/core4d/results/E106"
SCRIPTS_ROOT = REPO / "workspace/core4d/scripts/E106"
CANDIDATES_TSV = SCRIPTS_ROOT / "candidates.tsv"
VARIANTS_TSV = SCRIPTS_ROOT / "variants.tsv"
PREPROCESS_FAILURES_TSV = RESULTS_ROOT / "preprocess_failures.tsv"
E104_CANDIDATES_TSV = REPO / "workspace/core4d/results/E104/v2_candidates_3cm_with_fingertip.tsv"
HOLOSOMA_V2_ROOT = Path("/home/ubuntu/Workspace/holosoma/workspace/v3/data_construction_v2")
HOLOSOMA_STAGE_ROOT = HOLOSOMA_V2_ROOT / "results/stage2b_medium/results"
PIPELINE_CASE_FILE = HOLOSOMA_V2_ROOT / "inputs/cases_e106_box026_30candidate_pipeline.tsv"

PERSON_IDX = {"person1": "0", "person2": "1"}

CANDIDATE_FIELDS = [
    "ordinal",
    "e104_rank",
    "e104_route",
    "variant",
    "source_task",
    "derived_task",
    "split",
    "date",
    "seq",
    "person",
    "person_idx",
    "object_key",
    "object_name",
    "object_model_rel",
    "source_scene_task",
    "score",
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
    "e104_rank",
    "e104_route",
    "score",
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


def read_preprocess_failures(path: Path = PREPROCESS_FAILURES_TSV) -> dict[str, dict[str, str]]:
    if not path.is_file():
        return {}
    return {row["source_task"]: row for row in read_tsv(path)}


def slot_for_ordinal(ordinal: int) -> str:
    if 1 <= ordinal <= 10:
        return "local-gpu0"
    if 11 <= ordinal <= 20:
        return "remote-gpu0"
    if 21 <= ordinal <= 30:
        return "remote-gpu1"
    raise ValueError(f"E106 expects exactly 30 candidates, got ordinal={ordinal}")


def stage_case_dir(task: str) -> Path:
    return HOLOSOMA_STAGE_ROOT / f"holosoma_{task}"


def first_npz(root: Path, subdir: str) -> Path | None:
    paths = sorted((root / subdir).glob("*_original.npz"))
    return paths[0] if paths else None
