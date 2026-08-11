#!/usr/bin/env python3
"""Frozen contract and I/O helpers for E194 G1 72-case expansion."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[5]
SCRIPT_DIR = REPO / "workspace/core4d/scripts/experiments/E194"
RESULTS = REPO / "workspace/core4d/results/E194"
TASK_ROOT = REPO / "example_datasets/processed/core4d/unitree_g1/humanoid_object"

E173_MANIFEST = REPO / "workspace/core4d/results/E173/s6_downstream/manifests/cem_full_manifest.tsv"
E170_VARIANTS = REPO / "workspace/core4d/scripts/experiments/E170/variants.tsv"
E173_Z_METRICS = REPO / "workspace/core4d/results/E173/s6_downstream/eval/full/e173_object_tracking_position_error_z_by_case.tsv"

OBJECT_COUNTS = {"box001": 28, "box023": 16, "box021": 28}
OBJECT_ORDER = ("box001", "box023", "box021")
N_CASES = 72
ARM = "G1"
KP_POS = 500.0
KP_ROT = 50.0
GRAVCOMP = 1.0
FULL_SAMPLES, FULL_OPT_STEPS = 1024, 32
CANARY_SAMPLES, CANARY_OPT_STEPS = 64, 4
CEM_SEED = 0
WORKERS = ("local-gpu0", "ada-gpu0", "ada-gpu1")
WORKER_GPU = {"local-gpu0": "0", "ada-gpu0": "0", "ada-gpu1": "1"}
SCENE_NAME = "scene_act_E194_G1_expansion_rubberHull_PRG_gravcomp"
SIDECAR_FILE = f"{SCENE_NAME}.xml"

MANIFEST_DIR = RESULTS / "s6_downstream/manifests"
FULL_MANIFEST = MANIFEST_DIR / "g1_expansion_full_manifest.tsv"
CANARY_MANIFEST = MANIFEST_DIR / "g1_expansion_canary_manifest.tsv"
SENTINEL_MANIFEST = MANIFEST_DIR / "g1_expansion_sentinel_manifest.tsv"
SOURCE_AUTHORITY = MANIFEST_DIR / "g1_expansion_source_authority.tsv"
A0_METRICS = MANIFEST_DIR / "g1_expansion_a0_metrics.tsv"


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def repo_path(value: str | Path) -> Path:
    path = Path(str(value))
    if path.is_file() or path.is_dir():
        return path
    text = str(value)
    for marker in ("example_datasets/", "workspace/", "logs/"):
        if marker in text:
            return REPO / (marker + text.split(marker, 1)[1])
    return path if path.is_absolute() else REPO / path


def rel(value: str | Path) -> str:
    path = repo_path(value)
    try:
        return path.relative_to(REPO).as_posix()
    except ValueError:
        return Path(str(value)).as_posix()


def require_file(value: str | Path, label: str) -> Path:
    path = repo_path(value)
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"missing {label}: {value}")
    return path


def sha256(value: str | Path) -> str:
    path = require_file(value, "sha256 input")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_tsv(value: str | Path) -> list[dict[str, str]]:
    with require_file(value, "tsv").open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def read_with_fields(value: str | Path) -> tuple[list[dict[str, str]], list[str]]:
    with require_file(value, "tsv").open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        rows = list(reader)
        return rows, list(reader.fieldnames or [])


def serial(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return value


def write_tsv(value: str | Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path = repo_path(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: serial(row.get(key, "")) for key in fields})


def write_json(value: str | Path, payload: Any) -> None:
    path = repo_path(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def truth(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def source_rows() -> list[dict[str, str]]:
    """Load the exact 72 A0 rows without path-template reconstruction."""
    rows: list[dict[str, str]] = []
    for raw in read_tsv(E173_MANIFEST):
        if raw.get("object_key") not in {"box001", "box023"}:
            continue
        row = dict(raw)
        row["source_exp"] = "E173"
        row["execution_source"] = "E173"
        row["reused_full"] = "false"
        rows.append(row)
    for raw in read_tsv(E170_VARIANTS):
        if raw.get("object_key") != "box021":
            continue
        row = dict(raw)
        row["source_exp"] = "E170"
        rows.append(row)
    counts = Counter(row["object_key"] for row in rows)
    if dict(counts) != OBJECT_COUNTS:
        raise ValueError(f"source object counts differ: {dict(counts)} != {OBJECT_COUNTS}")
    if len(rows) != N_CASES or len({row["case_id"] for row in rows}) != N_CASES:
        raise ValueError(f"expected {N_CASES} unique source rows, got {len(rows)}")
    rank = {key: i for i, key in enumerate(OBJECT_ORDER)}
    return sorted(rows, key=lambda row: (rank[row["object_key"]], row["case_id"]))


def worker_for_object_index(index: int) -> str:
    """Per-object local,local,ada0,ada1 pattern gives exact 36/18/18."""
    return ("local-gpu0", "local-gpu0", "ada-gpu0", "ada-gpu1")[index % 4]


def validate_worker_balance(rows: list[dict[str, Any]]) -> None:
    totals = Counter(row["worker"] for row in rows)
    if totals != Counter({"local-gpu0": 36, "ada-gpu0": 18, "ada-gpu1": 18}):
        raise ValueError(f"worker balance differs: {dict(totals)}")
    for object_key in OBJECT_ORDER:
        workers = {row["worker"] for row in rows if row["object_key"] == object_key}
        if workers != set(WORKERS):
            raise ValueError(f"{object_key} missing worker coverage: {workers}")
