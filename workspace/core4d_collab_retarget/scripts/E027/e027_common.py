#!/usr/bin/env python3
"""Shared helpers for E027 offline data/timing audit."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

THIS = Path(__file__).resolve()
REPO = THIS.parents[4]
WS = REPO / "workspace/core4d_collab_retarget"
RESULTS = WS / "results/E027"
DATA_QUALITY = RESULTS / "data_quality"
TIMING = RESULTS / "timing_panels"
E018B = WS / "results/E018b"
E020 = WS / "results/E020_audit"
E026 = WS / "results/E026_full_eval"
HOLOSOMA = WS / "results/holosoma_v2_kinematic"

CASES_13 = [
    "box021_p1",
    "box021_p2",
    "box023_p1",
    "box023_p2",
    "box025_p1",
    "box025_p2",
    "bucket001_p1",
    "bucket001_p2",
    "bucket005_s2_p1",
    "bucket005_s2_p2",
    "bucket007_p1",
    "bucket007_p2",
    "desk021_p1",
]

P0_CASES = {
    "box023_p1",
    "box023_p2",
    "box025_p1",
    "box025_p2",
    "bucket001_p2",
    "bucket005_s2_p1",
    "bucket005_s2_p2",
    "bucket007_p1",
    "bucket007_p2",
}

TIMING_PRIORITY = ["box023_p1", "box023_p2", "box025_p1", "bucket007_p2"]


def ensure_repo_path() -> None:
    for path in [REPO, WS / "scripts/eval", WS / "scripts/E020_audit"]:
        s = str(path)
        if s not in sys.path:
            sys.path.insert(0, s)


def read_rows(path: Path, *, delimiter: str = ",") -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter=delimiter))


def write_rows(path: Path, rows: list[dict[str, Any]], *, delimiter: str = ",") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter=delimiter)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def as_float(value: Any, default: float = float("nan")) -> float:
    try:
        if value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def fmt_pct(value: Any) -> str:
    v = as_float(value)
    if not np.isfinite(v):
        return "n/a"
    return f"{v:.1f}%"


def index_by_case(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row["case"]: row for row in rows if row.get("case")}


def short_case_from_variant(variant: str) -> str:
    label = variant
    for prefix in ["E018b_", "E022_", "E023_", "E024_", "E025_"]:
        if label.startswith(prefix):
            label = label[len(prefix) :]
            break
    if label.endswith("_canonical_t02"):
        label = label[: -len("_canonical_t02")]
    parts = label.split("_")
    if len(parts) >= 2 and parts[-1].startswith("p") and parts[-1][1:].isdigit():
        return "_".join(parts[-2:])
    return label


def result_dir_for_variant(variant: str) -> Path:
    prefix = variant.split("_", 1)[0]
    return WS / "results" / prefix


def load_manifest_by_case() -> dict[str, dict[str, str]]:
    rows = read_rows(E018B / "manifest.tsv", delimiter="\t")
    return {short_case_from_variant(row["variant"]): row for row in rows}


def load_quality_rows() -> list[dict[str, str]]:
    return read_rows(DATA_QUALITY / "case_quality_audit.csv")


def load_best_selection() -> dict[str, dict[str, str]]:
    return index_by_case(read_rows(E026 / "best_dynamic_selection.csv"))


def load_e026_method_rows() -> list[dict[str, str]]:
    return read_rows(E026 / "method_case_metrics.csv")


def best_dynamic_metrics_by_case() -> dict[str, dict[str, str]]:
    rows = [
        row
        for row in load_e026_method_rows()
        if row.get("method") == "spider_best_E018b_E022_E025"
    ]
    return index_by_case(rows)


def load_holosoma_by_case() -> dict[str, dict[str, str]]:
    return index_by_case(read_rows(HOLOSOMA / "comparison.csv"))


def load_e020_tables() -> dict[str, dict[str, dict[str, str]]]:
    return {
        "root": index_by_case(read_rows(E020 / "root_cause_attribution.csv")),
        "anchor": index_by_case(read_rows(E020 / "anchor_vs_raw.csv")),
        "mask": index_by_case(read_rows(E020 / "mask_vs_raw.csv")),
        "ref": index_by_case(read_rows(E020 / "ref_physics.csv")),
        "sim": index_by_case(read_rows(E020 / "sim_ref_overlay.csv")),
    }


def mask_summary_for_manifest(row: dict[str, str]) -> dict[str, Any]:
    slug = row.get("mask_slug", "")
    path = E018B / "contact_masks" / slug / "audit_summary_3cm.json"
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def load_mask_npz_for_case(case: str) -> tuple[np.lib.npyio.NpzFile | None, int]:
    manifest = load_manifest_by_case().get(case, {})
    slug = manifest.get("mask_slug", "")
    path = E018B / "contact_masks" / slug / "raw_contact_mask_3cm.npz"
    if not path.is_file():
        return None, int(manifest.get("person_idx", 0) or 0)
    return np.load(path, allow_pickle=True), int(manifest.get("person_idx", 0) or 0)


def align_array(arr: np.ndarray, target_len: int) -> np.ndarray:
    arr = np.asarray(arr)
    if len(arr) == target_len:
        return arr
    if len(arr) == 0:
        return np.zeros((target_len,) + arr.shape[1:], dtype=arr.dtype)
    idx = np.round(np.linspace(0, len(arr) - 1, target_len)).astype(int)
    return arr[idx]


def selected_variant_for_case(case: str) -> str:
    best = load_best_selection().get(case, {})
    return best.get("selected_variant") or f"E018b_{case}_canonical_t02"


def timeseries_path_for_variant(variant: str) -> Path:
    return result_dir_for_variant(variant) / f"timeseries_{variant}.csv"


def legobj_timeseries_path_for_variant(variant: str) -> Path:
    return result_dir_for_variant(variant) / f"legobj_timeseries_{variant}.csv"


def panel_path(case: str) -> Path:
    return E020 / "per_case" / f"E018b_{case}_canonical_t02" / "attribution_panel.png"


def video_path(case: str) -> str:
    manifest = load_manifest_by_case().get(case, {})
    return manifest.get("online_video_path", "")
