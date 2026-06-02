#!/usr/bin/env python3
"""Shared helpers for Spider vs OmniRetarget fair-eval scripts."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable


REPO = Path(__file__).resolve().parents[4]

DEFAULT_THRESHOLDS_M = (0.03, 0.05, 0.08)
DEEP_PENETRATION_THRESHOLD_M = -0.02
FALL_PELVIS_THRESHOLD_M = 0.45


def repo_rel(path: str | Path | None) -> str:
    if path is None:
        return ""
    text = str(path)
    if not text:
        return ""
    p = Path(text)
    try:
        if p.is_absolute():
            return str(p.relative_to(REPO))
    except ValueError:
        return text
    return text


def resolve_path(path: str | Path | None, *, base: Path | None = None) -> Path | None:
    if path is None:
        return None
    text = str(path)
    if not text:
        return None
    p = Path(text)
    if p.is_absolute():
        return p
    if base is not None and (base / p).exists():
        return base / p
    return REPO / p


def read_csv(path: Path, *, delimiter: str = ",") -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter=delimiter))


def write_tsv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def as_float(value: Any, default: float = math.nan) -> float:
    try:
        if value is None or value == "":
            return default
        out = float(value)
        return out if math.isfinite(out) else default
    except Exception:
        return default


def as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "pass", "work"}


def threshold_tag(threshold_m: float) -> str:
    return f"{int(round(threshold_m * 100)):02d}cm"


def infer_experiment(path: str | Path) -> str:
    text = str(path)
    for part in Path(text).parts:
        if part.startswith("E") and part[1:].isdigit():
            return part
    return ""


def first_nonempty(row: dict[str, Any], keys: Iterable[str]) -> str:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return ""
