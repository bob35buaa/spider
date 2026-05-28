#!/usr/bin/env python3
"""E084 evaluator: reuse E083 diagnostics and add main-gate decisions."""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[4]
os.environ.setdefault("RESULTS", str(REPO / "workspace/core4d/results/E084"))
os.environ.setdefault(
    "VARIANTS_FILE", str(REPO / "workspace/core4d/scripts/E084/variants.tsv")
)
EVAL_DIR = REPO / "workspace/core4d/scripts/eval"
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

import eval_E083 as e083  # noqa: E402


RESULTS = Path(os.environ["RESULTS"])
VARIANTS_FILE = Path(os.environ["VARIANTS_FILE"])


def _bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _float(row: dict[str, object], key: str, default: float = 0.0) -> float:
    try:
        raw = row.get(key, default)
        if raw == "":
            return default
        return float(raw)
    except (TypeError, ValueError):
        return default


def _group(variant: str) -> str:
    if variant.startswith("E084A_"):
        return "A_safety"
    if variant.startswith("E084B_"):
        return "B_upright"
    if variant.startswith("E084C_"):
        return "C_semantic"
    return "unknown"


def _read_rows(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    keys = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def add_e084_gate_fields(rows: list[dict[str, object]]) -> dict[str, object]:
    accepted: list[str] = []
    for row in rows:
        variant = str(row["variant"])
        row["E084_group"] = _group(variant)
        contact_ok = _float(row, "case_window_sim_contact_frames_pct") >= 34.0
        bottom_ok = _float(row, "case_window_object_bottom_mean_gap_vs_ref_m") >= -0.08
        obj_ok = _float(row, "case_window_obj_err_mean_m") <= 0.80
        phys_ok = _bool(row.get("E083_success_upperbody_physical_proxy", False))
        row["E084_main_gate_contact_ok"] = bool(contact_ok)
        row["E084_main_gate_bottom_ok"] = bool(bottom_ok)
        row["E084_main_gate_obj_ok"] = bool(obj_ok)
        row["E084_main_gate_accept_for_guard"] = bool(
            row.get("role") == "main" and phys_ok and contact_ok and bottom_ok and obj_ok
        )
        if _bool(row["E084_main_gate_accept_for_guard"]):
            accepted.append(variant)

    accepted_groups = sorted({_group(v) for v in accepted})
    guard_splits = {
        "A_safety": "guard-A",
        "B_upright": "guard-B",
        "C_semantic": "guard-C",
    }
    aggregate = {
        "num_results": len(rows),
        "num_main_results": sum(row.get("role") == "main" for row in rows),
        "num_main_gate_accept_for_guard": len(accepted),
        "accepted_main_variants": accepted,
        "accepted_groups": accepted_groups,
        "guard_splits_to_run": [guard_splits[g] for g in accepted_groups],
        "stop_before_guard": len(accepted) == 0,
        "gate_rule": (
            "main role, E083 upperbody physical proxy, contact>=34%, "
            "bottom_gap_vs_ref>=-0.08m, obj_err_mean<=0.80m"
        ),
    }
    return aggregate


def main() -> None:
    e083.main()
    comparison = RESULTS / "comparison.csv"
    rows = _read_rows(comparison)
    gate = add_e084_gate_fields(rows)
    _write_rows(comparison, rows)

    previous_aggregate_path = RESULTS / "aggregate_summary.json"
    previous = {}
    if previous_aggregate_path.is_file():
        previous = json.loads(previous_aggregate_path.read_text(encoding="utf-8"))
    merged = {**previous, **gate}
    previous_aggregate_path.write_text(
        json.dumps(merged, indent=2, sort_keys=True), encoding="utf-8"
    )
    (RESULTS / "e084_gate_summary.json").write_text(
        json.dumps(gate, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"Wrote {comparison}")
    print(f"Wrote {RESULTS / 'e084_gate_summary.json'}")
    print(json.dumps(gate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
