#!/usr/bin/env python3
"""E086 evaluator: E083 diagnostics plus a simple iteration gate."""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[4]
os.environ.setdefault("RESULTS", str(REPO / "workspace/core4d/results/E086"))
os.environ.setdefault(
    "VARIANTS_FILE", str(REPO / "workspace/core4d/scripts/E086/variants.tsv")
)
EVAL_DIR = REPO / "workspace/core4d/scripts/eval"
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

import eval_E083 as e083  # noqa: E402


RESULTS = Path(os.environ["RESULTS"])


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


def _read_rows(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    keys = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def add_e086_fields(rows: list[dict[str, object]]) -> dict[str, object]:
    accepted: list[str] = []
    for row in rows:
        phys_ok = _bool(row.get("E083_success_upperbody_physical_proxy", False))
        contact_ok = _float(row, "case_window_sim_contact_frames_pct") >= 50.0
        head_ok = _float(row, "case_window_sim_head_collision_object_penetration_pct") < 5.0
        upper_ok = _float(row, "case_window_sim_upperbody_object_penetration_pct") < 15.0
        hand_pen_ok = (
            _float(row, "case_window_sim_lh_object_penetration_pct") < 20.0
            and _float(row, "case_window_sim_rh_object_penetration_pct") < 20.0
        )
        row["E086_gate_phys_ok"] = bool(phys_ok)
        row["E086_gate_contact_ok"] = bool(contact_ok)
        row["E086_gate_head_ok"] = bool(head_ok)
        row["E086_gate_upper_ok"] = bool(upper_ok)
        row["E086_gate_hand_pen_ok"] = bool(hand_pen_ok)
        row["E086_gate_accept"] = bool(
            phys_ok and contact_ok and head_ok and upper_ok and hand_pen_ok
        )
        if _bool(row["E086_gate_accept"]):
            accepted.append(str(row["variant"]))
    return {
        "num_results": len(rows),
        "num_gate_accept": len(accepted),
        "accepted_variants": accepted,
        "gate_rule": (
            "E083 physical proxy, contact>=50%, head penetration<5%, "
            "upperbody penetration<15%, lh/rh penetration<20%"
        ),
    }


def main() -> None:
    e083.main()
    comparison = RESULTS / "comparison.csv"
    rows = _read_rows(comparison)
    gate = add_e086_fields(rows)
    _write_rows(comparison, rows)
    previous_aggregate_path = RESULTS / "aggregate_summary.json"
    previous = {}
    if previous_aggregate_path.is_file():
        previous = json.loads(previous_aggregate_path.read_text(encoding="utf-8"))
    merged = {**previous, **gate}
    previous_aggregate_path.write_text(
        json.dumps(merged, indent=2, sort_keys=True), encoding="utf-8"
    )
    (RESULTS / "e086_gate_summary.json").write_text(
        json.dumps(gate, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(gate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
