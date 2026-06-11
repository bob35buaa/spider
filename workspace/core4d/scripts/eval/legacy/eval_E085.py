#!/usr/bin/env python3
"""E085 evaluator: E083 diagnostics plus raw-target provenance fields."""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[4]
os.environ.setdefault("RESULTS", str(REPO / "workspace/core4d/results/E085"))
os.environ.setdefault(
    "VARIANTS_FILE", str(REPO / "workspace/core4d/scripts/E085/variants.tsv")
)
EVAL_DIR = REPO / "workspace/core4d/scripts/eval"
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

import eval_E083 as e083  # noqa: E402


RESULTS = Path(os.environ["RESULTS"])
VARIANTS_FILE = Path(os.environ["VARIANTS_FILE"])
RAW_TARGET_ROOT = REPO / "workspace/core4d/results/E085/raw_targets"


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


def _summary_for_variant(variant: str) -> dict[str, object]:
    path = RAW_TARGET_ROOT / variant / "summary.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def add_e085_fields(rows: list[dict[str, object]]) -> dict[str, object]:
    accepted: list[str] = []
    for row in rows:
        variant = str(row["variant"])
        row["E085_group"] = "A_rawtarget" if variant.startswith("E085A_") else "unknown"
        target_summary = _summary_for_variant(variant)
        row["E085_target_summary_exists"] = bool(target_summary)
        row["E085_target_kind"] = target_summary.get("target_kind", "")
        row["E085_left_face_counts"] = json.dumps(
            target_summary.get("hands", {}).get("left", {}).get("face_counts", {}),
            sort_keys=True,
        )
        row["E085_right_face_counts"] = json.dumps(
            target_summary.get("hands", {}).get("right", {}).get("face_counts", {}),
            sort_keys=True,
        )
        old = target_summary.get("old_g1_target_comparison", {})
        for hand in ("left", "right"):
            vals = old.get(hand, {}).get("surface_delta_cm_mean_min_max", [])
            row[f"E085_old_g1_{hand}_target_delta_cm_mean"] = vals[0] if vals else ""

        phys_ok = _bool(row.get("E083_success_upperbody_physical_proxy", False))
        contact_ok = _float(row, "case_window_sim_contact_frames_pct") >= 50.0
        hand_floor_ok = (
            _float(row, "case_window_sim_lh_floor_contact_pct") <= 5.0
            and _float(row, "case_window_sim_rh_floor_contact_pct") <= 5.0
        )
        obj_ok = _float(row, "case_window_obj_err_mean_m") <= 0.80
        row["E085_gate_phys_ok"] = bool(phys_ok)
        row["E085_gate_contact_ok"] = bool(contact_ok)
        row["E085_gate_hand_floor_ok"] = bool(hand_floor_ok)
        row["E085_gate_obj_ok"] = bool(obj_ok)
        row["E085_gate_accept"] = bool(phys_ok and contact_ok and hand_floor_ok and obj_ok)
        if _bool(row["E085_gate_accept"]):
            accepted.append(variant)

    aggregate = {
        "num_results": len(rows),
        "accepted_variants": accepted,
        "num_gate_accept": len(accepted),
        "gate_rule": (
            "E083 upperbody physical proxy, case-window contact>=50%, "
            "lh/rh floor contact<=5%, obj_err_mean<=0.80m"
        ),
    }
    return aggregate


def main() -> None:
    e083.main()
    comparison = RESULTS / "comparison.csv"
    rows = _read_rows(comparison)
    gate = add_e085_fields(rows)
    _write_rows(comparison, rows)

    previous_aggregate_path = RESULTS / "aggregate_summary.json"
    previous = {}
    if previous_aggregate_path.is_file():
        previous = json.loads(previous_aggregate_path.read_text(encoding="utf-8"))
    merged = {**previous, **gate}
    previous_aggregate_path.write_text(
        json.dumps(merged, indent=2, sort_keys=True), encoding="utf-8"
    )
    (RESULTS / "e085_gate_summary.json").write_text(
        json.dumps(gate, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"Wrote {comparison}")
    print(f"Wrote {RESULTS / 'e085_gate_summary.json'}")
    print(json.dumps(gate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
