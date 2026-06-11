#!/usr/bin/env python3
"""E087 evaluator: E083 diagnostics plus E087 gate fields."""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[4]
os.environ.setdefault("RESULTS", str(REPO / "workspace/core4d/results/E087"))
os.environ.setdefault(
    "VARIANTS_FILE", str(REPO / "workspace/core4d/scripts/E087/variants.tsv")
)
EVAL_DIR = REPO / "workspace/core4d/scripts/eval"
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

import eval_E083 as e083  # noqa: E402


RESULTS = Path(os.environ["RESULTS"])
VARIANTS_FILE = Path(os.environ["VARIANTS_FILE"])


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


def _variant_meta() -> dict[str, dict[str, str]]:
    out = {}
    with VARIANTS_FILE.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(
            (line for line in f if line.strip() and not line.startswith("#")),
            delimiter="\t",
            fieldnames=[
                "variant",
                "source_task",
                "derived_task",
                "mask_source_dir",
                "mask_slug",
                "person_idx",
                "split",
                "role",
                "group",
                "mass_kg",
                "notes",
            ],
        )
        for row in reader:
            out[row["variant"]] = row
    return out


def main() -> None:
    e083.main()
    comparison = RESULTS / "comparison.csv"
    rows = _read_rows(comparison)
    meta = _variant_meta()
    accepted = []
    for row in rows:
        variant = str(row["variant"])
        m = meta.get(variant, {})
        row["E087_group"] = m.get("group", "")
        row["E087_mass_kg"] = m.get("mass_kg", "")
        contact_ok = _float(row, "case_window_sim_contact_frames_pct") >= 50.0
        obj_ok = _float(row, "case_window_obj_err_mean_m") <= 0.765
        head_ok = _float(row, "case_window_sim_head_collision_object_penetration_pct") < 5.0
        upper_ok = _float(row, "case_window_sim_upperbody_object_penetration_pct") < 15.0
        floor_ok = (
            _float(row, "case_window_sim_lh_floor_contact_pct") <= 5.0
            and _float(row, "case_window_sim_rh_floor_contact_pct") <= 5.0
        )
        accept = contact_ok and obj_ok and head_ok and upper_ok and floor_ok
        row["E087_gate_contact_ok"] = bool(contact_ok)
        row["E087_gate_obj_ok"] = bool(obj_ok)
        row["E087_gate_head_ok"] = bool(head_ok)
        row["E087_gate_upper_ok"] = bool(upper_ok)
        row["E087_gate_floor_ok"] = bool(floor_ok)
        row["E087_gate_accept"] = bool(accept)
        if accept:
            accepted.append(variant)
    _write_rows(comparison, rows)
    gate = {
        "num_results": len(rows),
        "accepted_variants": accepted,
        "num_gate_accept": len(accepted),
        "gate_rule": "contact>=50, obj_mean<=0.765m, head<5%, upper<15%, hand-floor<=5%",
    }
    previous_path = RESULTS / "aggregate_summary.json"
    previous = {}
    if previous_path.is_file():
        previous = json.loads(previous_path.read_text(encoding="utf-8"))
    previous_path.write_text(
        json.dumps({**previous, **gate}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (RESULTS / "e087_gate_summary.json").write_text(
        json.dumps(gate, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"Wrote {comparison}")
    print(json.dumps(gate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
