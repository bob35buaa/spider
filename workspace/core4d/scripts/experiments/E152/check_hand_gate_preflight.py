#!/usr/bin/env python3
"""Preflight checks for E152 hand-object CEM gate experiments."""

from __future__ import annotations

import csv
import json
import argparse
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
VARIANTS = REPO / "workspace/core4d/scripts/E152/variants.tsv"
RESULT_ROOT = REPO / "workspace/core4d/results/E152/axis1_hand_object_physics_gate"


def repo_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO / p


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-missing-reuse", action="store_true")
    args = parser.parse_args()

    rows = read_rows(VARIANTS)
    errors: list[str] = []
    if len(rows) != 12:
        errors.append(f"expected 12 rows, got {len(rows)}")
    todo = [r for r in rows if r["run_status"] == "to_run"]
    if len(todo) != 6:
        errors.append(f"expected 6 to_run rows, got {len(todo)}")
    split_counts: dict[str, int] = {}
    for row in todo:
        split_counts[row["split"]] = split_counts.get(row["split"], 0) + 1
    if split_counts != {"local-gpu0": 2, "remote-gpu0": 2, "remote-gpu1": 2}:
        errors.append(f"expected split 2/2/2, got {split_counts}")

    gate_rows = [r for r in rows if r["method"] in {"gateA", "gateA_b1"}]
    for row in gate_rows:
        override = repo_path(row["override"])
        if not override.is_file():
            errors.append(f"{row['variant']}: missing override {row['override']}")
            continue
        text = override.read_text(encoding="utf-8")
        for token in (
            "cem_hand_gate_enabled: true",
            'cem_hand_gate_geom_names: ["lh", "rh"]',
            "cem_hand_gate_min_sdf_m:",
            "cem_hand_gate_max_violation_pct:",
        ):
            if token not in text:
                errors.append(f"{row['variant']}: override missing {token}")
        try:
            if float(row["cem_hand_gate_min_sdf_m"]) >= 0.0:
                errors.append(f"{row['variant']}: hand gate min_sdf should allow light contact")
            if not (0.0 <= float(row["cem_hand_gate_max_violation_pct"]) <= 1.0):
                errors.append(f"{row['variant']}: invalid hand gate max violation pct")
        except Exception as exc:
            errors.append(f"{row['variant']}: invalid hand gate numeric fields: {exc}")

    for row in rows:
        for key in ("target_scene", "trajectory", "rubber_scene_act", "object_asset", "mask_path", "override"):
            if row.get(key) and not repo_path(row[key]).is_file():
                errors.append(f"{row['variant']}: missing {key}: {row[key]}")
        if row["run_status"] != "to_run" and not args.allow_missing_reuse:
            for key in ("result_npz", "outdir_npz"):
                if not repo_path(row[key]).is_file():
                    errors.append(f"{row['variant']}: missing reuse {key}: {row[key]}")

    subprocess.run(
        [
            ".venv/bin/python",
            "-m",
            "py_compile",
            "spider/config.py",
            "spider/simulators/mjwp.py",
            "spider/optimizers/sampling.py",
            "spider/optimizers/sampling_fast.py",
            "workspace/core4d/scripts/E152/build_axis1_hand_gate_manifest.py",
            "workspace/core4d/scripts/E152/check_hand_gate_preflight.py",
        ],
        cwd=REPO,
        check=True,
    )

    summary = {
        "rows": len(rows),
        "to_run": len(todo),
        "split_counts": split_counts,
        "gate_rows": len(gate_rows),
        "errors": errors,
    }
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    (RESULT_ROOT / "preflight_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if errors:
        raise SystemExit("E152 preflight failed:\n" + "\n".join(errors))
    print(f"E152 preflight OK: rows={len(rows)} to_run={len(todo)} split={split_counts}")


if __name__ == "__main__":
    main()
