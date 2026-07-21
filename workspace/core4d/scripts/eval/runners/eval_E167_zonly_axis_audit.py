#!/usr/bin/env python3
"""Axis-contract audit for E167 Holosoma z-only experiments."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from pathlib import Path
from typing import Any

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


REPO = Path(__file__).resolve().parents[5]
BUILDER = REPO / "workspace/core4d/scripts/experiments/E167/build_zonly_manifest.py"
VARIANTS = REPO / "workspace/core4d/scripts/experiments/E167/variants.tsv"
EVAL_ROOT = REPO / "workspace/core4d/results/E167/holosoma_zonly/eval/axis_audit"
TARGET_CASES = {
    "box023_person2",
    "box021_029_p2",
    "box021_035_p1",
    "box021_035_p2",
    "box004_082_p1",
    "box004_083_p1",
    "box004_083_p2",
}
ARMS = {"E167A", "E167A_B1", "E167A_B2"}
XY_TOL = 1e-6


def rel(path: str | Path) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(REPO.resolve()))
    except Exception:
        return str(path)


def repo_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO / p


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_yaml(path: Path) -> dict[str, Any]:
    if yaml is None or not path.is_file():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def close_enough(value: Any, target: float, tol: float = 1e-8) -> bool:
    try:
        return abs(float(value) - target) <= tol
    except Exception:
        return False


def boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def report_path(row: dict[str, str]) -> Path:
    out = repo_path(row["postprocess_output_npz"])
    return out.with_suffix(out.suffix + ".zonly_smooth_report.json")


def b2_report_metrics(row: dict[str, str], mode: str) -> dict[str, Any]:
    out = repo_path(row["postprocess_output_npz"])
    report = report_path(row)
    item: dict[str, Any] = {
        "postprocess_output_exists": out.is_file(),
        "zonly_report": rel(report),
        "zonly_report_exists": report.is_file(),
        "xy_max_abs_delta": "",
        "b2_xy_invariance_ok": mode != "full",
    }
    if report.is_file():
        data = json.loads(report.read_text(encoding="utf-8"))
        xy = float(data.get("xy_max_abs_delta", math.inf))
        item["xy_max_abs_delta"] = xy
        item["b2_xy_invariance_ok"] = xy <= XY_TOL
    elif mode == "full":
        item["b2_xy_invariance_ok"] = False
    return item


def audit_row(row: dict[str, str], mode: str) -> dict[str, Any]:
    item: dict[str, Any] = {
        "variant": row["variant"],
        "short_case_id": row["short_case_id"],
        "arm": row["arm"],
        "arm_kind": row["arm_kind"],
        "split": row["split"],
        "override": row["override"],
        "manifest_foot_slip_disabled": row["foot_slip_enabled"] == "false",
        "manifest_ankle_weight_neutral": close_enough(row["local_frame_ankle_weight"], 1.0),
        "manifest_b1_axis_z": row["arm"] != "E167A_B1" or row["cem_smooth_axis"] == "z",
        "manifest_b1_smooth_enabled": row["arm"] != "E167A_B1" or row["cem_smooth_enabled"] == "true",
        "manifest_a_smooth_disabled": row["arm"] != "E167A" or row["cem_smooth_enabled"] == "false",
    }
    if row["arm_kind"] == "cem":
        override = repo_path(row["override"])
        cfg = load_yaml(override)
        item.update(
            {
                "override_exists": override.is_file(),
                "override_e167_body_z_enabled": boolish(cfg.get("e167_body_z_enabled")),
                "override_e167_ground_z_enabled": boolish(cfg.get("e167_ground_z_enabled")),
                "override_foot_slip_disabled": not boolish(cfg.get("foot_slip_enabled")),
                "override_foot_ground_disabled": not boolish(cfg.get("foot_ground_enabled")),
                "override_ankle_weight_neutral": close_enough(cfg.get("local_frame_ankle_weight"), 1.0),
                "override_b1_axis_z": row["arm"] != "E167A_B1" or cfg.get("cem_smooth_axis") == "z",
                "override_b1_smooth_enabled": row["arm"] != "E167A_B1" or boolish(cfg.get("cem_smooth_enabled")),
                "override_a_smooth_disabled": row["arm"] != "E167A" or not boolish(cfg.get("cem_smooth_enabled")),
            }
        )
        if mode == "full":
            item.update(
                {
                    "result_npz_exists": repo_path(row["result_npz"]).is_file(),
                    "outdir_npz_exists": repo_path(row["outdir_npz"]).is_file(),
                    "video_exists": repo_path(row["video"]).is_file(),
                }
            )
    else:
        item.update(b2_report_metrics(row, mode))
    bool_keys = [
        key
        for key in item
        if key.startswith("manifest_")
        or key.startswith("override_")
        or key.endswith("_ok")
        or (
            key.endswith("_exists")
            and not (
                mode == "preflight"
                and row["arm_kind"] == "postprocess"
                and key in {"postprocess_output_exists", "zonly_report_exists"}
            )
        )
    ]
    item["axis_audit_ok"] = all(bool(item[key]) for key in bool_keys)
    return item


def run(mode: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    subprocess.run([str(REPO / ".venv/bin/python"), str(BUILDER)], cwd=REPO, check=True)
    rows = read_tsv(VARIANTS)
    audit = [audit_row(row, mode) for row in rows]
    cases = {row["short_case_id"] for row in rows}
    arms = {row["arm"] for row in rows}
    structural = {
        "rows_21": len(rows) == 21,
        "cases_ok": cases == TARGET_CASES,
        "arms_ok": arms == ARMS,
        "box026_absent": all("box026" not in row["short_case_id"] and "box026" not in row["variant"] for row in rows),
    }
    summary = {
        "mode": mode,
        "rows": len(rows),
        "cases": sorted(cases),
        "arms": sorted(arms),
        **structural,
        "row_audit_pass": sum(1 for row in audit if row["axis_audit_ok"]),
        "row_audit_total": len(audit),
        "b2_reports_present": sum(1 for row in audit if row.get("zonly_report_exists") is True),
    }
    summary["axis_audit_pass"] = all(structural.values()) and summary["row_audit_pass"] == summary["row_audit_total"]
    return audit, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", nargs="?", default="preflight", choices=["preflight", "full"])
    args = parser.parse_args()
    audit, summary = run(args.mode)
    out_dir = EVAL_ROOT / args.mode
    write_tsv(out_dir / "axis_audit.tsv", audit)
    write_json(out_dir / "axis_audit_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    if not summary["axis_audit_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
