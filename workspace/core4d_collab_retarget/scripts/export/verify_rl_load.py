#!/usr/bin/env python3
"""Validate exported Holosoma RL npz files without starting the RL trainer."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
EXP_WS = SCRIPT_DIR.parents[1]
DEFAULT_MANIFEST = SCRIPT_DIR / "manifest_rl.tsv"
DEFAULT_OUTPUT_DIR = Path("/home/ubuntu/Workspace/holosoma/workspace/data/spider_best_E018b_E022_E025_for_rl_rename")
DEFAULT_CONVERSION_LOG = EXP_WS / "results/E021_rl_export_manifest_rename/conversion_log.csv"
DEFAULT_VERIFY_CSV = EXP_WS / "results/E021_rl_export_manifest_rename/verify_load.csv"
DEFAULT_VERIFY_JSON = EXP_WS / "results/E021_rl_export_manifest_rename/verify_load.json"

REQUIRED_KEYS = (
    "fps",
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
    "object_pos_w",
    "object_quat_w",
    "object_lin_vel_w",
    "object_ang_vel_w",
    "joint_names",
    "body_names",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--conversion-log", type=Path, default=DEFAULT_CONVERSION_LOG)
    parser.add_argument("--verify-csv", type=Path, default=DEFAULT_VERIFY_CSV)
    parser.add_argument("--verify-json", type=Path, default=DEFAULT_VERIFY_JSON)
    return parser.parse_args()


def read_conversion_outputs(path: Path) -> dict[str, Path]:
    outputs: dict[str, Path] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row["status"] in {"ok", "skipped_existing"}:
                outputs[row["case"]] = Path(row["output_npz"])
    return outputs


def validate_file(path: Path) -> tuple[str, dict[str, object]]:
    info: dict[str, object] = {"path": str(path)}
    if not path.exists():
        return "missing", info

    with np.load(path, allow_pickle=True) as data:
        missing = [k for k in REQUIRED_KEYS if k not in data.files]
        if missing:
            info["missing_keys"] = missing
            return "bad", info

        joint_pos = data["joint_pos"]
        joint_vel = data["joint_vel"]
        object_pos = data["object_pos_w"]
        object_quat = data["object_quat_w"]
        body_pos = data["body_pos_w"]
        body_quat = data["body_quat_w"]
        fps = data["fps"].tolist()

        checks = [
            joint_pos.ndim == 2 and joint_pos.shape[1] == 36,
            joint_vel.ndim == 2 and joint_vel.shape[1] == 35,
            object_pos.shape == (joint_pos.shape[0], 3),
            object_quat.shape == (joint_pos.shape[0], 4),
            body_pos.ndim == 3 and body_pos.shape[0] == joint_pos.shape[0] and body_pos.shape[2] == 3,
            body_quat.ndim == 3 and body_quat.shape[0] == joint_pos.shape[0] and body_quat.shape[2] == 4,
            int(np.asarray(data["fps"]).reshape(-1)[0]) == 50,
        ]
        info.update(
            {
                "fps": fps,
                "frames": int(joint_pos.shape[0]),
                "joint_pos_shape": list(joint_pos.shape),
                "joint_vel_shape": list(joint_vel.shape),
                "body_pos_w_shape": list(body_pos.shape),
                "object_pos_w_shape": list(object_pos.shape),
                "joint_names": int(len(data["joint_names"])),
                "body_names": int(len(data["body_names"])),
            }
        )
        return ("ok" if all(checks) else "bad"), info


def main() -> int:
    args = parse_args()
    outputs = read_conversion_outputs(args.conversion_log)
    rows: list[dict[str, object]] = []
    failures = 0

    with args.manifest.open(newline="") as f:
        for man in csv.DictReader(f, delimiter="\t"):
            case = man["case"]
            output_name = f"{man.get('original_prefix', man['selected_variant'])}_v2_mj_w_obj.npz"
            output = outputs.get(case, args.output_dir / output_name)
            status, info = validate_file(output)
            if status != "ok":
                failures += 1
            rows.append(
                {
                    "case": case,
                    "selected_variant": man["selected_variant"],
                    "object_name": man["object_name"],
                    "original_prefix": man.get("original_prefix", ""),
                    "recommended_for_rl": man["recommended_for_rl"],
                    "status": status,
                    **info,
                }
            )

    args.verify_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.verify_csv.open("w", newline="") as f:
        fieldnames = [
            "case",
            "selected_variant",
            "object_name",
            "original_prefix",
            "recommended_for_rl",
            "status",
            "path",
            "fps",
            "frames",
            "joint_pos_shape",
            "joint_vel_shape",
            "body_pos_w_shape",
            "object_pos_w_shape",
            "joint_names",
            "body_names",
            "missing_keys",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    summary = {"total": len(rows), "ok": len(rows) - failures, "failures": failures, "rows": rows}
    args.verify_json.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n")
    print(json.dumps({k: summary[k] for k in ("total", "ok", "failures")}, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
