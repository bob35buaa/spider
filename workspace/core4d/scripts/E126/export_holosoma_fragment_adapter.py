#!/usr/bin/env python3
"""Export E125 fragment-only rows into Holosoma motion adapter artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


REPO = Path(__file__).resolve().parents[4]
HOLOSOMA = Path("/home/ubuntu/Workspace/holosoma")
DEFAULT_INPUT = REPO / "workspace/core4d/results/E125/rl_hand_support_preflight/selected_preflight.tsv"
DEFAULT_OUT = REPO / "workspace/core4d/results/E126/holosoma_fragment_adapter_preflight"
DEFAULT_PYTHON = Path("/home/ubuntu/.holosoma_deps/miniconda3/envs/hsretargeting/bin/python")
CONVERT_SCRIPT = HOLOSOMA / "src/holosoma_retargeting/holosoma_retargeting/data_conversion/convert_data_format_mj.py"
CONVERT_CWD = HOLOSOMA / "src/holosoma_retargeting/holosoma_retargeting"
WRIST_NAMES = ["left_wrist_yaw_link", "right_wrist_yaw_link"]
DEFAULT_PAIR = ("box021_035_p1", "box021_035_p2")


@dataclass
class AdapterRow:
    case_id: str
    partner_case_id: str
    experiment: str
    variant: str
    source_decision: str
    rl_train_allowed: str
    object_name: str
    input_qpos43_npz: str
    stripped_converter_input_npz: str
    single_export_npz: str
    paired_export_npz: str
    partner_source_export_npz: str
    input_frames: int
    input_fps: float
    output_frames: int
    output_fps: float
    paired_frames: int
    nan_count: int
    partner_left_dist_min_m: float
    partner_left_dist_mean_m: float
    partner_left_dist_max_m: float
    partner_right_dist_min_m: float
    partner_right_dist_mean_m: float
    partner_right_dist_max_m: float
    alignment_policy: str
    adapter_status: str
    failure_mode: str
    notes: str


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def repo_path(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else REPO / path


def safe_id(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in text).strip("_")


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def write_tsv(path: Path, rows: list[AdapterRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(AdapterRow.__dataclass_fields__), delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def load_qpos43(path: Path) -> tuple[np.ndarray, float]:
    with np.load(path, allow_pickle=True) as data:
        qpos = np.asarray(data["qpos"], dtype=np.float64)
        fps_arr = np.asarray(data["fps"]) if "fps" in data.files else np.asarray(30.0)
    if qpos.ndim != 2 or qpos.shape[1] != 43:
        raise ValueError(f"{path}: expected qpos shape (T,43), got {qpos.shape}")
    if not np.isfinite(qpos).all():
        raise ValueError(f"{path}: qpos contains non-finite values")
    fps = float(fps_arr.reshape(-1)[0]) if fps_arr.size else 30.0
    return qpos, fps


def save_converter_input_without_fps(qpos: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Holosoma converter currently treats a stored fps key as dt. Use CLI fps.
    np.savez(path, qpos=qpos.astype(np.float32))


def run_convert(
    *,
    python: Path,
    input_file: Path,
    output_file: Path,
    object_name: str,
    input_fps: int,
    output_fps: int,
    log_path: Path,
    force: bool,
) -> None:
    if output_file.is_file() and not force:
        return
    cmd = [
        str(python),
        str(CONVERT_SCRIPT),
        "--input-file",
        str(input_file),
        "--input-fps",
        str(input_fps),
        "--output-fps",
        str(output_fps),
        "--has-dynamic-object",
        "--object-name",
        object_name,
        "--output-name",
        str(output_file),
        "--once",
    ]
    started = time.perf_counter()
    proc = subprocess.run(cmd, cwd=CONVERT_CWD, text=True, capture_output=True, check=False)
    elapsed = time.perf_counter() - started
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write("\n" + "=" * 96 + "\n")
        f.write(" ".join(cmd) + f"\nreturncode={proc.returncode} elapsed={elapsed:.3f}s\n")
        f.write("\n[stdout]\n" + proc.stdout)
        f.write("\n[stderr]\n" + proc.stderr)
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout)[-1600:].strip()
        raise RuntimeError(tail or f"convert failed: {input_file}")


def load_motion(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def normalize_names(arr: np.ndarray) -> list[str]:
    out: list[str] = []
    for item in list(arr):
        if isinstance(item, bytes):
            out.append(item.decode("utf-8"))
        else:
            out.append(str(item))
    return out


def crop_motion(motion: dict[str, Any], frames: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in motion.items():
        arr = np.asarray(value)
        if arr.ndim > 0 and arr.shape[0] > frames and np.issubdtype(arr.dtype, np.number):
            out[key] = arr[:frames]
        elif arr.ndim > 0 and arr.shape[0] == frames and np.issubdtype(arr.dtype, np.number):
            out[key] = arr
        elif arr.ndim > 0 and arr.shape[0] > frames and key in {"body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"}:
            out[key] = arr[:frames]
        else:
            out[key] = value
    return out


def add_partner(base_path: Path, partner_path: Path, out_path: Path) -> dict[str, float | int]:
    base = load_motion(base_path)
    partner = load_motion(partner_path)
    body_names = normalize_names(np.asarray(partner["body_names"]))
    wrist_indices = [body_names.index(name) for name in WRIST_NAMES]

    base_frames = int(np.asarray(base["joint_pos"]).shape[0])
    partner_frames = int(np.asarray(partner["joint_pos"]).shape[0])
    frames = min(base_frames, partner_frames)
    if frames < 2:
        raise ValueError(f"invalid paired frame count: base={base_frames}, partner={partner_frames}")
    base = crop_motion(base, frames)
    partner_pos = np.asarray(partner["body_pos_w"][:frames, wrist_indices, :], dtype=np.float64)
    partner_quat = np.asarray(partner["body_quat_w"][:frames, wrist_indices, :], dtype=np.float64)

    obj = np.asarray(base["object_pos_w"][:frames], dtype=np.float64)
    left_dist = np.linalg.norm(partner_pos[:, 0, :] - obj, axis=-1)
    right_dist = np.linalg.norm(partner_pos[:, 1, :] - obj, axis=-1)
    base["partner_hand_pos_w"] = partner_pos
    base["partner_hand_quat_w"] = partner_quat
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **base)
    return {
        "paired_frames": frames,
        "partner_left_dist_min_m": float(left_dist.min()),
        "partner_left_dist_mean_m": float(left_dist.mean()),
        "partner_left_dist_max_m": float(left_dist.max()),
        "partner_right_dist_min_m": float(right_dist.min()),
        "partner_right_dist_mean_m": float(right_dist.mean()),
        "partner_right_dist_max_m": float(right_dist.max()),
    }


def inspect_paired(path: Path) -> tuple[int, int, float]:
    required = [
        "fps",
        "joint_pos",
        "joint_vel",
        "body_pos_w",
        "body_quat_w",
        "object_pos_w",
        "object_quat_w",
        "partner_hand_pos_w",
        "partner_hand_quat_w",
    ]
    with np.load(path, allow_pickle=True) as data:
        missing = [key for key in required if key not in data.files]
        if missing:
            raise ValueError(f"{path}: missing keys {missing}")
        frames = int(data["joint_pos"].shape[0])
        if data["joint_pos"].ndim != 2 or data["joint_pos"].shape[1] != 36:
            raise ValueError(f"{path}: bad joint_pos shape {data['joint_pos'].shape}")
        if data["joint_vel"].ndim != 2 or data["joint_vel"].shape[1] != 35:
            raise ValueError(f"{path}: bad joint_vel shape {data['joint_vel'].shape}")
        if tuple(data["partner_hand_pos_w"].shape) != (frames, 2, 3):
            raise ValueError(f"{path}: bad partner_hand_pos_w shape {data['partner_hand_pos_w'].shape}")
        if tuple(data["partner_hand_quat_w"].shape) != (frames, 2, 4):
            raise ValueError(f"{path}: bad partner_hand_quat_w shape {data['partner_hand_quat_w'].shape}")
        nan_count = 0
        for key in data.files:
            arr = np.asarray(data[key])
            if np.issubdtype(arr.dtype, np.number):
                nan_count += int(np.isnan(arr).sum())
        fps = float(np.asarray(data["fps"]).reshape(-1)[0])
    return frames, nan_count, fps


def row_by_case(rows: list[dict[str, str]], case_ids: tuple[str, str]) -> dict[str, dict[str, str]]:
    out = {row["case_id"]: row for row in rows if row.get("case_id") in case_ids}
    missing = [case_id for case_id in case_ids if case_id not in out]
    if missing:
        raise ValueError(f"missing selected_preflight rows: {missing}")
    for case_id, row in out.items():
        if row.get("decision") != "FRAGMENT_HOLDOUT_ONLY":
            raise ValueError(f"{case_id}: expected FRAGMENT_HOLDOUT_ONLY, got {row.get('decision')}")
    return out


def export_pair(args: argparse.Namespace) -> list[AdapterRow]:
    source_rows = row_by_case(read_tsv(args.input_tsv), tuple(args.case_ids))
    qpos_data: dict[str, tuple[np.ndarray, float]] = {}
    single_exports: dict[str, Path] = {}
    stripped_inputs: dict[str, Path] = {}
    converter_dir = args.out_dir / "converter_inputs"
    export_dir = args.out_dir / "exports"
    for case_id in args.case_ids:
        source = source_rows[case_id]
        qpos_path = repo_path(source["converter_input_npz"])
        qpos, fps = load_qpos43(qpos_path)
        qpos_data[case_id] = (qpos, fps)
        stem = safe_id(f"E126_{case_id}_{source['variant']}")
        stripped = converter_dir / f"{stem}_qpos43_no_fps.npz"
        export = export_dir / f"{stem}_mj_w_obj.npz"
        save_converter_input_without_fps(qpos, stripped)
        run_convert(
            python=args.python,
            input_file=stripped,
            output_file=export,
            object_name=args.object_name,
            input_fps=round(fps),
            output_fps=args.output_fps,
            log_path=args.out_dir / "logs/convert.log",
            force=args.force,
        )
        stripped_inputs[case_id] = stripped
        single_exports[case_id] = export

    rows: list[AdapterRow] = []
    for case_id in args.case_ids:
        partner_id = args.case_ids[1] if case_id == args.case_ids[0] else args.case_ids[0]
        source = source_rows[case_id]
        paired = export_dir / f"E126_{case_id}_with_partner_{partner_id}_mj_w_obj_w_partner.npz"
        stats = add_partner(single_exports[case_id], single_exports[partner_id], paired)
        paired_frames, nan_count, output_fps = inspect_paired(paired)
        qpos, input_fps = qpos_data[case_id]
        rows.append(
            AdapterRow(
                case_id=case_id,
                partner_case_id=partner_id,
                experiment=source["experiment"],
                variant=source["variant"],
                source_decision=source["decision"],
                rl_train_allowed="false",
                object_name=args.object_name,
                input_qpos43_npz=source["converter_input_npz"],
                stripped_converter_input_npz=rel(stripped_inputs[case_id]),
                single_export_npz=rel(single_exports[case_id]),
                paired_export_npz=rel(paired),
                partner_source_export_npz=rel(single_exports[partner_id]),
                input_frames=int(qpos.shape[0]),
                input_fps=float(input_fps),
                output_frames=int(load_motion(single_exports[case_id])["joint_pos"].shape[0]),
                output_fps=output_fps,
                paired_frames=paired_frames,
                nan_count=nan_count,
                partner_left_dist_min_m=float(stats["partner_left_dist_min_m"]),
                partner_left_dist_mean_m=float(stats["partner_left_dist_mean_m"]),
                partner_left_dist_max_m=float(stats["partner_left_dist_max_m"]),
                partner_right_dist_min_m=float(stats["partner_right_dist_min_m"]),
                partner_right_dist_mean_m=float(stats["partner_right_dist_mean_m"]),
                partner_right_dist_max_m=float(stats["partner_right_dist_max_m"]),
                alignment_policy="min_frames_head_crop_fragment_pair_no_raw_window",
                adapter_status="pass" if nan_count == 0 else "fail",
                failure_mode="" if nan_count == 0 else "nan_in_paired_export",
                notes="fragment-only adapter preflight; not RL-ready evidence",
            )
        )
    return rows


def write_summary(out_dir: Path, rows: list[AdapterRow]) -> None:
    counts = Counter(row.adapter_status for row in rows)
    summary = {
        "stage": "E126_holosoma_fragment_adapter_preflight",
        "rows": len(rows),
        "adapter_status_counts": dict(counts),
        "rl_ready_rows": 0,
        "training_launched": False,
        "outputs": {
            "manifest": rel(out_dir / "e126_adapter_manifest.tsv"),
            "summary_md": rel(out_dir / "e126_adapter_summary.md"),
            "exports_dir": rel(out_dir / "exports"),
        },
    }
    (out_dir / "e126_adapter_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# E126 Holosoma Fragment Adapter Preflight Summary",
        "",
        f"- rows: `{len(rows)}`",
        f"- pass rows: `{counts.get('pass', 0)}`",
        "- RL-ready rows: `0`",
        "- training launched: `False`",
        "",
        "| case | partner | status | input frames | paired frames | output fps | partner L mean | partner R mean | paired export |",
        "|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| `{row.case_id}` | `{row.partner_case_id}` | `{row.adapter_status}` | "
            f"{row.input_frames} | {row.paired_frames} | {row.output_fps:.1f} | "
            f"{row.partner_left_dist_mean_m:.3f}m | {row.partner_right_dist_mean_m:.3f}m | "
            f"`{row.paired_export_npz}` |"
        )
    lines.extend(
        [
            "",
            "Interpretation: these files are adapter/reward-inspection artifacts only. They preserve the E125 `FRAGMENT_HOLDOUT_ONLY` label and do not solve the main `box021_029_p2` gate.",
        ]
    )
    (out_dir / "e126_adapter_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-tsv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--object-name", default="Box021")
    parser.add_argument("--output-fps", type=int, default=50)
    parser.add_argument("--case-ids", nargs=2, default=list(DEFAULT_PAIR))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    args.out_dir = args.out_dir if args.out_dir.is_absolute() else REPO / args.out_dir
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if not args.python.is_file():
        raise FileNotFoundError(args.python)
    if not CONVERT_SCRIPT.is_file():
        raise FileNotFoundError(CONVERT_SCRIPT)
    rows = export_pair(args)
    write_tsv(args.out_dir / "e126_adapter_manifest.tsv", rows)
    write_summary(args.out_dir, rows)
    failures = sum(row.adapter_status != "pass" for row in rows)
    print(f"wrote {rel(args.out_dir / 'e126_adapter_summary.md')} rows={len(rows)} failures={failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
