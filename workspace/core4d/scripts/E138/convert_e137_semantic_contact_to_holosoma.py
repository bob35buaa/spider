#!/usr/bin/env python3
"""Convert E137 qpos-style semantic contact files into Holosoma WBT motions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


REPO = Path(__file__).resolve().parents[4]
HOLOSOMA = Path(os.environ.get("HOLOSOMA_ROOT", "/home/ubuntu/Workspace/holosoma"))
DEFAULT_CONVERTER_PY = Path(
    os.environ.get(
        "HOLOSOMA_CONVERTER_PY",
        "/home/ubuntu/.holosoma_deps/miniconda3/envs/hsretargeting/bin/python",
    )
)
CONVERT_SCRIPT = (
    HOLOSOMA
    / "src/holosoma_retargeting/holosoma_retargeting/data_conversion/convert_data_format_mj.py"
)
CONVERT_CWD = HOLOSOMA / "src/holosoma_retargeting/holosoma_retargeting"
E137_DIR = REPO / "workspace/core4d/results/E137/e107_semantic_object_contact_export"
E137_MANIFEST = E137_DIR / "e137_e107_semantic_object_contact_manifest.tsv"
OUT_DIR = REPO / "workspace/core4d/results/E138/e137_semantic_contact_converter_preflight"
MANIFEST = OUT_DIR / "e138_manifest.tsv"
SUMMARY_JSON = OUT_DIR / "e138_summary.json"
SUMMARY_MD = OUT_DIR / "e138_summary.md"

FIELDS = [
    "case_id",
    "raw_case_id",
    "person",
    "source_export_kind",
    "e137_npz",
    "input_fps",
    "output_fps",
    "input_frames",
    "converter_output_frames",
    "time_grid_expected_frames",
    "time_grid_match",
    "contact_mapping_policy",
    "source_index_min",
    "source_index_max",
    "source_index_sha256",
    "converter_input_npz",
    "converted_raw_npz",
    "converted_contact_npz",
    "converted_contact_sha256",
    "object_contact_shape",
    "object_contact_dtype",
    "left_active_frac",
    "right_active_frac",
    "both_active_frac",
    "either_active_frac",
    "required_keys_status",
    "finite_status",
    "convert_status",
    "inject_status",
    "motionloader_has_object",
    "motionloader_has_object_contact",
    "motionloader_contact_shape",
    "motionloader_status",
    "rl_ready",
    "training_launched",
    "cem_launched",
    "remote_jobs_launched",
    "notes",
]

REQUIRED_KEYS = [
    "fps",
    "joint_names",
    "body_names",
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
    "object_pos_w",
    "object_quat_w",
    "object_lin_vel_w",
]


def rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return str(p.relative_to(REPO))
    except ValueError:
        return str(p)


def as_abs(path: Path | str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO / p


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_array(arr: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(arr).view(np.uint8))
    return h.hexdigest()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIELDS})


def scalar_float(value: Any, default: float) -> float:
    if value is None:
        return default
    arr = np.asarray(value)
    if arr.size == 0:
        return default
    return float(arr.reshape(-1)[0])


def contact_stats(contact: np.ndarray) -> dict[str, str]:
    left = contact[:, 0].astype(bool)
    right = contact[:, 1].astype(bool)
    both = left & right
    either = left | right
    return {
        "left_active_frac": f"{float(np.mean(left)):.6f}",
        "right_active_frac": f"{float(np.mean(right)):.6f}",
        "both_active_frac": f"{float(np.mean(both)):.6f}",
        "either_active_frac": f"{float(np.mean(either)):.6f}",
    }


def converter_times(input_frames: int, input_fps: float, output_fps: float) -> np.ndarray:
    duration = (input_frames - 1) / input_fps
    return np.arange(0.0, duration, 1.0 / output_fps, dtype=np.float64)


def map_contact_to_converter_grid(
    contact: np.ndarray,
    *,
    input_fps: float,
    output_fps: float,
    output_frames: int,
) -> tuple[np.ndarray, np.ndarray, int, bool]:
    input_frames = int(contact.shape[0])
    times = converter_times(input_frames, input_fps, output_fps)
    expected_frames = int(times.shape[0])
    if expected_frames != output_frames:
        return np.zeros((0, 2), dtype=np.bool_), np.zeros((0,), dtype=np.int64), expected_frames, False
    duration = (input_frames - 1) / input_fps
    if duration <= 0:
        source_idx = np.zeros(output_frames, dtype=np.int64)
    else:
        raw_float = (times / duration) * float(input_frames - 1)
        source_idx = np.floor(raw_float + 0.5).astype(np.int64)
        source_idx = np.clip(source_idx, 0, input_frames - 1)
    return contact[source_idx].astype(np.bool_), source_idx, expected_frames, True


def run_converter(
    *,
    converter_py: Path,
    input_npz: Path,
    output_npz: Path,
    input_fps: int,
    output_fps: int,
    log_path: Path,
) -> tuple[bool, str]:
    cmd = [
        str(converter_py),
        str(CONVERT_SCRIPT),
        "--input-file",
        str(input_npz),
        "--input-fps",
        str(input_fps),
        "--output-fps",
        str(output_fps),
        "--has-dynamic-object",
        "--object-name",
        "Box021",
        "--output-name",
        str(output_npz),
        "--once",
    ]
    env = os.environ.copy()
    env.pop("DISPLAY", None)
    env.pop("WAYLAND_DISPLAY", None)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("COMMAND " + " ".join(cmd) + "\n\n")
        proc = subprocess.run(
            cmd,
            cwd=str(CONVERT_CWD),
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        log.write(proc.stdout)
        log.write(f"\nRETURN_CODE {proc.returncode}\n")
    if proc.returncode != 0:
        return False, f"converter_return_code={proc.returncode}; log={rel(log_path)}"
    if not output_npz.exists():
        return False, f"converter_output_missing; log={rel(log_path)}"
    return True, f"converter_pass; log={rel(log_path)}"


def validate_required(data: dict[str, np.ndarray]) -> tuple[str, str]:
    missing = [key for key in REQUIRED_KEYS if key not in data]
    if missing:
        return "fail", "missing_required_keys=" + ",".join(missing)
    numeric_keys = [
        "joint_pos",
        "joint_vel",
        "body_pos_w",
        "body_quat_w",
        "body_lin_vel_w",
        "body_ang_vel_w",
        "object_pos_w",
        "object_quat_w",
        "object_lin_vel_w",
    ]
    nonfinite = [key for key in numeric_keys if not np.all(np.isfinite(data[key]))]
    if nonfinite:
        return "pass", "nonfinite_keys=" + ",".join(nonfinite)
    return "pass", ""


def convert_stage(args: argparse.Namespace) -> list[dict[str, Any]]:
    if not E137_MANIFEST.exists():
        raise FileNotFoundError(E137_MANIFEST)
    if not CONVERT_SCRIPT.exists():
        raise FileNotFoundError(CONVERT_SCRIPT)
    if not args.converter_py.exists():
        raise FileNotFoundError(args.converter_py)

    rows = [row for row in read_tsv(E137_MANIFEST) if row["source_export_kind"] == "E107_trimmed"]
    out_rows: list[dict[str, Any]] = []
    for src in rows:
        case_id = src["case_id"]
        e137_npz = as_abs(src["output_npz"])
        stem = f"{case_id}_e137_semantic"
        converter_input = OUT_DIR / "converter_inputs" / f"{stem}_qpos_only.npz"
        converted_raw = OUT_DIR / "converted_raw" / f"{stem}_mj_w_obj_raw.npz"
        converted_contact = OUT_DIR / "converted" / f"{stem}_mj_w_obj_object_contact.npz"
        log_path = OUT_DIR / "logs" / f"{stem}_convert.log"
        base: dict[str, Any] = {
            "case_id": case_id,
            "raw_case_id": src["raw_case_id"],
            "person": src["person"],
            "source_export_kind": src["source_export_kind"],
            "e137_npz": rel(e137_npz),
            "output_fps": args.output_fps,
            "contact_mapping_policy": "converter_time_grid_nearest_source_frame",
            "rl_ready": "false",
            "training_launched": "false",
            "cem_launched": "false",
            "remote_jobs_launched": "false",
        }
        try:
            with np.load(e137_npz, allow_pickle=True) as e137:
                qpos = np.asarray(e137["qpos"])
                contact = np.asarray(e137["object_contact"], dtype=np.bool_)
                contact_3cm = np.asarray(e137["object_contact_3cm"], dtype=np.bool_)
                contact_5cm = np.asarray(e137["object_contact_5cm"], dtype=np.bool_)
                input_fps = scalar_float(e137.get("fps"), float(args.input_fps))
            if qpos.ndim != 2 or qpos.shape[1] != 43:
                raise ValueError(f"expected qpos shape (T,43), got {qpos.shape}")
            if contact.shape != (qpos.shape[0], 2):
                raise ValueError(f"expected object_contact {(qpos.shape[0], 2)}, got {contact.shape}")
            converter_input.parent.mkdir(parents=True, exist_ok=True)
            np.savez(converter_input, qpos=qpos)
            ok, note = run_converter(
                converter_py=args.converter_py,
                input_npz=converter_input,
                output_npz=converted_raw,
                input_fps=int(round(input_fps)),
                output_fps=args.output_fps,
                log_path=log_path,
            )
            base.update(
                {
                    "input_fps": f"{input_fps:.6f}",
                    "input_frames": int(qpos.shape[0]),
                    "converter_input_npz": rel(converter_input),
                    "converted_raw_npz": rel(converted_raw),
                    "convert_status": "pass" if ok else "fail",
                    "notes": note,
                }
            )
            if not ok:
                out_rows.append(base)
                continue

            with np.load(converted_raw, allow_pickle=True) as converted:
                payload = {key: converted[key] for key in converted.files}
            converted_frames = int(payload["joint_pos"].shape[0])
            mapped_contact, source_idx, expected_frames, grid_match = map_contact_to_converter_grid(
                contact,
                input_fps=input_fps,
                output_fps=float(args.output_fps),
                output_frames=converted_frames,
            )
            mapped_3cm, _, _, _ = map_contact_to_converter_grid(
                contact_3cm,
                input_fps=input_fps,
                output_fps=float(args.output_fps),
                output_frames=converted_frames,
            )
            mapped_5cm, _, _, _ = map_contact_to_converter_grid(
                contact_5cm,
                input_fps=input_fps,
                output_fps=float(args.output_fps),
                output_frames=converted_frames,
            )
            required_status, finite_note = validate_required(payload)
            finite_status = "fail" if finite_note.startswith("nonfinite") else "pass"
            inject_status = "pass"
            notes = [note]
            if not grid_match:
                inject_status = "fail"
                notes.append("time_grid_frame_mismatch")
            if required_status != "pass":
                inject_status = "fail"
                notes.append("required_key_failure")
            if finite_status != "pass":
                inject_status = "fail"
                notes.append(finite_note)
            if inject_status == "pass":
                payload.update(
                    {
                        "object_contact": mapped_contact,
                        "object_contact_3cm": mapped_3cm,
                        "object_contact_5cm": mapped_5cm,
                        "object_contact_source": np.array("E137_E135_semantic_contact_3cm"),
                        "object_contact_source_npz": np.array(rel(e137_npz)),
                        "object_contact_mapping_policy": np.array(
                            "converter_time_grid_nearest_source_frame"
                        ),
                        "object_contact_source_frame_index": source_idx.astype(np.int32),
                        "object_contact_input_fps": np.array(input_fps, dtype=np.float32),
                        "object_contact_output_fps": np.array(float(args.output_fps), dtype=np.float32),
                        "object_contact_input_frames": np.array(qpos.shape[0], dtype=np.int32),
                        "object_contact_output_frames": np.array(converted_frames, dtype=np.int32),
                        "object_contact_note": np.array(
                            "E138 post-injected after Holosoma qpos-to-WBT conversion"
                        ),
                    }
                )
                converted_contact.parent.mkdir(parents=True, exist_ok=True)
                np.savez(converted_contact, **payload)
                converted_sha = sha256(converted_contact)
            else:
                converted_sha = ""
            mapped_stats = contact_stats(mapped_contact) if mapped_contact.size else {}
            base.update(
                {
                    "converter_output_frames": converted_frames,
                    "time_grid_expected_frames": expected_frames,
                    "time_grid_match": str(bool(grid_match)).lower(),
                    "source_index_min": int(source_idx.min()) if source_idx.size else "",
                    "source_index_max": int(source_idx.max()) if source_idx.size else "",
                    "source_index_sha256": sha256_array(source_idx) if source_idx.size else "",
                    "converted_contact_npz": rel(converted_contact) if inject_status == "pass" else "",
                    "converted_contact_sha256": converted_sha,
                    "object_contact_shape": "x".join(str(x) for x in mapped_contact.shape),
                    "object_contact_dtype": str(mapped_contact.dtype),
                    "required_keys_status": required_status,
                    "finite_status": finite_status,
                    "inject_status": inject_status,
                    "notes": "; ".join(x for x in notes if x),
                    **mapped_stats,
                }
            )
        except Exception as exc:  # noqa: BLE001 - manifest should capture row-level failure.
            base.update(
                {
                    "convert_status": "fail",
                    "inject_status": "fail",
                    "notes": f"{type(exc).__name__}: {exc}",
                }
            )
        out_rows.append(base)
    write_tsv(MANIFEST, out_rows)
    write_summary(out_rows, stage="convert")
    return out_rows


def decode_names(arr: np.ndarray) -> list[str]:
    out = []
    for item in arr.tolist():
        out.append(item.decode("utf-8") if isinstance(item, bytes) else str(item))
    return out


def motionloader_stage() -> list[dict[str, Any]]:
    from holosoma.managers.command.terms.wbt import MotionLoader

    rows = read_tsv(MANIFEST)
    out_rows: list[dict[str, Any]] = []
    for row in rows:
        row = dict(row)
        if row.get("inject_status") != "pass":
            row["motionloader_status"] = "not_run"
            out_rows.append(row)
            continue
        path = as_abs(row["converted_contact_npz"])
        try:
            with np.load(path, allow_pickle=True) as data:
                body_names = decode_names(data["body_names"])
                joint_names = decode_names(data["joint_names"])
                frames = int(data["joint_pos"].shape[0])
            motion = MotionLoader(str(path), body_names, joint_names, device="cpu")
            contact = motion.object_contact.cpu().numpy().astype(bool)
            status = "pass"
            if not bool(motion.has_object):
                status = "fail"
            if not bool(motion.has_object_contact):
                status = "fail"
            if contact.shape != (frames, 2):
                status = "fail"
            if not bool(np.any(contact)):
                status = "fail"
            row.update(
                {
                    "motionloader_has_object": str(bool(motion.has_object)).lower(),
                    "motionloader_has_object_contact": str(bool(motion.has_object_contact)).lower(),
                    "motionloader_contact_shape": "x".join(str(x) for x in contact.shape),
                    "motionloader_status": status,
                    "rl_ready": "false",
                }
            )
        except Exception as exc:  # noqa: BLE001 - manifest should capture row-level failure.
            row.update(
                {
                    "motionloader_status": "fail",
                    "notes": (row.get("notes", "") + f"; motionloader {type(exc).__name__}: {exc}").strip("; "),
                }
            )
        out_rows.append(row)
    write_tsv(MANIFEST, out_rows)
    write_summary(out_rows, stage="motionloader")
    return out_rows


def write_summary(rows: list[dict[str, Any]], *, stage: str) -> None:
    convert_pass = sum(1 for row in rows if row.get("convert_status") == "pass")
    inject_pass = sum(1 for row in rows if row.get("inject_status") == "pass")
    loader_values = [row.get("motionloader_status", "") for row in rows]
    loader_pass = sum(1 for value in loader_values if value == "pass")
    loader_ran = any(value for value in loader_values)
    status = "pass" if rows and inject_pass == len(rows) and (not loader_ran or loader_pass == len(rows)) else "fail"
    summary = {
        "experiment": "E138",
        "stage": stage,
        "status": status,
        "rows": len(rows),
        "trimmed_rows": len(rows),
        "convert_pass_rows": convert_pass,
        "inject_pass_rows": inject_pass,
        "motionloader_pass_rows": loader_pass,
        "motionloader_ran": loader_ran,
        "output_fps": 50,
        "mapping_policy": "converter_time_grid_nearest_source_frame",
        "rl_ready_rows": 0,
        "training_launched": False,
        "cem_launched": False,
        "remote_jobs_launched": False,
        "notes": [
            "E138 converts trimmed E137 qpos-style semantic contact files into Holosoma WBT format.",
            "object_contact is post-injected after conversion because the converter drops extra keys.",
            "No CEM, PPO, remote jobs, or checkpoint creation was launched.",
        ],
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# E138 E137 Semantic Contact Converter Preflight Summary",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| rows | {summary['rows']} |",
        f"| convert pass rows | {summary['convert_pass_rows']} |",
        f"| inject pass rows | {summary['inject_pass_rows']} |",
        f"| MotionLoader pass rows | {summary['motionloader_pass_rows']} |",
        "| RL-ready rows | 0 |",
        "| training launched | false |",
        "| CEM launched | false |",
        "| remote jobs launched | false |",
        "",
        "## Rows",
        "",
        "| case | input frames | converted frames | contact shape | convert | inject | MotionLoader | either active |",
        "|---|---:|---:|---|---|---|---|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row.get('case_id', '')}`",
                    str(row.get("input_frames", "")),
                    str(row.get("converter_output_frames", "")),
                    f"`{row.get('object_contact_shape', '')}`",
                    f"`{row.get('convert_status', '')}`",
                    f"`{row.get('inject_status', '')}`",
                    f"`{row.get('motionloader_status', '')}`",
                    str(row.get("either_active_frac", "")),
                ]
            )
            + " |"
        )
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["convert", "motionloader", "all"], default="convert")
    parser.add_argument("--input-fps", type=int, default=30)
    parser.add_argument("--output-fps", type=int, default=50)
    parser.add_argument("--converter-py", type=Path, default=DEFAULT_CONVERTER_PY)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.stage in {"convert", "all"}:
        convert_stage(args)
    if args.stage in {"motionloader", "all"}:
        motionloader_stage()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
