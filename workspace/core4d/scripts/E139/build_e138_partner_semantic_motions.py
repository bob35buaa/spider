#!/usr/bin/env python3
"""Build partner-injected E138 semantic WBT motions for E139 env probing."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


REPO = Path(__file__).resolve().parents[4]
E138_DIR = REPO / "workspace/core4d/results/E138/e137_semantic_contact_converter_preflight"
E138_MANIFEST = E138_DIR / "e138_manifest.tsv"
OUT_DIR = REPO / "workspace/core4d/results/E139/e138_semantic_ref_object_contact_env_probe"
EXPORT_DIR = OUT_DIR / "partner_semantic_motions"
MANIFEST = OUT_DIR / "e139_partner_injection_manifest.tsv"
SUMMARY_JSON = OUT_DIR / "e139_partner_injection_summary.json"
WRIST_NAMES = ["left_wrist_yaw_link", "right_wrist_yaw_link"]

FIELDS = [
    "case_id",
    "partner_case_id",
    "base_motion_npz",
    "partner_motion_npz",
    "output_npz",
    "output_sha256",
    "frames",
    "partner_frames",
    "object_contact_shape",
    "partner_hand_pos_shape",
    "partner_hand_quat_shape",
    "alignment_policy",
    "base_raw_frame_min",
    "base_raw_frame_max",
    "partner_raw_frame_min",
    "partner_raw_frame_max",
    "nearest_raw_frame_diff_mean",
    "nearest_raw_frame_diff_max",
    "left_partner_dist_mean_m",
    "right_partner_dist_mean_m",
    "finite_status",
    "partner_injection_status",
    "rl_ready",
    "training_launched",
    "cem_launched",
    "remote_jobs_launched",
    "notes",
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


def load_npz(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def decode_names(arr: np.ndarray) -> list[str]:
    out = []
    for item in arr.tolist():
        out.append(item.decode("utf-8") if isinstance(item, bytes) else str(item))
    return out


def trim_start_from_e137(e138_motion: dict[str, Any]) -> int:
    source_npz = as_abs(str(np.asarray(e138_motion["object_contact_source_npz"]).reshape(-1)[0]))
    with np.load(source_npz, allow_pickle=True) as data:
        return int(np.asarray(data["object_contact_trim_start"]).reshape(-1)[0])


def raw_frame_index(e138_motion: dict[str, Any]) -> np.ndarray:
    trim_start = trim_start_from_e137(e138_motion)
    source_idx = np.asarray(e138_motion["object_contact_source_frame_index"], dtype=np.int64)
    return trim_start + source_idx


def nearest_indices(source_values: np.ndarray, target_values: np.ndarray) -> np.ndarray:
    source_values = np.asarray(source_values, dtype=np.int64)
    target_values = np.asarray(target_values, dtype=np.int64)
    positions = np.searchsorted(source_values, target_values, side="left")
    left = np.clip(positions - 1, 0, len(source_values) - 1)
    right = np.clip(positions, 0, len(source_values) - 1)
    choose_right = np.abs(source_values[right] - target_values) < np.abs(source_values[left] - target_values)
    return np.where(choose_right, right, left).astype(np.int64)


def pair_case_id(case_id: str) -> str:
    if case_id.endswith("_p1"):
        return case_id[:-2] + "p2"
    if case_id.endswith("_p2"):
        return case_id[:-2] + "p1"
    raise ValueError(f"cannot derive pair for case_id={case_id}")


def finite_status(payload: dict[str, Any]) -> str:
    for key, value in payload.items():
        arr = np.asarray(value)
        if np.issubdtype(arr.dtype, np.number) and not np.all(np.isfinite(arr)):
            return f"fail:{key}"
    return "pass"


def build_one(row: dict[str, str], rows_by_case: dict[str, dict[str, str]]) -> dict[str, Any]:
    case_id = row["case_id"]
    partner_case_id = pair_case_id(case_id)
    partner_row = rows_by_case[partner_case_id]
    base_path = as_abs(row["converted_contact_npz"])
    partner_path = as_abs(partner_row["converted_contact_npz"])
    base = load_npz(base_path)
    partner = load_npz(partner_path)
    base_raw = raw_frame_index(base)
    partner_raw = raw_frame_index(partner)
    partner_idx = nearest_indices(partner_raw, base_raw)
    body_names = decode_names(np.asarray(partner["body_names"]))
    wrist_indices = [body_names.index(name) for name in WRIST_NAMES]
    partner_pos = np.asarray(partner["body_pos_w"][partner_idx][:, wrist_indices, :], dtype=np.float64)
    partner_quat = np.asarray(partner["body_quat_w"][partner_idx][:, wrist_indices, :], dtype=np.float64)

    out = dict(base)
    out["partner_hand_pos_w"] = partner_pos
    out["partner_hand_quat_w"] = partner_quat
    out["partner_hand_source_case_id"] = np.array(partner_case_id)
    out["partner_hand_source_npz"] = np.array(rel(partner_path))
    out["partner_hand_alignment_policy"] = np.array("nearest_partner_converted_frame_by_raw_contact_frame")
    out["partner_hand_base_raw_frame_index"] = base_raw.astype(np.int32)
    out["partner_hand_source_frame_index"] = partner_idx.astype(np.int32)
    out["partner_hand_source_raw_frame_index"] = partner_raw[partner_idx].astype(np.int32)

    frames = int(np.asarray(base["joint_pos"]).shape[0])
    output = EXPORT_DIR / f"{case_id}_with_partner_{partner_case_id}_e138_semantic_mj_w_obj_w_partner.npz"
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, **out)

    raw_diff = np.abs(partner_raw[partner_idx] - base_raw)
    obj = np.asarray(base["object_pos_w"], dtype=np.float64)
    left_dist = np.linalg.norm(partner_pos[:, 0, :] - obj, axis=-1)
    right_dist = np.linalg.norm(partner_pos[:, 1, :] - obj, axis=-1)
    status = "pass"
    notes = "partner hands injected from paired E138 semantic motion by nearest raw-frame alignment"
    if partner_pos.shape != (frames, 2, 3) or partner_quat.shape != (frames, 2, 4):
        status = "fail"
        notes = "bad partner field shape"
    finite = finite_status(out)
    if finite != "pass":
        status = "fail"
    return {
        "case_id": case_id,
        "partner_case_id": partner_case_id,
        "base_motion_npz": rel(base_path),
        "partner_motion_npz": rel(partner_path),
        "output_npz": rel(output),
        "output_sha256": sha256(output),
        "frames": frames,
        "partner_frames": int(np.asarray(partner["joint_pos"]).shape[0]),
        "object_contact_shape": "x".join(str(x) for x in np.asarray(base["object_contact"]).shape),
        "partner_hand_pos_shape": "x".join(str(x) for x in partner_pos.shape),
        "partner_hand_quat_shape": "x".join(str(x) for x in partner_quat.shape),
        "alignment_policy": "nearest_partner_converted_frame_by_raw_contact_frame",
        "base_raw_frame_min": int(base_raw.min()),
        "base_raw_frame_max": int(base_raw.max()),
        "partner_raw_frame_min": int(partner_raw.min()),
        "partner_raw_frame_max": int(partner_raw.max()),
        "nearest_raw_frame_diff_mean": f"{float(raw_diff.mean()):.6f}",
        "nearest_raw_frame_diff_max": int(raw_diff.max()),
        "left_partner_dist_mean_m": f"{float(left_dist.mean()):.6f}",
        "right_partner_dist_mean_m": f"{float(right_dist.mean()):.6f}",
        "finite_status": finite,
        "partner_injection_status": status,
        "rl_ready": "false",
        "training_launched": "false",
        "cem_launched": "false",
        "remote_jobs_launched": "false",
        "notes": notes,
    }


def main() -> int:
    rows = [row for row in read_tsv(E138_MANIFEST) if row["source_export_kind"] == "E107_trimmed"]
    rows_by_case = {row["case_id"]: row for row in rows}
    out_rows = [build_one(row, rows_by_case) for row in rows]
    write_tsv(MANIFEST, out_rows)
    pass_rows = [row for row in out_rows if row["partner_injection_status"] == "pass"]
    summary = {
        "experiment": "E139",
        "stage": "partner_injection",
        "status": "pass" if len(pass_rows) == len(out_rows) == 4 else "fail",
        "rows": len(out_rows),
        "partner_injection_pass_rows": len(pass_rows),
        "alignment_policy": "nearest_partner_converted_frame_by_raw_contact_frame",
        "rl_ready_rows": 0,
        "training_launched": False,
        "cem_launched": False,
        "remote_jobs_launched": False,
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if summary["status"] != "pass":
        raise SystemExit(1)
    print(f"wrote {MANIFEST} rows={summary['rows']} status={summary['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
