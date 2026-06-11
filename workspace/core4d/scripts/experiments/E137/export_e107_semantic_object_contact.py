#!/usr/bin/env python3
"""Write E107 qpos-style semantic object_contact candidates from E135 masks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


REPO = Path(__file__).resolve().parents[4]
DEFAULT_E136_MANIFEST = (
    REPO
    / "workspace/core4d/results/E136/e135_semantic_contact_holosoma_bridge/"
    / "e136_e135_semantic_contact_bridge_manifest.tsv"
)
DEFAULT_OUT_DIR = REPO / "workspace/core4d/results/E137/e107_semantic_object_contact_export"
DEFAULT_LABEL = "3cm"
THRESHOLDS = ("3cm", "5cm")

MANIFEST_FIELDS = [
    "case_id",
    "raw_case_id",
    "person",
    "person_idx",
    "source_export_kind",
    "source_export_npz",
    "source_export_sha256",
    "source_frame_key",
    "source_frames",
    "mapping_type",
    "mapping_proof",
    "trim_window_json",
    "trim_start",
    "trim_end",
    "output_npz",
    "output_exists",
    "output_frames",
    "object_contact_shape",
    "object_contact_dtype",
    "object_contact_source",
    "object_contact_label",
    "object_contact_left_active_frac",
    "object_contact_right_active_frac",
    "object_contact_both_active_frac",
    "object_contact_3cm_shape",
    "object_contact_3cm_left_active_frac",
    "object_contact_3cm_right_active_frac",
    "object_contact_3cm_both_active_frac",
    "object_contact_5cm_shape",
    "object_contact_5cm_left_active_frac",
    "object_contact_5cm_right_active_frac",
    "object_contact_5cm_both_active_frac",
    "raw_contact_npz",
    "raw_contact_key_3cm",
    "raw_contact_key_5cm",
    "raw_frames",
    "semantic_export_status",
    "motionloader_runtime_probe_launched",
    "training_launched",
    "cem_launched",
    "remote_jobs_launched",
    "notes",
]


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def repo_path(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else REPO / path


def rel(path: Path | str) -> str:
    path_obj = Path(path)
    try:
        return str(path_obj.resolve().relative_to(REPO))
    except ValueError:
        return str(path_obj)


def slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frame_key(data: dict[str, np.ndarray]) -> tuple[str, int]:
    for key in ("joint_pos", "qpos", "body_pos_w", "object_pos_w", "object_pos"):
        if key in data:
            arr = np.asarray(data[key])
            if arr.ndim >= 1:
                return key, int(arr.shape[0])
    raise ValueError("source export has no supported frame key")


def load_npz_dict(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def fraction(mask: np.ndarray) -> str:
    if mask.size == 0:
        return ""
    return f"{float(mask.mean()):.6f}"


def contact_stats(contact: np.ndarray, prefix: str) -> dict[str, str]:
    if contact.ndim != 2 or contact.shape[1] != 2:
        raise ValueError(f"{prefix}: expected (T,2), got {contact.shape}")
    left = contact[:, 0]
    right = contact[:, 1]
    both = left & right
    return {
        f"{prefix}_shape": "x".join(map(str, contact.shape)),
        f"{prefix}_left_active_frac": fraction(left),
        f"{prefix}_right_active_frac": fraction(right),
        f"{prefix}_both_active_frac": fraction(both),
    }


def mapped_contact(row: dict[str, str]) -> np.ndarray:
    raw_path = repo_path(row["raw_contact_npz"])
    raw_key = row["raw_contact_key"]
    person_idx = int(row["person_idx"])
    with np.load(raw_path, allow_pickle=True) as data:
        raw = np.asarray(data[raw_key]).astype(bool)
    if raw.ndim != 3 or raw.shape[2] != 2:
        raise ValueError(f"{raw_path}:{raw_key}: expected (raw_frames, persons, hands), got {raw.shape}")
    if person_idx >= raw.shape[1]:
        raise ValueError(f"{raw_path}:{raw_key}: person_idx={person_idx} out of axis size {raw.shape[1]}")
    start = int(row["trim_start"] or 0)
    end = int(row["trim_end"] or raw.shape[0])
    if not (0 <= start < end <= raw.shape[0]):
        raise ValueError(f"{row['case_id']} {row['threshold_label']}: invalid trim slice {start}:{end} for raw frames {raw.shape[0]}")
    return raw[start:end, person_idx, :].astype(np.bool_)


def candidate_groups(rows: list[dict[str, str]]) -> dict[tuple[str, str, str], dict[str, dict[str, str]]]:
    groups: dict[tuple[str, str, str], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        if row.get("semantic_bridge_candidate") != "true":
            continue
        if not row.get("export_kind", "").startswith("E107_"):
            continue
        key = (row["case_id"], row["export_kind"], row["export_npz"])
        groups[key][row["threshold_label"]] = row
    return groups


def export_group(key: tuple[str, str, str], thresholds: dict[str, dict[str, str]], out_dir: Path) -> dict[str, Any]:
    missing = [label for label in THRESHOLDS if label not in thresholds]
    if missing:
        raise ValueError(f"{key}: missing threshold rows {missing}")
    row = thresholds[DEFAULT_LABEL]
    source_path = repo_path(row["export_npz"])
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    source_data = load_npz_dict(source_path)
    source_frame_key, source_frames = frame_key(source_data)

    contacts = {label: mapped_contact(thresholds[label]) for label in THRESHOLDS}
    for label, contact in contacts.items():
        if int(contact.shape[0]) != source_frames:
            raise ValueError(f"{key} {label}: contact frames {contact.shape[0]} != source frames {source_frames}")

    case_id, export_kind, _export_npz = key
    source_stem = source_path.stem
    output_dir = out_dir / "exports" / export_kind / case_id
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{slug(source_stem)}_e135_semantic_object_contact_{DEFAULT_LABEL}.npz"
    object_contact = contacts[DEFAULT_LABEL].astype(np.bool_)

    out_data: dict[str, Any] = dict(source_data)
    out_data["object_contact"] = object_contact
    out_data["object_contact_3cm"] = contacts["3cm"].astype(np.bool_)
    out_data["object_contact_5cm"] = contacts["5cm"].astype(np.bool_)
    out_data["object_contact_source"] = np.asarray(f"E135_v3_raw_contact_{DEFAULT_LABEL}")
    out_data["object_contact_source_npz"] = np.asarray(row["raw_contact_npz"])
    out_data["object_contact_mapping_type"] = np.asarray(row["mapping_type"])
    out_data["object_contact_mapping_proof"] = np.asarray(row["mapping_proof"])
    out_data["object_contact_label"] = np.asarray(DEFAULT_LABEL)
    out_data["object_contact_threshold_labels"] = np.asarray(THRESHOLDS)
    out_data["object_contact_person_idx"] = np.asarray(int(row["person_idx"]), dtype=np.int32)
    out_data["object_contact_raw_case_id"] = np.asarray(row["raw_case_id"])
    out_data["object_contact_trim_start"] = np.asarray(int(row["trim_start"] or 0), dtype=np.int32)
    out_data["object_contact_trim_end"] = np.asarray(int(row["trim_end"] or row["raw_frames"]), dtype=np.int32)
    out_data["object_contact_export_note"] = np.asarray("E137 audit/preflight artifact; qpos-style export, not MotionLoader runtime proof")
    np.savez_compressed(output_path, **out_data)

    with np.load(output_path, allow_pickle=True) as written:
        output_frame_key, output_frames = frame_key({key: written[key] for key in written.files})
        written_contact = np.asarray(written["object_contact"]).astype(bool)
        status = "pass"
        if output_frame_key != source_frame_key or output_frames != source_frames:
            status = "fail"
        if written_contact.shape != (source_frames, 2):
            status = "fail"
        if not np.array_equal(written_contact, object_contact):
            status = "fail"

    stats_default = contact_stats(object_contact, "object_contact")
    stats_3cm = contact_stats(contacts["3cm"], "object_contact_3cm")
    stats_5cm = contact_stats(contacts["5cm"], "object_contact_5cm")
    return {
        "case_id": case_id,
        "raw_case_id": row["raw_case_id"],
        "person": row["person"],
        "person_idx": row["person_idx"],
        "source_export_kind": export_kind,
        "source_export_npz": row["export_npz"],
        "source_export_sha256": sha256_file(source_path),
        "source_frame_key": source_frame_key,
        "source_frames": source_frames,
        "mapping_type": row["mapping_type"],
        "mapping_proof": row["mapping_proof"],
        "trim_window_json": row["trim_window_json"],
        "trim_start": row["trim_start"],
        "trim_end": row["trim_end"],
        "output_npz": rel(output_path),
        "output_exists": str(output_path.is_file()).lower(),
        "output_frames": output_frames,
        "object_contact_shape": stats_default["object_contact_shape"],
        "object_contact_dtype": "bool",
        "object_contact_source": f"E135_v3_raw_contact_{DEFAULT_LABEL}",
        "object_contact_label": DEFAULT_LABEL,
        "object_contact_left_active_frac": stats_default["object_contact_left_active_frac"],
        "object_contact_right_active_frac": stats_default["object_contact_right_active_frac"],
        "object_contact_both_active_frac": stats_default["object_contact_both_active_frac"],
        "object_contact_3cm_shape": stats_3cm["object_contact_3cm_shape"],
        "object_contact_3cm_left_active_frac": stats_3cm["object_contact_3cm_left_active_frac"],
        "object_contact_3cm_right_active_frac": stats_3cm["object_contact_3cm_right_active_frac"],
        "object_contact_3cm_both_active_frac": stats_3cm["object_contact_3cm_both_active_frac"],
        "object_contact_5cm_shape": stats_5cm["object_contact_5cm_shape"],
        "object_contact_5cm_left_active_frac": stats_5cm["object_contact_5cm_left_active_frac"],
        "object_contact_5cm_right_active_frac": stats_5cm["object_contact_5cm_right_active_frac"],
        "object_contact_5cm_both_active_frac": stats_5cm["object_contact_5cm_both_active_frac"],
        "raw_contact_npz": row["raw_contact_npz"],
        "raw_contact_key_3cm": thresholds["3cm"]["raw_contact_key"],
        "raw_contact_key_5cm": thresholds["5cm"]["raw_contact_key"],
        "raw_frames": row["raw_frames"],
        "semantic_export_status": status,
        "motionloader_runtime_probe_launched": "false",
        "training_launched": "false",
        "cem_launched": "false",
        "remote_jobs_launched": "false",
        "notes": "qpos-style E107 semantic object_contact candidate; runtime loader proof is a later step",
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "status": "pass" if rows and all(row["semantic_export_status"] == "pass" for row in rows) else "fail",
        "export_rows": len(rows),
        "e107_semantic_export_npz": len(rows),
        "retargeted_untrimmed_exports": sum(1 for row in rows if row["source_export_kind"] == "E107_retargeted_untrimmed"),
        "trimmed_exports": sum(1 for row in rows if row["source_export_kind"] == "E107_trimmed"),
        "e126_e131_exports_written": 0,
        "default_object_contact_label": DEFAULT_LABEL,
        "motionloader_runtime_probe_launched": False,
        "training_launched": False,
        "cem_launched": False,
        "remote_jobs_launched": False,
        "notes": "E137 writes isolated qpos-style E107 semantic contact candidates only.",
    }


def markdown_summary(summary: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    lines = [
        "# E137 E107 semantic object-contact export summary",
        "",
        f"- export rows: `{summary['export_rows']}`",
        f"- retargeted-untrimmed exports: `{summary['retargeted_untrimmed_exports']}`",
        f"- trimmed exports: `{summary['trimmed_exports']}`",
        f"- E126/E131 exports written: `{summary['e126_e131_exports_written']}`",
        f"- default object_contact label: `{summary['default_object_contact_label']}`",
        f"- MotionLoader runtime probe launched: `{str(summary['motionloader_runtime_probe_launched']).lower()}`",
        f"- training/CEM/remote launched: `{str(summary['training_launched']).lower()}`/"
        f"`{str(summary['cem_launched']).lower()}`/`{str(summary['remote_jobs_launched']).lower()}`",
        "",
        "## Exports",
        "",
        "| case | source | frames | mapping | object contact | 3cm L/R/both | 5cm L/R/both | status |",
        "|---|---|---:|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['case_id']}` | `{row['source_export_kind']}` | {row['output_frames']} | "
            f"`{row['mapping_type']}` | `{row['object_contact_shape']}` | "
            f"{row['object_contact_3cm_left_active_frac']}/{row['object_contact_3cm_right_active_frac']}/"
            f"{row['object_contact_3cm_both_active_frac']} | "
            f"{row['object_contact_5cm_left_active_frac']}/{row['object_contact_5cm_right_active_frac']}/"
            f"{row['object_contact_5cm_both_active_frac']} | `{row['semantic_export_status']}` |"
        )
    lines.extend(
        [
            "",
            "Interpretation: these are isolated qpos-style E107 artifacts with semantic `object_contact` arrays added from E135 raw-contact evidence. They are not E126/E131 paired fragment exports and are not Holosoma runtime-loader proof.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--e136-manifest", type=Path, default=DEFAULT_E136_MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    e136_manifest = args.e136_manifest.resolve()
    out_dir = args.out_dir.resolve()
    groups = candidate_groups(read_tsv(e136_manifest))
    rows = [export_group(key, thresholds, out_dir) for key, thresholds in sorted(groups.items())]
    summary = summarize(rows)
    write_tsv(out_dir / "e137_e107_semantic_object_contact_manifest.tsv", rows, MANIFEST_FIELDS)
    write_json(out_dir / "e137_e107_semantic_object_contact_summary.json", summary)
    (out_dir / "e137_e107_semantic_object_contact_summary.md").write_text(markdown_summary(summary, rows), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
