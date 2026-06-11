#!/usr/bin/env python3
"""Summarize bounded Box021 v3 S1 raw-contact remine outputs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


REPO = Path(__file__).resolve().parents[4]
DEFAULT_OUT_ROOT = REPO / "workspace/core4d/results/E135/box021_v3_s1_raw_contact_remine"
TARGET_CASE_IDS = (
    "box021_20231011_035_p1",
    "box021_20231011_035_p2",
    "box021_20231018_029_p1",
    "box021_20231018_029_p2",
)
LABELS = ("3cm", "5cm")

MANIFEST_FIELDS = [
    "threshold_label",
    "case_id",
    "sequence",
    "person",
    "person_idx",
    "raw_contact_decision",
    "raw_contact_decision_group",
    "raw_contact_score",
    "left_active_frac",
    "right_active_frac",
    "both_active_frac",
    "target_any_active_frac",
    "target_both_active_frac",
    "partner_any_active_frac",
    "raw_frames",
    "active_frames",
    "contact_mask_npz",
    "contact_mask_exists",
    "npz_has_raw_contact_mask_3cm",
    "npz_has_raw_contact_mask_5cm",
    "npz_has_centroid_world_3cm",
    "npz_has_centroid_world_5cm",
    "npz_has_centroid_object_local_3cm",
    "npz_has_centroid_object_local_5cm",
    "raw_to_trimmed_mapping_status",
    "contact_target_status",
    "registry_raw_contact_3cm_status",
    "registry_raw_contact_5cm_status",
    "registry_contact_mask_3cm_npz",
    "registry_contact_mask_5cm_npz",
    "registry_common_contact_mask_label",
]


def read_tsv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
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


def repo_relative(path: str | Path) -> str:
    if not path:
        return ""
    value = Path(path)
    try:
        return str(value.resolve().relative_to(REPO))
    except ValueError:
        return str(value)


def file_exists(path: str) -> bool:
    if not path:
        return False
    value = Path(path)
    return value.is_file() or (REPO / path).is_file()


def load_npz_keys(path: str, cache: dict[str, set[str]]) -> set[str]:
    if not path:
        return set()
    value = Path(path)
    if not value.is_file():
        value = REPO / path
    if not value.is_file():
        return set()
    key = str(value.resolve())
    if key not in cache:
        with np.load(value, allow_pickle=False) as data:
            cache[key] = set(data.files)
    return cache[key]


def registry_by_case(out_root: Path) -> dict[str, dict[str, str]]:
    rows = read_tsv(out_root / "registries_combined_5cm_then_3cm/case_state_registry.tsv")
    return {row.get("case_id", ""): row for row in rows}


def threshold_counts(rows_by_label: dict[str, list[dict[str, str]]]) -> dict[str, dict[str, Any]]:
    counts: dict[str, dict[str, Any]] = {}
    for label, rows in rows_by_label.items():
        pass_rows = [row for row in rows if row.get("raw_contact_decision") == "raw_contact_pass"]
        counts[label] = {
            "candidate_rows": len(rows),
            "pass_rows": len(pass_rows),
            "decision_counts": dict(Counter(row.get("raw_contact_decision", "") for row in rows)),
            "decision_group_counts": dict(Counter(row.get("raw_contact_decision_group", "") for row in rows)),
            "case_ids": sorted(row.get("case_id", "") for row in rows),
        }
    return counts


def build_manifest(out_root: Path) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, str]]]]:
    rows_by_label = {
        label: read_tsv(out_root / "s1_raw_contact/raw_contact" / f"raw_contact_candidates_{label}.tsv")
        for label in LABELS
    }
    registry = registry_by_case(out_root)
    npz_cache: dict[str, set[str]] = {}
    manifest: list[dict[str, Any]] = []
    for label in LABELS:
        for row in rows_by_label[label]:
            case_id = row.get("case_id", "")
            reg = registry.get(case_id, {})
            npz_path = row.get("contact_mask_npz", "")
            keys = load_npz_keys(npz_path, npz_cache)
            manifest.append(
                {
                    "threshold_label": label,
                    "case_id": case_id,
                    "sequence": row.get("sequence", ""),
                    "person": row.get("person", ""),
                    "person_idx": row.get("person_idx", ""),
                    "raw_contact_decision": row.get("raw_contact_decision", ""),
                    "raw_contact_decision_group": row.get("raw_contact_decision_group", ""),
                    "raw_contact_score": row.get("raw_contact_score", ""),
                    "left_active_frac": row.get("left_active_frac", ""),
                    "right_active_frac": row.get("right_active_frac", ""),
                    "both_active_frac": row.get("both_active_frac", ""),
                    "target_any_active_frac": row.get("target_any_active_frac", ""),
                    "target_both_active_frac": row.get("target_both_active_frac", ""),
                    "partner_any_active_frac": row.get("partner_any_active_frac", ""),
                    "raw_frames": row.get("raw_frames", row.get("raw_frame_count", "")),
                    "active_frames": row.get("active_frames", ""),
                    "contact_mask_npz": repo_relative(npz_path),
                    "contact_mask_exists": str(file_exists(npz_path)).lower(),
                    "npz_has_raw_contact_mask_3cm": str("raw_contact_mask_3cm" in keys).lower(),
                    "npz_has_raw_contact_mask_5cm": str("raw_contact_mask_5cm" in keys).lower(),
                    "npz_has_centroid_world_3cm": str("raw_contact_centroid_world_3cm" in keys).lower(),
                    "npz_has_centroid_world_5cm": str("raw_contact_centroid_world_5cm" in keys).lower(),
                    "npz_has_centroid_object_local_3cm": str("raw_contact_centroid_object_local_3cm" in keys).lower(),
                    "npz_has_centroid_object_local_5cm": str("raw_contact_centroid_object_local_5cm" in keys).lower(),
                    "raw_to_trimmed_mapping_status": row.get("raw_to_trimmed_mapping_status", ""),
                    "contact_target_status": row.get("contact_target_status", ""),
                    "registry_raw_contact_3cm_status": reg.get("raw_contact_3cm_status", ""),
                    "registry_raw_contact_5cm_status": reg.get("raw_contact_5cm_status", ""),
                    "registry_contact_mask_3cm_npz": repo_relative(reg.get("contact_mask_3cm_npz", "")),
                    "registry_contact_mask_5cm_npz": repo_relative(reg.get("contact_mask_5cm_npz", "")),
                    "registry_common_contact_mask_label": reg.get("contact_mask_label", ""),
                }
            )
    return manifest, rows_by_label


def per_sequence_artifacts(manifest: list[dict[str, Any]]) -> list[dict[str, Any]]:
    paths = sorted({row["contact_mask_npz"] for row in manifest if row.get("contact_mask_npz")})
    artifacts: list[dict[str, Any]] = []
    for path in paths:
        value = REPO / path
        if not value.is_file():
            value = Path(path)
        if not value.is_file():
            artifacts.append({"path": path, "exists": False})
            continue
        with np.load(value, allow_pickle=False) as data:
            keys = set(data.files)
            artifacts.append(
                {
                    "path": repo_relative(value),
                    "exists": True,
                    "raw_contact_mask_3cm_shape": "x".join(map(str, data["raw_contact_mask_3cm"].shape))
                    if "raw_contact_mask_3cm" in keys
                    else "",
                    "raw_contact_mask_5cm_shape": "x".join(map(str, data["raw_contact_mask_5cm"].shape))
                    if "raw_contact_mask_5cm" in keys
                    else "",
                    "has_centroid_object_local_3cm": "raw_contact_centroid_object_local_3cm" in keys,
                    "has_centroid_object_local_5cm": "raw_contact_centroid_object_local_5cm" in keys,
                    "has_raw_to_trimmed_frame_index": "raw_to_trimmed_frame_index" in keys,
                }
            )
    return artifacts


def markdown_summary(summary: dict[str, Any], manifest: list[dict[str, Any]]) -> str:
    lines = [
        "# E135 Box021 v3 S1 raw-contact remine summary",
        "",
        f"- selected case-person rows: `{summary['selected_case_person_rows']}`",
        f"- unique raw sequences: `{summary['unique_sequences']}`",
        f"- per-sequence proxy NPZ files: `{summary['per_sequence_proxy_npz_files']}`",
        f"- global registry updated: `{str(summary['global_registry_updated']).lower()}`",
        f"- training/CEM/remote launched: `{str(summary['training_launched']).lower()}`/"
        f"`{str(summary['cem_launched']).lower()}`/`{str(summary['remote_jobs_launched']).lower()}`",
        "",
        "## Threshold Counts",
        "",
        "| threshold | candidates | pass | decisions |",
        "|---|---:|---:|---|",
    ]
    for label, counts in summary["threshold_counts"].items():
        lines.append(
            f"| `{label}` | {counts['candidate_rows']} | {counts['pass_rows']} | "
            f"`{counts['decision_counts']}` |"
        )
    lines.extend(
        [
            "",
            "## Candidate Rows",
            "",
            "| threshold | case | decision | score | L/R/both | partner any | registry 3cm/5cm |",
            "|---|---|---|---:|---|---:|---|",
        ]
    )
    for row in manifest:
        lines.append(
            f"| `{row['threshold_label']}` | `{row['case_id']}` | `{row['raw_contact_decision']}` | "
            f"{row['raw_contact_score']} | {row['left_active_frac']}/{row['right_active_frac']}/"
            f"{row['both_active_frac']} | {row['partner_any_active_frac']} | "
            f"`{row['registry_raw_contact_3cm_status']}`/`{row['registry_raw_contact_5cm_status']}` |"
        )
    lines.extend(["", "## Per-Sequence Artifacts", "", "| artifact | 3cm shape | 5cm shape |", "|---|---|---|"])
    for artifact in summary["per_sequence_artifacts"]:
        lines.append(
            f"| `{artifact['path']}` | `{artifact.get('raw_contact_mask_3cm_shape', '')}` | "
            f"`{artifact.get('raw_contact_mask_5cm_shape', '')}` |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    args = parser.parse_args()

    out_root = args.out_root.resolve()
    target_inventory = read_tsv(out_root / "s1_raw_contact/inventory/inventory_box021_bounded.tsv")
    manifest, rows_by_label = build_manifest(out_root)
    artifacts = per_sequence_artifacts(manifest)
    sequences = sorted({row.get("sequence", "") for rows in rows_by_label.values() for row in rows if row.get("sequence")})
    registry_rows = read_tsv(out_root / "registries_combined_5cm_then_3cm/case_state_registry.tsv")

    summary = {
        "status": "pass" if manifest and len(target_inventory) == len(TARGET_CASE_IDS) else "review",
        "target_case_ids": list(TARGET_CASE_IDS),
        "selected_case_person_rows": len(target_inventory),
        "unique_sequences": len(sequences),
        "sequences": sequences,
        "threshold_counts": threshold_counts(rows_by_label),
        "manifest_rows": len(manifest),
        "registry_rows": len(registry_rows),
        "per_sequence_proxy_npz_files": sum(1 for item in artifacts if item.get("exists")),
        "per_sequence_artifacts": artifacts,
        "global_registry_updated": False,
        "training_launched": False,
        "cem_launched": False,
        "remote_jobs_launched": False,
        "notes": (
            "Raw contact remains a geometric proxy on raw CORE4D time axes. "
            "E135 does not write Holosoma semantic object_contact exports."
        ),
    }

    summary_dir = out_root / "summary"
    write_tsv(summary_dir / "e135_box021_v3_s1_raw_contact_manifest.tsv", manifest, MANIFEST_FIELDS)
    write_json(summary_dir / "e135_box021_v3_s1_raw_contact_summary.json", summary)
    (summary_dir / "e135_box021_v3_s1_raw_contact_summary.md").write_text(
        markdown_summary(summary, manifest),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
