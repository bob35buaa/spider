#!/usr/bin/env python3
"""Select E181 canonical CoACD sets or persist the ASSET_REJECTED terminal."""

from __future__ import annotations

import csv
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from build_authority import REPO_ROOT, relative_to_repo, sha256_file

ASSET_ROOT = REPO_ROOT / "workspace/core4d/results/E181/s2_asset_eval"
COACD_ROOT = REPO_ROOT / "workspace/core4d/results/E181/s2_coacd"
OBJECT_KEYS = ("bucket003", "bucket004", "bucket007")


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    """Write deterministic JSON atomically."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def read_rows() -> list[dict[str, str]]:
    """Read the 54-row fidelity table."""
    path = ASSET_ROOT / "candidate_metrics.tsv"
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if len(rows) != 54:
        raise RuntimeError(f"expected 54 fidelity rows, got {len(rows)}")
    return rows


def candidate_manifest(row: dict[str, str]) -> dict[str, Any]:
    """Load a candidate build manifest."""
    path = COACD_ROOT / row["object_key"] / row["candidate_id"] / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest["candidate_asset_sha256"] != row["candidate_asset_sha256"]:
        raise RuntimeError(f"candidate SHA mismatch: {path}")
    return manifest


def selection_key(row: dict[str, str]) -> tuple[Any, ...]:
    """Frozen lexicographic selection rule for already-passing candidates."""
    manifest = candidate_manifest(row)
    symmetric_p99 = max(
        float(row["m_to_c_surface_p99_m"]),
        float(row["c_to_m_surface_p99_m"]),
    )
    return (
        manifest["hull_count"],
        float(row["broader_cavity_false_occupied_fraction"]),
        symmetric_p99,
        manifest["parameters"]["threshold_m"],
        manifest["parameters"]["max_ch_vertex"],
    )


def diagnostic_key(row: dict[str, str]) -> tuple[Any, ...]:
    """Rank rejected assets for diagnosis only; never release this ranking."""
    return (
        float(row["broader_cavity_false_occupied_fraction"]),
        len([gate for gate in row["failed_dev_gates"].split(",") if gate]),
        -float(row["must_cover_recall_15mm"]),
        float(row["c_to_m_surface_p99_m"]),
        row["candidate_id"],
    )


def plot_heatmaps(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Render required Gate B cavity and task-metric heatmaps."""
    labels = [
        f"t{threshold:03d}/k{hulls:02d}"
        for threshold in (5, 10, 20)
        for hulls in (8, 16, 32)
    ]
    candidate_order = [
        f"t{threshold:03d}_k{hulls:02d}_v{vertices:03d}"
        for threshold in (5, 10, 20)
        for hulls in (8, 16, 32)
        for vertices in (32, 64)
    ]
    row_lookup = {(row["object_key"], row["candidate_id"]): row for row in rows}

    cavity_figure, cavity_axes = plt.subplots(1, 3, figsize=(12, 8))
    task_figure, task_axes = plt.subplots(2, 3, figsize=(12, 12))
    for column, object_key in enumerate(OBJECT_KEYS):
        ordered = [
            row_lookup[(object_key, identifier)] for identifier in candidate_order
        ]
        cavity = np.asarray(
            [
                float(row["broader_cavity_false_occupied_fraction"]) * 100
                for row in ordered
            ]
        ).reshape(9, 2)
        cover = np.asarray(
            [float(row["must_cover_recall_15mm"]) * 100 for row in ordered]
        ).reshape(9, 2)
        contact = np.asarray(
            [float(row["dev_contact_excess_p90_m"]) * 1000 for row in ordered]
        ).reshape(9, 2)
        cavity_image = cavity_axes[column].imshow(
            cavity,
            aspect="auto",
            cmap="magma",
        )
        cavity_axes[column].set_title(f"{object_key} cavity false occupied (%)")
        cavity_axes[column].set_xticks((0, 1), ("v32", "v64"))
        cavity_axes[column].set_yticks(range(9), labels)
        cavity_figure.colorbar(cavity_image, ax=cavity_axes[column], shrink=0.7)
        for axis, values, title, color_map in (
            (
                task_axes[0, column],
                cover,
                f"{object_key} must-cover (%)",
                "viridis",
            ),
            (
                task_axes[1, column],
                contact,
                f"{object_key} dev contact excess p90 (mm)",
                "cividis",
            ),
        ):
            image = axis.imshow(values, aspect="auto", cmap=color_map)
            axis.set_title(title)
            axis.set_xticks((0, 1), ("v32", "v64"))
            axis.set_yticks(range(9), labels)
            task_figure.colorbar(image, ax=axis, shrink=0.7)
    cavity_figure.tight_layout()
    task_figure.tight_layout()
    cavity_path = ASSET_ROOT / "cavity_false_positive_heatmap.png"
    task_path = ASSET_ROOT / "must_cover_contact_excess_heatmap.png"
    cavity_figure.savefig(cavity_path, dpi=180)
    task_figure.savefig(task_path, dpi=180)
    plt.close(cavity_figure)
    plt.close(task_figure)
    return [
        {
            "path": relative_to_repo(path),
            "sha256": sha256_file(path),
        }
        for path in (cavity_path, task_path)
    ]


def main() -> int:
    """Select passing assets or persist no-selection evidence."""
    summary_path = ASSET_ROOT / "pareto_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("candidate_count") != 54:
        raise RuntimeError("Gate B summary is incomplete")
    rows = read_rows()
    visual_evidence = plot_heatmaps(rows)
    selected: dict[str, Any] = {}
    diagnostics: dict[str, Any] = {}
    for object_key in OBJECT_KEYS:
        object_rows = [row for row in rows if row["object_key"] == object_key]
        passing = [row for row in object_rows if row["status"] == "PASS"]
        rejected_best = min(object_rows, key=diagnostic_key)
        failed_counts = Counter(
            gate
            for row in object_rows
            for gate in row["failed_dev_gates"].split(",")
            if gate
        )
        diagnostics[object_key] = {
            "pass_count": len(passing),
            "closest_rejected_candidate_id": rejected_best["candidate_id"],
            "closest_rejected_candidate_asset_sha256": rejected_best[
                "candidate_asset_sha256"
            ],
            "closest_rejected_broader_cavity_false_occupied_fraction": float(
                rejected_best["broader_cavity_false_occupied_fraction"]
            ),
            "closest_rejected_failed_dev_gates": [
                gate for gate in rejected_best["failed_dev_gates"].split(",") if gate
            ],
            "failed_gate_counts": dict(sorted(failed_counts.items())),
        }
        if passing:
            winner = min(passing, key=selection_key)
            manifest = candidate_manifest(winner)
            selected[object_key] = {
                "candidate_id": winner["candidate_id"],
                "candidate_asset_sha256": winner["candidate_asset_sha256"],
                "candidate_manifest": manifest,
                "selection_key": selection_key(winner),
            }
    status = "PASS" if len(selected) == len(OBJECT_KEYS) else "ASSET_REJECTED"
    if status == "ASSET_REJECTED":
        selected = {}
    result = {
        "experiment_id": "E181",
        "stage": "S2_canonical_selection",
        "status": status,
        "reason": (
            "At least one object has zero candidates passing all frozen dev hard gates; "
            "the plan forbids threshold relaxation or manual hull edits."
            if status == "ASSET_REJECTED"
            else "All objects selected by the frozen lexicographic rule."
        ),
        "heldout_status": (
            "SEALED_NO_C_STAR" if status == "ASSET_REJECTED" else "READY_TO_UNLOCK"
        ),
        "gate_b_summary": {
            "path": relative_to_repo(summary_path),
            "sha256": sha256_file(summary_path),
        },
        "candidate_metrics": summary["candidate_metrics"],
        "selected": selected,
        "diagnostics": diagnostics,
        "visual_evidence": visual_evidence,
        "downstream_status": (
            "S3_S6_NOT_AUTHORIZED_BY_GATE_B"
            if status == "ASSET_REJECTED"
            else "S3_AUTHORIZED"
        ),
    }
    output_path = ASSET_ROOT / "selected_canonical_sets.json"
    atomic_json(output_path, result)
    print(
        f"E181_CANONICAL_SELECTION={status} selected={len(selected)}/3 "
        f"heldout={result['heldout_status']}"
    )
    return 0 if status == "PASS" else 3


if __name__ == "__main__":
    raise SystemExit(main())
