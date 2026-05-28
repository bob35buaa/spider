#!/usr/bin/env python3
"""Preflight E091 medium-box Stage2b backlog before template creation."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_V2_ROOT = Path("/home/ubuntu/Workspace/holosoma/workspace/v3/data_construction_v2")
DEFAULT_CORE4D_ROOT = Path(
    "/mnt/a0ccc676-9496-49f8-a861-f8a1797dec52/mocap_data/CORE4D/CORE4D_Real"
)
DEFAULT_SPIDER_TASK_ROOT = Path(
    "/home/ubuntu/Workspace/spider/example_datasets/processed/core4d/unitree_g1/humanoid_object"
)

BASE_TEMPLATE_PREFERENCES = {
    "box026": ["box021_person1", "box023_person1", "box025_person2"],
    "box004": ["box023_person1", "box021_person1", "box025_person2"],
    "box022": ["box021_person1", "box023_person1", "box025_person2"],
}


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def parse_obj_extents(path: Path) -> tuple[float, float, float] | None:
    if not path.is_file():
        return None
    vertices: list[list[float]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.startswith("v "):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            except ValueError:
                continue
    if not vertices:
        return None
    arr = np.asarray(vertices, dtype=np.float64)
    ext = arr.max(axis=0) - arr.min(axis=0)
    return float(ext[0]), float(ext[1]), float(ext[2])


def choose_base_template(object_key: str, task_root: Path) -> tuple[str, bool, str]:
    for name in BASE_TEMPLATE_PREFERENCES.get(object_key, []):
        scene = task_root / name / "scene.xml"
        if scene.is_file():
            return name, True, str(scene)
    return "", False, ""


def preflight_row(row: dict[str, str], core4d_root: Path, task_root: Path) -> dict[str, Any]:
    mesh_path = core4d_root / "object_models" / row["object_model_rel"]
    ext = parse_obj_extents(mesh_path)
    half_ext = tuple(round(v / 2.0, 6) for v in ext) if ext else ("", "", "")
    volume = float(np.prod(ext)) if ext else None
    source_scene = task_root / row["source_scene_task"] / "scene.xml"
    base_name, base_exists, base_path = choose_base_template(row["target_task"].split("_")[1], task_root)
    source_exists = source_scene.is_file()
    mesh_exists = mesh_path.is_file()
    if not mesh_exists:
        status = "blocked_missing_object_mesh"
    elif not base_exists:
        status = "blocked_missing_base_template"
    elif not source_exists:
        status = "needs_source_scene_template"
    else:
        status = "source_scene_ready"

    return {
        "target_task": row["target_task"],
        "case_role": row["case_role"],
        "date": row["date"],
        "seq": row["seq"],
        "person": row["person"],
        "object_name": row["object_name"],
        "object_key": row["target_task"].split("_")[1],
        "object_model_rel": row["object_model_rel"],
        "object_mesh_path": str(mesh_path),
        "object_mesh_exists": mesh_exists,
        "mesh_extent_x_m": round(ext[0], 6) if ext else "",
        "mesh_extent_y_m": round(ext[1], 6) if ext else "",
        "mesh_extent_z_m": round(ext[2], 6) if ext else "",
        "mesh_half_extent_x_m": half_ext[0],
        "mesh_half_extent_y_m": half_ext[1],
        "mesh_half_extent_z_m": half_ext[2],
        "mesh_volume_m3": round(volume, 9) if volume is not None else "",
        "source_scene_task": row["source_scene_task"],
        "source_scene_path": str(source_scene),
        "source_scene_exists": source_exists,
        "recommended_base_template": base_name,
        "recommended_base_template_exists": base_exists,
        "recommended_base_scene_path": base_path,
        "preflight_status": status,
        "next_action": "create source scene template, then rerun preflight" if status == "needs_source_scene_template" else "",
    }


def plot_preflight(rows: list[dict[str, Any]], out_dir: Path) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    status_counts = Counter(row["preflight_status"] for row in rows)
    object_counts = Counter(row["object_key"] for row in rows)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    axes[0].barh(list(status_counts.keys()), list(status_counts.values()), color="#cc6677")
    axes[0].set_xlabel("count")
    axes[0].set_title("Template preflight status")
    axes[1].bar(list(object_counts.keys()), list(object_counts.values()), color="#4477aa")
    axes[1].set_ylabel("case-person count")
    axes[1].set_title("Stage2b backlog by object")
    path = out_dir / "template_preflight_status.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(str(path))
    return paths


def write_summary(root: Path, rows: list[dict[str, Any]], plots: list[str]) -> None:
    summary = {
        "stage": "E091 template preflight",
        "rows": len(rows),
        "status_counts": dict(Counter(row["preflight_status"] for row in rows)),
        "object_counts": dict(Counter(row["object_key"] for row in rows)),
        "source_scene_ready_count": sum(1 for row in rows if row["preflight_status"] == "source_scene_ready"),
        "plots": plots,
    }
    write_json(root / "results/template_preflight/summary.json", summary)
    lines = [
        "# E091 Template Preflight Summary",
        "",
        f"- Backlog rows checked: `{len(rows)}`",
        f"- Source scene ready: `{summary['source_scene_ready_count']}`",
        "",
        "Status counts:",
        "",
        "| status | count |",
        "|---|---:|",
    ]
    for status, count in Counter(row["preflight_status"] for row in rows).most_common():
        lines.append(f"| `{status}` | {count} |")
    lines.extend(["", "Object counts:", "", "| object | count |", "|---|---:|"])
    for obj, count in Counter(row["object_key"] for row in rows).most_common():
        lines.append(f"| `{obj}` | {count} |")
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "- Object meshes exist for rows marked `needs_source_scene_template`.",
            "- Stage2b remains disabled until each object's source scene template is created and preflight is rerun.",
            "- Recommended base templates are same-category CORE4D box scenes; object mesh/collision/mass must be replaced before use.",
            "",
            "Visualizations:",
            "",
        ]
    )
    lines.extend(f"- `{path}`" for path in plots)
    (root / "results/template_preflight/summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v2-root", type=Path, default=DEFAULT_V2_ROOT)
    parser.add_argument("--core4d-root", type=Path, default=DEFAULT_CORE4D_ROOT)
    parser.add_argument("--spider-task-root", type=Path, default=DEFAULT_SPIDER_TASK_ROOT)
    args = parser.parse_args()

    rows = read_tsv(args.v2_root / "inputs/cases_stage2b_medium_backlog.tsv")
    out_rows = [preflight_row(row, args.core4d_root, args.spider_task_root) for row in rows]
    fields = list(out_rows[0].keys()) if out_rows else []
    write_tsv(args.v2_root / "results/template_preflight/template_preflight_summary.tsv", out_rows, fields)
    write_json(args.v2_root / "results/template_preflight/template_preflight_summary.json", out_rows)
    plots = plot_preflight(out_rows, args.v2_root / "visualizations/dashboard")
    write_summary(args.v2_root, out_rows, plots)
    print(f"Preflighted {len(out_rows)} Stage2b backlog rows")
    print(dict(Counter(row["preflight_status"] for row in out_rows)))


if __name__ == "__main__":
    main()
