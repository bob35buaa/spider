#!/usr/bin/env python3
"""Generate raw-contact visualizations for E091 medium-box candidates."""

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


def as_float(value: Any, default: float = 0.0) -> float:
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def image_nonblank(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size < 2048:
        return False
    try:
        arr = plt.imread(path)
    except Exception:
        return False
    return bool(np.asarray(arr).std() > 1e-4)


def threshold_index(thresholds: np.ndarray, target: float = 0.03) -> int:
    arr = np.asarray(thresholds, dtype=float).reshape(-1)
    return int(np.argmin(np.abs(arr - target)))


def shade_active(ax: plt.Axes, frames: np.ndarray, active: np.ndarray) -> None:
    if len(frames) == 0:
        return
    ax.fill_between(
        frames,
        0,
        1,
        where=active,
        transform=ax.get_xaxis_transform(),
        color="#999999",
        alpha=0.12,
        linewidth=0,
        label="active object motion",
    )


def plot_case(row: dict[str, str], out_dir: Path) -> dict[str, Any]:
    proxy = Path(row["raw_contact_proxy_path"])
    target_task = row["planned_target_task"]
    out_path = out_dir / f"{target_task}_raw_contact.png"
    if not proxy.is_file():
        return {
            "planned_target_task": target_task,
            "object_key": row["object_key"],
            "sequence": row["sequence"],
            "person": row["person"],
            "status": "missing_raw_contact_proxy",
            "raw_contact_png": "",
            "nonblank": False,
        }

    with np.load(proxy, allow_pickle=True) as data:
        min_dist = np.asarray(data["min_dist_m"], dtype=float)
        masks = np.asarray(data["masks"], dtype=bool)
        active = np.asarray(data["active_mask"], dtype=bool)
        object_pos = np.asarray(data["object_pos"], dtype=float)
        persons = [str(x) for x in np.asarray(data["persons"]).tolist()]
        hands = [str(x) for x in np.asarray(data["hands"]).tolist()]
        thresholds = np.asarray(data["thresholds_m"], dtype=float)

    if row["person"] not in persons:
        person_idx = 0
        status = f"person_not_found_used_{persons[0]}"
    else:
        person_idx = persons.index(row["person"])
        status = "ok"
    th_idx = threshold_index(thresholds, 0.03)
    th_cm = float(thresholds.reshape(-1)[th_idx] * 100.0)
    frames = np.arange(min_dist.shape[0])
    hand_dist_cm = min_dist[:, person_idx, :] * 100.0
    hand_masks = masks[:, person_idx, :, th_idx]
    both = np.logical_and(hand_masks[:, 0], hand_masks[:, 1])

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), constrained_layout=True)
    title = (
        f"{target_task} raw contact | {row['sequence']} {row['person']} "
        f"{row['object_key']} | score={row['stage1_raw_contact_score']}"
    )
    fig.suptitle(title)

    for i, hand in enumerate(hands):
        axes[0].plot(frames, hand_dist_cm[:, i], label=f"{hand} min dist")
    axes[0].axhline(th_cm, color="#cc3311", linestyle="--", linewidth=1.2, label=f"{th_cm:.1f} cm")
    shade_active(axes[0], frames, active)
    axes[0].set_ylabel("distance (cm)")
    axes[0].set_ylim(bottom=0)
    axes[0].legend(loc="upper right")
    axes[0].set_title("Target hand to object surface distance")

    axes[1].plot(frames, hand_masks[:, 0].astype(int), label=f"{hands[0]} <= {th_cm:.1f}cm")
    axes[1].plot(frames, hand_masks[:, 1].astype(int) + 1.2, label=f"{hands[1]} <= {th_cm:.1f}cm")
    axes[1].plot(frames, both.astype(int) + 2.4, label="both hands")
    axes[1].plot(frames, active.astype(int) + 3.6, label="object active")
    axes[1].set_yticks([0, 1.2, 2.4, 3.6])
    axes[1].set_yticklabels([hands[0], hands[1], "both", "active"])
    axes[1].set_ylim(-0.2, 4.9)
    axes[1].legend(loc="upper right")
    axes[1].set_title("3cm contact masks")

    labels = ["x", "y", "z"]
    for i, label in enumerate(labels):
        axes[2].plot(frames, object_pos[:, i], label=f"object {label}")
    shade_active(axes[2], frames, active)
    axes[2].set_ylabel("position (m)")
    axes[2].legend(loc="upper right")
    axes[2].set_title("Raw object trajectory")

    axes[3].plot(object_pos[:, 0], object_pos[:, 1], color="#4477aa")
    axes[3].scatter(object_pos[0, 0], object_pos[0, 1], color="#228833", label="start")
    axes[3].scatter(object_pos[-1, 0], object_pos[-1, 1], color="#cc3311", label="end")
    axes[3].set_xlabel("object x (m)")
    axes[3].set_ylabel("object y (m)")
    axes[3].axis("equal")
    axes[3].legend(loc="best")
    axes[3].set_title("Object XY path")

    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)

    return {
        "planned_target_task": target_task,
        "object_key": row["object_key"],
        "sequence": row["sequence"],
        "person": row["person"],
        "case_role": row["case_role"],
        "stage1_decision": row["stage1_decision"],
        "stage1_raw_contact_score": row["stage1_raw_contact_score"],
        "target_both_active_frac_3cm": row["target_both_active_frac_3cm"],
        "partner_any_active_frac_3cm": row["partner_any_active_frac_3cm"],
        "status": status,
        "raw_contact_png": str(out_path),
        "nonblank": image_nonblank(out_path),
        "threshold_cm": round(th_cm, 3),
        "frame_count": int(min_dist.shape[0]),
        "both_mask_frac_from_npz": round(float(both.mean()), 6),
        "active_frac_from_npz": round(float(active.mean()), 6),
    }


def plot_dashboard(rows: list[dict[str, str]], visual_rows: list[dict[str, Any]], out_dir: Path) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    plotted = [r for r in visual_rows if r["status"] == "ok"]

    if plotted:
        labels = [r["planned_target_task"].replace("e091_", "") for r in plotted]
        scores = [as_float(r["stage1_raw_contact_score"]) for r in plotted]
        both = [as_float(r["target_both_active_frac_3cm"]) for r in plotted]
        partner = [as_float(r["partner_any_active_frac_3cm"]) for r in plotted]
        colors = ["#4477aa" if r["object_key"] == "box026" else "#66ccee" for r in plotted]

        fig, ax = plt.subplots(figsize=(12, max(5, 0.35 * len(plotted))), constrained_layout=True)
        y = np.arange(len(plotted))
        ax.barh(y - 0.2, scores, height=0.22, color=colors, label="raw score")
        ax.barh(y + 0.05, [b * 100 for b in both], height=0.22, color="#228833", label="target both 3cm %")
        ax.barh(y + 0.30, [p * 100 for p in partner], height=0.22, color="#cc6677", label="partner any 3cm %")
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("score / percent")
        ax.set_title("E091 raw-contact candidates")
        ax.legend(loc="lower right")
        path = out_dir / "raw_contact_candidate_scores.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(str(path))

    object_counts = Counter(r["object_key"] for r in rows)
    visual_counts = Counter(r["object_key"] for r in visual_rows if r["raw_contact_png"])
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    keys = sorted(set(object_counts) | set(visual_counts))
    x = np.arange(len(keys))
    ax.bar(x - 0.18, [object_counts[k] for k in keys], width=0.36, label="manifest")
    ax.bar(x + 0.18, [visual_counts[k] for k in keys], width=0.36, label="visualized")
    ax.set_xticks(x)
    ax.set_xticklabels(keys)
    ax.set_ylabel("count")
    ax.set_title("Raw-contact visualization coverage")
    ax.legend()
    path = out_dir / "raw_contact_visual_coverage.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(str(path))
    return paths


def write_summary(root: Path, visual_rows: list[dict[str, Any]], dashboard_paths: list[str]) -> None:
    summary = {
        "stage": "E091 raw-contact visuals",
        "candidate_rows": len(visual_rows),
        "png_count": sum(1 for row in visual_rows if row["raw_contact_png"]),
        "nonblank_png_count": sum(1 for row in visual_rows if row["nonblank"]),
        "status_counts": dict(Counter(row["status"] for row in visual_rows)),
        "object_counts": dict(Counter(row["object_key"] for row in visual_rows)),
        "dashboard_paths": dashboard_paths,
    }
    write_json(root / "results/raw_contact_visuals/summary.json", summary)
    lines = [
        "# E091 Raw Contact Visual Summary",
        "",
        f"- Candidate rows with raw proxy: `{len(visual_rows)}`",
        f"- PNG written: `{summary['png_count']}`",
        f"- Nonblank PNG: `{summary['nonblank_png_count']}`",
        "",
        "Status counts:",
        "",
        "| status | count |",
        "|---|---:|",
    ]
    for status, count in Counter(row["status"] for row in visual_rows).most_common():
        lines.append(f"| `{status}` | {count} |")
    lines.extend(["", "Dashboard:", ""])
    lines.extend(f"- `{path}`" for path in dashboard_paths)
    (root / "results/raw_contact_visuals/summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v2-root", type=Path, default=DEFAULT_V2_ROOT)
    parser.add_argument("--include-fail", action="store_true", help="Also plot D002 raw-contact fail rows.")
    args = parser.parse_args()

    manifest_path = args.v2_root / "inputs/medium_box_manifest.tsv"
    rows = read_tsv(manifest_path)
    wanted_roles = {"primary_stage2b_template_backlog", "secondary_stage2b_template_backlog"}
    if args.include_fail:
        wanted_roles.add("hold_raw_contact_failed")
    candidates = [
        row
        for row in rows
        if row["case_role"] in wanted_roles and row.get("raw_contact_proxy_path") and Path(row["raw_contact_proxy_path"]).is_file()
    ]

    visual_dir = args.v2_root / "visualizations/raw_contact"
    visual_rows = [plot_case(row, visual_dir) for row in candidates]
    fields = list(visual_rows[0].keys()) if visual_rows else []
    write_tsv(args.v2_root / "results/raw_contact_visuals/raw_contact_visual_manifest.tsv", visual_rows, fields)
    write_json(args.v2_root / "results/raw_contact_visuals/raw_contact_visual_manifest.json", visual_rows)
    dashboard_paths = plot_dashboard(rows, visual_rows, args.v2_root / "visualizations/dashboard")
    write_summary(args.v2_root, visual_rows, dashboard_paths)

    print(f"Plotted {len(visual_rows)} raw-contact rows")
    print(f"Nonblank PNG: {sum(1 for row in visual_rows if row['nonblank'])}/{len(visual_rows)}")


if __name__ == "__main__":
    main()
