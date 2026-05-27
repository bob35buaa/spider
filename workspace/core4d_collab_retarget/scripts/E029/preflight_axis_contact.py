#!/usr/bin/env python3
"""Axis/contact preflight for E029 D6 support endpoint selection."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import e029_common as common  # noqa: E402


def _axis_world_alignment(model, qpos: np.ndarray) -> tuple[np.ndarray, str, float, str]:
    qadr = common.object_qadr(model)
    local_axes = np.eye(3, dtype=np.float64)
    dots = np.zeros((3, 3), dtype=np.float64)
    for q in qpos:
        quat = q[qadr + 3 : qadr + 7]
        world_axes = np.asarray([common.quat_apply(quat, axis) for axis in local_axes])
        dots += np.abs(world_axes)
    dots /= max(1, len(qpos))
    dom = [common.WORLD_AXES[int(np.argmax(dots[i]))] for i in range(3)]
    height_idx = int(np.argmax(dots[:, 2]))
    height_local = common.AXES[height_idx]
    height_dot = float(dots[height_idx, 2])
    dom_string = ";".join(f"local_{common.AXES[i]}->{dom[i]}:{dots[i, int(np.argmax(dots[i]))]:.3f}" for i in range(3))
    return dots, height_local, height_dot, dom_string


def _cloud_summary(points: np.ndarray, half: np.ndarray) -> dict[str, object]:
    counts = common.face_counts(points, half)
    total = int(len(points))
    side, side_frac, margin = common.side_margin(counts, total)
    top = common.top_face(counts)
    centroid = points.mean(axis=0) if total else np.full(3, np.nan, dtype=np.float64)
    robust = common.robust_centroid(points)
    labels = np.asarray([common.face_label(point, half) for point in points]) if total else np.asarray([])
    selected = labels == side
    side_points = points[selected] if np.any(selected) else np.zeros((0, 3), dtype=np.float64)
    side_centroid = common.robust_centroid(side_points)
    return {
        "total": total,
        "counts": counts,
        "top_face": top,
        "side_face": side,
        "side_frac": side_frac,
        "side_margin": margin,
        "centroid": centroid,
        "robust_centroid": robust,
        "side_centroid": side_centroid,
        "side_count": int(len(side_points)),
    }


def _mask_counts(mask_path: Path, person_idx: int) -> tuple[int, int, int]:
    selected = common.load_contact_mask(mask_path, person_idx)
    counterpart = common.load_contact_mask(mask_path, 1 - person_idx)
    selected_count = int(np.count_nonzero(selected)) if selected is not None else -1
    counterpart_count = int(np.count_nonzero(counterpart)) if counterpart is not None else -1
    frames = int(selected.shape[0]) if selected is not None else -1
    return selected_count, counterpart_count, frames


def _load_counterpart_cloud(row: dict[str, str], half: np.ndarray) -> tuple[dict[str, object], str]:
    task = common.counterpart_source_task(row["source_task"])
    if task is None or not (common.BASE / task).is_dir():
        return _cloud_summary(np.zeros((0, 3), dtype=np.float64), half), "false"
    try:
        model, qpos, contact_pos, active, _mask = common.load_source_case(task)
        points, _hands, _frames = common.contact_points_local(model, qpos, contact_pos, active)
    except Exception:
        return _cloud_summary(np.zeros((0, 3), dtype=np.float64), half), "error"
    return _cloud_summary(points, half), "true"


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        return float("nan")
    return float(np.linalg.norm(a - b))


def _free_axis_distance(anchor: np.ndarray, centroid: np.ndarray, face: str) -> float:
    if not (np.all(np.isfinite(anchor)) and np.all(np.isfinite(centroid))) or not face:
        return float("nan")
    axis = common.AXES.index(face[1])
    keep = [idx for idx in range(3) if idx != axis]
    return float(np.linalg.norm(anchor[keep] - centroid[keep]))


def _decision(row: dict[str, str], selected: dict[str, object], height_local: str, height_dot: float) -> tuple[str, str]:
    reasons: list[str] = []
    axis_ok = height_local == "z" and height_dot >= 0.75
    if not axis_ok:
        reasons.append(f"axis_height_local_{height_local}_{height_dot:.2f}")
    side = str(selected["side_face"])
    side_frac = float(selected["side_frac"])
    margin = float(selected["side_margin"])
    if int(selected["total"]) < 20:
        reasons.append("too_few_selected_points")
    if not side:
        reasons.append("no_selected_side")
    if side_frac < 0.45:
        reasons.append(f"weak_side_frac_{side_frac:.2f}")
    if margin < 0.12:
        reasons.append(f"ambiguous_side_margin_{margin:.2f}")
    if common.as_bool(row.get("anchor_face_review", "false")):
        reasons.append(f"manifest_review_{row.get('anchor_face_review_reason', '')}")
    if side and side != row.get("anchor_face", ""):
        reasons.append(f"manifest_anchor_face_mismatch_{row.get('anchor_face', '')}_vs_{side}")
    if not reasons:
        return "ready_for_d6_sanity", "ok"
    if any(reason.startswith("axis_height") for reason in reasons):
        return "review_axis_before_d6", ";".join(reasons)
    if any(reason.startswith("manifest_anchor_face_mismatch") for reason in reasons):
        return "review_endpoint_before_d6", ";".join(reasons)
    if side and side_frac >= 0.45 and margin >= 0.12:
        return "ready_with_endpoint_review", ";".join(reasons)
    return "reject_unstable_side", ";".join(reasons)


def _plot_panel(
    variant: str,
    half: np.ndarray,
    anchor: np.ndarray,
    points: np.ndarray,
    hands: np.ndarray,
    selected: dict[str, object],
    axis_matrix: np.ndarray,
    status: str,
    reason: str,
    out_path: Path,
) -> None:
    colors = np.asarray(["tab:blue", "tab:red"])
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), dpi=140)
    views = [
        (axes[0, 0], "XY", 0, 1),
        (axes[0, 1], "XZ", 0, 2),
        (axes[1, 0], "YZ", 1, 2),
    ]
    robust = np.asarray(selected["side_centroid"], dtype=np.float64)
    for ax, title, a, b in views:
        pad_a = max(0.08, float(half[a]) * 0.35)
        pad_b = max(0.08, float(half[b]) * 0.35)
        xs = [-half[a], half[a], half[a], -half[a], -half[a]]
        ys = [-half[b], -half[b], half[b], half[b], -half[b]]
        ax.plot(xs, ys, color="0.3", linewidth=1.1)
        if len(points):
            for hand in (0, 1):
                sel = hands == hand
                if np.any(sel):
                    ax.scatter(points[sel, a], points[sel, b], s=8, alpha=0.38, color=colors[hand], label=f"hand{hand}")
        ax.scatter(anchor[a], anchor[b], marker="*", s=190, color="#e6b800", edgecolors="black", linewidths=0.8)
        if np.all(np.isfinite(robust)):
            ax.scatter(robust[a], robust[b], marker="x", s=90, color="#1b8a5a", linewidths=2.0)
        ax.set_xlim(-half[a] - pad_a, half[a] + pad_a)
        ax.set_ylim(-half[b] - pad_b, half[b] + pad_b)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(common.AXES[a])
        ax.set_ylabel(common.AXES[b])
        ax.set_title(title)
        ax.grid(True, color="0.88", linewidth=0.6)
    axes[0, 0].legend(loc="upper right", fontsize=8)

    ax = axes[1, 1]
    ax.axis("off")
    counts = selected["counts"]
    matrix_lines = [
        "axis mean | local rows vs world xyz",
        *[
            f"local {common.AXES[i]}: {axis_matrix[i,0]:.3f} {axis_matrix[i,1]:.3f} {axis_matrix[i,2]:.3f}"
            for i in range(3)
        ],
        "",
        f"top_face={selected['top_face']} side_face={selected['side_face']}",
        "counts=" + ", ".join(f"{face}:{counts.get(face, 0)}" for face in common.FACES),
        f"side_frac={float(selected['side_frac']):.3f} margin={float(selected['side_margin']):.3f}",
        f"status={status}",
        f"reason={reason[:120]}",
    ]
    ax.text(0.02, 0.98, "\n".join(matrix_lines), va="top", ha="left", family="monospace", fontsize=9)
    fig.suptitle(
        f"{variant}\nanchor=[{common.point_fmt(anchor)}] selected side centroid=[{common.point_fmt(robust)}]",
        fontsize=10,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path)
    plt.close(fig)


def preflight_row(row: dict[str, str], out_dir: Path) -> dict[str, str]:
    half = common.object_half(row)
    anchor = common.anchor_local(row)
    person_idx = int(row["person_idx"])
    mask_path = Path(row["mask_path_source"])
    model, qpos, contact_pos, active, _mask = common.load_source_case(row["source_task"], mask_npz=mask_path, person_idx=person_idx)
    points, hands, _frames = common.contact_points_local(model, qpos, contact_pos, active)
    axis_matrix, height_local, height_dot, axis_dom = _axis_world_alignment(model, qpos)
    selected = _cloud_summary(points, half)
    counterpart, counterpart_available = _load_counterpart_cloud(row, half)
    selected_mask_count, counterpart_mask_count, mask_frames = _mask_counts(mask_path, person_idx)
    status, reason = _decision(row, selected, height_local, height_dot)

    centroid = np.asarray(selected["robust_centroid"], dtype=np.float64)
    side_centroid = np.asarray(selected["side_centroid"], dtype=np.float64)
    panel = out_dir / f"{row['variant']}_axis_contact_panel.jpg"
    _plot_panel(row["variant"], half, anchor, points, hands, selected, axis_matrix, status, reason, panel)

    counts = selected["counts"]
    counterpart_counts = counterpart["counts"]
    return {
        "variant": row["variant"],
        "source_task": row["source_task"],
        "person_idx": row["person_idx"],
        "manifest_anchor_face": row["anchor_face"],
        "manifest_anchor_review": row["anchor_face_review"],
        "manifest_anchor_review_reason": row["anchor_face_review_reason"],
        "anchor_local": common.point_fmt(anchor),
        "object_half": common.point_fmt(half),
        "axis_dominant_world": axis_dom,
        "height_local_axis": height_local,
        "height_axis_world_z_dot": f"{height_dot:.8g}",
        "selected_points": str(selected["total"]),
        "selected_mask_contact_count": str(selected_mask_count),
        "counterpart_mask_contact_count": str(counterpart_mask_count),
        "mask_frames": str(mask_frames),
        "selected_top_face": str(selected["top_face"]),
        "selected_side_face": str(selected["side_face"]),
        "selected_side_frac": f"{float(selected['side_frac']):.8g}",
        "selected_side_margin": f"{float(selected['side_margin']):.8g}",
        "selected_side_count": str(selected["side_count"]),
        "selected_robust_centroid": common.point_fmt(centroid),
        "selected_side_centroid": common.point_fmt(side_centroid),
        "anchor_to_selected_robust_centroid_m": f"{_dist(anchor, centroid):.8g}",
        "anchor_to_selected_side_centroid_m": f"{_dist(anchor, side_centroid):.8g}",
        "anchor_to_selected_side_centroid_free_axes_m": f"{_free_axis_distance(anchor, side_centroid, str(selected['side_face'])):.8g}",
        "face_count_pos_x": str(counts.get("+x", 0)),
        "face_count_neg_x": str(counts.get("-x", 0)),
        "face_count_pos_y": str(counts.get("+y", 0)),
        "face_count_neg_y": str(counts.get("-y", 0)),
        "face_count_pos_z": str(counts.get("+z", 0)),
        "face_count_neg_z": str(counts.get("-z", 0)),
        "counterpart_cloud_available": counterpart_available,
        "counterpart_points": str(counterpart["total"]),
        "counterpart_side_face": str(counterpart["side_face"]),
        "counterpart_side_frac": f"{float(counterpart['side_frac']):.8g}",
        "counterpart_side_margin": f"{float(counterpart['side_margin']):.8g}",
        "counterpart_face_count_pos_x": str(counterpart_counts.get("+x", 0)),
        "counterpart_face_count_neg_x": str(counterpart_counts.get("-x", 0)),
        "counterpart_face_count_pos_y": str(counterpart_counts.get("+y", 0)),
        "counterpart_face_count_neg_y": str(counterpart_counts.get("-y", 0)),
        "counterpart_face_count_pos_z": str(counterpart_counts.get("+z", 0)),
        "counterpart_face_count_neg_z": str(counterpart_counts.get("-z", 0)),
        "d6_preflight_status": status,
        "d6_preflight_reason": reason,
        "panel_path": common.rel_or_abs(panel),
    }


def _write_csv(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_report(rows: list[dict[str, str]], path: Path, csv_path: Path) -> None:
    total = len(rows)
    ready = sum(row["d6_preflight_status"].startswith("ready") for row in rows)
    reject = sum(row["d6_preflight_status"].startswith("reject") for row in rows)
    axis_review = sum(row["d6_preflight_status"] == "review_axis_before_d6" for row in rows)
    lines = [
        "# E029 Axis/Contact Preflight",
        "",
        f"- Candidates: {total}",
        f"- Ready or ready-with-review: {ready}/{total}",
        f"- Axis review: {axis_review}/{total}",
        f"- Rejected unstable side: {reject}/{total}",
        f"- CSV: `{common.rel_or_abs(csv_path)}`",
        "",
        "## Interpretation",
        "",
        (
            "This preflight does not decide dynamic success. It only checks whether a "
            "candidate has a stable object-local support side and whether the previous "
            "single anchor is close to the active contact cloud. D6 sanity should use "
            "the stable side plus robust free-axis centroid, not the fixed canonical "
            "`0.62 * half_z` height."
        ),
        "",
        "## Per Candidate",
        "",
        "| Variant | selected side | frac/margin | anchor face | anchor-side dist | axis height | status |",
        "|---|---|---:|---|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['variant']}`",
                    row["selected_side_face"],
                    f"{row['selected_side_frac']}/{row['selected_side_margin']}",
                    row["manifest_anchor_face"],
                    row["anchor_to_selected_side_centroid_m"],
                    f"{row['height_local_axis']} ({row['height_axis_world_z_dot']})",
                    row["d6_preflight_status"],
                ]
            )
            + " |"
        )
    lines.extend(["", "## Review Reasons", ""])
    for row in rows:
        lines.append(f"- `{row['variant']}`: {row['d6_preflight_reason']}")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=common.MANIFEST)
    parser.add_argument("--candidates", type=Path, default=common.CANDIDATES)
    parser.add_argument("--out-dir", type=Path, default=common.E029_RESULTS / "preflight")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        preflight_row(row, args.out_dir)
        for row in common.candidate_rows(manifest_path=args.manifest, candidates_path=args.candidates)
    ]
    csv_path = args.out_dir / "axis_contact_summary.csv"
    report_path = args.out_dir / "preflight_report.md"
    _write_csv(rows, csv_path)
    _write_report(rows, report_path, csv_path)
    print(f"Wrote {common.rel_or_abs(csv_path)}")
    print(f"Wrote {common.rel_or_abs(report_path)}")


if __name__ == "__main__":
    main()

