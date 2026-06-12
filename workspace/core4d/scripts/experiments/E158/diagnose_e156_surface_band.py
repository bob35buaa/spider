#!/usr/bin/env python3
"""Generate E158 Stage-A SDF/mask diagnostic curves from E156 clean6 results."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from eval.core.core_metrics import (  # noqa: E402
    EvalConfig,
    HAND_GEOMS,
    geom_object_sdf,
    mj_id,
    npz_qpos,
    object_collision_geoms,
)


REPO = Path(__file__).resolve().parents[5]
E156_VARIANTS = REPO / "workspace/core4d/scripts/experiments/E156/variants.tsv"
OUT_ROOT = REPO / "workspace/core4d/results/E158/gate_surface_band/diagnostics"

CLEAN6_CASES = [
    "box021_035_p1",
    "box021_035_p2",
    "box021_029_p2",
    "box004_083_p1",
    "box004_083_p2",
    "box023_person2",
]
METHODS = ["spider-rubberhand", "+gateA", "E155_decay"]
BANDS = [
    ("deep_lt_neg5mm", lambda x: x < -0.005),
    ("pen3_neg5_to_neg3mm", lambda x: (-0.005 <= x) & (x < -0.003)),
    ("shallow_neg3_to_0mm", lambda x: (-0.003 <= x) & (x < 0.0)),
    ("surface_0_to_30mm", lambda x: (0.0 <= x) & (x <= 0.030)),
    ("far_gt_30mm", lambda x: x > 0.030),
]


def repo_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO / p


def rel(path: str | Path) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(REPO.resolve()))
    except Exception:
        return str(path)


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def load_mask(path: Path, person_idx: int) -> np.ndarray:
    data = np.load(path, allow_pickle=True)
    mask = np.asarray(data["spider_contact_mask_3cm"])
    return mask[:, person_idx, 0].astype(bool) | mask[:, person_idx, 1].astype(bool)


def sdf_series(row: dict[str, str], mesh_sample_count: int) -> tuple[np.ndarray, np.ndarray]:
    qpos, _ = npz_qpos(repo_path(row["outdir_npz"]))
    scene = repo_path(row["rubber_scene_act"])
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    object_gids = object_collision_geoms(model)
    hand_gids = [mj_id(model, mujoco.mjtObj.mjOBJ_GEOM, name) for name in HAND_GEOMS]
    hand_gids = [gid for gid in hand_gids if gid >= 0]
    vals: list[float] = []
    for q in qpos:
        data.qpos[:] = q
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        hand_vals = [geom_object_sdf(model, data, gid, object_gids, mesh_sample_count) for gid in hand_gids]
        vals.append(float(min(hand_vals)) if hand_vals else math.nan)
    mask = load_mask(repo_path(row["mask_path"]), int(row["person_idx"]))
    if mask.shape[0] != len(vals):
        raise ValueError(
            f"contact mask length mismatch for {row['short_case_id']}:{row['method']}: "
            f"mask={mask.shape[0]} qpos={len(vals)} path={row['mask_path']}"
        )
    return np.asarray(vals, dtype=np.float64), mask


def release_mask(contact_mask: np.ndarray) -> np.ndarray:
    out = np.zeros_like(contact_mask, dtype=bool)
    if contact_mask.any():
        last = int(len(contact_mask) - 1 - np.argmax(contact_mask[::-1]))
        out[last + 1 :] = True
    return out


def band_fraction_rows(case: str, method: str, sdf: np.ndarray, mask: np.ndarray) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    phases = {
        "all": np.ones_like(mask, dtype=bool),
        "in_mask": mask,
        "release": release_mask(mask),
    }
    for phase, pmask in phases.items():
        denom = int(pmask.sum())
        row: dict[str, Any] = {
            "short_case_id": case,
            "method": method,
            "phase": phase,
            "frames": denom,
        }
        for name, pred in BANDS:
            row[name] = float(np.mean(pred(sdf[pmask]))) if denom else math.nan
        rows.append(row)
    return rows


def plot_time_curve(case: str, method: str, sdf: np.ndarray, mask: np.ndarray, out: Path) -> None:
    t = np.arange(len(sdf)) / 30.0
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axhspan(0.0, 0.030, color="#d9ead3", alpha=0.45, label="surface 0..30mm")
    ax.fill_between(t, -0.08, 0.08, where=mask, color="#fff2cc", alpha=0.35, step="pre", label="contact mask")
    ax.plot(t, sdf, color="#1f77b4", linewidth=1.4, label="min hand-object SDF")
    for y, label, color in [
        (0.0, "0mm", "#222222"),
        (-0.003, "-3mm", "#ff7f0e"),
        (-0.005, "-5mm", "#d62728"),
        (0.030, "+30mm", "#2ca02c"),
    ]:
        ax.axhline(y, color=color, linestyle="--", linewidth=0.9, label=label)
    ax.set_title(f"{case} / {method}")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("SDF (m)")
    ax.set_ylim(-0.08, 0.08)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", ncol=3, fontsize=8)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_case_compare(case: str, series: dict[str, tuple[np.ndarray, np.ndarray]], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    max_len = max(len(sdf) for sdf, _ in series.values())
    t = np.arange(max_len) / 30.0
    any_mask = np.zeros(max_len, dtype=bool)
    for sdf, mask in series.values():
        any_mask[: len(mask)] |= mask
    ax.axhspan(0.0, 0.030, color="#d9ead3", alpha=0.45, label="surface 0..30mm")
    ax.fill_between(t, -0.08, 0.08, where=any_mask, color="#fff2cc", alpha=0.30, step="pre", label="contact mask")
    for method, (sdf, _) in series.items():
        ax.plot(np.arange(len(sdf)) / 30.0, sdf, linewidth=1.2, label=method)
    for y, color in [(0.0, "#222222"), (-0.003, "#ff7f0e"), (-0.005, "#d62728"), (0.030, "#2ca02c")]:
        ax.axhline(y, color=color, linestyle="--", linewidth=0.8)
    ax.set_title(f"{case} method comparison")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("SDF (m)")
    ax.set_ylim(-0.08, 0.08)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", ncol=3, fontsize=8)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_band_stack(rows: list[dict[str, Any]], out: Path) -> None:
    phases = ["in_mask", "release"]
    labels = [f"{row['short_case_id']}:{row['method']}:{row['phase']}" for row in rows if row["phase"] in phases]
    data_rows = [row for row in rows if row["phase"] in phases]
    fig, ax = plt.subplots(figsize=(14, max(4, 0.35 * len(data_rows))))
    left = np.zeros(len(data_rows))
    colors = ["#d62728", "#ff9896", "#ffbb78", "#2ca02c", "#bdbdbd"]
    for (name, _), color in zip(BANDS, colors):
        vals = np.asarray([float(row[name]) if row[name] == row[name] else 0.0 for row in data_rows])
        ax.barh(np.arange(len(data_rows)), vals, left=left, label=name, color=color)
        left += vals
    ax.set_yticks(np.arange(len(data_rows)), labels=labels, fontsize=7)
    ax.set_xlim(0, 1)
    ax.set_xlabel("fraction")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh-sample-count", type=int, default=800)
    args = parser.parse_args()

    rows = read_tsv(E156_VARIANTS)
    selected = {
        (row["short_case_id"], row["method"]): row
        for row in rows
        if row["short_case_id"] in CLEAN6_CASES and row["method"] in METHODS
    }
    missing = [(case, method) for case in CLEAN6_CASES for method in METHODS if (case, method) not in selected]
    if missing:
        raise SystemExit(f"missing E156 rows: {missing}")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    timeseries_rows: list[dict[str, Any]] = []
    band_rows: list[dict[str, Any]] = []
    for case in CLEAN6_CASES:
        case_series: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for method in METHODS:
            row = selected[(case, method)]
            sdf, mask = sdf_series(row, args.mesh_sample_count)
            case_series[method] = (sdf, mask)
            plot_time_curve(case, method, sdf, mask, OUT_ROOT / "time_curves" / f"{case}_{method.replace('+', 'plus')}.png")
            band_rows.extend(band_fraction_rows(case, method, sdf, mask))
            for frame, value in enumerate(sdf):
                timeseries_rows.append(
                    {
                        "short_case_id": case,
                        "method": method,
                        "frame": frame,
                        "time_s": frame / 30.0,
                        "sdf_m": float(value),
                        "contact_mask": int(mask[frame]),
                    }
                )
        plot_case_compare(case, case_series, OUT_ROOT / "method_compare" / f"{case}_sdf_compare.png")

    plot_band_stack(band_rows, OUT_ROOT / "band_fraction_stacked.png")
    write_tsv(
        OUT_ROOT / "e158_stageA_sdf_timeseries.tsv",
        timeseries_rows,
        ["short_case_id", "method", "frame", "time_s", "sdf_m", "contact_mask"],
    )
    write_tsv(
        OUT_ROOT / "e158_stageA_band_fractions.tsv",
        band_rows,
        ["short_case_id", "method", "phase", "frames"] + [name for name, _ in BANDS],
    )
    print(
        "E158 diagnostics: "
        f"timeseries_rows={len(timeseries_rows)} band_rows={len(band_rows)} out={rel(OUT_ROOT)}"
    )


if __name__ == "__main__":
    main()
