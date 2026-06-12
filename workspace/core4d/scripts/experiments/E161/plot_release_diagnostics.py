#!/usr/bin/env python3
"""Plot E161 release-side diagnostics for selected clean8 cases."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO / "workspace/core4d/scripts/eval/runners"))

from eval_E161_surface_release_ablation import (  # noqa: E402
    E161_METHODS,
    RESULT_ROOT,
    aligned_contact_mask,
    release_mask,
    repo_path,
    stage_qpos,
)

VARIANTS = REPO / "workspace/core4d/scripts/experiments/E161/variants.tsv"
OUT_ROOT = RESULT_ROOT / "diagnostics/release_curves"

DEFAULT_CASES = [
    "box021_029_p2",
    "box021_035_p2",
    "box004_083_p2",
    "box026_139_p1",
]


def read_rows() -> list[dict[str, str]]:
    with VARIANTS.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def row_for(rows: list[dict[str, str]], case: str, method: str) -> dict[str, str] | None:
    for row in rows:
        if row["short_case_id"] == case and row["method"] == method:
            return row
    return None


def mean_series(data: np.lib.npyio.NpzFile, key: str) -> np.ndarray | None:
    if key not in data.files:
        return None
    arr = np.asarray(data[key], dtype=np.float64)
    if arr.ndim == 1:
        return arr
    return np.nanmean(arr.reshape(arr.shape[0], -1), axis=1)


def plot_case(case: str, rows: list[dict[str, str]], out_dir: Path) -> None:
    methods = E161_METHODS
    fig, axes = plt.subplots(len(methods), 1, figsize=(12, 8), sharex=True)
    if len(methods) == 1:
        axes = [axes]

    csv_rows: list[dict[str, str | float | int]] = []
    for ax, method in zip(axes, methods):
        row = row_for(rows, case, method)
        if row is None:
            ax.set_title(f"{method}: missing manifest row")
            continue
        qpos_path = stage_qpos(row, "full")
        if not qpos_path.is_file():
            ax.set_title(f"{method}: missing {qpos_path}")
            continue

        data = np.load(qpos_path, allow_pickle=True)
        reward = mean_series(data, "surface_band_rew_mean")
        gate = mean_series(data, "surface_band_gate_mean")
        decay = mean_series(data, "surface_band_decay_factor_mean")
        sdf = mean_series(data, "surface_band_sdf_mean")
        frames = len(reward) if reward is not None else int(np.asarray(data["qpos"]).shape[0])
        x = np.arange(frames)

        mask = aligned_contact_mask(row, frames)
        ref_gate = (mask[:, 0] | mask[:, 1]).astype(np.float64) if mask is not None else np.zeros(frames)
        rel = release_mask(row, frames)
        strict_current = ref_gate if row.get("method_group") == "strictMask" else None

        ax.fill_between(x, 0, 1, where=ref_gate > 0.5, alpha=0.08, color="tab:green", label="ref contact")
        ax.fill_between(x, 0, 1, where=rel, alpha=0.08, color="tab:red", label="release")
        if reward is not None:
            ax.plot(x, reward, label="surface reward", color="tab:blue", linewidth=1.4)
        if gate is not None:
            ax.plot(x, gate, label="rollout gate mean", color="tab:orange", linewidth=1.0)
        if decay is not None:
            ax.plot(x, decay, label="decay factor", color="tab:purple", linewidth=1.0)
        if strict_current is not None:
            ax.plot(x, strict_current, label="strict current gate", color="black", linewidth=1.0)
        if sdf is not None:
            ax2 = ax.twinx()
            ax2.plot(x, sdf, label="surface sdf mean", color="tab:gray", alpha=0.45, linewidth=0.8)
            ax2.axhline(0.0, color="tab:gray", alpha=0.25, linewidth=0.8)
            ax2.set_ylabel("sdf m")
        ax.set_ylim(-0.05, 1.15)
        ax.set_ylabel("reward/gate")
        ax.set_title(method)
        ax.grid(alpha=0.2)

        if reward is not None:
            for i in range(frames):
                csv_rows.append(
                    {
                        "case": case,
                        "method": method,
                        "frame": i,
                        "ref_contact": float(ref_gate[i]),
                        "release": int(bool(rel[i])),
                        "surface_reward": float(reward[i]),
                        "rollout_gate_mean": float(gate[i]) if gate is not None else np.nan,
                        "decay_factor": float(decay[i]) if decay is not None else np.nan,
                        "surface_sdf_mean_m": float(sdf[i]) if sdf is not None else np.nan,
                        "strict_current_gate": float(strict_current[i]) if strict_current is not None else np.nan,
                    }
                )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", fontsize=8)
    fig.suptitle(f"E161 release diagnostics: {case}")
    fig.tight_layout(rect=(0, 0, 0.92, 0.96))
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{case}_release_diagnostics.png", dpi=160)
    plt.close(fig)

    if csv_rows:
        with (out_dir / f"{case}_release_diagnostics.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", nargs="*", default=DEFAULT_CASES)
    parser.add_argument("--out-dir", type=Path, default=OUT_ROOT)
    args = parser.parse_args()

    rows = read_rows()
    for case in args.cases:
        plot_case(case, rows, args.out_dir)
    print(f"Wrote E161 release diagnostics to {args.out_dir}")


if __name__ == "__main__":
    main()
