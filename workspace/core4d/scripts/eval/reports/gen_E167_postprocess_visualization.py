#!/usr/bin/env python3
"""Generate E167 B2 z-only postprocess visualization dashboard."""
from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_ROOT = Path("workspace/core4d/results/E167/holosoma_zonly/postprocess/full")


def case_from_name(path: Path) -> str:
    name = path.name
    prefix = "E167_"
    suffix = "_E167A_B2.npz"
    if not name.startswith(prefix) or not name.endswith(suffix):
        raise ValueError(f"Unexpected E167 postprocess npz name: {name}")
    return name[len(prefix) : -len(suffix)]


def positive_median_dt(time_arr: np.ndarray | None, n: int) -> float:
    if time_arr is None:
        return 1.0
    t = np.asarray(time_arr)
    if t.ndim == 0 or len(t) < 2:
        return 1.0
    if t.ndim > 1:
        t = t[:, 0]
    diff = np.diff(t.astype(float))
    diff = diff[np.isfinite(diff) & (diff > 0)]
    if diff.size == 0:
        return 1.0
    return float(np.median(diff))


def p95_abs_diff(values: np.ndarray, order: int, dt: float) -> float:
    if values.shape[0] <= order:
        return 0.0
    deriv = np.diff(values.astype(float), n=order, axis=0) / (dt**order)
    return float(np.percentile(np.abs(deriv), 95))


def p95_abs(values: np.ndarray) -> float:
    return float(np.percentile(np.abs(values.astype(float)), 95))


def finite_reduction(before: float, after: float) -> float:
    if not np.isfinite(before) or abs(before) < 1e-12:
        return 0.0
    return float((before - after) / before)


def load_pair(report_path: Path) -> tuple[str, dict[str, Any], dict[str, np.ndarray], dict[str, np.ndarray]]:
    report = json.loads(report_path.read_text())
    out_path = Path(report["output"])
    in_path = Path(report["input"])
    if not out_path.exists():
        out_path = report_path.with_suffix("")
    if not in_path.exists():
        raise FileNotFoundError(f"Input npz from report does not exist: {in_path}")
    case = case_from_name(out_path)
    before_npz = np.load(in_path, allow_pickle=True)
    after_npz = np.load(out_path, allow_pickle=True)
    before = {k: before_npz[k] for k in before_npz.files}
    after = {k: after_npz[k] for k in after_npz.files}
    return case, report, before, after


def summarize_case(case: str, report: dict[str, Any], before: dict[str, np.ndarray], after: dict[str, np.ndarray]) -> dict[str, Any]:
    qpos_before = np.asarray(before["qpos"])
    qpos_after = np.asarray(after["qpos"])
    qvel_before = np.asarray(before["qvel"])
    qvel_after = np.asarray(after["qvel"])
    time_arr = np.asarray(after["time"]) if "time" in after else None
    dt = positive_median_dt(time_arr, qpos_after.shape[0])

    qpos_z_before = qpos_before[..., 2]
    qpos_z_after = qpos_after[..., 2]
    qvel_z_before = qvel_before[..., 2]
    qvel_z_after = qvel_after[..., 2]

    qpos_non_z_mask = np.ones(qpos_before.shape[-1], dtype=bool)
    qpos_non_z_mask[2] = False
    qvel_non_z_mask = np.ones(qvel_before.shape[-1], dtype=bool)
    qvel_non_z_mask[2] = False

    accel_before = p95_abs_diff(qpos_z_before, 2, dt)
    accel_after = p95_abs_diff(qpos_z_after, 2, dt)
    jerk_before = p95_abs_diff(qpos_z_before, 3, dt)
    jerk_after = p95_abs_diff(qpos_z_after, 3, dt)

    xy_checks = report.get("xy_checks", {})
    z_checks = report.get("z_checks", {})
    return {
        "case": case,
        "frames": int(qpos_after.shape[0]),
        "tracks": int(qpos_after.shape[1]) if qpos_after.ndim > 1 else 1,
        "dt_s": dt,
        "window": report.get("window"),
        "polyorder": report.get("polyorder"),
        "qpos_z_max_delta_m": float(np.max(np.abs(qpos_z_after - qpos_z_before))),
        "qvel_z_max_delta": float(np.max(np.abs(qvel_z_after - qvel_z_before))),
        "qpos_xy_max_delta_report": float(xy_checks.get("qpos_xy_max_abs_delta", np.nan)),
        "qvel_xy_max_delta_report": float(xy_checks.get("qvel_xy_max_abs_delta", np.nan)),
        "xy_max_delta_report": float(report.get("xy_max_abs_delta", np.nan)),
        "qpos_non_z_max_delta": float(np.max(np.abs(qpos_after[..., qpos_non_z_mask] - qpos_before[..., qpos_non_z_mask]))),
        "qvel_non_z_max_delta": float(np.max(np.abs(qvel_after[..., qvel_non_z_mask] - qvel_before[..., qvel_non_z_mask]))),
        "qpos_z_max_delta_report": float(z_checks.get("qpos_z_max_abs_delta", np.nan)),
        "qvel_z_max_delta_report": float(z_checks.get("qvel_z_max_abs_delta", np.nan)),
        "qpos_z_accel_p95_before": accel_before,
        "qpos_z_accel_p95_after": accel_after,
        "qpos_z_accel_p95_reduction": finite_reduction(accel_before, accel_after),
        "qpos_z_jerk_p95_before": jerk_before,
        "qpos_z_jerk_p95_after": jerk_after,
        "qpos_z_jerk_p95_reduction": finite_reduction(jerk_before, jerk_after),
        "qvel_z_p95_before": p95_abs(qvel_z_before),
        "qvel_z_p95_after": p95_abs(qvel_z_after),
        "qvel_z_p95_reduction": finite_reduction(p95_abs(qvel_z_before), p95_abs(qvel_z_after)),
    }


def plot_summary(rows: list[dict[str, Any]], out_path: Path) -> None:
    cases = [r["case"] for r in rows]
    x = np.arange(len(cases))

    fig, axs = plt.subplots(2, 2, figsize=(14, 8.5))
    ax = axs[0, 0]
    ax.bar(x, [r["qpos_z_max_delta_m"] * 100 for r in rows], color="#2b6cb0")
    ax.set_title("Root-z smoothing magnitude")
    ax.set_ylabel("max |delta z| (cm)")
    ax.set_xticks(x, cases, rotation=35, ha="right", fontsize=8)

    ax = axs[0, 1]
    width = 0.38
    ax.bar(x - width / 2, [r["qpos_z_jerk_p95_before"] for r in rows], width, label="before", color="#b7791f")
    ax.bar(x + width / 2, [r["qpos_z_jerk_p95_after"] for r in rows], width, label="after", color="#2f855a")
    ax.set_title("Root-z jerk p95")
    ax.set_ylabel("abs jerk p95")
    ax.set_xticks(x, cases, rotation=35, ha="right", fontsize=8)
    ax.legend(fontsize=8)

    ax = axs[1, 0]
    ax.bar(x, [r["qpos_non_z_max_delta"] for r in rows], label="qpos non-z", color="#4a5568")
    ax.bar(x, [r["qvel_non_z_max_delta"] for r in rows], bottom=[r["qpos_non_z_max_delta"] for r in rows], label="qvel non-z", color="#718096")
    ax.set_title("Non-z invariance audit")
    ax.set_ylabel("max abs delta")
    ax.set_xticks(x, cases, rotation=35, ha="right", fontsize=8)
    ax.legend(fontsize=8)

    ax = axs[1, 1]
    ax.bar(x - width / 2, [r["qvel_z_p95_before"] for r in rows], width, label="before", color="#805ad5")
    ax.bar(x + width / 2, [r["qvel_z_p95_after"] for r in rows], width, label="after", color="#319795")
    ax.set_title("Root-z velocity p95")
    ax.set_ylabel("abs qvel-z p95")
    ax.set_xticks(x, cases, rotation=35, ha="right", fontsize=8)
    ax.legend(fontsize=8)

    fig.suptitle("E167 B2 z-only postprocess summary", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_case(case: str, before: dict[str, np.ndarray], after: dict[str, np.ndarray], out_path: Path) -> None:
    qpos_before = np.asarray(before["qpos"])
    qpos_after = np.asarray(after["qpos"])
    qvel_before = np.asarray(before["qvel"])
    qvel_after = np.asarray(after["qvel"])
    time_arr = np.asarray(after["time"]) if "time" in after else None
    if time_arr is None:
        t = np.arange(qpos_after.shape[0])
    else:
        t = time_arr[:, 0] if time_arr.ndim > 1 else time_arr

    fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    tracks = qpos_after.shape[1] if qpos_after.ndim > 2 else 1
    colors = ["#2b6cb0", "#c05621", "#2f855a", "#805ad5"]
    for track in range(tracks):
        label = f"track {track}"
        c = colors[track % len(colors)]
        axs[0].plot(t, qpos_before[:, track, 2], "--", color=c, alpha=0.45, lw=1.0, label=f"{label} before")
        axs[0].plot(t, qpos_after[:, track, 2], "-", color=c, lw=1.4, label=f"{label} after")
        axs[1].plot(t, qvel_before[:, track, 2], "--", color=c, alpha=0.45, lw=1.0, label=f"{label} before")
        axs[1].plot(t, qvel_after[:, track, 2], "-", color=c, lw=1.4, label=f"{label} after")
        axs[2].plot(t, qpos_after[:, track, 2] - qpos_before[:, track, 2], "-", color=c, lw=1.2, label=label)

    axs[0].set_title(f"{case}: root z before/after")
    axs[0].set_ylabel("qpos z (m)")
    axs[1].set_title("root z velocity before/after")
    axs[1].set_ylabel("qvel z")
    axs[2].set_title("z-only smoothing delta")
    axs[2].set_ylabel("after - before (m)")
    axs[2].set_xlabel("time (s)")
    for ax in axs:
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def write_tsv(rows: list[dict[str, Any]], out_path: Path) -> None:
    keys = [
        "case",
        "frames",
        "tracks",
        "dt_s",
        "window",
        "polyorder",
        "qpos_z_max_delta_m",
        "qvel_z_max_delta",
        "qpos_non_z_max_delta",
        "qvel_non_z_max_delta",
        "xy_max_delta_report",
        "qpos_z_accel_p95_before",
        "qpos_z_accel_p95_after",
        "qpos_z_accel_p95_reduction",
        "qpos_z_jerk_p95_before",
        "qpos_z_jerk_p95_after",
        "qpos_z_jerk_p95_reduction",
        "qvel_z_p95_before",
        "qvel_z_p95_after",
        "qvel_z_p95_reduction",
    ]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def format_float(value: Any, digits: int = 4) -> str:
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    try:
        v = float(value)
    except (TypeError, ValueError):
        return html.escape(str(value))
    if not np.isfinite(v):
        return "nan"
    return f"{v:.{digits}g}"


def write_html(rows: list[dict[str, Any]], case_images: list[tuple[str, str]], out_path: Path) -> None:
    summary_cols = [
        ("case", "case"),
        ("qpos_z_max_delta_m", "max dz (m)"),
        ("qvel_z_max_delta", "max dvz"),
        ("qpos_non_z_max_delta", "qpos non-z delta"),
        ("qvel_non_z_max_delta", "qvel non-z delta"),
        ("qpos_z_jerk_p95_reduction", "jerk p95 reduction"),
        ("qvel_z_p95_reduction", "qvel p95 reduction"),
    ]
    table_rows = []
    for row in rows:
        cells = "".join(f"<td>{html.escape(format_float(row[k]))}</td>" for k, _ in summary_cols)
        table_rows.append(f"<tr>{cells}</tr>")
    cards = []
    for case, img in case_images:
        cards.append(
            f"""
            <section class="case">
              <h2>{html.escape(case)}</h2>
              <img src="{html.escape(img)}" alt="{html.escape(case)} root z postprocess plot">
            </section>
            """
        )
    headers = "".join(f"<th>{html.escape(label)}</th>" for _, label in summary_cols)
    out_path.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>E167 B2 z-only postprocess visualization</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #172033;
      --muted: #5b6475;
      --line: #d8dde7;
      --bg: #f7f8fb;
      --panel: #ffffff;
    }}
    body {{
      margin: 0;
      font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background: var(--bg);
    }}
    main {{
      max-width: 1240px;
      margin: 0 auto;
      padding: 28px 22px 48px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 26px;
      font-weight: 700;
    }}
    h2 {{
      margin: 0 0 12px;
      font-size: 18px;
    }}
    p {{
      margin: 0 0 18px;
      color: var(--muted);
    }}
    .panel, .case {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
      margin: 16px 0;
    }}
    img {{
      width: 100%;
      height: auto;
      display: block;
    }}
    table {{
      border-collapse: collapse;
      width: 100%;
      font-variant-numeric: tabular-nums;
    }}
    th, td {{
      border-bottom: 1px solid var(--line);
      padding: 8px 7px;
      text-align: right;
      white-space: nowrap;
    }}
    th:first-child, td:first-child {{
      text-align: left;
    }}
    th {{
      color: var(--muted);
      font-size: 12px;
      font-weight: 600;
    }}
    .scroll {{
      overflow-x: auto;
    }}
  </style>
</head>
<body>
<main>
  <h1>E167 B2 z-only postprocess visualization</h1>
  <p>Generated from 7 postprocess reports. The audit compares each B2 output against its E167A input and verifies that only root-z channels changed.</p>
  <section class="panel">
    <h2>Summary</h2>
    <img src="summary.png" alt="E167 postprocess summary plots">
  </section>
  <section class="panel">
    <h2>Audit Table</h2>
    <div class="scroll">
      <table>
        <thead><tr>{headers}</tr></thead>
        <tbody>{''.join(table_rows)}</tbody>
      </table>
    </div>
  </section>
  {''.join(cards)}
</main>
</body>
</html>
""",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()

    root = args.root
    out_dir = args.out_dir or root / "viz"
    out_dir.mkdir(parents=True, exist_ok=True)

    reports = sorted(root.glob("*.zonly_smooth_report.json"))
    if not reports:
        raise SystemExit(f"No zonly smooth reports found under {root}")

    rows: list[dict[str, Any]] = []
    case_images: list[tuple[str, str]] = []
    for report_path in reports:
        case, report, before, after = load_pair(report_path)
        rows.append(summarize_case(case, report, before, after))
        img_name = f"{case}_root_z.png"
        plot_case(case, before, after, out_dir / img_name)
        case_images.append((case, img_name))

    rows.sort(key=lambda r: r["case"])
    case_images.sort(key=lambda p: p[0])
    write_tsv(rows, out_dir / "summary.tsv")
    plot_summary(rows, out_dir / "summary.png")
    write_html(rows, case_images, out_dir / "index.html")
    print(f"Wrote {out_dir / 'index.html'}")
    print(f"Wrote {out_dir / 'summary.tsv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
