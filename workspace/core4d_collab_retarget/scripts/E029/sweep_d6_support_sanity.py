#!/usr/bin/env python3
"""Sweep E029 compliant D6 no-training support-command settings."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import e029_common as common  # noqa: E402
import check_d6_support_load_path as sanity  # noqa: E402


MANIFEST = common.E029_RESULTS / "d6/manifest.tsv"
OUT_DIR = common.E029_RESULTS / "d6/sanity_sweep"


def _tag(target_mode: str, tau: float, ramp: float, clamp: float, pos_kp: float) -> str:
    return f"{target_mode}_tau{tau:g}_ramp{ramp:g}_f{clamp:g}_kp{pos_kp:g}".replace(".", "p")


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


def _write_md(rows: list[dict[str, str]], path: Path) -> None:
    ranked = sorted(
        rows,
        key=lambda row: (
            row["pass_gate"] != "true",
            float(row["object_pos_err_mean_m"]),
            float(row["force_saturation_frac"]),
        ),
    )
    lines = [
        "# E029 Compliant D6 Support Sanity Sweep",
        "",
        f"- Rows: {len(rows)}",
        f"- Pass: {sum(row['pass_gate'] == 'true' for row in rows)}/{len(rows)}",
        "",
        "| rank | target | tau | ramp | clamp | kp | drift mean/max | target err mean/max | object mean/max | force sat | pass |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for idx, row in enumerate(ranked[:20], start=1):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(idx),
                    row["target_mode"],
                    row["target_lowpass_tau"],
                    row["target_ramp_time"],
                    row["force_clamp"],
                    row["pos_kp"],
                    f"{row['support_drift_mean_m']}/{row['support_drift_max_m']}",
                    f"{row['support_target_err_mean_m']}/{row['support_target_err_max_m']}",
                    f"{row['object_pos_err_mean_m']}/{row['object_pos_err_max_m']}",
                    row["force_saturation_frac"],
                    row["pass_gate"],
                ]
            )
            + " |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--subset", choices=["representative", "candidates"], default="representative")
    parser.add_argument("--force-clamps", type=float, nargs="+", default=[120.0, 250.0, 400.0])
    parser.add_argument("--pos-kps", type=float, nargs="+", default=[200.0])
    parser.add_argument("--render-best-video", action="store_true")
    args = parser.parse_args()

    all_rows = sanity.read_manifest(args.manifest)
    rows = sanity._row_for_mode(all_rows, "d6-compliant-support", args.subset)
    combos: list[tuple[str, float, float, float, float]] = []
    for target_mode, tau, ramp in [
        ("raw_ref", 0.0, 0.0),
        ("raw_ref", 0.0, 0.5),
        ("raw_ref", 0.0, 1.0),
        ("raw_ref", 0.15, 0.0),
        ("raw_ref", 0.25, 0.0),
        ("raw_ref", 0.25, 0.5),
        ("smooth_final", 0.0, 0.0),
        ("smooth_final", 0.0, 0.5),
    ]:
        for clamp in args.force_clamps:
            for pos_kp in args.pos_kps:
                combos.append((target_mode, tau, ramp, clamp, pos_kp))

    summary: list[dict[str, str]] = []
    for target_mode, tau, ramp, clamp, pos_kp in combos:
        tag = _tag(target_mode, tau, ramp, clamp, pos_kp)
        for row in rows:
            summary.append(
                sanity.run_row(
                    row,
                    out_dir=args.out_dir,
                    render_video=False,
                    max_frames=None,
                    target_mode=target_mode,
                    target_lowpass_tau=tau,
                    target_ramp_time=ramp,
                    force_clamp_override=clamp,
                    pos_kp_override=pos_kp,
                    file_tag=tag,
                )
            )

    csv_path = args.out_dir / f"{args.subset}_compliant_sweep.csv"
    md_path = args.out_dir / f"{args.subset}_compliant_sweep.md"
    _write_csv(summary, csv_path)
    _write_md(summary, md_path)

    if args.render_best_video and summary:
        best = sorted(
            summary,
            key=lambda row: (
                row["pass_gate"] != "true",
                float(row["object_pos_err_mean_m"]),
                float(row["force_saturation_frac"]),
            ),
        )[0]
        target_mode = best["target_mode"]
        tau = float(best["target_lowpass_tau"])
        ramp = float(best["target_ramp_time"])
        clamp = float(best["force_clamp"])
        pos_kp = float(best["pos_kp"])
        tag = "best_" + _tag(target_mode, tau, ramp, clamp, pos_kp)
        for row in rows:
            if row["variant"] == best["variant"]:
                sanity.run_row(
                    row,
                    out_dir=args.out_dir,
                    render_video=True,
                    max_frames=None,
                    target_mode=target_mode,
                    target_lowpass_tau=tau,
                    target_ramp_time=ramp,
                    force_clamp_override=clamp,
                    pos_kp_override=pos_kp,
                    file_tag=tag,
                )
                break

    print(f"Wrote {common.rel_or_abs(csv_path)}")
    print(f"Wrote {common.rel_or_abs(md_path)}")


if __name__ == "__main__":
    main()
