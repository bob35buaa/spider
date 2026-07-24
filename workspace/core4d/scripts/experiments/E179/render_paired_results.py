#!/usr/bin/env python3
"""Render all E179 Full rows and build 16 E173-vs-E179 paired videos."""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e179_common as C  # noqa: E402

sys.path.insert(
    0,
    str(
        C.REPO
        / "workspace/core4d/scripts/experiments/E168"
    ),
)
from render_a100_cem_videos import render_row  # noqa: E402


COMPLETE_STATUSES = {
    "run_complete_pending_eval",
    "run_complete",
    "eval_complete",
}
PAIR_ROOT = (
    C.RESULTS / "s6_downstream/render/full/paired"
)


def baseline_videos() -> dict[str, Path]:
    rows = C.read_tsv(C.E173_FULL_MANIFEST)
    selected = {
        row["case_id"]: C.repo_path(row["video"])
        for row in rows
        if row.get("object_key") == "box023"
    }
    if len(selected) != 16:
        raise ValueError(
            f"E173 box023 baseline videos rows={len(selected)}"
        )
    return selected


def pair_video(
    *, case_id: str, baseline: Path, treatment: Path, overwrite: bool
) -> tuple[str, Path]:
    output = (
        PAIR_ROOT
        / f"{case_id}_E173_PRG_vs_E179_noPRG.mp4"
    )
    if not baseline.is_file() or not treatment.is_file():
        return "not_ready", output
    if output.is_file() and not overwrite:
        return "existing", output
    output.parent.mkdir(parents=True, exist_ok=True)
    filters = (
        "[0:v]scale=960:540:force_original_aspect_ratio=decrease,"
        "pad=960:540:(ow-iw)/2:(oh-ih)/2:black,setsar=1,"
        "drawtext=text='E173 PRG':x=18:y=16:fontsize=30:"
        "fontcolor=white:box=1:boxcolor=black@0.65[v0];"
        "[1:v]scale=960:540:force_original_aspect_ratio=decrease,"
        "pad=960:540:(ow-iw)/2:(oh-ih)/2:black,setsar=1,"
        "drawtext=text='E179 E167A no-PRG':x=18:y=16:fontsize=30:"
        "fontcolor=white:box=1:boxcolor=black@0.65[v1];"
        "[v0][v1]hstack=inputs=2[out]"
    )
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-nostdin",
            "-loglevel",
            "error",
            "-i",
            str(baseline),
            "-i",
            str(treatment),
            "-filter_complex",
            filters,
            "-map",
            "[out]",
            "-an",
            "-c:v",
            "libx264",
            "-crf",
            "22",
            "-preset",
            "medium",
            "-pix_fmt",
            "yuv420p",
            "-shortest",
            str(output),
        ],
        check=True,
    )
    return "rendered", output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=(
            C.RESULTS
            / "s6_downstream/manifests/cem_full_manifest.tsv"
        ),
    )
    parser.add_argument("--cases", nargs="*", default=[])
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--require-all", action="store_true")
    args = parser.parse_args()
    selectors = set(args.cases)
    rows = [
        row
        for row in C.read_tsv(args.manifest)
        if not selectors or row["case_id"] in selectors
    ]
    if not rows:
        raise SystemExit("no E179 manifest rows selected")
    baseline = baseline_videos()
    counts: Counter[str] = Counter()
    evidence = []
    failures = []
    for row in rows:
        case_id = row["case_id"]
        treatment = C.repo_path(row["video"])
        required = [
            C.repo_path(row[key])
            for key in (
                "outdir_npz",
                "config_act",
                "scene_act",
                "trajectory",
            )
        ]
        ready = (
            row["status"] in COMPLETE_STATUSES
            and all(path.is_file() for path in required)
        )
        if not baseline[case_id].is_file():
            failures.append(
                f"{case_id}:missing_e173_baseline_video"
            )
            counts["baseline_missing"] += 1
            continue
        counts["baseline_ready"] += 1
        try:
            if treatment.is_file() and not args.overwrite:
                counts["treatment_existing"] += 1
            elif not ready:
                counts["treatment_not_ready"] += 1
            elif args.dry_run:
                counts["treatment_ready"] += 1
            else:
                render_row(
                    row,
                    out_path=treatment,
                    max_frames=args.max_frames,
                )
                counts["treatment_rendered"] += 1

            if args.dry_run:
                pair_status = (
                    "ready"
                    if baseline[case_id].is_file()
                    and treatment.is_file()
                    else "not_ready"
                )
                pair_path = (
                    PAIR_ROOT
                    / f"{case_id}_E173_PRG_vs_E179_noPRG.mp4"
                )
            else:
                pair_status, pair_path = pair_video(
                    case_id=case_id,
                    baseline=baseline[case_id],
                    treatment=treatment,
                    overwrite=args.overwrite,
                )
            counts[f"paired_{pair_status}"] += 1
            evidence.append(
                {
                    "case_id": case_id,
                    "e173_video": C.rel(baseline[case_id]),
                    "e179_video": C.rel(treatment),
                    "paired_video": C.rel(pair_path),
                    "paired_status": pair_status,
                }
            )
        except Exception as exc:
            failures.append(
                f"{case_id}:{type(exc).__name__}:{exc}"
            )
            counts["failed"] += 1

    output = C.RESULTS / "s6_downstream/render/full"
    C.write_tsv(output / "paired_video_manifest.tsv", evidence)
    C.write_json(
        output / "render_summary.json",
        {
            "created_at": C.now(),
            "selected_rows": len(rows),
            "counts": dict(counts),
            "failures": failures,
            "status": "pass" if not failures else "fail",
        },
    )
    print(
        " ".join(
            f"{key}={value}" for key, value in sorted(counts.items())
        )
    )
    if failures:
        return 1
    if args.require_all:
        ready_pairs = sum(
            row["paired_status"] in {"rendered", "existing"}
            for row in evidence
        )
        if len(rows) != 16 or ready_pairs != 16:
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
