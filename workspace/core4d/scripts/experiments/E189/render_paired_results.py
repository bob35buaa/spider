#!/usr/bin/env python3
"""Render E189 Full CEM results and build E172/E173-vs-E189 paired videos.

E189 reran 43 box004/box024/box001 cases (E172=box004, E173=box024/box001)
with E170 PRG disabled. The CEM stage itself ran with save_video=false, so
self videos are produced offline here from the saved trajectory_mjwp_act.npz
rollout (via the shared E168 render_row helper, unchanged). For each case we
also locate the historical PRG video from the case's E172/E173 source
manifest (workspace/core4d/scripts/experiments/E189/e189_common.py::SOURCES)
and hstack the two into a left(PRG)/right(no-PRG) paired video.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e189_common as C  # noqa: E402

sys.path.insert(
    0, str(C.REPO / "workspace/core4d/scripts/experiments/E168")
)
from render_a100_cem_videos import render_row  # noqa: E402


DEFAULT_MANIFEST = (
    C.RESULTS / "s6_downstream/manifests/cem_full_manifest.tsv"
)
SELF_RENDER_DIR = C.RESULTS / "s6_downstream/render/full"
PAIR_ROOT = C.RESULTS / "s6_downstream/render/paired_e172_e173"
OBJECT_KEYS = ("box004", "box024", "box001")


def load_baseline_index() -> dict[str, dict[str, dict[str, str]]]:
    """object_key -> case_id -> row from that object's E172/E173 manifest."""
    index: dict[str, dict[str, dict[str, str]]] = {}
    for object_key, meta in C.SOURCES.items():
        rows = C.read_tsv(meta["full_manifest"])
        index[object_key] = {
            row["case_id"]: row
            for row in rows
            if row["object_key"] == object_key
        }
    return index


def required_inputs_ready(row: dict[str, str]) -> tuple[bool, list[str]]:
    missing = [
        key
        for key in ("outdir_npz", "config_act", "scene_act")
        if not C.repo_path(row[key]).is_file()
    ]
    return (not missing, missing)


def pair_video(
    *,
    case_id: str,
    object_key: str,
    baseline_experiment_id: str,
    baseline_video: Path,
    self_video: Path,
    skip_existing: bool,
) -> tuple[str, Path]:
    output = PAIR_ROOT / object_key / f"{case_id}.mp4"
    if not baseline_video.is_file():
        return "missing_baseline_video", output
    if not self_video.is_file():
        return "missing_self_video", output
    if output.is_file() and skip_existing:
        return "existing", output
    output.parent.mkdir(parents=True, exist_ok=True)
    filters = (
        "[0:v]scale=960:540:force_original_aspect_ratio=decrease,"
        "pad=960:540:(ow-iw)/2:(oh-ih)/2:black,setsar=1,"
        f"drawtext=text='{baseline_experiment_id} PRG':x=18:y=16:"
        "fontsize=30:fontcolor=white:box=1:boxcolor=black@0.65[v0];"
        "[1:v]scale=960:540:force_original_aspect_ratio=decrease,"
        "pad=960:540:(ow-iw)/2:(oh-ih)/2:black,setsar=1,"
        "drawtext=text='E189 E167A no-PRG':x=18:y=16:fontsize=30:"
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
            str(baseline_video),
            "-i",
            str(self_video),
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--object",
        choices=(*OBJECT_KEYS, "all"),
        default="all",
        help="Restrict to one object_key (default: all three)",
    )
    parser.add_argument(
        "--case-id",
        nargs="*",
        default=[],
        help="Restrict to specific case_id(s), e.g. for single-case debug",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip self/paired videos that already exist on disk",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print/record the plan; do not render anything",
    )
    args = parser.parse_args()

    rows = C.read_tsv(args.manifest)
    if args.object != "all":
        rows = [row for row in rows if row["object_key"] == args.object]
    if args.case_id:
        selectors = set(args.case_id)
        rows = [row for row in rows if row["case_id"] in selectors]
    if not rows:
        raise SystemExit("no E189 manifest rows selected")

    baseline_index = load_baseline_index()

    counts: Counter[str] = Counter()
    evidence: list[dict[str, str]] = []
    failures: list[str] = []

    for row in rows:
        case_id = row["case_id"]
        object_key = row["object_key"]
        variant = row["variant"]
        self_video = C.repo_path(row["video"])
        ready, missing = required_inputs_ready(row)

        baseline_row = baseline_index.get(object_key, {}).get(case_id)
        baseline_experiment_id = C.SOURCES[object_key]["experiment_id"]
        baseline_video = (
            C.repo_path(baseline_row["video"]) if baseline_row else None
        )

        note = ""
        try:
            if not ready:
                self_status = "missing_inputs"
                counts["self_missing_inputs"] += 1
                note = f"missing:{','.join(missing)}"
            elif self_video.is_file() and args.skip_existing:
                self_status = "skipped"
                counts["self_skipped"] += 1
            elif args.dry_run:
                self_status = "would_render"
                counts["self_would_render"] += 1
            else:
                render_row(row, out_path=self_video, max_frames=0)
                self_status = "rendered"
                counts["self_rendered"] += 1
                print(
                    f"[rendered] {case_id} -> {C.rel(self_video)}",
                    flush=True,
                )
        except Exception as exc:
            self_status = "failed"
            counts["self_failed"] += 1
            note = f"{type(exc).__name__}: {exc}"
            failures.append(f"{case_id}:self_render:{note}")
            print(f"[failed] {case_id} self render: {exc}", file=sys.stderr)

        if baseline_row is None:
            paired_status = "missing_baseline_row"
            paired_video = PAIR_ROOT / object_key / f"{case_id}.mp4"
            counts["paired_missing_baseline_row"] += 1
            failures.append(f"{case_id}:no matching {baseline_experiment_id} manifest row")
        elif args.dry_run:
            paired_video = PAIR_ROOT / object_key / f"{case_id}.mp4"
            if baseline_video is not None and baseline_video.is_file():
                if self_video.is_file() or self_status in (
                    "would_render",
                    "rendered",
                ):
                    paired_status = "would_render"
                else:
                    paired_status = "missing_self_video"
            else:
                paired_status = "missing_baseline_video"
            counts[f"paired_{paired_status}"] += 1
        else:
            try:
                paired_status, paired_video = pair_video(
                    case_id=case_id,
                    object_key=object_key,
                    baseline_experiment_id=baseline_experiment_id,
                    baseline_video=baseline_video,
                    self_video=self_video,
                    skip_existing=args.skip_existing,
                )
                counts[f"paired_{paired_status}"] += 1
                if paired_status == "missing_baseline_video":
                    failures.append(
                        f"{case_id}:missing_{baseline_experiment_id}_baseline_video:"
                        f"{C.rel(baseline_video)}"
                    )
                elif paired_status == "rendered":
                    print(
                        f"[paired] {case_id} -> {C.rel(paired_video)}",
                        flush=True,
                    )
            except Exception as exc:
                paired_status = "failed"
                paired_video = PAIR_ROOT / object_key / f"{case_id}.mp4"
                counts["paired_failed"] += 1
                failures.append(f"{case_id}:pair_render:{exc}")
                print(f"[failed] {case_id} pair render: {exc}", file=sys.stderr)

        evidence.append(
            {
                "case_id": case_id,
                "object_key": object_key,
                "variant": variant,
                "self_status": self_status,
                "self_video": C.rel(self_video),
                "baseline_experiment_id": baseline_experiment_id,
                "baseline_video": (
                    C.rel(baseline_video) if baseline_video else ""
                ),
                "paired_status": paired_status,
                "paired_video": C.rel(paired_video),
                "note": note,
            }
        )

    C.write_tsv(SELF_RENDER_DIR / "render_manifest.tsv", evidence)
    C.write_json(
        SELF_RENDER_DIR / "render_summary.json",
        {
            "created_at": C.now(),
            "selected_rows": len(rows),
            "counts": dict(counts),
            "failures": failures,
            "dry_run": args.dry_run,
        },
    )
    print(
        " ".join(f"{key}={value}" for key, value in sorted(counts.items()))
    )
    # Missing baseline videos are an expected, reportable condition (not a
    # pipeline bug) and must not abort the run; only real render exceptions
    # (self_failed / paired_failed) or an authority-join gap
    # (missing_baseline_row, which should never happen for the frozen 43-row
    # E189 set) are treated as failures.
    hard_failures = (
        counts["self_failed"]
        + counts["paired_failed"]
        + counts["paired_missing_baseline_row"]
    )
    return 1 if hard_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
