#!/usr/bin/env python3
"""Render E195 A3 rollouts and pair them with the same-case E192 A2 videos."""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e195_common as C  # noqa: E402

sys.path.insert(0, str(C.REPO / "workspace/core4d/scripts/experiments/E168"))
from render_a100_cem_videos import render_row  # noqa: E402


DEFAULT_MANIFEST = C.RESULTS / "s6_downstream/manifests/cem_full_manifest.tsv"
SELF_RENDER_DIR = C.RESULTS / "s6_downstream/render/full"
PAIR_ROOT = C.RESULTS / "s6_downstream/render/paired_e192_e195"


def pair_video(baseline: Path, current: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    filters = (
        "[0:v]scale=960:540:force_original_aspect_ratio=decrease,"
        "pad=960:540:(ow-iw)/2:(oh-ih)/2:black,setsar=1,"
        "drawtext=text='E192 A2':x=18:y=16:fontsize=30:"
        "fontcolor=white:box=1:boxcolor=black@0.65[v0];"
        "[1:v]scale=960:540:force_original_aspect_ratio=decrease,"
        "pad=960:540:(ow-iw)/2:(oh-ih)/2:black,setsar=1,"
        "drawtext=text='E195 A3':x=18:y=16:fontsize=30:"
        "fontcolor=white:box=1:boxcolor=black@0.65[v1];"
        "[v0][v1]hstack=inputs=2[out]"
    )
    subprocess.run([
        "ffmpeg", "-y", "-nostdin", "-loglevel", "error",
        "-i", str(baseline), "-i", str(current),
        "-filter_complex", filters, "-map", "[out]", "-an",
        "-c:v", "libx264", "-crf", "22", "-preset", "medium",
        "-pix_fmt", "yuv420p", "-shortest", str(output),
    ], check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--case-id", nargs="*", default=[])
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    rows = C.read_tsv(C.repo_path(args.manifest))
    if args.case_id:
        selected = set(args.case_id)
        rows = [row for row in rows if row["case_id"] in selected]
    if not rows:
        raise SystemExit("no E195 rows selected")
    baseline = C.load_e192_rows()
    counts: Counter[str] = Counter()
    evidence: list[dict[str, str]] = []
    failures: list[str] = []
    for row in rows:
        case_id = row["case_id"]
        current_video = C.repo_path(row["video"])
        baseline_video = C.repo_path(baseline[case_id]["video"])
        pair_path = PAIR_ROOT / row["object_key"] / f"{case_id}.mp4"
        note = ""
        try:
            if args.dry_run:
                current_status = "would_render"
            elif current_video.is_file() and args.skip_existing:
                current_status = "existing"
            else:
                render_row(row, out_path=current_video, max_frames=args.max_frames)
                current_status = "rendered"
            counts[f"self_{current_status}"] += 1
            print(f"[{current_status}] {case_id} -> {C.rel(current_video)}", flush=True)
        except Exception as exc:
            current_status = "failed"
            note = f"self:{type(exc).__name__}:{exc}"
            failures.append(f"{case_id}:{note}")
            counts["self_failed"] += 1
        try:
            if not baseline_video.is_file():
                paired_status = "missing_e192_video"
            elif current_status == "failed":
                paired_status = "missing_e195_video"
            elif args.dry_run:
                paired_status = "would_render"
            elif pair_path.is_file() and args.skip_existing:
                paired_status = "existing"
            else:
                pair_video(baseline_video, current_video, pair_path)
                paired_status = "rendered"
            counts[f"pair_{paired_status}"] += 1
            if paired_status.startswith("missing"):
                failures.append(f"{case_id}:{paired_status}")
            print(f"[{paired_status}] paired {case_id} -> {C.rel(pair_path)}", flush=True)
        except Exception as exc:
            paired_status = "failed"
            failures.append(f"{case_id}:pair:{type(exc).__name__}:{exc}")
            counts["pair_failed"] += 1
        evidence.append({
            "case_id": case_id, "object_key": row["object_key"],
            "e195_video": C.rel(current_video), "e195_status": current_status,
            "e192_video": C.rel(baseline_video), "paired_video": C.rel(pair_path),
            "paired_status": paired_status, "note": note,
        })
    C.write_tsv(SELF_RENDER_DIR / "render_manifest.tsv", evidence, list(evidence[0]))
    C.write_json(SELF_RENDER_DIR / "render_summary.json", {
        "created_at": C.now(), "selected_rows": len(rows),
        "counts": dict(counts), "failures": failures, "dry_run": args.dry_run,
    })
    print("summary " + " ".join(f"{key}={value}" for key, value in sorted(counts.items())))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

