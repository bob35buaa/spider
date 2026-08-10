#!/usr/bin/env python3
"""Render E192 A2 rollouts and pair them with frozen E172/E173 A0 videos."""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e192_common as C  # noqa: E402

sys.path.insert(0, str(C.REPO / "workspace/core4d/scripts/experiments/E168"))
from render_a100_cem_videos import render_row  # noqa: E402


DEFAULT_MANIFEST = C.RESULTS / "s6_downstream/manifests/cem_full_manifest.tsv"
SELF_RENDER_DIR = C.RESULTS / "s6_downstream/render/full"
PAIR_ROOT = C.RESULTS / "s6_downstream/render/paired_a0_a2"


def pair_video(baseline: Path, a2: Path, output: Path, baseline_label: str) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    filters = (
        "[0:v]scale=960:540:force_original_aspect_ratio=decrease,"
        "pad=960:540:(ow-iw)/2:(oh-ih)/2:black,setsar=1,"
        f"drawtext=text='{baseline_label} A0':x=18:y=16:fontsize=30:"
        "fontcolor=white:box=1:boxcolor=black@0.65[v0];"
        "[1:v]scale=960:540:force_original_aspect_ratio=decrease,"
        "pad=960:540:(ow-iw)/2:(oh-ih)/2:black,setsar=1,"
        "drawtext=text='E192 A2':x=18:y=16:fontsize=30:"
        "fontcolor=white:box=1:boxcolor=black@0.65[v1];"
        "[v0][v1]hstack=inputs=2[out]"
    )
    subprocess.run(
        [
            "ffmpeg", "-y", "-nostdin", "-loglevel", "error",
            "-i", str(baseline), "-i", str(a2),
            "-filter_complex", filters, "-map", "[out]", "-an",
            "-c:v", "libx264", "-crf", "22", "-preset", "medium",
            "-pix_fmt", "yuv420p", "-shortest", str(output),
        ],
        check=True,
    )


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
        raise SystemExit("no E192 rows selected")
    sources = C.load_source_rows()
    counts: Counter[str] = Counter()
    evidence: list[dict[str, str]] = []
    failures: list[str] = []

    for row in rows:
        case_id = row["case_id"]
        source = sources.get(case_id)
        a2_video = C.repo_path(row["video"])
        baseline_video = C.repo_path(source["video"]) if source else Path("")
        pair_path = PAIR_ROOT / row["object_key"] / f"{case_id}.mp4"
        self_status = pair_status = ""
        note = ""
        try:
            if args.dry_run:
                self_status = "would_render"
            elif a2_video.is_file() and args.skip_existing:
                self_status = "existing"
            else:
                render_row(row, out_path=a2_video, max_frames=args.max_frames)
                self_status = "rendered"
            counts[f"self_{self_status}"] += 1
            print(f"[{self_status}] {case_id} -> {C.rel(a2_video)}", flush=True)
        except Exception as exc:
            self_status = "failed"
            note = f"self:{type(exc).__name__}:{exc}"
            failures.append(f"{case_id}:{note}")
            counts["self_failed"] += 1

        try:
            if source is None:
                pair_status = "missing_baseline_row"
            elif not baseline_video.is_file():
                pair_status = "missing_baseline_video"
            elif self_status == "failed":
                pair_status = "missing_a2_video"
            elif args.dry_run:
                pair_status = "would_render"
            elif pair_path.is_file() and args.skip_existing:
                pair_status = "existing"
            else:
                pair_video(baseline_video, a2_video, pair_path, source["_source_exp"])
                pair_status = "rendered"
            counts[f"pair_{pair_status}"] += 1
            print(f"[{pair_status}] paired {case_id} -> {C.rel(pair_path)}", flush=True)
            if pair_status in {"missing_baseline_row", "missing_baseline_video", "missing_a2_video"}:
                failures.append(f"{case_id}:{pair_status}")
        except Exception as exc:
            pair_status = "failed"
            failures.append(f"{case_id}:pair:{type(exc).__name__}:{exc}")
            counts["pair_failed"] += 1

        evidence.append({
            "case_id": case_id,
            "object_key": row["object_key"],
            "a2_video": C.rel(a2_video),
            "a2_status": self_status,
            "baseline_video": C.rel(baseline_video) if source else "",
            "baseline_experiment": source.get("_source_exp", "") if source else "",
            "paired_video": C.rel(pair_path),
            "paired_status": pair_status,
            "note": note,
        })

    C.write_tsv(
        SELF_RENDER_DIR / "render_manifest.tsv",
        evidence,
        list(evidence[0]) if evidence else ["case_id"],
    )
    C.write_json(SELF_RENDER_DIR / "render_summary.json", {
        "created_at": C.now(), "selected_rows": len(rows),
        "counts": dict(counts), "failures": failures, "dry_run": args.dry_run,
    })
    print("summary " + " ".join(f"{key}={value}" for key, value in sorted(counts.items())))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
