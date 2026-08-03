#!/usr/bin/env python3
"""Build and audit E178-left versus E187-right paired review videos."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[5]
E178_METRICS = (
    REPO / "workspace/core4d/results/E178/s6_downstream/eval/full/e178_case_metrics.tsv"
)
E187_METRICS = (
    REPO / "workspace/core4d/results/E187/s6_downstream/eval/full/e187_case_metrics.tsv"
)
OUTPUT_ROOT = (
    REPO / "workspace/core4d/results/E187/s6_downstream/render/full/paired_e178_vs_e187"
)
MANIFEST = OUTPUT_ROOT / "paired_video_manifest.tsv"
SUMMARY = OUTPUT_ROOT / "summary.json"
EXPECTED_ROWS = 22
WIDTH = 1920
HEIGHT = 540
FPS = 50


def read_tsv(path: Path) -> list[dict[str, str]]:
    """Read a tab-separated table."""
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def sha256(path: Path) -> str:
    """Return a streaming SHA256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repo_path(raw: str | Path) -> Path:
    """Relocate foreign artifact paths onto the current repository."""
    text = str(raw).strip()
    path = Path(text)
    if path.is_file():
        return path
    if "spider_workdirs/" in text:
        return REPO / "workspace" / text.split("spider_workdirs/", 1)[1]
    for marker in ("workspace/", "example_datasets/", "logs/"):
        if marker in text:
            return REPO / (marker + text.split(marker, 1)[1])
    return path if path.is_absolute() else REPO / path


def rel(path: Path) -> str:
    """Keep logical repository-relative paths without resolving result symlinks."""
    absolute = path if path.is_absolute() else REPO / path
    try:
        return str(absolute.relative_to(REPO))
    except ValueError:
        return str(absolute)


def unique(rows: list[dict[str, str]], label: str) -> dict[str, dict[str, str]]:
    """Index rows by unique case ID."""
    output: dict[str, dict[str, str]] = {}
    for row in rows:
        case_id = row["case_id"]
        if case_id in output:
            raise ValueError(f"duplicate {label} case_id: {case_id}")
        output[case_id] = row
    return output


def transition(old: str, new: str) -> str:
    """Return the paired numeric transition label."""
    old_pass = old.strip().lower() == "true"
    new_pass = new.strip().lower() == "true"
    old_label = "PASS" if old_pass else "FAIL"
    new_label = "PASS" if new_pass else "FAIL"
    return f"E178_{old_label}_TO_E187_{new_label}"


def authority_rows() -> list[dict[str, Any]]:
    """Resolve the exact keep22 video pairs from frozen evaluation metrics."""
    e187 = read_tsv(E187_METRICS)
    e178 = unique(read_tsv(E178_METRICS), "E178 metrics")
    case_ids = [row["case_id"] for row in e187]
    if len(case_ids) != EXPECTED_ROWS or len(set(case_ids)) != EXPECTED_ROWS:
        raise ValueError("E187 metrics must contain 22 unique cases")
    missing = sorted(set(case_ids) - set(e178))
    if missing:
        raise ValueError(f"E178 metrics missing paired cases: {missing}")
    output = []
    for ordinal, new in enumerate(e187, 1):
        case_id = new["case_id"]
        old = e178[case_id]
        old_video = repo_path(old["video"])
        new_video = repo_path(new["video"])
        if not old_video.is_file() or not new_video.is_file():
            raise FileNotFoundError(
                f"{case_id}: E178={old_video.is_file()} E187={new_video.is_file()}"
            )
        output.append(
            {
                "ordinal": ordinal,
                "case_id": case_id,
                "object_key": new["object_key"],
                "transition": transition(
                    old["numeric_release_pass"], new["numeric_release_pass"]
                ),
                "e178_numeric_release_pass": old["numeric_release_pass"],
                "e187_numeric_release_pass": new["numeric_release_pass"],
                "e178_video": old_video,
                "e187_video": new_video,
                "output": OUTPUT_ROOT / f"{case_id}_E178_vs_E187.mp4",
            }
        )
    return output


def probe(path: Path) -> dict[str, Any]:
    """Probe the first video stream and container duration."""
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,width,height,pix_fmt,avg_frame_rate",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(path),
    ]
    payload = json.loads(subprocess.check_output(command, text=True))
    stream = payload["streams"][0]
    return {
        "codec": stream["codec_name"],
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "pix_fmt": stream.get("pix_fmt", ""),
        "frame_rate": stream["avg_frame_rate"],
        "duration_s": float(payload["format"]["duration"]),
    }


def filter_graph(case_id: str, label: str) -> str:
    """Return the fixed left/right ffmpeg filter graph."""
    common = (
        "scale=960:540:force_original_aspect_ratio=decrease,"
        "pad=960:540:(ow-iw)/2:(oh-ih)/2:black,setsar=1,fps=50"
    )
    text_style = "fontsize=28:fontcolor=white:box=1:boxcolor=black@0.72"
    return (
        f"[0:v]{common},drawtext=text='LEFT - E178 BASELINE':"
        f"x=18:y=18:{text_style}[left];"
        f"[1:v]{common},drawtext=text='RIGHT - E187 CONTINUATION':"
        f"x=18:y=18:{text_style}[right];"
        "[left][right]hstack=inputs=2,"
        "drawbox=x=958:y=0:w=4:h=540:color=white@0.8:t=fill,"
        f"drawtext=text='{case_id} | {label}':"
        "x=(w-text_w)/2:y=h-48:fontsize=24:fontcolor=white:"
        "box=1:boxcolor=black@0.72[out]"
    )


def render(row: dict[str, Any], overwrite: bool) -> str:
    """Render one pair atomically, or reuse an existing output."""
    output = Path(row["output"])
    if output.is_file() and not overwrite:
        return "existing"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.mp4")
    command = [
        "ffmpeg",
        "-y",
        "-nostdin",
        "-loglevel",
        "error",
        "-i",
        str(row["e178_video"]),
        "-i",
        str(row["e187_video"]),
        "-filter_complex",
        filter_graph(row["case_id"], row["transition"]),
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
        "-movflags",
        "+faststart",
        "-shortest",
        str(temporary),
    ]
    try:
        subprocess.run(command, check=True)
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return "rendered"


def evidence_row(row: dict[str, Any], status: str) -> dict[str, Any]:
    """Build one fully probed provenance row."""
    output = Path(row["output"])
    old_probe = probe(Path(row["e178_video"]))
    new_probe = probe(Path(row["e187_video"]))
    out_probe = probe(output)
    expected_duration = min(old_probe["duration_s"], new_probe["duration_s"])
    valid = (
        out_probe["codec"] == "h264"
        and out_probe["width"] == WIDTH
        and out_probe["height"] == HEIGHT
        and out_probe["pix_fmt"] == "yuv420p"
        and out_probe["frame_rate"] == f"{FPS}/1"
        and out_probe["duration_s"] > 0
        and abs(out_probe["duration_s"] - expected_duration) <= 0.05
    )
    return {
        "ordinal": row["ordinal"],
        "case_id": row["case_id"],
        "object_key": row["object_key"],
        "transition": row["transition"],
        "e178_numeric_release_pass": row["e178_numeric_release_pass"],
        "e187_numeric_release_pass": row["e187_numeric_release_pass"],
        "e178_video": rel(Path(row["e178_video"])),
        "e178_video_sha256": sha256(Path(row["e178_video"])),
        "e187_video": rel(Path(row["e187_video"])),
        "e187_video_sha256": sha256(Path(row["e187_video"])),
        "paired_video": rel(output),
        "paired_video_sha256": sha256(output),
        "codec": out_probe["codec"],
        "width": out_probe["width"],
        "height": out_probe["height"],
        "pix_fmt": out_probe["pix_fmt"],
        "frame_rate": out_probe["frame_rate"],
        "duration_s": out_probe["duration_s"],
        "render_status": status,
        "audit_status": "PASS" if valid else "FAIL",
    }


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write uniform dictionaries as a tab-separated table."""
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(rows[0]),
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def audit(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    """Probe all expected outputs and reject missing or unexpected MP4 files."""
    expected = {Path(row["output"]) for row in rows}
    existing = set(OUTPUT_ROOT.glob("*_E178_vs_E187.mp4"))
    failures = [f"missing:{path.name}" for path in sorted(expected - existing)]
    failures.extend(f"unexpected:{path.name}" for path in sorted(existing - expected))
    evidence = []
    for row in rows:
        output = Path(row["output"])
        if not output.is_file():
            continue
        item = evidence_row(row, "existing")
        evidence.append(item)
        if item["audit_status"] != "PASS":
            failures.append(f"invalid:{row['case_id']}")
    return evidence, failures


def write_summary(
    evidence: list[dict[str, Any]],
    failures: list[str],
    counts: Counter[str],
) -> None:
    """Persist the paired-video manifest and summary."""
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    if evidence:
        write_tsv(MANIFEST, evidence)
    payload = {
        "schema": "e187_vs_e178_paired_video_v1",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "layout": "LEFT_E178_REF_SIM__RIGHT_E187_REF_SIM",
        "expected_rows": EXPECTED_ROWS,
        "counts": dict(counts),
        "audited_rows": len(evidence),
        "failures": failures,
        "status": (
            "pass"
            if len(evidence) == EXPECTED_ROWS
            and not failures
            and all(row["audit_status"] == "PASS" for row in evidence)
            else "fail"
        ),
        "c9_technical_status": "FAIL",
        "progression_authority": "USER_WAIVED",
    }
    SUMMARY.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    """Run preflight, render, or output audit."""
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("preflight", "run", "audit"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    rows = authority_rows()
    input_formats = Counter()
    for row in rows:
        for side in ("e178_video", "e187_video"):
            info = probe(Path(row[side]))
            input_formats[
                (
                    side,
                    info["codec"],
                    info["width"],
                    info["height"],
                    info["pix_fmt"],
                    info["frame_rate"],
                )
            ] += 1
    if args.mode == "preflight":
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "rows": len(rows),
                    "input_formats": {
                        str(key): value for key, value in input_formats.items()
                    },
                    "output_root": rel(OUTPUT_ROOT),
                },
                sort_keys=True,
            )
        )
        return 0
    if args.mode == "run":
        counts: Counter[str] = Counter()
        for index, row in enumerate(rows, 1):
            status = render(row, args.overwrite)
            counts[status] += 1
            print(f"[{index}/{len(rows)}] {row['case_id']} {status}", flush=True)
        evidence, failures = audit(rows)
        write_summary(evidence, failures, counts)
    else:
        evidence, failures = audit(rows)
        counts = Counter({"existing": len(evidence)})
        write_summary(evidence, failures, counts)
    payload = json.loads(SUMMARY.read_text(encoding="utf-8"))
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0 if payload["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
