#!/usr/bin/env python3
"""Render and audit the eight compute-only E188 rows on the local machine."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO / "workspace/core4d/scripts/experiments/E168"))
from render_a100_cem_videos import render_row, repo_path  # noqa: E402

MANIFEST = REPO / "workspace/core4d/results/E188/s6_downstream/manifests/e188_full_evaluation_manifest.tsv"
OUTPUT_ROOT = REPO / "workspace/core4d/results/E188/s6_downstream/render/full/e188_videos"
VIDEO_MANIFEST = OUTPUT_ROOT / "e188_video_manifest.tsv"
SUMMARY = OUTPUT_ROOT / "summary.json"
EXPECTED = 15
EXPECTED_DEFERRED = 8


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def logical_rel(path: Path) -> str:
    if not path.is_absolute():
        return str(path)
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        text = str(path)
        if "spider_workdirs/" in text:
            return "workspace/" + text.split("spider_workdirs/", 1)[1]
        for marker in ("workspace/", "example_datasets/", "logs/"):
            if marker in text:
                return marker + text.split(marker, 1)[1]
        return text


def probe(path: Path) -> dict[str, Any]:
    payload = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name,width,height,pix_fmt,avg_frame_rate,nb_frames", "-show_entries", "format=duration", "-of", "json", str(path)], text=True))
    stream = payload["streams"][0]
    return {"codec": stream["codec_name"], "width": int(stream["width"]), "height": int(stream["height"]), "pix_fmt": stream.get("pix_fmt", ""), "frame_rate": stream["avg_frame_rate"], "frames": int(stream.get("nb_frames") or 0), "duration_s": float(payload["format"]["duration"])}


def authority() -> list[dict[str, str]]:
    rows = read_tsv(MANIFEST)
    if len(rows) != EXPECTED or len({row["case_id"] for row in rows}) != EXPECTED:
        raise ValueError("E188 render authority must contain 15 unique rows")
    if sum(row["render_mode"] == "DEFERRED_LOCAL_RENDER" for row in rows) != EXPECTED_DEFERRED:
        raise ValueError("E188 render authority must contain exactly eight deferred rows")
    return rows


def audit(rows: list[dict[str, str]], counts: Counter[str]) -> tuple[list[dict[str, Any]], list[str]]:
    evidence, failures = [], []
    for row in rows:
        video = repo_path(row["video"])
        if not video.is_file():
            failures.append(f"missing:{row['case_id']}")
            continue
        try:
            info = probe(video)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"probe:{row['case_id']}:{exc}")
            continue
        valid = info["duration_s"] > 0 and info["width"] == 1440 and info["height"] == 480 and info["frame_rate"] == "50/1"
        if not valid:
            failures.append(f"invalid:{row['case_id']}:{info}")
        evidence.append({"ordinal": row["ordinal"], "case_id": row["case_id"], "render_mode": row["render_mode"], "e188_worker": row["e188_worker"], "e188_device": row["e188_device"], "video": logical_rel(video), "video_sha256": sha256(video), **info, "audit_status": "PASS" if valid else "FAIL"})
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    if evidence:
        with VIDEO_MANIFEST.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(evidence[0]), delimiter="\t", lineterminator="\n")
            writer.writeheader(); writer.writerows(evidence)
    payload = {"schema": "e188_final_video_v1", "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"), "expected_rows": EXPECTED, "expected_deferred": EXPECTED_DEFERRED, "counts": dict(counts), "audited_rows": len(evidence), "failures": failures, "status": "pass" if len(evidence) == EXPECTED and not failures else "fail"}
    SUMMARY.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return evidence, failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("preflight", "run", "audit"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    rows = authority()
    deferred = [row for row in rows if row["render_mode"] == "DEFERRED_LOCAL_RENDER"]
    missing_inputs = []
    for row in deferred:
        for field in ("outdir_npz", "config_act", "scene_act", "trajectory"):
            if not repo_path(row[field]).is_file():
                missing_inputs.append(f"{row['case_id']}:{field}")
    if missing_inputs:
        raise RuntimeError(f"offline render inputs missing: {missing_inputs}")
    if args.mode == "preflight":
        print(json.dumps({"status": "PASS", "rows": len(rows), "deferred": len(deferred), "existing": sum(repo_path(row["video"]).is_file() for row in rows)}, sort_keys=True))
        return 0
    counts: Counter[str] = Counter()
    if args.mode == "run":
        selected = deferred[: args.limit] if args.limit > 0 else deferred
        for index, row in enumerate(selected, 1):
            output = repo_path(row["video"])
            if output.is_file() and not args.overwrite:
                counts["existing"] += 1
                print(f"[{index}/{len(selected)}] {row['case_id']} existing", flush=True)
                continue
            print(f"[{index}/{len(selected)}] {row['case_id']} render", flush=True)
            frame_count, fps = render_row(row, out_path=output, max_frames=0)
            counts["rendered"] += 1
            print(f"  frames={frame_count} fps={fps}", flush=True)
    evidence, failures = audit(rows, counts)
    payload = json.loads(SUMMARY.read_text(encoding="utf-8"))
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0 if payload["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
