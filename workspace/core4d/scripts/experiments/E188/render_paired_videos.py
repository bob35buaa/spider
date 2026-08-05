#!/usr/bin/env python3
"""Build and audit E187-left versus E188-right paired review videos."""

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
E187_METRICS = REPO / "workspace/core4d/results/E187/s6_downstream/eval/full/e187_case_metrics.tsv"
E188_METRICS = REPO / "workspace/core4d/results/E188/s6_downstream/eval/full/e188_case_metrics.tsv"
PAIRED = REPO / "workspace/core4d/results/E188/s6_downstream/eval/full/e188_vs_e187_paired_deltas.tsv"
OUTPUT_ROOT = REPO / "workspace/core4d/results/E188/s6_downstream/render/full/paired_e187_vs_e188"
MANIFEST = OUTPUT_ROOT / "paired_video_manifest.tsv"
SUMMARY = OUTPUT_ROOT / "summary.json"
EXPECTED = 15


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def unique(rows: list[dict[str, str]], label: str) -> dict[str, dict[str, str]]:
    result = {row["case_id"]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"duplicate {label} case_id")
    return result


def repo_path(raw: str | Path) -> Path:
    text = str(raw); path = Path(text)
    if path.is_file(): return path if path.is_absolute() else REPO / path
    for marker in ("workspace/", "example_datasets/", "logs/"):
        if marker in text: return REPO / (marker + text.split(marker, 1)[1])
    return path if path.is_absolute() else REPO / path


def rel(path: Path) -> str:
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""): digest.update(chunk)
    return digest.hexdigest()


def probe(path: Path) -> dict[str, Any]:
    payload = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name,width,height,pix_fmt,avg_frame_rate", "-show_entries", "format=duration", "-of", "json", str(path)], text=True))
    stream = payload["streams"][0]
    return {"codec": stream["codec_name"], "width": int(stream["width"]), "height": int(stream["height"]), "pix_fmt": stream.get("pix_fmt", ""), "frame_rate": stream["avg_frame_rate"], "duration_s": float(payload["format"]["duration"])}


def authority() -> list[dict[str, Any]]:
    new_rows = read_tsv(E188_METRICS)
    old = unique(read_tsv(E187_METRICS), "E187")
    deltas = unique(read_tsv(PAIRED), "paired")
    if len(new_rows) != EXPECTED or len({row["case_id"] for row in new_rows}) != EXPECTED:
        raise ValueError("E188 metrics must contain 15 unique rows")
    rows = []
    for ordinal, new in enumerate(new_rows, 1):
        case = new["case_id"]; before = old[case]; delta = deltas[case]
        left, right = repo_path(before["video"]), repo_path(new["video"])
        rows.append({"ordinal": ordinal, "case_id": case, "object_key": new["object_key"], "transition": delta["gate_transition"], "device_scope": delta["device_scope"], "e188_worker": delta["e188_worker"], "e188_device": delta["e188_device"], "contact_improvement": float(delta["improvement_hand_object_physics_contact_in_mask_frac"]), "hand_pen_improvement": float(delta["improvement_hand_object_physics_penetration_3mm_frame_frac"]), "leg_improvement": float(delta["improvement_leg_penetration_frac"]), "e187_video": left, "e188_video": right, "output": OUTPUT_ROOT / f"{case}_E187_vs_E188.mp4"})
    return rows


def escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("'", "\\'").replace(":", "\\:")


def filter_graph(row: dict[str, Any]) -> str:
    common = "scale=960:540:force_original_aspect_ratio=decrease,pad=960:540:(ow-iw)/2:(oh-ih)/2:black,setsar=1,fps=50"
    style = "fontsize=27:fontcolor=white:box=1:boxcolor=black@0.72"
    detail = escape(f"{row['case_id']} | 2kg -> 5kg | {row['device_scope']} | {row['e188_worker']} | contact {row['contact_improvement']:+.3f} | handPen {row['hand_pen_improvement']:+.3f} | leg {row['leg_improvement']:+.3f}")
    return f"[0:v]{common},drawtext=text='LEFT - E187 2KG':x=18:y=18:{style}[left];[1:v]{common},drawtext=text='RIGHT - E188 5KG':x=18:y=18:{style}[right];[left][right]hstack=inputs=2,drawbox=x=958:y=0:w=4:h=540:color=white@0.8:t=fill,drawtext=text='{detail}':x=(w-text_w)/2:y=h-46:fontsize=22:fontcolor=white:box=1:boxcolor=black@0.72[out]"


def render(row: dict[str, Any], overwrite: bool) -> str:
    output = Path(row["output"])
    if output.is_file() and not overwrite: return "existing"
    output.parent.mkdir(parents=True, exist_ok=True); temporary = output.with_suffix(".tmp.mp4")
    command = ["ffmpeg", "-y", "-nostdin", "-loglevel", "error", "-i", str(row["e187_video"]), "-i", str(row["e188_video"]), "-filter_complex", filter_graph(row), "-map", "[out]", "-an", "-c:v", "libx264", "-crf", "22", "-preset", "medium", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-shortest", str(temporary)]
    try:
        subprocess.run(command, check=True); os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return "rendered"


def evidence(row: dict[str, Any], status: str) -> dict[str, Any]:
    left, right, output = Path(row["e187_video"]), Path(row["e188_video"]), Path(row["output"])
    lp, rp, op = probe(left), probe(right), probe(output)
    expected_duration = min(lp["duration_s"], rp["duration_s"])
    valid = op["codec"] == "h264" and op["width"] == 1920 and op["height"] == 540 and op["pix_fmt"] == "yuv420p" and op["frame_rate"] == "50/1" and op["duration_s"] > 0 and abs(op["duration_s"] - expected_duration) <= 0.05
    return {key: row[key] for key in ("ordinal", "case_id", "object_key", "transition", "device_scope", "e188_worker", "e188_device", "contact_improvement", "hand_pen_improvement", "leg_improvement")} | {"e187_video": rel(left), "e187_video_sha256": sha256(left), "e188_video": rel(right), "e188_video_sha256": sha256(right), "paired_video": rel(output), "paired_video_sha256": sha256(output), **op, "render_status": status, "audit_status": "PASS" if valid else "FAIL"}


def audit(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    failures, items = [], []
    expected = {Path(row["output"]) for row in rows}; existing = set(OUTPUT_ROOT.glob("*_E187_vs_E188.mp4"))
    failures.extend(f"missing:{path.name}" for path in sorted(expected - existing)); failures.extend(f"unexpected:{path.name}" for path in sorted(existing - expected))
    for row in rows:
        if not Path(row["output"]).is_file(): continue
        item = evidence(row, "existing"); items.append(item)
        if item["audit_status"] != "PASS": failures.append(f"invalid:{row['case_id']}")
    return items, failures


def write_summary(items: list[dict[str, Any]], failures: list[str], counts: Counter[str]) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    if items:
        with MANIFEST.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(items[0]), delimiter="\t", lineterminator="\n"); writer.writeheader(); writer.writerows(items)
    payload = {"schema": "e188_vs_e187_paired_video_v1", "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"), "layout": "LEFT_E187_2KG__RIGHT_E188_5KG", "expected_rows": EXPECTED, "counts": dict(counts), "audited_rows": len(items), "failures": failures, "status": "pass" if len(items) == EXPECTED and not failures else "fail"}
    SUMMARY.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("mode", choices=("preflight", "run", "audit")); parser.add_argument("--overwrite", action="store_true"); args = parser.parse_args()
    rows = authority(); missing = [f"{row['case_id']}:{side}" for row in rows for side in ("e187_video", "e188_video") if not Path(row[side]).is_file()]
    if args.mode == "preflight":
        print(json.dumps({"status": "PASS" if not missing else "BLOCKED", "rows": len(rows), "missing": missing}, sort_keys=True)); return 0 if not missing else 2
    counts: Counter[str] = Counter()
    if args.mode == "run":
        if missing: raise FileNotFoundError(missing)
        for index, row in enumerate(rows, 1):
            status = render(row, args.overwrite); counts[status] += 1; print(f"[{index}/{EXPECTED}] {row['case_id']} {status}", flush=True)
    items, failures = audit(rows); write_summary(items, failures, counts if counts else Counter(existing=len(items)))
    payload = json.loads(SUMMARY.read_text(encoding="utf-8")); print(json.dumps(payload, ensure_ascii=False, sort_keys=True)); return 0 if payload["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
