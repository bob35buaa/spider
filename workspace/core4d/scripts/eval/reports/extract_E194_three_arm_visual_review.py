#!/usr/bin/env python3
"""Freeze mandatory E194 visual cases and extract 3-arm four-phase contact sheets."""

from __future__ import annotations

import statistics
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments/E194"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runners"))
import e194_g1_expansion_common as C  # noqa: E402
import eval_E194_three_arm_comparison as three  # noqa: E402

EVAL = C.RESULTS / "s6_downstream/eval/full_g1_expansion"
REVIEW = EVAL / "visual_review"
FRAMES = REVIEW / "frames"
SHEETS = REVIEW / "case_sheets"
ATLASES = REVIEW / "atlases"
PHASES = (("grasp", 0.22), ("lift", 0.42), ("carry", 0.65), ("place", 0.88))


def ffprobe_duration(path: Path) -> float:
    result = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
                            check=True, capture_output=True, text=True)
    return float(result.stdout.strip())


def extract(path: Path, timestamp: float, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.is_file() and output.stat().st_size > 0:
        return
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", f"{timestamp:.3f}", "-i", str(path),
                    "-frames:v", "1", "-vf", "scale=640:-2", "-q:v", "2", str(output)], check=True)


def video_maps() -> dict[str, dict[str, str]]:
    noprg, _ = three.no_prg_authority()
    prg = C.source_rows()
    g1 = C.read_tsv(C.FULL_MANIFEST)
    maps = {arm: {row["case_id"]: row["video"] for row in rows}
            for arm, rows in (("noPRG", noprg), ("PRG", prg), ("G1", g1))}
    for arm, mapping in maps.items():
        if len(mapping) != C.N_CASES:
            raise ValueError(f"{arm} video map rows={len(mapping)}")
        for case_id, value in mapping.items():
            path = C.repo_path(value)
            if not path.is_file() or path.stat().st_size == 0:
                raise FileNotFoundError(f"{arm}:{case_id}:{path}")
    return maps


def selected_cases() -> list[dict[str, Any]]:
    paired = [row for row in C.read_tsv(EVAL / "e194_three_arm_paired_deltas.tsv")
              if row["comparison"] == "PRG_to_G1"]
    old = {row["case_id"]: row for row in C.read_tsv(EVAL / "e194_g1_expansion_paired_deltas.tsv")}
    selected: dict[str, dict[str, Any]] = {}

    def add(row: dict[str, str], reason: str) -> None:
        selected.setdefault(row["case_id"], {"case_id": row["case_id"], "object_key": row["object_key"], "reasons": []})["reasons"].append(reason)

    for row in paired:
        flips = [gate for gate in three.base.ALL_GATES
                 if C.truth(row[f"before_{gate}_gate_pass"]) and not C.truth(row[f"after_{gate}_gate_pass"])]
        if flips:
            add(row, "gate_PASS_TO_FAIL:" + ",".join(flips))
        if C.truth(row["before_fall_gate_pass"]) and not C.truth(row["after_fall_gate_pass"]):
            add(row, "new_fall")
        if float(row["delta_track_obj_z_abs_err_cm_mean"]) > 1.0:
            add(row, "delta_z_gt_1cm")
        if float(row["delta_track_obj_pos_err_cm_mean"]) > 2.0:
            add(row, "delta_3d_gt_2cm")
    for object_key in C.OBJECT_ORDER:
        group = [row for row in paired if row["object_key"] == object_key]
        add(max(group, key=lambda row: float(row["before_track_obj_z_abs_err_cm_mean"])), "PRG_z_max")
        median = statistics.median(float(row["before_track_obj_z_abs_err_cm_mean"]) for row in group)
        add(min(group, key=lambda row: abs(float(row["before_track_obj_z_abs_err_cm_mean"]) - median)), "PRG_z_nearest_median")
    profile_counts = Counter(old[case_id]["execution_profile"] for case_id in selected)
    for profile in C.WORKERS:
        if not profile_counts[profile]:
            add(next(row for row in paired if old[row["case_id"]]["execution_profile"] == profile),
                f"execution_profile_coverage:{profile}")
    rank = {key: index for index, key in enumerate(C.OBJECT_ORDER)}
    output = []
    paired_map = {row["case_id"]: row for row in paired}
    for case_id, item in selected.items():
        row = paired_map[case_id]
        output.append({"case_id": case_id, "object_key": item["object_key"],
                       "execution_profile": old[case_id]["execution_profile"],
                       "selection_reason": ";".join(item["reasons"]),
                       "delta_z_cm": row["delta_track_obj_z_abs_err_cm_mean"],
                       "delta_3d_cm": row["delta_track_obj_pos_err_cm_mean"]})
    output.sort(key=lambda row: (rank[row["object_key"]], row["case_id"]))
    if len(output) != 36:
        raise ValueError(f"mandatory selection changed: {len(output)} != 36")
    return output


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return ImageFont.truetype(str(path), size=size) if path.is_file() else ImageFont.load_default()


def make_case_sheet(case: dict[str, Any]) -> Path:
    thumb_w, thumb_h, label_w, title_h = 480, 270, 90, 54
    canvas = Image.new("RGB", (label_w + thumb_w * 4, title_h + thumb_h * 3), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 8), f"{case['case_id']} | {case['execution_profile']} | dz={float(case['delta_z_cm']):+.3f} cm | d3D={float(case['delta_3d_cm']):+.3f} cm",
              fill="black", font=font(18))
    for col, (phase, _) in enumerate(PHASES):
        draw.text((label_w + col * thumb_w + 8, title_h - 24), phase, fill="#17365D", font=font(16))
    for row, arm in enumerate(("noPRG", "PRG", "G1")):
        draw.text((8, title_h + row * thumb_h + thumb_h // 2 - 10), arm, fill="#17365D", font=font(16))
        for col, (phase, _) in enumerate(PHASES):
            image = Image.open(FRAMES / case["case_id"] / f"{arm}_{phase}.jpg").convert("RGB")
            image.thumbnail((thumb_w, thumb_h))
            left = label_w + col * thumb_w + (thumb_w - image.width) // 2
            top = title_h + row * thumb_h + (thumb_h - image.height) // 2
            canvas.paste(image, (left, top))
    output = SHEETS / f"{case['case_id']}.jpg"
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=92)
    return output


def make_atlases(cases: list[dict[str, Any]], sheets: list[Path]) -> None:
    ATLASES.mkdir(parents=True, exist_ok=True)
    for page_index in range(0, len(sheets), 4):
        subset = sheets[page_index:page_index + 4]
        images = [Image.open(path).convert("RGB") for path in subset]
        width = max(image.width for image in images)
        height = sum(image.height for image in images)
        canvas = Image.new("RGB", (width, height), "white")
        top = 0
        for image in images:
            canvas.paste(image, (0, top)); top += image.height
        page = page_index // 4 + 1
        canvas.save(ATLASES / f"mandatory_review_{page:02d}.jpg", quality=90)


def main() -> int:
    cases = selected_cases()
    maps = video_maps()
    for case_index, case in enumerate(cases, 1):
        for arm in ("noPRG", "PRG", "G1"):
            video = C.repo_path(maps[arm][case["case_id"]])
            duration = ffprobe_duration(video)
            case[f"{arm}_video"] = C.rel(video)
            case[f"{arm}_duration_s"] = duration
            for phase, fraction in PHASES:
                output = FRAMES / case["case_id"] / f"{arm}_{phase}.jpg"
                extract(video, max(0.0, min(duration - 0.05, duration * fraction)), output)
        print(f"[frames {case_index:02d}/36] {case['case_id']}")
    C.write_tsv(EVAL / "e194_three_arm_visual_selection.tsv", cases)
    sheets = [make_case_sheet(case) for case in cases]
    make_atlases(cases, sheets)
    print(f"wrote {len(cases)} case sheets and {(len(cases) + 3) // 4} atlases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
