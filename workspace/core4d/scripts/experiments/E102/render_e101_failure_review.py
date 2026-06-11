"""Render E101 Phase 1 key-frame overlays and REVIEW.md for E102 Phase 0."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
from PIL import Image, ImageDraw, ImageFont


FRAME_RATIOS = [("early", 0.12), ("contact", 0.50), ("end", 0.88)]


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def wrap(text: str, max_chars: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    cur: list[str] = []
    for word in words:
        if sum(len(w) + 1 for w in cur) + len(word) > max_chars and cur:
            lines.append(" ".join(cur))
            cur = [word]
        else:
            cur.append(word)
    if cur:
        lines.append(" ".join(cur))
    return lines


def draw_overlay(frame_bgr, title: str, lines: list[str]) -> Image.Image:
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    draw = ImageDraw.Draw(img, "RGBA")
    title_font = font(24)
    body_font = font(18)
    panel_h = 34 + 24 * min(7, len(lines))
    draw.rectangle([(0, 0), (img.width, panel_h)], fill=(0, 0, 0, 170))
    draw.text((14, 8), title, fill=(255, 255, 255, 255), font=title_font)
    y = 40
    for line in lines[:7]:
        draw.text((14, y), line, fill=(230, 230, 230, 255), font=body_font)
        y += 24
    return img


def extract_overlays(video: Path, row: dict[str, str], out_dir: Path) -> list[Path]:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video}")
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n_frames <= 0:
        raise RuntimeError(f"video has no frames: {video}")

    variant = row["variant"]
    tags = row["primary_failure_tag"]
    if row.get("secondary_failure_tags"):
        tags = f"{tags};{row['secondary_failure_tags']}"
    metric_line = (
        f"gate={row.get('evidence_level')} status={row.get('e101_status')} "
        f"pelvis_min={row.get('pelvis_min_z_m')} pelvis_end={row.get('pelvis_end_z_m')} "
        f"tilt_end={row.get('pelvis_tilt_end_deg')} lie={row.get('lie_on_box_frac')}"
    )
    reason_lines = wrap(row.get("reason", ""), 105)
    lines = [f"task={row.get('task', '')}", f"tags={tags}", metric_line, *reason_lines]

    paths: list[Path] = []
    for label, ratio in FRAME_RATIOS:
        idx = max(0, min(n_frames - 1, int(round((n_frames - 1) * ratio))))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        title = f"{variant} | {label} | frame {idx}/{n_frames - 1}"
        img = draw_overlay(frame, title, lines)
        out_path = out_dir / f"{variant}_{label}.jpg"
        img.save(out_path, quality=92)
        paths.append(out_path)
    cap.release()
    return paths


def write_review(rows: list[dict[str, str]], overlay_map: dict[str, list[Path]], out_dir: Path) -> None:
    review = out_dir / "REVIEW.md"
    lines = [
        "# E102 Phase 0 E101 Failure Review",
        "",
        "Scope: E101 Phase 1 rollouts after E098 replay gate, E099 fingertip audit, and E100 fingertip-aware target generation.",
        "",
        "Evidence levels:",
        "- `current_negative`: post-fix E101 failure; hard negative for E102 mining.",
        "- `positive_guard`: post-fix E101 WORK guard; not a negative.",
        "",
        "## Summary",
        "",
        "| variant | evidence | tags | metrics | visual | artifacts |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        tags = row["primary_failure_tag"]
        if row.get("secondary_failure_tags"):
            tags += ";" + row["secondary_failure_tags"]
        metrics = (
            f"pelvis_min {row.get('pelvis_min_z_m')}, end {row.get('pelvis_end_z_m')}, "
            f"tilt {row.get('pelvis_tilt_end_deg')}, lie {row.get('lie_on_box_frac')}"
        )
        artifacts = []
        for p in overlay_map.get(row["variant"], []):
            artifacts.append(f"[{p.stem}]({p.name})")
        if row.get("sheet_path"):
            artifacts.append(f"[sheet]({Path(row['sheet_path']).resolve()})")
        if row.get("mp4_path"):
            artifacts.append(f"[mp4]({Path(row['mp4_path']).resolve()})")
        lines.append(
            "| {variant} | {evidence} | {tags} | {metrics} | {visual} | {artifacts} |".format(
                variant=row["variant"],
                evidence=row["evidence_level"],
                tags=tags,
                metrics=metrics,
                visual=row.get("visual_classification", ""),
                artifacts="<br>".join(artifacts),
            )
        )
    lines += [
        "",
        "## Decision",
        "",
        "- The two `box004_083_p2` guard rollouts remain `positive_guard`.",
        "- `box021_030_p1` seed0/1 are `current_negative` with `tilted_no_transport + object_miss + motion_level_H2_binding`.",
        "- `box021_11035/035_p2` is `current_negative` with `pelvis_collapse_residual + lie_on_box`.",
        "- `box021_18029_p2` is `current_negative` with `reward_hacking_residual + tilted_no_transport`.",
        "- Historical failures before E098-E100 are not upgraded here; they stay `legacy_failure_prior` in the registry.",
        "",
    ]
    review.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase1-dir", type=Path, required=True)
    parser.add_argument("--taxonomy", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = read_tsv(args.taxonomy)
    overlay_map: dict[str, list[Path]] = {}
    for row in rows:
        video = args.phase1_dir / f"{row['variant']}.mp4"
        if not video.is_file():
            print(f"[WARN] missing video for {row['variant']}: {video}")
            overlay_map[row["variant"]] = []
            continue
        overlay_map[row["variant"]] = extract_overlays(video, row, args.out_dir)
        print(f"[OK] {row['variant']} overlays={len(overlay_map[row['variant']])}")
    write_review(rows, overlay_map, args.out_dir)
    print(f"review -> {args.out_dir / 'REVIEW.md'}")


if __name__ == "__main__":
    main()
