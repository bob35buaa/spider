"""Write Box022 preflight REVIEW.md from inventory/preflight tables."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def read_tsv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def render_evidence(row: dict[str, str], out_dir: Path) -> str:
    json_path = row.get("json_path", "")
    if not json_path or not Path(json_path).is_file():
        return ""
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    img = Image.new("RGB", (1120, 520), "white")
    draw = ImageDraw.Draw(img)
    title_font = font(26)
    body_font = font(20)
    small_font = font(16)
    threshold = row.get("contact_threshold_label", "") or row.get("contact_threshold_m", "")
    min_frames = row.get("min_contact_frames", "")
    draw.text((24, 18), row["target_task"], fill="black", font=title_font)
    draw.text((24, 58), f"decision={row.get('decision')} reason={row.get('reason')}", fill=(120, 0, 0), font=body_font)
    draw.text(
        (24, 90),
        f"T={data.get('T')} threshold={threshold} min_frames={min_frames} half={data.get('half')} half_source={data.get('half_source')}",
        fill="black",
        font=body_font,
    )
    for idx, hand in enumerate(["L", "R"]):
        h = data.get(hand, {})
        base_x = 80 + idx * 520
        row_contact_frac = row.get(f"{hand}_contact_frac", "")
        draw.text(
            (base_x, 140),
            f"{hand} hand: close-contact frames = {h.get('contact_frames', 0)} ({row_contact_frac})",
            fill="black",
            font=body_font,
        )
        counts = h.get("counts_far", {})
        total = max(1, sum(int(v) for v in counts.values()))
        y = 190
        if not counts:
            draw.text((base_x, y), "no far-face counts", fill="gray", font=small_font)
        for face, value in sorted(counts.items()):
            value = int(value)
            bar_w = int(360 * value / total)
            draw.text((base_x, y), f"{face}: {value}", fill="black", font=small_font)
            draw.rectangle((base_x + 90, y, base_x + 90 + bar_w, y + 22), fill=(80, 140, 210))
            y += 40
    draw.text(
        (24, 470),
        "Visual evidence: thresholded raw fingertip face-vote contact evidence.",
        fill=(80, 80, 80),
        font=small_font,
    )
    out = out_dir / f"{row['target_task']}_contact_evidence.png"
    img.save(out)
    return out.name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--missing", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    inventory = read_tsv(args.inventory)
    preflight = read_tsv(args.preflight)
    missing = read_tsv(args.missing)
    evidence_names = {row["target_task"]: render_evidence(row, args.out_dir) for row in preflight}

    lines = [
        "# E102 Phase 1 Box022 Preflight Review",
        "",
        "Scope: recover Box022 inventory and run raw-contact/fingertip preflight when source files exist.",
        "",
        "## Inventory",
        "",
        f"- Discovered Box022 rows: {len(inventory)}",
        f"- E102 scoped rows: {sum(1 for r in inventory if r.get('e102_scope') == 'True')}",
        f"- Missing-source rows: {len(missing)}",
        "",
        "## Preflight",
        "",
        "| target_task | decision | reason | evidence | raw_person | raw_obj | source_scene | notes |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in preflight:
        lines.append(
            "| {target_task} | {decision} | {reason} | {evidence} | {raw_person} | {raw_obj} | {scene} | {notes} |".format(
                target_task=row.get("target_task", ""),
                decision=row.get("decision", ""),
                reason=row.get("reason", ""),
                evidence=(f"[png]({evidence_names.get(row.get('target_task', ''), '')})" if evidence_names.get(row.get("target_task", "")) else ""),
                raw_person=row.get("raw_person_npz", ""),
                raw_obj=row.get("raw_obj_poses", ""),
                scene=row.get("source_scene_xml", ""),
                notes=row.get("notes", "").replace("|", "/"),
            )
        )

    lines += [
        "",
        "## Decision",
        "",
        "- No Box022 row is promoted to CEM from Phase 1 unless it reaches `PREFLIGHT_PASS`.",
        "- `RAW_PREFLIGHT_PASS_NEEDS_SCENE_TEMPLATE` means raw contact exists, but CEM is still blocked until `box022_person*/scene.xml` templates exist.",
        "- `SOURCE_BLOCKED` is not treated as behavioral failure and is not a visual reject; source files must be recovered before visual review is possible.",
        "- Phase 2 may continue mining other medium-box inventory because Box022 selected rows are rejected by fingertip preflight.",
        "",
    ]
    (args.out_dir / "REVIEW.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"review -> {args.out_dir / 'REVIEW.md'}")


if __name__ == "__main__":
    main()
