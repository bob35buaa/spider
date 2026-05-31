"""Build E102 Phase 0 taxonomy from E101 Phase 1 outcome rows."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


ALLOWED_TAGS = {
    "motion_level_H2_binding",
    "pelvis_collapse_residual",
    "lie_on_box",
    "reward_hacking_residual",
    "tilted_no_transport",
    "object_miss",
    "other",
}


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def merge_summary(matrix_rows: list[dict[str, str]], summary_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    by_variant = {r["variant"]: r for r in summary_rows}
    merged: list[dict[str, str]] = []
    for row in matrix_rows:
        out = dict(by_variant.get(row["variant"], {}))
        out.update(row)
        merged.append(out)
    return merged


def classify(row: dict[str, str]) -> tuple[str, str, str, str]:
    """Return primary_tag, secondary_tags, evidence_level, short_reason."""
    status = row.get("status", "")
    visual = row.get("visual_classification", "")
    gate_pelvis = row.get("gate_pelvis_fail", "") == "True"
    gate_tilt = row.get("gate_tilt_fail", "") == "True"
    gate_lie = row.get("gate_lie_fail", "") == "True"
    obj_err = row.get("final_obj_pos_err_m", "")

    if status == "WORK" or row.get("gate_pass", "") == "True":
        return (
            "other",
            "",
            "positive_guard",
            f"gate PASS; visual={visual or 'n/a'}",
        )

    tags: list[str] = []
    if gate_pelvis:
        tags.append("pelvis_collapse_residual")
    if gate_lie:
        tags.append("lie_on_box")
    if gate_tilt:
        tags.append("tilted_no_transport")
    if obj_err:
        try:
            if float(obj_err) > 0.20:
                tags.append("object_miss")
                tags.append("motion_level_H2_binding")
        except ValueError:
            pass
    if visual == "upperbody_lean_tilt" and "reward_hacking_residual" not in tags:
        tags.insert(0, "reward_hacking_residual")
    if visual == "pelvis_collapse_lie_on_box":
        if "pelvis_collapse_residual" not in tags:
            tags.append("pelvis_collapse_residual")
        if "lie_on_box" not in tags:
            tags.append("lie_on_box")
    if visual == "tilted_no_transport" and "tilted_no_transport" not in tags:
        tags.append("tilted_no_transport")

    tags = [t for t in dict.fromkeys(tags) if t in ALLOWED_TAGS]
    if not tags:
        tags = ["other"]

    reason_parts = [
        f"status={status}",
        f"visual={visual or 'n/a'}",
        f"pelvis_min={row.get('pelvis_min_z_m', '')}",
        f"pelvis_end={row.get('pelvis_end_z_m', '')}",
        f"tilt_end={row.get('pelvis_tilt_end_deg', '')}",
        f"lie={row.get('lie_on_box_frac', '')}",
    ]
    if obj_err:
        reason_parts.append(f"final_obj_pos_err={obj_err}")
    return tags[0], ";".join(tags[1:]), "current_negative", ", ".join(reason_parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--e101-summary", type=Path, required=True)
    parser.add_argument("--e101-matrix", type=Path, required=True)
    parser.add_argument("--phase1-dir", type=Path, default=Path("workspace/core4d/results/E101/phase1"))
    parser.add_argument("--sheet-dir", type=Path, default=Path("workspace/core4d/results/E101/visuals/phase1_sheets"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rows = merge_summary(read_tsv(args.e101_matrix), read_tsv(args.e101_summary))
    out_rows: list[dict[str, str]] = []
    for row in rows:
        primary, secondary, evidence_level, reason = classify(row)
        variant = row["variant"]
        out_rows.append(
            {
                "variant": variant,
                "task": row.get("task", ""),
                "seed": row.get("seed", ""),
                "role": row.get("role", ""),
                "e101_status": row.get("status", ""),
                "evidence_level": evidence_level,
                "primary_failure_tag": primary,
                "secondary_failure_tags": secondary,
                "visual_classification": row.get("visual_classification", ""),
                "T": row.get("T", ""),
                "pelvis_min_z_m": row.get("pelvis_min_z_m", ""),
                "pelvis_end_z_m": row.get("pelvis_end_z_m", ""),
                "pelvis_tilt_end_deg": row.get("pelvis_tilt_end_deg", ""),
                "lie_on_box_frac": row.get("lie_on_box_frac", ""),
                "gate_pelvis_fail": row.get("gate_pelvis_fail", ""),
                "gate_tilt_fail": row.get("gate_tilt_fail", ""),
                "gate_lie_fail": row.get("gate_lie_fail", ""),
                "gate_pass": row.get("gate_pass", ""),
                "final_obj_pos_err_m": row.get("final_obj_pos_err_m", ""),
                "final_obj_quat_err": row.get("final_obj_quat_err", ""),
                "mp4_path": str(args.phase1_dir / f"{variant}.mp4"),
                "npz_path": str(args.phase1_dir / f"{variant}.npz"),
                "sheet_path": str(args.sheet_dir / f"{variant}_sheet.jpg"),
                "reason": reason,
            }
        )

    fieldnames = [
        "variant",
        "task",
        "seed",
        "role",
        "e101_status",
        "evidence_level",
        "primary_failure_tag",
        "secondary_failure_tags",
        "visual_classification",
        "T",
        "pelvis_min_z_m",
        "pelvis_end_z_m",
        "pelvis_tilt_end_deg",
        "lie_on_box_frac",
        "gate_pelvis_fail",
        "gate_tilt_fail",
        "gate_lie_fail",
        "gate_pass",
        "final_obj_pos_err_m",
        "final_obj_quat_err",
        "mp4_path",
        "npz_path",
        "sheet_path",
        "reason",
    ]
    write_tsv(args.out, out_rows, fieldnames)

    fail_rows = [r for r in out_rows if r["evidence_level"] == "current_negative"]
    pos_rows = [r for r in out_rows if r["evidence_level"] == "positive_guard"]
    print(f"wrote {args.out}")
    print(f"current_negative={len(fail_rows)} positive_guard={len(pos_rows)}")


if __name__ == "__main__":
    main()
