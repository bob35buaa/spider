#!/usr/bin/env python3
"""Build E103 Phase 3 selected-target regeneration status table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


FIRST_BATCH = [
    {
        "target_task": "box021_person1",
        "source_scene_task": "box021_person1",
        "role": "old_guard_equivalent",
    },
    {
        "target_task": "e091_box026_20231018_039_p2",
        "source_scene_task": "box026_person2",
        "role": "box026_selected_target",
    },
    {
        "target_task": "e091_box026_20231020_135_p2",
        "source_scene_task": "box026_person2",
        "role": "box026_selected_target",
    },
    {
        "target_task": "e091_box022_20231023_125_p1",
        "source_scene_task": "box022_person1",
        "role": "box022_selected_target",
    },
    {
        "target_task": "e091_box022_20231023_125_p2",
        "source_scene_task": "box022_person2",
        "role": "box022_selected_target",
    },
    {
        "target_task": "e091_box022_20231023_126_p1",
        "source_scene_task": "box022_person1",
        "role": "box022_selected_target",
    },
    {
        "target_task": "e091_box022_20231023_126_p2",
        "source_scene_task": "box022_person2",
        "role": "box022_selected_target",
    },
]


FIELDNAMES = [
    "target_task",
    "role",
    "source_scene_task",
    "phase3_status",
    "data_gate",
    "source_scene_clean",
    "target_scene_clean",
    "verify_summary",
    "trimmed_qpos_matches_spider_qpos",
    "spider_qpos_shape",
    "scene_dims",
    "scene_act_dims",
    "replay_sheet",
    "replay_video",
    "reason",
]


def load_tsv(path: Path, key: str) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open(newline="") as f:
        return {row[key]: row for row in csv.DictReader(f, delimiter="\t")}


def bool_clean(row: dict[str, str] | None) -> str:
    if not row:
        return "missing"
    return str(row.get("action") == "keep_clean" and row.get("inertial_status") == "clean")


def dims_text(obj: dict[str, object] | None) -> str:
    if not obj:
        return ""
    return f"nq={obj.get('nq')},nv={obj.get('nv')},nu={obj.get('nu')}"


def build_rows(args: argparse.Namespace) -> list[dict[str, str]]:
    registry = load_tsv(args.registry, "task")
    box022_preflight = load_tsv(args.box022_preflight, "target_task")

    rows: list[dict[str, str]] = []
    for item in FIRST_BATCH:
        task = item["target_task"]
        source_task = item["source_scene_task"]
        verify_path = args.verify_dir / f"{task}_verify_summary.json"
        replay_sheet = args.replay_dir / f"{task}_replay_sheet.png"
        replay_video = args.replay_dir / f"{task}_kinematic_replay.mp4"
        target_registry = registry.get(task)
        source_registry = registry.get(source_task)

        row = {
            "target_task": task,
            "role": item["role"],
            "source_scene_task": source_task,
            "phase3_status": "",
            "data_gate": "",
            "source_scene_clean": bool_clean(source_registry),
            "target_scene_clean": bool_clean(target_registry),
            "verify_summary": "",
            "trimmed_qpos_matches_spider_qpos": "",
            "spider_qpos_shape": "",
            "scene_dims": "",
            "scene_act_dims": "",
            "replay_sheet": "",
            "replay_video": "",
            "reason": "",
        }

        if verify_path.exists():
            summary = json.loads(verify_path.read_text())
            row.update(
                {
                    "phase3_status": "regenerated_and_verified",
                    "data_gate": "rebuilt_target_available",
                    "verify_summary": str(verify_path),
                    "trimmed_qpos_matches_spider_qpos": str(
                        summary.get("trimmed_qpos_matches_spider_qpos")
                    ),
                    "spider_qpos_shape": str(summary.get("spider_qpos_shape")),
                    "scene_dims": dims_text(summary.get("scene")),
                    "scene_act_dims": dims_text(summary.get("scene_act")),
                    "replay_sheet": str(replay_sheet) if replay_sheet.exists() else "",
                    "replay_video": str(replay_video) if replay_video.exists() else "",
                    "reason": "clean source template used; target scene and scene_act regenerated from trimmed qpos",
                }
            )
            rows.append(row)
            continue

        if task in box022_preflight:
            pf = box022_preflight[task]
            row.update(
                {
                    "phase3_status": "skipped_by_data_gate",
                    "data_gate": "box022_raw_contact_reject",
                    "target_scene_clean": "missing_or_not_generated",
                    "reason": (
                        f"{pf.get('decision')}: {pf.get('reason')}; "
                        f"L_contact={pf.get('L_contact')}, R_contact={pf.get('R_contact')}; "
                        "no target scene/trajectory should be generated for CEM"
                    ),
                }
            )
            rows.append(row)
            continue

        if task == "box021_person1":
            row.update(
                {
                    "phase3_status": "source_template_only_no_target_reuse",
                    "data_gate": "old_runtime_artifact_quarantined",
                    "target_scene_clean": bool_clean(target_registry),
                    "reason": (
                        "canonical source template rebuilt clean, but stale source runtime artifacts were removed; "
                        "old dynamics label/trajectory is not reused as target evidence"
                    ),
                }
            )
            rows.append(row)
            continue

        row.update({"phase3_status": "unknown", "data_gate": "missing_evidence", "reason": "no status evidence found"})
        rows.append(row)

    return rows


def write_markdown(rows: list[dict[str, str]], path: Path) -> None:
    lines = [
        "# E103 Phase 3 Target Regeneration Status",
        "",
        "| target | status | data_gate | source_clean | target_clean | reason |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        reason = row["reason"].replace("|", "/")
        lines.append(
            f"| `{row['target_task']}` | `{row['phase3_status']}` | `{row['data_gate']}` | "
            f"{row['source_scene_clean']} | {row['target_scene_clean']} | {reason} |"
        )
    lines.append("")
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, default=Path("workspace/core4d/results/E103/affected_scene_registry.tsv"))
    parser.add_argument("--box022-preflight", type=Path, default=Path("workspace/core4d/results/E103/rebuilt_box022_preflight.tsv"))
    parser.add_argument("--verify-dir", type=Path, default=Path("workspace/core4d/results/E103/verify"))
    parser.add_argument("--replay-dir", type=Path, default=Path("workspace/core4d/results/E103/visuals/rebuilt_target_replay"))
    parser.add_argument("--out", type=Path, default=Path("workspace/core4d/results/E103/rebuilt_target_regeneration_status.tsv"))
    parser.add_argument("--summary", type=Path, default=Path("workspace/core4d/results/E103/rebuilt_target_regeneration_status.md"))
    args = parser.parse_args()

    rows = build_rows(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    write_markdown(rows, args.summary)
    print(f"wrote {args.out} rows={len(rows)}")
    print(f"summary -> {args.summary}")


if __name__ == "__main__":
    main()
