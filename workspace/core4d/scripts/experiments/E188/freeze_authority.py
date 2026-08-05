#!/usr/bin/env python3
"""Freeze the exact 15-row E188 single-variable authority."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from common import (
    E187_EVAL,
    E187_METRICS,
    E187_OVERRIDES,
    E187_QUEUE,
    REPO,
    S0,
    e187_eval_rows,
    e187_override_index,
    e187_queue_index,
    queue_case_set,
    read_tsv,
    relative,
    repo_path,
    sha256,
    verify_upstream,
    write_tsv,
)

SCENES = S0 / "scene_manifest.tsv"
OUTPUT = S0 / "authority_manifest.tsv"


def build_rows() -> list[dict[str, Any]]:
    verify_upstream()
    scenes = {row["case_id"]: row for row in read_tsv(SCENES)}
    selected = queue_case_set()
    if set(scenes) != selected or len(scenes) != 15:
        raise RuntimeError("scene manifest is not the exact E188 15-case set")
    queue = e187_queue_index()
    overrides = e187_override_index()
    eval_by_case = {row["case_id"]: row for row in e187_eval_rows()}
    rows: list[dict[str, Any]] = []
    for e187_row in e187_eval_rows():
        case_id = e187_row["case_id"]
        if case_id not in selected:
            continue
        scene = scenes[case_id]
        old_queue = queue[case_id]
        override = overrides[case_id]
        for value, expected, label in (
            (scene["scene_act"], scene["scene_sha256"], "E188 scene"),
            (e187_row["trajectory"], e187_row["result_sha256"] if False else e187_row.get("trajectory_sha256", ""), "trajectory"),
            (e187_row["contact_mask"], "", "contact"),
            (override["override_path"], override["override_sha256"], "E187 override"),
            (override["grid_manifest"], override["grid_manifest_sha256"], "grid manifest"),
        ):
            path = repo_path(value)
            if not path.is_file():
                raise FileNotFoundError(f"{case_id}: missing {label}: {path}")
            if expected and sha256(path) != expected:
                raise RuntimeError(f"{case_id}: {label} SHA changed")
        trajectory_sha = sha256(repo_path(e187_row["trajectory"]))
        contact_sha = sha256(repo_path(e187_row["contact_mask"]))
        rows.append(
            {
                "ordinal": len(rows) + 1,
                "e187_ordinal": int(e187_row["ordinal"]),
                "case_id": case_id,
                "object_key": e187_row["object_key"],
                "person": e187_row["person"],
                "retarget_variant_id": e187_row["retarget_variant_id"],
                "target_variant_id": e187_row["target_variant_id"],
                "target_task": scene["target_task"],
                "old_mass_kg": scene["old_mass_kg"],
                "new_mass_kg": scene["new_mass_kg"],
                "inertia_scale": scene["inertia_scale"],
                "source_scene": scene["source_scene"],
                "source_scene_sha256": scene["source_scene_sha256"],
                "scene_act": scene["scene_act"],
                "scene_sha256": scene["scene_sha256"],
                "trajectory": e187_row["trajectory"],
                "trajectory_sha256": trajectory_sha,
                "contact_mask": e187_row["contact_mask"],
                "contact_mask_sha256": contact_sha,
                "e187_override_id": override["override_id"],
                "e187_override_path": override["override_path"],
                "e187_override_sha256": override["override_sha256"],
                "grid_manifest": override["grid_manifest"],
                "grid_manifest_sha256": override["grid_manifest_sha256"],
                "epsilon_grid_m": override["epsilon_grid_m"],
                "e187_worker": old_queue["e187_worker"],
                "e187_predicted_wall_seconds": old_queue["predicted_wall_seconds"],
                "e187_result_npz": e187_row["result_npz"],
                "e187_config_act": e187_row["config_act"],
                "e187_video": e187_row["video"],
                "e187_row_manifest": e187_row["row_manifest"],
                "e187_result_sha256": e187_row["result_sha256"],
                "e187_config_sha256": e187_row["config_sha256"],
                "e187_video_sha256": e187_row["video_sha256"],
                "cem_samples": 1024,
                "cem_opt_steps": 32,
                "cem_seed": 0,
                "status": "AUTHORITY_FROZEN",
            }
        )
    if len(rows) != 15 or {row["case_id"] for row in rows} != selected:
        raise RuntimeError("E188 authority is not the exact 15-row treatment set")
    if sum(row["object_key"] == "bucket003" for row in rows) != 2 or any(
        row["object_key"] == "bucket004" for row in rows
    ):
        raise RuntimeError("E188 object counts differ from plan212")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("check", "freeze"), default="freeze", nargs="?")
    args = parser.parse_args()
    rows = build_rows()
    if args.mode == "freeze":
        write_tsv(OUTPUT, rows)
    payload = {
        "status": "PASS",
        "rows": len(rows),
        "unique": len({row['case_id'] for row in rows}),
        "bucket003": sum(row["object_key"] == "bucket003" for row in rows),
        "bucket004": sum(row["object_key"] == "bucket004" for row in rows),
        "bucket007": sum(row["object_key"] == "bucket007" for row in rows),
        "skipped_5_to_5": 7,
        "upstream": {
            "evaluation_manifest": {"path": relative(E187_EVAL), "sha256": sha256(E187_EVAL)},
            "queue": {"path": relative(E187_QUEUE), "sha256": sha256(E187_QUEUE)},
            "overrides": {"path": relative(E187_OVERRIDES), "sha256": sha256(E187_OVERRIDES)},
            "metrics": {"path": relative(E187_METRICS), "sha256": sha256(E187_METRICS)},
        },
        "output": relative(OUTPUT),
        "written": args.mode == "freeze",
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
