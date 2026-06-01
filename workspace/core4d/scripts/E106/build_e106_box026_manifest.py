#!/usr/bin/env python3
"""Freeze E104 Box026 candidates and write E106 readiness/pipeline files."""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS))

from e106_common import (  # noqa: E402
    CANDIDATE_FIELDS,
    CANDIDATES_TSV,
    E104_CANDIDATES_TSV,
    HOLOSOMA_STAGE_ROOT,
    PERSON_IDX,
    PIPELINE_CASE_FILE,
    PREPROCESS_FAILURES_TSV,
    REPO,
    RESULTS_ROOT,
    TASK_ROOT,
    rel,
    slot_for_ordinal,
    stage_case_dir,
    first_npz,
    read_preprocess_failures,
    write_tsv,
)


READINESS_TSV = RESULTS_ROOT / "data_readiness.tsv"
SUMMARY_MD = RESULTS_ROOT / "manifest_summary.md"
SUMMARY_JSON = RESULTS_ROOT / "manifest_summary.json"

READINESS_FIELDS = CANDIDATE_FIELDS + [
    "source_scene_xml",
    "source_scene_exists_live",
    "spider_task_exists",
    "spider_traj_exists",
    "holosoma_case_dir",
    "retargeted_npz",
    "retargeted_exists",
    "trimmed_npz",
    "trimmed_exists",
    "ready_for_clean_task",
    "ready_for_cem",
    "preprocess_failed",
    "preprocess_failure_reason",
    "missing_reason",
]

PIPELINE_FIELDS = [
    "# enabled",
    "date",
    "seq",
    "person",
    "object_name",
    "object_model_rel",
    "source_scene_task",
    "target_task",
    "trim_start",
    "trim_frames",
    "data_id",
    "mask_slug",
]


def read_e104_rows() -> list[dict[str, str]]:
    with E104_CANDIDATES_TSV.open("r", encoding="utf-8", newline="") as f:
        rows = [
            row
            for row in csv.DictReader(f, delimiter="\t")
            if row.get("object_key") == "box026"
            and row.get("route") in {"candidate_legacy_risk_needs_visual", "candidate_executable"}
        ]
    rows.sort(key=lambda row: int(row["rank"]))
    if len(rows) != 30:
        raise RuntimeError(f"Expected 30 E104 Box026 3cm candidates, got {len(rows)}")
    return rows


def split_sequence(sequence: str) -> tuple[str, str]:
    date, seq = sequence.split("/", 1)
    return date, seq


def candidate_rows() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for ordinal, row in enumerate(read_e104_rows(), start=1):
        date, seq = split_sequence(row["sequence"])
        person = row["person"]
        source_task = row["target_task"]
        short = source_task.removeprefix("e091_")
        variant = f"E106B{ordinal:02d}_{short}_ref_fk_clean"
        out.append(
            {
                "ordinal": ordinal,
                "e104_rank": row["rank"],
                "e104_route": row["route"],
                "variant": variant,
                "source_task": source_task,
                "derived_task": f"{source_task}_e106_clean",
                "split": slot_for_ordinal(ordinal),
                "date": date,
                "seq": seq,
                "person": person,
                "person_idx": PERSON_IDX[person],
                "object_key": "box026",
                "object_name": row.get("object_name") or "Box026",
                "object_model_rel": "box/box026_m.obj",
                "source_scene_task": row["source_scene_task"],
                "score": row["score"],
                "target_both_active_frac_3cm": row.get("target_both_active_frac_3cm", ""),
                "target_both_active_frac_5cm": row.get("target_both_active_frac_5cm", ""),
                "raw_contact_proxy_path": row.get("raw_contact_proxy_path", ""),
            }
        )
    return out


def readiness_row(row: dict[str, Any]) -> dict[str, Any]:
    source_task = str(row["source_task"])
    source_scene = TASK_ROOT / str(row["source_scene_task"]) / "scene.xml"
    spider_dir = TASK_ROOT / source_task
    spider_traj = spider_dir / "0/trajectory_kinematic.npz"
    case_dir = stage_case_dir(source_task)
    retargeted = first_npz(case_dir, "retargeted")
    trimmed = first_npz(case_dir, "trimmed")
    failures = read_preprocess_failures()
    failure = failures.get(source_task)
    missing = []
    if not source_scene.is_file():
        missing.append("source_scene_missing")
    if not spider_dir.is_dir():
        missing.append("spider_task_missing")
    if not spider_traj.is_file():
        missing.append("spider_traj_missing")
    if retargeted is None:
        missing.append("retargeted_missing")
    if trimmed is None:
        missing.append("trimmed_missing")
    if failure:
        missing.append("preprocess_failed")
    ready_for_clean = source_scene.is_file() and spider_traj.is_file()
    ready_for_cem = ready_for_clean and retargeted is not None and trimmed is not None and not failure
    return {
        **row,
        "source_scene_xml": rel(source_scene),
        "source_scene_exists_live": str(source_scene.is_file()),
        "spider_task_exists": str(spider_dir.is_dir()),
        "spider_traj_exists": str(spider_traj.is_file()),
        "holosoma_case_dir": str(case_dir),
        "retargeted_npz": str(retargeted or ""),
        "retargeted_exists": str(retargeted is not None),
        "trimmed_npz": str(trimmed or ""),
        "trimmed_exists": str(trimmed is not None),
        "ready_for_clean_task": str(ready_for_clean),
        "ready_for_cem": str(ready_for_cem),
        "preprocess_failed": str(bool(failure)),
        "preprocess_failure_reason": failure.get("reason", "") if failure else "",
        "missing_reason": ",".join(missing),
    }


def write_pipeline_case_file(rows: list[dict[str, Any]], readiness: list[dict[str, Any]]) -> None:
    ready_by_task = {row["source_task"]: row for row in readiness}
    pipeline_rows = []
    for row in rows:
        status = ready_by_task[row["source_task"]]
        # Existing pipeline is idempotent; enable rows missing SPIDER or Holosoma outputs.
        missing_pipeline_input = (
            status["spider_traj_exists"] != "True"
            or status["retargeted_exists"] != "True"
            or status["trimmed_exists"] != "True"
        )
        preprocess_failed = status.get("preprocess_failed") == "True"
        pipeline_rows.append(
            {
                "# enabled": 1 if missing_pipeline_input and not preprocess_failed else 0,
                "date": row["date"],
                "seq": row["seq"],
                "person": row["person"],
                "object_name": "Box026",
                "object_model_rel": row["object_model_rel"],
                "source_scene_task": row["source_scene_task"],
                "target_task": row["source_task"],
                "trim_start": "auto",
                "trim_frames": "auto",
                "data_id": "0",
                "mask_slug": row["source_task"],
            }
        )
    PIPELINE_CASE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with PIPELINE_CASE_FILE.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PIPELINE_FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(pipeline_rows)


def write_summary(rows: list[dict[str, Any]], readiness: list[dict[str, Any]]) -> None:
    route_counts = Counter(row["e104_route"] for row in rows)
    split_counts = Counter(row["split"] for row in rows)
    ready_clean = sum(row["ready_for_clean_task"] == "True" for row in readiness)
    ready_cem = sum(row["ready_for_cem"] == "True" for row in readiness)
    enabled_pipeline = 0
    failures = read_preprocess_failures()
    with PIPELINE_CASE_FILE.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            enabled_pipeline += int(row["# enabled"] == "1")
    summary = {
        "candidate_count": len(rows),
        "route_counts": dict(route_counts),
        "split_counts": dict(split_counts),
        "ready_for_clean_task": ready_clean,
        "ready_for_cem": ready_cem,
        "pipeline_enabled_rows": enabled_pipeline,
        "preprocess_failures": len(failures),
        "preprocess_failures_tsv": str(PREPROCESS_FAILURES_TSV),
        "holosoma_stage_root": str(HOLOSOMA_STAGE_ROOT),
        "pipeline_case_file": str(PIPELINE_CASE_FILE),
    }
    SUMMARY_JSON.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# E106 Box026 30-candidate Manifest Summary",
        "",
        f"- candidates: `{len(rows)}`",
        f"- route counts: `{dict(route_counts)}`",
        f"- split counts: `{dict(split_counts)}`",
        f"- ready for clean task now: `{ready_clean}/30`",
        f"- ready for CEM now: `{ready_cem}/30`",
        f"- data_construction_v2 pipeline enabled rows: `{enabled_pipeline}`",
        f"- preprocess failures recorded: `{len(failures)}`",
        f"- frozen candidates: `{rel(CANDIDATES_TSV)}`",
        f"- readiness: `{rel(READINESS_TSV)}`",
        f"- pipeline case file: `{PIPELINE_CASE_FILE}`",
        f"- preprocess failures: `{rel(PREPROCESS_FAILURES_TSV)}`",
        "",
        "CEM must not start until `ready_for_cem=True` for every launched variant.",
    ]
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    rows = candidate_rows()
    readiness = [readiness_row(row) for row in rows]
    write_tsv(CANDIDATES_TSV, rows, CANDIDATE_FIELDS, "E106 frozen Box026 30-candidate manifest")
    write_tsv(READINESS_TSV, readiness, READINESS_FIELDS)
    write_pipeline_case_file(rows, readiness)
    write_summary(rows, readiness)
    print(f"wrote {rel(CANDIDATES_TSV)} rows={len(rows)}")
    print(f"wrote {rel(READINESS_TSV)}")
    print(f"wrote {PIPELINE_CASE_FILE}")
    print(f"wrote {rel(SUMMARY_MD)}")


if __name__ == "__main__":
    main()
