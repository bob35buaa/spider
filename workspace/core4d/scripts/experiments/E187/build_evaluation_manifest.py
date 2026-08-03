#!/usr/bin/env python3
"""Build the immutable 22-row E187 Full evaluation manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parents[5]
E187 = REPO / "workspace/core4d/results/E187"
KEEP22 = E187 / "s0_environment/keep22_protocol_manifest.tsv"
QUEUE = E187 / "s5_full/queue/queue_manifest.json"
ROW_ROOT = E187 / "s5_full/rows"
E178_MANIFEST = (
    REPO / "workspace/core4d/results/E178/s6_downstream/manifests/"
    "semantic_bucket_full_manifest.tsv"
)
E178_METRICS = (
    REPO / "workspace/core4d/results/E178/s6_downstream/eval/full/e178_case_metrics.tsv"
)
OUTPUT = E187 / "s6_downstream/manifests/e187_full_evaluation_manifest.tsv"
EXPECTED_ROWS = 22


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repo_path(value: str | Path) -> Path:
    path = Path(value)
    if path.exists():
        return path if path.is_absolute() else REPO / path
    text = str(value)
    for marker in ("example_datasets/", "workspace/", "logs/", "examples/"):
        if marker in text:
            return REPO / (marker + text.split(marker, 1)[1])
    return path if path.is_absolute() else REPO / path


def rel(path: Path) -> str:
    absolute = path if path.is_absolute() else REPO / path
    try:
        return str(absolute.relative_to(REPO))
    except ValueError:
        text = str(absolute)
        for marker in ("example_datasets/", "workspace/", "logs/", "examples/"):
            if marker in text:
                return marker + text.split(marker, 1)[1]
        return text


def require_file(path: Path, expected_sha: str = "") -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256(path)
    if expected_sha and actual != expected_sha:
        raise ValueError(f"SHA mismatch: {path}: {actual} != {expected_sha}")
    return actual


def unique(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        case_id = str(row["case_id"])
        if case_id in output:
            raise ValueError(f"duplicate {label} case_id: {case_id}")
        output[case_id] = row
    return output


def build_rows() -> list[dict[str, Any]]:
    keep = read_tsv(KEEP22)
    e178_manifest = unique(read_tsv(E178_MANIFEST), "E178 manifest")
    e178_metrics = unique(read_tsv(E178_METRICS), "E178 metrics")
    queue_payload = json.loads(QUEUE.read_text(encoding="utf-8"))
    queue_rows = [
        row for worker_rows in queue_payload["queues"].values() for row in worker_rows
    ]
    queue = unique(queue_rows, "E187 queue")
    keep_ids = [row["case_id"] for row in keep]
    if len(keep_ids) != EXPECTED_ROWS or len(set(keep_ids)) != EXPECTED_ROWS:
        raise ValueError("keep22 must contain 22 unique case_id values")
    for label, authority in (
        ("queue", queue),
        ("E178 manifest", e178_manifest),
        ("E178 metrics", e178_metrics),
    ):
        missing = sorted(set(keep_ids) - set(authority))
        if missing:
            raise ValueError(f"{label} missing keep22 rows: {missing}")
    if queue_payload.get("gate0_technical_status") != "FAIL":
        raise ValueError("C9/Gate0 technical status must remain FAIL")
    if queue_payload.get("progression_authority") != "USER_WAIVED":
        raise ValueError("progression authority must remain USER_WAIVED")
    queue_sha = require_file(QUEUE)

    output = []
    for keep_row in keep:
        case_id = keep_row["case_id"]
        source = e178_manifest[case_id]
        queue_row = queue[case_id]
        manifest_path = ROW_ROOT / case_id / "manifest.json"
        manifest_sha = require_file(manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "PASS" or manifest.get("case_id") != case_id:
            raise ValueError(f"row manifest is not PASS authority: {case_id}")
        if manifest.get("c9_technical_status") != "FAIL":
            raise ValueError(f"C9 technical drift: {case_id}")
        if manifest.get("c9_progression_authority") != "USER_WAIVED":
            raise ValueError(f"C9 authority drift: {case_id}")
        if manifest.get("queue_manifest_sha256") != queue_sha:
            raise ValueError(f"queue SHA drift: {case_id}")

        result_path = repo_path(manifest["result"]["path"])
        config_path = repo_path(manifest["config"]["path"])
        video_path = repo_path(manifest["video"]["path"])
        result_sha = require_file(result_path, manifest["result"]["sha256"])
        config_sha = require_file(config_path, manifest["config"]["sha256"])
        video_sha = require_file(video_path, manifest["video"]["sha256"])
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        scene_path = repo_path(config["model_path"])
        scene_sha = require_file(scene_path, queue_row["scene_sha256"])
        trajectory = repo_path(keep_row["trajectory"])
        contact_mask = repo_path(keep_row["contact_mask"])
        require_file(trajectory, keep_row["trajectory_sha256"])
        require_file(contact_mask, keep_row["contact_mask_sha256"])

        output.append(
            {
                "ordinal": manifest["ordinal"],
                "case_id": case_id,
                "variant": f"E187_{case_id}_distanceContinuation_full",
                "object_key": keep_row["object_key"],
                "person": source["person"],
                "retarget_variant_id": keep_row["retarget_variant_id"],
                "target_variant_id": keep_row["target_variant_id"],
                "hand_collision_variant_id": "object_specific_coacd_compound",
                "spider_method_id": keep_row["spider_method_id"],
                "status": "run_complete_pending_eval",
                "result_npz": rel(result_path),
                "outdir_npz": rel(result_path),
                "config_act": rel(config_path),
                "scene_act": rel(scene_path),
                "trajectory": rel(trajectory),
                "contact_mask": rel(contact_mask),
                "video": rel(video_path),
                "execution_kind": manifest["execution_kind"],
                "row_manifest": rel(manifest_path),
                "row_manifest_sha256": manifest_sha,
                "result_sha256": result_sha,
                "config_sha256": config_sha,
                "video_sha256": video_sha,
                "scene_sha256": scene_sha,
                "queue_manifest_sha256": queue_sha,
                "source_e178_variant": source["variant"],
                "source_e178_result_npz": source["result_npz"],
                "source_e178_manifest_sha256": require_file(E178_MANIFEST),
                "source_e178_metrics_sha256": require_file(E178_METRICS),
                "cem_samples": manifest["budget"]["samples"],
                "cem_opt_steps": manifest["budget"]["iterations"],
                "cem_seed": manifest["budget"]["seed"],
                "c9_technical_status": "FAIL",
                "c9_progression_authority": "USER_WAIVED",
            }
        )
    return sorted(output, key=lambda row: int(row["ordinal"]))


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(rows[0]),
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    rows = build_rows()
    if not args.check:
        write_tsv(args.output, rows)
    print(
        json.dumps(
            {
                "status": "PASS",
                "rows": len(rows),
                "promoted_canary": sum(
                    row["execution_kind"] == "PROMOTED_CANARY" for row in rows
                ),
                "full_cem": sum(row["execution_kind"] == "FULL_CEM" for row in rows),
                "output": str(args.output),
                "written": not args.check,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
