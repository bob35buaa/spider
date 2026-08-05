#!/usr/bin/env python3
"""Build the exact 15-row E188 Full evaluation manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[5]
E188 = REPO / "workspace/core4d/results/E188"
AUTHORITY = E188 / "s0_environment/authority_manifest.tsv"
ROW_ROOT = E188 / "s5_full/rows"
OUTPUT = E188 / "s6_downstream/manifests/e188_full_evaluation_manifest.tsv"
VIDEO_ROOT = E188 / "s6_downstream/render/full/e188_videos"
EXPECTED = 15


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repo_path(raw: str | Path) -> Path:
    text = str(raw)
    path = Path(text)
    if path.is_file():
        return path if path.is_absolute() else REPO / path
    for marker in ("workspace/", "example_datasets/", "logs/", "examples/"):
        if marker in text:
            return REPO / (marker + text.split(marker, 1)[1])
    return path if path.is_absolute() else REPO / path


def rel(path: Path) -> str:
    absolute = path if path.is_absolute() else REPO / path
    return str(absolute.relative_to(REPO))


def require(path: Path, expected: str = "") -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256(path)
    if expected and actual != expected:
        raise ValueError(f"SHA mismatch: {path}: {actual} != {expected}")
    return actual


def build_rows() -> list[dict[str, Any]]:
    authority = read_tsv(AUTHORITY)
    if len(authority) != EXPECTED or len({row["case_id"] for row in authority}) != EXPECTED:
        raise ValueError("E188 authority must contain exactly 15 unique cases")
    output: list[dict[str, Any]] = []
    for source in authority:
        case_id = source["case_id"]
        manifest_path = ROW_ROOT / case_id / "manifest.json"
        manifest_sha = require(manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "PASS" or manifest.get("case_id") != case_id:
            raise ValueError(f"row is not PASS authority: {case_id}")
        if manifest.get("budget") != {"samples": 1024, "iterations": 32, "seed": 0}:
            raise ValueError(f"budget drift: {case_id}")
        result = repo_path(manifest["result"]["path"])
        config = repo_path(manifest["config"]["path"])
        log = repo_path(manifest["log"]["path"])
        scene = repo_path(source["scene_act"])
        trajectory = repo_path(source["trajectory"])
        contact = repo_path(source["contact_mask"])
        result_sha = require(result, manifest["result"]["sha256"])
        config_sha = require(config, manifest["config"]["sha256"])
        log_sha = require(log, manifest["log"]["sha256"])
        scene_sha = require(scene, source["scene_sha256"])
        trajectory_sha = require(trajectory, source["trajectory_sha256"])
        contact_sha = require(contact, source["contact_mask_sha256"])
        render_mode = manifest["render_mode"]
        if render_mode == "INLINE_CEM":
            video = repo_path(manifest["video"]["path"])
            video_sha = require(video, manifest["video"]["sha256"])
        elif render_mode == "DEFERRED_LOCAL_RENDER":
            if manifest.get("video") is not None:
                raise ValueError(f"deferred row unexpectedly declares video: {case_id}")
            video = VIDEO_ROOT / f"{case_id}_E188_mass5kg.mp4"
            video_sha = sha256(video) if video.is_file() else ""
        else:
            raise ValueError(f"unknown render mode: {case_id}: {render_mode}")
        e187_worker = source["e187_worker"]
        same_device = e187_worker == "local-0" and manifest["worker"] == "local-0"
        output.append(
            {
                "ordinal": source["ordinal"],
                "case_id": case_id,
                "variant": f"E188_{case_id}_mass5kg_full",
                "object_key": source["object_key"],
                "person": source["person"],
                "retarget_variant_id": source["retarget_variant_id"],
                "target_variant_id": source["target_variant_id"],
                "hand_collision_variant_id": "object_specific_coacd_compound",
                "spider_method_id": "core4d_e188_mass5kg_distance_continuation",
                "status": "run_complete_pending_eval",
                "result_npz": rel(result),
                "outdir_npz": rel(result),
                "config_act": rel(config),
                "scene_act": rel(scene),
                "trajectory": rel(trajectory),
                "contact_mask": rel(contact),
                "video": rel(video),
                "row_manifest": rel(manifest_path),
                "row_manifest_sha256": manifest_sha,
                "result_sha256": result_sha,
                "config_sha256": config_sha,
                "log_sha256": log_sha,
                "scene_sha256": scene_sha,
                "trajectory_sha256": trajectory_sha,
                "contact_mask_sha256": contact_sha,
                "video_sha256": video_sha,
                "render_mode": render_mode,
                "execution_kind": manifest["execution_kind"],
                "e188_worker": manifest["worker"],
                "e188_physical_gpu": manifest["physical_gpu"],
                "e188_device": "RTX5090" if manifest["worker"] == "local-0" else f"A100_GPU{manifest['physical_gpu']}",
                "e187_worker": e187_worker,
                "e187_device": "RTX5090" if e187_worker == "local-0" else "RTX6000_Ada",
                "device_scope": "same_device_local4" if same_device else "cross_device11",
                "old_mass_kg": source["old_mass_kg"],
                "new_mass_kg": source["new_mass_kg"],
                "inertia_scale": source["inertia_scale"],
                "cem_samples": manifest["budget"]["samples"],
                "cem_opt_steps": manifest["budget"]["iterations"],
                "cem_seed": manifest["budget"]["seed"],
            }
        )
    if sum(row["device_scope"] == "same_device_local4" for row in output) != 4:
        raise ValueError("same-device local authority must remain exactly 4 cases")
    return sorted(output, key=lambda row: int(row["ordinal"]))


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
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
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["e188_worker"]] = counts.get(row["e188_worker"], 0) + 1
    print(json.dumps({"status": "PASS", "rows": len(rows), "e188_workers": counts, "same_device_local4": sum(row["device_scope"] == "same_device_local4" for row in rows), "written": not args.check}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
