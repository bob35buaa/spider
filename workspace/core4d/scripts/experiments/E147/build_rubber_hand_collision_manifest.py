#!/usr/bin/env python3
"""Build E147 rubber-hand collision CEM manifest, scenes, and overrides."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parents[4]
DCV3 = REPO / "workspace/core4d/scripts/data_construction_v3"
for _path in (DCV3 / "lib", DCV3 / "state", DCV3 / "stages/s5_handoff"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from common import read_tsv, sha256_file, timestamp, write_json, write_tsv  # noqa: E402
from patch_hand_collision import patch_scene  # noqa: E402


TASK_ROOT = REPO / "example_datasets/processed/core4d/unitree_g1/humanoid_object"
RESULT_ROOT = REPO / "workspace/core4d/results/E147/rubber_hand_collision"
SCRIPT_ROOT = REPO / "workspace/core4d/scripts/E147"
OVERRIDE_ROOT = REPO / "examples/config/override"
SCENE_NAME = "scene_act_E147_rubber_hull"
EXISTING_CASES = REPO / "workspace/core4d/data_construction_v3/existing_cases.tsv"

CASE_SPECS = [
    ("e091_box004_20231003_2_083_p1", "E096bP1_box004_083_p1_mask_cem", "remote-gpu0", "pass"),
    ("e091_box004_20231003_2_082_p1", "E096bP2_box004_082_p1_mask_cem", "remote-gpu1", "pass"),
    ("d003_box021_20231011_035_p1", "E107C02_box021_20231011_035_p1_ref_fk_clean", "remote-gpu0", "pass"),
    ("d003_box021_20231011_035_p2", "E107C03_box021_20231011_035_p2_ref_fk_clean", "remote-gpu1", "fail"),
    ("box023_person2", "E081_box023_p2_legobj", "remote-gpu0", "pass"),
    ("box023_person1", "E079_box023_p1", "remote-gpu1", "fail"),
    ("e091_box026_20231020_134_p1", "E106B05_box026_20231020_134_p1_ref_fk_clean", "remote-gpu0", "pass"),
    ("e091_box026_20231020_134_p2", "E106B06_box026_20231020_134_p2_ref_fk_clean", "remote-gpu1", "fail"),
    ("bucket004_20231003_1_012_p1", "E108B01_bucket004_20231003_1_012_p1_ref_fk_nonbox", "remote-gpu0", "pass"),
    ("bucket004_20231002_021_p1", "E108B03_bucket004_20231002_021_p1_ref_fk_nonbox", "remote-gpu1", "fail"),
]

FIELDS = [
    "ordinal",
    "variant",
    "case_id",
    "object_key",
    "object_category",
    "person",
    "person_idx",
    "split",
    "ablation",
    "retarget_variant_id",
    "target_variant_id",
    "hand_collision_variant_id",
    "derived_task",
    "target_scene",
    "trajectory",
    "base_scene_act",
    "rubber_scene_act",
    "scene_name",
    "sphere_run_id",
    "sphere_npz",
    "sphere_video",
    "sphere_outdir_npz",
    "sphere_config_act",
    "override",
    "object_asset",
    "remote_sync_key",
    "historical_cem_status",
    "historical_rl_status",
    "expected_quality",
    "run_status",
]


def rel(path: Path | str) -> str:
    if not path:
        return ""
    p = Path(path)
    try:
        return str(p.resolve().relative_to(REPO.resolve()))
    except Exception:
        return str(path)


def repo_path(text: str) -> Path:
    p = Path(text)
    return p if p.is_absolute() else REPO / p


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:
        return "unknown"


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def case_object_category(object_key: str) -> str:
    if object_key.startswith("box"):
        return "box"
    if object_key.startswith("bucket"):
        return "bucket"
    return object_key.rstrip("0123456789") or object_key


def person_from_case_id(case_id: str, fallback: str) -> str:
    if "_person" in case_id:
        return case_id.rsplit("_", 1)[-1]
    tail = case_id.rsplit("_", 1)[-1]
    if tail in {"p1", "p2"}:
        return f"person{tail[1:]}"
    return fallback


def config_act_for(run_id: str, npz_path: str) -> Path:
    npz = repo_path(npz_path)
    parent = npz.parent
    candidates = [
        parent / f"{run_id}_outdir_full/config_act.yaml",
        parent / f"{run_id}_outdir/config_act.yaml",
        parent / f"{npz.stem}_outdir_full/config_act.yaml",
        parent / f"{npz.stem}_outdir/config_act.yaml",
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"config_act.yaml not found for run_id={run_id} npz={npz_path}")


def sphere_outdir_npz_for(config_act: Path, run_id: str) -> Path:
    outdir = config_act.parent
    candidate = outdir / "trajectory_mjwp_act.npz"
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(f"sphere outdir trajectory missing for {run_id}: {candidate}")


def override_for(run_id: str) -> Path:
    path = OVERRIDE_ROOT / f"core4d_{run_id}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"historical override missing: {path}")
    return path


def object_asset_for(object_key: str, task_dir: Path) -> str:
    assets = sorted((REPO / f"example_datasets/processed/core4d/assets/objects/{object_key}").glob("*"))
    mesh_assets = [p for p in assets if p.suffix.lower() in {".obj", ".stl", ".ply"}]
    if mesh_assets:
        return rel(mesh_assets[0])
    scene = task_dir / "scene_act.xml"
    return rel(scene)


def write_override(path: Path, *, base_run_id: str, task: str) -> None:
    base = f"core4d_{base_run_id}"
    text = "\n".join(
        [
            "# @package _global_",
            "# Auto-generated by workspace/core4d/scripts/E147/build_rubber_hand_collision_manifest.py.",
            "# E147: same historical CEM override, only swapping the robot hand collision scene sidecar.",
            "defaults:",
            f"  - {base}",
            "  - _self_",
            "",
            f"task: {task}",
            f"scene_name: {SCENE_NAME}",
            "video_camera: auto",
            "",
        ]
    )
    path.write_text(text, encoding="utf-8")


def snapshot_scene(task: str, base_scene: Path, rubber_scene: Path, out_dir: Path) -> dict[str, str]:
    snap_dir = out_dir / "scene_snapshot" / task
    snap_dir.mkdir(parents=True, exist_ok=True)
    copied = {}
    for src in [base_scene, rubber_scene, base_scene.parent / "scene.xml", base_scene.parent / "scene_act_meta.json", base_scene.parent / "task_info.json"]:
        if src.is_file():
            dst = snap_dir / src.name
            shutil.copy2(src, dst)
            copied[src.name] = rel(dst)
    manifest = out_dir / "scene_snapshot/manifest.txt"
    return {
        "task": task,
        "base_scene_act": rel(base_scene),
        "rubber_scene_act": rel(rubber_scene),
        "base_sha256": sha(base_scene),
        "rubber_sha256": sha(rubber_scene),
        "snapshot_dir": rel(snap_dir),
        "copied_json": json.dumps(copied, sort_keys=True),
        "manifest": rel(manifest),
    }


def write_scene_snapshot_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# E147 scene snapshot",
        f"# Captured: {timestamp()}",
        f"# Git HEAD: {git_head()}",
        "",
        "task\tfile\tsha256\tpath",
    ]
    for row in rows:
        for key in ("base_scene_act", "rubber_scene_act"):
            p = repo_path(row[key])
            lines.append(f"{row['task']}\t{p.name}\t{sha256_file(p)}\t{row[key]}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build() -> list[dict[str, str]]:
    existing = read_tsv(EXISTING_CASES)
    case_rows = {(row.get("case_id", ""), row.get("cem_run_id", "")): row for row in existing if row.get("cem_result_npz")}
    rows: list[dict[str, str]] = []
    adapter_rows: list[dict[str, Any]] = []
    snapshot_rows: list[dict[str, str]] = []

    for ordinal, (case_id, sphere_run_id, split, expected_quality) in enumerate(CASE_SPECS, start=1):
        src = case_rows.get((case_id, sphere_run_id))
        if not src:
            raise KeyError(f"missing existing_cases row for {case_id} run={sphere_run_id}")
        config_act = config_act_for(sphere_run_id, src["cem_result_npz"])
        config = load_yaml(config_act)
        task = str(config.get("task") or "")
        if not task:
            raise ValueError(f"missing task in {config_act}")
        task_dir = TASK_ROOT / task
        base_scene = task_dir / "scene_act.xml"
        traj = task_dir / str(config.get("data_id", 0)) / "trajectory_kinematic.npz"
        if not base_scene.is_file():
            raise FileNotFoundError(base_scene)
        if not traj.is_file():
            raise FileNotFoundError(traj)

        variant = f"E147_{case_id}_rubber_hull"
        override = OVERRIDE_ROOT / f"core4d_{variant}.yaml"
        write_override(override, base_run_id=sphere_run_id, task=task)

        scene_out = RESULT_ROOT / "hand_collision_scenes" / variant
        adapter = patch_scene(
            base_scene_act=base_scene,
            out_dir=scene_out,
            case_id=case_id,
            hand_collision_variant_id="rubber_hull",
            scene_name=SCENE_NAME,
            install_dir=task_dir,
            repo=REPO,
        )
        adapter_rows.append(adapter)
        rubber_scene = task_dir / f"{SCENE_NAME}.xml"
        snapshot_rows.append(snapshot_scene(task, base_scene, rubber_scene, RESULT_ROOT))

        object_key = src.get("object_key", "")
        person = person_from_case_id(case_id, src.get("person", ""))
        row = {
            "ordinal": str(ordinal),
            "variant": variant,
            "case_id": case_id,
            "object_key": object_key,
            "object_category": case_object_category(object_key),
            "person": person,
            "person_idx": str(config.get("contact_hdmi_mask_person_idx", src.get("person_idx", ""))),
            "split": split,
            "ablation": "rubber_hull",
            "retarget_variant_id": src.get("retarget_variant_id", ""),
            "target_variant_id": src.get("target_variant_id", "ref_fk"),
            "hand_collision_variant_id": "rubber_hull",
            "derived_task": task,
            "target_scene": rel(task_dir / "scene.xml"),
            "trajectory": rel(traj),
            "base_scene_act": rel(base_scene),
            "rubber_scene_act": rel(rubber_scene),
            "scene_name": SCENE_NAME,
            "sphere_run_id": sphere_run_id,
            "sphere_npz": src.get("cem_result_npz", ""),
            "sphere_video": src.get("cem_video", ""),
            "sphere_outdir_npz": rel(sphere_outdir_npz_for(config_act, sphere_run_id)),
            "sphere_config_act": rel(config_act),
            "override": rel(override),
            "object_asset": object_asset_for(object_key, task_dir),
            "remote_sync_key": f"humanoid_object/{task}",
            "historical_cem_status": src.get("cem_status", ""),
            "historical_rl_status": src.get("rl_status", ""),
            "expected_quality": expected_quality,
            "run_status": "to_run",
        }
        rows.append(row)

    write_tsv(SCRIPT_ROOT / "variants.tsv", rows, FIELDS)
    write_tsv(RESULT_ROOT / "cases_manifest.tsv", rows, FIELDS)
    write_json(RESULT_ROOT / "cases_manifest.json", rows)
    write_tsv(RESULT_ROOT / "hand_collision_adapter_manifest.tsv", adapter_rows, list(adapter_rows[0].keys()))
    write_json(RESULT_ROOT / "hand_collision_adapter_manifest.json", adapter_rows)
    write_tsv(RESULT_ROOT / "scene_snapshot/scene_snapshot_manifest.tsv", snapshot_rows, list(snapshot_rows[0].keys()))
    write_scene_snapshot_manifest(RESULT_ROOT / "scene_snapshot/manifest.txt", snapshot_rows)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--print", action="store_true", dest="print_rows")
    args = parser.parse_args()
    rows = build()
    if args.print_rows:
        print(json.dumps(rows, indent=2, ensure_ascii=False))
    else:
        print(f"[E147] wrote {len(rows)} rows to {SCRIPT_ROOT / 'variants.tsv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
