#!/usr/bin/env python3
"""Build E148 E143-24 rubber-hand collision manifest with E147 reuse."""

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

REPO = Path(__file__).resolve().parents[4]
DCV3 = REPO / "workspace/core4d/scripts/data_construction_v3"
for _path in (DCV3 / "lib", DCV3 / "stages/s5_handoff"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from common import sha256_file, timestamp  # noqa: E402
from patch_hand_collision import patch_scene  # noqa: E402


TASK_ROOT = REPO / "example_datasets/processed/core4d/unitree_g1/humanoid_object"
OBJECT_ROOT = REPO / "example_datasets/processed/core4d/assets/objects"
OVERRIDE_ROOT = REPO / "examples/config/override"

E143_VARIANTS = REPO / "workspace/core4d/scripts/E143/variants.tsv"
E147_VARIANTS = REPO / "workspace/core4d/scripts/E147/variants.tsv"
E147_COMPARISON = REPO / "workspace/core4d/results/E147/rubber_hand_collision/comparison/e147_omni_sphere_rubber_case_comparison.tsv"
E147_REGISTRY = REPO / "workspace/core4d/results/E147/registries/case_state_registry.tsv"
E147_EVIDENCE = REPO / "workspace/core4d/results/E147/s6_downstream/evidence/downstream_evidence_manifest.tsv"

SCRIPT_ROOT = REPO / "workspace/core4d/scripts/E148"
RESULT_ROOT = REPO / "workspace/core4d/results/E148/e143_24case_rubber_hand_collision"
E148_CEM_ROOT = RESULT_ROOT / "cem/full"
SCENE_NAME = "scene_act_E148_rubber_hull"
E147_RESULT_ROOT = REPO / "workspace/core4d/results/E147/rubber_hand_collision/cem/full"

E143_FIELDS = [
    "ordinal",
    "variant",
    "e109_case_id",
    "case_id",
    "object_key",
    "target_variant_id",
    "source_task",
    "derived_task",
    "person_idx",
    "split",
    "ablation",
    "mask_path",
    "mask_kind",
    "source_mask_path",
    "baseline_npz_path",
    "baseline_run_id",
    "omni_qpos_path",
    "omni_scene_xml",
    "spider_scene_xml",
    "override",
    "run_status",
    "reuse_source_exp",
    "reuse_npz_path",
    "reuse_video_path",
    "reuse_outdir_npz_path",
]

FIELDS = [
    "ordinal",
    "variant",
    "e143_variant",
    "e109_case_id",
    "case_id",
    "object_key",
    "object_category",
    "person_idx",
    "split",
    "run_status",
    "reuse_source_exp",
    "e147_variant",
    "target_variant_id",
    "hand_collision_variant_id",
    "derived_task",
    "target_scene",
    "trajectory",
    "base_scene_act",
    "rubber_scene_act",
    "scene_name",
    "sphere_npz",
    "sphere_outdir_npz",
    "sphere_video",
    "sphere_scene_act",
    "omni_qpos_path",
    "omni_scene_xml",
    "override",
    "object_asset",
    "mask_path",
    "remote_sync_key",
    "expected_quality",
    "rubber_npz",
    "rubber_outdir_npz",
    "rubber_video",
    "spider_cem_status",
]


def rel(path: Path | str) -> str:
    if not path:
        return ""
    p = Path(path)
    try:
        return str(p.resolve().relative_to(REPO.resolve()))
    except Exception:
        return str(path)


def repo_path(text: str | Path) -> Path:
    p = Path(text)
    return p if p.is_absolute() else REPO / p


def sha(path: Path | str) -> str:
    p = repo_path(path)
    return hashlib.sha256(p.read_bytes()).hexdigest()


def git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:
        return "unknown"


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def read_e143_variants(path: Path = E143_VARIANTS) -> list[dict[str, str]]:
    rows = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            values = line.rstrip("\n").split("\t")
            rows.append(dict(zip(E143_FIELDS, values)))
    return rows


def write_tsv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: "" if row.get(field) is None else row.get(field, "") for field in fields})


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def object_category(object_key: str) -> str:
    if object_key.startswith("box"):
        return "box"
    if object_key.startswith("bucket"):
        return "bucket"
    return object_key.rstrip("0123456789") or object_key


def variant_base(e143_variant: str) -> str:
    text = e143_variant
    if text.startswith("E143_"):
        text = text[len("E143_") :]
    if text.endswith("_raw_mask_ref_fk"):
        text = text[: -len("_raw_mask_ref_fk")]
    return text


def expected_e143_paths(row: dict[str, str]) -> tuple[str, str, str]:
    if row.get("run_status") == "already_done":
        return row["reuse_npz_path"], row["reuse_outdir_npz_path"], row["reuse_video_path"]
    root = f"workspace/core4d/results/E143/cem/full/{row['variant']}.npz"
    outdir = f"workspace/core4d/results/E143/cem/full/{row['variant']}_outdir_full/trajectory_mjwp_act.npz"
    video = f"workspace/core4d/results/E143/cem/full/{row['variant']}_full.mp4"
    return root, outdir, video


def object_asset_for(object_key: str) -> str:
    obj_dir = OBJECT_ROOT / object_key
    if obj_dir.is_dir():
        mesh_assets = sorted(p for p in obj_dir.glob("*") if p.suffix.lower() in {".obj", ".stl", ".ply"})
        if mesh_assets:
            return rel(mesh_assets[0])
    return rel(obj_dir)


def write_override(path: Path, *, e143_override: str, task: str) -> None:
    base = Path(e143_override).stem
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "# @package _global_",
                "# Auto-generated by workspace/core4d/scripts/E148/build_rubber_hand_collision_manifest.py.",
                "# E148: same E143 raw_mask_ref_fk CEM override, only swapping robot hand collision sidecar.",
                "defaults:",
                f"  - {base}",
                "  - _self_",
                "",
                f"task: {task}",
                f"scene_name: {SCENE_NAME}",
                "video_camera: auto",
                "",
            ]
        ),
        encoding="utf-8",
    )


def snapshot_scene(task: str, base_scene: Path, rubber_scene: Path) -> dict[str, str]:
    snap_dir = RESULT_ROOT / "scene_snapshot" / task
    snap_dir.mkdir(parents=True, exist_ok=True)
    copied: dict[str, str] = {}
    for src in [
        base_scene,
        rubber_scene,
        base_scene.parent / "scene.xml",
        base_scene.parent / "scene_act_meta.json",
        base_scene.parent / "task_info.json",
    ]:
        if src.is_file():
            dst = snap_dir / src.name
            shutil.copy2(src, dst)
            copied[src.name] = rel(dst)
    return {
        "task": task,
        "base_scene_act": rel(base_scene),
        "rubber_scene_act": rel(rubber_scene),
        "base_sha256": sha(base_scene),
        "rubber_sha256": sha(rubber_scene),
        "snapshot_dir": rel(snap_dir),
        "copied_json": json.dumps(copied, sort_keys=True),
    }


def write_scene_snapshot_manifest(rows: list[dict[str, str]]) -> None:
    path = RESULT_ROOT / "scene_snapshot/manifest.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# E148 scene snapshot",
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


def e147_reuse_maps() -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    variants = {row["case_id"]: row for row in read_tsv(E147_VARIANTS)}
    comparison = {row["case_id"]: row for row in read_tsv(E147_COMPARISON)}
    registry = {
        row["case_id"]: row
        for row in read_tsv(E147_REGISTRY)
        if row.get("hand_collision_variant_id") == "rubber_hull"
    }
    _evidence = read_tsv(E147_EVIDENCE)
    return variants, comparison, registry


def reuse_row(e143: dict[str, str], e147: dict[str, str], comparison: dict[str, str], registry: dict[str, str]) -> dict[str, str] | None:
    case_id = e143["e109_case_id"]
    src = e147.get(case_id)
    cmp = comparison.get(case_id)
    reg = registry.get(case_id)
    if not src or not cmp or not reg:
        return None
    rubber_outdir = repo_path(cmp.get("rubber_npz", ""))
    e147_root = E147_RESULT_ROOT / f"{src['variant']}.npz"
    e147_video = E147_RESULT_ROOT / f"{src['variant']}_full.mp4"
    if not rubber_outdir.is_file() or not e147_root.is_file() or not e147_video.is_file():
        return None
    if sha(e143["spider_scene_xml"]) != sha(src["base_scene_act"]):
        return None
    return {
        "run_status": "reuse_e147",
        "reuse_source_exp": "E147",
        "e147_variant": src["variant"],
        "rubber_npz": rel(e147_root),
        "rubber_outdir_npz": rel(rubber_outdir),
        "rubber_video": rel(e147_video),
        "rubber_scene_act": src["rubber_scene_act"],
        "override": src["override"],
    }


def build() -> list[dict[str, str]]:
    e143_rows = read_e143_variants()
    e147, e147_cmp, e147_registry = e147_reuse_maps()
    rows: list[dict[str, str]] = []
    adapter_rows: list[dict[str, Any]] = []
    snapshot_rows: list[dict[str, str]] = []
    to_run_counter = 0

    for ordinal, e143 in enumerate(e143_rows, start=1):
        case_id = e143["e109_case_id"]
        base = variant_base(e143["variant"])
        variant = f"E148_{base}_rubber_hull"
        task = e143["derived_task"]
        task_dir = TASK_ROOT / task
        base_scene = task_dir / "scene_act.xml"
        target_scene = task_dir / "scene.xml"
        trajectory = task_dir / "0/trajectory_kinematic.npz"
        sphere_npz, sphere_outdir, sphere_video = expected_e143_paths(e143)
        expected_quality = "known_visual_fail" if case_id == "bucket004_20231003_1_012_p1" else "review"

        for required in [base_scene, target_scene, trajectory, repo_path(sphere_outdir), repo_path(e143["mask_path"])]:
            if not required.is_file():
                raise FileNotFoundError(f"missing required input for {case_id}: {required}")

        reuse = reuse_row(e143, e147, e147_cmp, e147_registry)
        if reuse:
            split = "reuse"
            run_status = "reuse_e147"
            override = reuse["override"]
            rubber_scene = reuse["rubber_scene_act"]
            rubber_npz = reuse["rubber_npz"]
            rubber_outdir = reuse["rubber_outdir_npz"]
            rubber_video = reuse["rubber_video"]
            e147_variant = reuse["e147_variant"]
            reuse_source = "E147"
        else:
            split = "remote-gpu0" if to_run_counter % 2 == 0 else "remote-gpu1"
            to_run_counter += 1
            run_status = "to_run"
            override_path = OVERRIDE_ROOT / f"core4d_{variant}.yaml"
            write_override(override_path, e143_override=e143["override"], task=task)
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
            adapter["variant"] = variant
            adapter["e143_variant"] = e143["variant"]
            adapter_rows.append(adapter)
            installed = task_dir / f"{SCENE_NAME}.xml"
            snapshot_rows.append(snapshot_scene(task, base_scene, installed))
            override = rel(override_path)
            rubber_scene = rel(installed)
            rubber_npz = rel(E148_CEM_ROOT / f"{variant}.npz")
            rubber_outdir = rel(E148_CEM_ROOT / f"{variant}_outdir_full/trajectory_mjwp_act.npz")
            rubber_video = rel(E148_CEM_ROOT / f"{variant}_full.mp4")
            e147_variant = ""
            reuse_source = ""

        row = {
            "ordinal": str(ordinal),
            "variant": variant,
            "e143_variant": e143["variant"],
            "e109_case_id": case_id,
            "case_id": case_id,
            "object_key": e143["object_key"],
            "object_category": object_category(e143["object_key"]),
            "person_idx": e143["person_idx"],
            "split": split,
            "run_status": run_status,
            "reuse_source_exp": reuse_source,
            "e147_variant": e147_variant,
            "target_variant_id": e143["target_variant_id"],
            "hand_collision_variant_id": "rubber_hull",
            "derived_task": task,
            "target_scene": rel(target_scene),
            "trajectory": rel(trajectory),
            "base_scene_act": rel(base_scene),
            "rubber_scene_act": rubber_scene,
            "scene_name": SCENE_NAME if run_status == "to_run" else Path(rubber_scene).stem,
            "sphere_npz": sphere_npz,
            "sphere_outdir_npz": sphere_outdir,
            "sphere_video": sphere_video,
            "sphere_scene_act": rel(base_scene),
            "omni_qpos_path": e143["omni_qpos_path"],
            "omni_scene_xml": e143["omni_scene_xml"],
            "override": override,
            "object_asset": object_asset_for(e143["object_key"]),
            "mask_path": e143["mask_path"],
            "remote_sync_key": f"humanoid_object/{task}",
            "expected_quality": expected_quality,
            "rubber_npz": rubber_npz,
            "rubber_outdir_npz": rubber_outdir,
            "rubber_video": rubber_video,
            "spider_cem_status": "",
        }
        rows.append(row)

    if len(rows) != 24:
        raise RuntimeError(f"E148 expected 24 rows, got {len(rows)}")
    reuse_count = sum(row["run_status"] == "reuse_e147" for row in rows)
    to_run_count = sum(row["run_status"] == "to_run" for row in rows)
    if reuse_count != 8 or to_run_count != 16:
        raise RuntimeError(f"E148 expected reuse=8/to_run=16, got reuse={reuse_count} to_run={to_run_count}")

    SCRIPT_ROOT.mkdir(parents=True, exist_ok=True)
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    write_tsv(SCRIPT_ROOT / "variants.tsv", rows, FIELDS)
    write_tsv(RESULT_ROOT / "cases_manifest.tsv", rows, FIELDS)
    write_json(RESULT_ROOT / "cases_manifest.json", rows)
    if adapter_rows:
        adapter_fields = sorted({key for row in adapter_rows for key in row})
        write_tsv(RESULT_ROOT / "hand_collision_adapter_manifest.tsv", adapter_rows, adapter_fields)
        write_json(RESULT_ROOT / "hand_collision_adapter_manifest.json", adapter_rows)
    if snapshot_rows:
        write_tsv(RESULT_ROOT / "scene_snapshot/scene_snapshot_manifest.tsv", snapshot_rows, list(snapshot_rows[0].keys()))
        write_scene_snapshot_manifest(snapshot_rows)
    summary = {
        "rows": len(rows),
        "reuse_e147": reuse_count,
        "to_run": to_run_count,
        "remote_gpu0": sum(row["split"] == "remote-gpu0" for row in rows),
        "remote_gpu1": sum(row["split"] == "remote-gpu1" for row in rows),
        "reuse_cases": [row["case_id"] for row in rows if row["run_status"] == "reuse_e147"],
    }
    write_json(RESULT_ROOT / "manifest_summary.json", summary)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--print", action="store_true", dest="print_rows")
    args = parser.parse_args()
    rows = build()
    if args.print_rows:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
    else:
        print(
            f"[E148] wrote {len(rows)} rows to {SCRIPT_ROOT / 'variants.tsv'} "
            f"(reuse={sum(r['run_status'] == 'reuse_e147' for r in rows)}, "
            f"to_run={sum(r['run_status'] == 'to_run' for r in rows)})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
