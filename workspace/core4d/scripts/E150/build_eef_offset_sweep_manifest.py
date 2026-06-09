#!/usr/bin/env python3
"""Build E150 contact-anchor eef_offset sweep manifest."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[4]
E148_VARIANTS = REPO / "workspace/core4d/scripts/E148/variants.tsv"
SCRIPT_ROOT = REPO / "workspace/core4d/scripts/E150"
RESULT_ROOT = REPO / "workspace/core4d/results/E150/contact_anchor_eef_offset_sweep"
CEM_ROOT = RESULT_ROOT / "cem/full"

RELAXED8 = [
    "box021_035_p2",
    "box023_person2",
    "box021_029_p2",
    "box004_082_p1",
    "box004_083_p2",
    "box021_035_p1",
    "box026_139_p1",
    "box004_083_p1",
]

RUN_OFFSETS = [0.08, 0.11]
BASELINE_OFFSET = 0.05

FIELDS = [
    "ordinal",
    "variant",
    "short_case_id",
    "e148_variant",
    "e109_case_id",
    "case_id",
    "object_key",
    "object_category",
    "person_idx",
    "split",
    "run_status",
    "source_exp",
    "anchor_variant",
    "eef_offset_x",
    "eef_offset_tag",
    "target_variant_id",
    "hand_collision_variant_id",
    "derived_task",
    "target_scene",
    "trajectory",
    "base_scene_act",
    "rubber_scene_act",
    "scene_name",
    "override",
    "object_asset",
    "mask_path",
    "baseline_variant",
    "baseline_npz",
    "baseline_outdir_npz",
    "baseline_video",
    "result_npz",
    "outdir_npz",
    "video",
    "expected_quality",
    "remote_sync_key",
]


def rel(path: str | Path) -> str:
    if not path:
        return ""
    p = Path(path)
    try:
        return str(p.resolve().relative_to(REPO.resolve()))
    except Exception:
        return str(path)


def repo_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO / p


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


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


def git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:
        return "unknown"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def short_case_id(row: dict[str, str]) -> str:
    variant = row["e143_variant"]
    if not variant.startswith("E143_") or not variant.endswith("_raw_mask_ref_fk"):
        raise ValueError(f"unexpected E143 variant: {variant}")
    return variant.removeprefix("E143_").removesuffix("_raw_mask_ref_fk")


def offset_tag(value: float) -> str:
    return f"off{int(round(value * 100)):02d}"


def source_exp(row: dict[str, str]) -> str:
    if row.get("run_status") == "reuse_e147":
        return row.get("reuse_source_exp") or "E147"
    return "E148"


def result_paths(variant: str) -> tuple[str, str, str]:
    return (
        rel(CEM_ROOT / f"{variant}.npz"),
        rel(CEM_ROOT / f"{variant}_outdir_full/trajectory_mjwp_act.npz"),
        rel(CEM_ROOT / f"{variant}_full.mp4"),
    )


def rubber_task_inputs(e148: dict[str, str]) -> dict[str, str]:
    """Return the task-side files that the rubber override actually loads."""
    task_dir = repo_path(e148["rubber_scene_act"]).parent
    return {
        "derived_task": task_dir.name,
        "target_scene": rel(task_dir / "scene.xml"),
        "trajectory": rel(task_dir / "0/trajectory_kinematic.npz"),
        "base_scene_act": rel(task_dir / "scene_act.xml"),
    }


def base_row(e148: dict[str, str], *, ordinal: int, offset_x: float, split: str, run_status: str) -> dict[str, Any]:
    short = short_case_id(e148)
    tag = offset_tag(offset_x)
    variant = f"E150_{short}_{tag}"
    run_inputs = rubber_task_inputs(e148)
    if run_status == "reuse_e148":
        result_npz = e148["rubber_npz"]
        outdir_npz = e148["rubber_outdir_npz"]
        video = e148["rubber_video"]
    else:
        result_npz, outdir_npz, video = result_paths(variant)
    return {
        "ordinal": ordinal,
        "variant": variant,
        "short_case_id": short,
        "e148_variant": e148["variant"],
        "e109_case_id": e148["e109_case_id"],
        "case_id": e148["case_id"],
        "object_key": e148["object_key"],
        "object_category": e148["object_category"],
        "person_idx": e148["person_idx"],
        "split": split,
        "run_status": run_status,
        "source_exp": source_exp(e148) if run_status == "reuse_e148" else "E150",
        "anchor_variant": tag,
        "eef_offset_x": f"{offset_x:.2f}",
        "eef_offset_tag": tag,
        "target_variant_id": e148["target_variant_id"],
        "hand_collision_variant_id": "rubber_hull",
        "derived_task": run_inputs["derived_task"],
        "target_scene": run_inputs["target_scene"],
        "trajectory": run_inputs["trajectory"],
        "base_scene_act": run_inputs["base_scene_act"],
        "rubber_scene_act": e148["rubber_scene_act"],
        "scene_name": e148["scene_name"],
        "override": e148["override"],
        "object_asset": e148["object_asset"],
        "mask_path": e148["mask_path"],
        "baseline_variant": f"E150_{short}_{offset_tag(BASELINE_OFFSET)}",
        "baseline_npz": e148["rubber_npz"],
        "baseline_outdir_npz": e148["rubber_outdir_npz"],
        "baseline_video": e148["rubber_video"],
        "result_npz": result_npz,
        "outdir_npz": outdir_npz,
        "video": video,
        "expected_quality": e148["expected_quality"],
        "remote_sync_key": e148["remote_sync_key"],
    }


def copy_scene_snapshot(rows: list[dict[str, Any]]) -> None:
    snap_root = RESULT_ROOT / "scene_snapshot"
    snap_root.mkdir(parents=True, exist_ok=True)
    unique: dict[str, dict[str, Any]] = {}
    for row in rows:
        unique.setdefault(row["derived_task"], row)
    lines = [
        "# E150 scene snapshot",
        f"# Git HEAD: {git_head()}",
        "",
        "task\tfile\tsha256\tpath",
    ]
    for task, row in sorted(unique.items()):
        dst_dir = snap_root / task
        dst_dir.mkdir(parents=True, exist_ok=True)
        candidates = [
            repo_path(row["target_scene"]),
            repo_path(row["base_scene_act"]),
            repo_path(row["rubber_scene_act"]),
            repo_path(row["base_scene_act"]).parent / "scene_act_meta.json",
            repo_path(row["base_scene_act"]).parent / "task_info.json",
        ]
        seen: set[Path] = set()
        for src in candidates:
            if src in seen or not src.is_file():
                continue
            seen.add(src)
            dst = dst_dir / src.name
            shutil.copy2(src, dst)
            lines.append(f"{task}\t{src.name}\t{sha256(src)}\t{rel(dst)}")
    (snap_root / "manifest.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate(rows: list[dict[str, Any]]) -> None:
    if len(rows) != 24:
        raise SystemExit(f"E150 manifest expected 24 rows, got {len(rows)}")
    reuse = [row for row in rows if row["run_status"] == "reuse_e148"]
    todo = [row for row in rows if row["run_status"] == "to_run"]
    if len(reuse) != 8 or len(todo) != 16:
        raise SystemExit(f"E150 expected reuse=8/to_run=16, got reuse={len(reuse)} to_run={len(todo)}")
    splits = {row["split"]: 0 for row in todo}
    for row in todo:
        splits[row["split"]] = splits.get(row["split"], 0) + 1
    if splits != {"remote-gpu0": 8, "remote-gpu1": 8}:
        raise SystemExit(f"E150 expected remote-gpu0=8/remote-gpu1=8, got {splits}")
    missing = []
    for row in rows:
        for key in ("override", "target_scene", "base_scene_act", "rubber_scene_act", "object_asset", "mask_path", "baseline_outdir_npz"):
            p = repo_path(row[key])
            if not p.is_file():
                missing.append(f"{row['variant']}:{key}:{row[key]}")
    if missing:
        raise SystemExit("E150 manifest missing files:\n" + "\n".join(missing))


def main() -> None:
    e148_rows = read_tsv(E148_VARIANTS)
    by_short = {short_case_id(row): row for row in e148_rows}
    missing = [case for case in RELAXED8 if case not in by_short]
    if missing:
        raise SystemExit(f"E148 variants missing relaxed8 cases: {missing}")

    rows: list[dict[str, Any]] = []
    ordinal = 1
    for short in RELAXED8:
        e148 = by_short[short]
        rows.append(base_row(e148, ordinal=ordinal, offset_x=BASELINE_OFFSET, split="reuse", run_status="reuse_e148"))
        ordinal += 1
        for offset_x, split in ((0.08, "remote-gpu0"), (0.11, "remote-gpu1")):
            rows.append(base_row(e148, ordinal=ordinal, offset_x=offset_x, split=split, run_status="to_run"))
            ordinal += 1

    validate(rows)
    SCRIPT_ROOT.mkdir(parents=True, exist_ok=True)
    write_tsv(SCRIPT_ROOT / "variants.tsv", rows, FIELDS)
    write_tsv(RESULT_ROOT / "cases_manifest.tsv", rows, FIELDS)
    copy_scene_snapshot(rows)
    write_json(
        RESULT_ROOT / "manifest_summary.json",
        {
            "rows": len(rows),
            "baseline_reuse": 8,
            "to_run": 16,
            "offsets": [BASELINE_OFFSET, *RUN_OFFSETS],
            "benchmark": "E149_relaxed8_valid_like",
            "git_head": git_head(),
        },
    )
    print(f"wrote {rel(SCRIPT_ROOT / 'variants.tsv')}")
    print("rows=24 reuse_e148=8 to_run=16 remote-gpu0=8 remote-gpu1=8")


if __name__ == "__main__":
    main()
