#!/usr/bin/env python3
"""Build E151 route-B rubber-hand surface reward manifest and overrides."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[4]
SCRIPT_ROOT = REPO / "workspace/core4d/scripts/E151"
RESULT_ROOT = REPO / "workspace/core4d/results/E151/route_b_hand_surface_contact"
CEM_ROOT = RESULT_ROOT / "cem/full"
OVERRIDE_ROOT = REPO / "examples/config/override"
E148_VARIANTS = REPO / "workspace/core4d/scripts/E148/variants.tsv"
E093_MANIFEST = REPO / "workspace/core4d/results/E093/contact_geometry/case_manifest.tsv"
E094_SCRIPT = REPO / "workspace/core4d/scripts/E094/build_handbox_target_projection.py"
E100_SCRIPT = REPO / "workspace/core4d/scripts/E100/build_fingertip_aware_target.py"

CASES = ["box021_029_p2", "box004_083_p2", "box023_person2"]
METHODS = ["baseline", "b2_sup", "b2_tip", "b1_mesh"]

B2_SUP_TARGETS = {
    "box021_029_p2": REPO / "workspace/core4d/results/E094/handbox_target_projection/targets/box021_d003_029_p2_d003_box021_20231018_029_p2_adaptive_support_targets.npz",
    "box004_083_p2": REPO / "workspace/core4d/results/E094/handbox_target_projection/targets/box004_083_p2_e091_box004_20231003_2_083_p2_adaptive_support_targets.npz",
    "box023_person2": REPO / "workspace/core4d/results/E094/handbox_target_projection/targets/box023_p2_box023_person2_adaptive_support_targets.npz",
}

B2_TIP_TARGETS = {
    "box021_029_p2": REPO / "workspace/core4d/results/E100/fingertip_targets/d003_box021_20231018_029_p2/spider_contact_target_object_local.npz",
    "box004_083_p2": REPO / "workspace/core4d/results/E100/fingertip_targets/e091_box004_20231003_2_083_p2/spider_contact_target_object_local.npz",
    "box023_person2": REPO / "workspace/core4d/results/E100/fingertip_targets/box023_person2/spider_contact_target_object_local.npz",
}

BOX021_CLEAN_TASK = "d003_box021_20231018_029_p2_e107_clean"
BOX021_CLEAN_SCENE = REPO / f"example_datasets/processed/core4d/unitree_g1/humanoid_object/{BOX021_CLEAN_TASK}/scene.xml"
BOX021_CLEAN_TRAJ = REPO / f"example_datasets/processed/core4d/unitree_g1/humanoid_object/{BOX021_CLEAN_TASK}/0/trajectory_kinematic.npz"
TARGET_DIFF_THRESHOLD_M = 0.005

FIELDS = [
    "ordinal",
    "variant",
    "short_case_id",
    "method",
    "method_group",
    "e148_variant",
    "e109_case_id",
    "case_id",
    "object_key",
    "object_category",
    "person_idx",
    "split",
    "run_status",
    "source_exp",
    "target_variant_id",
    "target_npz",
    "target_clean_max_diff_m",
    "target_clean_selected",
    "hand_support_scale",
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


def rubber_task_inputs(e148: dict[str, str]) -> dict[str, str]:
    task_dir = repo_path(e148["rubber_scene_act"]).parent
    return {
        "derived_task": task_dir.name,
        "target_scene": rel(task_dir / "scene.xml"),
        "trajectory": rel(task_dir / "0/trajectory_kinematic.npz"),
        "base_scene_act": rel(task_dir / "scene_act.xml"),
    }


def result_paths(variant: str) -> tuple[str, str, str]:
    return (
        rel(CEM_ROOT / f"{variant}.npz"),
        rel(CEM_ROOT / f"{variant}_outdir_full/trajectory_mjwp_act.npz"),
        rel(CEM_ROOT / f"{variant}_full.mp4"),
    )


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def npz_target(path: Path) -> np.ndarray:
    data = np.load(path, allow_pickle=True)
    if "spider_contact_target_object_local" not in data:
        raise KeyError(f"{path} missing spider_contact_target_object_local")
    arr = data["spider_contact_target_object_local"]
    if arr.ndim != 3 or arr.shape[1:] != (2, 3):
        raise ValueError(f"{path} target shape should be (T,2,3), got {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{path} target contains non-finite values")
    return arr.astype(np.float64)


def compare_targets(old_path: Path, new_path: Path) -> dict[str, Any]:
    old = npz_target(old_path)
    new = npz_target(new_path)
    if old.shape != new.shape:
        raise ValueError(f"target shape mismatch: {old_path} {old.shape} vs {new_path} {new.shape}")
    diff = np.linalg.norm(old - new, axis=-1)
    return {
        "old_target": rel(old_path),
        "clean_target": rel(new_path),
        "shape": "x".join(map(str, old.shape)),
        "max_diff_m": float(diff.max()),
        "mean_diff_m": float(diff.mean()),
        "p95_diff_m": float(np.percentile(diff, 95)),
    }


def ensure_box021_clean_targets() -> dict[str, dict[str, Any]]:
    """Regenerate box021 targets on the E107 clean scene and choose old vs clean."""
    out: dict[str, dict[str, Any]] = {}
    audit_root = RESULT_ROOT / "target_clean_check"
    audit_root.mkdir(parents=True, exist_ok=True)

    e094 = load_module(E094_SCRIPT, "e094_for_e151")
    e093_rows = read_tsv(E093_MANIFEST)
    source = next(row for row in e093_rows if row["case_id"] == "box021_d003_029_p2")
    clean_row = dict(source)
    clean_row["task"] = BOX021_CLEAN_TASK
    clean_row["task_dir"] = str(BOX021_CLEAN_SCENE.parent)
    clean_row["scene_xml"] = str(BOX021_CLEAN_SCENE)
    clean_row["trajectory_npz"] = str(BOX021_CLEAN_TRAJ)
    handbox = e094.parse_handbox_urdf(e094.HANDBOX_URDF)
    clean_sup_root = audit_root / "E094_clean"
    clean_sup = clean_sup_root / "targets" / f"{clean_row['case_id']}_{clean_row['task']}_adaptive_support_targets.npz"
    if not clean_sup.is_file():
        e094.compute_case(
            clean_row,
            clean_sup_root,
            handbox,
            0.015,
            6000,
            "adaptive_support",
            0.30,
        )
    sup_cmp = compare_targets(B2_SUP_TARGETS["box021_029_p2"], clean_sup)
    sup_use_clean = sup_cmp["max_diff_m"] > TARGET_DIFF_THRESHOLD_M
    out["b2_sup"] = {
        **sup_cmp,
        "method": "b2_sup",
        "selected_target": sup_cmp["clean_target"] if sup_use_clean else sup_cmp["old_target"],
        "selected": "clean" if sup_use_clean else "old",
    }

    e100 = load_module(E100_SCRIPT, "e100_for_e151")
    vote_path = REPO / "workspace/core4d/results/E099/fingertip_vote_per_case/d003_box021_20231018_029_p2.json"
    vote = json.loads(vote_path.read_text(encoding="utf-8"))
    clean_tip_dir = audit_root / "E100_clean" / "d003_box021_20231018_029_p2"
    clean_tip = clean_tip_dir / "spider_contact_target_object_local.npz"
    if not clean_tip.is_file():
        clean_tip_dir.mkdir(parents=True, exist_ok=True)
        res = e100.build_target("d003_box021_20231018_029_p2", BOX021_CLEAN_SCENE, BOX021_CLEAN_TRAJ, vote)
        np.savez(
            clean_tip,
            spider_contact_target_object_local=res["spider_contact_target_object_local"],
            palm_local_record=res["palm_local_record"],
            active=res["active"],
        )
        (clean_tip_dir / "summary.json").write_text(
            json.dumps(res["summary"], indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    tip_cmp = compare_targets(B2_TIP_TARGETS["box021_029_p2"], clean_tip)
    tip_use_clean = tip_cmp["max_diff_m"] > TARGET_DIFF_THRESHOLD_M
    out["b2_tip"] = {
        **tip_cmp,
        "method": "b2_tip",
        "selected_target": tip_cmp["clean_target"] if tip_use_clean else tip_cmp["old_target"],
        "selected": "clean" if tip_use_clean else "old",
    }

    audit_rows = [
        {k: item.get(k, "") for k in ["method", "old_target", "clean_target", "selected_target", "selected", "shape", "max_diff_m", "mean_diff_m", "p95_diff_m"]}
        for item in out.values()
    ]
    write_tsv(
        audit_root / "box021_clean_target_diff.tsv",
        audit_rows,
        ["method", "old_target", "clean_target", "selected_target", "selected", "shape", "max_diff_m", "mean_diff_m", "p95_diff_m"],
    )
    write_json(audit_root / "box021_clean_target_diff.json", out)
    return out


def target_for(short: str, method: str, clean_audit: dict[str, dict[str, Any]]) -> tuple[str, str, str]:
    if method == "b2_sup":
        if short == "box021_029_p2":
            item = clean_audit["b2_sup"]
            return item["selected_target"], f"{item['max_diff_m']:.8f}", item["selected"]
        return rel(B2_SUP_TARGETS[short]), "", "old"
    if method == "b2_tip":
        if short == "box021_029_p2":
            item = clean_audit["b2_tip"]
            return item["selected_target"], f"{item['max_diff_m']:.8f}", item["selected"]
        return rel(B2_TIP_TARGETS[short]), "", "old"
    return "", "", ""


def split_for(short: str) -> str:
    if short == "box021_029_p2":
        return "local-gpu0"
    if short == "box004_083_p2":
        return "remote-gpu0"
    if short == "box023_person2":
        return "remote-gpu1"
    raise ValueError(short)


def write_override(row: dict[str, Any], base_override: str) -> str:
    path = OVERRIDE_ROOT / f"core4d_{row['variant']}.yaml"
    base_stem = Path(base_override).stem
    lines = [
        "# @package _global_",
        "# Auto-generated by workspace/core4d/scripts/E151/build_route_b_manifest.py.",
        f"# E151 route-B method={row['method']} case={row['short_case_id']}.",
        "defaults:",
        f"  - {base_stem}",
        "  - _self_",
        "",
        f"task: {row['derived_task']}",
        f"scene_name: {row['scene_name']}",
        "video_camera: auto",
        "",
        "contact_hdmi_dynamic_target: true",
    ]
    if row["method"] in {"b2_sup", "b2_tip"}:
        lines += [
            "contact_hdmi_target_source: external",
            f"contact_hdmi_target_path: {row['target_npz']}",
            "contact_hdmi_target_time_axis: auto",
            "contact_hdmi_target_uses_eef_offset: false",
            "contact_hdmi_gain: 5.0",
            "",
            "hand_support_rew_scale: 0.0",
            "hand_support_geom_ids: []",
            "hand_object_deep_penalty_scale: 0.0",
            "hand_object_deep_penalty_geom_ids: []",
        ]
    elif row["method"] == "b1_mesh":
        lines += [
            "contact_hdmi_target_source: ref_fk",
            'contact_hdmi_target_path: ""',
            "contact_hdmi_target_time_axis: auto",
            "contact_hdmi_target_uses_eef_offset: true",
            "contact_hdmi_gain: 5.0",
            "",
            "hand_support_rew_scale: 3.0",
            "hand_support_sigma: 0.015",
            "hand_support_margin_m: 0.01",
            "hand_support_gate_source: contact_mask",
            "hand_support_start_eval_time: 0.0",
            "hand_support_end_eval_time: 0.0",
            'hand_support_geom_names: ["lh", "rh"]',
            "hand_support_geom_ids: []",
            "hand_object_deep_penalty_scale: 0.0",
            "hand_object_deep_penalty_geom_ids: []",
        ]
    else:
        raise ValueError(row["method"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rel(path)


def build_row(
    e148: dict[str, str],
    *,
    ordinal: int,
    method: str,
    clean_audit: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    short = short_case_id(e148)
    variant = f"E151_{short}_{method}"
    run_inputs = rubber_task_inputs(e148)
    baseline_variant = f"E151_{short}_baseline"
    target_npz, clean_max_diff, clean_selected = target_for(short, method, clean_audit)
    if method == "baseline":
        result_npz = e148["rubber_npz"]
        outdir_npz = e148["rubber_outdir_npz"]
        video = e148["rubber_video"]
        split = "reuse"
        run_status = "reuse_e148"
        source_exp = e148.get("reuse_source_exp") or ("E147" if e148["run_status"] == "reuse_e147" else "E148")
        override = e148["override"]
        method_group = "baseline"
    else:
        result_npz, outdir_npz, video = result_paths(variant)
        split = split_for(short)
        run_status = "to_run"
        source_exp = "E151"
        method_group = "B2" if method.startswith("b2_") else "B1"
        override = ""
    row: dict[str, Any] = {
        "ordinal": ordinal,
        "variant": variant,
        "short_case_id": short,
        "method": method,
        "method_group": method_group,
        "e148_variant": e148["variant"],
        "e109_case_id": e148["e109_case_id"],
        "case_id": e148["case_id"],
        "object_key": e148["object_key"],
        "object_category": e148["object_category"],
        "person_idx": e148["person_idx"],
        "split": split,
        "run_status": run_status,
        "source_exp": source_exp,
        "target_variant_id": method,
        "target_npz": target_npz,
        "target_clean_max_diff_m": clean_max_diff,
        "target_clean_selected": clean_selected,
        "hand_support_scale": "3.0" if method == "b1_mesh" else "0.0",
        "hand_collision_variant_id": "rubber_hull",
        "derived_task": run_inputs["derived_task"],
        "target_scene": run_inputs["target_scene"],
        "trajectory": run_inputs["trajectory"],
        "base_scene_act": run_inputs["base_scene_act"],
        "rubber_scene_act": e148["rubber_scene_act"],
        "scene_name": e148["scene_name"],
        "override": override,
        "object_asset": e148["object_asset"],
        "mask_path": e148["mask_path"],
        "baseline_variant": baseline_variant,
        "baseline_npz": e148["rubber_npz"],
        "baseline_outdir_npz": e148["rubber_outdir_npz"],
        "baseline_video": e148["rubber_video"],
        "result_npz": result_npz,
        "outdir_npz": outdir_npz,
        "video": video,
        "expected_quality": e148["expected_quality"],
        "remote_sync_key": e148["remote_sync_key"],
    }
    if method != "baseline":
        row["override"] = write_override(row, e148["override"])
    return row


def copy_scene_snapshot(rows: list[dict[str, Any]]) -> None:
    snap_root = RESULT_ROOT / "scene_snapshot"
    snap_root.mkdir(parents=True, exist_ok=True)
    unique: dict[str, dict[str, Any]] = {}
    for row in rows:
        unique.setdefault(row["derived_task"], row)
    lines = [
        "# E151 scene snapshot",
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
    if len(rows) != 12:
        raise SystemExit(f"E151 expected 12 rows, got {len(rows)}")
    todo = [row for row in rows if row["run_status"] == "to_run"]
    reuse = [row for row in rows if row["run_status"] == "reuse_e148"]
    if len(todo) != 9 or len(reuse) != 3:
        raise SystemExit(f"E151 expected todo=9/reuse=3, got todo={len(todo)} reuse={len(reuse)}")
    splits: dict[str, int] = {}
    for row in todo:
        splits[row["split"]] = splits.get(row["split"], 0) + 1
    if splits != {"local-gpu0": 3, "remote-gpu0": 3, "remote-gpu1": 3}:
        raise SystemExit(f"E151 expected split 3/3/3, got {splits}")
    missing = []
    for row in rows:
        for key in ("override", "target_scene", "trajectory", "base_scene_act", "rubber_scene_act", "object_asset", "mask_path", "baseline_outdir_npz"):
            if not repo_path(row[key]).is_file():
                missing.append(f"{row['variant']}:{key}:{row[key]}")
        if row["method"] in {"b2_sup", "b2_tip"} and not repo_path(row["target_npz"]).is_file():
            missing.append(f"{row['variant']}:target_npz:{row['target_npz']}")
        if row["method"] in {"b2_sup", "b2_tip"}:
            qpos_t = int(np.load(repo_path(row["trajectory"]), allow_pickle=True)["qpos"].shape[0])
            target_t = int(npz_target(repo_path(row["target_npz"])).shape[0])
            if qpos_t != target_t:
                missing.append(f"{row['variant']}:target_len:{target_t} != qpos_len:{qpos_t}")
    if missing:
        raise SystemExit("E151 manifest missing/invalid files:\n" + "\n".join(missing))


def main() -> None:
    clean_audit = ensure_box021_clean_targets()
    e148_rows = read_tsv(E148_VARIANTS)
    by_short = {short_case_id(row): row for row in e148_rows}
    missing = [case for case in CASES if case not in by_short]
    if missing:
        raise SystemExit(f"E148 variants missing E151 cases: {missing}")

    rows: list[dict[str, Any]] = []
    ordinal = 1
    for short in CASES:
        for method in METHODS:
            rows.append(build_row(by_short[short], ordinal=ordinal, method=method, clean_audit=clean_audit))
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
            "baseline_reuse": 3,
            "to_run": 9,
            "cases": CASES,
            "methods": METHODS,
            "split": {"local-gpu0": 3, "remote-gpu0": 3, "remote-gpu1": 3},
            "box021_clean_target_diff_threshold_m": TARGET_DIFF_THRESHOLD_M,
            "git_head": git_head(),
        },
    )
    print(f"wrote {rel(SCRIPT_ROOT / 'variants.tsv')}")
    print("rows=12 reuse=3 to_run=9 local-gpu0=3 remote-gpu0=3 remote-gpu1=3")


if __name__ == "__main__":
    main()
