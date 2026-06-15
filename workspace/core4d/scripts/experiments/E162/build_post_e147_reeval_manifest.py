#!/usr/bin/env python3
"""Build the E162 post-E147 RL-safe re-evaluation manifest.

E162 is eval-only. It normalizes reusable post-E147 trajectories into one
manifest so the evaluator can recompute every row with the current shared
metric standard and a single E147 spider-rubberhand baseline.
"""

from __future__ import annotations

import csv
import json
import re
import subprocess
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[5]
SCRIPT_ROOT = REPO / "workspace/core4d/scripts/experiments/E162"
RESULT_ROOT = REPO / "workspace/core4d/results/E162/post_e147_rl_safe_reeval"
CONTACT_MASK_ROOT = REPO / "workspace/core4d/results/E143/contact_masks"

EXPERIMENT_VARIANTS = [
    "E147",
    "E148",
    "E150",
    "E151",
    "E152",
    "E156",
    "E158",
    "E159",
    "E160",
    "E161",
]

FIELDS = [
    "ordinal",
    "row_id",
    "manifest_exp_id",
    "source_exp_id",
    "source_method_id",
    "canonical_method_id",
    "method_display",
    "case_id",
    "short_case_id",
    "variant",
    "qpos_path",
    "scene_xml",
    "video_path",
    "contact_mask_path",
    "kin_ref_path",
    "person_idx",
    "hand_collision_variant_id",
    "object_key",
    "object_category",
    "expected_quality",
    "baseline_method_id",
    "baseline_source_exp_id",
    "baseline_variant",
    "baseline_qpos_path",
    "baseline_scene_xml",
    "baseline_contact_mask_path",
    "baseline_kin_ref_path",
    "baseline_status",
    "row_status",
    "notes",
]

E153_CASES: dict[str, dict[str, str]] = {
    "box021_029_p2": {
        "case_id": "d003_box021_20231018_029_p2",
        "object_key": "box021",
        "person_idx": "1",
        "scene": "example_datasets/processed/core4d/unitree_g1/humanoid_object/d003_box021_20231018_029_p2_e107_clean/scene_act_E148_rubber_hull.xml",
        "kin_ref": "example_datasets/processed/core4d/unitree_g1/humanoid_object/d003_box021_20231018_029_p2_e107_clean/0/trajectory_kinematic.npz",
        "baseline": "workspace/core4d/results/E148/e143_24case_rubber_hand_collision/cem/full/E148_box021_029_p2_rubber_hull_outdir_full/trajectory_mjwp_act.npz",
        "b1": "workspace/core4d/results/E151/route_b_hand_surface_contact/cem/full/E151_box021_029_p2_b1_mesh_outdir_full/trajectory_mjwp_act.npz",
    },
    "box004_083_p2": {
        "case_id": "e091_box004_20231003_2_083_p2",
        "object_key": "box004",
        "person_idx": "1",
        "scene": "example_datasets/processed/core4d/unitree_g1/humanoid_object/e091_box004_20231003_2_083_p2_e092_dyn/scene_act_E148_rubber_hull.xml",
        "kin_ref": "example_datasets/processed/core4d/unitree_g1/humanoid_object/e091_box004_20231003_2_083_p2_e092_dyn/0/trajectory_kinematic.npz",
        "baseline": "workspace/core4d/results/E148/e143_24case_rubber_hand_collision/cem/full/E148_box004_083_p2_rubber_hull_outdir_full/trajectory_mjwp_act.npz",
        "b1": "workspace/core4d/results/E151/route_b_hand_surface_contact/cem/full/E151_box004_083_p2_b1_mesh_outdir_full/trajectory_mjwp_act.npz",
    },
    "box023_person2": {
        "case_id": "box023_person2",
        "object_key": "box023",
        "person_idx": "1",
        "scene": "example_datasets/processed/core4d/unitree_g1/humanoid_object/box023_person2_legobj/scene_act_E147_rubber_hull.xml",
        "kin_ref": "example_datasets/processed/core4d/unitree_g1/humanoid_object/box023_person2_legobj/0/trajectory_kinematic.npz",
        "baseline": "workspace/core4d/results/E147/rubber_hand_collision/cem/full/E147_box023_person2_rubber_hull_outdir_full/trajectory_mjwp_act.npz",
        "b1": "workspace/core4d/results/E151/route_b_hand_surface_contact/cem/full/E151_box023_person2_b1_mesh_outdir_full/trajectory_mjwp_act.npz",
    },
}

E153_GRID = [
    ("sdf005_v05", -0.005, 0.05),
    ("sdf005_v10", -0.005, 0.10),
    ("sdf010_v05", -0.010, 0.05),
    ("sdf010_v10", -0.010, 0.10),
    ("sdf015_v05", -0.015, 0.05),
    ("sdf015_v10", -0.015, 0.10),
]
E155_METHODS = ["ramp5", "ramp10", "decay", "neutral"]


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


def safe_id(text: str) -> str:
    out = text.strip()
    out = out.replace("+", "_").replace("-", "")
    out = re.sub(r"[^A-Za-z0-9_]+", "_", out)
    out = re.sub(r"_+", "_", out).strip("_")
    return out or "method"


def short_case_id(row: dict[str, str]) -> str:
    if row.get("short_case_id"):
        return row["short_case_id"]
    if row.get("e143_variant", "").startswith("E143_"):
        return row["e143_variant"].removeprefix("E143_").removesuffix("_raw_mask_ref_fk")
    case = row.get("case_id", "")
    if case.startswith(("d003_", "e091_")):
        m = re.search(r"(box\d+).*_(\d{3})_(p[12])$", case)
        if m:
            return f"{m.group(1)}_{m.group(2)}_{m.group(3)}"
    return case


def default_mask(short: str) -> str:
    return rel(CONTACT_MASK_ROOT / short / "raw_contact_mask_3cm.npz")


def default_kin_ref(scene_xml: str) -> str:
    if not scene_xml:
        return ""
    return rel(repo_path(scene_xml).parent / "0" / "trajectory_kinematic.npz")


def e147_qpos_path(variant: str) -> str:
    return rel(
        REPO
        / "workspace/core4d/results/E147/rubber_hand_collision/cem/full"
        / f"{variant}_outdir_full/trajectory_mjwp_act.npz"
    )


def e147_video_path(variant: str) -> str:
    return rel(REPO / "workspace/core4d/results/E147/rubber_hand_collision/cem/full" / f"{variant}_full.mp4")


def method_for(exp_id: str, row: dict[str, str]) -> str:
    if exp_id == "E147":
        return "spider-rubberhand"
    if exp_id == "E148":
        return "spider-rubberhand"
    if exp_id == "E150":
        return row.get("anchor_variant") or row.get("method") or "anchor"
    return row.get("method") or row.get("method_group") or row.get("anchor_variant") or "method"


def canonical_method_id(exp_id: str, method: str) -> str:
    if exp_id == "E147":
        return "E147_spider_rubberhand"
    return f"{exp_id}_{safe_id(method)}"


def qpos_for(exp_id: str, row: dict[str, str]) -> str:
    if exp_id == "E147":
        return e147_qpos_path(row["variant"])
    for key in ("outdir_npz", "rubber_outdir_npz", "qpos_path", "result_npz"):
        if row.get(key):
            return row[key]
    return ""


def video_for(exp_id: str, row: dict[str, str]) -> str:
    if exp_id == "E147":
        return e147_video_path(row["variant"])
    for key in ("video", "rubber_video"):
        if row.get(key):
            return row[key]
    return ""


def scene_for(row: dict[str, str]) -> str:
    for key in ("rubber_scene_act", "scene_xml", "base_scene_act"):
        if row.get(key):
            return row[key]
    return ""


def baseline_map() -> dict[str, dict[str, Any]]:
    rows = read_tsv(REPO / "workspace/core4d/scripts/experiments/E147/variants.tsv")
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        short = short_case_id(row)
        scene = scene_for(row)
        out[short] = {
            "baseline_source_exp_id": "E147",
            "baseline_variant": row["variant"],
            "baseline_qpos_path": e147_qpos_path(row["variant"]),
            "baseline_scene_xml": scene,
            "baseline_contact_mask_path": default_mask(short),
            "baseline_kin_ref_path": row.get("trajectory") or default_kin_ref(scene),
        }
    return out


def status_for(row: dict[str, Any]) -> str:
    missing = []
    for label, key in [
        ("qpos", "qpos_path"),
        ("scene", "scene_xml"),
        ("contact_mask", "contact_mask_path"),
        ("kin_ref", "kin_ref_path"),
    ]:
        if not row.get(key) or not repo_path(row[key]).is_file():
            missing.append(f"missing_{label}")
    return "ok" if not missing else ";".join(missing)


def baseline_status(row: dict[str, Any]) -> str:
    if not row.get("baseline_qpos_path"):
        return "missing_e147_case"
    missing = []
    for label, key in [
        ("qpos", "baseline_qpos_path"),
        ("scene", "baseline_scene_xml"),
        ("contact_mask", "baseline_contact_mask_path"),
        ("kin_ref", "baseline_kin_ref_path"),
    ]:
        if not row.get(key) or not repo_path(row[key]).is_file():
            missing.append(f"missing_e147_{label}")
    return "ok" if not missing else ";".join(missing)


def normalized_row(exp_id: str, row: dict[str, str], base: dict[str, dict[str, Any]], notes: str = "") -> dict[str, Any]:
    short = short_case_id(row)
    method = method_for(exp_id, row)
    scene = scene_for(row)
    item: dict[str, Any] = {
        "manifest_exp_id": exp_id,
        "source_exp_id": row.get("source_exp") or row.get("reuse_source_exp") or exp_id,
        "source_method_id": method,
        "canonical_method_id": canonical_method_id(exp_id, method),
        "method_display": row.get("method_display") or method,
        "case_id": row.get("case_id") or short,
        "short_case_id": short,
        "variant": row.get("variant") or f"{exp_id}_{short}_{safe_id(method)}",
        "qpos_path": qpos_for(exp_id, row),
        "scene_xml": scene,
        "video_path": video_for(exp_id, row),
        "contact_mask_path": row.get("mask_path") or default_mask(short),
        "kin_ref_path": row.get("trajectory") or default_kin_ref(scene),
        "person_idx": row.get("person_idx") or ("0" if "_p1" in short or "person1" in short else "1"),
        "hand_collision_variant_id": row.get("hand_collision_variant_id") or "rubber_hull",
        "object_key": row.get("object_key") or short.split("_")[0],
        "object_category": row.get("object_category") or ("box" if short.startswith("box") else ""),
        "expected_quality": row.get("expected_quality", ""),
        "baseline_method_id": "E147_spider_rubberhand",
        "notes": notes,
    }
    item.update(
        base.get(
            short,
            {
                "baseline_source_exp_id": "",
                "baseline_variant": "",
                "baseline_qpos_path": "",
                "baseline_scene_xml": "",
                "baseline_contact_mask_path": "",
                "baseline_kin_ref_path": "",
            },
        )
    )
    item["baseline_status"] = baseline_status(item)
    item["row_status"] = status_for(item)
    return item


def load_variant_rows(base: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for exp_id in EXPERIMENT_VARIANTS:
        path = REPO / f"workspace/core4d/scripts/experiments/{exp_id}/variants.tsv"
        if not path.is_file():
            continue
        for row in read_tsv(path):
            out.append(normalized_row(exp_id, row, base))
    return out


def e153_rows(base: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for short, meta in E153_CASES.items():
        for method, qpos_key in [("baseline", "baseline"), ("b1", "b1")]:
            row = {
                "variant": f"E153_{short}_{method}",
                "case_id": meta["case_id"],
                "short_case_id": short,
                "method": method,
                "source_exp": "E148" if method == "baseline" and short != "box023_person2" else ("E147" if method == "baseline" else "E151"),
                "outdir_npz": meta[qpos_key],
                "rubber_scene_act": meta["scene"],
                "trajectory": meta["kin_ref"],
                "mask_path": default_mask(short),
                "person_idx": meta["person_idx"],
                "object_key": meta["object_key"],
                "object_category": "box",
            }
            out.append(normalized_row("E153", row, base, notes="reconstructed_from_E153_evaluator"))
        for tag, _, _ in E153_GRID:
            variant = f"E153_{short}_gateA_b1_{tag}"
            qpos = (
                REPO
                / "workspace/core4d/results/E153/gate_threshold_sweep/cem/full"
                / f"{variant}_outdir_full/trajectory_mjwp_act.npz"
            )
            row = {
                "variant": variant,
                "case_id": meta["case_id"],
                "short_case_id": short,
                "method": f"gateA_b1_{tag}",
                "source_exp": "E153",
                "outdir_npz": rel(qpos),
                "video": rel(qpos.parent.parent / f"{variant}_full.mp4"),
                "rubber_scene_act": meta["scene"],
                "trajectory": meta["kin_ref"],
                "mask_path": default_mask(short),
                "person_idx": meta["person_idx"],
                "object_key": meta["object_key"],
                "object_category": "box",
            }
            out.append(normalized_row("E153", row, base, notes="reconstructed_from_E153_grid"))
    return out


def e155_rows(base: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for short, meta in E153_CASES.items():
        for method in E155_METHODS:
            variant = f"E155_{short}_{method}"
            qpos = REPO / "workspace/core4d/results/E155/cem/full" / f"{variant}_outdir_full/trajectory_mjwp_act.npz"
            row = {
                "variant": variant,
                "case_id": meta["case_id"],
                "short_case_id": short,
                "method": method,
                "source_exp": "E155",
                "outdir_npz": rel(qpos),
                "video": rel(qpos.parent.parent / f"{variant}_full.mp4"),
                "rubber_scene_act": meta["scene"],
                "trajectory": meta["kin_ref"],
                "mask_path": default_mask(short),
                "person_idx": meta["person_idx"],
                "object_key": meta["object_key"],
                "object_category": "box",
            }
            out.append(normalized_row("E155", row, base, notes="reconstructed_from_E155_results"))
    return out


def main() -> None:
    base = baseline_map()
    rows = load_variant_rows(base) + e153_rows(base) + e155_rows(base)

    seen: set[tuple[str, str, str, str, str]] = set()
    unique: list[dict[str, Any]] = []
    for row in rows:
        key = (
            row["manifest_exp_id"],
            row["canonical_method_id"],
            row["short_case_id"],
            row["qpos_path"],
            row["scene_xml"],
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)

    for idx, row in enumerate(unique, start=1):
        row["ordinal"] = idx
        row["row_id"] = f"E162_{idx:04d}_{row['canonical_method_id']}_{row['short_case_id']}"

    write_tsv(SCRIPT_ROOT / "variants.tsv", unique, FIELDS)
    by_status: dict[str, int] = {}
    by_baseline_status: dict[str, int] = {}
    by_method: dict[str, int] = {}
    for row in unique:
        by_status[row["row_status"]] = by_status.get(row["row_status"], 0) + 1
        by_baseline_status[row["baseline_status"]] = by_baseline_status.get(row["baseline_status"], 0) + 1
        by_method[row["canonical_method_id"]] = by_method.get(row["canonical_method_id"], 0) + 1
    write_json(
        RESULT_ROOT / "manifest/e162_manifest_summary.json",
        {
            "git_head": git_head(),
            "rows": len(unique),
            "row_status": by_status,
            "baseline_status": by_baseline_status,
            "canonical_method_rows": by_method,
            "baseline_cases": sorted(base),
            "view_only_experiments": {
                "E149": "E148 clean benchmark view only; no independent trajectory rows.",
                "E154": "metric standard only",
                "E157": "handoff/export only",
            },
        },
    )
    print(
        f"E162 manifest rows={len(unique)} ok={by_status.get('ok', 0)} "
        f"baseline_ok={by_baseline_status.get('ok', 0)} output={SCRIPT_ROOT / 'variants.tsv'}"
    )
    if any(status.startswith("missing_qpos") or status.startswith("missing_scene") for status in by_status):
        print("row_status:", by_status)
    print("baseline_status:", by_baseline_status)


if __name__ == "__main__":
    main()
