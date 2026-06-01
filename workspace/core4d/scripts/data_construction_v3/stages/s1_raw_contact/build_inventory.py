#!/usr/bin/env python3
"""Build S1 raw CORE4D inventory without reading legacy data-construction outputs."""

from __future__ import annotations

import argparse
import math
import os
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import sys

SCRIPT_ROOT = next(parent for parent in Path(__file__).resolve().parents if parent.name == "data_construction_v3")
for _path in (SCRIPT_ROOT / "lib", SCRIPT_ROOT / "state", SCRIPT_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
from typing import Any

import numpy as np

from common import SCHEMA_VERSION, find_spider_repo, json_dumps, timestamp, write_json, write_tsv
from geometry import aabb_extents, load_obj_vertices


MAIN_ACTIONS = {"move2_obs0", "move1_obs0"}
MOVE_ACTION_RE = re.compile(r"^move[12]_obs[013]$")
HIGH_RISK_ACTION_PREFIXES = ("pass", "raise", "rot", "strike", "join", "leave")
PERSONS = ("person1", "person2")


@dataclass(frozen=True)
class ObjectInfo:
    name: str
    category: str
    mesh_rel: str
    extent: tuple[float, float, float]
    sorted_extent: tuple[float, float, float]
    volume: float
    min_extent: float
    mid_extent: float
    max_extent: float


def canonical_name(name: str) -> str:
    return name.strip().lower()


def read_json(path: Path) -> Any:
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def build_object_index(core4d_root: Path) -> dict[str, ObjectInfo]:
    object_root = core4d_root / "object_models"
    index: dict[str, ObjectInfo] = {}
    for mesh in sorted(object_root.glob("*/*_m.obj")):
        extent_arr = aabb_extents(load_obj_vertices(mesh))
        sorted_extent = tuple(float(x) for x in sorted(extent_arr))
        name = mesh.stem.removesuffix("_m")
        key = canonical_name(name)
        index[key] = ObjectInfo(
            name=name,
            category=mesh.parent.name,
            mesh_rel=str(mesh.relative_to(object_root)),
            extent=tuple(float(x) for x in extent_arr),
            sorted_extent=sorted_extent,
            volume=float(np.prod(extent_arr)),
            min_extent=sorted_extent[0],
            mid_extent=sorted_extent[1],
            max_extent=sorted_extent[2],
        )
    return index


def require_refs(object_index: dict[str, ObjectInfo]) -> tuple[ObjectInfo, ObjectInfo, ObjectInfo]:
    missing = [key for key in ("bucket001", "box023", "box025") if key not in object_index]
    if missing:
        raise KeyError(f"missing size reference objects: {missing}")
    return object_index["bucket001"], object_index["box023"], object_index["box025"]


def object_name_from_metadata(metadata: dict[str, Any]) -> str:
    if metadata.get("obj_name"):
        return str(metadata["obj_name"])
    if metadata.get("object_name"):
        return str(metadata["object_name"])
    model_path = str(metadata.get("obj_model_path", ""))
    return Path(model_path).parent.name if model_path else ""


def object_key_from_metadata(metadata: dict[str, Any]) -> str:
    name = object_name_from_metadata(metadata)
    if name:
        return canonical_name(name)
    model_path = str(metadata.get("obj_model_path", ""))
    return canonical_name(Path(model_path).stem.removesuffix("_m")) if model_path else ""


def load_motion_stats(seq_dir: Path) -> dict[str, float | int | bool]:
    poses_path = seq_dir / "smooth_objposes.npy"
    if not poses_path.is_file():
        return {
            "motion_ok": False,
            "raw_frames": 0,
            "object_xy_displacement_m": math.nan,
            "object_xy_path_m": math.nan,
            "object_z_min_m": math.nan,
            "object_z_max_m": math.nan,
            "object_z_range_m": math.nan,
        }
    poses = np.load(poses_path)
    pos = np.asarray(poses[:, :3, 3], dtype=np.float64)
    xy_steps = np.linalg.norm(np.diff(pos[:, :2], axis=0), axis=1) if len(pos) > 1 else np.asarray([0.0])
    z_min = float(np.min(pos[:, 2]))
    z_max = float(np.max(pos[:, 2]))
    return {
        "motion_ok": True,
        "raw_frames": int(pos.shape[0]),
        "object_xy_displacement_m": float(np.linalg.norm(pos[-1, :2] - pos[0, :2])),
        "object_xy_path_m": float(np.sum(xy_steps)),
        "object_z_min_m": z_min,
        "object_z_max_m": z_max,
        "object_z_range_m": z_max - z_min,
    }


def classify_action(action: str) -> str:
    if action in MAIN_ACTIONS:
        return "main_move_obs0"
    if MOVE_ACTION_RE.match(action):
        return "move_obstacle_review"
    if action.startswith("move"):
        return "move_other_review"
    if action.startswith(HIGH_RISK_ACTION_PREFIXES):
        return "high_risk_nonmove"
    return "other_action"


def hard_reject_reason(key: str, obj: ObjectInfo | None, bucket001: ObjectInfo, box025: ObjectInfo) -> str:
    if obj is None:
        return "reject_missing_object_mesh"
    if key == "box025" or obj.volume >= box025.volume:
        return "reject_too_large_box025_or_larger"
    if obj.mid_extent >= box025.mid_extent and obj.max_extent >= box025.max_extent:
        return "reject_too_large_box025_or_larger"
    if key == "bucket001" or obj.volume <= bucket001.volume:
        return "reject_too_small_bucket001_or_smaller"
    if obj.min_extent <= bucket001.min_extent and obj.mid_extent <= bucket001.mid_extent and obj.max_extent <= bucket001.max_extent:
        return "reject_too_small_bucket001_or_smaller"
    return ""


def size_band(obj: ObjectInfo | None, bucket001: ObjectInfo, box023: ObjectInfo, box025: ObjectInfo) -> str:
    if obj is None:
        return "missing_mesh"
    if hard_reject_reason(canonical_name(obj.name), obj, bucket001, box025):
        return "hard_reject_size"
    if obj.volume <= box023.volume * 1.05:
        return "small_above_bucket001_to_box023"
    if obj.volume >= box025.volume * 0.75:
        return "near_box025_large_review"
    return "target_medium_between_box023_and_box025"


def size_fit_score(obj: ObjectInfo | None, bucket001: ObjectInfo, box023: ObjectInfo, box025: ObjectInfo) -> float:
    if obj is None or hard_reject_reason(canonical_name(obj.name), obj, bucket001, box025):
        return 0.0
    if box023.volume * 1.05 < obj.volume < box025.volume * 0.75:
        return 30.0
    if box023.volume < obj.volume < box025.volume:
        return 20.0
    if bucket001.volume < obj.volume <= box023.volume:
        return 8.0
    return 4.0


def action_score(action_family: str) -> float:
    if action_family == "main_move_obs0":
        return 25.0
    if action_family == "move_obstacle_review":
        return 10.0
    if action_family == "move_other_review":
        return 6.0
    return 0.0


def motion_quality(stats: dict[str, float | int | bool]) -> str:
    if not stats.get("motion_ok"):
        return "missing_motion"
    frames = int(stats["raw_frames"])
    xy_path = float(stats["object_xy_path_m"])
    xy_disp = float(stats["object_xy_displacement_m"])
    z_range = float(stats["object_z_range_m"])
    if frames >= 60 and (xy_path >= 0.35 or xy_disp >= 0.15 or z_range >= 0.08):
        return "motion_pass"
    if frames >= 40 and (xy_path >= 0.15 or xy_disp >= 0.08 or z_range >= 0.04):
        return "motion_review"
    return "motion_weak"


def motion_score(stats: dict[str, float | int | bool]) -> float:
    quality = motion_quality(stats)
    if quality == "motion_pass":
        score = 18.0
        frames = int(stats["raw_frames"])
        if 80 <= frames <= 320:
            score += 5.0
        if float(stats["object_xy_path_m"]) >= 1.0:
            score += 5.0
        if float(stats["object_z_range_m"]) >= 0.12:
            score += 2.0
        return score
    if quality == "motion_review":
        return 8.0
    return 0.0


def review_flags(obj: ObjectInfo | None, band: str, action_family: str, motion_value: str) -> list[str]:
    flags: list[str] = []
    if obj is None:
        return ["missing_object_mesh"]
    if obj.category != "box":
        flags.append("manual_template_review_required_non_box")
    if obj.category == "bucket":
        flags.append("bucket_surface_semantics_review")
    if obj.category in {"desk", "board"}:
        flags.append("desk_board_surface_semantics_review")
    if band == "near_box025_large_review":
        flags.append("large_size_near_box025_review")
    if band == "small_above_bucket001_to_box023":
        flags.append("small_size_collaboration_value_review")
    if action_family in {"move_obstacle_review", "move_other_review"}:
        flags.append(action_family)
    if action_family == "high_risk_nonmove":
        flags.append("high_risk_nonmove_action")
    if motion_value != "motion_pass":
        flags.append(motion_value)
    return flags


def stage0_decision(hard_reject: str, has_pose: bool, action_family: str, motion_value: str, flags: list[str]) -> str:
    if hard_reject:
        return hard_reject
    if not has_pose:
        return "reject_missing_person_pose"
    if action_family in {"high_risk_nonmove", "other_action"}:
        return "reject_action_not_first_line"
    if motion_value in {"motion_weak", "missing_motion"}:
        return "reject_motion_weak"
    if flags:
        return "inventory_review_to_raw_contact"
    return "inventory_pass_to_raw_contact"


def decision_group(decision: str) -> str:
    if decision == "inventory_pass_to_raw_contact":
        return "进入下一轮"
    if decision.startswith("reject_"):
        return "抛弃/跳过"
    return "边界/review"


def bool_text(value: bool) -> str:
    return str(bool(value))


def source_template_path(source_scene_root: Path, object_key: str, person: str) -> Path:
    return source_scene_root / f"{object_key}_{person}" / "scene.xml"


def build_rows(core4d_root: Path, spider_repo: Path, source_scene_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    object_index = build_object_index(core4d_root)
    bucket001, box023, box025 = require_refs(object_index)
    labels_path = core4d_root / "action_labels.json"
    labels = read_json(labels_path) if labels_path.is_file() else {}
    rows: list[dict[str, Any]] = []

    for seq_dir in sorted((core4d_root / "human_object_motions").glob("*/*")):
        if not seq_dir.is_dir():
            continue
        metadata_path = seq_dir / "object_metadata.json"
        if not metadata_path.is_file():
            continue
        date = seq_dir.parent.name
        seq = seq_dir.name
        sequence = f"{date}/{seq}"
        metadata = read_json(metadata_path)
        action = str(labels.get(sequence, "missing_action"))
        action_family = classify_action(action)
        object_name = object_name_from_metadata(metadata)
        object_key = object_key_from_metadata(metadata)
        obj = object_index.get(object_key)
        motion_stats = load_motion_stats(seq_dir)
        motion_value = motion_quality(motion_stats)
        hard_reject = hard_reject_reason(object_key, obj, bucket001, box025)
        band = size_band(obj, bucket001, box023, box025)

        for person in PERSONS:
            has_pose = (seq_dir / f"{person}_poses.npz").is_file()
            template = source_template_path(source_scene_root, object_key, person)
            has_template = template.is_file()
            flags = review_flags(obj, band, action_family, motion_value)
            decision = stage0_decision(hard_reject, has_pose, action_family, motion_value, flags)
            score = 0.0
            if decision_group(decision) != "抛弃/跳过":
                score = size_fit_score(obj, bucket001, box023, box025) + action_score(action_family) + motion_score(motion_stats)
                score += 5.0 if has_pose else 0.0
                if decision_group(decision) == "边界/review":
                    score *= 0.75
            route = ["route_has_template" if has_template else "route_needs_template", "route_needs_omniretarget_spider"]
            rows.append(
                {
                    "case_id": f"{object_key}_{date}_{seq}_{'p1' if person == 'person1' else 'p2'}",
                    "stage": "S1_inventory",
                    "date": date,
                    "seq": seq,
                    "sequence": sequence,
                    "person": person,
                    "person_idx": 0 if person == "person1" else 1,
                    "object_name": object_name,
                    "object_key": object_key,
                    "object_category": obj.category if obj else "unknown",
                    "action": action,
                    "action_family": action_family,
                    "inventory_rank_score": round(float(score), 3),
                    "inventory_decision": decision,
                    "decision_group": decision_group(decision),
                    "review_flags": ",".join(flags),
                    "hard_reject_reason": hard_reject,
                    "route_tags": ",".join(route),
                    "has_person_pose": bool_text(has_pose),
                    "source_scene_template_status": "clean_or_existing" if has_template else "backlog",
                    "source_scene_xml": str(template if has_template else ""),
                    "object_mesh_rel": obj.mesh_rel if obj else "",
                    "size_band": band,
                    "size_vs_bucket001_volume_ratio": round(float(obj.volume / bucket001.volume), 6) if obj else "",
                    "size_vs_box023_volume_ratio": round(float(obj.volume / box023.volume), 6) if obj else "",
                    "size_vs_box025_volume_ratio": round(float(obj.volume / box025.volume), 6) if obj else "",
                    "extent_x_m": obj.extent[0] if obj else "",
                    "extent_y_m": obj.extent[1] if obj else "",
                    "extent_z_m": obj.extent[2] if obj else "",
                    "extent_min_m": obj.min_extent if obj else "",
                    "extent_mid_m": obj.mid_extent if obj else "",
                    "extent_max_m": obj.max_extent if obj else "",
                    "aabb_volume_m3": obj.volume if obj else "",
                    **motion_stats,
                    "motion_quality": motion_value,
                    "schema_version": SCHEMA_VERSION,
                    "updated_at": timestamp(),
                }
            )

    order = {"进入下一轮": 0, "边界/review": 1, "抛弃/跳过": 2}
    rows.sort(key=lambda r: (order[str(r["decision_group"])], -float(r["inventory_rank_score"]), str(r["date"]), str(r["seq"]), str(r["person"])))
    refs = {
        "bucket001": object_summary(bucket001),
        "box023": object_summary(box023),
        "box025": object_summary(box025),
        "spider_repo": str(spider_repo),
        "source_scene_root": str(source_scene_root),
    }
    return rows, refs


def object_summary(obj: ObjectInfo) -> dict[str, Any]:
    return {
        "name": obj.name,
        "category": obj.category,
        "mesh_rel": obj.mesh_rel,
        "extent_m": list(obj.extent),
        "sorted_extent_m": list(obj.sorted_extent),
        "volume_m3": obj.volume,
    }


def sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: sanitize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def summarize(rows: list[dict[str, Any]], refs: dict[str, Any]) -> dict[str, Any]:
    return {
        "stage": "S1_inventory",
        "created_at": timestamp(),
        "schema_version": SCHEMA_VERSION,
        "total_case_person_rows": len(rows),
        "unique_sequences": len({row["sequence"] for row in rows}),
        "unique_objects": len({row["object_key"] for row in rows}),
        "decision_group_counts": dict(Counter(row["decision_group"] for row in rows)),
        "decision_counts": dict(Counter(row["inventory_decision"] for row in rows)),
        "object_counts": dict(Counter(row["object_key"] for row in rows)),
        "size_band_counts": dict(Counter(row["size_band"] for row in rows)),
        "route_counts": dict(Counter(tag for row in rows for tag in str(row["route_tags"]).split(",") if tag)),
        "reference_objects": refs,
    }


def markdown_summary(summary: dict[str, Any]) -> str:
    lines = [
        "# S1 raw inventory summary",
        "",
        f"- created_at: `{summary['created_at']}`",
        f"- total case-person rows: `{summary['total_case_person_rows']}`",
        f"- unique sequences: `{summary['unique_sequences']}`",
        f"- unique objects: `{summary['unique_objects']}`",
        "",
        "## decision counts",
        "",
        "| decision | count |",
        "|---|---:|",
    ]
    for key, count in summary["decision_counts"].items():
        lines.append(f"| `{key}` | {count} |")
    lines.extend(["", "## route counts", "", "| route | count |", "|---|---:|"])
    for key, count in summary["route_counts"].items():
        lines.append(f"| `{key}` | {count} |")
    lines.extend(
        [
            "",
            "说明：inventory 只做 raw 数据存在性、尺寸、动作、motion、source template backlog 标记；source scene missing 不作为数据 reject。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--core4d-raw-root", type=Path, default=None)
    parser.add_argument("--spider-repo", type=Path, default=None)
    parser.add_argument("--source-scene-root", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    raw_root = args.core4d_raw_root or (Path(os.environ["CORE4D_RAW_ROOT"]) if os.environ.get("CORE4D_RAW_ROOT") else None)
    if raw_root is None:
        raise SystemExit("missing --core4d-raw-root or CORE4D_RAW_ROOT")
    raw_root = raw_root.expanduser().resolve()
    spider_repo = (args.spider_repo or find_spider_repo()).resolve()
    source_scene_root = args.source_scene_root or (
        spider_repo / "example_datasets/processed/core4d/unitree_g1/humanoid_object"
    )
    rows, refs = build_rows(raw_root, spider_repo, source_scene_root.resolve())
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else []
    write_tsv(out_dir / "inventory.tsv", rows, fields)
    write_json(out_dir / "inventory.json", sanitize(rows))
    summary = summarize(rows, refs)
    write_json(out_dir / "inventory_summary.json", sanitize(summary))
    (out_dir / "inventory_summary.md").write_text(markdown_summary(summary), encoding="utf-8")
    print(json_dumps({"rows": len(rows), "out_dir": str(out_dir), "decision_counts": summary["decision_counts"]}))


if __name__ == "__main__":
    main()
