#!/usr/bin/env python3
"""Run S1 raw hand-object contact proxy with separate 3cm/5cm candidate outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
import sys

SCRIPT_ROOT = next(parent for parent in Path(__file__).resolve().parents if parent.name == "data_construction_v3")
for _path in (SCRIPT_ROOT / "lib", SCRIPT_ROOT / "state", SCRIPT_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
from typing import Any

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from common import SCHEMA_VERSION, json_dumps, read_tsv, timestamp, write_json, write_tsv


PERSONS = ("person1", "person2")
HANDS = ("left", "right")
DEFAULT_THRESHOLDS_M = (0.03, 0.05)
HAND_RANGES = {
    "left": np.arange(4700, 5500, dtype=np.int64),
    "right": np.arange(7500, 8150, dtype=np.int64),
}
FINGERTIP_IDS = {
    "left": np.array([5361, 4933, 5058, 5169, 5286], dtype=np.int64),
    "right": np.array([8079, 7669, 7794, 7905, 8022], dtype=np.int64),
}


@dataclass(frozen=True)
class SequenceCandidate:
    sequence: str
    date: str
    seq: str
    object_name: str
    object_key: str
    object_category: str
    action: str
    object_mesh_rel: str
    selected_persons: tuple[str, ...]
    inventory_rows: tuple[dict[str, str], ...]


def threshold_label(threshold_m: float) -> str:
    return f"{int(round(threshold_m * 100))}cm"


def threshold_field(base: str, threshold_m: float) -> str:
    return f"{base}_{threshold_label(threshold_m)}"


def as_float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, ""))
    except (TypeError, ValueError):
        return default


def inventory_sort_key(row: dict[str, str]) -> tuple[float, str, int]:
    person_rank = PERSONS.index(row["person"]) if row.get("person") in PERSONS else 99
    return (-as_float(row, "inventory_rank_score"), row.get("sequence", ""), person_rank)


def row_matches_queue(row: dict[str, str], queue: str) -> bool:
    group = row.get("decision_group", "")
    object_category = row.get("object_category", "")
    action_family = row.get("action_family", "")
    object_key = row.get("object_key", "")
    size_band = row.get("size_band", "")
    if queue == "clean":
        return group == "进入下一轮"
    if queue == "stage1-input":
        return group in {"进入下一轮", "边界/review"}
    if queue == "selected-medium-box":
        return object_category == "box" and size_band in {
            "target_medium_between_box023_and_box025",
            "near_box025_large_review",
        }
    if queue == "box-object":
        return object_category == "box"
    if queue == "object-key":
        raise ValueError("queue=object-key requires --object-keys and is handled separately")
    if queue == "review-main-box":
        return group == "边界/review" and object_category == "box" and action_family == "main_move_obs0"
    if queue == "rebuilt-box-family":
        return object_key in {"box021", "box022", "box026"}
    raise ValueError(f"unknown queue: {queue}")


def select_inventory_rows(
    rows: list[dict[str, str]],
    queue: str,
    object_keys: set[str],
    max_case_persons: int | None,
    max_sequences: int | None,
) -> list[dict[str, str]]:
    if queue == "object-key":
        selected = [row for row in rows if row.get("object_key") in object_keys]
    else:
        selected = [row for row in rows if row_matches_queue(row, queue)]
        if object_keys:
            selected = [row for row in selected if row.get("object_key") in object_keys]
    selected.sort(key=inventory_sort_key)
    if max_case_persons is not None:
        selected = selected[:max_case_persons]
    if max_sequences is not None:
        out: list[dict[str, str]] = []
        seen: set[str] = set()
        for row in selected:
            seq = row["sequence"]
            if seq not in seen and len(seen) >= max_sequences:
                continue
            seen.add(seq)
            out.append(row)
        selected = out
    return selected


def group_sequence_candidates(rows: list[dict[str, str]]) -> list[SequenceCandidate]:
    by_sequence: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_sequence[row["sequence"]].append(row)
    candidates: list[SequenceCandidate] = []
    for sequence, seq_rows in sorted(by_sequence.items(), key=lambda item: inventory_sort_key(item[1][0])):
        seq_rows = sorted(seq_rows, key=inventory_sort_key)
        first = seq_rows[0]
        selected_persons = tuple(person for person in PERSONS if any(row.get("person") == person for row in seq_rows))
        candidates.append(
            SequenceCandidate(
                sequence=sequence,
                date=first["date"],
                seq=first["seq"],
                object_name=first["object_name"],
                object_key=first["object_key"],
                object_category=first["object_category"],
                action=first["action"],
                object_mesh_rel=first["object_mesh_rel"],
                selected_persons=selected_persons,
                inventory_rows=tuple(seq_rows),
            )
        )
    return candidates


def row_for_person(candidate: SequenceCandidate, person: str) -> dict[str, str]:
    for row in candidate.inventory_rows:
        if row.get("person") == person:
            return row
    return candidate.inventory_rows[0]


def load_person_vertices(seq_dir: Path, person: str) -> np.ndarray:
    path = seq_dir / f"{person}_poses.npz"
    data = np.load(path, allow_pickle=True)["arr_0"].item()
    vertices = np.asarray(data["vertices"], dtype=np.float32)
    if vertices.ndim != 3 or vertices.shape[1] <= max(FINGERTIP_IDS["right"]):
        raise ValueError(f"{path} has unexpected vertices shape {vertices.shape}")
    return vertices


def active_motion_mask(object_pos: np.ndarray) -> np.ndarray:
    n = object_pos.shape[0]
    if n <= 2:
        return np.ones(n, dtype=bool)
    xy_step = np.linalg.norm(np.diff(object_pos[:, :2], axis=0), axis=1)
    total = float(xy_step.sum())
    if total < 0.10:
        return np.ones(n, dtype=bool)
    cumulative = np.concatenate([[0.0], np.cumsum(xy_step)])
    mask = (cumulative >= 0.10 * total) & (cumulative <= 0.90 * total)
    if mask.sum() < max(5, int(0.10 * n)):
        return np.ones(n, dtype=bool)
    return mask


def compute_contact_proxy(
    core4d_root: Path,
    candidate: SequenceCandidate,
    thresholds_m: tuple[float, ...],
    sample_count: int,
    seed: int,
) -> dict[str, Any]:
    seq_dir = core4d_root / "human_object_motions" / candidate.sequence
    mesh_path = core4d_root / "object_models" / candidate.object_mesh_rel
    poses = np.load(seq_dir / "smooth_objposes.npy")
    object_pos = np.asarray(poses[:, :3, 3], dtype=np.float64)
    n_frames = int(poses.shape[0])
    people = {person: load_person_vertices(seq_dir, person) for person in PERSONS}
    for person, vertices in people.items():
        if vertices.shape[0] != n_frames:
            raise ValueError(f"{candidate.sequence} {person} frame mismatch: vertices={vertices.shape[0]} poses={n_frames}")

    mesh = trimesh.load(mesh_path, process=False)
    rng = np.random.default_rng(seed)
    surface_points, _ = trimesh.sample.sample_surface(mesh, sample_count, seed=rng)
    surface_points = np.asarray(surface_points, dtype=np.float64)

    min_dist = np.full((n_frames, 2, 2), np.nan, dtype=np.float32)
    tip_min_dist = np.full((n_frames, 2, 2), np.nan, dtype=np.float32)
    vertex_count = np.zeros((n_frames, 2, 2, len(thresholds_m)), dtype=np.int32)
    for frame in range(n_frames):
        obj_world = surface_points @ poses[frame, :3, :3].T + poses[frame, :3, 3]
        tree = cKDTree(obj_world)
        for pi, person in enumerate(PERSONS):
            vertices = people[person][frame]
            for hi, hand in enumerate(HANDS):
                hand_dists, _ = tree.query(vertices[HAND_RANGES[hand]], k=1)
                tip_dists, _ = tree.query(vertices[FINGERTIP_IDS[hand]], k=1)
                min_dist[frame, pi, hi] = float(hand_dists.min())
                tip_min_dist[frame, pi, hi] = float(tip_dists.min())
                for ti, threshold in enumerate(thresholds_m):
                    vertex_count[frame, pi, hi, ti] = int((hand_dists < threshold).sum())

    return {
        "candidate": candidate,
        "seq_dir": seq_dir,
        "mesh_path": mesh_path,
        "object_pos": object_pos,
        "object_xy_path_m": float(np.linalg.norm(np.diff(object_pos[:, :2], axis=0), axis=1).sum()) if n_frames > 1 else 0.0,
        "object_xy_displacement_m": float(np.linalg.norm(object_pos[-1, :2] - object_pos[0, :2])),
        "object_z_min_m": float(object_pos[:, 2].min()),
        "object_z_max_m": float(object_pos[:, 2].max()),
        "active_mask": active_motion_mask(object_pos),
        "min_dist_m": min_dist,
        "tip_min_dist_m": tip_min_dist,
        "vertex_count": vertex_count,
        "masks": vertex_count > 0,
        "thresholds_m": np.asarray(thresholds_m, dtype=np.float32),
        "sample_count": sample_count,
        "seed": seed,
    }


def fraction(mask: np.ndarray) -> float:
    return float(np.mean(mask)) if mask.size else 0.0


def longest_run_fraction(mask: np.ndarray) -> float:
    best = 0
    cur = 0
    for value in mask.astype(bool):
        if value:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return float(best / mask.size) if mask.size else 0.0


def metrics_for(result: dict[str, Any], person: str, threshold_index: int) -> dict[str, float]:
    pi = PERSONS.index(person)
    partner_i = 1 - pi
    masks = result["masks"][:, :, :, threshold_index]
    active = result["active_mask"]
    target_left = masks[:, pi, 0]
    target_right = masks[:, pi, 1]
    target_any = target_left | target_right
    target_both = target_left & target_right
    partner_left = masks[:, partner_i, 0]
    partner_right = masks[:, partner_i, 1]
    partner_any = partner_left | partner_right
    partner_both = partner_left & partner_right
    return {
        "target_left_active_frac": fraction(target_left[active]),
        "target_right_active_frac": fraction(target_right[active]),
        "target_any_active_frac": fraction(target_any[active]),
        "target_both_active_frac": fraction(target_both[active]),
        "target_both_longest_run_active_frac": longest_run_fraction(target_both[active]),
        "partner_any_active_frac": fraction(partner_any[active]),
        "partner_both_active_frac": fraction(partner_both[active]),
    }


def route_after_raw_contact(decision: str, inventory_row: dict[str, str]) -> str:
    if decision == "raw_contact_fail":
        return "hold_raw_contact_failed"
    if decision == "raw_contact_error":
        return "hold_raw_contact_error"
    if "route_needs_template" in inventory_row.get("route_tags", ""):
        return "template_then_omniretarget_spider"
    return "omniretarget_spider_production"


def decision_group(decision: str) -> str:
    if decision == "raw_contact_pass":
        return "进入下一轮"
    if decision == "raw_contact_review":
        return "边界/review"
    return "抛弃/跳过"


def score_case_person(result: dict[str, Any], person: str, threshold_m: float, threshold_index: int) -> dict[str, Any]:
    candidate: SequenceCandidate = result["candidate"]
    inv = row_for_person(candidate, person)
    current = metrics_for(result, person, threshold_index)
    target_left = current["target_left_active_frac"]
    target_right = current["target_right_active_frac"]
    target_any = current["target_any_active_frac"]
    target_both = current["target_both_active_frac"]
    partner_any = current["partner_any_active_frac"]
    balanced = min(target_left, target_right)
    longest = current["target_both_longest_run_active_frac"]
    score = 35.0 * target_both + 20.0 * balanced + 20.0 * target_any + 15.0 * partner_any + 10.0 * longest
    if candidate.object_category != "box":
        score -= 8.0
    if target_any > 0.75 and target_both < 0.20:
        score -= 8.0
    if target_both >= 0.25 and balanced >= 0.35 and partner_any >= 0.25:
        decision = "raw_contact_pass"
    elif target_any >= 0.40 and balanced >= 0.20 and partner_any >= 0.15:
        decision = "raw_contact_review"
    else:
        decision = "raw_contact_fail"

    notes: list[str] = []
    if candidate.object_category != "box":
        notes.append("non_box_requires_manual_template_review")
    if target_both < 0.15:
        notes.append("weak_two_hand_overlap")
    if balanced < 0.20:
        notes.append("unbalanced_left_right_contact")
    if partner_any < 0.15:
        notes.append("weak_partner_support_proxy")

    threshold_metrics: dict[str, float] = {}
    thresholds = tuple(float(x) for x in result["thresholds_m"])
    for i, threshold in enumerate(thresholds):
        for key, value in metrics_for(result, person, i).items():
            threshold_metrics[threshold_field(key, threshold)] = round(value, 4)

    label = threshold_label(threshold_m)
    return {
        "case_id": inv.get("case_id", f"{candidate.object_key}_{candidate.date}_{candidate.seq}_{person}"),
        "stage": "S1_raw_contact",
        "contact_threshold_m": threshold_m,
        "contact_threshold_label": label,
        "sequence": candidate.sequence,
        "date": candidate.date,
        "seq": candidate.seq,
        "person": person,
        "person_idx": PERSONS.index(person),
        "object_name": candidate.object_name,
        "object_key": candidate.object_key,
        "object_category": candidate.object_category,
        "object_model_rel": inv.get("object_mesh_rel", ""),
        "action": candidate.action,
        "inventory_rank_score": inv.get("inventory_rank_score", ""),
        "inventory_decision": inv.get("inventory_decision", ""),
        "inventory_decision_group": inv.get("decision_group", ""),
        "inventory_review_flags": inv.get("review_flags", ""),
        "inventory_route_tags": inv.get("route_tags", ""),
        "source_scene_template_status": inv.get("source_scene_template_status", ""),
        "size_band": inv.get("size_band", ""),
        "raw_contact_score": round(max(0.0, score), 3),
        "raw_contact_decision": decision,
        "raw_contact_decision_group": decision_group(decision),
        "stage2_route": route_after_raw_contact(decision, inv),
        "raw_contact_notes": ",".join(notes),
        "active_frames": int(result["active_mask"].sum()),
        "raw_frames": int(result["active_mask"].size),
        "target_left_active_frac": round(target_left, 4),
        "target_right_active_frac": round(target_right, 4),
        "target_any_active_frac": round(target_any, 4),
        "target_both_active_frac": round(target_both, 4),
        "target_both_longest_run_active_frac": round(longest, 4),
        "partner_any_active_frac": round(partner_any, 4),
        "partner_both_active_frac": round(current["partner_both_active_frac"], 4),
        **threshold_metrics,
        "object_xy_displacement_m": round(float(result["object_xy_displacement_m"]), 4),
        "object_xy_path_m": round(float(result["object_xy_path_m"]), 4),
        "object_z_min_m": round(float(result["object_z_min_m"]), 4),
        "object_z_max_m": round(float(result["object_z_max_m"]), 4),
        "sample_count": int(result["sample_count"]),
        "schema_version": SCHEMA_VERSION,
        "updated_at": timestamp(),
    }


def error_rows(candidate: SequenceCandidate, message: str, threshold_m: float) -> list[dict[str, Any]]:
    label = threshold_label(threshold_m)
    rows: list[dict[str, Any]] = []
    for person in candidate.selected_persons:
        inv = row_for_person(candidate, person)
        rows.append(
            {
                "case_id": inv.get("case_id", f"{candidate.object_key}_{candidate.date}_{candidate.seq}_{person}"),
                "stage": "S1_raw_contact",
                "contact_threshold_m": threshold_m,
                "contact_threshold_label": label,
                "sequence": candidate.sequence,
                "date": candidate.date,
                "seq": candidate.seq,
                "person": person,
                "person_idx": PERSONS.index(person),
                "object_name": candidate.object_name,
                "object_key": candidate.object_key,
                "object_category": candidate.object_category,
                "object_model_rel": inv.get("object_mesh_rel", ""),
                "action": candidate.action,
                "inventory_rank_score": inv.get("inventory_rank_score", ""),
                "inventory_decision": inv.get("inventory_decision", ""),
                "inventory_decision_group": inv.get("decision_group", ""),
                "inventory_review_flags": inv.get("review_flags", ""),
                "inventory_route_tags": inv.get("route_tags", ""),
                "source_scene_template_status": inv.get("source_scene_template_status", ""),
                "size_band": inv.get("size_band", ""),
                "raw_contact_score": 0.0,
                "raw_contact_decision": "raw_contact_error",
                "raw_contact_decision_group": "抛弃/跳过",
                "stage2_route": "hold_raw_contact_error",
                "raw_contact_notes": message,
                "active_frames": 0,
                "raw_frames": 0,
                "target_left_active_frac": 0.0,
                "target_right_active_frac": 0.0,
                "target_any_active_frac": 0.0,
                "target_both_active_frac": 0.0,
                "target_both_longest_run_active_frac": 0.0,
                "partner_any_active_frac": 0.0,
                "partner_both_active_frac": 0.0,
                "object_xy_displacement_m": inv.get("object_xy_displacement_m", ""),
                "object_xy_path_m": inv.get("object_xy_path_m", ""),
                "object_z_min_m": inv.get("object_z_min_m", ""),
                "object_z_max_m": inv.get("object_z_max_m", ""),
                "sample_count": 0,
                "schema_version": SCHEMA_VERSION,
                "updated_at": timestamp(),
            }
        )
    return rows


def save_sequence_npz(result: dict[str, Any], out_dir: Path) -> None:
    candidate: SequenceCandidate = result["candidate"]
    slug = candidate.sequence.replace("/", "_")
    seq_dir = out_dir / "per_sequence" / f"{slug}_{candidate.object_key}"
    seq_dir.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, Any] = {
        "min_dist_m": result["min_dist_m"],
        "tip_min_dist_m": result["tip_min_dist_m"],
        "vertex_count": result["vertex_count"],
        "active_mask": result["active_mask"],
        "object_pos": result["object_pos"],
        "persons": np.asarray(PERSONS),
        "hands": np.asarray(HANDS),
        "thresholds_m": result["thresholds_m"],
        "sample_count": np.asarray(result["sample_count"]),
        "seed": np.asarray(result["seed"]),
        "selected_persons": np.asarray(candidate.selected_persons),
    }
    for ti, threshold in enumerate(result["thresholds_m"]):
        arrays[f"raw_contact_mask_{threshold_label(float(threshold))}"] = result["masks"][:, :, :, ti]
    np.savez_compressed(seq_dir / "raw_contact_proxy.npz", **arrays)


def sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: sanitize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize(v) for v in value]
    if isinstance(value, np.ndarray):
        return sanitize(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def write_threshold_outputs(rows: list[dict[str, Any]], out_dir: Path, threshold_m: float) -> dict[str, Any]:
    label = threshold_label(threshold_m)
    rows.sort(key=lambda row: (-float(row["raw_contact_score"]), row["sequence"], row["person"]))
    fields = list(rows[0].keys()) if rows else []
    write_tsv(out_dir / f"raw_contact_candidates_{label}.tsv", rows, fields)
    write_json(out_dir / f"raw_contact_candidates_{label}.json", sanitize(rows))
    pass_rows = [row for row in rows if row["raw_contact_decision"] == "raw_contact_pass"]
    write_tsv(out_dir / f"raw_contact_pass_{label}.tsv", pass_rows, fields)
    write_json(out_dir / f"raw_contact_pass_{label}.json", sanitize(pass_rows))
    summary = {
        "stage": "S1_raw_contact",
        "contact_threshold_m": threshold_m,
        "contact_threshold_label": label,
        "created_at": timestamp(),
        "schema_version": SCHEMA_VERSION,
        "evaluated_case_person": len(rows),
        "pass_case_person": len(pass_rows),
        "decision_counts": dict(Counter(row["raw_contact_decision"] for row in rows)),
        "decision_group_counts": dict(Counter(row["raw_contact_decision_group"] for row in rows)),
        "object_counts": dict(Counter(row["object_key"] for row in rows)),
        "route_counts": dict(Counter(row["stage2_route"] for row in rows)),
    }
    write_json(out_dir / f"raw_contact_summary_{label}.json", sanitize(summary))
    (out_dir / f"raw_contact_summary_{label}.md").write_text(markdown_summary(summary, rows), encoding="utf-8")
    return summary


def markdown_summary(summary: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    label = summary["contact_threshold_label"]
    lines = [
        f"# S1 raw contact summary @{label}",
        "",
        f"- evaluated case-person: `{summary['evaluated_case_person']}`",
        f"- pass case-person: `{summary['pass_case_person']}`",
        "- raw contact 是几何 proxy，不是人工 GT。",
        "",
        "## decision counts",
        "",
        "| decision | count |",
        "|---|---:|",
    ]
    for key, count in summary["decision_counts"].items():
        lines.append(f"| `{key}` | {count} |")
    lines.extend(["", "## top rows", "", "| score | decision | case | target both | target L/R | partner any | notes |", "|---:|---|---|---:|---|---:|---|"])
    for row in rows[:30]:
        lines.append(
            f"| {float(row['raw_contact_score']):.1f} | `{row['raw_contact_decision']}` | `{row['case_id']}` | {float(row['target_both_active_frac']):.2f} | {float(row['target_left_active_frac']):.2f}/{float(row['target_right_active_frac']):.2f} | {float(row['partner_any_active_frac']):.2f} | `{row['raw_contact_notes']}` |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--core4d-raw-root", type=Path, default=None)
    parser.add_argument("--inventory-tsv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--queue", default="clean", choices=("clean", "stage1-input", "selected-medium-box", "box-object", "object-key", "review-main-box", "rebuilt-box-family"))
    parser.add_argument("--object-keys", default="", help="comma-separated object keys for queue narrowing or queue=object-key")
    parser.add_argument("--thresholds-m", default="0.03,0.05")
    parser.add_argument("--max-case-persons", type=int, default=None)
    parser.add_argument("--max-sequences", type=int, default=None)
    parser.add_argument("--sample-count", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=203)
    args = parser.parse_args()

    raw_root = args.core4d_raw_root or (Path(os.environ["CORE4D_RAW_ROOT"]) if os.environ.get("CORE4D_RAW_ROOT") else None)
    if raw_root is None:
        raise SystemExit("missing --core4d-raw-root or CORE4D_RAW_ROOT")
    thresholds_m = tuple(float(x) for x in args.thresholds_m.split(",") if x.strip())
    if not thresholds_m:
        raise SystemExit("no thresholds provided")
    object_keys = {x.strip().lower() for x in args.object_keys.split(",") if x.strip()}
    inventory_rows = read_tsv(args.inventory_tsv)
    selected_rows = select_inventory_rows(inventory_rows, args.queue, object_keys, args.max_case_persons, args.max_sequences)
    candidates = group_sequence_candidates(selected_rows)
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows_by_threshold: dict[float, list[dict[str, Any]]] = {threshold: [] for threshold in thresholds_m}
    for idx, candidate in enumerate(candidates):
        try:
            result = compute_contact_proxy(raw_root.expanduser().resolve(), candidate, thresholds_m, args.sample_count, args.seed + idx)
        except Exception as exc:  # noqa: BLE001 - raw contact mining must retain per-sequence failure evidence.
            for threshold in thresholds_m:
                rows_by_threshold[threshold].extend(error_rows(candidate, f"{type(exc).__name__}: {exc}", threshold))
            continue
        save_sequence_npz(result, out_dir)
        for ti, threshold in enumerate(thresholds_m):
            for person in candidate.selected_persons:
                rows_by_threshold[threshold].append(score_case_person(result, person, threshold, ti))

    summaries = {
        threshold_label(threshold): write_threshold_outputs(rows, out_dir, threshold)
        for threshold, rows in rows_by_threshold.items()
    }
    run_summary = {
        "stage": "S1_raw_contact",
        "created_at": timestamp(),
        "schema_version": SCHEMA_VERSION,
        "queue": args.queue,
        "thresholds_m": list(thresholds_m),
        "selected_case_person": len(selected_rows),
        "unique_sequences": len(candidates),
        "summaries": summaries,
        "note": "3cm/5cm outputs are separate candidate sets; threshold is a geometric proxy, not GT contact.",
    }
    write_json(out_dir / "raw_contact_run_summary.json", sanitize(run_summary))
    print(json_dumps(sanitize(run_summary)))


if __name__ == "__main__":
    main()
