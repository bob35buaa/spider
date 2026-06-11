#!/usr/bin/env python3
"""E081 evaluation for leg/foot-object collision variants."""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import mujoco
import numpy as np


REPO = Path(__file__).resolve().parents[4]
EVAL_DIR = REPO / "workspace/core4d/scripts/eval"
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))
COLLAB_EVAL_DIR = REPO / "workspace/core4d_collab_retarget/scripts/eval"
if str(COLLAB_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(COLLAB_EVAL_DIR))

import eval_E078 as e078  # noqa: E402
import eval_E079 as e079  # noqa: E402
import eval_E072 as e072  # noqa: E402
import paper_metrics  # noqa: E402


def repo_path_from_env(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    path = Path(raw) if raw else default
    return path if path.is_absolute() else REPO / path


RESULTS = repo_path_from_env("RESULTS", REPO / "workspace/core4d/results/E081")
VARIANTS_FILE = repo_path_from_env(
    "VARIANTS_FILE", REPO / "workspace/core4d/scripts/E081/variants.tsv"
)
FPS = 50.0

LEG_FOOT_GEOMS = [
    "left_hip_collision",
    "right_hip_collision",
    "left_thigh_collision",
    "right_thigh_collision",
    "left_shin_collision",
    "right_shin_collision",
    "left_linkage_brace_collision",
    "right_linkage_brace_collision",
    "lf0",
    "lf1",
    "lf2",
    "lf3",
    "rf0",
    "rf1",
    "rf2",
    "rf3",
]


def read_variants() -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    fieldnames = [
        "variant",
        "source_task",
        "derived_task",
        "mask_source_exp",
        "mask_slug",
        "person_idx",
        "split",
        "role",
    ]
    with VARIANTS_FILE.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(
            (line for line in f if line.strip() and not line.startswith("#")),
            delimiter="\t",
            fieldnames=fieldnames,
        )
        for row in reader:
            if row["role"] not in {"main", "guard"}:
                continue
            out[row["variant"]] = {
                "name": row["variant"],
                "case": row["derived_task"],
                "source_task": row["source_task"],
                "override": f"core4d_{row['variant']}",
                "person_idx": row["person_idx"],
                "split": row["split"],
                "role": row["role"],
            }
    return out


def geom_name(model: mujoco.MjModel, gid: int) -> str:
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or f"geom{gid}"


def signed_point_box(
    point: np.ndarray,
    obj_pos: np.ndarray,
    obj_mat: np.ndarray,
    half: np.ndarray,
) -> float:
    local = obj_mat.T @ (point - obj_pos)
    q = np.abs(local) - half
    outside = np.linalg.norm(np.maximum(q, 0.0))
    inside = min(max(q[0], q[1], q[2]), 0.0)
    return float(outside + inside)


def geom_box_adjusted_sdf(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    geom_id: int,
    obj_pos: np.ndarray,
    obj_mat: np.ndarray,
    half: np.ndarray,
) -> float:
    geom_type = int(model.geom_type[geom_id])
    radius = float(model.geom_size[geom_id, 0])
    center = data.geom_xpos[geom_id].copy()
    mat = data.geom_xmat[geom_id].reshape(3, 3).copy()
    points = [center]
    if geom_type == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
        half_len = float(model.geom_size[geom_id, 1])
        axis = mat[:, 2]
        points = [center + axis * s for s in np.linspace(-half_len, half_len, 9)]
    signed = min(signed_point_box(p, obj_pos, obj_mat, half) for p in points)
    return float(signed - radius)


def leg_object_timeseries(
    model: mujoco.MjModel,
    qpos: np.ndarray,
) -> list[dict[str, object]]:
    data = mujoco.MjData(model)
    object_geom = e072.name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "object_collision")
    floor_geom = e072.name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    half = model.geom_size[object_geom, :3].copy()
    leg_gids = [
        e072.name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        for name in LEG_FOOT_GEOMS
    ]
    hand_gids = [
        e072.name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "lh"),
        e072.name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "rh"),
    ]

    rows: list[dict[str, object]] = []
    for frame, q in enumerate(qpos):
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        obj_pos = data.geom_xpos[object_geom].copy()
        obj_mat = data.geom_xmat[object_geom].reshape(3, 3).copy()

        leg_vals = [
            (
                geom_box_adjusted_sdf(model, data, gid, obj_pos, obj_mat, half),
                geom_name(model, gid),
            )
            for gid in leg_gids
        ]
        hand_vals = [
            (
                geom_box_adjusted_sdf(model, data, gid, obj_pos, obj_mat, half),
                geom_name(model, gid),
            )
            for gid in hand_gids
        ]
        leg_sdf, leg_arg = min(leg_vals, key=lambda x: x[0])
        hand_sdf, hand_arg = min(hand_vals, key=lambda x: x[0])

        leg_contacts = 0
        hand_contacts = 0
        floor_contacts = 0
        leg_min_dist = np.nan
        hand_min_dist = np.nan
        for ci in range(data.ncon):
            con = data.contact[ci]
            pair = {int(con.geom1), int(con.geom2)}
            if object_geom in pair and floor_geom in pair:
                floor_contacts += 1
            if object_geom not in pair:
                continue
            other = next(iter(pair - {object_geom}))
            if other in leg_gids:
                leg_contacts += 1
                leg_min_dist = (
                    float(con.dist)
                    if not np.isfinite(leg_min_dist)
                    else min(leg_min_dist, float(con.dist))
                )
            if other in hand_gids:
                hand_contacts += 1
                hand_min_dist = (
                    float(con.dist)
                    if not np.isfinite(hand_min_dist)
                    else min(hand_min_dist, float(con.dist))
                )

        rows.append(
            {
                "frame": frame,
                "eval_time_s": float(frame / FPS),
                "leg_box_sdf_min_m": float(leg_sdf),
                "leg_box_sdf_arg": leg_arg,
                "hand_box_sdf_min_m": float(hand_sdf),
                "hand_box_sdf_arg": hand_arg,
                "leg_object_contact_count": int(leg_contacts),
                "hand_object_contact_count": int(hand_contacts),
                "object_floor_contact_count": int(floor_contacts),
                "leg_object_contact_min_dist_m": float(leg_min_dist)
                if np.isfinite(leg_min_dist)
                else "",
                "hand_object_contact_min_dist_m": float(hand_min_dist)
                if np.isfinite(hand_min_dist)
                else "",
                "object_bottom_proxy_m": float(obj_pos[2] - half[2]),
                "object_z_m": float(obj_pos[2]),
            }
        )
    return rows


def write_leg_timeseries(variant: str, sim_rows: list[dict[str, object]], ref_rows: list[dict[str, object]]) -> None:
    path = RESULTS / f"legobj_timeseries_{variant}.csv"
    keys = ["variant", "kind"] + list(sim_rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for kind, rows in [("sim", sim_rows), ("ref", ref_rows)]:
            for row in rows:
                writer.writerow({"variant": variant, "kind": kind, **row})


def _array(rows: list[dict[str, object]], key: str) -> np.ndarray:
    return np.asarray([float(row[key]) for row in rows], dtype=np.float64)


def _contact_array(rows: list[dict[str, object]], key: str) -> np.ndarray:
    return np.asarray([int(row[key]) > 0 for row in rows], dtype=bool)


def _summarize_prefix(rows: list[dict[str, object]], start: int, end: int, prefix: str) -> dict[str, object]:
    window = rows[start : end + 1]
    leg_sdf = _array(window, "leg_box_sdf_min_m")
    hand_sdf = _array(window, "hand_box_sdf_min_m")
    obj_bottom = _array(window, "object_bottom_proxy_m")
    leg_contact = _contact_array(window, "leg_object_contact_count")
    hand_contact = _contact_array(window, "hand_object_contact_count")
    floor_contact = _contact_array(window, "object_floor_contact_count")
    return {
        f"{prefix}_leg_box_sdf_min_m": float(leg_sdf.min()),
        f"{prefix}_leg_box_sdf_mean_m": float(leg_sdf.mean()),
        f"{prefix}_leg_box_interference_frames_pct": float((leg_sdf < 0.0).mean() * 100.0),
        f"{prefix}_leg_box_near_2cm_frames_pct": float((leg_sdf < 0.02).mean() * 100.0),
        f"{prefix}_leg_object_contact_frames_pct": float(leg_contact.mean() * 100.0),
        f"{prefix}_hand_box_sdf_min_m": float(hand_sdf.min()),
        f"{prefix}_hand_box_sdf_mean_m": float(hand_sdf.mean()),
        f"{prefix}_hand_object_contact_frames_pct": float(hand_contact.mean() * 100.0),
        f"{prefix}_object_floor_contact_frames_pct": float(floor_contact.mean() * 100.0),
        f"{prefix}_object_bottom_proxy_min_m": float(obj_bottom.min()),
        f"{prefix}_object_bottom_proxy_mean_m": float(obj_bottom.mean()),
        f"{prefix}_object_bottom_proxy_max_m": float(obj_bottom.max()),
    }


def leg_object_metrics(summary: dict[str, object]) -> dict[str, object]:
    variant = str(summary["variant"])
    case = str(summary["case"])
    override = str(summary["override"])
    model, scene_used = e078.load_scene_model(case)

    data_npz = np.load(RESULTS / f"{variant}.npz", allow_pickle=True)
    qpos = e072.flatten_time_major(data_npz["qpos"])
    qpos_ref, _ctrl_ref = e078.load_ref(override, case)
    T = min(len(qpos), len(qpos_ref), int(summary["T"]))
    qpos = qpos[:T]
    qpos_ref = qpos_ref[:T]

    sim_rows = leg_object_timeseries(model, qpos)
    ref_rows = leg_object_timeseries(model, qpos_ref)
    write_leg_timeseries(variant, sim_rows, ref_rows)

    cw_start = int(summary["case_window_start_frame"])
    cw_end = min(int(summary["case_window_end_frame"]), T - 1)
    out: dict[str, object] = {
        "legobj_scene_used": str(scene_used.relative_to(REPO))
        if scene_used.is_absolute() and scene_used.is_relative_to(REPO)
        else str(scene_used),
        "legobj_geom_count": len(LEG_FOOT_GEOMS),
    }
    out.update(_summarize_prefix(sim_rows, 0, T - 1, "full_sim"))
    out.update(_summarize_prefix(ref_rows, 0, T - 1, "full_ref"))
    out.update(_summarize_prefix(sim_rows, cw_start, cw_end, "case_window_sim"))
    out.update(_summarize_prefix(ref_rows, cw_start, cw_end, "case_window_ref"))

    out["case_window_object_bottom_mean_gap_vs_ref_m"] = float(
        out["case_window_sim_object_bottom_proxy_mean_m"]
        - out["case_window_ref_object_bottom_proxy_mean_m"]
    )
    out["E081_success_case_window"] = bool(summary["E079_success_case_window"])
    out["E081_success_legobj_strict_proxy"] = bool(
        out["E081_success_case_window"]
        and out["case_window_sim_leg_box_interference_frames_pct"] <= 5.0
        and out["case_window_sim_object_bottom_proxy_mean_m"]
        >= out["case_window_ref_object_bottom_proxy_mean_m"] - 0.05
    )
    return out


def add_e081_paper_metrics(summary: dict[str, object]) -> dict[str, object]:
    variant = str(summary["variant"])
    case = str(summary["case"])
    override = str(summary["override"])
    person_idx = int(summary.get("person_idx", 0))

    model, _scene_used = e078.load_scene_model(case)
    data_npz = np.load(RESULTS / f"{variant}.npz", allow_pickle=True)
    qpos = e072.flatten_time_major(data_npz["qpos"])
    qpos_ref, _ctrl_ref = e078.load_ref(override, case)
    return paper_metrics.add_paper_metrics(
        summary,
        repo=REPO,
        results_dir=RESULTS,
        model=model,
        qpos=qpos,
        qpos_ref=qpos_ref,
        person_idx=person_idx,
    )


def write_variant_summary(summary: dict[str, object]) -> None:
    variant = str(summary["variant"])
    summary_json = RESULTS / f"eval_summary_{variant}.json"
    summary_csv = RESULTS / f"eval_summary_{variant}.csv"
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)


def main() -> None:
    variants = read_variants()
    selected = sys.argv[1:] or list(variants.keys())

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "plots").mkdir(parents=True, exist_ok=True)
    e078.RESULTS = RESULTS
    e078.VARIANTS = variants
    e079.RESULTS = RESULTS
    e079.VARIANTS_FILE = VARIANTS_FILE

    summaries = []
    for variant in selected:
        if variant not in variants:
            print(f"[SKIP] unknown variant {variant}")
            continue
        try:
            summary = e078.evaluate_variant(variant)
        except FileNotFoundError as exc:
            print(f"[SKIP] {exc}")
            continue
        summary.update(e079.case_window_metrics(summary))
        summary.update(leg_object_metrics(summary))
        summary["split"] = variants[variant]["split"]
        summary["role"] = variants[variant]["role"]
        summary["source_task"] = variants[variant]["source_task"]
        summary["person_idx"] = int(variants[variant]["person_idx"])
        summary.update(add_e081_paper_metrics(summary))
        summary["E081_success_numeric"] = bool(
            summary["post2_pelvis_z_min_m"] >= 0.55
            and summary["post2_sim_contact_frames_pct"] >= 50.0
            and summary["post2_obj_err_mean_m"] <= 0.20
        )
        write_variant_summary(summary)
        summaries.append(summary)

    if not summaries:
        raise SystemExit("No E081 variant results found.")

    keys = sorted({k for row in summaries for k in row.keys()})
    comparison = RESULTS / "comparison.csv"
    with comparison.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(summaries)

    main_rows = [r for r in summaries if r["role"] == "main"]
    aggregate = {
        "num_results": len(summaries),
        "num_main_results": len(main_rows),
        "num_main_case_window_success": sum(
            bool(r["E081_success_case_window"]) for r in main_rows
        ),
        "main_case_window_success_pct": (
            100.0
            * sum(bool(r["E081_success_case_window"]) for r in main_rows)
            / len(main_rows)
            if main_rows
            else 0.0
        ),
        "num_main_legobj_strict_proxy_success": sum(
            bool(r["E081_success_legobj_strict_proxy"]) for r in main_rows
        ),
        "main_legobj_strict_proxy_success_pct": (
            100.0
            * sum(bool(r["E081_success_legobj_strict_proxy"]) for r in main_rows)
            / len(main_rows)
            if main_rows
            else 0.0
        ),
        "guard_results": [r["variant"] for r in summaries if r["role"] == "guard"],
    }
    (RESULTS / "aggregate_summary.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True), encoding="utf-8"
    )

    print(f"Wrote {comparison}")
    print(json.dumps(aggregate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
