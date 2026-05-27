#!/usr/bin/env python3
"""Diagnose E082 body fall/head-torso penetration against E081 box023 guard."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

REPO = Path(__file__).resolve().parents[4]
EVAL_DIR = REPO / "workspace/core4d/scripts/eval"
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

import eval_E072 as e072  # noqa: E402
import eval_E078 as e078  # noqa: E402
import eval_E081 as e081  # noqa: E402

FPS = 50.0
OBJECT_GEOM = "object_collision"
BODY_GEOMS = [
    "head_collision",
    "torso_collision",
    "pelvis_collision",
    "lh",
    "rh",
]
FLOOR_CONTACT_GEOMS = ["lh", "rh"]


def _first(mask: np.ndarray, start: int = 0) -> int:
    idx = np.flatnonzero(mask[start:])
    return int(idx[0] + start) if len(idx) else -1


def _pct(mask: np.ndarray, start: int, end: int) -> float:
    if end < start:
        return 0.0
    return float(mask[start : end + 1].mean() * 100.0)


def _load_model(summary: dict[str, object]) -> tuple[mujoco.MjModel, Path]:
    scene = REPO / str(summary["scene_used"])
    if scene.is_file():
        return mujoco.MjModel.from_xml_path(str(scene)), scene
    scene = REPO / str(summary["legobj_scene_used"])
    return mujoco.MjModel.from_xml_path(str(scene)), scene


def _has_pair(model: mujoco.MjModel, geom_a: str, geom_b: str) -> bool:
    gid_a = e072.name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_a)
    gid_b = e072.name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_b)
    for i in range(model.npair):
        pair = {int(model.pair_geom1[i]), int(model.pair_geom2[i])}
        if pair == {gid_a, gid_b}:
            return True
    return False


def _replay(
    model: mujoco.MjModel,
    qpos: np.ndarray,
) -> dict[str, np.ndarray | dict[str, np.ndarray]]:
    data = mujoco.MjData(model)
    object_gid = e072.name2id(model, mujoco.mjtObj.mjOBJ_GEOM, OBJECT_GEOM)
    floor_gid = e072.name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    gids = {
        name: e072.name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        for name in BODY_GEOMS
    }
    T = len(qpos)
    sdf = {name: np.zeros(T, dtype=np.float64) for name in BODY_GEOMS}
    z = {name: np.zeros(T, dtype=np.float64) for name in BODY_GEOMS}
    floor_contact = {name: np.zeros(T, dtype=bool) for name in FLOOR_CONTACT_GEOMS}
    object_contact = {name: np.zeros(T, dtype=bool) for name in BODY_GEOMS}
    torso_axis_z = np.zeros(T, dtype=np.float64)
    obj_half = model.geom_size[object_gid, :3].copy()

    for frame, q in enumerate(qpos):
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        obj_pos = data.geom_xpos[object_gid].copy()
        obj_mat = data.geom_xmat[object_gid].reshape(3, 3).copy()
        for name, gid in gids.items():
            sdf[name][frame] = e081.geom_box_adjusted_sdf(
                model, data, gid, obj_pos, obj_mat, obj_half
            )
            z[name][frame] = float(data.geom_xpos[gid, 2])
        torso_gid = gids["torso_collision"]
        torso_mat = data.geom_xmat[torso_gid].reshape(3, 3)
        torso_axis_z[frame] = float(abs(torso_mat[:, 2].dot(np.array([0.0, 0.0, 1.0]))))
        for ci in range(data.ncon):
            con = data.contact[ci]
            pair = {int(con.geom1), int(con.geom2)}
            if floor_gid in pair:
                for name in FLOOR_CONTACT_GEOMS:
                    if gids[name] in pair:
                        floor_contact[name][frame] = True
            if object_gid in pair:
                for name, gid in gids.items():
                    if gid in pair:
                        object_contact[name][frame] = True
    return {
        "sdf": sdf,
        "z": z,
        "floor_contact": floor_contact,
        "object_contact": object_contact,
        "torso_axis_z": torso_axis_z,
    }


def _summarize(summary_path: Path) -> dict[str, object]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    results = summary_path.parent
    variant = str(summary["variant"])
    model, scene = _load_model(summary)
    qpos = e072.flatten_time_major(
        np.load(results / f"{variant}.npz", allow_pickle=True)["qpos"]
    )
    override = str(summary["override"])
    case = str(summary["case"])
    qpos_ref, _ctrl_ref = e078.load_ref(override, case)
    T = min(len(qpos), len(qpos_ref), int(summary["T"]))
    qpos = qpos[:T]
    qpos_ref = qpos_ref[:T]
    sim = _replay(model, qpos)
    ref = _replay(model, qpos_ref)
    start = int(summary.get("case_window_start_frame", 0))
    end = min(int(summary.get("case_window_end_frame", T - 1)), T - 1)

    row: dict[str, object] = {
        "variant": variant,
        "case": case,
        "T": T,
        "scene": str(scene.relative_to(REPO)) if scene.is_absolute() else str(scene),
        "object_half_xyz_m": "/".join(f"{x:.4f}" for x in model.geom_size[e072.name2id(model, mujoco.mjtObj.mjOBJ_GEOM, OBJECT_GEOM), :3]),
        "has_head_object_pair": _has_pair(model, "head_collision", OBJECT_GEOM),
        "has_torso_object_pair": _has_pair(model, "torso_collision", OBJECT_GEOM),
        "has_lh_floor_pair": _has_pair(model, "lh", "floor"),
        "has_rh_floor_pair": _has_pair(model, "rh", "floor"),
        "case_window_start_frame": start,
        "case_window_end_frame": end,
        "first_pelvis_z_lt_45cm_frame": summary.get("first_pelvis_z_lt_45cm_frame"),
        "first_sim_zero_contact_frame": summary.get("first_sim_zero_contact_frame"),
        "first_sim_min_hand_sdf_gt_10cm_frame": summary.get("first_sim_min_hand_sdf_gt_10cm_frame"),
        "case_window_obj_err_mean_m": summary.get("case_window_obj_err_mean_m"),
        "case_window_sim_contact_frames_pct": summary.get("case_window_sim_contact_frames_pct"),
        "case_window_ref_contact_frames_pct": summary.get("case_window_ref_contact_frames_pct"),
        "case_window_pelvis_z_min_m": summary.get("case_window_pelvis_z_min_m"),
    }
    for kind, replay in [("sim", sim), ("ref", ref)]:
        sdf = replay["sdf"]
        z = replay["z"]
        floor = replay["floor_contact"]
        obj_contact = replay["object_contact"]
        torso_axis_z = replay["torso_axis_z"]
        assert isinstance(sdf, dict) and isinstance(z, dict)
        assert isinstance(floor, dict) and isinstance(obj_contact, dict)
        assert isinstance(torso_axis_z, np.ndarray)
        for name in BODY_GEOMS:
            row[f"{kind}_{name}_object_sdf_min_m"] = float(sdf[name][start : end + 1].min())
            row[f"{kind}_{name}_object_penetration_pct"] = _pct(sdf[name] < 0.0, start, end)
            row[f"{kind}_{name}_object_near_2cm_pct"] = _pct(sdf[name] < 0.02, start, end)
            row[f"{kind}_{name}_z_min_m"] = float(z[name][start : end + 1].min())
            row[f"{kind}_{name}_object_contact_pct"] = _pct(obj_contact[name], start, end)
            row[f"{kind}_first_{name}_object_penetration_frame"] = _first(sdf[name] < 0.0, start)
        for name in FLOOR_CONTACT_GEOMS:
            row[f"{kind}_{name}_floor_contact_pct"] = _pct(floor[name], start, end)
            row[f"{kind}_first_{name}_floor_contact_frame"] = _first(floor[name], start)
        row[f"{kind}_torso_axis_vertical_abs_min"] = float(torso_axis_z[start : end + 1].min())
    return row


def main() -> None:
    paths = [
        REPO / "workspace/core4d/results/E082/eval_summary_E082_d003_box021_20231018_029_p2_legobj.json",
        REPO / "workspace/core4d/results/E082/eval_summary_E082_d003_box021_20231011_035_p2_legobj.json",
        REPO / "workspace/core4d/results/E082/eval_summary_E082_d003_box021_20231020_019_p1_legobj.json",
        REPO / "workspace/core4d/results/E081/eval_summary_E081_box023_p2_legobj.json",
    ]
    rows = [_summarize(p) for p in paths]
    out_dir = REPO / "workspace/core4d/results/E082/diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "body_fall_vs_box023.csv"
    keys = sorted({k for row in rows for k in row})
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(out_csv)
    focus = [
        "variant",
        "object_half_xyz_m",
        "has_head_object_pair",
        "has_torso_object_pair",
        "first_pelvis_z_lt_45cm_frame",
        "first_sim_zero_contact_frame",
        "case_window_obj_err_mean_m",
        "case_window_pelvis_z_min_m",
        "case_window_sim_contact_frames_pct",
        "sim_head_collision_object_sdf_min_m",
        "sim_head_collision_object_penetration_pct",
        "sim_torso_collision_object_sdf_min_m",
        "sim_torso_collision_object_penetration_pct",
        "sim_lh_floor_contact_pct",
        "sim_rh_floor_contact_pct",
        "ref_head_collision_object_sdf_min_m",
        "ref_torso_collision_object_sdf_min_m",
        "sim_torso_axis_vertical_abs_min",
    ]
    print("\t".join(focus))
    for row in rows:
        print("\t".join(str(row.get(k, "")) for k in focus))


if __name__ == "__main__":
    main()
