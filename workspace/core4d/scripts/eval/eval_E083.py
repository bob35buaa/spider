#!/usr/bin/env python3
"""E083 evaluator with E081 leg metrics plus upper-body/object diagnostics."""

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

import eval_E072 as e072  # noqa: E402
import eval_E081 as e081  # noqa: E402


def _env_path(name: str, default: Path) -> Path:
    path = Path(os.environ.get(name, str(default)))
    if not path.is_absolute():
        path = REPO / path
    os.environ[name] = str(path)
    return path


RESULTS = _env_path("RESULTS", REPO / "workspace/core4d/results/E083")
VARIANTS_FILE = _env_path("VARIANTS_FILE", REPO / "workspace/core4d/scripts/E083/variants.tsv")
FPS = 50.0
OBJECT_GEOM = "object_collision"

UPPER_BODY_GEOMS = [
    "head_collision",
    "torso_collision",
    "pelvis_collision",
    "left_shoulder_yaw_collision",
    "right_shoulder_yaw_collision",
    "left_elbow_yaw_collision",
    "right_elbow_yaw_collision",
]
HAND_GEOMS = ["lh", "rh"]
REPLAY_GEOMS = UPPER_BODY_GEOMS + HAND_GEOMS


def _first(mask: np.ndarray, start: int = 0) -> int:
    idx = np.flatnonzero(mask[start:])
    return int(idx[0] + start) if len(idx) else -1


def _first_between(mask: np.ndarray, start: int, end: int) -> int:
    if end < start:
        return -1
    idx = np.flatnonzero(mask[start : end + 1])
    return int(idx[0] + start) if len(idx) else -1


def _pct(mask: np.ndarray, start: int, end: int) -> float:
    if end < start:
        return 0.0
    return float(mask[start : end + 1].mean() * 100.0)


def _has_pair(model: mujoco.MjModel, geom_a: str, geom_b: str) -> bool:
    gid_a = e072.name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_a)
    gid_b = e072.name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_b)
    for i in range(model.npair):
        pair = {int(model.pair_geom1[i]), int(model.pair_geom2[i])}
        if pair == {gid_a, gid_b}:
            return True
    return False


def _geom_name(model: mujoco.MjModel, gid: int) -> str:
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or f"geom{gid}"


def _replay(model: mujoco.MjModel, qpos: np.ndarray) -> dict[str, object]:
    data = mujoco.MjData(model)
    object_gid = e072.name2id(model, mujoco.mjtObj.mjOBJ_GEOM, OBJECT_GEOM)
    floor_gid = e072.name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    gids = {
        name: e072.name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        for name in REPLAY_GEOMS
    }
    T = len(qpos)
    obj_half = model.geom_size[object_gid, :3].copy()
    sdf = {name: np.zeros(T, dtype=np.float64) for name in REPLAY_GEOMS}
    z = {name: np.zeros(T, dtype=np.float64) for name in REPLAY_GEOMS}
    object_contact = {name: np.zeros(T, dtype=bool) for name in REPLAY_GEOMS}
    floor_contact = {name: np.zeros(T, dtype=bool) for name in HAND_GEOMS}
    torso_axis_z = np.zeros(T, dtype=np.float64)
    contact_arg = {name: [""] * T for name in REPLAY_GEOMS}

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
                for name in HAND_GEOMS:
                    if gids[name] in pair:
                        floor_contact[name][frame] = True
            if object_gid in pair:
                for name, gid in gids.items():
                    if gid in pair:
                        object_contact[name][frame] = True
                        other = next(iter(pair - {object_gid}))
                        contact_arg[name][frame] = _geom_name(model, other)

    return {
        "sdf": sdf,
        "z": z,
        "object_contact": object_contact,
        "floor_contact": floor_contact,
        "torso_axis_z": torso_axis_z,
        "contact_arg": contact_arg,
    }


def _summarize_replay(replay: dict[str, object], start: int, end: int, prefix: str) -> dict[str, object]:
    sdf = replay["sdf"]
    z = replay["z"]
    object_contact = replay["object_contact"]
    floor_contact = replay["floor_contact"]
    torso_axis_z = replay["torso_axis_z"]
    assert isinstance(sdf, dict) and isinstance(z, dict)
    assert isinstance(object_contact, dict) and isinstance(floor_contact, dict)
    assert isinstance(torso_axis_z, np.ndarray)

    upper_stack = np.stack([sdf[name] for name in UPPER_BODY_GEOMS], axis=0)
    upper_window = upper_stack[:, start : end + 1]
    out: dict[str, object] = {
        f"{prefix}_upperbody_object_sdf_min_m": float(upper_window.min()),
        f"{prefix}_upperbody_object_penetration_pct": float(
            (upper_window.min(axis=0) < 0.0).mean() * 100.0
        ),
        f"{prefix}_upperbody_object_near_2cm_pct": float(
            (upper_window.min(axis=0) < 0.02).mean() * 100.0
        ),
        f"{prefix}_torso_axis_vertical_abs_min": float(torso_axis_z[start : end + 1].min()),
    }
    for name in REPLAY_GEOMS:
        vals = sdf[name]
        out[f"{prefix}_{name}_object_sdf_min_m"] = float(vals[start : end + 1].min())
        out[f"{prefix}_{name}_object_sdf_mean_m"] = float(vals[start : end + 1].mean())
        out[f"{prefix}_{name}_object_penetration_pct"] = _pct(vals < 0.0, start, end)
        out[f"{prefix}_{name}_object_near_2cm_pct"] = _pct(vals < 0.02, start, end)
        out[f"{prefix}_{name}_object_contact_pct"] = _pct(object_contact[name], start, end)
        out[f"{prefix}_{name}_z_min_m"] = float(z[name][start : end + 1].min())
    for name in HAND_GEOMS:
        out[f"{prefix}_{name}_floor_contact_pct"] = _pct(floor_contact[name], start, end)
    return out


def _write_timeseries(
    variant: str,
    sim: dict[str, object],
    ref: dict[str, object],
    T: int,
) -> None:
    path = RESULTS / f"upperbody_timeseries_{variant}.csv"
    keys = [
        "variant",
        "kind",
        "frame",
        "eval_time_s",
        "torso_axis_vertical_abs",
    ]
    for name in REPLAY_GEOMS:
        keys.extend(
            [
                f"{name}_object_sdf_m",
                f"{name}_object_contact",
                f"{name}_z_m",
            ]
        )
    for name in HAND_GEOMS:
        keys.append(f"{name}_floor_contact")

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for kind, replay in [("sim", sim), ("ref", ref)]:
            sdf = replay["sdf"]
            z = replay["z"]
            object_contact = replay["object_contact"]
            floor_contact = replay["floor_contact"]
            torso_axis_z = replay["torso_axis_z"]
            assert isinstance(sdf, dict) and isinstance(z, dict)
            assert isinstance(object_contact, dict) and isinstance(floor_contact, dict)
            assert isinstance(torso_axis_z, np.ndarray)
            for frame in range(T):
                row: dict[str, object] = {
                    "variant": variant,
                    "kind": kind,
                    "frame": frame,
                    "eval_time_s": frame / FPS,
                    "torso_axis_vertical_abs": float(torso_axis_z[frame]),
                }
                for name in REPLAY_GEOMS:
                    row[f"{name}_object_sdf_m"] = float(sdf[name][frame])
                    row[f"{name}_object_contact"] = bool(object_contact[name][frame])
                    row[f"{name}_z_m"] = float(z[name][frame])
                for name in HAND_GEOMS:
                    row[f"{name}_floor_contact"] = bool(floor_contact[name][frame])
                writer.writerow(row)


def upperbody_metrics(summary: dict[str, object]) -> dict[str, object]:
    variant = str(summary["variant"])
    case = str(summary["case"])
    override = str(summary["override"])
    model, scene_used = e081.e078.load_scene_model(case)

    data_npz = np.load(RESULTS / f"{variant}.npz", allow_pickle=True)
    qpos = e072.flatten_time_major(data_npz["qpos"])
    qpos_ref, _ctrl_ref = e081.e078.load_ref(override, case)
    T = min(len(qpos), len(qpos_ref), int(summary["T"]))
    qpos = qpos[:T]
    qpos_ref = qpos_ref[:T]

    sim = _replay(model, qpos)
    ref = _replay(model, qpos_ref)
    _write_timeseries(variant, sim, ref, T)

    start = int(summary.get("case_window_start_frame", 0))
    end = min(int(summary.get("case_window_end_frame", T - 1)), T - 1)

    out: dict[str, object] = {
        "upperobj_scene_used": str(scene_used.relative_to(REPO))
        if scene_used.is_absolute() and scene_used.is_relative_to(REPO)
        else str(scene_used),
        "upperobj_upperbody_geom_count": len(UPPER_BODY_GEOMS),
        "upperobj_object_half_xyz_m": "/".join(
            f"{x:.4f}"
            for x in model.geom_size[
                e072.name2id(model, mujoco.mjtObj.mjOBJ_GEOM, OBJECT_GEOM), :3
            ]
        ),
    }
    for name in UPPER_BODY_GEOMS:
        out[f"has_{name}_object_pair"] = _has_pair(model, name, OBJECT_GEOM)
    for name in HAND_GEOMS:
        out[f"has_{name}_floor_pair"] = _has_pair(model, name, "floor")

    out.update(_summarize_replay(sim, 0, T - 1, "full_sim"))
    out.update(_summarize_replay(ref, 0, T - 1, "full_ref"))
    out.update(_summarize_replay(sim, start, end, "case_window_sim"))
    out.update(_summarize_replay(ref, start, end, "case_window_ref"))

    sim_sdf = sim["sdf"]
    sim_floor = sim["floor_contact"]
    assert isinstance(sim_sdf, dict) and isinstance(sim_floor, dict)
    for name in UPPER_BODY_GEOMS + HAND_GEOMS:
        out[f"first_sim_{name}_object_penetration_frame"] = _first(sim_sdf[name] < 0.0)
        out[f"first_case_window_sim_{name}_object_penetration_frame"] = _first_between(
            sim_sdf[name] < 0.0, start, end
        )
    for name in HAND_GEOMS:
        out[f"first_sim_{name}_floor_contact_frame"] = _first(sim_floor[name])
        out[f"first_case_window_sim_{name}_floor_contact_frame"] = _first_between(
            sim_floor[name], start, end
        )

    out["E083_success_upperbody_physical_proxy"] = bool(
        out["case_window_sim_head_collision_object_penetration_pct"] <= 1.0
        and out["case_window_sim_torso_collision_object_penetration_pct"] <= 1.0
        and out["case_window_sim_upperbody_object_penetration_pct"] <= 5.0
        and out["case_window_sim_lh_floor_contact_pct"] <= 5.0
        and out["case_window_sim_rh_floor_contact_pct"] <= 5.0
        and float(summary.get("case_window_pelvis_z_min_m", 0.0)) >= 0.55
    )
    return out


def write_upperbody_diagnostics(summaries: list[dict[str, object]]) -> None:
    focus = [
        "variant",
        "role",
        "split",
        "case",
        "upperobj_object_half_xyz_m",
        "case_window_start_frame",
        "case_window_end_frame",
        "case_window_obj_err_mean_m",
        "case_window_pelvis_z_min_m",
        "case_window_sim_contact_frames_pct",
        "case_window_sim_head_collision_object_penetration_pct",
        "case_window_sim_torso_collision_object_penetration_pct",
        "case_window_sim_pelvis_collision_object_penetration_pct",
        "case_window_sim_upperbody_object_penetration_pct",
        "case_window_sim_lh_floor_contact_pct",
        "case_window_sim_rh_floor_contact_pct",
        "first_case_window_sim_head_collision_object_penetration_frame",
        "first_case_window_sim_torso_collision_object_penetration_frame",
        "first_case_window_sim_lh_floor_contact_frame",
        "first_case_window_sim_rh_floor_contact_frame",
        "E081_success_case_window",
        "E083_success_upperbody_physical_proxy",
    ]
    path = RESULTS / "upperbody_diagnostics.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=focus)
        writer.writeheader()
        for summary in summaries:
            writer.writerow({key: summary.get(key, "") for key in focus})


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "plots").mkdir(parents=True, exist_ok=True)
    e081.RESULTS = RESULTS
    e081.VARIANTS_FILE = VARIANTS_FILE
    e081.e078.RESULTS = RESULTS
    e081.e078.VARIANTS = {}
    e081.e079.RESULTS = RESULTS
    e081.e079.VARIANTS_FILE = VARIANTS_FILE

    variants = e081.read_variants()
    e081.e078.VARIANTS = variants
    selected = sys.argv[1:] or list(variants.keys())
    summaries: list[dict[str, object]] = []

    for variant in selected:
        if variant not in variants:
            print(f"[SKIP] unknown variant {variant}")
            continue
        try:
            summary = e081.e078.evaluate_variant(variant)
        except FileNotFoundError as exc:
            print(f"[SKIP] {exc}")
            continue
        summary.update(e081.e079.case_window_metrics(summary))
        summary.update(e081.leg_object_metrics(summary))
        summary["split"] = variants[variant]["split"]
        summary["role"] = variants[variant]["role"]
        summary["source_task"] = variants[variant]["source_task"]
        summary["person_idx"] = int(variants[variant]["person_idx"])
        summary.update(e081.add_e081_paper_metrics(summary))
        summary["E081_success_numeric"] = bool(
            summary["post2_pelvis_z_min_m"] >= 0.55
            and summary["post2_sim_contact_frames_pct"] >= 50.0
            and summary["post2_obj_err_mean_m"] <= 0.20
        )
        summary.update(upperbody_metrics(summary))
        e081.write_variant_summary(summary)
        summaries.append(summary)

    if not summaries:
        raise SystemExit("No E083 variant results found.")

    keys = sorted({k for row in summaries for k in row.keys()})
    comparison = RESULTS / "comparison.csv"
    with comparison.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(summaries)
    write_upperbody_diagnostics(summaries)

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
        "num_main_upperbody_physical_proxy_success": sum(
            bool(r["E083_success_upperbody_physical_proxy"]) for r in main_rows
        ),
        "main_upperbody_physical_proxy_success_pct": (
            100.0
            * sum(bool(r["E083_success_upperbody_physical_proxy"]) for r in main_rows)
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
    print(f"Wrote {RESULTS / 'upperbody_diagnostics.csv'}")
    print(json.dumps(aggregate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
