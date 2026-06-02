#!/usr/bin/env python3
"""Evaluate E108 non-box bucket004 full CEM rollouts.

This follows the E090/E105 metric style but handles bucket proxy templates with
multiple object collision boxes: object_collision bottom plus bucket wall geoms.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
from typing import Any

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

REPO = Path(__file__).resolve().parents[4]
TASK_ROOT = REPO / "example_datasets/processed/core4d/unitree_g1/humanoid_object"
DEFAULT_VARIANTS = REPO / "workspace/core4d/scripts/E108/nonbox_bucket004_variants.tsv"
REPLAY_GATE = REPO / "workspace/core4d/scripts/E098"
if str(REPLAY_GATE) not in sys.path:
    sys.path.insert(0, str(REPLAY_GATE))
import replay_gate as rg  # noqa: E402

EEF_OFFSET = np.array([0.05, 0.0, 0.0], dtype=np.float64)
LOWERBODY_INTERFERENCE_STRICT_FRAC = 0.05

UPPER_BODY_NAMES = [
    "torso_link",
    "left_shoulder_yaw_link",
    "right_shoulder_yaw_link",
    "left_shoulder_pitch_link",
    "right_shoulder_pitch_link",
    "pelvis_contour_link",
    "left_elbow_link",
    "right_elbow_link",
]

LOWERBODY_GEOMS = [
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

HAND_GEOMS = ["lh", "rh"]


def repo_path_from_env(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    path = Path(raw) if raw else default
    return path if path.is_absolute() else REPO / path


def read_variants(path: Path) -> dict[str, dict[str, str]]:
    fields = [
        "ordinal",
        "variant",
        "case_id",
        "derived_task",
        "person_idx",
        "split",
        "source_scene_task",
        "visual_qc_decision",
    ]
    out: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(
            (line for line in f if line.strip() and not line.startswith("#")),
            delimiter="\t",
            fieldnames=fields,
        )
        for row in reader:
            out[row["variant"]] = row
    return out


def body_id(model: mujoco.MjModel, name: str) -> int:
    return int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name))


def geom_id(model: mujoco.MjModel, name: str) -> int:
    return int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name))


def geom_name(model: mujoco.MjModel, gid: int) -> str:
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or f"geom{gid}"


def object_collision_geoms(model: mujoco.MjModel) -> list[int]:
    gids: list[int] = []
    for gid in range(model.ngeom):
        name = geom_name(model, gid)
        if name.startswith("object_collision"):
            gids.append(gid)
    if not gids:
        raise ValueError("no object_collision* geoms found")
    return gids


def signed_point_box(point: np.ndarray, box_pos: np.ndarray, box_mat: np.ndarray, half: np.ndarray) -> float:
    local = box_mat.T @ (point - box_pos)
    q = np.abs(local) - half
    outside = np.linalg.norm(np.maximum(q, 0.0))
    inside = min(max(q[0], q[1], q[2]), 0.0)
    return float(outside + inside)


def point_distance_to_object_proxy(model: mujoco.MjModel, data: mujoco.MjData, point: np.ndarray, object_gids: list[int]) -> float:
    vals = []
    for gid in object_gids:
        vals.append(
            signed_point_box(
                point,
                data.geom_xpos[gid].copy(),
                data.geom_xmat[gid].reshape(3, 3).copy(),
                model.geom_size[gid, :3].copy(),
            )
        )
    return float(min(vals))


def geom_box_adjusted_sdf(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    geom: int,
    object_gid: int,
) -> float:
    geom_type = int(model.geom_type[geom])
    radius = float(model.geom_size[geom, 0])
    center = data.geom_xpos[geom].copy()
    mat = data.geom_xmat[geom].reshape(3, 3).copy()
    points = [center]
    if geom_type == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
        half_len = float(model.geom_size[geom, 1])
        axis = mat[:, 2]
        points = [center + axis * s for s in np.linspace(-half_len, half_len, 9)]
    obj_pos = data.geom_xpos[object_gid].copy()
    obj_mat = data.geom_xmat[object_gid].reshape(3, 3).copy()
    half = model.geom_size[object_gid, :3].copy()
    return float(min(signed_point_box(p, obj_pos, obj_mat, half) for p in points) - radius)


def contact_fraction(model: mujoco.MjModel, data: mujoco.MjData, qpos: np.ndarray, object_gids: list[int]) -> tuple[float, float, float]:
    hand_gids = {gid for name in HAND_GEOMS if (gid := geom_id(model, name)) >= 0}
    lower_gids = {gid for name in LOWERBODY_GEOMS if (gid := geom_id(model, name)) >= 0}
    object_set = set(object_gids)
    hand_contact = []
    lower_contact = []
    floor_contact = []
    floor = geom_id(model, "floor")
    for q in qpos:
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        h = l = fl = False
        for ci in range(data.ncon):
            con = data.contact[ci]
            pair = {int(con.geom1), int(con.geom2)}
            if object_set & pair:
                other = next(iter(pair - (object_set & pair))) if pair - (object_set & pair) else -1
                if other in hand_gids:
                    h = True
                if other in lower_gids:
                    l = True
                if floor >= 0 and floor in pair:
                    fl = True
        hand_contact.append(h)
        lower_contact.append(l)
        floor_contact.append(fl)
    return float(np.mean(hand_contact)), float(np.mean(lower_contact)), float(np.mean(floor_contact))


def lowerbody_proxy_metrics(model: mujoco.MjModel, data: mujoco.MjData, qpos: np.ndarray, object_gids: list[int]) -> dict[str, Any]:
    lower_gids = [gid for name in LOWERBODY_GEOMS if (gid := geom_id(model, name)) >= 0]
    vals = []
    arg_names = []
    for q in qpos:
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        best = (float("inf"), "")
        for lower_gid in lower_gids:
            for object_gid in object_gids:
                sdf = geom_box_adjusted_sdf(model, data, lower_gid, object_gid)
                if sdf < best[0]:
                    best = (sdf, f"{geom_name(model, lower_gid)}->{geom_name(model, object_gid)}")
        vals.append(best[0])
        arg_names.append(best[1])
    arr = np.asarray(vals, dtype=np.float64)
    argmin = int(np.argmin(arr))
    frac = float((arr < 0.0).mean())
    return {
        "leg_box_sdf_min_m": float(arr.min()),
        "leg_box_sdf_mean_m": float(arr.mean()),
        "leg_box_sdf_argmin": arg_names[argmin],
        "leg_box_interference_frac": frac,
        "leg_box_near_2cm_frac": float((arr < 0.02).mean()),
        "lowerbody_strict_pass": bool(frac <= LOWERBODY_INTERFERENCE_STRICT_FRAC),
    }


def compute_metrics(npz_path: Path, scene_xml: Path) -> dict[str, Any]:
    data_npz = np.load(npz_path, allow_pickle=True)
    qpos = np.asarray(data_npz["qpos"], dtype=np.float64)
    if qpos.ndim != 3 or qpos.shape[1] < 2:
        raise ValueError(f"expected qpos shape (T,2,nq), got {qpos.shape} in {npz_path}")
    sim_q = qpos[:, 0]
    ref_q = qpos[:, 1]
    T = min(len(sim_q), len(ref_q))
    sim_q = sim_q[:T]
    ref_q = ref_q[:T]

    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    data = mujoco.MjData(model)
    object_gids = object_collision_geoms(model)
    obj_b = body_id(model, "object")
    head_b = body_id(model, "head_link")
    upper_bs = [bid for name in UPPER_BODY_NAMES if (bid := body_id(model, name)) >= 0]
    wrist_l = body_id(model, "left_wrist_yaw_link")
    wrist_r = body_id(model, "right_wrist_yaw_link")

    pelvis_z = sim_q[:, 2]
    head_in = []
    upper_in = []
    hand_l_floor = []
    hand_r_floor = []
    hand_l_dist = []
    hand_r_dist = []
    obj_xyz_sim = []
    obj_xyz_ref = []
    for sq, rq in zip(sim_q, ref_q, strict=True):
        data.qpos[:] = sq
        mujoco.mj_forward(model, data)
        obj_xyz_sim.append(data.xpos[obj_b].copy())
        if head_b >= 0:
            head_in.append(point_distance_to_object_proxy(model, data, data.xpos[head_b].copy(), object_gids) < 0.0)
        else:
            head_in.append(False)
        upper_in.append(any(point_distance_to_object_proxy(model, data, data.xpos[bid].copy(), object_gids) < 0.0 for bid in upper_bs))
        for bid, out_floor, out_dist in [(wrist_l, hand_l_floor, hand_l_dist), (wrist_r, hand_r_floor, hand_r_dist)]:
            pos = data.xpos[bid].copy()
            quat = data.xquat[bid].copy()
            rot = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
            eef = pos + rot.apply(EEF_OFFSET)
            out_floor.append(float(eef[2]))
            out_dist.append(point_distance_to_object_proxy(model, data, eef, object_gids))

        data.qpos[:] = rq
        mujoco.mj_forward(model, data)
        obj_xyz_ref.append(data.xpos[obj_b].copy())

    obj_err = np.linalg.norm(np.asarray(obj_xyz_sim) - np.asarray(obj_xyz_ref), axis=1)
    hand_contact_physics, lower_contact_physics, floor_contact = contact_fraction(model, data, sim_q, object_gids)
    lower = lowerbody_proxy_metrics(model, data, sim_q, object_gids)
    replay = rg.evaluate(sim_q, scene_xml)
    contact_l = float((np.asarray(hand_l_dist) < 0.08).mean())
    contact_r = float((np.asarray(hand_r_dist) < 0.08).mean())
    metrics: dict[str, Any] = {
        "T": int(T),
        "obj_err_mean_m": float(obj_err.mean()),
        "obj_err_max_m": float(obj_err.max()),
        "pelvis_min_m": float(pelvis_z.min()),
        "head_pen_frac": float(np.mean(head_in)),
        "upper_pen_frac": float(np.mean(upper_in)),
        "handL_floor_lt_5cm_frac": float((np.asarray(hand_l_floor) < 0.05).mean()),
        "handR_floor_lt_5cm_frac": float((np.asarray(hand_r_floor) < 0.05).mean()),
        "contact_frac_L_8cm": contact_l,
        "contact_frac_R_8cm": contact_r,
        "contact_frac_either": max(contact_l, contact_r),
        "hand_object_contact_physics_frac": hand_contact_physics,
        "lowerbody_object_contact_physics_frac": lower_contact_physics,
        "object_floor_contact_frac": floor_contact,
        "replay_gate_pass": not bool(replay["gate_overall_fail"]),
        "pelvis_end_z_m": float(replay["pelvis_end_z"]),
        "pelvis_tilt_end_deg": float(replay["pelvis_tilt_end_deg"]),
        "lie_on_box_frac": float(replay["lie_on_box_frac"]),
        "gate_pelvis_low": bool(replay["gate_pelvis_low"]),
        "gate_pelvis_tilt": bool(replay["gate_pelvis_tilt"]),
        "gate_lie_on_box": bool(replay["gate_lie_on_box"]),
        "lowerbody_strict_threshold_frac": LOWERBODY_INTERFERENCE_STRICT_FRAC,
        **lower,
    }
    metrics["full_strict_pass"] = bool(
        metrics["T"] >= 80
        and metrics["obj_err_mean_m"] <= 0.10
        and metrics["obj_err_max_m"] <= 0.30
        and metrics["pelvis_min_m"] >= 0.55
        and max(
            metrics["head_pen_frac"],
            metrics["upper_pen_frac"],
            metrics["handL_floor_lt_5cm_frac"],
            metrics["handR_floor_lt_5cm_frac"],
        )
        <= 0.05
        and metrics["contact_frac_either"] >= 0.30
        and metrics["replay_gate_pass"]
        and metrics["lowerbody_strict_pass"]
    )
    if metrics["full_strict_pass"]:
        metrics["cem_status"] = "pass"
    elif (
        metrics["T"] >= 80
        and metrics["obj_err_mean_m"] <= 0.12
        and metrics["obj_err_max_m"] <= 0.35
        and metrics["pelvis_min_m"] >= 0.45
        and metrics["replay_gate_pass"]
    ):
        metrics["cem_status"] = "review"
    else:
        metrics["cem_status"] = "fail"
    return metrics


def write_outputs(results: dict[str, dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "E108_cem_eval_summary.json").write_text(
        json.dumps(results, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    fields = sorted({key for row in results.values() for key in row})
    with (out_dir / "E108_cem_eval_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results.values())
    lines = [
        "# E108 non-box bucket004 CEM 评估",
        "",
        "| variant | case | T | contact | obj mean | obj max | pelvis min | replay | lower-body | CEM |",
        "|---|---|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for row in results.values():
        lines.append(
            f"| `{row['variant']}` | `{row['case_id']}` | {row['T']} | "
            f"{row['contact_frac_either'] * 100:.1f}% | "
            f"{row['obj_err_mean_m']:.3f} | {row['obj_err_max_m']:.3f} | "
            f"{row['pelvis_min_m']:.3f} | {'PASS' if row['replay_gate_pass'] else 'FAIL'} | "
            f"{'PASS' if row['lowerbody_strict_pass'] else 'FAIL'} | {row['cem_status']} |"
        )
    lines.extend(
        [
            "",
            "## 失败/复核指标",
            "",
            "| variant | head | upper | LH floor | RH floor | leg interference | hand physics contact | object floor | replay detail |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in results.values():
        lines.append(
            f"| `{row['variant']}` | {row['head_pen_frac'] * 100:.1f}% | "
            f"{row['upper_pen_frac'] * 100:.1f}% | "
            f"{row['handL_floor_lt_5cm_frac'] * 100:.1f}% | "
            f"{row['handR_floor_lt_5cm_frac'] * 100:.1f}% | "
            f"{row['leg_box_interference_frac'] * 100:.1f}% | "
            f"{row['hand_object_contact_physics_frac'] * 100:.1f}% | "
            f"{row['object_floor_contact_frac'] * 100:.1f}% | "
            f"pelvis_end={row['pelvis_end_z_m']:.3f}, tilt_end={row['pelvis_tilt_end_deg']:.1f}, lie={row['lie_on_box_frac'] * 100:.1f}% |"
        )
    (out_dir / "E108_cem_eval_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["full", "smoke"], default="full")
    parser.add_argument("--variants-file", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("variants", nargs="*")
    args = parser.parse_args()

    variants_file = args.variants_file or repo_path_from_env("VARIANTS_FILE", DEFAULT_VARIANTS)
    results_dir = args.results_dir or repo_path_from_env("RESULTS", REPO / f"workspace/core4d/results/E108/s6_downstream/cem/{args.stage}")
    variants = read_variants(variants_file)
    selected = args.variants or list(variants)
    results: dict[str, dict[str, Any]] = {}
    for variant in selected:
        row = variants.get(variant)
        if not row:
            print(f"[SKIP] unknown variant {variant}")
            continue
        npz = results_dir / f"{variant}.npz"
        fallback = results_dir / f"{variant}_outdir_{args.stage}/trajectory_mjwp_act.npz"
        if not npz.is_file() and fallback.is_file():
            npz = fallback
        scene = TASK_ROOT / row["derived_task"] / "scene_act.xml"
        if not npz.is_file() or not scene.is_file():
            print(f"[SKIP] missing npz/scene for {variant}: {npz} {scene}")
            continue
        metrics = compute_metrics(npz, scene)
        metrics.update(
            {
                "variant": variant,
                "case_id": row["case_id"],
                "derived_task": row["derived_task"],
                "split": row["split"],
                "npz_path": str(npz),
                "scene_xml": str(scene),
                "video_path": str(results_dir / f"{variant}_{args.stage}.mp4"),
            }
        )
        results[variant] = metrics
    if not results:
        raise SystemExit("No E108 CEM results found.")
    write_outputs(results, results_dir)
    for variant, row in results.items():
        print(
            f"{variant}: cem={row['cem_status']} T={row['T']} "
            f"obj_mean={row['obj_err_mean_m']:.4f} obj_max={row['obj_err_max_m']:.4f} "
            f"contact={row['contact_frac_either']:.3f} lower={row['leg_box_interference_frac']:.3f}"
        )
    print(f"wrote {results_dir / 'E108_cem_eval_summary.md'}")


if __name__ == "__main__":
    main()
