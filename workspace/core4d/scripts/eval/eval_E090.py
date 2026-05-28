#!/usr/bin/env python3
"""Evaluate E090 SPIDER smoke rollouts.

This intentionally uses the E089 headline metric style instead of the
mask-driven E083 evaluator, because the repaired topface-preIK tasks do not
have a raw CORE4D 3cm contact mask. Smoke pass/fail is about dynamic safety:
head/upper penetration, hand-floor shortcuts, object tracking, and pelvis.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R


os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")

REPO = Path(__file__).resolve().parents[4]
TASK_ROOT = REPO / "example_datasets/processed/core4d/unitree_g1/humanoid_object"
DEFAULT_VARIANTS = REPO / "workspace/core4d/scripts/E090/variants_smoke.tsv"
GEOMETRY_CSV = REPO / "workspace/core4d/results/E090/geometry/geometry_summary.csv"
EVAL_SUMMARY = REPO / "workspace/core4d/results/E090/eval_summary.json"
EEF_OFFSET = np.array([0.05, 0.0, 0.0], dtype=np.float64)

BASELINES = {
    "E085A_raw_main_29kg": {"contact": 0.829, "obj_mean_m": 0.665, "pelvis_min_m": 0.659, "head_pen": 0.186, "upper_pen": 0.535, "hand_floor_max": 0.736},
    "E087B_10kg_raw_main": {"contact": 0.829, "obj_mean_m": 0.735, "pelvis_min_m": 0.629, "head_pen": 0.698, "upper_pen": 0.760, "hand_floor_max": 0.744},
    "E088A_m10_gate": {"contact": 0.822, "obj_mean_m": 0.695, "pelvis_min_m": 0.181, "head_pen": 0.279, "upper_pen": 0.550, "hand_floor_max": 0.108},
    "E089A_box021_person1_full": {"contact": 0.602, "obj_mean_m": 0.013, "pelvis_min_m": 0.687, "head_pen": 0.0, "upper_pen": 0.0, "hand_floor_max": 0.0},
}


def _repo_path_from_env(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    path = Path(raw) if raw else default
    return path if path.is_absolute() else REPO / path


def _read_variants(path: Path) -> dict[str, dict[str, str]]:
    fieldnames = [
        "variant",
        "source_task",
        "derived_task",
        "mask_source_dir",
        "mask_slug",
        "person_idx",
        "split",
        "role",
    ]
    out: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(
            (line for line in f if line.strip() and not line.startswith("#")),
            delimiter="\t",
            fieldnames=fieldnames,
        )
        for row in reader:
            out[row["variant"]] = row
    return out


def _parse_box_half(scene_xml: Path) -> np.ndarray:
    tree = ET.parse(scene_xml)
    root = tree.getroot()
    for body in root.iter("body"):
        if body.get("name") != "object":
            continue
        for geom in body.iter("geom"):
            if geom.get("type") == "box" and "collision" in geom.get("name", "").lower():
                return np.array([float(x) for x in geom.get("size", "").split()], dtype=np.float64)
    raise ValueError(f"object collision box not found in {scene_xml}")


def _body_id(model: mujoco.MjModel, name: str) -> int:
    return int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name))


def compute_sim_metrics(npz_path: Path, scene_xml: Path) -> dict[str, object]:
    data_npz = np.load(npz_path, allow_pickle=True)
    qpos = data_npz["qpos"]
    if qpos.ndim != 3 or qpos.shape[1] < 2:
        raise ValueError(f"expected qpos shape (T,2,nq), got {qpos.shape} in {npz_path}")
    sim_q = qpos[:, 0]
    # Keep the E089 metric convention: the rollout file carries the target
    # object trajectory used by run_mjwp in qpos[:, 1]. The raw source ref can
    # have a different resampling horizon, so it is not used for object error.
    ref_q = qpos[:, 1]
    T = min(len(sim_q), len(ref_q))
    sim_q = sim_q[:T]
    ref_q = ref_q[:T]

    half = _parse_box_half(scene_xml)
    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    mj = mujoco.MjData(model)
    obj_b = _body_id(model, "object")
    head_b = _body_id(model, "head_link")
    upper_bs = [
        bid
        for name in [
            "torso_link",
            "left_shoulder_yaw_link",
            "right_shoulder_yaw_link",
            "left_shoulder_pitch_link",
            "right_shoulder_pitch_link",
            "pelvis_contour_link",
            "left_elbow_link",
            "right_elbow_link",
        ]
        for bid in [_body_id(model, name)]
        if bid != -1
    ]
    wrist_l = _body_id(model, "left_wrist_yaw_link")
    wrist_r = _body_id(model, "right_wrist_yaw_link")

    def in_box(pos_world: np.ndarray, obj_pos: np.ndarray, obj_mat: np.ndarray) -> bool:
        local = obj_mat.T @ (pos_world - obj_pos)
        return bool(np.all(np.abs(local) < half))

    pelvis_z = sim_q[:, 2]
    head_in = np.zeros(T, dtype=bool)
    upper_in = np.zeros(T, dtype=bool)
    hand_l_floor_z = np.zeros(T, dtype=np.float64)
    hand_r_floor_z = np.zeros(T, dtype=np.float64)
    hand_l_world = np.zeros((T, 3), dtype=np.float64)
    hand_r_world = np.zeros((T, 3), dtype=np.float64)
    obj_xyz_sim = np.zeros((T, 3), dtype=np.float64)
    obj_xyz_ref = np.zeros((T, 3), dtype=np.float64)

    for t in range(T):
        mj.qpos[:] = sim_q[t]
        mujoco.mj_forward(model, mj)
        obj_pos = mj.xpos[obj_b].copy()
        obj_mat = mj.xmat[obj_b].reshape(3, 3).copy()
        obj_xyz_sim[t] = obj_pos
        if head_b != -1:
            head_in[t] = in_box(mj.xpos[head_b].copy(), obj_pos, obj_mat)
        upper_in[t] = any(in_box(mj.xpos[bid].copy(), obj_pos, obj_mat) for bid in upper_bs)
        for bid, store in [(wrist_l, hand_l_world), (wrist_r, hand_r_world)]:
            pos = mj.xpos[bid].copy()
            quat = mj.xquat[bid].copy()
            rot = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
            store[t] = pos + rot.apply(EEF_OFFSET)
        hand_l_floor_z[t] = hand_l_world[t, 2]
        hand_r_floor_z[t] = hand_r_world[t, 2]

        mj.qpos[:] = ref_q[t]
        mujoco.mj_forward(model, mj)
        obj_xyz_ref[t] = mj.xpos[obj_b].copy()

    def contact_frac(hand_world: np.ndarray) -> float:
        near = 0
        for t in range(T):
            mj.qpos[:] = sim_q[t]
            mujoco.mj_forward(model, mj)
            obj_pos = mj.xpos[obj_b].copy()
            obj_mat = mj.xmat[obj_b].reshape(3, 3).copy()
            local = obj_mat.T @ (hand_world[t] - obj_pos)
            outside = np.maximum(np.abs(local) - half, 0.0)
            if float(np.linalg.norm(outside)) < 0.08:
                near += 1
        return near / T if T else 0.0

    obj_err = np.linalg.norm(obj_xyz_sim - obj_xyz_ref, axis=1)
    c_l = contact_frac(hand_l_world)
    c_r = contact_frac(hand_r_world)
    return {
        "T": int(T),
        "pelvis_min_m": float(pelvis_z.min()),
        "head_pen_frac": float(head_in.mean()),
        "upper_pen_frac": float(upper_in.mean()),
        "handL_floor_lt_5cm_frac": float((hand_l_floor_z < 0.05).mean()),
        "handR_floor_lt_5cm_frac": float((hand_r_floor_z < 0.05).mean()),
        "obj_err_mean_m": float(obj_err.mean()),
        "obj_err_max_m": float(obj_err.max()),
        "contact_frac_L_8cm": float(c_l),
        "contact_frac_R_8cm": float(c_r),
        "contact_frac_either": float(max(c_l, c_r)),
    }


def _load_geometry_rows() -> list[dict[str, str]]:
    if not GEOMETRY_CSV.is_file():
        return []
    with GEOMETRY_CSV.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _passes_stage(stage: str, metrics: dict[str, object]) -> bool:
    if stage == "smoke":
        return bool(
            float(metrics["head_pen_frac"]) <= 0.10
            and float(metrics["upper_pen_frac"]) <= 0.10
            and float(metrics["handL_floor_lt_5cm_frac"]) <= 0.10
            and float(metrics["handR_floor_lt_5cm_frac"]) <= 0.10
            and float(metrics["obj_err_mean_m"]) <= 0.10
        )
    return bool(
        float(metrics["head_pen_frac"]) <= 0.05
        and float(metrics["upper_pen_frac"]) <= 0.05
        and float(metrics["handL_floor_lt_5cm_frac"]) <= 0.05
        and float(metrics["handR_floor_lt_5cm_frac"]) <= 0.05
        and float(metrics["obj_err_mean_m"]) <= 0.10
        and float(metrics["pelvis_min_m"]) >= 0.55
    )


def _print_table(results: dict[str, dict[str, object]]) -> None:
    print("=" * 112)
    print(
        f"{'variant':<38} {'T':>4} {'cont':>7} {'obj_mean':>9} {'pelv_min':>9} "
        f"{'head':>7} {'upper':>7} {'LH_fl':>7} {'RH_fl':>7} {'pass':>6}"
    )
    print("-" * 112)
    for variant, m in results.items():
        print(
            f"{variant:<38} {int(m['T']):>4} "
            f"{float(m['contact_frac_either']) * 100:>6.1f}% "
            f"{float(m['obj_err_mean_m']):>8.3f}m "
            f"{float(m['pelvis_min_m']):>8.3f}m "
            f"{float(m['head_pen_frac']) * 100:>6.1f}% "
            f"{float(m['upper_pen_frac']) * 100:>6.1f}% "
            f"{float(m['handL_floor_lt_5cm_frac']) * 100:>6.1f}% "
            f"{float(m['handR_floor_lt_5cm_frac']) * 100:>6.1f}% "
            f"{str(m['stage_pass']):>6}"
        )
    print("-" * 112)
    print("Baselines:")
    for name, b in BASELINES.items():
        print(
            f"{name:<38} {'-':>4} {b['contact'] * 100:>6.1f}% "
            f"{b['obj_mean_m']:>8.3f}m {b['pelvis_min_m']:>8.3f}m "
            f"{b['head_pen'] * 100:>6.1f}% {b['upper_pen'] * 100:>6.1f}% "
            f"{b['hand_floor_max'] * 100:>6.1f}% {'-':>7} {'-':>6}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["smoke", "full"], default="smoke")
    parser.add_argument("variants", nargs="*")
    args = parser.parse_args()

    default_results = REPO / f"workspace/core4d/results/E090/{args.stage}"
    results_dir = _repo_path_from_env("RESULTS", default_results)
    variants_file = _repo_path_from_env("VARIANTS_FILE", DEFAULT_VARIANTS)
    variants = _read_variants(variants_file)
    selected = args.variants or list(variants)

    results: dict[str, dict[str, object]] = {}
    for variant in selected:
        if variant not in variants:
            print(f"[SKIP] unknown E090 variant {variant}")
            continue
        row = variants[variant]
        npz = results_dir / f"{variant}_outdir_{args.stage}/trajectory_mjwp_act.npz"
        scene = TASK_ROOT / row["derived_task"] / "scene_act.xml"
        if not npz.is_file():
            print(f"[SKIP] missing rollout {npz}")
            continue
        metrics = compute_sim_metrics(npz, scene)
        metrics.update(
            {
                "variant": variant,
                "source_task": row["source_task"],
                "derived_task": row["derived_task"],
                "npz_path": str(npz.relative_to(REPO)),
                "scene_xml": str(scene.relative_to(REPO)),
            }
        )
        metrics["stage"] = args.stage
        metrics["stage_pass"] = _passes_stage(args.stage, metrics)
        # Keep the smoke key for backward compatibility with the first E090 run.
        if args.stage == "smoke":
            metrics["smoke_pass"] = metrics["stage_pass"]
        else:
            metrics["full_pass"] = metrics["stage_pass"]
        results[variant] = metrics

    if not results:
        raise SystemExit("No E090 smoke rollout results found.")

    results_dir.mkdir(parents=True, exist_ok=True)
    out_json = results_dir / f"{args.stage}_eval_summary.json"
    out_json.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    keys = sorted({key for row in results.values() for key in row})
    out_csv = results_dir / f"{args.stage}_eval_summary.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results.values())

    geometry_rows = _load_geometry_rows()
    summary = json.loads(EVAL_SUMMARY.read_text(encoding="utf-8")) if EVAL_SUMMARY.is_file() else {}
    summary[f"E090_{args.stage.upper()}"] = {
        "results": results,
        "num_results": len(results),
        "num_pass": sum(bool(row["stage_pass"]) for row in results.values()),
        "geometry_rows": geometry_rows,
        "baselines": BASELINES,
    }
    EVAL_SUMMARY.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    _print_table(results)
    print(f"wrote {out_json.relative_to(REPO)}")
    print(f"wrote {out_csv.relative_to(REPO)}")
    print(f"updated {EVAL_SUMMARY.relative_to(REPO)}")


if __name__ == "__main__":
    main()
