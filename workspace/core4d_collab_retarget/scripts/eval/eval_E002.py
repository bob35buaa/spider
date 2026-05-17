#!/usr/bin/env python3
"""E002 evaluation for true-freejoint leg/foot-object variants."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import mujoco
import numpy as np


REPO = Path(__file__).resolve().parents[4]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
EVAL_DIR = REPO / "workspace/core4d/scripts/eval"
DEBUG_DIR = REPO / "workspace/core4d/scripts/debug"
for extra in (EVAL_DIR, DEBUG_DIR):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import eval_E072 as e072  # noqa: E402
import eval_E073 as e073  # noqa: E402
import eval_E079 as e079  # noqa: E402
import eval_E081 as e081  # noqa: E402
from diagnose_E068_init_drift import _build_config, _normalize_angle_deg  # noqa: E402
from spider.io import load_data  # noqa: E402


RESULTS = REPO / "workspace/core4d_collab_retarget/results/E002"
VARIANTS_FILE = REPO / "workspace/core4d_collab_retarget/scripts/E002/variants.tsv"
BASE = REPO / "example_datasets/processed/core4d/unitree_g1/humanoid_object"
FPS = 50.0


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
                "override": override_for_variant(row["variant"]),
                "split": row["split"],
                "role": row["role"],
            }
    return out


def override_for_variant(variant: str) -> str:
    if variant == "E002_box025_p2_freejoint":
        return "core4d_collab_E002_box025_p2_freejoint"
    if variant == "E002_box023_p2_freejoint":
        return "core4d_collab_E002_box023_p2_freejoint"
    return f"core4d_collab_{variant}"


def relative_or_abs(path: Path) -> str:
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def load_scene_model(case: str) -> tuple[mujoco.MjModel, Path]:
    snapshot = RESULTS / "scene_snapshot" / case / "scene.xml"
    active = BASE / case / "scene.xml"
    for path in (snapshot, active):
        if not path.is_file():
            continue
        try:
            return mujoco.MjModel.from_xml_path(str(path)), path
        except Exception:
            text = path.read_text(encoding="utf-8")
            patched = text.replace(
                "../../../../../../spider/assets",
                str(REPO / "spider/assets"),
            ).replace(
                "../../../../../example_datasets",
                str(REPO / "example_datasets"),
            )
            patched_path = RESULTS / f"_scene_abs_{case}.xml"
            patched_path.parent.mkdir(parents=True, exist_ok=True)
            patched_path.write_text(patched, encoding="utf-8")
            try:
                return mujoco.MjModel.from_xml_path(str(patched_path)), patched_path
            except Exception:
                continue
    raise FileNotFoundError(f"No loadable freejoint scene.xml for {case}")


def load_ref(override: str, case: str) -> tuple[np.ndarray, np.ndarray, object]:
    cfg = _build_config(override, case, "cpu", 4)
    if cfg.contact_guidance:
        raise ValueError(f"E002 override must have contact_guidance=false: {override}")
    qpos_ref, _qvel_ref, ctrl_ref, _contact, _contact_pos = load_data(cfg, cfg.data_path)
    return qpos_ref.detach().cpu().numpy(), ctrl_ref.detach().cpu().numpy(), cfg


def actuator_index(model: mujoco.MjModel, name: str) -> int:
    return int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name))


def foot_xy(model: mujoco.MjModel, qpos: np.ndarray) -> dict[str, np.ndarray]:
    data = mujoco.MjData(model)
    left_foot = e072.name2id(model, mujoco.mjtObj.mjOBJ_SITE, "left_foot")
    right_foot = e072.name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_foot")
    out = {
        "left": np.zeros((len(qpos), 2), dtype=float),
        "right": np.zeros((len(qpos), 2), dtype=float),
    }
    for t, q in enumerate(qpos):
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        out["left"][t] = data.site_xpos[left_foot, :2]
        out["right"][t] = data.site_xpos[right_foot, :2]
    return out


def write_timeseries(variant: str, rows: list[dict[str, object]]) -> None:
    path = RESULTS / f"timeseries_{variant}.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_variant_summary(summary: dict[str, object]) -> None:
    variant = str(summary["variant"])
    summary_json = RESULTS / f"eval_summary_{variant}.json"
    summary_csv = RESULTS / f"eval_summary_{variant}.csv"
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)


def _ctrl_diff_arrays(ctrl: np.ndarray, ctrl_ref: np.ndarray, T: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dim = min(ctrl.shape[1], ctrl_ref.shape[1])
    diff = np.zeros((T, max(ctrl.shape[1], ctrl_ref.shape[1])), dtype=np.float64)
    diff[:, :dim] = ctrl[:T, :dim] - ctrl_ref[:T, :dim]
    robot_dim = min(29, diff.shape[1])
    robot_linf = np.max(np.abs(diff[:, :robot_dim]), axis=1)
    robot_l2 = np.linalg.norm(diff[:, :robot_dim], axis=1)
    return diff, robot_linf, robot_l2


def evaluate_variant(variant: str, variants: dict[str, dict[str, str]]) -> dict[str, object]:
    if variant not in variants:
        raise ValueError(f"Unknown E002 variant: {variant}")
    meta = variants[variant]
    name = str(meta["name"])
    case = str(meta["case"])
    override = str(meta["override"])
    npz_path = RESULTS / f"{name}.npz"
    if not npz_path.is_file():
        raise FileNotFoundError(npz_path)

    model, scene_used = load_scene_model(case)
    data_npz = np.load(npz_path, allow_pickle=True)
    qpos = e072.flatten_time_major(data_npz["qpos"])
    qvel = e072.flatten_time_major(data_npz["qvel"])
    ctrl = e072.flatten_time_major(data_npz["ctrl"])
    time = e072.flatten_time_major(data_npz["time"]).astype(float)
    opt_steps = np.repeat(data_npz["opt_steps"].reshape(-1), 2)
    qpos_ref, ctrl_ref, cfg = load_ref(override, case)

    T = min(
        len(qpos),
        len(qvel),
        len(ctrl),
        len(time),
        len(opt_steps),
        len(qpos_ref),
        len(ctrl_ref),
    )
    qpos, qvel, ctrl, time, opt_steps = qpos[:T], qvel[:T], ctrl[:T], time[:T], opt_steps[:T]
    qpos_ref, ctrl_ref = qpos_ref[:T], ctrl_ref[:T]

    sim = e072.replay_metrics(model, qpos)
    ref = e072.replay_metrics(model, qpos_ref)
    sim_foot_xy = foot_xy(model, qpos)
    ref_foot_xy = foot_xy(model, qpos_ref)

    obj_err = np.linalg.norm(sim["obj_pos"] - ref["obj_pos"], axis=1)
    pelvis_err = np.linalg.norm(sim["pelvis_pos"] - ref["pelvis_pos"], axis=1)
    sim_min_hand_sdf = np.minimum(sim["left_sdf_m"], sim["right_sdf_m"])
    ref_min_hand_sdf = np.minimum(ref["left_sdf_m"], ref["right_sdf_m"])
    sim_total_contacts = sim["left_contact_count"] + sim["right_contact_count"]
    ref_total_contacts = ref["left_contact_count"] + ref["right_contact_count"]
    ctrl_diff, robot_ctrl_linf, robot_ctrl_l2 = _ctrl_diff_arrays(ctrl, ctrl_ref, T)
    object_ctrl_linf = np.zeros(T)
    object_ctrl_l2 = np.zeros(T)
    qpos_abs_max = np.max(np.abs(qpos - qpos_ref), axis=1)

    rhip_idx = actuator_index(model, "right_hip_pitch_joint")
    rhip_diff = ctrl_diff[:, rhip_idx] if 0 <= rhip_idx < ctrl_diff.shape[1] else np.zeros(T)
    sim_rstep = np.zeros(T)
    ref_rstep = np.zeros(T)
    sim_rstep[1:] = np.linalg.norm(np.diff(sim_foot_xy["right"], axis=0), axis=1)
    ref_rstep[1:] = np.linalg.norm(np.diff(ref_foot_xy["right"], axis=0), axis=1)

    rows: list[dict[str, object]] = []
    for i in range(T):
        rows.append(
            {
                "variant": variant,
                "case": case,
                "frame": i,
                "eval_time_s": float(i / FPS),
                "npz_time_s": float(time[i]),
                "opt_step": int(opt_steps[i]),
                "obj_err_m": float(obj_err[i]),
                "pelvis_err_m": float(pelvis_err[i]),
                "pelvis_z_qpos_m": float(qpos[i, 2]),
                "sim_pelvis_z_m": float(sim["pelvis_pos"][i, 2]),
                "ref_pelvis_z_m": float(ref["pelvis_pos"][i, 2]),
                "sim_object_x_m": float(sim["obj_pos"][i, 0]),
                "sim_object_y_m": float(sim["obj_pos"][i, 1]),
                "sim_object_z_m": float(sim["obj_pos"][i, 2]),
                "ref_object_x_m": float(ref["obj_pos"][i, 0]),
                "ref_object_y_m": float(ref["obj_pos"][i, 1]),
                "ref_object_z_m": float(ref["obj_pos"][i, 2]),
                "sim_left_sdf_m": float(sim["left_sdf_m"][i]),
                "sim_right_sdf_m": float(sim["right_sdf_m"][i]),
                "sim_min_hand_sdf_m": float(sim_min_hand_sdf[i]),
                "ref_left_sdf_m": float(ref["left_sdf_m"][i]),
                "ref_right_sdf_m": float(ref["right_sdf_m"][i]),
                "ref_min_hand_sdf_m": float(ref_min_hand_sdf[i]),
                "sim_left_contact_count": int(sim["left_contact_count"][i]),
                "sim_right_contact_count": int(sim["right_contact_count"][i]),
                "sim_total_contact_count": int(sim_total_contacts[i]),
                "sim_object_floor_contact_count": int(sim["object_floor_contact_count"][i]),
                "sim_all_contact_count": int(sim["total_contact_count"][i]),
                "ref_left_contact_count": int(ref["left_contact_count"][i]),
                "ref_right_contact_count": int(ref["right_contact_count"][i]),
                "ref_total_contact_count": int(ref_total_contacts[i]),
                "ref_object_floor_contact_count": int(ref["object_floor_contact_count"][i]),
                "ref_all_contact_count": int(ref["total_contact_count"][i]),
                "sim_left_foot_z_m": float(sim["left_foot_z"][i]),
                "sim_right_foot_z_m": float(sim["right_foot_z"][i]),
                "ref_left_foot_z_m": float(ref["left_foot_z"][i]),
                "ref_right_foot_z_m": float(ref["right_foot_z"][i]),
                "sim_right_foot_xy_step_m": float(sim_rstep[i]),
                "ref_right_foot_xy_step_m": float(ref_rstep[i]),
                "right_hip_pitch_ctrl_diff": float(rhip_diff[i]),
                "robot_ctrl_linf": float(robot_ctrl_linf[i]),
                "robot_ctrl_l2": float(robot_ctrl_l2[i]),
                "object_ctrl_linf": float(object_ctrl_linf[i]),
                "object_ctrl_l2": float(object_ctrl_l2[i]),
                "qpos_max_abs_diff_vs_ref": float(qpos_abs_max[i]),
                "qvel_max_abs": float(np.max(np.abs(qvel[i]))),
            }
        )
    write_timeseries(variant, rows)

    start = min(int(2.0 * FPS), T - 1)
    end = min(int(3.6 * FPS) + 1, T)
    post = slice(start, end)
    frame100_145 = slice(min(100, T - 1), min(146, T))
    frame115_130 = slice(min(115, T - 1), min(131, T))
    frame119_125_steps = slice(min(120, T - 1), min(126, T))
    frame120_124 = slice(min(120, T - 1), min(125, T))
    pre_end = min(int(2.0 * FPS) + 1, T)

    summary: dict[str, object] = {
        "variant": variant,
        "case": case,
        "source_task": meta["source_task"],
        "override": override,
        "npz_path": relative_or_abs(npz_path),
        "scene_used": relative_or_abs(scene_used),
        "scene_mode": "scene.xml_freejoint",
        "T": T,
        "config_contact_guidance": bool(cfg.contact_guidance),
        "config_scene_name": str(cfg.scene_name),
        "config_nq": int(cfg.nq),
        "config_nv": int(cfg.nv),
        "config_nu": int(cfg.nu),
        "config_nq_obj": int(cfg.nq_obj),
        "config_object_action_dims": int(cfg.object_action_dims),
        "config_object_actuator_ids": list(cfg.object_actuator_ids),
        "model_nq": int(model.nq),
        "model_nv": int(model.nv),
        "model_nu": int(model.nu),
        "model_npair": int(model.npair),
        "yaw_err_t0017_deg": float(_normalize_angle_deg(e073.yaw_deg(qpos[0, 3:7]) - e073.yaw_deg(qpos_ref[min(1, T - 1), 3:7]))),
        "yaw_err_t0033_deg": float(_normalize_angle_deg(e073.yaw_deg(qpos[min(1, T - 1), 3:7]) - e073.yaw_deg(qpos_ref[min(2, T - 1), 3:7]))),
        "B1_pre_contact_max_foot_z_m": float(max(sim["left_foot_z"][:pre_end].max(), sim["right_foot_z"][:pre_end].max())),
        "B2_single_foot_runs": e073.count_single_foot_runs(sim["left_foot_z"], sim["right_foot_z"]),
        "first_obj_err_gt_25cm_frame": e072.first_after(obj_err > 0.25, start),
        "first_sim_zero_contact_frame": e072.first_after(sim_total_contacts == 0, start),
        "first_sim_min_hand_sdf_gt_10cm_frame": e072.first_after(sim_min_hand_sdf > 0.10, start),
        "first_pelvis_z_lt_45cm_frame": e072.first_after(sim["pelvis_pos"][:, 2] < 0.45, start),
        "first_robot_ctrl_linf_gt_0p5_frame": e072.first_after(robot_ctrl_linf > 0.50, start),
        "post2_obj_err_max_m": float(obj_err[post].max()),
        "post2_obj_err_mean_m": float(obj_err[post].mean()),
        "post2_sim_contact_frames_pct": float((sim_total_contacts[post] > 0).mean() * 100.0),
        "post2_ref_contact_frames_pct": float((ref_total_contacts[post] > 0).mean() * 100.0),
        "post2_sim_min_hand_sdf_mean_m": float(sim_min_hand_sdf[post].mean()),
        "post2_sim_min_hand_sdf_max_m": float(sim_min_hand_sdf[post].max()),
        "post2_pelvis_z_min_m": float(sim["pelvis_pos"][post, 2].min()),
        "post2_robot_ctrl_linf_max": float(robot_ctrl_linf[post].max()),
        "post2_object_ctrl_linf_max": float(object_ctrl_linf[post].max()),
        "frame100_145_contact_frames_pct": float((sim_total_contacts[frame100_145] > 0).mean() * 100.0),
        "frame100_145_ref_contact_frames_pct": float((ref_total_contacts[frame100_145] > 0).mean() * 100.0),
        "frame100_145_obj_err_max_m": float(obj_err[frame100_145].max()),
        "frame115_130_sim_right_foot_xy_step_sum_m": float(sim_rstep[frame115_130].sum()),
        "frame115_130_ref_right_foot_xy_step_sum_m": float(ref_rstep[frame115_130].sum()),
        "frame119_125_sim_right_foot_xy_step_sum_m": float(sim_rstep[frame119_125_steps].sum()),
        "frame119_125_ref_right_foot_xy_step_sum_m": float(ref_rstep[frame119_125_steps].sum()),
        "frame119_125_sim_right_foot_xy_step_max_m": float(sim_rstep[frame119_125_steps].max()),
        "frame119_125_ref_right_foot_xy_step_max_m": float(ref_rstep[frame119_125_steps].max()),
        "frame120_124_right_hip_pitch_ctrl_diff_mean": float(rhip_diff[frame120_124].mean()),
        "frame120_124_right_hip_pitch_ctrl_diff_abs_max": float(np.abs(rhip_diff[frame120_124]).max()),
    }
    for key in [k for k in summary if k.endswith("_frame")]:
        e078_frame_time(summary, key, time)

    e079.RESULTS = RESULTS
    summary.update(e079.case_window_metrics(summary))
    sim_leg_rows = e081.leg_object_timeseries(model, qpos)
    ref_leg_rows = e081.leg_object_timeseries(model, qpos_ref)
    e081.RESULTS = RESULTS
    e081.write_leg_timeseries(variant, sim_leg_rows, ref_leg_rows)
    cw_start = int(summary["case_window_start_frame"])
    cw_end = min(int(summary["case_window_end_frame"]), T - 1)
    summary["legobj_geom_count"] = len(e081.LEG_FOOT_GEOMS)
    summary.update(e081._summarize_prefix(sim_leg_rows, 0, T - 1, "full_sim"))
    summary.update(e081._summarize_prefix(ref_leg_rows, 0, T - 1, "full_ref"))
    summary.update(e081._summarize_prefix(sim_leg_rows, cw_start, cw_end, "case_window_sim"))
    summary.update(e081._summarize_prefix(ref_leg_rows, cw_start, cw_end, "case_window_ref"))
    summary["case_window_object_bottom_mean_gap_vs_ref_m"] = float(
        summary["case_window_sim_object_bottom_proxy_mean_m"]
        - summary["case_window_ref_object_bottom_proxy_mean_m"]
    )
    summary["E002_success_numeric"] = bool(
        summary["post2_pelvis_z_min_m"] >= 0.55
        and summary["post2_sim_contact_frames_pct"] >= 50.0
        and summary["post2_obj_err_mean_m"] <= 0.20
    )
    summary["E002_success_case_window"] = bool(summary["E079_success_case_window"])
    summary["E002_success_legobj_strict_proxy"] = bool(
        summary["E002_success_case_window"]
        and summary["case_window_sim_leg_box_interference_frames_pct"] <= 5.0
        and summary["case_window_sim_object_bottom_proxy_mean_m"]
        >= summary["case_window_ref_object_bottom_proxy_mean_m"] - 0.05
    )
    summary["split"] = meta["split"]
    summary["role"] = meta["role"]

    e072.plot_timeline(rows, RESULTS / "plots" / f"{variant}_post2_failure_timeline.png")
    write_variant_summary(summary)
    return summary


def e078_frame_time(summary: dict[str, object], key: str, time: np.ndarray) -> None:
    frame = int(summary[key])
    summary[key.replace("_frame", "_eval_time_s")] = e072.eval_time_or_neg(frame)
    summary[key.replace("_frame", "_npz_time_s")] = e072.npz_time_or_neg(frame, time)


def main() -> None:
    variants = read_variants()
    selected = sys.argv[1:] or list(variants.keys())

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "plots").mkdir(parents=True, exist_ok=True)

    summaries = []
    for variant in selected:
        if variant not in variants:
            print(f"[SKIP] unknown variant {variant}")
            continue
        try:
            summary = evaluate_variant(variant, variants)
        except FileNotFoundError as exc:
            print(f"[SKIP] {exc}")
            continue
        write_variant_summary(summary)
        summaries.append(summary)

    if not summaries:
        raise SystemExit("No E002 variant results found.")

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
        "num_main_numeric_success": sum(bool(r["E002_success_numeric"]) for r in main_rows),
        "main_numeric_success_pct": (
            100.0 * sum(bool(r["E002_success_numeric"]) for r in main_rows) / len(main_rows)
            if main_rows
            else 0.0
        ),
        "num_main_case_window_success": sum(bool(r["E002_success_case_window"]) for r in main_rows),
        "main_case_window_success_pct": (
            100.0 * sum(bool(r["E002_success_case_window"]) for r in main_rows) / len(main_rows)
            if main_rows
            else 0.0
        ),
        "num_main_legobj_strict_proxy_success": sum(
            bool(r["E002_success_legobj_strict_proxy"]) for r in main_rows
        ),
        "main_legobj_strict_proxy_success_pct": (
            100.0
            * sum(bool(r["E002_success_legobj_strict_proxy"]) for r in main_rows)
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
