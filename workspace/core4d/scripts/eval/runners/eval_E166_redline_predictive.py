#!/usr/bin/env python
"""E166 Phase0: Tier-1 redline metrics vs downstream staggered success.

This is a training-free audit. It reads SUGAR handoff motions, downstream
staggered eval CSVs, and Isaac contact probes, then checks whether motion
smoothness / foot consistency predict downstream success better than contact.
"""

from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ANKLES = [3, 6]
WRISTS = [10, 13]
TRACKED = ANKLES + WRISTS


@dataclass(frozen=True)
class CaseSpec:
    name: str
    short_case: str
    motion_folder: str
    eval_rel: str | None
    contact_probe_rel: str
    success_override: float | None = None


CASES = [
    CaseSpec(
        name="box021_r160(029_p2)",
        short_case="box021_029_p2",
        motion_folder="Core4D_E163N_Box021_R160",
        eval_rel="e163_refiner_rl/box021_r160/spider_e163/eval_staggered_phase_mw30",
        contact_probe_rel="core4d_e163_threecase_contact_probe/box021_r160/isaac_contact_framewise.csv",
    ),
    CaseSpec(
        name="box021_035_p1",
        short_case="box021_035_p1",
        motion_folder="Core4D_E163N8_d003_box021_20231011_035_p1",
        eval_rel=(
            "e163_spider_e163_8case_refiner_rl/d003_box021_20231011_035_p1/"
            "spider_e163/eval_staggered_phase_mw30_latest_checkpoint"
        ),
        contact_probe_rel=(
            "core4d/e163_spider_e163_8case_refiner_rl/d003_box021_20231011_035_p1/"
            "spider_e163/contact_probe/data_000/isaac_contact_framewise.csv"
        ),
    ),
    CaseSpec(
        name="box021_035_p2",
        short_case="box021_035_p2",
        motion_folder="Core4D_E163N8_d003_box021_20231011_035_p2",
        eval_rel=(
            "e163_spider_e163_8case_refiner_rl/d003_box021_20231011_035_p2/"
            "spider_e163/eval_staggered_phase_mw30_latest_checkpoint"
        ),
        contact_probe_rel=(
            "core4d/e163_spider_e163_8case_refiner_rl/d003_box021_20231011_035_p2/"
            "spider_e163/contact_probe/data_000/isaac_contact_framewise.csv"
        ),
    ),
    CaseSpec(
        name="box004_r161(083_p2)",
        short_case="box004_083_p2",
        motion_folder="Core4D_E163N_Box004_R161",
        eval_rel="e163_refiner_rl/box004_r161/spider_e163/eval_staggered_phase_mw30",
        contact_probe_rel="core4d_e163_threecase_contact_probe/box004_r161/isaac_contact_framewise.csv",
    ),
    CaseSpec(
        name="box004_082_p1",
        short_case="box004_082_p1",
        motion_folder="Core4D_E163N8_e091_box004_20231003_2_082_p1",
        eval_rel=(
            "e163_spider_e163_8case_refiner_rl/e091_box004_20231003_2_082_p1/"
            "spider_e163/eval_staggered_phase_mw30_latest_checkpoint"
        ),
        contact_probe_rel=(
            "core4d/e163_spider_e163_8case_refiner_rl/e091_box004_20231003_2_082_p1/"
            "spider_e163/contact_probe/data_000/isaac_contact_framewise.csv"
        ),
    ),
    CaseSpec(
        name="box004_083_p1",
        short_case="box004_083_p1",
        motion_folder="Core4D_E163N8_e091_box004_20231003_2_083_p1",
        eval_rel=(
            "e163_spider_e163_8case_refiner_rl/e091_box004_20231003_2_083_p1/"
            "spider_e163/eval_staggered_phase_mw30_latest_checkpoint"
        ),
        contact_probe_rel=(
            "core4d/e163_spider_e163_8case_refiner_rl/e091_box004_20231003_2_083_p1/"
            "spider_e163/contact_probe/data_000/isaac_contact_framewise.csv"
        ),
    ),
    CaseSpec(
        name="box023_r158",
        short_case="box023_person2",
        motion_folder="Core4D_E163N_Box023_R158",
        eval_rel="e163_refiner_rl/box023_r158/spider_e163/eval_staggered_phase_mw30",
        contact_probe_rel="core4d_e163_threecase_contact_probe/box023_r158/isaac_contact_framewise.csv",
        success_override=0.0,
    ),
    CaseSpec(
        name="box026_139_p1",
        short_case="box026_139_p1",
        motion_folder="Core4D_E163N8_e091_box026_20231023_139_p1",
        eval_rel=None,
        contact_probe_rel=(
            "core4d/e163_spider_e163_8case_refiner_rl/e091_box026_20231023_139_p1/"
            "spider_e163/contact_probe/data_000/isaac_contact_framewise.csv"
        ),
    ),
]


def read_success(sugar_outputs: Path, case: CaseSpec) -> tuple[float | None, str | None, int | None]:
    if case.success_override is not None:
        return case.success_override, "override", 64
    if case.eval_rel is None:
        return None, None, None
    path = sugar_outputs / "core4d" / case.eval_rel / "analysis" / "success_vs_phase.csv"
    if not path.exists():
        return None, str(path), None
    df = pd.read_csv(path)
    return float(df["success"].mean()), str(path), int(len(df))


def load_motion(sugar_root: Path, motion_folder: str) -> tuple[dict[str, Any], dict[str, Any]]:
    data_dir = sugar_root / "data" / motion_folder / "data_000"
    robot_npz = data_dir / "robot_50hz.npz"
    obj_pkl = data_dir / "obj_motion_global_50hz.pkl"
    if not robot_npz.exists():
        raise FileNotFoundError(robot_npz)
    z = np.load(robot_npz, allow_pickle=True)
    with obj_pkl.open("rb") as fp:
        obj = pickle.load(fp)
    return {k: z[k] for k in z.files}, obj


def contiguous_segments(mask: np.ndarray) -> list[tuple[int, int]]:
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return []
    breaks = np.where(np.diff(idx) > 1)[0]
    starts = np.r_[idx[0], idx[breaks + 1]]
    ends = np.r_[idx[breaks], idx[-1]]
    return [(int(s), int(e) + 1) for s, e in zip(starts, ends)]


def foot_metrics(body_pos_w: np.ndarray, fps: float) -> dict[str, float]:
    ankles = body_pos_w[:, ANKLES]
    dt = 1.0 / fps
    ground_z = np.percentile(ankles[:, :, 2], 5, axis=0)
    grounded = ankles[:, :, 2] <= (ground_z[None, :] + 0.05)

    slip_values = []
    ground_dev_values = []
    grounded_xy_speed = []
    xy_drift_0p5_values = []
    z_std_values = []
    z_p95_dev_values = []
    win = max(2, int(round(0.5 * fps)))
    for foot_idx in range(2):
        for start, end in contiguous_segments(grounded[:, foot_idx]):
            if end - start < 2:
                continue
            xy = ankles[start:end, foot_idx, :2]
            slip_values.append(float(np.linalg.norm(xy - xy[0], axis=-1).max()))
            z = ankles[start:end, foot_idx, 2]
            ground_dev_values.append(float(np.abs(z - ground_z[foot_idx]).max()))
            z_std_values.append(float(np.std(z)))
            z_p95_dev_values.append(float(np.percentile(np.abs(z - np.median(z)), 95)))
            if len(xy) >= win:
                xy_drift_0p5_values.append(float(np.linalg.norm(xy[win - 1 :] - xy[: -(win - 1)], axis=-1).max()))
            else:
                xy_drift_0p5_values.append(float(np.linalg.norm(xy[-1] - xy[0])))
            vel = np.linalg.norm(np.diff(xy, axis=0), axis=-1) / dt
            grounded_xy_speed.extend(vel.tolist())

    return {
        "foot_slip_max_m": float(max(slip_values)) if slip_values else np.nan,
        "foot_ground_dev_max_m": float(max(ground_dev_values)) if ground_dev_values else np.nan,
        "foot_stance_xy_drift_0p5s_max_m": float(max(xy_drift_0p5_values)) if xy_drift_0p5_values else np.nan,
        "foot_stance_z_std_max_m": float(max(z_std_values)) if z_std_values else np.nan,
        "foot_stance_z_p95_dev_max_m": float(max(z_p95_dev_values)) if z_p95_dev_values else np.nan,
        "foot_grounded_xy_speed_p95": float(np.percentile(grounded_xy_speed, 95))
        if grounded_xy_speed
        else np.nan,
        "foot_grounded_frame_frac": float(np.mean(np.any(grounded, axis=1))),
    }


def motion_metrics(sugar_root: Path, case: CaseSpec) -> dict[str, float]:
    d, obj = load_motion(sugar_root, case.motion_folder)
    body = np.asarray(d["body_pos_w"], dtype=np.float64)
    fps = float(np.asarray(d.get("fps", [50]))[0])
    dt = 1.0 / fps
    tracked = body[:, TRACKED]
    ankles = body[:, ANKLES]
    wrists = body[:, WRISTS]
    obj_t = np.asarray(obj["obj_trans"], dtype=np.float64)

    track_speed = np.linalg.norm(np.diff(tracked, axis=0), axis=-1) / dt
    ankle_speed = np.linalg.norm(np.diff(ankles, axis=0), axis=-1) / dt
    wrist_speed = np.linalg.norm(np.diff(wrists, axis=0), axis=-1) / dt
    track_acc = np.linalg.norm(np.diff(tracked, n=2, axis=0), axis=-1) / (dt**2)
    ankle_acc = np.linalg.norm(np.diff(ankles, n=2, axis=0), axis=-1) / (dt**2)
    track_jerk = np.linalg.norm(np.diff(tracked, n=3, axis=0), axis=-1) / (dt**3)
    ankle_jerk = np.linalg.norm(np.diff(ankles, n=3, axis=0), axis=-1) / (dt**3)
    obj_speed = np.linalg.norm(np.diff(obj_t, axis=0), axis=-1) / dt

    out = {
        "frames": float(body.shape[0]),
        "duration_s": float(body.shape[0] / fps),
        "trackbody_speed_max": float(track_speed.max()),
        "ankle_speed_max": float(ankle_speed.max()),
        "wrist_speed_max": float(wrist_speed.max()),
        "trackbody_acc_max": float(track_acc.max()),
        "ankle_acc_max": float(ankle_acc.max()),
        "trackbody_jerk_p95": float(np.percentile(track_jerk, 95)),
        "ankle_jerk_p95": float(np.percentile(ankle_jerk, 95)),
        "obj_speed_max": float(obj_speed.max()),
        "obj_z_lift_m": float(obj_t[:, 2].max() - obj_t[:, 2].min()),
        "root_z_min": float(body[:, 0, 2].min()),
    }
    out.update(foot_metrics(body, fps))
    return out


def contact_metrics(sugar_outputs: Path, case: CaseSpec) -> dict[str, float | str]:
    path = sugar_outputs / case.contact_probe_rel
    if not path.exists():
        return {
            "isaac_contact_path": str(path),
            "isaac_both_contact_frac": np.nan,
            "isaac_contact_iou": np.nan,
            "source_contact_frac": np.nan,
        }
    df = pd.read_csv(path)
    both_col = None
    for col in ["isaac_both_contact_gt_0p1n", "isaac_both_contact", "isaac_both_net_contact"]:
        if col in df.columns:
            both_col = col
            break
    if both_col is None:
        raise ValueError(f"No Isaac both-contact column in {path}")
    source = df["source_contact"].astype(bool).to_numpy()
    both = df[both_col].astype(bool).to_numpy()
    union = np.logical_or(source, both)
    iou = float(np.logical_and(source, both).sum() / union.sum()) if union.sum() else np.nan
    return {
        "isaac_contact_path": str(path),
        "isaac_both_contact_frac": float(both.mean()),
        "isaac_contact_iou": iou,
        "source_contact_frac": float(source.mean()),
    }


def add_correlations(df: pd.DataFrame) -> pd.DataFrame:
    labeled = df.dropna(subset=["staggered_success"]).copy()
    metrics = [
        "trackbody_jerk_p95",
        "ankle_jerk_p95",
        "trackbody_acc_max",
        "ankle_acc_max",
        "foot_slip_max_m",
        "foot_ground_dev_max_m",
        "foot_stance_xy_drift_0p5s_max_m",
        "foot_stance_z_std_max_m",
        "foot_stance_z_p95_dev_max_m",
        "foot_grounded_xy_speed_p95",
        "obj_speed_max",
        "isaac_both_contact_frac",
        "isaac_contact_iou",
    ]
    rows = []
    for metric in metrics:
        sub = labeled.dropna(subset=[metric, "staggered_success"])
        if len(sub) < 3 or sub[metric].nunique() < 2:
            rho, pval = np.nan, np.nan
        else:
            rho, pval = spearmanr(sub[metric], sub["staggered_success"])
        rows.append(
            {
                "metric": metric,
                "n": int(len(sub)),
                "spearman_rho_vs_success": float(rho) if np.isfinite(rho) else np.nan,
                "spearman_abs_rho": float(abs(rho)) if np.isfinite(rho) else np.nan,
                "p_value": float(pval) if np.isfinite(pval) else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("spearman_abs_rho", ascending=False)


def add_subgroup_correlations(df: pd.DataFrame) -> pd.DataFrame:
    groups = {
        "all_labeled": df.dropna(subset=["staggered_success"]),
        "exclude_box023_self_collision": df[
            df["staggered_success"].notna() & ~df["pathology_self_collision"].astype(bool)
        ],
    }
    rows = []
    for group_name, sub in groups.items():
        corr = add_correlations(sub)
        corr.insert(0, "group", group_name)
        rows.append(corr)
    return pd.concat(rows, ignore_index=True)


def add_pathology_flags(df: pd.DataFrame, spider_root: Path) -> pd.DataFrame:
    out = df.copy()
    out["object_pathology_mode"] = ""
    out["pathology_self_collision"] = False
    out["pathology_low_contact"] = False
    out["pathology_reason"] = ""

    onrails_path = spider_root / "workspace/core4d/results/E165/E1_onrails_probe/onrails_probe_scalars.json"
    modes: dict[str, str] = {}
    init_force: dict[str, float] = {}
    if onrails_path.exists():
        data = json.loads(onrails_path.read_text())
        for obj, info in data.get("cases", {}).items():
            modes[obj] = str(info.get("mode", ""))
            init_force[obj] = float(info.get("max_init_net_force", 0.0))

    reasons = []
    for idx, row in out.iterrows():
        short = str(row["short_case"])
        obj = "box023" if short.startswith("box023") else "box021" if short.startswith("box021") else "box004" if short.startswith("box004") else "box026"
        mode = modes.get(obj, "")
        reason_parts = []
        self_collision = bool(obj == "box023" and init_force.get(obj, 0.0) > 500.0)
        low_contact = bool(row["isaac_both_contact_frac"] < 0.05 and row["isaac_contact_iou"] < 0.05)
        if self_collision:
            reason_parts.append(f"onrails_init_net_force={init_force.get(obj, 0.0):.1f}N")
        if low_contact:
            reason_parts.append("low_isaac_both_and_iou")
        out.at[idx, "object_pathology_mode"] = mode
        out.at[idx, "pathology_self_collision"] = self_collision
        out.at[idx, "pathology_low_contact"] = low_contact
        out.at[idx, "pathology_reason"] = "; ".join(reason_parts)
        reasons.append((short, reason_parts))
    return out


def precision_recall(pred_fail: np.ndarray, true_fail: np.ndarray) -> tuple[float, float, int, int, int]:
    tp = int(np.logical_and(pred_fail, true_fail).sum())
    fp = int(np.logical_and(pred_fail, ~true_fail).sum())
    fn = int(np.logical_and(~pred_fail, true_fail).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return precision, recall, tp, fp, fn


def calibrate_tier1(df: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    labeled = df.dropna(subset=["staggered_success"]).copy()
    true_fail = (labeled["staggered_success"].to_numpy() < 0.10).astype(bool)
    metric_names = [
        "trackbody_jerk_p95",
        "trackbody_acc_max",
        "foot_stance_xy_drift_0p5s_max_m",
        "foot_stance_z_p95_dev_max_m",
    ]

    threshold_grid: dict[str, list[float]] = {}
    for metric in metric_names:
        vals = labeled[metric].dropna().unique()
        threshold_grid[metric] = sorted(float(v) for v in vals)

    rows = []
    best: dict[str, Any] | None = None
    for jerk_thr in threshold_grid["trackbody_jerk_p95"]:
        for acc_thr in threshold_grid["trackbody_acc_max"]:
            for slip_thr in threshold_grid["foot_stance_xy_drift_0p5s_max_m"]:
                for ground_thr in threshold_grid["foot_stance_z_p95_dev_max_m"]:
                    pred = (
                        (labeled["trackbody_jerk_p95"].to_numpy() >= jerk_thr)
                        | (labeled["trackbody_acc_max"].to_numpy() >= acc_thr)
                        | (labeled["foot_stance_xy_drift_0p5s_max_m"].to_numpy() >= slip_thr)
                        | (labeled["foot_stance_z_p95_dev_max_m"].to_numpy() >= ground_thr)
                    )
                    precision, recall, tp, fp, fn = precision_recall(pred, true_fail)
                    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
                    row = {
                        "trackbody_jerk_p95": jerk_thr,
                        "trackbody_acc_max": acc_thr,
                        "foot_stance_xy_drift_0p5s_max_m": slip_thr,
                        "foot_stance_z_p95_dev_max_m": ground_thr,
                        "precision": precision,
                        "recall": recall,
                        "f1": f1,
                        "tp": tp,
                        "fp": fp,
                        "fn": fn,
                    }
                    rows.append(row)
                    key = (f1, precision, recall, -fp)
                    if best is None or key > best["key"]:
                        best = {**row, "key": key}
    pr_df = pd.DataFrame(rows).sort_values(["f1", "precision", "recall"], ascending=False)
    assert best is not None
    best.pop("key", None)
    best["negative_label"] = "staggered_success < 0.10"
    best["rule"] = (
        "fail if jerk>=threshold OR acc>=threshold OR foot_slip>=threshold "
        "OR foot_stance_z_p95_dev>=threshold"
    )
    best["n_labeled_cases"] = int(len(labeled))
    best["n_negative_cases"] = int(true_fail.sum())
    return best, pr_df


def _metric_thresholds(df: pd.DataFrame, metrics: list[str]) -> dict[str, list[float]]:
    threshold_grid: dict[str, list[float]] = {}
    for metric in metrics:
        vals = sorted(float(v) for v in df[metric].dropna().unique())
        threshold_grid[metric] = vals + [float("inf")]
    return threshold_grid


def _two_stage_predict(df: pd.DataFrame, thresholds: dict[str, float]) -> np.ndarray:
    pred = df["pathology_self_collision"].astype(bool).to_numpy().copy()
    non_pathology = ~df["pathology_self_collision"].astype(bool).to_numpy()
    for metric, threshold in thresholds.items():
        if not np.isfinite(threshold):
            continue
        pred |= non_pathology & (df[metric].to_numpy() >= threshold)
    return pred


def calibrate_two_stage(df: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    labeled = df.dropna(subset=["staggered_success"]).copy()
    true_fail = (labeled["staggered_success"].to_numpy() < 0.10).astype(bool)
    metrics = ["trackbody_jerk_p95", "ankle_acc_max", "obj_speed_max", "foot_slip_max_m"]
    threshold_grid = _metric_thresholds(labeled, metrics)

    rows = []
    best: dict[str, Any] | None = None
    for jerk_thr in threshold_grid["trackbody_jerk_p95"]:
        for ankle_acc_thr in threshold_grid["ankle_acc_max"]:
            for obj_speed_thr in threshold_grid["obj_speed_max"]:
                for foot_slip_thr in threshold_grid["foot_slip_max_m"]:
                    thresholds = {
                        "trackbody_jerk_p95": jerk_thr,
                        "ankle_acc_max": ankle_acc_thr,
                        "obj_speed_max": obj_speed_thr,
                        "foot_slip_max_m": foot_slip_thr,
                    }
                    pred = _two_stage_predict(labeled, thresholds)
                    precision, recall, tp, fp, fn = precision_recall(pred, true_fail)
                    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
                    n_enabled = sum(np.isfinite(v) for v in thresholds.values())
                    row = {
                        **thresholds,
                        "precision": precision,
                        "recall": recall,
                        "f1": f1,
                        "tp": tp,
                        "fp": fp,
                        "fn": fn,
                        "n_enabled_metrics": n_enabled,
                    }
                    rows.append(row)
                    passes_gate = precision >= 0.75 and recall >= 0.75
                    key = (passes_gate, f1, precision, recall, -n_enabled, -fp)
                    if best is None or key > best["key"]:
                        best = {**row, "key": key}
    grid_df = pd.DataFrame(rows).sort_values(
        ["f1", "precision", "recall", "n_enabled_metrics"],
        ascending=[False, False, False, True],
    )
    assert best is not None
    best.pop("key", None)
    best["stage0_rule"] = "pathology_self_collision == true"
    best["stage1_rule"] = "for non-self-collision cases: fail if any enabled metric exceeds threshold"
    best["negative_label"] = "staggered_success < 0.10"
    best["n_labeled_cases"] = int(len(labeled))
    best["n_negative_cases"] = int(true_fail.sum())

    case_rows = []
    thresholds = {metric: float(best[metric]) for metric in metrics}
    pred = _two_stage_predict(labeled, thresholds)
    for (_, row), pred_fail, actual_fail in zip(labeled.iterrows(), pred, true_fail):
        trigger_parts = []
        if bool(row["pathology_self_collision"]):
            trigger_parts.append("stage0:self_collision")
        else:
            for metric, threshold in thresholds.items():
                if np.isfinite(threshold) and float(row[metric]) >= threshold:
                    trigger_parts.append(f"stage1:{metric}>={threshold:.4g}")
        case_rows.append(
            {
                "short_case": row["short_case"],
                "staggered_success": row["staggered_success"],
                "actual_fail": bool(actual_fail),
                "pred_fail": bool(pred_fail),
                "is_tp": bool(pred_fail and actual_fail),
                "is_fp": bool(pred_fail and not actual_fail),
                "is_fn": bool((not pred_fail) and actual_fail),
                "trigger": "; ".join(trigger_parts),
            }
        )
    case_df = pd.DataFrame(case_rows)
    return best, grid_df, case_df


def leave_one_two_stage(df: pd.DataFrame) -> pd.DataFrame:
    labeled = df.dropna(subset=["staggered_success"]).copy().reset_index(drop=True)
    rows = []
    for i, heldout in labeled.iterrows():
        train = labeled.drop(index=i).copy()
        best, _, _ = calibrate_two_stage(train)
        thresholds = {
            "trackbody_jerk_p95": float(best["trackbody_jerk_p95"]),
            "ankle_acc_max": float(best["ankle_acc_max"]),
            "obj_speed_max": float(best["obj_speed_max"]),
            "foot_slip_max_m": float(best["foot_slip_max_m"]),
        }
        held_df = heldout.to_frame().T
        pred_fail = bool(_two_stage_predict(held_df, thresholds)[0])
        actual_fail = bool(float(heldout["staggered_success"]) < 0.10)
        rows.append(
            {
                "heldout_case": heldout["short_case"],
                "heldout_success": heldout["staggered_success"],
                "actual_fail": actual_fail,
                "pred_fail": pred_fail,
                "correct": pred_fail == actual_fail,
                "train_precision": best["precision"],
                "train_recall": best["recall"],
                "train_f1": best["f1"],
                "train_thresholds_json": json.dumps(thresholds, ensure_ascii=False),
            }
        )
    return pd.DataFrame(rows)


def plot_outputs(df: pd.DataFrame, corr_df: pd.DataFrame, pr_df: pd.DataFrame, out_dir: Path) -> None:
    labeled = df.dropna(subset=["staggered_success"]).copy()
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    specs = [
        ("trackbody_jerk_p95", "jerk P95"),
        ("trackbody_acc_max", "acc max"),
        ("foot_slip_max_m", "foot slip max"),
        ("foot_ground_dev_max_m", "foot-ground dev max"),
        ("obj_speed_max", "object Vmax"),
        ("isaac_both_contact_frac", "Isaac both@0.1"),
    ]
    rho_map = dict(zip(corr_df["metric"], corr_df["spearman_rho_vs_success"]))
    for ax, (metric, title) in zip(axes.ravel(), specs):
        ax.scatter(labeled[metric], labeled["staggered_success"], s=48)
        for _, row in labeled.iterrows():
            ax.annotate(row["short_case"], (row[metric], row["staggered_success"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
        rho = rho_map.get(metric, np.nan)
        ax.set_title(f"{title} (rho={rho:.2f})")
        ax.set_xlabel(metric)
        ax.set_ylabel("staggered success")
        ax.grid(True, alpha=0.25)
    fig.savefig(out_dir / "redline_scatter.png", dpi=180)
    plt.close(fig)

    top = pr_df.head(60).sort_values("recall")
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    ax.scatter(top["recall"], top["precision"], c=top["f1"], cmap="viridis", s=38)
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_title("Tier-1 redline threshold PR candidates")
    ax.grid(True, alpha=0.25)
    fig.savefig(out_dir / "tier1_pr.png", dpi=180)
    plt.close(fig)


def markdown_table(df: pd.DataFrame) -> str:
    show = df.copy()
    show = show.where(pd.notna(show), "")
    headers = [str(c) for c in show.columns]
    rows = [[str(v) for v in row] for row in show.to_numpy()]
    widths = [len(h) for h in headers]
    for row in rows:
        widths = [max(width, len(cell)) for width, cell in zip(widths, row)]

    def fmt(row: list[str]) -> str:
        return "| " + " | ".join(cell.ljust(width) for cell, width in zip(row, widths)) + " |"

    sep = "| " + " | ".join("-" * width for width in widths) + " |"
    return "\n".join([fmt(headers), sep] + [fmt(row) for row in rows])


def jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, float) and np.isinf(obj):
        return "inf"
    return obj


def write_summary(
    df: pd.DataFrame,
    corr_df: pd.DataFrame,
    subgroup_corr_df: pd.DataFrame,
    thresholds: dict[str, Any],
    two_stage_thresholds: dict[str, Any],
    two_stage_cases: pd.DataFrame,
    two_stage_loo: pd.DataFrame,
    out_dir: Path,
    sugar_root: Path,
    sugar_outputs: Path,
) -> None:
    labeled_n = int(df["staggered_success"].notna().sum())
    unlabeled = df[df["staggered_success"].isna()]["short_case"].tolist()
    jerk_rho = corr_df.loc[corr_df["metric"].eq("trackbody_jerk_p95"), "spearman_abs_rho"].iloc[0]
    acc_rho = corr_df.loc[corr_df["metric"].eq("trackbody_acc_max"), "spearman_abs_rho"].iloc[0]
    contact_rho = corr_df[corr_df["metric"].isin(["isaac_both_contact_frac", "isaac_contact_iou"])]["spearman_abs_rho"].max()
    phase0_pass = bool((max(jerk_rho, acc_rho) >= 0.60) and (max(jerk_rho, acc_rho) > contact_rho))
    tier1_pass = bool(thresholds["precision"] >= 0.75 and thresholds["recall"] >= 0.75)
    two_stage_pass = bool(two_stage_thresholds["precision"] >= 0.75 and two_stage_thresholds["recall"] >= 0.75)
    two_stage_loo_acc = float(two_stage_loo["correct"].mean()) if len(two_stage_loo) else np.nan
    subgroup_best = subgroup_corr_df[
        (subgroup_corr_df["group"] == "exclude_box023_self_collision")
        & subgroup_corr_df["metric"].isin(
            [
                "trackbody_jerk_p95",
                "trackbody_acc_max",
                "ankle_acc_max",
                "ankle_jerk_p95",
                "obj_speed_max",
                "foot_stance_xy_drift_0p5s_max_m",
                "foot_stance_z_p95_dev_max_m",
            ]
        )
    ]["spearman_abs_rho"].max()
    r3b_pass = bool(np.isfinite(subgroup_best) and subgroup_best >= 0.60)

    case_cols = [
        "short_case",
        "staggered_success",
        "pathology_self_collision",
        "pathology_low_contact",
        "trackbody_acc_max",
        "trackbody_jerk_p95",
        "foot_stance_xy_drift_0p5s_max_m",
        "foot_stance_z_p95_dev_max_m",
        "obj_speed_max",
        "isaac_both_contact_frac",
        "isaac_contact_iou",
    ]
    text = [
        "# E166-R3 redline predictive audit",
        "",
        f"- sugar_root: `{sugar_root}`",
        f"- sugar_outputs: `{sugar_outputs}`",
        f"- output_dir: `{out_dir}`",
        f"- Tier-1 metrics computed for: {len(df)}/8 cases",
        f"- correlation/PR labeled cases: {labeled_n}/8",
        f"- unlabeled downstream cases excluded from correlation: {unlabeled if unlabeled else 'none'}",
        "",
        "## Claim checks",
        "",
        f"- C-R3 smoothness vs contact: {'PASS' if phase0_pass else 'FAIL'} "
        f"(max(|rho(jerk)|, |rho(acc)|)={max(jerk_rho, acc_rho):.3f}, "
        f"max contact |rho|={contact_rho:.3f})",
        f"- C-R3b subgroup after self-collision split: {'PASS' if r3b_pass else 'FAIL'} "
        f"(best subgroup |rho|={subgroup_best:.3f})",
        f"- C-C Tier-1 PR: {'PASS' if tier1_pass else 'FAIL'} "
        f"(precision={thresholds['precision']:.3f}, recall={thresholds['recall']:.3f})",
        f"- C-R3c two-stage redline: {'PASS' if two_stage_pass else 'FAIL'} "
        f"(fit precision={two_stage_thresholds['precision']:.3f}, "
        f"recall={two_stage_thresholds['recall']:.3f}, leave-one acc={two_stage_loo_acc:.3f})",
        "",
        "## Case metrics",
        "",
        markdown_table(df[case_cols].round(4)),
        "",
        "## Correlations",
        "",
        markdown_table(corr_df.round(4)),
        "",
        "## Subgroup correlations",
        "",
        markdown_table(subgroup_corr_df.round(4)),
        "",
        "## Pathology split",
        "",
        markdown_table(
            df[
                [
                    "short_case",
                    "staggered_success",
                    "object_pathology_mode",
                    "pathology_self_collision",
                    "pathology_low_contact",
                    "pathology_reason",
                ]
            ].round(4)
        ),
        "",
        "## Selected Tier-1 thresholds",
        "",
        "```json",
        json.dumps(jsonable(thresholds), indent=2, ensure_ascii=False),
        "```",
        "",
        "## Selected Two-Stage Thresholds",
        "",
        "```json",
        json.dumps(jsonable(two_stage_thresholds), indent=2, ensure_ascii=False),
        "```",
        "",
        "## Two-Stage Case Decisions",
        "",
        markdown_table(two_stage_cases.round(4)),
        "",
        "## Two-Stage Leave-One Audit",
        "",
        markdown_table(two_stage_loo.round(4)),
        "",
        "## Artifacts",
        "",
        "- `redline_case_metrics.tsv`",
        "- `redline_correlations.tsv`",
        "- `redline_subgroup_correlations.tsv`",
        "- `redline_pathology_split.tsv`",
        "- `redline_thresholds.json`",
        "- `redline_two_stage_thresholds.json`",
        "- `redline_two_stage_grid.tsv`",
        "- `redline_two_stage_case_decisions.tsv`",
        "- `redline_two_stage_leave_one.tsv`",
        "- `redline_scatter.png`",
        "- `tier1_pr.png`",
    ]
    (out_dir / "redline_summary.md").write_text("\n".join(text) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sugar-root", type=Path, default=Path("/home/ubuntu/Workspace/Loco-Manipulation/SUGAR"))
    parser.add_argument("--sugar-outputs", type=Path, default=Path("/home/ubuntu/Workspace/Loco-Manipulation/SUGAR/outputs"))
    parser.add_argument("--spider-root", type=Path, default=Path("/home/ubuntu/Workspace/spider"))
    parser.add_argument("--out-dir", type=Path, default=Path("workspace/core4d/results/E166/redline_predictive"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for case in CASES:
        success, success_path, success_rows = read_success(args.sugar_outputs, case)
        row: dict[str, Any] = {
            "case": case.name,
            "short_case": case.short_case,
            "motion_folder": case.motion_folder,
            "staggered_success": success,
            "success_path": success_path,
            "success_rows": success_rows,
        }
        row.update(motion_metrics(args.sugar_root, case))
        row.update(contact_metrics(args.sugar_outputs, case))
        rows.append(row)

    df = add_pathology_flags(pd.DataFrame(rows), args.spider_root)
    corr_df = add_correlations(df)
    subgroup_corr_df = add_subgroup_correlations(df)
    thresholds, pr_df = calibrate_tier1(df)
    two_stage_thresholds, two_stage_grid_df, two_stage_case_df = calibrate_two_stage(df)
    two_stage_loo_df = leave_one_two_stage(df)

    df.to_csv(out_dir / "redline_case_metrics.tsv", sep="\t", index=False)
    corr_df.to_csv(out_dir / "redline_correlations.tsv", sep="\t", index=False)
    subgroup_corr_df.to_csv(out_dir / "redline_subgroup_correlations.tsv", sep="\t", index=False)
    df[
        [
            "short_case",
            "staggered_success",
            "object_pathology_mode",
            "pathology_self_collision",
            "pathology_low_contact",
            "pathology_reason",
        ]
    ].to_csv(out_dir / "redline_pathology_split.tsv", sep="\t", index=False)
    pr_df.to_csv(out_dir / "redline_threshold_grid.tsv", sep="\t", index=False)
    (out_dir / "redline_thresholds.json").write_text(json.dumps(jsonable(thresholds), indent=2, ensure_ascii=False) + "\n")
    two_stage_grid_df.to_csv(out_dir / "redline_two_stage_grid.tsv", sep="\t", index=False)
    two_stage_case_df.to_csv(out_dir / "redline_two_stage_case_decisions.tsv", sep="\t", index=False)
    two_stage_loo_df.to_csv(out_dir / "redline_two_stage_leave_one.tsv", sep="\t", index=False)
    (out_dir / "redline_two_stage_thresholds.json").write_text(
        json.dumps(jsonable(two_stage_thresholds), indent=2, ensure_ascii=False) + "\n"
    )
    plot_outputs(df, corr_df, pr_df, out_dir)
    write_summary(
        df,
        corr_df,
        subgroup_corr_df,
        thresholds,
        two_stage_thresholds,
        two_stage_case_df,
        two_stage_loo_df,
        out_dir,
        args.sugar_root,
        args.sugar_outputs,
    )

    print(f"[written] {out_dir}/redline_case_metrics.tsv")
    print(f"[written] {out_dir}/redline_correlations.tsv")
    print(f"[written] {out_dir}/redline_summary.md")


if __name__ == "__main__":
    main()
