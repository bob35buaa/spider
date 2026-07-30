#!/usr/bin/env python3
"""E180: audit sparse linear separability of CORE4D downstream RL labels.

The script intentionally separates three questions:

1. Can one metric split the observed RL labels in-sample?
2. Can a 1--5 feature linear model split them, and does the full fitting
   procedure survive leave-one-case/object-out validation?
3. Does the model frozen on RL-observed cases reject pre-RL proxy negatives?

Failure is the positive class throughout.  Identity fields, manual decisions,
existing pass/fail gates, and experiment identifiers are never model inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import re
import warnings
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


R018_ROOT = Path(
    "/home/ubuntu/Workspace/Loco-Manipulation/SUGAR-private-worktrees/R018"
)

RL_CASES: tuple[tuple[str, str, str], ...] = (
    # case_id, RL outcome, authority note
    ("box021_20231011_034_p1", "success", "R018 Box021 all11 validation"),
    ("box021_20231011_034_p2", "success", "R018 Box021 all11 validation"),
    ("box021_20231011_036_p1", "success", "R018 Box021 all11 validation"),
    ("box021_20231011_036_p2", "success", "R018 Box021 all11 validation"),
    ("box021_20231011_037_p2", "success", "R018 Box021 all11 validation"),
    ("box021_20231011_038_p1", "success", "R018 Box021 all11 validation"),
    ("box021_20231020_020_p1", "success", "R018 Box021 all11 validation"),
    ("box021_20231020_022_p1", "success", "R018 Box021 all11 validation"),
    ("box021_20231020_023_p1", "success", "R018 Box021 all11 validation"),
    ("box021_20231018_029_p2", "success", "E167/E167A bridge RL validation"),
    ("box021_20231011_035_p1", "success", "E167/E167A bridge RL validation"),
    ("box004_20231003_2_082_p1", "fail", "user-reported RL training failure"),
    ("box004_20231003_2_082_p2", "fail", "user-reported RL training failure"),
    ("box004_20231003_2_083_p1", "success", "user-reported RL validation"),
    ("box004_20231003_2_083_p2", "success", "user-reported RL validation"),
    ("box001_20231003_1_039_p1", "success", "Box001 13-case RL validation"),
    ("box001_20231003_1_040_p2", "success", "Box001 13-case RL validation"),
    ("box001_20231003_1_041_p1", "success", "Box001 13-case RL validation"),
    ("box001_20231003_2_037_p1", "success", "Box001 13-case RL validation"),
    ("box001_20231003_2_038_p1", "success", "Box001 13-case RL validation"),
    ("box001_20231003_2_039_p1", "success", "Box001 13-case RL validation"),
    ("box001_20231003_2_041_p1", "success", "Box001 13-case RL validation"),
    ("box001_20231020_011_p1", "success", "Box001 13-case RL validation"),
    ("box001_20231020_014_p2", "fail", "user-reported RL training failure"),
    ("box001_20231023_107_p2", "success", "Box001 13-case RL validation"),
    ("box001_20231023_108_p1", "success", "Box001 13-case RL validation"),
    ("box001_20231023_108_p2", "success", "Box001 13-case RL validation"),
    ("box001_20231023_109_p2", "success", "Box001 13-case RL validation"),
    ("box024_20231011_026_p2", "success", "Box024 3-case RL validation"),
    ("box024_20231011_027_p2", "success", "Box024 3-case RL validation"),
    ("box024_20231011_028_p2", "fail", "user-reported RL training failure"),
    ("box023_20231008_045_p1", "success", "Box023 exact7 RL validation"),
    ("box023_20231008_046_p1", "success", "Box023 exact7 RL validation"),
    ("box023_20231011_021_p1", "success", "Box023 exact7 RL validation"),
    ("box023_20231011_021_p2", "success", "Box023 exact7 RL validation"),
    ("box023_20231020_040_p2", "fail", "user-reported RL training failure"),
    ("box023_20231020_041_p1", "success", "Box023 exact7 RL validation"),
    ("box023_20231020_042_p2", "fail", "user-reported RL training failure"),
)

STANDARD_TABLES = {
    "E170": "results/E170/s6_downstream/eval/full/e170_case_metrics.tsv",
    "E172": "results/E172/s6_downstream/eval/full/e171_case_metrics.tsv",
    "E173": "results/E173/s6_downstream/eval/full/e173_case_metrics.tsv",
}

R018_EXPERIMENT_BY_OBJECT = {
    "box021": "R018-4_all11_mass78_ddp_refiner",
    "box001": "R018-5_box001_13case_refiner",
    "box004": "R018-7_cross_object_multirefiner",
    "box024": "R018-7_cross_object_multirefiner",
    "box023": "R018-8_box023_exact7_refiner_ddp30k",
}

IDENTITY_OR_LABEL_PATTERNS = (
    r"(^|_)id($|_)",
    r"case",
    r"variant",
    r"object_key",
    r"object_category",
    r"sequence",
    r"source_",
    r"assigned_gpu",
    r"execution",
    r"method$",
    r"status",
    r"manual",
    r"user_",
    r"codex_",
    r"review",
    r"decision",
    r"quality",
    r"expected",
    r"usable",
    r"use$",
    r"pass$",
    r"failure",
    r"fall_flag",
    r"success",
    r"gate_status",
    r"gate_applicable",
    r"has_.*health",
    r"^e168_",
    r"^delta_",
    r"^improvement_",
    r"ordinal",
)


@dataclass
class LinearFit:
    procedure: str
    features: list[str]
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    coef_z: np.ndarray
    intercept_z: float
    train_metrics: dict[str, float | int]
    geometric_margin: float
    hyperparameter: str

    @property
    def coef_raw(self) -> np.ndarray:
        return self.coef_z / self.scaler_scale

    @property
    def intercept_raw(self) -> float:
        return float(
            self.intercept_z
            - np.sum(self.coef_z * self.scaler_mean / self.scaler_scale)
        )

    def decision(self, frame: pd.DataFrame) -> np.ndarray:
        x = frame[self.features].to_numpy(dtype=float)
        return self.intercept_raw + x @ self.coef_raw

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return (self.decision(frame) >= 0.0).astype(int)


@dataclass
class ThresholdFit:
    feature: str
    direction: str
    threshold: float
    train_metrics: dict[str, float | int]
    class_gap: float

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        values = frame[self.feature].to_numpy(dtype=float)
        if self.direction == ">=":
            return (values >= self.threshold).astype(int)
        return (values <= self.threshold).astype(int)


def object_key(case_id: str) -> str:
    return case_id.split("_", 1)[0].lower()


def frozen_labels() -> pd.DataFrame:
    rows = []
    for case_id, outcome, authority in RL_CASES:
        rows.append(
            {
                "case_id": case_id,
                "object_key": object_key(case_id),
                "rl_outcome": outcome,
                "y_fail": int(outcome == "fail"),
                "label_authority": authority,
            }
        )
    frame = pd.DataFrame(rows)
    if len(frame) != 38 or frame["case_id"].nunique() != 38:
        raise AssertionError("RL label manifest must contain 38 unique cases")
    counts = frame["rl_outcome"].value_counts().to_dict()
    if counts != {"success": 32, "fail": 6}:
        raise AssertionError(f"unexpected label counts: {counts}")
    return frame


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scalar(array: np.ndarray) -> float:
    values = np.asarray(array).reshape(-1)
    if len(values) != 1:
        raise ValueError(f"expected scalar-like array, got shape {array.shape}")
    return float(values[0])


def norm_rows(array: np.ndarray) -> np.ndarray:
    values = np.asarray(array, dtype=float)
    return np.linalg.norm(values.reshape(values.shape[0], -1), axis=1)


def summary_stats(prefix: str, values: np.ndarray) -> dict[str, float]:
    data = np.asarray(values, dtype=float)
    data = data[np.isfinite(data)]
    if data.size == 0:
        return {
            f"{prefix}_mean": math.nan,
            f"{prefix}_p95": math.nan,
            f"{prefix}_max": math.nan,
        }
    return {
        f"{prefix}_mean": float(np.mean(data)),
        f"{prefix}_p95": float(np.quantile(data, 0.95)),
        f"{prefix}_max": float(np.max(data)),
    }


def quaternion_angle(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    left = np.asarray(q1, dtype=float)
    right = np.asarray(q2, dtype=float)
    left /= np.maximum(np.linalg.norm(left, axis=-1, keepdims=True), 1e-12)
    right /= np.maximum(np.linalg.norm(right, axis=-1, keepdims=True), 1e-12)
    dots = np.abs(np.sum(left * right, axis=-1))
    return 2.0 * np.arccos(np.clip(dots, 0.0, 1.0))


def nested_get(mapping: dict[str, Any], dotted: str) -> float:
    value: Any = mapping
    for key in dotted.split("."):
        if not isinstance(value, dict) or key not in value:
            return math.nan
        value = value[key]
    if isinstance(value, (int, float, np.number)) and np.isfinite(value):
        return float(value)
    return math.nan


def extract_reference_features(release_dir: Path) -> dict[str, float]:
    with np.load(release_dir / "robot_50hz.npz", allow_pickle=False) as robot:
        fps = scalar(robot["fps"])
        joint_vel = np.asarray(robot["joint_vel"], dtype=float)
        body_lin_vel = np.asarray(robot["body_lin_vel_w"], dtype=float)
        body_ang_vel = np.asarray(robot["body_ang_vel_w"], dtype=float)
        frames = int(joint_vel.shape[0])

    with np.load(release_dir / "partner_50hz.npz", allow_pickle=False) as partner:
        valid_overlap = np.asarray(partner["valid_overlap"], dtype=bool)
        geometry_valid = np.asarray(partner["partner_geometry_valid_mask"], dtype=bool)
        target_contact = np.asarray(partner["target_contact_label"], dtype=bool)
        partner_any = np.asarray(partner["partner_contact_any"], dtype=bool)
        partner_both = np.asarray(partner["partner_contact_both"], dtype=bool)
        hand_distance = np.asarray(
            partner["partner_hand_surface_distance_m"], dtype=float
        )
        partner_object_pos = np.asarray(partner["partner_object_pos_w"], dtype=float)
        target_object_pos = np.asarray(partner["target_object_pos_w"], dtype=float)
        partner_object_quat = np.asarray(
            partner["partner_object_quat_w"], dtype=float
        )
        target_object_quat = np.asarray(partner["target_object_quat_w"], dtype=float)
        phase_offset = scalar(partner["phase_residual_offset_50hz"])
        total_offset = scalar(partner["total_offset_50hz"])
        alignment = json.loads(str(partner["alignment_metadata_json"][0]))

    with (release_dir / "obj_motion_global_50hz.pkl").open("rb") as stream:
        obj = pickle.load(stream)
    object_lin_vel = np.asarray(obj["obj_lin_vel"], dtype=float)
    object_ang_vel = np.asarray(obj["obj_ang_vel"], dtype=float)

    dt = 1.0 / fps
    joint_speed = norm_rows(joint_vel)
    joint_acc = norm_rows(np.diff(joint_vel, axis=0) / dt)
    joint_jerk = norm_rows(np.diff(joint_vel, n=2, axis=0) / (dt * dt))
    body_speed = norm_rows(body_lin_vel)
    body_ang_speed = norm_rows(body_ang_vel)
    body_acc = norm_rows(np.diff(body_lin_vel, axis=0) / dt)
    body_jerk = norm_rows(np.diff(body_lin_vel, n=2, axis=0) / (dt * dt))
    object_speed = norm_rows(object_lin_vel)
    object_ang_speed = norm_rows(object_ang_vel)
    object_acc = norm_rows(np.diff(object_lin_vel, axis=0) / dt)
    object_jerk = norm_rows(np.diff(object_lin_vel, n=2, axis=0) / (dt * dt))
    object_ang_acc = norm_rows(np.diff(object_ang_vel, axis=0) / dt)
    object_ang_jerk = norm_rows(
        np.diff(object_ang_vel, n=2, axis=0) / (dt * dt)
    )

    valid_pose = valid_overlap & np.all(np.isfinite(partner_object_pos), axis=1)
    valid_pose &= np.all(np.isfinite(target_object_pos), axis=1)
    pos_mismatch = norm_rows(
        partner_object_pos[valid_pose] - target_object_pos[valid_pose]
    )
    ori_mismatch = quaternion_angle(
        partner_object_quat[valid_pose], target_object_quat[valid_pose]
    )
    valid_hand = valid_overlap & geometry_valid

    features: dict[str, float] = {
        "release_frames": float(frames),
        "duration_s": float((frames - 1) / fps),
        "target_contact_frac": float(np.mean(target_contact)),
        "partner_contact_any_frac": float(np.mean(partner_any)),
        "partner_contact_both_frac": float(np.mean(partner_both)),
        "partner_both_given_any_frac": float(
            np.sum(partner_both) / max(np.sum(partner_any), 1)
        ),
        "valid_overlap_frac": float(np.mean(valid_overlap)),
        "geometry_valid_frac": float(np.mean(geometry_valid)),
        "phase_residual_offset_50hz": float(phase_offset),
        "abs_phase_residual_offset_50hz": float(abs(phase_offset)),
        "total_offset_50hz": float(total_offset),
        "abs_total_offset_50hz": float(abs(total_offset)),
    }
    for prefix, values in (
        ("joint_speed_l2", joint_speed),
        ("joint_acc_l2", joint_acc),
        ("joint_jerk_l2", joint_jerk),
        ("body_lin_speed", body_speed),
        ("body_ang_speed", body_ang_speed),
        ("body_lin_acc", body_acc),
        ("body_lin_jerk", body_jerk),
        ("object_lin_speed", object_speed),
        ("object_ang_speed", object_ang_speed),
        ("object_lin_acc", object_acc),
        ("object_lin_jerk", object_jerk),
        ("object_ang_acc", object_ang_acc),
        ("object_ang_jerk", object_ang_jerk),
        ("partner_target_obj_pos_mismatch_m", pos_mismatch),
        ("partner_target_obj_ori_mismatch_rad", ori_mismatch),
        (
            "partner_hand_surface_distance_m",
            hand_distance[valid_hand].reshape(-1),
        ),
    ):
        features.update(summary_stats(prefix, values))

    alignment_fields = {
        "alignment_selected_score": "selected_phase_candidate.score",
        "alignment_translation_rmse_m": (
            "selected_phase_candidate.translation_rmse_m"
        ),
        "alignment_orientation_rmse_rad": (
            "selected_phase_candidate.orientation_rmse_rad"
        ),
        "alignment_linear_speed_rmse_mps": (
            "selected_phase_candidate.linear_speed_rmse_mps"
        ),
        "alignment_angular_speed_rmse_radps": (
            "selected_phase_candidate.angular_speed_rmse_radps"
        ),
        "alignment_overlap_ratio": "selected_phase_candidate.overlap_ratio",
        "alignment_score_ratio": "zero_residual_score_ratio_to_best",
    }
    for name, dotted in alignment_fields.items():
        features[name] = nested_get(alignment, dotted)
    if not np.isfinite(features["alignment_overlap_ratio"]):
        overlap_frames = nested_get(
            alignment, "selected_phase_candidate.overlap_frames"
        )
        features["alignment_overlap_ratio"] = float(overlap_frames / frames)
    if not np.isfinite(features["alignment_score_ratio"]):
        features["alignment_score_ratio"] = nested_get(
            alignment, "constrained_phase_score_ratio"
        )
    return features


def discover_release_map(
    labels: pd.DataFrame, sugar_root: Path
) -> tuple[dict[str, Path], pd.DataFrame]:
    experiment_base = (
        sugar_root
        / "outputs/experiments/R018_core4d_box021_multimotion_student_tracker"
    )
    found: dict[str, set[Path]] = {}
    config_sources: dict[tuple[str, Path], set[Path]] = {}
    for obj, experiment in R018_EXPERIMENT_BY_OBJECT.items():
        eval_root = experiment_base / experiment / "eval"
        for config_path in eval_root.rglob("resolved_config.json"):
            try:
                config = json.loads(config_path.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            target = config.get("offline_motion_alias", {}).get("target")
            if not target:
                continue
            target_path = Path(target)
            if not target_path.is_absolute():
                target_path = sugar_root / target_path
            if not target_path.is_dir():
                continue
            provenance_path = target_path / "provenance.json"
            if not provenance_path.is_file():
                continue
            provenance = json.loads(provenance_path.read_text())
            case_id = str(provenance["case_id"])
            if object_key(case_id) != obj:
                continue
            found.setdefault(case_id, set()).add(target_path.resolve())
            config_sources.setdefault((case_id, target_path.resolve()), set()).add(
                config_path
            )

    # The two bridge release directories use short historical case IDs.
    aliases = {
        "box021_20231018_029_p2": "box021_029_p2",
        "box021_20231011_035_p1": "box021_035_p1",
    }
    release_map: dict[str, Path] = {}
    lineage_rows: list[dict[str, Any]] = []
    for case_id in labels["case_id"]:
        lookup = aliases.get(case_id, case_id)
        targets = found.get(lookup, set())
        if len(targets) != 1:
            raise RuntimeError(
                f"{case_id}: expected one frozen release, found {len(targets)}: "
                f"{sorted(map(str, targets))}"
            )
        target = next(iter(targets))
        release_map[case_id] = target
        config_paths = sorted(config_sources[(lookup, target)])
        provenance = json.loads((target / "provenance.json").read_text())
        lineage_rows.append(
            {
                "case_id": case_id,
                "release_case_id": lookup,
                "release_id": provenance.get("release_id", ""),
                "release_dir": str(target),
                "resolved_config_count": len(config_paths),
                "example_resolved_config": str(config_paths[0]),
                "robot_50hz_sha256": sha256_file(target / "robot_50hz.npz"),
                "partner_50hz_sha256": sha256_file(target / "partner_50hz.npz"),
                "object_50hz_sha256": sha256_file(
                    target / "obj_motion_global_50hz.pkl"
                ),
                "spider_git_commit": provenance.get("spider_git_commit", ""),
                "builder_git_commit": provenance.get("builder_git_commit", ""),
            }
        )
    return release_map, pd.DataFrame(lineage_rows)


def build_reference_panel(
    labels: pd.DataFrame, sugar_root: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    release_map, lineage = discover_release_map(labels, sugar_root)
    rows: list[dict[str, Any]] = []
    for label in labels.to_dict("records"):
        row = dict(label)
        row.update(extract_reference_features(release_map[label["case_id"]]))
        rows.append(row)
    panel = pd.DataFrame(rows)
    feature_columns = [
        column
        for column in panel.columns
        if column
        not in {
            "case_id",
            "object_key",
            "rl_outcome",
            "y_fail",
            "label_authority",
        }
    ]
    if panel[feature_columns].isna().any().any():
        missing = panel[feature_columns].isna().sum()
        missing = missing[missing > 0].to_dict()
        raise RuntimeError(f"reference feature panel contains missing values: {missing}")
    return panel, lineage


def standard_feature_exclusion(column: str, series: pd.Series) -> str | None:
    if not pd.api.types.is_numeric_dtype(series):
        return "non_numeric"
    if pd.api.types.is_bool_dtype(series):
        return "boolean_or_existing_decision"
    lowered = column.lower()
    if any(re.search(pattern, lowered) for pattern in IDENTITY_OR_LABEL_PATTERNS):
        return "identity_label_or_existing_gate"
    if series.isna().any():
        return "missing_in_rl_or_proxy_panel"
    values = series.to_numpy(dtype=float)
    if not np.all(np.isfinite(values)):
        return "non_finite"
    if float(np.nanstd(values)) <= 1e-12:
        return "constant"
    return None


def build_standardized_panels(
    labels: pd.DataFrame, workspace_root: Path
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tables = []
    for experiment, relative in STANDARD_TABLES.items():
        frame = pd.read_csv(workspace_root / relative, sep="\t")
        frame = frame.assign(source_experiment=experiment)
        tables.append(frame)
    all_metrics = pd.concat(tables, ignore_index=True, sort=False)

    label_lookup = labels.set_index("case_id")
    rl_metrics = all_metrics[all_metrics["case_id"].isin(label_lookup.index)].copy()
    rl_metrics = rl_metrics.merge(
        labels[
            [
                "case_id",
                "object_key",
                "rl_outcome",
                "y_fail",
                "label_authority",
            ]
        ],
        on="case_id",
        how="left",
        suffixes=("", "_label"),
        validate="one_to_one",
    )
    if "object_key_label" in rl_metrics:
        rl_metrics["object_key"] = rl_metrics["object_key_label"]
        rl_metrics = rl_metrics.drop(columns=["object_key_label"])

    bridge_cases = {
        "box021_20231018_029_p2",
        "box021_20231011_035_p1",
    }
    expected_standard = set(labels["case_id"]) - bridge_cases
    actual_standard = set(rl_metrics["case_id"])
    if actual_standard != expected_standard or len(rl_metrics) != 36:
        raise RuntimeError(
            "standardized RL panel mismatch: "
            f"missing={sorted(expected_standard - actual_standard)}, "
            f"extra={sorted(actual_standard - expected_standard)}"
        )

    rl_case_set = set(labels["case_id"])
    proxy_mask = (
        ((all_metrics["source_experiment"] == "E170")
         & (all_metrics["manual_use_decision"] == "DO_NOT_USE"))
        | ((all_metrics["source_experiment"].isin(["E172", "E173"]))
           & ~all_metrics["case_id"].isin(rl_case_set))
    )
    proxy = all_metrics[proxy_mask].copy()
    proxy["object_key"] = proxy["case_id"].map(object_key)
    proxy["rl_outcome"] = "not_observed"
    proxy["y_fail"] = 1
    proxy["label_authority"] = np.where(
        proxy["source_experiment"] == "E170",
        "manual DO_NOT_USE proxy negative",
        "not selected for RL export proxy negative",
    )
    proxy["proxy_reason"] = np.where(
        proxy["source_experiment"] == "E170",
        "manual_do_not_use",
        "not_selected_for_rl_export",
    )
    if len(proxy) != 42 or proxy["case_id"].nunique() != 42:
        raise RuntimeError(
            f"expected 42 unique proxy negatives, got {len(proxy)} rows / "
            f"{proxy['case_id'].nunique()} cases"
        )

    combined_raw = pd.concat([rl_metrics, proxy], ignore_index=True, sort=False)
    metadata = {
        "case_id",
        "object_key",
        "rl_outcome",
        "y_fail",
        "label_authority",
        "proxy_reason",
        "source_experiment",
    }
    feature_columns: list[str] = []
    dictionary_rows: list[dict[str, str]] = []
    seen_vectors: dict[str, str] = {}
    for column in combined_raw.columns:
        if column in metadata:
            dictionary_rows.append(
                {
                    "feature": column,
                    "included": "false",
                    "reason": "metadata",
                }
            )
            continue
        reason = standard_feature_exclusion(column, combined_raw[column])
        if reason is None:
            vector_hash = hashlib.sha256(
                np.asarray(combined_raw[column], dtype=np.float64).tobytes()
            ).hexdigest()
            if vector_hash in seen_vectors:
                reason = f"exact_duplicate_of:{seen_vectors[vector_hash]}"
            else:
                seen_vectors[vector_hash] = column
        included = reason is None
        dictionary_rows.append(
            {
                "feature": column,
                "included": str(included).lower(),
                "reason": "continuous_physical_metric" if included else str(reason),
            }
        )
        if included:
            feature_columns.append(column)

    meta_columns = [
        "case_id",
        "object_key",
        "rl_outcome",
        "y_fail",
        "label_authority",
        "source_experiment",
    ]
    rl_panel = rl_metrics[meta_columns + feature_columns].copy()
    proxy_panel = proxy[meta_columns + ["proxy_reason"] + feature_columns].copy()
    dictionary = pd.DataFrame(dictionary_rows)
    return rl_panel, proxy_panel, dictionary


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    success_recall = tn / max(tn + fp, 1)
    failure_recall = tp / max(tp + fn, 1)
    return {
        "n": int(len(y_true)),
        "success_n": int(np.sum(y_true == 0)),
        "failure_n": int(np.sum(y_true == 1)),
        "tn_success_accept": tn,
        "fp_success_reject": fp,
        "fn_failure_accept": fn,
        "tp_failure_reject": tp,
        "errors": int(fp + fn),
        "accuracy": float((tn + tp) / max(len(y_true), 1)),
        "balanced_accuracy": float((success_recall + failure_recall) / 2.0),
        "failure_precision": float(tp / max(tp + fp, 1)),
        "failure_recall": float(failure_recall),
        "success_precision": float(tn / max(tn + fn, 1)),
        "success_recall": float(success_recall),
    }


def feature_columns(panel: pd.DataFrame) -> list[str]:
    metadata = {
        "case_id",
        "object_key",
        "rl_outcome",
        "y_fail",
        "label_authority",
        "source_experiment",
        "proxy_reason",
        "label_kind",
    }
    return [
        column
        for column in panel.columns
        if column not in metadata and pd.api.types.is_numeric_dtype(panel[column])
    ]


def usable_feature_columns(
    panel: pd.DataFrame, columns: Sequence[str] | None = None
) -> list[str]:
    candidates = list(columns) if columns is not None else feature_columns(panel)
    usable = []
    for column in candidates:
        values = panel[column].to_numpy(dtype=float)
        if np.all(np.isfinite(values)) and float(np.std(values)) > 1e-12:
            usable.append(column)
    return usable


def threshold_candidates(values: np.ndarray) -> np.ndarray:
    unique = np.unique(np.asarray(values, dtype=float))
    if len(unique) == 1:
        return unique.copy()
    midpoints = (unique[:-1] + unique[1:]) / 2.0
    span = max(float(unique[-1] - unique[0]), 1.0)
    return np.concatenate(
        ([unique[0] - span], midpoints, [unique[-1] + span])
    )


def threshold_sort_key(
    metrics: dict[str, float | int], class_gap: float
) -> tuple[float, ...]:
    return (
        float(metrics["errors"]),
        -float(metrics["balanced_accuracy"]),
        -float(metrics["failure_recall"]),
        -float(metrics["success_recall"]),
        -float(class_gap),
    )


def best_threshold_for_feature(
    values: np.ndarray, y_fail: np.ndarray, feature: str
) -> ThresholdFit:
    values = np.asarray(values, dtype=float)
    y_fail = np.asarray(y_fail, dtype=int)
    best: ThresholdFit | None = None
    for threshold in threshold_candidates(values):
        for direction in (">=", "<="):
            prediction = (
                (values >= threshold).astype(int)
                if direction == ">="
                else (values <= threshold).astype(int)
            )
            metrics = binary_metrics(y_fail, prediction)
            success_values = values[y_fail == 0]
            failure_values = values[y_fail == 1]
            if direction == ">=":
                class_gap = float(
                    np.min(failure_values) - np.max(success_values)
                )
            else:
                class_gap = float(
                    np.min(success_values) - np.max(failure_values)
                )
            candidate = ThresholdFit(
                feature, direction, float(threshold), metrics, class_gap
            )
            if best is None or threshold_sort_key(
                candidate.train_metrics, candidate.class_gap
            ) < threshold_sort_key(best.train_metrics, best.class_gap):
                best = candidate
    assert best is not None
    return best


def single_metric_sweep(panel: pd.DataFrame) -> pd.DataFrame:
    y = panel["y_fail"].to_numpy(dtype=int)
    rows = []
    for column in usable_feature_columns(panel):
        fit = best_threshold_for_feature(panel[column].to_numpy(dtype=float), y, column)
        rows.append(
            {
                "feature": fit.feature,
                "direction": fit.direction,
                "threshold": fit.threshold,
                "class_gap": fit.class_gap,
                "exact_separation": fit.train_metrics["errors"] == 0,
                **fit.train_metrics,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["errors", "balanced_accuracy", "failure_recall", "success_recall", "class_gap"],
        ascending=[True, False, False, False, False],
        kind="stable",
    )


def linear_sort_key(fit: LinearFit) -> tuple[float, ...]:
    return (
        float(fit.train_metrics["errors"]),
        float(len(fit.features)),
        -float(fit.train_metrics["balanced_accuracy"]),
        -float(fit.train_metrics["failure_recall"]),
        -float(fit.train_metrics["success_recall"]),
        -float(fit.geometric_margin),
    )


def fit_linear_svc_subset(
    x_scaled: np.ndarray,
    y: np.ndarray,
    columns: Sequence[str],
    subset: tuple[int, ...],
    scaler: StandardScaler,
    random_seed: int,
) -> LinearFit:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model = LinearSVC(
            C=1e5,
            class_weight="balanced",
            dual="auto",
            max_iter=100_000,
            tol=1e-7,
            random_state=random_seed,
        )
        model.fit(x_scaled[:, subset], y)
    decision = model.decision_function(x_scaled[:, subset])
    prediction = (decision >= 0).astype(int)
    norm = max(float(np.linalg.norm(model.coef_[0])), 1e-12)
    geometric_margin = float(np.min(np.abs(decision)) / norm)
    return LinearFit(
        procedure="beam_linear_svm",
        features=[columns[index] for index in subset],
        scaler_mean=scaler.mean_[list(subset)],
        scaler_scale=scaler.scale_[list(subset)],
        coef_z=np.asarray(model.coef_[0], dtype=float),
        intercept_z=float(model.intercept_[0]),
        train_metrics=binary_metrics(y, prediction),
        geometric_margin=geometric_margin,
        hyperparameter="C=1e5",
    )


def fit_beam_linear(
    panel: pd.DataFrame,
    *,
    max_features: int = 5,
    candidate_limit: int = 18,
    beam_width: int = 30,
    random_seed: int = 180,
) -> LinearFit:
    columns = usable_feature_columns(panel)
    if not columns:
        raise ValueError("no usable numeric features")
    y = panel["y_fail"].to_numpy(dtype=int)
    if len(np.unique(y)) != 2:
        raise ValueError("both classes are required")
    x = panel[columns].to_numpy(dtype=float)
    scaler = StandardScaler().fit(x)
    x_scaled = scaler.transform(x)

    threshold_ranking = []
    for index, column in enumerate(columns):
        threshold_fit = best_threshold_for_feature(x[:, index], y, column)
        threshold_ranking.append(
            (
                threshold_sort_key(
                    threshold_fit.train_metrics, threshold_fit.class_gap
                ),
                index,
            )
        )
    threshold_ranking.sort()
    candidate_indices = [
        index for _, index in threshold_ranking[: min(candidate_limit, len(columns))]
    ]

    all_fits: list[LinearFit] = []
    beam: list[tuple[tuple[int, ...], LinearFit]] = []
    for size in range(1, min(max_features, len(candidate_indices)) + 1):
        if size == 1:
            subsets: Iterable[tuple[int, ...]] = (
                (index,) for index in candidate_indices
            )
        else:
            generated: set[tuple[int, ...]] = set()
            for subset, _ in beam:
                for index in candidate_indices:
                    if index not in subset:
                        generated.add(tuple(sorted((*subset, index))))
            subsets = sorted(generated)
        candidates: list[tuple[tuple[int, ...], LinearFit]] = []
        for subset in subsets:
            fit = fit_linear_svc_subset(
                x_scaled, y, columns, subset, scaler, random_seed
            )
            candidates.append((subset, fit))
            all_fits.append(fit)
        candidates.sort(key=lambda item: linear_sort_key(item[1]))
        beam = candidates[:beam_width]
    if not all_fits:
        raise RuntimeError("beam search produced no fit")
    return min(all_fits, key=linear_sort_key)


def fit_l1_logistic(
    panel: pd.DataFrame,
    *,
    max_features: int = 5,
    random_seed: int = 180,
) -> LinearFit:
    columns = usable_feature_columns(panel)
    y = panel["y_fail"].to_numpy(dtype=int)
    if not columns or len(np.unique(y)) != 2:
        raise ValueError("usable features and both classes are required")
    x = panel[columns].to_numpy(dtype=float)
    scaler = StandardScaler().fit(x)
    x_scaled = scaler.transform(x)
    fits: list[LinearFit] = []
    fallback: list[LinearFit] = []
    for c_value in np.logspace(-3, 3, 19):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model = LogisticRegression(
                C=float(c_value),
                l1_ratio=1.0,
                solver="liblinear",
                class_weight="balanced",
                max_iter=20_000,
                random_state=random_seed,
            )
            model.fit(x_scaled, y)
        nonzero = np.flatnonzero(np.abs(model.coef_[0]) > 1e-10)
        if len(nonzero) == 0:
            continue
        decision = model.decision_function(x_scaled)
        prediction = (decision >= 0).astype(int)
        norm = max(float(np.linalg.norm(model.coef_[0, nonzero])), 1e-12)
        fit = LinearFit(
            procedure="l1_logistic",
            features=[columns[index] for index in nonzero],
            scaler_mean=scaler.mean_[nonzero],
            scaler_scale=scaler.scale_[nonzero],
            coef_z=np.asarray(model.coef_[0, nonzero], dtype=float),
            intercept_z=float(model.intercept_[0]),
            train_metrics=binary_metrics(y, prediction),
            geometric_margin=float(np.min(np.abs(decision)) / norm),
            hyperparameter=f"C={c_value:.8g}",
        )
        fallback.append(fit)
        if len(nonzero) <= max_features:
            fits.append(fit)
    if fits:
        return min(fits, key=linear_sort_key)
    if not fallback:
        return fit_beam_linear(
            panel, max_features=1, candidate_limit=18, beam_width=18
        )
    # This path is unlikely, but still guarantees the promised 1--5 feature output.
    too_dense = min(fallback, key=linear_sort_key)
    ranked = np.argsort(-np.abs(too_dense.coef_z))[:max_features]
    selected = [too_dense.features[index] for index in ranked]
    reduced = panel[
        [
            "case_id",
            "object_key",
            "rl_outcome",
            "y_fail",
            *selected,
        ]
    ].copy()
    return fit_beam_linear(
        reduced,
        max_features=max_features,
        candidate_limit=max_features,
        beam_width=max_features,
        random_seed=random_seed,
    )


def fit_all_feature_linear(
    panel: pd.DataFrame, *, random_seed: int = 180
) -> LinearFit:
    """Fit a deliberately high-dimensional diagnostic linear SVM.

    This model is included only to answer whether the available feature space
    is linearly separable at all.  It is not an interpretable gate candidate.
    """
    columns = usable_feature_columns(panel)
    y = panel["y_fail"].to_numpy(dtype=int)
    if not columns or len(np.unique(y)) != 2:
        raise ValueError("usable features and both classes are required")
    x = panel[columns].to_numpy(dtype=float)
    scaler = StandardScaler().fit(x)
    x_scaled = scaler.transform(x)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model = LinearSVC(
            C=1e5,
            class_weight="balanced",
            dual="auto",
            max_iter=100_000,
            tol=1e-7,
            random_state=random_seed,
        )
        model.fit(x_scaled, y)
    decision = model.decision_function(x_scaled)
    prediction = (decision >= 0).astype(int)
    norm = max(float(np.linalg.norm(model.coef_[0])), 1e-12)
    return LinearFit(
        procedure="all_feature_linear_svm_diagnostic",
        features=columns,
        scaler_mean=scaler.mean_,
        scaler_scale=scaler.scale_,
        coef_z=np.asarray(model.coef_[0], dtype=float),
        intercept_z=float(model.intercept_[0]),
        train_metrics=binary_metrics(y, prediction),
        geometric_margin=float(np.min(np.abs(decision)) / norm),
        hyperparameter="C=1e5",
    )


def formula_string(fit: LinearFit) -> str:
    if len(fit.features) > 10:
        return (
            f"{len(fit.features)}-feature raw-unit formula; "
            "see model_coefficients.tsv"
        )
    terms = [f"{fit.intercept_raw:+.10g}"]
    terms.extend(
        f"{coefficient:+.10g}*{feature}"
        for feature, coefficient in zip(fit.features, fit.coef_raw)
    )
    return " ".join(terms) + " >= 0 => predict failure"


def apparent_model_outputs(
    panel_name: str, panel: pd.DataFrame, random_seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, LinearFit]]:
    fits = {
        "beam_linear_svm": fit_beam_linear(panel, random_seed=random_seed),
        "l1_logistic": fit_l1_logistic(panel, random_seed=random_seed),
        "all_feature_linear_svm_diagnostic": fit_all_feature_linear(
            panel, random_seed=random_seed
        ),
    }
    model_rows = []
    coefficient_rows = []
    prediction_rows = []
    for procedure, fit in fits.items():
        prediction = fit.predict(panel)
        decision = fit.decision(panel)
        model_rows.append(
            {
                "panel": panel_name,
                "procedure": procedure,
                "feature_count": len(fit.features),
                "features": ";".join(fit.features),
                "hyperparameter": fit.hyperparameter,
                "geometric_margin_z": fit.geometric_margin,
                "formula_raw_units": formula_string(fit),
                **fit.train_metrics,
            }
        )
        coefficient_rows.append(
            {
                "panel": panel_name,
                "procedure": procedure,
                "feature": "__intercept__",
                "coefficient_z": fit.intercept_z,
                "coefficient_raw": fit.intercept_raw,
                "scaler_mean": math.nan,
                "scaler_scale": math.nan,
            }
        )
        for index, feature in enumerate(fit.features):
            coefficient_rows.append(
                {
                    "panel": panel_name,
                    "procedure": procedure,
                    "feature": feature,
                    "coefficient_z": fit.coef_z[index],
                    "coefficient_raw": fit.coef_raw[index],
                    "scaler_mean": fit.scaler_mean[index],
                    "scaler_scale": fit.scaler_scale[index],
                }
            )
        for row_index, row in panel.reset_index(drop=True).iterrows():
            prediction_rows.append(
                {
                    "panel": panel_name,
                    "procedure": procedure,
                    "case_id": row["case_id"],
                    "object_key": row["object_key"],
                    "label_kind": row.get("label_kind", "rl_observed"),
                    "actual_y_fail": int(row["y_fail"]),
                    "predicted_y_fail": int(prediction[row_index]),
                    "decision_score": float(decision[row_index]),
                    "correct": int(prediction[row_index]) == int(row["y_fail"]),
                }
            )
    return (
        pd.DataFrame(model_rows),
        pd.DataFrame(coefficient_rows),
        pd.DataFrame(prediction_rows),
        fits,
    )


def choose_best_threshold(panel: pd.DataFrame) -> ThresholdFit:
    y = panel["y_fail"].to_numpy(dtype=int)
    fits = [
        best_threshold_for_feature(panel[column].to_numpy(dtype=float), y, column)
        for column in usable_feature_columns(panel)
    ]
    return min(
        fits,
        key=lambda fit: (
            *threshold_sort_key(fit.train_metrics, fit.class_gap),
            fit.feature,
        ),
    )


def cv_splits(
    panel: pd.DataFrame, protocol: str, random_seed: int
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    indices = np.arange(len(panel))
    if protocol == "loocv":
        return [
            (f"case:{panel.iloc[test[0]]['case_id']}", train, test)
            for train, test in LeaveOneOut().split(indices)
        ]
    if protocol == "leave_one_object_out":
        return [
            (
                f"object:{obj}",
                indices[panel["object_key"].to_numpy() != obj],
                indices[panel["object_key"].to_numpy() == obj],
            )
            for obj in sorted(panel["object_key"].unique())
        ]
    if protocol == "stratified_5fold":
        splitter = StratifiedKFold(
            n_splits=5, shuffle=True, random_state=random_seed
        )
        return [
            (f"fold:{fold}", train, test)
            for fold, (train, test) in enumerate(
                splitter.split(indices, panel["y_fail"].to_numpy(dtype=int))
            )
        ]
    raise ValueError(f"unknown CV protocol: {protocol}")


def cross_validate(
    panel_name: str,
    panel: pd.DataFrame,
    protocol: str,
    procedure: str,
    random_seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = []
    for fold_id, train_indices, test_indices in cv_splits(
        panel, protocol, random_seed
    ):
        train = panel.iloc[train_indices].copy()
        test = panel.iloc[test_indices].copy()
        if procedure == "best_single_threshold":
            fit: ThresholdFit | LinearFit = choose_best_threshold(train)
            selected = fit.feature
            formula = f"{fit.feature} {fit.direction} {fit.threshold:.10g}"
        elif procedure == "beam_linear_svm":
            fit = fit_beam_linear(train, random_seed=random_seed)
            selected = ";".join(fit.features)
            formula = formula_string(fit)
        elif procedure == "l1_logistic":
            fit = fit_l1_logistic(train, random_seed=random_seed)
            selected = ";".join(fit.features)
            formula = formula_string(fit)
        elif procedure == "all_feature_linear_svm_diagnostic":
            fit = fit_all_feature_linear(train, random_seed=random_seed)
            selected = f"all:{len(fit.features)}"
            formula = formula_string(fit)
        else:
            raise ValueError(procedure)
        prediction = fit.predict(test)
        decision = (
            fit.decision(test)
            if isinstance(fit, LinearFit)
            else np.full(len(test), math.nan)
        )
        for offset, (_, row) in enumerate(test.iterrows()):
            rows.append(
                {
                    "panel": panel_name,
                    "protocol": protocol,
                    "procedure": procedure,
                    "fold": fold_id,
                    "case_id": row["case_id"],
                    "object_key": row["object_key"],
                    "label_kind": row.get("label_kind", "rl_observed"),
                    "actual_y_fail": int(row["y_fail"]),
                    "predicted_y_fail": int(prediction[offset]),
                    "decision_score": float(decision[offset]),
                    "selected_features": selected,
                    "frozen_rule": formula,
                    "correct": int(prediction[offset]) == int(row["y_fail"]),
                }
            )
    predictions = pd.DataFrame(rows)
    metrics = binary_metrics(
        predictions["actual_y_fail"].to_numpy(dtype=int),
        predictions["predicted_y_fail"].to_numpy(dtype=int),
    )
    summary = {
        "panel": panel_name,
        "protocol": protocol,
        "procedure": procedure,
        "folds": int(predictions["fold"].nunique()),
        **metrics,
    }
    return predictions, summary


def stratified_bootstrap_indices(
    y: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    groups = []
    for label in (0, 1):
        indices = np.flatnonzero(y == label)
        groups.append(rng.choice(indices, size=len(indices), replace=True))
    result = np.concatenate(groups)
    rng.shuffle(result)
    return result


def bootstrap_stability(
    panel_name: str,
    panel: pd.DataFrame,
    repetitions: int,
    random_seed: int,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    rng = np.random.default_rng(random_seed)
    y = panel["y_fail"].to_numpy(dtype=int)
    all_features = usable_feature_columns(panel)
    records = []
    exact_count = 0
    for repetition in range(repetitions):
        indices = stratified_bootstrap_indices(y, rng)
        sample = panel.iloc[indices].copy()
        fit = fit_beam_linear(
            sample,
            candidate_limit=min(12, len(all_features)),
            beam_width=12,
            random_seed=random_seed + repetition,
        )
        exact_count += int(fit.train_metrics["errors"] == 0)
        coefficients = dict(zip(fit.features, fit.coef_raw))
        for feature in fit.features:
            records.append(
                {
                    "panel": panel_name,
                    "repetition": repetition,
                    "feature": feature,
                    "coefficient_raw": coefficients[feature],
                    "coefficient_sign": int(np.sign(coefficients[feature])),
                }
            )
    raw = pd.DataFrame(records)
    rows = []
    for feature in all_features:
        selected = raw[raw["feature"] == feature]
        values = selected["coefficient_raw"].to_numpy(dtype=float)
        rows.append(
            {
                "panel": panel_name,
                "feature": feature,
                "selection_frequency": float(len(selected) / repetitions),
                "selected_count": int(len(selected)),
                "positive_sign_fraction_given_selected": (
                    float(np.mean(values > 0)) if len(values) else math.nan
                ),
                "coef_raw_median_given_selected": (
                    float(np.median(values)) if len(values) else math.nan
                ),
                "coef_raw_p05_given_selected": (
                    float(np.quantile(values, 0.05)) if len(values) else math.nan
                ),
                "coef_raw_p95_given_selected": (
                    float(np.quantile(values, 0.95)) if len(values) else math.nan
                ),
            }
        )
    summary = {
        "panel": panel_name,
        "repetitions": repetitions,
        "bootstrap_exact_fit_fraction": float(exact_count / repetitions),
    }
    return (
        pd.DataFrame(rows).sort_values(
            "selection_frequency", ascending=False, kind="stable"
        ),
        summary,
    )


def permutation_baseline(
    panel_name: str,
    panel: pd.DataFrame,
    repetitions: int,
    random_seed: int,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    rng = np.random.default_rng(random_seed)
    base = panel.copy()
    actual_predictions, actual_summary = cross_validate(
        panel_name,
        base,
        "stratified_5fold",
        "l1_logistic",
        random_seed,
    )
    del actual_predictions
    actual_score = float(actual_summary["balanced_accuracy"])
    rows = []
    original_y = base["y_fail"].to_numpy(dtype=int)
    for repetition in range(repetitions):
        shuffled = base.copy()
        shuffled["y_fail"] = rng.permutation(original_y)
        _, summary = cross_validate(
            panel_name,
            shuffled,
            "stratified_5fold",
            "l1_logistic",
            random_seed + repetition + 1,
        )
        rows.append(
            {
                "panel": panel_name,
                "repetition": repetition,
                "permuted_balanced_accuracy": summary["balanced_accuracy"],
            }
        )
    distribution = pd.DataFrame(rows)
    p_value = float(
        (1 + np.sum(distribution["permuted_balanced_accuracy"] >= actual_score))
        / (repetitions + 1)
    )
    summary = {
        "panel": panel_name,
        "procedure": "l1_logistic_stratified_5fold",
        "actual_balanced_accuracy": actual_score,
        "permutations": repetitions,
        "permutation_mean": float(
            distribution["permuted_balanced_accuracy"].mean()
        ),
        "permutation_p95": float(
            distribution["permuted_balanced_accuracy"].quantile(0.95)
        ),
        "permutation_p_value": p_value,
    }
    return distribution, summary


def proxy_stress_test(
    rl_panel: pd.DataFrame,
    proxy_panel: pd.DataFrame,
    fits: dict[str, LinearFit],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prediction_rows = []
    summary_rows = []
    for procedure, fit in fits.items():
        complete = proxy_panel[fit.features].notna().all(axis=1)
        covered = proxy_panel[complete].copy()
        prediction = fit.predict(covered) if len(covered) else np.array([], dtype=int)
        decision = fit.decision(covered) if len(covered) else np.array([], dtype=float)
        for offset, (_, row) in enumerate(covered.iterrows()):
            prediction_rows.append(
                {
                    "procedure": procedure,
                    "case_id": row["case_id"],
                    "object_key": row["object_key"],
                    "proxy_reason": row["proxy_reason"],
                    "predicted_y_fail": int(prediction[offset]),
                    "decision_score": float(decision[offset]),
                    "rejected_as_failure": bool(prediction[offset] == 1),
                }
            )
        summary_rows.append(
            {
                "procedure": procedure,
                "frozen_training_panel": "standardized_rl36",
                "selected_features": ";".join(fit.features),
                "rl_train_success_n": int(np.sum(rl_panel["y_fail"] == 0)),
                "rl_train_failure_n": int(np.sum(rl_panel["y_fail"] == 1)),
                "proxy_total_n": int(len(proxy_panel)),
                "proxy_covered_n": int(np.sum(complete)),
                "proxy_rejected_n": int(np.sum(prediction == 1)),
                "proxy_rejection_rate": (
                    float(np.mean(prediction == 1)) if len(prediction) else math.nan
                ),
            }
        )
    return pd.DataFrame(prediction_rows), pd.DataFrame(summary_rows)


def write_tsv(frame: pd.DataFrame, path: Path) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        frame.to_csv(path, sep="\t", index=False, float_format="%.10g")


def parse_args() -> argparse.Namespace:
    script_path = Path(__file__).resolve()
    default_workspace = script_path.parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, default=default_workspace)
    parser.add_argument("--sugar-root", type=Path, default=R018_ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_workspace
        / "results/E180/rl_metric_separability",
    )
    parser.add_argument("--bootstrap", type=int, default=100)
    parser.add_argument("--permutations", type=int, default=200)
    parser.add_argument("--random-seed", type=int, default=180)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    labels = frozen_labels()
    reference, lineage = build_reference_panel(labels, args.sugar_root)
    standard_rl, proxy, feature_dictionary = build_standardized_panels(
        labels, args.workspace_root
    )
    combined = pd.concat(
        [
            standard_rl.assign(label_kind="rl_observed"),
            proxy.assign(label_kind="proxy_negative"),
        ],
        ignore_index=True,
        sort=False,
    )

    write_tsv(labels, args.output_dir / "frozen_rl_labels.tsv")
    write_tsv(lineage, args.output_dir / "reference_release_lineage.tsv")
    write_tsv(reference, args.output_dir / "reference_input_features.tsv")
    write_tsv(standard_rl, args.output_dir / "standardized_rl_features.tsv")
    write_tsv(proxy, args.output_dir / "proxy_negative_features.tsv")
    write_tsv(combined, args.output_dir / "standardized_rl_plus_proxy_features.tsv")
    write_tsv(feature_dictionary, args.output_dir / "standard_feature_dictionary.tsv")

    panels = {
        "reference_rl38": reference,
        "standardized_rl36": standard_rl,
        "standardized_rl_plus_proxy78": combined,
    }
    sweep_frames = []
    model_frames = []
    coefficient_frames = []
    apparent_prediction_frames = []
    fits_by_panel: dict[str, dict[str, LinearFit]] = {}
    for panel_name, panel in panels.items():
        sweep = single_metric_sweep(panel)
        sweep.insert(0, "panel", panel_name)
        sweep_frames.append(sweep)
        models, coefficients, predictions, fits = apparent_model_outputs(
            panel_name, panel, args.random_seed
        )
        model_frames.append(models)
        coefficient_frames.append(coefficients)
        apparent_prediction_frames.append(predictions)
        fits_by_panel[panel_name] = fits

    single_sweep = pd.concat(sweep_frames, ignore_index=True)
    apparent_models = pd.concat(model_frames, ignore_index=True)
    coefficients = pd.concat(coefficient_frames, ignore_index=True)
    apparent_predictions = pd.concat(
        apparent_prediction_frames, ignore_index=True
    )
    write_tsv(single_sweep, args.output_dir / "single_metric_sweep.tsv")
    write_tsv(apparent_models, args.output_dir / "apparent_linear_models.tsv")
    write_tsv(coefficients, args.output_dir / "model_coefficients.tsv")
    write_tsv(
        apparent_predictions, args.output_dir / "apparent_case_predictions.tsv"
    )

    cv_prediction_frames = []
    cv_summaries = []
    for panel_name, panel in panels.items():
        protocols = (
            ["loocv", "leave_one_object_out"]
            if panel_name != "standardized_rl_plus_proxy78"
            else ["stratified_5fold", "leave_one_object_out"]
        )
        for protocol in protocols:
            for procedure in (
                "best_single_threshold",
                "beam_linear_svm",
                "l1_logistic",
                "all_feature_linear_svm_diagnostic",
            ):
                predictions, summary = cross_validate(
                    panel_name,
                    panel,
                    protocol,
                    procedure,
                    args.random_seed,
                )
                cv_prediction_frames.append(predictions)
                cv_summaries.append(summary)
    cv_predictions = pd.concat(cv_prediction_frames, ignore_index=True)
    cv_summary = pd.DataFrame(cv_summaries)
    write_tsv(cv_predictions, args.output_dir / "cross_validation_predictions.tsv")
    write_tsv(cv_summary, args.output_dir / "cross_validation_summary.tsv")

    bootstrap_frames = []
    bootstrap_summaries = []
    for panel_name in ("reference_rl38", "standardized_rl36"):
        stability, summary = bootstrap_stability(
            panel_name,
            panels[panel_name],
            args.bootstrap,
            args.random_seed,
        )
        bootstrap_frames.append(stability)
        bootstrap_summaries.append(summary)
    write_tsv(
        pd.concat(bootstrap_frames, ignore_index=True),
        args.output_dir / "bootstrap_feature_stability.tsv",
    )
    write_tsv(
        pd.DataFrame(bootstrap_summaries),
        args.output_dir / "bootstrap_summary.tsv",
    )

    permutation_frames = []
    permutation_summaries = []
    for panel_name in ("reference_rl38", "standardized_rl36"):
        distribution, summary = permutation_baseline(
            panel_name,
            panels[panel_name],
            args.permutations,
            args.random_seed,
        )
        permutation_frames.append(distribution)
        permutation_summaries.append(summary)
    write_tsv(
        pd.concat(permutation_frames, ignore_index=True),
        args.output_dir / "permutation_baseline.tsv",
    )
    write_tsv(
        pd.DataFrame(permutation_summaries),
        args.output_dir / "permutation_summary.tsv",
    )

    proxy_predictions, proxy_summary = proxy_stress_test(
        standard_rl, proxy, fits_by_panel["standardized_rl36"]
    )
    write_tsv(
        proxy_predictions, args.output_dir / "proxy_negative_predictions.tsv"
    )
    write_tsv(proxy_summary, args.output_dir / "proxy_negative_stress_test.tsv")

    audit_summary = {
        "experiment": "E180",
        "failure_is_positive_class": True,
        "label_counts": {
            "rl_total": int(len(labels)),
            "rl_success": int(np.sum(labels["y_fail"] == 0)),
            "rl_fail": int(np.sum(labels["y_fail"] == 1)),
            "standardized_rl_total": int(len(standard_rl)),
            "proxy_negative_total": int(len(proxy)),
        },
        "panel_feature_counts": {
            panel_name: len(usable_feature_columns(panel))
            for panel_name, panel in panels.items()
        },
        "single_metric_exact_counts": {
            panel_name: int(
                np.sum(
                    (single_sweep["panel"] == panel_name)
                    & single_sweep["exact_separation"]
                )
            )
            for panel_name in panels
        },
        "apparent_models": apparent_models.to_dict("records"),
        "cross_validation": cv_summary.to_dict("records"),
        "bootstrap": bootstrap_summaries,
        "permutation": permutation_summaries,
        "proxy_stress": proxy_summary.to_dict("records"),
        "caveat": (
            "Apparent separation is descriptive. Proxy negatives are not observed "
            "RL failures, and no model is a production gate without independent "
            "held-out RL validation."
        ),
    }
    (args.output_dir / "audit_summary.json").write_text(
        json.dumps(audit_summary, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(audit_summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
