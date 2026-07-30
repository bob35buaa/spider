from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


MODULE_PATH = Path(__file__).with_name(
    "eval_E180_rl_metric_separability.py"
)
SPEC = importlib.util.spec_from_file_location("eval_e180", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
E180 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = E180
SPEC.loader.exec_module(E180)


def test_frozen_labels_have_expected_authority_counts() -> None:
    labels = E180.frozen_labels()
    assert len(labels) == 38
    assert labels["case_id"].nunique() == 38
    assert labels["rl_outcome"].value_counts().to_dict() == {
        "success": 32,
        "fail": 6,
    }
    assert set(labels.loc[labels["y_fail"] == 1, "case_id"]) == {
        "box004_20231003_2_082_p1",
        "box004_20231003_2_082_p2",
        "box001_20231020_014_p2",
        "box024_20231011_028_p2",
        "box023_20231020_040_p2",
        "box023_20231020_042_p2",
    }


def test_single_threshold_finds_exact_split_in_either_direction() -> None:
    low_fail = E180.best_threshold_for_feature(
        np.array([4.0, 5.0, 1.0, 2.0]),
        np.array([0, 0, 1, 1]),
        "metric",
    )
    assert low_fail.direction == "<="
    assert low_fail.train_metrics["errors"] == 0
    assert low_fail.class_gap > 0

    high_fail = E180.best_threshold_for_feature(
        np.array([1.0, 2.0, 4.0, 5.0]),
        np.array([0, 0, 1, 1]),
        "metric",
    )
    assert high_fail.direction == ">="
    assert high_fail.train_metrics["errors"] == 0


def test_raw_formula_matches_standardized_decision() -> None:
    frame = pd.DataFrame(
        {
            "case_id": ["a", "b", "c", "d", "e", "f"],
            "object_key": ["x"] * 6,
            "rl_outcome": ["success"] * 3 + ["fail"] * 3,
            "y_fail": [0, 0, 0, 1, 1, 1],
            "f1": [0.0, 0.2, 0.4, 1.0, 1.2, 1.4],
            "f2": [1.0, 0.8, 0.9, 0.2, 0.1, 0.0],
        }
    )
    fit = E180.fit_beam_linear(
        frame, max_features=2, candidate_limit=2, beam_width=2
    )
    raw_decision = fit.decision(frame)
    x = frame[fit.features].to_numpy(dtype=float)
    z = (x - fit.scaler_mean) / fit.scaler_scale
    standardized_decision = fit.intercept_z + z @ fit.coef_z
    np.testing.assert_allclose(raw_decision, standardized_decision, atol=1e-10)
    assert fit.train_metrics["errors"] == 0


def test_standard_feature_filter_blocks_identity_and_existing_decisions() -> None:
    numeric = pd.Series([0.1, 0.2, 0.3])
    assert E180.standard_feature_exclusion("body_z_err_p95_m", numeric) is None
    for blocked in (
        "case_id",
        "object_key",
        "numeric_release_pass",
        "manual_use_decision",
        "fall_gate_pass",
        "expected_quality",
        "delta_body_z_err_p95_m",
    ):
        assert (
            E180.standard_feature_exclusion(blocked, numeric)
            == "identity_label_or_existing_gate"
        )
