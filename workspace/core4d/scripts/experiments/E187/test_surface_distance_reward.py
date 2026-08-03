#!/usr/bin/env python3
"""Pure-function contracts for the E187 distance-continuation reward."""

from __future__ import annotations

import math

import torch

from spider.config import Config
from spider.rewards.surface_distance import (
    surface_distance_score,
    surface_distance_support_mask,
    validate_surface_distance_score_parameters,
)


def continuation_formula(distance_m: torch.Tensor) -> torch.Tensor:
    """Independent preregistered E187 formula authority."""
    smooth_abs = torch.sqrt(torch.square(distance_m) + 0.001**2) - 0.001
    return 0.25 * torch.exp(-smooth_abs / 0.050) + 0.75 * torch.exp(-smooth_abs / 0.015)


def test_default_off_config() -> None:
    """New fields must not opt historical configs into E187 behavior."""
    config = Config()
    assert config.surface_band_score_mode == "one_sided"
    assert config.surface_band_continuation_far_weight == 0.25
    assert config.surface_band_continuation_near_weight == 0.75
    assert config.surface_band_continuation_far_scale_m == 0.050
    assert config.surface_band_continuation_near_scale_m == 0.015
    assert config.surface_band_continuation_smooth_delta_m == 0.001


def test_legacy_scores_and_support_are_exact() -> None:
    """Pure legacy branches reproduce the former inline expressions exactly."""
    distance = torch.linspace(-0.010, 0.010, 401, dtype=torch.float64)
    sigma = 0.0015
    one_sided = surface_distance_score(distance, mode="one_sided", sigma_m=sigma)
    symmetric = surface_distance_score(distance, mode="symmetric_abs", sigma_m=sigma)
    torch.testing.assert_close(
        one_sided,
        torch.exp(-torch.clamp(distance, min=0.0) / sigma),
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        symmetric,
        torch.exp(-torch.abs(distance) / sigma),
        atol=0.0,
        rtol=0.0,
    )
    expected_mask = (distance >= -0.001) & (distance <= 0.003)
    for mode in ("one_sided", "symmetric_abs"):
        actual_mask = surface_distance_support_mask(
            distance,
            mode=mode,
            band_min_sdf_m=-0.001,
            band_width_m=0.003,
        )
        assert torch.equal(actual_mask, expected_mask)


def test_continuation_formula_and_shape_contracts() -> None:
    """The implementation matches the preregistered formula and shape gates."""
    distance = torch.linspace(-0.100, 0.100, 20001, dtype=torch.float64)
    actual = surface_distance_score(
        distance,
        mode="distance_continuation",
        sigma_m=0.0015,
    )
    expected = continuation_formula(distance)
    assert float(torch.max(torch.abs(actual - expected))) <= 1e-7
    assert bool(torch.isfinite(actual).all())
    torch.testing.assert_close(
        actual, torch.flip(actual, dims=(0,)), atol=1e-14, rtol=0.0
    )
    center = len(distance) // 2
    assert actual[center].item() == 1.0
    assert bool(torch.all(torch.diff(actual[center:]) <= 0.0))
    assert bool(torch.all(torch.diff(actual[: center + 1]) >= 0.0))
    support = surface_distance_support_mask(
        distance,
        mode="distance_continuation",
        band_min_sdf_m=-0.001,
        band_width_m=0.003,
    )
    assert bool(support.all())


def test_bucket003_far_field_support() -> None:
    """The observed E186 bucket003 48.6mm gap is non-trivial under E187."""
    distance = torch.tensor([0.0458, 0.0486], dtype=torch.float64)
    score = surface_distance_score(
        distance,
        mode="distance_continuation",
        sigma_m=0.0015,
    )
    assert bool(torch.all(score > 0.05))
    assert math.isclose(score[1].item(), continuation_formula(distance)[1].item())


def test_invalid_continuation_parameters_fail_closed() -> None:
    """Invalid opt-in parameters must fail before a GPU run starts."""
    valid = {
        "mode": "distance_continuation",
        "sigma_m": 0.0015,
        "continuation_far_weight": 0.25,
        "continuation_near_weight": 0.75,
        "continuation_far_scale_m": 0.050,
        "continuation_near_scale_m": 0.015,
        "continuation_smooth_delta_m": 0.001,
    }
    validate_surface_distance_score_parameters(**valid)
    invalid_updates = (
        {"mode": "unknown"},
        {"continuation_far_weight": 0.5},
        {"continuation_near_scale_m": 0.0},
        {"continuation_smooth_delta_m": float("nan")},
    )
    for update in invalid_updates:
        parameters = valid | update
        try:
            validate_surface_distance_score_parameters(**parameters)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid parameters accepted: {update}")


def main() -> int:
    """Run all direct-main contracts without pytest collection dependencies."""
    tests = (
        test_default_off_config,
        test_legacy_scores_and_support_are_exact,
        test_continuation_formula_and_shape_contracts,
        test_bucket003_far_field_support,
        test_invalid_continuation_parameters_fail_closed,
    )
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"E187_SURFACE_DISTANCE_TESTS=PASS count={len(tests)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
