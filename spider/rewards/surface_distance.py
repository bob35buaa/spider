"""Pure surface-distance score functions shared by runtime and audits."""

from __future__ import annotations

import math

import torch

LEGACY_SURFACE_DISTANCE_SCORE_MODES = frozenset({"one_sided", "symmetric_abs"})
DISTANCE_CONTINUATION_SCORE_MODE = "distance_continuation"
SURFACE_DISTANCE_SCORE_MODES = LEGACY_SURFACE_DISTANCE_SCORE_MODES | {
    DISTANCE_CONTINUATION_SCORE_MODE
}


def validate_surface_distance_score_parameters(
    *,
    mode: str,
    sigma_m: float,
    continuation_far_weight: float,
    continuation_near_weight: float,
    continuation_far_scale_m: float,
    continuation_near_scale_m: float,
    continuation_smooth_delta_m: float,
) -> None:
    """Fail closed on unsupported or non-finite surface-score parameters."""
    if mode not in SURFACE_DISTANCE_SCORE_MODES:
        raise ValueError(f"Unsupported surface_band_score_mode={mode!r}")
    if not math.isfinite(sigma_m) or sigma_m <= 0.0:
        raise ValueError("surface_band_sigma must be finite and positive")
    if mode != DISTANCE_CONTINUATION_SCORE_MODE:
        return

    parameters = {
        "surface_band_continuation_far_weight": continuation_far_weight,
        "surface_band_continuation_near_weight": continuation_near_weight,
        "surface_band_continuation_far_scale_m": continuation_far_scale_m,
        "surface_band_continuation_near_scale_m": continuation_near_scale_m,
        "surface_band_continuation_smooth_delta_m": continuation_smooth_delta_m,
    }
    non_finite = [
        name for name, value in parameters.items() if not math.isfinite(value)
    ]
    if non_finite:
        raise ValueError(
            "distance-continuation parameters must be finite: " + ", ".join(non_finite)
        )
    if continuation_far_weight < 0.0 or continuation_near_weight < 0.0:
        raise ValueError("distance-continuation weights must be non-negative")
    if not math.isclose(
        continuation_far_weight + continuation_near_weight,
        1.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("distance-continuation weights must sum to 1")
    if continuation_far_scale_m <= 0.0 or continuation_near_scale_m <= 0.0:
        raise ValueError("distance-continuation scales must be positive")
    if continuation_smooth_delta_m <= 0.0:
        raise ValueError("distance-continuation smooth delta must be positive")


def surface_distance_score(
    sdf_m: torch.Tensor,
    *,
    mode: str,
    sigma_m: float,
    continuation_far_weight: float = 0.25,
    continuation_near_weight: float = 0.75,
    continuation_far_scale_m: float = 0.050,
    continuation_near_scale_m: float = 0.015,
    continuation_smooth_delta_m: float = 0.001,
) -> torch.Tensor:
    """Evaluate the selected nominal-distance score without temporal gating."""
    validate_surface_distance_score_parameters(
        mode=mode,
        sigma_m=sigma_m,
        continuation_far_weight=continuation_far_weight,
        continuation_near_weight=continuation_near_weight,
        continuation_far_scale_m=continuation_far_scale_m,
        continuation_near_scale_m=continuation_near_scale_m,
        continuation_smooth_delta_m=continuation_smooth_delta_m,
    )
    if mode == "one_sided":
        return torch.exp(-torch.clamp(sdf_m, min=0.0) / sigma_m)
    if mode == "symmetric_abs":
        return torch.exp(-torch.abs(sdf_m) / sigma_m)

    delta = torch.as_tensor(
        continuation_smooth_delta_m,
        device=sdf_m.device,
        dtype=sdf_m.dtype,
    )
    smooth_abs = torch.sqrt(torch.square(sdf_m) + torch.square(delta)) - delta
    far = torch.exp(-smooth_abs / continuation_far_scale_m)
    near = torch.exp(-smooth_abs / continuation_near_scale_m)
    return continuation_far_weight * far + continuation_near_weight * near


def surface_distance_support_mask(
    sdf_m: torch.Tensor,
    *,
    mode: str,
    band_min_sdf_m: float,
    band_width_m: float,
) -> torch.Tensor:
    """Return legacy hard-band support or continuation's global support."""
    if mode not in SURFACE_DISTANCE_SCORE_MODES:
        raise ValueError(f"Unsupported surface_band_score_mode={mode!r}")
    if mode == DISTANCE_CONTINUATION_SCORE_MODE:
        return torch.ones_like(sdf_m, dtype=torch.bool)
    return (sdf_m >= band_min_sdf_m) & (sdf_m <= band_width_m)
