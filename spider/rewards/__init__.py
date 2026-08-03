"""Reusable reward primitives with behavior-preserving opt-in modes."""

from spider.rewards.surface_distance import (
    surface_distance_score,
    surface_distance_support_mask,
    validate_surface_distance_score_parameters,
)

__all__ = [
    "surface_distance_score",
    "surface_distance_support_mask",
    "validate_surface_distance_score_parameters",
]
