"""Eval input adapters (E019 P1).

Provides a common ``EvalInputs`` dataclass and source-specific loaders that
let ``paper_metrics`` evaluate trajectories from spider physical rollouts,
holosoma v2 kinematic outputs, and future baselines under one schema.
"""

from .common_inputs import EvalInputs, detect_object_body, repo_root
from .kinematic_to_common import (
    CASE_MAP as HOLOSOMA_V2_CASE_MAP,
    list_available_cases as list_holosoma_v2_cases,
    load_kinematic_inputs,
)

__all__ = [
    "EvalInputs",
    "detect_object_body",
    "repo_root",
    "HOLOSOMA_V2_CASE_MAP",
    "list_holosoma_v2_cases",
    "load_kinematic_inputs",
]
