#!/usr/bin/env python3
"""Deterministic CPU checks for the E169 leg gate contract."""

from __future__ import annotations

import torch

from spider.config import Config
from spider.optimizers.sampling import (
    _cem_any_gate_enabled,
    _cem_min_valid_frac,
    _compute_sample_gate_info,
)


def main() -> int:
    config = Config()
    assert not _cem_any_gate_enabled(config)
    assert _compute_sample_gate_info(config, {}) is None

    config.cem_leg_gate_enabled = True
    config.cem_leg_gate_min_sdf_m = 0.005
    config.cem_leg_gate_max_violation_pct = 0.02
    config.cem_leg_gate_hard_floor_m = -0.005
    config.cem_leg_gate_min_valid_frac = 0.03
    info = {
        "cem_leg_gate_min_sdf": torch.tensor(
            [[0.010, 0.004, -0.006], [0.006, 0.006, 0.010]]
        ),
        "cem_leg_gate_violation": torch.tensor(
            [[0.0, 1.0, 1.0], [0.0, 0.0, 0.0]]
        ),
        "cem_leg_gate_violation_depth": torch.tensor(
            [[0.0, 0.001, 0.011], [0.0, 0.0, 0.0]]
        ),
    }
    result = _compute_sample_gate_info(config, info)
    assert result is not None
    expected = [True, False, False]
    assert result["sample_leg_gate_valid_mask"].tolist() == expected
    assert result["sample_gate_valid_mask"].tolist() == expected
    assert abs(_cem_min_valid_frac(config) - 0.03) < 1e-12

    config.cem_safety_gate_enabled = True
    info.update(
        {
            "cem_body_gate_min_sdf": torch.full((2, 3), 0.02),
            "cem_body_gate_violation": torch.zeros((2, 3)),
            "cem_body_gate_violation_depth": torch.zeros((2, 3)),
        }
    )
    combined = _compute_sample_gate_info(config, info)
    assert combined is not None
    assert combined["sample_body_gate_valid_mask"].all()
    assert combined["sample_gate_valid_mask"].tolist() == expected
    print("E169 leg gate plumbing tests: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
