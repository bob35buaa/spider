#!/usr/bin/env python3
"""Contract tests for E192's observation-only gate diagnostics."""

from __future__ import annotations

import sys
from pathlib import Path

import torch


REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO))

from spider.optimizers.sampling import (  # noqa: E402
    _compute_weights_with_gate_impl,
    _record_selected_hand_gate_stats,
)


def test_selected_stats_do_not_change_selection() -> None:
    rewards = torch.tensor([0.1, 0.9, 0.7, 0.4], dtype=torch.float32)
    valid = torch.tensor([True, True, True, False])
    violation_pct = torch.tensor([0.0, 0.0, 0.02, 0.4], dtype=torch.float32)
    violation_depth = torch.tensor([0.0, 0.0, 0.001, 0.02], dtype=torch.float32)
    args = (rewards, 4, 1.0, 0.5, valid, violation_pct, violation_depth, 0.25, None)
    before = _compute_weights_with_gate_impl(*args)

    selected_before = before[2].clone()
    info: dict[str, float] = {}
    hand_sdf = torch.tensor([-0.008, -0.011, -0.014, -0.025])
    rollout = {"sample_hand_gate_min_sdf": hand_sdf.clone()}
    _record_selected_hand_gate_stats(info, rollout, before[2])

    after = _compute_weights_with_gate_impl(*args)
    assert torch.equal(selected_before, after[2])
    assert torch.equal(rollout["sample_hand_gate_min_sdf"], hand_sdf)
    selected = hand_sdf[selected_before]
    assert info["cem_hand_gate_selected_min_sdf_m"] == selected.min().item()
    assert info["cem_hand_gate_selected_mean_sdf_m"] == selected.mean().item()
    assert info["cem_hand_gate_selected_p05_sdf_m"] == torch.quantile(
        selected.float(), 0.05
    ).item()


def test_missing_hand_gate_is_noop() -> None:
    info = {"sentinel": 1.0}
    _record_selected_hand_gate_stats(info, {}, torch.tensor([0]))
    assert info == {"sentinel": 1.0}


if __name__ == "__main__":
    test_selected_stats_do_not_change_selection()
    test_missing_hand_gate_is_noop()
    print("E192 observation-only diagnostic contract: PASS")
