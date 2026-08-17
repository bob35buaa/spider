#!/usr/bin/env python3
"""E201 three-tier data-filter funnel — single source of truth for thresholds.

14 gates split into:
  * 4 HARD gates (no wide/narrow band, enforced in every layer): fall / body_z /
    ankle_jerk_p95 / obj_speed_max. Any hard-gate failure => L1 reject.
  * 10 BANDED gates (wide/narrow dual thresholds): 6 tracking + contact / release
    / hand_pen / leg_pen.

Layer decision (see classify_funnel.py):
  hard fail                              -> L1 reject
  any banded fails WIDE                  -> L1 reject
  all banded pass WIDE, >=1 fails NARROW -> L2 middle band (human review)
  all banded pass NARROW                 -> L3 (orig auto-accept; aug needs family
                                            consistency, else L3-review)

Decisions frozen by user 2026-08-17:
  - contact wide = 0.40 (fixes the draft's 0.51 direction bug -> restores wide⊇narrow)
  - narrow leg_pen<=0.20 / hand_pen<=0.32 intentionally looser than E178 canonical
  - motion health (ankle_jerk / obj_speed) is a global hard gate
  - contact gate uses the standard in_mask contact frac (not the 3mm variant)
  - tracking wide band = narrow + 1 (kept deliberately tight)
  - body_z is a hard gate at <=0.20 (no band)
"""

from __future__ import annotations

import math
from typing import Any

# --- hard gates: (name, field, op, threshold). op "fall" == fall_flag must be false.
HARD_GATES: list[tuple[str, str, str, float]] = [
    ("fall", "fall_flag", "fall", 0.0),
    ("body_z", "body_z_err_p95_m", "<=", 0.20),
    ("ankle_jerk", "ankle_jerk_p95", "<", 1000.0),
    ("obj_speed", "obj_speed_max", "<", 3.0),
]

# --- banded gates: (name, field, op, narrow, wide). op in {"<=", ">="}.
BANDED_GATES: list[tuple[str, str, str, float, float]] = [
    ("root_pos", "track_root_pos_err_cm_mean", "<=", 20.0, 21.0),
    ("root_ori", "track_root_ori_err_deg_mean", "<=", 20.0, 21.0),
    ("eef_pos", "track_eef_pos_err_cm_mean", "<=", 20.0, 21.0),
    ("eef_ori", "track_eef_ori_err_deg_mean", "<=", 20.0, 21.0),
    ("obj_pos", "track_obj_pos_err_cm_mean", "<=", 20.0, 21.0),
    ("obj_ori", "track_obj_ori_err_deg_mean", "<=", 10.0, 11.0),
    ("contact", "hand_object_physics_contact_in_mask_frac", ">=", 0.50, 0.40),
    ("release", "hand_object_release_false_contact_3mm_frac", "<=", 0.30, 0.60),
    ("hand_pen", "hand_object_physics_penetration_3mm_frame_frac", "<=", 0.32, 0.55),
    ("leg_pen", "leg_penetration_frac", "<=", 0.20, 0.40),
]

# family = the four variants under one case_id
FAMILY_VARIANTS = ("orig", "trans0", "trans1", "trans2")


def passes(op: str, value: float, thr: float) -> bool:
    """Threshold test; a non-finite metric never passes."""
    if not math.isfinite(value):
        return False
    if op == "<=":
        return value <= thr
    if op == "<":
        return value < thr
    if op == ">=":
        return value >= thr
    raise ValueError(f"unknown op {op!r}")


def assert_monotonic() -> None:
    """Verify the funnel invariant: narrow-pass ⟹ wide-pass for every banded gate.

    For "<=" gates wide must be >= narrow; for ">=" gates wide must be <= narrow.
    A violation would let a narrow-passing rollout fail the wide (L1) filter.
    """
    for name, _field, op, narrow, wide in BANDED_GATES:
        if op == "<=" and wide < narrow:
            raise AssertionError(f"{name}: wide {wide} < narrow {narrow} (<=)")
        if op == ">=" and wide > narrow:
            raise AssertionError(f"{name}: wide {wide} > narrow {narrow} (>=)")


def hard_gate_result(row: dict[str, Any]) -> tuple[bool, list[str]]:
    """Return (all_hard_pass, failed_hard_gate_names)."""
    failed: list[str] = []
    for name, field, op, thr in HARD_GATES:
        if op == "fall":
            fell = str(row.get(field, "")).strip().lower() in ("true", "1")
            if fell:
                failed.append(name)
            continue
        if not passes(op, _finite(row.get(field)), thr):
            failed.append(name)
    return (not failed), failed


def banded_gate_result(row: dict[str, Any], caliber: str) -> tuple[bool, list[str]]:
    """Evaluate the 10 banded gates at 'narrow' or 'wide'. Returns (all_pass, failed)."""
    idx = 3 if caliber == "narrow" else 4
    failed: list[str] = []
    for gate in BANDED_GATES:
        name, field, op = gate[0], gate[1], gate[2]
        thr = gate[idx]
        if not passes(op, _finite(row.get(field)), thr):
            failed.append(name)
    return (not failed), failed


def _finite(value: Any) -> float:
    try:
        f = float(value)
        return f if math.isfinite(f) else math.nan
    except (TypeError, ValueError):
        return math.nan


if __name__ == "__main__":
    assert_monotonic()
    print("[funnel_config] monotonic invariant OK "
          f"({len(HARD_GATES)} hard + {len(BANDED_GATES)} banded = "
          f"{len(HARD_GATES) + len(BANDED_GATES)} gates)")
