#!/usr/bin/env python3
"""Compare short E178 replays from historical and current source trees."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

EXACT_ATOL = 1e-5
PRIMARY_KEYS = (
    "qpos",
    "qvel",
    "ctrl",
    "rew_mean",
    "surface_band_rew_mean",
    "surface_band_score_mean",
    "cem_gate_valid_frac",
    "cem_gate_selected_valid_frac",
    "cem_gate_fallback_used",
    "sample_gate_valid_mask_mean",
)


def _aligned(
    left: np.ndarray, right: np.ndarray
) -> tuple[np.ndarray, np.ndarray] | None:
    """Align equal arrays or a short run with the prefix of a full run."""
    if left.shape == right.shape:
        return left, right
    if left.ndim and right.ndim and left.shape[1:] == right.shape[1:]:
        count = min(left.shape[0], right.shape[0])
        return left[:count], right[:count]
    return None


def _array_diff(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    """Return finite/bool exact-difference evidence for two aligned arrays."""
    aligned = _aligned(left, right)
    if aligned is None:
        return {
            "status": "SHAPE_MISMATCH",
            "left_shape": list(left.shape),
            "right_shape": list(right.shape),
            "pass": False,
        }
    first, second = aligned
    if first.dtype.kind == "b" or second.dtype.kind == "b":
        mismatch = np.asarray(first != second)
        flat = np.flatnonzero(mismatch.reshape(-1))
        return {
            "status": "COMPARED",
            "shape": list(first.shape),
            "mismatch_count": int(flat.size),
            "first_mismatch_flat_index": int(flat[0]) if flat.size else None,
            "max_abs": 0.0 if not flat.size else None,
            "pass": not flat.size,
        }
    first64 = first.astype(np.float64)
    second64 = second.astype(np.float64)
    both_nan = np.isnan(first64) & np.isnan(second64)
    finite = np.isfinite(first64) & np.isfinite(second64)
    invalid_mismatch = ~(finite | both_nan)
    delta = np.zeros(first.shape, dtype=np.float64)
    delta[finite] = np.abs(first64[finite] - second64[finite])
    delta[invalid_mismatch] = math.inf
    flat_fail = np.flatnonzero((delta > EXACT_ATOL).reshape(-1))
    max_abs = float(np.max(delta)) if delta.size else 0.0
    return {
        "status": "COMPARED",
        "shape": list(first.shape),
        "mismatch_count": int(flat_fail.size),
        "first_mismatch_flat_index": int(flat_fail[0]) if flat_fail.size else None,
        "max_abs": max_abs if math.isfinite(max_abs) else None,
        "pass": not flat_fail.size,
    }


def compare(left_path: Path, right_path: Path) -> dict[str, Any]:
    """Compare all common numeric/bool arrays, including primary-key summaries."""
    comparisons: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    with (
        np.load(left_path, allow_pickle=True) as left,
        np.load(right_path, allow_pickle=True) as right,
    ):
        for key in sorted(set(left.files) & set(right.files)):
            first = np.asarray(left[key])
            second = np.asarray(right[key])
            if first.dtype.kind not in "biufc" or second.dtype.kind not in "biufc":
                continue
            comparisons[key] = _array_diff(first, second)
        missing = sorted(set(left.files) ^ set(right.files))
    failed = sorted(key for key, item in comparisons.items() if not item["pass"])
    return {
        "left": str(left_path),
        "right": str(right_path),
        "compared_arrays": len(comparisons),
        "missing_union": missing,
        "failed_arrays": failed,
        "pass": not missing and not failed,
        "primary": {
            key: comparisons.get(key, {"status": "MISSING", "pass": False})
            for key in PRIMARY_KEYS
        },
    }


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomically write one JSON report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--historical-full", type=Path)
    parser.add_argument("--historical-source-short", type=Path)
    parser.add_argument("--current-source-short", type=Path)
    parser.add_argument("--left", type=Path)
    parser.add_argument("--right", type=Path)
    parser.add_argument("--label", default="generic_short_replay")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.left is not None or args.right is not None:
        if args.left is None or args.right is None:
            parser.error("--left and --right must be provided together")
        payload = {
            "experiment_id": "E187",
            "stage": "S0_E178_same_source_repeat",
            "label": args.label,
            "exact_atol": EXACT_ATOL,
            "comparison": compare(args.left, args.right),
        }
    else:
        required = (
            args.historical_full,
            args.historical_source_short,
            args.current_source_short,
        )
        if any(path is None for path in required):
            parser.error(
                "bisect mode requires --historical-full, "
                "--historical-source-short, and --current-source-short"
            )
        payload = {
            "experiment_id": "E187",
            "stage": "S0_E178_same_device_short_bisect",
            "exact_atol": EXACT_ATOL,
            "historical_source_vs_current_source": compare(
                args.historical_source_short, args.current_source_short
            ),
            "historical_full_prefix_vs_historical_source": compare(
                args.historical_full, args.historical_source_short
            ),
            "historical_full_prefix_vs_current_source": compare(
                args.historical_full, args.current_source_short
            ),
        }
    atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
