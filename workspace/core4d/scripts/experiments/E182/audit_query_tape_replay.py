#!/usr/bin/env python3
"""Audit E182 query-tape on/off and repeated-run exactness."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from e182_common import atomic_json, relative_to_repo, sha256_file
from run_query_tape_replay import DEFAULT_RESULT_ROOT, load_dev_rows


def compare_npz_arrays(left: Path, right: Path) -> dict[str, Any]:
    """Compare NPZ key sets, dtypes, shapes, values, and NaN placement exactly."""
    left_npz = np.load(left, allow_pickle=True)
    right_npz = np.load(right, allow_pickle=True)
    left_keys = set(left_npz.files)
    right_keys = set(right_npz.files)
    missing_left = sorted(right_keys - left_keys)
    missing_right = sorted(left_keys - right_keys)
    mismatched: list[str] = []
    for key in sorted(left_keys & right_keys):
        left_value = left_npz[key]
        right_value = right_npz[key]
        if (
            left_value.shape != right_value.shape
            or left_value.dtype != right_value.dtype
        ):
            mismatched.append(key)
            continue
        try:
            equal = np.array_equal(left_value, right_value, equal_nan=True)
        except TypeError:
            equal = np.array_equal(left_value, right_value)
        if not equal:
            mismatched.append(key)
    status = "PASS" if not (missing_left or missing_right or mismatched) else "FAIL"
    return {
        "status": status,
        "left": relative_to_repo(left),
        "right": relative_to_repo(right),
        "key_count": len(left_keys & right_keys),
        "missing_left": missing_left,
        "missing_right": missing_right,
        "mismatched_keys": mismatched,
    }


def _exact(left: Any, right: Any) -> bool:
    """Return exact equality, including matching NaN placement."""
    try:
        return bool(np.array_equal(left, right, equal_nan=True))
    except TypeError:
        return bool(np.array_equal(left, right))


def audit_same_run_integrity(
    result_path: Path,
    chunk_manifest_path: Path,
) -> dict[str, Any]:
    """Match frozen final-iteration chunks to summaries from the same optimizer run."""
    manifest = json.loads(chunk_manifest_path.read_text(encoding="utf-8"))
    mismatches: list[dict[str, Any]] = []
    nonfinite: list[dict[str, Any]] = []
    if manifest.get("status") != "COMPLETE":
        mismatches.append(
            {"field": "manifest.status", "actual": manifest.get("status")}
        )
    if len(str(manifest.get("content_sha256", ""))) != 64:
        mismatches.append({"field": "manifest.content_sha256", "actual": None})
    chunks = manifest.get("chunks", [])
    if manifest.get("chunk_count") != len(chunks):
        mismatches.append(
            {
                "field": "manifest.chunk_count",
                "actual": manifest.get("chunk_count"),
                "expected": len(chunks),
            }
        )

    with np.load(result_path, allow_pickle=True) as output:
        active_ticks = np.flatnonzero(np.asarray(output["opt_steps"]).reshape(-1) > 0)
        if len(active_ticks) != len(chunks):
            mismatches.append(
                {
                    "field": "active_tick_count",
                    "actual": len(active_ticks),
                    "expected": len(chunks),
                }
            )
        for expected_index, (entry, tick) in enumerate(
            zip(chunks, active_ticks, strict=False)
        ):
            if int(entry["chunk_index"]) != expected_index:
                mismatches.append(
                    {
                        "chunk": expected_index,
                        "field": "chunk_index",
                        "actual": entry["chunk_index"],
                    }
                )
            chunk_path = Path(entry["path"])
            if not chunk_path.is_file() or sha256_file(chunk_path) != entry["sha256"]:
                mismatches.append(
                    {"chunk": expected_index, "field": "chunk_file_or_sha"}
                )
                continue
            iteration = int(np.asarray(output["opt_steps"])[tick].reshape(-1)[0]) - 1
            with np.load(chunk_path, allow_pickle=False) as chunk:
                for key in ("qpos", "rewards"):
                    values = np.asarray(chunk[key])
                    if not np.isfinite(values).all():
                        nonfinite.append({"chunk": expected_index, "field": key})
                summaries = {
                    "rew_max": np.max(chunk["rewards"]),
                    "rew_min": np.min(chunk["rewards"]),
                    "rew_median": np.median(chunk["rewards"]),
                    "rew_mean": np.mean(chunk["rewards"]),
                    "cem_selected_index0": chunk["selected_indices"][0],
                }
                for sample_key in (
                    key
                    for key in chunk.files
                    if key.startswith("sample_") and chunk[key].ndim == 1
                ):
                    values = chunk[sample_key]
                    if (
                        np.issubdtype(values.dtype, np.number)
                        and not np.isfinite(values).all()
                    ):
                        nonfinite.append({"chunk": expected_index, "field": sample_key})
                    summaries.update(
                        {
                            f"{sample_key}_max": np.max(values),
                            f"{sample_key}_min": np.min(values),
                            f"{sample_key}_median": np.median(values),
                            f"{sample_key}_mean": np.mean(values),
                        }
                    )
                for key, expected in summaries.items():
                    if key not in output.files:
                        mismatches.append(
                            {"chunk": expected_index, "field": key, "actual": "MISSING"}
                        )
                        continue
                    actual = output[key][tick, iteration]
                    if not _exact(actual, expected):
                        mismatches.append(
                            {
                                "chunk": expected_index,
                                "field": key,
                                "actual": np.asarray(actual).tolist(),
                                "expected": np.asarray(expected).tolist(),
                            }
                        )

    status = "PASS" if not mismatches and not nonfinite else "FAIL"
    return {
        "status": status,
        "result": relative_to_repo(result_path),
        "chunk_manifest": relative_to_repo(chunk_manifest_path),
        "chunk_count": len(chunks),
        "active_tick_count": len(active_ticks),
        "mismatch_count": len(mismatches),
        "nonfinite_count": len(nonfinite),
        "mismatches": mismatches,
        "nonfinite": nonfinite,
        "content_sha256": manifest.get("content_sha256"),
        "provenance": manifest.get("provenance", {}),
    }


def result_path(root: Path, mode: str, case_id: str) -> Path:
    """Return one replay NPZ path."""
    return root / "replays" / mode / f"{case_id}_outdir/trajectory_mjwp_act.npz"


def chunk_manifest_path(root: Path, mode: str, case_id: str) -> Path:
    """Return one raw-chunk manifest path."""
    return root / "raw_chunks" / mode / case_id / "chunk_manifest.json"


def compare_chunk_runs(root: Path, case_id: str) -> dict[str, Any]:
    """Compare on_a/on_b query chunks array-by-array."""
    manifests = [
        json.loads(chunk_manifest_path(root, mode, case_id).read_text(encoding="utf-8"))
        for mode in ("on_a", "on_b")
    ]
    count_equal = manifests[0]["chunk_count"] == manifests[1]["chunk_count"]
    comparisons = []
    if count_equal:
        for left_entry, right_entry in zip(
            manifests[0]["chunks"], manifests[1]["chunks"], strict=True
        ):
            comparisons.append(
                compare_npz_arrays(Path(left_entry["path"]), Path(right_entry["path"]))
            )
    status = (
        "PASS"
        if count_equal and all(item["status"] == "PASS" for item in comparisons)
        else "FAIL"
    )
    return {
        "status": status,
        "chunk_counts": [manifest["chunk_count"] for manifest in manifests],
        "comparisons": comparisons,
    }


def summarize_replay_divergence(left: Path, right: Path) -> dict[str, Any]:
    """Report CUDA replay divergence without treating bitwise drift as a gate."""
    exact = compare_npz_arrays(left, right)
    metrics = {}
    with (
        np.load(left, allow_pickle=True) as left_npz,
        np.load(right, allow_pickle=True) as right_npz,
    ):
        for key in ("qpos", "qvel", "ctrl", "rew_mean", "cem_selected_index0"):
            left_value = np.asarray(left_npz[key])
            right_value = np.asarray(right_npz[key])
            delta = np.abs(left_value - right_value)
            finite = np.isfinite(delta)
            metrics[key] = {
                "shape": list(left_value.shape),
                "exact_fraction": float(
                    np.mean(
                        (left_value == right_value)
                        | (np.isnan(left_value) & np.isnan(right_value))
                    )
                ),
                "mean_abs_delta": (
                    float(delta[finite].mean()) if finite.any() else None
                ),
                "max_abs_delta": (float(delta[finite].max()) if finite.any() else None),
            }
    return {
        "role": "REPORT_ONLY_CUDA_REPLAY_DIVERGENCE",
        "bitwise_status": exact["status"],
        "mismatched_key_count": len(exact["mismatched_keys"]),
        "mismatched_keys": exact["mismatched_keys"],
        "metrics": metrics,
    }


def evaluate_case(root: Path, case_id: str) -> dict[str, Any]:
    """Evaluate one case using same-run integrity as the only numerical hard gate."""
    same_run = {
        mode: audit_same_run_integrity(
            result_path(root, mode, case_id),
            chunk_manifest_path(root, mode, case_id),
        )
        for mode in ("on_a", "on_b")
    }
    off_on = summarize_replay_divergence(
        result_path(root, "off", case_id),
        result_path(root, "on_a", case_id),
    )
    on_on = summarize_replay_divergence(
        result_path(root, "on_a", case_id),
        result_path(root, "on_b", case_id),
    )
    chunk_repeat = compare_chunk_runs(root, case_id)
    status = (
        "PASS"
        if all(result["status"] == "PASS" for result in same_run.values())
        else "FAIL"
    )
    return {
        "case_id": case_id,
        "status": status,
        "hard_gate": {"same_run_integrity": same_run},
        "report_only": {
            "off_vs_on_a": off_on,
            "on_a_vs_on_b": on_on,
            "on_a_vs_on_b_chunks": {
                "bitwise_status": chunk_repeat["status"],
                "chunk_counts": chunk_repeat["chunk_counts"],
                "mismatched_chunk_count": sum(
                    item["status"] != "PASS" for item in chunk_repeat["comparisons"]
                ),
            },
        },
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--case-id", action="append", default=[])
    return parser.parse_args()


def main() -> int:
    """Audit same-run integrity and report cross-run CUDA divergence."""
    args = parse_args()
    dev_rows = load_dev_rows()
    if args.case_id:
        requested = set(args.case_id)
        dev_rows = [row for row in dev_rows if row["case_id"] in requested]
        if {row["case_id"] for row in dev_rows} != requested:
            raise RuntimeError("one or more requested cases are outside dev3")
    if args.output:
        output = args.output
    elif args.case_id:
        suffix = "_".join(row["case_id"] for row in dev_rows)
        output = args.result_root / f"same_run_integrity_audit_{suffix}.json"
    else:
        output = args.result_root / "same_run_integrity_audit.json"
    rows = [evaluate_case(args.result_root, row["case_id"]) for row in dev_rows]
    payload = {
        "experiment_id": "E182",
        "gate": "S1_same_run_integrity",
        "cross_run_policy": "REPORT_ONLY_CUDA_NON_BITWISE",
        "status": "PASS" if all(row["status"] == "PASS" for row in rows) else "FAIL",
        "rows": rows,
    }
    atomic_json(output, payload)
    print(f"E182_QUERY_INTEGRITY_AUDIT={payload['status']} rows={len(rows)}")
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
