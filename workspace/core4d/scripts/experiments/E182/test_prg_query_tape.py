#!/usr/bin/env python3
"""Direct-main tests for E182 object-local P/R/G query-tape construction."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from build_prg_query_tape import (
    build_case_query_tape,
    extract_query_chunk,
    load_case_context,
    load_final_qpos,
    load_reference_qpos,
    materialize_query_points,
    verify_case_query_tape,
)
from run_query_tape_replay import load_dev_rows

CASE_ID = "bucket007_20231020_055_p1"


def bucket007_row() -> dict[str, str]:
    """Return the frozen representative row."""
    return next(row for row in load_dev_rows() if row["case_id"] == CASE_ID)


def test_reference_and_final_pose_sources() -> None:
    """Reference conversion and E178 final flattening must share model nq exactly."""
    row = bucket007_row()
    context = load_case_context(row)
    reference = load_reference_qpos(row, context)
    final = load_final_qpos(row, context)
    assert context.model.nq == 42
    assert reference.shape == (216, 42)
    assert final.shape == (166, 42)
    assert np.isfinite(reference).all()
    assert np.isfinite(final).all()
    assert np.allclose(reference[:, :36], context.raw_reference_qpos[:, :36])


def test_object_local_query_schema() -> None:
    """Real consumer groups must produce finite canonical points and provenance."""
    row = bucket007_row()
    context = load_case_context(row)
    reference = load_reference_qpos(row, context)
    chunk = extract_query_chunk(context, reference[:2])
    assert "points_object_local" not in chunk
    assert chunk["geom_pos_object_local"].shape[0] == 2
    assert (
        chunk["geom_mat_object_local"].shape[:2]
        == chunk["geom_pos_object_local"].shape[:2]
    )
    points = materialize_query_points(chunk)
    assert points.ndim == 3
    assert points.shape[0] == 2
    assert points.shape[2] == 3
    assert chunk["point_radius_m"].shape == chunk["point_geom_id"].shape
    assert chunk["point_consumer_mask"].shape[0] == chunk["point_geom_id"].shape[0]
    assert chunk["point_consumer_mask"].shape[1] == len(context.consumer_names)
    assert np.isfinite(points).all()
    assert np.isfinite(chunk["point_radius_m"]).all()
    assert "P_collision" in context.consumer_names
    assert len(context.consumer_geom_ids["P_collision"]) > 0
    assert "R_robot_penalty" in context.consumer_names
    assert "R_leg_penalty" in context.consumer_names
    assert "R_surface_band" in context.consumer_names
    assert "G_safety" in context.consumer_names
    assert "G_hand" in context.consumer_names
    assert "G_leg" in context.consumer_names


def test_case_tape_manifest() -> None:
    """A bounded real case build must freeze ref/final/CEM chunks with SHA."""
    row = bucket007_row()
    with tempfile.TemporaryDirectory(prefix="e182_prg_tape_") as directory:
        payload = build_case_query_tape(
            row,
            output_root=Path(directory),
            max_cem_chunks=1,
        )
        assert payload["status"] == "COMPLETE"
        assert payload["case_id"] == CASE_ID
        assert payload["selection_cem_mode"] == "on_a"
        assert payload["source_chunk_counts"] == {
            "reference": 1,
            "e178_final": 1,
            "cem_on_a": 1,
        }
        assert payload["chunk_count"] == 3
        assert all(len(entry["sha256"]) == 64 for entry in payload["chunks"])
        assert all(
            (Path(directory) / entry["relative_path"]).is_file()
            for entry in payload["chunks"]
        )
        assert payload["storage_contract"] == "FACTORED_GEOM_POSE_PLUS_FIXED_OFFSETS"
        assert payload["estimated_expanded_point_bytes"] > payload["stored_size_bytes"]
        verified = verify_case_query_tape(Path(directory), payload)
        assert verified["status"] == "PASS"
        first = Path(directory) / payload["chunks"][0]["relative_path"]
        with first.open("ab") as stream:
            stream.write(b"tamper")
        failed = verify_case_query_tape(Path(directory), payload)
        assert failed["status"] == "FAIL"
        assert failed["mismatch_count"] >= 1
        assert any(item["field"] == "sha256" for item in failed["mismatches"])


def main() -> int:
    """Run tests without pytest discovery."""
    tests = (
        test_reference_and_final_pose_sources,
        test_object_local_query_schema,
        test_case_tape_manifest,
    )
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"E182_PRG_QUERY_TAPE_TESTS=PASS count={len(tests)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
