#!/usr/bin/env python3
"""Direct-main contracts for v8 task-aware local pre-segmentation."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import build_task_aware_preseg_v8 as v8
import numpy as np
from build_task_aware_preseg_v8 import (
    P_FLOOR,
    TARGET_HULLS,
    THRESHOLDS_M,
    candidate_family,
    derive_transition_atlas,
    protocol_payload,
    remap_original_segment_index,
    select_static_p_candidates,
    split_segment3,
    static_p_gate,
    unaffected_base_evidence,
)


def test_transition_atlas_determines_complete_plane_family() -> None:
    """All adjacent midpoints on the unique dominant axis must be retained."""
    atlas = derive_transition_atlas()
    assert atlas["pose_count"] == 3
    assert atlas["labels"] == ["TP", "PHANTOM", "PHANTOM"]
    assert atlas["dominant_axis"] == 0
    np.testing.assert_allclose(
        atlas["axis_ranges_m"],
        [0.04864747, 0.00491346, 0.00523118],
        atol=1e-8,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        atlas["candidate_planes_m"],
        [-0.14311002844145554, -0.11878629238288715],
        atol=1e-12,
        rtol=0.0,
    )
    assert atlas["removed_contact_count"] == 0


def test_local_split_is_exact_and_adds_one_segment() -> None:
    """Each frozen plane must replace segment3 with two closed positive solids."""
    atlas = derive_transition_atlas()
    for plane in atlas["candidate_planes_m"]:
        split = split_segment3(float(plane))
        assert split["source_segment_index"] == 3
        assert split["segment_count_after_split"] == 9
        assert len(split["children"]) == 2
        assert abs(split["volume_delta_m3"]) <= 1e-8
        assert all(child["watertight"] for child in split["children"])
        assert all(child["winding_consistent"] for child in split["children"])
        assert all(child["volume_m3"] > 0.0 for child in split["children"])


def test_v8_candidate_family_is_complete_without_k8_or_k4() -> None:
    """The family is the full 2-plane × 3-threshold × 2-K Cartesian product."""
    atlas = derive_transition_atlas()
    rows = candidate_family(atlas)
    assert len(rows) == 12
    assert len({row["candidate_id"] for row in rows}) == 12
    assert {row["max_hulls"] for row in rows} == set(TARGET_HULLS) == {16, 32}
    assert {row["threshold_m"] for row in rows} == set(THRESHOLDS_M)
    assert {row["plane_index"] for row in rows} == {0, 1}
    assert all(row["minimum_segment_hulls"] == 9 for row in rows)


def test_unaffected_segments_reuse_all_frozen_v4_bases() -> None:
    """Seven unchanged segments × three thresholds must be SHA-bound, not rebuilt."""
    evidence = unaffected_base_evidence()
    assert evidence["source_segment_indices"] == [0, 1, 2, 4, 5, 6, 7]
    assert evidence["thresholds_m"] == list(THRESHOLDS_M)
    assert evidence["manifest_count"] == 21
    assert len(evidence["manifests"]) == 21
    assert all(len(row["sha256"]) == 64 for row in evidence["manifests"])


def test_static_p_gate_matches_unchanged_precision_recall_floor() -> None:
    """The third-work-point shorthand must remain equivalent to both P floors."""
    assert P_FLOOR == 0.70
    passed = static_p_gate(tp=19, phantom=8, missed=8)
    assert passed["status"] == "PASS"
    assert passed["precision"] >= P_FLOOR and passed["recall"] >= P_FLOOR
    assert static_p_gate(tp=18, phantom=8, missed=9)["status"] == "FAIL"
    assert static_p_gate(tp=19, phantom=9, missed=8)["status"] == "FAIL"
    assert static_p_gate(tp=19, phantom=8, missed=9)["status"] == "FAIL"


def test_original_segments_have_one_deterministic_nine_segment_remap() -> None:
    """Only old segment3 is replaced; later original indices shift by one."""
    assert [remap_original_segment_index(index) for index in (0, 1, 2)] == [0, 1, 2]
    assert [remap_original_segment_index(index) for index in (4, 5, 6, 7)] == [
        5,
        6,
        7,
        8,
    ]
    for invalid in (3, -1, 8):
        try:
            remap_original_segment_index(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid original segment accepted: {invalid}")


def test_protocol_payload_binds_complete_prebuild_authority() -> None:
    """The frozen payload must cover family, topology, parents, and runtime math."""
    payload = protocol_payload()
    assert payload["status"] == "FROZEN_BEFORE_V8_SEGMENT_OR_COACD_OR_SCORE"
    assert payload["candidate_count"] == 12
    assert len(payload["topology"]["task_derived_splits"]) == 2
    assert payload["reused_v4_bases"]["manifest_count"] == 21
    assert payload["static_p_gate"]["third_work_point"] == "TP_GE_19_AND_PHANTOM_LE_8"
    assert payload["heldout_access"] == "NOT_ACCESSED_DEV3_ONLY"
    roles = [row["role"] for row in payload["source_dependencies"]]
    assert roles == [
        "UNCHANGED_P_CONTACT_MATH",
        "V6_STATIC_QUERY_AND_CONTACT_MATH",
        "V7_SIGNATURE_RECONSTRUCTION",
        "V1_EXACT_SEGMENT_EXPORT",
        "V2_ISOLATED_COACD_EXPORT",
        "V4_REDUCER_PART_CHECK_AND_EXPORT",
        "V5_DETERMINISTIC_REDUCER",
        "V8_BUILD_AND_STATIC_P",
    ]
    assert len({row["path"] for row in payload["source_dependencies"]}) == len(roles)


def test_static_p_selection_never_rescues_failed_rows() -> None:
    """Only true floor passes survive; ordering cannot override the hard gate."""
    rows = [
        {
            "candidate_id": "fail_better_score",
            "static_p_gate": {"status": "FAIL"},
            "p_score": 0.1,
            "actual_hulls": 9,
            "plane_index": 0,
            "threshold_m": 0.005,
            "max_hulls": 16,
        },
        {
            "candidate_id": "pass_more_hulls",
            "static_p_gate": {"status": "PASS"},
            "p_score": 0.4,
            "actual_hulls": 16,
            "plane_index": 1,
            "threshold_m": 0.010,
            "max_hulls": 16,
        },
        {
            "candidate_id": "pass_best",
            "static_p_gate": {"status": "PASS"},
            "p_score": 0.3,
            "actual_hulls": 15,
            "plane_index": 0,
            "threshold_m": 0.020,
            "max_hulls": 16,
        },
    ]
    selected = select_static_p_candidates(rows)
    assert [row["candidate_id"] for row in selected] == [
        "pass_best",
        "pass_more_hulls",
    ]


def test_complete_base_child_resume_has_no_subprocess_or_log_rewrite() -> None:
    """A complete child must be a true no-op resume, including its native log."""
    with tempfile.TemporaryDirectory(prefix="e182_v8_resume_") as directory:
        root = Path(directory)
        child_root = root / "segment_003"
        child_root.mkdir(parents=True)
        log_path = child_root / "native.log"
        log_path.write_text("frozen native output\n", encoding="utf-8")
        manifest_path = child_root / "manifest.json"
        payload = {
            "status": "BUILD_PASS",
            "native_log": {
                "path": str(log_path),
                "sha256": v8.sha256_file(log_path),
            },
        }
        manifest_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        manifest_before = manifest_path.read_bytes()
        log_before = log_path.read_bytes()

        original_root = v8._new_base_root
        original_build = v8.build_base_child
        original_run = v8.subprocess.run
        original_load_segments = v8._load_plane_segments
        original_validate_child = v8._validate_base_child_manifest
        try:
            v8._new_base_root = lambda _plane, _threshold: root
            v8.build_base_child = lambda **_kwargs: payload
            v8._load_plane_segments = lambda _plane: {}
            v8._validate_base_child_manifest = lambda value, **_kwargs: value

            def forbidden_subprocess(*_args: object, **_kwargs: object) -> object:
                raise AssertionError("complete v8 child must not relaunch")

            v8.subprocess.run = forbidden_subprocess
            resumed = v8._run_base_subprocess(0, 0.005, 3)
        finally:
            v8._new_base_root = original_root
            v8.build_base_child = original_build
            v8.subprocess.run = original_run
            v8._load_plane_segments = original_load_segments
            v8._validate_base_child_manifest = original_validate_child

        assert resumed == payload
        assert manifest_path.read_bytes() == manifest_before
        assert log_path.read_bytes() == log_before


def test_candidate_resume_rejects_empty_vacuous_part_inventory() -> None:
    """An empty/stale candidate manifest must not satisfy resume by vacuous truth."""
    row = {
        "candidate_id": "synthetic_v8_candidate",
        "method": "TASK_AWARE_LOCAL_EXACT_PRESEG_THEN_ISOLATED_COACD",
        "plane_index": 0,
        "plane_m": -0.1,
        "threshold_m": 0.005,
        "max_hulls": 16,
        "max_vertices": 256,
    }
    protocol = {
        "candidate_family": [row],
        "base_decomposition": {},
        "reducer": {},
        "topology": {},
        "physics_contract": {},
    }
    parameters = {
        "method": row["method"],
        "plane_index": row["plane_index"],
        "plane_m": row["plane_m"],
        "threshold_m": row["threshold_m"],
        "max_convex_hull": row["max_hulls"],
        "max_ch_vertex": row["max_vertices"],
        "base_decomposition": {},
        "reducer": {},
        "topology": {},
        "physics_contract": {},
    }
    with tempfile.TemporaryDirectory(prefix="e182_v8_candidate_resume_") as directory:
        candidate_root = Path(directory)
        manifest_path = candidate_root / row["candidate_id"] / "manifest.json"
        manifest_path.parent.mkdir(parents=True)
        manifest_path.write_text(
            json.dumps(
                {
                    "status": "BUILD_PASS",
                    "parameters": parameters,
                    "parts": [],
                }
            ),
            encoding="utf-8",
        )
        original_candidate_root = v8.CANDIDATE_ROOT
        original_load_protocol = v8._load_protocol
        try:
            v8.CANDIDATE_ROOT = candidate_root
            v8._load_protocol = lambda: protocol
            try:
                v8.build_candidate(row)
            except RuntimeError:
                pass
            else:
                raise AssertionError("empty v8 candidate manifest was resumed")
        finally:
            v8.CANDIDATE_ROOT = original_candidate_root
            v8._load_protocol = original_load_protocol


def test_static_p_resume_requires_current_protocol_and_frozen_row() -> None:
    """A static-P result must bind protocol and row, not only candidate SHA."""
    row = {
        "candidate_id": "synthetic_v8_candidate",
        "plane_index": 0,
        "plane_m": -0.1,
        "threshold_m": 0.005,
        "max_hulls": 16,
    }
    with tempfile.TemporaryDirectory(prefix="e182_v8_static_resume_") as directory:
        root = Path(directory)
        candidate_root = root / "candidates"
        static_root = root / "static_p"
        protocol_path = root / "protocol.json"
        protocol_path.write_text("{}\n", encoding="utf-8")
        manifest_path = candidate_root / row["candidate_id"] / "manifest.json"
        manifest_path.parent.mkdir(parents=True)
        manifest_path.write_text(
            json.dumps({"status": "BUILD_PASS"}) + "\n", encoding="utf-8"
        )
        result_path = static_root / "results" / f"{row['candidate_id']}.json"
        result_path.parent.mkdir(parents=True)
        result_path.write_text(
            json.dumps(
                {
                    "status": "COMPLETE",
                    "candidate_manifest": {
                        "sha256": v8.sha256_file(manifest_path),
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )
        originals = (
            v8.CANDIDATE_ROOT,
            v8.STATIC_P_ROOT,
            v8.PROTOCOL_PATH,
            v8._load_protocol,
        )
        try:
            v8.CANDIDATE_ROOT = candidate_root
            v8.STATIC_P_ROOT = static_root
            v8.PROTOCOL_PATH = protocol_path
            v8._load_protocol = lambda: {"candidate_family": [row]}
            try:
                v8.evaluate_static_p(row)
            except RuntimeError:
                pass
            else:
                raise AssertionError("stale static-P result was resumed")
        finally:
            (
                v8.CANDIDATE_ROOT,
                v8.STATIC_P_ROOT,
                v8.PROTOCOL_PATH,
                v8._load_protocol,
            ) = originals


def test_base_child_resume_rejects_threshold_and_protocol_drift() -> None:
    """Segment SHA alone cannot authorize a stale base-child manifest."""
    with tempfile.TemporaryDirectory(prefix="e182_v8_child_tamper_") as directory:
        root = Path(directory)
        child_root = root / "plane_00" / "t005" / "segment_003"
        part_path = child_root / "parts" / "part_000.obj"
        part_path.parent.mkdir(parents=True)
        part_path.write_text("synthetic part\n", encoding="utf-8")
        manifest_path = child_root / "manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "status": "BUILD_PASS",
                    "threshold_m": 0.010,
                    "segment_sha256": "segment-sha",
                    "parts": [
                        {
                            "path": str(part_path),
                            "sha256": v8.sha256_file(part_path),
                        }
                    ],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        originals = (
            v8.BASE_ROOT,
            v8._frozen_family_values,
            v8._load_plane_segments,
        )
        try:
            v8.BASE_ROOT = root
            v8._frozen_family_values = lambda **_kwargs: (0, 0.005)
            v8._load_plane_segments = lambda _plane: {
                "plane_m": -0.1,
                "children": [
                    {
                        "segment_index": 3,
                        "path": "synthetic-segment.obj",
                        "sha256": "segment-sha",
                    }
                ],
            }
            try:
                v8.build_base_child(
                    plane_index=0,
                    threshold_m=0.005,
                    segment_index=3,
                )
            except RuntimeError:
                pass
            else:
                raise AssertionError("stale v8 base child was resumed")
        finally:
            (
                v8.BASE_ROOT,
                v8._frozen_family_values,
                v8._load_plane_segments,
            ) = originals


def test_artifact_reference_rejects_sha_tamper_with_order_metadata() -> None:
    """Reference validation permits ordering metadata but never a SHA mismatch."""
    with tempfile.TemporaryDirectory(prefix="e182_v8_reference_") as directory:
        path = Path(directory) / "artifact.json"
        path.write_text("{}\n", encoding="utf-8")
        reference = {
            "plane_index": 0,
            "path": v8.relative_to_repo(path),
            "sha256": v8.sha256_file(path),
        }
        v8._require_artifact_reference(reference, path, label="synthetic")
        reference["sha256"] = "0" * 64
        try:
            v8._require_artifact_reference(reference, path, label="synthetic")
        except RuntimeError:
            pass
        else:
            raise AssertionError("tampered v8 artifact reference was accepted")


def test_complete_stage_summaries_are_validation_only_noops() -> None:
    """Completed build/static stages must not re-enter reducer or evaluator work."""
    with tempfile.TemporaryDirectory(prefix="e182_v8_stage_resume_") as directory:
        root = Path(directory)
        build_summary_path = root / "build_summary.json"
        build_summary_path.write_text('{"kind": "build"}\n', encoding="utf-8")
        static_root = root / "static_p"
        aggregate_path = static_root / "static_p_aggregate.json"
        aggregate_path.parent.mkdir(parents=True)
        aggregate_path.write_text('{"kind": "static"}\n', encoding="utf-8")
        protocol = {"candidate_family": []}
        originals = (
            v8.ATTEMPT_ROOT,
            v8.STATIC_P_ROOT,
            v8._load_protocol,
            v8._validate_build_summary,
            v8._validate_static_p_aggregate,
            v8.build_candidate,
            v8.evaluate_static_p,
        )
        try:
            v8.ATTEMPT_ROOT = root
            v8.STATIC_P_ROOT = static_root
            v8._load_protocol = lambda: protocol
            v8._validate_build_summary = lambda payload, **_kwargs: {
                **payload,
                "validated": True,
            }
            v8._validate_static_p_aggregate = lambda payload, **_kwargs: {
                **payload,
                "validated": True,
            }

            def forbidden_work(*_args: object, **_kwargs: object) -> object:
                raise AssertionError("completed v8 stage re-entered expensive work")

            v8.build_candidate = forbidden_work
            v8.evaluate_static_p = forbidden_work
            assert v8.build_candidates()["validated"] is True
            assert v8.run_static_p()["validated"] is True
        finally:
            (
                v8.ATTEMPT_ROOT,
                v8.STATIC_P_ROOT,
                v8._load_protocol,
                v8._validate_build_summary,
                v8._validate_static_p_aggregate,
                v8.build_candidate,
                v8.evaluate_static_p,
            ) = originals


def main() -> int:
    """Run v8 contracts without pytest discovery."""
    tests = (
        test_transition_atlas_determines_complete_plane_family,
        test_local_split_is_exact_and_adds_one_segment,
        test_v8_candidate_family_is_complete_without_k8_or_k4,
        test_unaffected_segments_reuse_all_frozen_v4_bases,
        test_static_p_gate_matches_unchanged_precision_recall_floor,
        test_original_segments_have_one_deterministic_nine_segment_remap,
        test_protocol_payload_binds_complete_prebuild_authority,
        test_static_p_selection_never_rescues_failed_rows,
        test_complete_base_child_resume_has_no_subprocess_or_log_rewrite,
        test_candidate_resume_rejects_empty_vacuous_part_inventory,
        test_static_p_resume_requires_current_protocol_and_frozen_row,
        test_base_child_resume_rejects_threshold_and_protocol_drift,
        test_artifact_reference_rejects_sha_tamper_with_order_metadata,
        test_complete_stage_summaries_are_validation_only_noops,
    )
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"E182_TASK_AWARE_PRESEG_V8_TESTS=PASS count={len(tests)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
