#!/usr/bin/env python3
"""Direct-main contracts for approved v9 simultaneous double-plane topology."""

from __future__ import annotations

import json
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import build_task_aware_preseg_v9 as v9
import numpy as np


@contextmanager
def temporary_artifact_roots() -> Iterator[Path]:
    """Redirect every writable v9 artifact into one repo-local temporary root."""
    original = {
        "ATTEMPT_ROOT": v9.ATTEMPT_ROOT,
        "PROTOCOL_PATH": v9.PROTOCOL_PATH,
        "SEGMENT_ROOT": v9.SEGMENT_ROOT,
        "BASE_ROOT": v9.BASE_ROOT,
        "CANDIDATE_ROOT": v9.CANDIDATE_ROOT,
        "STATIC_P_ROOT": v9.STATIC_P_ROOT,
    }
    with tempfile.TemporaryDirectory(
        prefix="e182_v9_pipeline_", dir=v9.v8.REPO_ROOT
    ) as directory:
        root = Path(directory)
        v9.ATTEMPT_ROOT = root
        v9.PROTOCOL_PATH = root / "protocol.json"
        v9.SEGMENT_ROOT = root / "segments"
        v9.BASE_ROOT = root / "coacd_bases"
        v9.CANDIDATE_ROOT = root / "candidates"
        v9.STATIC_P_ROOT = root / "static_p_screen"
        try:
            yield root
        finally:
            for name, value in original.items():
                setattr(v9, name, value)


def install_fake_coacd() -> tuple[Any, list[tuple[float, int]]]:
    """Replace native CoACD with one deterministic convex hull per invocation."""
    original = v9._coacd_parts
    calls: list[tuple[float, int]] = []

    def fake_coacd(
        mesh: Any, *, threshold_m: float, max_hulls: int
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        calls.append((threshold_m, max_hulls))
        hull = mesh.bounding_box
        return [
            (
                np.asarray(hull.vertices, dtype=np.float64),
                np.asarray(hull.faces, dtype=np.int64),
            )
        ]

    v9._coacd_parts = fake_coacd
    return original, calls


def run_fake_middle_isolated(threshold_m: float) -> dict[str, Any]:
    """Emulate the subprocess boundary while retaining the in-process fake CoACD."""
    payload = v9.build_middle_base(threshold_m)
    root = v9._base_root(threshold_m) / "segment_004"
    log_path = root / "native.log"
    log_path.write_text("FAKE_ISOLATED_COACD=PASS\n", encoding="utf-8")
    isolated = {**payload, "native_log": v9._artifact_reference(log_path)}
    v9.atomic_json(root / "manifest.json", isolated)
    return isolated


def build_fake_pipeline_through_candidates() -> dict[str, Any]:
    """Build the writable v9 stages using deterministic fake middle CoACD."""
    v9.freeze_protocol()
    v9.build_segments()
    original_coacd, calls = install_fake_coacd()
    original_runner = v9._run_middle_base_subprocess
    v9._run_middle_base_subprocess = run_fake_middle_isolated
    try:
        bases = v9.build_bases()
    finally:
        v9._coacd_parts = original_coacd
        v9._run_middle_base_subprocess = original_runner
    assert calls == [(0.005, 4), (0.01, 4), (0.02, 4)]
    build = v9.build_candidates()
    return {"bases": bases, "build": build}


def test_double_plane_topology_is_exact_and_isolates_transitions() -> None:
    """Both frozen planes must create three closed solids and three regions."""
    topology = v9.double_plane_topology()
    assert topology["source_segment_index"] == 3
    assert topology["source_segment_count"] == 8
    assert topology["final_segment_count"] == 10
    assert topology["minimum_hulls"] == 10
    assert topology["planes_m"] == [
        -0.14311002844145554,
        -0.11878629238288715,
    ]
    assert len(topology["children"]) == 3
    assert abs(topology["volume_delta_m3"]) <= 1e-8
    assert all(row["watertight"] for row in topology["children"])
    assert all(row["winding_consistent"] for row in topology["children"])
    assert all(row["volume_m3"] > 0.0 for row in topology["children"])
    assert [row["region_index"] for row in topology["transition_assignments"]] == [
        0,
        2,
        1,
    ]
    assert [row["label"] for row in topology["transition_assignments"]] == [
        "TP",
        "PHANTOM",
        "PHANTOM",
    ]
    assert sorted(
        row["region_index"] for row in topology["transition_assignments"]
    ) == [
        0,
        1,
        2,
    ]


def test_v9_family_is_exactly_three_thresholds_by_two_k() -> None:
    """K8/K4 must not appear because ten segments each require one hull."""
    rows = v9.candidate_family()
    assert len(rows) == 6
    assert len({row["candidate_id"] for row in rows}) == 6
    assert {row["threshold_m"] for row in rows} == {0.005, 0.010, 0.020}
    assert {row["max_hulls"] for row in rows} == {16, 32}
    assert all(row["minimum_segment_hulls"] == 10 for row in rows)
    assert all(row["simultaneous_plane_count"] == 2 for row in rows)
    assert all("k08" not in row["candidate_id"] for row in rows)
    assert all("k04" not in row["candidate_id"] for row in rows)


def test_original_segments_have_one_ten_segment_remap() -> None:
    """Only old segment3 is replaced by three children."""
    assert [v9.remap_original_segment_index(index) for index in (0, 1, 2)] == [
        0,
        1,
        2,
    ]
    assert [v9.remap_original_segment_index(index) for index in (4, 5, 6, 7)] == [
        6,
        7,
        8,
        9,
    ]
    for invalid in (3, -1, 8):
        try:
            v9.remap_original_segment_index(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid original segment accepted: {invalid}")


def test_reuse_evidence_is_complete_and_middle_only_is_fresh() -> None:
    """Seven v4 segments plus two v8 outers are reused for each threshold."""
    evidence = v9.reuse_evidence()
    assert evidence["unchanged_v4_manifest_count"] == 21
    assert evidence["outer_v8_manifest_count"] == 6
    assert evidence["fresh_middle_coacd_count"] == 3
    assert evidence["source_segment_indices"] == [0, 1, 2, 4, 5, 6, 7]
    assert len(evidence["unchanged_v4_manifests"]) == 21
    assert len(evidence["outer_v8_manifests"]) == 6
    assert all(len(row["sha256"]) == 64 for row in evidence["unchanged_v4_manifests"])
    assert all(len(row["sha256"]) == 64 for row in evidence["outer_v8_manifests"])
    assert all(
        row["geometry_fingerprint_exact"] for row in evidence["outer_v8_manifests"]
    )
    assert all(
        row["geometry_fingerprint"]["sorted_face_vertex_max_abs_m"]
        <= v9.GEOMETRY_ROUNDTRIP_TOLERANCE_M
        for row in evidence["outer_v8_manifests"]
    )


def test_static_p_gate_is_unchanged_and_selection_cannot_rescue_failure() -> None:
    """The approved topology does not alter the P launch floor."""
    assert v9.P_FLOOR == 0.70
    assert v9.static_p_gate(tp=19, phantom=8, missed=8)["status"] == "PASS"
    assert v9.static_p_gate(tp=18, phantom=7, missed=9)["status"] == "FAIL"
    assert v9.static_p_gate(tp=19, phantom=9, missed=8)["status"] == "FAIL"
    rows = [
        {
            "candidate_id": "failed",
            "static_p_gate": {"status": "FAIL"},
            "p_score": 0.0,
            "actual_hulls": 10,
            "threshold_m": 0.005,
            "max_hulls": 16,
        },
        {
            "candidate_id": "passed",
            "static_p_gate": {"status": "PASS"},
            "p_score": 0.5,
            "actual_hulls": 16,
            "threshold_m": 0.010,
            "max_hulls": 16,
        },
    ]
    assert [row["candidate_id"] for row in v9.select_static_p_candidates(rows)] == [
        "passed"
    ]


def test_protocol_payload_binds_parent_topology_reuse_and_code() -> None:
    """The pre-score protocol must bind every authority and six-row family."""
    payload = v9.protocol_payload()
    assert payload["status"] == "FROZEN_BEFORE_V9_SEGMENT_OR_COACD_OR_SCORE"
    assert payload["candidate_count"] == 6
    assert payload["heldout_access"] == "NOT_ACCESSED_DEV3_ONLY"
    assert payload["topology"]["final_segment_count"] == 10
    assert payload["topology"]["minimum_hulls"] == 10
    assert payload["topology"]["source_segment_manifest"] == v9._artifact_reference(
        v9.v8.V1_SEGMENT_MANIFEST_PATH
    )
    split_manifests = payload["topology"]["v8_split_manifests"]
    assert [row["plane_index"] for row in split_manifests] == [0, 1]
    assert [
        {"path": row["path"], "sha256": row["sha256"]} for row in split_manifests
    ] == [
        v9._artifact_reference(v9.v8._plane_segment_root(plane_index) / "manifest.json")
        for plane_index in (0, 1)
    ]
    assert payload["reuse"]["fresh_middle_coacd_count"] == 3
    assert payload["static_p_gate"]["third_work_point"] == "TP_GE_19_AND_PHANTOM_LE_8"
    assert payload["stop_action"] == "STOP_V9_NO_MORE_PLANES_OR_FLOOR_CHANGES"
    roles = [row["role"] for row in payload["source_dependencies"]]
    assert roles == [
        "UNCHANGED_EXACT_C_AND_P_MATH",
        "V5_DETERMINISTIC_REDUCER",
        "V5_QUERY_EVALUATOR_AUTHORITY",
        "V8_PARENT_BUILD_AND_STATIC_P",
        "V9_BUILD_AND_STATIC_P",
    ]
    evaluator = next(
        row
        for row in payload["source_dependencies"]
        if row["role"] == "V5_QUERY_EVALUATOR_AUTHORITY"
    )
    assert evaluator["sha256"] == (
        "677839039d8813987447d2beb4c32c07981d2ebdbc5a2f88bd3936dd11304560"
    )
    for key in ("v8_protocol", "v8_build", "v8_static", "v8_visual"):
        assert len(payload["parent_bindings"][key]["sha256"]) == 64


def test_protocol_freeze_rejects_nonempty_v9_root() -> None:
    """Pre-score freeze must fail closed if any v9 artifact already exists."""
    with tempfile.TemporaryDirectory(prefix="e182_v9_nonempty_") as directory:
        root = Path(directory)
        (root / "unexpected.txt").write_text("contamination\n", encoding="utf-8")
        original_root = v9.ATTEMPT_ROOT
        original_protocol = v9.PROTOCOL_PATH
        try:
            v9.ATTEMPT_ROOT = root
            v9.PROTOCOL_PATH = root / "protocol.json"
            try:
                v9.freeze_protocol()
            except RuntimeError:
                pass
            else:
                raise AssertionError("nonempty v9 root was frozen")
        finally:
            v9.ATTEMPT_ROOT = original_root
            v9.PROTOCOL_PATH = original_protocol


def test_protocol_resume_is_validation_only_and_rejects_sha_tamper() -> None:
    """A frozen protocol may resume only when its exact payload is unchanged."""
    with tempfile.TemporaryDirectory(prefix="e182_v9_protocol_") as directory:
        root = Path(directory)
        protocol_path = root / "protocol.json"
        original_root = v9.ATTEMPT_ROOT
        original_protocol = v9.PROTOCOL_PATH
        try:
            v9.ATTEMPT_ROOT = root
            v9.PROTOCOL_PATH = protocol_path
            payload = v9.freeze_protocol()
            before = protocol_path.read_bytes()
            resumed = v9.freeze_protocol()
            assert resumed == payload
            assert protocol_path.read_bytes() == before
            tampered = json.loads(protocol_path.read_text(encoding="utf-8"))
            tampered["candidate_count"] = 7
            protocol_path.write_text(
                json.dumps(tampered, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            try:
                v9.freeze_protocol()
            except RuntimeError:
                pass
            else:
                raise AssertionError("tampered v9 protocol resumed")
        finally:
            v9.ATTEMPT_ROOT = original_root
            v9.PROTOCOL_PATH = original_protocol


def test_segment_export_is_exact_resumable_and_tamper_evident() -> None:
    """The single v9 split manifest must bind exactly three exported children."""
    with temporary_artifact_roots():
        v9.freeze_protocol()
        summary = v9.build_segments()
        assert summary["status"] == "PASS"
        assert summary["child_count"] == 3
        assert [row["segment_index"] for row in summary["children"]] == [3, 4, 5]
        manifest_before = (v9.SEGMENT_ROOT / "manifest.json").read_bytes()
        resumed = v9.build_segments()
        assert resumed == summary
        assert (v9.SEGMENT_ROOT / "manifest.json").read_bytes() == manifest_before
        tampered = json.loads(manifest_before)
        tampered["child_count"] = 4
        (v9.SEGMENT_ROOT / "manifest.json").write_text(
            json.dumps(tampered, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        try:
            v9.build_segments()
        except RuntimeError:
            pass
        else:
            raise AssertionError("tampered v9 segment manifest resumed")


def test_only_three_middle_bases_are_fresh_and_composites_cover_ten_segments() -> None:
    """Outer and unchanged segments must remain frozen references, never rebuilt."""
    with temporary_artifact_roots():
        v9.freeze_protocol()
        v9.build_segments()
        original_coacd, calls = install_fake_coacd()
        original_runner = v9._run_middle_base_subprocess
        v9._run_middle_base_subprocess = run_fake_middle_isolated
        try:
            summary = v9.build_bases()
            assert summary["fresh_middle_count"] == 3
            assert summary["composite_base_count"] == 3
            assert calls == [(0.005, 4), (0.01, 4), (0.02, 4)]
            for threshold_m in v9.THRESHOLDS_M:
                base_path = v9._base_root(threshold_m) / "manifest.json"
                base = json.loads(base_path.read_text(encoding="utf-8"))
                assert base["segment_count"] == 10
                assert [
                    row["segment_index"] for row in base["child_manifests"]
                ] == list(range(10))
                sources = [row["source"] for row in base["child_manifests"]]
                assert sources.count("V9_FRESH_MIDDLE_SEGMENT") == 1
                assert sources.count("REUSED_V8_OUTER_SEGMENT") == 2
                assert sources.count("REUSED_V4_UNCHANGED_SEGMENT") == 7
            before = (v9.ATTEMPT_ROOT / "base_summary.json").read_bytes()
            assert v9.build_bases() == summary
            assert (v9.ATTEMPT_ROOT / "base_summary.json").read_bytes() == before
            assert calls == [(0.005, 4), (0.01, 4), (0.02, 4)]
        finally:
            v9._coacd_parts = original_coacd
            v9._run_middle_base_subprocess = original_runner


def test_base_summary_rejects_nonisolated_middle_builds() -> None:
    """A complete base stage must require native logs from all three subprocesses."""
    with temporary_artifact_roots():
        v9.freeze_protocol()
        v9.build_segments()
        original_coacd, _ = install_fake_coacd()
        original_runner = v9._run_middle_base_subprocess
        v9._run_middle_base_subprocess = v9.build_middle_base
        try:
            try:
                v9.build_bases()
            except RuntimeError:
                pass
            else:
                raise AssertionError("nonisolated v9 middle bases were accepted")
        finally:
            v9._coacd_parts = original_coacd
            v9._run_middle_base_subprocess = original_runner


def test_six_candidates_preserve_ten_segments_and_reject_manifest_tamper() -> None:
    """The reducer must emit six immutable K16/K32 ten-segment candidates."""
    with temporary_artifact_roots():
        pipeline = build_fake_pipeline_through_candidates()
        build = pipeline["build"]
        assert build["candidate_count"] == 6
        assert len(build["candidate_ids"]) == 6
        for candidate_id in build["candidate_ids"]:
            manifest_path = v9.CANDIDATE_ROOT / candidate_id / "manifest.json"
            candidate = json.loads(manifest_path.read_text(encoding="utf-8"))
            assert 10 <= candidate["hull_count"] <= candidate["max_hulls"]
            assert [row["segment_index"] for row in candidate["per_segment"]] == list(
                range(10)
            )
        summary_path = v9.ATTEMPT_ROOT / "build_summary.json"
        summary_before = summary_path.read_bytes()
        assert v9.build_candidates() == build
        assert summary_path.read_bytes() == summary_before
        first_path = v9.CANDIDATE_ROOT / build["candidate_ids"][0] / "manifest.json"
        tampered = json.loads(first_path.read_text(encoding="utf-8"))
        tampered["hull_count"] += 1
        first_path.write_text(
            json.dumps(tampered, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        try:
            v9.build_candidates()
        except RuntimeError:
            pass
        else:
            raise AssertionError("tampered v9 candidate resumed")


def test_static_p_zero_of_six_freezes_required_stop_and_resumes_noop() -> None:
    """A 0/6 screen must freeze the v9 stop action without querying heldout."""
    with temporary_artifact_roots():
        build_fake_pipeline_through_candidates()
        original_queries = v9._static_p_queries
        original_scenes = v9._candidate_segment_scenes
        original_distance = v9.core._scene_signed_distance
        points = np.zeros((882, 1, 3), dtype=np.float64)
        radii = np.zeros((1,), dtype=np.float64)
        oracle_contact = np.zeros((882,), dtype=bool)
        oracle_contact[:27] = True
        v9._static_p_queries = lambda: (
            [{"source_family": "DEV3_FAKE", "points": points, "radii": radii}],
            oracle_contact,
        )
        v9._candidate_segment_scenes = lambda candidate: [object()]
        v9.core._scene_signed_distance = lambda scene, query_points: np.ones(
            query_points.shape[:2], dtype=np.float64
        )
        try:
            aggregate = v9.run_static_p()
            assert aggregate["pass_count"] == 0
            assert aggregate["full_prg_eligible"] is False
            assert (
                aggregate["zero_pass_action"]
                == "STOP_V9_NO_MORE_PLANES_OR_FLOOR_CHANGES"
            )
            before = (v9.STATIC_P_ROOT / "static_p_aggregate.json").read_bytes()
            assert v9.run_static_p() == aggregate
            assert (v9.STATIC_P_ROOT / "static_p_aggregate.json").read_bytes() == before
            tampered = json.loads(before)
            tampered["pass_count"] = 1
            (v9.STATIC_P_ROOT / "static_p_aggregate.json").write_text(
                json.dumps(tampered, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            try:
                v9.run_static_p()
            except RuntimeError:
                pass
            else:
                raise AssertionError("tampered v9 static-P aggregate resumed")
        finally:
            v9._static_p_queries = original_queries
            v9._candidate_segment_scenes = original_scenes
            v9.core._scene_signed_distance = original_distance


def main() -> int:
    """Run v9 contracts without pytest discovery."""
    tests = (
        test_double_plane_topology_is_exact_and_isolates_transitions,
        test_v9_family_is_exactly_three_thresholds_by_two_k,
        test_original_segments_have_one_ten_segment_remap,
        test_reuse_evidence_is_complete_and_middle_only_is_fresh,
        test_static_p_gate_is_unchanged_and_selection_cannot_rescue_failure,
        test_protocol_payload_binds_parent_topology_reuse_and_code,
        test_protocol_freeze_rejects_nonempty_v9_root,
        test_protocol_resume_is_validation_only_and_rejects_sha_tamper,
        test_segment_export_is_exact_resumable_and_tamper_evident,
        test_only_three_middle_bases_are_fresh_and_composites_cover_ten_segments,
        test_base_summary_rejects_nonisolated_middle_builds,
        test_six_candidates_preserve_ten_segments_and_reject_manifest_tamper,
        test_static_p_zero_of_six_freezes_required_stop_and_resumes_noop,
    )
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"E182_TASK_AWARE_PRESEG_V9_TESTS=PASS count={len(tests)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
