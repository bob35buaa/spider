#!/usr/bin/env python3
"""Unit contracts for E187's direct-core E178 compatibility evaluator."""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO / "workspace/core4d/scripts/eval/runners"))

import eval_E187_e178_compat as MODULE  # noqa: E402


def test_gate_boundaries() -> None:
    """All twelve thresholds are inclusive and one epsilon above fails."""
    item = {
        "fall_flag": False,
        "body_z_err_p95_m": 0.20,
        "hand_object_physics_contact_in_mask_frac": 0.50,
        "release_gate_applicable": True,
        "hand_object_release_false_contact_3mm_frac": 0.30,
        "hand_object_physics_penetration_3mm_frame_frac": 0.30,
        "leg_penetration_frac": 0.10,
        "track_root_pos_err_cm_mean": 20.0,
        "track_root_ori_err_deg_mean": 20.0,
        "track_eef_pos_err_cm_mean": 20.0,
        "track_eef_ori_err_deg_mean": 20.0,
        "track_obj_pos_err_cm_mean": 20.0,
        "track_obj_ori_err_deg_mean": 10.0,
    }
    MODULE.apply_gates(item)
    assert item["numeric_release_pass"]
    item["track_obj_ori_err_deg_mean"] = 10.000001
    MODULE.apply_gates(item)
    assert not item["numeric_release_pass"]
    assert item["numeric_failure_modes"] == "object_ori"


def test_live_config_whitelist() -> None:
    """The in-progress bucket007 config has only frozen default additions."""
    manifest = MODULE.read_tsv(MODULE.DEFAULT_MANIFEST)
    row = next(row for row in manifest if row["case_id"] == "bucket007_20231020_055_p1")
    result = MODULE.config_whitelist_diff(row)
    assert result["pass"]
    assert result["added"] == MODULE.ALLOWED_ADDED_CONFIG
    assert set(result["changed"]) == MODULE.ALLOWED_CHANGED_CONFIG


def test_preregistered_device_routing() -> None:
    """Same-device exact and cross-GPU semantic gates must not be conflated."""
    common = {
        "manifest_contract_pass": True,
        "config_pass": True,
        "decision_match": True,
    }
    assert MODULE.compatibility_gate(
        mode="same_device_golden",
        same_device_golden_pass=True,
        semantic_pass=False,
        **common,
    )
    assert MODULE.compatibility_gate(
        mode="cross_gpu_semantic",
        same_device_golden_pass=False,
        semantic_pass=True,
        **common,
    )
    assert not MODULE.compatibility_gate(
        mode="same_device_golden",
        same_device_golden_pass=False,
        semantic_pass=True,
        **common,
    )
    assert not MODULE.compatibility_gate(
        mode="cross_gpu_semantic",
        same_device_golden_pass=True,
        semantic_pass=False,
        **common,
    )


def test_missing_historical_query_evidence_fails_closed() -> None:
    """The E178 trajectory NPZ cannot impersonate a missing query-tape golden."""
    manifest = MODULE.read_tsv(MODULE.DEFAULT_MANIFEST)
    row = next(row for row in manifest if row["case_id"] == "bucket007_20231020_055_p1")
    result = MODULE.same_device_query_tape_diff(row)
    assert not result["evidence_complete"]
    assert not result["pass"]
    assert result["missing_manifest_fields"] == list(MODULE.QUERY_TAPE_MANIFEST_FIELDS)


def test_live_manifest_device_mismatch_is_explicit() -> None:
    """The first manifest's all-local placeholders cannot authorize Ada replay."""
    rows = MODULE.read_tsv(MODULE.DEFAULT_MANIFEST)
    audits = {row["case_id"]: MODULE.manifest_contract_diff(row) for row in rows}
    assert audits["bucket007_20231020_055_p1"]["pass"]
    assert not audits["bucket003_20231018_003_p1"]["pass"]
    assert not audits["bucket004_20231002_021_p1"]["pass"]
    assert audits["bucket003_20231018_003_p1"]["mode"] == "cross_gpu_semantic"


def main() -> int:
    """Run direct-main contracts."""
    tests = (
        test_gate_boundaries,
        test_live_config_whitelist,
        test_preregistered_device_routing,
        test_missing_historical_query_evidence_fails_closed,
        test_live_manifest_device_mismatch_is_explicit,
    )
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"E187_COMPAT_EVAL_TESTS=PASS count={len(tests)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
