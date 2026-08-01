#!/usr/bin/env python3
"""Direct-main contract tests for E182 diagnostic MuJoCo P-contact replay."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import mujoco
import numpy as np
from e182_common import repo_path, sha256_file
from evaluate_task_queries import (
    _load_candidate_and_oracle,
    _resolve_case_and_candidate,
)
from replay_p_contact_mujoco import (
    DEFAULT_DIAGNOSTIC,
    _binary_confusion,
    build_compound_sidecar,
)

FIXTURE_PATH = repo_path(
    "workspace/core4d/results/E182/s2_task_query_eval/query_fixture_manifest.json"
)
REPLAY_MANIFEST_PATH = repo_path(
    "workspace/core4d/results/E182/s2_task_query_eval/"
    "p_contact_mujoco_replay/mujoco_replay_manifest.json"
)


def test_binary_confusion_contract() -> None:
    """Boolean contact comparison must preserve all four confusion cells."""
    result = _binary_confusion(
        np.asarray([True, True, False, False]),
        np.asarray([True, False, True, False]),
    )
    assert result == {
        "count": 4,
        "true_positive_count": 1,
        "false_positive_count": 1,
        "false_negative_count": 1,
        "true_negative_count": 1,
        "precision": 0.5,
        "recall": 0.5,
    }


def test_real_bucket003_sidecar_contract() -> None:
    """One real diagnostic candidate must compile with all compound pairs and inertia."""
    diagnostic = json.loads(DEFAULT_DIAGNOSTIC.read_text(encoding="utf-8"))
    assert diagnostic["selection_eligible"] is False
    assert diagnostic["heldout_access"] == "NOT_ACCESSED_DEV3_ONLY"
    row = diagnostic["candidates"][0]
    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    case, fixture_candidate = _resolve_case_and_candidate(
        fixture, row["case_id"], row["candidate_id"]
    )
    candidate, _, _ = _load_candidate_and_oracle(fixture_candidate)
    prg_path = repo_path(case["prg_manifest"]["path"])
    assert sha256_file(prg_path) == case["prg_manifest"]["sha256"]
    prg = json.loads(prg_path.read_text(encoding="utf-8"))
    base_scene = repo_path(prg["model"])
    with tempfile.TemporaryDirectory(prefix="e182_p_replay_") as temporary:
        sidecar_path = Path(temporary) / "sidecar.xml"
        sidecar = build_compound_sidecar(
            base_scene=base_scene,
            candidate_manifest=candidate,
            output_path=sidecar_path,
        )
        assert sidecar["base_scene_sha256"] == sha256_file(base_scene)
        assert sidecar["object_geom_count"] == row["actual_hulls"]
        assert sidecar["explicit_pair_count"] == row["actual_hulls"] * 19
        model = mujoco.MjModel.from_xml_path(str(sidecar_path))
        object_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
        assert object_body_id >= 0
        assert np.isclose(model.body_mass[object_body_id], 2.0)
        assert model.npair == sidecar["model_npair"]
        assert all(
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name) >= 0
            for name in sidecar["object_geom_names"]
        )


def test_real_replay_artifact_contract() -> None:
    """The completed replay must prove point phantoms are true compound contacts."""
    replay = json.loads(REPLAY_MANIFEST_PATH.read_text(encoding="utf-8"))
    assert replay["status"] == "COMPLETE"
    assert replay["selection_eligible"] is False
    assert replay["heldout_access"] == "NOT_ACCESSED_DEV3_ONLY"
    assert replay["candidate_count"] == 3
    diagnostic_path = repo_path(replay["diagnostic"]["path"])
    fixture_path = repo_path(replay["fixture"]["path"])
    assert sha256_file(diagnostic_path) == replay["diagnostic"]["sha256"]
    assert sha256_file(fixture_path) == replay["fixture"]["sha256"]
    for row in replay["candidates"]:
        assert row["selection_role"] == "DIAGNOSTIC_ONLY_NOT_FINALIST"
        assert row["pose_count"] == 882
        assert sum(source["pose_count"] for source in row["source_rows"]) == 882
        assert row["point_phantom_pose_count"] > 0
        assert (
            row["point_phantom_confirmed_by_mujoco_count"]
            == row["point_phantom_pose_count"]
        )
        assert row["point_phantom_confirmed_by_mujoco_fraction"] == 1.0
        assert row["point_C_vs_mujoco_C"]["precision"] == 1.0
        assert row["representative_short_rollout"]["contacts"][0]["contact"] is True
        sidecar = row["sidecar"]
        sidecar_path = repo_path(sidecar["path"])
        assert sha256_file(sidecar_path) == sidecar["sha256"]
        assert sidecar["object_geom_count"] == row["actual_hulls"]
        assert sidecar["explicit_pair_count"] == row["actual_hulls"] * 19


def main() -> int:
    """Run replay tests without pytest discovery."""
    tests = (
        test_binary_confusion_contract,
        test_real_bucket003_sidecar_contract,
        test_real_replay_artifact_contract,
    )
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"E182_P_MUJOCO_REPLAY_TESTS=PASS count={len(tests)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
