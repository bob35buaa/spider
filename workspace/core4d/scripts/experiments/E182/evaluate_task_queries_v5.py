#!/usr/bin/env python3
"""Authority adapter for evaluating the frozen one-case v5 query fixture."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import evaluate_task_queries as core
from build_segmented_coacd_attempt2_v5 import (
    ATTEMPT_ROOT,
    FIXTURE_PATH,
    PROTOCOL_PATH,
)
from e182_common import atomic_json, relative_to_repo, repo_path, sha256_file

_ORIGINAL_LOAD_FROZEN_FIXTURE = core._load_frozen_fixture
_ORIGINAL_LOAD_CASE_INDICES = core._load_case_indices
BOUNDED_ROOT = ATTEMPT_ROOT / "bounded_real_adapter"
BOUNDED_CASE_ID = "bucket003_20231018_001_p1"
BOUNDED_CANDIDATE_ID = "segx2y4_gmerge5_t010_k16_v256"


def load_v5_fixture(path: Path) -> tuple[dict[str, Any], str]:
    """Accept only the immutable bucket003 one-case/nine-candidate v5 fixture."""
    if path.resolve() != FIXTURE_PATH.resolve():
        raise RuntimeError("v5 evaluator only accepts the frozen v5 fixture path")
    fixture = json.loads(path.read_text(encoding="utf-8"))
    if fixture.get("status") != "FROZEN":
        raise RuntimeError("v5 query fixture is not FROZEN")
    if fixture.get("heldout_access") != "NOT_ACCESSED_DEV3_ONLY":
        raise RuntimeError("v5 heldout isolation changed")
    if fixture.get("candidate_count") != 9 or len(fixture.get("cases", [])) != 1:
        raise RuntimeError("v5 fixture authority count changed")
    case = fixture["cases"][0]
    if (
        case.get("case_id") != "bucket003_20231018_001_p1"
        or case.get("object_key") != "bucket003"
    ):
        raise RuntimeError("v5 fixture case authority changed")
    if any(
        candidate.get("object_key") != "bucket003"
        for candidate in fixture["candidates"]
    ):
        raise RuntimeError("v5 fixture includes a non-bucket003 candidate")
    protocol = fixture.get("attempt2_v5_protocol", {})
    if repo_path(
        protocol.get("path", "")
    ).resolve() != PROTOCOL_PATH.resolve() or protocol.get("sha256") != sha256_file(
        PROTOCOL_PATH
    ):
        raise RuntimeError("v5 fixture protocol authority changed")
    original = fixture.get("original_fixture", {})
    if repo_path(
        original.get("path", "")
    ).resolve() != core.DEFAULT_FIXTURE.resolve() or original.get(
        "sha256"
    ) != sha256_file(core.DEFAULT_FIXTURE):
        raise RuntimeError("v5 original fixture authority changed")
    return fixture, sha256_file(path)


def load_v5_case_indices(case: Mapping[str, Any], tier: str) -> tuple[Any, str]:
    """Resolve unchanged nested indices against their original frozen root."""
    return _ORIGINAL_LOAD_CASE_INDICES(core.DEFAULT_FIXTURE.parent, case, tier)


def install_v5_authority_adapter() -> None:
    """Patch only fixture/index authority; leave all geometry and P/R/G math intact."""
    core._load_frozen_fixture = load_v5_fixture
    core._load_case_indices = lambda _root, case, tier: load_v5_case_indices(case, tier)


def build_oracle_cache(**kwargs: Any) -> dict[str, Any]:
    """Delegate to the frozen core evaluator under v5 authority resolution."""
    install_v5_authority_adapter()
    return core.build_oracle_cache(**kwargs)


def evaluate_candidate_streaming(**kwargs: Any) -> dict[str, Any]:
    """Delegate unchanged streaming P/R/G computation under v5 authority."""
    install_v5_authority_adapter()
    return core.evaluate_candidate_streaming(**kwargs)


def run_bounded_real_adapter_test() -> dict[str, Any]:
    """Run one selection-forbidden real chunk through the v5 authority adapter."""
    fixture, fixture_sha = load_v5_fixture(FIXTURE_PATH)
    candidates = [
        candidate
        for candidate in fixture["candidates"]
        if candidate["candidate_id"] == BOUNDED_CANDIDATE_ID
    ]
    if len(candidates) != 1:
        raise RuntimeError("bounded v5 candidate identity changed")
    oracle_root = BOUNDED_ROOT / "oracle_cache"
    result_path = BOUNDED_ROOT / "candidate_result.json"
    oracle = build_oracle_cache(
        fixture_path=FIXTURE_PATH,
        case_id=BOUNDED_CASE_ID,
        candidate_id=BOUNDED_CANDIDATE_ID,
        tier="screen",
        output_root=oracle_root,
        max_cem_chunks=1,
    )
    result = evaluate_candidate_streaming(
        fixture_path=FIXTURE_PATH,
        case_id=BOUNDED_CASE_ID,
        candidate_id=BOUNDED_CANDIDATE_ID,
        tier="screen",
        oracle_cache_root=oracle_root,
        output_path=result_path,
        max_cem_chunks=1,
    )
    checks = {
        "oracle_complete": oracle["status"] == "COMPLETE",
        "candidate_complete": result["status"] == "COMPLETE",
        "fixture_sha_exact": result["fixture"]["sha256"] == fixture_sha,
        "candidate_asset_exact": result["candidate_asset_sha256"]
        == candidates[0]["candidate_asset_sha256"],
        "one_cem_chunk": result["r_shadow"]["chunk_count"] == 1
        and result["g_shadow"]["chunk_count"] == 1,
        "finite": result["point_metrics"]["nonfinite_count"] == 0,
        "selection_forbidden": not oracle["selection_eligible"]
        and not result["selection_eligible"],
        "heldout_not_accessed": result["heldout_access"] == "NOT_ACCESSED_DEV3_ONLY",
    }
    payload = {
        "experiment_id": "E182",
        "stage": "S2_bounded_real_v5_authority_adapter",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "selection_eligible": False,
        "heldout_access": "NOT_ACCESSED_DEV3_ONLY",
        "case_id": BOUNDED_CASE_ID,
        "candidate_id": BOUNDED_CANDIDATE_ID,
        "tier": "screen",
        "max_cem_chunks": 1,
        "checks": checks,
        "query_count": result["point_metrics"]["query_count"],
        "oracle_manifest": {
            "path": relative_to_repo(oracle_root / "manifest.json"),
            "sha256": sha256_file(oracle_root / "manifest.json"),
        },
        "candidate_result": {
            "path": relative_to_repo(result_path),
            "sha256": sha256_file(result_path),
        },
        "adapter_source": {
            "path": relative_to_repo(Path(__file__)),
            "sha256": sha256_file(Path(__file__)),
        },
    }
    atomic_json(BOUNDED_ROOT / "audit.json", payload)
    return payload


def main() -> int:
    """Run the explicitly selection-forbidden v5 adapter canary."""
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("bounded-real",))
    args = parser.parse_args()
    if args.command == "bounded-real":
        payload = run_bounded_real_adapter_test()
        print(
            f"E182_S2_BOUNDED_REAL_V5={payload['status']} "
            f"queries={payload['query_count']}"
        )
        return 0 if payload["status"] == "PASS" else 1
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
