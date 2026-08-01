#!/usr/bin/env python3
"""Freeze, execute, and aggregate the bucket003 nine-candidate v5 screen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import evaluate_task_queries as evaluator_core
import evaluate_task_queries_v5 as evaluator_adapter
import run_task_query_screen as original_screen
from build_segmented_coacd_attempt2_v5 import ATTEMPT_ROOT, FIXTURE_PATH
from e182_common import atomic_json, atomic_tsv, relative_to_repo, sha256_file

SCREEN_ROOT = ATTEMPT_ROOT / "screen"
PROTOCOL_PATH = SCREEN_ROOT / "screen_protocol_manifest.json"
CASE_ID = "bucket003_20231018_001_p1"
OBJECT_KEY = "bucket003"
EXPECTED_CANDIDATES = 9
EXPECTED_PER_K = 3
SCORING_CONTRACT = original_screen.SCORING_CONTRACT


def _load_fixture(path: Path) -> dict[str, Any]:
    """Load exactly the v5 fixture through its strict authority adapter."""
    return evaluator_adapter.load_v5_fixture(path)[0]


def freeze_screen_protocol(
    *, fixture_path: Path = FIXTURE_PATH, output_root: Path = SCREEN_ROOT
) -> dict[str, Any]:
    """Freeze all scoring and authority sources before selection-eligible results."""
    fixture = _load_fixture(fixture_path)
    if list(output_root.glob("results/*/*.json")):
        raise RuntimeError("cannot freeze v5 screen after candidate results exist")
    sources = {
        "evaluator_core_source": Path(evaluator_core.__file__).resolve(),
        "evaluator_adapter_source": Path(evaluator_adapter.__file__).resolve(),
        "scoring_source": Path(original_screen.__file__).resolve(),
        "runner_source": Path(__file__).resolve(),
    }
    payload = {
        "experiment_id": "E182",
        "stage": "S2_screen_protocol_attempt2_v5",
        "status": "FROZEN_BEFORE_SELECTION_ELIGIBLE_V5_SCORES",
        "heldout_access": "NOT_ACCESSED_DEV3_ONLY",
        "tier": "screen",
        "max_cem_chunks": None,
        "pose_policy": "ALL_STATIC_FULL_AND_ALL_CEM_MESH64",
        "fixture": {
            "path": relative_to_repo(fixture_path),
            "sha256": sha256_file(fixture_path),
            "candidate_identity_sha256": fixture["candidate_identity_sha256"],
        },
        "candidate_count": fixture["candidate_count"],
        "case_ids": [case["case_id"] for case in fixture["cases"]],
        "candidate_ids": [
            candidate["candidate_id"] for candidate in fixture["candidates"]
        ],
        "group_contract": {
            "object_key": OBJECT_KEY,
            "max_hulls": [8, 16, 32],
            "candidates_per_K": EXPECTED_PER_K,
            "thresholds_m": [0.005, 0.010, 0.020],
        },
        "scoring_contract": SCORING_CONTRACT,
        "authority_adapter_role": (
            "ONLY_FIXTURE_COUNT_AND_ORIGINAL_INDEX_ROOT;P_R_G_MATH_UNCHANGED"
        ),
        "broader_cavity_role": "REPORT_ONLY_NOT_READ_BY_SCREEN",
        "heldout_role": "SELECTION_FORBIDDEN_NOT_ACCESSED",
        "sources": {
            key: {"path": relative_to_repo(path), "sha256": sha256_file(path)}
            for key, path in sources.items()
        },
    }
    path = output_root / "screen_protocol_manifest.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise RuntimeError("frozen v5 screen protocol differs")
        return existing
    atomic_json(path, payload)
    return payload


def _load_protocol(output_root: Path, fixture_path: Path) -> dict[str, Any]:
    """Require exact pre-score v5 protocol and source hashes."""
    path = output_root / "screen_protocol_manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "FROZEN_BEFORE_SELECTION_ELIGIBLE_V5_SCORES":
        raise RuntimeError("v5 screen protocol is not frozen")
    if payload["fixture"]["sha256"] != sha256_file(fixture_path):
        raise RuntimeError("v5 screen fixture SHA changed")
    for source in payload["sources"].values():
        path = Path(source["path"])
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[5] / path
        if source["sha256"] != sha256_file(path):
            raise RuntimeError(f"v5 screen source SHA changed: {path}")
    return payload


def run_case_screen(
    case_id: str = CASE_ID,
    *,
    fixture_path: Path = FIXTURE_PATH,
    output_root: Path = SCREEN_ROOT,
) -> list[dict[str, Any]]:
    """Build one D_M cache and evaluate the nine v5 candidates serially."""
    protocol = _load_protocol(output_root, fixture_path)
    fixture = _load_fixture(fixture_path)
    cases = [case for case in fixture["cases"] if case["case_id"] == case_id]
    if len(cases) != 1 or case_id != CASE_ID:
        raise RuntimeError(f"case is outside frozen v5 authority: {case_id}")
    case = cases[0]
    candidates = [
        candidate
        for candidate in fixture["candidates"]
        if candidate["object_key"] == case["object_key"]
    ]
    if len(candidates) != EXPECTED_CANDIDATES:
        raise RuntimeError("v5 candidate count changed")
    candidates.sort(key=lambda value: value["candidate_id"])
    oracle_root = output_root / "oracle_cache" / case_id
    oracle_manifest = oracle_root / "manifest.json"
    if not oracle_manifest.exists():
        evaluator_adapter.build_oracle_cache(
            fixture_path=fixture_path,
            case_id=case_id,
            candidate_id=candidates[0]["candidate_id"],
            tier="screen",
            output_root=oracle_root,
            max_cem_chunks=None,
        )
    results = []
    for index, candidate in enumerate(candidates, start=1):
        result_path = (
            output_root
            / "results"
            / case["object_key"]
            / f"{candidate['candidate_id']}.json"
        )
        if result_path.exists():
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if (
                result.get("status") != "COMPLETE"
                or result.get("selection_eligible") is not True
                or result.get("fixture", {}).get("sha256")
                != protocol["fixture"]["sha256"]
                or result.get("candidate_asset_sha256")
                != candidate["candidate_asset_sha256"]
            ):
                raise RuntimeError(f"non-resumable v5 result: {result_path}")
        else:
            result = evaluator_adapter.evaluate_candidate_streaming(
                fixture_path=fixture_path,
                case_id=case_id,
                candidate_id=candidate["candidate_id"],
                tier="screen",
                oracle_cache_root=oracle_root,
                output_path=result_path,
                max_cem_chunks=None,
            )
        results.append(result)
        print(
            "E182_S2_SCREEN_V5_PROGRESS "
            f"candidate={index}/{EXPECTED_CANDIDATES} id={candidate['candidate_id']} "
            f"floor={result['launch_floor']['status']} wall={result['wall_seconds']:.3f}"
        )
    return results


def _candidate_row(
    candidate: dict[str, Any], result: dict[str, Any], result_path: Path
) -> dict[str, Any]:
    score = original_screen.normalized_task_score(result)
    gate = result["g_shadow"]
    optimizer_gate = gate["optimizer_combined"]
    return {
        "object_key": candidate["object_key"],
        "candidate_id": candidate["candidate_id"],
        "max_hulls": int(candidate["max_hulls"]),
        "actual_hulls": int(candidate["actual_hulls"]),
        "threshold_m": float(candidate["threshold_m"]),
        "max_vertices": int(candidate["max_vertices"]),
        "launch_floor_status": result["launch_floor"]["status"],
        "launch_floor_passed": int(result["launch_floor"]["passed"]),
        "score_P": score["P"],
        "score_R": score["R"],
        "score_G": score["G"],
        "score_worst": score["worst"],
        "sign_disagreement_fraction": result["point_metrics"][
            "sign_disagreement_fraction"
        ],
        "deep_sign_mismatch_fraction": result["point_metrics"][
            "deep_sign_mismatch_fraction"
        ],
        "sdf_p90_m": result["point_metrics"]["absolute_error_p90_m"],
        "p_precision": result["p_pose_contact"]["precision"],
        "p_recall": result["p_pose_contact"]["recall"],
        "r_normalized_error_p90": result["r_shadow"]["normalized_error_p90"],
        "r_spearman_median": result["r_shadow"]["geometry_spearman_median"],
        "g_mask_flip_fraction": optimizer_gate["mask_flip_fraction"],
        "g_false_reject_fraction": optimizer_gate["false_reject_fraction"],
        "g_false_accept_fraction": optimizer_gate["false_accept_fraction"],
        "fallback_delta": gate["candidate_fallback_fraction"]
        - gate["raw_fallback_fraction"],
        "topk_overlap_mean": gate["topk_overlap_mean"],
        "wall_seconds": float(result["wall_seconds"]),
        "result_path": relative_to_repo(result_path),
        "result_sha256": sha256_file(result_path),
    }


def aggregate_screen(
    *, fixture_path: Path = FIXTURE_PATH, output_root: Path = SCREEN_ROOT
) -> dict[str, Any]:
    """Require 9/9 results and select one provisional candidate per K."""
    protocol = _load_protocol(output_root, fixture_path)
    fixture = _load_fixture(fixture_path)
    rows = []
    for candidate in fixture["candidates"]:
        result_path = (
            output_root
            / "results"
            / candidate["object_key"]
            / f"{candidate['candidate_id']}.json"
        )
        if not result_path.is_file():
            raise RuntimeError(f"missing v5 screen result: {result_path}")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("selection_eligible") is not True:
            raise RuntimeError(f"v5 result is not selection eligible: {result_path}")
        if result["fixture"]["sha256"] != protocol["fixture"]["sha256"]:
            raise RuntimeError("v5 result fixture SHA drift")
        rows.append(_candidate_row(candidate, result, result_path))
    if len(rows) != EXPECTED_CANDIDATES:
        raise RuntimeError(f"v5 screen row count mismatch: {len(rows)}")
    groups = []
    for max_hulls in (8, 16, 32):
        group_rows = [row for row in rows if row["max_hulls"] == max_hulls]
        if len(group_rows) != EXPECTED_PER_K:
            raise RuntimeError(f"v5 candidate group size changed: K{max_hulls}")
        groups.append(
            {
                "object_key": OBJECT_KEY,
                "max_hulls": max_hulls,
                **original_screen._select_group(group_rows),
            }
        )
    fields = list(rows[0])
    atomic_tsv(
        output_root / "screen_candidates.tsv",
        [{key: str(row[key]) for key in fields} for row in rows],
        fields,
    )
    payload = {
        "experiment_id": "E182",
        "stage": "S2_screen_aggregate_attempt2_v5",
        "status": "COMPLETE",
        "heldout_access": "NOT_ACCESSED_DEV3_ONLY",
        "protocol": {
            "path": relative_to_repo(PROTOCOL_PATH),
            "sha256": sha256_file(PROTOCOL_PATH),
        },
        "candidate_count": len(rows),
        "launch_floor_pass_count": sum(
            row["launch_floor_status"] == "PASS" for row in rows
        ),
        "group_count": len(groups),
        "selected_group_count": sum(group["selected"] is not None for group in groups),
        "scoring_contract": SCORING_CONTRACT,
        "groups": groups,
        "candidate_table": {
            "path": relative_to_repo(output_root / "screen_candidates.tsv"),
            "sha256": sha256_file(output_root / "screen_candidates.tsv"),
        },
    }
    atomic_json(output_root / "screen_aggregate.json", payload)
    return payload


def parse_args() -> argparse.Namespace:
    """Parse v5 protocol, run, and aggregate actions."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command", choices=("freeze-protocol", "run-case", "run-all", "aggregate")
    )
    parser.add_argument("--fixture", type=Path, default=FIXTURE_PATH)
    parser.add_argument("--output-root", type=Path, default=SCREEN_ROOT)
    parser.add_argument("--case-id", default=CASE_ID)
    return parser.parse_args()


def main() -> int:
    """Execute one resume-safe v5 screen action."""
    args = parse_args()
    if args.command == "freeze-protocol":
        payload = freeze_screen_protocol(
            fixture_path=args.fixture, output_root=args.output_root
        )
        print(
            f"E182_S2_SCREEN_V5_PROTOCOL={payload['status']} "
            f"candidates={payload['candidate_count']}"
        )
        return 0
    if args.command in ("run-case", "run-all"):
        results = run_case_screen(
            args.case_id, fixture_path=args.fixture, output_root=args.output_root
        )
        print(f"E182_S2_SCREEN_V5_RUN=PASS results={len(results)}")
        return 0
    payload = aggregate_screen(fixture_path=args.fixture, output_root=args.output_root)
    print(
        f"E182_S2_SCREEN_V5_AGGREGATE={payload['status']} "
        f"pass={payload['launch_floor_pass_count']}/{EXPECTED_CANDIDATES} "
        f"selected={payload['selected_group_count']}/3"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
