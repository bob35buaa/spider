#!/usr/bin/env python3
"""Freeze, execute, and aggregate the E182 dev3 54-candidate task-query screen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from build_task_query_fixture import DEFAULT_OUTPUT_ROOT
from e182_common import atomic_json, atomic_tsv, relative_to_repo, sha256_file
from evaluate_task_queries import (
    DEFAULT_FIXTURE,
    build_oracle_cache,
    evaluate_candidate_streaming,
)

SCREEN_ROOT = DEFAULT_OUTPUT_ROOT / "screen"
PROTOCOL_PATH = SCREEN_ROOT / "screen_protocol_manifest.json"
SCORING_CONTRACT = {
    "name": "P_R_G_WORST_NORMALIZED_LAUNCH_FLOOR_ERROR_V1",
    "P": {
        "one_minus_contact_precision_denominator": 0.30,
        "one_minus_contact_recall_denominator": 0.30,
    },
    "R": {
        "point_sign_disagreement_denominator": 0.15,
        "point_deep_sign_mismatch_denominator": 0.05,
        "point_sdf_p90_denominator_m": 0.03,
        "component_normalized_error_p90_denominator": 0.30,
        "one_minus_geometry_spearman_denominator": 0.20,
    },
    "G": {
        "optimizer_mask_flip_denominator": 0.15,
        "optimizer_false_reject_denominator": 0.15,
        "optimizer_false_accept_denominator": 0.15,
        "positive_fallback_increase_denominator": 0.20,
        "one_minus_topk_overlap_denominator": 1.0,
    },
    "worst": "MAX(P,R,G)",
    "near_tie": "score <= best_score * 1.10; if best_score=0 require score=0",
    "selection": [
        "launch_floor_PASS_only",
        "minimum_worst_normalized_task_error",
        "within_10pct_choose_fewer_actual_hulls",
        "candidate_wall_seconds",
        "threshold_m",
        "max_vertices",
        "candidate_id",
    ],
}


def _load_fixture(path: Path) -> dict[str, Any]:
    """Load only the frozen dev3 pre-score fixture."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "FROZEN":
        raise RuntimeError("query fixture is not frozen")
    if payload.get("heldout_access") != "NOT_ACCESSED_DEV3_ONLY":
        raise RuntimeError("heldout isolation changed")
    if payload.get("candidate_count") != 54:
        raise RuntimeError("candidate count changed")
    return payload


def freeze_screen_protocol(
    *, fixture_path: Path = DEFAULT_FIXTURE, output_root: Path = SCREEN_ROOT
) -> dict[str, Any]:
    """Freeze score/selection semantics before any selection-eligible result exists."""
    fixture = _load_fixture(fixture_path)
    candidate_results = list(output_root.glob("results/*/*.json"))
    if candidate_results:
        raise RuntimeError("cannot freeze protocol after candidate results exist")
    payload = {
        "experiment_id": "E182",
        "stage": "S2_screen_protocol",
        "status": "FROZEN_BEFORE_SELECTION_ELIGIBLE_SCORES",
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
            f"{candidate['object_key']}/{candidate['candidate_id']}"
            for candidate in fixture["candidates"]
        ],
        "scoring_contract": SCORING_CONTRACT,
        "broader_cavity_role": "REPORT_ONLY_NOT_READ_BY_SCREEN",
        "heldout_role": "SELECTION_FORBIDDEN_NOT_ACCESSED",
        "bounded_real_role": "INTEGRATION_ONLY_NOT_READ_BY_SELECTION",
        "evaluator_source": {
            "path": relative_to_repo(
                Path(__file__).with_name("evaluate_task_queries.py")
            ),
            "sha256": sha256_file(Path(__file__).with_name("evaluate_task_queries.py")),
        },
        "runner_source": {
            "path": relative_to_repo(Path(__file__)),
            "sha256": sha256_file(Path(__file__)),
        },
    }
    path = output_root / "screen_protocol_manifest.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise RuntimeError("frozen screen protocol differs from current sources")
        return existing
    atomic_json(path, payload)
    return payload


def _load_protocol(output_root: Path, fixture_path: Path) -> dict[str, Any]:
    """Require an exact pre-score protocol before cache or candidate execution."""
    path = output_root / "screen_protocol_manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "FROZEN_BEFORE_SELECTION_ELIGIBLE_SCORES":
        raise RuntimeError("screen protocol is not frozen")
    if payload["fixture"]["sha256"] != sha256_file(fixture_path):
        raise RuntimeError("screen protocol fixture SHA changed")
    for key in ("evaluator_source", "runner_source"):
        source = Path(__file__).with_name(
            "evaluate_task_queries.py"
            if key == "evaluator_source"
            else Path(__file__).name
        )
        if sha256_file(source) != payload[key]["sha256"]:
            raise RuntimeError(f"screen protocol {key} SHA changed")
    return payload


def run_case_screen(
    case_id: str,
    *,
    fixture_path: Path = DEFAULT_FIXTURE,
    output_root: Path = SCREEN_ROOT,
) -> list[dict[str, Any]]:
    """Build one D_M cache and evaluate all 18 same-object candidates serially."""
    protocol = _load_protocol(output_root, fixture_path)
    fixture = _load_fixture(fixture_path)
    case_rows = [case for case in fixture["cases"] if case["case_id"] == case_id]
    if len(case_rows) != 1:
        raise RuntimeError(f"case is outside frozen dev3: {case_id}")
    case = case_rows[0]
    candidates = [
        candidate
        for candidate in fixture["candidates"]
        if candidate["object_key"] == case["object_key"]
    ]
    if len(candidates) != 18:
        raise RuntimeError("object candidate count changed")
    candidates.sort(key=lambda value: value["candidate_id"])
    oracle_root = output_root / "oracle_cache" / case_id
    oracle_manifest = oracle_root / "manifest.json"
    if not oracle_manifest.exists():
        build_oracle_cache(
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
                raise RuntimeError(f"non-resumable candidate result: {result_path}")
        else:
            result = evaluate_candidate_streaming(
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
            "E182_S2_SCREEN_PROGRESS "
            f"case={case_id} candidate={index}/18 id={candidate['candidate_id']} "
            f"floor={result['launch_floor']['status']} "
            f"wall={result['wall_seconds']:.3f}"
        )
    return results


def normalized_task_score(result: dict[str, Any]) -> dict[str, float]:
    """Apply the fully frozen P/R/G worst-error formula to one screen result."""
    point = result["point_metrics"]
    contact = result["p_pose_contact"]
    reward = result["r_shadow"]
    gate = result["g_shadow"]
    p_score = max(
        (1.0 - float(contact["precision"])) / 0.30,
        (1.0 - float(contact["recall"])) / 0.30,
    )
    r_score = max(
        float(point["sign_disagreement_fraction"]) / 0.15,
        float(point["deep_sign_mismatch_fraction"]) / 0.05,
        float(point["absolute_error_p90_m"]) / 0.03,
        float(reward["normalized_error_p90"]) / 0.30,
        (1.0 - float(reward["geometry_spearman_median"])) / 0.20,
    )
    optimizer_gate = gate["optimizer_combined"]
    g_score = max(
        float(optimizer_gate["mask_flip_fraction"]) / 0.15,
        float(optimizer_gate["false_reject_fraction"]) / 0.15,
        float(optimizer_gate["false_accept_fraction"]) / 0.15,
        max(
            0.0,
            float(gate["candidate_fallback_fraction"])
            - float(gate["raw_fallback_fraction"]),
        )
        / 0.20,
        (1.0 - float(gate["topk_overlap_mean"])) / 1.0,
    )
    return {
        "P": p_score,
        "R": r_score,
        "G": g_score,
        "worst": max(p_score, r_score, g_score),
    }


def _select_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Select one object×K screen finalist using the pre-frozen lexicographic rule."""
    passing = [row for row in rows if row["launch_floor_status"] == "PASS"]
    if not passing:
        return {
            "status": "NO_LAUNCH_FLOOR_CANDIDATE",
            "candidate_count": len(rows),
            "launch_floor_pass_count": 0,
            "selected": None,
        }
    best_score = min(row["score_worst"] for row in passing)
    ceiling = best_score * 1.10 if best_score > 0.0 else 0.0
    near = [row for row in passing if row["score_worst"] <= ceiling + 1e-12]
    selected = min(
        near,
        key=lambda row: (
            row["actual_hulls"],
            row["wall_seconds"],
            row["threshold_m"],
            row["max_vertices"],
            row["candidate_id"],
        ),
    )
    return {
        "status": "SELECTED_FOR_FINALIST_REVIEW",
        "candidate_count": len(rows),
        "launch_floor_pass_count": len(passing),
        "best_score": best_score,
        "near_tie_ceiling": ceiling,
        "near_tie_count": len(near),
        "selected": selected,
    }


def aggregate_screen(
    *,
    fixture_path: Path = DEFAULT_FIXTURE,
    output_root: Path = SCREEN_ROOT,
) -> dict[str, Any]:
    """Require 54/54 exact results, score them, and select nine provisional finalists."""
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
            raise RuntimeError(f"missing screen result: {result_path}")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("selection_eligible") is not True:
            raise RuntimeError(
                f"screen result is not selection eligible: {result_path}"
            )
        if result["fixture"]["sha256"] != protocol["fixture"]["sha256"]:
            raise RuntimeError("screen result fixture SHA drift")
        score = normalized_task_score(result)
        rows.append(
            {
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
                "g_mask_flip_fraction": result["g_shadow"]["optimizer_combined"][
                    "mask_flip_fraction"
                ],
                "g_false_reject_fraction": result["g_shadow"]["optimizer_combined"][
                    "false_reject_fraction"
                ],
                "g_false_accept_fraction": result["g_shadow"]["optimizer_combined"][
                    "false_accept_fraction"
                ],
                "fallback_delta": result["g_shadow"]["candidate_fallback_fraction"]
                - result["g_shadow"]["raw_fallback_fraction"],
                "topk_overlap_mean": result["g_shadow"]["topk_overlap_mean"],
                "wall_seconds": float(result["wall_seconds"]),
                "result_path": relative_to_repo(result_path),
                "result_sha256": sha256_file(result_path),
            }
        )
    if len(rows) != 54:
        raise RuntimeError(f"screen row count mismatch: {len(rows)}")
    groups = []
    for object_key in ("bucket003", "bucket004", "bucket007"):
        for max_hulls in (8, 16, 32):
            group_rows = [
                row
                for row in rows
                if row["object_key"] == object_key and row["max_hulls"] == max_hulls
            ]
            if len(group_rows) != 6:
                raise RuntimeError(
                    f"candidate group size changed: {object_key}/K{max_hulls}"
                )
            groups.append(
                {
                    "object_key": object_key,
                    "max_hulls": max_hulls,
                    **_select_group(group_rows),
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
        "stage": "S2_screen_aggregate",
        "status": "COMPLETE",
        "heldout_access": "NOT_ACCESSED_DEV3_ONLY",
        "protocol": {
            "path": relative_to_repo(output_root / "screen_protocol_manifest.json"),
            "sha256": sha256_file(output_root / "screen_protocol_manifest.json"),
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
    """Parse protocol, per-case screen, and aggregate actions."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command", choices=("freeze-protocol", "run-case", "run-all", "aggregate")
    )
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--output-root", type=Path, default=SCREEN_ROOT)
    parser.add_argument("--case-id")
    return parser.parse_args()


def main() -> int:
    """Execute one resume-safe screen action."""
    args = parse_args()
    if args.command == "freeze-protocol":
        payload = freeze_screen_protocol(
            fixture_path=args.fixture, output_root=args.output_root
        )
        print(
            f"E182_S2_SCREEN_PROTOCOL={payload['status']} candidates={payload['candidate_count']}"
        )
        return 0
    fixture = _load_fixture(args.fixture)
    if args.command == "run-case":
        if not args.case_id:
            raise RuntimeError("run-case requires --case-id")
        results = run_case_screen(
            args.case_id, fixture_path=args.fixture, output_root=args.output_root
        )
        print(f"E182_S2_SCREEN_CASE=PASS case={args.case_id} results={len(results)}")
        return 0
    if args.command == "run-all":
        for case in fixture["cases"]:
            run_case_screen(
                case["case_id"], fixture_path=args.fixture, output_root=args.output_root
            )
        print("E182_S2_SCREEN_ALL=PASS results=54")
        return 0
    payload = aggregate_screen(fixture_path=args.fixture, output_root=args.output_root)
    print(
        f"E182_S2_SCREEN_AGGREGATE={payload['status']} "
        f"pass={payload['launch_floor_pass_count']}/54 "
        f"selected={payload['selected_group_count']}/9"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
