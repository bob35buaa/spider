#!/usr/bin/env python3
"""Freeze E187 A1 continuation reward and object-specific production grids."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[5]
RESULTS_E187 = REPO / "workspace/core4d/results/E187"
OUTPUT = RESULTS_E187 / "s2_canonical_grid_sdf/reward_grid_lock.json"
PLAN205 = (
    REPO
    / "workspace/core4d/plan/205_E187_canonical_distance_continuation_reward_plan.md"
)
PLAN206 = (
    REPO / "workspace/core4d/plan/206_E187_user_waived_gate0_full_continuation_plan.md"
)
LOG254 = REPO / "workspace/core4d/log/254_E186_authority_grid_sdf_robust_p_results.md"
WAIVER = RESULTS_E187 / "s0_environment/gate0_user_waiver/waiver_manifest.json"
REWARD_SOURCE = REPO / "spider/rewards/surface_distance.py"
OBJECTS = {
    "bucket003": {
        "resolution_m": 0.005,
        "grid": REPO
        / "workspace/core4d/results/E186/s1_canonical_grid_sdf_v4/bucket003/manifest.json",
        "formal_fidelity": RESULTS_E187
        / "s2_canonical_grid_sdf/formal_fidelity_v2/eval/"
        "bucket003_20231018_003_p1.json",
        "efficiency": RESULTS_E187
        / "s2_canonical_grid_sdf/efficiency_v1/bucket003_20231018_003_p1.json",
        "decision": "5mm_FIRST_CANDIDATE_PASS",
    },
    "bucket004": {
        "resolution_m": 0.0025,
        "grid": REPO
        / "workspace/core4d/results/E186/s1_canonical_grid_sdf_v4/bucket004/manifest.json",
        "formal_fidelity": RESULTS_E187
        / "s2_canonical_grid_sdf/formal_fidelity_v2/eval/"
        "bucket004_20231002_021_p1.json",
        "efficiency": RESULTS_E187
        / "s2_canonical_grid_sdf/efficiency_v1/bucket004_20231002_021_p1.json",
        "formal_tape_manifest": RESULTS_E187
        / "s2_canonical_grid_sdf/bucket004_formal_tape/manifests/"
        "bucket004_20231002_021_p1.json",
        "decision": "5mm_E186_SMOKE_MAX_12P73MM_THEN_2P5MM_PASS",
    },
    "bucket007": {
        "resolution_m": 0.0025,
        "grid": RESULTS_E187
        / "s2_canonical_grid_sdf/candidates/bucket007_2p5mm/manifest.json",
        "formal_fidelity": RESULTS_E187
        / "s2_canonical_grid_sdf/formal_fidelity_bucket007_2p5mm_v1/eval/"
        "bucket007_20231020_055_p1.json",
        "rejected_5mm_fidelity": RESULTS_E187
        / "s2_canonical_grid_sdf/formal_fidelity_v2/eval/"
        "bucket007_20231020_055_p1.json",
        "efficiency": RESULTS_E187
        / "s2_canonical_grid_sdf/efficiency_v1/bucket007_20231020_055_p1.json",
        "decision": "5mm_TOPK_0P843_FAIL_THEN_2P5MM_TOPK_0P941_PASS",
    },
}


def sha256(path: Path) -> str:
    """Return a streaming SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path: Path) -> str:
    """Render lexical repository-relative artifact paths."""
    absolute = path.absolute()
    try:
        return absolute.relative_to(REPO.absolute()).as_posix()
    except ValueError:
        return str(absolute)


def artifact(path: Path) -> dict[str, Any]:
    """Freeze one required artifact path, SHA, and size."""
    if not path.is_file():
        raise RuntimeError(f"required A1 artifact is missing: {path}")
    return {
        "path": relative(path),
        "sha256": sha256(path),
        "size_bytes": path.stat().st_size,
    }


def load_json(path: Path) -> dict[str, Any]:
    """Load one required JSON object."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"A1 JSON is not an object: {path}")
    return value


def validate_fidelity(payload: dict[str, Any], object_key: str) -> None:
    """Require every preregistered formal fidelity gate to pass."""
    if (
        payload.get("object_key") != object_key
        or payload.get("fidelity_status") != "PASS"
        or not payload.get("gates")
        or not all(payload["gates"].values())
        or payload.get("gate0_technical_status") != "FAIL"
        or payload.get("progression_authority") != "USER_WAIVED"
    ):
        raise RuntimeError(f"{object_key}: formal fidelity is not a waived PASS")


def validate_efficiency(payload: dict[str, Any], object_key: str) -> None:
    """Require same-tape throughput and memory gates to pass."""
    if (
        payload.get("object_key") != object_key
        or payload.get("status") != "PASS"
        or not all(payload.get("gates", {}).values())
        or float(payload["chosen_over_baseline_ratio"]) > 1.25
        or float(payload["chosen_peak_allocated_mib"]) > 6144.0
    ):
        raise RuntimeError(f"{object_key}: efficiency gate did not pass")


def build_payload() -> dict[str, Any]:
    """Build and validate the deterministic A1 lock payload."""
    waiver = load_json(WAIVER)
    if (
        waiver.get("technical_gate_status") != "FAIL"
        or waiver.get("progression_authority") != "USER_WAIVED"
    ):
        raise RuntimeError("Gate0 waiver semantics changed before A1 lock")
    object_payload: dict[str, Any] = {}
    for object_key, spec in OBJECTS.items():
        grid = load_json(spec["grid"])
        fidelity = load_json(spec["formal_fidelity"])
        efficiency = load_json(spec["efficiency"])
        validate_fidelity(fidelity, object_key)
        validate_efficiency(efficiency, object_key)
        if (
            grid.get("status") != "GRID_FROZEN"
            or grid.get("object_key") != object_key
            or float(grid["grid"]["voxel_size_m"]) != spec["resolution_m"]
            or grid["validation"]["status"] != "PASS"
            or grid["validation"]["validation_point_count"] != 1_000_000
            or grid["validation"]["minkowski_support_coverage"][
                "minimum_actual_padding_m"
            ]
            < 0.110
        ):
            raise RuntimeError(f"{object_key}: selected production grid is invalid")
        row = {
            "decision": spec["decision"],
            "resolution_m": spec["resolution_m"],
            "epsilon_grid_m": grid["validation"]["epsilon_grid_m"],
            "grid_payload_sha256": grid["grid"]["sha256"],
            "grid_manifest": artifact(spec["grid"]),
            "formal_fidelity": artifact(spec["formal_fidelity"]),
            "formal_fidelity_scientific_payload_sha256": fidelity[
                "scientific_payload_sha256"
            ],
            "efficiency": artifact(spec["efficiency"]),
            "metrics": {
                "false_safe_accept_count": fidelity["metrics"][
                    "false_safe_accept_count"
                ],
                "selected_exact_valid_frac": fidelity["metrics"][
                    "grid_selected_exact_valid_frac"
                ],
                "surface_component_abs_p99": fidelity["metrics"][
                    "surface_component_grid_exact_abs_p99"
                ],
                "total_reward_spearman": fidelity["metrics"]["total_reward_spearman"],
                "topk_overlap_frac": fidelity["metrics"]["topk_overlap_frac"],
                "selected0_exact_regret_frac": fidelity["metrics"][
                    "grid_selected0_exact_regret_frac"
                ],
                "selected0_exact_percentile": fidelity["metrics"][
                    "grid_selected0_exact_percentile"
                ],
                "query_kernel_ratio": efficiency["chosen_over_baseline_ratio"],
                "query_peak_allocated_mib": efficiency["chosen_peak_allocated_mib"],
            },
        }
        if "formal_tape_manifest" in spec:
            tape = load_json(spec["formal_tape_manifest"])
            if tape.get("status") != "PASS" or tape["summary"]["sample_count"] != 1024:
                raise RuntimeError("bucket004 formal tape is not PASS")
            row["formal_tape_manifest"] = artifact(spec["formal_tape_manifest"])
            row["formal_tape_content_sha256"] = tape["chunk_content_sha256"]
        if "rejected_5mm_fidelity" in spec:
            rejected = load_json(spec["rejected_5mm_fidelity"])
            if (
                rejected.get("fidelity_status") != "FAIL"
                or rejected["gates"].get("elite_topk_overlap_ge_0p90") is not False
            ):
                raise RuntimeError("bucket007 5mm rejection evidence changed")
            row["rejected_5mm_fidelity"] = artifact(spec["rejected_5mm_fidelity"])
            row["rejected_5mm_topk_overlap_frac"] = rejected["metrics"][
                "topk_overlap_frac"
            ]
        object_payload[object_key] = row

    payload = {
        "schema": "e187_reward_grid_lock_v1",
        "experiment_id": "E187",
        "stage": "A1_S1_S2_REWARD_GRID_LOCK",
        "status": "FROZEN",
        "gate0_technical_status": "FAIL",
        "progression_authority": "USER_WAIVED",
        "gate0_waiver": artifact(WAIVER),
        "authority": {
            "plan205": artifact(PLAN205),
            "plan206": artifact(PLAN206),
            "bucket004_5mm_rejection_log": artifact(LOG254),
            "reward_source": artifact(REWARD_SOURCE),
        },
        "reward": {
            "mode": "distance_continuation",
            "surface_scale": 1.5,
            "temporal_gate": "contact_mask",
            "tail_decay_frac": 0.15,
            "far_weight": 0.25,
            "near_weight": 0.75,
            "far_scale_m": 0.050,
            "near_scale_m": 0.015,
            "smooth_delta_m": 0.001,
            "legacy_modes_unchanged": ["one_sided", "symmetric_abs"],
        },
        "gate": {
            "object_distance_rule": "D_C_MINUS_EPSILON_GRID",
            "reward_distance_rule": "NOMINAL_D_C_NO_EPSILON_SUBTRACTION",
            "false_safe_accept_total": 0,
            "selected_exact_valid_required_frac": 1.0,
        },
        "coarse_to_fine_policy": [0.005, 0.0025, 0.00125],
        "objects": object_payload,
        "global_gates": {
            "object_count_3": len(object_payload) == 3,
            "all_fidelity_pass": True,
            "all_false_safe_zero": all(
                row["metrics"]["false_safe_accept_count"] == 0
                for row in object_payload.values()
            ),
            "all_selected_exact_valid_100pct": all(
                row["metrics"]["selected_exact_valid_frac"] == 1.0
                for row in object_payload.values()
            ),
            "all_query_ratio_le_1p25": all(
                row["metrics"]["query_kernel_ratio"] <= 1.25
                for row in object_payload.values()
            ),
            "all_query_peak_le_6144mib": all(
                row["metrics"]["query_peak_allocated_mib"] <= 6144.0
                for row in object_payload.values()
            ),
            "no_finer_grid_inspected_after_pass": True,
        },
        "downstream_contract": {
            "immutable_after_freeze": True,
            "full_prohibited_until_a2_a3_pass": True,
            "use_a100": False,
            "process_interference_allowed": False,
        },
    }
    if not all(payload["global_gates"].values()):
        raise RuntimeError("E187 A1 global gates did not all pass")
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["scientific_payload_sha256"] = hashlib.sha256(canonical).hexdigest()
    return payload


def write_immutable(payload: dict[str, Any]) -> None:
    """Create or verify the byte-identical reward/grid lock."""
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    if OUTPUT.exists():
        if OUTPUT.read_bytes() != encoded:
            raise RuntimeError("immutable E187 reward/grid lock mismatch")
        return
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{OUTPUT.name}.", dir=OUTPUT.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
        os.replace(temporary, OUTPUT)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def main() -> int:
    """Validate all A1 evidence and freeze its downstream lock."""
    payload = build_payload()
    write_immutable(payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
