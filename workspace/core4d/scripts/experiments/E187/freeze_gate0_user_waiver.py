#!/usr/bin/env python3
"""Freeze the user-authorized E187 Gate S0 progression waiver."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import freeze_authority

REPO_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "workspace/core4d/results/E187/s0_environment/gate0_user_waiver"
)
SOURCES = {
    "plan205": REPO_ROOT
    / "workspace/core4d/plan/205_E187_canonical_distance_continuation_reward_plan.md",
    "plan206": REPO_ROOT
    / "workspace/core4d/plan/206_E187_user_waived_gate0_full_continuation_plan.md",
    "log256": REPO_ROOT
    / "workspace/core4d/log/256_E187_e178_compatibility_gate0_blocker_results.md",
    "log257": REPO_ROOT
    / "workspace/core4d/log/257_E187_gate0_contract_audit_results.md",
    "compat_summary_v1": REPO_ROOT
    / "workspace/core4d/results/E187/s0_environment/e178_compat/eval/summary.json",
    "compat_summary_v2": REPO_ROOT
    / "workspace/core4d/results/E187/s0_environment/e178_compat/eval_contract_audit_v2/summary.json",
    "s0_protocol": REPO_ROOT
    / "workspace/core4d/results/E187/s0_environment/protocol_manifest.json",
    "keep22_protocol": REPO_ROOT
    / "workspace/core4d/results/E187/s0_environment/keep22_protocol_manifest.tsv",
    "e178_inventory": REPO_ROOT
    / "workspace/core4d/results/E187/s0_environment/e178_authority_sha_inventory.tsv",
    "e178_manifest": REPO_ROOT
    / "workspace/core4d/results/E178/s6_downstream/manifests/semantic_bucket_full_manifest.tsv",
}
EXPECTED_SHA256 = {
    "plan205": "9cc2fc9d275af5a53c850a1f99a4fef919d3d60195a0569837c1837a553b15fe",
    "plan206": "e89bc72c2307049eb4e7bee3b9ee57ef2e497fcf2bff46f3bc18a95c22ddad2f",
    "log256": "73f08f8c4f8c4c7dd63d484461e81c19e3336d17cb5738f3c32fec0a56030e50",
    "log257": "14ecfce1d8fa882b7f5bef38ad42e6c81c674450f872d43f5112caf52f4df848",
    "compat_summary_v1": "c9b42d8271193bff032c9d2b068697848516bcedd8d42a88123b2101dc4ce38d",
    "compat_summary_v2": "70a1fe8ec164fbd9ce4e286e80d7e3450a866c9555883da69978783c8d5f83c8",
    "s0_protocol": "35233dac151528776527113fae60aecfbabae8c124115555a56fd84e3c73a35b",
    "keep22_protocol": "b1122a3a99d9760ff2bcd22b90783ca93999b17f620923e8ecef44801d6a4924",
    "e178_inventory": "2a396baf6f13c9761f31c34f10e85a65307961a4cb5823e1e35f98ff3d308237",
    "e178_manifest": "de9a3d165301318049da208aad52b1f5bc4d3ed671631f0097958734a0f022a8",
}


def _validate_sources() -> None:
    """Require all preregistered negative evidence and authority SHAs."""
    freeze_authority.validate()
    for name, path in SOURCES.items():
        if not path.is_file():
            raise RuntimeError(f"missing waiver source: {name}: {path}")
        if freeze_authority.sha256_file(path) != EXPECTED_SHA256[name]:
            raise RuntimeError(f"waiver source SHA changed: {name}: {path}")


def waiver_payload() -> dict[str, Any]:
    """Return the result-independent user-waiver authority payload."""
    _validate_sources()
    return {
        "experiment_id": "E187",
        "stage": "A0_GATE0_USER_WAIVER",
        "status": "PASS",
        "technical_gate_status": "FAIL",
        "progression_authority": "USER_WAIVED",
        "authorization": {
            "date": "2026-08-02",
            "user_text": "先跳过吧,恢复E187,进入Full CEM吧",
            "interpretation": (
                "waive only the irrecoverable historical query/selected golden; "
                "preserve all negative evidence and all S1-S4 safety gates"
            ),
        },
        "waived_evidence": [
            "historical_query_qpos_reward_exact",
            "historical_per_sample_valid_mask_exact",
            "historical_selected_index_exact",
        ],
        "not_waived": [
            "e178_authority_and_source_isolation",
            "effective_config_whitelist",
            "legacy_pure_reward_regression",
            "continuation_reward_and_grid_fidelity",
            "keep22_physics_and_gate_regression",
            "three_device_production_canary",
            "efficiency_and_memory_gates",
            "full_output_and_visual_closure",
        ],
        "full_contract": {
            "cem_samples": 1024,
            "cem_opt_steps": 32,
            "cem_seed": 0,
            "devices": [
                "local_nvidia_geforce_rtx_5090_gpu0",
                "spider_remote_nvidia_rtx_6000_ada_gpu0",
                "spider_remote_nvidia_rtx_6000_ada_gpu1",
            ],
            "use_a100": False,
            "existing_process_policy": "coexist_no_kill_pause_or_preempt",
        },
        "sources": {
            name: {
                "path": freeze_authority._display_path(path),
                "sha256": EXPECTED_SHA256[name],
            }
            for name, path in SOURCES.items()
        },
    }


def freeze(output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    """Create or byte-validate the immutable waiver manifest."""
    output = output_root / "waiver_manifest.json"
    freeze_authority._write_immutable(
        output,
        freeze_authority._json_bytes(waiver_payload()),
    )
    return {
        "status": "PASS",
        "technical_gate_status": "FAIL",
        "progression_authority": "USER_WAIVED",
        "waiver_manifest": {
            "path": freeze_authority._display_path(output),
            "sha256": freeze_authority.sha256_file(output),
            "size_bytes": output.stat().st_size,
        },
    }


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    print(json.dumps(freeze(args.output_root), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
