#!/usr/bin/env python3
"""Close E182 Gate 0 from authority, deployment, and environment evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from e182_common import REPO_ROOT, atomic_json, relative_to_repo, sha256_file

DEFAULT_ROOT = REPO_ROOT / "workspace/core4d/results/E182/s0_environment"
DEFAULT_AUTHORITY = DEFAULT_ROOT / "authority_manifest.json"
DEFAULT_DEPLOYMENT = DEFAULT_ROOT / "remote_deployment_manifest.json"
DEFAULT_ENVIRONMENT = DEFAULT_ROOT / "environment_manifest.json"
DEFAULT_OUTPUT = DEFAULT_ROOT / "gate0_audit.json"


def check(name: str, passed: bool, evidence: Any) -> dict[str, Any]:
    """Build one explicit audit check."""
    return {"name": name, "pass": bool(passed), "evidence": evidence}


def evaluate_gate(
    authority: dict[str, Any],
    deployment: dict[str, Any],
    environment: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate Gate 0 without relying on file locations."""
    execution = environment.get("execution_contract", {})
    remote_environment = environment.get("remote_deployment", {})
    checks = [
        check(
            "authority_status",
            authority.get("status") == "PASS",
            authority.get("status"),
        ),
        check(
            "authority_rows",
            authority.get("case_count") == 27,
            authority.get("case_count"),
        ),
        check(
            "dev_heldout_disjoint",
            authority.get("dev_heldout_overlap") == [],
            authority.get("dev_heldout_overlap"),
        ),
        check(
            "cem_contract",
            authority.get("cem_contract")
            == {"samples": 1024, "opt_steps": 32, "seed": 0},
            authority.get("cem_contract"),
        ),
        check(
            "candidate_snapshot",
            (authority.get("snapshot") or {}).get("candidate_manifests") == 54,
            authority.get("snapshot"),
        ),
        check(
            "deployment_status",
            deployment.get("status") == "PASS",
            deployment.get("status"),
        ),
        check(
            "source_verification",
            deployment.get("source_verification", {}).get("status") == "PASS",
            deployment.get("source_verification"),
        ),
        check(
            "dependency_verification",
            deployment.get("dependencies", {}).get("status") == "PASS",
            deployment.get("dependencies"),
        ),
        check(
            "environment_status",
            environment.get("status") == "PASS",
            environment.get("status"),
        ),
        check(
            "remote_root_parity",
            deployment.get("remote_root") == remote_environment.get("remote_root"),
            {
                "deployment": deployment.get("remote_root"),
                "environment": remote_environment.get("remote_root"),
            },
        ),
        check(
            "source_manifest_parity",
            deployment.get("source_manifest_sha256")
            == remote_environment.get("expected", {}).get("source_manifest_sha"),
            {
                "deployment": deployment.get("source_manifest_sha256"),
                "environment": remote_environment.get("expected", {}).get(
                    "source_manifest_sha"
                ),
            },
        ),
        check(
            "overlap_authorized",
            execution.get("allow_existing_compute_overlap") is True,
            execution,
        ),
        check(
            "existing_processes_untouched",
            deployment.get("existing_processes_modified") is False
            and execution.get("kill_existing_processes") is False
            and execution.get("pause_existing_processes") is False
            and execution.get("preempt_existing_processes") is False,
            {"deployment": deployment.get("existing_processes_modified"), **execution},
        ),
    ]
    failed = [item["name"] for item in checks if not item["pass"]]
    return {
        "experiment_id": "E182",
        "gate": "Gate_0",
        "status": "PASS" if not failed else "FAIL",
        "checks": checks,
        "failed_checks": failed,
    }


def load_json(path: Path) -> dict[str, Any]:
    """Load one JSON object."""
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--authority", type=Path, default=DEFAULT_AUTHORITY)
    parser.add_argument("--deployment", type=Path, default=DEFAULT_DEPLOYMENT)
    parser.add_argument("--environment", type=Path, default=DEFAULT_ENVIRONMENT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    """Evaluate and persist Gate 0."""
    args = parse_args()
    payload = evaluate_gate(
        load_json(args.authority),
        load_json(args.deployment),
        load_json(args.environment),
    )
    payload["inputs"] = {
        "authority": {
            "path": relative_to_repo(args.authority),
            "sha256": sha256_file(args.authority),
        },
        "deployment": {
            "path": relative_to_repo(args.deployment),
            "sha256": sha256_file(args.deployment),
        },
        "environment": {
            "path": relative_to_repo(args.environment),
            "sha256": sha256_file(args.environment),
        },
    }
    atomic_json(args.output, payload)
    print(
        f"E182_GATE0={payload['status']} checks={len(payload['checks'])} "
        f"failed={','.join(payload['failed_checks']) or 'none'}"
    )
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
