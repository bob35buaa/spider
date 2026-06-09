#!/usr/bin/env python3
"""Audit codebase-level release invariants for Core4D data-construction v3."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
import sys

SCRIPT_ROOT = next(parent for parent in Path(__file__).resolve().parents if parent.name == "data_construction_v3")
for _path in (SCRIPT_ROOT / "lib", SCRIPT_ROOT / "state", SCRIPT_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
from typing import Any

from common import SCHEMA_VERSION, SCRIPT_RELATIVE_PATHS, find_spider_repo, json_dumps, timestamp, write_json


SCRIPT_DIR = SCRIPT_ROOT
DOC_DIR = SCRIPT_DIR.parents[1] / "docs/data_construction_v3"
PLAN_PATH = SCRIPT_DIR.parents[1] / "plan/116_data_construction_v3_reproducible_pipeline_plan.md"


REQUIRED_SCRIPTS = [
    "check_environment.py",
    "init_workspace.sh",
    "run_pipeline.py",
    "verify_reproducibility.py",
    "run_smoke_suite.py",
    "run_release_checks.sh",
    "build_inventory.py",
    "run_raw_contact.py",
    "build_or_audit_templates.py",
    "render_template_review_package.py",
    "register_retarget_variant.py",
    "build_fingertip_route_diagnostics.py",
    "run_stage2b.py",
    "run_target_gate.py",
    "render_visual_qc_package.py",
    "make_visual_qc.py",
    "patch_hand_collision.py",
    "export_handoff.py",
    "export_cem_overrides.py",
    "export_rl_inputs.py",
    "record_downstream_evidence.py",
    "update_case_state_registry.py",
    "import_legacy_snapshot.py",
    "build_existing_cases_seed.py",
    "write_manual_seed_template.py",
    "geometry.py",
    "interfaces.py",
]


REQUIRED_DOCS = [
    "README.md",
    "00_environment.md",
    "01_data_layout.md",
    "02_pipeline_stages.md",
    "03_manifest_schema.md",
    "04_scene_template_policy.md",
    "05_reproducibility.md",
    "06_failure_taxonomy.md",
    "07_troubleshooting.md",
    "08_retarget_variants.md",
    "09_extension_interfaces.md",
    "10_diagnostic_contracts.md",
    "11_legacy_migration.md",
    "12_completion_audit.md",
    "13_requirements_traceability.md",
    "14_release_readiness.md",
    "15_hand_collision_variants.md",
]


REQUIRED_SEEDS = [
    "workspace/core4d/data_construction_v3/README.md",
    "workspace/core4d/data_construction_v3/existing_cases.tsv",
    "workspace/core4d/data_construction_v3/existing_cases_summary.md",
]


CHECKS: list[dict[str, str]] = [
    {
        "id": "gitignore_results",
        "path": ".gitignore",
        "pattern": r"results/.*workspace/v3/data_construction_v3_runs/",
        "description": "results and v3 run roots are git-ignored",
    },
    {
        "id": "pipeline_default_variants",
        "path": "workspace/core4d/scripts/data_construction_v3/orchestration/run_pipeline.py",
        "pattern": r"args\.retarget_variant_id = dedupe\(args\.retarget_variant_id or \[\"omnirt_v1\"\]\).*args\.target_variant_id = dedupe\(args\.target_variant_id or \[\"ref_fk\"\]\)",
        "description": "default route is omnirt_v1/ref_fk",
    },
    {
        "id": "pipeline_exports_cem_overrides",
        "path": "workspace/core4d/scripts/data_construction_v3/orchestration/run_pipeline.py",
        "pattern": r"export_cem_overrides\.py.*s5_handoff/cem_overrides",
        "description": "S5 handoff exports CEM override configs",
    },
    {
        "id": "stage2b_rejects_non_ref_fk_execute",
        "path": "workspace/core4d/scripts/data_construction_v3/stages/s3_retarget/run_stage2b.py",
        "pattern": r"args\.execute and args\.target_variant_id != \"ref_fk\".*non-ref_fk target routes need a dedicated Stage2b target adapter",
        "description": "legacy Stage2b execute cannot masquerade as non-ref_fk route",
    },
    {
        "id": "stage2b_fingertip_contract",
        "path": "workspace/core4d/scripts/data_construction_v3/stages/s3_retarget/run_stage2b.py",
        "pattern": r"target_variant_id == \"fingertip_aware\".*fingertip_vote_status.*palm_vote_status.*quat_audit_status.*target_active_mask_status.*e101_route_evidence_status",
        "description": "fingertip-aware route requires E099-E101 diagnostics",
    },
    {
        "id": "registry_fingertip_reject",
        "path": "workspace/core4d/scripts/data_construction_v3/state/update_case_state_registry.py",
        "pattern": r"target_variant_id\"\) == \"fingertip_aware\".*route_diagnostic_status\"\) not in \{\"pass\"\}.*REJECT_FINGERTIP_ROUTE_CONTRACT",
        "description": "registry rejects fingertip-aware rows without route diagnostics",
    },
    {
        "id": "raw_contact_dual_threshold",
        "path": "workspace/core4d/scripts/data_construction_v3/stages/s1_raw_contact/run_raw_contact.py",
        "pattern": r"(?=.*thresholds-m)(?=.*raw_contact_candidates_)(?=.*raw_contact_pass_)",
        "description": "raw contact writes separate 3cm/5cm candidate and pass files",
    },
    {
        "id": "template_missing_backlog",
        "path": "workspace/core4d/scripts/data_construction_v3/stages/s2_templates/build_or_audit_templates.py",
        "pattern": r"template_status\": \"backlog\".*build_box_source_template",
        "description": "source scene missing becomes template backlog, not data reject",
    },
    {
        "id": "template_robot_inertial_audit",
        "path": "workspace/core4d/scripts/data_construction_v3/stages/s2_templates/build_or_audit_templates.py",
        "pattern": r"polluted_robot_inertial.*robot_polluted_mass_29_632",
        "description": "source template audit catches E103 robot inertial pollution",
    },
    {
        "id": "target_gate_robot_inertial_audit",
        "path": "workspace/core4d/scripts/data_construction_v3/stages/s4_gate_visual_qc/run_target_gate.py",
        "pattern": r"scene_robot_polluted_mass_29_632.*scene_act_robot_polluted_mass_29_632",
        "description": "target gate rechecks source/act scene robot inertial pollution",
    },
    {
        "id": "retarget_original_pinned",
        "path": "workspace/core4d/scripts/data_construction_v3/stages/s3_retarget/register_retarget_variant.py",
        "pattern": r"OMNIRT_ORIGINAL_PUBLIC_SHA.*initial_public_release.*requires_solver_checkout",
        "description": "omnirt_original is pinned and cannot execute through current v1 checkout",
    },
    {
        "id": "cem_override_external_validation",
        "path": "workspace/core4d/scripts/data_construction_v3/stages/s5_handoff/export_cem_overrides.py",
        "pattern": r"spider_contact_target_object_local.*eval_contact_target_object_local.*contact_target_sha256.*contact_hdmi_target_source: external",
        "description": "external target CEM overrides validate target NPZ and sha256",
    },
    {
        "id": "legacy_python_env_guard",
        "path": "workspace/core4d/data_preprocess/pipeline.sh",
        "pattern": r"RETARGET_PYTHON_BIN.*CONDA_PREFIX.*retarget_python",
        "description": "legacy Stage2b wrapper uses retargeting Python instead of SPIDER venv python",
    },
    {
        "id": "docs_e098_e101_contracts",
        "path": "workspace/core4d/docs/data_construction_v3/10_diagnostic_contracts.md",
        "pattern": r"E098 是全路线基础 contract.*E099-E101 是 `target_variant_id=fingertip_aware`",
        "description": "docs distinguish global E098 from fingertip-aware E099-E101 contracts",
    },
    {
        "id": "script_structure_documented",
        "path": "workspace/core4d/scripts/data_construction_v3/README.md",
        "pattern": r"lib/.*orchestration/.*stages/s1_raw_contact/.*state/.*qa/",
        "description": "script implementation is organized by functional directories",
    },
]


def read_text(repo: Path, rel: str, errors: list[str]) -> str:
    path = repo / rel
    if not path.is_file():
        errors.append(f"missing file: {rel}")
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def compact(text: str) -> str:
    return re.sub(r"\s+", " ", text)


def check_required_files(repo: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name in REQUIRED_SCRIPTS:
        path = SCRIPT_DIR / SCRIPT_RELATIVE_PATHS.get(name, name)
        rows.append(
            {
                "check_id": f"script_exists:{name}",
                "status": "pass" if path.is_file() else "fail",
                "description": "required v3 script exists",
                "path": str(path.relative_to(repo)) if path.exists() else str(path),
                "notes": "",
            }
        )
    for name in REQUIRED_DOCS:
        path = DOC_DIR / name
        rows.append(
            {
                "check_id": f"doc_exists:{name}",
                "status": "pass" if path.is_file() else "fail",
                "description": "required v3 document exists",
                "path": str(path.relative_to(repo)) if path.exists() else str(path),
                "notes": "",
            }
        )
    for rel in REQUIRED_SEEDS:
        path = repo / rel
        rows.append(
            {
                "check_id": f"seed_exists:{rel}",
                "status": "pass" if path.is_file() else "fail",
                "description": "required git-trackable v3 seed exists",
                "path": rel,
                "notes": "",
            }
        )
    rows.append(
        {
            "check_id": "plan_exists",
            "status": "pass" if PLAN_PATH.is_file() else "fail",
            "description": "release plan exists",
            "path": str(PLAN_PATH.relative_to(repo)) if PLAN_PATH.exists() else str(PLAN_PATH),
            "notes": "",
        }
    )
    root_symlinks = sorted(path.name for path in SCRIPT_DIR.iterdir() if path.is_symlink())
    root_flat_scripts = sorted(
        path.name for path in SCRIPT_DIR.iterdir() if path.is_file() and path.suffix in {".py", ".sh"}
    )
    rows.append(
        {
            "check_id": "script_root_has_no_symlinks",
            "status": "pass" if not root_symlinks else "fail",
            "description": "v3 script root does not keep compatibility symlinks",
            "path": str(SCRIPT_DIR.relative_to(repo)),
            "notes": "" if not root_symlinks else ",".join(root_symlinks),
        }
    )
    rows.append(
        {
            "check_id": "script_root_has_no_flat_scripts",
            "status": "pass" if not root_flat_scripts else "fail",
            "description": "v3 script root does not keep flat .py/.sh script entries",
            "path": str(SCRIPT_DIR.relative_to(repo)),
            "notes": "" if not root_flat_scripts else ",".join(root_flat_scripts),
        }
    )
    return rows


def check_regexes(repo: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for item in CHECKS:
        rel = item["path"]
        text = compact(read_text(repo, rel, errors))
        ok = bool(text and re.search(item["pattern"], text))
        rows.append(
            {
                "check_id": item["id"],
                "status": "pass" if ok else "fail",
                "description": item["description"],
                "path": rel,
                "notes": "" if ok else "pattern not found",
            }
        )
    for error in errors:
        rows.append(
            {
                "check_id": "file_read",
                "status": "fail",
                "description": "required file readable",
                "path": "",
                "notes": error,
            }
        )
    return rows


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Core4D data-construction v3 release audit",
        "",
        f"- status: `{report['status']}`",
        f"- schema_version: `{report['schema_version']}`",
        f"- created_at: `{report['created_at']}`",
        f"- passed: `{report['passed']}`",
        f"- failed: `{report['failed']}`",
        "",
        "## Failed Checks",
        "",
        "| check | path | notes |",
        "|---|---|---|",
    ]
    failed = [row for row in report["checks"] if row["status"] != "pass"]
    if failed:
        for row in failed:
            lines.append(f"| `{row['check_id']}` | `{row['path']}` | {row['notes']} |")
    else:
        lines.append("| - | - | - |")
    lines.extend(["", "## All Checks", "", "| check | status | description |", "|---|---|---|"])
    for row in report["checks"]:
        lines.append(f"| `{row['check_id']}` | `{row['status']}` | {row['description']} |")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spider-repo", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    repo = (args.spider_repo or find_spider_repo()).resolve()
    checks = check_required_files(repo) + check_regexes(repo)
    failed = [row for row in checks if row["status"] != "pass"]
    report = {
        "stage": "audit_pipeline_release",
        "created_at": timestamp(),
        "schema_version": SCHEMA_VERSION,
        "status": "fail" if failed else "pass",
        "passed": len(checks) - len(failed),
        "failed": len(failed),
        "checks": checks,
    }
    if args.out_dir:
        out_dir = args.out_dir.expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        write_json(out_dir / "pipeline_release_audit.json", report)
        (out_dir / "pipeline_release_audit.md").write_text(markdown_report(report), encoding="utf-8")
    print(json_dumps(report))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
