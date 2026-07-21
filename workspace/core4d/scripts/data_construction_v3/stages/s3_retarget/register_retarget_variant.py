#!/usr/bin/env python3
"""Create or update the retarget variant registry."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
import sys

SCRIPT_ROOT = next(parent for parent in Path(__file__).resolve().parents if parent.name == "data_construction_v3")
for _path in (SCRIPT_ROOT / "lib", SCRIPT_ROOT / "state", SCRIPT_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from common import (
    SCHEMA_VERSION,
    find_spider_repo,
    git_state,
    read_tsv,
    resolve_holosoma_repo,
    stable_params_json,
    timestamp,
    write_json,
    write_tsv,
)


FIELDS = [
    "retarget_variant_id",
    "display_name",
    "solver_family",
    "solver_version",
    "solver_repo_path",
    "solver_git_sha",
    "solver_dirty",
    "converter_script",
    "converter_git_sha",
    "params_json",
    "input_rewrite_policy",
    "branch_from_stage",
    "target_route_contract",
    "created_at",
    "schema_version",
    "notes",
]

OMNIRT_ORIGINAL_PUBLIC_SHA = "9c238cf80f531c0e65d818348c3c1a5cc2764f5b"
OMNIRT_V1_IMPROVEMENTS_SHA = "805376bc291ba3310c5f4e36009c5ab64cb8d454"


def git_ref_available(repo: Path, ref: str) -> bool:
    try:
        subprocess.check_call(
            ["git", "cat-file", "-e", f"{ref}^{{commit}}"],
            cwd=repo,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except Exception:
        return False


def default_variants(spider_repo: Path, holosoma_repo: Path) -> list[dict[str, str]]:
    spider_git = git_state(spider_repo)
    holosoma_git = git_state(holosoma_repo)
    converter = holosoma_repo / "workspace/pipeline/convert_core4d_to_omniretarget.py"
    now = timestamp()
    original_available = git_ref_available(holosoma_repo, OMNIRT_ORIGINAL_PUBLIC_SHA)
    common = {
        "solver_family": "omniretarget",
        "solver_repo_path": str(holosoma_repo),
        "solver_git_sha": holosoma_git.get("git_sha", ""),
        "solver_dirty": holosoma_git.get("git_dirty", ""),
        "converter_script": str(converter),
        "converter_git_sha": holosoma_git.get("git_sha", ""),
        "branch_from_stage": "conversion",
        "created_at": now,
        "schema_version": SCHEMA_VERSION,
    }
    return [
        {
            **common,
            "retarget_variant_id": "omnirt_original",
            "display_name": "原始 OmniRetarget",
            "solver_version": "initial_public_release",
            "solver_repo_path": str(holosoma_repo) if original_available else "REQUIRES_ORIGINAL_CHECKOUT",
            "solver_git_sha": OMNIRT_ORIGINAL_PUBLIC_SHA if original_available else "",
            "solver_dirty": "false",
            "converter_git_sha": "",
            "params_json": stable_params_json(
                {
                    "replace_wrist_with_fingertip": False,
                    "include_fingertip_centers": False,
                    "original_public_release_sha": OMNIRT_ORIGINAL_PUBLIC_SHA,
                    "v1_improvements_sha": OMNIRT_V1_IMPROVEMENTS_SHA,
                    "requires_solver_checkout": True,
                    "solver_ref_available": original_available,
                    "core4d_converter_available_at_ref": False,
                    "target_variant_id_default": "ref_fk",
                }
            ),
            "input_rewrite_policy": "wrist",
            "target_route_contract": "selected_at_stage2b",
            "notes": "已根据本地 git 历史 pin 到 Initial public release；该提交没有当前 CORE4D converter，执行前必须单独 checkout 并实现/指定 original Stage2b adapter。",
        },
        {
            **common,
            "retarget_variant_id": "omnirt_v1",
            "display_name": "OmniRetarget v1 默认 ref_fk",
            "solver_version": "v1_current",
            "params_json": stable_params_json(
                {
                    "replace_wrist_with_fingertip": False,
                    "include_fingertip_centers": False,
                    "target_variant_id_default": "ref_fk",
                    "spider_repo_sha": spider_git.get("git_sha", ""),
                }
            ),
            "input_rewrite_policy": "wrist",
            "target_route_contract": "selected_at_stage2b",
            "notes": "medium box 默认 retarget variant；target route 默认由 S3 选择 ref_fk，不强制执行 E099-E101。",
        },
        {
            **common,
            "retarget_variant_id": "omnirt_v1_fingertip_replacement",
            "display_name": "OmniRetarget v1 + fingertip replacement",
            "solver_version": "v1_current",
            "params_json": stable_params_json(
                {
                    "replace_wrist_with_fingertip": True,
                    "include_fingertip_centers": True,
                    "target_variant_id_default": "ref_fk",
                    "spider_repo_sha": spider_git.get("git_sha", ""),
                }
            ),
            "input_rewrite_policy": "fingertip_mean_replaces_wrist",
            "target_route_contract": "selected_at_stage2b",
            "notes": "显式 input rewrite/retarget 分支；target route 仍由 S3 选择。只有选择 fingertip_aware target 时才要求 E099-E101。",
        },
        {
            **common,
            "retarget_variant_id": "omnirt_v2",
            "display_name": "OmniRetarget v2 Phase4 rescue",
            "solver_version": "v2_phase4_current",
            "params_json": stable_params_json(
                {
                    "replace_wrist_with_fingertip": False,
                    "include_fingertip_centers": False,
                    "enable_constraint_relaxation": True,
                    "enable_foot_z_constraint": True,
                    "foot_slide_penalty_weight": 1.0,
                    "enable_contact_preservation": True,
                    "object_penetration_tolerance_scale": 0.8,
                    "target_variant_id_default": "ref_fk",
                    "spider_repo_sha": spider_git.get("git_sha", ""),
                }
            ),
            "input_rewrite_policy": "wrist",
            "target_route_contract": "selected_at_stage2b",
            "notes": "E168 rescue-only variant: Phase4 flags enabled, fingertip replacement disabled. Do not use before an explicit omnirt_v1 infeasible result.",
        },
    ]


def merge_rows(existing: list[dict[str, str]], incoming: list[dict[str, str]]) -> list[dict[str, str]]:
    merged = {row["retarget_variant_id"]: row for row in existing if row.get("retarget_variant_id")}
    for row in incoming:
        merged[row["retarget_variant_id"]] = row
    return [merged[key] for key in sorted(merged)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry-dir", type=Path, required=True)
    parser.add_argument("--spider-repo", type=Path, default=None)
    parser.add_argument("--holosoma-repo", type=Path, default=None)
    parser.add_argument("--init-defaults", action="store_true")
    parser.add_argument("--variant-json", type=Path, default=None)
    args = parser.parse_args()

    spider_repo = (args.spider_repo or find_spider_repo()).resolve()
    holosoma_repo = (args.holosoma_repo or resolve_holosoma_repo()).resolve()
    registry_dir = args.registry_dir.resolve()
    tsv_path = registry_dir / "retarget_variant_registry.tsv"
    existing = read_tsv(tsv_path) if tsv_path.is_file() else []
    incoming: list[dict[str, str]] = []
    if args.init_defaults:
        incoming.extend(default_variants(spider_repo, holosoma_repo))
    if args.variant_json:
        data = json.loads(args.variant_json.read_text(encoding="utf-8"))
        incoming.append({field: str(data.get(field, "")) for field in FIELDS})

    rows = merge_rows(existing, incoming)
    write_tsv(tsv_path, rows, FIELDS)
    write_json(registry_dir / "retarget_variant_registry.json", rows)
    print(f"wrote {tsv_path} rows={len(rows)}")


if __name__ == "__main__":
    main()
