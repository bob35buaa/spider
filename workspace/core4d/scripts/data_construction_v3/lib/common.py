#!/usr/bin/env python3
"""Shared helpers for Core4D data-construction v3 wrappers."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_ROOT = next(parent for parent in Path(__file__).resolve().parents if parent.name == "data_construction_v3")
for _path in (SCRIPT_ROOT / "lib", SCRIPT_ROOT / "state", SCRIPT_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
from typing import Any, Iterable


SCHEMA_VERSION = "core4d_data_construction_v3.0"
DEFAULT_HOLOSOMA_REPO = Path.home() / "Workspace/holosoma"
DEFAULT_HAND_COLLISION_VARIANT_ID = "sphere5cm"
HAND_COLLISION_VARIANTS = {"sphere5cm", "rubber_hull"}
HAND_COLLISION_VARIANT_SPECS = {
    "sphere5cm": {
        "description": "Current SPIDER default: one 5cm sphere per G1 wrist-yaw hand geom.",
        "patch_mode": "noop",
    },
    "rubber_hull": {
        "description": "Rubber hand visual mesh reused as a convex collision hull.",
        "patch_mode": "mesh_convex_hull",
        "left_mesh": "left_rubber_hand",
        "right_mesh": "right_rubber_hand",
        "left_pos": "0.0415 0.003 0",
        "right_pos": "0.0415 -0.003 0",
        "maxhullvert": "64",
    },
}
SCRIPT_RELATIVE_PATHS = {
    "common.py": "lib/common.py",
    "geometry.py": "lib/geometry.py",
    "interfaces.py": "lib/interfaces.py",
    "init_workspace.sh": "orchestration/init_workspace.sh",
    "run_pipeline.py": "orchestration/run_pipeline.py",
    "run_release_checks.sh": "orchestration/run_release_checks.sh",
    "audit_pipeline_release.py": "qa/audit_pipeline_release.py",
    "run_smoke_suite.py": "qa/run_smoke_suite.py",
    "verify_reproducibility.py": "qa/verify_reproducibility.py",
    "check_environment.py": "stages/s0_environment/check_environment.py",
    "build_fingertip_route_diagnostics.py": "stages/s1_raw_contact/build_fingertip_route_diagnostics.py",
    "build_inventory.py": "stages/s1_raw_contact/build_inventory.py",
    "run_raw_contact.py": "stages/s1_raw_contact/run_raw_contact.py",
    "build_or_audit_templates.py": "stages/s2_templates/build_or_audit_templates.py",
    "render_template_review_package.py": "stages/s2_templates/render_template_review_package.py",
    "render_template_mesh_collision_review_package.py": "stages/s2_templates/render_template_mesh_collision_review_package.py",
    "register_retarget_variant.py": "stages/s3_retarget/register_retarget_variant.py",
    "run_stage2b.py": "stages/s3_retarget/run_stage2b.py",
    "make_visual_qc.py": "stages/s4_gate_visual_qc/make_visual_qc.py",
    "render_visual_qc_package.py": "stages/s4_gate_visual_qc/render_visual_qc_package.py",
    "run_target_gate.py": "stages/s4_gate_visual_qc/run_target_gate.py",
    "patch_hand_collision.py": "stages/s5_handoff/patch_hand_collision.py",
    "export_cem_overrides.py": "stages/s5_handoff/export_cem_overrides.py",
    "export_handoff.py": "stages/s5_handoff/export_handoff.py",
    "export_rl_inputs.py": "stages/s6_downstream/export_rl_inputs.py",
    "record_downstream_evidence.py": "stages/s6_downstream/record_downstream_evidence.py",
    "evaluate_contact_alignment.py": "stages/s6_downstream/evaluate_contact_alignment.py",
    "import_legacy_snapshot.py": "migration/import_legacy_snapshot.py",
    "build_existing_cases_seed.py": "migration/build_existing_cases_seed.py",
    "update_case_state_registry.py": "state/update_case_state_registry.py",
    "write_manual_seed_template.py": "state/write_manual_seed_template.py",
}


def script_path(script_root: Path, script_name: str) -> Path:
    rel = SCRIPT_RELATIVE_PATHS.get(script_name, script_name)
    return script_root / rel


def timestamp() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def find_spider_repo(start: Path | None = None) -> Path:
    here = (start or Path.cwd()).resolve()
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=here,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return Path(out).resolve()
    except Exception:
        for parent in [here, *here.parents]:
            if (parent / ".git").exists():
                return parent
    return here


def resolve_holosoma_repo(value: str | None = None) -> Path:
    raw = value or os.environ.get("HOLOSOMA_REPO") or str(DEFAULT_HOLOSOMA_REPO)
    return Path(raw).expanduser().resolve()


def resolve_run_root(holosoma_repo: Path, value: str | None = None) -> Path:
    raw = value or os.environ.get("DATA_CONSTRUCTION_RUN_ROOT")
    if raw:
        return Path(raw).expanduser().resolve()
    return (holosoma_repo / "workspace/v3/data_construction_v3_runs").resolve()


def git_state(repo: Path) -> dict[str, Any]:
    state: dict[str, Any] = {
        "repo_path": str(repo),
        "exists": repo.exists(),
        "is_git_repo": False,
        "git_sha": "",
        "git_dirty": "",
        "git_status_porcelain": "",
    }
    if not repo.exists():
        return state
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True, stderr=subprocess.DEVNULL
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=repo, text=True, stderr=subprocess.DEVNULL
        )
        state.update(
            {
                "is_git_repo": True,
                "git_sha": sha,
                "git_dirty": str(bool(status.strip())).lower(),
                "git_status_porcelain": status,
            }
        )
    except Exception as exc:
        state["git_error"] = str(exc)
    return state


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_dumps(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json_dumps(data) + "\n", encoding="utf-8")


def write_tsv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def command_exists(name: str) -> bool:
    for item in os.environ.get("PATH", "").split(os.pathsep):
        path = Path(item) / name
        if path.is_file() and os.access(path, os.X_OK):
            return True
    return False


def package_version(module_name: str) -> dict[str, str]:
    try:
        module = __import__(module_name)
        return {"available": "true", "version": str(getattr(module, "__version__", "unknown"))}
    except Exception as exc:
        return {"available": "false", "version": "", "error": str(exc)}


def run_id_from_time(prefix: str = "dcv3") -> str:
    return datetime.now().strftime(f"{prefix}_%Y%m%d_%H%M%S")


def stable_params_json(params: dict[str, Any]) -> str:
    return json.dumps(params, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def print_path(path: Path) -> None:
    sys.stdout.write(str(path) + "\n")
