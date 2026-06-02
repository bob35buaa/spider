#!/usr/bin/env python3
"""Run a compact Core4D data-construction v3 smoke suite."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

SCRIPT_ROOT = next(parent for parent in Path(__file__).resolve().parents if parent.name == "data_construction_v3")
for _path in (SCRIPT_ROOT / "lib", SCRIPT_ROOT / "state", SCRIPT_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
from typing import Any

import numpy as np

from common import (
    SCHEMA_VERSION,
    find_spider_repo,
    json_dumps,
    resolve_holosoma_repo,
    script_path,
    timestamp,
    write_json,
)
from update_case_state_registry import FIELDS as REGISTRY_FIELDS


SCRIPT_DIR = SCRIPT_ROOT


def py(script: str) -> str:
    return str(script_path(SCRIPT_DIR, script))


def run_step(name: str, cmd: list[str], cwd: Path, log_dir: Path) -> dict[str, Any]:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{name}.log"
    started = timestamp()
    with log_path.open("w", encoding="utf-8") as log:
        log.write("+ " + " ".join(str(x) for x in cmd) + "\n")
        completed = subprocess.run(cmd, cwd=cwd, stdout=log, stderr=subprocess.STDOUT, text=True, check=False)
    return {
        "name": name,
        "command": cmd,
        "started_at": started,
        "ended_at": timestamp(),
        "returncode": completed.returncode,
        "log": str(log_path),
        "status": "pass" if completed.returncode == 0 else "fail",
    }


def require_pass(step: dict[str, Any]) -> None:
    if step["returncode"] != 0:
        raise RuntimeError(f"smoke step failed: {step['name']} log={step['log']}")


def write_visual_fixture(root: Path) -> Path:
    evidence = root / "visual_fixture/evidence"
    evidence.mkdir(parents=True, exist_ok=True)
    scene = evidence / "scene.xml"
    scene.write_text(
        """<mujoco model="visual_render_smoke">
  <worldbody>
    <body name="object" pos="0 0 0.25">
      <freejoint/>
      <geom type="box" size="0.12 0.10 0.08" rgba="0.2 0.5 0.9 1"/>
    </body>
  </worldbody>
</mujoco>
""",
        encoding="utf-8",
    )
    qpos = np.asarray([[-0.2 + 0.04 * i, 0.0, 0.25, 1.0, 0.0, 0.0, 0.0] for i in range(12)], dtype=float)
    trajectory = evidence / "trajectory_kinematic.npz"
    np.savez(trajectory, qpos=qpos)
    manifest = root / "visual_fixture/target_gate_manifest.tsv"
    row = {
        "stage": "S4_target_gate",
        "case_id": "smoke_visual_001",
        "retarget_variant_id": "omnirt_v1",
        "target_variant_id": "ref_fk",
        "target_task": "smoke_task",
        "object_key": "box999",
        "object_name": "box999",
        "date": "20260101",
        "seq": "001",
        "person": "person1",
        "person_idx": "0",
        "stage2b_decision": "stage2b_ready",
        "target_gate_status": "pass",
        "visual_qc_status": "not_run",
        "failure_mode": "",
        "decision_notes": "machine gate pass",
        "target_scene": str(scene),
        "scene_act": "",
        "trajectory": str(trajectory),
        "schema_version": SCHEMA_VERSION,
        "updated_at": timestamp(),
    }
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerow(row)
    return manifest


def write_template_visual_fixture(root: Path) -> Path:
    evidence = root / "template_visual_fixture/evidence"
    evidence.mkdir(parents=True, exist_ok=True)
    scene = evidence / "scene.xml"
    scene.write_text(
        """<mujoco model="template_visual_smoke">
  <worldbody>
    <body name="object" pos="0 0 0.25">
      <freejoint/>
      <geom name="object_collision" type="box" size="0.12 0.10 0.08" rgba="0.2 0.5 0.9 1"/>
    </body>
  </worldbody>
</mujoco>
""",
        encoding="utf-8",
    )
    manifest = root / "template_visual_fixture/template_backlog.tsv"
    row = {
        "source_scene_task": "smoke_template_box999_person1",
        "object_key": "box999",
        "object_name": "box999",
        "object_category": "box",
        "person": "person1",
        "required_by_count": "1",
        "required_by_cases": "smoke_template_case",
        "scene_xml": str(scene),
        "scene_exists": "True",
        "template_status": "clean",
        "recommended_action": "use_existing_clean_template",
        "schema_version": SCHEMA_VERSION,
        "updated_at": timestamp(),
    }
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerow(row)
    return manifest


def write_cem_override_fixture(root: Path) -> Path:
    fixture = root / "cem_override_fixture"
    target_dir = fixture / "targets/smoke_external"
    target_dir.mkdir(parents=True, exist_ok=True)
    target_npz = target_dir / "spider_contact_target_object_local.npz"
    target = np.zeros((8, 2, 3), dtype=np.float32)
    active = np.ones((8, 2), dtype=np.bool_)
    np.savez(target_npz, spider_contact_target_object_local=target, active=active)
    manifest = fixture / "handoff_manifest.tsv"
    rows = [
        {
            "case_id": "smoke_cem_ref_fk",
            "object_key": "box004",
            "object_name": "box004",
            "date": "20260101",
            "seq": "001",
            "person": "person1",
            "person_idx": "0",
            "retarget_variant_id": "omnirt_v1",
            "target_variant_id": "ref_fk",
            "candidate_decision": "REVIEW",
            "handoff_decision": "HANDOFF_REVIEW_VISUAL_QC",
            "target_gate_status": "pass",
            "visual_qc_status": "review",
            "stage2b_target_task": "smoke_ref_fk_task",
            "target_npz": "",
            "schema_version": SCHEMA_VERSION,
            "updated_at": timestamp(),
        },
        {
            "case_id": "smoke_cem_fingertip",
            "object_key": "box004",
            "object_name": "box004",
            "date": "20260101",
            "seq": "002",
            "person": "person1",
            "person_idx": "0",
            "retarget_variant_id": "omnirt_v1",
            "target_variant_id": "fingertip_aware",
            "candidate_decision": "REVIEW",
            "handoff_decision": "HANDOFF_REVIEW_VISUAL_QC",
            "target_gate_status": "pass",
            "visual_qc_status": "review",
            "stage2b_target_task": "smoke_fingertip_task",
            "target_npz": str(target_npz),
            "schema_version": SCHEMA_VERSION,
            "updated_at": timestamp(),
        },
    ]
    fields = list(rows[0].keys())
    with manifest.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return manifest


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Core4D data-construction v3 smoke suite",
        "",
        f"- status: `{report['status']}`",
        f"- run_root: `{report['run_root']}`",
        f"- created_at: `{report['created_at']}`",
        "",
        "## steps",
        "",
        "| step | status | log |",
        "|---|---|---|",
    ]
    for step in report["steps"]:
        lines.append(f"| `{step['name']}` | `{step['status']}` | `{step['log']}` |")
    if report["warnings"]:
        lines.extend(["", "## warnings", ""])
        for warning in report["warnings"]:
            lines.append(f"- {warning}")
    lines.append("")
    return "\n".join(lines)


def first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path and path.exists():
            return path
    return None


def assert_visual_render_pass(path: Path) -> None:
    summary = json.loads(path.read_text(encoding="utf-8"))
    rendered = int(summary.get("rendered_rows", 0))
    pass_count = int(summary.get("render_status_counts", {}).get("pass", 0))
    if rendered < 1 or pass_count < 1:
        raise RuntimeError(f"visual render smoke produced no pass rows: {path}")


def assert_template_visual_pass(path: Path) -> None:
    summary = json.loads(path.read_text(encoding="utf-8"))
    rendered = int(summary.get("rendered_rows", 0))
    pass_count = int(summary.get("render_status_counts", {}).get("pass", 0))
    if rendered < 1 or pass_count < 1:
        raise RuntimeError(f"template visual smoke produced no pass rows: {path}")


def assert_cem_override_pass(path: Path) -> None:
    summary = json.loads(path.read_text(encoding="utf-8"))
    pass_count = int(summary.get("override_status_counts", {}).get("pass", 0))
    if pass_count < 2:
        raise RuntimeError(f"cem override smoke expected at least 2 pass rows: {path}")


def assert_retarget_variant_registry(path: Path) -> None:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = {row["retarget_variant_id"]: row for row in csv.DictReader(f, delimiter="\t")}
    for variant_id in ["omnirt_original", "omnirt_v1", "omnirt_v1_fingertip_replacement"]:
        if variant_id not in rows:
            raise RuntimeError(f"missing default retarget variant: {variant_id}")
    original = rows["omnirt_original"]
    if original.get("solver_repo_path") in {"", "REQUIRES_CONFIRMATION"}:
        raise RuntimeError("omnirt_original still lacks an exact source reference")
    params = json.loads(original.get("params_json", "{}"))
    if not params.get("requires_solver_checkout") or not params.get("original_public_release_sha"):
        raise RuntimeError("omnirt_original must be pinned and marked as requiring a separate checkout/adapter")


def write_state_matrix_fixture(root: Path) -> tuple[Path, Path, Path]:
    matrix = root / "state_matrix"
    evidence_root = matrix / "evidence"
    evidence_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    cases = [
        {
            "case_id": "smoke_raw_contact_fail",
            "raw_contact_3cm_status": "reject",
            "raw_contact_5cm_status": "reject",
            "template_status": "not_run",
            "retarget_variant_id": "shared",
            "stage2b_status": "not_run",
            "target_gate_status": "not_run",
            "visual_qc_status": "not_run",
            "cem_status": "not_run",
            "rl_status": "not_run",
        },
        {
            "case_id": "smoke_stage2b_pass_cem_not_run",
            "raw_contact_3cm_status": "pass",
            "raw_contact_5cm_status": "pass",
            "template_status": "clean",
            "retarget_variant_id": "omnirt_v1",
            "stage2b_status": "pass",
            "target_gate_status": "not_run",
            "visual_qc_status": "not_run",
            "cem_status": "not_run",
            "rl_status": "not_run",
        },
        {
            "case_id": "smoke_cem_pass_rl_not_run",
            "raw_contact_3cm_status": "pass",
            "raw_contact_5cm_status": "pass",
            "template_status": "clean",
            "retarget_variant_id": "omnirt_v1",
            "stage2b_status": "pass",
            "target_gate_status": "pass",
            "visual_qc_status": "pass",
            "cem_status": "pass",
            "rl_status": "not_run",
        },
        {
            "case_id": "smoke_full_chain_pass",
            "raw_contact_3cm_status": "pass",
            "raw_contact_5cm_status": "pass",
            "template_status": "clean",
            "retarget_variant_id": "omnirt_v1",
            "stage2b_status": "pass",
            "target_gate_status": "pass",
            "visual_qc_status": "pass",
            "cem_status": "pass",
            "rl_status": "pass",
        },
    ]
    for idx, item in enumerate(cases, start=1):
        case_evidence = evidence_root / item["case_id"]
        case_evidence.mkdir(parents=True, exist_ok=True)
        row = {field: "" for field in REGISTRY_FIELDS}
        row.update(
            {
                "object_key": "box004",
                "object_name": "box004",
                "date": "20260101",
                "seq": f"{idx:03d}",
                "person": "person1",
                "person_idx": "0",
                "raw_inventory_status": "present",
                "target_variant_id": "ref_fk",
                "diagnostic_contracts": "E098_global",
                "evidence_root": str(case_evidence),
                "source_type": "manual_seed",
                "source_ref": "run_smoke_suite_state_matrix",
                "schema_version": SCHEMA_VERSION,
                "updated_at": timestamp(),
                "notes": "state matrix smoke row",
            }
        )
        row.update(item)
        rows.append(row)
    seed_tsv = matrix / "state_matrix_seed.tsv"
    with seed_tsv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REGISTRY_FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    evidence_tsv = matrix / "state_matrix_downstream_evidence.tsv"
    evidence_fields = ["case_id", "retarget_variant_id", "target_variant_id", "cem_status", "rl_status", "cem_run_id", "rl_run_id"]
    evidence_rows = [
        {
            "case_id": "smoke_cem_pass_rl_not_run",
            "retarget_variant_id": "omnirt_v1",
            "target_variant_id": "ref_fk",
            "cem_status": "pass",
            "rl_status": "not_run",
            "cem_run_id": "smoke_cem_pass",
            "rl_run_id": "",
        },
        {
            "case_id": "smoke_full_chain_pass",
            "retarget_variant_id": "omnirt_v1",
            "target_variant_id": "ref_fk",
            "cem_status": "pass",
            "rl_status": "pass",
            "cem_run_id": "smoke_cem_pass",
            "rl_run_id": "smoke_rl_pass",
        },
    ]
    with evidence_tsv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=evidence_fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(evidence_rows)
    return matrix, seed_tsv, evidence_tsv


def assert_state_matrix_pass(registry_path: Path, handoff_summary_path: Path, downstream_summary_path: Path) -> None:
    def read_rows(path: Path) -> list[dict[str, str]]:
        with path.open("r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f, delimiter="\t"))

    registry = {row["case_id"]: row for row in read_rows(registry_path)}
    expected_current = {
        "smoke_raw_contact_fail": "REJECT_RAW_CONTACT",
        "smoke_stage2b_pass_cem_not_run": "STAGE2B_PASS",
        "smoke_cem_pass_rl_not_run": "VISUAL_QC_PASS",
        "smoke_full_chain_pass": "VISUAL_QC_PASS",
    }
    for case_id, expected in expected_current.items():
        actual = registry.get(case_id, {}).get("current_decision", "")
        if actual != expected:
            raise RuntimeError(f"state matrix current_decision mismatch: {case_id} expected={expected} actual={actual}")
    if registry["smoke_cem_pass_rl_not_run"].get("downstream_decision") != "DOWNSTREAM_CEM_PASS":
        raise RuntimeError("state matrix missing DOWNSTREAM_CEM_PASS")
    if registry["smoke_full_chain_pass"].get("downstream_decision") != "DOWNSTREAM_RL_PASS":
        raise RuntimeError("state matrix missing DOWNSTREAM_RL_PASS")

    handoff = json.loads(handoff_summary_path.read_text(encoding="utf-8"))
    downstream = json.loads(downstream_summary_path.read_text(encoding="utf-8"))
    if handoff.get("candidate_decision_counts", {}).get("PASS", 0) < 2:
        raise RuntimeError(f"state matrix handoff did not preserve PASS rows: {handoff_summary_path}")
    expected_downstream = downstream.get("downstream_decision_counts", {})
    if expected_downstream.get("DOWNSTREAM_CEM_PASS", 0) < 1 or expected_downstream.get("DOWNSTREAM_RL_PASS", 0) < 1:
        raise RuntimeError(f"state matrix downstream summary missing pass decisions: {downstream_summary_path}")


def write_bad_resume_registry(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {field: "" for field in REGISTRY_FIELDS}
    row.update(
        {
            "case_id": "smoke_bad_resume_missing_evidence",
            "object_key": "box004",
            "object_name": "box004",
            "date": "20260101",
            "seq": "999",
            "person": "person1",
            "person_idx": "0",
            "raw_inventory_status": "present",
            "raw_contact_3cm_status": "pass",
            "raw_contact_5cm_status": "pass",
            "template_status": "clean",
            "retarget_variant_id": "omnirt_v1",
            "stage2b_status": "pass",
            "target_variant_id": "ref_fk",
            "target_gate_status": "pass",
            "visual_qc_status": "pass",
            "cem_status": "not_run",
            "rl_status": "not_run",
            "current_decision": "VISUAL_QC_PASS",
            "diagnostic_contracts": "E098_global",
            "evidence_root": str(path.parent / "missing_evidence_dir"),
            "source_type": "manual_seed",
            "source_ref": "run_smoke_suite_bad_resume",
            "schema_version": SCHEMA_VERSION,
            "updated_at": timestamp(),
        }
    )
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REGISTRY_FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=Path("/tmp/core4d_dcv3_smoke_suite"))
    parser.add_argument("--core4d-raw-root", type=Path, default=None)
    parser.add_argument("--smplx-model-dir", type=Path, default=None)
    parser.add_argument("--spider-repo", type=Path, default=None)
    parser.add_argument("--holosoma-repo", type=Path, default=None)
    parser.add_argument("--object-keys", default="box004")
    parser.add_argument("--sample-count", type=int, default=300)
    parser.add_argument("--max-sequences", type=int, default=1)
    parser.add_argument("--keep-existing", action="store_true")
    args = parser.parse_args()

    spider_repo = (args.spider_repo or find_spider_repo()).resolve()
    holosoma_repo = (args.holosoma_repo or resolve_holosoma_repo()).resolve()
    raw_root = args.core4d_raw_root or (Path(os.environ["CORE4D_RAW_ROOT"]) if os.environ.get("CORE4D_RAW_ROOT") else None)
    smplx_model_dir = args.smplx_model_dir or (Path(os.environ["SMPLX_MODEL_DIR"]) if os.environ.get("SMPLX_MODEL_DIR") else None)
    run_root = args.run_root.expanduser().resolve()
    if run_root.exists() and not args.keep_existing:
        shutil.rmtree(run_root)
    run_root.mkdir(parents=True, exist_ok=True)
    log_dir = run_root / "logs"

    report: dict[str, Any] = {
        "stage": "run_smoke_suite",
        "created_at": timestamp(),
        "schema_version": SCHEMA_VERSION,
        "status": "pass",
        "run_root": str(run_root),
        "steps": [],
        "warnings": [],
    }

    def add(name: str, cmd: list[str], must_pass: bool = True) -> dict[str, Any]:
        step = run_step(name, cmd, spider_repo, log_dir)
        report["steps"].append(step)
        if must_pass:
            require_pass(step)
        elif step["returncode"] != 0:
            report["warnings"].append(f"{name} failed but was optional; log={step['log']}")
        return step

    def add_expected_fail(name: str, cmd: list[str]) -> dict[str, Any]:
        step = run_step(name, cmd, spider_repo, log_dir)
        if step["returncode"] == 0:
            step["status"] = "fail"
            report["steps"].append(step)
            raise RuntimeError(f"smoke step unexpectedly passed: {name} log={step['log']}")
        step["expected_failure"] = True
        step["expected_returncode"] = step["returncode"]
        step["status"] = "pass"
        report["steps"].append(step)
        return step

    try:
        py_files = sorted(path for path in SCRIPT_DIR.rglob("*.py") if "__pycache__" not in path.parts)
        add("py_compile", [sys.executable, "-m", "py_compile", *[str(path) for path in py_files]])
        add("geometry_self_test", [sys.executable, py("geometry.py")])
        add("interfaces_self_test", [sys.executable, py("interfaces.py")])
        add(
            "pipeline_release_audit",
            [
                sys.executable,
                py("audit_pipeline_release.py"),
                "--spider-repo",
                str(spider_repo),
                "--out-dir",
                str(run_root / "release_audit"),
            ],
        )
        variant_root = run_root / "variant_registry"
        add(
            "retarget_variant_registry",
            [
                sys.executable,
                py("register_retarget_variant.py"),
                "--registry-dir",
                str(variant_root),
                "--spider-repo",
                str(spider_repo),
                "--holosoma-repo",
                str(holosoma_repo),
                "--init-defaults",
            ],
        )
        assert_retarget_variant_registry(variant_root / "retarget_variant_registry.tsv")
        add_expected_fail(
            "execute_rejects_non_ref_fk_target_route",
            [
                sys.executable,
                py("run_stage2b.py"),
                "--raw-contact-tsv",
                str(run_root / "missing_raw_contact.tsv"),
                "--template-backlog-tsv",
                str(run_root / "missing_template_backlog.tsv"),
                "--retarget-variant-registry",
                str(variant_root / "retarget_variant_registry.tsv"),
                "--retarget-variant-id",
                "omnirt_v1",
                "--target-variant-id",
                "fingertip_aware",
                "--core4d-raw-root",
                str(run_root / "missing_core4d_raw"),
                "--smplx-model-dir",
                str(run_root / "missing_smplx"),
                "--out-dir",
                str(run_root / "non_ref_fk_execute_guard"),
                "--execute",
                "--allow-legacy-stage2b-wrapper",
            ],
        )

        manual_root = run_root / "manual_seed"
        evidence = manual_root / "evidence"
        evidence.mkdir(parents=True, exist_ok=True)
        add(
            "manual_seed_template",
            [
                sys.executable,
                py("write_manual_seed_template.py"),
                "--out-tsv",
                str(manual_root / "manual_seed.tsv"),
                "--include-example",
                "--example-evidence-root",
                str(evidence),
                "--write-help-md",
            ],
        )
        add(
            "manual_seed_import",
            [
                sys.executable,
                py("import_legacy_snapshot.py"),
                "--input-tsv",
                str(manual_root / "manual_seed.tsv"),
                "--out-dir",
                str(manual_root / "imported_snapshots"),
                "--snapshot-id",
                "smoke_manual_seed",
                "--source-type",
                "manual_seed",
                "--source-ref",
                "run_smoke_suite",
                "--require-existing-evidence",
            ],
        )

        matrix_root, matrix_seed, matrix_evidence = write_state_matrix_fixture(run_root)
        matrix_registry = matrix_root / "registries"
        add(
            "state_matrix_registry",
            [
                sys.executable,
                py("update_case_state_registry.py"),
                "--registry-dir",
                str(matrix_registry),
                "--input-tsv",
                str(matrix_seed),
                "--source-type",
                "manual_seed",
                "--source-ref",
                "run_smoke_suite_state_matrix",
            ],
        )
        add(
            "state_matrix_handoff",
            [
                sys.executable,
                py("export_handoff.py"),
                "--case-state-registry",
                str(matrix_registry / "case_state_registry.tsv"),
                "--out-dir",
                str(matrix_root / "s5_handoff"),
            ],
        )
        add(
            "state_matrix_downstream",
            [
                sys.executable,
                py("record_downstream_evidence.py"),
                "--handoff-manifest-tsv",
                str(matrix_root / "s5_handoff/handoff_manifest.tsv"),
                "--evidence-tsv",
                str(matrix_evidence),
                "--out-dir",
                str(matrix_root / "s6_downstream"),
                "--evidence-root",
                str(matrix_root / "evidence"),
                "--source-ref",
                "run_smoke_suite_state_matrix",
            ],
        )
        add(
            "state_matrix_registry_downstream",
            [
                sys.executable,
                py("update_case_state_registry.py"),
                "--registry-dir",
                str(matrix_registry),
                "--from-downstream-evidence-tsv",
                str(matrix_root / "s6_downstream/downstream_evidence_manifest.tsv"),
                "--evidence-root",
                str(matrix_root / "s6_downstream"),
                "--source-ref",
                "run_smoke_suite_state_matrix",
            ],
        )
        assert_state_matrix_pass(
            matrix_registry / "case_state_registry.tsv",
            matrix_root / "s5_handoff/handoff_summary.json",
            matrix_root / "s6_downstream/downstream_evidence_summary.json",
        )
        bad_resume = matrix_root / "bad_resume_registry.tsv"
        write_bad_resume_registry(bad_resume)
        add_expected_fail(
            "resume_rejects_missing_evidence",
            [
                sys.executable,
                py("run_pipeline.py"),
                "--mode",
                "resume-from-summary",
                "--run-id",
                "smoke_bad_resume_missing_evidence",
                "--run-root",
                str(run_root),
                "--resume-registry",
                str(bad_resume),
            ],
        )

        visual_manifest = write_visual_fixture(run_root)
        template_manifest = write_template_visual_fixture(run_root)
        cem_override_manifest = write_cem_override_fixture(run_root)
        add(
            "template_visual_render",
            [
                sys.executable,
                py("render_template_review_package.py"),
                "--template-backlog-tsv",
                str(template_manifest),
                "--out-dir",
                str(run_root / "template_visual_fixture/template_visual_review"),
                "--frames",
                "8",
                "--overwrite",
            ],
        )
        assert_template_visual_pass(run_root / "template_visual_fixture/template_visual_review/template_visual_summary.json")
        add(
            "visual_qc_render",
            [
                sys.executable,
                py("render_visual_qc_package.py"),
                "--target-gate-manifest-tsv",
                str(visual_manifest),
                "--out-dir",
                str(run_root / "visual_fixture/visual_qc_render"),
                "--max-video-frames",
                "12",
                "--overwrite",
            ],
        )
        assert_visual_render_pass(run_root / "visual_fixture/visual_qc_render/visual_qc_render_summary.json")
        add(
            "cem_override_handoff",
            [
                sys.executable,
                py("export_cem_overrides.py"),
                "--handoff-manifest-tsv",
                str(cem_override_manifest),
                "--out-dir",
                str(run_root / "cem_override_fixture/cem_overrides"),
                "--spider-repo",
                str(spider_repo),
            ],
        )
        assert_cem_override_pass(run_root / "cem_override_fixture/cem_overrides/cem_override_summary.json")

        if raw_root and raw_root.exists() and smplx_model_dir and smplx_model_dir.exists():
            full_run_id = "smoke_full_from_raw"
            add(
                "pipeline_full_from_raw",
                [
                    sys.executable,
                    py("run_pipeline.py"),
                    "--mode",
                    "full-from-raw",
                    "--run-id",
                    full_run_id,
                    "--run-root",
                    str(run_root),
                    "--core4d-raw-root",
                    str(raw_root),
                    "--smplx-model-dir",
                    str(smplx_model_dir),
                    "--queue",
                    "object-key",
                    "--object-keys",
                    args.object_keys,
                    "--max-sequences",
                    str(args.max_sequences),
                    "--sample-count",
                    str(args.sample_count),
                    "--retarget-variant-id",
                    "omnirt_v1",
                    "--target-variant-id",
                    "ref_fk",
                ],
            )
            full_dir = run_root / full_run_id
            add("verify_full_from_raw", [sys.executable, py("verify_reproducibility.py"), "--run-dir", str(full_dir)])

            registry = full_dir / "registries/case_state_registry.tsv"
            stage2b = sorted((full_dir / "s3_retarget").rglob("stage2b_manifest_*.tsv"))
            target_gate = sorted((full_dir / "s4_gate_visual_qc").rglob("target_gate_manifest.tsv"))
            resume_cmd = [
                sys.executable,
                py("run_pipeline.py"),
                "--mode",
                "resume-from-summary",
                "--run-id",
                "smoke_resume_from_summary",
                "--run-root",
                str(run_root),
                "--resume-registry",
                str(registry),
            ]
            if first_existing(stage2b):
                resume_cmd.extend(["--resume-stage2b-manifest", str(first_existing(stage2b))])
            if first_existing(target_gate):
                resume_cmd.extend(["--resume-target-gate-manifest", str(first_existing(target_gate))])
            add("pipeline_resume_from_summary", resume_cmd)
            add(
                "verify_resume_from_summary",
                [sys.executable, py("verify_reproducibility.py"), "--run-dir", str(run_root / "smoke_resume_from_summary")],
            )
        else:
            report["warnings"].append("Skipped raw full-from-raw/resume smoke because CORE4D raw root or SMPLX model dir is unavailable.")
    except Exception as exc:  # noqa: BLE001
        report["status"] = "fail"
        report["error"] = f"{type(exc).__name__}: {exc}"

    write_json(run_root / "smoke_suite_report.json", report)
    (run_root / "smoke_suite_report.md").write_text(markdown_report(report), encoding="utf-8")
    print(json_dumps({"status": report["status"], "report": str(run_root / "smoke_suite_report.json")}))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
