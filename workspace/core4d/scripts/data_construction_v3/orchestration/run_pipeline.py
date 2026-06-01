#!/usr/bin/env python3
"""Orchestrate Core4D data-construction v3 S0-S5 runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

SCRIPT_ROOT = next(parent for parent in Path(__file__).resolve().parents if parent.name == "data_construction_v3")
for _path in (SCRIPT_ROOT / "lib", SCRIPT_ROOT / "state", SCRIPT_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
from typing import Any

from common import (
    SCHEMA_VERSION,
    find_spider_repo,
    git_state,
    json_dumps,
    read_tsv,
    resolve_holosoma_repo,
    resolve_run_root,
    run_id_from_time,
    script_path,
    sha256_file,
    timestamp,
    write_json,
)


SCRIPT_DIR = SCRIPT_ROOT


def config_hash(config: dict[str, Any]) -> str:
    payload = json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def run_command(cmd: list[str], *, cwd: Path, dry_run: bool = False) -> dict[str, Any]:
    started = timestamp()
    item: dict[str, Any] = {
        "command": cmd,
        "started_at": started,
        "dry_run": dry_run,
        "returncode": "",
    }
    print("+ " + " ".join(str(x) for x in cmd))
    if dry_run:
        item.update({"returncode": 0, "ended_at": timestamp()})
        return item
    completed = subprocess.run(cmd, cwd=cwd, check=False)
    item.update({"returncode": completed.returncode, "ended_at": timestamp()})
    if completed.returncode != 0:
        raise RuntimeError(f"command failed with exit code {completed.returncode}: {' '.join(str(x) for x in cmd)}")
    return item


def mkdirs(run_dir: Path) -> None:
    for rel in [
        "config",
        "registries",
        "inputs",
        "stage_s0_environment",
        "stage_s1_raw_contact/inventory",
        "stage_s1_raw_contact/raw_contact",
        "stage_s2_templates",
        "stage_s3_retarget",
        "stage_s4_gate_visual_qc",
        "stage_s5_handoff",
        "stage_s6_downstream",
        "logs",
        "imported_snapshots",
    ]:
        (run_dir / rel).mkdir(parents=True, exist_ok=True)


def py(script: str) -> str:
    return str(script_path(SCRIPT_DIR, script))


def safe_id(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", text).strip("_")


RESUME_STATUS_FIELDS = [
    "raw_inventory_status",
    "raw_contact_3cm_status",
    "raw_contact_5cm_status",
    "template_status",
    "stage2b_status",
    "target_gate_status",
    "visual_qc_status",
    "cem_status",
    "rl_status",
]


def dedupe(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        if value and value not in out:
            out.append(value)
    return out


def row_has_completed_state(row: dict[str, str]) -> bool:
    if row.get("current_decision") and row.get("current_decision") != "not_run":
        return True
    for field in RESUME_STATUS_FIELDS:
        value = row.get(field, "")
        if value and value not in {"not_run", "not_seen"}:
            return True
    return False


def validate_resume_registry(path: Path) -> list[str]:
    errors: list[str] = []
    if not path.is_file():
        return [f"resume registry missing: {path}"]
    rows = read_tsv(path)
    for idx, row in enumerate(rows, start=2):
        schema = row.get("schema_version", "")
        if schema and schema != SCHEMA_VERSION:
            errors.append(f"{path}:{idx} schema_version={schema} expected={SCHEMA_VERSION}")
        if not row_has_completed_state(row):
            continue
        evidence_root = row.get("evidence_root", "")
        if not evidence_root:
            errors.append(f"{path}:{idx} case_id={row.get('case_id', '')} completed state has empty evidence_root")
            continue
        if not Path(evidence_root).expanduser().exists():
            errors.append(f"{path}:{idx} case_id={row.get('case_id', '')} evidence_root does not exist: {evidence_root}")
    return errors


def validate_resume_stage_manifest(path: Path, required_fields: list[str]) -> list[str]:
    errors: list[str] = []
    if not path.is_file():
        return [f"resume stage manifest missing: {path}"]
    rows = read_tsv(path)
    if not rows:
        return []
    missing = [field for field in required_fields if field not in rows[0]]
    if missing:
        errors.append(f"{path} missing required fields: {missing}")
    for idx, row in enumerate(rows, start=2):
        schema = row.get("schema_version", "")
        if schema and schema != SCHEMA_VERSION:
            errors.append(f"{path}:{idx} schema_version={schema} expected={SCHEMA_VERSION}")
    return errors


def validate_resume_inputs(args: argparse.Namespace) -> None:
    errors: list[str] = []
    if args.resume_registry:
        errors.extend(validate_resume_registry(args.resume_registry.expanduser().resolve()))
    for path in args.resume_stage2b_manifest or []:
        errors.extend(
            validate_resume_stage_manifest(
                path.expanduser().resolve(),
                [
                    "case_id",
                    "retarget_variant_id",
                    "target_variant_id",
                    "stage2b_decision",
                    "result_root",
                    "converted_npz",
                    "omniretarget_output_npz",
                    "trimmed_npz",
                    "spider_trajectory",
                ],
            )
        )
    for path in args.resume_target_gate_manifest or []:
        errors.extend(
            validate_resume_stage_manifest(
                path.expanduser().resolve(),
                ["case_id", "retarget_variant_id", "target_variant_id", "target_gate_status"],
            )
        )
    if args.resume_retarget_variant_registry:
        errors.extend(
            validate_resume_stage_manifest(
                args.resume_retarget_variant_registry.expanduser().resolve(),
                ["retarget_variant_id", "params_json", "schema_version"],
            )
        )
    if errors:
        raise SystemExit("resume input validation failed:\n" + "\n".join(f"- {error}" for error in errors[:50]))


def full_from_raw(args: argparse.Namespace, paths: dict[str, Path], manifest: dict[str, Any]) -> None:
    spider_repo = paths["spider_repo"]
    run_dir = paths["run_dir"]
    raw_root = paths["core4d_raw_root"]
    holosoma_repo = paths["holosoma_repo"]
    run_root = paths["run_root"]
    stage_decisions: dict[str, Any] = manifest["stage_decisions"]
    commands: list[dict[str, Any]] = manifest["commands"]

    env_cmd = [
        sys.executable,
        py("check_environment.py"),
        "--spider-repo",
        str(spider_repo),
        "--holosoma-repo",
        str(holosoma_repo),
        "--core4d-raw-root",
        str(raw_root),
        "--run-root",
        str(run_root),
        "--out-dir",
        str(run_dir / "stage_s0_environment"),
    ]
    if args.allow_missing_guard_scene:
        env_cmd.append("--allow-missing-guard-scene")
    commands.append(run_command(env_cmd, cwd=spider_repo))
    stage_decisions["S0_environment"] = "pass"

    commands.append(
        run_command(
            [
                sys.executable,
                py("register_retarget_variant.py"),
                "--registry-dir",
                str(run_dir / "registries"),
                "--spider-repo",
                str(spider_repo),
                "--holosoma-repo",
                str(holosoma_repo),
                "--init-defaults",
            ],
            cwd=spider_repo,
        )
    )
    commands.append(
        run_command(
            [
                sys.executable,
                py("update_case_state_registry.py"),
                "--registry-dir",
                str(run_dir / "registries"),
                "--init-empty",
            ],
            cwd=spider_repo,
        )
    )
    stage_decisions["S0b_registries"] = "pass"

    inv_dir = run_dir / "stage_s1_raw_contact/inventory"
    raw_contact_dir = run_dir / "stage_s1_raw_contact/raw_contact"
    commands.append(
        run_command(
            [
                sys.executable,
                py("build_inventory.py"),
                "--core4d-raw-root",
                str(raw_root),
                "--spider-repo",
                str(spider_repo),
                "--out-dir",
                str(inv_dir),
            ],
            cwd=spider_repo,
        )
    )
    commands.append(
        run_command(
            [
                sys.executable,
                py("update_case_state_registry.py"),
                "--registry-dir",
                str(run_dir / "registries"),
                "--from-inventory-tsv",
                str(inv_dir / "inventory.tsv"),
                "--evidence-root",
                str(inv_dir),
                "--source-ref",
                "S1_inventory",
            ],
            cwd=spider_repo,
        )
    )
    raw_cmd = [
        sys.executable,
        py("run_raw_contact.py"),
        "--core4d-raw-root",
        str(raw_root),
        "--inventory-tsv",
        str(inv_dir / "inventory.tsv"),
        "--out-dir",
        str(raw_contact_dir),
        "--queue",
        args.queue,
        "--thresholds-m",
        args.thresholds_m,
        "--sample-count",
        str(args.sample_count),
    ]
    if args.object_keys:
        raw_cmd.extend(["--object-keys", args.object_keys])
    if args.max_case_persons is not None:
        raw_cmd.extend(["--max-case-persons", str(args.max_case_persons)])
    if args.max_sequences is not None:
        raw_cmd.extend(["--max-sequences", str(args.max_sequences)])
    commands.append(run_command(raw_cmd, cwd=spider_repo))
    for label in ["3cm", "5cm"]:
        contact_path = raw_contact_dir / f"raw_contact_candidates_{label}.tsv"
        if contact_path.is_file():
            commands.append(
                run_command(
                    [
                        sys.executable,
                        py("update_case_state_registry.py"),
                        "--registry-dir",
                        str(run_dir / "registries"),
                        "--from-raw-contact-tsv",
                        str(contact_path),
                        "--raw-contact-label",
                        label,
                        "--evidence-root",
                        str(raw_contact_dir),
                        "--source-ref",
                        f"S1_raw_contact_{label}",
                    ],
                    cwd=spider_repo,
                )
            )
    stage_decisions["S1_raw_inventory_contact"] = "pass"

    template_dir = run_dir / "stage_s2_templates"
    stage2b_contact_label = args.stage2b_contact_label
    stage2b_raw_contact_pass = raw_contact_dir / f"raw_contact_pass_{stage2b_contact_label}.tsv"
    if not stage2b_raw_contact_pass.is_file():
        raise FileNotFoundError(f"selected Stage2b raw-contact pass TSV missing: {stage2b_raw_contact_pass}")
    s2_cmd = [
        sys.executable,
        py("build_or_audit_templates.py"),
        "--input-tsv",
        str(stage2b_raw_contact_pass),
        "--core4d-raw-root",
        str(raw_root),
        "--out-dir",
        str(template_dir),
    ]
    if args.apply_build_templates:
        s2_cmd.append("--apply-build")
    commands.append(run_command(s2_cmd, cwd=spider_repo))
    commands.append(
        run_command(
            [
                sys.executable,
                py("update_case_state_registry.py"),
                "--registry-dir",
                str(run_dir / "registries"),
                "--from-template-backlog-tsv",
                str(template_dir / "template_backlog.tsv"),
                "--evidence-root",
                str(template_dir),
                "--source-ref",
                "S2_template_backlog_audit",
            ],
            cwd=spider_repo,
        )
    )
    template_visual_dir = template_dir / "template_visual_review"
    commands.append(
        run_command(
            [
                sys.executable,
                py("render_template_review_package.py"),
                "--template-backlog-tsv",
                str(template_dir / "template_backlog.tsv"),
                "--out-dir",
                str(template_visual_dir),
            ],
            cwd=spider_repo,
        )
    )
    stage_decisions["S2_templates"] = "pass"
    stage_decisions["S2_template_visual_review"] = "pass"

    route_diagnostic_tsvs = list(args.route_diagnostic_tsv)
    if args.build_fingertip_route_diagnostics:
        diag_dir = run_dir / "stage_s1_raw_contact/fingertip_route_diagnostics"
        diag_cmd = [
            sys.executable,
            py("build_fingertip_route_diagnostics.py"),
            "--input-tsv",
            str(stage2b_raw_contact_pass),
            "--out-dir",
            str(diag_dir),
            "--fingertip-face-stats",
            str(args.fingertip_face_stats),
            "--palm-face-stats",
            str(args.palm_face_stats),
            "--quat-audit",
            str(args.quat_audit),
            "--target-gap-summary",
            str(args.target_gap_summary),
            "--target-dir",
            str(args.fingertip_target_dir),
            "--e101-outcome-matrix",
            str(args.e101_outcome_matrix),
        ]
        commands.append(run_command(diag_cmd, cwd=spider_repo))
        route_diagnostic_tsvs.append(diag_dir / "fingertip_route_diagnostics.tsv")
        stage_decisions["S1b_fingertip_route_diagnostics"] = "pass"

    stage2b_manifests: list[Path] = []
    target_gate_manifests: list[Path] = []
    for variant_id in args.retarget_variant_id:
        for target_variant_id in args.target_variant_id:
            run_slug = safe_id(f"{variant_id}_{target_variant_id}")
            variant_dir = run_dir / "stage_s3_retarget" / variant_id / target_variant_id
            s3_cmd = [
                sys.executable,
                py("run_stage2b.py"),
                "--raw-contact-tsv",
                str(stage2b_raw_contact_pass),
                "--template-backlog-tsv",
                str(template_dir / "template_backlog.tsv"),
                "--retarget-variant-registry",
                str(run_dir / "registries/retarget_variant_registry.tsv"),
                "--retarget-variant-id",
                variant_id,
                "--target-variant-id",
                target_variant_id,
                "--inventory-tsv",
                str(inv_dir / "inventory.tsv"),
                "--core4d-raw-root",
                str(raw_root),
                "--smplx-model-dir",
                str(paths["smplx_model_dir"]),
                "--out-dir",
                str(variant_dir),
            ]
            if args.execute_stage2b:
                s3_cmd.extend(["--execute", "--allow-legacy-stage2b-wrapper"])
            for path in route_diagnostic_tsvs:
                s3_cmd.extend(["--route-diagnostic-tsv", str(path)])
            commands.append(run_command(s3_cmd, cwd=spider_repo))
            s3_manifest = variant_dir / f"stage2b_manifest_{run_slug}.tsv"
            stage2b_manifests.append(s3_manifest)
            commands.append(
                run_command(
                    [
                        sys.executable,
                        py("update_case_state_registry.py"),
                        "--registry-dir",
                        str(run_dir / "registries"),
                        "--from-stage2b-manifest-tsv",
                        str(s3_manifest),
                        "--evidence-root",
                        str(variant_dir),
                        "--source-ref",
                        f"S3_stage2b_{variant_id}_{target_variant_id}",
                    ],
                    cwd=spider_repo,
                )
            )

            gate_dir = run_dir / "stage_s4_gate_visual_qc" / variant_id / target_variant_id
            commands.append(
                run_command(
                    [
                        sys.executable,
                        py("run_target_gate.py"),
                        "--stage2b-manifest-tsv",
                        str(s3_manifest),
                        "--out-dir",
                        str(gate_dir),
                        "--spider-repo",
                        str(spider_repo),
                    ],
                    cwd=spider_repo,
                )
            )
            gate_manifest = gate_dir / "target_gate_manifest.tsv"
            target_gate_manifests.append(gate_manifest)
            commands.append(
                run_command(
                    [
                        sys.executable,
                        py("update_case_state_registry.py"),
                        "--registry-dir",
                        str(run_dir / "registries"),
                        "--from-target-gate-manifest-tsv",
                        str(gate_manifest),
                        "--evidence-root",
                        str(gate_dir),
                        "--source-ref",
                        f"S4_target_gate_{variant_id}_{target_variant_id}",
                    ],
                    cwd=spider_repo,
                )
            )
            visual_dir = gate_dir / "visual_qc"
            commands.append(
                run_command(
                    [
                        sys.executable,
                        py("make_visual_qc.py"),
                        "--target-gate-manifest-tsv",
                        str(gate_manifest),
                        "--out-dir",
                        str(visual_dir),
                    ],
                    cwd=spider_repo,
                )
            )
            visual_manifest = visual_dir / "visual_qc_manifest.tsv"
            commands.append(
                run_command(
                    [
                        sys.executable,
                        py("update_case_state_registry.py"),
                        "--registry-dir",
                        str(run_dir / "registries"),
                        "--from-visual-qc-manifest-tsv",
                        str(visual_manifest),
                        "--evidence-root",
                        str(visual_dir),
                        "--source-ref",
                        f"S4_visual_qc_{variant_id}_{target_variant_id}",
                    ],
                    cwd=spider_repo,
                )
            )
    stage_decisions["S3_stage2b"] = "pass"
    stage_decisions["S4_target_gate"] = "pass"
    stage_decisions["S4_visual_qc_manifest"] = "pass"

    run_export_handoff(spider_repo, run_dir, stage2b_manifests, target_gate_manifests, commands)
    stage_decisions["S5_handoff"] = "pass"


def run_export_handoff(
    spider_repo: Path,
    run_dir: Path,
    stage2b_manifests: list[Path],
    target_gate_manifests: list[Path],
    commands: list[dict[str, Any]],
) -> None:
    cmd = [
        sys.executable,
        py("export_handoff.py"),
        "--case-state-registry",
        str(run_dir / "registries/case_state_registry.tsv"),
        "--out-dir",
        str(run_dir / "stage_s5_handoff"),
    ]
    for path in stage2b_manifests:
        if path.is_file():
            cmd.extend(["--stage2b-manifest-tsv", str(path)])
    for path in target_gate_manifests:
        if path.is_file():
            cmd.extend(["--target-gate-manifest-tsv", str(path)])
    commands.append(run_command(cmd, cwd=spider_repo))
    commands.append(
        run_command(
            [
                sys.executable,
                py("export_cem_overrides.py"),
                "--handoff-manifest-tsv",
                str(run_dir / "stage_s5_handoff/handoff_manifest.tsv"),
                "--out-dir",
                str(run_dir / "stage_s5_handoff/cem_overrides"),
                "--spider-repo",
                str(spider_repo),
            ],
            cwd=spider_repo,
        )
    )


def copy_resume_input(path: Path, out_dir: Path, prefix: str) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / f"{prefix}_{path.name}"
    shutil.copy2(path, dst)
    return {
        "source_path": str(path),
        "copied_path": str(dst),
        "source_sha256": sha256_file(path),
        "copied_sha256": sha256_file(dst),
        "bytes": str(dst.stat().st_size),
    }


def resume_from_summary(args: argparse.Namespace, paths: dict[str, Path], manifest: dict[str, Any]) -> None:
    run_dir = paths["run_dir"]
    spider_repo = paths["spider_repo"]
    holosoma_repo = paths["holosoma_repo"]
    commands: list[dict[str, Any]] = manifest["commands"]
    stage_decisions: dict[str, Any] = manifest["stage_decisions"]
    (run_dir / "registries").mkdir(parents=True, exist_ok=True)
    if not args.resume_registry:
        raise SystemExit("resume-from-summary requires --resume-registry")
    validate_resume_inputs(args)
    resume_dir = run_dir / "imported_snapshots/resume_inputs"
    registry_input = copy_resume_input(args.resume_registry.expanduser().resolve(), resume_dir, "case_state")
    stage2b_inputs = [
        copy_resume_input(path.expanduser().resolve(), resume_dir, f"stage2b_{idx:02d}")
        for idx, path in enumerate(args.resume_stage2b_manifest or [], start=1)
    ]
    target_gate_inputs = [
        copy_resume_input(path.expanduser().resolve(), resume_dir, f"target_gate_{idx:02d}")
        for idx, path in enumerate(args.resume_target_gate_manifest or [], start=1)
    ]
    retarget_variant_input: dict[str, str] = {}
    if args.resume_retarget_variant_registry:
        source_variant = args.resume_retarget_variant_registry.expanduser().resolve()
        copied_variant = run_dir / "registries/retarget_variant_registry.tsv"
        shutil.copy2(source_variant, copied_variant)
        retarget_variant_input = {
            "source_path": str(source_variant),
            "copied_path": str(copied_variant),
            "source_sha256": sha256_file(source_variant),
            "copied_sha256": sha256_file(copied_variant),
            "bytes": str(copied_variant.stat().st_size),
        }
    else:
        commands.append(
            run_command(
                [
                    sys.executable,
                    py("register_retarget_variant.py"),
                    "--registry-dir",
                    str(run_dir / "registries"),
                    "--spider-repo",
                    str(spider_repo),
                    "--holosoma-repo",
                    str(holosoma_repo),
                    "--init-defaults",
                ],
                cwd=spider_repo,
            )
        )
    commands.append(
        run_command(
            [
                sys.executable,
                py("update_case_state_registry.py"),
                "--registry-dir",
                str(run_dir / "registries"),
                "--input-tsv",
                registry_input["copied_path"],
            ],
            cwd=spider_repo,
        )
    )
    write_json(
        resume_dir / "resume_input_manifest.json",
        {
            "stage": "resume_from_summary_inputs",
            "created_at": timestamp(),
            "schema_version": SCHEMA_VERSION,
            "registry": registry_input,
            "stage2b_manifests": stage2b_inputs,
            "target_gate_manifests": target_gate_inputs,
            "retarget_variant_registry": retarget_variant_input,
            "source_registry": str(args.resume_registry),
            "copied_registry": registry_input["copied_path"],
            "source_stage2b_manifests": [item["source_path"] for item in stage2b_inputs],
            "copied_stage2b_manifests": [item["copied_path"] for item in stage2b_inputs],
            "source_target_gate_manifests": [item["source_path"] for item in target_gate_inputs],
            "copied_target_gate_manifests": [item["copied_path"] for item in target_gate_inputs],
            "source_retarget_variant_registry": retarget_variant_input.get("source_path", ""),
        },
    )
    stage2b_manifests = [Path(item["copied_path"]) for item in stage2b_inputs]
    target_gate_manifests = [Path(item["copied_path"]) for item in target_gate_inputs]
    run_export_handoff(spider_repo, run_dir, stage2b_manifests, target_gate_manifests, commands)
    stage_decisions["resume_import_registry"] = "pass"
    stage_decisions["S5_handoff"] = "pass"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("full-from-raw", "resume-from-summary"), required=True)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--spider-repo", type=Path, default=None)
    parser.add_argument("--holosoma-repo", type=Path, default=None)
    parser.add_argument("--core4d-raw-root", type=Path, default=None)
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--smplx-model-dir", type=Path, default=None)
    parser.add_argument("--queue", default="selected-medium-box")
    parser.add_argument("--object-keys", default="")
    parser.add_argument("--thresholds-m", default="0.03,0.05")
    parser.add_argument("--stage2b-contact-label", choices=("3cm", "5cm"), default="3cm")
    parser.add_argument("--sample-count", type=int, default=12000)
    parser.add_argument("--max-case-persons", type=int, default=None)
    parser.add_argument("--max-sequences", type=int, default=None)
    parser.add_argument("--retarget-variant-id", action="append", default=None)
    parser.add_argument("--target-variant-id", action="append", default=None)
    parser.add_argument("--route-diagnostic-tsv", type=Path, action="append", default=[])
    parser.add_argument("--build-fingertip-route-diagnostics", action="store_true")
    parser.add_argument("--fingertip-face-stats", type=Path, default=Path("workspace/core4d/results/E099/fingertip_face_stats.tsv"))
    parser.add_argument("--palm-face-stats", type=Path, default=Path("workspace/core4d/results/E099/palm_face_stats.tsv"))
    parser.add_argument("--quat-audit", type=Path, default=Path("workspace/core4d/results/E099/quat_audit.tsv"))
    parser.add_argument("--target-gap-summary", type=Path, default=Path("workspace/core4d/results/E100/target_gap_summary.tsv"))
    parser.add_argument("--fingertip-target-dir", type=Path, default=Path("workspace/core4d/results/E100/fingertip_targets"))
    parser.add_argument("--e101-outcome-matrix", type=Path, default=Path("workspace/core4d/results/E101/cem_outcome_matrix.tsv"))
    parser.add_argument("--apply-build-templates", action="store_true")
    parser.add_argument("--execute-stage2b", action="store_true")
    parser.add_argument("--allow-missing-guard-scene", action="store_true")
    parser.add_argument("--resume-registry", type=Path, default=None)
    parser.add_argument("--resume-retarget-variant-registry", type=Path, default=None)
    parser.add_argument("--resume-stage2b-manifest", type=Path, action="append", default=[])
    parser.add_argument("--resume-target-gate-manifest", type=Path, action="append", default=[])
    args = parser.parse_args()
    args.retarget_variant_id = dedupe(args.retarget_variant_id or ["omnirt_v1"])
    args.target_variant_id = dedupe(args.target_variant_id or ["ref_fk"])

    spider_repo = (args.spider_repo or find_spider_repo()).resolve()
    holosoma_repo = (args.holosoma_repo or resolve_holosoma_repo()).resolve()
    run_root = (args.run_root or resolve_run_root(holosoma_repo)).resolve()
    run_id = args.run_id or run_id_from_time(f"dcv3_{args.mode.replace('-', '_')}")
    run_dir = (run_root / run_id).resolve()
    raw_root = args.core4d_raw_root or (Path(os.environ["CORE4D_RAW_ROOT"]) if os.environ.get("CORE4D_RAW_ROOT") else None)
    smplx_model_dir = args.smplx_model_dir or (Path(os.environ["SMPLX_MODEL_DIR"]) if os.environ.get("SMPLX_MODEL_DIR") else None)
    if args.mode == "full-from-raw":
        if raw_root is None:
            raise SystemExit("full-from-raw requires --core4d-raw-root or CORE4D_RAW_ROOT")
        if smplx_model_dir is None:
            raise SystemExit("full-from-raw requires --smplx-model-dir or SMPLX_MODEL_DIR")
    paths = {
        "spider_repo": spider_repo,
        "holosoma_repo": holosoma_repo,
        "run_root": run_root,
        "run_dir": run_dir,
        "core4d_raw_root": raw_root.expanduser().resolve() if raw_root else Path(""),
        "smplx_model_dir": smplx_model_dir.expanduser().resolve() if smplx_model_dir else Path(""),
    }
    mkdirs(run_dir)
    config = {
        "mode": args.mode,
        "run_id": run_id,
        "spider_repo": str(spider_repo),
        "holosoma_repo": str(holosoma_repo),
        "run_root": str(run_root),
        "run_dir": str(run_dir),
        "core4d_raw_root": str(paths["core4d_raw_root"]),
        "smplx_model_dir": str(paths["smplx_model_dir"]),
        "queue": args.queue,
        "object_keys": args.object_keys,
        "thresholds_m": args.thresholds_m,
        "stage2b_contact_label": args.stage2b_contact_label,
        "sample_count": args.sample_count,
        "max_case_persons": args.max_case_persons,
        "max_sequences": args.max_sequences,
        "retarget_variant_id": args.retarget_variant_id,
        "target_variant_id": args.target_variant_id,
        "route_diagnostic_tsv": [str(path) for path in args.route_diagnostic_tsv],
        "build_fingertip_route_diagnostics": args.build_fingertip_route_diagnostics,
        "fingertip_face_stats": str(args.fingertip_face_stats),
        "palm_face_stats": str(args.palm_face_stats),
        "quat_audit": str(args.quat_audit),
        "target_gap_summary": str(args.target_gap_summary),
        "fingertip_target_dir": str(args.fingertip_target_dir),
        "e101_outcome_matrix": str(args.e101_outcome_matrix),
        "apply_build_templates": args.apply_build_templates,
        "execute_stage2b": args.execute_stage2b,
        "resume_registry": str(args.resume_registry or ""),
        "resume_retarget_variant_registry": str(args.resume_retarget_variant_registry or ""),
        "resume_stage2b_manifest": [str(path) for path in args.resume_stage2b_manifest],
        "resume_target_gate_manifest": [str(path) for path in args.resume_target_gate_manifest],
        "schema_version": SCHEMA_VERSION,
    }
    config["config_hash"] = config_hash(config)
    write_json(run_dir / "config_resolved.json", config)
    write_json(run_dir / "git_state.json", {"spider": git_state(spider_repo), "holosoma": git_state(holosoma_repo)})
    manifest: dict[str, Any] = {
        "run_id": run_id,
        "mode": args.mode,
        "start_time": timestamp(),
        "end_time": "",
        "config_path": str(run_dir / "config_resolved.json"),
        "config_hash": config["config_hash"],
        "raw_data_root": str(paths["core4d_raw_root"]),
        "output_root": str(run_dir),
        "spider_git": git_state(spider_repo),
        "holosoma_git": git_state(holosoma_repo),
        "command_line": sys.argv,
        "stage_decisions": {},
        "known_warnings": [],
        "commands": [],
        "schema_version": SCHEMA_VERSION,
    }
    try:
        if args.mode == "full-from-raw":
            full_from_raw(args, paths, manifest)
        else:
            resume_from_summary(args, paths, manifest)
    except Exception as exc:
        manifest["end_time"] = timestamp()
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        write_json(run_dir / "run_manifest.json", manifest)
        raise
    manifest["end_time"] = timestamp()
    manifest["status"] = "pass"
    write_json(run_dir / "run_manifest.json", manifest)
    print(json_dumps({"status": "pass", "run_dir": str(run_dir), "run_manifest": str(run_dir / "run_manifest.json")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
