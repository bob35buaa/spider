#!/usr/bin/env python3
"""Verify a Core4D data-construction v3 run directory is self-describing."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
import sys

SCRIPT_ROOT = next(parent for parent in Path(__file__).resolve().parents if parent.name == "data_construction_v3")
for _path in (SCRIPT_ROOT / "lib", SCRIPT_ROOT / "state", SCRIPT_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
from typing import Any

from common import SCHEMA_VERSION, find_spider_repo, json_dumps, read_tsv, sha256_file, timestamp, write_json


STATUS_FIELDS = [
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

OPTIONAL_STAGE_DIRS = [
    "stage_s1_raw_contact/fingertip_route_diagnostics",
    "imported_snapshots/resume_inputs",
]


def load_json(path: Path, errors: list[str]) -> Any:
    if not path.is_file():
        errors.append(f"missing file: {path}")
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        errors.append(f"invalid json: {path}: {type(exc).__name__}: {exc}")
        return None


def config_hash(config: dict[str, Any]) -> str:
    payload = dict(config)
    payload.pop("config_hash", None)
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    import hashlib

    return hashlib.sha256(raw).hexdigest()


def check_schema_rows(rows: list[dict[str, str]], path: Path, warnings: list[str]) -> None:
    for idx, row in enumerate(rows, start=2):
        schema = row.get("schema_version", "")
        if schema and schema != SCHEMA_VERSION:
            warnings.append(f"{path}:{idx} schema_version={schema} expected={SCHEMA_VERSION}")


def evidence_required(row: dict[str, str]) -> bool:
    if row.get("source_type") in {"manual_seed", "legacy_import"}:
        return bool(row.get("current_decision") and row.get("current_decision") != "not_run")
    for field in STATUS_FIELDS:
        value = row.get(field, "")
        if value and value != "not_run":
            return True
    return False


def check_registry(run_dir: Path, errors: list[str], warnings: list[str]) -> dict[str, Any]:
    registry_path = run_dir / "registries/case_state_registry.tsv"
    variant_path = run_dir / "registries/retarget_variant_registry.tsv"
    summary: dict[str, Any] = {}
    if not registry_path.is_file():
        errors.append(f"missing case_state_registry.tsv: {registry_path}")
    else:
        rows = read_tsv(registry_path)
        check_schema_rows(rows, registry_path, warnings)
        summary["case_state_rows"] = len(rows)
        summary["current_decision_counts"] = dict(Counter(row.get("current_decision", "") for row in rows))
        missing_evidence = []
        for row in rows:
            if not evidence_required(row):
                continue
            evidence_root = row.get("evidence_root", "")
            if not evidence_root:
                missing_evidence.append(f"{row.get('case_id', '')}:{row.get('current_decision', '')}:empty_evidence_root")
                continue
            if not Path(evidence_root).expanduser().exists():
                missing_evidence.append(f"{row.get('case_id', '')}:{row.get('current_decision', '')}:{evidence_root}")
        if missing_evidence:
            sample = "; ".join(missing_evidence[:20])
            errors.append(f"registry rows have missing evidence_root count={len(missing_evidence)} sample={sample}")

    if not variant_path.is_file():
        errors.append(f"missing retarget_variant_registry.tsv: {variant_path}")
    else:
        rows = read_tsv(variant_path)
        check_schema_rows(rows, variant_path, warnings)
        ids = [row.get("retarget_variant_id", "") for row in rows]
        duplicates = [item for item, count in Counter(ids).items() if item and count > 1]
        if duplicates:
            errors.append(f"duplicate retarget_variant_id: {duplicates}")
        bad_params = []
        for row in rows:
            try:
                json.loads(row.get("params_json", "") or "{}")
            except json.JSONDecodeError:
                bad_params.append(row.get("retarget_variant_id", ""))
        if bad_params:
            errors.append(f"invalid params_json for variants: {bad_params}")
        summary["retarget_variant_rows"] = len(rows)
    return summary


def check_imported_snapshots(run_dir: Path, mode: str, errors: list[str], warnings: list[str]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    snapshot_root = run_dir / "imported_snapshots"
    manifests = sorted(snapshot_root.glob("*/import_manifest.json")) if snapshot_root.is_dir() else []
    resume_inputs = snapshot_root / "resume_inputs/resume_input_manifest.json"
    summary["import_manifest_count"] = len(manifests)
    if resume_inputs.is_file():
        data = load_json(resume_inputs, errors)
        summary["resume_input_manifest"] = str(resume_inputs)
        if isinstance(data, dict):
            copied_entries: list[dict[str, Any]] = []
            for key in ("registry", "retarget_variant_registry"):
                entry = data.get(key, {})
                if isinstance(entry, dict) and entry.get("copied_path"):
                    copied_entries.append(entry)
            for key in ("stage2b_manifests", "target_gate_manifests"):
                entries = data.get(key, [])
                if isinstance(entries, list):
                    copied_entries.extend(entry for entry in entries if isinstance(entry, dict))
            summary["resume_input_copied_files"] = len(copied_entries)
            for entry in copied_entries:
                copied_path = Path(str(entry.get("copied_path", ""))).expanduser()
                expected_sha = str(entry.get("copied_sha256", ""))
                if not copied_path.is_file():
                    errors.append(f"resume copied input missing: {copied_path}")
                    continue
                actual_sha = sha256_file(copied_path)
                if expected_sha and actual_sha != expected_sha:
                    errors.append(f"resume copied input sha256 mismatch: {copied_path} expected={expected_sha} actual={actual_sha}")
                source_sha = str(entry.get("source_sha256", ""))
                if source_sha and expected_sha and source_sha != expected_sha:
                    errors.append(f"resume source/copy sha256 mismatch at copy time: {entry.get('source_path', '')} -> {copied_path}")
            if mode == "resume-from-summary" and not copied_entries:
                errors.append(f"resume_input_manifest has no copied file entries: {resume_inputs}")
    elif mode == "resume-from-summary":
        warnings.append(f"resume run has no resume_input_manifest.json: {resume_inputs}")
    for path in manifests:
        data = load_json(path, errors)
        if isinstance(data, dict):
            if data.get("schema_version") != SCHEMA_VERSION:
                warnings.append(f"{path} schema_version={data.get('schema_version')} expected={SCHEMA_VERSION}")
            if data.get("status") != "pass":
                errors.append(f"{path} status={data.get('status')}")
    return summary


def check_stage_files(run_dir: Path, mode: str, errors: list[str], warnings: list[str]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    inventory = run_dir / "stage_s1_raw_contact/inventory/inventory.tsv"
    require_raw_stages = mode != "resume-from-summary"
    if not inventory.is_file() and require_raw_stages:
        errors.append(f"missing inventory.tsv: {inventory}")
    elif inventory.is_file():
        rows = read_tsv(inventory)
        check_schema_rows(rows, inventory, warnings)
        summary["inventory_rows"] = len(rows)

    for label in ("3cm", "5cm"):
        candidates = run_dir / f"stage_s1_raw_contact/raw_contact/raw_contact_candidates_{label}.tsv"
        if not candidates.is_file() and require_raw_stages:
            errors.append(f"missing raw contact candidates {label}: {candidates}")
        elif candidates.is_file():
            rows = read_tsv(candidates)
            check_schema_rows(rows, candidates, warnings)
            summary[f"raw_contact_candidates_{label}"] = len(rows)

    template = run_dir / "stage_s2_templates/template_backlog.tsv"
    if not template.is_file() and require_raw_stages:
        errors.append(f"missing template_backlog.tsv: {template}")
    elif template.is_file():
        rows = read_tsv(template)
        check_schema_rows(rows, template, warnings)
        summary["template_rows"] = len(rows)

    template_visual = run_dir / "stage_s2_templates/template_visual_review/template_visual_manifest.tsv"
    if template_visual.is_file():
        rows = read_tsv(template_visual)
        check_schema_rows(rows, template_visual, warnings)
        summary["template_visual_rows"] = len(rows)
        status_counts = Counter(row.get("render_status", "") for row in rows)
        summary["template_visual_render_status_counts"] = dict(status_counts)
        missing = [
            field
            for field in ["source_scene_task", "template_status", "render_status", "scene_xml"]
            if rows and field not in rows[0]
        ]
        if missing:
            errors.append(f"{template_visual} missing required fields: {missing}")
        render_errors = [f"{row.get('source_scene_task', '')}:{row.get('render_notes', '')}" for row in rows if row.get("render_status") == "render_error"]
        missing_assets: list[str] = []
        for row in rows:
            if row.get("render_status") != "pass":
                continue
            for field in ("video_path", "sheet_path"):
                asset = Path(row.get(field, "")).expanduser()
                if not asset.is_file() or asset.stat().st_size <= 0:
                    missing_assets.append(f"{row.get('source_scene_task', '')}:{field}={asset}")
        if render_errors:
            errors.append(f"template visual render errors count={len(render_errors)} sample={'; '.join(render_errors[:20])}")
        if missing_assets:
            errors.append(f"template visual assets missing/empty count={len(missing_assets)} sample={'; '.join(missing_assets[:20])}")

    stage2b = sorted((run_dir / "stage_s3_retarget").rglob("stage2b_manifest_*.tsv"))
    summary["stage2b_manifest_count"] = len(stage2b)
    for path in stage2b:
        rows = read_tsv(path)
        check_schema_rows(rows, path, warnings)
        required = [
            "case_id",
            "retarget_variant_id",
            "target_variant_id",
            "stage2b_decision",
            "result_root",
            "converted_npz",
            "omniretarget_output_npz",
            "trimmed_npz",
            "spider_trajectory",
        ]
        missing = [field for field in required if field not in rows[0]] if rows else []
        if missing:
            errors.append(f"{path} missing required fields: {missing}")
        missing_outputs = []
        output_fields = [
            "converted_npz",
            "omniretarget_output_npz",
            "trimmed_npz",
            "spider_trajectory",
            "contact_mask_npz",
            "verify_summary",
        ]
        for row in rows:
            if row.get("stage2b_status") != "pass":
                continue
            for field in output_fields:
                asset = Path(row.get(field, "")).expanduser()
                if not asset.is_file() or asset.stat().st_size <= 0:
                    missing_outputs.append(f"{row.get('case_id', '')}:{field}={asset}")
        if missing_outputs:
            errors.append(f"{path} Stage2b pass outputs missing/empty count={len(missing_outputs)} sample={'; '.join(missing_outputs[:20])}")

    gate = sorted((run_dir / "stage_s4_gate_visual_qc").rglob("target_gate_manifest.tsv"))
    summary["target_gate_manifest_count"] = len(gate)
    for path in gate:
        rows = read_tsv(path)
        check_schema_rows(rows, path, warnings)
        missing = [field for field in ["case_id", "retarget_variant_id", "target_variant_id", "target_gate_status"] if field not in rows[0]] if rows else []
        if missing:
            errors.append(f"{path} missing required fields: {missing}")

    visual_qc = sorted((run_dir / "stage_s4_gate_visual_qc").rglob("visual_qc_manifest.tsv"))
    summary["visual_qc_manifest_count"] = len(visual_qc)
    for path in visual_qc:
        rows = read_tsv(path)
        check_schema_rows(rows, path, warnings)
        missing = [field for field in ["case_id", "retarget_variant_id", "target_variant_id", "visual_qc_status"] if rows and field not in rows[0]]
        if missing:
            errors.append(f"{path} missing required fields: {missing}")

    render_manifests = sorted((run_dir / "stage_s4_gate_visual_qc").rglob("visual_qc_render_manifest.tsv"))
    summary["visual_qc_render_manifest_count"] = len(render_manifests)
    render_status_counts: Counter[str] = Counter()
    missing_render_assets: list[str] = []
    render_errors: list[str] = []
    for path in render_manifests:
        rows = read_tsv(path)
        check_schema_rows(rows, path, warnings)
        missing = [field for field in ["case_id", "retarget_variant_id", "target_variant_id", "target_gate_status", "render_status"] if rows and field not in rows[0]]
        if missing:
            errors.append(f"{path} missing required fields: {missing}")
            continue
        for row in rows:
            status = row.get("render_status", "")
            render_status_counts[status] += 1
            if status == "render_error":
                render_errors.append(f"{path}:{row.get('case_id', '')}:{row.get('render_notes', '')}")
            if status == "not_rendered" and row.get("target_gate_status") == "pass":
                render_errors.append(f"{path}:{row.get('case_id', '')}:pass gate was not rendered")
            if status != "pass":
                continue
            for field in ("video_path", "sheet_path"):
                asset = Path(row.get(field, "")).expanduser()
                if not asset.is_file() or asset.stat().st_size <= 0:
                    missing_render_assets.append(f"{path}:{row.get('case_id', '')}:{field}={asset}")
    if render_status_counts:
        summary["visual_qc_render_status_counts"] = dict(render_status_counts)
    if render_errors:
        errors.append(f"visual QC render errors count={len(render_errors)} sample={'; '.join(render_errors[:20])}")
    if missing_render_assets:
        errors.append(f"visual QC render assets missing/empty count={len(missing_render_assets)} sample={'; '.join(missing_render_assets[:20])}")

    handoff = run_dir / "stage_s5_handoff/candidate_bank.tsv"
    if not handoff.is_file():
        errors.append(f"missing candidate_bank.tsv: {handoff}")
    else:
        rows = read_tsv(handoff)
        check_schema_rows(rows, handoff, warnings)
        summary["candidate_bank_rows"] = len(rows)

    cem_overrides = run_dir / "stage_s5_handoff/cem_overrides/cem_override_manifest.tsv"
    if cem_overrides.is_file():
        rows = read_tsv(cem_overrides)
        check_schema_rows(rows, cem_overrides, warnings)
        summary["cem_override_rows"] = len(rows)
        summary["cem_override_status_counts"] = dict(Counter(row.get("override_status", "") for row in rows))
        summary["cem_target_adapter_status_counts"] = dict(Counter(row.get("target_adapter_status", "") for row in rows))
        required = ["case_id", "retarget_variant_id", "target_variant_id", "override_status", "target_adapter_status"]
        missing = [field for field in required if rows and field not in rows[0]]
        if missing:
            errors.append(f"{cem_overrides} missing required fields: {missing}")
        repo = find_spider_repo()
        for row in rows:
            if row.get("override_status") != "pass":
                continue
            cfg = Path(row.get("override_config", "")).expanduser()
            if not cfg.is_file() or cfg.stat().st_size <= 0:
                errors.append(f"cem override pass row missing/empty config: {row.get('case_id', '')}:{cfg}")
            if row.get("contact_target_source") != "external":
                continue
            target = Path(row.get("contact_target_path", "")).expanduser()
            if not target.is_absolute():
                target = repo / target
            if not target.is_file():
                errors.append(f"cem override external target missing: {row.get('case_id', '')}:{target}")
            elif row.get("contact_target_sha256") and sha256_file(target) != row.get("contact_target_sha256"):
                errors.append(f"cem override external target sha256 mismatch: {row.get('case_id', '')}:{target}")

    downstream = run_dir / "stage_s6_downstream/downstream_evidence_manifest.tsv"
    if downstream.is_file():
        rows = read_tsv(downstream)
        check_schema_rows(rows, downstream, warnings)
        summary["downstream_evidence_rows"] = len(rows)
        missing = [
            field
            for field in ["case_id", "retarget_variant_id", "target_variant_id", "cem_status", "rl_status", "downstream_decision"]
            if rows and field not in rows[0]
        ]
        if missing:
            errors.append(f"{downstream} missing required fields: {missing}")

    for rel in OPTIONAL_STAGE_DIRS:
        path = run_dir / rel
        if path.exists():
            summary[f"optional_{rel.replace('/', '_')}"] = "present"
    return summary


def check_manifest(run_dir: Path, errors: list[str], warnings: list[str]) -> dict[str, Any]:
    config = load_json(run_dir / "config_resolved.json", errors)
    manifest = load_json(run_dir / "run_manifest.json", errors)
    load_json(run_dir / "git_state.json", errors)
    summary: dict[str, Any] = {}
    if isinstance(config, dict):
        expected = config.get("config_hash", "")
        actual = config_hash(config)
        summary["config_hash"] = expected
        if expected != actual:
            errors.append(f"config_hash mismatch expected={expected} recomputed={actual}")
        if config.get("schema_version") != SCHEMA_VERSION:
            warnings.append(f"config schema_version={config.get('schema_version')} expected={SCHEMA_VERSION}")
    if isinstance(manifest, dict):
        summary["mode"] = manifest.get("mode", "")
        summary["run_status"] = manifest.get("status", "")
        if manifest.get("schema_version") != SCHEMA_VERSION:
            warnings.append(f"run_manifest schema_version={manifest.get('schema_version')} expected={SCHEMA_VERSION}")
        if Path(str(manifest.get("output_root", ""))).resolve() != run_dir.resolve():
            errors.append(f"run_manifest output_root mismatch: {manifest.get('output_root')} != {run_dir}")
        if manifest.get("status") == "pass":
            bad_commands = [
                item
                for item in manifest.get("commands", [])
                if str(item.get("returncode", "")) not in {"0", "0.0"}
            ]
            if bad_commands:
                errors.append(f"run_manifest status=pass but command returncode failures count={len(bad_commands)}")
        summary["command_count"] = len(manifest.get("commands", []))
    return summary


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Core4D v3 reproducibility verification",
        "",
        f"- status: `{report['status']}`",
        f"- run_dir: `{report['run_dir']}`",
        f"- checked_at: `{report['checked_at']}`",
        f"- errors: `{len(report['errors'])}`",
        f"- warnings: `{len(report['warnings'])}`",
        "",
        "## Summary",
        "",
        "| key | value |",
        "|---|---:|",
    ]
    for key, value in report["summary"].items():
        if isinstance(value, dict):
            value = json.dumps(value, ensure_ascii=False, sort_keys=True)
        lines.append(f"| `{key}` | `{value}` |")
    if report["errors"]:
        lines.extend(["", "## Errors", ""])
        for item in report["errors"]:
            lines.append(f"- `{item}`")
    if report["warnings"]:
        lines.extend(["", "## Warnings", ""])
        for item in report["warnings"]:
            lines.append(f"- `{item}`")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--allow-warnings", action="store_true")
    args = parser.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    out_dir = (args.out_dir or (run_dir / "stage_s5_handoff/reproducibility")).resolve()
    errors: list[str] = []
    warnings: list[str] = []
    summary: dict[str, Any] = {}
    if not run_dir.is_dir():
        errors.append(f"run_dir does not exist: {run_dir}")
    else:
        summary.update(check_manifest(run_dir, errors, warnings))
        mode = str(summary.get("mode", ""))
        summary.update(check_registry(run_dir, errors, warnings))
        summary.update(check_imported_snapshots(run_dir, mode, errors, warnings))
        summary.update(check_stage_files(run_dir, mode, errors, warnings))

    status = "pass" if not errors and (args.allow_warnings or not warnings) else "fail"
    report = {
        "stage": "verify_reproducibility",
        "checked_at": timestamp(),
        "schema_version": SCHEMA_VERSION,
        "run_dir": str(run_dir),
        "status": status,
        "errors": errors,
        "warnings": warnings,
        "summary": summary,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "reproducibility_report.json", report)
    (out_dir / "reproducibility_report.md").write_text(markdown_report(report), encoding="utf-8")
    print(json_dumps(report))
    return 0 if status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
