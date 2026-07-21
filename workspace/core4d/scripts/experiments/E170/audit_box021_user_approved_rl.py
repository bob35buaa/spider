#!/usr/bin/env python3
"""Independently audit the formal E170 paired RL-ready export."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[5]
RUN_ROOT = REPO / "workspace/core4d/results/E170"
RL_DIR = RUN_ROOT / "s6_downstream/rl_export"
AUTHORITY = RUN_ROOT / "s6_downstream/eval/full/user_manual_review_template.tsv"
ANALYSIS = RUN_ROOT / "s6_downstream/manifests/analysis_manifest.tsv"
EVIDENCE = RUN_ROOT / "s6_downstream/evidence/box021_user_approved/downstream_evidence_manifest.tsv"
SOURCE_AUDIT = RL_DIR / "box021_user_approved_source_rows.tsv"
RL_INPUT = RL_DIR / "rl_export_input.tsv"
PARTNER = RL_DIR / "partner_omnirt/rl_partner_omnirt_manifest.tsv"
PAIRED = RL_DIR / "paired_rl_export_input.tsv"
OUT = RUN_ROOT / "s6_downstream/evidence/completion/rl_export_audit_final.json"
METHOD_ID = "E170_PRG_lowerbodyPhysics_softPenalty_candidateGate"


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def resolve_path(raw: str) -> Path:
    path = Path(raw)
    if path.exists():
        return path.resolve()
    for marker in ("example_datasets/", "workspace/", "logs/"):
        if marker in raw:
            return (REPO / (marker + raw.split(marker, 1)[1])).resolve()
    return path.resolve() if path.is_absolute() else (REPO / path).resolve()


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def partner_case_id(case_id: str) -> str:
    if case_id.endswith("_p1"):
        return case_id[:-3] + "_p2"
    if case_id.endswith("_p2"):
        return case_id[:-3] + "_p1"
    raise RuntimeError(f"invalid person-case id: {case_id}")


def unique(rows: list[dict[str, str]], key: str, label: str) -> dict[str, dict[str, str]]:
    output = {row[key]: row for row in rows}
    if len(output) != len(rows):
        raise RuntimeError(f"duplicate {key} in {label}")
    return output


def check_file(raw: str, expected_sha: str = "") -> tuple[bool, str]:
    path = resolve_path(raw)
    if not path.is_file() or path.stat().st_size == 0:
        return False, f"missing_or_empty:{raw}"
    actual = digest(path)
    if expected_sha and actual != expected_sha:
        return False, f"sha_mismatch:{raw}"
    return True, actual


def main() -> int:
    checks: list[dict[str, Any]] = []

    def record(name: str, passed: bool, detail: Any) -> None:
        checks.append({"name": name, "pass": bool(passed), "detail": detail})

    authority_rows = read_tsv(AUTHORITY)
    approved = {row["case_id"] for row in authority_rows if row["manual_use_decision"] == "USE"}
    rejected = {row["case_id"] for row in authority_rows if row["manual_use_decision"] == "DO_NOT_USE"}
    record(
        "final_authority",
        len(authority_rows) == 28 and len(approved) == 18 and len(rejected) == 10,
        {"rows": len(authority_rows), "use": len(approved), "do_not_use": len(rejected)},
    )

    analysis_by_case = unique(read_tsv(ANALYSIS), "case_id", "analysis")
    source_by_case = unique(read_tsv(SOURCE_AUDIT), "case_id", "source audit")
    evidence_by_case = unique(read_tsv(EVIDENCE), "case_id", "downstream evidence")
    rl_by_case = unique(read_tsv(RL_INPUT), "case_id", "RL input")
    partner_by_source = unique(read_tsv(PARTNER), "source_case_id", "partner manifest")
    paired_by_case = unique(read_tsv(PAIRED), "case_id", "paired RL input")

    exported_sets = {
        "source_audit": set(source_by_case),
        "evidence": set(evidence_by_case),
        "rl_input": set(rl_by_case),
        "partner": set(partner_by_source),
        "paired": set(paired_by_case),
    }
    set_pass = all(case_set == approved for case_set in exported_sets.values())
    record(
        "approved_set_equality",
        set_pass,
        {name: len(case_set) for name, case_set in exported_sets.items()},
    )
    rejected_entered = sorted(set().union(*exported_sets.values()) & rejected)
    record("rejected_zero_entry", not rejected_entered, rejected_entered)

    source_failures: list[str] = []
    source_path_fields = {
        "scene_act": "scene_act_sha256",
        "trajectory": "trajectory_sha256",
        "contact_mask": "contact_mask_sha256",
        "cem_result_npz": "cem_result_sha256",
        "cem_video": "cem_video_sha256",
        "config_act": "config_act_sha256",
    }
    alignment_fields = ("scene_act", "trajectory", "contact_mask", "result_npz", "video", "config_act")
    for case_id, source in source_by_case.items():
        if source["manual_use_decision"] != "USE" or source["spider_method_id"] != METHOD_ID:
            source_failures.append(f"{case_id}:authority_or_method")
        for field, sha_field in source_path_fields.items():
            passed, detail = check_file(source[field], source[sha_field])
            if not passed:
                source_failures.append(f"{case_id}:{field}:{detail}")
        analysis = analysis_by_case[case_id]
        source_to_analysis = {
            "scene_act": "scene_act",
            "trajectory": "trajectory",
            "contact_mask": "contact_mask",
            "result_npz": "cem_result_npz",
            "video": "cem_video",
            "config_act": "config_act",
        }
        for analysis_field in alignment_fields:
            source_field = source_to_analysis[analysis_field]
            if resolve_path(analysis[analysis_field]) != resolve_path(source[source_field]):
                source_failures.append(f"{case_id}:analysis_alignment:{analysis_field}")
    record("source_paths_sha_and_analysis_alignment", not source_failures, source_failures)

    rl_failures: list[str] = []
    for case_id, row in rl_by_case.items():
        if row["rl_export_decision"] != "RL_EXPORT_READY":
            rl_failures.append(f"{case_id}:decision={row['rl_export_decision']}")
        if row["source_exp_id"] != "E170" or row["spider_method_id"] != METHOD_ID:
            rl_failures.append(f"{case_id}:provenance")
        for field in ("scene_act", "trajectory", "contact_mask", "cem_result_npz"):
            passed, detail = check_file(row[field])
            if not passed:
                rl_failures.append(f"{case_id}:{field}:{detail}")
    record("rl_ready_contract", not rl_failures, rl_failures)

    evidence_failures = [
        case_id
        for case_id, row in evidence_by_case.items()
        if row["cem_status"] != "pass"
        or row["downstream_decision"] != "DOWNSTREAM_CEM_PASS"
        or row["source_exp_id"] != "E170"
        or row["spider_method_id"] != METHOD_ID
    ]
    record("downstream_evidence_contract", not evidence_failures, evidence_failures)

    partner_failures: list[str] = []
    partner_sha_fields = {
        "converted_npz": "converted_sha256",
        "omniretarget_output_npz": "omniretarget_output_sha256",
        "retargeted_npz": "retargeted_sha256",
        "trimmed_npz": "trimmed_sha256",
        "trim_window_json": "trim_window_sha256",
    }
    for source_case, row in partner_by_source.items():
        if row["partner_case_id"] != partner_case_id(source_case):
            partner_failures.append(f"{source_case}:opposite_person")
        if row["date"] not in source_case or f"_{row['seq']}_" not in source_case:
            partner_failures.append(f"{source_case}:sequence")
        if row["partner_status"] != "pass" or row["pair_status"] != "PAIR_COMPLETE":
            partner_failures.append(f"{source_case}:status")
        if row["partner_stage2b_status"] != "pass" or row["generation_mode"] != "reuse_e168_stage2b_partner":
            partner_failures.append(f"{source_case}:stage2b_provenance")
        for field, sha_field in partner_sha_fields.items():
            passed, detail = check_file(row[field], row[sha_field])
            if not passed:
                partner_failures.append(f"{source_case}:{field}:{detail}")
        passed, detail = check_file(row["partner_verify_summary"])
        if not passed:
            partner_failures.append(f"{source_case}:verify:{detail}")
    record("partner_pairing_paths_and_sha", not partner_failures, partner_failures)

    paired_failures = [
        case_id
        for case_id, row in paired_by_case.items()
        if row["paired_rl_export_decision"] != "RL_EXPORT_READY"
        or row["rl_export_decision"] != "RL_EXPORT_READY"
        or row["pair_status"] != "PAIR_COMPLETE"
        or row["partner_case_id"] != partner_case_id(case_id)
    ]
    record("paired_rl_ready_contract", not paired_failures, paired_failures)

    manifest_paths = [AUTHORITY, ANALYSIS, SOURCE_AUDIT, EVIDENCE, RL_INPUT, PARTNER, PAIRED]
    payload = {
        "experiment": "E170",
        "stage": "S6_paired_rl_export_independent_audit",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": "pass" if all(check["pass"] for check in checks) else "fail",
        "checks_passed": sum(check["pass"] for check in checks),
        "checks_total": len(checks),
        "approved_rows": len(approved),
        "rejected_rows": len(rejected),
        "partner_variant_counts": dict(
            sorted(
                {
                    variant: sum(row["retarget_variant_id"] == variant for row in partner_by_source.values())
                    for variant in {row["retarget_variant_id"] for row in partner_by_source.values()}
                }.items()
            )
        ),
        "checks": checks,
        "manifest_sha256": {str(path.relative_to(REPO)): digest(path) for path in manifest_paths},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if payload["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
