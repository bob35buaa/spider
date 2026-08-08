#!/usr/bin/env python3
"""Independent audit of the E190 38-case noPRG RL export.

Verifies: case_id set parity with the PRG-side authority, artifact existence +
sha256 integrity, a per-row PRG-negative config_act check, partner parity, and
RL_EXPORT_READY completeness.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from e190_common import (  # noqa: E402
    RL_DIR,
    load_authority,
    now,
    read_tsv,
    repo_path,
    rel,
    require_file,
    sha256,
    write_json,
)

CONFIG_SCALAR = re.compile(r"^(?P<key>[a-zA-Z0-9_]+):\s*(?P<value>.*?)\s*$")


def read_config_flags(config_path: str) -> dict[str, str]:
    path = repo_path(config_path)
    flags: dict[str, str] = {}
    wanted = {
        "scene_name",
        "leg_object_penalty_scale",
        "leg_object_penalty_geom_names",
        "cem_leg_gate_enabled",
    }
    for line in path.read_text(encoding="utf-8").splitlines():
        match = CONFIG_SCALAR.match(line)
        if match and match.group("key") in wanted:
            flags[match.group("key")] = match.group("value")
    return flags


def prg_negative_ok(flags: dict[str, str]) -> tuple[bool, str]:
    scene_name = flags.get("scene_name", "")
    if re.search(r"lowerbody_physics|_PRG", scene_name, re.IGNORECASE):
        return False, f"scene_name looks PRG-patched: {scene_name}"
    scale = flags.get("leg_object_penalty_scale", "")
    if scale not in ("0.0", "0", ""):
        return False, f"leg_object_penalty_scale={scale}"
    geoms = flags.get("leg_object_penalty_geom_names", "")
    if geoms not in ("[]", ""):
        return False, f"leg_object_penalty_geom_names={geoms}"
    gate = flags.get("cem_leg_gate_enabled", "")
    if gate not in ("false", "False", ""):
        return False, f"cem_leg_gate_enabled={gate}"
    return True, ""


def audit_case_id_parity(authority: dict[str, list[str]], combined_rows: list[dict[str, str]]) -> dict[str, Any]:
    expected = {cid for ids in authority.values() for cid in ids}
    actual = {row["case_id"] for row in combined_rows}
    return {
        "expected_count": len(expected),
        "actual_count": len(actual),
        "missing_from_export": sorted(expected - actual),
        "unexpected_in_export": sorted(actual - expected),
        "pass": expected == actual and len(expected) == 38,
    }


def audit_artifacts(combined_rows: list[dict[str, str]]) -> dict[str, Any]:
    failures = []
    for row in combined_rows:
        for field in ("scene_act", "trajectory", "contact_mask", "cem_result_npz", "cem_video"):
            value = row.get(field, "")
            if not value:
                failures.append(f"{row['case_id']}: empty {field}")
                continue
            path = repo_path(value)
            if not path.is_file() or path.stat().st_size == 0:
                failures.append(f"{row['case_id']}: missing {field} -> {value}")
    return {"failures": failures, "pass": not failures}


def audit_source_row_hashes(authority: dict[str, list[str]]) -> dict[str, Any]:
    failures = []
    checked = 0
    hash_field_for = {
        "scene_act": "scene_act_sha256",
        "trajectory": "trajectory_sha256",
        "contact_mask": "contact_mask_sha256",
        "cem_result_npz": "cem_result_sha256",
    }
    for object_key in authority:
        source_rows = read_tsv(RL_DIR / f"{object_key}_noPRG_user_approved" / f"{object_key}_noprg_source_rows.tsv")
        for row in source_rows:
            checked += 1
            for field, hash_field in hash_field_for.items():
                recorded = row.get(hash_field, "")
                if not recorded:
                    failures.append(f"{row['case_id']}: no recorded sha256 for {field}")
                    continue
                actual = sha256(row[field])
                if actual != recorded:
                    failures.append(f"{row['case_id']}: {field} sha256 mismatch")
    return {"checked_rows": checked, "failures": failures, "pass": not failures}


def audit_prg_negative(authority: dict[str, list[str]]) -> dict[str, Any]:
    failures = []
    checked = 0
    for object_key in authority:
        scope_dir = RL_DIR / f"{object_key}_noPRG_user_approved"
        config_by_case = {row["case_id"]: row["config_act"] for row in read_tsv(scope_dir / f"{object_key}_noprg_config_act.tsv")}
        for case_id, config_act in config_by_case.items():
            checked += 1
            require_file(config_act, f"{case_id} config_act")
            flags = read_config_flags(config_act)
            ok, reason = prg_negative_ok(flags)
            if not ok:
                failures.append(f"{case_id}: {reason}")
    return {"checked_rows": checked, "failures": failures, "pass": not failures}


def audit_partner_parity(authority: dict[str, list[str]]) -> dict[str, Any]:
    failures = []
    checked = 0
    for object_key, case_ids in authority.items():
        paired_rows = {row["case_id"]: row for row in read_tsv(RL_DIR / f"{object_key}_noPRG_user_approved" / "paired_rl_export_input.tsv")}
        for case_id in case_ids:
            checked += 1
            row = paired_rows.get(case_id)
            if not row:
                failures.append(f"{case_id}: missing paired row")
                continue
            if row.get("pair_status") != "PAIR_COMPLETE":
                failures.append(f"{case_id}: pair_status={row.get('pair_status')}")
            if row.get("paired_rl_export_decision") != "RL_EXPORT_READY":
                failures.append(f"{case_id}: paired_rl_export_decision={row.get('paired_rl_export_decision')}")
            for field in ("partner_converted_npz", "partner_retargeted_npz", "partner_trimmed_npz"):
                value = row.get(field, "")
                if not value or not repo_path(value).is_file():
                    failures.append(f"{case_id}: partner artifact missing -> {field}={value}")
    return {"checked_rows": checked, "failures": failures, "pass": not failures}


def audit_rl_ready_counts(authority: dict[str, list[str]], combined_rows: list[dict[str, str]]) -> dict[str, Any]:
    not_ready = [row["case_id"] for row in combined_rows if row.get("rl_export_decision") != "RL_EXPORT_READY"]
    per_object_ok = True
    per_object: dict[str, Any] = {}
    for object_key, case_ids in authority.items():
        rows = read_tsv(RL_DIR / f"{object_key}_noPRG_user_approved" / "rl_export_input.tsv")
        ready = sum(1 for row in rows if row["rl_export_decision"] == "RL_EXPORT_READY")
        per_object[object_key] = {"expected": len(case_ids), "ready": ready}
        per_object_ok = per_object_ok and ready == len(case_ids) == len(rows)
    return {
        "combined_not_ready": not_ready,
        "combined_total": len(combined_rows),
        "per_object": per_object,
        "pass": not not_ready and len(combined_rows) == 38 and per_object_ok,
    }


def main() -> int:
    authority = load_authority()
    combined_rows = read_tsv(require_file(RL_DIR / "rl_export_input.tsv", "combined rl_export_input.tsv"))

    checks = {
        "case_id_parity": audit_case_id_parity(authority, combined_rows),
        "artifact_existence": audit_artifacts(combined_rows),
        "source_row_hash_integrity": audit_source_row_hashes(authority),
        "prg_negative_config_check": audit_prg_negative(authority),
        "partner_parity": audit_partner_parity(authority),
        "rl_ready_counts": audit_rl_ready_counts(authority, combined_rows),
    }
    overall_pass = all(check["pass"] for check in checks.values())

    audit = {
        "experiment": "E190",
        "created_at": now(),
        "scope": "38-case noPRG counterpart of the frozen PRG-side RL export",
        "status": "pass" if overall_pass else "fail",
        "checks": checks,
        "known_provenance_notes": [
            "box021's 2 bridge cases (box021_20231018_029_p2, box021_20231011_035_p1) were never run "
            "under PRG at all; approved via the pre-PRG E167/E167A method as USER_APPROVED_BRIDGE_OVERRIDE "
            "(see workspace/core4d/exp_analysis_0726.md). Their noPRG row is a relabeled copy of the same "
            "E167 evidence already used on the PRG side, not a fresh ablation.",
            "box021's 9 standard cases source noPRG CEM evidence from E168 (pre-dates the PRG concept, "
            "introduced later in E169/E170); box001/box004/box024 source from E189; box023 sources from "
            "E179. No unified E179/E189-style paired ablation was run for box021 specifically.",
            "No fresh manual video review was performed for any of the 38 noPRG rows; approval is "
            "inherited from the PRG-side approved case_id list, per explicit user decision.",
        ],
    }
    write_json(RL_DIR / "E190_38case_noprg_rl_export_audit.json", audit)
    print(f"status={audit['status']}")
    for name, check in checks.items():
        print(f"  {name}: pass={check['pass']}")
    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
