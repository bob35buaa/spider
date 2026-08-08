#!/usr/bin/env python3
"""Build the E190 paired (source + partner) RL export.

Partner OmniRetarget motion generation does not depend on the P/R/G lower-body CEM
treatment (it only depends on which human performed the same task), so E190 reuses
the PRG-side partner_omnirt artifacts verbatim instead of regenerating them:
  - box004: E172's box004_user_approved/partner_omnirt
  - box001/box023/box024: E173's respective <object>_user_approved/partner_omnirt
  - box021 (9 standard cases): E170's partner_omnirt (18-row file, subset by case_id)
  - box021 (2 bridge cases): E167's own partner_omnirt (long-form case_id, renamed)
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from e190_common import (  # noqa: E402
    BOX021_BRIDGE_CASE_IDS,
    E167_PARTNER_MANIFEST,
    PRG_PARTNER_MANIFEST,
    RL_DIR,
    load_authority,
    now,
    read_tsv,
    rel,
    require_file,
    sha256,
    write_json,
    write_tsv,
)

PARTNER_FIELDS = [
    "source_case_id", "source_rl_export_decision", "source_handoff_decision",
    "source_cem_status", "source_person", "source_person_idx", "partner_case_id",
    "partner_person", "partner_person_idx", "object_key", "object_name",
    "object_model_rel", "date", "seq", "retarget_variant_id", "target_variant_id",
    "generation_mode", "pipeline_enabled", "partner_status", "failure_mode",
    "decision_notes", "partner_target_task", "result_root", "holosoma_case_root",
    "converted_npz", "omniretarget_output_npz", "retargeted_npz", "trimmed_npz",
    "trim_window_json", "trim_start", "trim_end", "trim_frames", "untrimmed_frames",
    "trimmed_frames", "raw_window_start_frame", "raw_window_end_frame",
    "raw_window_num_frames", "case_file", "run_script", "command_line",
    "replace_wrist_with_fingertip", "source_rl_export_input", "schema_version",
    "updated_at", "pair_status", "partner_stage2b_status",
    "partner_stage2b_manifest_ref", "partner_verify_summary", "solver_family",
    "solver_version", "solver_repo_path", "solver_git_sha", "solver_dirty",
    "converter_script", "converter_git_sha", "params_json", "converted_sha256",
    "omniretarget_output_sha256", "retargeted_sha256", "trimmed_sha256",
    "trim_window_sha256",
]

PAIRED_PARTNER_FIELDS = [
    "pair_status", "partner_case_id", "partner_person", "partner_person_idx",
    "partner_retarget_variant_id", "partner_target_variant_id", "partner_status",
    "partner_generation_mode", "partner_converted_npz", "partner_omniretarget_output_npz",
    "partner_retargeted_npz", "partner_trimmed_npz", "partner_trim_window_json",
    "partner_trim_start", "partner_trim_end", "partner_trim_frames",
    "partner_stage2b_manifest_ref", "partner_verify_summary", "partner_solver_family",
    "partner_solver_version", "partner_solver_git_sha", "partner_converter_script",
    "partner_converter_git_sha", "partner_params_json", "partner_converted_sha256",
    "partner_retargeted_sha256", "partner_trimmed_sha256", "partner_trim_window_sha256",
    "paired_rl_export_decision",
]

HASH_FIELD_SOURCE = {
    "converted_sha256": "converted_npz",
    "omniretarget_output_sha256": "omniretarget_output_npz",
    "retargeted_sha256": "retargeted_npz",
    "trimmed_sha256": "trimmed_npz",
    "trim_window_sha256": "trim_window_json",
}


def load_prg_partner_by_case(object_key: str) -> dict[str, dict[str, str]]:
    path = PRG_PARTNER_MANIFEST[object_key]
    rows = read_tsv(require_file(path, f"{object_key} PRG partner manifest"))
    return {row["source_case_id"]: row for row in rows}


def load_e167_partner_by_short_case(case_ids) -> dict[str, dict[str, str]]:
    rows = read_tsv(require_file(E167_PARTNER_MANIFEST, "E167 partner manifest"))
    by_long_id = {row["source_case_id"]: row for row in rows}
    output: dict[str, dict[str, str]] = {}
    for case_id in case_ids:
        long_id = f"d003_{case_id}__E167_E167A"
        row = by_long_id.get(long_id)
        if not row:
            raise SystemExit(f"E167 partner manifest missing bridge case: {long_id}")
        output[case_id] = row
    return output


def build_partner_row(case_id: str, source_row: dict[str, str], template: dict[str, str], origin: str) -> dict[str, Any]:
    # E167's partner manifest predates several PARTNER_FIELDS (pair_status, solver_*,
    # converter_*, params_json, sha256 columns); start from an all-fields-present base
    # so older/shorter templates don't KeyError on the newer schema.
    row: dict[str, Any] = {field: "" for field in PARTNER_FIELDS}
    row.update(template)
    row.setdefault("pair_status", "")
    if not row["pair_status"] and row.get("partner_status") == "pass":
        row["pair_status"] = "PAIR_COMPLETE"
    if not row.get("partner_stage2b_status"):
        row["partner_stage2b_status"] = row.get("partner_status", "")
    row["source_case_id"] = case_id
    row["source_rl_export_decision"] = source_row["rl_export_decision"]
    row["source_handoff_decision"] = source_row["handoff_decision"]
    row["source_cem_status"] = source_row["cem_status"]
    row["source_person"] = source_row["person"]
    row["source_person_idx"] = source_row["person_idx"]
    row["source_rl_export_input"] = rel(RL_DIR / "rl_export_input.tsv")
    row["schema_version"] = "core4d_data_construction_v3.0"
    row["updated_at"] = now()
    row["decision_notes"] = (
        f"E190 noPRG counterpart; partner artifacts reused verbatim from {origin} PRG-side "
        "partner_omnirt (partner motion generation is independent of the P/R/G lower-body "
        "CEM treatment)"
        + (
            "; source E167 manifest predates pair_status/solver_*/converter_*/params_json "
            "columns, backfilled from partner_status"
            if origin == "E167"
            else ""
        )
    )
    for hash_field, path_field in HASH_FIELD_SOURCE.items():
        path_value = row.get(path_field, "")
        row[hash_field] = sha256(path_value) if path_value else ""
    return row


def paired_row_from(source_row: dict[str, str], partner_row: dict[str, Any]) -> dict[str, Any]:
    return {
        **source_row,
        "pair_status": partner_row["pair_status"],
        "partner_case_id": partner_row["partner_case_id"],
        "partner_person": partner_row["partner_person"],
        "partner_person_idx": partner_row["partner_person_idx"],
        "partner_retarget_variant_id": partner_row["retarget_variant_id"],
        "partner_target_variant_id": partner_row["target_variant_id"],
        "partner_status": partner_row["partner_status"],
        "partner_generation_mode": partner_row["generation_mode"],
        "partner_converted_npz": partner_row["converted_npz"],
        "partner_omniretarget_output_npz": partner_row["omniretarget_output_npz"],
        "partner_retargeted_npz": partner_row["retargeted_npz"],
        "partner_trimmed_npz": partner_row["trimmed_npz"],
        "partner_trim_window_json": partner_row["trim_window_json"],
        "partner_trim_start": partner_row["trim_start"],
        "partner_trim_end": partner_row["trim_end"],
        "partner_trim_frames": partner_row["trim_frames"],
        "partner_stage2b_manifest_ref": partner_row["partner_stage2b_manifest_ref"],
        "partner_verify_summary": partner_row["partner_verify_summary"],
        "partner_solver_family": partner_row["solver_family"],
        "partner_solver_version": partner_row["solver_version"],
        "partner_solver_git_sha": partner_row["solver_git_sha"],
        "partner_converter_script": partner_row["converter_script"],
        "partner_converter_git_sha": partner_row["converter_git_sha"],
        "partner_params_json": partner_row["params_json"],
        "partner_converted_sha256": partner_row["converted_sha256"],
        "partner_retargeted_sha256": partner_row["retargeted_sha256"],
        "partner_trimmed_sha256": partner_row["trimmed_sha256"],
        "partner_trim_window_sha256": partner_row["trim_window_sha256"],
        "paired_rl_export_decision": "RL_EXPORT_READY",
    }


def process_object(object_key: str, case_ids: list[str]) -> list[dict[str, Any]]:
    scope_dir = RL_DIR / f"{object_key}_noPRG_user_approved"
    rl_rows = {row["case_id"]: row for row in read_tsv(scope_dir / "rl_export_input.tsv")}

    templates: dict[str, tuple[dict[str, str], str]] = {}
    if object_key == "box021":
        standard_ids = sorted(set(case_ids) - BOX021_BRIDGE_CASE_IDS)
        prg_partner = load_prg_partner_by_case("box021")
        for case_id in standard_ids:
            if case_id not in prg_partner:
                raise SystemExit(f"box021: missing E170 PRG partner row for {case_id}")
            templates[case_id] = (prg_partner[case_id], "E170")
        bridge_partner = load_e167_partner_by_short_case(BOX021_BRIDGE_CASE_IDS)
        for case_id in BOX021_BRIDGE_CASE_IDS:
            templates[case_id] = (bridge_partner[case_id], "E167")
    else:
        origin = {"box004": "E172", "box001": "E173", "box023": "E173", "box024": "E173"}[object_key]
        prg_partner = load_prg_partner_by_case(object_key)
        for case_id in case_ids:
            if case_id not in prg_partner:
                raise SystemExit(f"{object_key}: missing PRG partner row for {case_id}")
            templates[case_id] = (prg_partner[case_id], origin)

    partner_dir = scope_dir / "partner_omnirt"
    partner_rows: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    for case_id in sorted(case_ids):
        template, origin = templates[case_id]
        source_row = rl_rows[case_id]
        partner_row = build_partner_row(case_id, source_row, template, origin)
        partner_rows.append(partner_row)
        paired_rows.append(paired_row_from(source_row, partner_row))

    write_tsv(partner_dir / "rl_partner_omnirt_manifest.tsv", partner_rows, PARTNER_FIELDS)
    write_json(partner_dir / "rl_partner_omnirt_manifest.json", partner_rows)
    write_tsv(
        partner_dir / "cases_rl_partner_omnirt.tsv",
        [
            {
                "# enabled": "0",
                "source_case_id": row["source_case_id"],
                "partner_case_id": row["partner_case_id"],
                "generation_mode": row["generation_mode"],
            }
            for row in partner_rows
        ],
        ["# enabled", "source_case_id", "partner_case_id", "generation_mode"],
    )
    run_script = partner_dir / "run_rl_partner_omnirt.sh"
    run_script.write_text(
        f"#!/usr/bin/env bash\nset -euo pipefail\n"
        f"echo 'All {len(partner_rows)} partner OmniRetarget artifacts are reused from the PRG-side "
        f"export; no execution required.'\n",
        encoding="utf-8",
    )
    run_script.chmod(0o755)

    partner_summary = {
        "stage": "S6_rl_partner_omnirt",
        "experiment": "E190",
        "object_key": object_key,
        "created_at": now(),
        "schema_version": "core4d_data_construction_v3.0",
        "rows": len(partner_rows),
        "executed": False,
        "partner_status_counts": dict(Counter(row["partner_status"] for row in partner_rows)),
        "pair_status_counts": dict(Counter(row["pair_status"] for row in partner_rows)),
        "generation_mode_counts": dict(Counter(row["generation_mode"] for row in partner_rows)),
    }
    write_json(partner_dir / "rl_partner_omnirt_summary.json", partner_summary)
    lines = [
        f"# E190 {object_key} noPRG partner OmniRetarget",
        "",
        f"- rows: `{len(partner_rows)}`",
        f"- status: `{len(partner_rows)}/{len(partner_rows)} pass` (reused, not regenerated)",
        "",
        "| source | partner | variant | status |",
        "|---|---|---|---|",
    ]
    for row in partner_rows:
        lines.append(f"| `{row['source_case_id']}` | `{row['partner_case_id']}` | `{row['retarget_variant_id']}` | `{row['partner_status']}` |")
    (partner_dir / "rl_partner_omnirt_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    paired_fields = [*list(rl_rows[case_ids[0]].keys()), *PAIRED_PARTNER_FIELDS]
    write_tsv(scope_dir / "paired_rl_export_input.tsv", paired_rows, paired_fields)
    write_json(scope_dir / "paired_rl_export_input.json", paired_rows)
    return paired_rows


def main() -> int:
    authority = load_authority()
    all_paired: list[dict[str, Any]] = []
    for object_key, case_ids in authority.items():
        all_paired.extend(process_object(object_key, case_ids))

    if len(all_paired) != 38:
        raise SystemExit(f"expected 38 combined paired rows, got {len(all_paired)}")
    fields = list(all_paired[0].keys())
    write_tsv(RL_DIR / "paired_rl_export_input.tsv", all_paired, fields)
    write_json(RL_DIR / "paired_rl_export_input.json", all_paired)
    print(
        f"wrote {rel(RL_DIR / 'paired_rl_export_input.tsv')} rows={len(all_paired)} "
        f"pair_status={dict(Counter(row['pair_status'] for row in all_paired))}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
