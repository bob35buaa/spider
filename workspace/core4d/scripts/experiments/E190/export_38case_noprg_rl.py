#!/usr/bin/env python3
"""Run the S6 export_rl_inputs.py pipeline per object and assemble the E190 38-case
noPRG rl_export_input.tsv/json (combined + per-object).

Reuses the generic stage scripts (record_downstream_evidence.py, export_rl_inputs.py)
rather than reimplementing their join/decide() logic.
"""

from __future__ import annotations

import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from e190_common import (  # noqa: E402
    REPO,
    RL_DIR,
    RUN_ROOT,
    S6_SCRIPTS,
    SPIDER_METHOD_ID,
    load_authority,
    now,
    read_tsv,
    require_file,
    rel,
    sha256,
    write_json,
    write_tsv,
)

AUDIT_FIELDS = [
    "case_id", "object_key", "retarget_variant_id", "target_variant_id",
    "hand_collision_variant_id", "source_exp_id", "spider_method_id",
    "scene_act", "scene_act_sha256", "trajectory", "trajectory_sha256",
    "contact_mask", "contact_mask_sha256", "cem_result_npz", "cem_result_sha256",
    "cem_video", "cem_video_sha256", "config_act", "config_act_sha256",
    "prg_authority_note",
]


def run(command: list[str]) -> None:
    subprocess.run([str(item) for item in command], cwd=REPO, check=True)


def export_object(object_key: str, expected_rows: int) -> list[dict[str, Any]]:
    scope_dir = RL_DIR / f"{object_key}_noPRG_user_approved"
    evidence_dir = RUN_ROOT / f"s6_downstream/evidence/{object_key}_noPRG_user_approved"
    handoff_path = require_file(
        scope_dir / f"{object_key}_noprg_handoff_manifest.tsv", f"{object_key} noPRG handoff manifest"
    )
    evidence_input_path = require_file(
        evidence_dir / "downstream_evidence_input.tsv", f"{object_key} noPRG evidence input"
    )

    run([
        sys.executable, S6_SCRIPTS / "record_downstream_evidence.py",
        "--handoff-manifest-tsv", handoff_path,
        "--evidence-tsv", evidence_input_path,
        "--out-dir", evidence_dir,
        "--evidence-root", RUN_ROOT,
        "--source-ref", f"E190_{object_key}_noprg_counterpart_of_PRG_approved_38case",
    ])
    evidence_manifest = evidence_dir / "downstream_evidence_manifest.tsv"
    run([
        sys.executable, S6_SCRIPTS / "export_rl_inputs.py",
        "--handoff-manifest-tsv", handoff_path,
        "--cem-evidence-tsv", evidence_manifest,
        "--out-dir", scope_dir,
        "--spider-repo", REPO,
    ])

    rl_rows = read_tsv(scope_dir / "rl_export_input.tsv")
    if len(rl_rows) != expected_rows:
        raise SystemExit(f"{object_key}: expected {expected_rows} RL export rows, got {len(rl_rows)}")
    if any(row["rl_export_decision"] != "RL_EXPORT_READY" for row in rl_rows):
        not_ready = [row["case_id"] for row in rl_rows if row["rl_export_decision"] != "RL_EXPORT_READY"]
        raise SystemExit(f"{object_key}: rows not RL_EXPORT_READY: {not_ready}")
    if any(row["spider_method_id"] != SPIDER_METHOD_ID for row in rl_rows):
        raise SystemExit(f"{object_key}: spider_method_id drift from {SPIDER_METHOD_ID}")
    for row in rl_rows:
        for field in ("scene_act", "trajectory", "contact_mask", "cem_result_npz"):
            require_file(row[field], f"{row['case_id']} {field}")

    config_by_case = {
        row["case_id"]: row["config_act"]
        for row in read_tsv(scope_dir / f"{object_key}_noprg_config_act.tsv")
    }
    source_audit: list[dict[str, Any]] = []
    for row in rl_rows:
        config_act = config_by_case.get(row["case_id"], "")
        entry = {
            "case_id": row["case_id"],
            "object_key": row["object_key"],
            "retarget_variant_id": row["retarget_variant_id"],
            "target_variant_id": row["target_variant_id"],
            "hand_collision_variant_id": row["hand_collision_variant_id"],
            "source_exp_id": row["source_exp_id"],
            "spider_method_id": row["spider_method_id"],
            "scene_act": row["scene_act"],
            "scene_act_sha256": sha256(row["scene_act"]),
            "trajectory": row["trajectory"],
            "trajectory_sha256": sha256(row["trajectory"]),
            "contact_mask": row["contact_mask"],
            "contact_mask_sha256": sha256(row["contact_mask"]),
            "cem_result_npz": row["cem_result_npz"],
            "cem_result_sha256": sha256(row["cem_result_npz"]),
            "cem_video": row["cem_video"],
            "cem_video_sha256": sha256(row["cem_video"]) if row["cem_video"] else "",
            "config_act": config_act,
            "config_act_sha256": sha256(config_act) if config_act else "",
            "prg_authority_note": "case_id matches PRG-side approved export; no fresh manual review performed",
        }
        source_audit.append(entry)
    write_tsv(scope_dir / f"{object_key}_noprg_source_rows.tsv", source_audit, AUDIT_FIELDS)

    summary = {
        "experiment": "E190",
        "object_key": object_key,
        "created_at": now(),
        "scope": f"{object_key} noPRG counterpart of PRG-approved case set",
        "authority": "same case_id set as PRG-side <object>_user_approved export; no new manual review",
        "rows": len(rl_rows),
        "rl_export_decision_counts": dict(Counter(row["rl_export_decision"] for row in rl_rows)),
        "spider_method_id": SPIDER_METHOD_ID,
    }
    write_json(scope_dir / f"{object_key}_rl_export_summary.json", summary)
    lines = [
        f"# E190 {object_key} noPRG RL export",
        "",
        f"- authority: same case_id set as PRG-side `{object_key}_user_approved` export",
        "- manual review: none performed for noPRG; automated gate/status carried from source experiment",
        f"- RL_EXPORT_READY: `{len(rl_rows)}/{len(rl_rows)}`",
        f"- spider_method_id: `{SPIDER_METHOD_ID}`",
        "",
        "| case | source_exp_id | scene_act | cem_result_npz |",
        "|---|---|---|---|",
    ]
    for row in rl_rows:
        lines.append(f"| `{row['case_id']}` | `{row['source_exp_id']}` | `{row['scene_act']}` | `{row['cem_result_npz']}` |")
    (scope_dir / f"{object_key}_rl_export_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rl_rows


def main() -> int:
    authority = load_authority()
    all_rows: list[dict[str, Any]] = []
    for object_key, case_ids in authority.items():
        rows = export_object(object_key, len(case_ids))
        all_rows.extend(rows)

    if len(all_rows) != 38:
        raise SystemExit(f"expected 38 combined rows, got {len(all_rows)}")
    fields = list(all_rows[0].keys())
    write_tsv(RL_DIR / "rl_export_input.tsv", all_rows, fields)
    write_json(RL_DIR / "rl_export_input.json", all_rows)

    summary = {
        "experiment": "E190",
        "created_at": now(),
        "scope": "38-case noPRG counterpart of the frozen PRG-side RL export",
        "rows": len(all_rows),
        "rl_export_decision_counts": dict(Counter(row["rl_export_decision"] for row in all_rows)),
        "object_row_counts": dict(Counter(row["object_key"] for row in all_rows)),
        "source_exp_id_counts": dict(Counter(row["source_exp_id"] for row in all_rows)),
        "spider_method_id": SPIDER_METHOD_ID,
    }
    write_json(RL_DIR / "rl_export_summary.json", summary)
    lines = [
        "# E190 38-case noPRG RL export",
        "",
        f"- rows: `{len(all_rows)}`",
        f"- RL_EXPORT_READY: `{summary['rl_export_decision_counts'].get('RL_EXPORT_READY', 0)}/38`",
        f"- object counts: `{summary['object_row_counts']}`",
        f"- source_exp_id counts: `{summary['source_exp_id_counts']}`",
        "",
        "说明：本表是与 PRG 侧 38-case rl_export_input.tsv 完全同 case_id 集合的 noPRG 对照版本；未做新的人工 review，"
        "审批依据沿用 PRG 侧已批准 case 清单，自动化门控/状态沿用来源实验（E189/E179/E168/E167）。",
        "",
    ]
    (RL_DIR / "rl_export_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {rel(RL_DIR / 'rl_export_input.tsv')} rows={len(all_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
