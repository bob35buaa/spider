#!/usr/bin/env python3
"""Write a v3-compatible manual case-state seed TSV template."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

SCRIPT_ROOT = next(parent for parent in Path(__file__).resolve().parents if parent.name == "data_construction_v3")
for _path in (SCRIPT_ROOT / "lib", SCRIPT_ROOT / "state", SCRIPT_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from common import SCHEMA_VERSION, timestamp, write_json, write_tsv
from update_case_state_registry import FIELDS


FIELD_HELP = {
    "case_id": "必填；规范化 case id，例如 box026_20231018_039_p2。",
    "object_key": "物体 key，例如 box026。",
    "object_name": "CORE4D/OmniRetarget 物体名。",
    "date": "CORE4D 日期目录。",
    "seq": "CORE4D sequence id。",
    "person": "person1 或 person2。",
    "person_idx": "person1=0, person2=1。",
    "raw_inventory_status": "present / invalid_raw / not_seen。",
    "raw_contact_3cm_status": "pass / reject / review / not_run。",
    "raw_contact_5cm_status": "pass / reject / review / not_run。",
    "template_status": "clean / backlog / audit_fail / manual_review_required / not_run。",
    "retarget_variant_id": "S3 之后必填；共享上游状态可用 shared。",
    "stage2b_status": "pass / omniretarget_infeasible / preprocess_fail / not_run。",
    "target_variant_id": "ref_fk / adaptive / fingertip_aware。",
    "target_gate_status": "pass / review / reject / not_run。",
    "visual_qc_status": "pass / review / reject / not_run。",
    "cem_status": "pass / fail / not_run / not_required。",
    "rl_status": "pass / fail / not_run / not_required。",
    "downstream_decision": "DOWNSTREAM_*；只作为 S6 下游证据。",
    "current_decision": "可留空，import 时会按 registry 规则重算。",
    "evidence_root": "必填于任何非 not_run 状态；应指向可复查证据目录。",
    "source_type": "manual_seed。",
    "source_ref": "人工 seed 的来源说明，例如 E106_final_eval。",
    "schema_version": SCHEMA_VERSION,
    "updated_at": "可留空，import 时会更新。",
    "notes": "人工说明；记录为什么该状态可信。",
}


def example_row(evidence_root: str) -> dict[str, str]:
    row = {field: "" for field in FIELDS}
    row.update(
        {
            "case_id": "EXAMPLE_REMOVE_ME_box004_20231003_2_082_p1",
            "object_key": "box004",
            "object_name": "box004",
            "date": "20231003",
            "seq": "2_082",
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
            "current_decision": "",
            "evidence_root": evidence_root,
            "source_type": "manual_seed",
            "source_ref": "manual_seed_template_example",
            "schema_version": SCHEMA_VERSION,
            "updated_at": timestamp(),
            "notes": "EXAMPLE ROW; delete before import.",
        }
    )
    return row


def markdown_help() -> str:
    lines = [
        "# Manual seed TSV template",
        "",
        "用途：把已经人工确认可信的 case 状态导入 v3 registry。",
        "",
        "导入命令：",
        "",
        "```bash",
        "workspace/core4d/scripts/data_construction_v3/migration/import_legacy_snapshot.py \\",
        "  --input-tsv <manual_seed.tsv> \\",
        "  --out-dir \"$RUN_DIR/imported_snapshots\" \\",
        "  --snapshot-id <snapshot_id> \\",
        "  --source-type manual_seed \\",
        "  --source-ref <source_note>",
        "```",
        "",
        "非 `not_run` 状态必须有 `evidence_root`。`current_decision` 可以留空，导入时会重算。",
        "",
        "## 字段说明",
        "",
        "| field | 说明 |",
        "|---|---|",
    ]
    for field in FIELDS:
        lines.append(f"| `{field}` | {FIELD_HELP.get(field, '')} |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-tsv", type=Path, required=True)
    parser.add_argument("--include-example", action="store_true")
    parser.add_argument("--example-evidence-root", default="")
    parser.add_argument("--write-help-md", action="store_true")
    args = parser.parse_args()

    rows = [example_row(args.example_evidence_root)] if args.include_example else []
    out_tsv = args.out_tsv.expanduser().resolve()
    write_tsv(out_tsv, rows, FIELDS)
    metadata = {
        "stage": "write_manual_seed_template",
        "created_at": timestamp(),
        "schema_version": SCHEMA_VERSION,
        "template_tsv": str(out_tsv),
        "fields": FIELDS,
        "field_help": FIELD_HELP,
        "include_example": args.include_example,
    }
    write_json(out_tsv.with_suffix(".json"), metadata)
    if args.write_help_md:
        out_tsv.with_suffix(".md").write_text(markdown_help(), encoding="utf-8")
    print(f"wrote {out_tsv}")


if __name__ == "__main__":
    main()
