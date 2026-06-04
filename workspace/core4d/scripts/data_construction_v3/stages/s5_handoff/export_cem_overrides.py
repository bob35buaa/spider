#!/usr/bin/env python3
"""Build reproducible CEM override configs from the v3 S5 handoff manifest."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
import sys

SCRIPT_ROOT = next(parent for parent in Path(__file__).resolve().parents if parent.name == "data_construction_v3")
for _path in (SCRIPT_ROOT / "lib", SCRIPT_ROOT / "state", SCRIPT_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
from typing import Any

import numpy as np

from common import SCHEMA_VERSION, find_spider_repo, json_dumps, read_tsv, sha256_file, timestamp, write_json, write_tsv


FIELDS = [
    "case_id",
    "retarget_variant_id",
    "target_variant_id",
    "handoff_decision",
    "candidate_decision",
    "target_task",
    "override_status",
    "override_config",
    "target_adapter_status",
    "target_adapter_failure",
    "contact_target_source",
    "contact_target_path",
    "contact_target_sha256",
    "contact_target_key",
    "contact_target_shape",
    "contact_target_active_shape",
    "contact_mask_source",
    "contact_mask_path",
    "contact_mask_status",
    "contact_mask_label",
    "contact_mask_person_idx",
    "contact_mask_time_axis",
    "base_override",
    "schema_version",
    "updated_at",
]


def safe_id(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", text).strip("_")


def rel_to_repo(path: Path, repo: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo.resolve()))
    except ValueError:
        return str(path.resolve())


def validate_external_target(path_text: str, repo: Path) -> dict[str, str]:
    if not path_text:
        return {
            "target_adapter_status": "fail",
            "target_adapter_failure": "missing_contact_target_path",
            "contact_target_sha256": "",
            "contact_target_key": "",
            "contact_target_shape": "",
            "contact_target_active_shape": "",
        }
    path = Path(path_text)
    if not path.is_absolute():
        path = repo / path
    if not path.is_file():
        return {
            "target_adapter_status": "fail",
            "target_adapter_failure": "contact_target_file_missing",
            "contact_target_sha256": "",
            "contact_target_key": "",
            "contact_target_shape": "",
            "contact_target_active_shape": "",
        }
    try:
        data = np.load(path, allow_pickle=True)
        key = ""
        if "spider_contact_target_object_local" in data:
            key = "spider_contact_target_object_local"
        elif "eval_contact_target_object_local" in data:
            key = "eval_contact_target_object_local"
        else:
            return {
                "target_adapter_status": "fail",
                "target_adapter_failure": "contact_target_key_missing",
                "contact_target_sha256": sha256_file(path),
                "contact_target_key": "",
                "contact_target_shape": "",
                "contact_target_active_shape": "",
            }
        arr = np.asarray(data[key])
        if arr.ndim != 3 or arr.shape[1:] != (2, 3) or not np.isfinite(arr).all():
            return {
                "target_adapter_status": "fail",
                "target_adapter_failure": "contact_target_shape_or_finite_fail",
                "contact_target_sha256": sha256_file(path),
                "contact_target_key": key,
                "contact_target_shape": str(list(arr.shape)),
                "contact_target_active_shape": "",
            }
        active_shape = ""
        if "active" in data:
            active_shape = str(list(np.asarray(data["active"]).shape))
        elif "active_mask" in data:
            active_shape = str(list(np.asarray(data["active_mask"]).shape))
        return {
            "target_adapter_status": "pass",
            "target_adapter_failure": "",
            "contact_target_sha256": sha256_file(path),
            "contact_target_key": key,
            "contact_target_shape": str(list(arr.shape)),
            "contact_target_active_shape": active_shape,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "target_adapter_status": "fail",
            "target_adapter_failure": f"contact_target_load_error:{type(exc).__name__}",
            "contact_target_sha256": sha256_file(path),
            "contact_target_key": "",
            "contact_target_shape": "",
            "contact_target_active_shape": "",
        }


def validate_contact_mask(path_text: str, label: str, repo: Path) -> tuple[bool, str]:
    if not path_text:
        return False, "missing_contact_mask_path"
    path = Path(path_text)
    if not path.is_absolute():
        path = repo / path
    if not path.is_file() or path.stat().st_size <= 0:
        return False, "contact_mask_file_missing"
    try:
        data = np.load(path, allow_pickle=True)
    except Exception as exc:  # noqa: BLE001
        return False, f"contact_mask_load_error:{type(exc).__name__}"
    expected = f"raw_contact_mask_{label}" if label else ""
    accepted = [expected, f"method_contact_mask_{label}", f"contact_mask_{label}", "contact_mask", "method_contact_mask"]
    if not any(key and key in data for key in accepted):
        return False, "contact_mask_key_missing"
    return True, ""


def should_generate(row: dict[str, str]) -> bool:
    return row.get("handoff_decision") in {"HANDOFF_READY", "HANDOFF_REVIEW_VISUAL_QC"} and row.get("target_gate_status") == "pass"


def yaml_quote(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def write_override(
    path: Path,
    row: dict[str, str],
    base_override: str,
    contact_source: str,
    target_path: str,
    mask_source: str,
    mask_path: str,
    mask_time_axis: str,
) -> None:
    task = row.get("stage2b_target_task") or row.get("target_task") or row.get("task") or row.get("case_id")
    person_idx = row.get("person_idx", "")
    mask_source_yaml = yaml_quote(mask_source) if mask_source else '""'
    mask_path_yaml = yaml_quote(mask_path) if mask_path else '""'
    lines = [
        "# @package _global_",
        "# Auto-generated by workspace/core4d/scripts/data_construction_v3/stages/s5_handoff/export_cem_overrides.py.",
        f"# case_id: {row.get('case_id', '')}",
        f"# route: {row.get('retarget_variant_id', '')}/{row.get('target_variant_id', '')}",
        "defaults:",
        f"  - {base_override}",
        "  - _self_",
        "",
        f"task: {task}",
        "",
    ]
    if contact_source == "external":
        lines.extend(
            [
                "contact_hdmi_dynamic_target: true",
                "contact_hdmi_target_source: external",
                f"contact_hdmi_target_path: {target_path}",
                "contact_hdmi_target_time_axis: auto",
                "contact_hdmi_target_uses_eef_offset: false",
            ]
        )
    else:
        lines.extend(
            [
                "contact_hdmi_target_source: ref_fk",
                'contact_hdmi_target_path: ""',
                "contact_hdmi_target_uses_eef_offset: true",
            ]
        )
    lines.extend(
        [
            f"contact_hdmi_mask_source: {mask_source_yaml}",
            f"contact_hdmi_mask_path: {mask_path_yaml}",
            f"contact_hdmi_mask_person_idx: {person_idx}",
            f"contact_hdmi_mask_time_axis: {yaml_quote(mask_time_axis or 'auto')}",
            "video_camera: auto",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def build_rows(handoff_rows: list[dict[str, str]], out_dir: Path, repo: Path, base_override: str) -> list[dict[str, Any]]:
    config_dir = out_dir / "overrides"
    rows: list[dict[str, Any]] = []
    for row in handoff_rows:
        target_variant = row.get("target_variant_id", "ref_fk") or "ref_fk"
        target_task = row.get("stage2b_target_task") or row.get("target_task") or row.get("case_id", "")
        config_path = config_dir / f"core4d_dcv3_{safe_id(row.get('retarget_variant_id', ''))}_{safe_id(target_variant)}_{safe_id(row.get('case_id', ''))}.yaml"
        contact_source = "ref_fk" if target_variant == "ref_fk" else "external"
        target_path = row.get("target_npz", "")
        mask_path = row.get("contact_mask_npz") or row.get("contact_mask", "")
        mask_label = row.get("contact_mask_label") or row.get("raw_contact_threshold_label", "")
        mask_source = f"core4d_{mask_label}" if mask_path and mask_label else ("core4d_contact_mask" if mask_path else "")
        mask_time_axis = row.get("contact_mask_time_axis", "") or "auto"
        adapter = {
            "target_adapter_status": "skipped",
            "target_adapter_failure": "",
            "contact_target_sha256": "",
            "contact_target_key": "",
            "contact_target_shape": "",
            "contact_target_active_shape": "",
        }
        override_status = "skipped"
        if should_generate(row):
            if contact_source == "external":
                adapter = validate_external_target(target_path, repo)
                override_status = "pass" if adapter["target_adapter_status"] == "pass" else "fail"
            else:
                adapter["target_adapter_status"] = "pass"
                override_status = "pass"
            if mask_path:
                mask_ok, mask_failure = validate_contact_mask(mask_path, mask_label, repo)
                if not mask_ok:
                    adapter["target_adapter_status"] = "fail"
                    adapter["target_adapter_failure"] = mask_failure
                    override_status = "fail"
            if override_status == "pass":
                yaml_target_path = rel_to_repo((repo / target_path) if target_path and not Path(target_path).is_absolute() else Path(target_path), repo) if target_path else ""
                yaml_mask_path = rel_to_repo((repo / mask_path) if mask_path and not Path(mask_path).is_absolute() else Path(mask_path), repo) if mask_path else ""
                write_override(config_path, row, base_override, contact_source, yaml_target_path, mask_source, yaml_mask_path, mask_time_axis)
        rows.append(
            {
                "case_id": row.get("case_id", ""),
                "retarget_variant_id": row.get("retarget_variant_id", ""),
                "target_variant_id": target_variant,
                "handoff_decision": row.get("handoff_decision", ""),
                "candidate_decision": row.get("candidate_decision", ""),
                "target_task": target_task,
                "override_status": override_status,
                "override_config": str(config_path) if override_status == "pass" else "",
                "target_adapter_status": adapter["target_adapter_status"],
                "target_adapter_failure": adapter["target_adapter_failure"],
                "contact_target_source": contact_source,
                "contact_target_path": target_path,
                "contact_target_sha256": adapter["contact_target_sha256"],
                "contact_target_key": adapter["contact_target_key"],
                "contact_target_shape": adapter["contact_target_shape"],
                "contact_target_active_shape": adapter["contact_target_active_shape"],
                "contact_mask_source": mask_source,
                "contact_mask_path": mask_path,
                "contact_mask_status": row.get("contact_mask_status", ""),
                "contact_mask_label": mask_label,
                "contact_mask_person_idx": row.get("contact_mask_person_idx", row.get("person_idx", "")),
                "contact_mask_time_axis": mask_time_axis,
                "base_override": base_override,
                "schema_version": SCHEMA_VERSION,
                "updated_at": timestamp(),
            }
        )
    rows.sort(key=lambda item: (item["override_status"] != "pass", item["case_id"], item["retarget_variant_id"], item["target_variant_id"]))
    return rows


def summarize(rows: list[dict[str, Any]], out_dir: Path) -> dict[str, Any]:
    return {
        "stage": "S5_cem_override_handoff",
        "created_at": timestamp(),
        "schema_version": SCHEMA_VERSION,
        "rows": len(rows),
        "override_status_counts": dict(Counter(row["override_status"] for row in rows)),
        "target_adapter_status_counts": dict(Counter(row["target_adapter_status"] for row in rows)),
        "out_dir": str(out_dir),
    }


def markdown_summary(summary: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    lines = [
        "# CEM override handoff summary",
        "",
        f"- rows: `{summary['rows']}`",
        "",
        "## override status",
        "",
        "| status | count |",
        "|---|---:|",
    ]
    for key, count in summary["override_status_counts"].items():
        lines.append(f"| `{key}` | {count} |")
    lines.extend(["", "## generated overrides", "", "| case | route | status | config |", "|---|---|---|---|"])
    for row in rows:
        if row["override_status"] == "pass":
            lines.append(f"| `{row['case_id']}` | `{row['retarget_variant_id']}/{row['target_variant_id']}` | `{row['override_status']}` | `{row['override_config']}` |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--handoff-manifest-tsv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--spider-repo", type=Path, default=None)
    parser.add_argument("--base-override", default="core4d_E089A_box021_person1_upperobj")
    args = parser.parse_args()

    repo = (args.spider_repo or find_spider_repo()).resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = build_rows(read_tsv(args.handoff_manifest_tsv), out_dir, repo, args.base_override)
    write_tsv(out_dir / "cem_override_manifest.tsv", rows, FIELDS)
    write_json(out_dir / "cem_override_manifest.json", rows)
    summary = summarize(rows, out_dir)
    write_json(out_dir / "cem_override_summary.json", summary)
    (out_dir / "cem_override_summary.md").write_text(markdown_summary(summary, rows), encoding="utf-8")
    print(json_dumps(summary))


if __name__ == "__main__":
    main()
