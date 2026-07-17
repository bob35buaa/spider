#!/usr/bin/env python3
"""Build gated E168 E167A canary and production CEM manifests."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def safe_id(value: str) -> str:
    return "".join(char if char.isalnum() or char == "_" else "_" for char in value)


def write_tsv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def choose_canaries(rows: list[dict[str, Any]]) -> list[str]:
    predicates = [
        ("bucket_v1", lambda row: row["object_key"] == "bucket004" and row["retarget_variant_id"] == "omnirt_v1"),
        ("v2_rescue", lambda row: row["retarget_variant_id"] == "omnirt_v2"),
        ("obs1", lambda row: row["source_obstacle_level"] == "obs1" and row["retarget_variant_id"] == "omnirt_v1"),
        ("obs3", lambda row: row["source_obstacle_level"] == "obs3" and row["retarget_variant_id"] == "omnirt_v1"),
    ]
    selected = []
    used = set()
    for label, predicate in predicates:
        match = next((row for row in rows if row["case_id"] not in used and predicate(row)), None)
        if not match:
            raise SystemExit(f"cannot select required canary category: {label}")
        selected.append(match["case_id"])
        used.add(match["case_id"])
    return selected


def production_sort_key(row: dict[str, Any]) -> tuple[int, int, str]:
    object_priority = {"box021": 0, "box004": 1, "bucket004": 2}
    return (
        object_priority.get(row["object_key"], 99),
        int(row["ordinal"]),
        row["case_id"],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--handoff-manifest-tsv", type=Path, required=True)
    parser.add_argument("--override-install-manifest-tsv", type=Path, required=True)
    parser.add_argument("--config-audit-tsv", type=Path, required=True)
    parser.add_argument("--recall-tsv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    handoff_rows = read_tsv(args.handoff_manifest_tsv.expanduser().resolve())
    overrides = {
        (row["case_id"], row["retarget_variant_id"], row["target_variant_id"]): row
        for row in read_tsv(args.override_install_manifest_tsv.expanduser().resolve())
    }
    audits = {
        (row["case_id"], row["retarget_variant_id"], row["target_variant_id"]): row
        for row in read_tsv(args.config_audit_tsv.expanduser().resolve())
    }
    recall = {row["case_id"]: row for row in read_tsv(args.recall_tsv.expanduser().resolve())}
    rows: list[dict[str, Any]] = []
    for ordinal, source in enumerate(sorted(handoff_rows, key=lambda row: row["case_id"]), 1):
        key = (source["case_id"], source["retarget_variant_id"], source["target_variant_id"])
        override = overrides[key]
        audit = audits[key]
        recall_row = recall[source["case_id"]]
        if source["handoff_decision"] != "HANDOFF_READY" or audit["audit_status"] != "pass":
            raise SystemExit(f"CEM row not gated: {key}")
        variant = safe_id(f"E168_{source['case_id']}_E167A")
        rows.append(
            {
                "ordinal": ordinal,
                "variant": variant,
                "case_id": source["case_id"],
                "sequence_key": recall_row["sequence_key"],
                "source_person": recall_row["source_person"],
                "object_key": source["object_key"],
                "source_action": recall_row["source_action"],
                "source_obstacle_level": recall_row["source_obstacle_level"],
                "candidate_tier": recall_row["candidate_tier"],
                "obstacle_context_not_reconstructed": recall_row["obstacle_context_not_reconstructed"],
                "retarget_variant_id": source["retarget_variant_id"],
                "target_variant_id": source["target_variant_id"],
                "hand_collision_variant_id": source["hand_collision_variant_id"],
                "spider_method_id": "E167A_zOnlyBody",
                "target_task": source["stage2b_target_task"],
                "target_scene": source["target_scene"],
                "trajectory": source["trajectory"],
                "scene_act": source["scene_act"],
                "scene_name": source["scene_name"],
                "contact_mask": source["contact_mask_npz"],
                "contact_mask_label": source["contact_mask_label"],
                "override_id": override["override_id"],
                "override_path": override["installed_override"],
                "config_audit_status": audit["audit_status"],
                "execution_mode": "production",
                "execution_decision": "ready_pending_gpu_recheck",
                "preferred_pool": "a6000" if source["object_key"] in {"box004", "bucket004"} else "a100_then_a6000",
                "gpu_id": "",
                "status": "not_run",
                "failure_mode": "",
                "retry_of": "",
                "result_npz": f"workspace/core4d/results/E168/s6_downstream/cem/full/{variant}.npz",
                "outdir_npz": f"workspace/core4d/results/E168/s6_downstream/cem/full/{variant}_outdir_full/trajectory_mjwp_act.npz",
                "config_act": f"workspace/core4d/results/E168/s6_downstream/cem/full/{variant}_outdir_full/config_act.yaml",
                "video": f"workspace/core4d/results/E168/s6_downstream/cem/full/{variant}_full.mp4",
                "log": f"logs/E168/cem/full/{variant}.log",
                "updated_at": now(),
            }
        )
    if len(rows) != 40:
        raise SystemExit(f"expected 40 production CEM rows, got {len(rows)}")
    rows.sort(key=production_sort_key)
    for ordinal, row in enumerate(rows, 1):
        row["ordinal"] = ordinal
        row["preferred_pool"] = "a100"

    canary_ids = choose_canaries(rows)
    canary_rows = []
    for row in rows:
        if row["case_id"] not in canary_ids:
            continue
        canary = dict(row)
        canary["variant"] = f"{row['variant']}_canary"
        canary["execution_mode"] = "canary"
        canary["execution_decision"] = "ready_pending_local_gpu_confirmation"
        canary["preferred_pool"] = "local-gpu0"
        canary["result_npz"] = f"workspace/core4d/results/E168/s6_downstream/cem/canary/{canary['variant']}.npz"
        canary["outdir_npz"] = f"workspace/core4d/results/E168/s6_downstream/cem/canary/{canary['variant']}_outdir_smoke/trajectory_mjwp_act.npz"
        canary["config_act"] = f"workspace/core4d/results/E168/s6_downstream/cem/canary/{canary['variant']}_outdir_smoke/config_act.yaml"
        canary["video"] = f"workspace/core4d/results/E168/s6_downstream/cem/canary/{canary['variant']}_smoke.mp4"
        canary["log"] = f"logs/E168/cem/canary/{canary['variant']}.log"
        canary_rows.append(canary)

    fields = list(rows[0].keys())
    out_dir = args.out_dir.expanduser().resolve()
    write_tsv(out_dir / "cem_production_manifest.tsv", rows, fields)
    write_tsv(out_dir / "cem_canary_manifest.tsv", canary_rows, fields)
    summary = {
        "created_at": now(),
        "status": "pass",
        "production_rows": len(rows),
        "canary_rows": len(canary_rows),
        "canary_case_ids": canary_ids,
        "canary_excluded_from_production_yield": True,
        "e167_imported_reference_canary": "box004_20231003_2_082_p1 (existing E167 artifacts; reference-only, not scheduled)",
        "retarget_variant_counts": dict(Counter(row["retarget_variant_id"] for row in rows)),
        "object_counts": dict(Counter(row["object_key"] for row in rows)),
        "obstacle_counts": dict(Counter(row["source_obstacle_level"] for row in rows)),
        "execution_decision_counts": dict(Counter(row["execution_decision"] for row in rows)),
        "out_dir": str(out_dir),
    }
    (out_dir / "cem_execution_manifest_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
