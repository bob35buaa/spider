#!/usr/bin/env python3
"""Build E114 RL handoff gate artifacts from E113 Pareto decisions."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[4]
DEFAULT_E113 = REPO / "workspace/core4d/results/E113/cem/full/pareto_decisions.tsv"
DEFAULT_OUT = REPO / "workspace/core4d/results/E114/rl_handoff_gate"


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def write_table(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def as_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def as_float(row: dict[str, str], field: str) -> float:
    raw = row.get(field, "")
    return float(raw) if raw not in {"", None} else 0.0


def handoff_ready(row: dict[str, str]) -> bool:
    return bool(
        row.get("pareto_decision") == "release_candidate"
        and row.get("hold_strict_status") == "WORK"
        and as_bool(row.get("contact_ok", "False"))
        and as_bool(row.get("penetration_ok", "False"))
        and as_bool(row.get("lowerbody_ok", "False"))
        and as_bool(row.get("object_ok", "False"))
        and row.get("phase_scope") == "phaseA_release_candidate"
    )


def diagnostic_queue(row: dict[str, str]) -> tuple[str, str]:
    contact_ok = as_bool(row.get("contact_ok", "False"))
    lowerbody_ok = as_bool(row.get("lowerbody_ok", "False"))
    strict = row.get("hold_strict_status") == "WORK"
    contact_delta = as_float(row, "delta_contact_frac_either")
    physics_delta = as_float(row, "delta_physics_contact")

    if contact_ok and not lowerbody_ok:
        return "lowerbody_aware_contact", "contact improved but lower-body strict gate failed"
    if strict and not contact_ok:
        return "strict_contact_margin", "strict WORK but contact improvement is below E114 threshold"
    if not lowerbody_ok:
        return "lowerbody_repair", "lower-body strict gate failed before RL handoff"
    if contact_delta < 0.0:
        return "contact_target_repair", "geometric contact decreased under hold-band"
    if physics_delta < 0.08:
        return "contact_target_repair", "physics-contact gain is below E114 threshold"
    return "review", "manual review required"


def build(rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    export_rows: list[dict[str, Any]] = []
    blocked_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []

    for row in rows:
        ready = handoff_ready(row)
        queue, reason = ("rl_ready", "all E114 criteria passed") if ready else diagnostic_queue(row)
        out = {
            "variant": row.get("variant", ""),
            "case_id": row.get("case_id", ""),
            "object_key": row.get("object_key", ""),
            "phase_scope": row.get("phase_scope", ""),
            "source_task": row.get("source_task", ""),
            "derived_task": row.get("derived_task", ""),
            "hold_npz_path": row.get("hold_npz_path", ""),
            "hold_video_path": row.get("hold_video_path", ""),
            "baseline_npz_path": row.get("baseline_npz_path", ""),
            "hold_strict_status": row.get("hold_strict_status", ""),
            "pareto_decision": row.get("pareto_decision", ""),
            "rl_handoff_ready": ready,
            "diagnostic_queue": queue,
            "handoff_block_reason": reason,
            "delta_contact_frac_either": row.get("delta_contact_frac_either", ""),
            "delta_physics_contact": row.get("delta_physics_contact", ""),
            "delta_deep_penetration_2cm": row.get("delta_deep_penetration_2cm", ""),
            "hold_leg_interference_frac": row.get("hold_leg_interference_frac", ""),
            "contact_ok": row.get("contact_ok", ""),
            "penetration_ok": row.get("penetration_ok", ""),
            "lowerbody_ok": row.get("lowerbody_ok", ""),
            "object_ok": row.get("object_ok", ""),
        }
        if ready:
            export_rows.append(out)
        else:
            blocked_rows.append(out)
            diagnostic_rows.append(out)
    return export_rows, blocked_rows, diagnostic_rows


def write_summary(
    out_dir: Path,
    source: Path,
    rows: list[dict[str, str]],
    export_rows: list[dict[str, Any]],
    blocked_rows: list[dict[str, Any]],
    diagnostic_rows: list[dict[str, Any]],
) -> None:
    queue_counts = Counter(row["diagnostic_queue"] for row in diagnostic_rows)
    status_counts = Counter(row.get("hold_strict_status", "") for row in rows)
    summary = {
        "stage": "E114_contact_alignment_rl_handoff_gate",
        "source_pareto_decisions": rel(source),
        "total_rows": len(rows),
        "rl_ready_rows": len(export_rows),
        "blocked_rows": len(blocked_rows),
        "diagnostic_queue_counts": dict(queue_counts),
        "hold_strict_status_counts": dict(status_counts),
        "outputs": {
            "rl_export_list": rel(out_dir / "rl_export_list.tsv"),
            "blocked_candidates": rel(out_dir / "blocked_candidates.tsv"),
            "diagnostic_queues": rel(out_dir / "diagnostic_queues.tsv"),
            "no_handoff_manifest": rel(out_dir / "no_handoff_manifest.tsv"),
        },
    }
    (out_dir / "rl_handoff_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    lines = [
        "# E114 Contact Alignment RL Handoff Gate",
        "",
        f"- source: `{summary['source_pareto_decisions']}`",
        f"- evaluated rows: `{len(rows)}`",
        f"- RL-ready rows: `{len(export_rows)}`",
        f"- blocked rows: `{len(blocked_rows)}`",
        "",
        "## Diagnostic Queues",
        "",
        "| queue | count |",
        "|---|---:|",
    ]
    for key, count in sorted(queue_counts.items()):
        lines.append(f"| `{key}` | {count} |")
    lines.extend(
        [
            "",
            "## Blocked Rows",
            "",
            "| case | object | strict | contact delta | physics delta | leg hold | queue | reason |",
            "|---|---|---|---:|---:|---:|---|---|",
        ]
    )
    for row in blocked_rows:
        lines.append(
            f"| `{row['case_id']}` | `{row['object_key']}` | `{row['hold_strict_status']}` | "
            f"{float(row['delta_contact_frac_either']) * 100:+.1f}% | "
            f"{float(row['delta_physics_contact']) * 100:+.1f}% | "
            f"{float(row['hold_leg_interference_frac']) * 100:.1f}% | "
            f"`{row['diagnostic_queue']}` | {row['handoff_block_reason']} |"
        )
    (out_dir / "rl_handoff_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pareto-decisions", type=Path, default=DEFAULT_E113)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    source = args.pareto_decisions if args.pareto_decisions.is_absolute() else REPO / args.pareto_decisions
    out_dir = args.out_dir if args.out_dir.is_absolute() else REPO / args.out_dir
    rows = read_tsv(source)
    export_rows, blocked_rows, diagnostic_rows = build(rows)

    fields = [
        "variant",
        "case_id",
        "object_key",
        "phase_scope",
        "source_task",
        "derived_task",
        "hold_npz_path",
        "hold_video_path",
        "baseline_npz_path",
        "hold_strict_status",
        "pareto_decision",
        "rl_handoff_ready",
        "diagnostic_queue",
        "handoff_block_reason",
        "delta_contact_frac_either",
        "delta_physics_contact",
        "delta_deep_penetration_2cm",
        "hold_leg_interference_frac",
        "contact_ok",
        "penetration_ok",
        "lowerbody_ok",
        "object_ok",
    ]
    write_table(out_dir / "rl_export_list.tsv", export_rows, fields)
    write_table(out_dir / "blocked_candidates.tsv", blocked_rows, fields)
    write_table(out_dir / "diagnostic_queues.tsv", diagnostic_rows, fields)
    write_table(out_dir / "no_handoff_manifest.tsv", blocked_rows, fields)
    write_summary(out_dir, source, rows, export_rows, blocked_rows, diagnostic_rows)
    print(f"wrote {rel(out_dir / 'rl_handoff_summary.md')}")


if __name__ == "__main__":
    main()
