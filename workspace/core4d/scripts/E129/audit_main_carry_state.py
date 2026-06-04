#!/usr/bin/env python3
"""Audit main-case carry-state constraints from existing E113/E119-E124 evidence."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[4]
MAIN_CASE = "box021_029_p2"
OUT_DIR = REPO / "workspace/core4d/results/E129/main_carry_state_constraint_audit"

INPUTS = [
    ("E113", "full", REPO / "workspace/core4d/results/E113/cem/full/full_method_metrics.csv"),
    ("E119", "smoke", REPO / "workspace/core4d/results/E119/cem/smoke/smoke_method_metrics.csv"),
    ("E120", "smoke", REPO / "workspace/core4d/results/E120/cem/smoke/smoke_method_metrics.csv"),
    ("E121", "smoke", REPO / "workspace/core4d/results/E121/cem/smoke/smoke_method_metrics.csv"),
    ("E122", "smoke", REPO / "workspace/core4d/results/E122/cem/smoke/smoke_method_metrics.csv"),
    ("E123", "smoke", REPO / "workspace/core4d/results/E123/cem/smoke/smoke_method_metrics.csv"),
    ("E124", "smoke", REPO / "workspace/core4d/results/E124/cem/smoke/smoke_method_metrics.csv"),
]

CANDIDATE_FIELDS = [
    "experiment",
    "stage",
    "case_id",
    "variant",
    "ablation",
    "method",
    "work_status",
    "decision_label",
    "rank_score",
    "physics_contact_frac",
    "lowerbody_contact_frac",
    "nonhand_support_frac",
    "hand_near_zero_frac",
    "deep_pen_frac",
    "pelvis_min_m",
    "obj_err_mean_m",
    "object_floor_contact_frac",
    "npz_path",
    "video_path",
    "legobj_timeseries_csv",
    "support_decomp_timeseries_csv",
]

FRAME_FIELDS = [
    "experiment",
    "stage",
    "variant",
    "ablation",
    "frames",
    "full_constraint_evidence",
    "hand_contact_or_near_frac",
    "hand_physics_contact_frac",
    "hand_near_zero_frac",
    "hand_deep_clean_frac",
    "lowerbody_clean_frac",
    "nonhand_clean_frac",
    "object_floor_clean_frac",
    "clean_carry_frame_frac",
    "clean_carry_longest_run_frames",
    "clean_carry_longest_run_frac",
    "primary_frame_blocker",
    "legobj_timeseries_csv",
    "support_decomp_timeseries_csv",
]


def rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return str(p.relative_to(REPO))
    except ValueError:
        return str(p)


def repo_path(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else REPO / path


def safe_float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    raw = row.get(key, "")
    if raw in ("", None):
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return default if math.isnan(value) or math.isinf(value) else value


def safe_int(row: dict[str, str], key: str, default: int = 0) -> int:
    return int(round(safe_float(row, key, float(default))))


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_tsv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def longest_true_run(values: list[bool]) -> int:
    best = 0
    cur = 0
    for value in values:
        if value:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def frac(values: list[bool]) -> float:
    return sum(1 for value in values if value) / len(values) if values else 0.0


def classify_candidate(row: dict[str, Any]) -> str:
    if row["strict_rl_gate"]:
        return "STRICT_GATE_PASS"
    if row["physics_contact_frac"] < 0.57:
        return "CONTACT_BELOW_RL_GATE"
    if row["lowerbody_contact_frac"] > 0.05:
        return "LOWERBODY_SUPPORT_BLOCK"
    if row["nonhand_support_frac"] > 0.05:
        return "NONHAND_SUPPORT_BLOCK"
    if row["pelvis_min_m"] < 0.55:
        return "POSTURE_BLOCK"
    if row["obj_err_mean_m"] > 0.02:
        return "OBJECT_ERROR_BLOCK"
    if row["deep_pen_frac"] > 0.03:
        return "DEEP_PENETRATION_BLOCK"
    return "GLOBAL_GATE_INCOMPLETE"


def candidate_rows() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for experiment, stage, path in INPUTS:
        for row in read_csv(path):
            if row.get("case_id") != MAIN_CASE:
                continue
            if row.get("method") != row.get("ablation"):
                continue
            physics = safe_float(row, "hand_object_physics_contact")
            if physics == 0.0:
                physics = safe_float(row, "hand_object_contact_physics_frac")
            cand: dict[str, Any] = {
                "experiment": experiment,
                "stage": stage,
                "case_id": row.get("case_id", ""),
                "variant": row.get("variant", ""),
                "ablation": row.get("ablation", ""),
                "method": row.get("method", ""),
                "work_status": row.get("work_status", ""),
                "physics_contact_frac": physics,
                "lowerbody_contact_frac": safe_float(row, "leg_object_contact_frac"),
                "nonhand_support_frac": safe_float(row, "nonhand_object_support_frac"),
                "hand_near_zero_frac": safe_float(row, "hand_support_near_zero_frac"),
                "deep_pen_frac": safe_float(row, "hand_geom_deep_penetration_2cm"),
                "pelvis_min_m": safe_float(row, "pelvis_min_m"),
                "obj_err_mean_m": safe_float(row, "obj_err_mean_m"),
                "object_floor_contact_frac": safe_float(row, "object_floor_contact_frac"),
                "npz_path": row.get("npz_path", ""),
                "video_path": row.get("video_path", ""),
                "legobj_timeseries_csv": row.get("legobj_timeseries_csv", ""),
                "support_decomp_timeseries_csv": row.get("support_decomp_timeseries_csv", ""),
            }
            posture_gap = max(0.0, 0.55 - cand["pelvis_min_m"])
            cand["rank_score"] = (
                1.5 * cand["physics_contact_frac"]
                + cand["hand_near_zero_frac"]
                - 2.0 * cand["lowerbody_contact_frac"]
                - 2.0 * cand["nonhand_support_frac"]
                - 2.0 * cand["deep_pen_frac"]
                - posture_gap
            )
            cand["strict_rl_gate"] = (
                cand["physics_contact_frac"] >= 0.57
                and cand["lowerbody_contact_frac"] <= 0.05
                and cand["nonhand_support_frac"] <= 0.05
                and cand["hand_near_zero_frac"] >= 0.50
                and cand["deep_pen_frac"] <= 0.03
                and cand["pelvis_min_m"] >= 0.55
                and cand["obj_err_mean_m"] <= 0.02
                and cand["work_status"] == "PASS"
            )
            cand["decision_label"] = classify_candidate(cand)
            out.append(cand)
    out.sort(key=lambda r: (r["rank_score"], r["physics_contact_frac"]), reverse=True)
    return out


def load_legobj(path: str) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    if not path:
        return rows
    for row in read_csv(repo_path(path)):
        frame = safe_int(row, "frame", -1)
        if frame >= 0:
            rows[frame] = {
                "hand_object_contact_count": safe_float(row, "hand_object_contact_count"),
                "leg_object_contact_count": safe_float(row, "leg_object_contact_count"),
                "object_floor_contact_count": safe_float(row, "object_floor_contact_count"),
                "hand_box_sdf_min_m": safe_float(row, "hand_box_sdf_min_m", 999.0),
                "leg_box_sdf_min_m": safe_float(row, "leg_box_sdf_min_m", 999.0),
            }
    return rows


def load_support(path: str) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    if not path:
        return rows
    for row in read_csv(repo_path(path)):
        frame = safe_int(row, "frame", -1)
        if frame >= 0:
            rows[frame] = {
                "nonhand_support_sdf_min_m": safe_float(row, "nonhand_support_sdf_min_m", 999.0),
                "hand_support_sdf_min_m": safe_float(row, "hand_support_sdf_min_m", 999.0),
            }
    return rows


def frame_summary(cand: dict[str, Any]) -> dict[str, Any]:
    leg = load_legobj(str(cand.get("legobj_timeseries_csv", "")))
    support = load_support(str(cand.get("support_decomp_timeseries_csv", "")))
    frames = sorted(set(leg) | set(support))
    if not frames:
        return {
            "experiment": cand["experiment"],
            "stage": cand["stage"],
            "variant": cand["variant"],
            "ablation": cand["ablation"],
            "frames": 0,
            "full_constraint_evidence": False,
            "primary_frame_blocker": "missing_timeseries",
            "legobj_timeseries_csv": cand.get("legobj_timeseries_csv", ""),
            "support_decomp_timeseries_csv": cand.get("support_decomp_timeseries_csv", ""),
        }

    has_leg = bool(leg)
    has_support = bool(support)
    hand_physics: list[bool] = []
    hand_near: list[bool] = []
    hand_contact_or_near: list[bool] = []
    hand_deep_clean: list[bool] = []
    lower_clean: list[bool] = []
    nonhand_clean: list[bool] = []
    floor_clean: list[bool] = []
    clean_carry: list[bool] = []

    for frame in frames:
        lrow = leg.get(frame, {})
        srow = support.get(frame, {})
        physics = lrow.get("hand_object_contact_count", 0.0) > 0.0
        near = min(
            float(lrow.get("hand_box_sdf_min_m", 999.0)),
            float(srow.get("hand_support_sdf_min_m", 999.0)),
        ) <= 0.01
        deep_ok = float(lrow.get("hand_box_sdf_min_m", 999.0)) >= -0.02
        lower_ok = (
            lrow.get("leg_object_contact_count", 0.0) <= 0.0
            and float(lrow.get("leg_box_sdf_min_m", 999.0)) > 0.0
        )
        nonhand_ok = float(srow.get("nonhand_support_sdf_min_m", 999.0)) > 0.02
        floor_ok = lrow.get("object_floor_contact_count", 0.0) <= 0.0
        hand_physics.append(physics)
        hand_near.append(near)
        hand_contact_or_near.append(physics or near)
        hand_deep_clean.append(deep_ok)
        lower_clean.append(lower_ok if has_leg else False)
        nonhand_clean.append(nonhand_ok if has_support else False)
        floor_clean.append(floor_ok if has_leg else False)
        clean_carry.append((physics or near) and deep_ok and lower_ok and nonhand_ok and floor_ok)

    blocker_fracs = {
        "hand_contact_or_near_missing": 1.0 - frac(hand_contact_or_near),
        "hand_deep_not_clean": 1.0 - frac(hand_deep_clean),
        "lowerbody_not_clean": 1.0 - frac(lower_clean),
        "nonhand_not_clean": 1.0 - frac(nonhand_clean),
        "object_floor_not_clean": 1.0 - frac(floor_clean),
    }
    if not has_support:
        primary = "missing_nonhand_support_timeseries"
    elif not has_leg:
        primary = "missing_leg_object_timeseries"
    else:
        primary = max(blocker_fracs.items(), key=lambda item: item[1])[0]

    longest = longest_true_run(clean_carry) if has_leg and has_support else 0
    return {
        "experiment": cand["experiment"],
        "stage": cand["stage"],
        "variant": cand["variant"],
        "ablation": cand["ablation"],
        "frames": len(frames),
        "full_constraint_evidence": has_leg and has_support,
        "hand_contact_or_near_frac": frac(hand_contact_or_near),
        "hand_physics_contact_frac": frac(hand_physics),
        "hand_near_zero_frac": frac(hand_near),
        "hand_deep_clean_frac": frac(hand_deep_clean),
        "lowerbody_clean_frac": frac(lower_clean),
        "nonhand_clean_frac": frac(nonhand_clean),
        "object_floor_clean_frac": frac(floor_clean),
        "clean_carry_frame_frac": frac(clean_carry) if has_leg and has_support else "",
        "clean_carry_longest_run_frames": longest,
        "clean_carry_longest_run_frac": longest / len(frames) if frames else 0.0,
        "primary_frame_blocker": primary,
        "legobj_timeseries_csv": cand.get("legobj_timeseries_csv", ""),
        "support_decomp_timeseries_csv": cand.get("support_decomp_timeseries_csv", ""),
    }


def write_summary(candidates: list[dict[str, Any]], frames: list[dict[str, Any]]) -> dict[str, Any]:
    best = candidates[0] if candidates else {}
    best_clean = max(
        (row for row in frames if row.get("full_constraint_evidence")),
        key=lambda row: float(row.get("clean_carry_frame_frac") or 0.0),
        default={},
    )
    decision_counts = Counter(row["decision_label"] for row in candidates)
    frame_blockers = Counter(row.get("primary_frame_blocker", "") for row in frames if row.get("primary_frame_blocker"))
    strict_rows = [row for row in candidates if row.get("strict_rl_gate")]
    summary = {
        "experiment": "E129",
        "case_id": MAIN_CASE,
        "candidate_rows": len(candidates),
        "frame_rows": len(frames),
        "strict_gate_rows": len(strict_rows),
        "rl_ready_rows": 0,
        "cem_launched": False,
        "training_launched": False,
        "decision_counts": dict(decision_counts),
        "frame_blocker_counts": dict(frame_blockers),
        "best_ranked_variant": best.get("variant", ""),
        "best_ranked_decision": best.get("decision_label", ""),
        "best_ranked_physics_contact_frac": best.get("physics_contact_frac", 0.0),
        "best_ranked_lowerbody_contact_frac": best.get("lowerbody_contact_frac", 0.0),
        "best_ranked_nonhand_support_frac": best.get("nonhand_support_frac", 0.0),
        "best_ranked_pelvis_min_m": best.get("pelvis_min_m", 0.0),
        "best_clean_carry_variant": best_clean.get("variant", ""),
        "best_clean_carry_frame_frac": best_clean.get("clean_carry_frame_frac", 0.0),
        "best_clean_carry_longest_run_frames": best_clean.get("clean_carry_longest_run_frames", 0),
        "next_route": (
            "Do not launch RL or full CEM from current rows. Build an explicit "
            "stage-local constrained teacher/smoke that preserves E113/E119 contact "
            "while hard-filtering lower-body, non-hand, object-floor, and pelvis "
            "violations inside the raw-contact window."
        ),
        "status": "pass" if candidates and frames else "fail",
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "e129_main_carry_state_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# E129 Main Carry-State Constraint Audit Summary",
        "",
        f"- candidate rows: `{summary['candidate_rows']}`",
        f"- frame summary rows: `{summary['frame_rows']}`",
        f"- strict gate rows: `{summary['strict_gate_rows']}`",
        f"- RL-ready rows: `{summary['rl_ready_rows']}`",
        f"- CEM launched: `{summary['cem_launched']}`",
        f"- training launched: `{summary['training_launched']}`",
        f"- status: `{summary['status']}`",
        "",
        "## Best Ranked Main Row",
        "",
        "| variant | decision | physics | lower-body | non-hand | pelvis |",
        "|---|---|---:|---:|---:|---:|",
        (
            f"| `{summary['best_ranked_variant']}` | `{summary['best_ranked_decision']}` | "
            f"{summary['best_ranked_physics_contact_frac']:.1%} | "
            f"{summary['best_ranked_lowerbody_contact_frac']:.1%} | "
            f"{summary['best_ranked_nonhand_support_frac']:.1%} | "
            f"{summary['best_ranked_pelvis_min_m']:.3f}m |"
        ),
        "",
        "## Best Frame-Level Clean Carry",
        "",
        "| variant | clean carry frames | longest run |",
        "|---|---:|---:|",
        (
            f"| `{summary['best_clean_carry_variant']}` | "
            f"{float(summary['best_clean_carry_frame_frac'] or 0.0):.1%} | "
            f"{summary['best_clean_carry_longest_run_frames']} |"
        ),
        "",
        "## Decision Counts",
        "",
    ]
    for key, value in sorted(decision_counts.items()):
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Primary Frame Blockers", ""])
    for key, value in sorted(frame_blockers.items()):
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            summary["next_route"],
            "",
            "E129 is audit-only evidence. It does not make any fragment or main row "
            "release/RL-ready.",
            "",
        ]
    )
    (OUT_DIR / "e129_main_carry_state_summary.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def main() -> None:
    candidates = candidate_rows()
    frames = [frame_summary(row) for row in candidates]
    write_tsv(OUT_DIR / "main_candidate_rows.tsv", candidates, CANDIDATE_FIELDS)
    write_tsv(OUT_DIR / "frame_constraint_summary.tsv", frames, FRAME_FIELDS)
    summary = write_summary(candidates, frames)
    print(
        f"wrote {rel(OUT_DIR)} candidates={summary['candidate_rows']} "
        f"strict={summary['strict_gate_rows']} status={summary['status']}"
    )
    if summary["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
