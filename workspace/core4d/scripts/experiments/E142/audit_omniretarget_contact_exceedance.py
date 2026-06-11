#!/usr/bin/env python3
"""Audit whether contact-aware retarget/CEM variants exceed OmniRetarget contact."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def read_rows(path: Path, delimiter: str | None = None) -> list[dict[str, str]]:
    if delimiter is None:
        delimiter = "\t" if path.suffix == ".tsv" else ","
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter=delimiter))


def write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: fmt(row.get(key)) for key in fieldnames})


def finite(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    text = str(value).strip()
    if text == "":
        return default
    try:
        out = float(text)
    except ValueError:
        return default
    if out != out:
        return default
    return out


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def normalize_case_id(row: dict[str, str]) -> str:
    for key in ("source_task", "e109_case_id", "derived_task", "case_id"):
        value = row.get(key, "").strip()
        if value:
            return value
    return ""


def load_e110_pairs(path: Path) -> dict[str, dict[str, dict[str, str]]]:
    pairs: dict[str, dict[str, dict[str, str]]] = {}
    for row in read_rows(path, "\t"):
        case_id = row["case_id"]
        method = row["method"]
        pairs.setdefault(case_id, {})[method] = row
    return pairs


def e112_candidates(path: Path) -> list[dict[str, Any]]:
    rows = []
    for row in read_rows(path, ","):
        ablation = row.get("ablation", "")
        if ablation == "baseline_ref_fk":
            continue
        rows.append(
            {
                "source_exp": "E112",
                "candidate_method": ablation,
                "short_case_id": row.get("case_id", ""),
                "e109_case_id": normalize_case_id(row),
                "object_key": row.get("object_key", ""),
                "candidate_npz": row.get("root_npz_path") or row.get("npz_path", ""),
                "candidate_video": row.get("video_path", ""),
                "candidate_contact_near8": finite(row.get("contact_frac_either")),
                "candidate_physics_contact": finite(row.get("hand_object_contact_physics_frac")),
                "candidate_deep_penetration": finite(row.get("hand_geom_deep_penetration_2cm")),
                "candidate_leg_interference": finite(row.get("leg_box_interference_frac")),
                "candidate_obj_err_mean_m": finite(row.get("obj_err_mean_m")),
                "candidate_pelvis_min_m": finite(row.get("pelvis_min_m")),
                "candidate_work_status": row.get("work_status", ""),
                "candidate_lowerbody_status": row.get("work_status_lowerbody_strict", ""),
                "candidate_lowerbody_strict_pass": str(row.get("lowerbody_strict_pass", "")).lower(),
            }
        )
    return rows


def e113_candidates(path: Path) -> list[dict[str, Any]]:
    rows = []
    for row in read_rows(path, ","):
        if row.get("method", "") != "hold_band":
            continue
        rows.append(
            {
                "source_exp": "E113",
                "candidate_method": "hold_band",
                "short_case_id": row.get("case_id", ""),
                "e109_case_id": normalize_case_id(row),
                "object_key": row.get("object_key", ""),
                "candidate_npz": row.get("root_npz_path") or row.get("npz_path", ""),
                "candidate_video": row.get("video_path", ""),
                "candidate_contact_near8": finite(row.get("contact_frac_either")),
                "candidate_physics_contact": finite(row.get("hand_object_physics_contact"))
                or finite(row.get("hand_object_contact_physics_frac")),
                "candidate_deep_penetration": finite(row.get("hand_geom_deep_penetration_2cm")),
                "candidate_leg_interference": finite(row.get("leg_box_interference_frac")),
                "candidate_obj_err_mean_m": finite(row.get("obj_err_mean_m")),
                "candidate_pelvis_min_m": finite(row.get("pelvis_min_m")),
                "candidate_work_status": row.get("work_status", ""),
                "candidate_lowerbody_status": row.get("work_status_lowerbody_strict", ""),
                "candidate_lowerbody_strict_pass": str(row.get("lowerbody_strict_pass", "")).lower(),
            }
        )
    return rows


def attach_omni(candidate: dict[str, Any], pairs: dict[str, dict[str, dict[str, str]]]) -> dict[str, Any]:
    out = dict(candidate)
    pair = pairs.get(candidate["e109_case_id"], {})
    omni = pair.get("OmniRetarget")
    spider = pair.get("Spider CEM")
    out["join_status"] = "pass" if omni else "missing_omni_pair"
    out["omni_physics_contact"] = finite(omni.get("hand_object_physics_contact_frac")) if omni else None
    out["omni_eef_near8"] = finite(omni.get("eef_near_8cm_frac")) if omni else None
    out["omni_hand_near12"] = finite(omni.get("hand_geom_near_12cm_frac")) if omni else None
    out["omni_deep_penetration"] = finite(omni.get("hand_geom_deep_penetration_2cm_frac")) if omni else None
    out["omni_leg_penetration"] = finite(omni.get("leg_penetration_frac")) if omni else None
    out["omni_pelvis_min_m"] = finite(omni.get("pelvis_min_m")) if omni else None
    out["e110_spider_physics_contact"] = (
        finite(spider.get("hand_object_physics_contact_frac")) if spider else None
    )
    out["e110_spider_eef_near8"] = finite(spider.get("eef_near_8cm_frac")) if spider else None
    out["delta_candidate_physics_vs_e110_spider"] = diff(
        out.get("candidate_physics_contact"), out.get("e110_spider_physics_contact")
    )
    out["delta_candidate_physics_vs_omni"] = diff(
        out.get("candidate_physics_contact"), out.get("omni_physics_contact")
    )
    out["delta_candidate_near8_vs_omni_eef8"] = diff(
        out.get("candidate_contact_near8"), out.get("omni_eef_near8")
    )
    out["delta_candidate_near8_vs_omni_hand12"] = diff(
        out.get("candidate_contact_near8"), out.get("omni_hand_near12")
    )
    out["delta_candidate_deep_pen_vs_omni"] = diff(
        out.get("candidate_deep_penetration"), out.get("omni_deep_penetration")
    )
    out["delta_candidate_leg_vs_omni"] = diff(
        out.get("candidate_leg_interference"), out.get("omni_leg_penetration")
    )
    out["delta_candidate_pelvis_vs_omni"] = diff(
        out.get("candidate_pelvis_min_m"), out.get("omni_pelvis_min_m")
    )
    out["contact_exceeds_omni"] = bool(
        out["join_status"] == "pass"
        and out.get("candidate_physics_contact") is not None
        and out.get("omni_physics_contact") is not None
        and out["candidate_physics_contact"] > out["omni_physics_contact"]
    )
    out["lowerbody_guard_pass"] = bool(
        out.get("candidate_lowerbody_status") == "WORK"
        or out.get("candidate_lowerbody_strict_pass") == "true"
        or (
            out.get("candidate_leg_interference") is not None
            and out["candidate_leg_interference"] <= 0.05
        )
    )
    if out["join_status"] != "pass":
        decision = "missing_omni_pair"
    elif out["contact_exceeds_omni"] and out["lowerbody_guard_pass"]:
        decision = "contact_exceeds_omni_with_lowerbody_guard"
    elif out["contact_exceeds_omni"]:
        decision = "contact_exceeds_omni_but_lowerbody_risk"
    else:
        decision = "contact_not_exceed_omni"
    out["decision"] = decision
    return out


def diff(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return a - b


def best_by_case(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = row["e109_case_id"]
        current = best.get(key)
        if current is None:
            best[key] = row
            continue
        score = (
            row.get("candidate_physics_contact") or -1.0,
            row.get("candidate_contact_near8") or -1.0,
            1.0 if row.get("lowerbody_guard_pass") else 0.0,
        )
        current_score = (
            current.get("candidate_physics_contact") or -1.0,
            current.get("candidate_contact_near8") or -1.0,
            1.0 if current.get("lowerbody_guard_pass") else 0.0,
        )
        if score > current_score:
            best[key] = row
    return sorted(best.values(), key=lambda r: (r.get("object_key", ""), r.get("e109_case_id", "")))


FIELDS = [
    "source_exp",
    "candidate_method",
    "short_case_id",
    "e109_case_id",
    "object_key",
    "join_status",
    "candidate_physics_contact",
    "omni_physics_contact",
    "delta_candidate_physics_vs_omni",
    "e110_spider_physics_contact",
    "delta_candidate_physics_vs_e110_spider",
    "contact_exceeds_omni",
    "candidate_contact_near8",
    "omni_eef_near8",
    "delta_candidate_near8_vs_omni_eef8",
    "omni_hand_near12",
    "delta_candidate_near8_vs_omni_hand12",
    "candidate_deep_penetration",
    "omni_deep_penetration",
    "delta_candidate_deep_pen_vs_omni",
    "candidate_leg_interference",
    "omni_leg_penetration",
    "delta_candidate_leg_vs_omni",
    "candidate_pelvis_min_m",
    "omni_pelvis_min_m",
    "delta_candidate_pelvis_vs_omni",
    "candidate_obj_err_mean_m",
    "candidate_work_status",
    "candidate_lowerbody_status",
    "lowerbody_guard_pass",
    "decision",
    "candidate_npz",
    "candidate_video",
]


def write_summary(path: Path, summary: dict[str, Any], best_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# E142 OmniRetarget Contact Exceedance Audit",
        "",
        f"- status: `{summary['status']}`",
        f"- candidate rows: {summary['candidate_rows']}",
        f"- joined rows: {summary['joined_rows']}",
        f"- best case rows: {summary['best_case_rows']}",
        f"- best cases exceeding OmniRetarget physics contact: {summary['best_cases_contact_exceed_omni']}",
        f"- best cases exceeding OmniRetarget with lower-body guard: {summary['best_cases_contact_exceed_omni_with_lowerbody_guard']}",
        f"- missing Omni pairs: {summary['missing_omni_pairs']}",
        "- primary metric: `candidate hand_object_contact_physics_frac > E110 OmniRetarget hand_object_physics_contact_frac`",
        "- no CEM/RL/Holosoma jobs launched",
        "",
        "## Best By Case",
        "",
        "| case | source | method | cand physics | omni physics | delta vs omni | delta vs E110 Spider | near8 | lowerbody | decision |",
        "|---|---|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in best_rows:
        lines.append(
            "| {case} | {source} | {method} | {cand} | {omni} | {delta} | {delta_spider} | {near8} | {lower} | {decision} |".format(
                case=row.get("e109_case_id", ""),
                source=row.get("source_exp", ""),
                method=row.get("candidate_method", ""),
                cand=fmt(row.get("candidate_physics_contact")),
                omni=fmt(row.get("omni_physics_contact")),
                delta=fmt(row.get("delta_candidate_physics_vs_omni")),
                delta_spider=fmt(row.get("delta_candidate_physics_vs_e110_spider")),
                near8=fmt(row.get("candidate_contact_near8")),
                lower=row.get("candidate_lowerbody_status", ""),
                decision=row.get("decision", ""),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--e110-band", type=Path, required=True)
    parser.add_argument("--e112-full", type=Path, required=True)
    parser.add_argument("--e113-full", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    pairs = load_e110_pairs(args.e110_band)
    candidates = e112_candidates(args.e112_full) + e113_candidates(args.e113_full)
    rows = [attach_omni(row, pairs) for row in candidates]
    best_rows = best_by_case(rows)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows_tsv = args.out_dir / "e142_contact_exceedance_rows.tsv"
    best_tsv = args.out_dir / "e142_best_by_case.tsv"
    summary_json = args.out_dir / "e142_contact_exceedance_summary.json"
    summary_md = args.out_dir / "e142_contact_exceedance_summary.md"

    write_rows(rows_tsv, FIELDS, rows)
    write_rows(best_tsv, FIELDS, best_rows)

    summary = {
        "experiment": "E142",
        "status": "pass",
        "candidate_rows": len(rows),
        "joined_rows": sum(1 for row in rows if row["join_status"] == "pass"),
        "missing_omni_pairs": sum(1 for row in rows if row["join_status"] != "pass"),
        "candidate_rows_contact_exceed_omni": sum(1 for row in rows if row["contact_exceeds_omni"]),
        "candidate_rows_contact_exceed_omni_with_lowerbody_guard": sum(
            1 for row in rows if row["contact_exceeds_omni"] and row["lowerbody_guard_pass"]
        ),
        "best_case_rows": len(best_rows),
        "best_cases_contact_exceed_omni": sum(1 for row in best_rows if row["contact_exceeds_omni"]),
        "best_cases_contact_exceed_omni_with_lowerbody_guard": sum(
            1 for row in best_rows if row["contact_exceeds_omni"] and row["lowerbody_guard_pass"]
        ),
        "training_launched": False,
        "cem_launched": False,
        "rl_launched": False,
        "holosoma_touched": False,
        "rows_tsv": str(rows_tsv),
        "best_tsv": str(best_tsv),
    }
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_summary(summary_md, summary, best_rows)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
