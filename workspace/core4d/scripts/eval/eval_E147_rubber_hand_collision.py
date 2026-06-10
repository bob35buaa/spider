#!/usr/bin/env python3
"""Evaluate E147 sphere5cm vs rubber_hull CEM trajectories."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

# Ensure lib package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib.core_metrics import (
    METRIC_FIELDS,
    evaluate_sequence,
    rel as _lib_rel,
)

REPO = Path(__file__).resolve().parents[4]
VARIANTS_TSV = REPO / "workspace/core4d/scripts/E147/variants.tsv"
RESULT_ROOT = REPO / "workspace/core4d/results/E147/rubber_hand_collision"

# E147-specific constants kept for local use (build_summary, write_markdown)
NEAR_THRESHOLDS_M = (0.03, 0.05, 0.08, 0.10)
DEEP_PENETRATION_M = -0.02
MESH_SAMPLE_COUNT = 800

PAIR_FIELDS = [
    "case_id",
    "variant",
    "object_key",
    "object_category",
    "expected_quality",
    "sphere_historical_cem_status",
    "rubber_outputs_present",
    "sphere_hand_near_5cm",
    "rubber_hand_near_5cm",
    "delta_hand_near_5cm",
    "sphere_hand_deep_2cm",
    "rubber_hand_deep_2cm",
    "delta_hand_deep_2cm",
    "sphere_hand_penetration",
    "rubber_hand_penetration",
    "delta_hand_penetration",
    "sphere_physics_contact",
    "rubber_physics_contact",
    "delta_physics_contact",
    "sphere_pelvis_min_m",
    "rubber_pelvis_min_m",
    "rubber_fall_flag",
    "rubber_leg_penetration",
    "rubber_body_penetration",
    "rubber_object_floor_contact",
    "rubber_obj_err_mean_m",
    "ab_status",
]

EVIDENCE_FIELDS = [
    "case_id",
    "object_key",
    "object_name",
    "date",
    "seq",
    "person",
    "person_idx",
    "retarget_variant_id",
    "target_variant_id",
    "hand_collision_variant_id",
    "candidate_decision",
    "handoff_decision",
    "cem_status",
    "rl_status",
    "downstream_decision",
    "downstream_failure_mode",
    "downstream_notes",
    "cem_run_id",
    "cem_result_npz",
    "cem_video",
    "cem_metrics_ref",
    "rl_run_id",
    "rl_checkpoint",
    "rl_metrics_ref",
    "rl_video",
]


def rel(path: Path | str) -> str:
    return _lib_rel(path)


def repo_path(text: str) -> Path:
    p = Path(text)
    return p if p.is_absolute() else REPO / p


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return f"{value:.8g}"
    return str(value)


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: fmt(row.get(field, "")) for field in fields})


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def _evaluate_sequence_e147(
    *,
    row: dict[str, str],
    method: str,
    hand_collision_variant_id: str,
    qpos_path: Path,
    scene_xml: Path,
) -> dict[str, Any]:
    """E147-specific wrapper that sets the notes field."""
    result = evaluate_sequence(
        row=row,
        method=method,
        hand_collision_variant_id=hand_collision_variant_id,
        qpos_path=qpos_path,
        scene_xml=scene_xml,
    )
    result["notes"] = (
        "E147_eval_mesh_vertex_sdf_v1"
        if hand_collision_variant_id == "rubber_hull"
        else "E147_eval_sphere_sdf_v1"
    )
    return result


def rubber_paths(row: dict[str, str], stage: str, results_dir: Path) -> tuple[Path, Path]:
    outdir_npz = results_dir / f"{row['variant']}_outdir_{stage}/trajectory_mjwp_act.npz"
    return outdir_npz, repo_path(row["rubber_scene_act"])


def build_pair_rows(rows: list[dict[str, str]], metrics: list[dict[str, Any]], stage: str, results_dir: Path) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for metric in metrics:
        by_key[(metric["case_id"], metric["hand_collision_variant_id"])] = metric
    out = []
    for row in rows:
        sphere = by_key.get((row["case_id"], "sphere5cm"), {})
        rubber = by_key.get((row["case_id"], "rubber_hull"), {})
        rubber_npz = results_dir / f"{row['variant']}_outdir_{stage}/trajectory_mjwp_act.npz"

        def delta(key: str) -> float:
            return float(rubber.get(key, math.nan)) - float(sphere.get(key, math.nan))

        contact_ok = delta("hand_geom_near_5cm_frac") >= -0.02
        deep_ok = delta("hand_geom_deep_penetration_2cm_frac") < 0.0
        stable_ok = not bool(rubber.get("fall_flag", True))
        if rubber_npz.is_file():
            if contact_ok and deep_ok and stable_ok:
                status = "rubber_primary_pass"
            elif contact_ok and stable_ok:
                status = "rubber_contact_stable_but_penetration_not_improved"
            else:
                status = "rubber_primary_fail"
        else:
            status = "missing_rubber"
        out.append(
            {
                "case_id": row["case_id"],
                "variant": row["variant"],
                "object_key": row["object_key"],
                "object_category": row["object_category"],
                "expected_quality": row["expected_quality"],
                "sphere_historical_cem_status": row["historical_cem_status"],
                "rubber_outputs_present": rubber_npz.is_file(),
                "sphere_hand_near_5cm": sphere.get("hand_geom_near_5cm_frac", ""),
                "rubber_hand_near_5cm": rubber.get("hand_geom_near_5cm_frac", ""),
                "delta_hand_near_5cm": delta("hand_geom_near_5cm_frac"),
                "sphere_hand_deep_2cm": sphere.get("hand_geom_deep_penetration_2cm_frac", ""),
                "rubber_hand_deep_2cm": rubber.get("hand_geom_deep_penetration_2cm_frac", ""),
                "delta_hand_deep_2cm": delta("hand_geom_deep_penetration_2cm_frac"),
                "sphere_hand_penetration": sphere.get("hand_geom_penetration_frac", ""),
                "rubber_hand_penetration": rubber.get("hand_geom_penetration_frac", ""),
                "delta_hand_penetration": delta("hand_geom_penetration_frac"),
                "sphere_physics_contact": sphere.get("hand_object_physics_contact_frac", ""),
                "rubber_physics_contact": rubber.get("hand_object_physics_contact_frac", ""),
                "delta_physics_contact": delta("hand_object_physics_contact_frac"),
                "sphere_pelvis_min_m": sphere.get("pelvis_min_m", ""),
                "rubber_pelvis_min_m": rubber.get("pelvis_min_m", ""),
                "rubber_fall_flag": rubber.get("fall_flag", ""),
                "rubber_leg_penetration": rubber.get("leg_penetration_frac", ""),
                "rubber_body_penetration": rubber.get("body_penetration_frac", ""),
                "rubber_object_floor_contact": rubber.get("object_floor_contact_frac", ""),
                "rubber_obj_err_mean_m": rubber.get("obj_err_mean_m", ""),
                "ab_status": status,
            }
        )
    return out


def mean(values: list[float]) -> float:
    vals = [v for v in values if math.isfinite(v)]
    return sum(vals) / len(vals) if vals else math.nan


def build_summary(metrics: list[dict[str, Any]], pairs: list[dict[str, Any]]) -> dict[str, Any]:
    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for metric in metrics:
        by_method[metric["hand_collision_variant_id"]].append(metric)
    method_summary = {}
    for method, rows in sorted(by_method.items()):
        method_summary[method] = {
            "rows": len(rows),
            "fall_count": sum(bool(r.get("fall_flag")) for r in rows),
            "mean_hand_near_5cm": mean([float(r["hand_geom_near_5cm_frac"]) for r in rows]),
            "mean_hand_near_8cm": mean([float(r["hand_geom_near_8cm_frac"]) for r in rows]),
            "mean_hand_penetration": mean([float(r["hand_geom_penetration_frac"]) for r in rows]),
            "mean_hand_deep_2cm": mean([float(r["hand_geom_deep_penetration_2cm_frac"]) for r in rows]),
            "mean_physics_contact": mean([float(r["hand_object_physics_contact_frac"]) for r in rows]),
            "mean_pelvis_min_m": mean([float(r["pelvis_min_m"]) for r in rows]),
            "mean_obj_err_mean_m": mean([float(r["obj_err_mean_m"]) for r in rows]),
            "mean_leg_penetration": mean([float(r["leg_penetration_frac"]) for r in rows]),
            "mean_body_penetration": mean([float(r["body_penetration_frac"]) for r in rows]),
        }
    return {
        "metric_rows": len(metrics),
        "pair_rows": len(pairs),
        "rubber_outputs_present": sum(str(p["rubber_outputs_present"]).lower() == "true" for p in pairs),
        "ab_status_counts": dict((status, sum(p["ab_status"] == status for p in pairs)) for status in sorted({p["ab_status"] for p in pairs})),
        "method_summary": method_summary,
        "near_thresholds_m": NEAR_THRESHOLDS_M,
        "deep_penetration_m": DEEP_PENETRATION_M,
        "mesh_sample_count": MESH_SAMPLE_COUNT,
    }


def parse_case_identity(case_id: str, person: str) -> dict[str, str]:
    parts = case_id.split("_")
    date = ""
    seq = ""
    if len(parts) >= 5 and parts[0] in {"e091", "d003"}:
        date = parts[2]
        seq = parts[3]
    elif len(parts) >= 4 and parts[0].startswith("bucket"):
        date = parts[1]
        seq = parts[2]
    person_idx = ""
    if person.startswith("person"):
        try:
            person_idx = str(int(person.replace("person", "")) - 1)
        except Exception:
            person_idx = ""
    return {"date": date, "seq": seq, "person_idx": person_idx}


def rubber_gate(row: dict[str, Any]) -> tuple[str, str]:
    if str(row.get("rubber_outputs_present", "")).lower() != "true":
        return "not_run", "missing_rubber_output"
    if str(row.get("rubber_fall_flag", "")).lower() == "true":
        return "fail", "pelvis_fall"
    try:
        if float(row.get("rubber_leg_penetration", "nan")) > 0.05:
            return "fail", "lowerbody_interference"
        if float(row.get("rubber_body_penetration", "nan")) > 0.0:
            return "fail", "body_object_interference"
        if float(row.get("rubber_object_floor_contact", "nan")) > 0.5:
            return "fail", "object_floor_contact"
        if float(row.get("rubber_obj_err_mean_m", "nan")) > 0.10:
            return "fail", "cem_work_status_fail"
    except Exception:
        return "fail", "rubber_metric_parse_error"
    return "pass", ""


def build_evidence_rows(rows: list[dict[str, str]], pairs: list[dict[str, Any]], stage: str, results_dir: Path, metrics_ref: Path) -> list[dict[str, Any]]:
    by_case = {row["case_id"]: row for row in rows}
    out = []
    for pair in pairs:
        src = by_case[pair["case_id"]]
        cem_status, failure = rubber_gate(pair)
        identity = parse_case_identity(pair["case_id"], src.get("person", ""))
        notes = (
            f"ab_status={pair['ab_status']}; "
            f"delta_hand_near_5cm={fmt(pair.get('delta_hand_near_5cm'))}; "
            f"delta_hand_deep_2cm={fmt(pair.get('delta_hand_deep_2cm'))}; "
            f"rubber_fall={pair.get('rubber_fall_flag', '')}; "
            f"mesh_sdf_sample_count={MESH_SAMPLE_COUNT}"
        )
        out.append(
            {
                "case_id": pair["case_id"],
                "object_key": src.get("object_key", ""),
                "object_name": src.get("object_key", ""),
                "date": identity["date"],
                "seq": identity["seq"],
                "person": src.get("person", ""),
                "person_idx": src.get("person_idx", identity["person_idx"]),
                "retarget_variant_id": src.get("retarget_variant_id", ""),
                "target_variant_id": src.get("target_variant_id", "ref_fk"),
                "hand_collision_variant_id": "rubber_hull",
                "candidate_decision": "PASS",
                "handoff_decision": "HANDOFF_READY",
                "cem_status": cem_status,
                "rl_status": "not_run",
                "downstream_decision": "DOWNSTREAM_CEM_PASS" if cem_status == "pass" else ("WAIT_CEM_NOT_RUN" if cem_status == "not_run" else "DOWNSTREAM_CEM_FAIL"),
                "downstream_failure_mode": failure,
                "downstream_notes": notes,
                "cem_run_id": pair["variant"],
                "cem_result_npz": rel(results_dir / f"{pair['variant']}.npz") if cem_status != "not_run" else "",
                "cem_video": rel(results_dir / f"{pair['variant']}_{stage}.mp4") if cem_status != "not_run" else "",
                "cem_metrics_ref": rel(metrics_ref),
                "rl_run_id": "",
                "rl_checkpoint": "",
                "rl_metrics_ref": "",
                "rl_video": "",
            }
        )
    return out


def write_markdown(path: Path, pairs: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    lines = [
        "# E147 rubber hand collision A/B eval",
        "",
        f"- rubber outputs present: {summary['rubber_outputs_present']}/{summary['pair_rows']}",
        f"- mesh SDF sample count: {MESH_SAMPLE_COUNT}",
        f"- status counts: `{summary['ab_status_counts']}`",
        "",
        "## Method Summary",
        "",
        "| method | rows | fall | hand5 | hand8 | hand pen | hand deep2 | physics contact | pelvis min | obj err | leg pen | body pen |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method, row in summary["method_summary"].items():
        lines.append(
            f"| {method} | {row['rows']} | {row['fall_count']} | {row['mean_hand_near_5cm']:.4f} | "
            f"{row['mean_hand_near_8cm']:.4f} | {row['mean_hand_penetration']:.4f} | {row['mean_hand_deep_2cm']:.4f} | "
            f"{row['mean_physics_contact']:.4f} | {row['mean_pelvis_min_m']:.4f} | {row['mean_obj_err_mean_m']:.4f} | "
            f"{row['mean_leg_penetration']:.4f} | {row['mean_body_penetration']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Pair Delta",
            "",
            "| case | object | old | d hand5 | d deep2 | d pen | d phys | rubber pelvis | rubber fall | status |",
            "|---|---|---|---:|---:|---:|---:|---:|---|---|",
        ]
    )
    for row in pairs:
        lines.append(
            f"| `{row['case_id']}` | {row['object_key']} | {row['sphere_historical_cem_status']} | "
            f"{float(row['delta_hand_near_5cm']):.4f} | {float(row['delta_hand_deep_2cm']):.4f} | "
            f"{float(row['delta_hand_penetration']):.4f} | {float(row['delta_physics_contact']):.4f} | "
            f"{fmt(row['rubber_pelvis_min_m'])} | {row['rubber_fall_flag']} | {row['ab_status']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", default="full", choices=("smoke", "full"))
    parser.add_argument("--variants", type=Path, default=VARIANTS_TSV)
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--allow-missing-rubber", action="store_true")
    args = parser.parse_args()

    results_dir = args.results_dir or (RESULT_ROOT / "cem" / args.stage)
    out_dir = args.out_dir or (RESULT_ROOT / "eval" / args.stage)
    rows = read_tsv(args.variants)
    metric_rows: list[dict[str, Any]] = []
    missing = []
    for row in rows:
        sphere_qpos = repo_path(row["sphere_outdir_npz"])
        metric_rows.append(
            _evaluate_sequence_e147(
                row=row,
                method="sphere historical CEM",
                hand_collision_variant_id="sphere5cm",
                qpos_path=sphere_qpos,
                scene_xml=repo_path(row["base_scene_act"]),
            )
        )
        rubber_qpos, rubber_scene = rubber_paths(row, args.stage, results_dir)
        if not rubber_qpos.is_file():
            missing.append(rel(rubber_qpos))
            continue
        metric_rows.append(
            _evaluate_sequence_e147(
                row=row,
                method="rubber_hull CEM",
                hand_collision_variant_id="rubber_hull",
                qpos_path=rubber_qpos,
                scene_xml=rubber_scene,
            )
        )
    if missing and not args.allow_missing_rubber:
        raise FileNotFoundError("missing rubber outputs:\n" + "\n".join(missing))

    pair_rows = build_pair_rows(rows, metric_rows, args.stage, results_dir)
    summary = build_summary(metric_rows, pair_rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "e147_method_metrics.tsv"
    write_tsv(metrics_path, metric_rows, METRIC_FIELDS)
    write_tsv(out_dir / "e147_pair_delta.tsv", pair_rows, PAIR_FIELDS)
    write_tsv(out_dir / "e147_downstream_evidence_input.tsv", build_evidence_rows(rows, pair_rows, args.stage, results_dir, metrics_path), EVIDENCE_FIELDS)
    write_json(out_dir / "e147_eval_summary.json", summary)
    write_markdown(out_dir / "e147_eval_summary.md", pair_rows, summary)
    print(f"[E147 eval] wrote {len(metric_rows)} metric rows, missing rubber={len(missing)} to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
