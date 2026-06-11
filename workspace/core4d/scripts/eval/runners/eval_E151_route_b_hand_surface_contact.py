#!/usr/bin/env python3
"""Evaluate E151 route-B hand surface contact experiments."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Ensure lib package is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.core_metrics import METRIC_FIELDS, evaluate_sequence

REPO = Path(__file__).resolve().parents[5]
VARIANTS_TSV = REPO / "workspace/core4d/scripts/E151/variants.tsv"
RESULT_ROOT = REPO / "workspace/core4d/results/E151/route_b_hand_surface_contact"

METHODS = ["baseline", "b2_sup", "b2_tip", "b1_mesh"]
SUMMARY_METRICS = [
    "hand_geom_near_5cm_frac",
    "hand_geom_near_10cm_frac",
    "hand_geom_penetration_frac",
    "hand_geom_deep_penetration_2cm_frac",
    "hand_object_physics_contact_frac",
    "hand_object_con_dist_mean_m",
    "hand_object_con_dist_min_m",
    "hand_object_con_dist_frac_lt_neg5mm",
    "hand_object_con_deep5mm_frame_frac",
    "hand_floor_near_2cm_frac",
    "hand_floor_penetration_frac",
    "hand_floor_min_z_m",
    "hand_floor_physics_contact_frac",
    "hand_floor_con_dist_min_m",
    "hand_floor_con_dist_frac_lt_neg5mm",
    "hand_floor_con_deep5mm_frame_frac",
    "leg_penetration_frac",
    "object_floor_contact_frac",
    "pelvis_min_m",
    "obj_err_mean_m",
]
DELTA_METRICS = [
    "hand_geom_near_5cm_frac",
    "hand_geom_near_10cm_frac",
    "hand_geom_penetration_frac",
    "hand_object_physics_contact_frac",
    "hand_object_con_dist_frac_lt_neg5mm",
    "hand_object_con_deep5mm_frame_frac",
    "hand_floor_near_2cm_frac",
    "hand_floor_penetration_frac",
    "hand_floor_physics_contact_frac",
    "hand_floor_con_deep5mm_frame_frac",
    "leg_penetration_frac",
    "obj_err_mean_m",
]
LOWER_IS_WORST_METRICS = {
    "pelvis_min_m",
    "hand_floor_min_z_m",
    "hand_object_con_dist_mean_m",
    "hand_object_con_dist_min_m",
    "hand_floor_con_dist_mean_m",
    "hand_floor_con_dist_min_m",
}


def rel(path: Path | str) -> str:
    if not path:
        return ""
    p = Path(path)
    try:
        return str(p.resolve().relative_to(REPO.resolve()))
    except Exception:
        return str(path)


def repo_path(text: str | Path) -> Path:
    p = Path(text)
    return p if p.is_absolute() else REPO / p


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.8g}" if math.isfinite(value) else ""
    return str(value)


def finite(value: Any) -> float | None:
    if value in ("", None):
        return None
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def rows_for_stage(rows: list[dict[str, str]], stage: str) -> list[dict[str, str]]:
    """Manifest stores full paths; rewrite E151 run rows for smoke eval."""
    out: list[dict[str, str]] = []
    cem_dir = RESULT_ROOT / "cem" / stage
    for row in rows:
        item = dict(row)
        if item.get("run_status") == "to_run":
            variant = item["variant"]
            item["result_npz"] = rel(cem_dir / f"{variant}.npz")
            item["outdir_npz"] = rel(cem_dir / f"{variant}_outdir_{stage}" / "trajectory_mjwp_act.npz")
            item["video"] = rel(cem_dir / f"{variant}_{stage}.mp4")
        out.append(item)
    return out


def write_tsv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: fmt(row.get(field, "")) for field in fields})


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")




def resolve_scene_xml(row: dict[str, str]) -> Path:
    """Task-dir scene, falling back to the per-exp snapshot when absent locally.

    Clean/derived task dirs are remote-only; the E151 snapshot (experiment.md
    §7) holds the exact rubber scene. When falling back we rewrite the snapshot
    XML's relative asset paths to absolute so MuJoCo can load it locally.
    """
    scene_xml = repo_path(row["rubber_scene_act"])
    if scene_xml.is_file():
        return scene_xml
    snap = RESULT_ROOT / "scene_snapshot" / row["derived_task"] / scene_xml.name
    if not snap.is_file():
        raise FileNotFoundError(f"scene missing in task dir and snapshot: {scene_xml} / {snap}")
    import re
    import tempfile

    txt = snap.read_text(encoding="utf-8")
    txt = re.sub(r'meshdir="(\.\./)+spider', f'meshdir="{REPO}/spider', txt)
    txt = re.sub(r'file="(\.\./)+example_datasets', f'file="{REPO}/example_datasets', txt)
    fixed = Path(tempfile.gettempdir()) / f"e151_eval_{row['variant']}_{scene_xml.name}"
    fixed.write_text(txt, encoding="utf-8")
    return fixed


def eval_one(row: dict[str, str]) -> dict[str, Any]:
    qpos_path = repo_path(row["outdir_npz"])
    scene_xml = resolve_scene_xml(row)
    item = evaluate_sequence(
        row=row,
        method=f"E151 {row['method']}",
        hand_collision_variant_id=row["hand_collision_variant_id"],
        qpos_path=qpos_path,
        scene_xml=scene_xml,
    )
    item.update(
        {
            "variant": row["variant"],
            "short_case_id": row["short_case_id"],
            "method": row["method"],
            "method_group": row["method_group"],
            "split": row["split"],
            "run_status": row["run_status"],
            "source_exp": row["source_exp"],
            "target_npz": row["target_npz"],
            "result_npz": row["result_npz"],
            "video": row["video"],
        }
    )
    item.update(support_info(row))
    return item


def support_info(row: dict[str, str]) -> dict[str, Any]:
    out = {
        "hand_support_sdf_mean_mean": "",
        "hand_support_sdf_min_min": "",
        "hand_support_score_mean_mean": "",
        "hand_support_rew_mean_mean": "",
    }
    path = repo_path(row["outdir_npz"])
    if not path.is_file():
        return out
    data = np.load(path, allow_pickle=True)
    mapping = {
        "hand_support_sdf_mean": "hand_support_sdf_mean_mean",
        "hand_support_sdf_min": "hand_support_sdf_min_min",
        "hand_support_score_mean": "hand_support_score_mean_mean",
        "hand_support_rew_mean": "hand_support_rew_mean_mean",
    }
    for key, out_key in mapping.items():
        if key not in data:
            continue
        arr = np.asarray(data[key], dtype=np.float64)
        vals = arr[np.isfinite(arr)]
        if vals.size == 0:
            continue
        out[out_key] = float(vals.min() if key.endswith("_min") else vals.mean())
    return out


def build_metric_rows(rows: list[dict[str, str]], *, allow_missing: bool) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    metric_rows: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    fields = [
        "variant",
        "short_case_id",
        "method",
        "method_group",
        "split",
        "run_status",
        "source_exp",
        "target_npz",
        "result_npz",
        "video",
        *METRIC_FIELDS,
        "hand_support_sdf_mean_mean",
        "hand_support_sdf_min_min",
        "hand_support_score_mean_mean",
        "hand_support_rew_mean_mean",
    ]
    for row in rows:
        outdir = repo_path(row["outdir_npz"])
        if not outdir.is_file():
            missing.append({"variant": row["variant"], "method": row["method"], "missing": rel(outdir)})
            if allow_missing:
                continue
            raise FileNotFoundError(f"missing E151 qpos for {row['variant']}: {outdir}")
        metric_rows.append(eval_one(row))
    return metric_rows, missing, fields


def avg(values: list[float]) -> float:
    vals = [v for v in values if math.isfinite(v)]
    return sum(vals) / len(vals) if vals else math.nan


def stdev(values: list[float]) -> float:
    vals = [v for v in values if math.isfinite(v)]
    return statistics.pstdev(vals) if len(vals) > 1 else 0.0


def summary_rows(metric_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for method in METHODS:
        rows = [r for r in metric_rows if r["method"] == method]
        if not rows:
            continue
        item: dict[str, Any] = {
            "method": method,
            "case_count": len(rows),
            "fall_count": sum(1 for r in rows if r.get("fall_flag")),
        }
        for metric in SUMMARY_METRICS:
            vals = [float(r[metric]) for r in rows if finite(r.get(metric)) is not None]
            item[f"{metric}_mean"] = avg(vals)
            item[f"{metric}_std"] = stdev(vals)
            if vals:
                item[f"{metric}_worst"] = min(vals) if metric in LOWER_IS_WORST_METRICS else max(vals)
            else:
                item[f"{metric}_worst"] = math.nan
        out.append(item)
    return out


def delta_rows(metric_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_case_method = {(r["short_case_id"], r["method"]): r for r in metric_rows}
    out: list[dict[str, Any]] = []
    for short_case_id in sorted({r["short_case_id"] for r in metric_rows}):
        base = by_case_method.get((short_case_id, "baseline"))
        if not base:
            continue
        for method in ("b2_sup", "b2_tip", "b1_mesh"):
            row = by_case_method.get((short_case_id, method))
            if not row:
                continue
            item: dict[str, Any] = {
                "short_case_id": short_case_id,
                "case_id": row["case_id"],
                "object_key": row["object_key"],
                "method": method,
                "baseline_source_exp": base["source_exp"],
                "run_source_exp": row["source_exp"],
            }
            for metric in DELTA_METRICS:
                item[f"{metric}_baseline"] = base[metric]
                item[f"{metric}_run"] = row[metric]
                item[f"{metric}_delta"] = float(row[metric]) - float(base[metric])
            item["fall_flag"] = row["fall_flag"]
            item["success_near5_plus5pp_pen_not_up"] = (
                item["hand_geom_near_5cm_frac_delta"] >= 0.05
                and item["hand_geom_penetration_frac_delta"] <= 0.0
                and not bool(row["fall_flag"])
            )
            item["hand_support_sdf_mean_mean"] = row.get("hand_support_sdf_mean_mean", "")
            item["hand_support_score_mean_mean"] = row.get("hand_support_score_mean_mean", "")
            out.append(item)
    return out


def delta_summary_rows(deltas: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for method in ("b2_sup", "b2_tip", "b1_mesh"):
        rows = [r for r in deltas if r["method"] == method]
        if not rows:
            continue
        item: dict[str, Any] = {
            "method": method,
            "case_count": len(rows),
            "success_cases": sum(1 for r in rows if r["success_near5_plus5pp_pen_not_up"]),
        }
        for metric in DELTA_METRICS:
            vals = [float(r[f"{metric}_delta"]) for r in rows if finite(r.get(f"{metric}_delta")) is not None]
            item[f"{metric}_delta_mean"] = avg(vals)
            item[f"{metric}_delta_std"] = stdev(vals)
            if vals:
                item[f"{metric}_delta_worst"] = min(vals) if metric == "hand_geom_near_5cm_frac" else max(vals)
            else:
                item[f"{metric}_delta_worst"] = math.nan
        out.append(item)
    return out


def maybe_visual_sheets(metric_rows: list[dict[str, Any]], stage: str) -> list[dict[str, str]]:
    out_rows: list[dict[str, str]] = []
    try:
        from PIL import Image, ImageDraw
    except Exception:
        return out_rows
    if not metric_rows:
        return out_rows
    out_dir = RESULT_ROOT / "visual_inspection"
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = [25, 55, 85]
    by_case_method = {(r["short_case_id"], r["method"]): r for r in metric_rows}
    for short in sorted({r["short_case_id"] for r in metric_rows}):
        for frame in frames:
            imgs = []
            labels = []
            for method in METHODS:
                row = by_case_method.get((short, method))
                if not row:
                    continue
                video = repo_path(row["video"])
                if not video.is_file():
                    continue
                tmp = out_dir / f".tmp_{short}_{method}_f{frame}.jpg"
                subprocess.run(
                    [
                        "ffmpeg",
                        "-nostdin",
                        "-y",
                        "-loglevel",
                        "error",
                        "-i",
                        str(video),
                        "-vf",
                        f"select=eq(n\\,{frame})",
                        "-frames:v",
                        "1",
                        "-vsync",
                        "0",
                        str(tmp),
                    ],
                    check=False,
                )
                if tmp.is_file():
                    img = Image.open(tmp).convert("RGB")
                    img.thumbnail((360, 260))
                    imgs.append(img.copy())
                    labels.append(method)
                    tmp.unlink(missing_ok=True)
            if len(imgs) < 2:
                continue
            label_h = 24
            w = sum(img.width for img in imgs)
            h = max(img.height for img in imgs) + label_h
            sheet = Image.new("RGB", (w, h), "white")
            draw = ImageDraw.Draw(sheet)
            x = 0
            for label, img in zip(labels, imgs):
                draw.text((x + 4, 4), label, fill=(0, 0, 0))
                sheet.paste(img, (x, label_h))
                x += img.width
            out_path = out_dir / f"{short}_f{frame}_baseline_b2_b1.jpg"
            sheet.save(out_path, quality=92)
            out_rows.append({"short_case_id": short, "frame": str(frame), "sheet": rel(out_path)})
    return out_rows


def write_summary_md(path: Path, summaries: list[dict[str, Any]], delta_summaries: list[dict[str, Any]], missing: list[dict[str, Any]], visual_rows: list[dict[str, str]]) -> None:
    lines = [
        "# E151 route-B hand surface contact summary",
        "",
        f"- metric rows: {sum(int(r['case_count']) for r in summaries)}",
        f"- missing rows: {len(missing)}",
        "",
        "## Method Means",
        "",
        "| method | cases | 5cm | hand pen | physics | leg pen | obj err |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| `{row['method']}` | {row['case_count']} | "
            f"{float(row['hand_geom_near_5cm_frac_mean']):.4f} | "
            f"{float(row['hand_geom_penetration_frac_mean']):.4f} | "
            f"{float(row['hand_object_physics_contact_frac_mean']):.4f} | "
            f"{float(row['leg_penetration_frac_mean']):.4f} | "
            f"{float(row['obj_err_mean_m_mean']):.4f} |"
        )
    lines += [
        "",
        "## Mean Deltas vs Baseline",
        "",
        "| method | success | 5cm delta | hand pen delta | physics delta | leg pen delta |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in delta_summaries:
        lines.append(
            f"| `{row['method']}` | {row['success_cases']}/{row['case_count']} | "
            f"{float(row['hand_geom_near_5cm_frac_delta_mean']):+.4f} | "
            f"{float(row['hand_geom_penetration_frac_delta_mean']):+.4f} | "
            f"{float(row['hand_object_physics_contact_frac_delta_mean']):+.4f} | "
            f"{float(row['leg_penetration_frac_delta_mean']):+.4f} |"
        )
    if visual_rows:
        lines += ["", "## Visual Sheets", ""]
        for row in visual_rows:
            lines.append(f"- `{row['short_case_id']}` f{row['frame']}: `{row['sheet']}`")
    if missing:
        lines += ["", "## Missing", ""]
        for row in missing:
            lines.append(f"- `{row['variant']}`: `{row['missing']}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", nargs="?", default="full", choices=["smoke", "full"])
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--skip-visual", action="store_true")
    args = parser.parse_args()

    rows = rows_for_stage(read_tsv(VARIANTS_TSV), args.stage)
    eval_dir = RESULT_ROOT / "eval" / args.stage
    metric_rows, missing, metric_fields = build_metric_rows(rows, allow_missing=args.allow_missing)
    summaries = summary_rows(metric_rows)
    deltas = delta_rows(metric_rows)
    delta_summaries = delta_summary_rows(deltas)
    visual_rows = [] if args.skip_visual else maybe_visual_sheets(metric_rows, args.stage)

    write_tsv(eval_dir / "e151_method_metrics.tsv", metric_rows, metric_fields)
    write_tsv(eval_dir / "e151_method_summary.tsv", summaries, ["method", "case_count", "fall_count", *[f"{m}_{s}" for m in SUMMARY_METRICS for s in ("mean", "std", "worst")]])
    delta_fields = [
        "short_case_id",
        "case_id",
        "object_key",
        "method",
        "baseline_source_exp",
        "run_source_exp",
        *[f"{m}_{suffix}" for m in DELTA_METRICS for suffix in ("baseline", "run", "delta")],
        "fall_flag",
        "success_near5_plus5pp_pen_not_up",
        "hand_support_sdf_mean_mean",
        "hand_support_score_mean_mean",
    ]
    write_tsv(eval_dir / "e151_delta_vs_baseline.tsv", deltas, delta_fields)
    write_tsv(eval_dir / "e151_delta_summary.tsv", delta_summaries, ["method", "case_count", "success_cases", *[f"{m}_delta_{s}" for m in DELTA_METRICS for s in ("mean", "std", "worst")]])
    write_tsv(eval_dir / "e151_missing.tsv", missing, ["variant", "method", "missing"])
    write_tsv(eval_dir / "e151_visual_sheets.tsv", visual_rows, ["short_case_id", "frame", "sheet"])
    write_json(eval_dir / "e151_eval_summary.json", {"method_rows": len(metric_rows), "delta_rows": len(deltas), "missing": len(missing), "visual_sheets": len(visual_rows)})
    write_summary_md(eval_dir / "e151_summary.md", summaries, delta_summaries, missing, visual_rows)
    print(f"E151 eval: method_rows={len(metric_rows)} delta_rows={len(deltas)} missing={len(missing)} visual_sheets={len(visual_rows)}")


if __name__ == "__main__":
    main()
