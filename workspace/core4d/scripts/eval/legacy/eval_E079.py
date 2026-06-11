#!/usr/bin/env python3
"""E079 evaluation wrapper for multi-case CORE4D generalization."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
EVAL_DIR = REPO / "workspace/core4d/scripts/eval"
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

import eval_E078 as e078  # noqa: E402


RESULTS = REPO / "workspace/core4d/results/E079"
VARIANTS_FILE = REPO / "workspace/core4d/scripts/E079/variants.tsv"
FPS = 50.0
CASE_WINDOW_PAD_FRAMES = 10


def read_variants() -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    fieldnames = ["variant", "task", "mask_slug", "person_idx", "split", "role"]
    with VARIANTS_FILE.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(
            (line for line in f if line.strip() and not line.startswith("#")),
            delimiter="\t",
            fieldnames=fieldnames,
        )
        for row in reader:
            if row["role"] not in {"main", "guard"}:
                continue
            out[row["variant"]] = {
                "name": row["variant"],
                "case": row["task"],
                "override": f"core4d_{row['variant']}",
                "split": row["split"],
                "role": row["role"],
            }
    return out


def _read_timeseries(variant: str, target_len: int) -> list[dict[str, str]]:
    path = RESULTS / f"timeseries_{variant}.csv"
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return rows[:target_len]


def _float_array(rows: list[dict[str, str]], key: str) -> np.ndarray:
    return np.asarray([float(row[key]) for row in rows], dtype=np.float64)


def _bool_contact(rows: list[dict[str, str]], key: str) -> np.ndarray:
    return np.asarray([float(row[key]) > 0 for row in rows], dtype=bool)


def _load_case_mask(summary: dict[str, object], target_len: int) -> tuple[np.ndarray, str, str]:
    cfg = e078._build_config(str(summary["override"]), str(summary["case"]), "cpu", 4)
    if cfg.contact_hdmi_mask_source != "core4d_3cm":
        raise ValueError(f"{summary['variant']} does not use core4d_3cm mask")
    path = Path(cfg.contact_hdmi_mask_path)
    if not path.is_absolute():
        path = REPO / path
    data = np.load(path, allow_pickle=True)
    axis = cfg.contact_hdmi_mask_time_axis
    if axis == "auto":
        if (
            "spider_contact_mask_3cm" in data
            and data["spider_contact_mask_3cm"].shape[0] == target_len
        ):
            axis = "spider"
        elif (
            "eval_contact_mask_3cm" in data
            and data["eval_contact_mask_3cm"].shape[0] == target_len
        ):
            axis = "eval"
        else:
            axis = "eval" if "eval_contact_mask_3cm" in data else "spider"
    key = f"{axis}_contact_mask_3cm"
    raw = data[key]
    person_idx = int(cfg.contact_hdmi_mask_person_idx)
    mask = e078.resize_mask(raw[:, person_idx, :], target_len).astype(bool)
    return mask, key, str(path.relative_to(REPO))


def _case_window_from_mask(mask: np.ndarray) -> tuple[int, int, bool]:
    active = mask.any(axis=1)
    active_frames = np.flatnonzero(active)
    if active_frames.size == 0:
        return 0, len(mask) - 1, False
    start = max(0, int(active_frames[0]) - CASE_WINDOW_PAD_FRAMES)
    end = min(len(mask) - 1, int(active_frames[-1]) + CASE_WINDOW_PAD_FRAMES)
    return start, end, True


def case_window_metrics(summary: dict[str, object]) -> dict[str, object]:
    target_len = int(summary["T"])
    mask, mask_key, mask_path = _load_case_mask(summary, target_len)
    start, end, has_contact = _case_window_from_mask(mask)
    rows = _read_timeseries(str(summary["variant"]), target_len)
    window = rows[start : end + 1]

    obj_err = _float_array(window, "obj_err_m")
    pelvis_z = _float_array(window, "sim_pelvis_z_m")
    sim_contact = _bool_contact(window, "sim_total_contact_count")
    ref_contact = _bool_contact(window, "ref_total_contact_count")
    min_hand_sdf = _float_array(window, "sim_min_hand_sdf_m")
    robot_ctrl_linf = _float_array(window, "robot_ctrl_linf")
    object_ctrl_linf = _float_array(window, "object_ctrl_linf")
    full_pelvis_z = _float_array(rows, "sim_pelvis_z_m")

    mask_window = mask[start : end + 1]
    active = mask_window.any(axis=1)
    metrics: dict[str, object] = {
        "case_window_source": "mask_first_last_active_pad10",
        "case_window_mask_key": mask_key,
        "case_window_mask_path": mask_path,
        "case_window_has_mask_contact": bool(has_contact),
        "case_window_start_frame": int(start),
        "case_window_end_frame": int(end),
        "case_window_start_eval_time_s": float(start / FPS),
        "case_window_end_eval_time_s": float(end / FPS),
        "case_window_num_frames": int(end - start + 1),
        "case_window_mask_active_pct": float(active.mean() * 100.0),
        "case_window_mask_left_active_pct": float(mask_window[:, 0].mean() * 100.0),
        "case_window_mask_right_active_pct": float(mask_window[:, 1].mean() * 100.0),
        "case_window_obj_err_mean_m": float(obj_err.mean()),
        "case_window_obj_err_max_m": float(obj_err.max()),
        "case_window_pelvis_z_min_m": float(pelvis_z.min()),
        "case_window_sim_contact_frames_pct": float(sim_contact.mean() * 100.0),
        "case_window_ref_contact_frames_pct": float(ref_contact.mean() * 100.0),
        "case_window_sim_min_hand_sdf_mean_m": float(min_hand_sdf.mean()),
        "case_window_sim_min_hand_sdf_max_m": float(min_hand_sdf.max()),
        "case_window_robot_ctrl_linf_max": float(robot_ctrl_linf.max()),
        "case_window_object_ctrl_linf_max": float(object_ctrl_linf.max()),
        "full_pelvis_z_min_m": float(full_pelvis_z.min()),
    }
    metrics["E079_success_case_window"] = bool(
        metrics["case_window_pelvis_z_min_m"] >= 0.55
        and metrics["case_window_sim_contact_frames_pct"] >= 50.0
        and metrics["case_window_obj_err_mean_m"] <= 0.20
    )
    return metrics


def write_variant_summary(summary: dict[str, object]) -> None:
    variant = str(summary["variant"])
    summary_json = RESULTS / f"eval_summary_{variant}.json"
    summary_csv = RESULTS / f"eval_summary_{variant}.csv"
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)


def main() -> None:
    variants = read_variants()
    selected = sys.argv[1:] or list(variants.keys())

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "plots").mkdir(parents=True, exist_ok=True)
    e078.RESULTS = RESULTS
    e078.VARIANTS = variants

    summaries = []
    for variant in selected:
        if variant not in variants:
            print(f"[SKIP] unknown variant {variant}")
            continue
        try:
            summary = e078.evaluate_variant(variant)
        except FileNotFoundError as exc:
            print(f"[SKIP] {exc}")
            continue
        summary.update(case_window_metrics(summary))
        summary["split"] = variants[variant]["split"]
        summary["role"] = variants[variant]["role"]
        summary["E079_success_numeric"] = bool(
            summary["post2_pelvis_z_min_m"] >= 0.55
            and summary["post2_sim_contact_frames_pct"] >= 50.0
            and summary["post2_obj_err_mean_m"] <= 0.20
        )
        write_variant_summary(summary)
        summaries.append(summary)

    if not summaries:
        raise SystemExit("No E079 variant results found.")

    keys = sorted({k for row in summaries for k in row.keys()})
    comparison = RESULTS / "comparison.csv"
    with comparison.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(summaries)

    main_rows = [r for r in summaries if r["role"] == "main"]
    aggregate = {
        "num_results": len(summaries),
        "num_main_results": len(main_rows),
        "num_main_numeric_success": sum(bool(r["E079_success_numeric"]) for r in main_rows),
        "main_numeric_success_pct": (
            100.0 * sum(bool(r["E079_success_numeric"]) for r in main_rows) / len(main_rows)
            if main_rows
            else 0.0
        ),
        "num_main_case_window_success": sum(
            bool(r["E079_success_case_window"]) for r in main_rows
        ),
        "main_case_window_success_pct": (
            100.0
            * sum(bool(r["E079_success_case_window"]) for r in main_rows)
            / len(main_rows)
            if main_rows
            else 0.0
        ),
        "guard_results": [r["variant"] for r in summaries if r["role"] == "guard"],
    }
    (RESULTS / "aggregate_summary.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True), encoding="utf-8"
    )

    print(f"Wrote {comparison}")
    print(json.dumps(aggregate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
