#!/usr/bin/env python3
"""E027 contact timing diagnostics for low-contact cases."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

import e027_common as C


def _load_quality_by_case() -> dict[str, dict[str, str]]:
    return C.index_by_case(C.load_quality_rows())


def _bool_series_from_counts(rows: list[dict[str, str]], key: str) -> np.ndarray:
    return np.asarray([C.as_float(row.get(key), 0.0) > 0.0 for row in rows], dtype=bool)


def _float_series(rows: list[dict[str, str]], key: str) -> np.ndarray:
    return np.asarray([C.as_float(row.get(key), float("nan")) for row in rows], dtype=np.float64)


def _selected_raw_contact(case: str, target_len: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data, person_idx = C.load_mask_npz_for_case(case)
    if data is None or "eval_contact_mask_3cm" not in data.files:
        empty = np.zeros(target_len, dtype=bool)
        return empty, empty, empty
    contact = np.asarray(data["eval_contact_mask_3cm"], dtype=bool)
    selected = contact[:, person_idx, :]
    selected = C.align_array(selected, target_len)
    left = selected[:, 0].astype(bool)
    right = selected[:, 1].astype(bool)
    return left, right, np.logical_or(left, right)


def _best_shift(ref: np.ndarray, sim: np.ndarray) -> tuple[int, float, float]:
    if len(ref) == 0:
        return 0, 0.0, 0.0
    current = float((ref & sim).sum() / max(ref.sum(), 1) * 100.0)
    best_shift = 0
    best = current
    for shift in range(-6, 7):
        if shift < 0:
            shifted = np.pad(sim[-shift:], (0, -shift), constant_values=False)
        elif shift > 0:
            shifted = np.pad(sim[:-shift], (shift, 0), constant_values=False)
        else:
            shifted = sim
        score = float((ref & shifted).sum() / max(ref.sum(), 1) * 100.0)
        if score > best:
            best = score
            best_shift = shift
    return best_shift, current, best


def _classify_frame(
    *,
    label: str,
    ref_contact: bool,
    raw_contact: bool,
    sim_contact: bool,
    sim_sdf: float,
    ref_sdf: float,
    obj_err: float,
    pelvis_z: float,
) -> str:
    if sim_contact:
        return "hit"
    if label in {"discard_from_success_denominator", "raw_data_questionable", "retarget_questionable"}:
        return "raw_or_retarget_questionable"
    if not ref_contact and not raw_contact:
        return "no_contact_expected"
    if pelvis_z == pelvis_z and pelvis_z < 0.45:
        return "unstable_posture"
    if obj_err == obj_err and obj_err > 0.08:
        return "object_lag"
    if sim_sdf == sim_sdf and sim_sdf > 0.10:
        return "too_far"
    if ref_sdf == ref_sdf and ref_sdf < 0.03 and sim_sdf == sim_sdf and sim_sdf > 0.03:
        return "timing_or_wrong_side"
    return "timing_or_wrong_side"


def diagnose_case(case: str, quality: dict[str, str]) -> dict[str, Any]:
    variant = C.selected_variant_for_case(case)
    ts_path = C.timeseries_path_for_variant(variant)
    rows = C.read_rows(ts_path)
    if not rows:
        raise FileNotFoundError(f"missing timeseries for {case}: {ts_path}")

    sim_sdf = _float_series(rows, "sim_min_hand_sdf_m")
    ref_sdf = _float_series(rows, "ref_min_hand_sdf_m")
    # Use the same physical interpretation as the E019/E026 paper-facing
    # contact metric: hand-object proximity within 5cm. The raw contact count
    # columns can include other MuJoCo contacts and are not strict enough for
    # E027 timing diagnosis.
    sim_contact = np.asarray(sim_sdf <= 0.05, dtype=bool)
    ref_contact = np.asarray(ref_sdf <= 0.05, dtype=bool)
    raw_l, raw_r, raw_any = _selected_raw_contact(case, len(rows))
    obj_err = _float_series(rows, "obj_err_m")
    pelvis_z = _float_series(rows, "sim_pelvis_z_m")
    obj_lag = _float_series(rows, "obj_err_m")

    shift, current_overlap, best_overlap = _best_shift(np.logical_or(ref_contact, raw_any), sim_contact)
    label = quality.get("quality_label", "")

    per_frame: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        miss = _classify_frame(
            label=label,
            ref_contact=bool(ref_contact[i]),
            raw_contact=bool(raw_any[i]),
            sim_contact=bool(sim_contact[i]),
            sim_sdf=float(sim_sdf[i]),
            ref_sdf=float(ref_sdf[i]),
            obj_err=float(obj_err[i]),
            pelvis_z=float(pelvis_z[i]),
        )
        per_frame.append(
            {
                "frame": row.get("frame", i),
                "eval_time_s": row.get("eval_time_s", ""),
                "npz_time_s": row.get("npz_time_s", ""),
                "ref_contact_left": "",
                "ref_contact_right": "",
                "raw_contact_left": bool(raw_l[i]),
                "raw_contact_right": bool(raw_r[i]),
                "robot_contact_5cm_left": "",
                "robot_contact_5cm_right": "",
                "ref_contact_any": bool(ref_contact[i]),
                "raw_contact_any": bool(raw_any[i]),
                "sim_contact_any": bool(sim_contact[i]),
                "sim_hand_object_dist_left_m": "",
                "sim_hand_object_dist_right_m": "",
                "ref_hand_object_dist_left_m": "",
                "ref_hand_object_dist_right_m": "",
                "sim_min_hand_sdf_m": sim_sdf[i],
                "ref_min_hand_sdf_m": ref_sdf[i],
                "object_pos_lag_m": obj_lag[i],
                "support_proxy_lag_m": "",
                "contact_miss_reason": miss,
                "phase_shift_suggestion_frames": shift,
            }
        )

    counts = Counter(row["contact_miss_reason"] for row in per_frame)
    active = np.logical_or(ref_contact, raw_any)
    active_miss = [
        row["contact_miss_reason"]
        for row, is_active in zip(per_frame, active)
        if is_active and row["contact_miss_reason"] != "hit"
    ]
    active_counts = Counter(active_miss)
    expected_gain = max(0.0, best_overlap - current_overlap)
    best_contact = C.as_float(quality.get("best_contact5_pct"), 0.0)

    if label in {"discard_from_success_denominator", "raw_data_questionable", "retarget_questionable"}:
        next_action = "retarget_fix" if label == "retarget_questionable" else "discard"
    elif best_contact < 70.0 and expected_gain >= 10.0:
        next_action = "E027_full_variant"
    elif best_contact < 70.0:
        next_action = "E030_geometry"
    else:
        next_action = "E028_barrier" if C.as_float(quality.get("best_deep_pen_pct"), 0.0) > 20.0 else "no_timing_action"

    out_csv = C.TIMING / f"{case}.csv"
    C.write_rows(out_csv, per_frame)
    panel = C.TIMING / f"{case}.md"
    panel.write_text(
        "\n".join(
            [
                f"# E027 Timing Panel: {case}",
                "",
                f"- quality_label: `{label}`",
                f"- selected_variant: `{variant}`",
                f"- discard_from_p0: `{quality.get('discard_from_p0', '')}`",
                f"- discard_from_success_denominator: `{quality.get('discard_from_success_denominator', '')}`",
                f"- current contact overlap: `{current_overlap:.2f}%`",
                f"- best shift: `{shift}` frames",
                f"- shifted overlap: `{best_overlap:.2f}%`",
                f"- expected gain: `{expected_gain:.2f}pp`",
                f"- recommended_next_action: `{next_action}`",
                "",
                "## Miss Breakdown",
                "",
                "| Class | Frames | Active-contact frames |",
                "|---|---:|---:|",
                *[
                    f"| `{name}` | {counts.get(name, 0)} | {active_counts.get(name, 0)} |"
                    for name in sorted(set(counts) | set(active_counts))
                ],
                "",
                "## Evidence Links",
                "",
                f"- per-frame CSV: `{out_csv}`",
                f"- E020 panel: `{C.panel_path(case)}`",
                f"- online video: `{C.video_path(case)}`",
                "",
                f"Primary evidence: {quality.get('primary_evidence', '')}",
                f"Counter evidence: {quality.get('counter_evidence', '')}",
            ]
        ),
        encoding="utf-8",
    )

    return {
        "case": case,
        "quality_label": label,
        "selected_variant": variant,
        "timing_panel": str(panel),
        "timing_csv": str(out_csv),
        "frames": len(rows),
        "active_contact_frames": int(active.sum()),
        "sim_contact_frames": int(sim_contact.sum()),
        "current_overlap_pct": current_overlap,
        "recommended_shift_frames": shift,
        "shifted_overlap_pct": best_overlap,
        "expected_contact_gain_pp": expected_gain,
        "dominant_miss_class": active_counts.most_common(1)[0][0] if active_counts else "",
        "recommended_next_action": next_action,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--all", action="store_true", help="run the priority timing cases")
    ap.add_argument("--case", action="append", help="specific case; can be repeated")
    args = ap.parse_args()

    quality_by_case = _load_quality_by_case()
    if not quality_by_case:
        raise FileNotFoundError("Run audit_case_data_quality.py --all first")

    if args.case:
        cases = args.case
    elif args.all:
        cases = C.TIMING_PRIORITY
    else:
        ap.error("Use --all or --case")

    C.TIMING.mkdir(parents=True, exist_ok=True)
    summaries = [diagnose_case(case, quality_by_case[case]) for case in cases]
    C.write_rows(C.TIMING / "timing_summary.csv", summaries)
    C.write_json(
        C.TIMING / "timing_summary.json",
        {"num_cases": len(summaries), "cases": summaries},
    )
    print(f"[E027] wrote {len(summaries)} timing panels to {C.TIMING}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
