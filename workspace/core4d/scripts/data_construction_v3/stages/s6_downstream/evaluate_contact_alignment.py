#!/usr/bin/env python3
"""Evaluate contact alignment against S1 raw-contact artifacts.

This first E111 evaluator is intentionally mask-based. It reports raw-contact
coverage for every manifest row and computes precision/recall/F1/IoU only when
a method contact mask is explicitly available. MuJoCo replay/SDF dependent
metrics remain blank here and are handled by later E112/E113 evaluators.
"""

from __future__ import annotations

import argparse
import math
from collections import Counter
from pathlib import Path
import sys
from typing import Any

SCRIPT_ROOT = next(parent for parent in Path(__file__).resolve().parents if parent.name == "data_construction_v3")
for _path in (SCRIPT_ROOT / "lib", SCRIPT_ROOT / "state", SCRIPT_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import numpy as np

from common import SCHEMA_VERSION, read_tsv, timestamp, write_json, write_tsv


HANDS = ("left", "right")

FIELDS = [
    "case_id",
    "method",
    "retarget_variant_id",
    "target_variant_id",
    "object_key",
    "person",
    "person_idx",
    "contact_mask_label",
    "raw_contact_artifact_npz",
    "method_contact_mask_npz",
    "alignment_status",
    "failure_mode_label",
    "raw_frame_count",
    "method_frame_count",
    "raw_any_active_frac",
    "raw_left_active_frac",
    "raw_right_active_frac",
    "raw_both_active_frac",
    "raw_any_longest_run_frac",
    "raw_left_longest_run_frac",
    "raw_right_longest_run_frac",
    "method_any_active_frac",
    "method_left_active_frac",
    "method_right_active_frac",
    "raw_contact_precision",
    "raw_contact_recall",
    "raw_contact_f1",
    "raw_contact_iou",
    "left_precision",
    "left_recall",
    "left_f1",
    "right_precision",
    "right_recall",
    "right_f1",
    "physics_contact_frac",
    "sdf_deep_pen_frac",
    "sdf_shallow_pen_frac",
    "sdf_near_0_2cm_frac",
    "sdf_near_2_5cm_frac",
    "schema_version",
    "updated_at",
]


def fmt(value: float) -> str:
    return f"{value:.6f}" if math.isfinite(value) else ""


def fraction(mask: np.ndarray) -> float:
    return float(np.mean(mask.astype(bool))) if mask.size else math.nan


def longest_run_fraction(mask: np.ndarray) -> float:
    if not mask.size:
        return math.nan
    best = 0
    cur = 0
    for value in mask.astype(bool):
        if value:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return float(best / mask.size)


def binary_scores(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    tp = float(np.logical_and(pred, gt).sum())
    fp = float(np.logical_and(pred, ~gt).sum())
    fn = float(np.logical_and(~pred, gt).sum())
    union = float(np.logical_or(pred, gt).sum())
    precision = tp / (tp + fp) if (tp + fp) else math.nan
    recall = tp / (tp + fn) if (tp + fn) else math.nan
    f1 = 2.0 * precision * recall / (precision + recall) if math.isfinite(precision) and math.isfinite(recall) and (precision + recall) else math.nan
    iou = tp / union if union else math.nan
    return {"precision": precision, "recall": recall, "f1": f1, "iou": iou}


def resolve_path(path_text: str, repo_root: Path) -> Path:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path


def load_raw_mask(path: Path, label: str, person_idx: int) -> tuple[np.ndarray | None, str]:
    if not path.is_file() or path.stat().st_size <= 0:
        return None, "raw_contact_artifact_missing"
    try:
        data = np.load(path, allow_pickle=True)
    except Exception as exc:  # noqa: BLE001
        return None, f"raw_contact_npz_load_error:{type(exc).__name__}"
    key = f"raw_contact_mask_{label}"
    if key not in data:
        return None, f"raw_contact_key_missing:{key}"
    arr = np.asarray(data[key]).astype(bool)
    if arr.ndim == 4:
        if person_idx < 0 or person_idx >= arr.shape[1] or arr.shape[2] < 2:
            return None, "raw_contact_shape_person_hand_mismatch"
        return arr[:, person_idx, :2], ""
    if arr.ndim == 3 and arr.shape[2] >= 2:
        if person_idx < 0 or person_idx >= arr.shape[1]:
            return None, "raw_contact_shape_person_mismatch"
        return arr[:, person_idx, :2], ""
    if arr.ndim == 2 and arr.shape[1] >= 2:
        return arr[:, :2], ""
    return None, f"raw_contact_shape_unsupported:{list(arr.shape)}"


def load_method_mask(path: Path, label: str, person_idx: int) -> tuple[np.ndarray | None, str]:
    if not path.is_file() or path.stat().st_size <= 0:
        return None, "method_contact_mask_missing"
    try:
        data = np.load(path, allow_pickle=True)
    except Exception as exc:  # noqa: BLE001
        return None, f"method_contact_npz_load_error:{type(exc).__name__}"
    for key in (f"method_contact_mask_{label}", f"contact_mask_{label}", f"raw_contact_mask_{label}", "contact_mask", "method_contact_mask"):
        if key not in data:
            continue
        arr = np.asarray(data[key]).astype(bool)
        if arr.ndim == 4:
            if person_idx < 0 or person_idx >= arr.shape[1] or arr.shape[2] < 2:
                return None, "method_contact_shape_person_hand_mismatch"
            return arr[:, person_idx, :2], ""
        if arr.ndim == 3 and arr.shape[2] >= 2:
            if person_idx < 0 or person_idx >= arr.shape[1]:
                return None, "method_contact_shape_person_mismatch"
            return arr[:, person_idx, :2], ""
        if arr.ndim == 2 and arr.shape[1] >= 2:
            return arr[:, :2], ""
        if arr.ndim == 1:
            return np.stack([arr, arr], axis=1), ""
        return None, f"method_contact_shape_unsupported:{list(arr.shape)}"
    return None, "method_contact_key_missing"


def row_label(row: dict[str, str]) -> str:
    return row.get("contact_mask_label") or row.get("raw_contact_threshold_label") or row.get("contact_label") or "3cm"


def person_idx(row: dict[str, str]) -> int:
    for key in ("contact_mask_person_idx", "person_idx"):
        try:
            return int(str(row.get(key, "")).strip())
        except ValueError:
            pass
    return 0 if row.get("person") != "person2" else 1


def build_metric_row(row: dict[str, str], repo_root: Path, method_name: str) -> dict[str, Any]:
    label = row_label(row)
    pidx = person_idx(row)
    raw_path_text = row.get("raw_contact_artifact_npz") or row.get("contact_mask_npz") or row.get("contact_mask", "")
    method_path_text = row.get("method_contact_mask_npz") or row.get("contact_mask_npz") or row.get("contact_mask", "")
    raw_mask, raw_error = load_raw_mask(resolve_path(raw_path_text, repo_root), label, pidx) if raw_path_text else (None, "raw_contact_artifact_missing")
    method_mask: np.ndarray | None = None
    method_error = "method_contact_mask_missing"
    if method_path_text and method_path_text != raw_path_text:
        method_mask, method_error = load_method_mask(resolve_path(method_path_text, repo_root), label, pidx)
    elif row.get("method_contact_mask_npz"):
        method_mask, method_error = load_method_mask(resolve_path(method_path_text, repo_root), label, pidx)

    out: dict[str, Any] = {
        "case_id": row.get("case_id", ""),
        "method": row.get("method", method_name),
        "retarget_variant_id": row.get("retarget_variant_id", ""),
        "target_variant_id": row.get("target_variant_id", ""),
        "object_key": row.get("object_key", ""),
        "person": row.get("person", ""),
        "person_idx": str(pidx),
        "contact_mask_label": label,
        "raw_contact_artifact_npz": raw_path_text,
        "method_contact_mask_npz": row.get("method_contact_mask_npz", ""),
        "alignment_status": "",
        "failure_mode_label": "",
        "raw_frame_count": "",
        "method_frame_count": "",
        "raw_any_active_frac": "",
        "raw_left_active_frac": "",
        "raw_right_active_frac": "",
        "raw_both_active_frac": "",
        "raw_any_longest_run_frac": "",
        "raw_left_longest_run_frac": "",
        "raw_right_longest_run_frac": "",
        "method_any_active_frac": "",
        "method_left_active_frac": "",
        "method_right_active_frac": "",
        "raw_contact_precision": "",
        "raw_contact_recall": "",
        "raw_contact_f1": "",
        "raw_contact_iou": "",
        "left_precision": "",
        "left_recall": "",
        "left_f1": "",
        "right_precision": "",
        "right_recall": "",
        "right_f1": "",
        "physics_contact_frac": "",
        "sdf_deep_pen_frac": "",
        "sdf_shallow_pen_frac": "",
        "sdf_near_0_2cm_frac": "",
        "sdf_near_2_5cm_frac": "",
        "schema_version": SCHEMA_VERSION,
        "updated_at": timestamp(),
    }
    if raw_mask is None:
        out["alignment_status"] = "missing_raw_contact"
        out["failure_mode_label"] = raw_error
        return out

    raw_left = raw_mask[:, 0]
    raw_right = raw_mask[:, 1]
    raw_any = raw_left | raw_right
    raw_both = raw_left & raw_right
    out.update(
        {
            "raw_frame_count": str(raw_mask.shape[0]),
            "raw_any_active_frac": fmt(fraction(raw_any)),
            "raw_left_active_frac": fmt(fraction(raw_left)),
            "raw_right_active_frac": fmt(fraction(raw_right)),
            "raw_both_active_frac": fmt(fraction(raw_both)),
            "raw_any_longest_run_frac": fmt(longest_run_fraction(raw_any)),
            "raw_left_longest_run_frac": fmt(longest_run_fraction(raw_left)),
            "raw_right_longest_run_frac": fmt(longest_run_fraction(raw_right)),
        }
    )
    if method_mask is None:
        out["alignment_status"] = "raw_contact_only"
        out["failure_mode_label"] = method_error
        return out
    n = min(raw_mask.shape[0], method_mask.shape[0])
    raw_eval = raw_mask[:n]
    method_eval = method_mask[:n]
    method_left = method_eval[:, 0]
    method_right = method_eval[:, 1]
    method_any = method_left | method_right
    scores_any = binary_scores(method_any, raw_eval[:, 0] | raw_eval[:, 1])
    scores_left = binary_scores(method_left, raw_eval[:, 0])
    scores_right = binary_scores(method_right, raw_eval[:, 1])
    out.update(
        {
            "alignment_status": "evaluated",
            "failure_mode_label": "contact_alignment_evaluated",
            "method_frame_count": str(method_mask.shape[0]),
            "method_any_active_frac": fmt(fraction(method_any)),
            "method_left_active_frac": fmt(fraction(method_left)),
            "method_right_active_frac": fmt(fraction(method_right)),
            "raw_contact_precision": fmt(scores_any["precision"]),
            "raw_contact_recall": fmt(scores_any["recall"]),
            "raw_contact_f1": fmt(scores_any["f1"]),
            "raw_contact_iou": fmt(scores_any["iou"]),
            "left_precision": fmt(scores_left["precision"]),
            "left_recall": fmt(scores_left["recall"]),
            "left_f1": fmt(scores_left["f1"]),
            "right_precision": fmt(scores_right["precision"]),
            "right_recall": fmt(scores_right["recall"]),
            "right_f1": fmt(scores_right["f1"]),
        }
    )
    return out


def summarize(rows: list[dict[str, Any]], out_dir: Path) -> dict[str, Any]:
    evaluated = [row for row in rows if row.get("alignment_status") == "evaluated"]
    f1_values = [float(row["raw_contact_f1"]) for row in evaluated if row.get("raw_contact_f1")]
    return {
        "stage": "S6_contact_alignment",
        "created_at": timestamp(),
        "schema_version": SCHEMA_VERSION,
        "rows": len(rows),
        "alignment_status_counts": dict(Counter(row.get("alignment_status", "") for row in rows)),
        "failure_mode_counts": dict(Counter(row.get("failure_mode_label", "") for row in rows if row.get("failure_mode_label"))),
        "mean_raw_contact_f1": fmt(float(np.mean(f1_values))) if f1_values else "",
        "out_dir": str(out_dir),
        "notes": [
            "Mask PR/F1 is computed only when method_contact_mask_npz is explicit.",
            "Replay/SDF-dependent fields are intentionally blank in E111 smoke mode.",
        ],
    }


def markdown_summary(summary: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    lines = [
        "# S6 contact alignment summary",
        "",
        f"- rows: `{summary['rows']}`",
        f"- mean raw-contact F1: `{summary['mean_raw_contact_f1']}`",
        "",
        "## alignment status",
        "",
        "| status | count |",
        "|---|---:|",
    ]
    for key, count in summary["alignment_status_counts"].items():
        lines.append(f"| `{key}` | {count} |")
    lines.extend(["", "## rows", "", "| status | case | label | raw any | precision | recall | f1 |", "|---|---|---|---:|---:|---:|---:|"])
    for row in rows[:50]:
        lines.append(
            f"| `{row['alignment_status']}` | `{row['case_id']}` | `{row['contact_mask_label']}` | "
            f"{row['raw_any_active_frac']} | {row['raw_contact_precision']} | {row['raw_contact_recall']} | {row['raw_contact_f1']} |"
        )
    lines.extend(
        [
            "",
            "E111 does not claim physics contact or SDF quality unless replay-derived masks are explicitly supplied.",
            "",
        ]
    )
    return "\n".join(lines)


def make_smoke_fixture(out_dir: Path) -> Path:
    fixture = out_dir / "fixture"
    fixture.mkdir(parents=True, exist_ok=True)
    raw = np.zeros((8, 2, 2), dtype=bool)
    raw[[1, 2, 3], 0, 0] = True
    raw[[2, 3, 4], 0, 1] = True
    method = np.zeros((8, 2), dtype=bool)
    method[[2, 3, 4], 0] = True
    method[[3, 4], 1] = True
    mask_path = fixture / "contact_masks.npz"
    np.savez_compressed(mask_path, raw_contact_mask_3cm=raw, method_contact_mask_3cm=method)
    manifest = fixture / "contact_alignment_input.tsv"
    rows = [
        {
            "case_id": "smoke_contact_case",
            "method": "smoke_method",
            "retarget_variant_id": "omnirt_v1",
            "target_variant_id": "ref_fk",
            "object_key": "box004",
            "person": "person1",
            "person_idx": "0",
            "contact_mask_label": "3cm",
            "raw_contact_artifact_npz": str(mask_path),
            "method_contact_mask_npz": str(mask_path),
        }
    ]
    write_tsv(manifest, rows, list(rows[0].keys()))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-tsv", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--method-name", default="method")
    parser.add_argument("--make-smoke-fixture", action="store_true")
    args = parser.parse_args()

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = make_smoke_fixture(out_dir) if args.make_smoke_fixture else args.manifest_tsv
    if manifest is None:
        raise SystemExit("missing --manifest-tsv or --make-smoke-fixture")

    rows = [build_metric_row(row, args.repo_root.resolve(), args.method_name) for row in read_tsv(manifest)]
    write_tsv(out_dir / "contact_alignment_metrics.tsv", rows, FIELDS)
    summary = summarize(rows, out_dir)
    write_json(out_dir / "contact_alignment_summary.json", summary)
    (out_dir / "contact_alignment_summary.md").write_text(markdown_summary(summary, rows), encoding="utf-8")
    write_json(
        out_dir / "contact_sdf_histograms.json",
        {
            "stage": "S6_contact_alignment_histograms",
            "created_at": timestamp(),
            "schema_version": SCHEMA_VERSION,
            "status": "not_available_in_mask_only_e111",
            "histograms": [],
        },
    )
    print(f"[E111] wrote contact alignment to {out_dir}")
    print(f"[E111] rows={len(rows)} status_counts={summary['alignment_status_counts']}")


if __name__ == "__main__":
    main()
