#!/usr/bin/env python3
"""Build E164 bimanual global contact masks.

E164 only rewrites raw/spider masks for the target person:
global = fill_internal_holes(max(left, right)); left = right = global.
The eval mask is intentionally kept unchanged and E164 CEM uses spider axis.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np


REPO = Path(__file__).resolve().parents[5]
E156_VARIANTS = REPO / "workspace/core4d/scripts/experiments/E156/variants.tsv"
RESULT_ROOT = REPO / "workspace/core4d/results/E164/bimanual_global_mask_reward"
MASK_ROOT = RESULT_ROOT / "contact_masks"
DIAG_ROOT = RESULT_ROOT / "diagnostics/masks"

TARGET_CASES = ["box023_person2", "box004_082_p1", "box026_139_p1"]
MASK_KEYS = ["raw_contact_mask_3cm", "spider_contact_mask_3cm"]


def rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(REPO.resolve()))
    except Exception:
        return str(path)


def repo_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO / p


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def fill_internal_holes(mask: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int]]]:
    out = np.asarray(mask, dtype=bool).copy()
    active = np.flatnonzero(out)
    if active.size == 0:
        return out, []
    first = int(active[0])
    last = int(active[-1])
    holes: list[tuple[int, int]] = []
    i = first
    while i <= last:
        if out[i]:
            i += 1
            continue
        start = i
        while i <= last and not out[i]:
            i += 1
        end = i - 1
        holes.append((start, end))
    out[first : last + 1] = True
    return out, holes


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    arr = np.asarray(mask, dtype=bool)
    spans: list[tuple[int, int]] = []
    i = 0
    while i < arr.shape[0]:
        if not arr[i]:
            i += 1
            continue
        start = i
        while i < arr.shape[0] and arr[i]:
            i += 1
        spans.append((start, i - 1))
    return spans


def span_text(spans: list[tuple[int, int]]) -> str:
    return ",".join(f"{a}-{b}" for a, b in spans)


def process_key(
    arrays: dict[str, np.ndarray],
    key: str,
    person_idx: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    original = np.asarray(arrays[key])
    if original.ndim != 3 or original.shape[2] != 2:
        raise ValueError(f"{key} expected shape (T, persons, 2), got {original.shape}")
    if person_idx < 0 or person_idx >= original.shape[1]:
        raise ValueError(f"{key} person_idx={person_idx} out of shape {original.shape}")

    processed = original.copy()
    left = original[:, person_idx, 0].astype(bool)
    right = original[:, person_idx, 1].astype(bool)
    union = left | right
    global_mask, holes = fill_internal_holes(union)
    processed[:, person_idx, 0] = global_mask.astype(processed.dtype)
    processed[:, person_idx, 1] = global_mask.astype(processed.dtype)

    meta = {
        "key": key,
        "frames": int(original.shape[0]),
        "persons": int(original.shape[1]),
        "person_idx": int(person_idx),
        "before_left_active": int(left.sum()),
        "before_right_active": int(right.sum()),
        "before_union_active": int(union.sum()),
        "after_left_active": int(global_mask.sum()),
        "after_right_active": int(global_mask.sum()),
        "before_left_runs": span_text(runs(left)),
        "before_right_runs": span_text(runs(right)),
        "before_union_runs": span_text(runs(union)),
        "after_global_runs": span_text(runs(global_mask)),
        "filled_hole_segments": span_text(holes),
        "filled_hole_count": len(holes),
    }
    return processed, meta


def plot_case(case_id: str, before: dict[str, np.ndarray], after: dict[str, np.ndarray], person_idx: int, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(MASK_KEYS), 1, figsize=(12, 2.8 * len(MASK_KEYS)), squeeze=False)
    for ax, key in zip(axes[:, 0], MASK_KEYS, strict=True):
        rows = [
            before[key][:, person_idx, 0].astype(int),
            before[key][:, person_idx, 1].astype(int),
            np.maximum(before[key][:, person_idx, 0], before[key][:, person_idx, 1]).astype(int),
            after[key][:, person_idx, 0].astype(int),
            after[key][:, person_idx, 1].astype(int),
        ]
        img = np.vstack(rows)
        ax.imshow(img, aspect="auto", interpolation="nearest", cmap="Greys", vmin=0, vmax=1)
        ax.set_title(f"{case_id} {key}")
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels(["before L", "before R", "before union", "after L", "after R"])
        ax.set_xlabel("frame")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def source_rows() -> dict[str, dict[str, str]]:
    rows = read_tsv(E156_VARIANTS)
    return {
        row["short_case_id"]: row
        for row in rows
        if row.get("short_case_id") in TARGET_CASES and row.get("method") == "+gateA"
    }


def build() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_case = source_rows()
    missing = [case for case in TARGET_CASES if case not in by_case]
    if missing:
        raise RuntimeError(f"E156 +gateA source rows missing: {missing}")

    rows: list[dict[str, Any]] = []
    all_meta: list[dict[str, Any]] = []
    for case_id in TARGET_CASES:
        row = by_case[case_id]
        person_idx = int(row["person_idx"])
        src = repo_path(row["mask_path"])
        if not src.is_file():
            raise FileNotFoundError(src)
        data = np.load(src, allow_pickle=True)
        arrays = {key: data[key] for key in data.files}
        processed = dict(arrays)
        key_meta: list[dict[str, Any]] = []
        for key in MASK_KEYS:
            if key not in arrays:
                raise KeyError(f"{src} missing {key}")
            processed[key], meta = process_key(arrays, key, person_idx)
            meta.update({"case_id": case_id, "source_mask_path": rel(src)})
            key_meta.append(meta)
            all_meta.append(meta)

        out_dir = MASK_ROOT / case_id
        out_dir.mkdir(parents=True, exist_ok=True)
        out_npz = out_dir / "raw_contact_mask_3cm.npz"
        metadata = {
            "experiment": "E164",
            "case_id": case_id,
            "person_idx": person_idx,
            "source_mask_path": rel(src),
            "processed_mask_path": rel(out_npz),
            "processed_keys": MASK_KEYS,
            "eval_contact_mask_3cm_policy": "preserved_unmodified; E164 CEM forces spider axis",
            "keys": key_meta,
        }
        processed["e164_metadata_json"] = np.asarray(json.dumps(metadata, ensure_ascii=False))
        np.savez_compressed(out_npz, **processed)
        (out_dir / "mask_metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        plot_case(
            case_id,
            {key: arrays[key] for key in MASK_KEYS},
            {key: processed[key] for key in MASK_KEYS},
            person_idx,
            DIAG_ROOT / f"{case_id}_raw_spider_before_after.png",
        )
        rows.append(
            {
                "case_id": case_id,
                "person_idx": person_idx,
                "source_mask_path": rel(src),
                "processed_mask_path": rel(out_npz),
                "metadata_path": rel(out_dir / "mask_metadata.json"),
                "diagnostic_png": rel(DIAG_ROOT / f"{case_id}_raw_spider_before_after.png"),
                "processed_keys": ",".join(MASK_KEYS),
                "eval_mask_policy": "preserved_unmodified",
            }
        )

    fields = [
        "case_id",
        "person_idx",
        "source_mask_path",
        "processed_mask_path",
        "metadata_path",
        "diagnostic_png",
        "processed_keys",
        "eval_mask_policy",
    ]
    write_tsv(RESULT_ROOT / "contact_masks/e164_bimanual_global_masks.tsv", rows, fields)
    write_tsv(
        DIAG_ROOT / "e164_bimanual_global_mask_segments.tsv",
        all_meta,
        [
            "case_id",
            "key",
            "frames",
            "persons",
            "person_idx",
            "before_left_active",
            "before_right_active",
            "before_union_active",
            "after_left_active",
            "after_right_active",
            "before_left_runs",
            "before_right_runs",
            "before_union_runs",
            "after_global_runs",
            "filled_hole_segments",
            "filled_hole_count",
            "source_mask_path",
        ],
    )
    summary = {
        "experiment": "E164",
        "case_count": len(rows),
        "target_cases": TARGET_CASES,
        "processed_keys": MASK_KEYS,
        "mask_table": rel(RESULT_ROOT / "contact_masks/e164_bimanual_global_masks.tsv"),
        "segments_tsv": rel(DIAG_ROOT / "e164_bimanual_global_mask_segments.tsv"),
        "all_outputs_exist": all(repo_path(row["processed_mask_path"]).is_file() for row in rows),
    }
    (RESULT_ROOT / "contact_masks/e164_bimanual_global_masks_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return rows, summary


def main() -> None:
    rows, summary = build()
    print(
        "E164 masks: "
        f"cases={len(rows)} processed_keys={summary['processed_keys']} "
        f"all_outputs_exist={summary['all_outputs_exist']}"
    )


if __name__ == "__main__":
    main()
