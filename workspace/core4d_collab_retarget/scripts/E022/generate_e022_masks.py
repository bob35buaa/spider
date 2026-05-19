#!/usr/bin/env python3
"""Generate E022 variant tasks with patched contact masks."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path
from typing import Any

import numpy as np


REPO = Path(__file__).resolve().parents[4]
BASE = REPO / "example_datasets/processed/core4d/unitree_g1/humanoid_object"
WS = REPO / "workspace/core4d_collab_retarget"
VARIANTS = WS / "scripts/E022/variants.tsv"
E018B_RESULTS = WS / "results/E018b"
E018B_MANIFEST = E018B_RESULTS / "manifest.tsv"
RESULTS = WS / "results/E022"
MANIFEST = RESULTS / "manifest.tsv"

SOURCE_VARIANT = "E018b_box023_p1_canonical_t02"
SOURCE_TASK = "box023_person1_freejoint_legobj_e018b"
SOURCE_MASK_SLUG = "box023_person1"
PERSON_IDX = 0

FIELDNAMES = [
    "variant",
    "mask_mode",
    "mask_axis",
    "queue",
    "role",
    "wave",
    "hold_contact_rew_scale",
    "hold_contact_sigma",
    "hold_contact_start_eval_time",
    "hold_contact_end_eval_time",
    "hold_contact_require_ref_contact",
    "contact_hdmi_gain",
    "contact_hdmi_sigma",
    "notes",
]

EXTRA_FIELDS = [
    "E022_source_variant",
    "E022_source_task",
    "E022_mask_mode",
    "E022_mask_axis",
    "E022_contact_patch_applied",
    "E022_contact_patch_active_left_pct",
    "E022_contact_patch_active_right_pct",
    "E022_contact_patch_active_any_pct",
    "E022_notes",
    "contact_hdmi_mask_path",
    "contact_hdmi_mask_time_axis",
    "contact_hdmi_gain",
    "contact_hdmi_sigma",
]


def _read_tsv(path: Path, fieldnames: list[str] | None = None) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        lines = [line for line in f if line.strip() and not line.startswith("#")]
    return list(csv.DictReader(lines, delimiter="\t", fieldnames=fieldnames))


def _source_manifest_row() -> dict[str, str]:
    for row in _read_tsv(E018B_MANIFEST):
        if row["variant"] == SOURCE_VARIANT:
            return row
    raise KeyError(f"{SOURCE_VARIANT} not found in {E018B_MANIFEST}")


def _variant_slug(variant: str) -> str:
    prefix = "E022_box023_p1_"
    return variant[len(prefix) :] if variant.startswith(prefix) else variant


def _resize_mask(mask: np.ndarray, target_len: int) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if len(mask) == target_len:
        return mask
    if len(mask) == 0:
        return np.zeros((target_len,) + mask.shape[1:], dtype=bool)
    idx = np.round(np.linspace(0, len(mask) - 1, target_len)).astype(int)
    return mask[idx]


def _dilate_time(mask: np.ndarray, radius: int) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if radius <= 0 or len(mask) == 0:
        return mask.copy()
    out = np.zeros_like(mask, dtype=bool)
    for t in range(len(mask)):
        lo = max(0, t - radius)
        hi = min(len(mask), t + radius + 1)
        out[t] = mask[lo:hi].any(axis=0)
    return out


def _selected_mask(mask_npz: dict[str, np.ndarray], axis: str, mode: str) -> np.ndarray:
    key = f"{axis}_contact_mask_3cm"
    if key not in mask_npz:
        raise KeyError(f"{key} missing in source mask")
    raw = np.asarray(mask_npz[key], dtype=bool)
    if raw.ndim != 3 or raw.shape[1] <= PERSON_IDX:
        raise ValueError(f"{key} expected (T, person, hand), got {raw.shape}")
    selected = raw[:, PERSON_IDX, :2]
    if mode == "dilate3":
        selected = _dilate_time(selected, 3)
    elif mode not in {"baseline", "raw"}:
        raise ValueError(f"Unsupported mask_mode={mode}")
    return selected


def _write_mask_npz(src_npz: Path, dst_npz: Path, *, mode: str) -> None:
    src = np.load(src_npz, allow_pickle=True)
    arrays: dict[str, Any] = {key: src[key] for key in src.files}
    if mode == "dilate3":
        for axis in ("spider", "eval"):
            key = f"{axis}_contact_mask_3cm"
            if key in arrays:
                arr = np.asarray(arrays[key], dtype=bool).copy()
                arr[:, PERSON_IDX, :2] = _dilate_time(arr[:, PERSON_IDX, :2], 3)
                arrays[key] = arr.astype(np.float32)
    dst_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dst_npz, **arrays)


def _patch_task_contact(task: str, selected_mask: np.ndarray, *, apply_patch: bool) -> tuple[float, float, float]:
    path = BASE / task / "0/trajectory_kinematic.npz"
    data = np.load(path, allow_pickle=True)
    arrays: dict[str, Any] = {key: data[key] for key in data.files}
    contact = np.asarray(arrays["contact"]).copy()
    if contact.ndim != 2 or contact.shape[1] < 2:
        raise ValueError(f"{path} contact expected (T, >=2), got {contact.shape}")
    if apply_patch:
        mask = _resize_mask(selected_mask, len(contact)).astype(contact.dtype)
        contact[:, :2] = mask[:, :2]
        arrays["contact"] = contact
    active = contact[:, :2].astype(bool)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(f, **arrays)
    tmp.replace(path)
    return (
        float(active[:, 0].mean() * 100.0),
        float(active[:, 1].mean() * 100.0),
        float(active.any(axis=1).mean() * 100.0),
    )


def _copy_task(dst_task: str, *, force: bool) -> None:
    src = BASE / SOURCE_TASK
    dst = BASE / dst_task
    if force and dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, dirs_exist_ok=True)


def _load_npz_dict(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def generate(force: bool = False) -> None:
    source_meta = _source_manifest_row()
    source_mask = E018B_RESULTS / "contact_masks" / SOURCE_MASK_SLUG / "raw_contact_mask_3cm.npz"
    if not source_mask.is_file():
        raise FileNotFoundError(source_mask)
    source_mask_npz = _load_npz_dict(source_mask)

    rows: list[dict[str, str]] = []
    for row in _read_tsv(VARIANTS, FIELDNAMES):
        variant = row["variant"]
        mode = row["mask_mode"]
        axis = row["mask_axis"]
        slug = _variant_slug(variant)
        task = f"box023_person1_freejoint_legobj_e022_{slug}"
        mask_slug = variant
        mask_rel = (
            Path("workspace/core4d_collab_retarget/results/E022/contact_masks")
            / mask_slug
            / "raw_contact_mask_3cm.npz"
        )
        mask_abs = REPO / mask_rel

        _copy_task(task, force=force)
        _write_mask_npz(source_mask, mask_abs, mode=mode)
        if mode == "baseline":
            selected = _selected_mask(source_mask_npz, "spider", "raw")
            patch = False
        else:
            selected = _selected_mask(source_mask_npz, axis, mode)
            patch = True
        left_pct, right_pct, any_pct = _patch_task_contact(task, selected, apply_patch=patch)

        meta = dict(source_meta)
        meta.update(
            {
                "variant": variant,
                "source_task": "box023_person1",
                "derived_task": task,
                "mask_source_exp": "E022",
                "mask_slug": mask_slug,
                "person_idx": str(PERSON_IDX),
                "queue": row["queue"],
                "role": row["role"],
                "wave": row["wave"],
                "source_variant": SOURCE_VARIANT,
                "online_video_path": f"workspace/core4d_collab_retarget/results/E022/online_video/{variant}.mp4",
                "hold_contact_rew_scale": row["hold_contact_rew_scale"],
                "hold_contact_sigma": row["hold_contact_sigma"],
                "hold_contact_start_eval_time": row["hold_contact_start_eval_time"],
                "hold_contact_end_eval_time": row["hold_contact_end_eval_time"],
                "hold_contact_require_ref_contact": row["hold_contact_require_ref_contact"],
                "E022_source_variant": SOURCE_VARIANT,
                "E022_source_task": SOURCE_TASK,
                "E022_mask_mode": mode,
                "E022_mask_axis": axis,
                "E022_contact_patch_applied": str(patch).lower(),
                "E022_contact_patch_active_left_pct": f"{left_pct:.6g}",
                "E022_contact_patch_active_right_pct": f"{right_pct:.6g}",
                "E022_contact_patch_active_any_pct": f"{any_pct:.6g}",
                "E022_notes": row["notes"],
                "contact_hdmi_mask_path": str(mask_rel),
                "contact_hdmi_mask_time_axis": "auto" if mode == "baseline" else axis,
                "contact_hdmi_gain": row["contact_hdmi_gain"],
                "contact_hdmi_sigma": row["contact_hdmi_sigma"],
            }
        )
        rows.append(meta)

    fieldnames = list(source_meta.keys()) + [f for f in EXTRA_FIELDS if f not in source_meta]
    RESULTS.mkdir(parents=True, exist_ok=True)
    with MANIFEST.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {MANIFEST.relative_to(REPO)} ({len(rows)} variants)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    generate(force=args.force)


if __name__ == "__main__":
    main()
