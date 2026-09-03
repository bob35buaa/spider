#!/usr/bin/env python3
"""E206: semantic part-based collision proxies for desk / chair.

Rung 2 of plan236's fallback ladder, and the successor to the E177/E178 approach
that worked for buckets: instead of letting a voxel grid decide where the boxes
go, decompose the object into the parts a human names — tabletop, seat, back,
armrests, legs — and fit one axis-aligned box per part.

Why it beats the voxel proxy here: a voxel grid coarse enough to fit the box
budget cannot resolve the gap between a chair seat and the floor, so it bridges
it (measured: chair022 43% cavity over-fill at every budget setting).  A
semantic split puts a box on each leg and leaves the gap empty by construction.

Object-local frame: +Y is up for every CORE4D desk/chair (verified from the
per-axis surface-mass profile — the tabletop/seat shows up as a dominant Y
spike, the legs as a sparse tail).  This matches E177, which layered buckets
along local +Y as well.

Everything emitted is an axis-aligned box, because `object_collision_sdf_mode=
'union'` is fail-closed on non-box geoms (`spider/config.py:69-79`).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e206_common as C  # noqa: E402
import lowgeom_proxy_v2 as L  # noqa: E402
import manual_boxes as MB  # noqa: E402
from lowgeom_proxy import ProxyBox  # noqa: E402

UP = 1  # object-local +Y
HORIZ = (0, 2)
SAMPLES = 80_000
SEED = 0

# Per-object part plan, transcribed from the 2026-09-03 3D review.
# "keep_edited_below" = reuse the reviewer's own edited voxel boxes for the
# region below the plate, instead of re-deriving it semantically.
SEMANTIC_SPEC: dict[str, dict[str, Any]] = {
    "desk021": {"parts": ["top", "legs4"], "expect": 5},
    "desk020": {"parts": ["top", "keep_edited_below"], "expect": None},
    "chair005": {"parts": ["back", "rest"], "expect": 2},
    "chair020": {"parts": ["legs4", "seat", "back"], "expect": 6},
    "chair006": {"parts": ["legs4", "seat", "back", "arms"], "expect": 8},
    "chair022": {"parts": ["legs4", "seat", "back", "arms"], "expect": 8},
}


def sample_surface(mesh_path: Path, count: int = SAMPLES) -> np.ndarray:
    mesh = L.load_mesh(Path(mesh_path))
    state = np.random.get_state()
    try:
        np.random.seed(SEED)
        pts, _ = trimesh.sample.sample_surface(mesh, count)
    finally:
        np.random.set_state(state)
    return np.asarray(pts, dtype=np.float64)


def aabb_box(points: np.ndarray, *, min_half: float = 0.004) -> ProxyBox:
    lo, hi = points.min(axis=0), points.max(axis=0)
    center = (lo + hi) / 2.0
    half = np.maximum((hi - lo) / 2.0, min_half)
    return ProxyBox(center=center, half_size=half)


def footprint_profile(pts: np.ndarray, nbins: int = 40, grid: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """Per-Y-slice horizontal occupancy — a plate shows up as a spike."""
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    edges = np.linspace(lo[UP], hi[UP], nbins + 1)
    area = np.zeros(nbins)
    for i in range(nbins):
        sel = pts[(pts[:, UP] >= edges[i]) & (pts[:, UP] < edges[i + 1])]
        if len(sel) < 20:
            continue
        gx = np.clip(((sel[:, 0] - lo[0]) / max(hi[0] - lo[0], 1e-9) * grid).astype(int), 0, grid - 1)
        gz = np.clip(((sel[:, 2] - lo[2]) / max(hi[2] - lo[2], 1e-9) * grid).astype(int), 0, grid - 1)
        occupied = np.zeros(grid * grid, dtype=bool)
        occupied[gx * grid + gz] = True
        area[i] = occupied.sum() / (grid * grid)
    return area, edges


def plate_band(pts: np.ndarray) -> tuple[float, float]:
    """(y_low, y_high) of the tabletop / seat slab."""
    area, edges = footprint_profile(pts)
    peak = int(np.argmax(area))
    thresh = max(0.5 * area[peak], 0.15)
    lo_i = peak
    while lo_i > 0 and area[lo_i - 1] >= thresh:
        lo_i -= 1
    hi_i = peak
    while hi_i < len(area) - 1 and area[hi_i + 1] >= thresh:
        hi_i += 1
    return float(edges[lo_i]), float(edges[hi_i + 1])


def leg_zone_top(below: np.ndarray, pts: np.ndarray, nbins: int = 24) -> float:
    """Height where the legs stop being free-standing columns.

    Directly under the seat/top there is usually an apron (chair020) or side
    rails (chair006) tying the legs together.  Measured: at leg height the XZ
    occupancy is ~2-3% (four thin columns), and it jumps to 10-18% where the
    apron starts.  Clustering across that boundary is what made every "leg" box
    a fat quadrant of the whole under-seat volume (68-82% empty).

    Returns the height below which only leg columns exist.
    """
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    y0, y1 = below[:, UP].min(), below[:, UP].max()
    if y1 - y0 < 1e-6:
        return float(y1)
    edges = np.linspace(y0, y1, nbins + 1)
    grid = 40
    occ = np.zeros(nbins)
    for i in range(nbins):
        sel = below[(below[:, UP] >= edges[i]) & (below[:, UP] < edges[i + 1])]
        if len(sel) < 10:
            continue
        g = np.zeros((grid, grid), bool)
        gx = np.clip(((sel[:, 0] - lo[0]) / max(hi[0] - lo[0], 1e-9) * grid).astype(int), 0, grid - 1)
        gz = np.clip(((sel[:, 2] - lo[2]) / max(hi[2] - lo[2], 1e-9) * grid).astype(int), 0, grid - 1)
        g[gx, gz] = True
        occ[i] = g.mean()
    lower = occ[: max(1, nbins // 2)]
    baseline = float(np.median(lower[lower > 0])) if np.any(lower > 0) else 0.0
    if baseline <= 0:
        return float(y1)
    limit = max(2.0 * baseline, baseline + 0.02)
    top = nbins
    for i in range(nbins - 1, -1, -1):
        if occ[i] > limit:
            top = i
        else:
            break
    return float(edges[min(top, nbins)])


def split_legs(below: np.ndarray, k: int = 4) -> list[np.ndarray]:
    """Cluster the sub-plate points into k legs by horizontal position.

    KMeans on the XZ plane: legs are separated horizontally even when a stretcher
    makes them one connected component in 3D, so a purely topological split would
    merge them.
    """
    from scipy.cluster.vq import kmeans2

    xz = below[:, HORIZ]
    lo, hi = xz.min(axis=0), xz.max(axis=0)
    # Seed at the four horizontal corners so cluster<->leg assignment is stable.
    seeds = np.array([[lo[0], lo[1]], [lo[0], hi[1]], [hi[0], lo[1]], [hi[0], hi[1]]])[:k]
    centroid, label = kmeans2(xz, seeds, minit="matrix", iter=50, seed=SEED)
    return [below[label == i] for i in range(k) if np.any(label == i)]


def back_axis_and_side(above: np.ndarray, pts: np.ndarray) -> tuple[int, int]:
    """Which horizontal axis the backrest is offset along, and toward which end."""
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    best_ax, best_skew, best_side = HORIZ[0], 0.0, 1
    for ax in HORIZ:
        norm = (above[:, ax] - lo[ax]) / max(hi[ax] - lo[ax], 1e-9)
        skew = abs(float(norm.mean()) - 0.5)
        if skew > best_skew:
            best_ax, best_skew, best_side = ax, skew, 1 if norm.mean() > 0.5 else -1
    return best_ax, best_side


def _edited_voxel_boxes_below(object_key: str, y_split: float) -> list[ProxyBox]:
    """The reviewer's surviving voxel boxes that sit below the plate."""
    # target_cells must come from the EDIT RECORD, not the contract: once this
    # object switched to a semantic proxy the contract reports target_cells=n/a,
    # and feeding that back into the voxeliser yields a negative pitch (hang).
    # The edit record stores the exact voxel build its indices refer to.
    edits = L.load_box_edits().get(object_key, {})
    tc = int(edits.get("target_cells", 0))
    if tc <= 0:
        raise SystemExit(
            f"{object_key}: keep_edited_below needs a box_edits.json record with "
            "target_cells (re-do the 3D review for this object)"
        )
    boxes, _pitch = L._boxes_at(C.object_mesh_path(object_key), tc, C.N_MAX_TARGET)
    removed = {int(i) for i in edits.get("removed", [])}
    if edits and int(edits.get("n_boxes_original", len(boxes))) != len(boxes):
        raise SystemExit(
            f"{object_key}: edit record is stale ({edits.get('n_boxes_original')} vs "
            f"{len(boxes)} boxes) — re-do the 3D review before rebuilding"
        )
    return [
        b
        for i, b in enumerate(boxes)
        if i not in removed and float(b.center[UP]) < y_split
    ]


def build_semantic_boxes(object_key: str) -> tuple[list[ProxyBox], dict[str, Any]]:
    spec = SEMANTIC_SPEC.get(object_key)
    if spec is None:
        raise KeyError(f"no semantic spec for {object_key}")
    mesh_path = C.object_mesh_path(object_key)
    pts = sample_surface(mesh_path)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    y_lo, y_hi = plate_band(pts)

    boxes: list[ProxyBox] = []
    labels: list[str] = []
    below = pts[pts[:, UP] < y_lo]
    plate = pts[(pts[:, UP] >= y_lo) & (pts[:, UP] <= y_hi)]
    above = pts[pts[:, UP] > y_hi]

    parts = spec["parts"]

    if "rest" in parts:
        # chair005: one box for the backrest, one for everything else.
        ax, side = back_axis_and_side(above, pts) if len(above) > 50 else (HORIZ[0], -1)
        cut = np.quantile(above[:, ax], 0.5) if len(above) > 50 else 0.0
        back = above[above[:, ax] <= cut] if side < 0 else above[above[:, ax] >= cut]
        rest = pts[~np.isin(np.arange(len(pts)), np.array([], dtype=int))]
        mask = np.ones(len(pts), bool)
        if len(back):
            key = set(map(tuple, np.round(back, 9).tolist()))
            mask = np.array([tuple(np.round(p, 9)) not in key for p in pts])
        rest = pts[mask]
        if len(back):
            boxes.append(aabb_box(back)); labels.append("back")
        if len(rest):
            boxes.append(aabb_box(rest)); labels.append("rest")
        meta_extra = {"back_axis": "xyz"[ax], "back_side": side}
    else:
        # Split the sub-plate region at the apron: pure leg columns below,
        # apron/rails above.  The apron is folded into the seat/top box (it sits
        # directly against it), which keeps the reviewer's part count while
        # leaving the leg-swing gap genuinely open.
        y_legs_top = leg_zone_top(below, pts) if len(below) > 100 else y_lo
        legs_pts = below[below[:, UP] <= y_legs_top]
        apron_pts = below[below[:, UP] > y_legs_top]
        meta_apron = {
            "leg_zone_top": float(y_legs_top),
            "apron_points": int(len(apron_pts)),
            "apron_height_m": float(y_lo - y_legs_top),
        }

        if "legs4" in parts and len(legs_pts) > 100:
            for i, leg in enumerate(split_legs(legs_pts, 4)):
                if len(leg) < 20:
                    continue
                box = aabb_box(leg)
                # Extend each leg up to the plate so there is no floating gap;
                # the column is thin in XZ so this adds no cavity volume.
                top_y = y_lo
                lo_y = float(box.center[UP] - box.half_size[UP])
                box.center[UP] = (lo_y + top_y) / 2.0
                box.half_size[UP] = max((top_y - lo_y) / 2.0, 0.004)
                boxes.append(box); labels.append(f"leg{i}")

        if "keep_edited_below" in parts:
            # desk020: its sub-top structure is one connected panel base, not four
            # legs, so there is nothing semantic to split.  The reviewer already
            # pruned the voxel proxy there by hand; reuse exactly those boxes.
            kept = _edited_voxel_boxes_below(object_key, y_lo)
            for i, box in enumerate(kept):
                boxes.append(box); labels.append(f"base{i}")
            meta_apron["kept_edited_boxes"] = len(kept)

        plate_and_apron = np.vstack([plate, apron_pts]) if len(apron_pts) else plate
        if "keep_edited_below" in parts:
            plate_and_apron = plate  # base handled above; do not absorb it
        if "top" in parts and len(plate_and_apron):
            boxes.append(aabb_box(plate_and_apron)); labels.append("top")
        if "seat" in parts and len(plate_and_apron):
            boxes.append(aabb_box(plate_and_apron)); labels.append("seat")

        meta_extra = dict(meta_apron)
        if ("back" in parts or "arms" in parts) and len(above) > 50:
            ax, side = back_axis_and_side(above, pts)
            other = HORIZ[0] if ax == HORIZ[1] else HORIZ[1]
            span = hi[ax] - lo[ax]
            # Backrest slab: the third of the above-plate volume nearest the
            # offset end; armrests are what remains, split left/right.
            if side > 0:
                back_mask = above[:, ax] >= hi[ax] - 0.45 * span
            else:
                back_mask = above[:, ax] <= lo[ax] + 0.45 * span
            back = above[back_mask]
            remainder = above[~back_mask]
            if "back" in parts and len(back):
                boxes.append(aabb_box(back)); labels.append("back")
            if "arms" in parts and len(remainder) > 40:
                mid = (lo[other] + hi[other]) / 2.0
                for nm, sel in (
                    ("arm_lo", remainder[remainder[:, other] < mid]),
                    ("arm_hi", remainder[remainder[:, other] >= mid]),
                ):
                    if len(sel) > 20:
                        boxes.append(aabb_box(sel)); labels.append(nm)
            elif "arms" not in parts and "back" in parts and len(remainder) > 40:
                # No armrests declared: fold the remainder into the back box.
                boxes[-1] = aabb_box(above)
            meta_extra.update({"back_axis": "xyz"[ax], "back_side": side})

    meta: dict[str, Any] = {
        "object_key": object_key,
        "object_category": C.object_category(object_key),
        "policy": f"{C.object_category(object_key)}_semantic_parts",
        "parts": labels,
        "object_geom_count": len(boxes),
        "plate_y_low": y_lo,
        "plate_y_high": y_hi,
        "n_below": int(len(below)),
        "n_plate": int(len(plate)),
        "n_above": int(len(above)),
        "keep_edited_below": "keep_edited_below" in parts,
        **meta_extra,
    }
    expect = spec.get("expect")
    if expect is not None and len(boxes) != expect:
        meta["expect_mismatch"] = f"expected {expect}, got {len(boxes)}"
    return boxes, meta


_MEASURE_MEMO: dict[tuple[str, bytes], dict[str, Any]] = {}


def measure(object_key: str, boxes: list[ProxyBox]) -> dict[str, Any]:
    """Fidelity + cavity metrics for one box set.

    Memoised on the exact box geometry.  Every sampler underneath is seeded
    (`np.random.seed(0)` / `default_rng(0)`), so this is bit-for-bit equivalent
    to recomputing — it just stops the audit from measuring the same geometry
    twice.  A manual proxy is independent of `n_max`, so auditing the N=16 and
    N=9 budgets scores the identical box set both times.
    """
    key = (
        object_key,
        np.stack(
            [np.concatenate([b.center, b.half_size]) for b in boxes]
        ).astype(np.float64).tobytes()
        if boxes
        else b"",
    )
    hit = _MEASURE_MEMO.get(key)
    if hit is not None:
        return dict(hit)
    mesh_path = C.object_mesh_path(object_key)
    pitch = np.array([0.02, 0.02, 0.02])
    out = dict(L.fidelity_metrics(mesh_path, boxes))
    out.update(L.cavity_metrics(mesh_path, boxes, pitch))
    _MEASURE_MEMO[key] = dict(out)
    return out


if __name__ == "__main__":
    import json

    for key in SEMANTIC_SPEC:
        try:
            boxes, meta = build_semantic_boxes(key)
        except Exception as exc:  # noqa: BLE001
            print(f"{key}: ERROR {type(exc).__name__}: {exc}")
            continue
        m = measure(key, boxes)
        flag = " !! " + meta["expect_mismatch"] if "expect_mismatch" in meta else ""
        print(
            f"{key:9s} boxes={meta['object_geom_count']:2d} parts={','.join(meta['parts']):45s} "
            f"m2p_p90={m['mesh_to_proxy_p90_m']:.3f} p2m_p90={m['proxy_to_mesh_p90_m']:.3f} "
            f"of5={m['interior_overfill_frac_5cm']:.2f}{flag}"
        )


# --------------------------------------------------------------------------
# Dispatcher — the single entry point every downstream stage must use
# --------------------------------------------------------------------------
# Precedence, highest first:
#   1. manual   — boxes hand-placed in `edit_proxy_3d.py`, stored as absolute
#                 geometry in `manual_boxes.json`.  A human who has looked at the
#                 overlay outranks both automatic heuristics, so this wins
#                 outright and the auto paths become mere seeds.
#   2. semantic — hand-specified part decomposition (SEMANTIC_SPEC).
#   3. voxel    — auto merge, plus whatever boxes the reviewer deleted.
def build_effective_proxy(
    object_key: str,
    n_max: int = C.N_MAX_TARGET,
    target_cells: int | None = None,
    *,
    measure_metrics: bool = True,
) -> tuple[list[ProxyBox], dict[str, Any]]:
    """The proxy that actually ships for this object.

    `measure_metrics=False` returns the boxes without the fidelity/cavity pass.
    The 3D editor needs the box set as a starting point, not the numbers, and
    the exact metrics cost seconds per object — but it must go through THIS
    function so its starting point is genuinely what ships, precedence and
    recorded deletions included.
    """
    hit = MB.manual_boxes_for(object_key)
    if hit is not None:
        boxes, labels = hit
        MB.validate(boxes, object_key, n_max=n_max)
        record = MB.load_manual().get(object_key, {})
        meta: dict[str, Any] = {
            "object_key": object_key,
            "object_category": C.object_category(object_key),
            "mesh_path": str(C.object_mesh_path(object_key)),
            "proxy_kind": "manual",
            "n_max": n_max,
            "target_cells": -1,
            "object_geom_count": len(boxes),
            "parts": labels,
            "policy": f"{C.object_category(object_key)}_manual_boxes",
            "collision_policy": f"{C.object_category(object_key)}_manual_boxes",
            "editor": record.get("editor", ""),
            "edited_at": record.get("edited_at", ""),
            "manual_seed": record.get("seed", {}),
            "edited": True,
            "removed_indices": [],
        }
        if measure_metrics:
            meta.update(measure(object_key, boxes))
        return boxes, meta
    if object_key in SEMANTIC_SPEC:
        boxes, meta = build_semantic_boxes(object_key)
        all_parts = list(meta.get("parts", []))
        # Only replay a deletion record that was authored against the SEMANTIC
        # build.  A voxel-era record indexes a completely different box list:
        # for chair006 it is simply obsolete, and for desk020 it has already
        # been consumed — correctly — by `_edited_voxel_boxes_below`, so
        # replaying it here would apply the same edit twice.  Both objects were
        # failing `apply_box_edits`' staleness guard because of this.
        edit_record = L.load_box_edits().get(object_key, {})
        edit_info: dict[str, Any] = {"edited": False, "removed_indices": []}
        if edit_record.get("proxy_kind") == "semantic":
            boxes, edit_info = L.apply_box_edits(
                boxes, object_key, n_max=n_max, target_cells=-1, kind="semantic"
            )
        if edit_info.get("edited"):
            drop = set(edit_info["removed_indices"])
            meta["parts"] = [p for i, p in enumerate(all_parts) if i not in drop]
        meta.update(edit_info)
        meta["proxy_kind"] = "semantic"
        meta["n_max"] = n_max
        meta.setdefault("target_cells", -1)
        if measure_metrics:
            meta.update(measure(object_key, boxes))
        meta["collision_policy"] = meta["policy"]
        return boxes, meta
    boxes, meta = L.build_lowgeom_boxes(
        C.object_mesh_path(object_key),
        object_key,
        n_max=n_max,
        target_cells=target_cells,
        measure_metrics=measure_metrics,
    )
    meta["proxy_kind"] = "voxel_edited" if meta.get("edited") else "voxel"
    meta["parts"] = []
    return boxes, meta
