#!/usr/bin/env python3
"""E206 lowgeom proxy v2 — strict generalisation of E176 to desk + chair.

Three deltas vs `E176/lowgeom_proxy.py`, each with a reason:

1. **`sweep_target_cells` instead of a frozen per-object table.** E176 hard-coded
   `TARGET_CELLS_BY_OBJECT` for 5 buckets + desk007 and raises `KeyError` on
   anything else, so it cannot see a chair at all. The sweep picks the *largest
   feasible* target_cells (finest voxel) whose merge fits the budget — a
   deterministic rule, not a hand-tuned number.

2. **`cavity_metrics` replaces `center_inside_count`.** E176 raises
   "coarse proxy fills mesh AABB center" whenever any box contains the mesh AABB
   centre (`lowgeom_proxy.py:105-113`). That is a *hollowness heuristic* written
   for buckets and desks; a chair seat legitimately occupies the AABB centre, so
   all 5 chairs trip it. We do NOT drop the check — dropping it would lose the
   cavity guarantee E176 C4/C5 established. We replace it with a direct
   measurement: sample inside the proxy union, measure distance to the real
   mesh, report the over-filled fraction.

3. **`N_MAX` is a parameter, not a module constant.** E176 froze 9; the user
   authorised 16 and P3 measures whether it is affordable.

Everything else (voxelisation, greedy merge, inward shrink, fidelity metrics,
XML emission) is IMPORTED from E176 / the dcv3 template builder, so an E206
proxy at E176's settings reproduces E176 byte-for-byte.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e206_common as C  # noqa: E402  (sets up sys.path for the imports below)

from lowgeom_proxy import (  # noqa: E402  E176 authority
    ProxyBox,
    fidelity_metrics,
    load_mesh,
    point_to_proxy_surface_distance,
    proxy_xml,
)
from build_or_audit_templates import (  # noqa: E402  dcv3 authority
    geom_box_xml,
    merge_occupied_voxels,
)

__all__ = [
    "BOX_EDITS_PATH",
    "ProxyBox",
    "apply_box_edits",
    "build_lowgeom_boxes",
    "cavity_metrics",
    "fidelity_metrics",
    "load_box_edits",
    "load_mesh",
    "point_to_proxy_surface_distance",
    "proxy_xml",
    "save_box_edits",
    "sweep_target_cells",
    "union_geoms_are_boxes",
]

# E176's shrink recipe, kept verbatim so E206 is a generalisation not a new recipe.
SHRINK_PITCH_FRAC = 0.12
SHRINK_HALF_FRAC = 0.25
MIN_HALF_PITCH_FRAC = 0.22

SWEEP_LO = 2
SWEEP_HI = 16

# Rendering colour for E206 proxies (distinct from the dcv3 draft's blue-grey and
# from E178's orange, so an overlay sheet is unambiguous about which is on screen).
E206_RGBA = "0.10 0.75 0.35 0.40"


# --------------------------------------------------------------------------
# Sweep
# --------------------------------------------------------------------------
def _merge_at(mesh: trimesh.Trimesh, target_cells: int, n_max: int):
    """One (target_cells -> boxes) attempt. Returns (raw_boxes, left_unmerged, pitch)."""
    pitch_scalar = float(mesh.extents.max()) / float(target_cells)
    voxels = mesh.voxelized(pitch_scalar)
    pitch = np.asarray(voxels.pitch, dtype=np.float64)
    raw_boxes, left_unmerged = merge_occupied_voxels(
        voxels.matrix.astype(bool),
        pitch,
        np.asarray(voxels.transform, dtype=np.float64),
        max_boxes=n_max,
    )
    return raw_boxes, int(left_unmerged), pitch, int(voxels.matrix.sum())


VOXEL_CACHE_PATH = C.S2_PROXY_DIR / "voxel_box_cache.json"
_VOXEL_CACHE: dict[str, Any] | None = None


def _voxel_cache() -> dict[str, Any]:
    """Disk memo for `_boxes_at`.

    `trimesh.voxelized` subdivides the mesh to the voxel size, which for some of
    these meshes (desk020 at pitch 0.055) takes minutes.  The result is a pure
    function of (mesh, target_cells, n_max), so memoising it keeps the
    edit -> re-measure -> review loop interactive.
    """
    global _VOXEL_CACHE
    if _VOXEL_CACHE is None:
        import json

        _VOXEL_CACHE = (
            json.loads(VOXEL_CACHE_PATH.read_text(encoding="utf-8"))
            if VOXEL_CACHE_PATH.exists()
            else {}
        )
    return _VOXEL_CACHE


def _boxes_at(mesh_path: Path, target_cells: int, n_max: int) -> tuple[list[ProxyBox], np.ndarray]:
    """Shrunk proxy boxes at one target_cells (shared by scoring and building)."""
    import json

    key = f"{Path(mesh_path).stem}|tc{target_cells}|n{n_max}"
    cache = _voxel_cache()
    hit = cache.get(key)
    if hit is not None:
        return (
            [
                ProxyBox(
                    center=np.asarray(b["center"], dtype=np.float64),
                    half_size=np.asarray(b["half_size"], dtype=np.float64),
                )
                for b in hit["boxes"]
            ],
            np.asarray(hit["pitch"], dtype=np.float64),
        )
    boxes, pitch = _boxes_at_uncached(mesh_path, target_cells, n_max)
    cache[key] = {
        "pitch": [float(v) for v in np.atleast_1d(pitch)],
        "boxes": [
            {"center": list(map(float, b.center)), "half_size": list(map(float, b.half_size))}
            for b in boxes
        ],
    }
    VOXEL_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    VOXEL_CACHE_PATH.write_text(json.dumps(cache), encoding="utf-8")
    return boxes, pitch


def _boxes_at_uncached(mesh_path: Path, target_cells: int, n_max: int) -> tuple[list[ProxyBox], np.ndarray]:
    mesh = load_mesh(Path(mesh_path))
    raw_boxes, left_unmerged, pitch, _ = _merge_at(mesh, target_cells, n_max)
    if left_unmerged:
        raise ValueError(f"target_cells={target_cells} leaves {left_unmerged} voxels unmerged")
    boxes = []
    for center, half_size in raw_boxes:
        shrink = np.minimum(pitch * SHRINK_PITCH_FRAC, half_size * SHRINK_HALF_FRAC)
        shrunk = np.maximum(half_size - shrink, pitch * MIN_HALF_PITCH_FRAC)
        boxes.append(
            ProxyBox(
                center=np.asarray(center, dtype=np.float64),
                half_size=np.asarray(shrunk, dtype=np.float64),
            )
        )
    return boxes, pitch


def sweep_target_cells(
    mesh_path: Path,
    n_max: int,
    *,
    lo: int = SWEEP_LO,
    hi: int = SWEEP_HI,
) -> list[dict[str, Any]]:
    """Try every target_cells in [lo, hi]; report feasibility per value.

    Feasible == the greedy merge covered every occupied voxel within `n_max`
    boxes (`left_unmerged == 0`).
    """
    mesh = load_mesh(Path(mesh_path))
    table: list[dict[str, Any]] = []
    for target_cells in range(lo, hi + 1):
        try:
            raw_boxes, left_unmerged, pitch, occupied = _merge_at(mesh, target_cells, n_max)
        except Exception as exc:  # noqa: BLE001 - a failed voxelisation is a datum, not a crash
            table.append(
                {
                    "target_cells": target_cells,
                    "feasible": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        table.append(
            {
                "target_cells": target_cells,
                "feasible": left_unmerged == 0,
                "n_boxes": len(raw_boxes),
                "left_unmerged": left_unmerged,
                "voxel_pitch_m": float(pitch[0]),
                "occupied_voxels": occupied,
                "error": "",
            }
        )
    return table


# Fidelity bars used to rank sweep candidates.  Kept in sync with
# audit_lowgeom_contract.BARS (that module is the reporting authority; these are
# the same numbers used as a *selection* filter).
SELECT_BARS = {
    "mesh_to_proxy_p90_m": 0.08,
    "mesh_to_proxy_max_m": 0.16,
    "proxy_to_mesh_p90_m": 0.14,
    "interior_overfill_frac_pitch": 0.02,
}


def score_target_cells(
    mesh_path: Path,
    object_key: str,
    sweep: list[dict[str, Any]],
    n_max: int,
) -> list[dict[str, Any]]:
    """Measure fidelity for every feasible target_cells in the sweep.

    Necessary because voxel resolution is NOT monotone in proxy quality once a
    hard box cap is in play: the greedy merge has to cover more voxels with the
    same budget, so at a finer pitch it can emit fewer, larger blocks that bridge
    across cavities.  Measured on chair022 — tc=6 (finest feasible at n_max=16)
    scores mesh->proxy p90 0.100 / over-fill 0.56, while tc=4 scores 0.064 / 0.43.
    Picking "finest feasible" would therefore pick the worse proxy.
    """
    scored: list[dict[str, Any]] = []
    for row in sweep:
        if not row.get("feasible"):
            continue
        tc = int(row["target_cells"])
        try:
            boxes, pitch = _boxes_at(mesh_path, tc, n_max)
        except Exception as exc:  # noqa: BLE001
            scored.append({"target_cells": tc, "scorable": False, "error": str(exc)})
            continue
        entry: dict[str, Any] = {
            "target_cells": tc,
            "scorable": True,
            "n_boxes": len(boxes),
            "error": "",
        }
        entry.update(fidelity_metrics(mesh_path, boxes))
        entry.update(cavity_metrics(mesh_path, boxes, pitch))
        entry["bars_pass"] = all(
            entry[key] <= bar for key, bar in SELECT_BARS.items()
        )
        scored.append(entry)
    return scored


def select_target_cells(scored: list[dict[str, Any]]) -> int:
    """Pick by measurement, not by a resolution heuristic.

    Rule (declared, deterministic): among candidates that clear every fidelity
    bar, take the lowest cavity over-fill; break ties on lower mesh->proxy p90,
    then on fewer boxes.  If nothing clears the bars, fall back to the same
    ordering over all scorable candidates so the caller still gets the best
    available proxy and the contract audit reports the failure honestly.
    """
    usable = [s for s in scored if s.get("scorable")]
    if not usable:
        raise ValueError("no scorable target_cells in sweep")
    passing = [s for s in usable if s.get("bars_pass")]
    pool = passing or usable
    best = min(
        pool,
        key=lambda s: (
            round(s["interior_overfill_frac_5cm"], 4),
            round(s["mesh_to_proxy_p90_m"], 5),
            s["n_boxes"],
        ),
    )
    return int(best["target_cells"])


# --------------------------------------------------------------------------
# Cavity measurement (replaces E176 center_inside_count)
# --------------------------------------------------------------------------
def _sample_inside_boxes(boxes: list[ProxyBox], count: int, seed: int = 0) -> np.ndarray:
    """Volume-weighted uniform sample inside the union of the proxy boxes."""
    rng = np.random.default_rng(seed)
    volumes = np.array([float(np.prod(2.0 * b.half_size)) for b in boxes], dtype=np.float64)
    total = volumes.sum()
    if total <= 0:
        return np.zeros((0, 3), dtype=np.float64)
    counts = np.maximum(1, np.round(count * volumes / total).astype(int))
    chunks = []
    for box, n in zip(boxes, counts):
        chunks.append(
            box.center + rng.uniform(-box.half_size, box.half_size, size=(int(n), 3))
        )
    return np.concatenate(chunks, axis=0)


def cavity_metrics(
    mesh_path: Path,
    boxes: list[ProxyBox],
    pitch: np.ndarray,
    *,
    sample_count: int = 6000,
    seed: int = 0,
) -> dict[str, float]:
    """How much of the proxy volume is nowhere near the real mesh.

    This is the honest replacement for E176's AABB-centre heuristic: it answers
    "did we fill a cavity" by measurement, and therefore works for a chair (whose
    seat legitimately occupies the AABB centre) as well as for a hollow bucket.
    """
    mesh = load_mesh(Path(mesh_path))
    points = _sample_inside_boxes(boxes, sample_count, seed=seed)
    if points.shape[0] == 0:
        return {
            "interior_overfill_frac_5cm": 0.0,
            "interior_overfill_frac_pitch": 0.0,
            "interior_sample_count": 0,
            "proxy_volume_m3": 0.0,
            "proxy_volume_over_mesh_aabb": 0.0,
            "interior_dist_p90_m": 0.0,
        }
    _, distance, _ = trimesh.proximity.closest_point_naive(mesh, points)
    distance = np.asarray(distance, dtype=np.float64)
    pitch_max = float(np.max(pitch))
    proxy_volume = float(sum(np.prod(2.0 * b.half_size) for b in boxes))
    aabb_volume = float(np.prod(mesh.extents))
    return {
        "interior_overfill_frac_5cm": float(np.mean(distance > 0.05)),
        "interior_overfill_frac_pitch": float(np.mean(distance > pitch_max)),
        "interior_sample_count": int(points.shape[0]),
        "proxy_volume_m3": proxy_volume,
        "proxy_volume_over_mesh_aabb": proxy_volume / aabb_volume if aabb_volume > 0 else 0.0,
        "interior_dist_p90_m": float(np.quantile(distance, 0.90)),
    }


# --------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------
def build_lowgeom_boxes(
    mesh_path: Path,
    object_key: str,
    *,
    n_max: int,
    target_cells: int | None = None,
    sweep_lo: int = SWEEP_LO,
    sweep_hi: int = SWEEP_HI,
    apply_edits: bool = True,
) -> tuple[list[ProxyBox], dict[str, Any]]:
    """Deterministic <= n_max axis-aligned box proxy for one object mesh.

    If `target_cells` is None the sweep picks it by measured fidelity. Passing an
    explicit value reproduces a frozen configuration exactly.

    When `apply_edits` is set (default) any reviewer box deletions recorded in
    `box_edits.json` are applied AND all fidelity/cavity metrics are recomputed
    on the edited set — the reported numbers always describe the proxy that will
    actually be installed, never the pre-edit one.
    """
    mesh_path = Path(mesh_path)
    mesh = load_mesh(mesh_path)

    sweep: list[dict[str, Any]] = []
    scored: list[dict[str, Any]] = []
    if target_cells is None:
        sweep = sweep_target_cells(mesh_path, n_max, lo=sweep_lo, hi=sweep_hi)
        scored = score_target_cells(mesh_path, object_key, sweep, n_max)
        try:
            target_cells = select_target_cells(scored)
        except ValueError as exc:
            raise ValueError(
                f"{object_key}: no feasible target_cells in [{sweep_lo},{sweep_hi}] "
                f"at n_max={n_max}"
            ) from exc

    raw_boxes, left_unmerged, pitch, occupied = _merge_at(mesh, target_cells, n_max)
    if left_unmerged:
        raise ValueError(
            f"{object_key} exceeds {n_max} boxes at target_cells={target_cells}: "
            f"left_unmerged={left_unmerged}"
        )

    boxes: list[ProxyBox] = []
    for center, half_size in raw_boxes:
        shrink = np.minimum(pitch * SHRINK_PITCH_FRAC, half_size * SHRINK_HALF_FRAC)
        shrunk = np.maximum(half_size - shrink, pitch * MIN_HALF_PITCH_FRAC)
        boxes.append(
            ProxyBox(
                center=np.asarray(center, dtype=np.float64),
                half_size=np.asarray(shrunk, dtype=np.float64),
            )
        )
    if not 1 <= len(boxes) <= n_max:
        raise AssertionError(f"{object_key}: invalid box count {len(boxes)} (n_max={n_max})")

    edit_info: dict[str, Any] = {"edited": False, "removed_indices": []}
    if apply_edits:
        boxes, edit_info = apply_box_edits(
            boxes, object_key, n_max=n_max, target_cells=int(target_cells)
        )

    meta: dict[str, Any] = {
        "object_key": object_key,
        "object_category": C.object_category(object_key),
        "mesh_path": str(mesh_path),
        "n_max": n_max,
        "target_cells": int(target_cells),
        "object_geom_count": len(boxes),
        "voxel_pitch_m": float(pitch[0]),
        "occupied_voxels": occupied,
        "mesh_extent_x_m": float(mesh.extents[0]),
        "mesh_extent_y_m": float(mesh.extents[1]),
        "mesh_extent_z_m": float(mesh.extents[2]),
        "collision_policy": f"{C.object_category(object_key)}_lowgeom{n_max}_proxy",
        "sweep": sweep,
        "scored": scored,
        "selection_rule": "min_overfill_among_bar_passing" if scored else "explicit",
    }
    meta.update(edit_info)
    # Recomputed on the POST-edit box set, so the contract always describes what
    # actually gets installed.
    meta.update(fidelity_metrics(mesh_path, boxes))
    meta.update(cavity_metrics(mesh_path, boxes, pitch))
    return boxes, meta


def proxy_geom_xml(boxes: list[ProxyBox], *, rgba: str = E206_RGBA) -> tuple[str, list[str]]:
    """`object_collision` + `object_collision_coarse_NNN` box geoms.

    Reuses E176's `proxy_xml` when the count is within its 1..9 contract, and
    falls back to the same naming/emission for the 10..16 range E206 unlocks.
    """
    if 1 <= len(boxes) <= 9:
        return proxy_xml(boxes, rgba=rgba)
    geoms: list[str] = []
    names: list[str] = []
    for index, box in enumerate(boxes):
        name = "object_collision" if index == 0 else f"object_collision_coarse_{index:03d}"
        names.append(name)
        geoms.append(geom_box_xml(name, box.center, box.half_size, rgba=rgba))
    return "\n".join(geoms), names


# --------------------------------------------------------------------------
# Manual box edits (human deletions from the 3D reviewer)
# --------------------------------------------------------------------------
# The auto proxy is a starting point, not an oracle.  A reviewer looking at the
# overlay can see things the metrics cannot — e.g. a single block bridging the
# under-seat gap of chair022, where no budget setting helps.  Deleting that one
# box is the cheapest correct fix, and is exactly the "hand-authored shell"
# rung of plan236's fallback ladder, done surgically instead of wholesale.
#
# Edits are stored as ORIGINAL build-order indices and are only valid for the
# exact (n_max, target_cells, n_boxes) they were made against; `load_box_edits`
# refuses stale records rather than silently deleting the wrong box.
BOX_EDITS_PATH = C.S2_PROXY_DIR / "box_edits.json"


def load_box_edits(path: Path | None = None) -> dict[str, dict[str, Any]]:
    path = Path(path or BOX_EDITS_PATH)
    if not path.exists():
        return {}
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def save_box_edits(edits: dict[str, dict[str, Any]], path: Path | None = None) -> Path:
    import json

    path = Path(path or BOX_EDITS_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(edits, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def apply_box_edits(
    boxes: list[ProxyBox],
    object_key: str,
    *,
    n_max: int,
    target_cells: int,
    edits: dict[str, dict[str, Any]] | None = None,
    kind: str = "voxel",
) -> tuple[list[ProxyBox], dict[str, Any]]:
    """Drop the boxes a reviewer deleted. Refuses to apply a stale edit."""
    edits = load_box_edits() if edits is None else edits
    record = edits.get(object_key)
    info: dict[str, Any] = {
        "edited": False,
        "removed_indices": [],
        "n_boxes_before_edit": len(boxes),
    }
    if not record:
        return boxes, info

    removed = sorted({int(i) for i in record.get("removed", [])})
    if not removed:
        return boxes, info

    stale: list[str] = []
    if int(record.get("n_max", n_max)) != n_max:
        stale.append(f"n_max {record.get('n_max')} != {n_max}")
    if str(record.get("proxy_kind", kind)) != kind:
        stale.append(f"proxy_kind {record.get('proxy_kind')} != {kind}")
    if kind == "voxel" and int(record.get("target_cells", target_cells)) != target_cells:
        stale.append(f"target_cells {record.get('target_cells')} != {target_cells}")
    if int(record.get("n_boxes_original", len(boxes))) != len(boxes):
        stale.append(f"n_boxes_original {record.get('n_boxes_original')} != {len(boxes)}")
    if stale:
        raise ValueError(
            f"{object_key}: box edit is stale ({'; '.join(stale)}). The stored indices "
            "refer to a different build and would delete the wrong boxes. Re-do the "
            "3D review for this object."
        )
    if any(i < 0 or i >= len(boxes) for i in removed):
        raise ValueError(f"{object_key}: removed index out of range: {removed}")
    kept = [b for i, b in enumerate(boxes) if i not in set(removed)]
    if not kept:
        raise ValueError(f"{object_key}: edit removes every box")

    info.update(
        {
            "edited": True,
            "removed_indices": removed,
            "n_boxes_after_edit": len(kept),
            "editor": record.get("editor", ""),
            "edited_at": record.get("edited_at", ""),
            "edit_notes": record.get("notes", ""),
        }
    )
    return kept, info


# --------------------------------------------------------------------------
# Box-only guard (spider/config.py:69-79 is fail-closed on non-box geoms)
# --------------------------------------------------------------------------
def union_geoms_are_boxes(scene_xml: Path) -> dict[str, Any]:
    """Compile the scene and assert every `object_collision*` geom is mjGEOM_BOX.

    `object_collision_sdf_mode='union'` raises at load time otherwise, and every
    PRG reward/gate SDF path in `spider/simulators/mjwp.py` asserts the same.
    Checking here means we find it before spending GPU time.
    """
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    names: list[str] = []
    non_box: list[str] = []
    for gid in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
        if not name.startswith("object_collision"):
            continue
        names.append(name)
        if int(model.geom_type[gid]) != int(mujoco.mjtGeom.mjGEOM_BOX):
            non_box.append(f"{name}:type={int(model.geom_type[gid])}")
    return {
        "scene_xml": str(scene_xml),
        "object_collision_geom_count": len(names),
        "object_collision_geom_names": names,
        "non_box": non_box,
        "all_box": not non_box,
        "nq": int(model.nq),
        "nv": int(model.nv),
        "nu": int(model.nu),
    }
