#!/usr/bin/env python3
"""E088 evaluator: E083 diagnostics plus CEM gate and object clearance metrics."""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import mujoco
import numpy as np


REPO = Path(__file__).resolve().parents[4]
os.environ.setdefault("RESULTS", str(REPO / "workspace/core4d/results/E088"))
os.environ.setdefault(
    "VARIANTS_FILE", str(REPO / "workspace/core4d/scripts/E088/variants.tsv")
)
EVAL_DIR = REPO / "workspace/core4d/scripts/eval"
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

import eval_E083 as e083  # noqa: E402


RESULTS = Path(os.environ["RESULTS"])
VARIANTS_FILE = Path(os.environ["VARIANTS_FILE"])
CLEARANCE_THRESHOLD_M = 0.04


def _float(row: dict[str, object], key: str, default: float = 0.0) -> float:
    try:
        raw = row.get(key, default)
        if raw == "":
            return default
        return float(raw)
    except (TypeError, ValueError):
        return default


def _read_rows(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    keys = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _variant_meta() -> dict[str, dict[str, str]]:
    out = {}
    with VARIANTS_FILE.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(
            (line for line in f if line.strip() and not line.startswith("#")),
            delimiter="\t",
            fieldnames=[
                "variant",
                "source_task",
                "derived_task",
                "mask_source_dir",
                "mask_slug",
                "person_idx",
                "split",
                "role",
                "group",
                "mass_kg",
                "notes",
            ],
        )
        for row in reader:
            out[row["variant"]] = row
    return out


def _stats(values: np.ndarray, prefix: str) -> dict[str, object]:
    values = np.asarray(values, dtype=np.float64)
    return {
        f"{prefix}_mean_m": float(values.mean()),
        f"{prefix}_min_m": float(values.min()),
        f"{prefix}_max_m": float(values.max()),
        f"{prefix}_below_{int(CLEARANCE_THRESHOLD_M * 100)}cm_pct": float(
            (values < CLEARANCE_THRESHOLD_M).mean() * 100.0
        ),
    }


def _object_bottom_clearance(model: mujoco.MjModel, qpos: np.ndarray) -> np.ndarray:
    data = mujoco.MjData(model)
    object_gid = e083.e072.name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, e083.OBJECT_GEOM
    )
    half_z = float(model.geom_size[object_gid, 2])
    clearance = np.zeros(len(qpos), dtype=np.float64)
    for frame, q in enumerate(qpos):
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        clearance[frame] = float(data.geom_xpos[object_gid, 2] - half_z)
    return clearance


def _object_clearance_metrics(row: dict[str, object]) -> dict[str, object]:
    variant = str(row["variant"])
    case = str(row["case"])
    override = str(row["override"])
    model, _scene_used = e083.e081.e078.load_scene_model(case)

    data_npz = np.load(RESULTS / f"{variant}.npz", allow_pickle=True)
    qpos = e083.e072.flatten_time_major(data_npz["qpos"])
    qpos_ref, _ctrl_ref = e083.e081.e078.load_ref(override, case)
    T = min(len(qpos), len(qpos_ref), int(row.get("T", len(qpos))))
    qpos = qpos[:T]
    qpos_ref = qpos_ref[:T]

    sim = _object_bottom_clearance(model, qpos)
    ref = _object_bottom_clearance(model, qpos_ref)
    start = int(float(row.get("case_window_start_frame", 0)))
    end = min(int(float(row.get("case_window_end_frame", T - 1))), T - 1)
    sl = slice(start, end + 1)
    out: dict[str, object] = {
        "object_clearance_threshold_m": CLEARANCE_THRESHOLD_M,
        "full_sim_object_bottom_clearance_ref_delta_mean_m": float(
            (sim - ref).mean()
        ),
        "case_window_sim_object_bottom_clearance_ref_delta_mean_m": float(
            (sim[sl] - ref[sl]).mean()
        ),
    }
    out.update(_stats(sim, "full_sim_object_bottom_clearance"))
    out.update(_stats(ref, "full_ref_object_bottom_clearance"))
    out.update(_stats(sim[sl], "case_window_sim_object_bottom_clearance"))
    out.update(_stats(ref[sl], "case_window_ref_object_bottom_clearance"))
    return out


def _finite_arr(data: np.lib.npyio.NpzFile, key: str) -> np.ndarray | None:
    if key not in data.files:
        return None
    arr = np.asarray(data[key], dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    return arr


def _gate_npz_metrics(variant: str) -> dict[str, object]:
    path = RESULTS / f"{variant}.npz"
    if not path.is_file():
        return {}
    data = np.load(path, allow_pickle=True)
    out: dict[str, object] = {}
    for key in [
        "cem_gate_valid_frac",
        "cem_gate_selected_valid_frac",
        "sample_gate_min_sdf_min",
        "sample_gate_min_sdf_mean",
        "sample_gate_violation_pct_mean",
        "sample_gate_violation_pct_max",
        "sample_gate_violation_depth_mean_mean",
        "sample_gate_violation_depth_mean_max",
    ]:
        arr = _finite_arr(data, key)
        if arr is None:
            continue
        out[f"{key}_mean"] = float(arr.mean())
        out[f"{key}_min"] = float(arr.min())
        out[f"{key}_max"] = float(arr.max())
    arr = _finite_arr(data, "cem_gate_fallback_used")
    if arr is not None:
        out["cem_gate_fallback_used_any"] = bool((arr > 0.5).any())
        out["cem_gate_fallback_used_pct"] = float((arr > 0.5).mean() * 100.0)
    return out


def main() -> None:
    e083.main()
    comparison = RESULTS / "comparison.csv"
    rows = _read_rows(comparison)
    meta = _variant_meta()
    accepted = []
    for row in rows:
        variant = str(row["variant"])
        m = meta.get(variant, {})
        row["E088_group"] = m.get("group", "")
        row["E088_mass_kg"] = m.get("mass_kg", "")
        row.update(_object_clearance_metrics(row))
        row.update(_gate_npz_metrics(variant))

        contact_ok = _float(row, "case_window_sim_contact_frames_pct") >= 40.0
        obj_ok = _float(row, "case_window_obj_err_mean_m") <= 0.85
        head_ok = (
            _float(row, "case_window_sim_head_collision_object_penetration_pct")
            < 5.0
        )
        upper_ok = _float(row, "case_window_sim_upperbody_object_penetration_pct") < 15.0
        floor_ok = (
            _float(row, "case_window_sim_lh_floor_contact_pct") <= 5.0
            and _float(row, "case_window_sim_rh_floor_contact_pct") <= 5.0
        )
        gate_stats_ok = _float(row, "cem_gate_valid_frac_mean", 1.0) > 0.0
        accept = contact_ok and obj_ok and head_ok and upper_ok and floor_ok and gate_stats_ok
        row["E088_gate_contact_ok"] = bool(contact_ok)
        row["E088_gate_obj_ok"] = bool(obj_ok)
        row["E088_gate_head_ok"] = bool(head_ok)
        row["E088_gate_upper_ok"] = bool(upper_ok)
        row["E088_gate_floor_ok"] = bool(floor_ok)
        row["E088_gate_stats_ok"] = bool(gate_stats_ok)
        row["E088_gate_accept"] = bool(accept)
        if accept:
            accepted.append(variant)
    _write_rows(comparison, rows)
    gate = {
        "num_results": len(rows),
        "accepted_variants": accepted,
        "num_gate_accept": len(accepted),
        "gate_rule": (
            "contact>=40, obj_mean<=0.85m, head<5%, upper<15%, "
            "hand-floor<=5%, gate_valid_frac>0"
        ),
    }
    previous_path = RESULTS / "aggregate_summary.json"
    previous = {}
    if previous_path.is_file():
        previous = json.loads(previous_path.read_text(encoding="utf-8"))
    previous_path.write_text(
        json.dumps({**previous, **gate}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (RESULTS / "e088_gate_summary.json").write_text(
        json.dumps(gate, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"Wrote {comparison}")
    print(json.dumps(gate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
