#!/usr/bin/env python3
"""Offline Holosoma-style z-only body gate for E167 artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import mujoco
import numpy as np


REPO = Path(__file__).resolve().parents[5]
BUILDER = REPO / "workspace/core4d/scripts/experiments/E167/build_zonly_manifest.py"
VARIANTS = REPO / "workspace/core4d/scripts/experiments/E167/variants.tsv"
EVAL_ROOT = REPO / "workspace/core4d/results/E167/holosoma_zonly/eval/zgate"
BODY_NAMES = [
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
]
Z_THRESHOLD_M = 0.25
SUGAR_3D_THRESHOLD_M = 0.30


def rel(path: str | Path) -> str:
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


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def qpos_path(row: dict[str, str]) -> Path:
    if row["arm"] == "baseline":
        return repo_path(row["baseline_outdir_npz"])
    if row["arm_kind"] == "postprocess":
        return repo_path(row["postprocess_output_npz"])
    return repo_path(row["outdir_npz"])


def baseline_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    out = []
    seen = set()
    for row in rows:
        case = row["short_case_id"]
        if case in seen:
            continue
        seen.add(case)
        base = dict(row)
        base["variant"] = row["base_e163_variant"]
        base["arm"] = "baseline"
        base["arm_kind"] = "baseline"
        base["method"] = "E163_narrowSurfaceBand"
        base["method_display"] = "E163_narrowSurfaceBand"
        base["split"] = "baseline"
        out.append(base)
    return out


def load_qpos_pair(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    with np.load(path, allow_pickle=True) as data:
        qpos = np.asarray(data["qpos"], dtype=np.float64)
        time = np.asarray(data["time"], dtype=np.float64) if "time" in data.files else None
    if qpos.ndim == 3 and qpos.shape[1] >= 2:
        sim = qpos[:, 0, :]
        ref = qpos[:, 1, :]
    elif qpos.ndim == 2:
        raise ValueError(f"{path} has one qpos track; expected sim/ref pair")
    else:
        raise ValueError(f"unsupported qpos shape for {path}: {qpos.shape}")
    return sim, ref, time


def fps_from_time(time: np.ndarray | None, default: float = 50.0) -> float:
    if time is None:
        return default
    arr = np.asarray(time, dtype=np.float64)
    if arr.ndim > 1:
        arr = arr[:, 0]
    dt = np.diff(arr)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    return float(1.0 / np.median(dt)) if dt.size else default


def body_positions(scene_xml: Path, qpos: np.ndarray) -> tuple[np.ndarray, list[str]]:
    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    data = mujoco.MjData(model)
    body_ids: list[int] = []
    names: list[str] = []
    for name in BODY_NAMES:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid >= 0:
            body_ids.append(bid)
            names.append(name)
    if not body_ids:
        raise ValueError(f"no monitored body ids found in {scene_xml}")
    if qpos.shape[1] != model.nq:
        raise ValueError(f"{scene_xml} model.nq={model.nq}, qpos width={qpos.shape[1]}")
    out = np.zeros((qpos.shape[0], len(body_ids), 3), dtype=np.float64)
    for idx, q in enumerate(qpos):
        data.qpos[:] = q
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        out[idx] = data.xpos[body_ids]
    return out, names


def p95(values: np.ndarray) -> float:
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    return float(np.percentile(vals, 95)) if vals.size else math.nan


def row_metrics(row: dict[str, str]) -> dict[str, Any] | None:
    path = qpos_path(row)
    scene = repo_path(row["rubber_scene_act"])
    if not path.is_file() or not scene.is_file():
        return None
    sim_qpos, ref_qpos, time = load_qpos_pair(path)
    n = min(sim_qpos.shape[0], ref_qpos.shape[0])
    sim_pos, names = body_positions(scene, sim_qpos[:n])
    ref_pos, _ = body_positions(scene, ref_qpos[:n])
    diff = sim_pos - ref_pos
    z_err = np.abs(diff[..., 2])
    xy_err = np.linalg.norm(diff[..., :2], axis=-1)
    err3d = np.linalg.norm(diff, axis=-1)
    fps = fps_from_time(time)
    z_accel = np.diff(sim_pos[..., 2], n=2, axis=0) * (fps**2) if n >= 3 else np.empty((0,))
    z_jerk = np.diff(sim_pos[..., 2], n=3, axis=0) * (fps**3) if n >= 4 else np.empty((0,))
    z_peak = float(np.max(z_err)) if z_err.size else math.nan
    err3d_peak = float(np.max(err3d)) if err3d.size else math.nan
    item = {
        "short_case_id": row["short_case_id"],
        "variant": row["variant"],
        "arm": row["arm"],
        "arm_kind": row["arm_kind"],
        "method": row["method"],
        "qpos_path": rel(path),
        "scene": rel(scene),
        "frames": int(n),
        "monitored_bodies": ",".join(names),
        "body_z_err_peak_m": z_peak,
        "body_z_err_p95_m": p95(z_err),
        "body_z_over_frac": float(np.mean(z_err > Z_THRESHOLD_M)) if z_err.size else math.nan,
        "holosoma_z_gate_pass": bool(z_peak <= Z_THRESHOLD_M) if math.isfinite(z_peak) else False,
        "sugar_3d_err_peak_m": err3d_peak,
        "sugar_3d_err_p95_m": p95(err3d),
        "sugar_3d_over_frac": float(np.mean(err3d > SUGAR_3D_THRESHOLD_M)) if err3d.size else math.nan,
        "sugar_3d_gate_pass": bool(err3d_peak <= SUGAR_3D_THRESHOLD_M) if math.isfinite(err3d_peak) else False,
        "xy_err_peak_m": float(np.max(xy_err)) if xy_err.size else math.nan,
        "xy_err_p95_m": p95(xy_err),
        "xy_only_failure": bool(err3d_peak > SUGAR_3D_THRESHOLD_M and z_peak <= Z_THRESHOLD_M)
        if math.isfinite(err3d_peak) and math.isfinite(z_peak)
        else False,
        "body_z_accel_p95": p95(np.abs(z_accel)),
        "body_z_jerk_p95": p95(np.abs(z_jerk)),
    }
    return item


def run(mode: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    subprocess.run([str(REPO / ".venv/bin/python"), str(BUILDER)], cwd=REPO, check=True)
    rows = read_tsv(VARIANTS)
    eval_rows = baseline_rows(rows) + rows
    metrics: list[dict[str, Any]] = []
    missing: list[str] = []
    for row in eval_rows:
        try:
            item = row_metrics(row)
        except Exception as exc:
            missing.append(f"{row['variant']}: {exc}")
            continue
        if item is None:
            if mode == "full" or row["arm"] == "baseline":
                missing.append(row["variant"])
            continue
        metrics.append(item)
    by_arm = {}
    for arm in sorted({row["arm"] for row in metrics}):
        arm_rows = [row for row in metrics if row["arm"] == arm]
        by_arm[arm] = {
            "rows": len(arm_rows),
            "holosoma_z_pass": sum(1 for row in arm_rows if row["holosoma_z_gate_pass"]),
            "sugar_3d_pass": sum(1 for row in arm_rows if row["sugar_3d_gate_pass"]),
            "xy_only_failures": sum(1 for row in arm_rows if row["xy_only_failure"]),
            "mean_body_z_peak_m": float(np.mean([row["body_z_err_peak_m"] for row in arm_rows])) if arm_rows else math.nan,
            "mean_sugar_3d_peak_m": float(np.mean([row["sugar_3d_err_peak_m"] for row in arm_rows])) if arm_rows else math.nan,
        }
    summary = {
        "mode": mode,
        "rows_evaluated": len(metrics),
        "missing_count": len(missing),
        "missing": missing[:50],
        "by_arm": by_arm,
        "z_threshold_m": Z_THRESHOLD_M,
        "sugar_3d_threshold_m": SUGAR_3D_THRESHOLD_M,
    }
    summary["zgate_eval_pass"] = len(missing) == 0 if mode == "full" else len([m for m in missing if "baseline" in m]) == 0
    return metrics, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", nargs="?", default="preflight", choices=["preflight", "full"])
    args = parser.parse_args()
    rows, summary = run(args.mode)
    out_dir = EVAL_ROOT / args.mode
    write_tsv(out_dir / "holosoma_zgate.tsv", rows)
    write_json(out_dir / "holosoma_zgate_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    if not summary["zgate_eval_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
