"""Z-only smoothing for SPIDER handoff NPZ artifacts.

This is the E167 B2 postprocess hook. It intentionally leaves x/y, joints,
object state, contacts, masks, and reward traces unchanged. The first version
only smooths root z in qpos and qvel because those are the only axis-addressable
fields in the SPIDER NPZ schema before SUGAR reference export.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from spider.postprocess.smooth_handoff import _smooth_array


def _smooth_component(arr: np.ndarray, index: tuple[slice | int, ...], window: int, polyorder: int) -> np.ndarray:
    out = arr.copy()
    component = np.asarray(arr[index])
    out[index] = _smooth_array(component, window=window, polyorder=polyorder).astype(
        component.dtype,
        copy=False,
    )
    return out


def _max_delta(before: np.ndarray, after: np.ndarray, index: tuple[slice | int, ...]) -> float:
    return float(np.max(np.abs(after[index] - before[index]))) if before.size else 0.0


def smooth_npz_zonly(
    input_path: Path,
    output_path: Path,
    window: int = 7,
    polyorder: int = 2,
    force: bool = False,
) -> dict[str, object]:
    if output_path.exists() and not force:
        raise FileExistsError(f"output exists; pass --force to overwrite: {output_path}")
    if input_path.resolve() == output_path.resolve():
        raise ValueError("input and output must be different paths")

    with np.load(input_path, allow_pickle=True) as data:
        arrays = {key: data[key] for key in data.files}

    output: dict[str, np.ndarray] = dict(arrays)
    changed: list[str] = []
    xy_checks: dict[str, float] = {}
    z_checks: dict[str, float] = {}

    qpos = arrays.get("qpos")
    if isinstance(qpos, np.ndarray) and qpos.ndim >= 2 and qpos.shape[-1] > 2:
        index = (slice(None),) * (qpos.ndim - 1) + (2,)
        new_qpos = _smooth_component(qpos, index, window=window, polyorder=polyorder)
        output["qpos"] = new_qpos
        changed.append("qpos[...,2]")
        xy_checks["qpos_xy_max_abs_delta"] = _max_delta(
            qpos,
            new_qpos,
            (slice(None),) * (qpos.ndim - 1) + (slice(0, 2),),
        )
        z_checks["qpos_z_max_abs_delta"] = _max_delta(qpos, new_qpos, index)

    qvel = arrays.get("qvel")
    if isinstance(qvel, np.ndarray) and qvel.ndim >= 2 and qvel.shape[-1] > 2:
        index = (slice(None),) * (qvel.ndim - 1) + (2,)
        new_qvel = _smooth_component(qvel, index, window=window, polyorder=polyorder)
        output["qvel"] = new_qvel
        changed.append("qvel[...,2]")
        xy_checks["qvel_xy_max_abs_delta"] = _max_delta(
            qvel,
            new_qvel,
            (slice(None),) * (qvel.ndim - 1) + (slice(0, 2),),
        )
        z_checks["qvel_z_max_abs_delta"] = _max_delta(qvel, new_qvel, index)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **output)
    report = {
        "input": str(input_path),
        "output": str(output_path),
        "window": int(window),
        "polyorder": int(polyorder),
        "changed_components": changed,
        "xy_max_abs_delta": max(xy_checks.values(), default=0.0),
        "xy_checks": xy_checks,
        "z_checks": z_checks,
        "schema_note": "SPIDER NPZ z-only pass smooths qpos/qvel root-z only; SUGAR body-z audit remains required.",
    }
    report_path = output_path.with_suffix(output_path.suffix + ".zonly_smooth_report.json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    report["report"] = str(report_path)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--window", type=int, default=7)
    parser.add_argument("--polyorder", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = smooth_npz_zonly(
        input_path=args.input,
        output_path=args.output,
        window=args.window,
        polyorder=args.polyorder,
        force=args.force,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
