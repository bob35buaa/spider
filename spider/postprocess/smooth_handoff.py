"""CPU-only smoothing for SPIDER handoff NPZ artifacts.

This tool writes a new NPZ and never mutates the input. By default it smooths
floating arrays along axis 0 and leaves masks/contact/time-like arrays intact.
It is intended as the E166-B2 post-CEM hook; experiment launch scripts should
select explicit keys once the target handoff schema is fixed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np


SKIP_NAME_PARTS = (
    "mask",
    "contact",
    "phase",
    "time",
    "frame",
    "success",
    "valid",
)


def _normalize_window(window: int, length: int) -> int:
    window = max(3, int(window))
    if window % 2 == 0:
        window += 1
    if window > length:
        window = length if length % 2 == 1 else length - 1
    return max(3, window)


def _moving_average(arr: np.ndarray, window: int) -> np.ndarray:
    radius = window // 2
    padded = np.pad(arr, [(radius, radius)] + [(0, 0)] * (arr.ndim - 1), mode="edge")
    out = np.empty_like(arr, dtype=np.float64)
    for t in range(arr.shape[0]):
        out[t] = padded[t : t + window].mean(axis=0)
    return out


def _smooth_array(arr: np.ndarray, window: int, polyorder: int) -> np.ndarray:
    window = _normalize_window(window, arr.shape[0])
    polyorder = min(max(0, int(polyorder)), window - 1)
    try:
        from scipy.signal import savgol_filter  # type: ignore

        return savgol_filter(arr, window_length=window, polyorder=polyorder, axis=0)
    except Exception:
        return _moving_average(arr, window)


def _should_smooth(name: str, arr: np.ndarray, selected: set[str] | None) -> bool:
    if selected is not None:
        return name in selected
    if arr.ndim < 2 or arr.shape[0] < 3:
        return False
    if not np.issubdtype(arr.dtype, np.floating):
        return False
    lowered = name.lower()
    return not any(part in lowered for part in SKIP_NAME_PARTS)


def _boundary_restore_mask(
    arrays: dict[str, np.ndarray],
    pin_mask_keys: Iterable[str],
    length: int,
    radius: int,
) -> np.ndarray | None:
    restore = np.zeros(length, dtype=bool)
    found = False
    for key in pin_mask_keys:
        mask = arrays.get(key)
        if mask is None or mask.shape[0] != length:
            continue
        flat = np.asarray(mask).reshape(length, -1)
        active = flat.astype(bool).any(axis=1)
        transitions = np.flatnonzero(active[1:] != active[:-1]) + 1
        for idx in transitions:
            lo = max(0, idx - radius)
            hi = min(length, idx + radius + 1)
            restore[lo:hi] = True
        found = True
    return restore if found else None


def smooth_npz(
    input_path: Path,
    output_path: Path,
    keys: Iterable[str] | None = None,
    window: int = 7,
    polyorder: int = 2,
    pin_mask_keys: Iterable[str] = (),
    pin_radius: int = 1,
    force: bool = False,
) -> dict[str, object]:
    if output_path.exists() and not force:
        raise FileExistsError(f"output exists; pass --force to overwrite: {output_path}")
    if input_path.resolve() == output_path.resolve():
        raise ValueError("input and output must be different paths")

    selected = set(keys) if keys else None
    with np.load(input_path, allow_pickle=True) as data:
        arrays = {key: data[key] for key in data.files}

    output: dict[str, np.ndarray] = {}
    smoothed: list[str] = []
    skipped: list[str] = []
    for name, arr in arrays.items():
        if _should_smooth(name, arr, selected):
            restore = _boundary_restore_mask(
                arrays,
                pin_mask_keys,
                arr.shape[0],
                max(0, int(pin_radius)),
            )
            new_arr = _smooth_array(arr, window=window, polyorder=polyorder).astype(
                arr.dtype,
                copy=False,
            )
            if restore is not None:
                new_arr[restore] = arr[restore]
            output[name] = new_arr
            smoothed.append(name)
        else:
            output[name] = arr
            skipped.append(name)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **output)
    report = {
        "input": str(input_path),
        "output": str(output_path),
        "window": int(window),
        "polyorder": int(polyorder),
        "pin_mask_keys": list(pin_mask_keys),
        "pin_radius": int(pin_radius),
        "smoothed_keys": smoothed,
        "skipped_keys": skipped,
    }
    report_path = output_path.with_suffix(output_path.suffix + ".smooth_report.json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    report["report"] = str(report_path)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--key",
        action="append",
        default=None,
        help="Array key to smooth. Repeat to smooth multiple keys. Default: all eligible floating arrays.",
    )
    parser.add_argument("--window", type=int, default=7)
    parser.add_argument("--polyorder", type=int, default=2)
    parser.add_argument(
        "--pin-mask-key",
        action="append",
        default=[],
        help="Mask key whose on/off boundaries should keep original values.",
    )
    parser.add_argument("--pin-radius", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = smooth_npz(
        input_path=args.input,
        output_path=args.output,
        keys=args.key,
        window=args.window,
        polyorder=args.polyorder,
        pin_mask_keys=args.pin_mask_key,
        pin_radius=args.pin_radius,
        force=args.force,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
