#!/usr/bin/env python3
"""E208: verify every retarget artifact is internally consistent and usable.

After the 2026-09-05 concurrent-runner incidents the question "which process
wrote this file" is unanswerable for some artifacts.  Rather than reason about
provenance, this checks the property that actually matters downstream -- each
aug npz must be a loadable, finite, correctly-shaped trajectory that shares the
`_original` time base, because `fixed_window_trim` slices all of them at one
window and the CEM scene is built from the result.

Per (case, retarget-variant root, aug variant):
  V1  npz loads and carries `qpos`
  V2  frame count == the `_original` frame count in the same root
      (fixed_window_trim assumes this; a mismatch means a torn or stale file)
  V3  qpos width == the `_original` width
  V4  all finite -- no NaN/Inf from a half-written file
  V5  the object channel actually moved relative to `_original`
      (a byte-identical copy of orig would mean the augmentation silently did
       nothing, which a shape check alone would not catch)

Usage:
    .venv/bin/python .../E208/verify_retarget_artifacts.py
    ... --json-out PATH
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import e208_common as C  # noqa: E402

SEED = C.load_e208_module("seed_from_e206")

OBJ_POS = slice(36, 39)


def check_root(case: dict[str, str], variant: str) -> list[dict[str, Any]]:
    base = case["base_target_task"]
    meta = C.load_case_meta(base)
    holo = meta["holosoma_task"]
    root = C.holosoma_dir(base, variant) / "retargeted"
    rows: list[dict[str, Any]] = []

    orig_path = root / f"{holo}_original.npz"
    if not orig_path.is_file():
        return rows
    with np.load(orig_path, allow_pickle=True) as data:
        q_orig = np.asarray(data["qpos"], dtype=np.float64)

    for short, holo_name in C.BUILD_VARIANTS:
        path = root / f"{holo}_{holo_name}.npz"
        if not path.is_file():
            continue
        row: dict[str, Any] = {
            "case_id": case["case_id"], "object_key": case["object_key"],
            "variant_root": variant, "variant": short,
            "npz": C.rel(path), "sha256": C.sha256(path),
        }
        failures: list[str] = []
        try:
            with np.load(path, allow_pickle=True) as data:
                if "qpos" not in data:
                    failures.append("V1: no qpos array")
                    q = None
                else:
                    q = np.asarray(data["qpos"], dtype=np.float64)
        except (OSError, ValueError, EOFError) as exc:
            failures.append(f"V1: unreadable ({type(exc).__name__}: {exc})")
            q = None

        if q is not None:
            row["frames"] = int(q.shape[0])
            row["width"] = int(q.shape[1])
            if q.shape[0] != q_orig.shape[0]:
                failures.append(f"V2: {q.shape[0]} frames != _original {q_orig.shape[0]}")
            if q.shape[1] != q_orig.shape[1]:
                failures.append(f"V3: width {q.shape[1]} != _original {q_orig.shape[1]}")
            if not np.isfinite(q).all():
                failures.append(f"V4: {int((~np.isfinite(q)).sum())} non-finite values")
            if q.shape == q_orig.shape:
                delta = float(np.linalg.norm(q[:, OBJ_POS] - q_orig[:, OBJ_POS], axis=1).max())
                row["max_object_delta_m"] = round(delta, 6)
                if delta < 1e-6:
                    failures.append("V5: object channel identical to _original (no augmentation)")

        row["failures"] = "; ".join(failures)
        row["status"] = "pass" if not failures else "fail"
        rows.append(row)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv-out", type=Path,
                    default=C.PREFLIGHT_DIR / "retarget_artifact_verify.tsv")
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    rows: list[dict[str, Any]] = []
    for case in C.load_e208_cases():
        for variant in SEED.target_variants(case, C.RETARGET_VARIANTS):
            rows.extend(check_root(case, variant))

    failures = [r for r in rows if r["status"] != "pass"]
    for row in failures:
        print(f"[FAIL] {row['case_id']:32s} {row['variant_root']:10s} {row['variant']:7s} "
              f"{row['failures']}")

    C.write_tsv(args.tsv_out, rows,
                [k for k in rows[0] if k != "failures"] + ["failures"] if rows else None)
    by_root: dict[str, int] = {}
    for row in rows:
        by_root[row["variant_root"]] = by_root.get(row["variant_root"], 0) + 1
    payload = {
        "experiment": C.EXP_ID, "generated_at": C.now(),
        "n_artifacts": len(rows), "n_failures": len(failures),
        "by_variant_root": by_root,
        "verdict": "pass" if not failures else "fail",
        "note": ("Provenance-independent: after the concurrent-runner incidents these "
                 "checks establish the artifacts are usable regardless of which process "
                 "wrote them."),
    }
    C.write_json(args.json_out or args.tsv_out.with_suffix(".json"), payload)
    print(f"\n{len(rows) - len(failures)}/{len(rows)} aug artifacts pass "
          f"(by root: {by_root}) -> {C.rel(args.tsv_out)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
