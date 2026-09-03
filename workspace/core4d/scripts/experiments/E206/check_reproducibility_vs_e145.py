#!/usr/bin/env python3
"""E206 P5: is the rebuilt S3 trajectory identical to the E145/E174-era one?

plan236 P5 rebuilds all 65 task dirs rather than reusing the 43 existing ones,
which overwrites the trajectories E145/E174 produced.  Same raw clip + same
omnirt_v1 parameters + same SPIDER preprocess should reproduce bit-for-bit;
anything else means the pipeline carries hidden non-determinism, and that is a
finding worth reporting on its own (plan236 P5 "代价缓解").

The comparison baseline is the pre-P5 copy under
``results/E206/s3_retarget/e145_baseline/`` -- captured *before* S3 ran, because
S3 destroys the originals in place.

Exit code is 1 when any overlapping case diverges beyond --tol, so the pipeline
script surfaces it instead of burying it in a TSV nobody opens.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import e206_common as C  # noqa: E402

BASELINE_DIR = C.RESULTS / "s3_retarget/e145_baseline"
ARRAYS = ("qpos", "qvel", "ctrl", "contact", "contact_pos")


def compare(baseline: Path, rebuilt: Path, tol: float) -> dict[str, Any]:
    old = np.load(baseline)
    new = np.load(rebuilt)
    row: dict[str, Any] = {
        "baseline_frames": int(old["qpos"].shape[0]),
        "rebuilt_frames": int(new["qpos"].shape[0]),
    }
    worst = 0.0
    verdict = "identical"
    for name in ARRAYS:
        if name not in old.files or name not in new.files:
            row[f"{name}_max_abs_diff"] = "missing"
            verdict = "schema_mismatch"
            continue
        a, b = old[name], new[name]
        if a.shape != b.shape:
            # A different frame count is a real divergence, not a tolerance
            # question -- trimming decided differently on identical input.
            row[f"{name}_max_abs_diff"] = f"shape {a.shape}!={b.shape}"
            verdict = "shape_mismatch"
            continue
        diff = float(np.max(np.abs(a - b))) if a.size else 0.0
        row[f"{name}_max_abs_diff"] = diff
        worst = max(worst, diff)
    row["worst_max_abs_diff"] = worst
    if verdict == "identical" and worst > tol:
        verdict = "diverged"
    elif verdict == "identical" and worst > 0.0:
        verdict = "within_tol"
    row["verdict"] = verdict
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1-manifest-tsv", type=Path, required=True)
    ap.add_argument("--baseline-dir", type=Path, default=BASELINE_DIR)
    ap.add_argument("--out-dir", type=Path, default=C.RESULTS / "s3_retarget")
    ap.add_argument(
        "--tol",
        type=float,
        default=1e-9,
        help="max |delta| still called reproducible; default is ~bit-identical",
    )
    args = ap.parse_args()

    baselines = {p.stem: p for p in sorted(args.baseline_dir.glob("*.npz"))}
    if not baselines:
        print(f"[skip] no baselines under {args.baseline_dir}")
        return 0

    rows: list[dict[str, Any]] = []
    for case_id, baseline in baselines.items():
        rebuilt = (
            C.PROCESSED_ROOT
            / f"dcv3_omnirt_v1_ref_fk_{case_id}"
            / "0"
            / "trajectory_kinematic.npz"
        )
        row: dict[str, Any] = {
            "case_id": case_id,
            "object_key": case_id.split("_")[0],
        }
        if not rebuilt.is_file():
            # The case failed S3 (CVXPY infeasible etc). That is a Stage2b
            # outcome recorded in the manifest, not a reproducibility verdict.
            row.update({"verdict": "rebuilt_missing", "worst_max_abs_diff": ""})
        else:
            row.update(compare(baseline, rebuilt, args.tol))
        rows.append(row)

    fields = ["case_id", "object_key", "verdict", "worst_max_abs_diff",
              "baseline_frames", "rebuilt_frames"]
    fields += [f"{n}_max_abs_diff" for n in ARRAYS]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    C.write_tsv(args.out_dir / "reproducibility_vs_e145.tsv", rows, fields)

    tally: dict[str, int] = {}
    for row in rows:
        tally[str(row["verdict"])] = tally.get(str(row["verdict"]), 0) + 1
    diverged = [r for r in rows if r["verdict"] in ("diverged", "shape_mismatch",
                                                    "schema_mismatch")]
    payload = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "baseline_dir": str(args.baseline_dir.relative_to(C.REPO)),
        "tol": args.tol,
        "n_compared": len(rows),
        "verdicts": tally,
        "diverged_cases": [r["case_id"] for r in diverged],
    }
    (args.out_dir / "reproducibility_vs_e145.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 1 if diverged else 0


if __name__ == "__main__":
    raise SystemExit(main())
