#!/usr/bin/env python3
"""Recompute the fixed (42-dim, single-channel) qpos_jerk_l2_p95 for every
case feeding the box004/024/001 (E189 vs PRG) and box023/021 (PRG vs no-PRG)
reports.

The pre-existing ``qpos_jerk_l2_p95`` column in E168/E170/E173/E179's own
case_metrics.tsv files used a buggy formula (see
``eval.core.motion_health.qpos_kinematic_health`` fix) that flattened the
npz's ``(T, 2, nq)`` qpos array into an 84-dim vector instead of using only
the actual robot qpos channel (``npz_qpos()[0]``, 42-dim). This script
re-derives the corrected value directly from each case's raw
``trajectory_mjwp_act.npz`` and writes a small lookup JSON that the report
generators read from; it does not touch any historical case_metrics.tsv.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

SCRIPT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SCRIPT_ROOT / "experiments/E189"))

import e189_common as C  # noqa: E402

from eval.core.motion_health import qpos_kinematic_health  # noqa: E402

OUTPUT = C.REPO / "workspace/core4d/results/E189/s6_downstream/eval/full/qpos_jerk_fixed.json"

# (label, tsv_path, object_key_filter) — label is used purely for logging.
SOURCES = [
    ("E172_box004_prg", C.REPO / "workspace/core4d/results/E172/s6_downstream/eval/full/e171_case_metrics.tsv", "box004"),
    ("E173_box024_prg", C.REPO / "workspace/core4d/results/E173/s6_downstream/eval/full/e173_case_metrics.tsv", "box024"),
    ("E173_box001_prg", C.REPO / "workspace/core4d/results/E173/s6_downstream/eval/full/e173_case_metrics.tsv", "box001"),
    ("E173_box023_prg", C.REPO / "workspace/core4d/results/E173/s6_downstream/eval/full/e173_case_metrics.tsv", "box023"),
    ("E189_box004_noprg", C.REPO / "workspace/core4d/results/E189/s6_downstream/eval/full/e189_case_metrics.tsv", "box004"),
    ("E189_box024_noprg", C.REPO / "workspace/core4d/results/E189/s6_downstream/eval/full/e189_case_metrics.tsv", "box024"),
    ("E189_box001_noprg", C.REPO / "workspace/core4d/results/E189/s6_downstream/eval/full/e189_case_metrics.tsv", "box001"),
    ("E179_box023_noprg", C.REPO / "workspace/core4d/results/E179/s6_downstream/eval/full/e179_case_metrics.tsv", "box023"),
    ("E170_box021_prg", C.REPO / "workspace/core4d/results/E170/s6_downstream/eval/full/e170_case_metrics.tsv", "box021"),
    ("E168_box021_noprg", C.REPO / "workspace/core4d/results/E168/s6_downstream/cem/eval/box021_all28_reviewed/e168_case_metrics.tsv", "box021"),
]

def resolve_npz(row: dict[str, str]) -> Path:
    """Resolve a case's trajectory npz, tolerating stale absolute paths left
    over from experiments originally run on a different machine (result_npz
    is the canonical archived copy; qpos_path/outdir_npz are the raw
    workdir output, sometimes since cleaned up)."""
    for key in ("result_npz", "qpos_path", "outdir_npz"):
        value = row.get(key)
        if not value:
            continue
        candidate = C.repo_path(value)
        if candidate.is_file():
            return candidate
        parts = Path(value).parts
        if "core4d" in parts:
            remapped = C.REPO / "workspace" / Path(*parts[parts.index("core4d") :])
            if remapped.is_file():
                return remapped
    raise FileNotFoundError(f"no resolvable npz for case_id={row.get('case_id')}: {row}")


EXPECTED_COUNTS = {
    "E172_box004_prg": 6,
    "E173_box024_prg": 9,
    "E173_box001_prg": 28,
    "E173_box023_prg": 16,
    "E189_box004_noprg": 6,
    "E189_box024_noprg": 9,
    "E189_box001_noprg": 28,
    "E179_box023_noprg": 16,
    "E170_box021_prg": 28,
    "E168_box021_noprg": 28,
}


def main() -> int:
    by_label: dict[str, dict[str, float]] = {}
    for label, tsv_path, object_key in SOURCES:
        rows = C.read_tsv(tsv_path)
        filtered = [r for r in rows if r.get("object_key") == object_key]
        expected = EXPECTED_COUNTS[label]
        if len(filtered) != expected:
            raise ValueError(f"{label}: expected {expected} rows, got {len(filtered)} in {tsv_path}")
        values: dict[str, float] = {}
        for row in filtered:
            qpos_path = resolve_npz(row)
            result = qpos_kinematic_health(qpos_path, 30.0)
            if not math.isfinite(result["qpos_jerk_l2_p95"]):
                raise ValueError(f"{label}/{row['case_id']}: non-finite jerk from {qpos_path}")
            values[row["case_id"]] = result["qpos_jerk_l2_p95"]
        by_label[label] = values
        print(f"[ok] {label}: {len(values)} cases")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(by_label, indent=2, sort_keys=True), encoding="utf-8")
    print(OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
