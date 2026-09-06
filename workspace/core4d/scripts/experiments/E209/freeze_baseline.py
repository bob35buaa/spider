#!/usr/bin/env python3
"""E209 P0: freeze the three E206-PRG baselines as read-only, hashed TSVs.

E209's whole design is paired against E206's PRG arm, so every gate threshold in
plan239 is stated relative to a baseline number.  If those numbers are re-derived
ad hoc at report time they can drift silently; freezing them here (and hashing
them) makes each Claim checkable against a committed artifact.

Three baselines, all restricted to the 22 delivered cases:

  1. z diagnostic  -- measured with `gen_E178_object_z_diff_report.object_z_series`,
     the same function E178 and E207 used (rule 13).  Also re-asserts the frozen
     `e209_common.BASELINE_Z_BIAS_CM` that the S+/S- strata are derived from, so
     the split provably predates the run.
  2. 14-gate       -- the 22 prg rows of E206's `e206_two_arm_rollout.tsv`.
  3. manual review -- the 22 rows of E206's `user_manual_review_filled.tsv`.

Usage:
    MUJOCO_GL=disable .venv/bin/python \
      workspace/core4d/scripts/experiments/E209/freeze_baseline.py
"""

from __future__ import annotations

import csv
import json
import statistics
import sys
from pathlib import Path

import mujoco
import numpy as np

REPO = Path(__file__).resolve().parents[5]
for _p in (
    "workspace/core4d/scripts",
    "workspace/core4d/scripts/eval/reports",
    "workspace/core4d/scripts/experiments/E209",
):
    _s = str(REPO / _p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

from eval.core.core_metrics import npz_qpos  # noqa: E402
from gen_E178_object_z_diff_report import object_z_series  # noqa: E402

import e209_common as C  # noqa: E402

E206_EVAL = REPO / "workspace/core4d/results/E206/s6_downstream/eval/two_arm"
ROLLOUT_TSV = E206_EVAL / "e206_two_arm_rollout.tsv"
REVIEW_TSV = E206_EVAL / "user_manual_review_filled.tsv"

Z_FIELDS = [
    "case_id",
    "object_key",
    "frames",
    "z_bias_cm",
    "z_mae_cm",
    "z_rmse_cm",
    "z_abs_p95_cm",
    "z_abs_max_cm",
    "obj_pos_err_3d_cm",
    "ref_z_range_cm",
    "object_mass_kg",
    "stratum",
]


def measure_z(row: dict[str, str]) -> dict[str, object]:
    case_id = row["case_id"]
    scene = C.base_scene_path(row)
    model = mujoco.MjModel.from_xml_path(str(scene))
    run_qpos, _ = npz_qpos(C.baseline_npz(case_id))
    kin = np.asarray(np.load(C.kinematic_npz(row), allow_pickle=True)["qpos"], dtype=np.float64)
    if kin.ndim == 3:
        kin = kin[:, 0, :]
    z_sim, z_ref, err_3d = object_z_series(run_qpos, kin, model)
    dz = (z_sim - z_ref) * 100.0
    adz = np.abs(dz)
    return {
        "case_id": case_id,
        "object_key": row["object_key"],
        "frames": len(dz),
        "z_bias_cm": round(float(dz.mean()), 4),
        "z_mae_cm": round(float(adz.mean()), 4),
        "z_rmse_cm": round(float(np.sqrt((dz**2).mean())), 4),
        "z_abs_p95_cm": round(float(np.percentile(adz, 95)), 4),
        "z_abs_max_cm": round(float(adz.max()), 4),
        "obj_pos_err_3d_cm": round(float(err_3d.mean() * 100.0), 4),
        "ref_z_range_cm": round(float((z_ref.max() - z_ref.min()) * 100.0), 4),
        "object_mass_kg": round(C.object_mass(scene), 4),
        "stratum": "S+" if case_id in C.S_PLUS else "S-",
    }


def read_tsv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        return list(reader.fieldnames or []), list(reader)


def write_tsv(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    rows = C.sources()
    order = {c: i for i, c in enumerate(C.CASES)}
    C.BASELINE_DIR.mkdir(parents=True, exist_ok=True)

    # --- 1. z diagnostic ---------------------------------------------------
    z_rows = [measure_z(r) for r in rows]
    drift = [
        (r["case_id"], r["z_bias_cm"], C.BASELINE_Z_BIAS_CM[r["case_id"]])
        for r in z_rows
        if abs(r["z_bias_cm"] - C.BASELINE_Z_BIAS_CM[r["case_id"]]) > C.BASELINE_Z_TOL_CM
    ]
    if drift:
        raise SystemExit(
            "frozen BASELINE_Z_BIAS_CM disagrees with re-measurement "
            f"(tol {C.BASELINE_Z_TOL_CM} cm):\n"
            + "\n".join(f"  {c}: measured {m:+.4f} vs frozen {f:+.4f}" for c, m, f in drift)
        )
    write_tsv(C.BASELINE_DIR / "e209_baseline_z_prg.tsv", Z_FIELDS, z_rows)

    bias = np.array([r["z_bias_cm"] for r in z_rows])
    sminus = np.array([r["z_bias_cm"] for r in z_rows if r["stratum"] == "S-"])
    splus = np.array([r["z_bias_cm"] for r in z_rows if r["stratum"] == "S+"])

    # --- 2. 14-gate --------------------------------------------------------
    gate_fields, gate_all = read_tsv(ROLLOUT_TSV)
    gate_rows = [r for r in gate_all if r["arm"] == "prg" and r["case_id"] in order]
    gate_rows.sort(key=lambda r: order[r["case_id"]])
    if len(gate_rows) != C.EXPECTED_CASES:
        raise SystemExit(f"expected {C.EXPECTED_CASES} prg rollout rows, got {len(gate_rows)}")
    write_tsv(C.BASELINE_DIR / "e209_baseline_14gate_prg.tsv", gate_fields, gate_rows)
    narrow = sum(1 for r in gate_rows if r["narrow_pass"] == "True")
    wide = sum(1 for r in gate_rows if r["wide_pass"] == "True")
    hard = sum(1 for r in gate_rows if r["hard_pass"] == "True")

    # --- 3. manual review --------------------------------------------------
    # E206 keys its review rows "<case_id>#<ARM>" (65 rows = 65 cases, PRG only).
    rev_fields, rev_all = read_tsv(REVIEW_TSV)
    rev_rows = [r for r in rev_all if r["case_id"].split("#", 1)[0] in order]
    rev_rows.sort(key=lambda r: order[r["case_id"].split("#", 1)[0]])
    if len(rev_rows) != C.EXPECTED_CASES:
        raise SystemExit(f"expected {C.EXPECTED_CASES} review rows, got {len(rev_rows)}")
    write_tsv(C.BASELINE_DIR / "e209_baseline_review_prg.tsv", rev_fields, rev_rows)
    use = sum(1 for r in rev_rows if r["manual_use_decision"] == "USE")

    # --- summary + hashes --------------------------------------------------
    summary = {
        "exp": C.EXP,
        "n_cases": len(rows),
        "source_tsv": str(C.SOURCE_TSV.relative_to(REPO)),
        "source_tsv_sha256": C.EXPECTED_SOURCE_SHA256,
        "z": {
            "macro_bias_cm": round(float(bias.mean()), 4),
            "mean_abs_bias_cm": round(float(np.abs(bias).mean()), 4),
            "mean_mae_cm": round(float(statistics.fmean(r["z_mae_cm"] for r in z_rows)), 4),
            "n_negative": int((bias < 0).sum()),
            "s_minus": {"n": len(sminus), "macro_bias_cm": round(float(sminus.mean()), 4)},
            "s_plus": {"n": len(splus), "macro_bias_cm": round(float(splus.mean()), 4)},
        },
        "gates": {"hard_pass": hard, "wide_pass": wide, "narrow_pass": narrow},
        "review": {"use": use, "n": len(rev_rows)},
        "frozen_files": {},
    }
    for name in (
        "e209_baseline_z_prg.tsv",
        "e209_baseline_14gate_prg.tsv",
        "e209_baseline_review_prg.tsv",
    ):
        summary["frozen_files"][name] = C.sha256(C.BASELINE_DIR / name)
    (C.BASELINE_DIR / "e209_baseline_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    # --- plan239 P0 exit checks -------------------------------------------
    checks = [
        ("z macro bias == -2.517", abs(summary["z"]["macro_bias_cm"] - (-2.517)) <= 0.001),
        ("z mean|bias| == 3.241", abs(summary["z"]["mean_abs_bias_cm"] - 3.241) <= 0.001),
        ("z negative == 18/22", summary["z"]["n_negative"] == 18),
        ("S- n=18 macro -3.519", len(sminus) == 18 and abs(sminus.mean() + 3.519) <= 0.001),
        ("S+ n=4 macro +1.994", len(splus) == 4 and abs(splus.mean() - 1.994) <= 0.001),
        ("hard_pass == 22/22", hard == 22),
        ("narrow_pass == 13/22", narrow == 13),
        ("manual USE == 22/22", use == 22),
    ]
    print(f"\nfrozen -> {C.BASELINE_DIR.relative_to(REPO)}")
    for label, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    failed = [label for label, ok in checks if not ok]
    if failed:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        raise SystemExit(f"P0 exit check failed: {failed}")
    print(f"\nP0 baseline freeze PASS ({len(rows)} cases)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
