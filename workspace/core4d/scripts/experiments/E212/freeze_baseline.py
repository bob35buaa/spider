#!/usr/bin/env python3
"""E212 P0: re-measure the two frozen endpoints and assert they match plan242.

Stage A never re-runs g=0 (E206 PRG) or g=1 (E209 G1); it joins them to the
three new arms into one 5-point curve.  That only works if the endpoints are
byte-stable, so this script re-derives every number in
``e212_common.BASELINE_DESK023`` from E209's delivered TSVs and fails on any
drift.  Nothing here is computed a second way -- it reads the same authority the
E209 log reports from, which is the point: if E209's outputs move under us, the
whole comparison is void and we want to know before spending GPU.

It also copies the desk023 slices of those TSVs into ``results/E212/baseline/``
with their sha256, so the comparison stays reproducible even if E209's directory
is later regenerated.

Unlike desk007, all four desk023 cases have a finite ``release`` (no empty
release window), so ``BASELINE_RELEASE_N`` is a true 4 -- this script asserts
that rather than assuming it.

Usage:
    .venv/bin/python workspace/core4d/scripts/experiments/E212/freeze_baseline.py
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO / "workspace/core4d/scripts/experiments/E212"))

import e212_common as C  # noqa: E402

E209_EVAL = REPO / "workspace/core4d/results/E209/s6_downstream/eval/two_arm"
ROLLOUT_TSV = E209_EVAL / "e209_two_arm_rollout.tsv"
ZDIFF_TSV = E209_EVAL / "e209_object_z_diff_by_case.tsv"

#: rollout-TSV column -> BASELINE_DESK023 key.
ROLLOUT_FIELDS = {
    "track_obj_pos_err_cm_mean": "obj_pos_cm",
    "track_obj_ori_err_deg_mean": "obj_ori_deg",
    "track_eef_ori_err_deg_mean": "eef_ori_deg",
    "track_eef_pos_err_cm_mean": "eef_pos_cm",
    "track_root_ori_err_deg_mean": "root_ori_deg",
    "track_root_pos_err_cm_mean": "root_pos_cm",
    "hand_object_physics_contact_in_mask_frac": "contact",
    "hand_object_release_false_contact_3mm_frac": "release",
    "hand_object_physics_penetration_3mm_frame_frac": "hand_pen",
    "body_z_err_p95_m": "body_z_p95_m",
    "leg_penetration_frac": "leg_pen",
    "ankle_jerk_p95": "ankle_jerk_p95",
}


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def num(value: str) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(out) else out


def measure() -> tuple[dict[str, dict[str, float]], dict[str, int]]:
    rollout = [r for r in read_tsv(ROLLOUT_TSV) if r["object_key"] == C.OBJECT_KEY]
    zdiff = [r for r in read_tsv(ZDIFF_TSV) if r["object_key"] == C.OBJECT_KEY]
    cases = set(C.CASES)
    if {r["case_id"] for r in rollout} != cases:
        raise SystemExit(
            f"rollout TSV {C.OBJECT_KEY} cases != E212 scope: "
            f"{sorted({r['case_id'] for r in rollout})}"
        )

    out: dict[str, dict[str, float]] = {}
    counts: dict[str, int] = {}
    for arm, rollout_tag, z_tag in (("prg", "prg", "PRG"), ("g1", "g1", "G1")):
        rows = [r for r in rollout if r["arm"] == rollout_tag]
        zrows = [r for r in zdiff if r["arm"] == z_tag]
        if len(rows) != C.EXPECTED_CASES or len(zrows) != C.EXPECTED_CASES:
            raise SystemExit(
                f"{arm}: {len(rows)} rollout / {len(zrows)} z rows, want {C.EXPECTED_CASES}"
            )
        stats: dict[str, float] = {}
        for column, key in ROLLOUT_FIELDS.items():
            vals = [v for v in (num(r[column]) for r in rows) if v is not None]
            if not vals:
                raise SystemExit(f"{arm}/{column}: all values non-finite")
            stats[key] = sum(vals) / len(vals)
            counts[f"{arm}.{key}"] = len(vals)
        bias = [num(r["z_bias_cm"]) for r in zrows]
        mae = [num(r["z_mae_cm"]) for r in zrows]
        stats["z_bias_cm"] = sum(bias) / len(bias)
        stats["z_abs_bias_cm"] = sum(abs(b) for b in bias) / len(bias)
        stats["z_mae_cm"] = sum(mae) / len(mae)
        stats["narrow_pass"] = float(sum(r["narrow_pass"] == "True" for r in rows))
        stats["hard_pass"] = float(sum(r["hard_pass"] == "True" for r in rows))
        out[arm] = stats
    return out, counts


def main() -> int:
    for path in (ROLLOUT_TSV, ZDIFF_TSV):
        if not path.is_file():
            raise SystemExit(f"missing E209 authority: {path}")

    measured, counts = measure()
    problems: list[str] = []
    for arm, frozen in C.BASELINE_DESK023.items():
        for key, want in frozen.items():
            got = measured[arm][key]
            tol = C.BASELINE_TOL.get(key, C.BASELINE_TOL_DEFAULT)
            if abs(got - want) > tol:
                problems.append(f"{arm}.{key}: frozen {want} != measured {got:.6f} (tol {tol})")

    # Every arm must average `release` over the SAME cases or the C3 comparison
    # is between different populations.  desk023 has no empty release window, so
    # this must be a full n=4 -- assert rather than assume.
    for arm in C.BASELINE_DESK023:
        n = counts[f"{arm}.release"]
        if n != C.BASELINE_RELEASE_N:
            problems.append(f"{arm}.release n={n} != {C.BASELINE_RELEASE_N}")

    # Per-case pre-bias / delta, the inputs to P1 and to the g* prediction.
    zrows = [r for r in read_tsv(ZDIFF_TSV) if r["object_key"] == C.OBJECT_KEY]
    by = {(r["arm"], r["case_id"]): num(r["z_bias_cm"]) for r in zrows}
    for case_id, prereg in C.PREREG_PER_CASE.items():
        pre, post = by[("PRG", case_id)], by[("G1", case_id)]
        if abs(pre - prereg["pre_bias_cm"]) > 1e-3:
            problems.append(f"{case_id}: pre_bias frozen {prereg['pre_bias_cm']} != {pre:.4f}")
        if abs((post - pre) - prereg["delta_g1_cm"]) > 1e-3:
            problems.append(f"{case_id}: delta_g1 frozen {prereg['delta_g1_cm']} != {post - pre:.4f}")
        g_star = -pre / (post - pre)
        if abs(g_star - prereg["g_star"]) > 1e-3:
            problems.append(f"{case_id}: g_star frozen {prereg['g_star']} != {g_star:.4f}")

    # P1's line is fixed analytically by the two endpoints; check the frozen
    # coefficients really are those endpoints, so R^2 later measures only how far
    # the three new arms deviate.
    want_b0 = measured["prg"]["z_bias_cm"]
    want_slope = measured["g1"]["z_bias_cm"] - measured["prg"]["z_bias_cm"]
    if abs(want_b0 - C.PREREG_Z_LINEAR["intercept_cm"]) > 5e-3:
        problems.append(f"PREREG_Z_LINEAR intercept {C.PREREG_Z_LINEAR['intercept_cm']} != {want_b0:.4f}")
    if abs(want_slope - C.PREREG_Z_LINEAR["slope_cm_per_g"]) > 5e-3:
        problems.append(f"PREREG_Z_LINEAR slope {C.PREREG_Z_LINEAR['slope_cm_per_g']} != {want_slope:.4f}")

    C.BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    for src in (ROLLOUT_TSV, ZDIFF_TSV):
        rows = [r for r in read_tsv(src) if r["object_key"] == C.OBJECT_KEY]
        stem = src.stem.replace("e209_", "")
        dst = C.BASELINE_DIR / f"{C.EXP.lower()}_{stem}_{C.OBJECT_KEY}.tsv"
        with dst.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)

    summary = {
        "scope": {"object_key": C.OBJECT_KEY, "cases": list(C.CASES)},
        "source": {
            str(p.relative_to(REPO)): C.sha256(p) for p in (ROLLOUT_TSV, ZDIFF_TSV)
        },
        "measured": measured,
        "frozen": C.BASELINE_DESK023,
        "prereg_z_linear": C.PREREG_Z_LINEAR,
        "prereg_per_case": C.PREREG_PER_CASE,
        "gates": C.GATES,
        "release_n": {a: counts[f"{a}.release"] for a in C.BASELINE_DESK023},
        "problems": problems,
    }
    out = C.BASELINE_DIR / f"{C.EXP.lower()}_baseline_summary.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    for arm in ("prg", "g1"):
        m = measured[arm]
        print(
            f"  {arm:4s} z_bias={m['z_bias_cm']:+7.4f} z_mae={m['z_mae_cm']:6.4f} "
            f"eef_ori={m['eef_ori_deg']:7.4f} eef_pos={m['eef_pos_cm']:7.4f} "
            f"hand_pen={m['hand_pen']:.4f} contact={m['contact']:.4f} "
            f"narrow={int(m['narrow_pass'])}/{C.EXPECTED_CASES}"
        )
    if problems:
        print("\n".join(f"  FAIL {p}" for p in problems))
        raise SystemExit(f"P0 baseline freeze FAILED ({len(problems)} problems)")
    print(f"\nP0 PASS: both endpoints reproduce plan242 -> {out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
