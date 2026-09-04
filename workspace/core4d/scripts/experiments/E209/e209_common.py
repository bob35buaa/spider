#!/usr/bin/env python3
"""Shared contract for E209 (G1 = E206 PRG + object gravcomp, A0 hand-gate).

E209 reuses E206's 22 delivered desk/chair cases VERBATIM: same omnirt_v1/v2
ref_fk reference trajectory, same 3cm contact mask, same lowgeom proxy, same CEM
budget (1024x32 seed 0), same PRG lower-body triple.  The only change vs E206's
PRG arm is the object body ``gravcomp`` 0 -> 1 (E194's G1 intervention).

Why this is not just E207 again
-------------------------------
E207 concluded gravcomp is a ~constant ``+1.945 +- 0.711 cm`` lift, but that
estimate carries two confounds E207's own data cannot break:

  1. Mass.  Stratified, it is not constant at all: mass=2.0 (n=7) -> +1.691,
     mass=5.0 (n=2) -> +2.836, r(mass, delta) = +0.710.
  2. Mass is collinear with pre-bias.  E207's two mass=5 cases average
     pre-bias -2.08 vs -1.18 for the mass=2 cases, so "shrink toward zero" and
     "additive offset" are structurally indistinguishable there.

All 22 E209 objects are mass=5.000 kg (zero variance) while pre-bias spans
-5.646 .. +2.668, which severs that collinearity.  See PREREG_MODELS.

Nothing is copied here: the case authority, arm/scene/override wiring and the
E163 hand-gate baseline come from E206; the gravcomp assertion comes from E200
(rule 13).  ``e206_common`` is imported read-only -- it is also imported by E208
(``e208_common.py:53``), so mutating it would silently pollute that branch.
"""

from __future__ import annotations

import csv
import hashlib
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[5]
_EXP = REPO / "workspace/core4d/scripts/experiments"
for _d in ("E209", "E206", "E200", "E199", "E198"):
    _p = str(_EXP / _d)
    if _p not in sys.path:
        sys.path.insert(0, _p)

import e206_common as E206  # noqa: E402
from e200_common import assert_gravcomp_diff  # noqa: E402

# --- reused authority (single source of truth) -------------------------------
PROCESSED_ROOT = E206.PROCESSED_ROOT
OVERRIDE_DIR = E206.OVERRIDE_DIR
E163_HAND_GATE = E206.E163_HAND_GATE
CEM_NUM_SAMPLES = E206.CEM_NUM_SAMPLES
CEM_MAX_ITERATIONS = E206.CEM_MAX_ITERATIONS
CEM_SEED = E206.CEM_SEED
BASE_SCENE = E206.SCENE_BY_ARM["prg"]  # scene_act_E206_lowgeom_PRG

# --- E209 arm ----------------------------------------------------------------
EXP = "E209"
ARM_TAG = "G1"
SCENE = "scene_act_E209_lowgeom_PRG_gravcomp"
RESULTS = REPO / "workspace/core4d/results" / EXP
S6_DIR = RESULTS / "s6_downstream"
MANIFEST = S6_DIR / "manifests/e209_g1_full_manifest.tsv"
BASELINE_DIR = RESULTS / "baseline"

#: E206's P10 delivery table -- the authority for which 22 cases E209 covers.
#: Pinned by content hash: if E206 ever re-exports, E209 must be re-reviewed
#: rather than silently following a moved target.
SOURCE_TSV = (
    REPO / "workspace/core4d/results/E206/s6_downstream/rl_export/paired_rl_export_input.tsv"
)
EXPECTED_SOURCE_SHA256 = "af17e209a9cbce3cad7597f0782a60c78b8f3c76ac7656efcb60b2a6cb2e1d5e"

#: The 22 E206 manual-USE desk/chair cases (PRG arm), in sorted case_id order.
CASES: tuple[str, ...] = (
    "chair005_20231030_043_p1",
    "chair006_20231003_1_003_p1",
    "chair006_20231003_1_005_p1",
    "chair006_20231003_2_011_p2",
    "chair006_20231003_2_015_p1",
    "chair006_20231011_076_p2",
    "desk007_20231030_028_p1",
    "desk007_20231030_028_p2",
    "desk007_20231030_030_p2",
    "desk007_20231030_032_p2",
    "desk007_20231030_034_p1",
    "desk021_20231008_005_p1",
    "desk021_20231008_005_p2",
    "desk021_20231008_007_p1",
    "desk021_20231008_007_p2",
    "desk021_20231011_010_p1",
    "desk021_20231011_010_p2",
    "desk021_20231011_014_p2",
    "desk023_20231008_066_p1",
    "desk023_20231008_066_p2",
    "desk023_20231011_005_p1",
    "desk023_20231030_019_p1",
)
EXPECTED_CASES = 22
EXPECTED_OBJECT_COUNTS = {
    "desk021": 7,
    "chair006": 5,
    "desk007": 5,
    "desk023": 4,
    "chair005": 1,
}

#: Every object in scope weighs exactly this (verified per-XML in audit()).
#: This zero-variance mass is what makes E209 able to test the mass-scaling
#: branch of PREREG_MODELS against the shrink branch.
EXPECTED_OBJECT_MASS_KG = 5.000

#: `desk021_20231011_014_p2` is the only omnirt_v2 (rescue) case.  Resolving its
#: task through the v1 naming convention would silently load a different
#: reference trajectory, so task names are always read from the source row.
OMNIRT_V2_CASES: frozenset[str] = frozenset({"desk021_20231011_014_p2"})


# --------------------------------------------------------------------------
# Pre-registered predictions (plan239) -- FROZEN, do not refit after the run
# --------------------------------------------------------------------------
#: Candidate transfer functions post_bias = f(pre_bias), in cm.  Fitted on
#: E207's 9 bucket cases at plan time; E209 only compares their RMSE against
#: the measured post-bias.  Swapping a model in after seeing results would make
#: C4d unfalsifiable, so these coefficients are part of the committed contract.
PREREG_MODELS: dict[str, dict[str, float]] = {
    # OLS on E207 (r=0.705, residual sd=0.377 cm)
    "M_shrink": {"intercept": 1.0939, "slope": 0.3827},
    # E207 pooled mean delta (+1.945 +- 0.711)
    "M_pooled": {"intercept": 1.945, "slope": 1.0},
    # E207 mass=5.0 cell (+2.836 +- 0.550) -- mass-matched to E209
    "M_mass": {"intercept": 2.836, "slope": 1.0},
}
#: Measurement noise floor for the model discrimination (E207 OLS residual sd).
PREREG_RESIDUAL_SD_CM = 0.377


def predict(model: str, pre_bias_cm: float) -> float:
    """Pre-registered post-intervention z bias, cm."""
    m = PREREG_MODELS[model]
    return m["intercept"] + m["slope"] * pre_bias_cm


#: Baseline (E206 PRG) object z bias in cm, measured at plan time with
#: `gen_E178_object_z_diff_report.object_z_series` -- the same function E178 and
#: E207 used.  Frozen here so the S+/S- split is decided by pre-run data, not by
#: looking at the outcome.  P0 re-measures and asserts agreement.
BASELINE_Z_BIAS_CM: dict[str, float] = {
    "chair005_20231030_043_p1": -4.906,
    "chair006_20231003_1_003_p1": +2.668,
    "chair006_20231003_1_005_p1": +2.268,
    "chair006_20231003_2_011_p2": +0.651,
    "chair006_20231003_2_015_p1": -0.726,
    "chair006_20231011_076_p2": +2.386,
    "desk007_20231030_028_p1": -3.653,
    "desk007_20231030_028_p2": -2.736,
    "desk007_20231030_030_p2": -1.512,
    "desk007_20231030_032_p2": -2.557,
    "desk007_20231030_034_p1": -3.461,
    "desk021_20231008_005_p1": -4.780,
    "desk021_20231008_005_p2": -5.646,
    "desk021_20231008_007_p1": -3.405,
    "desk021_20231008_007_p2": -4.053,
    "desk021_20231011_010_p1": -3.905,
    "desk021_20231011_010_p2": -4.272,
    "desk021_20231011_014_p2": -3.133,
    "desk023_20231008_066_p1": -3.774,
    "desk023_20231008_066_p2": -3.908,
    "desk023_20231011_005_p1": -3.274,
    "desk023_20231030_019_p1": -3.638,
}
BASELINE_Z_TOL_CM = 0.01

#: Strata are a mechanical function of BASELINE_Z_BIAS_CM, NOT of object name.
#: `chair006_20231003_2_015_p1` (-0.726) stays in S_MINUS even though its four
#: siblings are the whole of S_PLUS.
S_PLUS: tuple[str, ...] = tuple(c for c in CASES if BASELINE_Z_BIAS_CM[c] >= 0.0)
S_MINUS: tuple[str, ...] = tuple(c for c in CASES if BASELINE_Z_BIAS_CM[c] < 0.0)

#: Frozen plan239 gate thresholds (see the Claims table).
GATES = {
    "C3_narrow_pass_min": 12,
    "C4_s_minus_abs_macro_bias_max_cm": 1.0,
    "C4_s_minus_improved_min": 16,
    "C4_s_minus_overshoot_max_cases": 2,
    "C4_s_minus_overshoot_abs_cm": 1.5,
    "C4b_s_plus_abs_macro_bias_max_cm": 3.99,
    "C4b_s_plus_abs_case_max_cm": 6.0,
    "C4c_all_abs_macro_bias_max_cm": 2.517,
    "C4c_all_mean_abs_bias_max_cm": 3.241,
    "C4d_rmse_win_ratio_min": 2.0,
    "C4d_rmse_max_cm": 1.0,
    "C5_contact_in_mask_min": 0.85,
    "C5_leg_pen_frac_max": 0.02,
    "C6_release_false_contact_max": 0.18,
    "C6_hand_pen_3mm_max": 0.26,
    "C7_use_min": 18,
    "C8_wall_median_min_min": 33.0,
    "C8_wall_median_max_min": 55.0,
}


# --------------------------------------------------------------------------
# Source rows / paths
# --------------------------------------------------------------------------
def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sources() -> list[dict[str, str]]:
    """The 22 E209 source rows, in CASES order, from E206's pinned delivery TSV."""
    actual = sha256(SOURCE_TSV)
    if actual != EXPECTED_SOURCE_SHA256:
        raise ValueError(
            f"E206 delivery TSV changed under E209:\n  expected {EXPECTED_SOURCE_SHA256}"
            f"\n  actual   {actual}\n  {SOURCE_TSV}"
        )
    with SOURCE_TSV.open(encoding="utf-8", newline="") as stream:
        by_case = {r["case_id"]: r for r in csv.DictReader(stream, delimiter="\t")}
    missing = [c for c in CASES if c not in by_case]
    if missing:
        raise KeyError(f"E209 cases absent from the E206 authority: {missing}")
    if len(set(CASES)) != len(CASES):
        raise ValueError("duplicate case_id in E209 CASES")
    if len(by_case) != EXPECTED_CASES:
        raise ValueError(f"E206 delivery has {len(by_case)} rows, expected {EXPECTED_CASES}")
    rows = [by_case[c] for c in CASES]
    dist = dict(Counter(r["object_key"] for r in rows))
    if dist != EXPECTED_OBJECT_COUNTS:
        raise ValueError(f"E209 object drift: expected={EXPECTED_OBJECT_COUNTS} got={dist}")
    return rows


def source_row(case_id: str) -> dict[str, str]:
    return {r["case_id"]: r for r in sources()}[case_id]


def target_task(row: dict[str, str]) -> str:
    """dcv3 task dir name, read from the source row (never reconstructed).

    `stage2b_target_task` carries the omnirt_v1/v2 choice per case; deriving it
    from a naming convention would silently mis-resolve OMNIRT_V2_CASES.
    """
    task = row["stage2b_target_task"]
    if not task:
        raise ValueError(f"{row['case_id']}: empty stage2b_target_task")
    expect_v2 = row["case_id"] in OMNIRT_V2_CASES
    if ("omnirt_v2" in task) != expect_v2:
        raise ValueError(
            f"{row['case_id']}: omnirt variant mismatch -- task={task!r}, "
            f"expected v2={expect_v2} (OMNIRT_V2_CASES)"
        )
    return task


def task_dir(row: dict[str, str]) -> Path:
    return PROCESSED_ROOT / target_task(row)


def base_scene_path(row: dict[str, str]) -> Path:
    """E206 PRG scene -- the baseline arm and the sidecar's parent."""
    return task_dir(row) / f"{BASE_SCENE}.xml"


def scene_path(row: dict[str, str]) -> Path:
    return task_dir(row) / f"{SCENE}.xml"


def override_id(case_id: str) -> str:
    return f"core4d_{EXP}_{case_id}_lowgeom_PRG_{ARM_TAG}"


def override_path(case_id: str) -> Path:
    return OVERRIDE_DIR / f"{override_id(case_id)}.yaml"


def base_override_id(case_id: str) -> str:
    return E206.override_name(case_id, "prg")


def out_dir(case_id: str, stage: str = "full") -> Path:
    """Stage separates the 64x4 smoke from the 1024x32 full run so a small-budget
    rollout can never shadow the full one via skip-already-done."""
    return S6_DIR / "cem" / stage / f"{EXP}_{case_id}_{ARM_TAG}"


def result_npz(case_id: str, stage: str = "full") -> Path:
    return out_dir(case_id, stage) / "trajectory_mjwp_act.npz"


def config_act(case_id: str, stage: str = "full") -> Path:
    return out_dir(case_id, stage) / "config_act.yaml"


def baseline_out_dir(case_id: str) -> Path:
    """E206 PRG rollout dir -- reused as the baseline arm, never re-run."""
    return E206.arm_out_dir("prg", case_id, "full")


def baseline_npz(case_id: str) -> Path:
    return baseline_out_dir(case_id) / "trajectory_mjwp_act.npz"


def kinematic_npz(row: dict[str, str]) -> Path:
    """Reference trajectory (43-qpos freejoint object), shared by both arms."""
    return REPO / row["trajectory"]


# --------------------------------------------------------------------------
# Single-variable gravcomp sidecar
# --------------------------------------------------------------------------
def build_sidecar(row: dict[str, str], *, overwrite: bool = False) -> Path:
    """Write ``scene_act_E209_lowgeom_PRG_gravcomp.xml`` next to the PRG scene.

    Deliberately NOT ``e200_common.build_gravcomp_sidecar``: that one hardcodes
    its output name to ``scene_act_E199_rubberHull_PRG_gravcomp`` (e200_common.py
    :161), so reusing it here would drop an E199-named file into desk/chair task
    dirs.  The assertion is reused verbatim; only the 12-line writer is local.

    Single-variable diff: the `object` body gravcomp 0/absent -> 1, nothing else.
    Idempotent: an existing sidecar is re-verified (not rewritten) unless overwrite.
    """
    base = base_scene_path(row)
    if not base.is_file():
        raise FileNotFoundError(f"{row['case_id']}: missing E206 PRG scene: {base}")
    out = scene_path(row)
    if out.is_file() and not overwrite:
        assert_gravcomp_diff(base, out)
        return out
    tree = ET.parse(base)
    objs = [b for b in tree.getroot().iter("body") if b.get("name") == "object"]
    if len(objs) != 1:
        raise ValueError(f"expected exactly one object body in {base}, found {len(objs)}")
    if objs[0].get("gravcomp") not in (None, "0", "0.0"):
        raise ValueError(f"object already has gravcomp in {base}: {objs[0].get('gravcomp')}")
    objs[0].set("gravcomp", "1")
    ET.indent(tree, space="  ")
    tree.write(out, encoding="utf-8", xml_declaration=True)
    assert_gravcomp_diff(base, out)
    return out


def object_mass(scene: Path) -> float:
    root = ET.parse(scene).getroot()
    obj = next(b for b in root.iter("body") if b.get("name") == "object")
    inertial = obj.find("inertial")
    if inertial is None or inertial.get("mass") is None:
        raise ValueError(f"no object inertial mass in {scene}")
    return float(inertial.get("mass"))


# --------------------------------------------------------------------------
# Contract audit
# --------------------------------------------------------------------------
def audit(verbose: bool = True) -> int:
    """Assert the E209 scene contract; fail loudly on the first drift."""
    rows = sources()
    for row in rows:
        case_id = row["case_id"]
        base, side = base_scene_path(row), scene_path(row)
        if not base.is_file():
            raise FileNotFoundError(f"{case_id}: missing E206 PRG scene: {base}")
        if not side.is_file():
            raise FileNotFoundError(f"{case_id}: missing gravcomp sidecar (run P1): {side}")
        assert_gravcomp_diff(base, side)
        mass = object_mass(base)
        if abs(mass - EXPECTED_OBJECT_MASS_KG) > 1e-6:
            raise ValueError(f"{case_id}: object mass {mass} != {EXPECTED_OBJECT_MASS_KG}")
        if verbose:
            print(f"  {case_id:36s} PASS  {SCENE} == E206 PRG + gravcomp=1  (m={mass:.3f} kg)")
    if verbose:
        print(
            f"audit PASS: {len(rows)}/{EXPECTED_CASES} single-variable gravcomp scenes "
            f"({EXPECTED_OBJECT_COUNTS}); S+={len(S_PLUS)} S-={len(S_MINUS)}"
        )
    return len(rows)


if __name__ == "__main__":
    raise SystemExit(0 if audit() == EXPECTED_CASES else 1)
