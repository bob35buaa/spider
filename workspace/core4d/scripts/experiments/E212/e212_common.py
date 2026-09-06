#!/usr/bin/env python3
"""Shared contract for E212 Stage A (desk023 partial-gravcomp sweep).

E211 swept ``gravcomp in {0.4, 0.6, 0.8}`` on the 5 ``desk007`` cases and failed
its main gate on all five g values, but produced a mechanism result worth
extrapolating: the cost of unloading the object comes in two forms with two
different dose-response curves -- contact collapse is load-driven
(``r(g, contact) = -0.408``) while posture collapse is driven by the CEM safety
gate falling back to ``least_violation`` (``r(fallback, eef_ori) = +0.495``,
partial correlation +0.504 after controlling for g), and ``r(g, fallback)`` is
only +0.016, i.e. the two are near-orthogonal.

E212 moves the same single variable onto ``desk023``.  This is NOT "the same
experiment on more cases" -- desk023 sits in a different regime:

    metric              desk007 (E211)        desk023 (E212)
    -----------------   -------------------   ----------------------
    n                   5                     4
    PRG narrow          2/5 (weak baseline)   4/4 (perfect baseline)
    G1 narrow           0/5                   1/4
    z_bias PRG->G1      -2.784 -> +1.116      -3.648 -> +0.353
    per-case g* median  0.710                 0.928
    contact PRG->G1     .880 -> .764          .910 -> .921  (improves)
    gates broken at G1  eef_ori+contact+      eef_ori (3/4) +
                        release               eef_pos/hand_pen (1/4)
    fallback subsystem  body / hand gate      leg gate

Two consequences drive the design (plan242):

  * desk023 has NO contact-collapse form.  E210 F2's "two mutually exclusive
    cost forms" degenerates to one here.
  * The object side is already near-optimal at g=1 (|bias| 0.413 cm, per-case
    g* median 0.928 vs desk007's 0.710), so lowering g can only *cost* z.  E212
    is therefore an explicit "pay z to buy back eef_ori" trade-off curve, not
    E211's "rescue both sides".  The main gate C1 is built accordingly.

Per-case, the fallback mechanism explains 3 of 4 cases and has one clean
counterexample (see ``PREREG_PER_CASE`` and prediction P5 in plan242 section 2):
066_p2 has ``cem_gate_fallback_used == 0`` at both endpoints and is the only
case that survives g=1; 019_p1 has the largest fallback rise and the worst
damage; but 066_p1's fallback does not move at all (0.007 -> 0.007) while its
eef_ori still degrades 4.5 deg and hand_pen 4x.

Read-only reuse
---------------
``e211_common`` is NOT imported: E212 must not be able to perturb an experiment
whose log is already written.  ``e209_common`` (and through it ``e206_common``)
is imported READ-ONLY as the authority for the case list, the pinned E206
delivery TSV, task resolution, the A0 hand-gate and the CEM budget.  The
value-parameterised gravcomp assertion is re-stated here rather than imported
from E211 for the same reason, but it still reuses ``e200_common._signature`` so
the two can never drift apart on what counts as "the same XML".
"""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

REPO = Path(__file__).resolve().parents[5]
_EXP = REPO / "workspace/core4d/scripts/experiments"
for _d in ("E212", "E209", "E206", "E200", "E199"):
    _p = str(_EXP / _d)
    if _p not in sys.path:
        sys.path.insert(0, _p)

import e209_common as E209  # noqa: E402
from e200_common import _signature  # noqa: E402

# --- reused authority (single source of truth, never re-typed) ---------------
PROCESSED_ROOT = E209.PROCESSED_ROOT
OVERRIDE_DIR = E209.OVERRIDE_DIR
E163_HAND_GATE = E209.E163_HAND_GATE
CEM_NUM_SAMPLES = E209.CEM_NUM_SAMPLES
CEM_MAX_ITERATIONS = E209.CEM_MAX_ITERATIONS
CEM_SEED = E209.CEM_SEED
BASE_SCENE = E209.BASE_SCENE  # scene_act_E206_lowgeom_PRG
G1_SCENE = E209.SCENE  # scene_act_E209_lowgeom_PRG_gravcomp
EXPECTED_OBJECT_MASS_KG = E209.EXPECTED_OBJECT_MASS_KG
sha256 = E209.sha256
object_mass = E209.object_mass

# --- E212 scope --------------------------------------------------------------
EXP = "E212"
OBJECT_KEY = "desk023"
RESULTS = REPO / "workspace/core4d/results" / EXP
S6_DIR = RESULTS / "s6_downstream"
MANIFEST_DIR = S6_DIR / "manifests"
BASELINE_DIR = RESULTS / "baseline"

#: The 4 desk023 cases, derived from E209's authority rather than re-typed.
CASES: tuple[str, ...] = tuple(c for c in E209.CASES if c.startswith(f"{OBJECT_KEY}_"))
EXPECTED_CASES = 4

#: Stage A arms: tag -> gravcomp multiplier on the object body.  Identical grid
#: to E211 (user-confirmed) so the two object families compare cell-by-cell.
#: 0.0 (E206 PRG) and 1.0 (E209 G1) are the frozen endpoints and are NOT re-run.
ARMS: dict[str, float] = {"G04": 0.4, "G06": 0.6, "G08": 0.8}
ARM_ORDER: tuple[str, ...] = tuple(sorted(ARMS))
EXPECTED_ROWS = EXPECTED_CASES * len(ARMS)  # 12


#: gravcomp is written with this exact formatting so the XML text, the manifest
#: and the runtime read-back all compare as the same string.
def gravcomp_str(arm: str) -> str:
    return f"{ARMS[arm]:.1f}"


def scene_name(arm: str) -> str:
    """e.g. G06 -> scene_act_E212_lowgeom_PRG_gc06."""
    return f"scene_act_{EXP}_lowgeom_PRG_gc{arm[1:]}"


SCENE_BY_ARM: dict[str, str] = {a: scene_name(a) for a in ARMS}

# --------------------------------------------------------------------------
# Frozen baselines (plan242 section 3) -- measured from E209's delivered TSVs
# at plan time, over these 4 cases only.  Frozen here so the success criteria
# cannot drift after seeing Stage A results.  P0 (freeze_baseline.py)
# re-measures and asserts every number below.
# --------------------------------------------------------------------------
#: desk023 macro means.  Unlike desk007, ALL FOUR cases have a finite `release`
#: (no empty release window), so `release` is a true n=4 mean -- see
#: BASELINE_RELEASE_N.
BASELINE_DESK023: dict[str, dict[str, float]] = {
    "prg": {
        "z_bias_cm": -3.6481, "z_abs_bias_cm": 3.6481, "z_mae_cm": 4.6158,
        "obj_pos_cm": 10.3506, "obj_ori_deg": 5.2336,
        "eef_ori_deg": 16.9583, "eef_pos_cm": 13.4131,
        "root_ori_deg": 8.0138, "root_pos_cm": 14.5716,
        "contact": 0.9104, "release": 0.0421, "hand_pen": 0.1651,
        "body_z_p95_m": 0.0809, "leg_pen": 0.0088, "ankle_jerk_p95": 598.0334,
        "narrow_pass": 4.0, "hard_pass": 4.0,
    },
    "g1": {
        "z_bias_cm": 0.3532, "z_abs_bias_cm": 0.4133, "z_mae_cm": 3.1752,
        "obj_pos_cm": 9.9442, "obj_ori_deg": 4.5995,
        "eef_ori_deg": 21.2777, "eef_pos_cm": 15.7323,
        "root_ori_deg": 9.4703, "root_pos_cm": 16.0636,
        "contact": 0.9211, "release": 0.0542, "hand_pen": 0.2152,
        "body_z_p95_m": 0.0874, "leg_pen": 0.0095, "ankle_jerk_p95": 720.3441,
        "narrow_pass": 1.0, "hard_pass": 4.0,
    },
}
BASELINE_TOL = {"narrow_pass": 0.0, "hard_pass": 0.0, "contact": 5e-4,
                "release": 5e-4, "hand_pen": 5e-4, "leg_pen": 5e-4,
                "body_z_p95_m": 5e-4}
BASELINE_TOL_DEFAULT = 5e-3
BASELINE_RELEASE_N = 4

#: Per-case E206-PRG z bias (cm), the measured E209 g=1 correction (cm), and the
#: per-case fallback rates that motivate P3/P4/P5.  ``g_star`` is a PREDICTION
#: used only to falsify P1; plan242 forbids using it as a per-case tuning knob
#: (object-level single g was the chosen granularity).
PREREG_PER_CASE: dict[str, dict[str, float]] = {
    "desk023_20231008_066_p1": {
        "pre_bias_cm": -3.774, "delta_g1_cm": 3.653, "g_star": 1.033,
        "fallback_prg": 0.007, "fallback_g1": 0.007,  # flat -> P5 counterexample
    },
    "desk023_20231008_066_p2": {
        "pre_bias_cm": -3.908, "delta_g1_cm": 4.176, "g_star": 0.936,
        "fallback_prg": 0.000, "fallback_g1": 0.000,  # zero -> P3 survivor
    },
    "desk023_20231011_005_p1": {
        "pre_bias_cm": -3.274, "delta_g1_cm": 3.554, "g_star": 0.921,
        "fallback_prg": 0.148, "fallback_g1": 0.207,
    },
    "desk023_20231030_019_p1": {
        "pre_bias_cm": -3.638, "delta_g1_cm": 4.622, "g_star": 0.787,
        "fallback_prg": 0.048, "fallback_g1": 0.105,  # worst case
    },
}

#: P1: z_bias(g) is linear in g with these coefficients (macro over 4 cases).
#: Intercept and slope are fixed analytically by the two frozen endpoints, so
#: R^2 measures ONLY how far the three new arms fall off that line.
PREREG_Z_LINEAR = {"intercept_cm": -3.6481, "slope_cm_per_g": 4.0013, "min_r2": 0.90}

#: P4: partial correlation r(fallback, eef_ori | g) must exceed this for E211's
#: fallback mechanism to be considered to extrapolate to desk023.
PREREG_P4_MIN_PARTIAL_R = 0.40

#: The case whose damage cannot be explained by gate fallback (flat fallback,
#: yet eef_ori +4.5 deg and hand_pen 4x at g=1).  P5 tests whether its eef_ori
#: is instead monotone in g, i.e. load-driven.
PREREG_P5_CASE = "desk023_20231008_066_p1"
#: The case with zero fallback at both endpoints and the only g=1 survivor.
PREREG_P3_CASE = "desk023_20231008_066_p2"
#: The worst case; plan242 section 3 requires macro means reported both with and
#: without it (n=4 means one case carries 25% of the weight).
OUTLIER_CASE = "desk023_20231030_019_p1"

#: plan242 section 3 main gate C1.  Frozen.  NOTE the clause layout differs from
#: E211: desk023 breaks on eef_pos/hand_pen and never on contact/release, so C1d
#: is (eef_pos AND hand_pen) and C1e is contact -- E211 had C1d=contact,
#: C1e=release.  The eval runner's ``c1_for()`` must be written against THIS
#: layout, not E211's; swapping only the dict would judge SUCCESS on wrong gates.
GATES = {
    "C1a_abs_z_bias_max_cm": 1.50,
    "C1a_z_mae_max_cm": 3.1752,     # G1 value -- the tightest z clause
    "C1b_obj_pos_max_cm": 10.3506,  # PRG
    "C1b_obj_ori_max_deg": 5.2336,  # PRG
    "C1c_eef_ori_max_deg": 17.958,  # PRG + 1.0
    "C1d_eef_pos_max_cm": 14.413,   # PRG + 1.0
    "C1d_hand_pen_max": 0.1951,     # PRG + 0.03
    "C1e_contact_min": 0.8804,      # PRG - 0.030
    "C1f_narrow_pass_min": 3,       # PRG is 4/4; allow losing one
    "C2_root_ori_max_deg": 9.014,   # PRG + 1.0
    "C2_root_pos_max_cm": 15.572,   # PRG + 1.0
    "C3_release_max": 0.0921,       # PRG + 0.05
    "C3_body_z_p95_max_m": 0.20,    # hard
    "C3_leg_pen_max": 0.20,         # hard
    "C3_ankle_jerk_max": 1000.0,    # hard
    "C4_wall_median_min_min": 33.0,
    "C4_wall_median_max_min": 55.0,
}


# --------------------------------------------------------------------------
# Source rows / paths
# --------------------------------------------------------------------------
def sources() -> list[dict[str, str]]:
    """The 4 desk023 source rows, in CASES order, from E206's pinned delivery TSV.

    Goes through ``E209.sources()`` so the delivery-TSV sha256 pin, the 22-row
    count and the object distribution are all re-checked here too: E212 must not
    be able to run against a drifted authority just because it looks at a subset.
    """
    by_case = {r["case_id"]: r for r in E209.sources()}
    rows = [by_case[c] for c in CASES]
    if len(rows) != EXPECTED_CASES:
        raise ValueError(f"E212 expected {EXPECTED_CASES} {OBJECT_KEY} cases, got {len(rows)}")
    bad = [r["case_id"] for r in rows if r["object_key"] != OBJECT_KEY]
    if bad:
        raise ValueError(f"non-{OBJECT_KEY} case in E212 scope: {bad}")
    return rows


def source_row(case_id: str) -> dict[str, str]:
    return {r["case_id"]: r for r in sources()}[case_id]


target_task = E209.target_task
task_dir = E209.task_dir
base_scene_path = E209.base_scene_path
kinematic_npz = E209.kinematic_npz


def scene_path(row: dict[str, str], arm: str) -> Path:
    return task_dir(row) / f"{SCENE_BY_ARM[arm]}.xml"


def override_id(case_id: str, arm: str) -> str:
    return f"core4d_{EXP}_{case_id}_lowgeom_PRG_{arm}"


def override_path(case_id: str, arm: str) -> Path:
    return OVERRIDE_DIR / f"{override_id(case_id, arm)}.yaml"


base_override_id = E209.base_override_id


def out_dir(case_id: str, arm: str, stage: str = "full") -> Path:
    """Stage separates the 64x4 smoke from the 1024x32 full run so a small-budget
    rollout can never shadow the full one via skip-already-done."""
    return S6_DIR / "cem" / stage / f"{EXP}_{case_id}_{arm}"


def result_npz(case_id: str, arm: str, stage: str = "full") -> Path:
    return out_dir(case_id, arm, stage) / "trajectory_mjwp_act.npz"


def config_act(case_id: str, arm: str, stage: str = "full") -> Path:
    return out_dir(case_id, arm, stage) / "config_act.yaml"


def prg_out_dir(case_id: str) -> Path:
    """g=0 endpoint: the E206 PRG rollout.  Frozen, never re-run."""
    return E209.baseline_out_dir(case_id)


def g1_out_dir(case_id: str) -> Path:
    """g=1 endpoint: the E209 G1 rollout.  Frozen, never re-run."""
    return E209.out_dir(case_id, "full")


def manifest_path(stage: str = "full", shard: str = "") -> Path:
    suffix = f".shard{shard}" if shard else ""
    return MANIFEST_DIR / f"{EXP.lower()}_stageA_{stage}_manifest{suffix}.tsv"


# --------------------------------------------------------------------------
# Two-machine shard split (plan242 section 4-2)
# --------------------------------------------------------------------------
#: Rows per shard.  The user runs 8 locally and 4 on the second 8-GPU box.
EXPECTED_SHARD_ROWS = {"A": 8, "B": 4}
SHARDS = tuple(sorted(EXPECTED_SHARD_ROWS))


def shard_of(case_id: str, arm: str) -> str:
    """Latin-square diagonal: (case_idx + arm_idx) % 3 == 2 -> shard B.

    E211 used ``SHARDS[index % 2]`` over the sorted (case, arm) list, which for
    12 rows yields 6/6 -- the user asked for 8/4.  The naive alternative
    ``index % 3 == 2`` gives 8/4 but puts the ENTIRE G08 arm on the remote
    machine, and G08 is the predicted-best arm (|z_bias| 0.447 cm); losing that
    box would lose a whole g value and leave an unreadable curve.  The diagonal
    keeps both shards spanning all 4 cases and all 3 arms, so either shard
    surviving alone is still an interpretable cross-section.
    """
    return "B" if (CASES.index(case_id) + ARM_ORDER.index(arm)) % 3 == 2 else "A"


# --------------------------------------------------------------------------
# Value-parameterised gravcomp sidecar
# --------------------------------------------------------------------------
def assert_gravcomp_diff_value(base: Path, sidecar: Path, value: str) -> None:
    """Fail unless sidecar == base with only object body gravcomp absent/0 -> value.

    Same contract and same recursive element signature as
    ``e200_common.assert_gravcomp_diff``, but the target value is a parameter.
    That function hardcodes ``"1"`` (e200_common.py:146), so it rejects every
    partial-compensation sidecar; it is imported by E198/E200/E209/E210 and must
    not be edited.  ``_signature`` is reused verbatim rather than re-implemented,
    so the two assertions can never drift apart in what they consider "the same
    XML".
    """
    base, sidecar = Path(base), Path(sidecar)
    base_root = ET.parse(base).getroot()
    objs = [b for b in base_root.iter("body") if b.get("name") == "object"]
    if len(objs) != 1 or objs[0].get("gravcomp") not in (None, "0", "0.0"):
        raise ValueError(f"unexpected base object gravcomp in {base}")
    expected = ET.parse(base).getroot()
    next(b for b in expected.iter("body") if b.get("name") == "object").set("gravcomp", value)
    side_root = ET.parse(sidecar).getroot()
    if _signature(side_root) != _signature(expected):
        raise AssertionError(
            f"sidecar is not a single-variable gravcomp={value} diff: {sidecar}"
        )


def build_sidecar(row: dict[str, str], arm: str, *, overwrite: bool = False) -> Path:
    """Write ``scene_act_E212_lowgeom_PRG_gc<NN>.xml`` next to the E206 PRG scene.

    Single-variable diff: the `object` body gravcomp absent/0 -> ARMS[arm].
    Idempotent: an existing sidecar is re-verified (not rewritten) unless overwrite.
    """
    base = base_scene_path(row)
    if not base.is_file():
        raise FileNotFoundError(f"{row['case_id']}: missing E206 PRG scene: {base}")
    value = gravcomp_str(arm)
    out = scene_path(row, arm)
    if out.is_file() and not overwrite:
        assert_gravcomp_diff_value(base, out, value)
        return out
    tree = ET.parse(base)
    objs = [b for b in tree.getroot().iter("body") if b.get("name") == "object"]
    if len(objs) != 1:
        raise ValueError(f"expected exactly one object body in {base}, found {len(objs)}")
    if objs[0].get("gravcomp") not in (None, "0", "0.0"):
        raise ValueError(f"object already has gravcomp in {base}: {objs[0].get('gravcomp')}")
    objs[0].set("gravcomp", value)
    ET.indent(tree, space="  ")
    tree.write(out, encoding="utf-8", xml_declaration=True)
    assert_gravcomp_diff_value(base, out, value)
    return out


def scene_gravcomp(scene: Path) -> str | None:
    """Read back the object body's gravcomp attribute from a compiled-source XML."""
    root = ET.parse(scene).getroot()
    obj = next(b for b in root.iter("body") if b.get("name") == "object")
    return obj.get("gravcomp")


# --------------------------------------------------------------------------
# Contract audit
# --------------------------------------------------------------------------
def audit(verbose: bool = True) -> int:
    """Assert the E212 scene contract; fail loudly on the first drift."""
    rows = sources()
    checked = 0
    for row in rows:
        case_id = row["case_id"]
        base = base_scene_path(row)
        if not base.is_file():
            raise FileNotFoundError(f"{case_id}: missing E206 PRG scene: {base}")
        mass = object_mass(base)
        if abs(mass - EXPECTED_OBJECT_MASS_KG) > 1e-6:
            raise ValueError(f"{case_id}: object mass {mass} != {EXPECTED_OBJECT_MASS_KG}")
        for arm in ARM_ORDER:
            side = scene_path(row, arm)
            if not side.is_file():
                raise FileNotFoundError(f"{case_id}/{arm}: missing sidecar (run P2): {side}")
            assert_gravcomp_diff_value(base, side, gravcomp_str(arm))
            got = scene_gravcomp(side)
            if got != gravcomp_str(arm):
                raise ValueError(f"{case_id}/{arm}: gravcomp={got!r} != {gravcomp_str(arm)!r}")
            checked += 1
        if verbose:
            arms = " ".join(f"{a}={gravcomp_str(a)}" for a in ARM_ORDER)
            print(f"  {case_id:32s} PASS  {arms}  (m={mass:.3f} kg)")
    if verbose:
        print(f"audit PASS: {checked}/{EXPECTED_ROWS} single-variable partial-gravcomp scenes")
    return checked


if __name__ == "__main__":
    raise SystemExit(0 if audit() == EXPECTED_ROWS else 1)
