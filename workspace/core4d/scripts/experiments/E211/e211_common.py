#!/usr/bin/env python3
"""Shared contract for E211 Stage A (desk007 partial-gravcomp sweep).

E209 put ``gravcomp=1`` on all 22 E206-PRG desk/chair cases.  The object side
improved everywhere, but the 14-gate main claim broke -- and splitting by object
shows the damage is almost entirely the 5 ``desk007`` cases (narrow 2/5 -> 0/5,
eef_ori 14.67 -> 25.62 deg, contact .880 -> .764, release .179 -> .307), while
``chair006`` improved on every single metric (narrow 3/5 -> 5/5).

Why a *partial* compensation is the right single variable
---------------------------------------------------------
The object is not a free body: ``scene_act`` replaces its freejoint with 3 slide
+ 3 hinge joints driven by position actuators (``init_pos_actuator_gain=500``).
The -z bias is servo droop under the share of the 49 N weight the hands do not
carry, and ``gravcomp`` cancels exactly that gravity term.  MuJoCo treats
``gravcomp`` as a float multiplier, not a flag --
``mujoco_warp/_src/passive.py:264-272`` computes
``force = -gravity * body_mass * gravcomp`` -- so any value in [0, 1] is honoured
by the runtime with no simulator change.

Two independent estimates say full compensation overshoots on desk007:

  * CORE4D is a two-person co-carry and only ONE robot is compiled into the
    scene (31 bodies: one G1 + the object).  The object COM sits 0.32-0.44 m
    horizontally outside the single robot's grasp line, so the position
    actuators are already standing in for the absent partner.  ``gravcomp=1``
    makes the servo carry 100% when the partner's share should be ~50-70%.
  * Per-case ``g* = -pre_bias / delta(g=1)`` over the 5 desk007 cases is
    0.603 / 0.969 / 0.738 / 0.710 / 0.697 -- median 0.710, matching E209's
    global shrink-model extrapolation of ~0.71.

Stage A therefore sweeps ``gravcomp in {0.4, 0.6, 0.8}`` and joins the two
frozen endpoints (E206 PRG = 0.0, E209 G1 = 1.0) into a 5-point curve.

Read-only reuse
---------------
``e209_common`` (and through it ``e206_common``) is imported READ-ONLY: it is
the authority for the case list, the pinned E206 delivery TSV, task resolution,
the A0 hand-gate and the CEM budget.  Mutating either would silently pollute
E208/E209/E210, which import the same modules.  Only the value-parameterised
gravcomp assertion is new here, because ``e200_common.assert_gravcomp_diff``
hardcodes the target value ``"1"`` (e200_common.py:146) and would reject a 0.6
sidecar as "not a single-variable gravcomp diff".
"""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

REPO = Path(__file__).resolve().parents[5]
_EXP = REPO / "workspace/core4d/scripts/experiments"
for _d in ("E211", "E209", "E206", "E200", "E199"):
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

# --- E211 scope --------------------------------------------------------------
EXP = "E211"
OBJECT_KEY = "desk007"
RESULTS = REPO / "workspace/core4d/results" / EXP
S6_DIR = RESULTS / "s6_downstream"
MANIFEST_DIR = S6_DIR / "manifests"
BASELINE_DIR = RESULTS / "baseline"

#: The 5 desk007 cases, derived from E209's authority rather than re-typed.
CASES: tuple[str, ...] = tuple(c for c in E209.CASES if c.startswith(f"{OBJECT_KEY}_"))
EXPECTED_CASES = 5

#: Stage A arms: tag -> gravcomp multiplier on the object body.
#: 0.0 (E206 PRG) and 1.0 (E209 G1) are the frozen endpoints and are NOT re-run.
ARMS: dict[str, float] = {"G04": 0.4, "G06": 0.6, "G08": 0.8}
ARM_ORDER: tuple[str, ...] = tuple(sorted(ARMS))
EXPECTED_ROWS = EXPECTED_CASES * len(ARMS)  # 15

#: gravcomp is written with this exact formatting so the XML text, the manifest
#: and the runtime read-back all compare as the same string.
def gravcomp_str(arm: str) -> str:
    return f"{ARMS[arm]:.1f}"


def scene_name(arm: str) -> str:
    """e.g. G06 -> scene_act_E211_lowgeom_PRG_gc06."""
    return f"scene_act_{EXP}_lowgeom_PRG_gc{arm[1:]}"


SCENE_BY_ARM: dict[str, str] = {a: scene_name(a) for a in ARMS}

# --------------------------------------------------------------------------
# Frozen baselines (plan241 section 3) -- measured from E209's delivered TSVs
# at plan time, over these 5 cases only.  Frozen here so the success criteria
# cannot drift after seeing Stage A results.  P0 re-measures and asserts.
# --------------------------------------------------------------------------
#: desk007 macro means. `release` is over n=4 (028_p2 has an empty release
#: window, so the metric is NaN there and stays NaN for every arm).
BASELINE_DESK007: dict[str, dict[str, float]] = {
    "prg": {
        "z_bias_cm": -2.7839, "z_abs_bias_cm": 2.7839, "z_mae_cm": 3.5947,
        "obj_pos_cm": 9.0780, "obj_ori_deg": 5.3104,
        "eef_ori_deg": 14.6707, "eef_pos_cm": 12.6093,
        "root_ori_deg": 6.9842, "root_pos_cm": 13.6415,
        "contact": 0.8799, "release": 0.1793, "hand_pen": 0.2540,
        "body_z_p95_m": 0.0560, "leg_pen": 0.0318, "ankle_jerk_p95": 542.4886,
        "narrow_pass": 2.0,
    },
    "g1": {
        "z_bias_cm": 1.1161, "z_abs_bias_cm": 1.1161, "z_mae_cm": 2.0990,
        "obj_pos_cm": 8.1202, "obj_ori_deg": 4.8278,
        "eef_ori_deg": 25.6215, "eef_pos_cm": 14.5040,
        "root_ori_deg": 14.7919, "root_pos_cm": 17.5438,
        "contact": 0.7644, "release": 0.3065, "hand_pen": 0.2147,
        "body_z_p95_m": 0.0800, "leg_pen": 0.0315, "ankle_jerk_p95": 514.9700,
        "narrow_pass": 0.0,
    },
}
BASELINE_TOL = {"narrow_pass": 0.0, "contact": 5e-4, "release": 5e-4,
                "hand_pen": 5e-4, "leg_pen": 5e-4, "body_z_p95_m": 5e-4}
BASELINE_TOL_DEFAULT = 5e-3
BASELINE_RELEASE_N = 4

#: Per-case E206-PRG z bias (cm) and the measured E209 g=1 correction (cm).
#: `g_star` is a PREDICTION used only to falsify A-P1; plan241 forbids using it
#: as a per-case tuning knob (object-level single g was the chosen granularity).
PREREG_PER_CASE: dict[str, dict[str, float]] = {
    "desk007_20231030_028_p1": {"pre_bias_cm": -3.653, "delta_g1_cm": 6.059, "g_star": 0.603},
    "desk007_20231030_028_p2": {"pre_bias_cm": -2.736, "delta_g1_cm": 2.823, "g_star": 0.969},
    "desk007_20231030_030_p2": {"pre_bias_cm": -1.512, "delta_g1_cm": 2.049, "g_star": 0.738},
    "desk007_20231030_032_p2": {"pre_bias_cm": -2.557, "delta_g1_cm": 3.600, "g_star": 0.710},
    "desk007_20231030_034_p1": {"pre_bias_cm": -3.461, "delta_g1_cm": 4.969, "g_star": 0.697},
}

#: A-P1: z_bias(g) is linear in g with these coefficients (macro over 5 cases).
PREREG_Z_LINEAR = {"intercept_cm": -2.7839, "slope_cm_per_g": 3.9000, "min_r2": 0.90}

#: plan241 section 3 main gate C1.  Frozen; the eval runner reads them from here.
GATES = {
    "C1a_abs_z_bias_max_cm": 1.50,
    "C1a_z_mae_max_cm": 2.80,
    "C1b_obj_pos_max_cm": 9.078,
    "C1b_obj_ori_max_deg": 5.310,
    "C1c_eef_ori_max_deg": 15.67,   # PRG + 1.0
    "C1d_contact_min": 0.850,       # PRG - 0.030
    "C1e_release_max": 0.229,       # PRG + 0.050
    "C1f_narrow_pass_min": 2,
    "C2_root_ori_max_deg": 7.98,
    "C2_root_pos_max_cm": 14.64,
    "C2_eef_pos_max_cm": 13.61,
    "C3_hand_pen_max": 0.284,
    "C4_wall_median_min_min": 33.0,
    "C4_wall_median_max_min": 55.0,
}


# --------------------------------------------------------------------------
# Source rows / paths
# --------------------------------------------------------------------------
def sources() -> list[dict[str, str]]:
    """The 5 desk007 source rows, in CASES order, from E206's pinned delivery TSV.

    Goes through ``E209.sources()`` so the delivery-TSV sha256 pin, the 22-row
    count and the object distribution are all re-checked here too: E211 must not
    be able to run against a drifted authority just because it looks at a subset.
    """
    by_case = {r["case_id"]: r for r in E209.sources()}
    rows = [by_case[c] for c in CASES]
    if len(rows) != EXPECTED_CASES:
        raise ValueError(f"E211 expected {EXPECTED_CASES} desk007 cases, got {len(rows)}")
    bad = [r["case_id"] for r in rows if r["object_key"] != OBJECT_KEY]
    if bad:
        raise ValueError(f"non-{OBJECT_KEY} case in E211 scope: {bad}")
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
    return MANIFEST_DIR / f"e211_stageA_{stage}_manifest{suffix}.tsv"


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
    """Write ``scene_act_E211_lowgeom_PRG_gc<NN>.xml`` next to the E206 PRG scene.

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
    """Assert the E211 scene contract; fail loudly on the first drift."""
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
