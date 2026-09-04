#!/usr/bin/env python3
"""Shared contract for E207 (G1only = E178 PRG + object gravcomp, A0 hand-gate).

E207 reuses E178's bucket cases VERBATIM: same omnirt_v1 ref_fk reference
trajectory, same 3cm contact mask, same CEM budget (1024x32 seed 0), same
contact-aligned 5-segment object proxy. The only change vs E178 is the object
body ``gravcomp`` 0 -> 1 (E194's G1 arm).

Why this arm did not exist yet: E205 ran G1A2, which moves *two* axes off E178
(gravcomp AND the A2 hand-gate retune). E207 fills the A0+gravcomp cell, giving
two clean single-variable paths:

    E178 -> E207   isolates gravcomp (G1)
    E207 -> E205   isolates the hand-gate (A2), conditional on gravcomp=on

Scene reuse: E207 does NOT emit its own sidecar. ``scene_act_E205_..._gravcomp``
is byte-verified to be exactly ``scene_act_E178_contactAlignedTop`` + object
``gravcomp=1`` (see :func:`audit`), so reusing it also makes the E207-vs-E205
scene identical -- the hand-gate contrast is then single-variable by construction.

Nothing is copied here: the 27-case authority, task resolution and E163 hand-gate
baseline come from E204_E205; the gravcomp assertion comes from E200 (rule 13).
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[5]
_EXP = REPO / "workspace/core4d/scripts/experiments"
for _d in ("E207", "E204_E205", "E178", "E177", "E176", "E175", "E200", "E198"):
    _p = str(_EXP / _d)
    if _p not in sys.path:
        sys.path.insert(0, _p)

import e204e205_common as E205C  # noqa: E402
from e200_common import assert_gravcomp_diff  # noqa: E402

# --- reused authority (single source of truth) -------------------------------
load_sources = E205C.load_sources
task_of = E205C.task_of
task_dir = E205C.task_dir
e178_scene_path = E205C.e178_scene_path
e178_override_id = E205C.e178_override_id
E163_HAND_GATE = E205C.E163_HAND_GATE
OVERRIDE_DIR = E205C.OVERRIDE_DIR
CEM_SEED = E205C.CEM_SEED
FULL_SAMPLES = E205C.FULL_SAMPLES
FULL_OPT_STEPS = E205C.FULL_OPT_STEPS

# --- E207 arm ----------------------------------------------------------------
EXP = "E207"
ARM_TAG = "G1only"
SCENE = E205C.E205_SCENE  # reused; equality with E178+gravcomp asserted in audit()
RESULTS = REPO / "workspace/core4d/results/E207"
MANIFEST = RESULTS / "s6_downstream/manifests/g1only_full_manifest.tsv"

#: The 9 bucket cases for E207 (user-selected subset of E178's 27).
#: bucket003=3, bucket007=6, bucket004=0 -- conclusions do NOT extrapolate to
#: bucket004, which is absent and is the highest-lift / worst-sag object.
CASES: tuple[str, ...] = (
    "bucket003_20231018_001_p2",
    "bucket003_20231018_005_p2",
    "bucket003_20231020_068_p1",
    "bucket007_20231003_2_021_p1",
    "bucket007_20231003_2_021_p2",
    "bucket007_20231020_059_p1",
    "bucket007_20231023_073_p1",
    "bucket007_20231023_075_p1",
    "bucket007_20231023_075_p2",
)
EXPECTED_OBJECT_COUNTS = {"bucket003": 3, "bucket007": 6}


def sources() -> list[dict]:
    """The 9 E207 source rows, in CASES order, filtered from E178's 27."""
    by_case = {s["case_id"]: s for s in load_sources()}
    missing = [c for c in CASES if c not in by_case]
    if missing:
        raise KeyError(f"E207 cases absent from the E178 authority: {missing}")
    if len(set(CASES)) != len(CASES):
        raise ValueError("duplicate case_id in E207 CASES")
    rows = [by_case[c] for c in CASES]
    from collections import Counter

    dist = dict(Counter(r["object_key"] for r in rows))
    if dist != EXPECTED_OBJECT_COUNTS:
        raise ValueError(f"E207 object drift: expected={EXPECTED_OBJECT_COUNTS} got={dist}")
    return rows


def scene_path(case_id: str) -> Path:
    return task_dir(case_id) / f"{SCENE}.xml"


def override_id(case_id: str) -> str:
    return f"core4d_{EXP}_{case_id}_contactAlignedTop_{ARM_TAG}"


def override_path(case_id: str) -> Path:
    return OVERRIDE_DIR / f"{override_id(case_id)}.yaml"


def out_dir(case_id: str, stage: str = "full") -> Path:
    """Stage separates the 64x4 smoke from the 1024x32 full run so a small-budget
    rollout can never shadow the full one via skip-already-done."""
    return RESULTS / f"s6_downstream/cem/{stage}" / f"{EXP}_{case_id}_{ARM_TAG}"


def result_npz(case_id: str, stage: str = "full") -> Path:
    return out_dir(case_id, stage) / "trajectory_mjwp_act.npz"


def config_act(case_id: str, stage: str = "full") -> Path:
    return out_dir(case_id, stage) / "config_act.yaml"


def audit(verbose: bool = True) -> int:
    """Assert the E207 scene contract: reused sidecar == E178 + object gravcomp=1.

    Fails loudly on the first drift; returns the number of verified cases.
    """
    rows = sources()
    for row in rows:
        case_id = row["case_id"]
        base, side = e178_scene_path(case_id), scene_path(case_id)
        for label, path in (("E178 scene", base), ("gravcomp sidecar", side)):
            if not path.is_file():
                raise FileNotFoundError(f"{case_id}: missing {label}: {path}")
        assert_gravcomp_diff(base, side)
        if verbose:
            print(f"  {case_id:32s} PASS  {SCENE} == E178 + object gravcomp=1")
    if verbose:
        print(
            f"audit PASS: {len(rows)}/{len(CASES)} single-variable gravcomp scenes "
            f"({EXPECTED_OBJECT_COUNTS})"
        )
    return len(rows)


if __name__ == "__main__":
    raise SystemExit(0 if audit() == len(CASES) else 1)
