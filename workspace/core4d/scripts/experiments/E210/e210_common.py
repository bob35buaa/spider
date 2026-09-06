#!/usr/bin/env python3
"""Shared contract for E210 (aug x G1only = E202 bucket aug + object gravcomp).

E210 reuses E202's bucket-augmentation assets VERBATIM -- same omnirt_v2 ref_fk
aug reference trajectory, same 3cm contact mask, same E178 contact-aligned
five-segment proxy, same E174 PRG reward/gate, same CEM budget (1024x32 seed 0).
The only change is the object body ``gravcomp`` 0 -> 1, i.e. exactly the E207
intervention transplanted onto the augmented variants.

That completes a 2x2 whose other three cells already exist on disk::

    A  orig x PRG       E178   scene_act_E178_contactAlignedTop
    B  orig x PRG+G1    E207   scene_act_E205_contactAlignedTop_gravcomp   <- baseline
    C  aug  x PRG       E202   scene_act_E202_bucketAlignedTop_PRG         <- single-variable partner
    D  aug  x PRG+G1    E210   scene_act_E210_bucketAlignedTop_PRG_gravcomp  (this)

``C -> D`` is the strict single-variable contrast (gravcomp only). ``B -> D`` is
the delivery judgement but moves two axes (object perturbation AND the omnirt_v1
-> v2 retarget variant that E202 had to adopt because v1 is often IK-infeasible
once the object moves); that confound is inherited from E202 and must stay
labelled in every report -- see plan240.

Scope note (plan240 round 2): translations only, no rotations. As a direct
consequence ``bucket007_20231023_075_p2`` is excluded -- E202 recorded all three
of its translations as ``runtime_initial_overlap`` (the reference's *first frame*
puts a leg inside the bucket once the object is shifted). That is a geometric
fact about the reference, unaffected by gravcomp, so re-running it would only
reproduce the same rejection. Its absence is asserted, not assumed: if E202 is
ever backfilled, :func:`load_variants` raises instead of silently growing.

Nothing is reimplemented here. IO helpers, repo roots, the manifest schema and
the frozen CEM/PRG constants come from E202 (which itself re-exports E199/E174);
the gravcomp assertion comes from E200 (rule 13).
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[5]
_EXP = REPO / "workspace/core4d/scripts/experiments"
for _d in ("E210", "E202", "E200", "E204_E205", "E178", "E177", "E176", "E175"):
    _p = str(_EXP / _d)
    if _p not in sys.path:
        sys.path.insert(0, _p)

import e202_common as E202C  # noqa: E402
import e204e205_common as E205C  # noqa: E402
from e200_common import assert_gravcomp_diff  # noqa: E402

#: A0 hand-gate. Imported from the SAME module E207 uses, so "E210's hand-gate is
#: identical to E207's" is structural rather than two copies that agree today.
E163_HAND_GATE = E205C.E163_HAND_GATE

# --- reused IO / path / schema helpers (single source of truth) --------------
now = E202C.now
rel = E202C.rel
repo_path = E202C.repo_path
sha256 = E202C.sha256
safe_id = E202C.safe_id
read_tsv = E202C.read_tsv
write_tsv = E202C.write_tsv
write_json = E202C.write_json
OVERRIDE_DIR = E202C.OVERRIDE_DIR
TASK_ROOT = E202C.TASK_ROOT
FIELDS = list(E202C.FIELDS)

# --- frozen CEM budget (identical to E178 / E202 / E207) ---------------------
CEM_SEED = E202C.CEM_SEED                    # 0
CEM_FULL_SAMPLES = E202C.CEM_FULL_SAMPLES    # 1024
CEM_FULL_OPT_STEPS = E202C.CEM_FULL_OPT_STEPS  # 32

# --- E210 arm ----------------------------------------------------------------
EXP = "E210"
ARM = "G1only_E202aug"
ARM_TAG = "G1only"
#: The gravcomp sidecar E210 writes next to each E202 aug PRG scene.
SCENE_NAME = "scene_act_E210_bucketAlignedTop_PRG_gravcomp"
#: The E202 sidecar it is a single-variable diff of.
BASE_SCENE_NAME = E202C.SCENE_NAME  # scene_act_E202_bucketAlignedTop_PRG

RESULTS = REPO / "workspace/core4d/results/E210"
MANIFEST_DIR = RESULTS / "s6_downstream/manifests"
FULL_MANIFEST = MANIFEST_DIR / "aug_g1only_full_manifest.tsv"
SMOKE_MANIFEST = MANIFEST_DIR / "aug_g1only_smoke_manifest.tsv"
CEM_ROOT_REL = "workspace/core4d/results/E210/s6_downstream/cem"
RENDER_ROOT_REL = "workspace/core4d/results/E210/s6_downstream/render"

# --- authority (both pinned by sha256; only read, never written) -------------
#: E207's delivered RL-export set -- the 6 cases this experiment is scoped to.
E207_PAIRED_TSV = (
    REPO / "workspace/core4d/results/E207/s6_downstream/rl_export/paired_rl_export_input.tsv"
)
E207_PAIRED_SHA = "9278cc0da48cf86119227107215ec11bfd61d4373a3b35007dbc89d64d0b7c2c"
#: E202's completed aug rollouts -- the source of the 15 variants reused here.
E202_MANIFEST = E202C.FULL_MANIFEST
E202_MANIFEST_SHA = "b7b187ce649938260f69e9279b6a74403c5d2c6e15d0078b1f702fbe37402a17"

#: The 6 bucket007 cases E207 delivered (order = paired_rl_export_input.tsv).
CASES: tuple[str, ...] = (
    "bucket007_20231003_2_021_p1",
    "bucket007_20231003_2_021_p2",
    "bucket007_20231020_059_p1",
    "bucket007_20231023_073_p1",
    "bucket007_20231023_075_p1",
    "bucket007_20231023_075_p2",
)
#: Excluded from the aug run, with the reason recorded rather than dropped.
EXCLUDED_CASES: dict[str, str] = {
    "bucket007_20231023_075_p2": (
        "E202 rejected all three translations with runtime_initial_overlap "
        "(reference first frame puts a leg inside the shifted bucket); geometric "
        "fact about the reference, unaffected by gravcomp. Rotations excluded by "
        "plan240 round-2 scope, so no untried variant remains."
    ),
}
AUG_CASES: tuple[str, ...] = tuple(c for c in CASES if c not in EXCLUDED_CASES)

#: Translation variants only (plan240 round 2). Names as E202 wrote them.
TRANS_VARIANTS: tuple[str, ...] = ("trans0", "trans1", "trans2")
EXPECTED_VARIANTS = len(AUG_CASES) * len(TRANS_VARIANTS)  # 5 x 3 = 15
#: E202 marks a finished rollout with this status; anything else is not reusable.
E202_DONE_STATUS = "run_complete_pending_eval"
#: bucket007's E178 contact-aligned proxy: 5 geoms -> 18 pairs/geom -> 90 pairs.
EXPECTED_GEOM_COUNT = 5
EXPECTED_PAIR_COUNT = 90


def _assert_sha(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} missing: {rel(path)}")
    got = sha256(path)
    if got != expected:
        raise AssertionError(
            f"{label} changed underneath E210: {rel(path)}\n"
            f"  expected {expected}\n  got      {got}\n"
            "Re-read the upstream experiment before re-pinning -- the 15 reused "
            "rollouts may no longer be the ones this contract was written against."
        )


def load_paired_cases() -> list[dict[str, str]]:
    """The 6 E207-delivered rows, sha-pinned, in CASES order."""
    _assert_sha(E207_PAIRED_TSV, E207_PAIRED_SHA, "E207 paired RL-export authority")
    rows = [r for r in read_tsv(E207_PAIRED_TSV) if r.get("case_id")]
    by_case = {r["case_id"]: r for r in rows}
    if len(rows) != len(CASES) or set(by_case) != set(CASES):
        raise AssertionError(
            f"E207 authority drift: expected {len(CASES)} cases {sorted(CASES)}, "
            f"got {len(rows)} {sorted(by_case)}"
        )
    return [by_case[c] for c in CASES]


def load_variants() -> list[dict[str, str]]:
    """The 15 reusable E202 aug rows for the in-scope cases, sha-pinned.

    Raises if any expected variant is missing/unfinished, or if the excluded
    case has acquired rows (i.e. E202 was backfilled and the plan240 exclusion
    rationale needs re-reading).
    """
    _assert_sha(E202_MANIFEST, E202_MANIFEST_SHA, "E202 aug priority manifest")
    rows = read_tsv(E202_MANIFEST)

    for case_id, reason in EXCLUDED_CASES.items():
        stray = [r for r in rows if r["case_id"] == case_id]
        if stray:
            raise AssertionError(
                f"{case_id} was excluded on the grounds that E202 has no feasible "
                f"variant for it, but E202 now lists {len(stray)} "
                f"({sorted(r['aug_variant'] for r in stray)}).\n"
                f"Recorded reason: {reason}\n"
                "Re-read plan240 'why 075_p2 is excluded' before widening the scope."
            )

    wanted = {(c, v) for c in AUG_CASES for v in TRANS_VARIANTS}
    picked: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        key = (row["case_id"], row["aug_variant"])
        if key not in wanted:
            continue
        if key in picked:
            raise AssertionError(f"duplicate E202 row for {key}")
        if row["status"] != E202_DONE_STATUS:
            raise AssertionError(
                f"{key} is not a finished E202 rollout: status={row['status']!r} "
                f"failure_mode={row.get('failure_mode')!r}"
            )
        picked[key] = row

    if missing := sorted(wanted - set(picked)):
        raise AssertionError(f"E202 rows missing for {missing}")
    if len(picked) != EXPECTED_VARIANTS:
        raise AssertionError(f"expected {EXPECTED_VARIANTS} variants, got {len(picked)}")

    ordered = [picked[(c, v)] for c in AUG_CASES for v in TRANS_VARIANTS]
    dist = dict(Counter(r["object_key"] for r in ordered))
    if dist != {"bucket007": EXPECTED_VARIANTS}:
        raise AssertionError(f"E210 is bucket007-only; got {dist}")
    return ordered


# --- per-variant naming ------------------------------------------------------

def variant_id(case_id: str, aug_variant: str) -> str:
    return safe_id(f"{EXP}_{case_id}_aug_{aug_variant}_{ARM_TAG}")


def base_override_id(row: dict[str, str]) -> str:
    """E202's arm override, which E210's override chains to via `defaults`."""
    return row["override_id"]


def override_id(case_id: str, aug_variant: str) -> str:
    return safe_id(f"core4d_{EXP}_{case_id}_aug_{aug_variant}_PRG_gravcomp")


def override_path(case_id: str, aug_variant: str) -> Path:
    return OVERRIDE_DIR / f"{override_id(case_id, aug_variant)}.yaml"


def base_scene_path(row: dict[str, str]) -> Path:
    """E202's PRG sidecar for this variant (the gravcomp diff base)."""
    return repo_path(row["scene_act"])


def scene_path(row: dict[str, str]) -> Path:
    return base_scene_path(row).with_name(f"{SCENE_NAME}.xml")


def artifact_paths(case_id: str, aug_variant: str, stage: str = "full") -> dict[str, str]:
    """Stage separates the 64x4 smoke from the 1024x32 full run so a small-budget
    rollout can never shadow the full one via the queue's skip-already-done."""
    vid = variant_id(case_id, aug_variant)
    return {
        "variant": vid,
        "result_npz": f"{CEM_ROOT_REL}/{stage}/{vid}.npz",
        "outdir_npz": f"{CEM_ROOT_REL}/{stage}/{vid}_outdir_{stage}/trajectory_mjwp_act.npz",
        "config_act": f"{CEM_ROOT_REL}/{stage}/{vid}_outdir_{stage}/config_act.yaml",
        "video": f"{RENDER_ROOT_REL}/{stage}/{vid}_{stage}.mp4",
        "log": f"logs/{EXP}/cem/{stage}/{vid}.log",
    }


# --- contract audit ----------------------------------------------------------

def audit(verbose: bool = True) -> int:
    """Assert the E210 scene contract for every in-scope variant.

    Only checks what must already be true *before* P2 writes the sidecars, plus
    the sidecar diff itself where a sidecar exists. Fails loudly on first drift.
    """
    load_paired_cases()
    rows = load_variants()
    checked = 0
    for row in rows:
        case_id, av = row["case_id"], row["aug_variant"]
        base = base_scene_path(row)
        if not base.is_file():
            raise FileNotFoundError(f"{case_id}/{av}: missing E202 PRG scene {rel(base)}")
        geoms = int(row.get("object_geom_count") or 0)
        pairs = int(row.get("compiled_robot_object_pair_count") or 0)
        if geoms != EXPECTED_GEOM_COUNT or pairs != EXPECTED_PAIR_COUNT:
            raise AssertionError(
                f"{case_id}/{av}: proxy drift geom={geoms} (want {EXPECTED_GEOM_COUNT}) "
                f"pair={pairs} (want {EXPECTED_PAIR_COUNT})"
            )
        side = scene_path(row)
        state = "sidecar PASS"
        if side.is_file():
            assert_gravcomp_diff(base, side)
        else:
            state = "sidecar PENDING (P2)"
        checked += 1
        if verbose:
            print(f"  {case_id:30s} {av:7s} geom={geoms} pair={pairs}  {state}")
    if verbose:
        excl = ", ".join(EXCLUDED_CASES) or "none"
        print(
            f"audit PASS: {checked}/{EXPECTED_VARIANTS} variants over "
            f"{len(AUG_CASES)} cases; excluded: {excl}; both authority sha pinned"
        )
    return checked


def summary() -> dict[str, Any]:
    return {
        "experiment": EXP,
        "arm": ARM,
        "scene_name": SCENE_NAME,
        "base_scene_name": BASE_SCENE_NAME,
        "cases_in_authority": list(CASES),
        "cases_augmented": list(AUG_CASES),
        "excluded_cases": EXCLUDED_CASES,
        "trans_variants": list(TRANS_VARIANTS),
        "expected_variants": EXPECTED_VARIANTS,
        "cem": {"seed": CEM_SEED, "samples": CEM_FULL_SAMPLES, "opt_steps": CEM_FULL_OPT_STEPS},
        "authority": {
            rel(E207_PAIRED_TSV): E207_PAIRED_SHA,
            rel(E202_MANIFEST): E202_MANIFEST_SHA,
        },
    }


if __name__ == "__main__":
    raise SystemExit(0 if audit() == EXPECTED_VARIANTS else 1)
