#!/usr/bin/env python3
"""E208 V5c: prove an aug scene differs from its E206 base ONLY by object pose.

The whole comparability argument of this experiment is that an ``__aug_*`` task
is E206's scene with the object started somewhere else.  If anything else drifted
-- a collision proxy box, a robot body, the ground, a contact pair -- then a
worse aug metric could be geometry rather than augmentation, and C4 would be
measuring the wrong thing.

So for every arm-chain XML the aug task owns, this masks the ``<body
name="object">`` element's ``pos``/``quat`` and requires the rest to be
byte-identical to the base task's file:

  scene.xml                            template output (inherits E206's lowgeom proxy)
  scene_act.xml                        + actuators, 2 robot<->object pairs
  scene_act_E206_lowgeom_rubberHull.xml  + rubber-hull hand patch
  scene_act_E206_lowgeom_noPRG.xml     + 2N hand pairs
  scene_act_E206_lowgeom_PRG.xml       + 16N leg pairs   <- the file CEM loads

Byte parity on the last one is the strongest single piece of evidence that
E206's hand patch and pair arithmetic landed identically in the aug dir, and it
covers the whole chain at once.

Two further checks, because masked parity alone cannot see a *mixed-up* object
pose (every variant would still pass it):

  P3  the aug scene's object pos/quat equals frame 0 of THIS variant's trimmed
      npz -- so the scene was built from the variant it claims, not from orig or
      from a sibling variant
  P4  the base scene's object pos/quat equals frame 0 of the seeded
      ``_original`` npz -- the same mapping, verified on the reference side

and P5 records the scene-level offset, which is an independent re-derivation of
``approach_trans_offset_m_max`` straight out of the XML rather than out of the
npz the manifest was built from.

**Why the object joint order is masked too.**  ``generate_scene_act.py:55-78``
picks the Euler convention per task, choosing whichever of the six orders
minimises the worst middle-axis angle over the trajectory (gimbal-lock margin).
Rotation augmentation changes the object's orientation trajectory, so a rot
variant can legitimately land on a different convention than its base -- measured
here: 11 of 42 rot variants, 0 of 63 trans variants.  That is not a geometry
difference and forcing the base's convention would be strictly worse (it is the
convention that is *further* from gimbal lock that gets dropped).  It is safe
because ``run_mjwp.py:583`` resolves the convention from the task dir's own
``scene_act_meta.json``, and E206's own 22 base cases already span four
conventions (XZY 16 / ZYX 4 / XYZ 1 / YZX 1) -- so nothing downstream can be
assuming a single one.  The object joints and their actuators are therefore
order-normalised before comparison; a change to the joint *set* still fails.

  P6  replicates ``resolve_scene_act_reference``'s runtime contract offline on
      both scenes -- metadata present, convention valid, and equal to the
      compiled hinge axis sequence.  That assertion is fail-closed at run time,
      so a stale ``scene_act_meta.json`` would abort a CEM run after it had
      already been dispatched to a GPU; checking it here costs a model compile.

Usage:
    .venv/bin/python .../E208/check_scene_parity_vs_e206.py
    ... --include-excluded      # also gate chair005 (default: report only)
    ... --skip-contract         # drop P6 (no mujoco compile)
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import e208_common as C  # noqa: E402

OBJ_POS = slice(36, 39)
OBJ_QUAT = slice(39, 43)

# scene.xml prints pos with 4 decimals and quat with 6, so parity of the parsed
# values is only meaningful to that precision.
POS_TOL_M = 5e-4
QUAT_TOL = 5e-6

CHAIN = (
    "scene.xml",
    "scene_act.xml",
    "scene_act_E206_lowgeom_rubberHull.xml",
    "scene_act_E206_lowgeom_noPRG.xml",
    "scene_act_E206_lowgeom_PRG.xml",
)

OBJECT_BODY = re.compile(
    r'(?P<head><body\s+name="object"\s+)pos="(?P<pos>[^"]*)"\s+quat="(?P<quat>[^"]*)"'
)
# The object's 6 DOF and their actuators: emitted in Euler-convention order, so
# their sequence is a property of the trajectory, not of the geometry.
OBJECT_DOF = re.compile(r'<(?P<tag>joint|position) name="object_(?:pos|rot)_[xyz]"[^>]*/>')


def _normalise_dof_order(text: str) -> tuple[str, dict[str, list[str]]]:
    """Sort the object joint/actuator elements in place, keeping content intact.

    Order-insensitive, set-sensitive: reordering passes, adding/removing/altering
    a DOF still shows up as a diff.
    """
    by_tag: dict[str, list[str]] = {}
    for m in OBJECT_DOF.finditer(text):
        by_tag.setdefault(m.group("tag"), []).append(m.group(0))
    ordered = {tag: sorted(items) for tag, items in by_tag.items()}
    cursor = {tag: 0 for tag in ordered}

    def sub(m: re.Match[str]) -> str:
        tag = m.group("tag")
        out = ordered[tag][cursor[tag]]
        cursor[tag] += 1
        return out

    return OBJECT_DOF.sub(sub, text), by_tag


def mask_object_pose(text: str) -> tuple[str, list[tuple[np.ndarray, np.ndarray]], dict[str, list[str]]]:
    """Placeholder the object pose and order-normalise its DOF elements."""
    found: list[tuple[np.ndarray, np.ndarray]] = []

    def sub(m: re.Match[str]) -> str:
        found.append((
            np.fromstring(m.group("pos"), sep=" "),
            np.fromstring(m.group("quat"), sep=" "),
        ))
        return f'{m.group("head")}pos="<MASKED>" quat="<MASKED>"'

    masked, dofs = _normalise_dof_order(OBJECT_BODY.sub(sub, text))
    return masked, found, dofs


def hinge_sequence(dofs: dict[str, list[str]]) -> str:
    """The compiled Euler convention, read off the joint element order."""
    order = [
        m.group(1)
        for j in dofs.get("joint", [])
        if (m := re.search(r'name="object_rot_([xyz])"', j))
    ]
    return "".join(a.upper() for a in order)


def yaw_deg(quat: np.ndarray) -> float:
    """MuJoCo wxyz -> yaw in degrees."""
    w, x, y, z = quat
    return math.degrees(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def frame0_pose(npz: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(npz, allow_pickle=True) as data:
        q = np.asarray(data["qpos"], dtype=np.float64)
    return q[0, OBJ_POS].copy(), q[0, OBJ_QUAT].copy()


def scene_act_contract(prg_scene: Path) -> tuple[str, str]:
    """Offline replay of ``resolve_scene_act_reference``; returns (convention, sha)."""
    import mujoco  # local: only needed for P6

    from spider.simulators.scene_act_reference import resolve_scene_act_reference

    model = mujoco.MjModel.from_xml_path(str(prg_scene))
    ref = resolve_scene_act_reference(prg_scene, model, emit_log=False)
    return ref.convention, ref.meta_sha256


def check_row(row: dict[str, str], *, run_contract: bool) -> dict[str, Any]:
    base_dir = C.TASK_ROOT / row["base_target_task"]
    aug_dir = C.TASK_ROOT / row["target_task"]
    out: dict[str, Any] = {
        "case_id": row["case_id"], "object_key": row["object_key"],
        "aug_variant": row["aug_variant"],
        "effective_retarget_variant": row["effective_retarget_variant"],
        "base_target_task": row["base_target_task"], "target_task": row["target_task"],
        "excluded": int(C.is_excluded(row["object_key"])),
    }
    failures: list[str] = []
    checked: list[str] = []

    aug_pose: tuple[np.ndarray, np.ndarray] | None = None
    base_pose: tuple[np.ndarray, np.ndarray] | None = None

    for name in CHAIN:
        base_f, aug_f = base_dir / name, aug_dir / name
        if not aug_f.is_file():
            failures.append(f"P1[{name}]: aug file missing")
            continue
        if not base_f.is_file():
            failures.append(f"P1[{name}]: base file missing")
            continue
        base_masked, base_found, base_dofs = mask_object_pose(base_f.read_text(encoding="utf-8"))
        aug_masked, aug_found, aug_dofs = mask_object_pose(aug_f.read_text(encoding="utf-8"))

        if len(aug_found) != 1 or len(base_found) != 1:
            failures.append(
                f"P2[{name}]: expected exactly one object body, got "
                f"base={len(base_found)} aug={len(aug_found)}"
            )
            continue
        if base_masked != aug_masked:
            # Locate the first differing line so the failure is actionable.
            bl, al = base_masked.splitlines(), aug_masked.splitlines()
            where = next(
                (i + 1 for i, (b, a) in enumerate(zip(bl, al)) if b != a),
                min(len(bl), len(al)) + 1,
            )
            failures.append(
                f"P1[{name}]: differs outside the object pose, first at line {where} "
                f"(base {len(bl)} lines, aug {len(al)} lines)"
            )
            continue
        checked.append(name)
        if name == "scene_act_E206_lowgeom_PRG.xml":
            aug_pose, base_pose = aug_found[0], base_found[0]
            for side, dofs in (("base", base_dofs), ("aug", aug_dofs)):
                n_j, n_a = len(dofs.get("joint", [])), len(dofs.get("position", []))
                if (n_j, n_a) != (6, 6):
                    failures.append(
                        f"P2[{name}]: {side} object has {n_j} joints / {n_a} actuators, expected 6/6"
                    )
            out["xml_hinge_base"] = hinge_sequence(base_dofs)
            out["xml_hinge_aug"] = hinge_sequence(aug_dofs)

    out["files_parity_ok"] = ";".join(checked)
    out["n_files_checked"] = len(checked)

    if aug_pose is not None and base_pose is not None:
        d_pos = aug_pose[0] - base_pose[0]
        out["scene_offset_m"] = round(float(np.linalg.norm(d_pos)), 6)
        out["scene_yaw_deg"] = round(
            abs((yaw_deg(aug_pose[1]) - yaw_deg(base_pose[1]) + 180.0) % 360.0 - 180.0), 4
        )
        out["scene_offset_band"] = C.offset_band(out["scene_offset_m"])

        # P3/P4: the scene's object pose must be frame 0 of the npz it claims.
        meta = C.load_case_meta(row["base_target_task"])
        holo = meta["holosoma_task"]
        root = C.holosoma_dir(row["base_target_task"], row["effective_retarget_variant"])
        holo_name = dict(C.BUILD_VARIANTS)[row["aug_variant"]]
        for tag, npz, pose in (
            ("P3", root / "trimmed" / f"{holo}_{holo_name}.npz", aug_pose),
            ("P4", root / "trimmed" / f"{holo}_original.npz", base_pose),
        ):
            if not npz.is_file():
                failures.append(f"{tag}: trimmed npz missing: {C.rel(npz)}")
                continue
            n_pos, n_quat = frame0_pose(npz)
            dp = float(np.linalg.norm(n_pos - pose[0]))
            dq = float(np.abs(n_quat - pose[1]).max())
            out[f"{tag.lower()}_pos_err_m"] = round(dp, 8)
            out[f"{tag.lower()}_quat_err"] = round(dq, 8)
            if dp > POS_TOL_M:
                failures.append(
                    f"{tag}: scene object pos is {dp:.6f} m from frame 0 of "
                    f"{npz.name} -- scene built from the wrong trajectory"
                )
            if dq > QUAT_TOL:
                failures.append(f"{tag}: scene object quat differs by {dq:.2e} from {npz.name}")

        # P5: an XML-derived cross-check of the manifest's offset number.
        manifest_offset = float(row["approach_trans_offset_m_max"])
        out["manifest_offset_m"] = round(manifest_offset, 6)
        out["offset_agreement_m"] = round(abs(manifest_offset - out["scene_offset_m"]), 6)
    else:
        failures.append("P3/P4: PRG scene parity did not complete, pose checks skipped")

    if run_contract:
        for side, d in (("base", base_dir), ("aug", aug_dir)):
            prg = d / "scene_act_E206_lowgeom_PRG.xml"
            try:
                conv, meta_sha = scene_act_contract(prg)
            except Exception as exc:  # the runtime raises many types; all are fatal there
                failures.append(f"P6[{side}]: {type(exc).__name__}: {exc}")
                continue
            out[f"euler_convention_{side}"] = conv
            out[f"scene_act_meta_sha256_{side}"] = meta_sha
            xml_seq = out.get(f"xml_hinge_{side}")
            if xml_seq and conv != xml_seq:
                failures.append(f"P6[{side}]: resolved {conv} but XML joint order is {xml_seq}")
        out["euler_convention_changed"] = int(
            out.get("euler_convention_base") != out.get("euler_convention_aug")
        )

    out["failures"] = "; ".join(failures)
    out["status"] = "pass" if not failures else "fail"
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv-out", type=Path, default=C.SCENE_PARITY_TSV)
    ap.add_argument("--json-out", type=Path, default=None)
    ap.add_argument("--include-excluded", action="store_true",
                    help="let excluded objects (chair005) count toward the verdict")
    ap.add_argument("--skip-contract", action="store_true",
                    help="skip P6 (the offline scene-act contract replay; needs mujoco)")
    args = ap.parse_args()

    # Every row whose task dir was built, including the degenerate-offset ones.
    # The gated/reported split is then driven by `is_excluded()` -- the actual
    # decision -- rather than falling out of the status filter by coincidence.
    rows = [r for r in C.read_tsv(C.ARTIFACTS_TSV) if r.get("status", "").startswith("built")]
    if not rows:
        raise SystemExit(f"no built rows in {C.rel(C.ARTIFACTS_TSV)}")

    results = [check_row(r, run_contract=not args.skip_contract) for r in rows]
    gated = [r for r in results if args.include_excluded or not r["excluded"]]
    failures = [r for r in gated if r["status"] != "pass"]
    excluded_fail = [r for r in results if r["excluded"] and r["status"] != "pass"]

    for row in results:
        if row["status"] != "pass":
            tag = "FAIL" if row in gated else "fail/excluded"
            print(f"[{tag}] {row['case_id']:32s} {row['aug_variant']:7s} {row['failures']}")

    C.write_tsv(args.tsv_out, results)

    bands: dict[str, int] = {}
    for row in gated:
        bands[row.get("scene_offset_band", "?")] = bands.get(row.get("scene_offset_band", "?"), 0) + 1
    worst_agreement = max((r.get("offset_agreement_m", 0.0) for r in results), default=0.0)

    conv_changed: dict[str, int] = {}
    conv_pairs: dict[str, int] = {}
    for row in results:
        v = row["aug_variant"]
        conv_changed[v] = conv_changed.get(v, 0) + int(row.get("euler_convention_changed", 0))
        if row.get("euler_convention_changed"):
            k = f"{row['euler_convention_base']}->{row['euler_convention_aug']}"
            conv_pairs[k] = conv_pairs.get(k, 0) + 1

    payload = {
        "experiment": C.EXP_ID, "check": "V5c", "generated_at": C.now(),
        "chain": list(CHAIN),
        "n_rows": len(results), "n_gated": len(gated), "n_failures": len(failures),
        "n_excluded_rows": len(results) - len(gated),
        "n_excluded_failures": len(excluded_fail),
        "excluded_objects": list(C.EXCLUDED_OBJECT_KEYS),
        "scene_offset_bands": bands,
        "max_manifest_vs_scene_offset_disagreement_m": round(worst_agreement, 8),
        "euler_convention_changed_by_variant": conv_changed,
        "euler_convention_transitions": conv_pairs,
        "verdict": "pass" if not failures else "fail",
        "note": ("Masked parity covers the whole arm chain, so E206's rubber-hull patch "
                 "and 18N pair arithmetic are proven identical in the aug dir. P3/P4 tie "
                 "each scene to frame 0 of the npz it claims, which masked parity alone "
                 "cannot detect. The object DOF order is normalised, not compared: the "
                 "Euler convention is auto-selected per trajectory for gimbal-lock margin "
                 "and rot variants legitimately re-select it; P6 replays the fail-closed "
                 "runtime contract instead."),
    }
    C.write_json(args.json_out or args.tsv_out.with_suffix(".json"), payload)

    print(f"\nV5c {len(gated) - len(failures)}/{len(gated)} gated rows pass "
          f"({len(results) - len(gated)} excluded rows reported separately)")
    print(f"  scene offset bands: {bands}")
    print(f"  manifest-vs-scene offset agreement: max {worst_agreement:.2e} m")
    if not args.skip_contract:
        print(f"  euler convention re-selected: "
              f"{ {k: v for k, v in conv_changed.items() if v} } {conv_pairs}")
    print(f"  -> {C.rel(args.tsv_out)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
