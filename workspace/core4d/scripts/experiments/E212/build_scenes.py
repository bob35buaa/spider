#!/usr/bin/env python3
"""E212 P2: emit the 12 partial-gravcomp sidecars and prove they are single-variable.

Writes ``scene_act_E212_lowgeom_PRG_gc{04,06,08}.xml`` beside each desk023 case's
``scene_act_E206_lowgeom_PRG.xml``.  Four independent proofs that the gravcomp
value is the only change:

  1. XML-level   -- ``e212_common.assert_gravcomp_diff_value`` compares full
     recursive element signatures against "base with object gravcomp=<value>".
     It reuses ``e200_common._signature`` verbatim so it cannot drift from the
     g=1 assertion E209/E210 use.
  2. Token count -- exactly one ``gravcomp`` token in the file.
  3. Read-back   -- the object body's gravcomp attribute parses back as the
     expected value string.  (1) already implies this, but (1) would also pass
     if BOTH sides were wrong in the same way; (3) pins the absolute value.
  4. Model-level -- MuJoCo compiles base and sidecar and ``ngeom``/``npair``/
     ``nq``/``nv``/``nu``/``nbody`` must match exactly.  A body attribute cannot
     change geometry or collision pairs; if these differ, the sidecar is not
     what it claims to be.  This is the check the XML signature cannot give,
     because it never asks the compiler.

``--self-test`` additionally tampers a sidecar in a temp copy and proves the
assertion rejects it -- the same reverse test E210 used
(``E210/build_gravcomp_sidecars.py:57-79``).  An assertion nobody has ever seen
fail is not evidence.

Usage:
    MUJOCO_GL=disable .venv/bin/python \
      workspace/core4d/scripts/experiments/E212/build_scenes.py [--overwrite] [--self-test]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO / "workspace/core4d/scripts/experiments/E212"))

import e212_common as C  # noqa: E402

MODEL_INVARIANTS = ("ngeom", "npair", "nq", "nv", "nu", "nbody")


def model_shape(path: Path) -> dict[str, int]:
    model = mujoco.MjModel.from_xml_path(str(path))
    return {k: int(getattr(model, k)) for k in MODEL_INVARIANTS}


def self_test(base: Path, sidecar: Path, value: str) -> list[str]:
    """Prove the assertion rejects (a) a wrong value and (b) a non-gravcomp edit."""
    results: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        # (a) right attribute, wrong value
        wrong = Path(tmp) / "wrong_value.xml"
        tree = ET.parse(base)
        next(b for b in tree.getroot().iter("body") if b.get("name") == "object").set(
            "gravcomp", "0.5" if value != "0.5" else "0.7"
        )
        ET.indent(tree, space="  ")
        tree.write(wrong, encoding="utf-8", xml_declaration=True)
        try:
            C.assert_gravcomp_diff_value(base, wrong, value)
        except AssertionError:
            results.append("rejects_wrong_value")
        else:
            raise SystemExit("SELF-TEST FAILED: assertion accepted a wrong gravcomp value")

        # (b) right value plus a second, unrelated edit
        tampered = Path(tmp) / "tampered.xml"
        shutil.copy2(sidecar, tampered)
        tree = ET.parse(tampered)
        obj = next(b for b in tree.getroot().iter("body") if b.get("name") == "object")
        inertial = obj.find("inertial")
        inertial.set("mass", f"{float(inertial.get('mass')) + 1.0:.3f}")
        tree.write(tampered, encoding="utf-8", xml_declaration=True)
        try:
            C.assert_gravcomp_diff_value(base, tampered, value)
        except AssertionError:
            results.append("rejects_extra_edit")
        else:
            raise SystemExit("SELF-TEST FAILED: assertion accepted a non-gravcomp-only diff")
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true", help="rewrite existing sidecars")
    parser.add_argument("--self-test", action="store_true", default=True)
    parser.add_argument("--no-self-test", dest="self_test", action="store_false")
    args = parser.parse_args()

    rows = C.sources()
    report: list[dict[str, object]] = []
    tests: list[str] = []
    for row in rows:
        case_id = row["case_id"]
        base = C.base_scene_path(row)
        base_shape = model_shape(base)
        for arm in C.ARM_ORDER:
            value = C.gravcomp_str(arm)
            side = C.build_sidecar(row, arm, overwrite=args.overwrite)

            hits = side.read_text(encoding="utf-8").count("gravcomp")
            if hits != 1:
                raise SystemExit(f"{case_id}/{arm}: expected exactly 1 gravcomp token, got {hits}")
            got = C.scene_gravcomp(side)
            if got != value:
                raise SystemExit(f"{case_id}/{arm}: gravcomp read back {got!r} != {value!r}")

            side_shape = model_shape(side)
            if base_shape != side_shape:
                raise SystemExit(
                    f"{case_id}/{arm}: gravcomp changed the compiled model -- NOT single-variable\n"
                    f"  base    {base_shape}\n  sidecar {side_shape}"
                )
            if args.self_test and not tests:
                tests = self_test(base, side, value)

            report.append(
                {
                    "case_id": case_id,
                    "arm": arm,
                    "gravcomp": value,
                    "object_key": row["object_key"],
                    "task": C.target_task(row),
                    "base_scene": str(base.relative_to(REPO)),
                    "sidecar": str(side.relative_to(REPO)),
                    "base_sha256": C.sha256(base),
                    "sidecar_sha256": C.sha256(side),
                    "object_mass_kg": C.object_mass(base),
                    **side_shape,
                }
            )
            print(
                f"  {case_id:32s} {arm}  gravcomp={value}  PASS  "
                f"npair={side_shape['npair']:4d} ngeom={side_shape['ngeom']:4d} nq={side_shape['nq']}"
            )

    shas = {r["sidecar_sha256"] for r in report}
    if len(shas) != len(report):
        raise SystemExit("duplicate sidecar sha256 -- two (case, arm) pairs share a scene file")
    if shas & {r["base_sha256"] for r in report}:
        raise SystemExit("a sidecar is byte-identical to a base scene -- gravcomp not applied")
    # A partial arm must also differ from the E209 g=1 scene, else the sweep has
    # a duplicated point masquerading as a new one.
    g1_shas = {
        C.sha256(p)
        for p in (C.task_dir(r) / f"{C.G1_SCENE}.xml" for r in rows)
        if p.is_file()
    }
    if shas & g1_shas:
        raise SystemExit("a partial sidecar equals the E209 g=1 scene byte-for-byte")

    out = C.MANIFEST_DIR / f"{C.EXP.lower()}_scene_audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"self_test": tests, "scenes": report}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    C.audit(verbose=False)
    print(f"\nP2 PASS: {len(report)}/{C.EXPECTED_ROWS} single-variable partial-gravcomp sidecars")
    if tests:
        print(f"  assertion self-test: {', '.join(tests)}")
    print(f"  audit -> {out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
