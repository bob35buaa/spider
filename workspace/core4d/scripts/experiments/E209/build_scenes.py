#!/usr/bin/env python3
"""E209 P1: emit the 22 gravcomp sidecars and prove they are single-variable.

Writes ``scene_act_E209_lowgeom_PRG_gravcomp.xml`` beside each case's
``scene_act_E206_lowgeom_PRG.xml``.  Two independent proofs that gravcomp is the
only change:

  1. XML-level  -- ``e200_common.assert_gravcomp_diff`` compares full element
     signatures against "base with object gravcomp=1" (reused verbatim, rule 13).
  2. Model-level -- MuJoCo compiles both scenes and ``ngeom``/``npair``/``nq``/
     ``nv``/``nu``/``nbody`` must match exactly.  A body attribute cannot change
     geometry or collision pairs; if these ever differ, the sidecar is not what
     it claims to be.  This is the check the XML signature cannot give us,
     because it never asks the compiler.

Usage:
    MUJOCO_GL=disable .venv/bin/python \
      workspace/core4d/scripts/experiments/E209/build_scenes.py [--overwrite]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mujoco

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO / "workspace/core4d/scripts/experiments/E209"))

import e209_common as C  # noqa: E402

MODEL_INVARIANTS = ("ngeom", "npair", "nq", "nv", "nu", "nbody")


def model_shape(path: Path) -> dict[str, int]:
    model = mujoco.MjModel.from_xml_path(str(path))
    return {k: int(getattr(model, k)) for k in MODEL_INVARIANTS}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true", help="rewrite existing sidecars")
    args = parser.parse_args()

    rows = C.sources()
    report: list[dict[str, object]] = []
    for row in rows:
        case_id = row["case_id"]
        base = C.base_scene_path(row)
        side = C.build_sidecar(row, overwrite=args.overwrite)

        gravcomp_hits = side.read_text(encoding="utf-8").count("gravcomp")
        if gravcomp_hits != 1:
            raise SystemExit(f"{case_id}: expected exactly 1 gravcomp token, got {gravcomp_hits}")

        base_shape, side_shape = model_shape(base), model_shape(side)
        if base_shape != side_shape:
            raise SystemExit(
                f"{case_id}: gravcomp changed the compiled model -- NOT single-variable\n"
                f"  base    {base_shape}\n  sidecar {side_shape}"
            )
        report.append(
            {
                "case_id": case_id,
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
            f"  {case_id:36s} PASS  npair={side_shape['npair']:4d} ngeom={side_shape['ngeom']:4d} "
            f"nq={side_shape['nq']}  m={C.object_mass(base):.3f}kg"
        )

    shas = {r["sidecar_sha256"] for r in report}
    if len(shas) != len(report):
        raise SystemExit("duplicate sidecar sha256 -- two cases share a scene file")
    if shas & {r["base_sha256"] for r in report}:
        raise SystemExit("a sidecar is byte-identical to a base scene -- gravcomp not applied")

    out = C.S6_DIR / "manifests/e209_scene_audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    C.audit(verbose=False)
    print(f"\nP1 PASS: {len(report)}/{C.EXPECTED_CASES} single-variable gravcomp sidecars")
    print(f"  audit -> {out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
