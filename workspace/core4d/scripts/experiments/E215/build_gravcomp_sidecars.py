#!/usr/bin/env python3
"""E215: gravcomp sidecars for the two G1 arm groups (bucket007, box001/004/024).

Single-variable diff: the ``object`` body's ``gravcomp`` 0/absent -> 1, nothing
else.  Idempotent -- an existing sidecar is re-verified, not rewritten.

Two output names, one writer:
  * bucket007 (bucket_prg_gravcomp)  base = scene_act_E202_bucketAlignedTop_PRG
                                     out  = scene_act_E210_bucketAlignedTop_PRG_gravcomp
  * box001/004/024 (box_prg_g1a2)    base = scene_act_E199_rubberHull_PRG
                                     out  = scene_act_E199_rubberHull_PRG_gravcomp

The *assertion* (:func:`e200_common.assert_gravcomp_diff`) is the generic, vetted
guarantee and is reused verbatim.  ``--self-test`` proves the guard has teeth by
perturbing a second attribute and requiring the assertion to reject it.

Standalone entry re-verifies every gravcomp scene named in the artifacts TSV;
build_augmented_tasks also calls :func:`write_gravcomp_sidecar` per task inline.

Usage:
    .venv/bin/python .../E215/build_gravcomp_sidecars.py            # verify/build all
    ... --overwrite --no-self-test
"""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "E200"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import e200_common as E200  # noqa: E402  (canonical assert_gravcomp_diff)
import e215_common as C  # noqa: E402

assert_gravcomp_diff = E200.assert_gravcomp_diff


def write_gravcomp_sidecar(base_scene: Path, out_scene_name: str, *,
                           overwrite: bool = False) -> tuple[Path, str]:
    """Write (or re-verify) ``out`` = ``base`` + object gravcomp=1.

    Returns (path, 'written'|'verified').  ``out_scene_name`` is the basename
    without extension (e.g. scene_act_E210_bucketAlignedTop_PRG_gravcomp).
    """
    base = C.repo_path(base_scene)
    if not base.is_file():
        raise FileNotFoundError(base)
    out = base.with_name(f"{out_scene_name}.xml")
    if out.is_file() and not overwrite:
        assert_gravcomp_diff(base, out)
        return out, "verified"
    tree = ET.parse(base)
    root = tree.getroot()
    objs = [b for b in root.iter("body") if b.get("name") == "object"]
    if len(objs) != 1:
        raise ValueError(f"expected exactly one 'object' body in {C.rel(base)}, found {len(objs)}")
    if objs[0].get("gravcomp") not in (None, "0", "0.0"):
        raise ValueError(f"object already has gravcomp in {C.rel(base)}: {objs[0].get('gravcomp')!r}")
    objs[0].set("gravcomp", "1")
    ET.indent(tree, space="  ")
    tree.write(out, encoding="utf-8", xml_declaration=True)
    assert_gravcomp_diff(base, out)
    return out, "written"


def self_test(base: Path) -> None:
    """Prove assert_gravcomp_diff rejects a diff that is NOT gravcomp-only."""
    tree = ET.parse(base)
    root = tree.getroot()
    obj = next(b for b in root.iter("body") if b.get("name") == "object")
    obj.set("gravcomp", "1")
    for geom in root.iter("geom"):
        if geom.get("name"):
            geom.set("name", geom.get("name") + "_tampered")
            break
    tamper = base.with_name("_e215_selftest_tampered.xml")
    ET.indent(tree, space="  ")
    tree.write(tamper, encoding="utf-8", xml_declaration=True)
    try:
        assert_gravcomp_diff(base, tamper)
    except AssertionError:
        print("  self-test PASS: guard rejected a gravcomp+extra-change sidecar")
        return
    finally:
        tamper.unlink(missing_ok=True)
    raise AssertionError(
        "self-test FAILED: assert_gravcomp_diff accepted a >gravcomp diff -- guarantee is vacuous")


def _gravcomp_rows() -> list[dict[str, str]]:
    """Artifacts rows whose arm group builds a gravcomp sidecar."""
    if not C.ARTIFACTS_TSV.is_file():
        raise SystemExit(f"missing {C.rel(C.ARTIFACTS_TSV)}; run build_augmented_tasks.py first")
    rows = []
    for r in C.read_tsv(C.ARTIFACTS_TSV):
        if not r.get("status", "").startswith("built"):
            continue
        if C.GROUPS[r["arm_group"]].get("gravcomp"):
            rows.append(r)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--self-test", action="store_true", default=True)
    ap.add_argument("--no-self-test", dest="self_test", action="store_false")
    args = ap.parse_args()

    rows = _gravcomp_rows()
    if not rows:
        print("no gravcomp arm-group rows built yet; nothing to verify")
        return 0

    if args.self_test:
        # tamper-test against the first PRG base scene (gravcomp source)
        first = C.repo_path(rows[0]["gravcomp_base_scene"])
        self_test(first)

    report = []
    for row in rows:
        base = C.repo_path(row["gravcomp_base_scene"])
        spec = C.GROUPS[row["arm_group"]]
        out, action = write_gravcomp_sidecar(base, spec["scene_name"], overwrite=args.overwrite)
        report.append({
            "case_id": row["case_id"], "aug_variant": row["aug_variant"],
            "arm_group": row["arm_group"], "action": action,
            "base_scene": C.rel(base), "scene_act": C.rel(out),
            "effective_scene_sha256": C.sha256(out),
        })
        print(f"  {row['case_id']:32s} {row['aug_variant']:5s} {action:8s} {C.rel(out)}")

    bases = {r["base_scene"] for r in report}
    if len(bases) != len(report):
        raise AssertionError(f"base scenes not per-variant: {len(bases)} distinct for {len(report)} rows")

    out_json = C.PREFLIGHT_DIR / "gravcomp_sidecars.json"
    C.write_json(out_json, {
        "created_at": C.now(), "count": len(report),
        "written": sum(r["action"] == "written" for r in report),
        "verified": sum(r["action"] == "verified" for r in report),
        "self_test": bool(args.self_test), "sidecars": report,
    })
    print(f"\nPASS: {len(report)} gravcomp sidecars (single-variable object gravcomp=1) "
          f"-> {C.rel(out_json)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
