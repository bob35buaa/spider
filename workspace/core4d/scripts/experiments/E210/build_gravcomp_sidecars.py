#!/usr/bin/env python3
"""E210 P2: write the gravcomp sidecar next to each E202 aug PRG scene.

Single-variable diff: the ``object`` body's ``gravcomp`` 0/absent -> 1, nothing
else. Idempotent -- an existing sidecar is re-verified, not rewritten.

Why this file exists instead of calling ``e200_common.build_gravcomp_sidecar``:
that helper hardcodes its output name to ``scene_act_E199_rubberHull_PRG_gravcomp``
(see e200_common.py:161), so reusing it here would silently write E199's name
next to an E202 scene. E209 logged the same trap. The *assertion*
(:func:`e200_common.assert_gravcomp_diff`) is generic and IS reused -- that is
the part that carries the guarantee.

``--self-test`` additionally proves the guard has teeth: it perturbs a second
attribute alongside gravcomp and requires the assertion to reject it. A guard
that has never been seen to fail is not evidence.
"""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e210_common as C  # noqa: E402


def write_sidecar(base: Path, out: Path, *, overwrite: bool = False) -> tuple[Path, str]:
    """Write (or re-verify) ``out`` = ``base`` + object gravcomp=1.

    Returns (path, action) where action is 'written' | 'verified'.
    """
    if not base.is_file():
        raise FileNotFoundError(base)
    if out.is_file() and not overwrite:
        C.assert_gravcomp_diff(base, out)
        return out, "verified"

    tree = ET.parse(base)
    root = tree.getroot()
    objs = [b for b in root.iter("body") if b.get("name") == "object"]
    if len(objs) != 1:
        raise ValueError(f"expected exactly one body named 'object' in {C.rel(base)}, found {len(objs)}")
    if objs[0].get("gravcomp") not in (None, "0", "0.0"):
        raise ValueError(f"object already has gravcomp in {C.rel(base)}: {objs[0].get('gravcomp')!r}")
    objs[0].set("gravcomp", "1")
    ET.indent(tree, space="  ")
    tree.write(out, encoding="utf-8", xml_declaration=True)
    C.assert_gravcomp_diff(base, out)
    return out, "written"


def self_test(base: Path) -> None:
    """Prove assert_gravcomp_diff rejects a diff that is NOT gravcomp-only."""
    tree = ET.parse(base)
    root = tree.getroot()
    obj = next(b for b in root.iter("body") if b.get("name") == "object")
    obj.set("gravcomp", "1")
    # a second, unrelated change riding along -- exactly what the guard must catch
    for geom in root.iter("geom"):
        if geom.get("name"):
            geom.set("name", geom.get("name") + "_tampered")
            break
    tamper = base.with_name("_e210_selftest_tampered.xml")
    ET.indent(tree, space="  ")
    tree.write(tamper, encoding="utf-8", xml_declaration=True)
    try:
        C.assert_gravcomp_diff(base, tamper)
    except AssertionError:
        print("  self-test PASS: guard rejected a gravcomp+extra-change sidecar")
        return
    finally:
        tamper.unlink(missing_ok=True)
    raise AssertionError(
        "self-test FAILED: assert_gravcomp_diff accepted a sidecar that changed "
        "more than gravcomp -- the P2 guarantee is vacuous, stop here."
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--overwrite", action="store_true", help="rewrite existing sidecars")
    ap.add_argument("--self-test", action="store_true", default=True,
                    help="prove the gravcomp guard rejects a non-single-variable diff")
    ap.add_argument("--no-self-test", dest="self_test", action="store_false")
    args = ap.parse_args()

    rows = C.load_variants()
    if args.self_test:
        self_test(C.base_scene_path(rows[0]))

    report = []
    for row in rows:
        base = C.base_scene_path(row)
        out, action = write_sidecar(base, C.scene_path(row), overwrite=args.overwrite)
        report.append({
            "case_id": row["case_id"], "aug_variant": row["aug_variant"],
            "target_task": row["target_task"], "action": action,
            "base_scene_act": C.rel(base), "base_scene_sha256": C.sha256(base),
            "scene_act": C.rel(out), "effective_scene_sha256": C.sha256(out),
            "object_geom_count": row.get("object_geom_count", ""),
            "compiled_robot_object_pair_count": row.get("compiled_robot_object_pair_count", ""),
        })
        print(f"  {row['case_id']:30s} {row['aug_variant']:7s} {action:8s} {C.rel(out)}")

    # A shared base scene would mean two variants pointing at the same file --
    # they must be per-task, otherwise the sidecars would overwrite each other.
    bases = {r["base_scene_act"] for r in report}
    if len(bases) != len(report):
        raise AssertionError(f"base scenes are not per-variant: {len(bases)} distinct for {len(report)} rows")

    out_json = C.RESULTS / "preflight/e210_gravcomp_sidecars.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    C.write_json(out_json, {
        "created_at": C.now(), "scene_name": C.SCENE_NAME,
        "base_scene_name": C.BASE_SCENE_NAME, "count": len(report),
        "written": sum(r["action"] == "written" for r in report),
        "verified": sum(r["action"] == "verified" for r in report),
        "self_test": bool(args.self_test), "sidecars": report,
    })
    print(f"\nP2 PASS: {len(report)}/{C.EXPECTED_VARIANTS} sidecars, "
          f"all == E202 PRG scene + object gravcomp=1 -> {C.rel(out_json)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
