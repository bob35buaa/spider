#!/usr/bin/env python3
"""E216 C0 contract test -- registry, per-object arm, trans-only 0.1m, seeds, templates.

Pure CPU; run before any GPU work.  Asserts the pins the whole pipeline relies on,
so a registry edit or a missing warm-start is caught here rather than mid-CEM.

    .venv/bin/python .../E216/test_e216_contract.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import e216_common as C  # noqa: E402

EXPECTED_BY_OBJECT = {
    "box001": 24, "box004": 6, "box021": 28, "box023": 16, "box024": 9,
    "bucket003": 9, "bucket007": 12,
}
EXPECTED_ARM = {
    "box021": "box_prg", "box023": "box_noprg",
    "box001": "box_prg_g1a2", "box004": "box_prg_g1a2", "box024": "box_prg_g1a2",
    "bucket003": "bucket_prg", "bucket007": "bucket_prg",
}


def main() -> int:
    fails: list[str] = []

    def check(cond: bool, msg: str) -> None:
        if not cond:
            fails.append(msg)

    cases = C.load_e216_cases()

    # A1 count + per-object
    check(len(cases) == 104, f"A1 expected 104 cases, got {len(cases)}")
    check(C.EXPECTED_CASES_BY_OBJECT == EXPECTED_BY_OBJECT,
          f"A1 per-object mismatch: {C.EXPECTED_CASES_BY_OBJECT} != {EXPECTED_BY_OBJECT}")

    # A2 arm mapping per object + group spec present
    for c in cases:
        obj, cid = c["object_key"], c["case_id"]
        check(c["arm_group"] == EXPECTED_ARM[obj],
              f"A2 {cid}: arm {c['arm_group']} != {EXPECTED_ARM[obj]}")
        check(c["arm_group"] in C.GROUPS, f"A2 {cid}: arm {c['arm_group']} not in GROUPS")

    # A3 trans-only 0.1m variants
    check([v for v, _ in C.BUILD_VARIANTS] == ["trans0", "trans1", "trans2"],
          f"A3 BUILD_VARIANTS not trans-only: {C.BUILD_VARIANTS}")
    check(C.OMNIRT_TRANS_OFFSET_M == "0.1", f"A3 OMNIRT_TRANS_OFFSET_M != 0.1: {C.OMNIRT_TRANS_OFFSET_M}")
    check(C.aug_translation("trans0") == "0.1,0,0", f"A3 trans0 offset {C.aug_translation('trans0')}")
    check(C.aug_translation("trans1") == "0,0.1,0", f"A3 trans1 offset {C.aug_translation('trans1')}")
    check(C.aug_translation("trans2") == "0,-0.1,0", f"A3 trans2 offset {C.aug_translation('trans2')}")
    check(all(C.aug_rotation_rad(v) == "0.0" for v, _ in C.BUILD_VARIANTS), "A3 rotation not 0")

    # A4 aug_task carries _0p1m suffix and differs from the 0.2m name
    for c in cases[:3] + cases[-3:]:
        t = C.aug_task_name(c["case_id"], "trans0")
        check(t.endswith("__aug_trans0_0p1m"), f"A4 {c['case_id']}: aug_task {t} missing 0p1m suffix")
        check(t != f"dcv3_omnirt_v2_ref_fk_{c['case_id']}__aug_trans0",
              f"A4 {c['case_id']}: aug_task collides with 0.2m name")

    # A5 warm-start seed source exists (converted + _original + trim_window) per case
    n_seed_ok = 0
    for c in cases:
        src = C.seed_source_dir(c["case_id"])
        meta_ok = src.is_dir()
        orig = src / "retargeted" / f"{C.load_case_meta(c['base_target_task'])['holosoma_task']}_original.npz" if meta_ok else None
        tw = src / "trim_window.json"
        if meta_ok and orig and orig.is_file() and tw.is_file():
            n_seed_ok += 1
        else:
            fails.append(f"A5 {c['case_id']}: missing warm-start under {C.rel(src) if meta_ok else src}")

    # A6 base geometry template scene.xml exists (needed by create_spider_scene)
    n_tmpl_ok = 0
    for c in cases:
        sx = C.TASK_ROOT / c["base_target_task"] / "scene.xml"
        if sx.is_file():
            n_tmpl_ok += 1
        else:
            fails.append(f"A6 {c['case_id']}: base scene.xml missing ({C.rel(sx)})")

    print(f"A1 cases=104 ✓  per-object={C.EXPECTED_CASES_BY_OBJECT}")
    print(f"A2 arm mapping ✓ ({len(cases)} cases)")
    print(f"A3 trans-only 0.1m ✓  variants={[v for v,_ in C.BUILD_VARIANTS]}")
    print(f"A4 aug_task 0p1m suffix ✓")
    print(f"A5 warm-start seeds present: {n_seed_ok}/{len(cases)}")
    print(f"A6 base scene.xml templates present: {n_tmpl_ok}/{len(cases)}")

    if fails:
        print(f"\nCONTRACT FAIL ({len(fails)}):")
        for f in fails[:40]:
            print(f"  - {f}")
        return 1
    print("\nCONTRACT ALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
