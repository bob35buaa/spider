#!/usr/bin/env python3
"""E213 offline contract self-check (no GPU).

A1  source TSV sha256 pinned; 21 RL_EXPORT_READY rows; per-object counts pinned.
A2  every base task dir exists with task_info.json + the orig selected-arm scene.
A3  selected-arm map (arm/scene_name/gravcomp) is internally consistent and the
    orig selected scene actually carries that gravcomp.
A4  every aug variant row from E208 is status 'built'; the aug __aug_* task dir and
    its E206-PRG base scene exist.
A5  the built selected-arm sidecars are single-variable gravcomp diffs (sampled).
A6  partner gap = exactly the 11 partner-only cases; each has an E206 converted npz.
A7  the 1 PRG-selected case maps to existing E208 PRG aug rollouts (reuse target).

Usage:
    .venv/bin/python workspace/core4d/scripts/experiments/E213/test_e213_contract.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import e213_common as C  # noqa: E402

#: Expected scene_name SUFFIX per gravcomp value.  The experiment prefix (E209 /
#: E211 / E212) is the *sweep* experiment for that object family, so it varies
#: (desk007 G06 = E211_gc06, desk023 G06 = E212_gc06); only the suffix is fixed by
#: the g value.  The pipeline uses the TSV's arm_scene_name verbatim, so this test
#: validates internal consistency (suffix <-> gravcomp), not the prefix.
def expected_scene_suffix(arm_gravcomp: str) -> str:
    g = float(arm_gravcomp)
    if abs(g - 1.0) < 1e-9:
        return "PRG_gravcomp"
    if abs(g) < 1e-9:
        return "lowgeom_PRG"  # PRG base
    return f"PRG_gc{int(round(g * 10)):02d}"


EXPECTED_PARTNER_GAP = 11


def main() -> int:
    fails: list[str] = []

    def check(cond: bool, msg: str) -> None:
        if not cond:
            fails.append(msg)

    cases = C.load_cases()

    # A1
    check(C.sha256(C.SOURCE_TSV) == C.EXPECTED_SOURCE_SHA256, "A1 source TSV sha256 drift")
    check(len(cases) == C.EXPECTED_CASES, f"A1 case count {len(cases)} != {C.EXPECTED_CASES}")
    from collections import Counter
    by_obj = Counter(c["object_key"] for c in cases)
    check(dict(by_obj) == C.EXPECTED_CASES_BY_OBJECT, f"A1 per-object counts {dict(by_obj)}")

    # A2 + A3
    for c in cases:
        base = C.TASK_ROOT / c["base_target_task"]
        check((base / "task_info.json").is_file(), f"A2 {c['case_id']}: no task_info.json")
        check(c["arm"] in ("G1", "G08", "G06", "G04", "PRG"), f"A3 {c['case_id']}: unknown arm {c['arm']}")
        suffix = expected_scene_suffix(c["arm_gravcomp"])
        check(c["arm_scene_name"].endswith(suffix),
              f"A3 {c['case_id']}: scene {c['arm_scene_name']} lacks suffix {suffix} (g={c['arm_gravcomp']})")
        sel = C.orig_selected_scene(c)
        check(sel.is_file(), f"A2 {c['case_id']}: orig selected scene missing {C.rel(sel)}")
        if sel.is_file() and not C.is_prg_case(c):
            g = C.scene_gravcomp(sel)
            check(g not in (None, "0", "0.0"), f"A3 {c['case_id']}: orig selected scene gravcomp={g}")

    # A4
    aug = C.aug_rows_by_case()
    n_variants = 0
    for c in cases:
        rows = aug.get(c["case_id"], [])
        check(len(rows) > 0, f"A4 {c['case_id']}: no built aug variants")
        for r in rows:
            n_variants += 1
            check(r["status"] == C.BUILT_STATUS, f"A4 {c['case_id']}/{r['aug_variant']}: status={r['status']}")
            check(C.aug_task_dir(r).is_dir(), f"A4 {r['target_task']}: task dir missing")
            check(C.aug_prg_scene(r).is_file(), f"A4 {r['target_task']}: aug PRG scene missing")

    # A5 sidecar single-variable (sample up to 6 non-PRG rows)
    sampled = 0
    for c in cases:
        if C.is_prg_case(c) or sampled >= 6:
            continue
        for r in aug.get(c["case_id"], [])[:1]:
            side = C.aug_selected_scene(r, c)
            if not side.is_file():
                fails.append(f"A5 {r['target_task']}: selected sidecar not built yet (run build_source_arm_scenes.py)")
                continue
            try:
                C.assert_gravcomp_diff_value(C.aug_prg_scene(r), side, C.selected_gravcomp_str(c))
                sampled += 1
            except Exception as exc:  # noqa: BLE001
                fails.append(f"A5 {r['target_task']}: {exc}")

    # A6 partner gap
    gap = C.partner_gap_cases()
    check(len(gap) == EXPECTED_PARTNER_GAP, f"A6 partner gap {len(gap)} != {EXPECTED_PARTNER_GAP}")
    for g in gap:
        # aug seed = the converted SMPL-X dir in the partner's OWN retarget tree
        # (standard s3_retarget for 10; partner_omnirt_direct_v2 for the temp case).
        # Derived from partner_trimmed_npz = .../holosoma_<base>/trimmed/<file>.
        if not g["partner_trimmed_npz"]:
            fails.append(f"A6 {g['partner_case_id']}: no partner_trimmed_npz to locate seed")
            continue
        conv = C.repo_path(g["partner_trimmed_npz"]).parent.parent / "converted"
        check(conv.is_dir(), f"A6 {g['partner_case_id']}: no converted dir {C.rel(conv)}")

    # A7 PRG reuse
    prg = [c for c in cases if C.is_prg_case(c)]
    check(len(prg) == 1, f"A7 PRG-selected cases {len(prg)} != 1")
    for c in prg:
        for r in aug.get(c["case_id"], []):
            reuse = C.e208_prg_result_npz(c["case_id"], r["aug_variant"])
            check(reuse.is_file(), f"A7 {c['case_id']}/{r['aug_variant']}: E208 PRG rollout missing {C.rel(reuse)}")

    print(f"cases={len(cases)} aug_variant_rows={n_variants} partner_gap={len(gap)} "
          f"prg_reuse={len(prg)} sidecar_sampled={sampled}")
    if fails:
        for f in fails:
            print(f"  FAIL {f}")
        raise SystemExit(f"E213 contract FAILED ({len(fails)} checks)")
    print("E213 contract PASS: A1-A7")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
