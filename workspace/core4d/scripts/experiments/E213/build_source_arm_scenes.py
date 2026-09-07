#!/usr/bin/env python3
"""E213 Phase A step 1: write the selected-arm gravcomp sidecar into each aug task dir.

For every (case, aug variant) that E208 built, write
``<arm_scene_name>.xml`` next to the aug task's own ``scene_act_E206_lowgeom_PRG.xml``,
as a single-variable diff: the object body gravcomp absent/0 -> the value the
ORIGINAL selected-arm scene used (read verbatim so CEM behaves identically).

PRG-selected cases (g=0.0) are skipped: their selected arm is the E206 PRG base
itself, which E208 already ran; nothing to write.

C1 (single-variable) is enforced per file by ``assert_gravcomp_diff_value`` and
re-asserted here against a compiled-model check would be redundant -- the runtime
scene-act contract is exercised at CEM launch.

Usage:
    .venv/bin/python workspace/core4d/scripts/experiments/E213/build_source_arm_scenes.py
    ... --cases <case_id,...> --overwrite
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import e213_common as C  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="", help="comma-separated case_id subset")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    cases = C.load_cases()
    if args.cases:
        keep = {c.strip() for c in args.cases.split(",") if c.strip()}
        cases = [c for c in cases if c["case_id"] in keep]
    by_id = {c["case_id"]: c for c in cases}
    aug = C.aug_rows_by_case()

    written = 0
    skipped_prg = 0
    audit: list[dict[str, str]] = []
    failures: list[str] = []
    C.PREFLIGHT_DIR.mkdir(parents=True, exist_ok=True)

    for case_id in sorted(by_id):
        case = by_id[case_id]
        if C.is_prg_case(case):
            skipped_prg += 1
            print(f"  [skip-PRG] {case_id}: selected arm is PRG (g=0), reuse E208")
            continue
        value = C.selected_gravcomp_str(case)
        for row in aug.get(case_id, []):
            try:
                out = C.build_selected_sidecar(row, case, overwrite=args.overwrite)
                written += 1
                audit.append({
                    "case_id": case_id, "aug_variant": row["aug_variant"],
                    "arm": case["arm"], "gravcomp": value,
                    "scene": C.rel(out), "sha256": C.sha256(out),
                })
            except Exception as exc:  # noqa: BLE001
                failures.append(f"{case_id}/{row['aug_variant']}: {type(exc).__name__}: {exc}")

    C.write_json(C.ARM_SCENE_AUDIT_JSON, {
        "written": written, "skipped_prg_cases": skipped_prg,
        "n_failures": len(failures), "failures": failures, "rows": audit,
        "updated_at": C.now(),
    })

    print(f"\nwrote {written} selected-arm gravcomp sidecars "
          f"({skipped_prg} PRG case(s) skipped) -> audit {C.rel(C.ARM_SCENE_AUDIT_JSON)}")
    if failures:
        for f in failures:
            print(f"  FAIL {f}")
        raise SystemExit(f"C1 sidecar build FAILED for {len(failures)} rows")
    print("C1 PASS: every sidecar is a single-variable object-gravcomp diff vs the aug PRG scene")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
