#!/usr/bin/env python3
"""E215 preflight: locate each case's SAME-ARM orig baseline rollout, report gaps.

The rot variants are compared (C4) against the orig CEM of the identical case
and arm.  Per plan247 3.3 those baselines live in different experiments:

    bucket_prg           bucket003   E202 (/E178) PRG orig
    bucket_prg_gravcomp  bucket007   E210 PRG+G1 orig
    box_prg              box021      E199 full-scale PRG orig
    box_noprg            box023      E200 noPRG orig (E190 lineage)
    box_prg_g1a2         box001/4/24 E200 prg_g1a2 orig (E198 G1A2 lineage)

This audit searches a per-group list of candidate manifests for an orig row of
the case, records the ``result_npz`` + retarget variant, and checks the npz
exists.  It does NOT decide anything: it emits the gap list so the user can
choose whether to re-run any ``orig x arm`` baseline (plan says default = reuse,
only backfill on a confirmed gap).  Retarget-variant parity is reported too --
E215 rot is v2, and a v1 orig baseline would be a single-variable confound to
flag (not auto-fix).

Usage:
    .venv/bin/python .../E215/preflight_baseline_audit.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import e215_common as C  # noqa: E402

RES = C.REPO / "workspace/core4d/results"

# Same-arm NO-AUGMENTATION orig baseline source per arm group (2026-09-12 dig).
# Each entry: (manifest, arm_column_value|None, note).  These are the source
# experiments that actually ran the orig (un-augmented) full CEM -- NOT the
# E199/E202/E200/E210 aug manifests (those hold only trans/rot rows).  case_id is
# p{N} in all of them (matches E215), so no format juggling is needed.
#   E198 four-arm cache: one row per (case, arm) orig CEM; arm in {A0,A2,G1,G1A2}.
#     box_prg (E199 PRG)          -> A0    (E199 fullscale's own orig source)
#     box_noprg (E167A rubberHull)-> A0    (base, no gravcomp/gate/leg-pairs)
#     box_prg_g1a2 (E200 prg_g1a2)-> G1A2  (E200's own orig source)
#   E178 bucket full: orig contact-aligned PRG CEM (no aug_variant column).
#     bucket_prg (bucket003)          -> E178 (clean, same PRG arm)
#     bucket_prg_gravcomp (bucket007) -> E178 (NO gravcomp: arm differs by gravcomp
#                                       -- flagged; no bucket007 PRG+G1 orig exists)
E198_ARM_CACHE = "E198/s6_downstream/eval/full_factorial/e198_arm_cache.tsv"
E178_BUCKET = "E178/s6_downstream/manifests/semantic_bucket_full_manifest.tsv"

CANDIDATES: dict[str, list[tuple[str, str | None, str]]] = {
    "box_prg": [(E198_ARM_CACHE, "A0", "clean")],
    "box_noprg": [(E198_ARM_CACHE, "A0", "clean")],
    "box_prg_g1a2": [(E198_ARM_CACHE, "G1A2", "clean")],
    "bucket_prg": [(E178_BUCKET, None, "clean")],
    "bucket_prg_gravcomp": [(E178_BUCKET, None, "arm_differs_by_gravcomp")],
}


def _result_npz(row: dict[str, str]) -> str:
    for key in ("result_npz", "cem_result_npz", "outdir_npz"):
        if row.get(key):
            return row[key]
    return ""


def _scene(row: dict[str, str]) -> str:
    return row.get("scene_act") or row.get("scene_xml") or ""


def find_baseline(case: dict[str, str]) -> dict[str, Any]:
    """Locate the case's same-arm orig CEM row (full record for eval reuse)."""
    group = case["arm_group"]
    case_id = case["case_id"]
    for rel_path, arm_filter, note in CANDIDATES.get(group, []):
        tsv = RES / rel_path
        if not tsv.is_file():
            continue
        for row in C.read_tsv(tsv):
            if row.get("case_id") != case_id:
                continue
            if arm_filter is not None and row.get("arm") != arm_filter:
                continue
            npz, scene = _result_npz(row), _scene(row)
            exists = bool(npz) and C.repo_path(npz).is_file()
            rv = row.get("retarget_variant_id") or row.get("selected_retarget_variant_id") \
                or row.get("effective_retarget_variant", "")
            return {
                "found_in": rel_path, "arm_filter": arm_filter or "", "arm_note": note,
                "result_npz": npz, "scene_act": scene,
                "trajectory": row.get("trajectory", ""), "contact_mask": row.get("contact_mask", ""),
                "npz_exists": exists, "retarget_variant_id": rv,
                "scene_name": row.get("scene_name", ""),
            }
    return {"found_in": "", "arm_filter": "", "arm_note": "", "result_npz": "", "scene_act": "",
            "trajectory": "", "contact_mask": "", "npz_exists": False,
            "retarget_variant_id": "", "scene_name": ""}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-out", type=Path, default=C.BASELINE_AUDIT_JSON)
    args = ap.parse_args()

    cases = C.load_e215_cases()
    rows: list[dict[str, Any]] = []
    for case in cases:
        base = find_baseline(case)
        rows.append({
            "case_id": case["case_id"], "object_key": case["object_key"],
            "arm_group": case["arm_group"], "base_variant": case["base_variant"],
            **base,
            "variant_parity": (base["retarget_variant_id"] in ("", "omnirt_v2")),
        })

    found = [r for r in rows if r["npz_exists"]]
    missing = [r for r in rows if not r["npz_exists"]]
    v1_baseline = [r for r in found if r["retarget_variant_id"] == "omnirt_v1"]

    by_group_missing: dict[str, list[str]] = {}
    for r in missing:
        by_group_missing.setdefault(r["arm_group"], []).append(r["case_id"])

    C.write_json(args.json_out, {
        "experiment": C.EXP_ID, "generated_at": C.now(),
        "n_cases": len(cases), "n_baseline_found": len(found), "n_missing": len(missing),
        "missing_by_group": by_group_missing,
        "n_v1_baseline_confound": len(v1_baseline),
        "v1_baseline_cases": [r["case_id"] for r in v1_baseline],
        "rows": rows,
    })

    print(f"baseline found {len(found)}/{len(cases)} -> {C.rel(args.json_out)}")
    if missing:
        print(f"  MISSING same-arm orig baseline ({len(missing)}):")
        for group, cids in sorted(by_group_missing.items()):
            print(f"    {group}: {cids}")
    if v1_baseline:
        print(f"  WARN v1 orig baseline (retarget-variant confound vs v2 rot) ({len(v1_baseline)}): "
              f"{[r['case_id'] for r in v1_baseline]}")
    print("  NOTE: audit only reports; backfilling any orig x arm CEM is a user decision.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
