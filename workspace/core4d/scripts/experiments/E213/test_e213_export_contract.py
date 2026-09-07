#!/usr/bin/env python3
"""Offline self-check for the E213-export contract (plan244 E0-E2/E5).

No CEM, no writes: validates the selection authority, unit resolution, partner
topology, variant-name mapping and dcv3 schema completeness against the on-disk
artifacts. Run before/after export_selected_arm_aug_rl.py.

    .venv/bin/python workspace/core4d/scripts/experiments/E213/test_e213_export_contract.py
"""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import e213_common as C  # noqa: E402
import e213_export_common as X  # noqa: E402

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    RESULTS.append((name, bool(cond), detail))


def main() -> int:
    units = X.load_units()
    seed = X.seed_index()
    man = X.source_manifest_index()
    pm = X.partner_manifest_index()
    e208 = X.e208_aug_index()
    source_ids = {u["case_id"] for u in units}

    # E0: authorities + counts
    check("E0.xlsx_sha_pinned", C.sha256(X.XLSX) == X.EXPECTED_XLSX_SHA256)
    check("E0.seed_sha_pinned", C.sha256(X.SEED_TSV) == X.EXPECTED_SEED_SHA256)
    check("E0.unit_count", len(units) == 32, f"{len(units)} units")
    check("E0.distinct_cases", len(source_ids) == 20, f"{len(source_ids)} cases")
    check("E0.cases_subset_of_e212", source_ids <= set(seed),
          f"missing={sorted(source_ids - set(seed))}")

    # E1: every unit has a cem_ok source rollout on its selected arm, arm scene matches
    e1_ok = e1_scene = 0
    for u in units:
        row = man.get((u["case_id"], u["variant"], u["arm"]))
        if row is None:
            check(f"E1.source_missing[{u['case_id']}/{u['variant']}/{u['arm']}]", False)
            continue
        e1_ok += 1
        arm_scene = seed[u["case_id"]]["arm_scene_name"]
        if arm_scene in Path(row["selected_scene_act"]).name:
            e1_scene += 1
    check("E1.all_source_rollouts", e1_ok == len(units), f"{e1_ok}/{len(units)}")
    check("E1.all_arm_scene_match", e1_scene == len(units), f"{e1_scene}/{len(units)}")

    # E2: every unit's partner variant resolves (two sources), files exist
    e2_ok = 0
    kinds = {"e213_partner_aug": 0, "e208_source_aug": 0}
    for u in units:
        partner = X.flip_case(u["case_id"])
        variant_u = X.VAR_UNDERSCORE[u["variant"]]
        try:
            aug = X.resolve_partner_aug(partner, variant_u, pm_idx=pm, e208_idx=e208,
                                        source_ids=source_ids)
            kinds[aug["source_kind"]] += 1
            e2_ok += 1
        except SystemExit as exc:  # noqa: PERF203
            check(f"E2.partner_resolve[{partner}/{variant_u}]", False, str(exc))
    check("E2.all_partner_resolve", e2_ok == len(units), f"{e2_ok}/{len(units)}")
    check("E2.partner_source_kinds", sum(kinds.values()) == len(units), str(kinds))

    # E5: schema field presence on the emitted paired export (if it exists)
    paired = X.OUT / "paired_rl_export_input.tsv"
    if paired.is_file():
        rows = C.read_tsv(paired)
        check("E5.paired_row_count", len(rows) == len(units), f"{len(rows)}")
        check("E5.schema_dcv3", all(r.get("schema_version") == "core4d_data_construction_v3.0"
                                    for r in rows))
        check("E5.all_ready", all(r.get("paired_rl_export_decision") == "RL_EXPORT_READY"
                                  for r in rows))
    else:
        check("E5.paired_export_present", False, "run export_selected_arm_aug_rl.py first")

    print(f"{'PASS':>6}  E213-export contract self-check")
    ok = True
    for name, passed, detail in RESULTS:
        flag = "ok " if passed else "FAIL"
        ok = ok and passed
        print(f"  [{flag}] {name}" + (f"  ({detail})" if detail else ""))
    print("\n" + ("ALL PASS" if ok else "SOME FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
