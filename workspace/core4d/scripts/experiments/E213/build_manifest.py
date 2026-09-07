#!/usr/bin/env python3
"""E213 Phase A step 3: build the source-side selected-arm CEM manifest + 4 shards.

One row per (case, aug variant) for the 20 non-PRG cases (the 1 PRG-selected case
reuses E208's PRG aug rollouts and is recorded separately, not re-run).  Every
input path carries its sha256 so the runner's preflight refuses to launch against
a drifted scene / trajectory / mask / override.

Four 8-GPU machines share this /mnt.  The runner rewrites the whole manifest on
every status change, so rows are split into 4 disjoint shard files (A local, B/C/D
remote); merge_shards.py reconciles them.  The split is a round-robin over a
canonical (arm, object, case, variant) order, so every shard spans every arm and
a dead machine leaves an interpretable cross-section.

Usage:
    .venv/bin/python workspace/core4d/scripts/experiments/E213/build_manifest.py
    ... --cases <id,...>
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import e213_common as C  # noqa: E402

ARM_PRIORITY = {"G1": 0, "G08": 1, "G06": 2, "G04": 3, "PRG": 9}


def build(cases: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[str]]:
    aug = C.aug_rows_by_case()
    budget = C.frozen_budget()
    reuse_prg: list[str] = []
    raw: list[dict[str, str]] = []

    for case in cases:
        case_id = case["case_id"]
        if C.is_prg_case(case):
            for row in aug.get(case_id, []):
                reuse_prg.append(f"{case_id}/{row['aug_variant']} -> "
                                 + C.rel(C.e208_prg_result_npz(case_id, row['aug_variant'])))
            continue
        arm = case["arm"]
        for row in aug.get(case_id, []):
            variant = row["aug_variant"]
            selected = C.aug_selected_scene(row, case)
            if not selected.is_file():
                raise SystemExit(
                    f"{case_id}/{variant}: missing selected sidecar {C.rel(selected)}; "
                    "run build_source_arm_scenes.py first")
            override = C.override_path(case_id, variant, arm)
            if not override.is_file():
                raise SystemExit(
                    f"{case_id}/{variant}: missing override {C.rel(override)}; "
                    "run build_source_overrides.py first")
            trajectory = C.repo_path(row["trajectory"])
            contact = C.repo_path(row["contact_mask"])
            for label, p in (("trajectory", trajectory), ("contact_mask", contact),
                             ("target_scene", C.repo_path(row["target_scene"]))):
                if not p.is_file():
                    raise SystemExit(f"{case_id}/{variant}: missing {label}: {C.rel(p)}")

            out = C.cem_out_dir(case_id, variant, arm)
            raw.append({
                "object_key": case["object_key"], "case_id": case_id,
                "aug_variant": variant, "arm": arm,
                "arm_gravcomp": case["arm_gravcomp"], "arm_experiment": case["arm_experiment"],
                "base_target_task": case["base_target_task"],
                "target_task": row["target_task"],
                "e208_prg_override_id": C.e208_prg_override_id(case_id, variant),
                "override_id": C.override_id(case_id, variant, arm),
                "override_path": C.rel(override), "override_sha256": C.sha256(override),
                "scene_name": case["arm_scene_name"],
                "selected_scene_act": C.rel(selected), "selected_scene_sha256": C.sha256(selected),
                "trajectory": row["trajectory"], "trajectory_sha256": C.sha256(trajectory),
                "contact_mask": row["contact_mask"], "contact_mask_sha256": C.sha256(contact),
                "target_scene": row["target_scene"], "base_scene_act": row["scene_act"],
                "effective_retarget_variant": row.get("effective_retarget_variant", ""),
                "object_geom_count": row.get("object_geom_count", ""),
                "offset_band": row.get("offset_band", ""),
                "approach_trans_offset_m_max": row.get("approach_trans_offset_m_max", ""),
                "aug_translation": row.get("aug_translation", ""),
                "aug_rotation_rad": row.get("aug_rotation_rad", ""),
                "cem_samples": str(budget["num_samples"]),
                "cem_opt_steps": str(budget["max_num_iterations"]),
                "cem_seed": str(budget["seed"]),
                "orig_result_npz": case["orig_result_npz"],
                "result_npz": C.rel(C.result_npz(case_id, variant, arm)),
                "outdir_npz": C.rel(C.result_npz(case_id, variant, arm)),
                "config_act": C.rel(out / "config_act.yaml"),
                "log": C.rel(C.cem_log_path(case_id, variant, arm)),
                "execution_mode": "production", "gpu_id": "", "host": "",
                "status": "READY_FOR_FULL", "failure_mode": "", "wall_min": "",
                "updated_at": C.now(),
            })

    # canonical order + round-robin shard assignment
    raw.sort(key=lambda r: (ARM_PRIORITY.get(r["arm"], 5), r["object_key"], r["case_id"], r["aug_variant"]))
    for i, r in enumerate(raw):
        r["ordinal"] = str(i)
        r["shard"] = C.SHARDS[i % C.N_SHARDS]
    return raw, reuse_prg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="")
    args = ap.parse_args()

    cases = C.load_cases()
    if args.cases:
        keep = {c.strip() for c in args.cases.split(",") if c.strip()}
        cases = [c for c in cases if c["case_id"] in keep]

    rows, reuse_prg = build(cases)
    C.MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    C.write_tsv(C.SOURCE_MANIFEST, rows, C.SOURCE_FIELDS)
    for shard in C.SHARDS:
        subset = [r for r in rows if r["shard"] == shard]
        if subset:
            C.write_tsv(C.manifest_path(shard), subset, C.SOURCE_FIELDS)

    by_shard = Counter(r["shard"] for r in rows)
    by_arm = Counter(r["arm"] for r in rows)
    print(f"wrote {len(rows)} source CEM rows -> {C.rel(C.SOURCE_MANIFEST)}")
    print(f"  arms={dict(by_arm)}")
    for s in C.SHARDS:
        arms_in = Counter(r["arm"] for r in rows if r["shard"] == s)
        print(f"  shard {s}: {by_shard[s]:2d} rows  arms={dict(arms_in)} -> {C.rel(C.manifest_path(s))}")
    print(f"  reuse-E208-PRG rows (not re-run): {len(reuse_prg)}")
    for r in reuse_prg:
        print(f"    {r}")
    # distinct-scene sanity: no two rows share an effective scene hash
    shas = [r["selected_scene_sha256"] for r in rows]
    print(f"  distinct selected-scene hashes: {len(set(shas))}/{len(shas)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
