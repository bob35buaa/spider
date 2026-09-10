#!/usr/bin/env python3
"""E214 build the single 200-row ablation CEM manifest (50 cases x 4 ablations).

Case-major ordering (case0 x {A1..A4}, case1 x {A1..A4}, ...) so an interrupted
run still leaves a cross-section spanning all four ablations for the cases done.

Each row records the resolved baseline config_act.yaml (the load_config_path
target) + its scene/traj/mask + sha256, the ablation toggle string, and the E214
output paths.  Nothing is written outside results/E214 / tmp.

Usage:
    .venv/bin/python workspace/core4d/scripts/experiments/E214/build_manifest.py [--dry-run]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e214_common as C  # noqa: E402


def build_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    ordinal = 0
    unresolved: list[str] = []
    for case_id in C.load_cases():
        bp = C.baseline_paths(case_id)
        cfg = bp.get("config_act")
        if cfg is None:
            unresolved.append(f"{case_id}: {bp.get('note')}")
            continue
        cfg_sha = C.sha256(cfg)
        run_inputs = bp.get("run_inputs", {})
        if "model_path" not in run_inputs or "data_path" not in run_inputs:
            unresolved.append(f"{case_id}: run inputs missing ({bp.get('note')})")
            continue
        for abl in C.ABLATION_ORDER:
            rows.append({
                "ordinal": str(ordinal),
                "object_key": bp["object_key"],
                "case_id": case_id,
                "ablation": abl,
                "ablation_label": C.ABLATIONS[abl]["label"],
                "source_exp": bp["exp"],
                "baseline_config_act": C.rel(cfg),
                "baseline_config_act_sha256": cfg_sha,
                "baseline_cem_npz": C.rel(bp["cem_npz"]) if bp.get("cem_npz") else "",
                "run_model_path": C.rel(run_inputs["model_path"]),
                "run_data_path": C.rel(run_inputs["data_path"]),
                "run_contact_mask_path": C.rel(run_inputs["contact_hdmi_mask_path"])
                if run_inputs.get("contact_hdmi_mask_path") else "",
                "toggles": " ".join(C.toggle_tokens(abl)),
                "outdir_npz": C.rel(C.result_npz(case_id, abl)),
                "config_act": C.rel(C.config_act_out(case_id, abl)),
                "log": C.rel(C.cem_log_path(case_id, abl)),
                "gpu_id": "", "host": "", "status": "", "failure_mode": "",
                "wall_min": "", "updated_at": "",
            })
            ordinal += 1
    if unresolved:
        raise SystemExit("unresolved baselines:\n  " + "\n  ".join(unresolved))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    rows = build_rows()
    n_cases = len({r["case_id"] for r in rows})
    print(f"{C.EXP_ID} manifest: {len(rows)} rows | {n_cases} cases x {len(C.ABLATIONS)} ablations")
    by_abl: dict[str, int] = {}
    for r in rows:
        by_abl[r["ablation"]] = by_abl.get(r["ablation"], 0) + 1
    for abl, n in by_abl.items():
        print(f"  {abl:24s} {n} rows | toggles: {C.ABLATIONS[abl]['label']}")

    if args.dry_run:
        for r in rows[:8]:
            print(f"  [{r['ordinal']:>3}] {r['object_key']:9s} {r['ablation']:22s} "
                  f"{r['case_id']:30s} src={r['source_exp']} toggles=({r['toggles']})")
        return 0

    C.write_tsv(C.MANIFEST, rows, C.MANIFEST_FIELDS)
    print(f"wrote {C.rel(C.MANIFEST)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
