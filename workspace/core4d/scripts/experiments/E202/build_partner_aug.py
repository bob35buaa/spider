#!/usr/bin/env python3
"""E202-export step 3: ensure the opposite-person (partner) aug retarget exists.

Object augmentation perturbs the SHARED object approach; a physically consistent
paired RL export therefore needs the partner (opposite CORE4D person of the same
object/date/seq) retargeted under the IDENTICAL trans_k perturbation. E202 already
ran the aug retarget for 25/27 E178 full-CEM persons, so for the 13 manual-USE
source cases 10 partners are pure REUSE and only 2 partners (059_p2, 073_p2) need a
fresh build.

Partner side is retarget-only (no CEM, same as log264): we run the E202 upstream
(`pipeline.sh --skip-spider`, omnirt_v2) + fixed-window trim, producing
`converted/ retargeted/ trimmed/ trim_window.json` in the SAME E202 data_preprocess
layout the export evidence resolver expects. We deliberately do NOT build the CEM
PRG scene / SPIDER task for the partner (that runtime-overlap gate is what blocked
075_p2/021_p2 as CEM *sources*; it is irrelevant to a retarget-only partner).

Idempotent: reused partners are only verified; missing partners are built once.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e202_common as C  # noqa: E402


def _load_e202_module(name: str):
    """Load an E202-local module by explicit path.

    `e202_common` inserts the E199 dir at sys.path[0], so a bare
    `import build_augmented_tasks` would resolve to E199's sibling module. Load
    E202's copy by file path to avoid that collision.
    """
    path = Path(__file__).resolve().parent / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"e202_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BAT = _load_e202_module("build_augmented_tasks")  # E202's, not E199's

# opposite CORE4D person suffix
_FLIP = {"p1": "p2", "p2": "p1"}


def partner_case_id(source_case_id: str) -> str:
    base, _, suffix = source_case_id.rpartition("_")
    if suffix not in _FLIP:
        raise SystemExit(f"unexpected person suffix in {source_case_id}")
    return f"{base}_{_FLIP[suffix]}"


def partner_base_task(source_base_task: str) -> str:
    # dcv3_omnirt_v1_ref_fk_bucket007_20231020_059_p1 -> ..._p2
    base, _, suffix = source_base_task.rpartition("_")
    if suffix not in _FLIP:
        raise SystemExit(f"unexpected person suffix in base task {source_base_task}")
    return f"{base}_{_FLIP[suffix]}"


def trimmed_trans_present(base_task: str, meta: dict[str, str]) -> dict[str, bool]:
    """Which trans variants have a trimmed NPZ under the E202 partner dir."""
    root = BAT.case_root(base_task)
    holo = meta["holosoma_task"]
    present: dict[str, bool] = {}
    for _e202_name, holo_name in C.TRANS_VARIANTS:
        p = root / "trimmed" / f"{holo}_{holo_name}.npz"
        present[holo_name] = p.is_file() and p.stat().st_size > 0
    return present


def ensure_partner(source_case_id: str, source_base_task: str, object_key: str,
                   *, force: bool, max_workers: int) -> dict[str, object]:
    p_case = partner_case_id(source_case_id)
    p_base = partner_base_task(source_base_task)
    meta = C.load_case_meta(p_base)
    holo = meta["holosoma_task"]
    root = BAT.case_root(p_base)
    trim_json = root / "trim_window.json"

    present = trimmed_trans_present(p_base, meta)
    already = trim_json.is_file() and all(present.values())
    mode = "reuse"
    if not already or force:
        mode = "build"
        print(f"[partner build] {p_case} (base={p_base})", flush=True)
        BAT.run_upstream(p_base, meta, force=force, max_workers=max_workers)
        trim_start, feasible = BAT.fixed_window_trim(p_base, meta, C.TRANS_VARIANTS)
        present = trimmed_trans_present(p_base, meta)
        print(f"[partner build] {p_case} trim_start={trim_start} feasible={sorted(feasible)}", flush=True)
    else:
        print(f"[partner reuse] {p_case}: trimmed trans present {present}", flush=True)

    if not trim_json.is_file():
        raise SystemExit(f"partner {p_case}: trim_window.json missing after {mode}")
    return {
        "source_case_id": source_case_id,
        "partner_case_id": p_case,
        "partner_base_task": p_base,
        "object_key": object_key,
        "holosoma_task": holo,
        "case_root": C.rel(root),
        "trim_window_json": C.rel(trim_json),
        "mode": mode,
        "trans_present": {k: bool(v) for k, v in present.items()},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="force rebuild even if present")
    ap.add_argument("--max-workers", type=int, default=1)
    ap.add_argument("--only", default="", help="comma source case_ids to restrict")
    args = ap.parse_args()

    import e202_export_common as X  # local import to avoid cycle at module load

    only = {s.strip() for s in args.only.split(",") if s.strip()}
    results = []
    for case in X.source_use_cases():
        if only and case["case_id"] not in only:
            continue
        results.append(ensure_partner(
            case["case_id"], case["base_target_task"], case["object_key"],
            force=args.force, max_workers=args.max_workers,
        ))
    out = X.PARTNER_AUG_MANIFEST
    C.write_json(out, results)
    built = [r["partner_case_id"] for r in results if r["mode"] == "build"]
    reused = [r["partner_case_id"] for r in results if r["mode"] == "reuse"]
    print(f"\n[done] partners: {len(results)} total, built={len(built)} {built}, reused={len(reused)}")
    print(f"       manifest -> {C.rel(out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
