#!/usr/bin/env python3
"""E215 P1: seed each case's `_original` retarget artifacts into the E215 tree.

Unlike E208 (which seeded from E206's s3_retarget tree), E215's warm start
already exists as a finished FLAT ``data_preprocess`` tree from E199 (box) or
E202 (bucket): ``converted/`` + ``retargeted/trimmed`` ``_original`` (+ the trans
variants) + ``trim_window.json``.  Those ``_original`` npz are the omnirt_v2
retargets E199/E202 froze for every variant, so seeding them into the E215 tree
is a v2->v2 warm start -- the rot aug then differs from the baseline in the
augmentation config ALONE.

Why seed instead of retarget-in-place
-------------------------------------
* ``parallel_robot_retarget.py`` short-circuits when an output npz already
  exists, so the seeded ``_original`` is *reused as the IK warm start* for rot_0
  and rot_1 and is never recomputed.  (Hence: run_upstream_retarget never passes
  ``--force``.)
* Seeding into a NEW E215 tree keeps the historical E199/E202 trees untouched --
  the rot npz land next to a copy of ``_original``, and provenance stays clean
  even if two drivers race.

Hardlink when possible (same volume, identical bytes), copy2 otherwise.

Usage:
    .venv/bin/python .../E215/seed_warmstart.py                 # all buildable cases
    ... --cases a,b --dry-run --force
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import e215_common as C  # noqa: E402


def link_or_copy(src: Path, dst: Path, *, force: bool, dry_run: bool) -> str:
    if dst.exists() and not force:
        return "exists"
    if dry_run:
        return "would_link"
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f".{dst.name}.e215tmp")
    tmp.unlink(missing_ok=True)
    try:
        os.link(src, tmp)
        mode = "hardlink"
    except OSError:
        shutil.copy2(src, tmp)
        mode = "copy"
    tmp.replace(dst)
    return mode


def seed_case(case: dict[str, str], *, force: bool, dry_run: bool) -> dict[str, Any]:
    case_id = case["case_id"]
    base = case["base_target_task"]
    meta = C.load_case_meta(base)
    holo = meta["holosoma_task"]

    src_dir = C.seed_source_dir(case_id)
    dst_dir = C.holosoma_dir(case_id)
    row: dict[str, Any] = {
        "case_id": case_id, "object_key": case["object_key"],
        "base_target_task": base, "holosoma_task": holo,
        "src": C.rel(src_dir), "dst": C.rel(dst_dir),
    }
    if not src_dir.is_dir():
        row["status"] = "fail_missing_source"
        row["error"] = f"no warm-start dir: {src_dir}"
        return row

    actions: dict[str, str] = {}

    src_conv = src_dir / "converted"
    if not src_conv.is_dir():
        row["status"] = "fail_missing_converted"
        return row
    for src in sorted(src_conv.glob("*.npz")):
        actions[f"converted/{src.name}"] = link_or_copy(
            src, dst_dir / "converted" / src.name, force=force, dry_run=dry_run)

    for sub in ("retargeted", "trimmed"):
        src = src_dir / sub / f"{holo}_original.npz"
        if not src.is_file():
            row["status"] = f"fail_missing_{sub}_original"
            row["error"] = str(src)
            return row
        actions[f"{sub}/{src.name}"] = link_or_copy(
            src, dst_dir / sub / src.name, force=force, dry_run=dry_run)

    # Also seed the trans_* RETARGETED npz so upstream short-circuits them
    # (parallel_robot_retarget skips a variant whose output already exists) and
    # computes ONLY rot_0/rot_1 -- ~60% less IK, and it matches the plan's
    # "produce only rot npz".  These are byte-identical hardlinks from E199/E202
    # and are never consumed downstream (build_augmented_tasks trims rot only);
    # a case whose trans was infeasible simply has none to seed (harmless).
    for holo_name in ("trans_0", "trans_1", "trans_2"):
        src = src_dir / "retargeted" / f"{holo}_{holo_name}.npz"
        if src.is_file():
            actions[f"retargeted/{src.name}"] = link_or_copy(
                src, dst_dir / "retargeted" / src.name, force=force, dry_run=dry_run)

    src_tw = src_dir / "trim_window.json"
    if not src_tw.is_file():
        row["status"] = "fail_missing_trim_window"
        return row
    dst_tw = dst_dir / "trim_window.json"
    if dry_run:
        actions["trim_window.json"] = "would_copy"
    elif dst_tw.exists() and not force:
        actions["trim_window.json"] = "exists"
    else:
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_tw, dst_tw)  # copied: pipeline.sh may rewrite it
        actions["trim_window.json"] = "copy"

    payload = json.loads(src_tw.read_text(encoding="utf-8"))
    row["trim_start"] = payload.get("trim_start")
    row["trim_frames"] = payload.get("trim_frames")
    row["actions"] = actions
    row["n_files"] = len(actions)
    row["status"] = "seeded"
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="", help="comma-separated case_ids (default: all buildable)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--json-out", type=Path, default=C.SEED_REPORT_JSON)
    args = ap.parse_args()

    cases = C.load_e215_cases()
    if args.cases:
        keep = {c.strip() for c in args.cases.split(",") if c.strip()}
        unknown = keep - {c["case_id"] for c in cases}
        if unknown:
            raise SystemExit(f"unknown/non-buildable case_ids: {sorted(unknown)}")
        cases = [c for c in cases if c["case_id"] in keep]

    rows: list[dict[str, Any]] = []
    for case in cases:
        row = seed_case(case, force=args.force, dry_run=args.dry_run)
        rows.append(row)
        print(f"[{row['status']:22s}] {row['case_id']:32s} files={row.get('n_files', 0):2d} "
              f"trim_start={row.get('trim_start', '?')}", flush=True)

    failures = [r for r in rows if r["status"] != "seeded"]
    modes: dict[str, int] = {}
    for row in rows:
        for mode in row.get("actions", {}).values():
            modes[mode] = modes.get(mode, 0) + 1

    C.write_json(args.json_out, {
        "experiment": C.EXP_ID, "generated_at": C.now(), "dry_run": args.dry_run,
        "n_cases": len(cases), "n_seeded": len(rows) - len(failures),
        "n_failures": len(failures), "action_modes": modes, "rows": rows,
    })
    print(f"\nseeded {len(rows) - len(failures)}/{len(rows)} cases, modes={modes} "
          f"-> {C.rel(args.json_out)}")
    for row in failures:
        print(f"  FAIL {row['case_id']}: {row['status']} {row.get('error', '')}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
