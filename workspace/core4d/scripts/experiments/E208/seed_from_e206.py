#!/usr/bin/env python3
"""E208 P1: seed each case's `_original` retarget artifacts from E206, byte-for-byte.

Why seed instead of recompute
-----------------------------
E206's F15 found the retarget pipeline is not reproducible everywhere: 19 of 22
re-run cases were bit-identical, 3 diverged.  **Two of those three are in the
E208 registry** -- ``chair005_20231030_043_p1`` (worst |dqpos| = 1.096, and the
*only* chair005 case, so n=1) and ``desk023_20231030_019_p1`` (0.097).  If we
recomputed `_original`, chair005's aug-vs-orig delta would mix the augmentation
effect with IK noise of comparable size and be unreadable.

Seeding removes the exposure by construction rather than by measurement.  It
works because of three upstream behaviours, all verified:

* ``parallel_robot_retarget.py:266-267`` short-circuits when an output npz
  already exists -- so the seeded ``_original`` is *reused as the warm-start
  source* for trans_0/1/2 and never recomputed.  (Hence: never pass ``--force``.)
* ``pipeline.sh:319`` skips the SMPL-X convert step when
  ``converted/{task}.npz`` exists.
* ``pipeline.sh:403`` skips the holosoma trim when ``trimmed/{task}_original.npz``
  exists -- and ``trim_no_contact.py:362`` only globs ``*_original.npz`` anyway,
  so aug variants were never going to be trimmed upstream (E208 does that itself
  with a fixed window).

Because ``trim_start`` is therefore identical to E206's, E206's 3cm contact mask
is valid frame-for-frame for every aug variant, so the driver runs pipeline.sh
with ``--skip-contact`` and the overrides keep pointing at E206's mask -- one
authority, no regenerated copy to drift from it.

Both retarget-variant roots are seeded: ``omnirt_v2`` needs the same
``_original`` so a rescue differs from the v1 pass in the solver contract *only*.

Usage:
    .venv/bin/python .../E208/seed_from_e206.py                  # both roots, all cases
    ... --cases a,b --variants omnirt_v1 --dry-run --force
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

import e208_common as C  # noqa: E402


def e206_root(retarget_variant: str) -> Path:
    """E206's RESULT_ROOT for one retarget variant (the dcv3 s3_retarget tree)."""
    return C.E206.S3_DIR / f"{retarget_variant}/ref_fk/results/{retarget_variant}_ref_fk"


def target_variants(case: dict[str, str], wanted: tuple[str, ...]) -> list[str]:
    """Which E208 variant roots this case needs.

    A case whose source is already omnirt_v2 has no v1 pass and nothing to
    escalate to, so it only ever lives under the v2 root.
    """
    if case["source_retarget_variant_id"] == "omnirt_v2":
        return [v for v in wanted if v == "omnirt_v2"]
    return list(wanted)


def link_or_copy(src: Path, dst: Path, *, force: bool, dry_run: bool) -> str:
    """Hardlink when possible (same storage volume), else copy2.

    Either way the bytes are identical, which is all R1-R4 assert; the hardlink
    is purely to avoid duplicating GBs of npz on the shared volume.
    """
    if dst.exists() and not force:
        return "exists"
    if dry_run:
        return "would_link"
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f".{dst.name}.e208tmp")
    tmp.unlink(missing_ok=True)
    try:
        os.link(src, tmp)
        mode = "hardlink"
    except OSError:
        shutil.copy2(src, tmp)
        mode = "copy"
    tmp.replace(dst)
    return mode


def seed_case(case: dict[str, str], variant: str, *, force: bool, dry_run: bool) -> dict[str, Any]:
    base = case["base_target_task"]
    meta = C.load_case_meta(base)
    task_name = meta["holosoma_task"]

    src_dir = e206_root(case["source_retarget_variant_id"]) / f"holosoma_{base}"
    dst_dir = C.holosoma_dir(base, variant)
    row: dict[str, Any] = {
        "case_id": case["case_id"], "object_key": case["object_key"],
        "base_target_task": base, "variant_root": variant,
        "holosoma_task": task_name,
        "src": C.rel(src_dir), "dst": C.rel(dst_dir),
    }
    if not src_dir.is_dir():
        row["status"] = "fail_missing_source"
        row["error"] = f"no E206 case dir: {src_dir}"
        return row

    actions: dict[str, str] = {}

    # 1. converted/ -- all three npz (person, object, person+object); the guard
    #    pipeline.sh checks is `{task_name}.npz`, but the retargeter's
    #    find_files() globs the whole dir, so seed it wholesale.
    src_conv = src_dir / "converted"
    if not src_conv.is_dir():
        row["status"] = "fail_missing_converted"
        return row
    for src in sorted(src_conv.glob("*.npz")):
        actions[f"converted/{src.name}"] = link_or_copy(
            src, dst_dir / "converted" / src.name, force=force, dry_run=dry_run)

    # 2/3. the two `_original` npz -- the whole point of this script
    for sub in ("retargeted", "trimmed"):
        src = src_dir / sub / f"{task_name}_original.npz"
        if not src.is_file():
            row["status"] = f"fail_missing_{sub}_original"
            row["error"] = str(src)
            return row
        actions[f"{sub}/{src.name}"] = link_or_copy(
            src, dst_dir / sub / src.name, force=force, dry_run=dry_run)

    # 4. trim_window.json -- copied (not linked) because pipeline.sh will
    #    overwrite it with re-inferred numbers and E208-local paths; keeping our
    #    own file means a hardlink would corrupt E206's copy.
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
        shutil.copy2(src_tw, dst_tw)
        actions["trim_window.json"] = "copy"

    payload = json.loads(src_tw.read_text(encoding="utf-8"))
    row["trim_start"] = payload.get("trim_start")
    row["trim_frames"] = payload.get("trim_frames")
    row["untrimmed_frames"] = payload.get("untrimmed_frames")
    row["actions"] = actions
    row["n_files"] = len(actions)
    row["status"] = "seeded"
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="", help="comma-separated case_ids (default: all 22)")
    ap.add_argument("--variants", default=",".join(C.RETARGET_VARIANTS),
                    help="which E208 variant roots to seed (default: both)")
    ap.add_argument("--force", action="store_true", help="re-link even if the target exists")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--json-out", type=Path, default=C.PREFLIGHT_DIR / "orig_seed_report.json")
    args = ap.parse_args()

    wanted = tuple(v.strip() for v in args.variants.split(",") if v.strip())
    for variant in wanted:
        if variant not in C.RETARGET_VARIANTS:
            raise SystemExit(f"unknown retarget variant {variant!r}")

    cases = C.load_e208_cases()
    if args.cases:
        keep = {c.strip() for c in args.cases.split(",") if c.strip()}
        unknown = keep - {c["case_id"] for c in cases}
        if unknown:
            raise SystemExit(f"unknown case_ids: {sorted(unknown)}")
        cases = [c for c in cases if c["case_id"] in keep]

    rows: list[dict[str, Any]] = []
    for case in cases:
        for variant in target_variants(case, wanted):
            row = seed_case(case, variant, force=args.force, dry_run=args.dry_run)
            rows.append(row)
            print(f"[{row['status']:22s}] {row['case_id']:32s} {variant:10s} "
                  f"files={row.get('n_files', 0):2d} trim_start={row.get('trim_start', '?')}",
                  flush=True)

    failures = [r for r in rows if r["status"] != "seeded"]
    modes: dict[str, int] = {}
    for row in rows:
        for mode in row.get("actions", {}).values():
            modes[mode] = modes.get(mode, 0) + 1

    payload = {
        "experiment": C.EXP_ID, "generated_at": C.now(),
        "dry_run": args.dry_run, "variants": list(wanted),
        "n_cases": len(cases), "n_seeded": len(rows) - len(failures),
        "n_failures": len(failures), "action_modes": modes, "rows": rows,
    }
    C.write_json(args.json_out, payload)
    print(f"\nseeded {len(rows) - len(failures)}/{len(rows)} case-variants, "
          f"modes={modes} -> {C.rel(args.json_out)}")
    if failures:
        for row in failures:
            print(f"  FAIL {row['case_id']}/{row['variant_root']}: "
                  f"{row['status']} {row.get('error', '')}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
