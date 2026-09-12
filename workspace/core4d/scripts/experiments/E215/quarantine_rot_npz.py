#!/usr/bin/env python3
"""E215: quarantine rot_* retarget outputs of uncertain provenance.

If two ``run_upstream_retarget.py`` instances ever race (the per-object files
pipeline.sh rewrites are the hazard the driver's flock + object-serial grouping
exist to prevent), the rot npz produced during the overlap window have unclear
provenance.  Move -- not delete -- them so the incident stays auditable and the
clean re-run can be diffed against them.  ``_original`` and the seeded npz are
never touched (upstream short-circuits on existing outputs).

Usage:
    .venv/bin/python .../E215/quarantine_rot_npz.py --reason "<why>" [--dry-run]
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import e215_common as C  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reason", default="concurrent-runner incident")
    ap.add_argument("--quarantine-dir", type=Path, default=C.DP / "quarantine/rot_npz")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    root = args.quarantine_dir

    def _live(paths):
        return [q for q in paths if root not in q.parents]

    hits = sorted(_live(C.DP.rglob("*_rot_*.npz")))
    trans = len(_live(C.DP.rglob("*_trans_*.npz")))
    orig = len(_live(C.DP.rglob("*_original.npz")))
    print(f"rot npz to quarantine: {len(hits)} (keeping trans={trans} original={orig})")

    moved = 0
    for src in hits:
        rel = src.relative_to(C.DP)
        dst = args.quarantine_dir / rel
        print(f"  {'would move' if args.dry_run else 'move'} {rel}")
        if args.dry_run:
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        moved += 1

    if not args.dry_run and hits:
        (args.quarantine_dir / "README.md").write_text(
            f"# Quarantined E215 rot_* retarget outputs\n\n"
            f"- moved_at: {C.now()}\n- reason: {args.reason}\n- count: {moved}\n\n"
            "Provenance uncertain (concurrent run_upstream_retarget instances raced on the\n"
            "per-object files pipeline.sh rewrites). Kept, not deleted, for audit.\n",
            encoding="utf-8")

    remaining = len(_live(C.DP.rglob("*_rot_*.npz")))
    print(f"\nmoved {moved}, remaining rot npz under DP: {remaining}")
    if not args.dry_run:
        print(f"quarantine -> {C.rel(args.quarantine_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
