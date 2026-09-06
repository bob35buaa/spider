#!/usr/bin/env python3
"""E208: quarantine rot_* retarget outputs of uncertain provenance.

Incident, 2026-09-05: a second `run_upstream_retarget.py --pass pass1
--force-rerun` was launched while the first was still running (the wait loop
polled a fixed number of times instead of waiting for the process to actually
exit, and `kill -0` also succeeds on a zombie).  Two instances then raced on the
per-object files `pipeline.sh` rewrites -- `sync_generated_object_model`'s
`cp -a` and `ensure_g1_object_xml`'s first-write -- which is exactly the hazard
`run_upstream_retarget.py` groups cases by object to avoid.

Scope of the damage, established by mtime rather than assumed:

* ``_original`` -- untouched.  Upstream short-circuits on an existing output
  (`parallel_robot_retarget.py:300`), so it is never rewritten.  C1 stands.
* ``*_trans_*`` -- untouched.  Every probe case's trans npz has an mtime between
  00:51 and 01:21, all before the second instance started at 01:40, so the
  short-circuit held and no trans file was rewritten.  63/66 trans results stand.
* ``*_rot_*`` -- **uncertain**.  These span 01:31-01:56, straddling the overlap
  window, and some were produced by the killed instance.  They are also the only
  artifacts whose determinism has not been measured (that is probe R6 / gate G4).

So only rot_* is quarantined.  Moving rather than deleting keeps the incident
auditable and lets the re-run be compared against it if the question of aug-IK
determinism comes up.

Usage:
    .venv/bin/python .../E208/quarantine_rot_npz.py --reason "<why>"
    ... --dry-run
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import e208_common as C  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reason", default="concurrent-runner incident 2026-09-05")
    ap.add_argument("--quarantine-dir", type=Path,
                    default=C.DP / "quarantine/rot_npz_2026-09-05_concurrent_runners")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    quarantine_root = args.quarantine_dir
    def _live(paths):
        # the quarantine dir lives under DP, so exclude it from every count
        return [q for q in paths if quarantine_root not in q.parents]
    hits = sorted(_live(C.DP.rglob("*_rot_*.npz")))
    trans = len(_live(C.DP.rglob("*_trans_*.npz")))
    orig = len(_live(C.DP.rglob("*_original.npz")))
    print(f"rot npz to quarantine: {len(hits)}")
    print(f"  keeping trans: {trans}   original: {orig}")

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
            f"# Quarantined rot_* retarget outputs\n\n"
            f"- moved_at: {C.now()}\n- reason: {args.reason}\n- count: {moved}\n\n"
            "Provenance is uncertain: two `run_upstream_retarget.py` instances ran\n"
            "concurrently and raced on the per-object files pipeline.sh rewrites.\n"
            "`_original` and `*_trans_*` were verified untouched by mtime (all predate\n"
            "the overlap window) and were NOT quarantined.\n\n"
            "These files are kept, not deleted, so the clean re-run can be diffed\n"
            "against them if aug-IK determinism (probe R6 / gate G4) is ever measured.\n",
            encoding="utf-8")

    remaining = len(_live(C.DP.rglob("*_rot_*.npz")))
    print(f"\nmoved {moved}, remaining rot npz under DP: {remaining}")
    if not args.dry_run:
        print(f"quarantine -> {C.rel(args.quarantine_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
