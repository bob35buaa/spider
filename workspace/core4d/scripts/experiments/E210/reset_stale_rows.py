#!/usr/bin/env python3
"""E210: recover manifest rows stranded in `running` by a killed queue instance.

The E199 priority queue is resume-safe for *finished* work -- on restart it
re-detects completed rollouts from their output files. It is NOT resume-safe for
*interrupted* work: ``ELIGIBLE`` (run_local_priority_queue.py:30) excludes
``running``, so a row whose queue process died after the status write-back but
before the child finished becomes a tombstone. A resumed queue silently skips it
forever and reports "0 pending" as if the manifest were complete.

That is what happened to E210's first wave: 8 rows dispatched at 23:48:56-59 got
``running`` written, their queue died within ~2.5 min (the next wave reused the
same GPU ids at 23:51:22), and the relaunched queue then ran only the remaining
7. Nothing marked the 8 as failed.

This resets such rows to pending so a relaunch picks them up. Safety rails,
because "the job is running on another machine that shares /mnt" is a real
configuration here and a wrong reset means two processes writing one output dir:

  * a row whose outputs all exist is NOT reset (it finished -- let the queue's
    own resume logic reclassify it);
  * a row whose log has grown past the header, or was touched within
    --stale-minutes, is NOT reset (it may be alive elsewhere);
  * dry-run by default; --apply is required to write.

Usage:
    .venv/bin/python .../E210/reset_stale_rows.py                 # dry-run
    .venv/bin/python .../E210/reset_stale_rows.py --apply
    ... --stale-minutes 30 --stage full
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e210_common as C  # noqa: E402

#: A queue log with only the three `# ...` header lines means the child never
#: produced a byte. run_mjwp runs under `python -u`, so a live job always has more.
HEADER_LINES = 3


def classify(row: dict[str, str], stale_minutes: float) -> tuple[str, str]:
    """Return (action, reason) for one row. action in {reset, keep, complete}."""
    if row["status"] != "running":
        return "keep", f"status={row['status']!r} is not 'running'"

    outs = {k: C.repo_path(row[k]) for k in ("result_npz", "outdir_npz", "config_act")}
    present = [k for k, p in outs.items() if p.is_file()]
    if len(present) == len(outs):
        return "complete", "all outputs present -- queue resume will reclassify it"
    if present:
        return "keep", f"partial outputs {present} -- inspect by hand, do not reset blindly"

    log = C.repo_path(row["log"])
    if not log.is_file():
        return "reset", "no log and no outputs"

    body = [ln for ln in log.read_text(errors="replace").splitlines() if ln.strip()]
    age_min = (time.time() - log.stat().st_mtime) / 60.0
    if len(body) > HEADER_LINES:
        return "keep", f"log has {len(body) - HEADER_LINES} output lines -- may be alive"
    if age_min < stale_minutes:
        return "keep", f"log header-only but touched {age_min:.0f} min ago (< {stale_minutes:.0f})"
    return "reset", f"log header-only, untouched for {age_min:.0f} min, no outputs"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=("full", "smoke"), default="full")
    ap.add_argument("--stale-minutes", type=float, default=60.0)
    ap.add_argument("--apply", action="store_true", help="write the manifest (default: dry-run)")
    args = ap.parse_args()

    manifest = C.FULL_MANIFEST if args.stage == "full" else C.SMOKE_MANIFEST
    rows, fields = C.E202C.E199.read_with_fields(manifest)

    n = {"reset": 0, "keep": 0, "complete": 0}
    for row in rows:
        action, reason = classify(row, args.stale_minutes)
        n[action] += 1
        if action == "keep" and row["status"] != "running":
            continue
        tag = row["variant"].replace("E210_bucket007_", "").replace("_G1only", "")
        print(f"  {action.upper():8s} {tag:30s} {reason}")
        if action == "reset" and args.apply:
            row["status"] = ""
            row["failure_mode"] = "reset_from_stale_running"
            row["gpu_id"] = ""
            row["updated_at"] = C.now()

    print(f"\nreset={n['reset']}  keep={n['keep']}  already-complete={n['complete']}")
    if not args.apply:
        print("dry-run: nothing written. Re-run with --apply once you have confirmed "
              "no queue is alive on any machine sharing this filesystem.")
        return 0
    C.write_tsv(manifest, rows, fields)
    print(f"wrote {C.rel(manifest)} -- {n['reset']} row(s) now pending")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
