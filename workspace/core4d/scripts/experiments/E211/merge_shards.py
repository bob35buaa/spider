#!/usr/bin/env python3
"""E211 P7 exit check: reconcile the two-machine shards back into the main manifest.

Stage A ran on two hosts against disjoint shard manifests (the queue rewrites the
whole TSV on every status change, so a shared file would corrupt).  Nothing may
enter P8 until the two halves are proven to compose into exactly the intended 15
rows, on the intended scenes, with a recorded host for each.

Checks, in the order a wrong answer would be cheapest to catch:

  1. Coverage   -- shard keys are disjoint and their union is exactly the main
     manifest's 15 (case_id, arm) pairs.  Catches "someone ran shard A twice"
     and "a row silently vanished".
  2. Completion -- every row is ``run_complete_pending_eval`` with its result
     npz and config_act on disk.  A row left in ``running`` is a tombstone, not
     a success (E210 F1: the queue's ELIGIBLE set excludes ``running``, so a
     killed queue reports "0 pending" and looks finished).
  3. Identity   -- each run's own ``config_act.yaml`` names the scene its arm is
     supposed to use.  This is what catches a shard/arm mis-wiring: without it,
     two rows could have run the same g and nothing else would notice.
  4. Provenance -- every row carries a host, and the per-host wall-time medians
     are reported so a hardware-linked difference cannot hide.

Usage:
    .venv/bin/python workspace/core4d/scripts/experiments/E211/merge_shards.py [--stage full]
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import re
import statistics as stats
import sys
from collections import defaultdict
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[5]
for _d in ("E211", "E199"):
    sys.path.insert(0, str(REPO / "workspace/core4d/scripts/experiments" / _d))

import e199_common as E199  # noqa: E402
import e211_common as C  # noqa: E402


def _load_sibling(name: str):
    """Import a same-directory module by path, not by sys.path search.

    ``e211_common`` puts E209/E206/E200/E199 on sys.path ahead of E211, and every
    one of those dirs has its own ``build_manifest.py``. A plain
    ``from build_manifest import FIELDS`` silently resolves to E200's and fails
    (or worse, succeeds with the wrong column list).
    """
    import importlib.util

    path = Path(__file__).resolve().parent / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"e211_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_bm = _load_sibling("build_manifest")
FIELDS, SHARDS = _bm.FIELDS, _bm.SHARDS

DONE = "run_complete_pending_eval"


def read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def wall_minutes(row: dict[str, str]) -> float | None:
    """Wall clock from the queue's log header to the result npz mtime."""
    log, npz = REPO / row["log"], REPO / row["outdir_npz"]
    if not (log.is_file() and npz.is_file()):
        return None
    start = None
    with log.open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            match = re.search(r"started_at=(\S+)", line)
            if match:
                start = match.group(1)
                break
            if not line.startswith("#"):
                break
    if start is None:
        return None
    t0 = dt.datetime.fromisoformat(start)
    t1 = dt.datetime.fromtimestamp(npz.stat().st_mtime, tz=t0.tzinfo)
    return (t1 - t0).total_seconds() / 60.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="full", choices=("full", "smoke"))
    args = ap.parse_args()

    main_path = C.manifest_path(args.stage)
    main_rows = read(main_path)
    want_keys = {(r["case_id"], r["arm"]) for r in main_rows}

    problems: list[str] = []
    merged: dict[tuple[str, str], dict[str, str]] = {}
    seen_in: dict[tuple[str, str], str] = {}

    for shard in SHARDS:
        path = C.manifest_path(args.stage, shard)
        if not path.is_file():
            problems.append(f"shard {shard}: manifest missing ({path})")
            continue
        for row in read(path):
            key = (row["case_id"], row["arm"])
            if key in seen_in:
                problems.append(f"{key}: present in both shard {seen_in[key]} and {shard}")
                continue
            if row["shard"] != shard:
                problems.append(f"{key}: row in shard{shard} file is labelled shard={row['shard']}")
            seen_in[key] = shard
            merged[key] = row

    # 1. coverage
    if set(merged) != want_keys:
        for key in sorted(want_keys - set(merged)):
            problems.append(f"{key}: in main manifest but in no shard")
        for key in sorted(set(merged) - want_keys):
            problems.append(f"{key}: in a shard but not in the main manifest")

    by_host: dict[str, list[float]] = defaultdict(list)
    for key, row in sorted(merged.items()):
        label = f"{row['case_id']}/{row['arm']}"

        # 2. completion
        if row["status"] != DONE:
            problems.append(f"{label}: status={row['status']!r} != {DONE!r} ({row['failure_mode']})")
        for field in ("outdir_npz", "result_npz", "config_act"):
            if not (REPO / row[field]).is_file():
                problems.append(f"{label}: missing {field} -> {row[field]}")

        # 3. identity -- the run's own resolved config must name this arm's scene
        cfg_path = REPO / row["config_act"]
        if cfg_path.is_file():
            cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
            want_scene = C.SCENE_BY_ARM[row["arm"]]
            if cfg.get("scene_name") != want_scene:
                problems.append(
                    f"{label}: ran scene_name={cfg.get('scene_name')!r}, arm expects {want_scene!r}"
                )
            if cfg.get("scene_name") != row["scene_name"]:
                problems.append(
                    f"{label}: manifest scene_name={row['scene_name']!r} != executed {cfg.get('scene_name')!r}"
                )

        # 4. provenance
        if not row["host"]:
            problems.append(f"{label}: empty host -- cannot attribute the run to a machine")
        wall = wall_minutes(row)
        if wall is not None:
            by_host[row["host"] or "?"].append(wall)

    # Write the reconciled main manifest (shard files stay untouched as evidence).
    if not problems:
        ordered = [merged[(r["case_id"], r["arm"])] for r in main_rows]
        E199.write_tsv(main_path, ordered, FIELDS)

    host_stats = {
        host: {
            "n": len(v),
            "median_min": round(stats.median(v), 1),
            "min_min": round(min(v), 1),
            "max_min": round(max(v), 1),
        }
        for host, v in sorted(by_host.items())
    }
    all_walls = [w for v in by_host.values() for w in v]
    summary = {
        "stage": args.stage,
        "rows": len(merged),
        "expected": len(want_keys),
        "shards": {s: sum(1 for k in seen_in if seen_in[k] == s) for s in SHARDS},
        "by_host": host_stats,
        "wall_median_min": round(stats.median(all_walls), 1) if all_walls else None,
        "C4_wall_window_min": [
            C.GATES["C4_wall_median_min_min"], C.GATES["C4_wall_median_max_min"]
        ],
        "problems": problems,
    }
    out = C.MANIFEST_DIR / f"e211_stageA_{args.stage}_merge.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    for host, s in host_stats.items():
        print(f"  {host:36s} n={s['n']:2d}  median={s['median_min']:5.1f} min  "
              f"[{s['min_min']:.1f}, {s['max_min']:.1f}]")
    if all_walls:
        med = stats.median(all_walls)
        lo, hi = C.GATES["C4_wall_median_min_min"], C.GATES["C4_wall_median_max_min"]
        verdict = "PASS" if lo <= med <= hi else "FAIL"
        print(f"  C4 throughput: median {med:.1f} min in [{lo}, {hi}] -> {verdict}")

    if problems:
        print("\n".join(f"  FAIL {p}" for p in problems))
        raise SystemExit(f"P7 merge FAILED ({len(problems)} problems) -- do NOT proceed to P8")
    print(f"\nP7 PASS: {len(merged)}/{len(want_keys)} rows reconciled "
          f"(shardA={summary['shards']['A']}, shardB={summary['shards']['B']}) -> {out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
