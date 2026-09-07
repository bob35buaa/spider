#!/usr/bin/env python3
"""E213 Phase A exit check: reconcile the 4 machine shards into the main manifest.

Four 8-GPU machines ran disjoint shard manifests (the runner rewrites the whole
TSV on every status change, so a shared file would corrupt).  Nothing enters eval
until the four shards compose into exactly the intended 100 rows, each on its
selected-arm scene, with a recorded host.

Checks, cheapest-to-catch first:
  1. Coverage   -- shard keys disjoint and their union == the main manifest's
     (case_id, aug_variant) set.
  2. Completion -- every row is ``cem_ok`` with its result npz + config_act on disk.
  3. Identity   -- each run's own config_act.scene_name == the arm's selected scene.
  4. Provenance -- every row has a host; per-host wall medians reported.

Usage:
    .venv/bin/python workspace/core4d/scripts/experiments/E213/merge_shards.py
"""

from __future__ import annotations

import json
import statistics as stats
import sys
from collections import defaultdict
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

import e213_common as C  # noqa: E402

DONE = "cem_ok"


def main() -> int:
    main_rows, fields = C.read_with_fields(C.SOURCE_MANIFEST)
    want_keys = {(r["case_id"], r["aug_variant"]) for r in main_rows}

    problems: list[str] = []
    merged: dict[tuple[str, str], dict[str, str]] = {}
    seen_in: dict[tuple[str, str], str] = {}

    for shard in C.SHARDS:
        path = C.manifest_path(shard)
        if not path.is_file():
            problems.append(f"shard {shard}: manifest missing ({C.rel(path)})")
            continue
        for row in C.read_tsv(path):
            key = (row["case_id"], row["aug_variant"])
            if key in seen_in:
                problems.append(f"{key}: in both shard {seen_in[key]} and {shard}")
                continue
            if row["shard"] != shard:
                problems.append(f"{key}: row in shard{shard} file labelled shard={row['shard']}")
            seen_in[key] = shard
            merged[key] = row

    if set(merged) != want_keys:
        for key in sorted(want_keys - set(merged)):
            problems.append(f"{key}: in main manifest but no shard")
        for key in sorted(set(merged) - want_keys):
            problems.append(f"{key}: in a shard but not the main manifest")

    by_host: dict[str, list[float]] = defaultdict(list)
    for key, row in sorted(merged.items()):
        label = f"{row['case_id']}/{row['aug_variant']}/{row['arm']}"
        if row["status"] != DONE:
            problems.append(f"{label}: status={row['status']!r} != {DONE!r} ({row['failure_mode']})")
        for field in ("outdir_npz", "config_act"):
            if not C.repo_path(row[field]).is_file():
                problems.append(f"{label}: missing {field} -> {row[field]}")
        cfg_path = C.repo_path(row["config_act"])
        if cfg_path.is_file():
            cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
            if cfg.get("scene_name") != row["scene_name"]:
                problems.append(f"{label}: ran scene_name={cfg.get('scene_name')!r}, "
                                f"expected {row['scene_name']!r}")
        if not row.get("host"):
            problems.append(f"{label}: empty host")
        if row.get("wall_min"):
            try:
                by_host[row.get("host") or "?"].append(float(row["wall_min"]))
            except ValueError:
                pass

    if not problems:
        ordered = [merged[(r["case_id"], r["aug_variant"])] for r in main_rows]
        C.write_tsv(C.SOURCE_MANIFEST, ordered, fields)

    host_stats = {h: {"n": len(v), "median_min": round(stats.median(v), 1),
                      "min_min": round(min(v), 1), "max_min": round(max(v), 1)}
                  for h, v in sorted(by_host.items())}
    all_walls = [w for v in by_host.values() for w in v]
    summary = {
        "rows": len(merged), "expected": len(want_keys),
        "shards": {s: sum(1 for k in seen_in if seen_in[k] == s) for s in C.SHARDS},
        "by_host": host_stats,
        "wall_median_min": round(stats.median(all_walls), 1) if all_walls else None,
        "n_problems": len(problems), "problems": problems,
    }
    out = C.MANIFEST_DIR / "e213_source_merge.json"
    C.write_json(out, summary)

    for host, s in host_stats.items():
        print(f"  {host:24s} n={s['n']:2d}  median={s['median_min']:5.1f} min  [{s['min_min']:.1f}, {s['max_min']:.1f}]")
    if problems:
        print("\n".join(f"  FAIL {p}" for p in problems[:20]))
        raise SystemExit(f"merge FAILED ({len(problems)} problems)")
    print(f"\nMERGE PASS: {len(merged)}/{len(want_keys)} rows reconciled "
          f"({', '.join(f'{s}={summary['shards'][s]}' for s in C.SHARDS)}) -> {C.rel(out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
