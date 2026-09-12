#!/usr/bin/env python3
"""E215: split the frozen 70-row CEM queue into N shards for multi-machine runs.

Three machines share the filesystem, and run_e215_cem rewrites the WHOLE manifest
on every status change -- so the three MUST run disjoint shard manifests, never
one shared TSV.  Output paths are keyed by variant_id (unique per case+variant),
so the shards' rollouts/logs never collide either.

Sharding is round-robin over the frozen ``ordinal`` (which is already a stratified
order: P0 first, then a diagonal on object x variant), so each shard gets a
balanced slice of tiers and arm groups -- if one machine dies, no whole tier or
arm group is lost.

Usage:
    .venv/bin/python .../E215/split_manifest_shards.py            # 3 shards A/B/C
    ... --shards 2
"""

from __future__ import annotations

import argparse
import string
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import e215_common as C  # noqa: E402


def shard_path(label: str) -> Path:
    return C.MANIFEST_DIR / f"e215_priority_manifest.shard{label}.tsv"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=int, default=3)
    ap.add_argument("--source", type=Path, default=C.FROZEN_MANIFEST)
    args = ap.parse_args()

    src = C.repo_path(args.source)
    if not src.is_file():
        raise SystemExit(f"missing frozen manifest {C.rel(src)} -- freeze first")
    rows, fields = C.read_with_fields(src)
    rows.sort(key=lambda r: int(r["ordinal"]))

    n = args.shards
    if n < 1 or n > len(string.ascii_uppercase):
        raise SystemExit(f"--shards must be 1..26, got {n}")
    labels = list(string.ascii_uppercase[:n])
    buckets: dict[str, list[dict[str, str]]] = {lab: [] for lab in labels}
    for i, row in enumerate(rows):
        buckets[labels[i % n]].append(row)

    print(f"splitting {len(rows)} rows into {n} shards (round-robin over ordinal):")
    for lab in labels:
        shard = buckets[lab]
        C.write_tsv(shard_path(lab), shard, fields)
        by_tier: dict[str, int] = {}
        by_group: dict[str, int] = {}
        for r in shard:
            by_tier[r["tier"]] = by_tier.get(r["tier"], 0) + 1
            by_group[r["arm_group"]] = by_group.get(r["arm_group"], 0) + 1
        print(f"  shard{lab}: {len(shard):2d} rows  tier={by_tier}  group={by_group}")
        print(f"           -> {C.rel(shard_path(lab))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
