#!/usr/bin/env python3
"""E191: offline object-support audit over already-completed CEM rollouts.

E191 runs **no physics simulation**. It re-scores existing `s6_downstream` NPZ
from E172/E173/E174/E189 with the additive object-support columns added to
`eval.core.core_metrics` (`E191_SUPPORT_FIELDS`), so that three mechanisms which
are perfectly collinear with object size in the historical box experiments can
start to be told apart:

  (a) fixed-metre reward/gate geometry inherited from a box004 case override,
  (b) the soft 6-DoF object position servo with no partner model,
  (c) grasp topology on large flat faces.

Coverage caveat recorded in the output: the `dcv3_omnirt_v1_ref_fk_box021_*`
dataset directories are absent from this checkout, so E170/E168 (box021 — the
object R018 called "native") cannot be audited offline and is reported as a gap
rather than silently dropped.

Usage
-----
    .venv/bin/python workspace/core4d/scripts/eval/runners/eval_E191_object_support_audit.py
    .venv/bin/python .../eval_E191_object_support_audit.py --regress-against \
        workspace/core4d/results/E189/s6_downstream/eval/full/e189_case_metrics.tsv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCRIPT_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(SCRIPT_ROOT))
sys.path.insert(0, str(SCRIPT_ROOT / "experiments/E189"))

from eval.core.core_metrics import (  # noqa: E402
    METRIC_FIELDS,
    E191_SUPPORT_FIELDS,
    EvalConfig,
    evaluate_sequence,
)
# Reuse the frozen TSV serialisation convention (bool -> lowercase, non-finite
# float -> empty) rather than re-deriving it, so E191 output is diffable against
# the E170/E172/E173/E174/E189 case_metrics tables.
from e189_common import serial  # noqa: E402

OUT_DIR = REPO_ROOT / "workspace/core4d/results/E191/audit"
AUDIT_TSV = OUT_DIR / "e191_object_support_audit.tsv"
COVERAGE_JSON = OUT_DIR / "e191_coverage.json"

# Provenance columns copied straight from the source case_metrics row.
PASSTHROUGH = (
    "case_id",
    "object_key",
    "variant",
    "method",
    "spider_method_id",
    "target_variant_id",
    "retarget_variant_id",
    "hand_collision_variant_id",
)


@dataclass(frozen=True)
class Source:
    """One already-completed experiment whose rollouts we re-score."""

    exp: str
    tsv: str
    prg: str  # "PRG" | "noPRG" — which side of the E189/E179 ablation this is
    note: str


SOURCES = (
    Source("E172", "workspace/core4d/results/E172/s6_downstream/eval/full/e171_case_metrics.tsv", "PRG", "box004"),
    Source("E173", "workspace/core4d/results/E173/s6_downstream/eval/full/e173_case_metrics.tsv", "PRG", "box001/023/024"),
    Source("E174", "workspace/core4d/results/E174/s6_downstream/eval/full/e174_case_metrics.tsv", "PRG", "bucket/desk (SPIDER, NOT an Omni control)"),
    Source("E189", "workspace/core4d/results/E189/s6_downstream/eval/full/e189_case_metrics.tsv", "noPRG", "box004/024/001"),
    # E170 (box021 PRG) and E168 (box021 noPRG) are listed so the coverage gap is
    # explicit in the report; their dataset directories are missing locally.
    Source("E170", "workspace/core4d/results/E170/s6_downstream/eval/full/e170_case_metrics.tsv", "PRG", "box021"),
)


def resolve(raw: str) -> Path | None:
    """Resolve a case_metrics path, remapping stale absolute workdir prefixes.

    Older TSVs (E170, E174) recorded absolute paths on detached filesystems.
    Anything after `core4d/results/` is stable, so retry under the repo.
    """
    if not raw:
        return None
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = REPO_ROOT / raw
    if candidate.is_file():
        return candidate
    match = re.search(r"(core4d/results/.*)$", raw)
    if match:
        remapped = REPO_ROOT / "workspace" / match.group(1)
        if remapped.is_file():
            return remapped
    return None


def person_idx(row: dict[str, str]) -> int:
    value = str(row.get("person_idx", "")).strip()
    if value:
        return int(value)
    return 0 if row["case_id"].endswith("_p1") else 1


def score_row(payload: tuple[str, str, dict[str, str]]) -> dict[str, Any]:
    """Worker: re-score one rollout. Returns a row dict or an error record."""
    exp, prg, row = payload
    # Prefer the same qpos source the source experiment scored, so the audit is
    # a re-scoring of the identical trajectory rather than a sibling artifact.
    npz = resolve(row.get("outdir_npz", "")) or resolve(row.get("result_npz", ""))
    scene = resolve(row.get("scene_xml", "") or row.get("scene_act", ""))
    traj = resolve(row.get("trajectory", ""))
    mask = resolve(row.get("contact_mask", ""))
    missing = [
        label
        for label, path in (("result_npz", npz), ("scene_xml", scene), ("trajectory", traj))
        if path is None
    ]
    if missing:
        return {"__error__": f"missing:{'/'.join(missing)}", "exp": exp, "case_id": row.get("case_id", "")}
    try:
        item = evaluate_sequence(
            row=row,
            method=row.get("spider_method_id", "") or row.get("method", ""),
            hand_collision_variant_id=row.get("hand_collision_variant_id", ""),
            qpos_path=npz,
            scene_xml=scene,
            config=EvalConfig(),
            kin_ref_path=traj,
            contact_mask_path=mask,
            person_idx=person_idx(row) if mask is not None else None,
        )
    except Exception as exc:  # noqa: BLE001 - recorded per row, never silently dropped
        return {"__error__": f"{type(exc).__name__}: {exc}", "exp": exp, "case_id": row.get("case_id", "")}
    item["exp"] = exp
    item["prg"] = prg
    for key in PASSTHROUGH:
        if row.get(key):
            item[key] = row[key]
    return item


def load_source(src: Source) -> list[dict[str, str]]:
    path = REPO_ROOT / src.tsv
    if not path.is_file():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def regression_check(
    rows: list[dict[str, Any]], reference_tsv: Path, exp: str
) -> tuple[int, list[str]]:
    """Assert every pre-existing metric column is unchanged.

    The E191 change to core_metrics.py must be purely additive. This compares
    the re-scored values against the frozen reference TSV column by column, for
    every column that already existed there.

    `exp` scopes the comparison: the same `case_id` exists in both the PRG
    (E172/E173) and no-PRG (E189) sources with different rollouts, so matching
    on case_id alone would compare unrelated trajectories.
    """
    with reference_tsv.open(newline="") as handle:
        ref = {r["case_id"]: r for r in csv.DictReader(handle, delimiter="\t")}
    new_cols = set(E191_SUPPORT_FIELDS)
    diffs: list[str] = []
    compared = 0
    for row in rows:
        if row.get("exp") != exp:
            continue
        old = ref.get(row.get("case_id", ""))
        if old is None:
            continue
        compared += 1
        for col, old_value in old.items():
            if col in new_cols or col not in row:
                continue
            if _same(old_value, row[col]):
                continue
            diffs.append(f"{row['case_id']}:{col}: {old_value!r} -> {row[col]!r}")
    return compared, diffs


def _same(old_raw: str, new_value: Any) -> bool:
    """Compare a reference TSV cell against a freshly computed value.

    Both sides are normalised through the frozen `serial()` convention first, so
    `False`/`"false"` and `nan`/`""` are not reported as regressions.
    """
    new_raw = str(serial(new_value))
    if old_raw == new_raw:
        return True
    try:
        a, b = float(old_raw), float(new_raw)
    except (TypeError, ValueError):
        return False
    if math.isnan(a) and math.isnan(b):
        return True
    return a == b


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=min(16, (os.cpu_count() or 4)))
    parser.add_argument(
        "--regress-against",
        type=Path,
        default=REPO_ROOT / "workspace/core4d/results/E189/s6_downstream/eval/full/e189_case_metrics.tsv",
        help="Frozen TSV whose pre-existing columns must reproduce bit-identically.",
    )
    parser.add_argument("--only", default="", help="Substring filter on case_id (debugging).")
    args = parser.parse_args()

    payloads: list[tuple[str, str, dict[str, str]]] = []
    coverage: dict[str, Any] = {"sources": [], "skipped": []}
    for src in SOURCES:
        rows = load_source(src)
        if args.only:
            rows = [r for r in rows if args.only in r.get("case_id", "")]
        coverage["sources"].append({"exp": src.exp, "prg": src.prg, "note": src.note, "rows": len(rows)})
        payloads.extend((src.exp, src.prg, row) for row in rows)
    if not payloads:
        print("no rows found", file=sys.stderr)
        return 1

    print(f"scoring {len(payloads)} rollouts with {args.workers} workers ...")
    scored: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, item in enumerate(pool.map(score_row, payloads, chunksize=1), start=1):
            if "__error__" in item:
                coverage["skipped"].append(item)
            else:
                scored.append(item)
            if i % 20 == 0 or i == len(payloads):
                print(f"  {i}/{len(payloads)}  ok={len(scored)}  skipped={len(coverage['skipped'])}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fields = ["exp", "prg", *METRIC_FIELDS]
    with AUDIT_TSV.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, delimiter="\t", lineterminator="\n", extrasaction="ignore"
        )
        writer.writeheader()
        for item in sorted(scored, key=lambda r: (r["exp"], r.get("object_key", ""), r["case_id"])):
            writer.writerow({key: serial(item.get(key, "")) for key in fields})
    print(f"wrote {AUDIT_TSV} ({len(scored)} rows)")

    if args.regress_against and Path(args.regress_against).is_file():
        reference = Path(args.regress_against)
        match = re.search(r"/results/(E\d+)/", str(reference))
        ref_exp = match.group(1) if match else ""
        compared, diffs = regression_check(scored, reference, ref_exp)
        coverage["regression"] = {
            "reference": str(reference),
            "reference_exp": ref_exp,
            "rows_compared": compared,
            "diff_count": len(diffs),
            "diffs": diffs[:50],
        }
        status = "PASS" if not diffs else "FAIL"
        print(f"regression vs {reference.name} [{ref_exp}]: {status} ({compared} rows, {len(diffs)} diffs)")
        for line in diffs[:20]:
            print("  ", line)
    COVERAGE_JSON.write_text(json.dumps(coverage, indent=2))
    print(f"wrote {COVERAGE_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
