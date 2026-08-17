#!/usr/bin/env python3
"""E201 · classify each rollout into the three-tier funnel.

Reads a case_metrics.tsv (E199 fullscale by default), recomputes body_z (via the
E199 xlsx helper, since it is not stored in the tsv), evaluates the 14 gates
(4 hard + 10 banded at wide/narrow) from FunnelConfig, and assigns each rollout
a layer:

  L1_reject        hard gate failed OR any banded gate fails WIDE
  L2_review        all banded pass WIDE, >=1 fails NARROW
  L3_auto          all banded pass NARROW, and whole family passes narrow
  L3_review        all banded pass NARROW, but a family arm does not

Family = the {orig,trans0,trans1,trans2} arms under one case_id. Any L3 row
(orig or aug) is auto-accepted iff every *present* arm of its family is itself
L3 (narrow pass); partial families (missing aug arms) impose no extra constraint
(user 2026-08-17). orig is family-checked too (validated: metric-clean orig
baselines with inconsistent families were the false-accepts).

Usage:
    .venv/bin/python workspace/core4d/scripts/experiments/E201/classify_funnel.py \
        --exp E199 [--assert-monotonic] [--out <tsv>]
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO / "workspace/core4d/scripts"))
sys.path.insert(0, str(REPO / "workspace/core4d/scripts/eval/runners"))
sys.path.insert(0, str(REPO / "workspace/core4d/scripts/eval/reports"))
sys.path.insert(0, str(REPO / "workspace/core4d/scripts/experiments/E201"))

import funnel_config as FC  # noqa: E402
# Reuse the E199 workbook's body_z recompute + path plumbing (DRY: identical FK).
import gen_E199_fullscale_gate_xlsx as X  # noqa: E402

# --- experiment -> (case_metrics tsv, manifest tsv) --------------------------
_E200 = REPO / "workspace/core4d/results/E200/s6_downstream"
EXPS: dict[str, tuple[Path, Path]] = {
    "E199": (X.CASE_METRICS, X.MANIFEST),
    # E200 arms (same funnel_config contract; both scored by eval_E199_augmentation.score).
    # These arms have only the 3 translation aug variants (no orig row -> family
    # arbitration uses the 3 present trans arms; orig baseline lives elsewhere:
    # PRG-side reuses E198 A0, noPRG-side reuses E190).
    "E200_noprg": (_E200 / "eval/noprg/e200_noprg_case_metrics.tsv",
                   _E200 / "manifests/e200_noprg_priority_manifest.tsv"),
    "E200_prg_g1a2": (_E200 / "eval/prg_g1a2/e200_prg_g1a2_case_metrics.tsv",
                      _E200 / "manifests/e200_prg_g1a2_priority_manifest.tsv"),
}

FAMILY_ORDER = {"orig": 0, "trans0": 1, "trans1": 2, "trans2": 3}


def family_key(case_id: str) -> str:
    """Normalize the actor suffix so orig (_p1/_p2) and aug (_person1/_person2)
    arms of the same physical case share one family key."""
    return re.sub(r"_p(\d)$", r"_person\1", case_id)


def _finite(value: Any) -> float:
    return X.finite(value)


def _body_z(row: dict[str, str], manifest: dict[tuple[str, str], dict[str, str]]) -> float:
    """Recompute body_z p95 the same way the E199 workbook does (CPU FK)."""
    scene_xml = X.G.repo_path(row["scene_xml"])
    qpos_path = Path(row["qpos_path"])
    if row["group"] == "aug":
        m = manifest.get((row["case_id"], row["aug_variant"]))
        trajectory = X.G.repo_path(m["trajectory"]) if m else None
    else:
        trajectory, _mask = X.orig_paths(scene_xml)
    try:
        if trajectory is None or not trajectory.is_file():
            raise FileNotFoundError(f"missing reference trajectory: {trajectory}")
        return X.G.body_z_p95(qpos_path, scene_xml, trajectory)
    except Exception as exc:  # noqa: BLE001
        print(f"[body_z] {row['case_id']}/{row.get('aug_variant')}: {type(exc).__name__}", file=sys.stderr)
        return float("nan")


def classify_row(row: dict[str, str], body_z: float) -> dict[str, Any]:
    """Per-row (pre-family) classification. Family arbitration happens later."""
    item = dict(row)
    item["body_z_err_p95_m"] = body_z

    hard_ok, hard_failed = FC.hard_gate_result(item)
    wide_ok, wide_failed = FC.banded_gate_result(item, "wide")
    narrow_ok, narrow_failed = FC.banded_gate_result(item, "narrow")

    if not hard_ok or not wide_ok:
        layer = "L1_reject"
    elif not narrow_ok:
        layer = "L2_review"
    else:
        layer = "L3_narrow"  # provisional; family arbitration refines aug rows

    out = {
        "object_key": row.get("object_key", ""),
        "case_id": row.get("case_id", ""),
        "group": row.get("group", ""),
        "aug_variant": row.get("aug_variant", ""),
        "layer": layer,
        "hard_pass": hard_ok,
        "hard_failed": ",".join(hard_failed),
        "wide_pass": wide_ok,
        "wide_failed": ",".join(wide_failed),
        "narrow_pass": narrow_ok,
        "narrow_failed": ",".join(narrow_failed),
    }
    # full 14-gate numeric values so the rollout TSV is self-contained (no need to
    # join back to case_metrics to see the numbers behind each gate).
    out["fall_flag"] = str(item.get("fall_flag", "")).strip().lower() in ("true", "1")
    out["body_z_err_p95_m"] = body_z
    for _name, field, _op, _thr in FC.HARD_GATES:
        if field in ("fall_flag", "body_z_err_p95_m"):
            continue
        out[field] = X.finite(row.get(field))
    for _name, field, *_ in FC.BANDED_GATES:
        out[field] = X.finite(row.get(field))
    return out


def apply_family(rows: list[dict[str, Any]], min_other_pass: str = "all") -> None:
    """Refine L3_narrow rows into L3_auto / L3_review, in place.

    A narrow-passing rollout (orig or aug alike) is auto-accepted iff enough of
    its *other* family arms also pass narrow:
      min_other_pass="all" (default): ALL other present arms must pass narrow.
      min_other_pass="2"/"1":         at least K other arms pass narrow.

    Default is "all" (strictest): validated on E199 to give 0 false-accepts.
    Relaxing to 2/1 raises auto-decision rate but admits the metric-clean-but-
    unusable "lucky narrow-passers" (E199 sweep: >=2 -> ~62% of the newly
    auto-accepted were human-DO_NOT_USE). See log 288.

    Family narrow-pass is read from a snapshot taken *before* any arbitration,
    so the outcome is independent of row order.
    """
    by_case: dict[str, list[dict[str, Any]]] = {}
    narrow_ok: dict[int, bool] = {}
    for r in rows:
        by_case.setdefault(family_key(r["case_id"]), []).append(r)
        narrow_ok[id(r)] = (r["layer"] == "L3_narrow")  # snapshot pre-arbitration

    for r in rows:
        if r["layer"] != "L3_narrow":
            continue
        others = [o for o in by_case[family_key(r["case_id"])] if o is not r]
        n_other_ok = sum(narrow_ok[id(o)] for o in others)
        if min_other_pass == "all":
            ok = (n_other_ok == len(others))
        else:
            ok = (n_other_ok >= int(min_other_pass))
        if ok:
            r["layer"] = "L3_auto"
            tag = "all" if min_other_pass == "all" else f">={min_other_pass}"
            r["family_flag"] = f"consistent({n_other_ok}/{len(others)}others,{tag})"
        else:
            r["layer"] = "L3_review"
            bad = [o["aug_variant"] or "orig" for o in others if not narrow_ok[id(o)]]
            r["family_flag"] = f"inconsistent:{','.join(sorted(set(bad)))}"
    for r in rows:
        r.setdefault("family_flag", "")


def summarize(rows: list[dict[str, Any]]) -> str:
    layers = ["L1_reject", "L2_review", "L3_auto", "L3_review"]
    lines: list[str] = []

    def tally(subset: list[dict[str, Any]], title: str) -> None:
        n = len(subset)
        counts = {lay: sum(1 for r in subset if r["layer"] == lay) for lay in layers}
        auto = counts["L1_reject"] + counts["L3_auto"]
        human = counts["L2_review"] + counts["L3_review"]
        lines.append(f"[{title}] total={n}")
        for lay in layers:
            frac = counts[lay] / n if n else 0.0
            lines.append(f"    {lay:12s} {counts[lay]:4d}  ({frac:5.1%})")
        lines.append(f"    -> auto-decided {auto}/{n} ({(auto/n if n else 0):.1%}); "
                     f"human {human}/{n} ({(human/n if n else 0):.1%})")

    tally(rows, "ALL")
    tally([r for r in rows if r["group"] == "aug"], "aug")
    tally([r for r in rows if r["group"] == "orig"], "orig")
    lines.append("--- per object ---")
    for obj in sorted({r["object_key"] for r in rows}):
        tally([r for r in rows if r["object_key"] == obj], obj)
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="E199", choices=sorted(EXPS))
    ap.add_argument("--assert-monotonic", action="store_true")
    ap.add_argument("--family-min-other-pass", default="all", choices=["all", "2", "1"],
                    help="auto-accept an L3 rollout if >=this many OTHER family arms "
                         "pass narrow ('all'=strict default, 0 false-accepts on E199)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    FC.assert_monotonic()
    if args.assert_monotonic:
        print("[assert] narrow⟹wide monotonic invariant OK for all banded gates")

    case_metrics, manifest_path = EXPS[args.exp]
    case_rows = X.read_tsv(case_metrics)
    man_rows = X.read_tsv(manifest_path)
    manifest = {(m["case_id"], m["aug_variant"]): m for m in man_rows}

    computed: list[dict[str, Any]] = []
    for i, row in enumerate(case_rows, start=1):
        bz = _body_z(row, manifest)
        computed.append(classify_row(row, bz))
        if i % 40 == 0 or i == len(case_rows):
            print(f"[classify] {i}/{len(case_rows)}", file=sys.stderr)

    apply_family(computed, min_other_pass=args.family_min_other_pass)

    # per-row monotonic self-check: narrow_pass must imply wide_pass
    violations = [r for r in computed if r["narrow_pass"] and not r["wide_pass"]]
    if violations:
        raise AssertionError(f"{len(violations)} rows violate narrow⟹wide")
    print(f"[assert] per-row narrow⟹wide: 0 violations over {len(computed)} rows")

    out = args.out or (REPO / f"workspace/core4d/results/E201/funnel/{args.exp}_funnel_rollout.tsv")
    out.parent.mkdir(parents=True, exist_ok=True)
    order = {"orig": 0, "trans0": 1, "trans1": 2, "trans2": 3}
    computed.sort(key=lambda r: (r["object_key"], r["case_id"].replace("_person", "_p"),
                                 order.get(r["aug_variant"], 9)))
    # full 14-gate numeric value columns, in FunnelConfig order (fall, body_z,
    # ankle_jerk, obj_speed, then the 10 banded fields) — self-contained.
    gate_val_cols = ["fall_flag", "body_z_err_p95_m"]
    gate_val_cols += [f for _n, f, _o, _t in FC.HARD_GATES
                      if f not in ("fall_flag", "body_z_err_p95_m")]
    gate_val_cols += [f for _n, f, *_ in FC.BANDED_GATES]
    cols = (["object_key", "case_id", "group", "aug_variant", "layer", "family_flag"]
            + gate_val_cols
            + ["hard_pass", "hard_failed", "wide_pass", "wide_failed",
               "narrow_pass", "narrow_failed"])

    def _fmt(v: Any) -> Any:
        if isinstance(v, float):
            return "" if not math.isfinite(v) else round(v, 4)
        return v

    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, delimiter="\t")
        w.writeheader()
        for r in computed:
            w.writerow({c: _fmt(r.get(c, "")) for c in cols})

    print(f"\n[done] wrote {out} ({len(computed)} rows)\n")
    print(summarize(computed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
