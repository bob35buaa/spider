#!/usr/bin/env python3
"""E208 P9: is the human review actually complete, and complete over what?

C5 is "every aug row reviewed, USE rate >= orig - 0.15".  A review that skipped
the hard cases would satisfy a naive count while meaning nothing, so this checks
coverage per stratum, not just in total -- and reports the USE rate the same way
the numeric side is reported (per variant, per object, per offset band), so a
degradation concentrated in one band cannot hide inside a healthy pooled number.

Run twice:
  * before reviewing -- confirms every row is playable and has an MP4
  * after reviewing  -- confirms every row is filled and computes C5/C6

C6 (visual-vs-numeric disagreement) is computed here too, because it needs both
sides joined: numeric-pass-but-human-rejected is the reward-hacking signal, and
numeric-fail-but-human-accepted tells us the gates are too strict.

Usage:
    .venv/bin/python .../E208/build_review_coverage.py
    ... --filled PATH      # the reviewer's filled TSV (default: *_filled.tsv)
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import e208_common as C  # noqa: E402

FEED = C.EVAL_DIR / "e208_aug_case_metrics.tsv"
FILLED = C.EVAL_DIR / "user_manual_review_filled.tsv"

C5_MAX_USE_RATE_DROP = 0.15
C6_MAX_NUMERIC_PASS_HUMAN_REJECT = 0.20


def rate(num: int, den: int) -> float | None:
    return num / den if den else None


def strata_of(row: dict[str, str]) -> dict[str, str]:
    return {
        "variant": row.get("aug_variant", ""),
        "object": row.get("object_key", ""),
        "offset_band": row.get("offset_band", ""),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feed", type=Path, default=FEED)
    ap.add_argument("--filled", type=Path, default=FILLED)
    ap.add_argument("--json-out", type=Path, default=C.EVAL_DIR / "e208_review_coverage.json")
    args = ap.parse_args()

    feed_path = C.repo_path(args.feed)
    if not feed_path.is_file():
        raise SystemExit(f"missing {C.rel(feed_path)} -- run build_aug_review_tsv.py first")
    rows = C.read_tsv(feed_path)

    filled_path = C.repo_path(args.filled)
    filled_rows = C.read_tsv(filled_path) if filled_path.is_file() else []
    decisions = {(r["case_id"], r.get("aug_variant") or r.get("variant", "")):
                 r.get("manual_use_decision", "").strip()
                 for r in filled_rows}

    aug = [r for r in rows if r["variant"] != "orig"]
    orig = [r for r in rows if r["variant"] == "orig"]

    # --- playability / media coverage (checkable before any human looks) ---
    no_npz = [f"{r['case_id']}/{r['variant']}" for r in rows if not r["outdir_npz"]]
    no_video = [f"{r['case_id']}/{r['variant']}" for r in aug if not r["video"]]

    # --- review completeness, per stratum ---
    by_stratum: dict[str, dict[str, dict[str, int]]] = defaultdict(lambda: defaultdict(Counter))
    unreviewed: list[str] = []
    use_aug = pass_aug = 0
    for r in aug:
        key = (r["case_id"], r["variant"])
        decision = decisions.get(key, "")
        for level, value in strata_of(r).items():
            cell = by_stratum[level][value]
            cell["n"] += 1
            if decision:
                cell["reviewed"] += 1
            if decision.upper() == "USE":
                cell["use"] += 1
            if C.truth(r.get("c4_all_gates_pass", "")):
                cell["numeric_pass"] += 1
        if not decision:
            unreviewed.append(f"{r['case_id']}/{r['variant']}")
        if decision.upper() == "USE":
            use_aug += 1
        if C.truth(r.get("c4_all_gates_pass", "")):
            pass_aug += 1

    # orig USE rate is E206's own verdict, carried in as a prefill
    use_orig = sum(1 for r in orig if r.get("orig_manual_use_decision", "").upper() == "USE")

    n_reviewed = len(aug) - len(unreviewed)
    aug_use_rate = rate(use_aug, n_reviewed)
    orig_use_rate = rate(use_orig, len(orig))

    # --- C6: where numbers and humans disagree ---
    numeric_pass_rows = [r for r in aug if C.truth(r.get("c4_all_gates_pass", ""))]
    hacked = [f"{r['case_id']}/{r['variant']}" for r in numeric_pass_rows
              if decisions.get((r["case_id"], r["variant"]), "").upper()
              not in ("", "USE")]
    strict = [f"{r['case_id']}/{r['variant']}" for r in aug
              if not C.truth(r.get("c4_all_gates_pass", ""))
              and decisions.get((r["case_id"], r["variant"]), "").upper() == "USE"]

    c5_bar = (orig_use_rate - C5_MAX_USE_RATE_DROP) if orig_use_rate is not None else None
    payload: dict[str, Any] = {
        "experiment": C.EXP_ID, "run_id": C.RUN_ID, "generated_at": C.now(),
        "feed": C.rel(feed_path), "feed_sha256": C.sha256(feed_path),
        "filled": C.rel(filled_path) if filled_path.is_file() else "",
        "filled_sha256": C.sha256(filled_path) if filled_path.is_file() else "",
        "n_rows": len(rows), "n_aug": len(aug), "n_orig": len(orig),
        "media": {
            "rows_without_rollout": no_npz,
            "aug_rows_without_video": no_video,
            "verdict": "pass" if not no_npz and not no_video else "incomplete",
        },
        "C5_review": {
            "n_reviewed": n_reviewed, "n_unreviewed": len(unreviewed),
            "unreviewed": unreviewed[:20],
            "aug_use_rate": aug_use_rate,
            "orig_use_rate": orig_use_rate,
            "bar": c5_bar,
            "verdict": (
                "incomplete" if unreviewed or not filled_rows
                else "pass" if aug_use_rate is not None and c5_bar is not None
                and aug_use_rate >= c5_bar else "fail"
            ),
        },
        "C6_visual_vs_numeric": {
            "n_numeric_pass": len(numeric_pass_rows),
            "numeric_pass_human_reject": hacked,
            "rate": rate(len(hacked), len(numeric_pass_rows)),
            "bar": C6_MAX_NUMERIC_PASS_HUMAN_REJECT,
            "numeric_fail_human_use": strict,
            "verdict": (
                "incomplete" if not filled_rows
                else "pass" if rate(len(hacked), len(numeric_pass_rows)) is not None
                and rate(len(hacked), len(numeric_pass_rows)) <= C6_MAX_NUMERIC_PASS_HUMAN_REJECT
                else "fail"
            ),
            "note": ("numeric_pass_human_reject is the reward-hacking signal; "
                     "numeric_fail_human_use says the gates are stricter than the "
                     "reviewer and is reported, not gated"),
        },
        "by_stratum": {
            level: {
                key: {
                    "n": c["n"], "reviewed": c["reviewed"], "use": c["use"],
                    "numeric_pass": c["numeric_pass"],
                    "use_rate": rate(c["use"], c["reviewed"]),
                    "indicative": c["n"] < 3,
                }
                for key, c in sorted(cells.items())
            }
            for level, cells in by_stratum.items()
        },
    }
    C.write_json(C.repo_path(args.json_out), payload)

    print(f"rows={len(rows)} (aug {len(aug)} + orig {len(orig)})")
    print(f"  media: {len(no_npz)} without rollout, {len(no_video)} aug without mp4 "
          f"-> {payload['media']['verdict']}")
    print(f"  C5: reviewed {n_reviewed}/{len(aug)}, aug USE "
          f"{aug_use_rate if aug_use_rate is None else round(aug_use_rate, 3)} vs orig "
          f"{orig_use_rate if orig_use_rate is None else round(orig_use_rate, 3)} "
          f"-> {payload['C5_review']['verdict']}")
    print(f"  C6: {len(hacked)}/{len(numeric_pass_rows)} numeric-pass but rejected, "
          f"{len(strict)} numeric-fail but accepted -> "
          f"{payload['C6_visual_vs_numeric']['verdict']}")
    for level in ("variant", "offset_band", "object"):
        cells = payload["by_stratum"].get(level, {})
        print(f"  {level}: " + "  ".join(
            f"{k}={c['reviewed']}/{c['n']}" for k, c in cells.items()))
    print(f"-> {C.rel(C.repo_path(args.json_out))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
