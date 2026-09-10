#!/usr/bin/env python3
"""E214 single-variable audit: prove each ablation config_act.yaml differs from
its baseline config_act.yaml only by the intended ablation toggle (+ run plumbing).

For every cem_ok manifest row, load baseline vs ablation config_act and require
    diff_keys  ⊆  ablation_toggle_keys ∪ PLUMBING_KEYS ∪ {new-field fills}
where a "new-field fill" is a key whose baseline value is None (added to Config
after the baseline was saved; process_config fills it with the current default).
Also assert every toggle key ACTUALLY changed to the requested value.

Run after run_ablation_cem.py:
    .venv/bin/python .../E214/audit_single_variable.py [--strict-newfields]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e214_common as C  # noqa: E402


def _num_eq(a, b) -> bool:
    try:
        fa, fb = float(a), float(b)
        if math.isnan(fa) and math.isnan(fb):  # nan==nan for config identity
            return True
        return abs(fa - fb) <= 1e-9
    except (TypeError, ValueError):
        return str(a) == str(b)


def audit_row(row: dict[str, str]) -> dict:
    base_p = C.repo_path(row["baseline_config_act"])
    abl_p = C.repo_path(row["config_act"])
    res = {"case_id": row["case_id"], "ablation": row["ablation"], "ok": True, "problems": []}
    if not abl_p.is_file():
        res["ok"] = False
        res["problems"].append("ablation config_act missing")
        return res
    base = yaml.safe_load(base_p.read_text(encoding="utf-8")) or {}
    abl = yaml.safe_load(abl_p.read_text(encoding="utf-8")) or {}

    toggle_keys = set(C.ABLATIONS[row["ablation"]]["toggles"])
    allowed = set(toggle_keys) | C.PLUMBING_KEYS

    # 1) every toggle key must have taken its requested value
    for key, want in C.ABLATIONS[row["ablation"]]["toggles"].items():
        got = abl.get(key)
        if isinstance(want, bool):
            if str(got).lower() != str(want).lower():
                res["ok"] = False
                res["problems"].append(f"toggle {key}={got}!={want}")
        elif not _num_eq(got, want):
            res["ok"] = False
            res["problems"].append(f"toggle {key}={got}!={want}")

    # 2) any OTHER differing key must be plumbing or a None->default new-field fill
    for key in set(base) | set(abl):
        if key in allowed:
            continue
        bv, av = base.get(key), abl.get(key)
        if bv == av or _num_eq(bv, av):
            continue
        if bv is None:  # new Config field filled with current default on reload
            continue
        if any(key.endswith(s) for s in C.DERIVED_KEY_SUFFIXES):
            continue
        res["ok"] = False
        res["problems"].append(f"unexpected diff {key}: {bv!r}->{av!r}")
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    rows, _ = C.read_with_fields(C.MANIFEST)
    done = [r for r in rows if r.get("status") == "cem_ok"]
    if args.limit:
        done = done[: args.limit]
    if not done:
        print("no cem_ok rows to audit yet")
        return 0
    results = [audit_row(r) for r in done]
    bad = [r for r in results if not r["ok"]]
    C.PREFLIGHT_DIR.mkdir(parents=True, exist_ok=True)
    out = C.PREFLIGHT_DIR / "e214_single_variable_audit.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"audited {len(results)} cem_ok rows -> {C.rel(out)}")
    for r in bad[:30]:
        print(f"  FAIL {r['case_id']} {r['ablation']}: {'; '.join(r['problems'])}")
    print("ALL PASS" if not bad else f"FAIL: {len(bad)}/{len(results)} rows")
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
