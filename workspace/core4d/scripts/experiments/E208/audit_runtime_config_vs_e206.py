#!/usr/bin/env python3
"""E208 V6d: what the CEM process ACTUALLY ran with, vs E206's same-case PRG run.

V5e audited the composed Hydra config -- what the override *would* produce.  This
audits ``config_act.yaml``, which the runner writes from the live ``Config`` after
every resolution, geom-id lookup and model compile.  It is the only artifact that
can catch a divergence introduced between compose time and run time.

Measured 2026-09-05 on the first finished pair, the real difference is exactly
four task-derived keys -- ``task``, ``model_path``, ``data_path``, ``output_dir``.
Everything else in 423 keys is identical, including two things worth asserting
positively rather than merely observing absent from the diff:

* ``contact_hdmi_mask_path`` -- the aug runs share E206's 3cm mask by design
  (F3: identical trim window). If a second copy of that mask ever appeared, the
  contact metrics would drift with no other symptom.
* every ``*_geom_ids`` list -- these are resolved by name against the COMPILED
  model, so equality means E206's hand-placed lowgeom proxy compiled to the same
  geoms in the same order in the aug scene. That is a stronger statement than
  V5c's XML parity: it survives the compiler.

``nan != nan`` is handled explicitly; ``cem_safety_gate_hard_floor_m`` is NaN in
both and would otherwise report as a permanent false difference.

Usage:
    .venv/bin/python .../E208/audit_runtime_config_vs_e206.py
    ... --strict-missing     # also fail when a run has no config_act yet
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

import e208_common as C  # noqa: E402

# Keys allowed to differ: task identity and the three paths Hydra derives from it.
# Deliberately NOT including *_geom_ids or the contact mask -- see module docstring.
ALLOWED_DIFF_KEYS = frozenset({"task", "model_path", "data_path", "output_dir"})

# `object_actuator_ids` is the runtime footprint of F8: the object's 6 actuators
# are emitted in Euler-convention order, and a rot variant may legitimately land
# on a different convention than its base (measured: 11 of 42 rot, 0 of 63 trans).
# The SET must still match -- a missing or extra actuator is a real fault -- so
# this is compared order-insensitively rather than excused, and the convention on
# each side is recorded so the reason is visible in the audit output rather than
# only in the log.
ORDER_INSENSITIVE_KEYS = frozenset({"object_actuator_ids"})

# Must be byte-equal, and are checked by name rather than by absence from the diff
# so that a key vanishing from both configs cannot pass silently.
MUST_MATCH_KEYS = ("contact_hdmi_mask_path", "contact_hdmi_target_source",
                   "scene_name", "object_collision_sdf_mode", "num_samples",
                   "max_num_iterations", "seed")


def e206_config_act(case_id: str) -> Path:
    return C.E206.S6_DIR / f"cem/full/E206_{case_id}_prg/config_act.yaml"


def _convention(task_dir: Path) -> str:
    """The task's own scene_act_meta.json Euler convention (the runtime authority)."""
    import json

    meta = task_dir / "scene_act_meta.json"
    if not meta.is_file():
        return ""
    try:
        return str(json.loads(meta.read_text(encoding="utf-8")).get("euler_convention", ""))
    except (OSError, ValueError):
        return ""


def same(a: Any, b: Any) -> bool:
    """Equality with NaN == NaN (a float NaN would otherwise never compare equal)."""
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return a == b


def audit_one(row: dict[str, str]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "case_id": row["case_id"], "aug_variant": row["aug_variant"],
        "object_key": row["object_key"],
    }
    aug_path = C.repo_path(row["config_act"])
    e206_path = e206_config_act(row["case_id"])
    failures: list[str] = []

    if not aug_path.is_file():
        out.update({"status": "missing_aug_config", "failures": C.rel(aug_path)})
        return out
    if not e206_path.is_file():
        out.update({"status": "missing_e206_config", "failures": C.rel(e206_path)})
        return out

    aug = yaml.safe_load(aug_path.read_text(encoding="utf-8"))
    ref = yaml.safe_load(e206_path.read_text(encoding="utf-8"))
    keys = set(aug) | set(ref)
    diff = sorted(k for k in keys if not same(aug.get(k, "__M"), ref.get(k, "__M")))

    # F8 footprint: the object's actuator order follows the Euler convention.
    reordered: list[str] = []
    for key in sorted(ORDER_INSENSITIVE_KEYS & set(diff)):
        a, b = aug.get(key), ref.get(key)
        if isinstance(a, list) and isinstance(b, list) and sorted(a) == sorted(b):
            reordered.append(key)
        else:
            failures.append(f"set_differs:{key}({b!r}->{a!r})")
    out["reordered_keys"] = ";".join(reordered)
    out["euler_convention_aug"] = _convention(C.TASK_ROOT / str(aug.get("task", "")))
    out["euler_convention_e206"] = _convention(C.TASK_ROOT / str(ref.get("task", "")))
    out["euler_convention_changed"] = int(
        out["euler_convention_aug"] != out["euler_convention_e206"])

    unexpected = [k for k in diff
                  if k not in ALLOWED_DIFF_KEYS and k not in set(reordered)]
    if unexpected:
        failures.append("unexpected: " + ", ".join(
            f"{k}({ref.get(k, '__missing__')!r}->{aug.get(k, '__missing__')!r})"[:160]
            for k in unexpected))

    for key in MUST_MATCH_KEYS:
        if key not in aug or key not in ref:
            failures.append(f"absent:{key}")
        elif not same(aug[key], ref[key]):
            failures.append(f"must_match:{key}({ref[key]!r}->{aug[key]!r})")

    geom_keys = sorted(k for k in keys if k.endswith("_geom_ids"))
    geom_bad = [k for k in geom_keys if not same(aug.get(k), ref.get(k))]
    if geom_bad:
        failures.append("geom_ids differ: " + ",".join(geom_bad))

    out.update({
        "n_keys": len(keys), "n_diff": len(diff), "diff_keys": ";".join(diff),
        "n_geom_id_keys": len(geom_keys), "geom_ids_equal": int(not geom_bad),
        "contact_mask": aug.get("contact_hdmi_mask_path", ""),
        "status": "pass" if not failures else "fail",
        "failures": "; ".join(failures),
    })
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv-out", type=Path,
                    default=C.PREFLIGHT_DIR / "runtime_config_audit.tsv")
    ap.add_argument("--json-out", type=Path, default=C.RUNTIME_CONFIG_AUDIT_JSON)
    ap.add_argument("--strict-missing", action="store_true")
    args = ap.parse_args()

    rows = [r for r in C.read_tsv(C.PRIORITY_MANIFEST) if r.get("status") == "cem_ok"]
    if not rows:
        raise SystemExit("no cem_ok rows yet")

    results = [audit_one(r) for r in rows]
    missing = [r for r in results if str(r["status"]).startswith("missing")]
    failures = [r for r in results if r["status"] == "fail"]
    if args.strict_missing:
        failures += missing

    for r in results:
        if r["status"] != "pass":
            print(f"[{r['status']}] {r['case_id']:32s} {r['aug_variant']:7s} {r['failures']}")

    C.write_tsv(args.tsv_out, results)
    observed: set[str] = set()
    for r in results:
        observed.update(k for k in str(r.get("diff_keys", "")).split(";") if k)
    payload = {
        "experiment": C.EXP_ID, "check": "V6d", "generated_at": C.now(),
        "n_audited": len(results), "n_failures": len(failures), "n_missing": len(missing),
        "allowed_diff_keys": sorted(ALLOWED_DIFF_KEYS),
        "observed_diff_keys": sorted(observed),
        "unused_allowed_diff_keys": sorted(ALLOWED_DIFF_KEYS - observed),
        "must_match_keys": list(MUST_MATCH_KEYS),
        "order_insensitive_keys": sorted(ORDER_INSENSITIVE_KEYS),
        "geom_ids_equal_rows": sum(1 for r in results if r.get("geom_ids_equal") == 1),
        "rows_with_reordered_object_actuators": sum(
            1 for r in results if r.get("reordered_keys")),
        "rows_with_changed_euler_convention": sum(
            1 for r in results if r.get("euler_convention_changed") == 1),
        "verdict": "pass" if not failures else "fail",
        "note": ("config_act is written from the live Config after geom-id resolution "
                 "and model compile, so equal *_geom_ids means E206's lowgeom proxy "
                 "compiled to the same geoms in the same order in the aug scene -- a "
                 "stronger claim than V5c's XML parity, because it survives the compiler."),
    }
    C.write_json(args.json_out, payload)

    print(f"\nV6d {len(results) - len(failures)}/{len(results)} pass "
          f"({len(missing)} without a config_act yet)")
    print(f"  observed diff keys: {sorted(observed)}")
    print(f"  geom_ids identical in {payload['geom_ids_equal_rows']}/{len(results)} rows")
    print(f"-> {C.rel(args.json_out)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
