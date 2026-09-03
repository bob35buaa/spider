#!/usr/bin/env python3
"""E206 P8 exit check: the arm contract as the CEM run ACTUALLY saw it.

`build_overrides.py` audits the *composed Hydra* config before launch. This one
audits the `config_act.yaml` that `run_mjwp.py` dumped at the start of each real
run, which additionally carries runtime-resolved fields (`npair`, `model_path`,
`*_geom_ids`). plan236 P8 requires spot-checking >=4 of these.

Three classes of legitimate cross-arm difference, beyond `ARM_DIFF_KEYS`:
  * DERIVED   - resolved from a key that is already allowed to differ
                (`*_geom_ids` from `*_geom_names`, `model_path` from `scene_name`)
  * BY_DESIGN - `output_dir`, which MUST differ or the two arms would overwrite
                each other's rollouts
  * THE VARIABLE ITSELF - `npair`; PRG minus noPRG must equal exactly 16*N

Anything else is a confound and fails the audit.

NaN needs care: `cem_safety_gate_hard_floor_m` defaults to NaN (config.py:387)
and `nan != nan`, so a naive comparison reports a difference that is not one.

Usage:
    .venv/bin/python .../audit_runtime_arm_contract.py            # every finished pair
    ... --limit 4 --stage full
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

import e206_common as C  # noqa: E402

DERIVED = {"model_path", "leg_object_penalty_geom_ids", "cem_leg_gate_geom_ids"}
BY_DESIGN = {"output_dir"}
THE_VARIABLE = {"npair"}
BENIGN = DERIVED | BY_DESIGN | THE_VARIABLE


def same(a: Any, b: Any) -> bool:
    """Equality that treats NaN == NaN (NaN is a legitimate 'unset' default)."""
    if isinstance(a, float) and isinstance(b, float):
        if math.isnan(a) and math.isnan(b):
            return True
    return a == b


def audit_case(case_id: str, stage: str) -> dict[str, Any] | None:
    cfgs: dict[str, dict] = {}
    for arm in C.ARMS:
        path = C.arm_out_dir(arm, case_id, stage) / "config_act.yaml"
        if not path.is_file():
            return None
        cfgs[arm] = yaml.safe_load(path.read_text(encoding="utf-8"))

    keys = set(cfgs["noprg"]) | set(cfgs["prg"])
    diff = {k for k in keys
            if not same(cfgs["noprg"].get(k, "__m"), cfgs["prg"].get(k, "__m"))}
    fails: list[str] = []

    unexpected = diff - C.ARM_DIFF_KEYS - BENIGN
    if unexpected:
        fails.append(f"unexpected diff keys: {sorted(unexpected)}")
    if "scene_name" not in diff:
        fails.append("scene_name identical -> both arms ran one scene")
    if "output_dir" not in diff:
        fails.append("output_dir identical -> arms would overwrite each other")

    # npair delta must be exactly 16*N: that is the whole experiment.
    n_boxes = None
    scene = Path(str(cfgs["noprg"].get("model_path", "")))
    if scene.is_file():
        import xml.etree.ElementTree as ET
        root = ET.parse(scene).getroot()
        n_boxes = sum(
            1 for body in root.iter("body") if body.get("name") == "object"
            for g in body.findall("geom")
            if (g.get("name") or "").startswith("object_collision")
        )
    delta = int(cfgs["prg"].get("npair", 0)) - int(cfgs["noprg"].get("npair", 0))
    if n_boxes:
        expected = C.N_LEG_GEOMS * n_boxes
        if delta != expected:
            fails.append(f"npair delta {delta} != 16x{n_boxes}={expected}")

    for arm, cfg in cfgs.items():
        if cfg.get("scene_name") != C.SCENE_BY_ARM[arm]:
            fails.append(f"{arm}: scene_name={cfg.get('scene_name')}")
        if cfg.get("object_collision_sdf_mode") != "union":
            fails.append(f"{arm}: sdf_mode={cfg.get('object_collision_sdf_mode')}")
        for k, v in C.E163_HAND_GATE.items():   # no A2 in E206
            got = cfg.get(k)
            ok = (bool(got) == v) if isinstance(v, bool) else (
                got is not None and abs(float(got) - float(v)) <= 1e-9)
            if not ok:
                fails.append(f"{arm}: {k}={got}!=E163({v})")
    if float(cfgs["noprg"].get("leg_object_penalty_scale", -1)) != 0.0:
        fails.append("noprg leg_object_penalty_scale!=0")
    if bool(cfgs["noprg"].get("cem_leg_gate_enabled", True)):
        fails.append("noprg cem_leg_gate_enabled!=false")
    if float(cfgs["prg"].get("leg_object_penalty_scale", -1)) != 2.0:
        fails.append("prg leg_object_penalty_scale!=2.0")
    if not bool(cfgs["prg"].get("cem_leg_gate_enabled", False)):
        fails.append("prg cem_leg_gate_enabled!=true")

    return {
        "case_id": case_id,
        "object_geom_count": n_boxes or "",
        "npair_noprg": cfgs["noprg"].get("npair"),
        "npair_prg": cfgs["prg"].get("npair"),
        "npair_delta": delta,
        "diff_keys": ",".join(sorted(diff)),
        "status": "pass" if not fails else "fail",
        "fails": ";".join(fails),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="full", choices=["full", "smoke"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    manifests = (
        C.S3_DIR / "omnirt_v1/ref_fk/stage2b_manifest_omnirt_v1_ref_fk.tsv",
        C.S3_DIR / "omnirt_v2/ref_fk/stage2b_manifest_omnirt_v2_ref_fk.tsv",
    )
    case_ids: set[str] = set()
    for m in manifests:
        if m.is_file():
            case_ids |= {r["case_id"] for r in C.read_tsv(m)
                         if r.get("stage2b_status") == "pass"}

    rows = []
    for case_id in sorted(case_ids):
        row = audit_case(case_id, args.stage)
        if row is None:
            continue
        rows.append(row)
        print(f"{row['case_id']:34s} {row['status']:5s} "
              f"npair {row['npair_noprg']}->{row['npair_prg']} "
              f"(+{row['npair_delta']}) {row['fails']}", flush=True)
        if args.limit and len(rows) >= args.limit:
            break

    if not rows:
        raise SystemExit("no config_act.yaml pairs on disk yet")
    out_dir = args.out_dir or (C.S6_DIR / "cem" / args.stage)
    out_dir.mkdir(parents=True, exist_ok=True)
    C.write_tsv(out_dir / "runtime_arm_contract.tsv", rows)
    failed = [r for r in rows if r["status"] != "pass"]
    (out_dir / "runtime_arm_contract.json").write_text(json.dumps({
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "stage": args.stage,
        "audited_pairs": len(rows),
        "passed": len(rows) - len(failed),
        "allowed_diff_keys": sorted(C.ARM_DIFF_KEYS),
        "benign_runtime_keys": {"derived": sorted(DERIVED),
                                "by_design": sorted(BY_DESIGN),
                                "the_variable": sorted(THE_VARIABLE)},
        "failures": failed,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\npass={len(rows) - len(failed)}/{len(rows)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
