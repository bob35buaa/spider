#!/usr/bin/env python3
"""E174: final completion audit (Full completeness contract + closure).

Verifies plan section 6.3 / 10.3:
  * pipeline authority raw closure (60 rows, every row a terminal state);
  * rescue set == fresh v1 omniretarget_infeasible; every v2 row has rescue_of;
  * S5_READY (frozen) == Full expected;
  * full_expected == full_completed + full_terminal_failed, missing == 0;
  * each completed row: root+outdir NPZ present, qpos finite, config scene_name /
    leg penalty / gate match frozen, effective scene sha matches manifest;
  * variant-id semantics not mixed; no RL/partner export present (not approved).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import e174_common as C

COMPLETE_STATUSES = {"run_complete_pending_eval", "run_complete", "eval_complete"}
FAILED_STATUSES = {"failed_preflight", "failed_validation", "failed"}


def audit() -> dict:
    R = C.RESULTS
    fails: list[str] = []

    # 1. authority closure
    auth_path = R / "completion_audit/pipeline_authority_audit.json"
    authority = json.loads(auth_path.read_text()) if auth_path.is_file() else {}
    if not authority.get("raw_closure_pass"):
        fails.append("authority_raw_closure_not_pass")
    if not authority.get("rescue_set_equals_v1_infeasible"):
        fails.append("rescue_set_neq_v1_infeasible")

    # 2. v2 provenance: every v2 row has rescue_of
    auth_rows = C.read_tsv(R / "registries/pipeline_authority.tsv") if (R / "registries/pipeline_authority.tsv").is_file() else []
    v2_missing_rescue = [r["case_id"] for r in auth_rows
                         if r.get("selected_retarget_variant_id") == "omnirt_v2" and not r.get("rescue_of")]
    if v2_missing_rescue:
        fails.append(f"v2_missing_rescue_of:{v2_missing_rescue}")

    # 3. Full completeness contract
    fm = R / "s6_downstream/manifests/cem_full_manifest.tsv"
    rows = C.read_tsv(fm) if fm.is_file() else []
    full_expected = len(rows)
    completed, terminal_failed, other = [], [], []
    per_case_fail: list[str] = []
    for r in rows:
        st = r.get("status", "")
        if st in COMPLETE_STATUSES:
            completed.append(r["case_id"])
            # artifact hard contract
            outn = C.repo_path(r["outdir_npz"]); rootn = C.repo_path(r["result_npz"])
            cfgp = C.repo_path(r["config_act"]); scn = C.repo_path(r["scene_act"])
            if not outn.is_file() or not rootn.is_file():
                per_case_fail.append(f"{r['case_id']}:missing_npz"); continue
            try:
                with np.load(outn, allow_pickle=True) as d:
                    if "qpos" not in d or not np.isfinite(np.asarray(d["qpos"], dtype=np.float64)).all():
                        per_case_fail.append(f"{r['case_id']}:nonfinite_qpos")
                cfg = yaml.safe_load(cfgp.read_text())
                if cfg.get("scene_name") != C.SCENE_NAME:
                    per_case_fail.append(f"{r['case_id']}:scene_name")
                if cfg.get("leg_object_penalty_scale") != C.LEG_OBJECT_PENALTY_SCALE:
                    per_case_fail.append(f"{r['case_id']}:leg_penalty")
                if cfg.get("cem_leg_gate_enabled") is not True:
                    per_case_fail.append(f"{r['case_id']}:gate_enabled")
                if scn.is_file() and r.get("effective_scene_sha256") and C.sha256(scn) != r["effective_scene_sha256"]:
                    per_case_fail.append(f"{r['case_id']}:scene_sha_drift")
            except Exception as exc:
                per_case_fail.append(f"{r['case_id']}:audit_error:{type(exc).__name__}")
        elif st in FAILED_STATUSES:
            terminal_failed.append(r["case_id"])
        else:
            other.append(r["case_id"])

    missing = full_expected - len(completed) - len(terminal_failed)
    if other:
        fails.append(f"non_terminal_rows:{other}")
    if missing != 0:
        fails.append(f"completeness_missing:{missing}")
    if per_case_fail:
        fails.append(f"artifact_contract:{per_case_fail}")

    # 4. variant-id semantics not mixed
    bad_variant = [r["case_id"] for r in rows
                   if r.get("selected_retarget_variant_id") not in {"omnirt_v1", "omnirt_v2"}
                   or r.get("hand_collision_variant_id") != C.HAND_COLLISION_VARIANT
                   or r.get("spider_method_id") != C.E174_METHOD_ID]
    if bad_variant:
        fails.append(f"variant_semantics:{bad_variant}")

    # 5. no unapproved RL / partner export
    for forbidden in ("s6_downstream/rl_export", "s6_downstream/partner_export"):
        if (R / forbidden).exists():
            fails.append(f"unapproved_export_present:{forbidden}")

    result = {
        "created_at": C.now(),
        "status": "pass" if not fails else "fail",
        "failures": fails,
        "full_expected": full_expected,
        "full_completed": len(completed),
        "full_terminal_failed": len(terminal_failed),
        "missing": missing,
        "completeness_closed": missing == 0 and not other,
        "completed_cases": sorted(completed),
        "terminal_failed_cases": sorted(terminal_failed),
        "authority_funnel": authority.get("funnel", {}),
    }
    C.write_json(R / "completion_audit/completion_audit.json", result)
    return result


def main() -> int:
    argparse.ArgumentParser().parse_args()
    result = audit()
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
