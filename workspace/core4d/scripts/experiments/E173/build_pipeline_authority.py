#!/usr/bin/env python3
"""E173: build the frozen pipeline authority, funnel and terminal-state audit.

Reads the E173 S1-S6 manifests and produces, for every expected Box022/Box026
raw person-case, a single traceable terminal state plus the selected retarget
variant and v1->v2 rescue provenance. Emits a per-case authority TSV, a stage
funnel summary, and a terminal-state closure audit (no silent drops).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import e173_common as C


def index_by_case(path: Path, key: str = "case_id") -> dict[str, dict]:
    if not path.is_file():
        return {}
    return {r[key]: r for r in C.read_tsv(path) if r.get(key)}


def build() -> dict:
    R = C.RESULTS
    inv = [r for r in C.read_tsv(R / "s1_raw_contact/inventory/inventory.tsv")
           if r.get("object_key") in C.OBJECT_KEYS]
    rc3 = index_by_case(R / "s1_raw_contact/raw_contact/raw_contact_candidates_3cm.tsv")
    rc5 = index_by_case(R / "s1_raw_contact/raw_contact/raw_contact_candidates_5cm.tsv")
    v1 = index_by_case(R / "s3_retarget/omnirt_v1/ref_fk/stage2b_manifest_omnirt_v1_ref_fk.tsv")
    v2 = index_by_case(R / "s3_retarget/omnirt_v2/ref_fk/stage2b_manifest_omnirt_v2_ref_fk.tsv")
    g1 = index_by_case(R / "s4_gate_visual_qc/omnirt_v1/ref_fk/target_gate_manifest.tsv")
    g2 = index_by_case(R / "s4_gate_visual_qc/omnirt_v2/ref_fk/target_gate_manifest.tsv")
    vq1 = index_by_case(R / "s4_gate_visual_qc/omnirt_v1/ref_fk/visual_qc/visual_qc_manifest.tsv")
    vq2 = index_by_case(R / "s4_gate_visual_qc/omnirt_v2/ref_fk/visual_qc/visual_qc_manifest.tsv")
    cem = index_by_case(R / "s6_downstream/manifests/cem_full_manifest.tsv")
    scene_audit = index_by_case(R / "s6_downstream/preflight/scene_audit.tsv")

    rows = []
    for src in sorted(inv, key=lambda r: r["case_id"]):
        cid = src["case_id"]
        rc = rc3.get(cid, {})
        rc_decision = rc.get("raw_contact_decision", "") or ("raw_contact_pass" if cid in v1 else "")
        v1s = v1.get(cid, {}).get("stage2b_status", "")
        v2s = v2.get(cid, {}).get("stage2b_status", "")
        # selected variant + rescue provenance
        selected, rescue_of = "", ""
        if v1s == "pass":
            selected = "omnirt_v1"
        elif v1s == C.V1_INFEASIBLE_STATUS and v2s == "pass":
            selected = "omnirt_v2"
            rescue_of = f"omnirt_v1:{cid}:omniretarget_infeasible"
        gate = (g2 if selected == "omnirt_v2" else g1).get(cid, {}).get("target_gate_status", "")
        vq = (vq2 if selected == "omnirt_v2" else vq1).get(cid, {}).get("visual_qc_status", "")
        # CEM-eligible = admitted to full CEM (not preflight-blocked); accepts
        # post-run statuses (run_complete_pending_eval / numeric_*) so the
        # authority stays valid after CEM+eval, not only at the pre-run snapshot.
        in_cem = cid in cem and cem[cid].get("status", "") not in {"", "preflight_blocked"}
        prg = scene_audit.get(cid, {})
        prg_status = prg.get("status", "")
        # terminal state (single, traceable)
        if cid in v1 and v1s not in {"pass", C.V1_INFEASIBLE_STATUS}:
            terminal = f"REJECT_STAGE2B_{v1s.upper()}"
        elif v1s == C.V1_INFEASIBLE_STATUS and v2s != "pass":
            terminal = "REJECT_DUAL_OMNIRT_INFEASIBLE"
        elif cid not in v1:
            terminal = "REJECT_RAW_CONTACT" if rc_decision != "raw_contact_pass" else "NOT_STAGE2B_ELIGIBLE"
        elif gate != "pass":
            terminal = "REJECT_TARGET_GATE"
        elif vq not in {"pass"}:
            terminal = "REJECT_VISUAL_QC"
        elif prg_status == "preflight_blocked":
            terminal = "REJECT_PRG_SCENE_CONTRACT"
        elif in_cem:
            terminal = "CEM_ELIGIBLE"
        else:
            terminal = "PENDING"
        rows.append({
            "case_id": cid, "object_key": src["object_key"], "date": src.get("date", ""),
            "seq": src.get("seq", ""), "person": src.get("person", ""), "action": src.get("action", ""),
            "raw_contact_decision": rc_decision,
            "raw_contact_5cm_status": rc5.get(cid, {}).get("contact_target_status", ""),
            "stage2b_v1_status": v1s, "stage2b_v2_status": v2s,
            "selected_retarget_variant_id": selected, "rescue_of": rescue_of,
            "target_gate_status": gate, "visual_qc_status": vq,
            "prg_scene_contract": prg_status or ("pass" if in_cem else ""),
            "prg_scene_detail": prg.get("failures", ""),
            "cem_eligible": "true" if in_cem else "false",
            "terminal_state": terminal,
        })

    funnel = {
        "raw_expected": C.EXPECTED_PERSON_CASES,
        "raw_seen": len(rows),
        "by_object": {k: sum(r["object_key"] == k for r in rows) for k in C.OBJECT_KEYS},
        "raw_contact_pass": sum(r["raw_contact_decision"] == "raw_contact_pass" for r in rows),
        "stage2b_v1_pass": sum(r["stage2b_v1_status"] == "pass" for r in rows),
        "stage2b_v1_infeasible": sum(r["stage2b_v1_status"] == C.V1_INFEASIBLE_STATUS for r in rows),
        "stage2b_v2_rescued_pass": sum(r["selected_retarget_variant_id"] == "omnirt_v2" for r in rows),
        "dual_infeasible": sum(r["terminal_state"] == "REJECT_DUAL_OMNIRT_INFEASIBLE" for r in rows),
        "target_gate_pass": sum(r["target_gate_status"] == "pass" for r in rows),
        "visual_qc_pass": sum(r["visual_qc_status"] == "pass" for r in rows),
        "prg_scene_contract_reject": sum(r["terminal_state"] == "REJECT_PRG_SCENE_CONTRACT" for r in rows),
        "cem_eligible": sum(r["cem_eligible"] == "true" for r in rows),
        "selected_variant": dict(Counter(r["selected_retarget_variant_id"] for r in rows if r["selected_retarget_variant_id"])),
        "terminal_states": dict(Counter(r["terminal_state"] for r in rows)),
    }
    # closure audit: every raw row has exactly one non-empty terminal state
    missing_terminal = [r["case_id"] for r in rows if not r["terminal_state"] or r["terminal_state"] == "PENDING"]
    audit = {
        "created_at": C.now(),
        "raw_closure_pass": funnel["raw_seen"] == C.EXPECTED_PERSON_CASES and not missing_terminal,
        "raw_seen": funnel["raw_seen"], "raw_expected": C.EXPECTED_PERSON_CASES,
        "missing_or_pending_terminal": missing_terminal,
        "rescue_set_equals_v1_infeasible": funnel["stage2b_v1_infeasible"] == (
            funnel["stage2b_v2_rescued_pass"] + funnel["dual_infeasible"]),
        "cem_eligible_equals_frozen": funnel["cem_eligible"],
        "funnel": funnel,
    }
    C.write_tsv(R / "registries/pipeline_authority.tsv", rows)
    C.write_json(R / "registries/pipeline_funnel.json", funnel)
    C.write_json(R / "completion_audit/pipeline_authority_audit.json", audit)
    return audit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    audit = build()
    print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
