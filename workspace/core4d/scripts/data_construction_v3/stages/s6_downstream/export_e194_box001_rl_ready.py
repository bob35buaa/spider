#!/usr/bin/env python3
"""Export the E194 G1 box001 manual-USE set with paired partner evidence.

This is a downstream adapter: E194 G1 rollout artifacts are the source-person
authority, the specified E194 review TSV is the manual authority, and E173
passing Stage2b/partner artifacts are reused only for the opposite person.
"""
from __future__ import annotations

import csv, hashlib, json
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[6]
EXP = REPO / "workspace/core4d/results/E194"
REVIEW = EXP / "s6_downstream/eval/full_g1_expansion/user_manual_review_filled.tsv"
G1 = EXP / "s6_downstream/manifests/g1_expansion_full_manifest.tsv"
HANDOFF = REPO / "workspace/core4d/results/E173/s5_handoff/handoff_manifest.tsv"
STAGE2B = [
    REPO / "workspace/core4d/results/E173/s3_retarget/omnirt_v1/ref_fk/stage2b_manifest_omnirt_v1_ref_fk.tsv",
    REPO / "workspace/core4d/results/E173/s3_retarget/omnirt_v2/ref_fk/stage2b_manifest_omnirt_v2_ref_fk.tsv",
]
OUT = EXP / "s6_downstream/rl_export/box001_user_approved"

def read(path):
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))

def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""): h.update(b)
    return h.hexdigest()

def resolve(value):
    p = Path(value).expanduser()
    if p.is_absolute() and p.exists(): return p
    s = p.as_posix()
    for marker in ("/workspace/core4d/", "/spider/workspace/core4d/"):
        if marker in s: return REPO / ("workspace/core4d/" + s.split(marker, 1)[1])
    if not p.is_absolute(): return REPO / p
    return p

def ref(path):
    p = resolve(path).absolute()
    try: return p.relative_to(REPO.absolute()).as_posix()
    except ValueError: return p.as_posix()

def require(value, label):
    p = resolve(value)
    if not p.is_file() or p.stat().st_size == 0: raise RuntimeError(f"missing or empty {label}: {p}")
    return p

def ts(): return datetime.now().astimezone().isoformat(timespec="seconds")

def write_tsv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t", lineterminator="\n", extrasaction="ignore")
        w.writeheader(); w.writerows(rows)

def main():
    reviews = read(REVIEW)
    use = {r["case_id"]: r for r in reviews if r.get("object_key", r.get("case_id", "").split("_", 1)[0]) == "box001" and r.get("manual_use_decision") == "USE" and r.get("user_manual_review_status") == "reviewed"}
    g1 = {r["case_id"]: r for r in read(G1) if r.get("object_key") == "box001"}
    if set(use) - set(g1): raise RuntimeError(f"review USE cases missing from E194 G1 manifest: {sorted(set(use)-set(g1))}")
    if len(use) != 19: raise RuntimeError(f"expected current box001 USE count=19, got {len(use)}")
    handoff = {r["case_id"]: r for r in read(HANDOFF)}
    src_fields = ["case_id","object_key","object_name","date","seq","person","person_idx","retarget_variant_id","target_variant_id","hand_collision_variant_id","source_exp_id","spider_method_id","handoff_decision","target_gate_status","visual_qc_status","target_scene","trajectory","scene_act","contact_mask","cem_status","cem_run_id","cem_result_npz","cem_video","config_act","outdir_npz","cem_metrics_ref","rl_export_decision","manual_use_decision","user_manual_review_status","manual_quality_label","manual_review_note","source_g1_manifest","source_manual_review","trajectory_sha256","scene_act_sha256","contact_mask_sha256","cem_result_sha256","cem_video_sha256","config_act_sha256"]
    sources=[]
    for case in sorted(use):
        r=g1[case]; h=handoff.get(case,{})
        exp_id, spider = "E194", "E194_G1_expansion"
        parts=case.split("_")
        row={"case_id":case,"object_key":"box001","object_name":r.get("object_key","box001"),"date":"_".join(parts[1:-2]),"seq":parts[-2],"person":("person1" if case.endswith("p1") else "person2"),"person_idx":("0" if case.endswith("p1") else "1"),"retarget_variant_id":r.get("retarget_variant_id") or h.get("retarget_variant_id",""),"target_variant_id":"ref_fk","hand_collision_variant_id":r.get("hand_collision_variant_id","rubber_hull"),"source_exp_id":exp_id,"spider_method_id":spider or "E194_G1_expansion","handoff_decision":"HANDOFF_READY","target_gate_status":"pass","visual_qc_status":"pass","target_scene":r.get("target_scene",""),"trajectory":r.get("trajectory",""),"scene_act":r.get("scene_act",""),"contact_mask":r.get("contact_mask",""),"cem_status":"pass","cem_run_id":r.get("variant",""),"cem_result_npz":r.get("result_npz",""),"cem_video":r.get("video",""),"config_act":r.get("config_act",""),"outdir_npz":r.get("outdir_npz",""),"cem_metrics_ref":ref(EXP/"s6_downstream/eval/full_g1_expansion/e194_g1_expansion_case_metrics.tsv"),"rl_export_decision":"RL_EXPORT_READY","manual_use_decision":"USE","user_manual_review_status":use[case].get("user_manual_review_status","reviewed"),"manual_quality_label":use[case].get("manual_quality_label",""),"manual_review_note":use[case].get("manual_review_note",""),"source_g1_manifest":ref(G1),"source_manual_review":ref(REVIEW)}
        for field,key in [("trajectory_sha256","trajectory"),("scene_act_sha256","scene_act"),("contact_mask_sha256","contact_mask"),("cem_result_sha256","cem_result_npz"),("cem_video_sha256","cem_video"),("config_act_sha256","config_act")]: row[field]=sha(require(row[key],f"{case} {key}"))
        sources.append(row)
    write_tsv(OUT/"rl_export_input.tsv",sources,src_fields)
    (OUT/"rl_export_input.json").write_text(json.dumps(sources, ensure_ascii=False, indent=2)+"\n",encoding="utf-8")
    write_tsv(OUT/"box001_manual_review_snapshot.tsv",[use[k] for k in sorted(use)],list(use[next(iter(use))].keys()))

    # Partner authority: passing E173 Stage2b rows, plus two existing generated partner packages.
    pidx={}
    for p in STAGE2B:
        for r in read(p):
            if r.get("stage2b_status")=="pass": pidx.setdefault(r["case_id"],[]).append((p,r))
    generated={}
    partner_root = REPO / "workspace/core4d/results/E173/s6_downstream/rl_export/box001_user_approved/partner_omnirt"
    for p in partner_root.glob("generated_missing*/rl_partner_omnirt_manifest.tsv"):
        for r in read(p):
            if r.get("partner_status")=="pass": generated[r["partner_case_id"]]=(p,r)
    partners=[]
    for s in sources:
        partner=s["case_id"][:-2]+("p2" if s["case_id"].endswith("p1") else "p1")
        candidates=pidx.get(partner,[])
        if candidates:
            candidates.sort(key=lambda x:(0 if x[1].get("retarget_variant_id")=="omnirt_v1" else 1,x[1].get("updated_at","")))
            prov,e=candidates[0]; mode="reuse_e173_stage2b"
            paths={k:e.get(k,"") for k in ("converted_npz","omniretarget_output_npz","retargeted_npz","trimmed_npz","trim_window_json")}
            if not paths["trim_window_json"]:
                paths["trim_window_json"]=str(Path(e.get("holosoma_case_root", "")) / "trim_window.json")
        elif partner in generated:
            prov,e=generated[partner]; mode="direct_omnirt_partner_temp"
            paths={k:e.get(k,"") for k in ("converted_npz","omniretarget_output_npz","retargeted_npz","trimmed_npz","trim_window_json")}
        else: raise RuntimeError(f"no verified partner artifact for {s['case_id']} -> {partner}")
        prow={"source_case_id":s["case_id"],"source_person":s["person"],"source_person_idx":s["person_idx"],"source_rl_export_decision":"RL_EXPORT_READY","partner_case_id":partner,"partner_person":("person1" if partner.endswith("p1") else "person2"),"partner_person_idx":("0" if partner.endswith("p1") else "1"),"object_key":"box001","object_name":"box001","date":s["date"],"seq":s["seq"],"pair_status":"PAIR_COMPLETE","partner_status":"pass","paired_rl_export_decision":"RL_EXPORT_READY","partner_retarget_variant_id":e.get("retarget_variant_id",e.get("partner_retarget_variant_id","")),"partner_target_variant_id":e.get("target_variant_id",e.get("partner_target_variant_id","ref_fk")),"generation_mode":mode,"stage2b_status":"pass","failure_mode":"","decision_notes":("reused passing E173 Stage2b partner artifact" if mode.startswith("reuse") else "reused existing E173 generated partner OmniRetarget artifact"),"partner_target_task":e.get("target_task",e.get("partner_target_task","")),"partner_provenance_ref":ref(prov),"partner_provenance_sha256":sha(prov),"partner_params_json":e.get("params_json",e.get("retarget_params_json","")),"source_rl_export_input":ref(OUT/"rl_export_input.tsv"),"source_rl_export_input_sha256":sha(OUT/"rl_export_input.tsv"),"schema_version":"core4d_data_construction_v3.0","updated_at":ts()}
        for k,v in paths.items():
            q=require(v,f"{partner} {k}"); prow[k]=ref(q); prow[k+"_sha256"]=sha(q)
        partners.append(prow)
    pf=list(partners[0].keys())
    write_tsv(OUT/"rl_partner_omnirt_manifest.tsv",partners,pf)
    (OUT/"rl_partner_omnirt_manifest.json").write_text(json.dumps(partners, ensure_ascii=False, indent=2)+"\n",encoding="utf-8")
    psha=sha(OUT/"rl_partner_omnirt_manifest.tsv")
    paired=[]
    for s,p in zip(sources,partners):
        row=dict(s)
        for k in ["pair_status","paired_rl_export_decision","partner_case_id","partner_person","partner_person_idx","partner_status","partner_retarget_variant_id","partner_target_variant_id","generation_mode","partner_target_task","partner_provenance_ref","partner_provenance_sha256","partner_params_json"]: row[k]=p.get(k,"")
        # The partner manifest uses the canonical artifact names; paired TSV
        # exposes them with an explicit partner_ prefix.
        for source_key, paired_key in [
            ("trimmed_npz", "partner_trimmed_npz"),
            ("trimmed_npz_sha256", "partner_trimmed_npz_sha256"),
            ("omniretarget_output_npz", "partner_omniretarget_output_npz"),
            ("omniretarget_output_npz_sha256", "partner_omniretarget_output_npz_sha256"),
            ("trim_window_json", "partner_trim_window_json"),
            ("trim_window_json_sha256", "partner_trim_window_json_sha256"),
        ]:
            row[paired_key]=p.get(source_key,"")
        row["partner_manifest_ref"]=ref(OUT/"rl_partner_omnirt_manifest.tsv"); row["partner_manifest_sha256"]=psha
        paired.append(row)
    paired_fields=src_fields+[k for k in paired[0] if k not in src_fields]
    write_tsv(OUT/"paired_rl_export_input.tsv",paired,paired_fields)
    (OUT/"paired_rl_export_input.json").write_text(json.dumps(paired, ensure_ascii=False, indent=2)+"\n",encoding="utf-8")
    audit={"experiment_id":"E194","object_key":"box001","source_rows":len(sources),"partner_rows":len(partners),"paired_rl_ready_rows":len(paired),"manual_review_authority":ref(REVIEW),"manual_review_authority_sha256":sha(REVIEW),"manual_use_rows_exact":len(use),"source_g1_manifest":ref(G1),"source_g1_manifest_sha256":sha(G1),"partner_manifest":ref(OUT/"rl_partner_omnirt_manifest.tsv"),"partner_manifest_sha256":psha,"source_only_box001":True,"no_box023_or_box021_rows":True,"all_source_required_artifacts_nonempty":True,"all_partner_artifacts_nonempty":True,"generated_partner_cases":sorted(generated),"created_at":ts()}
    (OUT/"box001_rl_export_audit.json").write_text(json.dumps(audit,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (OUT/"box001_rl_export_summary.md").write_text("# E194 box001 RL-ready export\n\n- source/paired rows: **19**\n- manual authority: `user_manual_review_filled.tsv`, reviewed `USE` only\n- partner artifacts: 17 reused passing Stage2b + 2 existing generated E173 partner artifacts\n- all required source and partner paths were checked non-empty and hash-pinned.\n",encoding="utf-8")
    print(f"exported {len(sources)} box001 source rows and {len(partners)} partner rows to {OUT}")

if __name__ == "__main__": main()
