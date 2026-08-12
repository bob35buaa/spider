#!/usr/bin/env python3
"""Export E197 OmniRetarget wide-v4 candidates to Holosoma RL contracts.

This is deliberately E197-specific.  The source qpos is scene-act (42 dims),
not the historical 43-dim free-joint layout.  We convert it explicitly,
invoke the Holosoma converter, then add the opposite person's wrist FK as a
partner contract.  Missing Omni partner data remains blocked and is never
silently replaced by PRG.
"""
from __future__ import annotations

import argparse, csv, hashlib, json, subprocess, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[6]
HOLOSOMA = Path('/home/ubuntu/Workspace/holosoma')
HS_PY = Path('/home/ubuntu/.holosoma_deps/miniconda3/envs/hsretargeting/bin/python')
CONVERTER = HOLOSOMA/'src/holosoma_retargeting/holosoma_retargeting/data_conversion/convert_data_format_mj.py'
CONVERT_CWD = HOLOSOMA/'src/holosoma_retargeting/holosoma_retargeting'
PERSON_PARTNER = {'person1':'person2','person2':'person1'}
PERSON_SHORT = {'person1':'p1','person2':'p2'}
PERSON_IDX = {'person1':'0','person2':'1'}

def sha(path):
    h=hashlib.sha256();
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def resolve(x):
    p=Path(str(x)).expanduser()
    return p if p.is_absolute() else REPO/p

def safe(x): return ''.join(c if c.isalnum() or c in '_-' else '_' for c in x)
def ref(p):
    p=Path(p).absolute()
    try: return str(p.relative_to(REPO))
    except ValueError: return str(p)

def scene_act_to_free(q42, scene):
    import mujoco
    from scipy.spatial.transform import Rotation as R
    model=mujoco.MjModel.from_xml_path(str(scene))
    bid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'object')
    if bid < 0: raise ValueError(f'object body missing: {scene}')
    meta=scene.with_name('scene_act_meta.json')
    conv='XYZ'
    if meta.exists(): conv=json.loads(meta.read_text()).get('euler_convention','XYZ')
    if q42.ndim!=2 or q42.shape[1]!=42: raise ValueError(f'expected (T,42), got {q42.shape}')
    base_pos=model.body_pos[bid].copy(); q=model.body_quat[bid].copy()
    base_rot=R.from_quat([q[1],q[2],q[3],q[0]])
    slide=q42[:,36:39]; euler=q42[:,39:42]
    pos=base_pos[None,:]+base_rot.apply(slide)
    rot=(base_rot*R.from_euler(conv,euler)).as_quat()
    out=np.zeros((len(q42),43),dtype=np.float64); out[:,:36]=q42[:,:36]
    out[:,36:39]=pos; out[:,39:43]=rot[:,[3,0,1,2]]
    return out, conv

def run_convert(inp, out, object_name, log):
    out.parent.mkdir(parents=True,exist_ok=True)
    cmd=[str(HS_PY),str(CONVERTER),'--input-file',str(inp),'--input-fps','30','--output-fps','50','--has-dynamic-object','--object-name',object_name,'--output-name',str(out),'--once']
    p=subprocess.run(cmd,cwd=CONVERT_CWD,text=True,capture_output=True)
    log.parent.mkdir(parents=True,exist_ok=True); log.write_text((p.stdout or '')+'\n'+(p.stderr or ''))
    if p.returncode: raise RuntimeError(f'converter failed ({p.returncode}): {p.stderr[-1200:]}')

def load_npz(path):
    with np.load(path,allow_pickle=True) as z: return {k:z[k] for k in z.files}

def nearest_contact(mask_path, person_idx, out_frames):
    if not mask_path or not resolve(mask_path).exists(): return None
    z=load_npz(resolve(mask_path)); key='raw_contact_mask_3cm'
    if key not in z: return None
    a=np.asarray(z[key]); a=a[:,int(person_idx),:].any(axis=-1).astype(np.float32)
    idx=np.rint(np.linspace(0,len(a)-1,out_frames)).astype(int)
    return a[idx]

def add_partner(base_path, partner_path, out_path):
    b=load_npz(base_path); p=load_npz(partner_path)
    names=[x.decode() if isinstance(x,bytes) else str(x) for x in p['body_names']]
    li=[names.index('left_wrist_yaw_link'),names.index('right_wrist_yaw_link')]
    n=min(len(b['joint_pos']),len(p['joint_pos']))
    for k,v in list(b.items()):
        if isinstance(v,np.ndarray) and v.ndim and v.shape[0]>=n and k not in ('body_names','joint_names'): b[k]=v[:n]
    b['partner_hand_pos_w']=np.asarray(p['body_pos_w'][:n,li,:],dtype=np.float32)
    b['partner_hand_quat_w']=np.asarray(p['body_quat_w'][:n,li,:],dtype=np.float32)
    out_path.parent.mkdir(parents=True,exist_ok=True); np.savez(out_path,**b)
    return n

def write_tsv(path, rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    fields=[]
    for r in rows:
        for k in r:
            if k not in fields: fields.append(k)
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields,delimiter='\t',extrasaction='ignore'); w.writeheader(); w.writerows(rows)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out-dir',type=Path,required=True); ap.add_argument('--force',action='store_true'); args=ap.parse_args()
    out=args.out_dir.expanduser().resolve(); motion=out/'motion'; source_dir=motion/'source'; partner_dir=motion/'partner'; paired_dir=motion/'paired'; q43dir=out/'qpos43_inputs'; logs=out/'converter_logs'
    for d in (source_dir,partner_dir,paired_dir,q43dir,logs): d.mkdir(parents=True,exist_ok=True)
    gate=REPO/'workspace/core4d/analysis/E197_full_cem_omnirt_vs_prg_metrics/e197_omni_absolute_wide_gate_filter.tsv'; methods=REPO/'workspace/core4d/analysis/E197_full_cem_omnirt_vs_prg_metrics/e197_method_metrics.tsv'
    gates=list(csv.DictReader(gate.open(),delimiter='\t')); pass_ids={r['case_id'] for r in gates if r.get('rl_filter_decision')=='RL_CANDIDATE_OMNI_WIDE_GATE_PASS'}
    all_omni=[r for r in csv.DictReader(methods.open(),delimiter='\t') if r.get('method')=='OmniRetarget']
    rows=[r for r in all_omni if r.get('case_id') in pass_ids]
    by={r['case_id']:r for r in all_omni}; source=[]; paired=[]
    for r in sorted(rows,key=lambda x:x['case_id']):
        cid=r['case_id']; partner_id=cid[:-2] + ('p1' if cid.endswith('p2') else 'p2')
        qpath=resolve(r['qpos_path']); scene=resolve(r['scene_xml']); obj=('box004' if r['object_key']=='box004' else r['object_key'].replace('box','Box'))
        q42=load_npz(qpath)['qpos']; q43, euler=scene_act_to_free(q42,scene)
        q43path=q43dir/(safe(cid)+'_qpos43.npz'); np.savez(q43path,qpos=q43.astype(np.float32))
        single=source_dir/(safe(cid)+'.npz')
        if args.force or not single.exists(): run_convert(q43path,single,obj,logs/(safe(cid)+'.log'))
        contact=nearest_contact(r.get('contact_mask',''),r.get('person_idx','0'),len(load_npz(single)['joint_pos']))
        if contact is not None:
            d=load_npz(single); d['object_contact']=contact.astype(np.float32); np.savez(single,**d)
        sr={'case_id':cid,'object_key':r['object_key'],'person_idx':r.get('person_idx',''),'retarget_variant_id':'omnirt_v1','target_variant_id':'ref_fk','source_exp_id':'E197','omni_gate_version':'E197-omni-absolute-wide-v4','target_source':'OmniRetarget','qpos42_path':ref(qpath),'qpos43_path':ref(q43path),'source_motion_npz':ref(single),'source_scene_act':ref(scene),'converter_scene_model':f'g1_29dof_w_{obj}.xml','scene_provenance_mismatch':'true','qpos_frames':str(len(q42)),'motion_frames':str(len(load_npz(single)['joint_pos'])),'partner_case_id':partner_id,'visual_qc_ref':'workspace/core4d/analysis/E197_full_cem_omnirt_vs_prg_metrics/omnirt_video_visual_audit.tsv','rl_export_decision':'RL_EXPORT_READY_SOURCE_ONLY'}
        if partner_id not in by:
            sr.update({'pair_status':'BLOCKED_PARTNER_MISSING','paired_rl_export_decision':'BLOCKED_PARTNER_MISSING','partner_source':'','failure_mode':'partner_omnirt_row_missing'})
        else:
            pr=by[partner_id]; pq42=load_npz(resolve(pr['qpos_path']))['qpos']; pscene=resolve(pr['scene_xml']); pq43,_=scene_act_to_free(pq42,pscene); pq43path=q43dir/(safe(partner_id)+'_qpos43.npz'); np.savez(pq43path,qpos=pq43.astype(np.float32)); pmotion=partner_dir/(safe(partner_id)+'.npz')
            pobj=('box004' if pr['object_key']=='box004' else pr['object_key'].replace('box','Box'))
            if args.force or not pmotion.exists(): run_convert(pq43path,pmotion,pobj,logs/(safe(partner_id)+'.log'))
            outpaired=paired_dir/(safe(cid)+'__paired.npz'); n=add_partner(single,pmotion,outpaired)
            sr.update({'pair_status':'PAIR_COMPLETE','paired_rl_export_decision':'RL_EXPORT_READY','partner_source_motion_npz':ref(pmotion),'paired_motion_npz':ref(outpaired),'alignment_policy':'truncate_to_min_frames','paired_frames':str(n),'partner_case_id':partner_id,'partner_source_exp_id':'E197','partner_target_source':'OmniRetarget','partner_qpos42_path':ref(resolve(pr['qpos_path']))})
        source.append(sr)
    write_tsv(out/'source_omnirt_manifest.tsv',source); write_tsv(out/'paired_rl_export_input.tsv',source)
    (out/'paired_rl_export_input.json').write_text(json.dumps(source,ensure_ascii=False,indent=2)+'\n')
    summary={'experiment_id':'E197','gate_version':'E197-omni-absolute-wide-v4','source_rows':len(source),'pair_complete':sum(x.get('pair_status')=='PAIR_COMPLETE' for x in source),'blocked_partner_missing':sum(x.get('pair_status')=='BLOCKED_PARTNER_MISSING' for x in source),'target_source':'OmniRetarget','partner_required':True,'alignment_policy':'truncate_to_min_frames','source_gate_manifest':str(gate.relative_to(REPO))}
    (out/'export_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    lines=['# E197 OmniRetarget RL export','',f"- source candidates: **{len(source)}**",f"- paired complete: **{summary['pair_complete']}**",f"- blocked partner missing: **{summary['blocked_partner_missing']}**",'- target: `OmniRetarget (omnirt_v1)`','- partner: opposite CORE4D person, wrist FK injected','- alignment: `truncate_to_min_frames`','- note: converter FK uses Holosoma g1 Box model; source scene_act is retained as provenance and is PRG-side rubberHull geometry.','', '| case | pair | source motion | paired motion |', '|---|---|---|---|']
    lines += [f"| `{x['case_id']}` | `{x.get('pair_status','')}` | `{x['source_motion_npz']}` | `{x.get('paired_motion_npz','')}` |" for x in source]
    (out/'export_summary.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(summary,ensure_ascii=False))
if __name__=='__main__': main()
