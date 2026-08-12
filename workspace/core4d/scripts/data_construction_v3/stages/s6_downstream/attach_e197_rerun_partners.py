#!/usr/bin/env python3
"""Attach successfully rerun E197 partner OmniRetarget motions to paired export."""
from pathlib import Path
import csv, hashlib, json, importlib.util
import numpy as np

REPO=Path(__file__).resolve().parents[6]
OUT=REPO/'workspace/core4d/results/E197/s6_downstream/rl_export'
RERUN=OUT/'partner_omnirt_rerun/omnirt_v1'
HS_PY=Path('/home/ubuntu/.holosoma_deps/miniconda3/envs/hsretargeting/bin/python')
HOLO=Path('/home/ubuntu/Workspace/holosoma')
CONVERTER=HOLO/'src/holosoma_retargeting/holosoma_retargeting/data_conversion/convert_data_format_mj.py'
CWD=HOLO/'src/holosoma_retargeting/holosoma_retargeting'
TARGETS={'box021_20231018_029_p1':'box021_20231018_029_p2','box021_20231018_034_p2':'box021_20231018_034_p1','box021_20231018_035_p2':'box021_20231018_035_p1','box023_20231020_039_p2':'box023_20231020_039_p1','box024_20231011_030_p1':'box024_20231011_030_p2'}

def safe(x): return ''.join(c if c.isalnum() or c in '_-' else '_' for c in x)
def ref(p):
 p=Path(p).absolute()
 try:return str(p.relative_to(REPO))
 except ValueError:return str(p)
def sha(p):
 h=hashlib.sha256();
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def npz(p):
 with np.load(p,allow_pickle=True) as z:return {k:z[k] for k in z.files}
def convert(src,dst,obj,log):
 raw=npz(src)['qpos']; inp=dst.with_name(dst.stem+'_qpos43_input.npz'); np.savez(inp,qpos=raw.astype(np.float32)); dst.parent.mkdir(parents=True,exist_ok=True)
 cmd=[str(HS_PY),str(CONVERTER),'--input-file',str(inp),'--input-fps','30','--output-fps','50','--has-dynamic-object','--object-name',obj,'--output-name',str(dst),'--once']
 p=__import__('subprocess').run(cmd,cwd=CWD,text=True,capture_output=True); log.write_text((p.stdout or '')+'\n'+(p.stderr or ''))
 if p.returncode: raise RuntimeError(p.stderr[-1000:])
def add(base,partner,dst):
 b=npz(base); p=npz(partner); names=[str(x) for x in p['body_names']]; wi=[names.index('left_wrist_yaw_link'),names.index('right_wrist_yaw_link')]; n=min(len(b['joint_pos']),len(p['joint_pos']))
 for k,v in list(b.items()):
  if isinstance(v,np.ndarray) and v.ndim and v.shape[0]>=n and k not in ('body_names','joint_names'): b[k]=v[:n]
 b['partner_hand_pos_w']=p['body_pos_w'][:n,wi,:].astype(np.float32); b['partner_hand_quat_w']=p['body_quat_w'][:n,wi,:].astype(np.float32); dst.parent.mkdir(parents=True,exist_ok=True); np.savez(dst,**b); return n

def main():
 rows=list(csv.DictReader((OUT/'source_omnirt_manifest.tsv').open(),delimiter='\t')); rerun=list(csv.DictReader((RERUN/'rl_partner_omnirt_manifest.tsv').open(),delimiter='\t')); rr={r['partner_case_id']:r for r in rerun}; partner_dir=OUT/'motion/partner'; paired_dir=OUT/'motion/paired'; logs=OUT/'converter_logs'
 for row in rows:
  if row['case_id'] not in TARGETS: continue
  pc=TARGETS[row['case_id']]; r=rr[pc]; src=Path(r['trimmed_npz']); obj=row['object_key'].replace('box','Box'); pobj=partner_dir/(safe(pc)+'.npz'); convert(src,pobj,obj,logs/(safe(pc)+'_rerun_v1.log')); pair=paired_dir/(safe(row['case_id'])+'__paired.npz'); n=add(Path(row['source_motion_npz']),pobj,pair)
  row.update(pair_status='PAIR_COMPLETE',paired_rl_export_decision='RL_EXPORT_READY',partner_source_motion_npz=ref(pobj),paired_motion_npz=ref(pair),alignment_policy='truncate_to_min_frames',paired_frames=str(n),partner_source_exp_id='E197_partner_rerun',partner_target_source='OmniRetarget',partner_retarget_variant_id='omnirt_v1',partner_rerun_variant='omnirt_v1',partner_rerun_manifest=ref(RERUN/'rl_partner_omnirt_manifest.tsv'),failure_mode='')
  row['partner_source_sha256']=sha(pobj); row['paired_motion_sha256']=sha(pair)
 fields=[]
 for r in rows:
  for k in r:
   if k not in fields:fields.append(k)
 with (OUT/'source_omnirt_manifest.tsv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields,delimiter='\t');w.writeheader();w.writerows(rows)
 with (OUT/'paired_rl_export_input.tsv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields,delimiter='\t');w.writeheader();w.writerows(rows)
 (OUT/'paired_rl_export_input.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
 summary=json.loads((OUT/'export_summary.json').read_text()); summary.update(pair_complete=sum(r['pair_status']=='PAIR_COMPLETE' for r in rows),blocked_partner_missing=sum(r['pair_status']=='BLOCKED_PARTNER_MISSING' for r in rows),partner_rerun_variant='omnirt_v1',partner_rerun_cases=list(TARGETS.values()))
 (OUT/'export_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps(summary,ensure_ascii=False))
if __name__=='__main__':main()
