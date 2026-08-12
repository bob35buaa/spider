#!/usr/bin/env python3
"""Render independent OmniRetarget-only E197 videos.

Each MP4 contains only the converted OmniRetarget scene_act qpos replay.  No
Spider/PRG rollout or reference ghost is rendered.
"""
from __future__ import annotations

import argparse, csv, json, os, sys, tempfile
from pathlib import Path
os.environ.setdefault("MUJOCO_GL", "egl")
REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "workspace/core4d/scripts/eval/review"))
from viser_review_player import _load_portable_spec  # noqa: E402

OUT = REPO / "workspace/core4d/analysis/E197_full_cem_omnirt_vs_prg_metrics"
GATE = OUT / "e197_omni_absolute_wide_gate_filter.tsv"
METHOD = OUT / "e197_method_metrics.tsv"

def repo_path(s: str) -> Path:
    p = Path(s)
    if p.is_file(): return p.resolve()
    for marker in ("workspace/", "example_datasets/"):
        if marker in s:
            q = REPO / (marker + s.split(marker,1)[1])
            if q.is_file(): return q.resolve()
    return p

def camera_for(model, qpos):
    import mujoco, numpy as np
    d = mujoco.MjData(model); pts=[]
    for q in qpos[::max(1,len(qpos)//20)]:
        d.qpos[:] = q; mujoco.mj_forward(model,d); pts.append(d.xpos[1:].copy())
    a=np.concatenate(pts); lo,hi=a.min(0),a.max(0); center=(lo+hi)/2; span=hi-lo
    cam=mujoco.MjvCamera(); mujoco.mjv_defaultFreeCamera(model,cam)
    cam.type=int(mujoco.mjtCamera.mjCAMERA_FREE); cam.lookat[:]=center
    cam.distance=max(3.8, max(float(np.linalg.norm(span[:2]))*.5, float(span[2])*.85,1.0)*3.0)
    cam.azimuth=135.; cam.elevation=-22.
    return cam

def render(rec, out_path, max_frames=0):
    import imageio.v2 as imageio, mujoco, numpy as np
    from spider.viewers import setup_renderer
    from spider.viewers.viser_viewer import _ensure_names
    spec=_load_portable_spec(repo_path(rec['scene_xml'])); _ensure_names(spec); model=spec.compile()
    q=np.asarray(np.load(repo_path(rec['qpos_path']),allow_pickle=False)['qpos'],dtype=np.float64)
    if q.ndim!=2 or q.shape[1]!=model.nq: raise ValueError(f"{rec['case_id']}: qpos {q.shape} model nq={model.nq}")
    class C: save_video=True
    renderer=setup_renderer(C(),model); cam=camera_for(model,q); data=mujoco.MjData(model)
    ids=list(range(len(q))); 
    if max_frames>0: ids=ids[:max_frames]
    out_path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix='.'+out_path.stem+'.',suffix='.tmp.mp4',dir=out_path.parent); os.close(fd)
    writer=imageio.get_writer(tmp,fps=30,codec='libx264',quality=8)
    try:
        for i in ids:
            data.qpos[:]=q[i]; data.qvel[:]=0; mujoco.mj_forward(model,data)
            renderer.update_scene(data,camera=cam); im=renderer.render()
            writer.append_data(im)
        writer.close(); renderer.close(); Path(tmp).replace(out_path)
    except Exception:
        writer.close(); renderer.close(); Path(tmp).unlink(missing_ok=True); raise
    return len(ids),30

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--overwrite',action='store_true'); ap.add_argument('--max-frames',type=int,default=0); ap.add_argument('--limit',type=int,default=0); args=ap.parse_args()
    gate={r['case_id']:r for r in csv.DictReader(GATE.open(),delimiter='\t') if r.get('rl_filter_decision')=='RL_CANDIDATE_OMNI_WIDE_GATE_PASS'}
    methods={r['case_id']:r for r in csv.DictReader(METHOD.open(),delimiter='\t') if r.get('method')=='OmniRetarget'}
    outdir=OUT/'omnirt_videos'; outdir.mkdir(exist_ok=True)
    rows=[]; selected=list(gate)
    if args.limit: selected=selected[:args.limit]
    for case in selected:
        rec=methods[case]; path=outdir/(case+'_omnirt_only.mp4'); status='existing'
        try:
            if args.overwrite or not path.is_file(): n,fps=render(rec,path,args.max_frames); status='rendered'
            else:
                import subprocess; n=int(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0','-show_entries','stream=nb_read_frames','-of','default=nw=1:nk=1',str(path)]).decode().strip() or 0); fps=30
            print(f'[{status}] {case} frames={n}',flush=True)
            rows.append({'case_id':case,'object_key':rec['object_key'],'video':str(path.relative_to(REPO)),'status':status,'frames':n,'fps':fps,'content':'OmniRetarget converted qpos only; no Spider/PRG'})
        except Exception as e:
            print(f'[failed] {case}: {type(e).__name__}: {e}',file=sys.stderr); rows.append({'case_id':case,'object_key':rec['object_key'],'video':str(path.relative_to(REPO)),'status':'failed','error':f'{type(e).__name__}: {e}','content':'OmniRetarget only'})
    with (OUT/'omnirt_video_manifest.tsv').open('w',encoding='utf-8',newline='') as f:
        cols=['case_id','object_key','video','status','frames','fps','content','error']; w=csv.DictWriter(f,fieldnames=cols,delimiter='\t'); w.writeheader(); w.writerows(rows)
    summary={'gate_version':'E197-omni-absolute-wide-v4','selected_cases':len(selected),'rendered':sum(r['status'] in ('rendered','existing') for r in rows),'failed':sum(r['status']=='failed' for r in rows),'videos_are':'OmniRetarget-only (converted scene_act qpos), excluding Spider/PRG'}
    (OUT/'omnirt_video_render_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8'); print(json.dumps(summary,ensure_ascii=False))
    return 1 if summary['failed'] else 0
if __name__=='__main__': raise SystemExit(main())
