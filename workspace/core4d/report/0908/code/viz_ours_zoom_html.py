"""Interactive, viewpoint-adjustable HTML demo of the "Ours" (SPIDER) hand/object close-up.

This is the browser twin of viz_ours_zoom.py: it exports the exact same geometry
(spider right-hand + object, left arm hideable, no floor) into a self-contained
Three.js page so you can orbit freely, read the camera angle live in OUR convention
(az-offset from robot facing, elevation, hand-radius framing), save/restore chosen
viewpoints with thumbnails, download a square PNG at the current viewpoint, and copy
the exact viz_ours_zoom.py CLI to reproduce it at high resolution offline.

Run (headless):
    PYOPENGL_PLATFORM=osmesa MUJOCO_GL=osmesa .venv/bin/python \
        workspace/core4d/report/0908/code/viz_ours_zoom_html.py
then open the printed .html in a browser.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "osmesa")

import argparse  # noqa: E402

import numpy as np  # noqa: E402
import mujoco  # noqa: E402

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from viz_methods_compare import DEFAULT_SCENE, apply_T, facing_yaw  # noqa: E402
from viz_paper_figure import build_alignment  # noqa: E402
from viz_mixed_robot_smplx import _geom_rgb, object_world_mesh  # noqa: E402

LEFT_ARM = ["left_shoulder_pitch_link", "left_shoulder_roll_link", "left_shoulder_yaw_link",
            "left_elbow_link", "left_wrist_roll_link", "left_wrist_pitch_link",
            "left_wrist_yaw_link"]
# Both upper limbs are kept (shoulder -> wrist, hand mesh rides on the wrist link); torso/
# legs/pelvis ("body") are dropped so the close-up shows just the grasp. Each arm is tagged
# by side so the viewer can mask either hand independently.
RIGHT_ARM_ROOT = "right_shoulder_pitch_link"
LEFT_ARM_ROOT = "left_shoulder_pitch_link"
BOX_RGB = [0.24, 0.36, 0.54]                 # matches render_style.BOX_COLOR

# ours_zoom.py defaults -- the viewer OPENS on the viewpoint the user picked below.
DEFAULT_AZ_OFFSET = -50.0
DEFAULT_EL = 25.0
DEFAULT_RADIUS = 0.307


def build_meshlib(model):
    """Local-frame vertices/faces for every unique mesh dataid (stored once)."""
    lib = {}
    for did in range(model.nmesh):
        va, vn = model.mesh_vertadr[did], model.mesh_vertnum[did]
        fa, fn = model.mesh_faceadr[did], model.mesh_facenum[did]
        v = model.mesh_vert[va:va + vn].reshape(-1, 3)
        f = model.mesh_face[fa:fa + fn].reshape(-1, 3)
        lib[did] = {"v": v.round(4).ravel().tolist(), "f": f.astype(int).ravel().tolist()}
    return lib


def subtree_bids(model, root_name):
    """Body ids of the kinematic subtree rooted at root_name (inclusive)."""
    root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, root_name)
    keep = {root}
    for bid in range(model.nbody):
        p = bid
        while p != 0:
            if p == root:
                keep.add(bid)
                break
            p = int(model.body_parentid[p])
    return keep


def instances(model, data, T, obj_bid, right_bids, left_bids):
    """Per-geom instance refs for both arms + the object, each tagged by side ('s')."""
    TR, Tp = T[:3, :3], T[:3, 3]
    out = []
    for gid in range(model.ngeom):
        if model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        bid = int(model.geom_bodyid[gid])
        if bid == obj_bid:
            s = "obj"
        elif bid in right_bids:
            s = "right"
        elif bid in left_bids:
            s = "left"
        else:
            continue
        R = TR @ data.geom_xmat[gid].reshape(3, 3)
        p = TR @ data.geom_xpos[gid] + Tp
        rgb = BOX_RGB if s == "obj" else [round(float(c), 3) for c in _geom_rgb(model, gid)]
        out.append({"m": int(model.geom_dataid[gid]),
                    "R": R.round(5).ravel().tolist(), "p": p.round(4).tolist(),
                    "c": rgb, "s": s})
    return out


def main():
    ap = argparse.ArgumentParser(description="Interactive Ours hand close-up (Three.js HTML).")
    ap.add_argument("--case", default="box021_20231011_037")
    ap.add_argument("--scene", default=str(DEFAULT_SCENE))
    ap.add_argument("--file-frame", type=int, default=12,
                    help="frame index in the build_alignment window (box021/E170 path)")
    # Direct-load mode: load a SPIDER-CEM rollout npz straight and pick a raw frame,
    # bypassing the box021-hardwired build_alignment (needed for other experiments, e.g.
    # box001/E198). --cem-npz + --scene + --frame together select the exact pose.
    ap.add_argument("--cem-npz", default=None,
                    help="SPIDER-CEM rollout npz; enables direct-load mode (T=identity)")
    ap.add_argument("--frame", type=int, default=0, help="raw qpos index for --cem-npz mode")
    ap.add_argument("--person", default="p2", help="person tag for the default output dir")
    ap.add_argument("--open-az", type=float, default=DEFAULT_AZ_OFFSET)
    ap.add_argument("--open-el", type=float, default=DEFAULT_EL)
    ap.add_argument("--open-radius", type=float, default=DEFAULT_RADIUS)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    scene = args.scene
    model = mujoco.MjModel.from_xml_path(scene)
    bid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
    wrist = {"right": bid("right_wrist_yaw_link"), "left": bid("left_wrist_yaw_link")}
    obj_bid = bid("object")
    right_bids = subtree_bids(model, RIGHT_ARM_ROOT)
    left_bids = subtree_bids(model, LEFT_ARM_ROOT)
    data = mujoco.MjData(model)

    # Resolve the full per-frame qpos list + which frame to open on + how to label frames.
    if args.cem_npz:                                     # direct-load mode (native frame)
        q = np.load(args.cem_npz)["qpos"]
        q = q[:, 0, :] if q.ndim == 3 else q
        assert 0 <= args.frame < len(q), f"frame {args.frame} outside [0,{len(q)})"
        T = np.eye(4)
        facing = facing_yaw(q)
        qvecs = list(q)                                  # labels == raw qpos index (matches PNGs)
        frame_num0, open_i = 0, args.frame
    else:                                                # box021/E170 build_alignment path
        qpos, ref, regs, (w0, w1) = build_alignment(scene, args.case)
        assert 0 <= args.file_frame < (w1 - w0), \
            f"file-frame {args.file_frame} outside window [0,{w1 - w0})"
        lag, T = regs["spider"]
        facing = facing_yaw(ref)
        qvecs = [qpos["spider"][w0 + ff - lag] for ff in range(w1 - w0)]
        frame_num0, open_i = 0, args.file_frame

    def _unit(v):
        v = np.asarray(v, float)
        n = np.linalg.norm(v)
        return (v / n).round(5).tolist() if n > 1e-9 else [0.0, 0.0, 1.0]

    def frame_record(qv):
        """Both-arm instances + per-hand contact target and away-directions for one frame."""
        data.qpos[:] = qv
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        inst = instances(model, data, T, obj_bid, right_bids, left_bids)
        ov, _, _ = object_world_mesh(model, data)
        ov = apply_T(ov, T)
        obj_center = ov.mean(0)
        rec = {"inst": inst}
        for side, wb in wrist.items():
            hand = apply_T(data.xpos[wb][None], T)[0]
            nearest = ov[np.argmin(np.linalg.norm(ov - hand, axis=1))]
            contact = (0.5 * (hand + nearest)).round(4).tolist()
            away = {"normal": _unit(hand - nearest), "box": _unit(hand - obj_center),
                    "horiz": _unit([*(hand - obj_center)[:2], 0.0]), "up": [0.0, 0.0, 1.0]}
            rec[side] = {"contact": contact, "away": away}
        return rec

    frames = [frame_record(qv) for qv in qvecs]

    data_js = {
        "case": args.case, "facing": facing, "meshlib": build_meshlib(model),
        "frames": frames, "frame_num0": frame_num0, "open": open_i,
        "default": {"az": args.open_az, "el": args.open_el, "radius": args.open_radius},
    }

    out = Path(args.out) if args.out else \
        REPO / (f"workspace/core4d/report/0908/paper_results/viz/"
                f"{args.case}_{args.person}/paper_figure/ours_zoom_viewer.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(HTML.replace("__DATA__", json.dumps(data_js)))
    print(f"[ok] {out}  ({len(frames)} frames, open on {frame_num0 + open_i})")
    print("open in a browser; scrub frames, orbit, pull the arm, save viewpoints, download PNGs.")


HTML = r"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Ours zoom viewer</title>
<style>
  :root{--panel:rgba(255,255,255,.94)}
  html,body{margin:0;height:100%;overflow:hidden;font-family:monospace}
  /* checkerboard body bg so a transparent render reads as transparent.
     Kept on <body> (non-positioned) so the static WebGL canvas paints ON TOP of it. */
  body{background-image:linear-gradient(45deg,#ccc 25%,transparent 25%),linear-gradient(-45deg,#ccc 25%,transparent 25%),linear-gradient(45deg,transparent 75%,#ccc 75%),linear-gradient(-45deg,transparent 75%,#ccc 75%);
    background-size:24px 24px;background-position:0 0,0 12px,12px -12px,-12px 0px;background-color:#e9ebee}
  body.white{background:#fff}
  canvas{display:block}
  #ui{position:fixed;top:10px;left:10px;background:var(--panel);padding:12px 14px;
      border-radius:10px;font-size:13px;line-height:1.55;box-shadow:0 2px 10px rgba(0,0,0,.25);
      max-width:340px}
  #ui b{font-size:14px}
  #readout{white-space:pre;margin:6px 0;color:#2c3e50}
  #cli{color:#c0392b;user-select:all;word-break:break-all;display:block;margin:2px 0 6px}
  button,select{font-family:monospace;font-size:13px;margin:2px 4px 2px 0;padding:3px 7px;
      border:1px solid #bbb;border-radius:6px;background:#f6f6f6;cursor:pointer}
  button:hover{background:#eceff1}
  label{cursor:pointer}
  #saved{margin-top:10px;max-height:46vh;overflow:auto;border-top:1px solid #ddd;padding-top:8px}
  .vp{display:flex;gap:8px;align-items:flex-start;margin-bottom:8px;padding-bottom:8px;border-bottom:1px dashed #e0e0e0}
  .vp img{width:72px;height:72px;border:1px solid #ccc;border-radius:4px;flex:0 0 auto;
      background-image:linear-gradient(45deg,#ddd 25%,transparent 25%),linear-gradient(-45deg,#ddd 25%,transparent 25%),linear-gradient(45deg,transparent 75%,#ddd 75%),linear-gradient(-45deg,transparent 75%,#ddd 75%);
      background-size:12px 12px;background-position:0 0,0 6px,6px -6px,-6px 0}
  .vp .meta{font-size:11px;color:#2c3e50;flex:1 1 auto}
  .vp .meta .cli{color:#c0392b;user-select:all;word-break:break-all}
  .vp .row{margin-top:3px}
  .hint{color:#888;font-size:11px;margin-top:4px}
</style></head><body>
<div id="ui">
  <b>Ours zoom viewer</b> <span id="title" style="font-weight:normal;color:#888"></span><br>
  <div class="row">
    frame: <button id="prev">◀</button>
    <input type="range" id="frame" min="0" max="0" step="1" value="0" style="width:150px;vertical-align:middle">
    <button id="next">▶</button>
    <button id="play">play</button>
    <span id="fnum">0</span>
  </div>
  <div class="row">
    bg: <select id="bg">
      <option value="transparent">transparent</option>
      <option value="white">white</option>
    </select>
    show: <label><input type="checkbox" id="show_right" checked> right</label>
    <label><input type="checkbox" id="show_left" checked> left</label>
  </div>
  <div class="row">
    pull: <select id="dir">
      <option value="normal">contact normal</option>
      <option value="box">box→hand</option>
      <option value="horiz">horizontal</option>
      <option value="up">straight up</option>
    </select>
    <input type="range" id="off" min="-15" max="20" step="0.5" value="0" style="width:120px;vertical-align:middle">
    <span id="offval">0.0 cm</span>
  </div>
  <div id="readout"></div>
  <span id="cli"></span>
  <div class="row">
    <button id="reset">Reset view</button>
    <button id="save">★ Save viewpoint</button>
    <button id="png">⬇ Download PNG</button>
    res <select id="res">
      <option>700</option><option selected>1400</option><option>2100</option>
    </select>
  </div>
  <div class="hint">drag = orbit · scroll = zoom · right-drag = pan · frame slider scrubs, view stays put</div>
  <div id="saved"></div>
</div>
<script src="https://unpkg.com/three@0.128.0/build/three.min.js"></script>
<script src="https://unpkg.com/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<script>
const DATA = __DATA__;
const FOV = 45;                       // vertical fov (deg) -- matches yfov=pi/4 in the render

const FRAMES = DATA.frames, NF = FRAMES.length;
let CURF = DATA.open;                  // current frame index into FRAMES
document.getElementById('title').textContent = `${DATA.case}  (${NF} frames)`;

const scene = new THREE.Scene();      // no scene.background -> canvas stays transparent
const renderer = new THREE.WebGLRenderer({antialias:true, alpha:true, preserveDrawingBuffer:true});
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setClearColor(0x000000, 0);
document.body.appendChild(renderer.domElement);

const camera = new THREE.PerspectiveCamera(FOV, window.innerWidth/window.innerHeight, 0.01, 100);
camera.up.set(0,0,1);                 // Z-up world
const controls = new THREE.OrbitControls(camera, renderer.domElement);

// lighting ~ soft phong close-up: ambient + a few directionals (no harsh specular)
scene.add(new THREE.AmbientLight(0xffffff, 0.8));
const dirs = [[2,-2,4,0.6],[-2,2,3,0.35],[0,0,5,0.25]];
dirs.forEach(([x,y,z,i])=>{const d=new THREE.DirectionalLight(0xffffff,i);d.position.set(x,y,z);scene.add(d);});

// shared geometry library (built once from local mesh verts/faces)
const GEO = {};
for (const did in DATA.meshlib){
  const s = DATA.meshlib[did];
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(s.v,3));
  g.setIndex(s.f); g.computeVertexNormals();
  GEO[did] = g;
}

const target = new THREE.Vector3();
let SIDE = {right: FRAMES[DATA.open].right, left: FRAMES[DATA.open].left};  // current-frame per-hand data

const rightArm = new THREE.Group(), leftArm = new THREE.Group(), box = new THREE.Group();
scene.add(rightArm); scene.add(leftArm); scene.add(box);
function groupFor(s){ return s === 'obj' ? box : (s === 'left' ? leftArm : rightArm); }
function shown(side){ return document.getElementById('show_' + side).checked; }
function applyVisibility(){ rightArm.visible = shown('right'); leftArm.visible = shown('left'); }
function instMesh(inst){
  const mat = new THREE.MeshStandardMaterial({color:new THREE.Color(inst.c[0],inst.c[1],inst.c[2]),
      roughness:0.55, metalness:0.0, side:THREE.DoubleSide});
  const mesh = new THREE.Mesh(GEO[inst.m], mat);
  const R = inst.R, p = inst.p;
  const m4 = new THREE.Matrix4();
  m4.set(R[0],R[1],R[2],p[0], R[3],R[4],R[5],p[1], R[6],R[7],R[8],p[2], 0,0,0,1);
  mesh.matrixAutoUpdate = false; mesh.matrix.copy(m4);
  return mesh;
}
function clearGroup(g){ while(g.children.length){ const c=g.children[0]; g.remove(c); if(c.material)c.material.dispose(); } }

// Centre the orbit target on the visible hands' contact(s) (both hidden -> midpoint of both).
function setTargetFromVisible(){
  const cs = [];
  if (shown('right')) cs.push(SIDE.right.contact);
  if (shown('left'))  cs.push(SIDE.left.contact);
  if (!cs.length) cs.push(SIDE.right.contact, SIDE.left.contact);
  let x=0,y=0,z=0; cs.forEach(p=>{x+=p[0];y+=p[1];z+=p[2];});
  target.set(x/cs.length, y/cs.length, z/cs.length);
  controls.target.copy(target);
}

// Swap in frame i. keepView preserves the current az/el/radius framing and only re-centers
// the orbit target on the visible hands, so they stay framed while scrubbing.
function loadFrame(i, keepView){
  CURF = ((i % NF) + NF) % NF;
  const fr = FRAMES[CURF];
  const v = keepView ? currentView() : DATA.default;
  clearGroup(rightArm); clearGroup(leftArm); clearGroup(box);
  fr.inst.forEach(inst=>{ groupFor(inst.s).add(instMesh(inst)); });
  SIDE = {right: fr.right, left: fr.left};
  applyVisibility();
  setTargetFromVisible();
  setView(v.az, v.el, v.radius);
  document.getElementById('frame').value = CURF;
  document.getElementById('fnum').textContent = (DATA.frame_num0 + CURF);
  applyOffset();
}

// pull each arm away from the box along the chosen direction (box stays fixed)
function offsetMeters(){ return parseFloat(document.getElementById('off').value)/100; }
function dirName(){ return document.getElementById('dir').value; }
function applyOffset(){
  const d = offsetMeters(), dn = dirName();
  const r = SIDE.right.away[dn], l = SIDE.left.away[dn];
  rightArm.position.set(r[0]*d, r[1]*d, r[2]*d);
  leftArm.position.set(l[0]*d, l[1]*d, l[2]*d);
  const cm = d*100;
  document.getElementById('offval').textContent =
      cm.toFixed(1)+' cm '+(cm > 0.05 ? 'away' : cm < -0.05 ? 'toward' : '');
  readout();
}

function applyBg(){
  const v = document.getElementById('bg').value;
  document.body.classList.toggle('white', v === 'white');
  renderer.setClearColor(v === 'white' ? 0xffffff : 0x000000, v === 'white' ? 1 : 0);
}

function setView(azOff, el, radius){
  const az = (DATA.facing + azOff)*Math.PI/180, e = el*Math.PI/180;
  const dir = new THREE.Vector3(Math.cos(e)*Math.cos(az), Math.cos(e)*Math.sin(az), Math.sin(e));
  const dist = radius/Math.tan(FOV/2*Math.PI/180);
  camera.position.copy(target).add(dir.multiplyScalar(dist));
  controls.update();
}
function currentView(){
  const dir = camera.position.clone().sub(target);
  const dist = dir.length(); dir.normalize();
  let azOff = Math.atan2(dir.y, dir.x)*180/Math.PI - DATA.facing;
  azOff = ((azOff+180)%360+360)%360 - 180;              // wrap to [-180,180]
  const el = Math.asin(dir.z)*180/Math.PI;
  const radius = dist*Math.tan(FOV/2*Math.PI/180);
  return {az:azOff, el:el, radius:radius};
}
function cliFor(v){
  const bg = document.getElementById('bg').value;
  return `uv run workspace/core4d/report/0908/code/viz_ours_zoom.py --case ${DATA.case} `
       + `--file-frame ${DATA.frame_num0 + CURF} --hand-az-offset ${v.az.toFixed(1)} `
       + `--hand-el ${v.el.toFixed(1)} --hand-radius ${v.radius.toFixed(3)} --bg ${bg}`;
}
function readout(){
  const v = currentView();
  const cm = offsetMeters()*100, dn = dirName();
  const word = cm > 0.05 ? ' away' : cm < -0.05 ? ' toward' : '';
  document.getElementById('readout').textContent =
     `az_offset = ${v.az.toFixed(1)} deg\nel        = ${v.el.toFixed(1)} deg\n`
   + `radius    = ${v.radius.toFixed(3)} m\narm_pull  = ${cm.toFixed(1)} cm${word}  [${dn}]`;
  document.getElementById('cli').textContent = cliFor(v);
}

// render the current view into a square canvas (matches the render's 1:1 aspect)
function snapshotDataURL(size){
  const w = window.innerWidth, h = window.innerHeight;
  renderer.setSize(size, size, false);
  camera.aspect = 1; camera.updateProjectionMatrix();
  renderer.render(scene, camera);
  const url = renderer.domElement.toDataURL('image/png');
  renderer.setSize(w, h, false);
  camera.aspect = w/h; camera.updateProjectionMatrix();
  return url;
}

function frameTag(){ return `f${DATA.frame_num0 + CURF}`; }
document.getElementById('bg').onchange = applyBg;
['show_right','show_left'].forEach(id=>{ document.getElementById(id).onchange = ()=>{
  const v = currentView(); applyVisibility(); setTargetFromVisible(); setView(v.az, v.el, v.radius); applyOffset();
};});
document.getElementById('off').oninput = applyOffset;
document.getElementById('dir').onchange = applyOffset;
document.getElementById('reset').onclick = ()=>setView(DATA.default.az, DATA.default.el, DATA.default.radius);
document.getElementById('png').onclick = ()=>{
  const size = parseInt(document.getElementById('res').value, 10);
  const a = document.createElement('a');
  const v = currentView();
  a.download = `${DATA.case}_${frameTag()}_az${v.az.toFixed(0)}_el${v.el.toFixed(0)}_r${v.radius.toFixed(2)}.png`;
  a.href = snapshotDataURL(size); a.click();
};
document.getElementById('save').onclick = ()=>{
  const v = currentView(), cli = cliFor(v), tag = frameTag(), sf = CURF;
  const thumb = snapshotDataURL(256);
  const div = document.createElement('div'); div.className = 'vp';
  const img = document.createElement('img'); img.src = thumb; div.appendChild(img);
  const meta = document.createElement('div'); meta.className = 'meta';
  meta.innerHTML =
    `${tag}  ·  az ${v.az.toFixed(1)}  el ${v.el.toFixed(1)}  r ${v.radius.toFixed(3)}`
    + `  ·  pull ${(offsetMeters()*100).toFixed(1)}cm [${dirName()}]`
    + `<div class="cli">${cli}</div>`;
  const row = document.createElement('div'); row.className = 'row';
  const go = document.createElement('button'); go.textContent = 'restore';
  go.onclick = ()=>{ loadFrame(sf, false); setView(v.az, v.el, v.radius); };
  const dl = document.createElement('button'); dl.textContent = 'png';
  dl.onclick = ()=>{ const a=document.createElement('a');
    a.download=`${DATA.case}_${tag}_az${v.az.toFixed(0)}_el${v.el.toFixed(0)}_r${v.radius.toFixed(2)}.png`;
    a.href=snapshotDataURL(parseInt(document.getElementById('res').value,10)); a.click(); };
  const rm = document.createElement('button'); rm.textContent = '✕';
  rm.onclick = ()=>div.remove();
  row.appendChild(go); row.appendChild(dl); row.appendChild(rm); meta.appendChild(row);
  div.appendChild(meta);
  document.getElementById('saved').prepend(div);
  console.log('[viewpoint]', tag, cli);
};

// frame scrubbing
const frameEl = document.getElementById('frame');
frameEl.max = NF - 1; frameEl.value = DATA.open;
frameEl.oninput = ()=> loadFrame(parseInt(frameEl.value, 10), true);
document.getElementById('prev').onclick = ()=> loadFrame(CURF - 1, true);
document.getElementById('next').onclick = ()=> loadFrame(CURF + 1, true);
let playTimer = null;
document.getElementById('play').onclick = ()=>{
  const btn = document.getElementById('play');
  if (playTimer){ clearInterval(playTimer); playTimer = null; btn.textContent = 'play'; return; }
  btn.textContent = 'pause';
  playTimer = setInterval(()=> loadFrame(CURF + 1, true), 100);
};

controls.addEventListener('change', readout);
window.addEventListener('resize', ()=>{camera.aspect=window.innerWidth/window.innerHeight;
  camera.updateProjectionMatrix(); renderer.setSize(window.innerWidth,window.innerHeight);});

applyBg();
loadFrame(DATA.open, false);
(function animate(){requestAnimationFrame(animate); renderer.render(scene,camera); })();
readout();
</script></body></html>"""


if __name__ == "__main__":
    main()
