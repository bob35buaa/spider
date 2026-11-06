"""Interactive, viewpoint-adjustable HTML demo of a methods_compare clip (full scene).

Browser twin of viz_methods_compare.py: exports the exact same geometry (G1 robot +
manipulated object, object-aligned world transform) of one method, for EVERY frame of
the common window, into a self-contained Three.js page. Mesh topology is shared across
frames (stored once); only the 38 per-geom transforms are stored per frame, so a whole
123-frame clip adds little over a single frame. You can scrub to any frame
(frame_{k:04d}.png), orbit freely, read the camera angle live in the methods_compare
convention (az-offset from the reference facing yaw, elevation), toggle background /
floor / left-arm, download a square PNG at any chosen viewpoint, and copy the
viz_methods_compare.py CLI to reproduce the same angle offline at high resolution.

Frame indexing matches viz_methods_compare.py exactly: frame k maps to the rendered
frame_{k:04d}.png (ref index = w0 + k over the common window). --file-frame only sets
which frame the viewer OPENS on; all frames are selectable in-browser.

Run (headless):
    PYOPENGL_PLATFORM=osmesa MUJOCO_GL=osmesa .venv/bin/python \
        workspace/core4d/report/0908/code/viz_methods_compare_html.py \
        --case box021_20231011_037 --method spider --file-frame 17
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
from viz_methods_compare import DEFAULT_SCENE, PERSON, facing_yaw  # noqa: E402
from viz_paper_figure import build_alignment  # noqa: E402

LEFT_ARM = ["left_shoulder_pitch_link", "left_shoulder_roll_link", "left_shoulder_yaw_link",
            "left_elbow_link", "left_wrist_roll_link", "left_wrist_pitch_link",
            "left_wrist_yaw_link"]
BOX_RGB = [0.24, 0.36, 0.54]                 # matches render_style.BOX_COLOR

# viz_methods_compare.py camera defaults (yfov = pi/4, 1.10 framing pad).
DEFAULT_AZ_OFFSET = 45.0
DEFAULT_EL = 12.0
FOV = 45.0
FRAME_PAD = 1.10


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


def mesh_gids(model):
    """Geom ids of mesh geoms, in a fixed order -> defines the per-frame slot order."""
    return [gid for gid in range(model.ngeom)
            if model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_MESH]


def build_slots(model, gids, obj_bid, left_bids):
    """Static per-slot refs (constant across frames): mesh dataid, colour, flags."""
    slots = []
    for gid in gids:
        bid = int(model.geom_bodyid[gid])
        is_obj = bid == obj_bid
        from viz_mixed_robot_smplx import _geom_rgb  # local import; only needed here
        rgb = BOX_RGB if is_obj else [round(float(c), 3) for c in _geom_rgb(model, gid)]
        slots.append({"m": int(model.geom_dataid[gid]), "c": rgb,
                      "obj": is_obj, "left": bid in left_bids})
    return slots


def frame_data(model, data, gids, slots, TR, Tp, local_verts):
    """One frame: per-slot world transforms + framing target (robot/object midpoint)."""
    xform, robot_v, obj_v = [], [], []
    for gid, slot, lv in zip(gids, slots, local_verts):
        R = TR @ data.geom_xmat[gid].reshape(3, 3)
        p = TR @ data.geom_xpos[gid] + Tp
        xform.append({"R": R.round(5).ravel().tolist(), "p": p.round(4).tolist()})
        (obj_v if slot["obj"] else robot_v).append(lv @ R.T + p)
    robot_all = np.concatenate(robot_v, 0)
    ov = np.concatenate(obj_v, 0)
    target = (robot_all.mean(0) + ov.mean(0)) / 2.0
    allv = np.concatenate([robot_all, ov], 0)
    return xform, target, allv


def build_data_js(model, data, frame_qpos, *, facing, default_az, default_el, pad, T,
                  case, method, start_frame, cli_tool, cli_fixed, cli_az, cli_el):
    """Assemble the self-contained viewer payload for a clip of frames.

    frame_qpos: iterable of per-frame qpos vectors (already lag-shifted if needed).
    T: a single 4x4 world transform applied to every frame (identity for most cases;
    the per-method object-align transform for viz_methods_compare). Mesh topology is
    emitted once; only the per-geom transforms + framing target are stored per frame."""
    TR, Tp = T[:3, :3], T[:3, 3]
    obj_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    left_bids = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in LEFT_ARM}
    gids = mesh_gids(model)
    slots = build_slots(model, gids, obj_bid, left_bids)
    local_verts = [model.mesh_vert[
        model.mesh_vertadr[model.geom_dataid[gid]]:
        model.mesh_vertadr[model.geom_dataid[gid]] + model.mesh_vertnum[model.geom_dataid[gid]]
    ].reshape(-1, 3) for gid in gids]

    # A single COMMON camera distance (max framing radius across frames) keeps scrubbing
    # stable; the per-frame target follows the robot/object midpoint -- the same framing
    # contract the offline renderers use for a single clip.
    frames, radius, floor_z = [], 0.0, np.inf
    for q in frame_qpos:
        data.qpos[:] = q
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        xform, target, allv = frame_data(model, data, gids, slots, TR, Tp, local_verts)
        frames.append({"x": xform, "t": target.round(4).tolist()})
        radius = max(radius, float(np.linalg.norm(allv - target, axis=1).max()))
        floor_z = min(floor_z, float(allv[:, 2].min()))
    dist = radius / np.tan(np.radians(FOV) / 2.0) * pad

    return {
        "case": case, "method": method, "start_frame": start_frame,
        "n_frames": len(frames), "facing": facing, "floor_z": round(float(floor_z), 4),
        "slots": slots, "frames": frames, "meshlib": build_meshlib(model),
        "default": {"az": default_az, "el": default_el, "dist": round(dist, 4)},
        "cli": {"tool": cli_tool, "fixed": cli_fixed, "az": cli_az, "el": cli_el},
    }


def main():
    ap = argparse.ArgumentParser(description="Interactive methods_compare clip (Three.js HTML).")
    ap.add_argument("--case", default="box021_20231011_037")
    ap.add_argument("--method", default="spider",
                    choices=["spider", "omniretarget", "sbto", "gmr"])
    ap.add_argument("--scene", default=str(DEFAULT_SCENE))
    ap.add_argument("--file-frame", type=int, default=17,
                    help="frame the viewer OPENS on (all frames are selectable in-browser)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    scene = args.scene
    qpos, ref, regs, (w0, w1) = build_alignment(scene, args.case)
    n_frames = w1 - w0
    assert 0 <= args.file_frame < n_frames, \
        f"file-frame {args.file_frame} outside [0,{n_frames})"

    model = mujoco.MjModel.from_xml_path(scene)
    data = mujoco.MjData(model)
    lag, T = regs[args.method]
    frame_qpos = [qpos[args.method][ref_i - lag] for ref_i in range(w0, w1)]

    data_js = build_data_js(
        model, data, frame_qpos, facing=facing_yaw(ref),
        default_az=DEFAULT_AZ_OFFSET, default_el=DEFAULT_EL, pad=FRAME_PAD, T=T,
        case=args.case, method=args.method, start_frame=args.file_frame,
        cli_tool="workspace/core4d/report/0908/code/viz_methods_compare.py",
        cli_fixed=f"--case {args.case}", cli_az="--az-offset", cli_el="--el")

    out = Path(args.out) if args.out else \
        REPO / (f"workspace/core4d/report/0908/paper_results/viz/{args.case}_{PERSON}"
                f"/methods_compare/{args.method}/frame_viewer.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(HTML.replace("__DATA__", json.dumps(data_js)))
    print(f"[ok] {out}  ({n_frames} frames, opens on f{args.file_frame})")
    print("open in a browser; scrub frames, orbit, toggle floor/bg, download a PNG at "
          "any viewpoint, and copy the viz_methods_compare.py CLI for a high-res render.")


HTML = r"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>methods_compare frame viewer</title>
<style>
  :root{--panel:rgba(255,255,255,.94)}
  html,body{margin:0;height:100%;overflow:hidden;font-family:monospace}
  #stage{position:fixed;inset:0;z-index:0;
    background-image:linear-gradient(45deg,#ccc 25%,transparent 25%),linear-gradient(-45deg,#ccc 25%,transparent 25%),linear-gradient(45deg,transparent 75%,#ccc 75%),linear-gradient(-45deg,transparent 75%,#ccc 75%);
    background-size:24px 24px;background-position:0 0,0 12px,12px -12px,-12px 0px;background-color:#e9ebee}
  #stage.white{background:#fff}
  canvas{position:fixed;top:0;left:0;z-index:1}
  #ui{position:fixed;top:10px;left:10px;z-index:3;background:var(--panel);padding:12px 14px;
      border-radius:10px;font-size:13px;line-height:1.55;box-shadow:0 2px 10px rgba(0,0,0,.25);
      max-width:360px}
  #ui b{font-size:14px}
  #readout{white-space:pre;margin:6px 0;color:#2c3e50}
  #cli{color:#c0392b;user-select:all;word-break:break-all;display:block;margin:2px 0 6px}
  button,select{font-family:monospace;font-size:13px;margin:2px 4px 2px 0;padding:3px 7px;
      border:1px solid #bbb;border-radius:6px;background:#f6f6f6;cursor:pointer}
  button:hover{background:#eceff1}
  label{cursor:pointer;margin-right:8px}
  #saved{margin-top:10px;max-height:40vh;overflow:auto;border-top:1px solid #ddd;padding-top:8px}
  .vp{display:flex;gap:8px;align-items:flex-start;margin-bottom:8px;padding-bottom:8px;border-bottom:1px dashed #e0e0e0}
  .vp img{width:72px;height:72px;border:1px solid #ccc;border-radius:4px;flex:0 0 auto;background:#e9ebee}
  .vp .meta{font-size:11px;color:#2c3e50;flex:1 1 auto}
  .vp .meta .cli{color:#c0392b;user-select:all;word-break:break-all}
  .vp .row{margin-top:3px}
  .hint{color:#888;font-size:11px;margin-top:4px}
</style></head><body>
<div id="stage"></div>
<div id="ui">
  <b>methods_compare viewer</b> <span id="title" style="font-weight:normal;color:#888"></span><br>
  <div class="row">
    bg: <select id="bg">
      <option value="transparent">transparent</option>
      <option value="white" selected>white</option>
      <option value="grey">grey</option>
    </select>
    <label><input type="checkbox" id="floor"> floor</label>
    <label><input type="checkbox" id="hideleft"> hide left arm</label>
  </div>
  <div class="row" style="display:flex;align-items:center;gap:6px;margin-top:4px">
    <button id="prev">◀</button>
    <input type="range" id="frame" min="0" step="1" style="flex:1 1 auto">
    <button id="next">▶</button>
    <input type="number" id="framenum" min="0" step="1" style="width:56px">
  </div>
  <div id="readout"></div>
  <span id="cli"></span>
  <div class="row">
    <button id="reset">Reset view</button>
    <button id="save">★ Save viewpoint</button>
    <button id="png">⬇ Download PNG</button>
    res <select id="res">
      <option>720</option><option selected>1080</option><option>1620</option>
    </select>
  </div>
  <div class="hint">drag = orbit · scroll = zoom · right-drag = pan</div>
  <div id="saved"></div>
</div>
<script src="https://unpkg.com/three@0.128.0/build/three.min.js"></script>
<script src="https://unpkg.com/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<script>
const DATA = __DATA__;
const FOV = 45;                       // vertical fov (deg) -- matches yfov=pi/4 in the render
let curFrame = DATA.start_frame;

const stage = document.getElementById('stage');
function setTitle(){ document.getElementById('title').textContent =
    `${DATA.case}  ${DATA.method}  f${curFrame}/${DATA.n_frames-1}`; }

const scene = new THREE.Scene();
const renderer = new THREE.WebGLRenderer({antialias:true, alpha:true, preserveDrawingBuffer:true});
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setClearColor(0x000000, 0);
document.body.appendChild(renderer.domElement);

const camera = new THREE.PerspectiveCamera(FOV, window.innerWidth/window.innerHeight, 0.01, 200);
camera.up.set(0,0,1);                 // Z-up world
const controls = new THREE.OrbitControls(camera, renderer.domElement);

// soft phong-ish lighting: ambient + a few directionals
scene.add(new THREE.AmbientLight(0xffffff, 0.75));
const dirs = [[2,-2,4,0.55],[-2,2,3,0.3],[0,0,6,0.25]];
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

// Build one mesh per static slot once; per-frame we only update each slot's matrix.
let mainGroup = new THREE.Group(), leftGroup = new THREE.Group();
scene.add(mainGroup); scene.add(leftGroup);
const slotMesh = DATA.slots.map(slot=>{
  const mat = new THREE.MeshStandardMaterial({color:new THREE.Color(slot.c[0],slot.c[1],slot.c[2]),
      roughness:0.55, metalness:0.0, side:THREE.DoubleSide});
  const mesh = new THREE.Mesh(GEO[slot.m], mat);
  mesh.matrixAutoUpdate = false;
  (slot.left ? leftGroup : mainGroup).add(mesh);
  return mesh;
});
function applyFrameGeom(k){
  const fr = DATA.frames[k];
  for (let i=0;i<slotMesh.length;i++){
    const R = fr.x[i].R, p = fr.x[i].p;
    slotMesh[i].matrix.set(R[0],R[1],R[2],p[0], R[3],R[4],R[5],p[1], R[6],R[7],R[8],p[2], 0,0,0,1);
  }
  target.fromArray(fr.t);
  floorMesh.position.set(target.x, target.y, DATA.floor_z);
  controls.target.copy(target);
}

// checker floor plane at floor_z (toggleable), matching the pyrender bg render
function makeCheckerTexture(){
  const n=512, c=document.createElement('canvas'); c.width=c.height=n;
  const g=c.getContext('2d'); const s=n/8;
  for(let i=0;i<8;i++)for(let j=0;j<8;j++){ g.fillStyle=((i+j)&1)?'#c7ccd2':'#e6e9ec';
    g.fillRect(i*s,j*s,s,s); }
  const t=new THREE.CanvasTexture(c); t.wrapS=t.wrapT=THREE.RepeatWrapping; t.repeat.set(14,14);
  return t;
}
const floorMesh = new THREE.Mesh(
  new THREE.PlaneGeometry(12,12),
  new THREE.MeshStandardMaterial({map:makeCheckerTexture(), roughness:0.95, metalness:0.0}));
floorMesh.visible = false;
scene.add(floorMesh);

// Switch frame, preserving the user's current orbit (az/el/dist) around the new target.
function setFrame(k){
  k = Math.max(0, Math.min(DATA.n_frames-1, k|0));
  const v = currentView();               // keep orientation across the frame change
  curFrame = k;
  applyFrameGeom(k);
  setView(v.az, v.el, v.dist);
  document.getElementById('frame').value = k;
  document.getElementById('framenum').value = k;
  setTitle(); readout();
}

function applyBg(){
  const v = document.getElementById('bg').value;
  stage.className = (v === 'white') ? 'white' : '';
  if (v === 'white'){ renderer.setClearColor(0xffffff, 1); }
  else if (v === 'grey'){ renderer.setClearColor(0xe6e8ea, 1); }
  else { renderer.setClearColor(0x000000, 0); }
}
function applyFloor(){ floorMesh.visible = document.getElementById('floor').checked; }
function applyLeft(){ leftGroup.visible = !document.getElementById('hideleft').checked; }

function setView(azOff, el, dist){
  const az = (DATA.facing + azOff)*Math.PI/180, e = el*Math.PI/180;
  const dir = new THREE.Vector3(Math.cos(e)*Math.cos(az), Math.cos(e)*Math.sin(az), Math.sin(e));
  camera.position.copy(target).add(dir.multiplyScalar(dist));
  controls.update();
}
function currentView(){
  const dir = camera.position.clone().sub(target);
  const dist = dir.length() || DATA.default.dist; dir.normalize();
  let azOff = Math.atan2(dir.y, dir.x)*180/Math.PI - DATA.facing;
  azOff = ((azOff+180)%360+360)%360 - 180;              // wrap to [-180,180]
  const el = Math.asin(Math.max(-1,Math.min(1,dir.z)))*180/Math.PI;
  return {az:azOff, el:el, dist:dist};
}
function cliFor(v){
  return `uv run ${DATA.cli.tool} ${DATA.cli.fixed} `
       + `${DATA.cli.az} ${v.az.toFixed(1)} ${DATA.cli.el} ${v.el.toFixed(1)}`;
}
function readout(){
  const v = currentView();
  document.getElementById('readout').textContent =
     `frame     = ${curFrame} / ${DATA.n_frames-1}\n`
   + `az_offset = ${v.az.toFixed(1)} deg\nel        = ${v.el.toFixed(1)} deg\ndist      = ${v.dist.toFixed(3)} m`;
  document.getElementById('cli').textContent = cliFor(v);
}

// render the current view into a square canvas (matches the 1:1 render aspect)
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
function pngName(v){
  return `${DATA.case}_${DATA.method}_f${curFrame}_az${v.az.toFixed(0)}_el${v.el.toFixed(0)}.png`;
}

document.getElementById('bg').onchange = applyBg;
document.getElementById('floor').onchange = applyFloor;
document.getElementById('hideleft').onchange = applyLeft;
document.getElementById('frame').oninput = (e)=>setFrame(parseInt(e.target.value,10));
document.getElementById('framenum').onchange = (e)=>setFrame(parseInt(e.target.value,10));
document.getElementById('prev').onclick = ()=>setFrame(curFrame-1);
document.getElementById('next').onclick = ()=>setFrame(curFrame+1);
document.getElementById('reset').onclick = ()=>setView(DATA.default.az, DATA.default.el, DATA.default.dist);
document.getElementById('png').onclick = ()=>{
  const size = parseInt(document.getElementById('res').value, 10);
  const a = document.createElement('a');
  a.download = pngName(currentView());
  a.href = snapshotDataURL(size); a.click();
};
document.getElementById('save').onclick = ()=>{
  const v = currentView(), cli = cliFor(v), frame = curFrame;
  const thumb = snapshotDataURL(256);
  const div = document.createElement('div'); div.className = 'vp';
  const img = document.createElement('img'); img.src = thumb; div.appendChild(img);
  const meta = document.createElement('div'); meta.className = 'meta';
  meta.innerHTML =
    `f${frame}  az ${v.az.toFixed(1)}  el ${v.el.toFixed(1)}  dist ${v.dist.toFixed(3)}`
    + `<div class="cli">${cli}</div>`;
  const row = document.createElement('div'); row.className = 'row';
  const go = document.createElement('button'); go.textContent = 'restore';
  go.onclick = ()=>{ setFrame(frame); setView(v.az, v.el, v.dist); };
  const dl = document.createElement('button'); dl.textContent = 'png';
  dl.onclick = ()=>{ const a=document.createElement('a');
    a.download=`${DATA.case}_${DATA.method}_f${frame}_az${v.az.toFixed(0)}_el${v.el.toFixed(0)}.png`;
    a.href=snapshotDataURL(parseInt(document.getElementById('res').value,10)); a.click(); };
  const rm = document.createElement('button'); rm.textContent = '✕';
  rm.onclick = ()=>div.remove();
  row.appendChild(go); row.appendChild(dl); row.appendChild(rm); meta.appendChild(row);
  div.appendChild(meta);
  document.getElementById('saved').prepend(div);
  console.log('[viewpoint]', `f${frame}`, cli);
};

controls.addEventListener('change', readout);
window.addEventListener('resize', ()=>{camera.aspect=window.innerWidth/window.innerHeight;
  camera.updateProjectionMatrix(); renderer.setSize(window.innerWidth,window.innerHeight);});

document.getElementById('frame').max = DATA.n_frames-1;
document.getElementById('framenum').max = DATA.n_frames-1;
applyBg(); applyFloor(); applyLeft();
applyFrameGeom(curFrame);
setView(DATA.default.az, DATA.default.el, DATA.default.dist);
document.getElementById('frame').value = curFrame;
document.getElementById('framenum').value = curFrame;
setTitle();
(function animate(){requestAnimationFrame(animate); renderer.render(scene,camera); })();
readout();
</script></body></html>"""


if __name__ == "__main__":
    main()
