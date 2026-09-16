"""Export one frame (all 4 methods) to a self-contained interactive Three.js HTML.

Orbit freely around the right-hand / object contact, switch method, toggle the
left arm, and read the camera angle live in OUR convention -- azimuth offset from
the robot facing yaw, elevation, and hand-radius (framing) -- plus the exact
viz_paper_figure.py CLI to reproduce it. A "save viewpoint" button logs the values.

Run:
    PYOPENGL_PLATFORM=osmesa MUJOCO_GL=osmesa .venv/bin/python \
        workspace/core4d/report/0908/code/viz_hand_html.py
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
from viz_methods_compare import DEFAULT_SCENE, apply_T, facing_yaw, load_qpos, method_qpos_paths  # noqa: E402
from viz_paper_figure import build_alignment  # noqa: E402
from viz_mixed_robot_smplx import _geom_rgb, object_world_mesh  # noqa: E402

METHODS = ["gmr", "omniretarget", "sbto", "spider"]
LABEL = {"gmr": "GMR", "omniretarget": "OmniRetarget", "sbto": "DynaRetarget", "spider": "Ours"}
LEFT_ARM = ["left_shoulder_pitch_link", "left_shoulder_roll_link", "left_shoulder_yaw_link",
            "left_elbow_link", "left_wrist_roll_link", "left_wrist_pitch_link",
            "left_wrist_yaw_link"]
BOX_RGB = [0.24, 0.36, 0.54]


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


def instances(model, data, T, obj_bid, left_bids):
    """Per-geom instance refs: mesh dataid + world transform (T applied) + colour + flags."""
    TR, Tp = T[:3, :3], T[:3, 3]
    out = []
    for gid in range(model.ngeom):
        if model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        bid = int(model.geom_bodyid[gid])
        is_obj = bid == obj_bid
        R = TR @ data.geom_xmat[gid].reshape(3, 3)
        p = TR @ data.geom_xpos[gid] + Tp
        rgb = BOX_RGB if is_obj else [round(float(c), 3) for c in _geom_rgb(model, gid)]
        out.append({"m": int(model.geom_dataid[gid]),
                    "R": R.round(5).ravel().tolist(), "p": p.round(4).tolist(),
                    "c": rgb, "obj": is_obj, "left": bid in left_bids})
    return out


def main():
    ap = argparse.ArgumentParser(description="Interactive frame viewer (Three.js HTML).")
    ap.add_argument("--case", default="box021_20231011_037")
    ap.add_argument("--scene", default=str(DEFAULT_SCENE))
    ap.add_argument("--file-frame", type=int, default=12)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    scene = args.scene
    qpos, ref, regs, (w0, w1) = build_alignment(scene, args.case)
    ref_i = w0 + args.file_frame
    facing = facing_yaw(ref)

    _m0 = mujoco.MjModel.from_xml_path(scene)
    rh_bid = mujoco.mj_name2id(_m0, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_yaw_link")
    obj_bid = mujoco.mj_name2id(_m0, mujoco.mjtObj.mjOBJ_BODY, "object")
    left_bids = {mujoco.mj_name2id(_m0, mujoco.mjtObj.mjOBJ_BODY, n) for n in LEFT_ARM}

    data_js = {"facing": facing, "methods": {}, "order": METHODS, "label": LABEL,
               "meshlib": build_meshlib(_m0)}
    floor_z = np.inf
    for m in METHODS:
        lag, T = regs[m]
        model = mujoco.MjModel.from_xml_path(scene)
        data = mujoco.MjData(model)
        data.qpos[:] = qpos[m][ref_i - lag]
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        inst = instances(model, data, T, obj_bid, left_bids)
        ov, _, _ = object_world_mesh(model, data)
        ov = apply_T(ov, T)
        rhand = apply_T(data.xpos[rh_bid][None], T)[0]
        nearest = ov[np.argmin(np.linalg.norm(ov - rhand, axis=1))]
        contact = (0.5 * (rhand + nearest)).round(4).tolist()
        floor_z = min(floor_z, float(ov[:, 2].min()))
        data_js["methods"][m] = {"inst": inst, "rhand": rhand.round(4).tolist(),
                                 "contact": contact}
    data_js["floor_z"] = float(floor_z)

    out = Path(args.out) if args.out else \
        REPO / f"workspace/core4d/report/0908/paper_results/viz/{args.case}_p2/paper_figure/frame_viewer.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(HTML.replace("__DATA__", json.dumps(data_js)))
    print(f"[ok] {out}")
    print("open in a browser; orbit, pick method / toggle left arm, read the angle, "
          "click 'Save viewpoint' and send me the printed --hand-az-offset/--hand-el/--hand-radius.")


HTML = r"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>frame viewer</title>
<style>
  body{margin:0;overflow:hidden;font-family:monospace;background:#e9eb ee}
  #ui{position:fixed;top:8px;left:8px;background:rgba(255,255,255,.92);padding:10px 12px;
      border-radius:8px;font-size:13px;line-height:1.5;box-shadow:0 1px 6px rgba(0,0,0,.2)}
  #ui b{font-size:14px}
  #readout{white-space:pre;margin-top:6px}
  #cli{color:#c0392b;user-select:all}
  button,select{font-family:monospace;font-size:13px;margin-top:4px}
  #saved{margin-top:8px;max-height:160px;overflow:auto;color:#2c3e50}
</style></head><body>
<div id="ui">
  <b>Frame viewer</b><br>
  method: <select id="method"></select>
  <label><input type="checkbox" id="hideleft"> hide left arm</label><br>
  <div id="readout"></div>
  CLI: <span id="cli"></span><br>
  <button id="save">Save viewpoint</button>
  <div id="saved"></div>
</div>
<script src="https://unpkg.com/three@0.128.0/build/three.min.js"></script>
<script src="https://unpkg.com/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<script>
const DATA = __DATA__;
const FOV = 45;                       // vertical fov (deg), matches the render
const scene = new THREE.Scene();
scene.background = new THREE.Color(0xe9ebee);
const renderer = new THREE.WebGLRenderer({antialias:true});
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(window.devicePixelRatio);
document.body.appendChild(renderer.domElement);
const camera = new THREE.PerspectiveCamera(FOV, window.innerWidth/window.innerHeight, 0.01, 100);
camera.up.set(0,0,1);                 // Z-up world
const controls = new THREE.OrbitControls(camera, renderer.domElement);
scene.add(new THREE.AmbientLight(0xffffff, 0.75));
const d1 = new THREE.DirectionalLight(0xffffff, 0.7); d1.position.set(2,-2,4); scene.add(d1);
const d2 = new THREE.DirectionalLight(0xffffff, 0.4); d2.position.set(-2,2,3); scene.add(d2);
// ground
const gg = new THREE.PlaneGeometry(20,20);
const gm = new THREE.MeshStandardMaterial({color:0x9a9186, roughness:1});
const ground = new THREE.Mesh(gg, gm); ground.position.z = DATA.floor_z; scene.add(ground);

// shared geometry library (built once from local mesh verts/faces)
const GEO = {};
for (const did in DATA.meshlib){
  const s = DATA.meshlib[did];
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(s.v,3));
  g.setIndex(s.f); g.computeVertexNormals();
  GEO[did] = g;
}
let group = new THREE.Group(); scene.add(group);
let leftGroup = new THREE.Group(); scene.add(leftGroup);
let target = new THREE.Vector3();

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
function load(method){
  scene.remove(group); scene.remove(leftGroup);
  group = new THREE.Group(); leftGroup = new THREE.Group();
  const md = DATA.methods[method];
  md.inst.forEach(inst=>{ (inst.left ? leftGroup : group).add(instMesh(inst)); });
  scene.add(group); scene.add(leftGroup);
  leftGroup.visible = !document.getElementById('hideleft').checked;
  target.fromArray(md.contact);
  controls.target.copy(target);
  setView(-45, 22, 0.5);          // open at az-45/el22, radius 0.5 (our convention)
}
function setView(azOff, el, radius){
  const az = (DATA.facing + azOff)*Math.PI/180, e = el*Math.PI/180;
  const dir = new THREE.Vector3(Math.cos(e)*Math.cos(az), Math.cos(e)*Math.sin(az), Math.sin(e));
  const dist = radius/Math.tan(FOV/2*Math.PI/180);
  camera.position.copy(target).add(dir.multiplyScalar(dist));
  controls.update();
}
function readout(){
  const dir = camera.position.clone().sub(target);
  const dist = dir.length(); dir.normalize();
  let azw = Math.atan2(dir.y, dir.x)*180/Math.PI;
  let azOff = azw - DATA.facing;
  azOff = ((azOff+180)%360+360)%360 - 180;      // wrap to [-180,180]
  const el = Math.asin(dir.z)*180/Math.PI;
  const radius = dist*Math.tan(FOV/2*Math.PI/180);
  document.getElementById('readout').textContent =
     `az_offset = ${azOff.toFixed(1)} deg\nel        = ${el.toFixed(1)} deg\nradius    = ${radius.toFixed(3)} m`;
  document.getElementById('cli').textContent =
     `--hand-az-offset ${azOff.toFixed(1)} --hand-el ${el.toFixed(1)} --hand-radius ${radius.toFixed(3)}`;
}
const sel = document.getElementById('method');
DATA.order.forEach(m=>{const o=document.createElement('option');o.value=m;o.textContent=DATA.label[m];sel.appendChild(o);});
sel.value = 'spider';
sel.onchange = ()=>load(sel.value);
document.getElementById('hideleft').onchange = ()=>{ leftGroup.visible = !document.getElementById('hideleft').checked; };
document.getElementById('save').onclick = ()=>{
  const t = document.getElementById('cli').textContent;
  const div = document.createElement('div'); div.textContent = '• '+sel.value+':  '+t;
  document.getElementById('saved').prepend(div);
  console.log('[viewpoint]', sel.value, t);
};
controls.addEventListener('change', readout);
window.addEventListener('resize', ()=>{camera.aspect=window.innerWidth/window.innerHeight;
  camera.updateProjectionMatrix(); renderer.setSize(window.innerWidth,window.innerHeight);});
load('spider');
(function animate(){requestAnimationFrame(animate); renderer.render(scene,camera); })();
readout();
</script></body></html>"""


if __name__ == "__main__":
    main()
