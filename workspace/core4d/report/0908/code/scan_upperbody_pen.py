"""Scan for UPPER-BODY vs OBJECT penetration frames across the 0908 report cases.

The paper metrics only measure HAND-object penetration. This script finds frames
where the robot's *upper body* (torso / waist / shoulders / upper arms) clips into
the manipulated object -- the classic artifact we want example figures of.

For each (method, case) we replay the scene_act qpos through mj_forward, take the
world-space vertices of the upper-body visual mesh geoms, and compute their signed
distance into the (watertight) object mesh. Positive = inside = penetration depth.
We report, per frame, the deepest-penetrating upper-body body and its depth.

Output: a TSV ranking frames by penetration depth, written to tmp/.
"""
from __future__ import annotations

import os
os.environ.setdefault("MUJOCO_GL", "osmesa")

import json
import sys
from pathlib import Path

import numpy as np
import mujoco
import trimesh

REPO = Path(__file__).resolve().parents[5]
R0908 = Path(__file__).resolve().parents[1]
PROV = json.load(open(R0908 / "paper_results" / "provenance.json"))
OMNI_QPOS_DIR = R0908 / "paper_results" / "omni_scene_act_qpos"

# Upper-body bodies, split into "core" (torso/chest/waist) and "arm" (shoulder/
# upper-arm/elbow). Wrist + hand links are EXCLUDED -- those are the grasp contact,
# not an upper-body artifact.
CORE_KEYS = ("waist", "torso")
ARM_KEYS = ("shoulder", "elbow")
EXCLUDE_KEYS = ("wrist", "hand", "ankle", "knee", "hip")


def body_class(name: str) -> str | None:
    if any(k in name for k in EXCLUDE_KEYS):
        return None
    if any(k in name for k in CORE_KEYS):
        return "core"
    if any(k in name for k in ARM_KEYS):
        return "arm"
    return None


def load_qpos_spider(npz: Path) -> np.ndarray:
    q = np.load(npz)["qpos"]
    return q[:, 0, :] if q.ndim == 3 else q


def load_qpos_omni(case: str) -> Path:
    return OMNI_QPOS_DIR / f"{case}_omnirt_scene_act_qpos.npz"


def upperbody_geoms(model):
    """Return {gid: (body_name, cls)} for upper-body visual mesh geoms."""
    out = {}
    for gid in range(model.ngeom):
        if model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        bid = int(model.geom_bodyid[gid])
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
        cls = body_class(bname)
        if cls is None:
            continue
        out[gid] = (bname, cls)
    return out


def geom_local_mesh(model, gid):
    mid = model.geom_dataid[gid]
    va, vn = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
    fa, fn = model.mesh_faceadr[mid], model.mesh_facenum[mid]
    v = model.mesh_vert[va:va + vn].reshape(-1, 3)
    f = model.mesh_face[fa:fa + fn].reshape(-1, 3)
    return v, f


def object_geom(model):
    obj_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    for gid in range(model.ngeom):
        if int(model.geom_bodyid[gid]) == obj_body and model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_MESH:
            return gid
    raise RuntimeError("no object mesh geom")


def scan(scene: str, qpos: np.ndarray, vstride: int = 1):
    """Return list of (frame, depth_m, body, cls).

    The object (a box) is treated as an axis-aligned box in its own geom-local
    frame (verified: center ~origin, local-axis-aligned, hull/box vol ~0.94).
    For every upper-body vertex we transform into object-local coords and evaluate
    the analytic box signed distance -- penetration depth = max(0, -sdf). Fully
    vectorised, no mesh proximity queries (orders of magnitude faster, exact for a
    box)."""
    model = mujoco.MjModel.from_xml_path(scene)
    data = mujoco.MjData(model)
    ub = upperbody_geoms(model)
    if not ub:
        return []
    locals_ = {gid: geom_local_mesh(model, gid)[0][::vstride] for gid in ub}
    ogid = object_geom(model)
    ov_local, _ = geom_local_mesh(model, ogid)
    c = (ov_local.max(0) + ov_local.min(0)) / 2.0          # box center (local)
    h = (ov_local.max(0) - ov_local.min(0)) / 2.0          # box half-extents

    def box_pen(vw, wp, R):
        pl = (vw - wp) @ R - c                              # world -> object-local
        q = np.abs(pl) - h
        outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
        inside = np.minimum(q.max(1), 0.0)
        sdf = outside + inside                              # <0 inside
        return np.maximum(-sdf, 0.0)                        # penetration depth >=0

    rows = []
    for t, q in enumerate(qpos):
        data.qpos[:] = q
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        wp = data.geom_xpos[ogid]
        R = data.geom_xmat[ogid].reshape(3, 3)
        best = (0.0, "", "")
        for gid, (bname, cls) in ub.items():
            v = locals_[gid]
            vw = v @ data.geom_xmat[gid].reshape(3, 3).T + data.geom_xpos[gid]
            pen = box_pen(vw, wp, R)
            d = float(pen.max()) if len(pen) else 0.0
            if d > best[0]:
                best = (d, bname, cls)
        rows.append((t, best[0], best[1], best[2]))
    return rows


def main():
    methods = sys.argv[1] if len(sys.argv) > 1 else "both"  # omni|spider|both
    only_obj = sys.argv[2] if len(sys.argv) > 2 else "box"  # object-key prefix filter
    recs = PROV["records"]
    results = []
    # index spider cem_npz per case
    by = {}
    for r in recs:
        by.setdefault(r["case_id"], {})[r["method"]] = r

    cases = [c for c in by if c.startswith(only_obj)]
    print(f"scanning {len(cases)} cases, methods={methods}, filter={only_obj}", flush=True)
    for ci, case in enumerate(sorted(cases)):
        entry = by[case]
        todo = []
        if methods in ("omni", "both") and "OmniRetarget" in entry:
            oq = load_qpos_omni(case)
            if oq.exists():
                todo.append(("OmniRetarget", entry["OmniRetarget"]["scene_act"], oq))
        if methods in ("spider", "both") and "SPIDER-CEM" in entry:
            todo.append(("SPIDER-CEM", entry["SPIDER-CEM"]["scene_act"], Path(entry["SPIDER-CEM"]["cem_npz"])))
        for method, scene_rel, qp in todo:
            scene = scene_rel if os.path.isabs(scene_rel) else str(REPO / scene_rel)
            if not os.path.exists(scene):
                print(f"  [skip] {case} {method}: scene missing", flush=True)
                continue
            if method == "OmniRetarget":
                qpos = np.load(qp)["qpos"]
                qpos = qpos[:, 0, :] if qpos.ndim == 3 else qpos
            else:
                qpos = load_qpos_spider(qp)
            try:
                rows = scan(scene, qpos)
            except Exception as e:
                print(f"  [err] {case} {method}: {e}", flush=True)
                continue
            if not rows:
                continue
            fr, depth, body, cls = max(rows, key=lambda r: r[1])
            n_pen = sum(1 for r in rows if r[1] > 0.005)  # frames with >5mm
            results.append((depth, case, method, fr, body, cls, n_pen, len(rows)))
            print(f"  [{ci+1}/{len(cases)}] {case:28s} {method:12s} "
                  f"max={depth*1000:6.1f}mm @f{fr:3d} {body:22s} ({cls}) "
                  f"npen>5mm={n_pen}", flush=True)

    results.sort(reverse=True)
    out = R0908 / "tmp" / f"upperbody_pen_{methods}_{only_obj}.tsv"
    out.parent.mkdir(exist_ok=True)
    with out.open("w") as fh:
        fh.write("depth_mm\tcase\tmethod\tframe\tbody\tclass\tn_frames_gt5mm\tn_frames\n")
        for depth, case, method, fr, body, cls, n_pen, n in results:
            fh.write(f"{depth*1000:.2f}\t{case}\t{method}\t{fr}\t{body}\t{cls}\t{n_pen}\t{n}\n")
    print(f"\n[ok] wrote {out}  ({len(results)} method-cases)", flush=True)
    print("\nTOP 20:")
    for row in results[:20]:
        depth, case, method, fr, body, cls, n_pen, n = row
        print(f"  {depth*1000:6.1f}mm  {case:28s} {method:12s} f{fr:3d} {body:20s} {cls} npen={n_pen}")


if __name__ == "__main__":
    main()
