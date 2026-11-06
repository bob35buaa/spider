"""Scan for TORSO (trunk) vs OBJECT penetration -- across ALL objects, any shape.

Distinct from scan_upperbody_pen.py (which covered arms on boxes): here we look
ONLY at the trunk (pelvis / waist / torso_link -- NO arms, NO legs, NO hands) and
handle arbitrary object meshes (bucket, chair, desk, box).

Speed: the object mesh is constant in its own geom-local frame, so we build ONE
trimesh ProximityQuery per case on that local mesh. Each frame we transform the
trunk vertices into object-local coords and query signed distance (positive =
inside = penetration). AABB prefilter keeps the query to the few candidate verts.
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

# Trunk only. torso_link = chest; waist_* = lower trunk; pelvis = trunk base.
TRUNK_KEYS = ("pelvis", "waist", "torso")


def is_trunk(name: str) -> bool:
    return any(k in name for k in TRUNK_KEYS)


def geom_local_mesh(model, gid):
    mid = model.geom_dataid[gid]
    va, vn = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
    fa, fn = model.mesh_faceadr[mid], model.mesh_facenum[mid]
    v = model.mesh_vert[va:va + vn].reshape(-1, 3)
    f = model.mesh_face[fa:fa + fn].reshape(-1, 3)
    return v, f


def trunk_geoms(model):
    out = {}
    for gid in range(model.ngeom):
        if model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        bid = int(model.geom_bodyid[gid])
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
        if is_trunk(bname):
            out[gid] = bname
    return out


def object_geom(model):
    ob = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    for gid in range(model.ngeom):
        if int(model.geom_bodyid[gid]) == ob and model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_MESH:
            return gid
    raise RuntimeError("no object mesh geom")


def scan(scene: str, qpos: np.ndarray, vstride: int = 2):
    model = mujoco.MjModel.from_xml_path(scene)
    data = mujoco.MjData(model)
    tg = trunk_geoms(model)
    if not tg:
        return []
    locals_ = {gid: geom_local_mesh(model, gid)[0][::vstride] for gid in tg}
    ogid = object_geom(model)
    ov, of = geom_local_mesh(model, ogid)
    omesh = trimesh.Trimesh(ov, of, process=False)
    pq = trimesh.proximity.ProximityQuery(omesh)
    lo, hi = ov.min(0), ov.max(0)

    rows = []
    for t, q in enumerate(qpos):
        data.qpos[:] = q
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        wp = data.geom_xpos[ogid]
        R = data.geom_xmat[ogid].reshape(3, 3)       # local->world
        best = (0.0, "")
        for gid, bname in tg.items():
            vw = locals_[gid] @ data.geom_xmat[gid].reshape(3, 3).T + data.geom_xpos[gid]
            pl = (vw - wp) @ R                        # world -> object-local
            cand = ((pl >= lo) & (pl <= hi)).all(1)
            if not cand.any():
                continue
            sd = pq.signed_distance(pl[cand])         # + inside
            d = float(sd.max()) if len(sd) else 0.0
            if d > best[0]:
                best = (d, bname)
        rows.append((t, best[0], best[1]))
    return rows


def load_omni(case):
    return OMNI_QPOS_DIR / f"{case}_omnirt_scene_act_qpos.npz"


def main():
    methods = sys.argv[1] if len(sys.argv) > 1 else "both"
    only = sys.argv[2] if len(sys.argv) > 2 else ""   # object-key prefix filter ("" = all)
    recs = PROV["records"]
    by = {}
    for r in recs:
        by.setdefault(r["case_id"], {})[r["method"]] = r
    cases = sorted(c for c in by if c.startswith(only))
    print(f"scanning {len(cases)} cases methods={methods} filter={only!r}", flush=True)

    results = []
    for ci, case in enumerate(cases):
        entry = by[case]
        todo = []
        if methods in ("omni", "both") and "OmniRetarget" in entry:
            oq = load_omni(case)
            if oq.exists():
                todo.append(("OmniRetarget", entry["OmniRetarget"]["scene_act"], oq, False))
        if methods in ("spider", "both") and "SPIDER-CEM" in entry:
            todo.append(("SPIDER-CEM", entry["SPIDER-CEM"]["scene_act"],
                         Path(entry["SPIDER-CEM"]["cem_npz"]), True))
        for method, scene_rel, qp, is_spider in todo:
            scene = scene_rel if os.path.isabs(scene_rel) else str(REPO / scene_rel)
            if not os.path.exists(scene) or not Path(qp).exists():
                continue
            q = np.load(qp)["qpos"]
            q = q[:, 0, :] if q.ndim == 3 else q
            try:
                rows = scan(scene, q)
            except Exception as e:
                print(f"  [err] {case} {method}: {e}", flush=True)
                continue
            if not rows:
                continue
            fr, depth, body = max(rows, key=lambda r: r[1])
            n5 = sum(1 for r in rows if r[1] > 0.005)
            results.append((depth, case, entry[method]["object_key"], method, fr, body, n5, len(rows)))
            flag = " <<<" if depth > 0.005 else ""
            print(f"  [{ci+1}/{len(cases)}] {case:26s} {method:12s} "
                  f"max={depth*1000:6.1f}mm @f{fr:3d} {body:14s} n>5mm={n5}{flag}", flush=True)

    results.sort(reverse=True)
    out = R0908 / "tmp" / f"torso_pen_{methods}_{only or 'all'}.tsv"
    out.parent.mkdir(exist_ok=True)
    with out.open("w") as fh:
        fh.write("depth_mm\tcase\tobject\tmethod\tframe\tbody\tn_frames_gt5mm\tn_frames\n")
        for depth, case, obj, method, fr, body, n5, n in results:
            fh.write(f"{depth*1000:.2f}\t{case}\t{obj}\t{method}\t{fr}\t{body}\t{n5}\t{n}\n")
    print(f"\n[ok] wrote {out} ({len(results)} method-cases)")
    print("\nTOP 25 by torso penetration depth:")
    for row in results[:25]:
        depth, case, obj, method, fr, body, n5, n = row
        print(f"  {depth*1000:6.1f}mm  {case:26s} {obj:10s} {method:12s} f{fr:3d} {body:12s} n>5mm={n5}")


if __name__ == "__main__":
    main()
