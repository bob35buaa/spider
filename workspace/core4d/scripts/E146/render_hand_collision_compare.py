#!/usr/bin/env python3
"""E146: hand-collision-geometry comparison via kinematic replay.

For each CORE4D case we replay the EXISTING SPIDER trajectory (kinematic
mj_forward, no physics) and render three side-by-side panels showing how three
hand collision geometries sit against the box:

  - sphere : the current 5cm `hand_collision` sphere (lh/rh).
  - 3box   : the HDMI-style forearm 3-box proxy (lh_1/2/3, rh_1/2/3).
  - rubber : the rubber_hand mesh as a (rendered) collision shell.

The hand POSE is identical across panels (same qpos); only the overlaid
collision geometry changes, so this is a pure geometry-fit comparison.

Per geometry we also compute an analytic point-box SDF to the object box
(min over hand sample points), reusing the math from unified_replay_eval.

Nothing here changes the SPIDER algorithm or the source scenes. Augmented
scenes are written under workspace/core4d/results/E145/scene_snapshot/.

Run (needs MUJOCO_GL=egl):
    MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 .venv/bin/python \
        workspace/core4d/scripts/E146/render_hand_collision_compare.py --case box023_person2
    ... --all
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[4]
E143_SNAP = REPO / "workspace/core4d/results/E143/scene_snapshot"
E143_CEM = REPO / "workspace/core4d/results/E143/cem/full"
OUT_ROOT = REPO / "workspace/core4d/results/E146"

# near-contact bands + deep-penetration threshold (match unified_replay_eval.py)
NEAR_THRESHOLDS_M = (0.03, 0.05, 0.08, 0.10)
DEEP_PENETRATION_M = -0.02

# case_id -> (snapshot dir under E143 scene_snapshot, E143 cem variant short id, label)
CASES: dict[str, dict[str, str]] = {
    "box021_035_p2": {
        "snapshot": "d003_box021_20231011_035_p2_e107_clean",
        "variant": "box021_035_p2",
        "quality": "good",
        "note": "rot 4deg / lift 0.37m clean",
    },
    "box023_person2": {
        "snapshot": "box023_person2_legobj_e026_e081",
        "variant": "box023_person2",
        "quality": "good",
        "note": "rot 5deg / lift 0.45m clean (small box)",
    },
    "box026_133_p1": {
        "snapshot": "e091_box026_20231020_133_p1_e106_clean",
        "variant": "box026_133_p1",
        "quality": "bad",
        "note": "rot 155deg (large rotation)",
    },
    "box026_134_p2": {
        "snapshot": "e091_box026_20231020_134_p2_e106_clean",
        "variant": "box026_134_p2",
        "quality": "bad",
        "note": "lift 0.03m (insufficient lift)",
    },
}

# HDMI-style forearm 3-box proxy, copied verbatim from
# example_datasets/.../box025_person1/scene_forearm.xml:284-286,330-332
FOREARM_BOXES = {
    "l": [
        ("lh_1", "0.05 0.025 0.025", "0.02 0 0", None),
        ("lh_2", "0.05 0.01 0.05", "0.09 0 0", None),
        ("lh_3", "0.025 0.01 0.05", "0.15 -0.01 0", "0.980067 0 0 -0.198669"),
    ],
    "r": [
        ("rh_1", "0.05 0.025 0.025", "0.02 0 0", None),
        ("rh_2", "0.05 0.01 0.05", "0.09 0 0", None),
        ("rh_3", "0.025 0.01 0.05", "0.15 -0.01 0", "0.980067 0 0 -0.198669"),
    ],
}

# rubber_hand visual geom placement, from the source scene (left/right differ in y)
RUBBER = {
    "l": ("lh_rubber", "left_rubber_hand", "0.0415 0.003 0"),
    "r": ("rh_rubber", "right_rubber_hand", "0.0415 -0.003 0"),
}

WRIST_BODY = {"l": "left_wrist_yaw_link", "r": "right_wrist_yaw_link"}
SPHERE_GEOMS = ["lh", "rh"]
BOX_GEOMS = ["lh_1", "lh_2", "lh_3", "rh_1", "rh_2", "rh_3"]
RUBBER_GEOMS = ["lh_rubber", "rh_rubber"]

# panel mode -> the collision geoms that belong to that mode
MODE_GEOMS = {
    "sphere": SPHERE_GEOMS,
    "3box": BOX_GEOMS,
    "rubber": RUBBER_GEOMS,
}
MODE_TITLE = {"sphere": "sphere 5cm", "3box": "forearm 3-box", "rubber": "rubber mesh"}


# --------------------------------------------------------------------------- #
# scene augmentation
# --------------------------------------------------------------------------- #
def augment_scene(src_xml: Path, out_xml: Path) -> Path:
    """Add 3-box and rubber-mesh collision geoms next to the existing sphere.

    The sphere geoms (lh/rh) already exist. We add lh_1/2/3, rh_1/2/3 (box) and
    lh_rubber/rh_rubber (mesh) into the wrist_yaw bodies. All three sets carry
    group=3 so they render only when we set their alpha; physics is irrelevant
    here (kinematic replay + analytic SDF).
    """
    tree = ET.parse(src_xml)
    root = tree.getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError(f"no worldbody in {src_xml}")

    # locate wrist bodies anywhere in the tree
    bodies = {b.get("name"): b for b in root.iter("body")}
    for side, body_name in WRIST_BODY.items():
        body = bodies.get(body_name)
        if body is None:
            raise ValueError(f"missing body {body_name} in {src_xml}")
        existing = {g.get("name") for g in body.findall("geom")}
        # 3-box
        for name, size, pos, quat in FOREARM_BOXES[side]:
            if name in existing:
                continue
            attrs = {
                "name": name,
                "type": "box",
                "size": size,
                "pos": pos,
                "group": "3",
                "rgba": "0.2 0.6 0.2 0.0",
            }
            if quat:
                attrs["quat"] = quat
            ET.SubElement(body, "geom", attrs)
        # rubber mesh as a named collision-shell geom (group 3, hidden by default)
        gname, mesh, pos = RUBBER[side]
        if gname not in existing:
            ET.SubElement(
                body,
                "geom",
                {
                    "name": gname,
                    "type": "mesh",
                    "mesh": mesh,
                    "pos": pos,
                    "quat": "1 0 0 0",
                    "group": "3",
                    "contype": "0",
                    "conaffinity": "0",
                    "rgba": "0.85 0.55 0.2 0.0",
                },
            )

    out_xml.parent.mkdir(parents=True, exist_ok=True)
    tree.write(out_xml, encoding="unicode")
    return out_xml


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


# --------------------------------------------------------------------------- #
# SDF math (reused from unified_replay_eval.py)
# --------------------------------------------------------------------------- #
def signed_point_box(
    point: np.ndarray, box_pos: np.ndarray, box_mat: np.ndarray, half: np.ndarray
) -> float:
    local = box_mat.T @ (point - box_pos)
    q = np.abs(local) - half
    outside = np.linalg.norm(np.maximum(q, 0.0))
    inside = min(max(q[0], q[1], q[2]), 0.0)
    return float(outside + inside)


def object_collision_gids(model: mujoco.MjModel) -> list[int]:
    gids = []
    for gid in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
        if name.startswith("object_collision"):
            gids.append(gid)
    if not gids:
        raise ValueError("no object_collision* geoms in scene")
    return gids


def geom_sample_points(
    model: mujoco.MjModel, data: mujoco.MjData, gid: int
) -> tuple[np.ndarray, float]:
    """World-frame sample points for a geom + a radius offset (for sphere).

    Returns (points[N,3], radius). For sphere we return the center + radius;
    for box the 8 corners; for mesh a subsample of mesh vertices.
    """
    gtype = int(model.geom_type[gid])
    pos = data.geom_xpos[gid].copy()
    mat = data.geom_xmat[gid].reshape(3, 3).copy()
    if gtype == int(mujoco.mjtGeom.mjGEOM_SPHERE):
        return pos[None, :], float(model.geom_size[gid, 0])
    if gtype == int(mujoco.mjtGeom.mjGEOM_BOX):
        half = model.geom_size[gid, :3]
        corners = []
        for sx in (-1, 1):
            for sy in (-1, 1):
                for sz in (-1, 1):
                    local = np.array([sx * half[0], sy * half[1], sz * half[2]])
                    corners.append(pos + mat @ local)
        return np.asarray(corners), 0.0
    if gtype == int(mujoco.mjtGeom.mjGEOM_MESH):
        mesh_id = int(model.geom_dataid[gid])
        v0 = int(model.mesh_vertadr[mesh_id])
        nv = int(model.mesh_vertnum[mesh_id])
        verts = model.mesh_vert[v0 : v0 + nv].reshape(-1, 3)
        if nv > 400:  # subsample for speed
            idx = np.linspace(0, nv - 1, 400).astype(int)
            verts = verts[idx]
        world = (mat @ verts.T).T + pos
        return world, 0.0
    return pos[None, :], 0.0


def geom_min_sdf(
    model: mujoco.MjModel, data: mujoco.MjData, gid: int, obj_gids: list[int]
) -> float:
    points, radius = geom_sample_points(model, data, gid)
    best = np.inf
    for og in obj_gids:
        opos = data.geom_xpos[og].copy()
        omat = data.geom_xmat[og].reshape(3, 3).copy()
        half = model.geom_size[og, :3].copy()
        for p in points:
            best = min(best, signed_point_box(p, opos, omat, half))
    return float(best - radius)


def mode_min_sdf(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    geom_names: list[str],
    obj_gids: list[int],
) -> float:
    best = np.inf
    for name in geom_names:
        gid = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name))
        if gid < 0:
            continue
        best = min(best, geom_min_sdf(model, data, gid, obj_gids))
    return float(best)


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #
def load_font(size: int):
    for path in (
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"),
    ):
        if path.is_file():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def npz_qpos(path: Path) -> np.ndarray:
    data = np.load(path, allow_pickle=True)
    arr = np.asarray(data["qpos"], dtype=np.float64)
    if arr.ndim == 3 and arr.shape[1] >= 2:
        return arr[:, 0, :]  # sim channel
    if arr.ndim == 2:
        return arr
    raise ValueError(f"unexpected qpos shape {arr.shape} in {path}")


def set_panel_visibility(
    model: mujoco.MjModel, base_rgba: np.ndarray, active_mode: str
) -> None:
    """Show object solid, fade robot mesh, show only the active hand geom set."""
    model.geom_rgba[:] = base_rgba
    color = {
        "sphere": np.array([0.9, 0.2, 0.1, 0.8]),
        "3box": np.array([0.2, 0.6, 0.2, 0.55]),
        "rubber": np.array([0.95, 0.6, 0.15, 0.85]),
    }[active_mode]
    active = set(MODE_GEOMS[active_mode])
    all_hand = set(SPHERE_GEOMS + BOX_GEOMS + RUBBER_GEOMS)
    for gid in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
        group = int(model.geom_group[gid])
        if name in all_hand:
            model.geom_rgba[gid] = color if name in active else np.array([0, 0, 0, 0.0])
        elif name.startswith("object_collision"):
            # hide collision box (object_visual mesh shows the box); SDF still uses it
            model.geom_rgba[gid, 3] = 0.0
        elif name == "object_visual":
            model.geom_rgba[gid] = np.array([0.15, 0.45, 0.85, 0.5])
        elif group == 2:  # robot visual mesh -> faint
            model.geom_rgba[gid, 3] = min(float(model.geom_rgba[gid, 3]), 0.18)
        elif group == 3:  # other collision proxies -> hide
            model.geom_rgba[gid, 3] = 0.0


def camera_for(model, data, azimuth: float) -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.azimuth = azimuth
    cam.elevation = -18
    cam.distance = 2.2
    obj_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    look = np.array([0.0, 0.0, 0.7])
    if obj_id >= 0 and pelvis_id >= 0:
        look = 0.6 * data.xpos[obj_id] + 0.4 * data.xpos[pelvis_id]
    elif obj_id >= 0:
        look = data.xpos[obj_id].copy()
    cam.lookat[:] = look
    return cam


def label(img: np.ndarray, lines: list[str], font) -> Image.Image:
    im = Image.fromarray(img)
    pad = 16 * len(lines) + 8
    out = Image.new("RGB", (im.width, im.height + pad), "white")
    out.paste(im, (0, pad))
    draw = ImageDraw.Draw(out)
    draw.rectangle([0, 0, im.width, pad], fill=(245, 245, 245))
    for i, ln in enumerate(lines):
        draw.text((8, 4 + 16 * i), ln, fill=(0, 0, 0), font=font)
    return out


def render_case(case_id: str, width: int, height: int, fps: int, stride: int) -> dict:
    spec = CASES[case_id]
    src = E143_SNAP / spec["snapshot"] / "scene_act.xml"
    if not src.is_file():
        raise FileNotFoundError(f"missing source scene_act: {src}")
    traj_dir = E143_CEM / f"E143_{spec['variant']}_raw_mask_ref_fk_outdir_full"
    traj = traj_dir / "trajectory_mjwp_act.npz"
    if not traj.is_file():
        raise FileNotFoundError(f"missing trajectory: {traj}")

    # 1. augment scene
    snap_dir = OUT_ROOT / "scene_snapshot" / case_id
    aug = augment_scene(src, snap_dir / "scene_act_handviz.xml")

    # 2. load model + qpos
    model = mujoco.MjModel.from_xml_path(str(aug))
    data = mujoco.MjData(model)
    qpos = npz_qpos(traj)
    if qpos.shape[1] != model.nq:
        raise ValueError(f"qpos nq={qpos.shape[1]} != model nq={model.nq}")
    obj_gids = object_collision_gids(model)
    base_rgba = model.geom_rgba.copy()
    font = load_font(15)

    out_dir = OUT_ROOT / "hand_collision_viz" / case_id
    out_dir.mkdir(parents=True, exist_ok=True)

    frames_idx = list(range(0, len(qpos), max(1, stride)))
    rendered: list[np.ndarray] = []
    sdf_rows: list[dict] = []
    # azimuth slowly rotates for depth perception
    azis = np.linspace(40.0, 140.0, len(frames_idx))

    # enable rendering of all geom groups (collision proxies live in group 3)
    scene_opt = mujoco.MjvOption()
    scene_opt.geomgroup[:] = 1

    with mujoco.Renderer(model, height=height, width=width) as renderer:
        for k, fi in enumerate(frames_idx):
            data.qpos[:] = qpos[fi]
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
            # per-mode SDF (geometry independent of rgba)
            sdf = {m: mode_min_sdf(model, data, MODE_GEOMS[m], obj_gids) for m in MODE_GEOMS}
            sdf_rows.append({"frame": fi, **{f"sdf_{m}_m": round(sdf[m], 4) for m in MODE_GEOMS}})

            panels = []
            cam = camera_for(model, data, float(azis[k]))
            for mode in ("sphere", "3box", "rubber"):
                set_panel_visibility(model, base_rgba, mode)
                renderer.update_scene(data, camera=cam, scene_option=scene_opt)
                img = renderer.render()
                panels.append(
                    label(
                        img,
                        [MODE_TITLE[mode], f"SDF={sdf[mode]*100:+.1f}cm  f{fi}"],
                        font,
                    )
                )
            gap = 6
            w = sum(p.width for p in panels) + gap * (len(panels) - 1)
            h = max(p.height for p in panels)
            comp = Image.new("RGB", (w, h), "white")
            x = 0
            for p in panels:
                comp.paste(p, (x, 0))
                x += p.width + gap
            rendered.append(np.asarray(comp))

    mp4 = out_dir / "compare.mp4"
    imageio.mimsave(mp4, rendered, fps=fps)

    # keyframe sheet: pick 4 frames where any geometry is most in contact (min SDF)
    sdf_arr = np.array([min(r[f"sdf_{m}_m"] for m in MODE_GEOMS) for r in sdf_rows])
    key_order = np.argsort(sdf_arr)[:4]  # most-penetrating / closest
    key_order = sorted(key_order.tolist())
    cols = 2
    rows_n = int(np.ceil(len(key_order) / cols))
    tile_h, tile_w = rendered[0].shape[0], rendered[0].shape[1]
    sheet = Image.new("RGB", (cols * tile_w, rows_n * tile_h + 30), "white")
    d = ImageDraw.Draw(sheet)
    d.text((8, 8), f"{case_id} [{spec['quality']}] {spec['note']}", fill=(0, 0, 0), font=load_font(18))
    for i, ki in enumerate(key_order):
        rr, cc = divmod(i, cols)
        sheet.paste(Image.fromarray(rendered[ki]), (cc * tile_w, 30 + rr * tile_h))
    keyimg = out_dir / "keyframes.png"
    sheet.save(keyimg)

    # sdf tsv
    tsv = out_dir / "sdf_per_geom.tsv"
    with tsv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sdf_rows[0].keys()), delimiter="\t")
        w.writeheader()
        w.writerows(sdf_rows)

    # manifest
    manifest = snap_dir / "manifest.txt"
    manifest.write_text(
        f"git_head={git_head()}\n"
        f"case_id={case_id}\n"
        f"source_scene_act={src}\n"
        f"source_sha256={sha256(src)}\n"
        f"augmented_scene={aug}\n"
        f"trajectory={traj}\n"
        f"forearm_3box_from=example_datasets/.../box025_person1/scene_forearm.xml:284-286,330-332\n"
        f"frames_total={len(qpos)} frames_rendered={len(frames_idx)} stride={stride}\n",
        encoding="utf-8",
    )

    # summary stats
    stat = {
        "case_id": case_id,
        "quality": spec["quality"],
        "frames": len(qpos),
        "mp4": str(mp4),
        "keyframes": str(keyimg),
        "sdf_tsv": str(tsv),
    }
    for m in MODE_GEOMS:
        vals = np.array([r[f"sdf_{m}_m"] for r in sdf_rows])
        stat[f"{m}_sdf_min_cm"] = round(float(vals.min()) * 100, 2)
        stat[f"{m}_sdf_mean_cm"] = round(float(vals.mean()) * 100, 2)
        # CONTACT (near-band): frac of frames where geom surface is within X of the box
        # (SDF <= X, includes penetration). Higher = better contact/adhesion.
        for thr in NEAR_THRESHOLDS_M:
            stat[f"{m}_near_{int(thr*100)}cm_frac"] = round(float((vals <= thr).mean()), 3)
        # PENETRATION (penalty): frac of frames where surface is inside the box.
        stat[f"{m}_frac_penetrate"] = round(float((vals < 0).mean()), 3)
        stat[f"{m}_frac_deep_pen_2cm"] = round(float((vals < DEEP_PENETRATION_M).mean()), 3)
    return stat


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", choices=list(CASES), help="single case id")
    ap.add_argument("--all", action="store_true", help="run all 4 cases")
    ap.add_argument("--width", type=int, default=420)
    ap.add_argument("--height", type=int, default=360)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--stride", type=int, default=2, help="render every Nth frame")
    args = ap.parse_args()

    if args.all:
        cases = list(CASES)
    elif args.case:
        cases = [args.case]
    else:
        ap.error("pass --case <id> or --all")

    stats = []
    for cid in cases:
        print(f"[E146] rendering {cid} ...", flush=True)
        stats.append(render_case(cid, args.width, args.height, args.fps, args.stride))

    summary_path = OUT_ROOT / "hand_collision_viz" / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
