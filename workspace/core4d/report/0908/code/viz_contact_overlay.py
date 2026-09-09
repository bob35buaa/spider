"""Contact/penetration overlay visualization: SPIDER-CEM vs OmniRetarget.

Makes the (otherwise invisible) contact/penetration metric visible by drawing the
*same* MuJoCo hand-object contacts that produce the numbers as colored spheres on
top of an A/B side-by-side replay:

    green   = clean contact          (penetration <= 0 mm)
    yellow  = shallow penetration    (0 < depth < 3 mm)
    red     = physical penetration   (depth >= 3 mm; radius grows with depth)

Both cells replay under the identical scene_act with mj_forward, so the contacts
shown are exactly the ones behind `hand_object_physics_contact_3mm_in_mask_frac`
and `hand_object_physics_penetration_3mm_frame_frac`.

Companion static figure: per-frame deepest hand-object penetration depth (mm) and
in-mask contact indicator over time, SPIDER vs Omni.

Prototype run (single case): headless osmesa.
    MUJOCO_GL=osmesa .venv/bin/python workspace/core4d/report/0908/code/viz_contact_overlay.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "osmesa")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import imageio.v2 as imageio  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO / "workspace/core4d/scripts"))
from eval.core.core_metrics import HAND_GEOMS, mj_id, object_collision_geoms  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "paper_results" / "viz"
OUT.mkdir(parents=True, exist_ok=True)

H, W = 380, 440            # per-view cell size
FPS = 20
PEN_3MM = 0.003            # physical penetration threshold used by the metric
FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 15)
FONT_S = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)

# Multi-view cameras. mode="body" auto-frames the full robot+object; mode="hands"
# zooms onto the contact region (mean of the two hand geoms).
VIEWS = [
    {"name": "3/4 full body", "azimuth": 135.0, "elevation": -12.0, "distance": 2.6, "mode": "body"},
    {"name": "left hand macro", "azimuth": 155.0, "elevation": -6.0, "distance": 0.7, "mode": "left"},
    {"name": "right hand macro", "azimuth": 105.0, "elevation": -6.0, "distance": 0.7, "mode": "right"},
]


def hand_object_contacts(model, data, object_set, hand_gids):
    """Return list of (world_pos[3], dist) for every hand-object contact this frame.

    Mirrors the contact selection in core_metrics.evaluate_sequence: a MuJoCo
    contact whose geom pair intersects the object collision set and whose other
    geom is a hand geom. `dist` < 0 means penetration (MuJoCo convention)."""
    out = []
    for ci in range(data.ncon):
        con = data.contact[ci]
        pair = {int(con.geom1), int(con.geom2)}
        if not (object_set & pair):
            continue
        other = list(pair - object_set)
        if other and other[0] in hand_gids:
            out.append((con.pos.copy(), float(con.dist), int(other[0])))
    return out


def deepest_contact_pos(cons, gid=None):
    """World position of the deepest-penetrating contact (optionally restricted
    to hand geom ``gid``), or None."""
    sub = cons if gid is None else [c for c in cons if c[2] == gid]
    if not sub:
        return None
    return min(sub, key=lambda c: c[1])[0]


def contact_rgba_and_radius(dist: float):
    """Map MuJoCo contact dist (m) to color + sphere radius. depth = -dist."""
    depth_mm = -dist * 1000.0
    if depth_mm < 0.0:                      # separated within margin -> faint green
        return (0.1, 0.85, 0.2, 0.55), 0.010
    if depth_mm < PEN_3MM * 1000.0:         # 0..3 mm shallow -> yellow
        return (0.95, 0.85, 0.1, 0.85), 0.014
    # >= 3 mm physical penetration -> red, grow radius with depth (cap 20 mm).
    # Small enough to point at the penetration without hiding the geometry clip.
    frac = min(depth_mm, 20.0) / 20.0
    return (0.95, 0.15, 0.1, 0.9), 0.010 + frac * 0.012


def add_sphere(scene, pos, radius, rgba):
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        g, int(mujoco.mjtGeom.mjGEOM_SPHERE),
        np.array([radius, 0, 0]), np.asarray(pos, dtype=np.float64),
        np.eye(3).flatten(), np.asarray(rgba, dtype=np.float32),
    )
    scene.ngeom += 1


def prep_scene(model, object_body, collision_gids):
    """Solid opaque object *visual* mesh (so a penetrating hand is visibly clipped
    by the surface), but hide the ``object_collision*`` proxy geoms from rendering
    (they only drive the metric and otherwise occlude the view as gray boxes).
    Also hide scene site markers that clutter the figure."""
    collision = set(collision_gids)
    for gid in range(model.ngeom):
        if int(model.geom_bodyid[gid]) != object_body:
            continue
        model.geom_rgba[gid, 3] = 0.0 if gid in collision else 1.0
    for sid in range(model.nsite):
        model.site_rgba[sid, 3] = 0.0


def _lookat(mode, data, object_body, hand_gids, pelvis_body, cons, side_gids):
    if mode in ("left", "right"):
        gid = side_gids.get(mode)
        if gid is None:
            mode = "hands"
        else:
            dp = deepest_contact_pos(cons, gid)          # deepest contact on this hand
            return np.asarray(dp) if dp is not None else data.geom_xpos[gid].copy()
    if mode == "hands" and hand_gids:
        return np.mean([data.geom_xpos[g] for g in hand_gids], axis=0)
    # body: midpoint of pelvis + object in xy, torso height in z -> frames full figure
    px = data.xpos[pelvis_body].copy()
    ox = data.xpos[object_body].copy()
    return np.array([(px[0] + ox[0]) / 2, (px[1] + ox[1]) / 2, 0.85])


def render_cell(qpos, model, object_set, hand_gids, object_body, pelvis_body):
    """Replay qpos, render every VIEW per frame with contact overlay.

    Returns (frames_by_view, deepest_pen_mm[T], contact_count[T]). The depth/
    count series are view-independent (computed once per frame)."""
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, H, W)
    side_gids = {
        "left": mj_id(model, mujoco.mjtObj.mjOBJ_GEOM, "lh"),
        "right": mj_id(model, mujoco.mjtObj.mjOBJ_GEOM, "rh"),
    }
    side_gids = {k: (v if v >= 0 else None) for k, v in side_gids.items()}
    cams = []
    for v in VIEWS:
        c = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(c)
        c.azimuth, c.elevation, c.distance = v["azimuth"], v["elevation"], v["distance"]
        cams.append(c)

    frames = {v["name"]: [] for v in VIEWS}
    deepest_mm, ncontact = [], []
    focus = {v["name"]: None for v in VIEWS}   # per-view EMA-smoothed lookat (anti-jitter)
    azfocus = {v["name"]: None for v in VIEWS}  # per-view EMA-smoothed azimuth vector
    for q in qpos:
        data.qpos[:] = q
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        cons = hand_object_contacts(model, data, object_set, hand_gids)
        deepest = max((-c[1] * 1000.0 for c in cons), default=0.0)
        deepest_mm.append(deepest)
        ncontact.append(len(cons))
        for v, cam in zip(VIEWS, cams):
            target = _lookat(v["mode"], data, object_body, hand_gids, pelvis_body, cons, side_gids)
            if v["mode"] in ("left", "right"):
                prev = focus[v["name"]]
                target = target if prev is None else 0.6 * prev + 0.4 * target
                focus[v["name"]] = target
                # aim the camera from outside the body toward the hand (pelvis->hand
                # bearing) so the torso never occludes; EMA-smooth the azimuth vector.
                pel = data.xpos[pelvis_body]
                bearing = np.array([target[0] - pel[0], target[1] - pel[1]])
                n = np.linalg.norm(bearing)
                if n > 1e-6:
                    bv = bearing / n
                    av = azfocus[v["name"]]
                    av = bv if av is None else 0.7 * av + 0.3 * bv
                    azfocus[v["name"]] = av
                    # camera sits on the far side of the hand from the pelvis
                    # (MuJoCo places the cam at lookat - dist*heading), so +180.
                    cam.azimuth = float(np.degrees(np.arctan2(av[1], av[0]))) + 180.0
            cam.lookat = target
            renderer.update_scene(data, camera=cam)
            for pos, dist, _gid in cons:
                rgba, rad = contact_rgba_and_radius(dist)
                add_sphere(renderer.scene, pos, rad, rgba)
            frames[v["name"]].append(renderer.render().copy())
    renderer.close()
    return frames, np.asarray(deepest_mm), np.asarray(ncontact)


def timeline_strip(depth_mm: np.ndarray, width: int, height: int = 34) -> np.ndarray:
    """Horizontal penetration-depth heat bar over time (green->red)."""
    T = len(depth_mm)
    bar = np.zeros((height, width, 3), dtype=np.uint8)
    for x in range(width):
        t = int(x / width * T)
        d = depth_mm[min(t, T - 1)]
        if d < PEN_3MM * 1000.0:
            col = (30, int(200 * (1 - d / 3.0)) + 40, 40)
        else:
            f = min(d, 20.0) / 20.0
            col = (int(120 + 135 * f), int(120 * (1 - f)), 40)
        bar[:, x] = col
    return bar


def annotate(frame, text, color=(255, 255, 255)):
    img = Image.fromarray(frame)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, 20], fill=(0, 0, 0))
    d.text((5, 3), text, font=FONT_S, fill=color)
    return np.asarray(img)


def _labeled_strip(depth, gw, t, T, tag):
    strip = timeline_strip(depth, gw)
    img = Image.fromarray(strip)
    d = ImageDraw.Draw(img)
    cx = int(t / max(T - 1, 1) * (gw - 1))
    d.line([(cx, 0), (cx, strip.shape[0])], fill=(255, 255, 255), width=2)
    d.rectangle([0, 0, 70, strip.shape[0]], fill=(0, 0, 0))
    d.text((5, 8), tag, font=FONT_S, fill=(255, 255, 255))
    return np.asarray(img)


def compose_frame(sp_views, om_views, t, T, ldepth, rdepth, lcon, rcon, lpen3, rpen3):
    """Grid: rows = method (SPIDER top, Omni bottom), cols = views; then a live
    stats bar and two full-width penetration timelines."""
    gw = len(VIEWS) * W
    sp_row = np.concatenate(
        [annotate(sp_views[v["name"]][t], f"SPIDER · {v['name']}", (140, 255, 160)) for v in VIEWS],
        axis=1,
    )
    om_row = np.concatenate(
        [annotate(om_views[v["name"]][t], f"Omni · {v['name']}", (255, 170, 150)) for v in VIEWS],
        axis=1,
    )
    # live stats bar
    bar = Image.new("RGB", (gw, 30), (0, 0, 0))
    d = ImageDraw.Draw(bar)
    d.text((8, 6),
           f"SPIDER  deepest {ldepth[t]:5.1f}mm  contacts {int(lcon[t])}  pen>=3mm {lpen3:.0%}",
           font=FONT, fill=(140, 255, 160))
    d.text((gw // 2 + 8, 6),
           f"Omni  deepest {rdepth[t]:5.1f}mm  contacts {int(rcon[t])}  pen>=3mm {rpen3:.0%}",
           font=FONT, fill=(255, 170, 150))
    strip_l = _labeled_strip(ldepth, gw, t, T, "SPIDER")
    strip_r = _labeled_strip(rdepth, gw, t, T, "Omni")
    return np.concatenate([sp_row, om_row, np.asarray(bar), strip_l, strip_r], axis=0)


def static_figure(ldepth, rdepth, lcon, rcon, out_png, case_id):
    T = len(ldepth)
    miss = (lcon > 0) & (rcon == 0)   # SPIDER contacts, Omni misses
    fig, axes = plt.subplots(3, 1, figsize=(9, 6.2), sharex=True)
    ax = axes[0]
    ax.plot(ldepth, color="#1f8f3a", lw=1.8, label="SPIDER-CEM")
    ax.plot(rdepth, color="#c0261a", lw=1.8, label="OmniRetarget")
    ax.axhline(PEN_3MM * 1000, color="k", ls="--", lw=1, label="3 mm threshold")
    ax.set_ylabel("deepest hand-object\npenetration (mm)")
    ax.legend(loc="upper right", fontsize=8, ncol=3)
    ax.set_title(f"{case_id}: per-frame contact penetration depth", fontsize=10)
    ax = axes[1]
    ax.fill_between(range(T), 0, (ldepth >= PEN_3MM * 1000).astype(float),
                    step="mid", color="#1f8f3a", alpha=0.5, label="SPIDER pen>=3mm")
    ax.fill_between(range(T), 0, -(rdepth >= PEN_3MM * 1000).astype(float),
                    step="mid", color="#c0261a", alpha=0.5, label="Omni pen>=3mm")
    ax.set_ylim(-1.1, 1.1)
    ax.set_yticks([-1, 0, 1])
    ax.set_yticklabels(["Omni", "", "SPIDER"])
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax = axes[2]
    ax.fill_between(range(T), 0, (lcon > 0).astype(float),
                    step="mid", color="#1f8f3a", alpha=0.5, label="SPIDER in contact")
    ax.fill_between(range(T), 0, -(rcon > 0).astype(float),
                    step="mid", color="#c0261a", alpha=0.5, label="Omni in contact")
    for x in np.where(miss)[0]:                       # SPIDER-hit / Omni-miss frames
        ax.axvspan(x - 0.5, x + 0.5, color="#1f6fff", alpha=0.25, lw=0)
    ax.set_ylim(-1.1, 1.1)
    ax.set_yticks([-1, 0, 1])
    ax.set_yticklabels(["Omni", "", "SPIDER"])
    ax.set_xlabel("frame")
    ax.set_title(f"contact presence  (blue band = SPIDER contact but Omni none; {int(miss.sum())} frames)",
                 fontsize=9)
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def npz_phys(path):
    q = np.load(path)["qpos"]
    return q[:, 0, :] if q.ndim == 3 else q


# --- per-case source resolution (SPIDER npz + scene) --------------------------------
# Object-group -> selected-rollout rl-export tsv (matches gen_paper_results.GROUPS).
MISSING_MOUNT = "/mnt/a0ccc676-9496-49f8-a861-f8a1797dec52/spider_workdirs"
_E198 = "workspace/core4d/results/E198/s6_downstream/rl_export"
_E212 = "workspace/core4d/results/E212/s6_downstream/rl_export/paired_rl_export_input.tsv"
RL_TSV = {
    "box001": f"{_E198}/box001_user_approved/rl_export_input.tsv",
    "box004": f"{_E198}/box004_user_approved/rl_export_input.tsv",
    "box024": f"{_E198}/box024_user_approved/rl_export_input.tsv",
    "box021": "workspace/core4d/results/E170/s6_downstream/rl_export/paired_rl_export_input.tsv",
    "box023": "workspace/core4d/results/E190/s6_downstream/rl_export/box023_noPRG_user_approved/paired_rl_export_input.tsv",
    "bucket003": "workspace/core4d/results/E178/s6_downstream/rl_export/paired_rl_export_input.tsv",
    "bucket007": "workspace/core4d/results/E207/s6_downstream/rl_export/paired_rl_export_input.tsv",
    "chair006": _E212, "desk007": _E212, "desk021": _E212, "desk023": _E212,
}


def _remap(value: str) -> str:
    if value.startswith(MISSING_MOUNT):
        tail = value[len(MISSING_MOUNT):].lstrip("/")
        cand = REPO / "workspace" / tail
        return str(cand if (cand.exists() or tail.startswith("core4d/")) else REPO / tail)
    return value


def _resolve(value: str):
    if not value:
        return None
    p = Path(_remap(value))
    if p.is_file():
        return p
    if not p.is_absolute() and (REPO / p).is_file():
        return REPO / p
    for marker in ("workspace/core4d/", "example_datasets/"):
        if marker in value:
            cand = REPO / (marker + value.split(marker, 1)[1])
            if cand.is_file():
                return cand
    return None


def resolve_case(case_id):
    """Return (spider_phys_npz, omni_qpos_npz, scene_xml) for a case, or None."""
    import csv
    obj = case_id.split("_", 1)[0]
    tsv = REPO / RL_TSV[obj]
    rows = list(csv.DictReader(tsv.open(newline=""), delimiter="\t"))
    row = next((r for r in rows if r.get("case_id") == case_id), None)
    if row is None:
        raise ValueError(f"{case_id}: not found in {tsv}")
    cem = _resolve(row.get("cem_result_npz", ""))
    scene = _resolve(row.get("scene_act", ""))
    omni = OUT.parent / "omni_scene_act_qpos" / f"{case_id}_omnirt_scene_act_qpos.npz"
    if cem is None or scene is None or not omni.is_file():
        raise ValueError(f"{case_id}: unresolved cem={cem} scene={scene} omni_exists={omni.is_file()}")
    return cem, omni, scene


def run_case(case_id, spider_npz, omni_npz, scene_xml):
    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    object_set = set(object_collision_geoms(model))
    hand_gids = [g for name in HAND_GEOMS if (g := mj_id(model, mujoco.mjtObj.mjOBJ_GEOM, name)) >= 0]
    object_body = mj_id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    pelvis_body = mj_id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    prep_scene(model, object_body, object_set)

    sp = npz_phys(spider_npz)
    om = np.load(omni_npz)["qpos"]
    T = min(len(sp), len(om))
    sp, om = sp[:T], om[:T]

    sp_views, ldepth, lcon = render_cell(sp, model, object_set, hand_gids, object_body, pelvis_body)
    om_views, rdepth, rcon = render_cell(om, model, object_set, hand_gids, object_body, pelvis_body)

    lpen3 = float(np.mean(ldepth >= PEN_3MM * 1000))
    rpen3 = float(np.mean(rdepth >= PEN_3MM * 1000))

    frames = [
        compose_frame(sp_views, om_views, t, T, ldepth, rdepth, lcon, rcon, lpen3, rpen3)
        for t in range(T)
    ]
    mp4 = OUT / f"{case_id}_contact_overlay.mp4"
    imageio.mimsave(mp4, frames, fps=FPS, quality=8)
    png = OUT / f"{case_id}_penetration_timeseries.png"
    static_figure(ldepth, rdepth, lcon, rcon, png, case_id)

    # Frames where SPIDER holds contact but OmniRetarget makes none.
    miss = np.where((lcon > 0) & (rcon == 0))[0]
    if miss.size:
        best = int(miss[np.argmax(lcon[miss])])   # SPIDER's strongest such contact
        still = OUT / f"{case_id}_spiderHit_omniMiss_f{best}.png"
        imageio.imwrite(still, frames[best])
        print(f"     SPIDER-hit / Omni-miss frames ({miss.size}): {miss.tolist()}")
        print(f"     representative still (frame {best}) -> {still}")
    else:
        print("     SPIDER-hit / Omni-miss frames: none")

    # Penetration-contrast frame: Omni penetrates deeply while SPIDER makes clean
    # (contacting, shallow) contact -> the money shot for "Omni buries, SPIDER holds".
    CLEAN_MM = 3.0
    contrast = np.where((lcon > 0) & (ldepth < CLEAN_MM) & (rcon > 0) & (rdepth >= CLEAN_MM))[0]
    if contrast.size:
        cbest = int(contrast[np.argmax(rdepth[contrast])])  # deepest Omni penetration
        cstill = OUT / f"{case_id}_omniPen_spiderClean_f{cbest}.png"
        imageio.imwrite(cstill, frames[cbest])
        print(f"     penetration-contrast frame {cbest} "
              f"(Omni {rdepth[cbest]:.1f}mm vs SPIDER {ldepth[cbest]:.1f}mm) -> {cstill}")
    else:
        print("     penetration-contrast frame: none (no clean-SPIDER + deep-Omni frame)")
    print(f"[ok] {case_id}: SPIDER pen>=3mm={lpen3:.1%}  Omni pen>=3mm={rpen3:.1%}")
    print(f"     video -> {mp4}")
    print(f"     figure -> {png}")


def render_orbit(case_id, frame, n_az=36, distance=0.9, elevation=-12.0, cell=520):
    """Orbit a fixed frame: render both methods from n_az azimuths around the
    hand/contact centroid. Returns (frames_by_method, deepest_mm_by_method, az_list)."""
    cem, omni, scene = resolve_case(case_id)
    model = mujoco.MjModel.from_xml_path(str(scene))
    object_set = set(object_collision_geoms(model))
    hand_gids = [g for name in HAND_GEOMS if (g := mj_id(model, mujoco.mjtObj.mjOBJ_GEOM, name)) >= 0]
    object_body = mj_id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    prep_scene(model, object_body, object_set)
    qframes = {"SPIDER": npz_phys(cem)[frame], "Omni": np.load(omni)["qpos"][frame]}
    az_list = np.linspace(0.0, 360.0, n_az, endpoint=False)
    renderer = mujoco.Renderer(model, cell, cell)
    frames, deepest = {}, {}
    for method, q in qframes.items():
        data = mujoco.MjData(model)
        data.qpos[:] = q
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        cons = hand_object_contacts(model, data, object_set, hand_gids)
        deepest[method] = max((-c[1] * 1000.0 for c in cons), default=0.0)
        # centroid of contact points (fall back to hand geoms) -> orbit focus
        if cons:
            focus = np.mean([c[0] for c in cons], axis=0)
        else:
            focus = np.mean([data.geom_xpos[g] for g in hand_gids], axis=0)
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(cam)
        cam.elevation, cam.distance = elevation, distance
        cam.lookat = focus
        mframes = []
        for az in az_list:
            cam.azimuth = float(az)
            renderer.update_scene(data, camera=cam)
            for pos, dist, _gid in cons:
                rgba, rad = contact_rgba_and_radius(dist)
                add_sphere(renderer.scene, pos, rad, rgba)
            mframes.append(renderer.render().copy())
        frames[method] = mframes
    renderer.close()
    return frames, deepest, az_list, cell


def run_orbit(case_id, frame, n_az=36):
    frames, deepest, az_list, cell = render_orbit(case_id, frame, n_az=n_az)
    tag = f"{case_id}_f{frame}"
    # side-by-side orbit video (SPIDER | Omni), labeled per azimuth
    vid = []
    for i, az in enumerate(az_list):
        pair = np.concatenate([frames["SPIDER"][i], frames["Omni"][i]], axis=1)
        img = Image.fromarray(pair)
        d = ImageDraw.Draw(img)
        d.rectangle([0, 0, 2 * cell, 22], fill=(0, 0, 0))
        d.text((6, 3), f"SPIDER  deepest {deepest['SPIDER']:.1f}mm", font=FONT, fill=(140, 255, 160))
        d.text((cell + 6, 3), f"Omni  deepest {deepest['Omni']:.1f}mm", font=FONT, fill=(255, 170, 150))
        d.rectangle([2 * cell - 96, 0, 2 * cell, 22], fill=(0, 0, 0))
        d.text((2 * cell - 92, 3), f"az {int(az):3d}°", font=FONT, fill=(230, 230, 230))
        vid.append(np.asarray(img))
    mp4 = OUT / f"{tag}_orbit.mp4"
    imageio.mimsave(mp4, vid, fps=15, quality=8)
    # contact sheet: 8 azimuths, SPIDER row over Omni row
    picks = np.linspace(0, n_az, 8, endpoint=False).astype(int)
    sh = 260
    def strip(method):
        cells = []
        for i in picks:
            im = Image.fromarray(frames[method][i]).resize((sh, sh))
            d = ImageDraw.Draw(im)
            d.rectangle([0, 0, sh, 18], fill=(0, 0, 0))
            d.text((4, 2), f"{method} az{int(az_list[i])}", font=FONT_S,
                   fill=(140, 255, 160) if method == "SPIDER" else (255, 170, 150))
            cells.append(np.asarray(im))
        return np.concatenate(cells, axis=1)
    sheet = np.concatenate([strip("SPIDER"), strip("Omni")], axis=0)
    png = OUT / f"{tag}_orbit_sheet.png"
    imageio.imwrite(png, sheet)
    print(f"[ok] {tag}: SPIDER {deepest['SPIDER']:.1f}mm  Omni {deepest['Omni']:.1f}mm")
    print(f"     orbit video -> {mp4}")
    print(f"     contact sheet -> {png}")


def scan_series(qpos, model, object_set, hand_gids):
    """Metric-only (no render): per-frame (deepest_pen_mm, contact_count)."""
    data = mujoco.MjData(model)
    depth, ncon = [], []
    for q in qpos:
        data.qpos[:] = q
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        cons = hand_object_contacts(model, data, object_set, hand_gids)
        depth.append(max((-c[1] * 1000.0 for c in cons), default=0.0))
        ncon.append(len(cons))
    return np.asarray(depth), np.asarray(ncon)


def scan_case(case_id, clean_mm=2.0, pen_mm=3.0):
    """Return the best penetration-contrast frame for a case, or None.

    Contrast frame: SPIDER contacting & clean (< clean_mm) while Omni contacts &
    penetrates >= pen_mm.  Score = Omni deepest penetration at that frame."""
    cem, omni, scene = resolve_case(case_id)
    model = mujoco.MjModel.from_xml_path(str(scene))
    object_set = set(object_collision_geoms(model))
    hand_gids = [g for name in HAND_GEOMS if (g := mj_id(model, mujoco.mjtObj.mjOBJ_GEOM, name)) >= 0]
    sp = npz_phys(cem)
    om = np.load(omni)["qpos"]
    T = min(len(sp), len(om))
    ld, lc = scan_series(sp[:T], model, object_set, hand_gids)
    rd, rc = scan_series(om[:T], model, object_set, hand_gids)
    ok = np.where((lc > 0) & (ld < clean_mm) & (rc > 0) & (rd >= pen_mm))[0]
    if not ok.size:
        return {"case": case_id, "score": 0.0, "frame": None,
                "spider_pen3": float(np.mean(ld >= pen_mm)), "omni_pen3": float(np.mean(rd >= pen_mm))}
    f = int(ok[np.argmax(rd[ok])])
    return {"case": case_id, "score": float(rd[f]), "frame": f,
            "omni_mm": float(rd[f]), "spider_mm": float(ld[f]),
            "spider_pen3": float(np.mean(ld >= pen_mm)), "omni_pen3": float(np.mean(rd >= pen_mm))}


def run_scan():
    cases = [l.strip() for l in (REPO / "tmp/paper_case_id.txt").read_text().splitlines() if l.strip()]
    results = []
    for cid in cases:
        try:
            results.append(scan_case(cid))
        except Exception as e:
            print(f"[skip] {cid}: {e}")
    results.sort(key=lambda r: r["score"], reverse=True)
    print("\n=== penetration-contrast ranking (Omni deep pen @ SPIDER-clean frame) ===")
    print(f"{'rank':>4} {'case':44} {'frame':>6} {'omni_mm':>8} {'spider_mm':>9} {'omni_pen3':>9}")
    for i, r in enumerate(results, 1):
        fr = r["frame"] if r["frame"] is not None else -1
        print(f"{i:>4} {r['case']:44} {fr:>6} {r['score']:>8.1f} "
              f"{r.get('spider_mm', 0):>9.1f} {r['omni_pen3']:>9.1%}")


DEFAULT_CASES = [
    "box001_20231023_108_p1",
    "box021_20231011_037_p2",
    "desk021_20231008_005_p2",
    "desk023_20231030_019_p1",
    "box004_20231003_2_082_p1",
    "box023_20231020_042_p2",
]

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("cases", nargs="*", default=DEFAULT_CASES)
    ap.add_argument("--scan", action="store_true", help="metric-only contrast ranking over all 52 cases")
    ap.add_argument("--orbit", nargs=2, metavar=("CASE", "FRAME"),
                    help="orbit a fixed frame from many azimuths (multi-view)")
    ap.add_argument("--n-az", type=int, default=36, help="orbit azimuth count")
    args = ap.parse_args()
    if args.scan:
        run_scan()
        raise SystemExit(0)
    if args.orbit:
        run_orbit(args.orbit[0], int(args.orbit[1]), n_az=args.n_az)
        raise SystemExit(0)
    cases = args.cases or DEFAULT_CASES
    for i, cid in enumerate(cases, 1):
        print(f"=== [{i}/{len(cases)}] {cid} ===")
        try:
            cem, omni, scene = resolve_case(cid)
            run_case(cid, cem, omni, scene)
        except Exception as e:  # keep going; report the failing case
            print(f"[FAIL] {cid}: {e}")
