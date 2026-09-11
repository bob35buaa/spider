"""Shared rendering style helpers for the 0908 paper visuals.

Two concerns live here:

1. `beautify_robot_scene_xml` -- lxml surgery on a MuJoCo scene XML to give the
   MuJoCo native renderer a look approaching the OmniRetarget paper figure:
   tiled/reflective floor, warm MDF box, studio-ish multi-light rig with soft
   shadows, gradient sky. One beautified XML drives BOTH deliverable versions;
   the render pass toggles floor / shadow / sky to switch between:
     - "bg"    : tiled floor + gradient sky + shadows + reflection
     - "white" : floor hidden, shadows off, background painted pure white.

2. `MDF_BOX_RGBA` / palette constants reused across scripts.
"""
from __future__ import annotations

import re
from pathlib import Path

from lxml import etree

# --- default data locations (override via CLI: --data-root / --smplx-model) ---
DEFAULT_DATA_ROOT = ("/mnt/ali-sh-1/usr/xiayibo/xyb_data_tidal_alsh/other-datasets/"
                     "mocap_data/CORE4D/CORE4D_Real")
DEFAULT_SMPLX_MODEL = ("/mnt/ali-sh-1/usr/xiayibo/xyb_data_tidal_alsh/other-datasets/"
                       "mocap_data/human_model_files/smplx/SMPLX_NEUTRAL.npz")
DEFAULT_CASE = "box021_20231011_037"          # <object><NNN>_<date>_<seq>
_VIZ_ROOT = Path(__file__).resolve().parents[1] / "paper_results" / "viz"


def parse_case(case: str):
    """'box021_20231011_037' -> (object, date, seq, category). The object-model
    subdir is the object name with its trailing digits stripped (box021 -> box)."""
    parts = case.split("_")
    if len(parts) != 3:
        raise ValueError(f"--case must be '<object>_<date>_<seq>', got {case!r}")
    obj, date, seq = parts
    category = re.sub(r"\d+$", "", obj)
    return obj, date, seq, category


def seq_dir(data_root: str, date: str, seq: str) -> Path:
    return Path(data_root) / "human_object_motions" / date / seq


def object_mesh_path(data_root: str, obj: str, category: str) -> Path:
    return Path(data_root) / "object_models" / category / f"{obj}_m.obj"


def default_out(case: str, primary: str = "p2") -> Path:
    return _VIZ_ROOT / f"{case}_{primary}"


def add_common_args(parser):
    """Args shared by every render script."""
    parser.add_argument("--case", default=DEFAULT_CASE,
                        help="sequence id '<object>_<date>_<seq>' (default: %(default)s)")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT,
                        help="CORE4D_Real root (default: %(default)s)")
    parser.add_argument("--smplx-model", default=DEFAULT_SMPLX_MODEL,
                        help="SMPLX_NEUTRAL.npz path (default: %(default)s)")
    parser.add_argument("--out", default=None,
                        help="output dir (default: <viz>/<case>_<primary>)")
    parser.add_argument("--res", type=int, default=1080, help="render resolution")
    parser.add_argument("--fps", type=int, default=20, help="output video fps")
    parser.add_argument("--preview", type=int, default=0,
                        help="render only N evenly-sampled frames (0 = full clip)")
    return parser

# Slate blue for the manipulated box, tuned per renderer to read as the SAME blue
# across all deliverables. MuJoCo (BOX_RGBA) keeps the original box021_material
# value; pyrender's stronger lighting washes that out, so BOX_COLOR is deepened to
# match the MuJoCo look visually.
BOX_RGBA = "0.40 0.50 0.60 1"          # robot scene (MuJoCo native renderer)
BOX_COLOR = [0.24, 0.36, 0.54, 1.0]    # SMPLX / mixed (pyrender; deeper to survive top light)
# CARI4D SMPL body colour (light periwinkle) -- used as the neutral human tone.
CARI4D_SMPL_COLOR = [0.65098039, 0.74117647, 0.85882353, 1.0]


def _set_attrs(elem, **attrs):
    for k, v in attrs.items():
        elem.set(k, str(v))


# --- pyrender helpers (SMPLX reference render; CARI4D-style soft Phong look) ---

def make_checker_floor(y: float, extent: float = 8.0, repeats: int = 16, up: str = "y"):
    """A large quad with a procedural warm-beige checker texture (matches the
    robot scene's tiled floor). `up` is the world up-axis ('y' or 'z')."""
    import numpy as np
    import trimesh
    from PIL import Image

    n, k = 512, 64  # 8x8 checks per texture tile
    c1 = np.array([214, 207, 194], np.uint8)   # light beige
    c2 = np.array([171, 161, 146], np.uint8)   # darker beige
    img = np.empty((n, n, 3), np.uint8)
    for i in range(8):
        for j in range(8):
            img[i * k:(i + 1) * k, j * k:(j + 1) * k] = c1 if (i + j) % 2 == 0 else c2
    e, r = extent, float(repeats)
    if up == "y":
        v = np.array([[-e, y, -e], [e, y, -e], [e, y, e], [-e, y, e]], float)
        f = np.array([[0, 2, 1], [0, 3, 2]])            # +Y normal
    else:  # z-up
        v = np.array([[-e, -e, y], [e, -e, y], [e, e, y], [-e, e, y]], float)
        f = np.array([[0, 1, 2], [0, 2, 3]])            # +Z normal
    uv = np.array([[0, 0], [r, 0], [r, r], [0, r]], float)
    mesh = trimesh.Trimesh(vertices=v, faces=f, process=False)
    mesh.visual = trimesh.visual.TextureVisuals(
        uv=uv, image=Image.fromarray(img))
    return mesh


def soft_phong_lights(look_at_fn, target, up_vec, floor_h, up: str = "y"):
    """CARI4D-style soft lighting: an overhead point light plus a directional key
    and a fill, all aimed at `target`. Returns [(light, pose), ...]."""
    import numpy as np
    import pyrender

    over = np.array(target, float)
    over[1 if up == "y" else 2] = floor_h + 3.5
    pl_pose = np.eye(4)
    pl_pose[:3, 3] = over
    key_dir = np.array([0.7, 1.0, 0.6]) if up == "y" else np.array([0.7, 0.6, 1.0])
    fill_dir = np.array([-0.8, 0.5, -0.4]) if up == "y" else np.array([-0.8, -0.4, 0.5])
    # Softer overhead point (was blowing out upward-facing box tops); lean on the
    # side-lit directional key/fill so the box keeps its blue from every angle.
    return [
        (pyrender.PointLight(color=[1, 1, 1], intensity=10.0), pl_pose),
        (pyrender.DirectionalLight(color=[1.0, 0.98, 0.95], intensity=4.0),
         look_at_fn(target + key_dir, target, up_vec)),
        (pyrender.DirectionalLight(color=[0.9, 0.92, 0.98], intensity=2.0),
         look_at_fn(target + fill_dir, target, up_vec)),
    ]


def beautify_robot_scene_xml(src_xml: str, out_xml: str) -> None:
    """Rewrite the <visual>/<asset> and worldbody lights of a MuJoCo scene so the
    native renderer produces a paper-quality frame. Robot link meshes keep their
    real silver/black materials; only floor, sky, box and lighting change."""
    parser = etree.XMLParser(remove_blank_text=True)
    tree = etree.parse(src_xml, parser)
    root = tree.getroot()

    # --- <visual>: soft ambient headlight, big crisp shadow map, warm haze ---
    visual = root.find("visual")
    if visual is None:
        visual = etree.SubElement(root, "visual")
    hl = visual.find("headlight")
    if hl is None:
        hl = etree.SubElement(visual, "headlight")
    _set_attrs(hl, ambient="0.40 0.40 0.41", diffuse="0.32 0.32 0.33",
               specular="0.12 0.12 0.12")
    quality = visual.find("quality")
    if quality is None:
        quality = etree.SubElement(visual, "quality")
    _set_attrs(quality, shadowsize="8192", offsamples="8")
    mp = visual.find("map")
    if mp is None:
        mp = etree.SubElement(visual, "map")
    _set_attrs(mp, shadowclip="4", shadowscale="0.7")
    rgba = visual.find("rgba")
    if rgba is None:
        rgba = etree.SubElement(visual, "rgba")
    _set_attrs(rgba, haze="0.86 0.90 0.95 1")
    gl = visual.find("global")
    if gl is not None:
        _set_attrs(gl, offwidth="1920", offheight="1920")

    # --- <asset>: gradient sky, visible gray floor tiles, MDF box material ---
    asset = root.find("asset")
    for tex in asset.findall("texture"):
        if tex.get("type") == "skybox":
            _set_attrs(tex, builtin="gradient", rgb1="0.73 0.72 0.70",
                       rgb2="0.95 0.95 0.94", width="512", height="512")
            tex.attrib.pop("mark", None)
        elif tex.get("name") == "groundplane":
            # warm beige stone tiles with darker grout (matches the reference figure)
            _set_attrs(tex, builtin="checker", mark="edge",
                       rgb1="0.84 0.81 0.76", rgb2="0.67 0.63 0.57",
                       markrgb="0.50 0.47 0.43", width="512", height="512")
    for m in asset.findall("material"):
        if m.get("name") == "groundplane":
            _set_attrs(m, texrepeat="3 3", texuniform="true", reflectance="0.15")
        # box material name varies per object (e.g. box021_material)
        if m.get("name", "").endswith("_material") and "box" in m.get("name", ""):
            _set_attrs(m, rgba=BOX_RGBA, specular="0.15", shininess="0.2",
                       reflectance="0.0")

    # --- worldbody lights: replace the lone directional light with a rig ---
    worldbody = root.find("worldbody")
    for lt in worldbody.findall("light"):
        worldbody.remove(lt)
    # key (casts the shadow), fill, rim -- all directional, warm/cool balance.
    key = etree.SubElement(worldbody, "light")
    _set_attrs(key, name="key", pos="2.5 -1.5 4.0", dir="-0.5 0.3 -0.9",
               directional="true", castshadow="true",
               diffuse="0.78 0.75 0.70", specular="0.28 0.28 0.28")
    fill = etree.SubElement(worldbody, "light")
    _set_attrs(fill, name="fill", pos="-3.0 -2.0 2.5", dir="0.6 0.4 -0.6",
               directional="true", castshadow="false",
               diffuse="0.3 0.32 0.36", specular="0.0 0.0 0.0")
    rim = etree.SubElement(worldbody, "light")
    _set_attrs(rim, name="rim", pos="0.0 3.0 3.0", dir="0.0 -0.6 -0.8",
               directional="true", castshadow="false",
               diffuse="0.25 0.25 0.28", specular="0.1 0.1 0.1")

    tree.write(out_xml, pretty_print=True, xml_declaration=False)
