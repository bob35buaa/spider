#!/usr/bin/env python3
"""Generate E028 derived freejoint tasks and support-proxy scenes."""

from __future__ import annotations

import argparse
import csv
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np


REPO = Path(__file__).resolve().parents[4]
BASE = REPO / "example_datasets/processed/core4d/unitree_g1/humanoid_object"
RESULTS = REPO / "workspace/core4d_collab_retarget/results/E028"
MANIFEST = RESULTS / "manifest.tsv"
E016_DIR = REPO / "workspace/core4d_collab_retarget/scripts/E016"
if str(E016_DIR) not in sys.path:
    sys.path.insert(0, str(E016_DIR))

import generate_e016_assets as e016_assets  # noqa: E402


ANCHOR_BODY = "support_weld_anchor"
ANCHOR_GEOM = "support_weld_anchor_geom"
WELD_NAME = "e028_support_weld"


def _read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def _remove_named_children(parent: ET.Element, tag: str, names: set[str]) -> None:
    for child in list(parent.findall(tag)):
        if child.get("name") in names:
            parent.remove(child)


def _indent(elem: ET.Element, level: int = 0) -> None:
    spacer = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = spacer + "  "
        for child in elem:
            _indent(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = spacer
    if level and (not elem.tail or not elem.tail.strip()):
        elem.tail = spacer


def _point(row: dict[str, str]) -> np.ndarray:
    return np.array(
        [
            float(row["support_proxy_point_local_x"]),
            float(row["support_proxy_point_local_y"]),
            float(row["support_proxy_point_local_z"]),
        ],
        dtype=np.float64,
    )


def generate_support_scene(
    task_dir: Path,
    scene_name: str,
    point_local: np.ndarray,
    *,
    solref_timeconst: str,
    solimp_1: str,
    solimp_2: str,
    solimp_width: str,
) -> Path:
    src = task_dir / "scene.xml"
    qpos0 = np.load(task_dir / "0/trajectory_kinematic.npz")["qpos"][0]
    obj_pos = qpos0[-7:-4].astype(np.float64)
    obj_quat = qpos0[-4:].astype(np.float64)
    support_pos = obj_pos + e016_assets._quat_apply(obj_quat, point_local)
    support_quat = obj_quat / np.clip(np.linalg.norm(obj_quat), 1e-8, None)

    tree = ET.parse(src)
    root = tree.getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError(f"{src} missing worldbody")
    equality = root.find("equality")
    if equality is None:
        equality = ET.SubElement(root, "equality")

    _remove_named_children(worldbody, "body", {ANCHOR_BODY, "object_target"})
    _remove_named_children(
        equality,
        "weld",
        {
            WELD_NAME,
            "e018b_support_weld",
            "e018_support_weld",
            "e017_support_weld",
            "e016_support_weld",
            "e014_support_weld",
            "object_weld",
        },
    )

    anchor = ET.Element(
        "body",
        {
            "name": ANCHOR_BODY,
            "mocap": "true",
            "pos": " ".join(f"{v:.8g}" for v in support_pos),
            "quat": " ".join(f"{v:.8g}" for v in support_quat),
        },
    )
    ET.SubElement(
        anchor,
        "geom",
        {
            "name": ANCHOR_GEOM,
            "type": "sphere",
            "size": "0.035",
            "rgba": "0.9 0.78 0.05 0.45",
            "group": "4",
            "contype": "0",
            "conaffinity": "0",
        },
    )
    ET.SubElement(anchor, "site", {"name": "trace_support_weld_anchor", "size": "0.025"})
    worldbody.append(anchor)
    ET.SubElement(
        equality,
        "weld",
        {
            "name": WELD_NAME,
            "body1": "object",
            "body2": ANCHOR_BODY,
            "relpose": f"{point_local[0]:.8g} {point_local[1]:.8g} {point_local[2]:.8g} 1 0 0 0",
            "solref": f"{solref_timeconst} 1",
            "solimp": f"{solimp_1} {solimp_2} {solimp_width}",
        },
    )
    _indent(root)
    out = task_dir / f"{scene_name}.xml"
    tree.write(out, encoding="unicode", xml_declaration=False)
    text = out.read_text(encoding="utf-8")
    if not text.endswith("\n"):
        out.write_text(text + "\n", encoding="utf-8")

    model = mujoco.MjModel.from_xml_path(str(out))
    if (model.nq, model.nv, model.nu) != (43, 41, 29):
        raise ValueError(f"{out} changed dims: nq/nv/nu={model.nq}/{model.nv}/{model.nu}")
    return out


def generate_assets(manifest: Path, *, force: bool) -> list[Path]:
    rows = _read_manifest(manifest)
    outputs: list[Path] = []
    prepared: set[str] = set()
    for row in rows:
        task = row["derived_task"]
        if task not in prepared:
            task_dir = e016_assets.copy_freejoint_case(row["source_task"], task, force=force)
            e016_assets.patch_leg_object_pairs(task_dir / "scene.xml", row["source_task"], task)
            prepared.add(task)
        else:
            task_dir = BASE / task
        out = generate_support_scene(
            task_dir,
            row["scene_name"],
            _point(row),
            solref_timeconst=row["weld_solref_timeconst"],
            solimp_1=row["weld_solimp_1"],
            solimp_2=row["weld_solimp_2"],
            solimp_width=row["weld_solimp_width"],
        )
        outputs.append(out)
        print(
            f"{row['variant']}: task={task} scene={out.relative_to(REPO)} "
            f"face={row['anchor_face']} point={_point(row).tolist()}"
        )
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    outputs = generate_assets(args.manifest, force=args.force)
    print(f"E028 assets done scenes={len(outputs)}")


if __name__ == "__main__":
    main()
