#!/usr/bin/env python3
"""Generate E023 derived tasks for lower-body geometry repair."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco


REPO = Path(__file__).resolve().parents[4]
BASE = REPO / "example_datasets/processed/core4d/unitree_g1/humanoid_object"
WS = REPO / "workspace/core4d_collab_retarget"
VARIANTS = WS / "scripts/E023/variants.tsv"
E018B_RESULTS = WS / "results/E018b"
E018B_MANIFEST = E018B_RESULTS / "manifest.tsv"
RESULTS = WS / "results/E023"
MANIFEST = RESULTS / "manifest.tsv"

FIELDNAMES = [
    "variant",
    "source_variant",
    "case_slug",
    "patch_mode",
    "queue",
    "role",
    "wave",
    "lowerbody_radius",
    "foot_radius",
    "notes",
]

LEG_FOOT_GEOMS = [
    "left_hip_collision",
    "right_hip_collision",
    "left_thigh_collision",
    "right_thigh_collision",
    "left_shin_collision",
    "right_shin_collision",
    "left_linkage_brace_collision",
    "right_linkage_brace_collision",
    "lf0",
    "lf1",
    "lf2",
    "lf3",
    "rf0",
    "rf1",
    "rf2",
    "rf3",
]
FOOT_GEOMS = {"lf0", "lf1", "lf2", "lf3", "rf0", "rf1", "rf2", "rf3"}
PAIR_NAMES = {f"{geom}_object" for geom in LEG_FOOT_GEOMS}

EXTRA_FIELDS = [
    "E023_source_variant",
    "E023_source_derived_task",
    "E023_patch_mode",
    "E023_disabled_leg_object_pairs",
    "E023_shrunk_leg_geoms",
    "E023_lowerbody_radius",
    "E023_foot_radius",
    "E023_notes",
    "contact_hdmi_mask_path",
    "contact_hdmi_mask_time_axis",
]


def _read_tsv(path: Path, fieldnames: list[str] | None = None) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        lines = [line for line in f if line.strip() and not line.startswith("#")]
    return list(csv.DictReader(lines, delimiter="\t", fieldnames=fieldnames))


def _e018b_manifest() -> dict[str, dict[str, str]]:
    return {row["variant"]: row for row in _read_tsv(E018B_MANIFEST)}


def _copy_task(src_task: str, dst_task: str, *, force: bool) -> None:
    src = BASE / src_task
    dst = BASE / dst_task
    if not src.is_dir():
        raise FileNotFoundError(src)
    if force and dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, dirs_exist_ok=True)


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


def _write_tree(tree: ET.ElementTree, path: Path) -> None:
    root = tree.getroot()
    _indent(root)
    tree.write(path, encoding="unicode", xml_declaration=False)
    text = path.read_text(encoding="utf-8")
    if not text.endswith("\n"):
        path.write_text(text + "\n", encoding="utf-8")


def _remove_leg_pairs(root: ET.Element) -> list[str]:
    removed: list[str] = []
    for contact in root.findall("contact"):
        for child in list(contact.findall("pair")):
            name = child.get("name", "")
            if name in PAIR_NAMES:
                contact.remove(child)
                removed.append(name)
    return sorted(set(removed))


def _shrink_leg_geoms(root: ET.Element, *, lowerbody_radius: float, foot_radius: float) -> list[str]:
    changed: list[str] = []
    for geom in root.iter("geom"):
        name = geom.get("name", "")
        if name not in LEG_FOOT_GEOMS:
            continue
        target = foot_radius if name in FOOT_GEOMS else lowerbody_radius
        size = geom.get("size", "").split()
        if not size:
            size = [f"{target:.8g}"]
        else:
            size[0] = f"{target:.8g}"
        geom.set("size", " ".join(size))
        changed.append(name)
    return sorted(set(changed))


def _validate_scene(path: Path) -> None:
    model = mujoco.MjModel.from_xml_path(str(path))
    if model.nq != 43 or model.nv != 41 or model.nu != 29:
        raise ValueError(f"{path} changed dims: nq/nv/nu={model.nq}/{model.nv}/{model.nu}")


def _patch_task_xmls(
    task: str,
    patch_mode: str,
    *,
    lowerbody_radius: float,
    foot_radius: float,
) -> tuple[list[str], list[str]]:
    task_dir = BASE / task
    disabled: set[str] = set()
    shrunk: set[str] = set()
    for xml_path in sorted(task_dir.glob("scene*.xml")):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        if patch_mode == "legpair_off":
            disabled.update(_remove_leg_pairs(root))
        elif patch_mode == "lowerbody_proxy_min":
            shrunk.update(
                _shrink_leg_geoms(
                    root,
                    lowerbody_radius=lowerbody_radius,
                    foot_radius=foot_radius,
                )
            )
        elif patch_mode == "none":
            pass
        else:
            raise ValueError(f"Unsupported patch_mode={patch_mode}")
        _write_tree(tree, xml_path)
        _validate_scene(xml_path)
    return sorted(disabled), sorted(shrunk)


def _copy_mask(row: dict[str, str], variant: str) -> str:
    src_dir = E018B_RESULTS / "contact_masks" / row["mask_slug"]
    if not src_dir.is_dir():
        raise FileNotFoundError(src_dir)
    dst_dir = RESULTS / "contact_masks" / variant
    dst_dir.mkdir(parents=True, exist_ok=True)
    for name in ("raw_contact_mask_3cm.npz", "raw_contact_mask_3cm.csv", "audit_summary_3cm.json"):
        src = src_dir / name
        if src.is_file():
            shutil.copy2(src, dst_dir / name)
    mask = dst_dir / "raw_contact_mask_3cm.npz"
    if not mask.is_file():
        raise FileNotFoundError(mask)
    return str(mask.relative_to(REPO))


def _dst_task(source_task: str, variant: str) -> str:
    slug = variant.removeprefix("E023_")
    return f"{source_task}_freejoint_legobj_e023_{slug}"


def generate(force: bool = False) -> None:
    e018b_rows = _e018b_manifest()
    rows: list[dict[str, str]] = []
    RESULTS.mkdir(parents=True, exist_ok=True)
    for variant_row in _read_tsv(VARIANTS, FIELDNAMES):
        source_variant = variant_row["source_variant"]
        source_meta = e018b_rows[source_variant]
        dst_task = _dst_task(source_meta["source_task"], variant_row["variant"])
        _copy_task(source_meta["derived_task"], dst_task, force=force)
        disabled, shrunk = _patch_task_xmls(
            dst_task,
            variant_row["patch_mode"],
            lowerbody_radius=float(variant_row["lowerbody_radius"] or 0.0),
            foot_radius=float(variant_row["foot_radius"] or 0.0),
        )
        mask_path = _copy_mask(source_meta, variant_row["variant"])

        meta = dict(source_meta)
        meta.update(
            {
                "variant": variant_row["variant"],
                "derived_task": dst_task,
                "queue": variant_row["queue"],
                "role": variant_row["role"],
                "wave": variant_row["wave"],
                "source_variant": source_variant,
                "online_video_path": (
                    f"workspace/core4d_collab_retarget/results/E023/online_video/{variant_row['variant']}.mp4"
                ),
                "E023_source_variant": source_variant,
                "E023_source_derived_task": source_meta["derived_task"],
                "E023_patch_mode": variant_row["patch_mode"],
                "E023_disabled_leg_object_pairs": ",".join(disabled),
                "E023_shrunk_leg_geoms": ",".join(shrunk),
                "E023_lowerbody_radius": variant_row["lowerbody_radius"],
                "E023_foot_radius": variant_row["foot_radius"],
                "E023_notes": variant_row["notes"],
                "contact_hdmi_mask_path": mask_path,
                "contact_hdmi_mask_time_axis": "auto",
            }
        )
        (BASE / dst_task / "e023_geometry_patch_meta.json").write_text(
            json.dumps(
                {
                    "variant": variant_row["variant"],
                    "source_variant": source_variant,
                    "source_derived_task": source_meta["derived_task"],
                    "patch_mode": variant_row["patch_mode"],
                    "disabled_leg_object_pairs": disabled,
                    "shrunk_leg_geoms": shrunk,
                    "lowerbody_radius": variant_row["lowerbody_radius"],
                    "foot_radius": variant_row["foot_radius"],
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        rows.append(meta)
        print(
            f"{variant_row['variant']}: task={dst_task} patch={variant_row['patch_mode']} "
            f"disabled={len(disabled)} shrunk={len(shrunk)}"
        )

    fieldnames = list(e018b_rows[next(iter(e018b_rows))].keys()) + [
        field for field in EXTRA_FIELDS if field not in e018b_rows[next(iter(e018b_rows))].keys()
    ]
    with MANIFEST.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {MANIFEST.relative_to(REPO)} ({len(rows)} variants)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    generate(force=args.force)


if __name__ == "__main__":
    main()
