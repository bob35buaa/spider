#!/usr/bin/env python3
"""Build the E204 (noPRG) and E205 (G1A2) per-case scenes for the 27 E178 buckets.

Pipeline per case (idempotent, run on the machine that has the task dirs+meshes):

  1. (Re)generate the canonical E178 contact-aligned scene via the E178 production
     ``build_scene`` -> ``scene_act_E178_contactAlignedTop.xml`` in the task dir.
     This guarantees byte-parity with the scene E178's own CEM used (so the PRG
     middle arm in the 3-arm comparison stays clean). Use --reuse-existing-e178-scene
     to skip regeneration when the file is already present.
  2. E205 gravcomp sidecar: object body ``gravcomp`` 0->1, single-variable diff
     (``assert_gravcomp_diff``). Everything else identical to the E178 scene.
  3. E204 noPRG scene: remove the leg<->object PRG pairs (geom in LOWER_BODY_GEOMS),
     keep the 5-segment object proxy + hand-object pairs. Assert 0 leg pairs remain,
     hand pairs == 2xN, object gravcomp stays 0.

Also snapshots the derived scenes into results/E204 and results/E205 (rule 10b)
and writes a per-case build manifest + the task-name list used by the launcher.

Usage:
    .venv/bin/python workspace/core4d/scripts/experiments/E204_E205/build_arm_scenes.py
    ... --reuse-existing-e178-scene        # skip step 1 if E178 scene already on disk
    ... --limit N                          # first N cases (smoke)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import e204e205_common as C

REPO = C.REPO


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ensure_e178_scene(source: dict, proxy_cache: dict, *, reuse: bool) -> Path:
    """Return the canonical E178 contactAlignedTop scene path, regenerating if needed."""
    case_id = source["case_id"]
    out = C.e178_scene_path(case_id)
    if reuse and out.is_file():
        return out
    _target, trajectory, _mask = C.base.local_authorities(source)
    scene, _rows = C.production.build_scene(source, proxy_cache, trajectory, overwrite=True)
    built = C.base.repo_path(scene["scene_act"])
    if built.resolve() != out.resolve():
        raise AssertionError(f"E178 scene path drift: {built} != {out}")
    return out


def build_e205_gravcomp(e178_scene: Path) -> Path:
    """E205: object gravcomp 0->1 single-variable diff off the E178 scene."""
    out = e178_scene.with_name(f"{C.E205_SCENE}.xml")
    tree = ET.parse(e178_scene)
    root = tree.getroot()
    objs = [b for b in root.iter("body") if b.get("name") == "object"]
    if len(objs) != 1:
        raise ValueError(f"expected exactly one object body in {e178_scene}, found {len(objs)}")
    if objs[0].get("gravcomp") not in (None, "0", "0.0"):
        raise ValueError(f"object already has gravcomp in {e178_scene}: {objs[0].get('gravcomp')}")
    objs[0].set("gravcomp", "1")
    ET.indent(tree, space="  ")
    tree.write(out, encoding="utf-8", xml_declaration=True)
    C.assert_gravcomp_diff(e178_scene, out)  # single-variable guarantee (reused E198/E200)
    return out


def build_e204_noprg(e178_scene: Path) -> tuple[Path, int, int]:
    """E204: strip leg<->object PRG pairs; keep 5-seg proxy + hand pairs, gravcomp 0."""
    out = e178_scene.with_name(f"{C.E204_SCENE}.xml")
    tree = ET.parse(e178_scene)
    root = tree.getroot()
    contact = root.find("contact")
    if contact is None:
        raise ValueError(f"no <contact> block in {e178_scene}")
    leg = set(C.LOWER_BODY_GEOMS)
    hand = set(C.HAND_GEOMS)

    def _leg_object(p: ET.Element) -> bool:
        # canonical robot<->object pair (one side is object_collision*) AND leg side.
        return C.base.is_robot_object_pair(p) and (
            p.get("geom1") in leg or p.get("geom2") in leg)

    def _hand_object(p: ET.Element) -> bool:
        return C.base.is_robot_object_pair(p) and (
            p.get("geom1") in hand or p.get("geom2") in hand)

    removed = 0
    for pair in list(contact.findall("pair")):
        if _leg_object(pair):
            contact.remove(pair)
            removed += 1
    # asserts: no leg-object pair survives; hand-object pairs preserved; gravcomp 0.
    # (leg-floor / leg self-collision pairs are NOT touched -- only leg<->object.)
    leg_left = [p for p in contact.findall("pair") if _leg_object(p)]
    if leg_left:
        raise AssertionError(f"leg-object pairs survived noPRG strip: {len(leg_left)}")
    hand_pairs = [p for p in contact.findall("pair") if _hand_object(p)]
    objs = [b for b in root.iter("body") if b.get("name") == "object"]
    if objs and objs[0].get("gravcomp") not in (None, "0", "0.0"):
        raise ValueError(f"noPRG scene must keep object gravcomp 0: {e178_scene}")
    ET.indent(tree, space="  ")
    tree.write(out, encoding="utf-8", xml_declaration=True)
    return out, removed, len(hand_pairs)


def snapshot(exp_id: str, task_names: list[str]) -> None:
    """rule 10b: copy every scene*.xml + meta for each task dir into results/<exp>."""
    if not task_names:
        return
    cmd = ["bash", "workspace/core4d/scripts/convert/snapshot_scenes.sh", exp_id, *task_names]
    subprocess.run(cmd, cwd=REPO, check=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reuse-existing-e178-scene", action="store_true",
                    help="skip E178 scene regeneration when already on disk")
    ap.add_argument("--limit", type=int, default=0, help="first N cases only (smoke)")
    ap.add_argument("--no-snapshot", action="store_true", help="skip rule-10b snapshot")
    args = ap.parse_args()

    sources = C.load_sources()
    if args.limit:
        sources = sources[: args.limit]

    proxy_cache: dict = {}
    rows = []
    task_names = []
    for i, source in enumerate(sources, 1):
        case_id = source["case_id"]
        print(f"[{i}/{len(sources)}] {case_id}", flush=True)
        e178 = ensure_e178_scene(source, proxy_cache, reuse=args.reuse_existing_e178_scene)
        e205, = (build_e205_gravcomp(e178),)
        e204, removed, hand_n = build_e204_noprg(e178)
        expected_boxes = C.production.EXPECTED_BOXES_BY_OBJECT[source["object_key"]]
        # E178 has 18 robot geoms x N boxes; noPRG strips 16 leg geoms x N.
        if removed != len(C.LOWER_BODY_GEOMS) * expected_boxes:
            raise AssertionError(
                f"{case_id}: removed {removed} leg pairs != {len(C.LOWER_BODY_GEOMS)}x{expected_boxes}")
        if hand_n != len(C.HAND_GEOMS) * expected_boxes:
            raise AssertionError(
                f"{case_id}: {hand_n} hand pairs != {len(C.HAND_GEOMS)}x{expected_boxes}")
        task_names.append(C.task_of(case_id))
        rows.append({
            "case_id": case_id, "object_key": source["object_key"],
            "expected_boxes": expected_boxes,
            "e178_scene": str(e178.relative_to(REPO)), "e178_sha256": _sha256(e178),
            "e204_scene": str(e204.relative_to(REPO)), "e204_sha256": _sha256(e204),
            "e204_leg_pairs_removed": removed, "e204_hand_pairs": hand_n,
            "e205_scene": str(e205.relative_to(REPO)), "e205_sha256": _sha256(e205),
        })
        print(f"   E204 noPRG: -{removed} leg pairs, {hand_n} hand pairs kept | "
              f"E205 gravcomp single-var OK", flush=True)

    for arm in C.ARMS:
        man_dir = C.RESULTS[arm] / "s6_downstream/manifests"
        man_dir.mkdir(parents=True, exist_ok=True)
        (man_dir / "arm_scene_build.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
        (C.RESULTS[arm] / "cases.txt").write_text(
            "\n".join(task_names) + "\n", encoding="utf-8")

    if not args.no_snapshot:
        snapshot("E204", task_names)
        snapshot("E205", task_names)

    print(f"\nbuilt {len(rows)} cases x 2 arms. scene build manifest + cases.txt written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
