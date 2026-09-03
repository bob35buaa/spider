#!/usr/bin/env python3
"""E206 P7: build the noPRG and PRG scenes for every Stage2b-passing case.

Direction is the OPPOSITE of E204/E205. Those start from an E178 scene that
already carries 18N robot<->object pairs and *strip* the leg ones. A dcv3
``scene_act.xml`` carries only **2** robot<->object pairs no matter how many
collision boxes the object has -- that is finding F7 in log234/plan236, and it
is why 40 of desk007's 41 draft boxes were invisible to the robot. So E206 has
to *build the pairs up*:

  1. dcv3 ``scene_act.xml``  (5cm ball hands + installed lowgeom proxy, 2 pairs)
  2. -> rubber_hull hands              -> scene_act_E206_lowgeom_rubberHull.xml
  3. -> 2N hand pairs, 0 leg pairs     -> scene_act_E206_lowgeom_noPRG.xml
  4. -> + 16N leg pairs                -> scene_act_E206_lowgeom_PRG.xml

Both arms share step 2, so the ONLY difference between them is the 16N leg
pairs (plan236 C4, zero tolerance). That is asserted here, per case, before any
GPU time is spent -- ``stripped_signature`` must match exactly once pairs are
ignored.

Usage:
    .venv/bin/python .../build_arm_scenes.py                 # all passing cases
    ... --cases a,b --limit N --dry-run
"""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import e206_common as C  # noqa: E402
import build_nonbox_multigeom_production as E175  # noqa: E402

sys.path.insert(
    0,
    str(C.REPO / "workspace/core4d/scripts/data_construction_v3/stages/s5_handoff"),
)
import patch_hand_collision as PHC  # noqa: E402

V1_MANIFEST = (
    C.S3_DIR / "omnirt_v1/ref_fk/stage2b_manifest_omnirt_v1_ref_fk.tsv"
)
V2_MANIFEST = (
    C.S3_DIR / "omnirt_v2/ref_fk/stage2b_manifest_omnirt_v2_ref_fk.tsv"
)


def passing_cases() -> list[dict[str, str]]:
    """Stage2b-passing rows, v2 rescue winning over a v1 failure for the same case."""
    by_case: dict[str, dict[str, str]] = {}
    for manifest in (V1_MANIFEST, V2_MANIFEST):
        if not manifest.is_file():
            continue
        for row in C.read_tsv(manifest):
            if row.get("stage2b_status") == "pass":
                by_case[row["case_id"]] = row
    return [by_case[k] for k in sorted(by_case)]


def task_dir(row: dict[str, str]) -> Path:
    """Task dir for a Stage2b row.

    Rescued cases live under `dcv3_omnirt_v2_ref_fk_*`, not v1 -- so take the
    name the manifest recorded instead of rebuilding it from the case_id.
    """
    task = row.get("target_task") or f"dcv3_omnirt_v1_ref_fk_{row['case_id']}"
    return C.PROCESSED_ROOT / task


def object_geom_names(root: ET.Element) -> list[str]:
    body = E175.find_object_body(root)
    return E175.object_collision_names(body)


def add_pairs(root: ET.Element, names: list[str], *, with_legs: bool) -> int:
    """Replace every robot<->object pair with 2N hand pairs (+16N leg if asked).

    Mirrors E175.replace_robot_object_pairs, which always emits 18N; noPRG needs
    the hand half only, so the leg loop is conditional rather than forked.
    """
    contact = root.find("contact")
    if contact is None:
        contact = ET.SubElement(root, "contact")
    for pair in list(contact.findall("pair")):
        if E175.is_robot_object_pair(pair):
            contact.remove(pair)

    made = 0
    for index, name in enumerate(names):
        for hand in E175.HAND_GEOMS:
            ET.SubElement(contact, "pair", {
                "name": f"E206_{hand}_obj{index:03d}",
                "geom1": hand, "geom2": name,
                "solref": "0.008 1", "friction": "2 1", "condim": "4",
            })
            made += 1
        if not with_legs:
            continue
        for leg in E175.LOWER_BODY_GEOMS:
            ET.SubElement(contact, "pair", {
                "name": f"E206_{leg}_obj{index:03d}",
                "geom1": leg, "geom2": name,
                "solref": "0.008 1", "margin": "0", "gap": "0", "condim": "1",
            })
            made += 1
    return made


def write_scene(root: ET.Element, out: Path) -> Path:
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(out, encoding="unicode", xml_declaration=False)
    return out


def check_compiled(scene: Path, *, expect_pairs: int, expect_boxes: int) -> dict[str, Any]:
    """Compile the scene and assert the contract the CEM run depends on."""
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(scene))
    gids = [
        g for g in range(model.ngeom)
        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "")
        .startswith("object_collision")
    ]
    non_box = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g)
        for g in gids if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_BOX
    ]
    if non_box:
        # spider/config.py:69-79 union mode is fail-closed on non-box geoms.
        raise AssertionError(f"{scene.name}: non-box object collision geoms {non_box}")
    if len(gids) != expect_boxes:
        raise AssertionError(
            f"{scene.name}: {len(gids)} object collision geoms != {expect_boxes}")
    object_names = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) for g in gids
    }
    robot = set(E175.ROBOT_OBJECT_GEOMS)
    pairs = 0
    for p in range(model.npair):
        n1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, model.pair_geom1[p]) or ""
        n2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, model.pair_geom2[p]) or ""
        if (n1 in robot and n2 in object_names) or (n2 in robot and n1 in object_names):
            pairs += 1
    if pairs != expect_pairs:
        raise AssertionError(
            f"{scene.name}: compiled {pairs} robot-object pairs != {expect_pairs}")
    for side in ("lh", "rh"):
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, side)
        if gid < 0 or model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_MESH:
            raise AssertionError(f"{scene.name}: {side} is not a mesh geom")
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    if bid >= 0 and float(model.body_gravcomp[bid]) != 0.0:
        raise AssertionError(f"{scene.name}: object gravcomp != 0 (no G1 in E206)")
    return {
        "nq": int(model.nq), "nv": int(model.nv), "nu": int(model.nu),
        "object_geoms": len(gids), "robot_object_pairs": pairs,
    }


def signature_without_pairs(scene: Path) -> Any:
    """stripped_signature with the <contact> block removed -- the C4 single-variable check."""
    root = ET.parse(scene).getroot()
    clone = copy.deepcopy(root)
    for contact in list(clone.findall("contact")):
        clone.remove(contact)
    return ET.tostring(clone, encoding="unicode")


def build_one(row: dict[str, str], *, dry_run: bool) -> dict[str, Any]:
    case_id = row["case_id"]
    out: dict[str, Any] = {"case_id": case_id, "object_key": row.get("object_key", "")}
    tdir = task_dir(row)
    out["target_task"] = tdir.name
    base = tdir / "scene_act.xml"
    if not base.is_file():
        out["status"] = "skip_missing_scene_act"
        return out

    n_boxes = len(object_geom_names(ET.parse(base).getroot()))
    expected = C.expected_pair_counts(n_boxes)
    out.update({"object_geom_count": n_boxes,
                "expected_pairs_noprg": expected["noprg"],
                "expected_pairs_prg": expected["prg"]})
    if dry_run:
        out["status"] = "dry_run"
        return out

    # 1. shared rubber-hull middle state -- both arms derive from this one file,
    #    so hand geometry can never become a second variable (plan236 R7).
    PHC.patch_scene(
        base_scene_act=base, out_dir=tdir, case_id=case_id,
        hand_collision_variant_id="rubber_hull",
        scene_name=C.SCENE_RUBBER_HULL, install_dir=None, repo=C.REPO,
    )
    hull = tdir / f"{C.SCENE_RUBBER_HULL}.xml"

    scenes = {}
    for arm, with_legs in (("noprg", False), ("prg", True)):
        root = ET.parse(hull).getroot()
        names = object_geom_names(root)
        made = add_pairs(root, names, with_legs=with_legs)
        if made != expected[arm]:
            raise AssertionError(f"{case_id}/{arm}: wrote {made} pairs != {expected[arm]}")
        scenes[arm] = write_scene(root, tdir / f"{C.SCENE_BY_ARM[arm]}.xml")

    # C4: ignoring <contact>, the two arms must be byte-identical.
    if signature_without_pairs(scenes["noprg"]) != signature_without_pairs(scenes["prg"]):
        raise AssertionError(f"{case_id}: noPRG/PRG differ outside the contact block")

    for arm in ("noprg", "prg"):
        out[f"{arm}_compiled"] = check_compiled(
            scenes[arm], expect_pairs=expected[arm], expect_boxes=n_boxes)
        out[f"{arm}_scene"] = str(scenes[arm].relative_to(C.REPO))
    out["status"] = "built"
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="", help="comma-separated case_ids (default: all passing)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=C.S5_DIR / "arm_scenes")
    ap.add_argument("--snapshot", action="store_true",
                    help="run snapshot_scenes.sh over the built task dirs (rule 7)")
    args = ap.parse_args()

    rows = passing_cases()
    if args.cases:
        want = {c.strip() for c in args.cases.split(",") if c.strip()}
        rows = [r for r in rows if r["case_id"] in want]
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit("no Stage2b-passing cases found; run S3 first")

    results, failures = [], []
    for i, row in enumerate(rows, 1):
        try:
            res = build_one(row, dry_run=args.dry_run)
        except Exception as exc:  # noqa: BLE001 - a bad case must not kill the batch
            res = {"case_id": row["case_id"], "object_key": row.get("object_key", ""),
                   "status": f"fail_{type(exc).__name__}", "error": str(exc)}
            failures.append(res)
        results.append(res)
        print(f"[{i:02d}/{len(rows)}] {res['case_id']}: {res['status']} "
              f"boxes={res.get('object_geom_count','?')} "
              f"pairs={res.get('expected_pairs_noprg','?')}/{res.get('expected_pairs_prg','?')}",
              flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "n_cases": len(results),
        "built": sum(1 for r in results if r["status"] == "built"),
        "failures": failures,
        "cases": results,
    }
    (args.out_dir / "arm_scene_build.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    flat = [{k: v for k, v in r.items() if not isinstance(v, dict)} for r in results]
    C.write_tsv(args.out_dir / "arm_scene_build.tsv", flat)

    if args.snapshot and not args.dry_run:
        tasks = [r["target_task"] for r in results if r["status"] == "built"]
        if tasks:
            subprocess.run(
                ["bash", "workspace/core4d/scripts/convert/snapshot_scenes.sh",
                 C.EXP_ID, *tasks], cwd=C.REPO, check=True)

    print(f"\nbuilt={payload['built']}/{len(results)} failures={len(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
