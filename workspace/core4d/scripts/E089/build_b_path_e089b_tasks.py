#!/usr/bin/env python3
"""Build E089B SPIDER-runnable tasks from B-path repaired _btop staged data.

The B subagent staged repaired trajectory_kinematic.npz under
`example_datasets/.../d003_box021_*_btop/` but those only have a bare
`scene.xml` (no scene_act, no e083 collision pairs). This script
takes each top-2 _btop task and creates a `*_btop_upperobj_e089b`
task with:
- scene.xml + scene_act.xml from `d003_box021_*_upperobj_e083` template,
  with object pos/quat patched to match the _btop case
- e083 leg+upper-body collision pairs (already in template's scene_act)
- 0/trajectory_kinematic.npz copied from the _btop case
"""
from __future__ import annotations
import json
import re
import shutil
from pathlib import Path
import sys

REPO = Path("/mnt/ali-sh-1/usr/xiayibo/work_dir/embodied/spider")
TASKS = REPO / "example_datasets/processed/core4d/unitree_g1/humanoid_object"

# Top-2 from B subagent's ranking_top10 (score 6/6 + best metrics)
# Use real existing _btop dirs that have npz
TOP_CASES = [
    ("d003_box021_20231018_031_p2_btop", "d003_box021_20231018_031_p2_upperobj_e083"),
    ("d003_box021_20231020_020_p1_btop", "d003_box021_20231020_020_p1_upperobj_e083"),
]

def find_template(source_btop: str):
    # First try the corresponding _upperobj_e083 task
    cand = TASKS / source_btop.replace("_btop", "_upperobj_e083")
    if cand.exists():
        return cand
    # Else use the canonical e083 template
    return TASKS / "d003_box021_20231018_029_p2_upperobj_e083"


def patch_object_xml(text: str, new_pos: str, new_quat: str) -> str:
    # Replace `<body name="object" pos="..." quat="...">` with new values
    pattern = r'(<body name="object")(\s+pos="[^"]+")?(\s+quat="[^"]+")?'
    def repl(m):
        return f'{m.group(1)} pos="{new_pos}" quat="{new_quat}"'
    return re.sub(pattern, repl, text, count=1)


def main(force: bool = False):
    created = []
    for btop, template_name in TOP_CASES:
        src_btop = TASKS / btop
        if not (src_btop / "0/trajectory_kinematic.npz").exists():
            print(f"[SKIP] no trajectory for {btop}")
            continue
        # Find template
        if not (TASKS / template_name).exists():
            template_dir = TASKS / "d003_box021_20231018_029_p2_upperobj_e083"
        else:
            template_dir = TASKS / template_name
        dst_name = btop + "_upperobj_e089b"
        dst = TASKS / dst_name
        if dst.exists():
            if not force:
                print(f"[SKIP exists] {dst_name}")
                continue
            shutil.rmtree(dst)
        (dst / "0").mkdir(parents=True, exist_ok=True)

        # Extract pos/quat from btop scene.xml
        btop_scene = (src_btop / "scene.xml").read_text()
        m = re.search(r'<body name="object"\s+pos="([^"]+)"\s+quat="([^"]+)"', btop_scene)
        if not m:
            print(f"[ERR] cannot parse object pos/quat from {btop}/scene.xml")
            continue
        new_pos, new_quat = m.group(1), m.group(2)

        # Copy scene.xml and scene_act.xml from template, patch object pos/quat
        for fname in ["scene.xml", "scene_act.xml"]:
            text = (template_dir / fname).read_text()
            text = patch_object_xml(text, new_pos, new_quat)
            (dst / fname).write_text(text)

        # Copy meta files
        for fname in ["scene_act_meta.json", "task_info.json", "upper_object_collision_meta.json"]:
            src_file = template_dir / fname
            if src_file.exists():
                shutil.copy2(src_file, dst / fname)
        # Patch task_info.json
        ti_path = dst / "task_info.json"
        if ti_path.exists():
            data = json.loads(ti_path.read_text())
            data["task"] = dst_name
            data["b_path_source"] = btop
            data["e089b_note"] = "B path top-face repaired npz + template e083 collision pairs."
            ti_path.write_text(json.dumps(data, indent=2, sort_keys=True))

        # Copy trajectory_kinematic.npz from _btop
        shutil.copy2(src_btop / "0/trajectory_kinematic.npz", dst / "0/trajectory_kinematic.npz")
        created.append(dst_name)
        print(f"created {dst_name} ← btop={btop}, template={template_dir.name}, pos={new_pos} quat={new_quat}")

    if created:
        # Print a variants.tsv stub
        out_tsv = REPO / "workspace/core4d/scripts/E089/variants_B4.tsv"
        out_tsv.write_text("# E089 B4 SPIDER smoke on B-path repaired top cases\n"
                           "# variant\tsource_task\tderived_task\tmask_source_dir\tmask_slug\tperson_idx\tsplit\trole\n"
                           + "\n".join(
                               f"E089B4_{c}\t{c.replace('_upperobj_e089b','')}\t{c}\t-\t-\t1\tlocal-gpu0\tmain"
                               for c in created
                           ) + "\n")
        print(f"\nvariants tsv: {out_tsv}")


if __name__ == "__main__":
    main(force="--force" in sys.argv)
