#!/usr/bin/env python3
"""Create E089 derived tasks by reusing E083's leg+upper-body collision pair patcher.

Reads workspace/core4d/scripts/E089/variants.tsv (TSV with header comment + columns:
variant, source_task, derived_task, mask_source_dir, mask_slug, person_idx, split, role)
and copies + patches each source_task to produce a derived_task with the same set of
collision pairs as the E083 derivations.
"""
from __future__ import annotations
import sys
from pathlib import Path

# Reuse the E083 implementation directly to keep pair set consistent
HERE = Path(__file__).resolve()
REPO = HERE.parents[4]
sys.path.insert(0, str(REPO / "workspace/core4d/scripts/E083"))

import create_upperobj_cases as up  # type: ignore  # noqa: E402

VARIANTS = REPO / "workspace/core4d/scripts/E089/variants.tsv"


def main() -> None:
    force = "--force" in sys.argv
    # Patch up.copy_case to skip optional files (older box021_person1 lacks meta)
    OPTIONAL = {"scene_act_meta.json", "task_info.json"}
    orig_copy = up.copy_case
    import json, shutil
    def copy_case(source_task: str, derived_task: str, *, force: bool):
        src = up.BASE / source_task
        dst = up.BASE / derived_task
        if not src.is_dir():
            raise FileNotFoundError(src)
        if dst.exists():
            if not force:
                print(f"[SKIP] derived task exists: {dst.relative_to(up.REPO)}")
                return dst
            shutil.rmtree(dst)
        (dst / "0").mkdir(parents=True, exist_ok=True)
        for rel in ["scene.xml", "scene_act.xml", "scene_act_meta.json",
                    "task_info.json", "0/trajectory_kinematic.npz"]:
            src_file = src / rel
            if not src_file.is_file():
                if Path(rel).name in OPTIONAL:
                    # Stub it
                    dst_file = dst / rel
                    dst_file.parent.mkdir(parents=True, exist_ok=True)
                    if rel.endswith("scene_act_meta.json"):
                        dst_file.write_text(json.dumps({"euler_convention": "XYZ"}))
                    else:  # task_info.json
                        dst_file.write_text(json.dumps({
                            "task": derived_task, "source_task": source_task,
                            "e089_note": "Derived from older box021_person1 lacking meta; XYZ assumed.",
                        }))
                    continue
                raise FileNotFoundError(src_file)
            dst_file = dst / rel
            dst_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_file, dst_file)
        up.patch_task_info(dst / "task_info.json", source_task, derived_task)
        return dst

    for row in up.read_variants(VARIANTS):
        dst = copy_case(row["source_task"], row["derived_task"], force=force)
        added_leg, added_upper = up.patch_scene_act(
            dst / "scene_act.xml", row["source_task"], row["derived_task"]
        )
        print(
            f"{row['source_task']} -> {row['derived_task']}: "
            f"leg_pairs={len(added_leg)} upper_pairs={len(added_upper)} "
            f"path={dst.relative_to(REPO)}"
        )


if __name__ == "__main__":
    main()
