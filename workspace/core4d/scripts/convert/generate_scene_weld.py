"""Generate scene_weld.xml: adds a mocap body + soft weld equality constraint for the object.

The weld constraint makes MuJoCo's solver handle position+orientation coupling internally,
avoiding the feedback loop that xfrc_applied torque + position spring creates.

Usage:
    python workspace/core4d/scripts/convert/generate_scene_weld.py
"""

import xml.etree.ElementTree as ET
from pathlib import Path

CASES = {
    "box025_person1": "example_datasets/processed/core4d/unitree_g1/humanoid_object/box025_person1/scene.xml",
    "bucket010_person1": "example_datasets/processed/core4d/unitree_g1/humanoid_object/bucket010_person1/scene.xml",
    "desk005_person1": "example_datasets/processed/core4d/unitree_g1/humanoid_object/desk005_person1/scene.xml",
    "chair022_person1": "example_datasets/processed/core4d/unitree_g1/humanoid_object/chair022_person1/scene.xml",
}

# Soft weld parameters:
# solref: [timeconst, dampratio] — negative values = spring-damper
# -timeconst: ~0.1s response time (larger = softer)
# -dampratio: 1.0 = critically damped
# solimp: [dmin, dmax, width] — constraint impedance
SOLREF = "-0.1 -1.0"  # soft spring: 0.1s time constant, critically damped
SOLIMP = "0.9 0.95 0.001"  # high impedance (stiff but not rigid)


def generate_scene_weld(input_path: str) -> str:
    """Add mocap body + weld constraint to scene XML."""
    tree = ET.parse(input_path)
    root = tree.getroot()

    # Find object body to get its initial position
    worldbody = root.find("worldbody")
    obj_body = worldbody.find(".//body[@name='object']")
    if obj_body is None:
        raise ValueError(f"No 'object' body in {input_path}")

    obj_pos = obj_body.get("pos", "0 0 0")

    # Add mocap body BEFORE the object body (as sibling)
    mocap_body = ET.SubElement(worldbody, "body")
    mocap_body.set("name", "object_target")
    mocap_body.set("mocap", "true")
    mocap_body.set("pos", obj_pos)
    # Visual marker (transparent)
    mocap_geom = ET.SubElement(mocap_body, "geom")
    mocap_geom.set("type", "box")
    mocap_geom.set("size", "0.05 0.05 0.05")
    mocap_geom.set("rgba", "1 0 0 0.3")
    mocap_geom.set("contype", "0")
    mocap_geom.set("conaffinity", "0")
    mocap_geom.set("group", "4")

    # Add equality section with weld constraint
    equality = root.find("equality")
    if equality is None:
        equality = ET.SubElement(root, "equality")

    weld = ET.SubElement(equality, "weld")
    weld.set("name", "object_weld")
    weld.set("body1", "object")
    weld.set("body2", "object_target")
    weld.set("solref", SOLREF)
    weld.set("solimp", SOLIMP)
    # relpose="0 0 0 1 0 0 0" means bodies should be aligned (no offset)
    weld.set("relpose", "0 0 0 1 0 0 0")

    # Write output
    output_path = input_path.replace("scene.xml", "scene_weld.xml")
    tree.write(output_path, xml_declaration=False)

    # Add XML declaration manually (MuJoCo prefers it)
    with open(output_path, "r") as f:
        content = f.read()
    with open(output_path, "w") as f:
        f.write(content)

    return output_path


def main():
    for case, path in CASES.items():
        if not Path(path).exists():
            print(f"SKIP {case}: {path} not found")
            continue
        out = generate_scene_weld(path)
        print(f"Generated: {out}")


if __name__ == "__main__":
    main()
