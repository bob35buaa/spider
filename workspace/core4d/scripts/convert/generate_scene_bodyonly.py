"""Generate scene_bodyonly.xml: disable object collision pairs for pure body retargeting.

Removes hand-object and object-floor contact pairs so the robot ignores the object.
Object remains in scene (for visualization) but is a ghost — no physics interaction.
"""

import xml.etree.ElementTree as ET
from pathlib import Path


CASES = {
    "box025_person1": "example_datasets/processed/core4d/unitree_g1/humanoid_object/box025_person1/scene.xml",
    "bucket010_person1": "example_datasets/processed/core4d/unitree_g1/humanoid_object/bucket010_person1/scene.xml",
    "chair022_person1": "example_datasets/processed/core4d/unitree_g1/humanoid_object/chair022_person1/scene.xml",
    "desk005_person2": "example_datasets/processed/core4d/unitree_g1/humanoid_object/desk005_person2/scene.xml",
}


def generate_scene_bodyonly(input_path: str) -> str:
    """Remove object collision pairs from scene XML."""
    tree = ET.parse(input_path)
    root = tree.getroot()

    # Find contact section and remove object-related pairs
    contact = root.find("contact")
    if contact is not None:
        pairs_to_remove = []
        for pair in contact.findall("pair"):
            name = pair.get("name", "")
            # Remove any pair involving "object"
            geom1 = pair.get("geom1", "")
            geom2 = pair.get("geom2", "")
            if "object" in name or "object" in geom1 or "object" in geom2:
                pairs_to_remove.append(pair)
        for pair in pairs_to_remove:
            contact.remove(pair)

    # Also set object geom to contype=0 conaffinity=0 (ghost)
    worldbody = root.find("worldbody")
    for geom in worldbody.iter("geom"):
        if "object" in (geom.get("name") or ""):
            geom.set("contype", "0")
            geom.set("conaffinity", "0")

    output_path = input_path.replace("scene.xml", "scene_bodyonly.xml")
    tree.write(output_path, xml_declaration=False)
    return output_path


def main():
    for case, path in CASES.items():
        if not Path(path).exists():
            print(f"SKIP {case}")
            continue
        out = generate_scene_bodyonly(path)
        print(f"Generated: {out}")


if __name__ == "__main__":
    main()
