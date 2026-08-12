#!/usr/bin/env python3
"""Direct-entry tests for the E196 scene-act reference contract."""

from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path

import mujoco

from spider.simulators.scene_act_reference import resolve_scene_act_reference


REPO = Path(__file__).resolve().parents[5]
REAL_MATCH_CASE = "box001_20231003_1_039_p2"
REAL_MATCH_SCENE = REPO / (
    "example_datasets/processed/core4d/unitree_g1/humanoid_object/"
    "dcv3_omnirt_v1_ref_fk_box001_20231003_1_039_p2/"
    "scene_act_E194_G1_expansion_rubberHull_PRG_gravcomp.xml"
)
REAL_MATCH_AUDIT = REPO / (
    "workspace/core4d/results/E194/s6_downstream/eval/full_g1_expansion/"
    "e194_g1_object_orientation_reference_conversion_audit.tsv"
)


def scene_xml(order: str) -> str:
    axes = {"X": "1 0 0", "Y": "0 1 0", "Z": "0 0 1"}
    hinges = "\n".join(
        f'<joint name="object_rot_{axis.lower()}" type="hinge" axis="{axes[axis]}"/>'
        for axis in order
    )
    return f"""<mujoco model="e196_reference_test">
  <worldbody>
    <body name="object">
      <joint name="object_pos_x" type="slide" axis="1 0 0"/>
      <joint name="object_pos_y" type="slide" axis="0 1 0"/>
      <joint name="object_pos_z" type="slide" axis="0 0 1"/>
      {hinges}
      <geom type="sphere" size="0.1" mass="1"/>
    </body>
  </worldbody>
</mujoco>
"""


def build(tmp: Path, order: str, payload: object | None) -> tuple[Path, mujoco.MjModel]:
    xml = tmp / "scene_act.xml"
    xml.write_text(scene_xml(order), encoding="utf-8")
    if payload is not None:
        (tmp / "scene_act_meta.json").write_text(
            json.dumps(payload) + "\n", encoding="utf-8"
        )
    return xml, mujoco.MjModel.from_xml_path(str(xml))


def expect_failure(error_type: type[BaseException], call, contains: str) -> None:
    try:
        call()
    except error_type as exc:
        assert contains in str(exc), str(exc)
    else:
        raise AssertionError(f"expected {error_type.__name__}: {contains}")


def test_valid_conventions() -> None:
    for order in ("XYZ", "XZY", "ZYX"):
        with tempfile.TemporaryDirectory(prefix="e196_reference_valid_") as raw:
            xml, model = build(Path(raw), order, {"euler_convention": order})
            resolved = resolve_scene_act_reference(xml, model, emit_log=False)
            assert resolved.convention == order
            assert resolved.xml_axis_sequence == order
            assert len(resolved.meta_sha256) == 64


def test_missing_and_invalid_metadata() -> None:
    with tempfile.TemporaryDirectory(prefix="e196_reference_invalid_") as raw:
        tmp = Path(raw)
        xml, model = build(tmp, "XZY", None)
        expect_failure(
            FileNotFoundError,
            lambda: resolve_scene_act_reference(xml, model, emit_log=False),
            "no Euler fallback",
        )
        (tmp / "scene_act_meta.json").write_text("{bad json", encoding="utf-8")
        expect_failure(
            ValueError,
            lambda: resolve_scene_act_reference(xml, model, emit_log=False),
            "invalid scene-act reference metadata",
        )
        for payload, message in (
            ({}, "must be one of"),
            ({"euler_convention": "xyz"}, "must be one of"),
            ({"euler_convention": "XXZ"}, "must be one of"),
            ({"euler_convention": "XYZ"}, "disagrees with compiled"),
        ):
            (tmp / "scene_act_meta.json").write_text(
                json.dumps(payload) + "\n", encoding="utf-8"
            )
            expect_failure(
                ValueError,
                lambda: resolve_scene_act_reference(xml, model, emit_log=False),
                message,
            )


def test_invalid_compiled_axes() -> None:
    with tempfile.TemporaryDirectory(prefix="e196_reference_axes_") as raw:
        xml, model = build(Path(raw), "XZY", {"euler_convention": "XZY"})
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
        start = int(model.body_jntadr[body])
        hinge = start + 3
        model.jnt_axis[hinge] = [-1.0, 0.0, 0.0]
        expect_failure(
            ValueError,
            lambda: resolve_scene_act_reference(xml, model, emit_log=False),
            "positive XYZ unit vectors",
        )
        model.jnt_axis[hinge] = [2.0, 0.0, 0.0]
        expect_failure(
            ValueError,
            lambda: resolve_scene_act_reference(xml, model, emit_log=False),
            "positive XYZ unit vectors",
        )
        model.jnt_axis[hinge] = [0.0, 0.0, 1.0]
        expect_failure(
            ValueError,
            lambda: resolve_scene_act_reference(xml, model, emit_log=False),
            "three unique XYZ hinges",
        )


def test_real_e194_match_case() -> None:
    assert REAL_MATCH_SCENE.is_file(), REAL_MATCH_SCENE
    assert REAL_MATCH_SCENE.with_name("scene_act_meta.json").is_file()
    model = mujoco.MjModel.from_xml_path(str(REAL_MATCH_SCENE))
    resolved = resolve_scene_act_reference(REAL_MATCH_SCENE, model, emit_log=False)
    with REAL_MATCH_AUDIT.open(encoding="utf-8", newline="") as stream:
        authority = next(
            row for row in csv.DictReader(stream, delimiter="\t")
            if row["case_id"] == REAL_MATCH_CASE
        )
    assert authority["runtime_convention_matches_xml_axes"].lower() == "true"
    assert resolved.convention == authority["runtime_euler_convention"] == "XZY"
    assert resolved.xml_axis_sequence == authority["xml_hinge_axis_sequence"] == "XZY"


def main() -> None:
    test_valid_conventions()
    test_missing_and_invalid_metadata()
    test_invalid_compiled_axes()
    test_real_e194_match_case()
    print("PASS: E196 scene-act reference contract")


if __name__ == "__main__":
    main()
