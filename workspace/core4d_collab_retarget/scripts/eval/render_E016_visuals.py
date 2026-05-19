#!/usr/bin/env python3
"""Render E016 comparison videos and contact sheets.

E016 uses a soft weld to a moving mocap support body. A generic qpos-only
renderer is misleading for these scenes because the mocap anchor is not part of
qpos. This script replays the saved qpos while restoring the recorded support
proxy position to the mocap body, and uses the same front-camera ref/sim layout
as examples/run_mjwp.py.
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import imageio
import mujoco
import numpy as np


REPO = Path(__file__).resolve().parents[4]
RESULTS = REPO / "workspace/core4d_collab_retarget/results/E016"
MANIFEST = RESULTS / "manifest.tsv"
COMPARISON = RESULTS / "comparison.csv"
VIS_DIR = RESULTS / "visual"
BASE = REPO / "example_datasets/processed/core4d/unitree_g1/humanoid_object"


def flatten_mjwp(data: dict[str, np.ndarray], key: str) -> np.ndarray:
    arr = data[key]
    if arr.ndim == 3:
        return arr[:, -1, :]
    return arr


def object_qadr(model: mujoco.MjModel) -> int | None:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    if body_id < 0:
        return None
    joint_id = model.body_jntadr[body_id]
    if joint_id < 0:
        return None
    return int(model.jnt_qposadr[joint_id])


def support_mocap_id(model: mujoco.MjModel) -> int | None:
    body_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "support_weld_anchor"
    )
    if body_id < 0:
        return None
    mocap_id = int(model.body_mocapid[body_id])
    return mocap_id if mocap_id >= 0 else None


def set_state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    qpos: np.ndarray,
    *,
    mocap_id: int | None,
    support_pos: np.ndarray | None,
    support_quat: np.ndarray | None,
) -> None:
    data.qpos[: model.nq] = qpos[: model.nq]
    if mocap_id is not None and support_pos is not None:
        data.mocap_pos[mocap_id] = support_pos
    if mocap_id is not None and support_quat is not None:
        data.mocap_quat[mocap_id] = support_quat
    mujoco.mj_forward(model, data)


def update_scene(
    renderer: mujoco.Renderer,
    data: mujoco.MjData,
    options: mujoco.MjvOption | None = None,
) -> None:
    try:
        renderer.update_scene(data, "front", options)
    except Exception:
        try:
            renderer.update_scene(data, 0, options)
        except Exception:
            renderer.update_scene(data, options=options)


def put_label(image: np.ndarray, text: str) -> np.ndarray:
    out = image.copy()
    cv2.putText(
        out,
        text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (128, 128, 128),
        2,
    )
    return out


def render_comparison(
    scene: Path,
    kin_npz: Path,
    phys_npz: Path,
    out: Path,
    *,
    width: int,
    height: int,
    fps: int,
) -> None:
    model = mujoco.MjModel.from_xml_path(str(scene))
    model.vis.global_.offwidth = width
    model.vis.global_.offheight = height
    renderer = mujoco.Renderer(model, height=height, width=width)
    data_sim = mujoco.MjData(model)
    data_ref = mujoco.MjData(model)

    kin = np.load(kin_npz)
    phys = dict(np.load(phys_npz))
    qpos_ref = kin["qpos"]
    qpos_sim = flatten_mjwp(phys, "qpos")
    support_pos = (
        flatten_mjwp(phys, "support_proxy_pos")
        if "support_proxy_pos" in phys
        else None
    )

    if qpos_ref.shape[1] != model.nq or qpos_sim.shape[1] != model.nq:
        raise ValueError(
            f"qpos/model mismatch for {scene}: ref={qpos_ref.shape}, "
            f"sim={qpos_sim.shape}, model.nq={model.nq}"
        )

    qadr = object_qadr(model)
    mocap_id = support_mocap_id(model)
    options = mujoco.MjvOption()
    mujoco.mjv_defaultOption(options)
    options.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True

    frame_count = min(qpos_ref.shape[0], qpos_sim.shape[0])
    frames: list[np.ndarray] = []
    for i in range(frame_count):
        idx = min(i, support_pos.shape[0] - 1) if support_pos is not None else i
        sp = support_pos[idx] if support_pos is not None else None
        sq = qpos_ref[i, qadr + 3 : qadr + 7] if qadr is not None else None

        set_state(
            model,
            data_ref,
            qpos_ref[i],
            mocap_id=mocap_id,
            support_pos=sp,
            support_quat=sq,
        )
        update_scene(renderer, data_ref)
        ref_image = put_label(renderer.render(), "ref")

        set_state(
            model,
            data_sim,
            qpos_sim[i],
            mocap_id=mocap_id,
            support_pos=sp,
            support_quat=sq,
        )
        update_scene(renderer, data_sim, options)
        sim_image = put_label(renderer.render(), "sim")

        frames.append(np.concatenate([ref_image, sim_image], axis=1))

    renderer.close()
    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(out), frames, fps=fps, codec="libx264", quality=8)
    print(f"Video saved: {out} ({len(frames)} frames, {fps} fps)")


def read_manifest() -> dict[str, dict[str, str]]:
    with MANIFEST.open("r", encoding="utf-8", newline="") as f:
        return {row["variant"]: row for row in csv.DictReader(f, delimiter="\t")}


def read_comparison() -> dict[str, dict[str, str]]:
    if not COMPARISON.is_file():
        return {}
    with COMPARISON.open("r", encoding="utf-8", newline="") as f:
        return {row["variant"]: row for row in csv.DictReader(f)}


def run(cmd: list[str], *, env: dict[str, str] | None = None) -> None:
    print(" ".join(cmd))
    subprocess.run(cmd, cwd=REPO, env=env, check=True)


def render_variant(
    variant: str,
    row: dict[str, str],
    *,
    force: bool,
    width: int,
    height: int,
    fps: int,
) -> tuple[Path, Path]:
    task = row["derived_task"]
    scene_name = row["scene_name"]
    scene = BASE / task / f"{scene_name}.xml"
    kin = BASE / task / "0/trajectory_kinematic.npz"
    phys = RESULTS / f"{variant}.npz"
    out = VIS_DIR / f"{variant}_comparison.mp4"
    sheet_dir = VIS_DIR / f"{variant}_frames"
    sheet = sheet_dir / "sheet.jpg"

    for path in (scene, kin, phys):
        if not path.is_file():
            raise FileNotFoundError(path)

    VIS_DIR.mkdir(parents=True, exist_ok=True)
    sheet_dir.mkdir(parents=True, exist_ok=True)
    if force or not out.is_file() or out.stat().st_size == 0:
        os.environ.setdefault("MUJOCO_GL", "egl")
        render_comparison(
            scene,
            kin,
            phys,
            out,
            width=width,
            height=height,
            fps=fps,
        )
    else:
        print(f"skip existing {out}")

    run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(out),
            "-vf",
            "fps=1,scale=320:-1,tile=5x2",
            "-frames:v",
            "1",
            str(sheet),
        ]
    )
    return out, sheet


def fmt_pct(row: dict[str, str], key: str) -> str:
    try:
        return f"{float(row[key]):.1f}"
    except Exception:
        return ""


def fmt_num(row: dict[str, str], key: str, digits: int = 3) -> str:
    try:
        return f"{float(row[key]):.{digits}f}"
    except Exception:
        return ""


def write_visual_index(
    rendered: list[tuple[str, Path, Path]],
    comparison: dict[str, dict[str, str]],
) -> None:
    lines = [
        "# E016 Visual Evaluation",
        "",
        "Videos use the run_mjwp-style front-camera layout: reference qpos (left) and MJWarp output (right).",
        "",
        "| Variant | Video | Sheet | Epos m | Erot deg | contact 5cm % | deep pen % | leg % | diagnosis |",
        "|---------|-------|-------|--------|----------|--------------|------------|-------|-----------|",
    ]
    for variant, video, sheet in rendered:
        row = comparison.get(variant, {})
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{variant}`",
                    f"`{video.relative_to(REPO)}`",
                    f"`{sheet.relative_to(REPO)}`",
                    fmt_num(row, "paper_object_Epos_case_m"),
                    fmt_num(row, "paper_object_Erot_case_deg", 1),
                    fmt_pct(row, "paper_omniretarget_contact_preservation_5cm_pct"),
                    fmt_pct(
                        row,
                        "paper_omniretarget_robot_object_deep_penetration_duration_pct",
                    ),
                    fmt_pct(row, "case_window_sim_leg_box_interference_frames_pct"),
                    row.get("E016_diagnostic_class", ""),
                ]
            )
            + " |"
        )
    (VIS_DIR / "visual_eval.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("variants", nargs="*", help="Variant names. Default: all.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()

    manifest = read_manifest()
    selected = args.variants or list(manifest.keys())
    comparison = read_comparison()
    rendered: list[tuple[str, Path, Path]] = []
    for variant in selected:
        if variant not in manifest:
            raise ValueError(f"Unknown E016 variant: {variant}")
        video, sheet = render_variant(
            variant,
            manifest[variant],
            force=args.force,
            width=args.width,
            height=args.height,
            fps=args.fps,
        )
        rendered.append((variant, video, sheet))
    write_visual_index(rendered, comparison)
    print(f"Wrote {VIS_DIR / 'visual_eval.md'}")


if __name__ == "__main__":
    main()
