"""Converter for CORE4D holosoma-retargeted data to SPIDER format.

Reads the kinematic retargeting output from holosoma (qpos(T,43), fps=30)
and converts it to SPIDER's trajectory_kinematic.npz format with qvel, ctrl,
contact, and contact_pos computed via MuJoCo forward kinematics.

Usage:
    uv run spider/process_datasets/core4d.py \
        --source-npz /path/to/holosoma/retarget.npz \
        --task box025_person1 --data-id 0
"""

import os

import imageio
import mujoco
import numpy as np
import tyro
from loop_rate_limiters import RateLimiter

from spider import ROOT
from spider.io import get_processed_data_dir
from spider.mujoco_utils import get_viewer


def main(
    source_npz: str = "",
    dataset_dir: str = f"{ROOT}/../example_datasets",
    dataset_name: str = "core4d",
    robot_type: str = "unitree_g1",
    embodiment_type: str = "humanoid_object",
    task: str = "box025_person1",
    data_id: int = 0,
    show_viewer: bool = False,
    save_video: bool = True,
    enable_rate_limiter: bool = False,
    contact_detection_mode: str = "one",
):
    """Convert CORE4D holosoma-retargeted data to SPIDER format.

    Args:
        source_npz: Path to holosoma retargeted .npz file.
        dataset_dir: SPIDER example_datasets directory.
        dataset_name: Dataset name (core4d).
        robot_type: Robot type (unitree_g1).
        embodiment_type: Embodiment type (humanoid_object).
        task: Task name (e.g., box025_person1).
        data_id: Data ID.
        show_viewer: Whether to show the MuJoCo viewer.
        save_video: Whether to save a visualization video.
        enable_rate_limiter: Whether to enable playback rate limiting.
        contact_detection_mode: Contact detection mode.
            "one": Always enable contact (default for cooperative carrying).
            "auto": Detect from hand-object distance.
    """
    dataset_dir = os.path.abspath(dataset_dir)
    processed_dir = get_processed_data_dir(
        dataset_dir=dataset_dir,
        dataset_name=dataset_name,
        robot_type=robot_type,
        embodiment_type=embodiment_type,
        task=task,
        data_id=data_id,
    )
    print(f"Processed directory: {processed_dir}")
    os.makedirs(processed_dir, exist_ok=True)

    # Load holosoma retargeted data
    if not source_npz:
        raise ValueError("--source-npz is required")
    if not os.path.exists(source_npz):
        raise FileNotFoundError(f"Source NPZ not found: {source_npz}")

    src = np.load(source_npz, allow_pickle=True)
    qpos = src["qpos"].astype(np.float64)
    fps = int(src["fps"])
    print(f"Loaded: qpos shape={qpos.shape}, fps={fps}")
    assert qpos.shape[1] == 43, f"Expected 43 DOF, got {qpos.shape[1]}"

    # Load scene XML
    scene_dir = os.path.dirname(processed_dir)
    scene_path = f"{scene_dir}/scene.xml"
    if not os.path.exists(scene_path):
        raise FileNotFoundError(f"Scene XML not found: {scene_path}")

    mj_model = mujoco.MjModel.from_xml_path(scene_path)
    mj_data = mujoco.MjData(mj_model)
    assert mj_model.nq == 43, f"Expected nq=43, got {mj_model.nq}"

    # Find contact site IDs (hand sites)
    contact_site_ids = []
    for i in range(mj_model.nsite):
        site_name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_SITE, i)
        if site_name and "contact" in site_name and "hand" in site_name:
            contact_site_ids.append(i)
    print(f"Contact sites ({len(contact_site_ids)}): "
          f"{[mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_SITE, i) for i in contact_site_ids]}")
    assert len(contact_site_ids) == 2, f"Expected 2 hand contact sites, got {len(contact_site_ids)}"

    run_viewer = get_viewer(show_viewer, mj_model, mj_data)
    rate_limiter = RateLimiter(fps)

    # Process each frame
    info_list = []
    if save_video:
        images = []
        mj_model.vis.global_.offwidth = 720
        mj_model.vis.global_.offheight = 480
        renderer = mujoco.Renderer(mj_model, height=480, width=720)

    with run_viewer() as gui:
        for i in range(qpos.shape[0]):
            mj_data.qpos[:] = qpos[i]

            # Compute qvel via finite differences
            if i > 0:
                mujoco.mj_differentiatePos(
                    mj_model, mj_data.qvel, 1.0 / fps, qpos[i - 1], qpos[i]
                )
            else:
                mj_data.qvel[:] = 0.0

            # Ctrl = joint position targets (29 joints, excluding freejoint and object)
            mj_data.ctrl[:] = qpos[i][7:36]
            mujoco.mj_forward(mj_model, mj_data)

            # Contact detection
            # B4 DEPRECATION NOTICE (exp_diagnostic_v2 §2/§4):
            # `contact_pos` written here is the IK FK PALM SITE position (after IK
            # has placed G1 in the retargeted pose), NOT raw human mocap fingertip.
            # All downstream "raw contact" face statistics in
            # workspace/core4d_collab_retarget/scripts/E017/E018/E020/E028 actually
            # consume this FK proxy. For box021 D003 where IK is itself flaky, this
            # is misleading. See exp_diagnostic_v2/findings/02 §3 (B4) + §4 (B6)
            # for the wrist≠contact root cause. A future fix is to write a separate
            # `contact_pos_fk` field and also store raw mocap fingertips.
            contact_pos = mj_data.site_xpos[contact_site_ids, :].copy()
            if contact_detection_mode == "one":
                contact = np.ones(len(contact_site_ids))
            else:
                # Auto: check hand-to-object distance
                obj_body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "object")
                obj_pos = mj_data.xpos[obj_body_id]
                distances = np.linalg.norm(contact_pos - obj_pos, axis=1)
                contact = (distances < 0.15).astype(float)

            info = {
                "qpos": mj_data.qpos.copy(),
                "qvel": mj_data.qvel.copy(),
                "ctrl": mj_data.ctrl.copy(),
                "contact": contact,
                "contact_pos": contact_pos,
            }
            info_list.append(info)

            if save_video:
                renderer.update_scene(mj_data, "track")
                images.append(renderer.render())
            if show_viewer:
                gui.sync()
            if enable_rate_limiter:
                rate_limiter.sleep()

    # Aggregate and save
    info_aggregated = {}
    for key in info_list[0]:
        info_aggregated[key] = np.stack([info[key] for info in info_list], axis=0)

    out_path = f"{processed_dir}/trajectory_kinematic.npz"
    np.savez(out_path, **info_aggregated)
    print(f"Saved trajectory to {out_path}")
    for k, v in info_aggregated.items():
        print(f"  {k}: shape={v.shape}, dtype={v.dtype}")

    if save_video:
        video_path = f"{processed_dir}/visualization_kinematic.mp4"
        imageio.mimsave(video_path, images, fps=fps)
        print(f"Saved video to {video_path}")


if __name__ == "__main__":
    tyro.cli(main)
