"""Render hybrid trajectory visualization: robot from physics + object from ref.

Uses MuJoCo renderer to show the combined result.
"""

import os
import cv2
import mujoco
import numpy as np


CASES = {
    "box025_person1": "example_datasets/processed/core4d/unitree_g1/humanoid_object/box025_person1",
    "bucket010_person1": "example_datasets/processed/core4d/unitree_g1/humanoid_object/bucket010_person1",
    "chair022_person1": "example_datasets/processed/core4d/unitree_g1/humanoid_object/chair022_person1",
    "desk005_person2": "example_datasets/processed/core4d/unitree_g1/humanoid_object/desk005_person2",
}

OUTPUT_DIR = "workspace/core4d/results/E030_hybrid_export"
FPS = 10  # Low fps since we only have ~11 frames


def render_hybrid(case_name: str, case_dir: str):
    """Render hybrid trajectory for one case."""
    hybrid_path = os.path.join(OUTPUT_DIR, f"{case_name}_hybrid.npz")
    if not os.path.exists(hybrid_path):
        print(f"SKIP {case_name}: no hybrid npz")
        return

    scene_xml = f"{case_dir}/scene.xml"
    model = mujoco.MjModel.from_xml_path(scene_xml)
    data = mujoco.MjData(model)

    hybrid_data = np.load(hybrid_path)
    qpos_traj = hybrid_data["qpos"]  # (n_frames, nq)

    # Setup renderer
    renderer = mujoco.Renderer(model, height=480, width=720)

    frames = []
    for i in range(qpos_traj.shape[0]):
        data.qpos[:] = qpos_traj[i]
        mujoco.mj_forward(model, data)

        renderer.update_scene(data, camera="track")
        frame = renderer.render()
        frames.append(frame.copy())

    renderer.close()

    # Save video
    video_path = os.path.join(OUTPUT_DIR, f"{case_name}_hybrid.mp4")
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(video_path, fourcc, FPS, (w, h))
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()

    # Save key frames
    for i, idx in enumerate([0, len(frames) // 3, len(frames) * 2 // 3]):
        if idx < len(frames):
            img = cv2.cvtColor(frames[idx], cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"{case_name}_hybrid_f{i}.png"), img)

    print(f"  {case_name}: {len(frames)} frames → {video_path}")


def main():
    os.environ.setdefault("MUJOCO_GL", "egl")
    print("Rendering hybrid trajectories...")
    for case_name, case_dir in CASES.items():
        render_hybrid(case_name, case_dir)
    print("Done.")


if __name__ == "__main__":
    main()
