"""Render kinematic and physics trajectories to video using MuJoCo EGL offscreen.

Generates side-by-side or sequential frames comparing kinematic input vs physics output.

Usage:
    MUJOCO_GL=egl python render_trajectory_video.py \
        --scene <scene.xml> \
        --kin <trajectory_kinematic.npz> \
        --phys <trajectory_mjwp.npz> \
        --output <output.mp4>
export MUJOCO_GL=egl  # 无头服务器必须
python render_trajectory_video.py \
    --scene .../move_largebox/scene.xml \
    --kin  .../move_largebox/0/trajectory_kinematic.npz \
    --phys .../move_largebox/0/trajectory_mjwp.npz \
    --output .../results/omomo_data0/comparison.mp4 \
    --fps 10
"""

import argparse
from pathlib import Path

import mujoco
import numpy as np


def flatten_mjwp(phys_data: dict) -> np.ndarray:
    qpos = phys_data["qpos"]
    if qpos.ndim == 3:
        return qpos[:, -1, :]
    return qpos


def _euler_xyz_to_quat_wxyz(euler: np.ndarray) -> np.ndarray:
    """Convert intrinsic XYZ euler angles to wxyz quaternion."""
    from scipy.spatial.transform import Rotation
    r = Rotation.from_euler("xyz", euler)
    q_xyzw = r.as_quat()  # scipy returns (x,y,z,w)
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])


def adapt_qpos_to_model(
    model: mujoco.MjModel, qpos_seq: np.ndarray
) -> np.ndarray:
    """Convert trajectory qpos to match model nq if needed.

    Handles contact guidance (nq=42, 6 slide/hinge suitcase joints) →
    freejoint (nq=43, 7-DOF suitcase freejoint) conversion.
    """
    traj_nq = qpos_seq.shape[1]
    if traj_nq == model.nq:
        return qpos_seq

    if traj_nq == model.nq - 1:
        # Contact guidance → freejoint: convert suitcase 6DOF to 7DOF
        # Find suitcase body to get its default position
        suitcase_bid = -1
        for bi in range(model.nbody):
            bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bi)
            if bname and "suitcase" in bname:
                suitcase_bid = bi
        if suitcase_bid < 0:
            print("Warning: no suitcase body found, zero-padding qpos")
            pad = np.zeros((qpos_seq.shape[0], 1), dtype=qpos_seq.dtype)
            return np.concatenate([qpos_seq, pad], axis=1)

        body_default_pos = model.body_pos[suitcase_bid].copy()

        # Robot part is qpos[:nq-7] in freejoint model
        robot_nq = model.nq - 7  # 36 for freejoint suitcase
        T = qpos_seq.shape[0]
        out = np.zeros((T, model.nq), dtype=qpos_seq.dtype)
        out[:, :robot_nq] = qpos_seq[:, :robot_nq]

        for t in range(T):
            # Suitcase slide offsets → absolute position
            slide_offset = qpos_seq[t, robot_nq:robot_nq + 3]
            out[t, robot_nq:robot_nq + 3] = slide_offset + body_default_pos
            # Euler xyz → quaternion wxyz
            euler = qpos_seq[t, robot_nq + 3:robot_nq + 6]
            out[t, robot_nq + 3:robot_nq + 7] = _euler_xyz_to_quat_wxyz(euler)

        print(f"Converted qpos: {traj_nq} → {model.nq} (contact guidance → freejoint)")
        return out

    print(f"Warning: trajectory nq={traj_nq} != model nq={model.nq}, truncating/padding")
    T = qpos_seq.shape[0]
    out = np.zeros((T, model.nq), dtype=qpos_seq.dtype)
    n = min(traj_nq, model.nq)
    out[:, :n] = qpos_seq[:, :n]
    return out


def render_frames(
    model: mujoco.MjModel,
    qpos_seq: np.ndarray,
    width: int = 640,
    height: int = 480,
    camera_name: str | None = None,
) -> list[np.ndarray]:
    """Render trajectory frames using offscreen renderer."""
    qpos_seq = adapt_qpos_to_model(model, qpos_seq)
    renderer = mujoco.Renderer(model, height=height, width=width)
    data = mujoco.MjData(model)
    frames = []

    for t in range(qpos_seq.shape[0]):
        data.qpos[:model.nq] = qpos_seq[t, :model.nq]
        mujoco.mj_forward(model, data)

        if camera_name:
            cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
            renderer.update_scene(data, camera=cam_id)
        else:
            renderer.update_scene(data)

        img = renderer.render()
        frames.append(img.copy())

    renderer.close()
    return frames


def frames_to_video(frames: list[np.ndarray], output_path: str, fps: int = 30) -> None:
    """Save frames as MP4 video using imageio."""
    import imageio.v3 as iio

    writer = iio.imopen(output_path, "w", plugin="pyav")
    writer.write(
        np.stack(frames),
        codec="libx264",
        fps=fps,
    )
    writer.close()
    print(f"Video saved: {output_path} ({len(frames)} frames, {fps} fps)")


def frames_to_video_cv2(frames: list[np.ndarray], output_path: str, fps: int = 30) -> None:
    """Save frames as MP4 video using OpenCV."""
    import cv2

    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    for f in frames:
        writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    writer.release()
    print(f"Video saved: {output_path} ({len(frames)} frames, {fps} fps)")


def save_frames_as_images(frames: list[np.ndarray], output_dir: str, prefix: str = "frame") -> None:
    """Fallback: save individual frames as PNG images."""
    from PIL import Image

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(frames):
        img = Image.fromarray(f)
        img.save(out / f"{prefix}_{i:04d}.png")
    print(f"Saved {len(frames)} frames to {output_dir}/{prefix}_*.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render trajectory comparison video")
    parser.add_argument("--scene", type=str, required=True)
    parser.add_argument("--kin", type=str, required=True)
    parser.add_argument("--phys", type=str, required=True)
    parser.add_argument("--output", type=str, default="comparison.mp4")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--camera", type=str, default=None)
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(args.scene)
    print(f"Scene: nq={model.nq}")

    # Load data
    kin_qpos = np.load(args.kin)["qpos"]
    phys_qpos = flatten_mjwp(dict(np.load(args.phys)))
    print(f"Kinematic: {kin_qpos.shape}, Physics: {phys_qpos.shape}")

    # Render both
    print("Rendering kinematic frames...")
    kin_frames = render_frames(model, kin_qpos, args.width, args.height, args.camera)
    print("Rendering physics frames...")
    phys_frames = render_frames(model, phys_qpos, args.width, args.height, args.camera)

    # Combine side-by-side (resample to same length)
    T = min(len(kin_frames), len(phys_frames))
    kin_idx = np.linspace(0, len(kin_frames) - 1, T).astype(int)
    phys_idx = np.linspace(0, len(phys_frames) - 1, T).astype(int)

    combined_frames = []
    for ki, pi in zip(kin_idx, phys_idx):
        combined = np.concatenate([kin_frames[ki], phys_frames[pi]], axis=1)
        combined_frames.append(combined)

    # Try different video backends
    output = args.output
    try:
        # Prefer imageio with h264 (VSCode compatible)
        import imageio
        imageio.mimsave(output, combined_frames, fps=args.fps, codec="libx264", quality=8)
        print(f"Video saved: {output} ({len(combined_frames)} frames, {args.fps} fps)")
    except Exception:
        try:
            frames_to_video_cv2(combined_frames, output, args.fps)
        except ImportError:
            try:
                frames_to_video(combined_frames, output, args.fps)
            except ImportError:
                out_dir = str(Path(output).with_suffix(""))
                save_frames_as_images(combined_frames, out_dir, "combined")


if __name__ == "__main__":
    main()
