"""Analyze and visualize contact points from reference trajectory.

Computes per-frame hand-to-object contact in the object's local frame,
identifies contact target points, and renders a video with contact markers.

Usage:
    MUJOCO_GL=egl python workspace/core4d/scripts/eval/analyze_contact_points.py \
        --task box025_person1 \
        --output workspace/core4d/results/E039_viz/

Outputs:
    - {task}_contact_analysis.npz: per-frame distances, offsets, masks
    - {task}_contact_curve.png: contact ratio over time
    - {task}_contact_video.mp4: video with contact point markers
"""

import argparse
import os
from pathlib import Path

import imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation


def surface_distance_rotated(hand_pos, obj_pos, obj_mat, half_ext):
    """Correct rotated box SDF (same as eval_comprehensive.py)."""
    local = obj_mat.T @ (hand_pos - obj_pos)
    clamped = np.clip(local, -half_ext, half_ext)
    return np.linalg.norm(local - clamped)


def compute_contact_analysis(model, data, qpos_ref, obj_bid, eef_bids, half_ext):
    """Compute per-frame contact data.

    Returns dict with:
        - surface_dist: (T, 2) rotated SDF distance per EEF
        - hand_pos_world: (T, 2, 3) EEF world positions
        - obj_pos_world: (T, 3) object world positions
        - obj_quat_world: (T, 4) object wxyz quaternions
        - hand_offset_local: (T, 2, 3) hand position in object local frame
        - contact_mask: (T, 2) bool, True if surface_dist < threshold
    """
    T = qpos_ref.shape[0]
    n_eef = len(eef_bids)

    surface_dist = np.zeros((T, n_eef))
    hand_pos_world = np.zeros((T, n_eef, 3))
    obj_pos_world = np.zeros((T, 3))
    obj_quat_world = np.zeros((T, 4))
    hand_offset_local = np.zeros((T, n_eef, 3))

    for t in range(T):
        data.qpos[:] = qpos_ref[t]
        mujoco.mj_forward(model, data)

        obj_pos = data.xpos[obj_bid].copy()
        obj_mat = data.xmat[obj_bid].reshape(3, 3)
        obj_quat = data.xquat[obj_bid].copy()  # wxyz

        obj_pos_world[t] = obj_pos
        obj_quat_world[t] = obj_quat

        r_inv = Rotation.from_quat(
            [obj_quat[1], obj_quat[2], obj_quat[3], obj_quat[0]]
        ).inv()

        for ei, bid in enumerate(eef_bids):
            hp = data.xpos[bid].copy()
            hand_pos_world[t, ei] = hp
            surface_dist[t, ei] = surface_distance_rotated(hp, obj_pos, obj_mat, half_ext)
            hand_offset_local[t, ei] = r_inv.apply(hp - obj_pos)

    return {
        "surface_dist": surface_dist,
        "hand_pos_world": hand_pos_world,
        "obj_pos_world": obj_pos_world,
        "obj_quat_world": obj_quat_world,
        "hand_offset_local": hand_offset_local,
    }


def compute_contact_targets(hand_offset_local, surface_dist, threshold=0.15):
    """Compute mean contact target offset from close-contact frames.

    Returns:
        targets: (2, 3) per-EEF mean offset in object local frame
        contact_mask: (T, 2) per-EEF contact mask
    """
    T, n_eef, _ = hand_offset_local.shape
    contact_mask = surface_dist < threshold
    targets = np.zeros((n_eef, 3))

    for ei in range(n_eef):
        mask = contact_mask[:, ei]
        if mask.sum() > 0:
            targets[ei] = hand_offset_local[mask, ei].mean(axis=0)
        else:
            # Fallback: use frames with smallest distance
            k = max(1, T // 10)
            idx = np.argsort(surface_dist[:, ei])[:k]
            targets[ei] = hand_offset_local[idx, ei].mean(axis=0)

    return targets, contact_mask


def plot_contact_curve(surface_dist, contact_mask, fps, output_path, task_name):
    """Plot contact distance and mask over time."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    T = surface_dist.shape[0]
    time = np.arange(T) / fps

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    # Top: distance curves
    ax = axes[0]
    ax.plot(time, surface_dist[:, 0] * 100, "g-", label="Left wrist", alpha=0.8)
    ax.plot(time, surface_dist[:, 1] * 100, "b-", label="Right wrist", alpha=0.8)
    ax.plot(time, surface_dist.min(axis=1) * 100, "r-", linewidth=2, label="Best hand")
    ax.axhline(15, color="orange", linestyle="--", alpha=0.5, label="15cm threshold")
    ax.axhline(10, color="red", linestyle="--", alpha=0.5, label="10cm threshold")
    ax.set_ylabel("Surface Distance (cm)")
    ax.set_title(f"{task_name} — Hand-to-Object Surface Distance (Rotated SDF)")
    ax.legend(loc="upper right")
    ax.set_ylim(0, min(100, surface_dist.max() * 100 + 5))
    ax.grid(True, alpha=0.3)

    # Bottom: contact mask
    ax = axes[1]
    ax.fill_between(time, 0, contact_mask[:, 0].astype(float), alpha=0.4, color="green", label="Left contact")
    ax.fill_between(time, 0, contact_mask[:, 1].astype(float), alpha=0.4, color="blue", label="Right contact")
    ax.set_ylabel("Contact Active")
    ax.set_xlabel("Time (s)")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["No", "Yes"])
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # Add summary stats
    total_contact = (contact_mask.any(axis=1)).mean() * 100
    fig.suptitle(f"{task_name}: {total_contact:.0f}% frames with any contact (<15cm)", fontsize=11, y=0.98)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved contact curve: {output_path}")


def render_contact_video(
    model, data, qpos_ref, obj_bid, eef_bids, half_ext,
    contact_data, contact_targets, output_path, fps=30
):
    """Render video with contact point markers overlaid."""
    T = qpos_ref.shape[0]
    width, height = 640, 480
    renderer = mujoco.Renderer(model, height=height, width=width)

    surface_dist = contact_data["surface_dist"]
    obj_pos_world = contact_data["obj_pos_world"]
    obj_quat_world = contact_data["obj_quat_world"]
    hand_pos_world = contact_data["hand_pos_world"]

    # Get camera matrix for world-to-pixel projection
    scene = mujoco.MjvScene(model, maxgeom=1000)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam)

    frames = []
    for t in range(T):
        data.qpos[:] = qpos_ref[t]
        mujoco.mj_forward(model, data)

        # Compute target positions in world frame
        obj_pos = obj_pos_world[t]
        obj_quat = obj_quat_world[t]
        rot = Rotation.from_quat([obj_quat[1], obj_quat[2], obj_quat[3], obj_quat[0]])

        target_worlds = []
        for ei in range(len(eef_bids)):
            tw = obj_pos + rot.apply(contact_targets[ei])
            target_worlds.append(tw)

        # Render base scene
        renderer.update_scene(data, camera=cam)
        img = renderer.render().copy()

        # Project 3D points to 2D pixel coords
        pil_img = Image.fromarray(img)
        draw = ImageDraw.Draw(pil_img)

        # Use mujoco's camera to project world → pixel
        # Get camera pos/orientation from scene
        scn = renderer._scene
        cam_pos = np.array(scn.camera[0].pos)
        cam_forward = np.array(scn.camera[0].forward)
        cam_up = np.array(scn.camera[0].up)
        cam_right = np.cross(cam_forward, cam_up)
        # Focal length from frustum
        fovy = model.vis.global_.fovy if model.vis.global_.fovy > 0 else 45.0
        f = height / (2.0 * np.tan(np.radians(fovy) / 2.0))

        def project_point(world_pt):
            """Project 3D world point to 2D pixel (x, y)."""
            d = world_pt - cam_pos
            # Camera coordinates: x=right, y=up, z=-forward
            cx = np.dot(d, cam_right)
            cy = np.dot(d, -cam_up)
            cz = np.dot(d, cam_forward)
            if cz < 0.01:
                return None
            px = int(width / 2 + f * cx / cz)
            py = int(height / 2 + f * cy / cz)
            if 0 <= px < width and 0 <= py < height:
                return (px, py)
            return None

        # Draw contact target points
        colors_target = [(0, 255, 0), (100, 150, 255)]  # green=left, blue=right
        colors_hand = [(200, 255, 200), (200, 200, 255)]
        labels = ["L", "R"]
        for ei in range(len(eef_bids)):
            # Draw target point (filled circle)
            pt = project_point(target_worlds[ei])
            if pt:
                r = 8
                draw.ellipse([pt[0]-r, pt[1]-r, pt[0]+r, pt[1]+r], fill=colors_target[ei], outline=(255, 255, 255))
                draw.text((pt[0]+r+2, pt[1]-6), f"{labels[ei]}t", fill=colors_target[ei])

            # Draw hand point (hollow circle)
            hp = project_point(hand_pos_world[t, ei])
            if hp:
                r = 6
                draw.ellipse([hp[0]-r, hp[1]-r, hp[0]+r, hp[1]+r], outline=colors_target[ei], width=2)

            # Draw line from hand to target
            if pt and hp:
                draw.line([hp, pt], fill=colors_target[ei], width=2)

        # Text overlay
        dl = surface_dist[t, 0] * 100
        dr = surface_dist[t, 1] * 100
        min_d = min(dl, dr)
        color = (0, 255, 0) if min_d < 10 else (255, 255, 0) if min_d < 15 else (255, 100, 100)
        draw.text((5, 5), f"t={t}/{T} ({t/T*100:.0f}%)", fill=(255, 255, 255))
        draw.text((5, 20), f"L surf: {dl:.1f}cm", fill=(100, 255, 100))
        draw.text((5, 35), f"R surf: {dr:.1f}cm", fill=(100, 150, 255))
        draw.text((5, 50), f"min: {min_d:.1f}cm", fill=color)

        # Legend
        draw.text((width-150, 5), "● = target", fill=(255, 255, 255))
        draw.text((width-150, 20), "○ = hand", fill=(255, 255, 255))
        draw.text((width-150, 35), "— = gap", fill=(255, 255, 255))

        # Contact progress bar
        bar_y = 65
        bar_w = 100
        contact_ratio = (surface_dist[:t+1].min(axis=1) < 0.10).mean() if t > 0 else 0
        draw.rectangle([5, bar_y, 5 + bar_w, bar_y + 10], outline=(128, 128, 128))
        draw.rectangle([5, bar_y, 5 + int(bar_w * contact_ratio), bar_y + 10], fill=(0, 200, 0))
        draw.text((bar_w + 10, bar_y - 2), f"<10cm: {contact_ratio*100:.0f}%", fill=(200, 200, 200))

        frames.append(np.array(pil_img))

    renderer.close()

    # Save video
    imageio.mimsave(output_path, frames, fps=fps, codec="libx264", quality=8)
    print(f"  Saved contact video: {output_path} ({T} frames)")


def main():
    parser = argparse.ArgumentParser(description="Analyze contact points from ref trajectory")
    parser.add_argument("--task", type=str, required=True, help="e.g. box025_person1")
    parser.add_argument("--output", type=str, default="workspace/core4d/results/E039_viz")
    parser.add_argument("--threshold", type=float, default=0.15, help="Contact threshold (m)")
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    base = "example_datasets/processed/core4d/unitree_g1/humanoid_object"
    scene_path = f"{base}/{args.task}/scene.xml"
    ref_path = f"{base}/{args.task}/0/trajectory_kinematic.npz"

    assert os.path.exists(scene_path), f"Scene not found: {scene_path}"
    assert os.path.exists(ref_path), f"Ref not found: {ref_path}"

    os.makedirs(args.output, exist_ok=True)

    # Load
    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)
    ref = np.load(ref_path)
    qpos_ref = ref["qpos"]

    obj_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    left_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_wrist_yaw_link")
    right_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_yaw_link")
    eef_bids = [left_bid, right_bid]

    # Get collision half extents
    half_ext = None
    for gi in range(model.ngeom):
        if model.geom_bodyid[gi] == obj_bid:
            gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gi)
            if "collision" in (gname or ""):
                half_ext = model.geom_size[gi].copy()
                break
    assert half_ext is not None, "No collision geom found for object"

    print(f"\n{'='*60}")
    print(f"  Contact Analysis: {args.task}")
    print(f"  T={qpos_ref.shape[0]} frames, half_ext={half_ext}")
    print(f"{'='*60}")

    # 1. Compute contact data
    print("\n[1/4] Computing per-frame contact data...")
    contact_data = compute_contact_analysis(model, data, qpos_ref, obj_bid, eef_bids, half_ext)

    # 2. Compute contact targets
    print("[2/4] Computing contact target points...")
    targets, contact_mask = compute_contact_targets(
        contact_data["hand_offset_local"],
        contact_data["surface_dist"],
        threshold=args.threshold,
    )

    print(f"\n  Contact targets (in object local frame):")
    print(f"    Left wrist:  [{targets[0, 0]:.4f}, {targets[0, 1]:.4f}, {targets[0, 2]:.4f}]")
    print(f"    Right wrist: [{targets[1, 0]:.4f}, {targets[1, 1]:.4f}, {targets[1, 2]:.4f}]")

    sd = contact_data["surface_dist"]
    min_sd = sd.min(axis=1)
    print(f"\n  Contact statistics (rotated SDF):")
    print(f"    Mean dist (best hand): {min_sd.mean()*100:.1f}cm")
    print(f"    <15cm: {(min_sd<0.15).mean()*100:.0f}% ({(min_sd<0.15).sum()}/{len(min_sd)})")
    print(f"    <10cm: {(min_sd<0.10).mean()*100:.0f}% ({(min_sd<0.10).sum()}/{len(min_sd)})")
    print(f"    < 5cm: {(min_sd<0.05).mean()*100:.0f}% ({(min_sd<0.05).sum()}/{len(min_sd)})")
    print(f"    < 3cm: {(min_sd<0.03).mean()*100:.0f}% ({(min_sd<0.03).sum()}/{len(min_sd)})")
    print(f"    Per-EEF contact frames (<{args.threshold*100:.0f}cm):")
    print(f"      Left:  {contact_mask[:, 0].sum()}/{len(contact_mask)} ({contact_mask[:, 0].mean()*100:.0f}%)")
    print(f"      Right: {contact_mask[:, 1].sum()}/{len(contact_mask)} ({contact_mask[:, 1].mean()*100:.0f}%)")

    # 3. Plot contact curve
    print("\n[3/4] Plotting contact curve...")
    plot_contact_curve(
        sd, contact_mask, args.fps,
        f"{args.output}/{args.task}_contact_curve.png",
        args.task,
    )

    # 4. Render video
    print("[4/4] Rendering contact video...")
    render_contact_video(
        model, data, qpos_ref, obj_bid, eef_bids, half_ext,
        contact_data, targets,
        f"{args.output}/{args.task}_contact_video.mp4",
        fps=args.fps,
    )

    # Save analysis NPZ
    npz_path = f"{args.output}/{args.task}_contact_analysis.npz"
    np.savez(
        npz_path,
        surface_dist=sd,
        contact_mask=contact_mask,
        contact_targets=targets,
        hand_offset_local=contact_data["hand_offset_local"],
        half_ext=half_ext,
        threshold=args.threshold,
    )
    print(f"  Saved analysis: {npz_path}")
    print(f"\n{'='*60}")
    print(f"  Done: {args.task}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
