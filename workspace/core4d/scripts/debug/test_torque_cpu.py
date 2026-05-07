"""Test xfrc_applied torque PD control on freejoint object in pure MuJoCo (CPU).

Goal: Verify that orientation torque via xfrc_applied can stabilize a tilted object.
This isolates the torque math from MuJoCo Warp / CUDA graph complications.

Usage:
    python workspace/core4d/scripts/debug/test_torque_cpu.py
"""

import sys
import numpy as np

try:
    import mujoco
except ImportError:
    print("mujoco not installed")
    sys.exit(1)

SCENE_XML = "example_datasets/processed/core4d/unitree_g1/humanoid_object/box025_person1/scene.xml"


def quat_conj(q):
    """Conjugate quaternion (w,x,y,z)."""
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_mul(a, b):
    """Multiply quaternions (w,x,y,z)."""
    w = a[0]*b[0] - a[1]*b[1] - a[2]*b[2] - a[3]*b[3]
    x = a[0]*b[1] + a[1]*b[0] + a[2]*b[3] - a[3]*b[2]
    y = a[0]*b[2] - a[1]*b[3] + a[2]*b[0] + a[3]*b[1]
    z = a[0]*b[3] + a[1]*b[2] - a[2]*b[1] + a[3]*b[0]
    return np.array([w, x, y, z])


def quat_to_axis_angle(q):
    """Convert quaternion (w,x,y,z) to axis-angle (3,). Handles sign ambiguity."""
    # Ensure w >= 0 for shortest path
    if q[0] < 0:
        q = -q
    axis = q[1:4]
    sin_half = np.linalg.norm(axis)
    if sin_half < 1e-8:
        return np.zeros(3)
    angle = 2.0 * np.arctan2(sin_half, q[0])
    # Wrap to [-pi, pi]
    if angle > np.pi:
        angle -= 2.0 * np.pi
    return axis / sin_half * angle


def axis_angle_to_quat(aa):
    """Convert axis-angle (3,) to quaternion (w,x,y,z)."""
    angle = np.linalg.norm(aa)
    if angle < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = aa / angle
    w = np.cos(angle / 2)
    xyz = axis * np.sin(angle / 2)
    return np.array([w, xyz[0], xyz[1], xyz[2]])


def test_torque_pd(kp_rot, kd_rot, tilt_deg=30.0, duration=3.0, dt=0.002):
    """Test torque PD on a tilted object. Returns (stable, max_angle, had_nan)."""
    model = mujoco.MjModel.from_xml_path(SCENE_XML)
    data = mujoco.MjData(model)
    model.opt.timestep = dt

    # Find object body and joint
    obj_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    if obj_body_id == -1:
        print("ERROR: 'object' body not found")
        return False, 999.0, False

    obj_jnt_id = model.body_jntadr[obj_body_id]
    obj_qadr = model.jnt_qposadr[obj_jnt_id]
    obj_vadr = model.jnt_dofadr[obj_jnt_id]

    # Reset and tilt object around X axis
    mujoco.mj_resetData(model, data)
    tilt_rad = np.radians(tilt_deg)
    tilt_quat = axis_angle_to_quat(np.array([tilt_rad, 0.0, 0.0]))
    data.qpos[obj_qadr + 3:obj_qadr + 7] = tilt_quat
    mujoco.mj_forward(model, data)

    # Reference: upright
    quat_ref = np.array([1.0, 0.0, 0.0, 0.0])

    obj_mass = model.body_mass[obj_body_id]
    obj_inertia = model.body_inertia[obj_body_id]  # (3,) diagonal
    avg_inertia = np.mean(obj_inertia)

    angles = []
    had_nan = False
    nsteps = int(duration / dt)

    for step in range(nsteps):
        # Get current orientation
        quat_cur = data.qpos[obj_qadr + 3:obj_qadr + 7].copy()
        omega_cur = data.qvel[obj_vadr + 3:obj_vadr + 6].copy()

        # Check NaN
        if np.any(np.isnan(quat_cur)) or np.any(np.isnan(omega_cur)):
            had_nan = True
            break

        # Quaternion error: q_err = q_ref * q_cur^(-1) → axis-angle in world frame
        q_err = quat_mul(quat_ref, quat_conj(quat_cur))
        aa_err = quat_to_axis_angle(q_err)

        # Clamp axis-angle magnitude
        aa_mag = np.linalg.norm(aa_err)
        if aa_mag > 0.5:
            aa_err = aa_err / aa_mag * 0.5

        # PD torque (world frame)
        torque = kp_rot * aa_err - kd_rot * omega_cur

        # Apply via xfrc_applied (world frame torque)
        data.xfrc_applied[obj_body_id, 3:6] = torque

        # Also apply gravity compensation to keep object floating
        data.xfrc_applied[obj_body_id, 2] = obj_mass * 9.81

        mujoco.mj_step(model, data)

        # Record angle from upright
        quat_now = data.qpos[obj_qadr + 3:obj_qadr + 7].copy()
        q_err_now = quat_mul(quat_ref, quat_conj(quat_now))
        angle_now = np.linalg.norm(quat_to_axis_angle(q_err_now))
        angles.append(np.degrees(angle_now))

    if had_nan:
        return False, 999.0, True

    max_angle = max(angles)
    final_angle = angles[-1] if angles else 999.0
    stable = final_angle < 5.0  # < 5 degrees at end

    return stable, final_angle, False


def main():
    print("=== E030 Step 1: xfrc_applied Torque PD Test (Pure MuJoCo CPU) ===\n")
    print(f"Scene: {SCENE_XML}")
    print(f"Test: Tilt object 30° around X, apply torque PD, check recovery\n")

    # Get object inertia for auto kd
    model = mujoco.MjModel.from_xml_path(SCENE_XML)
    obj_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    obj_inertia = model.body_inertia[obj_body_id]
    avg_I = np.mean(obj_inertia)
    obj_mass = model.body_mass[obj_body_id]
    print(f"Object mass: {obj_mass:.3f} kg, inertia: {obj_inertia}, avg_I: {avg_I:.4f}\n")

    results = []
    kp_values = [1.0, 5.0, 10.0, 20.0, 50.0]

    print(f"{'kp_rot':>8} {'kd_rot':>8} {'stable':>8} {'final_angle':>12} {'NaN':>5}")
    print("-" * 50)

    for kp in kp_values:
        # Critical damping: kd = 2*sqrt(I*kp)
        kd = 2.0 * np.sqrt(avg_I * kp)
        stable, final_angle, had_nan = test_torque_pd(kp, kd, tilt_deg=30.0)
        status = "NaN!" if had_nan else ("OK" if stable else "UNSTABLE")
        print(f"{kp:8.1f} {kd:8.3f} {status:>8} {final_angle:12.2f}° {had_nan!s:>5}")
        results.append((kp, kd, stable, final_angle, had_nan))

    # Also test without clamp to see if that's the issue
    print("\n--- Without axis-angle clamp (kp=10) ---")
    # We'll do a manual test inline
    kp, kd = 10.0, 2.0 * np.sqrt(avg_I * 10.0)
    # Run with large clamp (effectively no clamp)
    stable, final_angle, had_nan = test_torque_pd(kp, kd, tilt_deg=30.0)
    print(f"kp={kp}, kd={kd:.3f}: stable={stable}, final={final_angle:.2f}°, nan={had_nan}")

    # Test larger tilt
    print("\n--- Large tilt (60°, kp=10) ---")
    stable, final_angle, had_nan = test_torque_pd(kp, kd, tilt_deg=60.0)
    print(f"kp={kp}, kd={kd:.3f}: stable={stable}, final={final_angle:.2f}°, nan={had_nan}")

    # Test with position spring too
    print("\n--- Combined: position spring + orientation torque ---")
    kp_rot_best = None
    for kp, _, stable, _, had_nan in results:
        if stable and not had_nan:
            kp_rot_best = kp
            break
    if kp_rot_best:
        print(f"Best kp_rot={kp_rot_best} (first stable)")
    else:
        print("WARNING: No stable kp_rot found!")

    print("\nDone.")


if __name__ == "__main__":
    main()
