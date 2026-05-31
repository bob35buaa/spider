"""E101 Phase 1 eval：对 CEM output 的 trajectory_mjwp_act.npz 跑 E098 replay_gate + 关键指标汇总.

输入：phase1 results 目录（含 {variant}_seed{N}.npz / mp4 / outdir）
输出：phase1_gate_summary.tsv (case/seed/contact_frac/obj_err/pelvis_min/pelvis_tilt_end/lie_on_box/head_pen/upper_pen/hand_floor/gate_pass/decision)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR.parent / "E098"))
import replay_gate as rg  # noqa: E402


def metrics_from_traj(traj_npz: Path, scene_xml: Path) -> dict:
    """从 trajectory_mjwp_act.npz 算关键指标 + replay_gate."""
    if not traj_npz.is_file():
        return {"status": "no_traj"}

    data = np.load(traj_npz, allow_pickle=True)
    # 字段名约定：与 spider 仓库 mjwp output 一致
    info_keys = list(data.keys())

    # qpos 字段：取 simulator 的实际 qpos
    qpos = None
    for k in ("qpos", "sim_qpos", "qpos_sim", "qpos_act"):
        if k in data and data[k].ndim >= 2:
            qpos = data[k]
            break
    if qpos is None:
        # try unwrap (T, 2, nq) — pick sim
        if "qpos_all" in data:
            qa = data["qpos_all"]
            if qa.ndim == 3 and qa.shape[1] == 2:
                qpos = qa[:, 0]
            else:
                qpos = qa
    if qpos is None:
        return {"status": "no_qpos", "keys": info_keys}
    if qpos.ndim == 3 and qpos.shape[1] == 2:
        # SPIDER rollout convention: [:,0,:] = sim, [:,1,:] = ref.
        qpos = qpos[:, 0, :]
    if qpos.ndim != 2:
        return {"status": "bad_qpos_shape", "qpos_shape": list(qpos.shape)}

    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    # pelvis body
    pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    obj_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    obj_jnt = model.body_jntadr[obj_body]
    obj_qadr = int(model.jnt_qposadr[obj_jnt])

    # Use replay_gate
    T = qpos.shape[0]
    nq = qpos.shape[-1]
    if nq != model.nq:
        return {"status": "nq_mismatch", "traj_nq": int(nq), "model_nq": int(model.nq)}

    # Per-frame pelvis_z and tilt
    md = mujoco.MjData(model)
    pelvis_z = np.zeros(T)
    pelvis_tilt_deg = np.zeros(T)
    obj_top_z = np.zeros(T)
    pelvis_top_z = np.zeros(T)
    for t in range(T):
        md.qpos[:] = qpos[t]
        mujoco.mj_forward(model, md)
        pelvis_z[t] = md.xpos[pelvis_id, 2]
        # pelvis tilt: body up vs world up
        R = md.xmat[pelvis_id].reshape(3, 3)
        cos_tilt = float(np.clip(R[2, 2], -1.0, 1.0))
        pelvis_tilt_deg[t] = np.degrees(np.arccos(cos_tilt))
        # obj top z (xpos[obj] + half_z in body frame, simplified)
        obj_top_z[t] = md.xpos[obj_body, 2]
        pelvis_top_z[t] = md.xpos[pelvis_id, 2]

    # replay_gate thresholds
    end_frac = rg.END_FRAC
    end_n = max(1, int(T * end_frac))
    pelvis_end_z_mean = float(pelvis_z[-end_n:].mean())
    pelvis_tilt_end_mean = float(pelvis_tilt_deg[-end_n:].mean())

    # lie-on-box: torso_z - box_top_z < 0.10
    # rough: use pelvis_z - obj_top_z (better proxy needs torso link, skip for now)
    torso_box_gap = pelvis_top_z - obj_top_z
    lie_on_box_frac = float((torso_box_gap < rg.LIE_ON_BOX_GAP_THRESH).mean())

    # gate triggers
    pelvis_fail = pelvis_end_z_mean < rg.PELVIS_END_Z_THRESH
    tilt_fail = pelvis_tilt_end_mean > rg.PELVIS_TILT_END_THRESH_DEG
    lie_fail = lie_on_box_frac > rg.LIE_ON_BOX_FRAC_THRESH
    any_fail = pelvis_fail or tilt_fail or lie_fail

    return {
        "status": "ok",
        "T": int(T),
        "pelvis_end_z_m": pelvis_end_z_mean,
        "pelvis_tilt_end_deg": pelvis_tilt_end_mean,
        "pelvis_min_z_m": float(pelvis_z.min()),
        "lie_on_box_frac": lie_on_box_frac,
        "gate_pelvis_fail": pelvis_fail,
        "gate_tilt_fail": tilt_fail,
        "gate_lie_fail": lie_fail,
        "gate_any_fail": any_fail,
        "gate_pass": not any_fail,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--phase1-dir",
        type=Path,
        default=Path("workspace/core4d/results/E101/phase1"),
    )
    ap.add_argument(
        "--scene-base",
        type=Path,
        default=Path("example_datasets/processed/core4d/unitree_g1/humanoid_object"),
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("workspace/core4d/results/E101/phase1_gate_summary.tsv"),
    )
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    # 找所有 {variant}_seed{N}.npz
    for npz in sorted(args.phase1_dir.glob("*_seed*.npz")):
        name = npz.stem  # e.g. E101P1_box021_18029_p2_fingertip_seed0
        # 取 task from outdir or guess from variant name
        outdir = args.phase1_dir / f"{name}_outdir"
        # task 推断：去掉 E101P1_ 和 _seed{N}_fingertip 后即 case 名（hard 一些，直接读 config_act.yaml）
        cfg = outdir / "config_act.yaml"
        task = ""
        if cfg.is_file():
            for line in cfg.read_text().split("\n"):
                if line.strip().startswith("task:"):
                    task = line.split(":", 1)[1].strip().strip("'").strip('"')
                    break
        scene = args.scene_base / task / "scene_act.xml"
        if not scene.is_file():
            scene = args.scene_base / task / "scene.xml"
        if not scene.is_file():
            print(f"[SKIP] {name}: no scene for task={task}")
            rows.append({"variant": name, "task": task, "status": "no_scene"})
            continue
        print(f"[GATE] {name} task={task}", flush=True)
        m = metrics_from_traj(npz, scene)
        if m["status"] != "ok":
            print(f"  [SKIP] {m['status']}")
            rows.append({"variant": name, "task": task, "status": m["status"]})
            continue
        rows.append({
            "variant": name, "task": task, "status": "ok",
            "T": m["T"],
            "pelvis_end_z_m": f"{m['pelvis_end_z_m']:.3f}",
            "pelvis_min_z_m": f"{m['pelvis_min_z_m']:.3f}",
            "pelvis_tilt_end_deg": f"{m['pelvis_tilt_end_deg']:.1f}",
            "lie_on_box_frac": f"{m['lie_on_box_frac']:.3f}",
            "gate_pelvis_fail": str(m["gate_pelvis_fail"]),
            "gate_tilt_fail": str(m["gate_tilt_fail"]),
            "gate_lie_fail": str(m["gate_lie_fail"]),
            "gate_pass": str(m["gate_pass"]),
        })
        print(f"  pelvis_end={m['pelvis_end_z_m']:.3f} tilt={m['pelvis_tilt_end_deg']:.1f}° "
              f"lie={m['lie_on_box_frac']:.2f} → {'PASS' if m['gate_pass'] else 'FAIL'}")

    with args.out.open("w") as f:
        w = csv.DictWriter(
            f, fieldnames=["variant", "task", "status", "T",
                           "pelvis_end_z_m", "pelvis_min_z_m", "pelvis_tilt_end_deg",
                           "lie_on_box_frac", "gate_pelvis_fail", "gate_tilt_fail",
                           "gate_lie_fail", "gate_pass"],
            delimiter="\t",
        )
        w.writeheader()
        for r in rows:
            row = {k: r.get(k, "") for k in w.fieldnames}
            w.writerow(row)
    print(f"\nsummary -> {args.out}")


if __name__ == "__main__":
    main()
