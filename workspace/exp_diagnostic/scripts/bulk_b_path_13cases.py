"""Process all 13 D003 box021 case-persons:
  (a) For cases that already have a trajectory_kinematic.npz locally, repair in-place to *_btop variant.
  (b) For missing cases, regenerate trajectory_kinematic.npz from holosoma trimmed
      qpos using the box021_person2 scene template, then repair.

For each case, evaluate the G1-Feasibility gate on the repaired npz and emit JSON.
"""
from __future__ import annotations
import os
import sys
import csv
import json
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")

import numpy as np
import mujoco

REPO = Path("/mnt/ali-sh-1/usr/xiayibo/work_dir/embodied/spider")
TASKS_DIR = REPO / "example_datasets/processed/core4d/unitree_g1/humanoid_object"
TEMPLATE_TASK = "box021_person2"
HOLOSOMA_TRIMMED_ROOT = Path(
    "/mnt/ali-sh-1/usr/xiayibo/work_dir/embodied/holosoma/workspace/v3/"
    "data_construction/results/d003_omniretarget_spider_production/results"
)
MANIFEST = REPO / "workspace/core4d_collab_retarget/results/E028/manifest.tsv"

sys.path.insert(0, str(REPO / "workspace/exp_diagnostic/scripts"))
from wrist_repair_top_face import repair_one_case  # noqa: E402
from g1_feasibility_gate import evaluate  # noqa: E402
from gate_compare_b_path import world_up_face_frac  # noqa: E402


def load_manifest():
    rows = []
    with MANIFEST.open() as f:
        for row in csv.DictReader(f, delimiter="\t"):
            rows.append({
                "source_task": row["source_task"],
                "seq": row["d003_sequence"],
                "person": row["d003_person"],
                "object": row["d003_object"],
            })
    return rows


def derive_trajectory_kinematic(case_task: str, seq: str, person: str, object_name: str) -> Path:
    """For a missing case, regenerate scene.xml + trajectory_kinematic.npz from
    holosoma trimmed qpos using box021_person2 template."""
    case_dir = TASKS_DIR / case_task
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "0").mkdir(parents=True, exist_ok=True)

    # Copy template scene
    src_scene = TASKS_DIR / TEMPLATE_TASK / "scene.xml"
    dst_scene = case_dir / "scene.xml"
    if not dst_scene.exists():
        shutil.copy2(src_scene, dst_scene)

    # Locate holosoma trimmed NPZ for this case
    date = seq.split("/")[0]
    seq_id = seq.split("/")[1]
    person_short = "p" + person[-1]  # person1 -> p1, person2 -> p2
    holosoma_case_dir = HOLOSOMA_TRIMMED_ROOT / f"holosoma_d003_box021_{date}_{seq_id}_{person_short}"
    trimmed_dir = holosoma_case_dir / "trimmed"
    trim_files = sorted(trimmed_dir.glob("*.npz"))
    if not trim_files:
        raise FileNotFoundError(f"No trimmed NPZ in {trimmed_dir}")
    src_npz = trim_files[0]

    # Load trimmed qpos
    src_data = np.load(src_npz, allow_pickle=True)
    qpos = src_data["qpos"].astype(np.float64)  # (T, 43)
    assert qpos.shape[1] == 43, f"Expected 43 DOF, got {qpos.shape}"

    # Update scene object pos/quat to qpos[0, 36:43]
    obj_pos = qpos[0, 36:39]
    obj_quat = qpos[0, 39:43]
    tree = ET.parse(dst_scene)
    root = tree.getroot()
    for body in root.iter("body"):
        if body.attrib.get("name") == "object":
            body.set("pos", " ".join(f"{float(v):.4f}" for v in obj_pos))
            body.set("quat", " ".join(f"{float(v):.6f}" for v in obj_quat))
            break
    tree.write(dst_scene, encoding="unicode")

    # Run forward kinematics to compute contact_pos / qvel
    model = mujoco.MjModel.from_xml_path(str(dst_scene))
    data = mujoco.MjData(model)
    contact_site_ids = []
    for i in range(model.nsite):
        sname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i) or ""
        if "contact" in sname and "hand" in sname:
            contact_site_ids.append(i)
    assert len(contact_site_ids) == 2, f"contact_sites: {contact_site_ids}"

    fps = 30.0
    qvel_list = []
    ctrl_list = []
    contact_list = []
    contact_pos_list = []
    qpos_list = []
    for i in range(qpos.shape[0]):
        data.qpos[:] = qpos[i]
        if i == 0:
            data.qvel[:] = 0
        else:
            mujoco.mj_differentiatePos(model, data.qvel, 1.0 / fps, qpos[i - 1], qpos[i])
        mujoco.mj_forward(model, data)
        ctrl = qpos[i][7:36]  # 29 DOF
        cpos = data.site_xpos[contact_site_ids, :].copy()
        contact = np.ones(len(contact_site_ids))
        qpos_list.append(data.qpos.copy())
        qvel_list.append(data.qvel.copy())
        ctrl_list.append(ctrl)
        contact_list.append(contact)
        contact_pos_list.append(cpos)

    out_npz = case_dir / "0" / "trajectory_kinematic.npz"
    np.savez(out_npz,
             qpos=np.array(qpos_list),
             qvel=np.array(qvel_list),
             ctrl=np.array(ctrl_list),
             contact=np.array(contact_list),
             contact_pos=np.array(contact_pos_list))
    print(f"  derived: {out_npz} (T={qpos.shape[0]})")
    return out_npz


def main():
    out_dir = REPO / "workspace/exp_diagnostic/findings"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_manifest()
    results = {}

    print(f"Processing {len(rows)} D003 box021 cases")
    for row in rows:
        case_task = row["source_task"]
        # Determine base npz to repair
        base_task = case_task  # plain D003 task name
        base_dir = TASKS_DIR / base_task
        base_npz = base_dir / "0" / "trajectory_kinematic.npz"

        # If missing locally, regenerate
        derived = False
        if not base_npz.exists():
            try:
                derive_trajectory_kinematic(base_task, row["seq"], row["person"], row["object"])
                derived = True
            except Exception as e:
                results[case_task] = {"error": f"derive_failed: {e}"}
                print(f"  {case_task}: DERIVE FAILED — {e}")
                continue

        # Repair
        out_task = base_task + "_btop"
        out_path = TASKS_DIR / out_task
        try:
            repair_res = repair_one_case(base_task, out_task, verbose=False)
        except Exception as e:
            results[case_task] = {"error": f"repair_failed: {e}"}
            print(f"  {case_task}: REPAIR FAILED — {e}")
            continue

        # Evaluate gate on BOTH original and repaired
        try:
            orig = evaluate(base_dir)
            new = evaluate(out_path)
            wup_orig = world_up_face_frac(base_dir) or {}
            wup_new = world_up_face_frac(out_path) or {}
        except Exception as e:
            results[case_task] = {"error": f"eval_failed: {e}", "repair_res": repair_res}
            print(f"  {case_task}: EVAL FAILED — {e}")
            continue

        results[case_task] = {
            "derived_from_holosoma": derived,
            "repair_summary": repair_res,
            "original": {
                "gate_pass": orig.get("gate_pass"),
                "gate_reject_reasons": orig.get("gate_reject_reasons"),
                "T": orig.get("T"),
                "pelvis_z_min": orig.get("pelvis_z_min"),
                "L": orig.get("L"),
                "R": orig.get("R"),
                **wup_orig,
            },
            "repaired": {
                "gate_pass": new.get("gate_pass"),
                "gate_reject_reasons": new.get("gate_reject_reasons"),
                "T": new.get("T"),
                "pelvis_z_min": new.get("pelvis_z_min"),
                "L": new.get("L"),
                "R": new.get("R"),
                **wup_new,
            },
        }
        L_in_o = orig["L"]["inside_box_frac"]; R_in_o = orig["R"]["inside_box_frac"]
        L_in_n = new["L"]["inside_box_frac"]; R_in_n = new["R"]["inside_box_frac"]
        wuL = wup_new.get("L_world_up_face_frac", 0); wuR = wup_new.get("R_world_up_face_frac", 0)
        print(f"  {case_task}: derived={derived}  T={orig['T']}  "
              f"R_in {R_in_o*100:.0f}%->{R_in_n*100:.0f}%  "
              f"L_in {L_in_o*100:.0f}%->{L_in_n*100:.0f}%  "
              f"WUface L={wuL*100:.0f}% R={wuR*100:.0f}%  "
              f"gate {orig['gate_pass']}->{new['gate_pass']}  "
              f"reasons={new.get('gate_reject_reasons')}")

    # Compute "near-pass" rank: count of metrics PASSED if we accept world-up-face
    def near_pass_score(r):
        rep = r.get("repaired", {})
        if not rep:
            return -1
        score = 0
        L = rep.get("L", {}); R = rep.get("R", {})
        # 1. inside box <= 10%
        if L.get("inside_box_frac", 1) <= 0.10 and R.get("inside_box_frac", 1) <= 0.10: score += 1
        # 2. signed dist >= 3cm
        if L.get("signed_dist_mean_m", 0) >= 0.03 and R.get("signed_dist_mean_m", 0) >= 0.03: score += 1
        # 3. wrist-below-pelvis <= 30cm
        if L.get("wrist_below_pelvis_gap_m", 1) <= 0.30 and R.get("wrist_below_pelvis_gap_m", 1) <= 0.30: score += 1
        # 4. pelvis_z_min >= 60cm
        if rep.get("pelvis_z_min", 0) >= 0.60: score += 1
        # 5. trajectory_T >= 80
        if rep.get("T", 0) >= 80: score += 1
        # 6. world-up face frac >= 20% on either hand
        wL = rep.get("L_world_up_face_frac", 0)
        wR = rep.get("R_world_up_face_frac", 0)
        if max(wL, wR) >= 0.20: score += 1
        return score

    ranking = sorted(
        [(case_task, near_pass_score(r), r) for case_task, r in results.items() if "error" not in r],
        key=lambda x: (-x[1], x[0]),
    )

    print("\nRanking (out of 6 criteria, world-up-face substituted for top-face):")
    for case_task, score, _ in ranking:
        print(f"  {case_task}: score={score}/6")

    payload = {
        "n_cases": len(rows),
        "n_processed": sum(1 for r in results.values() if "error" not in r),
        "n_errors": sum(1 for r in results.values() if "error" in r),
        "ranking_top10": [{"case": c, "near_pass_score_out_of_6": s} for c, s, _ in ranking[:10]],
        "results": results,
    }
    out_json = out_dir / "08_B_path_gate_results.json"
    out_json.write_text(json.dumps(payload, indent=2))
    print(f"\nsaved {out_json}")


if __name__ == "__main__":
    main()
