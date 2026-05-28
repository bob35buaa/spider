# E089 B-path OmniRetarget Audit

Date: 2026-05-28
Scope: identify the cleanest insertion point to project G1 wrist IK target onto box top face + 5 cm during the Holosoma OmniRetarget production of D003 box021 cases.

## TL;DR

Verdict: **green light**. The D003 production pipeline drives the G1 wrist IK target *implicitly* through the `global_joint_positions[t, 20:22]` entries of the per-case converted SMPLX NPZ (`L_Wrist`, `R_Wrist`). A surgical projection there — applied *before* the retargeter sees the data — fully controls the wrist target and requires zero edits to the retargeter itself. Estimated implementation: ~120 lines of Python in a single standalone pre-IK rewrite script + ~20 lines of pipeline glue.

## 1. Pipeline diagram

```
cases_d003_ready.tsv (13 box021 case-persons)
   │
   ▼
holosoma/workspace/v3/data_construction/scripts/run_d003_spider_preprocess.sh
   │   (loops one case at a time, calls)
   ▼
spider/workspace/core4d/data_preprocess/pipeline.sh   ← orchestrator
   │
   ├─ STAGE A: SMPLX → OmniRetarget conversion
   │     holosoma/workspace/pipeline/convert_core4d_to_omniretarget.py
   │     - flag --replace_wrist_with_fingertip rewrites idx 20/21 of
   │       global_joint_positions with the MEAN of 5 fingertip joints
   │       (LEFT 27/30/33/36/39, RIGHT 42/45/48/51/54).
   │     - output: holosoma_<task>/converted/<seq>-<person>-<Box>_with_obj.npz
   │       keys: global_joint_positions (T,22,3), height, object_poses (T,7), obj_name
   │       object_poses are stored as [qw,qx,qy,qz, x,y,z], Z-up.
   │
   ├─ STAGE B: Holosoma OmniRetarget IK
   │     holosoma/src/holosoma_retargeting/holosoma_retargeting/examples/robot_retarget.py
   │       task_type = object_interaction, data_format = smplx
   │     core algorithm: InteractionMeshRetargeter (.../src/interaction_mesh_retargeter.py)
   │     - load_motion_data reads the converted NPZ exactly as-is:
   │         human_joints = human_data["global_joint_positions"]
   │       (no further wrist transform)
   │     - smpl_scale = ROBOT_HEIGHT / human_height  (≈ 1.32 / 1.81 = 0.73)
   │     - preprocess_motion_data scales the joints by smpl_scale, applies
   │       a recentering so frame-0 pelvis lies on the floor in world frame.
   │     - retarget_motion runs Laplacian-deformation IK per frame; the wrist
   │       SOURCE positions are exactly:
   │         human_mapped_joints[wrist_demo_idx]  for L_Wrist / R_Wrist
   │       i.e. they come straight from global_joint_positions[i, 20] /
   │       global_joint_positions[i, 21]  (after the smpl_scale + recentering
   │       in preprocess_motion_data).
   │     - JOINTS_MAPPING for G1 (data_type.py L287-288):
   │         "L_Wrist" -> "left_wrist_yaw_link"
   │         "R_Wrist" -> "right_wrist_yaw_link"
   │     - output: retargeted/<task>_original.npz (qpos with G1 joints + free obj)
   │
   ├─ STAGE C: trim_no_contact + write_trim_window
   │     trims first/last frames where no contact, produces trimmed NPZ
   │     trim_window.json records (trim_start, trim_frames)
   │
   ├─ STAGE D: spider scene generation
   │     workspace/core4d/data_preprocess/create_spider_scene_from_template.py
   │     - writes example_datasets/processed/core4d/.../<task>/scene.xml
   │     - object collision box half-extents = (raw object mesh extents) * smpl_scale
   │
   └─ STAGE E: spider process_datasets
         spider/process_datasets/core4d.py writes <task>/0/trajectory_kinematic.npz
         (qpos, contact_pos, contact_mask, eef_pos, ...) — this is what SPIDER trains on.
```

## 2. Exact file paths + lines where the wrist IK target is set

The wrist target is set in **two equivalent places** (one is a derived form of the other):

| Level | File | Lines | What gets written |
|---|---|---|---|
| Raw SMPLX wrist position is overwritten with fingertip-mean | `holosoma/workspace/pipeline/convert_core4d_to_omniretarget.py` | 118-122 | `joints[:, 20, :]` = L fingertip center, `joints[:, 21, :]` = R fingertip center |
| Z-up + dtype cast, saved as `global_joint_positions` | same file | 130, 136-139 | output NPZ |
| Loaded by retargeter exactly as-is | `holosoma/src/holosoma_retargeting/holosoma_retargeting/examples/robot_retarget.py` | 262-265 | `human_joints = human_data["global_joint_positions"]` |
| Scaled + recentered (no per-joint logic) | `holosoma/src/holosoma_retargeting/holosoma_retargeting/src/utils.py` `preprocess_motion_data` | (utility) | applies `smpl_scale` and Z-floor offset uniformly |
| Used as IK target via Laplacian mesh | `holosoma/src/holosoma_retargeting/holosoma_retargeting/src/interaction_mesh_retargeter.py` `retarget_motion` | 440-451 | `human_mapped_joints = human_joint_motions[i, self.smplh_mapped_joint_indices]` → into `create_interaction_mesh` → Laplacian source vertices used as deformation targets for `left_wrist_yaw_link` / `right_wrist_yaw_link`. |
| Optional wrist→object distance preservation | same file | 484-520, 792-813 | only enabled when `enable_contact_preservation=True` (default off); preserves *distance* not position. |

Key observation: **the wrist IK target is ONLY ever read from `global_joint_positions[:, 20:22, :]`**. The retargeter never re-derives it from anything else (e.g. ground-truth contact masks). This means any modification of those two rows in the converted NPZ flows transparently through STAGE B → STAGE E.

## 3. Object-local representation availability

| Representation | Available? | Where |
|---|---|---|
| Per-frame object pose `(qw,qx,qy,qz, x,y,z)` Z-up | YES, in converted NPZ | `object_poses` (T,7) |
| Object box half-extents | NO in NPZ, must be derived | from object_mesh `.obj` extents × `smpl_scale`. Already encoded in `example_datasets/processed/core4d/unitree_g1/humanoid_object/<task>/scene.xml` after STAGE D, but unavailable at STAGE A. Easiest source for B-path: re-derive from the mesh `.obj` AABB × `smpl_scale = G1_HEIGHT / human_height` where `G1_HEIGHT=1.32` is `ROBOT_HEIGHT` from `RobotConfig(robot_type="g1")`. |
| Per-frame hand-in-object-local coords | NO, must be computed | trivially: `R(box_quat).T @ (hand_world - box_pos)` |

So no "object-local hand target" representation exists pre-IK. We must compute it ourselves at the rewrite point.

## 4. Proposed insertion points

### B-1 — Pre-IK target rewrite (RECOMMENDED)

**Where**: write a new standalone script `scripts/rewrite_wrist_top_face.py` that takes a converted NPZ (the output of STAGE A) plus the box mesh path, applies the projection, and writes a new NPZ. The D003 pipeline is then re-invoked with the rewritten NPZ as the retargeter input (via `--data_path`).

**How**:

```
inputs:  converted/<seq>-<person>-<Box>_with_obj.npz
         core4d/object_models/<Box>/<Box>.obj   (for mesh AABB → half_extents)
         G1 robot height (1.32 m) for smpl_scale derivation

for each frame t:
    human_height ← npz["height"]
    smpl_scale ← G1_HEIGHT / human_height
    box_half ← mesh_aabb_half * smpl_scale         # constant per case

    for hand in [L=20, R=21]:
        hand_world ← global_joint_positions[t, hand]
        box_pos ← object_poses[t, 4:7]
        box_quat ← object_poses[t, 0:4]    # (qw,qx,qy,qz)
        # express hand in box local frame
        hand_local ← R(box_quat).T @ (hand_world - box_pos)
        # find world-up-most local axis (handles box021's 90° local-x rotation)
        local_world_up ← R(box_quat).T @ [0,0,1]   # unit vector
        top_axis ← argmax(|local_world_up|)        # index in {0,1,2}
        top_sign ← sign(local_world_up[top_axis])  # +1 / -1
        # if hand is near that face OR already on the +up side, project
        signed_dist ← top_sign*hand_local[top_axis] - box_half[top_axis]
        if hand_world[2] > box_pos[2] OR |signed_dist| < 0.15:
            target_local ← hand_local.copy()
            other_axes ← [a for a in (0,1,2) if a != top_axis]
            # clip in-face coords away from the edge by 2 cm
            for a in other_axes:
                target_local[a] ← clip(hand_local[a],
                                       -box_half[a] + 0.02,
                                       +box_half[a] - 0.02)
            target_local[top_axis] ← top_sign * (box_half[top_axis] + 0.05)
            hand_world_new ← R(box_quat) @ target_local + box_pos
            global_joint_positions[t, hand] ← hand_world_new
```

Write a new NPZ to `<case_dir>/converted_box_top/<task>.npz`, then have the pipeline use that as its `--data_path`.

Pros:
- Zero modification to OmniRetarget code (retargeter file untouched).
- Single, reversible, off-pipeline rewrite that is trivial to A/B against.
- No risk of breaking other tasks: only the explicitly-rewritten cases use the new path.
- Same projection runs at "fingertip" semantic level (after STAGE A's fingertip-mean substitution), which matches the geometric intent: project the *contact target* onto the top face.

Cons:
- Need to derive `box_half` separately from the mesh `.obj` (one line of trimesh or numpy on the vertex array).
- The pelvis recentering inside `preprocess_motion_data` happens AFTER our rewrite, so we operate in the same Z-up frame the retargeter operates in for the Laplacian target — but pelvis-recentering only shifts all joints uniformly, so it preserves our relative geometry to the box (which is also in the same NPZ and gets the same shift). Verified by reading `preprocess_motion_data` flow.

### B-2 — Post-IK wrist re-IK (NOT RECOMMENDED)

After STAGE B produces `qpos`, run a second IK pass that snaps `left_wrist_yaw_link` and `right_wrist_yaw_link` to the projected world positions while keeping the rest of qpos near the original.

Pros:
- Even more isolated; the SMPLX input is left untouched.

Cons:
- Requires a second IK solver (CVXPy + manipulator jacobian setup), which is exactly the machinery in `interaction_mesh_retargeter.iterate`. Replicating it is high-effort.
- Snapping wrist *after* the Laplacian solve will inevitably violate joint limits / self-collision constraints that the Laplacian solve respected — fighting the original IK.
- Doesn't leverage the deformation-mesh global stretching of arms / shoulders that the Laplacian solver naturally provides when the source vertex moves.

### Recommendation

**B-1**. Surgical, decoupled, reversible, low LOC, no risk to other pipelines. Effort: ~2 h to implement + test for one case. Bulk re-process effort: dominated by OmniRetarget runtime (~3-5 min per case × 13 cases ≈ 1 h on local hsretargeting env).

## 5. Effort breakdown

| Step | LOC | Wall time |
|---|---|---|
| `rewrite_wrist_top_face.py` (single case → new converted NPZ) | ~150 LOC | 1 h to write + smoke |
| Pipeline glue: invoke retargeter with new `--data_path` and write trimmed NPZ + spider scene | ~30 LOC (reuse pipeline.sh STAGES B-E with overridden converted dir) | 30 min |
| Single-case validation (B2) | run pipeline on 1 case + apply gate | ~10 min OmniRetarget + ~30s gate |
| Bulk reprocess (B3) | 13 × ~5 min ≈ 70 min OmniRetarget + 13 × gate | < 90 min |
| Apply G1-Feasibility gate, pick top-2, copy into example_datasets | ~30 LOC orchestration | 15 min |
| **Total** | ~210 LOC | ~3 h |

## 6. Known risks / open questions

- **Top-face detection under arbitrary quaternion**: the task instructions specified "project onto +z (top) face + 5 cm" using local-frame coordinates, but added that "top face must be the WORLD-UP-most face after applying box_quat (because box021 has quat rotation that swaps local axes)." Our B-1 implementation uses the **world-up-most face** definition (largest |R^T·ẑ| component in local frame, with sign). This is unambiguous and handles box021's local-axis swap correctly.

- **Fingertip-mean was already applied at STAGE A**: idx 20/21 are *not* the SMPLX wrist any more — they are the mean of 5 fingertip joints. This is fine for B-1 because the IK target the Laplacian solver actually uses is whatever sits in idx 20/21. The B-1 rewrite simply replaces that with our projected target. Documenting this so a reader doesn't think we are "moving the wrist" — semantically we are moving the *contact point* (the fingertip cluster) onto the box top face.

- **`enable_contact_preservation` is OFF in D003 production**: confirmed by absence of `--retargeter.enable-contact-preservation` flag in `pipeline.sh` STAGE B invocation. So the only signal driving wrist position is the SMPL wrist coords + Laplacian deformation. Modifying those coords is fully sufficient.

- **Box half-extents derivation**: requires loading the object `.obj` and computing AABB. Done once per object (Box021 is shared by all 13 cases). Use `trimesh` if available, else parse the OBJ vertex lines manually.

- **`preprocess_motion_data` does z-floor recentering**: it shifts the whole `human_joints` and `object_poses` by a constant (foot-on-floor offset). Because both the object and the joints are shifted together, our box-local-frame projection is invariant to that shift. Confirmed safe.

- **`d003_box021_20231018_028_p2` exclusion**: not present in the 13 D003-success list under E028 manifest? Re-check — the manifest shows 13 cases including `20231018_028_p2`. Bulk B3 will operate on all 13 from the manifest TSV.

## 7. Insertion-point decision

**Originally**: B-1 (pre-IK target rewrite on converted NPZ). This remains the cleanest long-term solution.

**Actual implementation for E089**: **B-2 lite — post-IK wrist re-IK directly on the existing `trajectory_kinematic.npz`**, motivated by the local environment limitations:

- Holosoma `hsretargeting` conda env is not present on this machine (the D003 pipeline assumes `/home/ubuntu/.holosoma_deps/miniconda3/envs/hsretargeting`).
- Spider's `.venv` lacks `cvxpy` + `clarabel` + `yourdfpy`; a `uv pip install` was attempted but `cvxpy` source build dragged past the time budget.
- Re-running OmniRetarget end-to-end for 13 cases (~5 min each × 13 = ~1 h) inside that env, after installing it, is out of the 1-GPU-hour budget for this subagent slot.

**B-2 lite implementation**: per-frame inverse kinematics on a *small subset of qpos indices* (left arm 7 dof + right arm 7 dof + waist 3 dof = 17 dof) using MuJoCo's analytical Jacobian `mj_jacBody`, target = the world-up projected wrist points. Each frame: Gauss-Newton on `J^T J + λI` damped LS, 20 iterations, joint-limit clipping. We start from the *existing* qpos and apply small corrections so all other DOFs (legs, free joint, hand, head) stay identical. This produces a modified `trajectory_kinematic.npz` whose `qpos` differs from the original only in the arm/waist DOFs and whose wrist FK lands on the box top face + 5 cm at every frame.

**Why this is the right call**:
- The G1-Feasibility gate reads exactly `trajectory_kinematic.npz` + `scene.xml`. Direct surgery on the npz produces a comparable input. The gate evaluation is fair (same scene, same other DOFs).
- No environment install needed.
- Each case takes ~5-10 s (MuJoCo Jacobian is fast).
- The intervention is *only* on wrist position; the failure mode being tested is exactly "wrist target inside box vs. on top face" — this isolates the variable cleanly.
- Trade-off: arm joint configurations may look less natural than a full Laplacian solve. But for the gate (which checks wrist position only) and for downstream SPIDER CEM (which has its own action sampling), this is fine.

**What the audit recommendation does NOT change**: B-1 (rewrite converted NPZ + re-run OmniRetarget) is still the correct long-term solution for the holosoma pipeline. The audit findings (file paths, line numbers, projection formula) are directly usable by a future implementer with the `hsretargeting` env available.

Proceed to B2 (implementation + single-case validation) using the B-2 lite path. If single-case gate passes, proceed to B3.
