# E061 Plan: Sphere Baseline Verification — Validate Box025 3-box Regression Diagnosis

## Context

Log 74 (`workspace/core4d/log/74_box025_3box_regression_and_E060_invalidation.md`) found that the box025 calibration case **regressed** after the 3-box hand port (commit `fa2e181`):

| Run | Hand collision | pelvis_min | Stable % |
|-----|----------------|-----------|---------|
| Historical sphere E041c (E048 video) | sphere @ wrist+10cm, r=5cm | **0.575m** | **100%** |
| New 3-box e041c | 3 boxes (wrist cuff + palm + finger pad to wrist+17.5cm) | **0.253m** | **61.7%** |
| New 3-box e041 | same | 0.193m | 69.2% |
| New 3-box e039 (no ori reward) | same | **0.156m** | **60.8%** |

**Even E039 (zero ori reward) regressed** → reward stack is NOT the cause. The diagnosis (log 74 §2) points to **3-box geometry + reward `contact_hdmi_eef_offset = [0.05,0,0]` mismatch**: box3 tip extends to wrist+17.5cm but reward looks at wrist+5cm — a 12.5cm error. CEM finds high-reward poses where box3 physically punts the object away.

**This invalidates E060.0/.1/.2 conclusions** — every "reward task-specific" finding in logs 71/72/73 is contaminated by a physics bug. E061 is the **mandatory regression test** (log 74 §5.1 rule) that should have run BEFORE any reward ablation on the new 3-box geometry.

## Goal

**One single, controlled question**: Does temporarily reverting hand collision to the sphere version (commit `fa2e181~1`) recover the historical box025 + E041c baseline (`pelvis_min ≥ 0.50m`, `Stable ≥ 90%`)?

The answer cleanly partitions the failure space (log 74 §4.2):

| Sphere box025 + E041c result | Diagnosis | Next phase |
|-----------------------------|-----------|-----------|
| `pelvis_min ≥ 0.50m`, `Stable ≥ 90%` | **3-box port is the SOLE regression source** | Decide B (fix 3-box geometry/eef_offset) vs A (revert sphere) |
| `0.30m ≤ pelvis_min < 0.50m` | Partial — additional confound | `git bisect` between b190785 and fa2e181 |
| `pelvis_min < 0.30m` | Sphere also fails — deeper bug | `git bisect` further back; may implicate E058 warmstart hook or other recent infra changes |

## Phase 1: Sphere Checkout + Smoke Test (~5 min)

### 1A. Snapshot E061 scenes BEFORE checkout (per dual-safeguard rule §10b)

```bash
# Snapshot the CURRENT (3-box) state first, so we have an audit trail of what we're temporarily reverting from.
bash workspace/core4d/scripts/convert/snapshot_scenes.sh E061_pre_checkout box025_person1
```

### 1B. Checkout sphere version of robot.xml + box025 scenes

```bash
git checkout fa2e181~1 -- \
  spider/assets/robots/unitree_g1/robot.xml \
  example_datasets/processed/core4d/unitree_g1/humanoid_object/box025_person1/scene.xml \
  example_datasets/processed/core4d/unitree_g1/humanoid_object/box025_person1/scene_act.xml
```

### 1C. Snapshot E061 scenes AFTER checkout (sphere state — what training will actually run on)

```bash
bash workspace/core4d/scripts/convert/snapshot_scenes.sh E061 box025_person1
```

### 1D. Smoke test: verify sphere geometry restored

```bash
.venv/bin/python -c "
import mujoco
m = mujoco.MjModel.from_xml_path('example_datasets/processed/core4d/unitree_g1/humanoid_object/box025_person1/scene_act.xml')
hand = sorted([mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i)
               for i in range(m.ngeom)
               if mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i)
               and mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i).startswith(('lh','rh'))])
print(f'hand geoms: {hand}')        # expect ['lh', 'rh']  (length 2, not 6)
print(f'npair: {m.npair}')          # expect ~26 (sphere era)
print(f'lh size: {m.geom_size[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, \"lh\")]}')  # expect [0.05, 0, 0] (sphere r=0.05)
"
```

If geometry is wrong → STOP, manually inspect & abort. Do NOT proceed to training.

## Phase 2: Run box025 + E041c on sphere geometry (~25 min wall, 1 GPU)

### 2A. Train script

**File**: `workspace/core4d/scripts/train/train_E061.sh` (new, minimal — single case, single config, sphere)

Skeleton:
```bash
#!/usr/bin/env bash
# E061: sphere baseline verification — does box025 + E041c recover historical 0.575m pelvis_min on sphere hand?
# REQUIRES: Phase 1 already ran (sphere robot.xml + box025 scenes checked out).
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

GPU="${1:-0}"
RESULTS=workspace/core4d/results/E061
LOGS=logs/E061
mkdir -p "$RESULTS" "$LOGS"

# Snapshot already done in Phase 1C — skip re-snapshot here.
echo "[$(date '+%H:%M:%S')] === E061 box025 + E041c sphere ==="
out_dir="$RESULTS/box025_e041c_sphere_outdir"
mkdir -p "$out_dir"
CUDA_VISIBLE_DEVICES=$GPU MUJOCO_GL=egl PYTHONUNBUFFERED=1 .venv/bin/python -u examples/run_mjwp.py \
  +override=core4d_e041c \
  task=box025_person1 \
  +use_torch_compile=false \
  output_dir="$out_dir" \
  video_output_path="$RESULTS/box025_e041c_sphere.mp4" \
  > "$LOGS/box025_e041c_sphere.log" 2>&1
cp "$out_dir/trajectory_mjwp_act.npz" "$RESULTS/box025_e041c_sphere.npz"
echo "[$(date '+%H:%M:%S')] done. npz/mp4/log under $RESULTS / $LOGS"
```

### 2B. Run

```bash
bash workspace/core4d/scripts/train/train_E061.sh 0
```

## Phase 3: Evaluate vs Historical Baseline (~5 min)

### 3A. Quantitative compare

```bash
.venv/bin/python -c "
import numpy as np
d = np.load('workspace/core4d/results/E061/box025_e041c_sphere.npz', allow_pickle=True)
qpos = d['qpos']
qpos = qpos[:,0,:] if qpos.ndim==3 else qpos
pz = qpos[:,2]
print(f'pelvis_min : {pz.min():.3f}m   (historical sphere E041c: 0.575m)')
print(f'pelvis_mean: {pz.mean():.3f}m   (historical: ~0.7m)')
print(f'stable %   : {(pz>=0.5).mean()*100:.1f}%   (historical: 100%)')
print(f'frames     : {len(pz)}')
"
```

### 3B. Visual keyframe extraction (5 frames evenly spaced)

Use `/video-frames` skill on `workspace/core4d/results/E061/box025_e041c_sphere.mp4` → output to `workspace/core4d/results/E061/keyframes/`. Confirm sim is doing real bimanual carry (matches sphere-era E048 description: "弯腰前倾, 手在箱顶, 手背朝向物体"), not 3-box-era pathological behavior (T-pose, dropped object).

### 3C. Decision per Goal table

Apply log 74 §4.2 judgment table to `pelvis_min`. The decision determines the next plan/log to write — but **do NOT plan B/A here**, defer per user instruction.

## Phase 4: Restore 3-box state (mandatory, before any commit)

```bash
git checkout HEAD -- \
  spider/assets/robots/unitree_g1/robot.xml \
  example_datasets/processed/core4d/unitree_g1/humanoid_object/box025_person1/scene.xml \
  example_datasets/processed/core4d/unitree_g1/humanoid_object/box025_person1/scene_act.xml
```

Verify hand geom is back to `lh/lh2/lh3/rh/rh2/rh3` (6 boxes) before proceeding to commit. Otherwise the commit could leave the working tree in mixed sphere/3-box state.

## Phase 5: Log + Commit

### 5A. Write log

**File**: `workspace/core4d/log/75_E061_sphere_baseline_verify.md`

Must include:
- Numerical comparison table (sphere E061 vs historical sphere E048 baseline)
- 5-keyframe visual description (filled, not "TBD" — per visualization rule §9)
- Decision per §3C
- Pointer to `74_box025_3box_regression_and_E060_invalidation.md` as motivation
- "改动文件" table including snapshot path + train script + log

### 5B. Update `EXPERIMENT_TRACKER.md`

- Add E061 row with sphere-vs-3-box result
- Add log 75 to log index
- If diagnosis confirmed: add a "Phase 18 entry: 3-box port confirmed as regression source; reward ablation deferred until physics fix"

### 5C. Commit

Format (per project rule §11):
```
exp(core4d): E061 sphere baseline verification — [PASS/PARTIAL/FAIL] confirms 3-box port is regression source

- Temporarily checked out fa2e181~1 robot.xml + box025 scenes; ran box025 + E041c.
- Result: pelvis_min = X.XXm (vs historical sphere 0.575m, vs new 3-box 0.253m).
- Decision: [next step].
- 3-box state restored before commit.
```

## Critical Files

### Modify (temporarily, then restore)
- `spider/assets/robots/unitree_g1/robot.xml`
- `example_datasets/processed/core4d/unitree_g1/humanoid_object/box025_person1/scene.xml`
- `example_datasets/processed/core4d/unitree_g1/humanoid_object/box025_person1/scene_act.xml`

### Create
- `workspace/core4d/plan/71_E061_sphere_baseline_verify_plan.md` — copy this plan
- `workspace/core4d/scripts/train/train_E061.sh` — Phase 2A
- `workspace/core4d/results/E061/scene_snapshot/` — Phase 1C (sphere snapshot)
- `workspace/core4d/results/E061_pre_checkout/scene_snapshot/` — Phase 1A (3-box snapshot, audit trail)
- `workspace/core4d/results/E061/box025_e041c_sphere.{npz,mp4,log}` — Phase 2B output
- `workspace/core4d/results/E061/keyframes/*.jpg` — Phase 3B
- `workspace/core4d/log/75_E061_sphere_baseline_verify.md` — Phase 5A

### Reuse (no modification)
- `examples/config/override/core4d_e041c.yaml` — same reward stack as historical
- `workspace/core4d/scripts/convert/snapshot_scenes.sh` — Phase 1A/C
- `examples/run_mjwp.py` — entry point

## Verification (end-to-end)

```bash
# After Phase 1
ls workspace/core4d/results/E061/scene_snapshot/box025_person1/
ls workspace/core4d/results/E061_pre_checkout/scene_snapshot/box025_person1/
# both should exist with manifest.txt

# After Phase 2
ls -lh workspace/core4d/results/E061/box025_e041c_sphere.{npz,mp4}
tail -30 logs/E061/box025_e041c_sphere.log  # check no errors, sees "trajectory saved"

# After Phase 3
cat workspace/core4d/results/E061/keyframes/*.jpg | wc -c   # 5 keyframes present

# After Phase 4
.venv/bin/python -c "
import mujoco
m = mujoco.MjModel.from_xml_path('example_datasets/processed/core4d/unitree_g1/humanoid_object/box025_person1/scene_act.xml')
hand = sorted([mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i)
               for i in range(m.ngeom)
               if mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i)
               and mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i).startswith(('lh','rh'))])
assert hand == ['lh', 'lh2', 'lh3', 'rh', 'rh2', 'rh3'], f'restore failed: got {hand}'
print('3-box restored: OK')
"

# After Phase 5
git log -1 --stat
```

## Risks & Mitigations

| Risk | Probability | Mitigation |
|------|-------------|-----------|
| Forget Phase 4 restore → commit mixed state | Medium | Phase 4 is a hard gate before Phase 5; Phase 5 verification asserts 6-box hand |
| Sphere also fails (`< 0.30m`) | Low (historical data is solid) | Plan §3C judgment table → next step is `git bisect` to fa2e181..b190785 range |
| `core4d_e041c.yaml` itself was modified between historical sphere E048 and now | Low | Verify with `git log examples/config/override/core4d_e041c.yaml` — if changed, also need to checkout that version |
| CEM noise gives one bad seed | Low | Single-seed run is fine for first pass; if result is borderline (0.45-0.55m), re-run with different seed before deciding |
| Other dependencies (e.g., simulator code) regressed since fa2e181~1 | Low | If sphere result is partial (0.30-0.50m), bisect simulator-side commits |

## Out of Scope (defer to follow-up plans)

- **Choosing B vs A** (3-box geometry fix vs sphere revert) — explicitly deferred per user instruction; will plan after E061 result is in
- **Other case sphere verification** (box023, bucket005_s2) — only box025 was the calibration case with known historical baseline; expanding the regression test to other cases is for E062+
- **Re-doing E060.0/.1/.2** in fixed physics environment — for E063+ once fix path is chosen
- **E060.3 stability_penalty experiment** — abandoned (reward ablation moot until physics fixed)
- **`contact_hdmi_eef_offset` re-tuning** — that's part of B path, not E061
- **`git bisect`** — only triggered if E061 fails to recover sphere baseline

## Estimated Cost

| Phase | Time | GPU |
|-------|------|-----|
| Phase 1 (snapshot + checkout + smoke) | 5 min | 0 |
| Phase 2 (train) | ~25 min wall | 1 GPU |
| Phase 3 (eval + keyframes) | 5 min | 0 |
| Phase 4 (restore) | 1 min | 0 |
| Phase 5 (log + tracker + commit) | 15 min | 0 |
| **Total** | **~50 min** | 1 GPU |

## Success Definition

- **Mandatory**: Phase 4 leaves working tree with 3-box hand restored (no regressions to other branches that depend on robot.xml current state).
- **Primary**: Phase 3 produces a clear decision per §3C — either confirms 3-box port as sole regression source (clean signal for B/A choice next) or surfaces additional confounds (triggers bisect).
- **Bonus**: If sphere recovers cleanly, the keyframes provide a visual gold standard for what "good box025 behavior" looks like — useful baseline for evaluating future B path fixes.
