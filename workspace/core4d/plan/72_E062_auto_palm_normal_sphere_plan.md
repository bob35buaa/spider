# E062 Plan: Auto Palm Normal — Sphere-based Case Generalization (X1, box025+box023)

## Context

Log 76 + 77 已经定位了 spider 项目 1 年来的核心问题: **E041c reward stack 完全是 box025+sphere 的过拟合, 没有泛化性**. 4-cell 矩阵 (sphere/3-box × box025/box023) 中只有 box025+sphere 一个组合 work, 因为它的两个 hardcoded reward 常数 (`palm_normal=[0,∓1,0]` for box025 motion, `eef_offset=[0.05,0,0]` for sphere geometry) 恰好都对.

E041c box025 真实视觉问题 (姿态扭曲) 经 log 77 重审定位为 **臂展物理硬限制 + body tracking + 下半身控制** (P1-P4), 跟 hand collision shape 关系不大. **3-box 唯一独有价值是跨 framework 一致性**, 在 spider 内部解决泛化完全可以基于 sphere. 用户决定 P1-P4 暂不修, 优先攻击 P5 (case 泛化), 改为基于 sphere 开发以避免 3-box × eef_offset 的额外 confound.

E062 是泛化方向的第一步: **X1 = per-case auto-derived palm_normal from ref motion**, 替代 hardcoded `[0,∓1,0]` (box025 fingerprint). 在 sphere 几何上验证 box025 baseline 不退化 + box023 (历史 sphere 下 0.193m 摔) 能否站住. 通过 → reward 真有泛化性, 推到剩余 4 个 B+C case (E063+).

## Goal

**单一可验证问题**: 用 ref motion 自动算出 per-case `palm_normal` 替代 hardcoded box025 fingerprint, 在 sphere geometry 下能否同时让 (1) box025 不退化 (≥ 0.6m) 和 (2) box023 站住 (≥ 0.5m)?

## Decision Tree

| 结果 | 解读 | 下一步 |
|------|------|--------|
| box025 ≥ 0.6m AND box023 ≥ 0.5m | **X1 成功**, reward 真有泛化性 | E063: 推到 bucket005_s2 / bucket007 / bucket001 / desk021 |
| box025 ≥ 0.6m AND box023 < 0.5m | X1 单独不够, 可能 box023 还有别的问题 (比如 eef_offset 在 box023 上也需要 X2) | 加 X2 (auto eef_offset) 或诊断 box023 reward 其他 case-specific 部分 |
| box025 < 0.6m | X1 破坏了 sphere baseline (说明 box025 的 hardcoded palm_normal 被 X1 算错了) | 检查 X1 算法 — 算出来的 box025 palm_normal 是否真等于 `[0,∓1,0]` |

## Phase 1: Sphere Revert (~10min, 写新 commit)

### 1A. 局部 revert robot.xml + 9 case scenes 到 fa2e181~1

```bash
# robot.xml: 全 revert (fa2e181 唯一改动就是 hand)
git checkout fa2e181~1 -- spider/assets/robots/unitree_g1/robot.xml

# 9 case scenes: fa2e181 改动了 hand inline geoms + box023 margin
# checkout fa2e181~1 版本会 revert hand 到 sphere, 同时 revert box023 margin 1.05→0.90 (我们要再加回来)
git checkout fa2e181~1 -- example_datasets/processed/core4d/unitree_g1/humanoid_object/{box025_person1,box023_person1,box021_person1,bucket001_person1,bucket005_s2_person1,bucket007_person1,desk021_person1}/{scene.xml,scene_act.xml}
# 注意: fa2e181 新增了一些 case (b190785 force-add 9 case 入 git, fa2e181 patch 这些 case), 上面列出的是确认存在的; 实际跑前用 `git diff fa2e181~1..fa2e181 --name-only example_datasets/processed/core4d/unitree_g1/humanoid_object/` 取完整列表
```

### 1B. 重新应用 box023 margin 0.90 (用现有工具)

```bash
.venv/bin/python workspace/core4d/scripts/convert/set_collision_margin.py \
    --cases box023_person1 --margin 0.90
```

### 1C. Smoke test: 验证 sphere + box023 margin 0.90 状态

```bash
.venv/bin/python -c "
import mujoco
for case in ['box025_person1', 'box023_person1']:
    m = mujoco.MjModel.from_xml_path(f'example_datasets/processed/core4d/unitree_g1/humanoid_object/{case}/scene_act.xml')
    hand = sorted([mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i)
                   for i in range(m.ngeom)
                   if mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i)
                   and mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i).startswith(('lh','rh'))])
    obj_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, 'object_collision')
    obj_size = m.geom_size[obj_id] if obj_id >= 0 else None
    print(f'{case}: hand={hand}, object_collision size={obj_size}')
# 期望:
#   box025: hand=['lh', 'rh'] (sphere), object_collision size 不变
#   box023: hand=['lh', 'rh'] (sphere), object_collision half-size ~ (0.153, 0.157, 0.177) (margin 0.90)
"
```

### 1D. 新 commit

```
infra(core4d): revert hand collision to sphere, keep box023 margin 0.90

- robot.xml + 7 case scene*.xml: revert hand from 3-box back to sphere
  (fa2e181~1 version), as 3-box port introduced 12.5cm reward/physics
  misalignment (log 74) and offers no unique value for spider-internal
  case generalization (log 77 §4).
- box023 margin re-applied to 0.90x via set_collision_margin.py (E053 result).
- Sphere geometry confirmed work for box025 (E061: pelvis_min 0.672m); now
  pivoting to attack reward generalization (X1 = auto palm_normal, E062).
- 3-box port (commit fa2e181) preserved in git history for future
  reconsideration if cross-framework consistency with HDMI/Holosoma needed.
```

## Phase 2: Implement `compute_palm_normal.py` (~1h)

### 2A. 新脚本

**File**: `workspace/core4d/scripts/convert/compute_palm_normal.py`

**Algorithm** (per audit §2.2 + Explore agent recommendation):

```python
import mujoco
import numpy as np
import yaml
from pathlib import Path

def compute_palm_normal_for_case(
    case_name: str,
    scene_xml: str,                  # path to scene_act.xml
    ref_npz: str,                    # path to trajectory_kinematic_dual.npz or _kinematic.npz
    obj_body_name: str = "object",
    wrist_body_names: tuple = ("left_wrist_yaw_link", "right_wrist_yaw_link"),
    proximity_threshold: float = 0.30,  # rough intent-window proxy
) -> dict[str, list[float]]:
    """Returns {'left': [x,y,z], 'right': [x,y,z]} in wrist-local frame."""
    m = mujoco.MjModel.from_xml_path(scene_xml)
    d = mujoco.MjData(m)
    obj_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, obj_body_name)
    wrist_bids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in wrist_body_names]

    qpos_ref = np.load(ref_npz)['qpos']  # (T, nq)

    # Per-frame: forward, get wrist xpos/xmat, obj xpos
    # Then compute approach mask (wrist-obj dist < threshold) and dot products
    candidate_axes = np.array([
        [+1,0,0],[-1,0,0],[0,+1,0],[0,-1,0],[0,0,+1],[0,0,-1]
    ], dtype=float)

    dot_sums = {bid: np.zeros(6) for bid in wrist_bids}
    counts = {bid: 0 for bid in wrist_bids}

    for t in range(qpos_ref.shape[0]):
        d.qpos[:m.nq] = qpos_ref[t, :m.nq]
        mujoco.mj_kinematics(m, d)
        obj_pos = d.xpos[obj_bid].copy()
        for bid in wrist_bids:
            wrist_pos = d.xpos[bid]
            wrist_mat = d.xmat[bid].reshape(3, 3)
            dir_to_obj = obj_pos - wrist_pos
            dist = np.linalg.norm(dir_to_obj)
            if dist > proximity_threshold:
                continue
            dir_unit = dir_to_obj / max(dist, 1e-6)
            for i, axis_local in enumerate(candidate_axes):
                axis_world = wrist_mat @ axis_local
                dot_sums[bid][i] += float(np.dot(axis_world, dir_unit))
            counts[bid] += 1

    # Pick max-mean axis per side; emit unit vector in wrist-LOCAL frame
    result = {}
    for side, bid in zip(["left", "right"], wrist_bids):
        n = counts[bid]
        if n == 0:
            print(f"[WARN] {case_name} {side}: no frames within {proximity_threshold}m, falling back to [0,0,0] (no prior)")
            result[side] = [0.0, 0.0, 0.0]
            continue
        means = dot_sums[bid] / n
        best_idx = int(np.argmax(means))
        result[side] = candidate_axes[best_idx].tolist()
        print(f"[{case_name}] {side} wrist: {n} frames; means {means.round(3)}; chose axis {result[side]} (dot={means[best_idx]:.3f})")
    return result

def write_yaml_override(case_name: str, palm_normal: dict, out_dir: str):
    """Writes examples/config/override/core4d_e062_<case>.yaml inheriting E041c, overriding only palm_normals."""
    yaml_path = Path(out_dir) / f"core4d_e062_{case_name.replace('_person1','')}.yaml"
    content = f"""# @package _global_
# E062 X1: auto-derived palm_normal for {case_name} (sphere geometry, E041c base).
# Generated by workspace/core4d/scripts/convert/compute_palm_normal.py.
# Method: ref-motion proximity-windowed dot product (audit log 70 §2.2 algorithm).
defaults:
  - core4d_e041c
  - _self_

contact_hdmi_palm_normal_left:  {palm_normal['left']}
contact_hdmi_palm_normal_right: {palm_normal['right']}
"""
    yaml_path.write_text(content)
    print(f"Wrote {yaml_path}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="+", required=True)
    ap.add_argument("--out-dir", default="examples/config/override")
    args = ap.parse_args()

    for case in args.cases:
        scene_xml = f"example_datasets/processed/core4d/unitree_g1/humanoid_object/{case}/scene_act.xml"
        # Try dual then single ref motion
        for ref_name in ["trajectory_kinematic_dual.npz", "trajectory_kinematic.npz"]:
            ref_npz = f"example_datasets/processed/core4d/unitree_g1/humanoid_object/{case}/0/{ref_name}"
            if Path(ref_npz).exists():
                break
        result = compute_palm_normal_for_case(case, scene_xml, ref_npz)
        write_yaml_override(case, result, args.out_dir)
```

### 2B. 验证 box025 算出 `[0,-1,0]/[0,1,0]` (跟 audit §2.2 匹配)

```bash
.venv/bin/python workspace/core4d/scripts/convert/compute_palm_normal.py --cases box025_person1
# 期望输出:
#   [box025_person1] left  wrist: ~50 frames; means [..., -y_dot=0.78~0.80, ...]; chose axis [0,-1,0]
#   [box025_person1] right wrist: ~50 frames; means [..., +y_dot=0.78~0.80, ...]; chose axis [0,1,0]
# 写出 examples/config/override/core4d_e062_box025.yaml
```

如果 box025 算出来不是 `[0,∓1,0]`, 算法有 bug 必须修, 不能继续 (因为这意味着算法跟 audit §2.2 不一致, 历史 baseline 会破).

### 2C. 算 box023

```bash
.venv/bin/python workspace/core4d/scripts/convert/compute_palm_normal.py --cases box023_person1
# 期望: 跟 audit §2.2 一致 — box023 L/R 都应该是 [+1,0,0] 或某个新方向 (看脚本算什么)
# 写出 examples/config/override/core4d_e062_box023.yaml
```

## Phase 3: Train E062 (~30min wall, 2 GPU 并行)

### 3A. Train script

**File**: `workspace/core4d/scripts/train/train_E062.sh`

```bash
#!/usr/bin/env bash
# E062: X1 auto palm_normal on sphere — box025 (baseline) + box023 (target).
# REQUIRES: Phase 1 (sphere revert) + Phase 2 (yamls generated) done.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

MODE="${1:-parallel}"
GPU_BOX025="${2:-0}"
GPU_BOX023="${3:-1}"
RESULTS=workspace/core4d/results/E062
LOGS=logs/E062
mkdir -p "$RESULTS" "$LOGS"

bash workspace/core4d/scripts/convert/snapshot_scenes.sh E062 box025_person1 box023_person1

run_one() {
  local name=$1 task=$2 override=$3 gpu=$4
  local out_dir="$RESULTS/${name}_outdir"
  mkdir -p "$out_dir"
  echo "[$(date '+%H:%M:%S')] === $name (override=$override, GPU $gpu) ==="
  CUDA_VISIBLE_DEVICES=$gpu MUJOCO_GL=egl PYTHONUNBUFFERED=1 .venv/bin/python -u examples/run_mjwp.py \
    +override=$override task=$task +use_torch_compile=false \
    output_dir="$out_dir" \
    video_output_path="$RESULTS/${name}.mp4" \
    > "$LOGS/${name}.log" 2>&1
  cp "$out_dir/trajectory_mjwp_act.npz" "$RESULTS/${name}.npz"
}

if [[ "$MODE" == "parallel" ]]; then
  run_one E062_box025_sphere_autopalm box025_person1 core4d_e062_box025 "$GPU_BOX025" &
  PID0=$!
  run_one E062_box023_sphere_autopalm box023_person1 core4d_e062_box023 "$GPU_BOX023" &
  PID1=$!
  wait $PID0; wait $PID1
else
  run_one E062_box025_sphere_autopalm box025_person1 core4d_e062_box025 "$GPU_BOX025"
  run_one E062_box023_sphere_autopalm box023_person1 core4d_e062_box023 "$GPU_BOX023"
fi
```

### 3B. Run

```bash
bash workspace/core4d/scripts/train/train_E062.sh parallel 0 1
```

## Phase 4: Evaluate (~10min)

### 4A. Quantitative

```bash
.venv/bin/python -c "
import numpy as np
for name, hist in [('box025', 0.672), ('box023', 0.193)]:
    d = np.load(f'workspace/core4d/results/E062/E062_{name}_sphere_autopalm.npz', allow_pickle=True)
    qpos = d['qpos']
    qpos = qpos[:,0,:] if qpos.ndim==3 else qpos
    pz = qpos[:,2]
    print(f'{name}: pelvis_min={pz.min():.3f}m (hist {hist}m), mean={pz.mean():.3f}m, stable={(pz>=0.5).mean()*100:.1f}%')
"
```

### 4B. Visual keyframes (5 per case)

```bash
mkdir -p workspace/core4d/results/E062/keyframes
for case in box025 box023; do
  for t in 0.5 1.5 2.5 3.5 4.5; do
    bash /root/.cc-mirror/codewiz-cc/config/skills/video-frames/scripts/frame.sh \
      workspace/core4d/results/E062/E062_${case}_sphere_autopalm.mp4 \
      --time $t --out workspace/core4d/results/E062/keyframes/${case}_t${t}s.jpg
  done
done
```

Read keyframes, fill log §3 visual table.

### 4C. Decision per Goal table

Apply judgment table from "Decision Tree" above.

## Phase 5: Log + Tracker + Commit

### 5A. Write log 78

**File**: `workspace/core4d/log/78_E062_auto_palm_normal_sphere.md`

Must include:
- Phase 2B/C output (per-case computed palm_normals)
- Phase 4A quantitative table (E062 vs E061 baseline vs E048 historical)
- Phase 4B 5+5 keyframe descriptions (filled, not "TBD")
- Decision per §4C
- "改动文件" table

### 5B. Update tracker

- E062 row
- Log 78 index
- If PASS: Phase 18 → "X1 sufficient for case generalization, push to remaining 4 cases"
- If PARTIAL: keep at "X1 alone insufficient, need X2 / other"

### 5C. Commit

```
exp(core4d): E062 X1 auto palm_normal on sphere — [PASS/PARTIAL/FAIL] on box025 + box023

- compute_palm_normal.py: per-case auto-derive palm_normal from ref motion
  via proximity-windowed dot product (audit §2.2 algorithm, now committed
  as a tool not just inline log analysis).
- box025 result: pelvis_min X.XXm (E061 baseline 0.672m, hist 0.575m)
- box023 result: pelvis_min Y.YYm (E048 hist 0.193m, +Z.ZZm)
- Decision: [next step].
```

## Critical Files

### Modify (Phase 1)
- `spider/assets/robots/unitree_g1/robot.xml` — revert to sphere
- `example_datasets/processed/core4d/unitree_g1/humanoid_object/{7 cases}/scene{,_act}.xml` — revert hand to sphere; box023 re-apply margin 0.90

### Create (Phase 2-5)
- `workspace/core4d/scripts/convert/compute_palm_normal.py` — Phase 2A
- `examples/config/override/core4d_e062_box025.yaml` — Phase 2B (auto-generated)
- `examples/config/override/core4d_e062_box023.yaml` — Phase 2C (auto-generated)
- `workspace/core4d/scripts/train/train_E062.sh` — Phase 3A
- `workspace/core4d/results/E062/scene_snapshot/` — Phase 3 dual-safeguard
- `workspace/core4d/results/E062/E062_{box025,box023}_sphere_autopalm.{npz,mp4}` — Phase 3B output
- `workspace/core4d/results/E062/keyframes/{box025,box023}_t*.jpg` — Phase 4B
- `workspace/core4d/log/78_E062_auto_palm_normal_sphere.md` — Phase 5A
- `workspace/core4d/plan/72_E062_auto_palm_normal_sphere_plan.md` — copy of this plan

### Reuse (no modification)
- `workspace/core4d/scripts/convert/set_collision_margin.py` — Phase 1B
- `workspace/core4d/scripts/convert/snapshot_scenes.sh` — Phase 3A
- `examples/config/override/core4d_e041c.yaml` — Phase 2A defaults inheritance
- `spider/simulators/mjwp.py:820-892` — `contact_hdmi_palm_normal_left/right` consumption (no change needed, only override)
- `examples/run_mjwp.py` — entry point

## Verification

```bash
# Phase 1 done
.venv/bin/python -c "
import mujoco
m = mujoco.MjModel.from_xml_path('example_datasets/processed/core4d/unitree_g1/humanoid_object/box025_person1/scene_act.xml')
hand = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) for i in range(m.ngeom)
        if mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) and
           mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i).startswith(('lh','rh'))]
assert hand == ['lh', 'rh'], f'sphere revert failed: got {hand}'
print('sphere OK')
"

# Phase 2 done
ls examples/config/override/core4d_e062_*.yaml  # 2 files

# Phase 3 done
ls -lh workspace/core4d/results/E062/E062_*.{npz,mp4}  # 4 files

# Phase 4 done
ls workspace/core4d/results/E062/keyframes/  # 10 jpg

# Phase 5 done
git log -1 --stat
```

## Risks & Mitigations

| Risk | Probability | Mitigation |
|------|-------------|-----------|
| Phase 1 1A 把 9 cases 列错 (有的 case 不在 git 里) | 低 | 先 `git diff fa2e181~1..fa2e181 --name-only -- example_datasets/...` 取确切列表 |
| compute_palm_normal.py 算出 box025 不是 [0,∓1,0] | 中 | Phase 2B 强制验证, 不通过禁止往后做 |
| Proximity threshold 0.30m 太严/太松 | 中 | 先用 0.30, 看 frame 数; <10 frame 增到 0.50, >150 frame 降到 0.20 |
| trajectory_kinematic_dual.npz 不存在某 case | 低 | 脚本 fallback 到 trajectory_kinematic.npz |
| sphere + box023 + auto palm_normal 仍摔 (box023 还有别的 issue) | **高** | log §Decision Tree, 进 X2 或其他 case-specific 调整 |
| sphere baseline (box025) 退化 | 低 | E061 已证 sphere+E041c work, 只改 palm_normal 不该退化太多 |
| Hydra `defaults` 继承 + per-package override 顺序问题 | 中 | 测试一个 yaml 跑通再生成第二个; 必要时改 yaml 直接 inline 全部 E041c 字段 (像 core4d_e060_2_box023.yaml 那样) |

## Out of Scope (defer)

- **X2 (auto eef_offset)**: 在 sphere 上价值降低, 跳过. 如果 X1 单独通过 box023 → 不做 X2; 如果不通过 → 加 X2 在 E063
- **推到剩余 4 cases (bucket005_s2/007/001, desk021)**: E063 任务
- **重做 E060.0/.1/.2**: 已 invalidated, 不需要
- **3-box 重新评估**: 长期可选, 当前不做
- **修 P1-P4 (body tracking / 臂展 / 下半身)**: 物理硬限制, 投入产出比低, log 77 决策暂不修
- **inline X1 (β 选项, 在 reward init 时算 palm_normal)**: 跟离线 (α) 等价但 debug 复杂, 等离线方案验证后再考虑

## Estimated Cost

| Phase | Time | GPU |
|-------|------|-----|
| Phase 1 (revert + smoke) | 10 min | 0 |
| Phase 2 (compute_palm_normal.py + 2 yamls) | 60 min | 0 |
| Phase 3 (train parallel) | ~30 min wall | 2 |
| Phase 4 (eval + keyframes) | 15 min | 0 |
| Phase 5 (log + tracker + commit) | 30 min | 0 |
| **Total** | **~2.5 h** | 2 GPU peak |

## Success Definition

- **Mandatory**: Phase 1 sphere revert clean (no 3-box residue, box023 margin 0.90 preserved); compute_palm_normal.py 通过 box025 self-check ([0,∓1,0])
- **Primary (X1 success)**: Phase 4A box025 ≥ 0.6m AND box023 ≥ 0.5m → reward 真有泛化性, E063 推 4 cases
- **Acceptable (partial)**: box025 不退化, box023 改善 (≥ 0.3m, +50%) 但仍未达 0.5m → X1 部分有效, 加 X2 或其他
- **Failure (X1 不工作)**: box025 退化 OR box023 完全没改善 → 算法有 bug 或假设错, log 78 详细诊断, 重新设计 reward 泛化方案
