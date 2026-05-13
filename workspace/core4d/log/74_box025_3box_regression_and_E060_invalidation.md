# Box025 3-box Regression Discovery + E060 Phase Invalidation

## 状态: 🚨 **CRITICAL FINDING — 3-box hand port (commit fa2e181) 在 E041c 校准 case (box025) 上 regression, E060.0/.1/.2 整套对比基础崩塌**

**TL;DR**: 用户在远程机器跑 `run_box025_3box_regression.sh` 验证 3-box hand 改造对历史 baseline 的影响. 三个 reward stack (E041c, E041, E039) **全部 regression**. pelvis_min 从 sphere 时代的 0.575m 跌到 0.16-0.25m (-30 到 -42cm), stable% 从 100% 跌到 60-69%. 视频显示物体被推开/掉地, sim 行为完全异常 (绕到 box 后方/趴在 box 上). E041c box025 是 E041c reward stack 的**校准 case**, 历史上稳定 100%, 现在跑不出来 → **3-box hand port 是问题源**, 不是 reward.

这意味着 **E060.0/.1/.2 全部建立在物理 bug 之上**. 它们的"摔倒"现象至少部分是 3-box geometry × reward eef_offset 错位导致, 不是单纯的 reward task-specific 问题. E060 phase 的 reward ablation 结论 (E060.1 case-divergent, E060.2 catastrophic) **不可靠**, 必须在修复物理基础后重做.

## 1. Box025 Regression 数据

### 1.1 三个 reward stack 数值

历史 sphere 时代 (sphere hand_collision = 5cm 球 at wrist+10cm):

| Run | 来源 | pelvis_min | Stable % (full) |
|-----|------|------------|-----------------|
| E041c sphere baseline | E041c log line 56 / E048 视频复查 | **0.575m** (E048) | **100%** |
| E053a sphere + 0.90 margin | E053 log line 19-23 | **0.660m** | **100%** |
| 全部 sphere 时代 box025 | TRACKER lines 53-69 | **>= 0.50m** | **>= 99%** |

**新 3-box 时代** (commit fa2e181 之后, 6 box per side, 12 contact pair):

| Run | reward override | pelvis_min | pelvis_mean | Stable % | recovers? |
|-----|----------------|------------|-------------|----------|-----------|
| `box025_e041c` | core4d_e041c | **0.253m** | 0.609 | **61.7%** | True |
| `box025_e041` | core4d_e041 (multiply ori) | **0.193m** | 0.632 | **69.2%** | False |
| `box025_e039` | core4d_e039 (no ori reward) | **0.156m** | 0.528 | **60.8%** | False |

**关键**: **E039 (无 ori reward) 也 regression**. 这排除"reward 是元凶"的假设 — 即使最简单的 contact + position reward 也跑不出 sphere 时代 0.575m baseline.

### 1.2 视觉行为变化 (4 keyframe 核实, 都摔)

历史 sphere E041c box025 (E041 log line 56-60): "sim 弯腰前倾, 手在箱顶, 手背朝向物体. 前倾程度与 E040 类似, 手掌方向不太明确. additive 模式不会完全 zero-out reward → CEM 保持接近行为".

新 3-box (4 帧目检, 不同 reward 行为模式不同但都异常):

| 时间 | box025_e041c (3-box) | box025_e039 (3-box) |
|------|---------------------|---------------------|
| t=2.00s | sim 双手在箱顶, 但箱姿态跟 ref 不同 (ref 已倾斜准备搬, sim 是水平) | sim **头部趴在箱顶**, 物体竖立 |
| t=4.00s | sim 双臂高举接近 T-pose, **箱子掉地, trace_object site 在地上** | sim 在 **box 后方**, 单手扶箱顶, 物体掉地 |

行为差异**不是同一种摔法**:
- sphere E041c box025: sim 弯腰扶箱, 跟 ref 在同一侧, 物体跟随 ref
- 3-box e041c box025: sim 在前方但 **物体被推走/掉**, 然后 sim 双臂高举 (想要重新接触但失败)
- 3-box e039 box025: CEM 找到的局部最优是 **绕到 box 后方推**

**CEM 在 3-box 几何下找到的局部最优解跟 sphere 时代完全不同**. 这是 collision shape 改变导致接触动力学突变的直接证据.

## 2. 根因诊断

### 2.1 几何 mismatch (主要嫌疑)

| 项 | sphere | 3-box | 差异影响 |
|----|--------|-------|---------|
| 形状对称性 | 各向同性 | 有方向 (box3 倾斜 ±22.9° around z) | 接触法向/摩擦方向变了 |
| x 轴长度 | 单点 (wrist+10cm 半径 5cm) | 长条 (从 x=-3cm 到 x=+17.5cm) | "手"实际长 7cm, 提前撞物体 |
| 总碰撞体积 | V ≈ 5.2e-4 m³ | V ≈ 5.5e-4 m³ | 接近, 但分布完全不同 |
| 与 contact_hdmi_eef_offset (`[0.05,0,0]`) 的对齐 | sphere center = wrist+10cm, reward 看 wrist+5cm, 偏 5cm | box3 末端 = wrist+17.5cm, reward 看 wrist+5cm, **偏 12.5cm** | reward 优化目标和实际接触点错位 → CEM 找到的 reward-高 解物理上是"碰撞末端推开物体" |

### 2.2 The smoking gun: contact_hdmi_eef_offset

`contact_hdmi_eef_offset = [0.05, 0, 0]` 这个值是 E039/E040 时代基于 sphere geometry 调的 (sphere center 在 wrist+10cm, eef_offset 5cm 取的是 sphere 中心略后位置). 当时 sphere 各向同性, eef_offset 落在 sphere 内部都"看得到接触". 

3-box 时代:
- box1 (wrist cuff): pos=(0.02, 0, 0), 范围 x∈[-0.03, 0.07], eef_offset 0.05 在 box1 内 ✓
- box2 (palm slab): pos=(0.09, 0, 0), 范围 x∈[0.04, 0.14]
- box3 (finger pad, 倾斜): pos=(0.15, ∓0.01, 0), 范围 x∈[0.125, 0.175]

CEM 优化 reward 让 wrist+[0.05,0,0] 靠近物体目标. 但**物理上 box2 中心和 box3 末端先撞到物体** (它们伸到了 wrist+8 ~ wrist+17.5cm). 物体被这个"前突的 box3"撞翻/推开 → reward 报告 "我离物体很近", 物理报告 "物体被你撞飞了".

这是个典型的 **reward 跟物理脱节** bug. CEM 局部最优收敛到一个数学高 reward 但物理灾难的姿态.

### 2.3 体积差异不是元凶

总碰撞体积 sphere vs 3-box 接近 (~5.5e-4 m³). 所以不是"手变重撞翻物体". inertial 也未改 (mass=0.254576), 是 carry 物体的能力没变, 是**接触位置的精确度**问题.

### 2.4 contact pair 配置 OK

`patch_hand_3box.py` 的 12 个 pair (3 box × 2 side × 2 surface = 12) 用的是 sphere 时代继承的 friction/condim/solref. 这部分应该没问题.

## 3. 对 E060 整套实验的影响

### 3.1 数据回顾 (E060.0 → .1 → .2)

| 实验 | 改动 | box023 stable_intent | bucket005_s2 stable_intent |
|------|------|---------------------|----------------------------|
| E060.0 | 数据层 fix (3-box + box023 0.90 margin), reward = E041c | 74% | 15% |
| E060.1 | + ori_weight = 0 | 53% (-21pp) | 54% (+39pp) |
| E060.2 | + case-correct palm_normal (+x) | **5%** | **2%** |

之前的解读:
- E060.1 case-divergent → 修订假设 "palm normal = implicit prior"
- E060.2 验证修订假设 → catastrophic 反向

### 3.2 加上 box025 regression 后的重新解读

**新解读**: 三个 sub-experiment 的不同结果可能不是 "reward 真的有 case-divergent 影响", 而是 **3-box geometry 在不同 palm normal 配置下让 CEM 收敛到不同 (但都不好的) 局部最优**:

- E060.0 (默认 hardcoded -y/+y): CEM 找到 "前扑+物体在地" 局部最优
- E060.1 (ori = 0): 没了朝向约束, CEM 找到 "深蹲弓步" (bucket005_s2) 或 "前扑加爬起" (box023) 局部最优
- E060.2 (case-correct +x): 让手指轴指物体, CEM 找到 "全程趴地" 局部最优

**所有这些"局部最优"都是 3-box 末端推翻物体之后的不同收尾姿态**. 不是 reward stack 的真实 task-specific 问题.

box025 regression 是干净证明: 同样 3 个 reward, 历史 sphere 时代有 100% stable, 现在 3-box 时代全 < 70%. 唯一变量是 hand collision geometry. 3-box port 就是 regression source.

### 3.3 哪些结论需要重新验证

| log/conclusion | 之前判断 | 现在的怀疑 |
|---------------|---------|-----------|
| log 71 §1 "数据层修复未实质改善" (E060.0 vs E058/E059) | 数据层 (3-box + 0.90 margin) 不是元凶 | 实际上**数据层 (3-box) 让事情更糟了**, 因为 sphere → 3-box 引入新 bug |
| log 72 §4.1 "case-divergent + implicit prior 假设" (E060.1) | hardcoded palm normal 是 implicit prior, 关掉 box023 退化 / bucket005_s2 改善 | 这个 case-divergent 可能是 3-box geometry 在不同 reward 下的随机表现, 不是真 reward 机制 |
| log 73 (E060.2) "case-correct +x 比 hardcoded 差" | +x 不是更好的 prior | 也可能是 +x 让 CEM 转 wrist 让手指方向指物体 → 撞得更深 |

**audit log §2.2 (palm normal 数值分析) 仍然 valid** — 它分析的是 ref motion 中 wrist 哪个轴指物体, 跟 CEM 优化无关, 跟 collision geometry 无关. 但**用 audit 结论指导 reward 调参的所有结论** (E060.1/.2) 都需要在修好 3-box 后重做.

## 4. 立即必须做: 验证 sphere 是否能复现历史 baseline

在做任何下一步前, 必须**控制变量**确认 box025 sphere 能复现历史 0.575m / 100% stable. 这排除两种可能:
- (a) **3-box port 是唯一 regression 源** → 修复方向明确 (B 重新校准 3-box 几何 / A revert)
- (b) **还有其他变化** (E058 加的 warmstart hook, 别的代码改动) 也影响了 baseline → 范围更广, 需要 git bisect

### 4.1 D 方案: sphere 验证 (执行步骤)

```bash
# 1. 临时退回 sphere (3-box port commit fa2e181 之前)
git checkout fa2e181~1 -- \
  spider/assets/robots/unitree_g1/robot.xml \
  example_datasets/processed/core4d/unitree_g1/humanoid_object/box025_person1/scene.xml \
  example_datasets/processed/core4d/unitree_g1/humanoid_object/box025_person1/scene_act.xml

# 2. 验证 sphere 状态 (应只有 ['lh', 'rh'])
.venv/bin/python -c "
import mujoco
m = mujoco.MjModel.from_xml_path('example_datasets/processed/core4d/unitree_g1/humanoid_object/box025_person1/scene_act.xml')
hand = sorted([mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i)
               for i in range(m.ngeom)
               if mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i)
               and mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i).startswith(('lh','rh'))])
print(f'hand geoms: {hand}')  # 期待 ['lh', 'rh']
print(f'npair: {m.npair}')    # 期待 ~26 (sphere 时代)
"

# 3. 跑 box025 + E041c (sphere 时代的 calibration)
mkdir -p workspace/core4d/results/E060_sphere_verify
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl PYTHONUNBUFFERED=1 .venv/bin/python -u examples/run_mjwp.py \
  +override=core4d_e041c task=box025_person1 +use_torch_compile=false \
  output_dir=workspace/core4d/results/E060_sphere_verify_outdir \
  video_output_path=workspace/core4d/results/E060_sphere_verify/box025_e041c_sphere.mp4 \
  > logs/E060_sphere_verify.log 2>&1
cp workspace/core4d/results/E060_sphere_verify_outdir/trajectory_mjwp_act.npz \
   workspace/core4d/results/E060_sphere_verify/box025_e041c_sphere.npz

# 4. 量化对比
.venv/bin/python -c "
import numpy as np
d = np.load('workspace/core4d/results/E060_sphere_verify/box025_e041c_sphere.npz', allow_pickle=True)
qpos = d['qpos']; qpos = qpos[:,0,:] if qpos.ndim==3 else qpos
pz = qpos[:,2]
print(f'pelvis_min: {pz.min():.3f}m  (历史 0.575m)')
print(f'stable %  : {(pz>=0.5).mean()*100:.1f}%  (历史 100%)')
"

# 5. 用完恢复 3-box (避免污染当前实验状态)
git checkout HEAD -- spider/assets/robots/unitree_g1/robot.xml \
  example_datasets/processed/core4d/unitree_g1/humanoid_object/box025_person1/scene.xml \
  example_datasets/processed/core4d/unitree_g1/humanoid_object/box025_person1/scene_act.xml
```

约 30min wall.

### 4.2 D 的判别表

| sphere box025 + E041c 结果 | 判断 | 下一步 |
|---------------------------|------|--------|
| pelvis_min ≥ 0.50m, stable ≥ 90% | **3-box port 是唯一 regression 源**. E060.0/.1/.2 全废 | 进 4.3a (B 修 3-box geometry) 或 4.3b (A revert sphere) |
| 0.30m ≤ pelvis_min < 0.50m | 部分恢复, 还有别的 confound | git bisect 找 fa2e181 之外的 regression commit |
| pelvis_min < 0.30m | sphere 也复现不了历史 → 还有更深的问题 | 可能 E058 warmstart hook 影响 baseline; 继续往前 bisect 到 0c767f1 之前 |

### 4.3 D 通过后的两个修复方向

**B (推荐)**: 修复 3-box geometry, 让其与 reward eef_offset 对齐
- 把 box1/2/3 的 pos 整体往 wrist 缩, 让 box3 末端 ≈ wrist+8cm (而非 wrist+17.5cm)
- 或者把 contact_hdmi_eef_offset 改成 `[0.10, 0, 0]` 让 reward 看 box3 末端附近
- 重新跑 box025 + E041c 验证, 直到达到 sphere 时代基线
- 然后再做 reward ablation (重新 E060.0/.1/.2)

**A (backup)**: 完全弃用 3-box, revert 到 sphere
- `git revert fa2e181` (或类似) 撤销 3-box port
- 接受 "sphere 各向同性, 手心手背物理无区分" 的限制
- 转到其他方向破解 E058/E059 摔倒 (如 E060.3 stability_penalty 或更大的方向变更)

## 5. 教训

### 5.1 物理 port 必须验证 sphere → 新形状的 baseline 等价性

3-box port (fa2e181) 之前没做"sphere 版本能跑通的 case 在 3-box 版本上还能跑通" 的回归测试. 直接就在 box023/bucket005_s2 上跑 E060.0, 默认 "数据层修复一定是改善". **应该的流程**:
1. 改物理 (3-box port)
2. **必做**: 在历史能跑通的 case (box025) 上回归, 确认新物理至少不退化
3. 通过后才在新 case 上做 reward ablation

这次 user 在远程跑 box025 regression 救了我们 — 没有这个回归测试, E060.0/.1/.2 的反直觉结果会被错误归因为 "reward task-specific" 而不是 "物理 bug + reward 噪声".

**强制规则 (从此实施)**: 任何修改 robot.xml / scene.xml 物理参数的 commit, **必须配套一个回归测试** — 在历史已知能跑通的 case 上跑, 确认 baseline 不退化. 没有这一步的物理 port 不允许往后做 reward 调参.

### 5.2 reward eef_offset 是 case-specific 假设, 不是常数

`contact_hdmi_eef_offset = [0.05, 0, 0]` 在 sphere 几何下"碰巧 work", 因为 sphere 各向同性. 改 collision geometry 时, 这个 offset 必须重新校准. 之前它从未被当作"几何相关参数", 是隐藏 bug.

类似的隐藏假设可能还有: `contact_hdmi_palm_normal`, `contact_hdmi_threshold`, `local_frame_pos_sigma` 等 — audit log §2 列的"task-specific" 实际是 "geometry-specific". 改 geometry 时全部要重审.

### 5.3 三次 over-optimism / 反向 假设 教训累计

| Over-optimism / 反向 | 我做的 | 用户 / 数据纠正 |
|---------------------|-------|----------------|
| 1 | E059 box023 t=1.65s "接近真实搬运" | 用户: 没有, box023 完全失败 |
| 2 | E060.0 box023 t=1.65s "53 实验首次单手抱箱站立" | 用户: 你视频看了个寂寞 |
| 3 | E060.1 box023 stable_intent 退化 → 修订 "implicit prior" 假设 | 数据 E060.2: 假设反转 |
| 4 | "E060 是诊断 reward task-specific 的好框架" | box025 regression: E060 整套都建立在物理 bug 上 |

教训累计后的强制流程:
1. 看视频前必先看 pelvis_z trace 全程 (log 71 §5)
2. 任何 reward ablation 必须 ≥2 case (log 72 教训 1)
3. 任何物理 port 必须配套历史 case 回归测试 (本 log §5.1)
4. 跨多个 sub-experiment 形成的"假设链"在每一步都重新验证根基, 不要传递性接受

## 6. 改动文件 (本次 audit/discovery)

| 文件 | 改动 |
|------|------|
| `workspace/core4d/log/74_box025_3box_regression_and_E060_invalidation.md` | 本文件 |
| `workspace/core4d/results/E060_box025_regression/` | 用户在远程跑的 3 个 reward 输出 (待 commit) |
| `workspace/core4d/scripts/run_box025_3box_regression.sh` | 脚本 (已 commit `f708834`) |

## 7. 关联 commit

- `fa2e181` infra(core4d): port 3-box hand collision + box023 margin 0.90x — **regression source**
- `784e792` E060.0 baseline test — **结论需重新验证**
- `165ba42` E060.1 ori_weight=0 — **结论需重新验证**
- (待) E060.2 commit (catastrophic fail)
- (待) sphere 验证 (D 方案)
- (待) 3-box 修复 commit 或 revert commit

## 8. 推荐立即下一步

**D 方案 (sphere 验证, ~30min)**, 然后根据结果选 A 或 B:

```
D 通过 (sphere ≥ 0.50m) → B (修 3-box 几何) — 推荐, 保留区分手心手背的能力
                       → A (revert sphere) — backup, 接受 sphere 限制

D 不通过 (sphere 也 < 0.30m) → 还有别的 bug, 用 git bisect 在 fa2e181 之前找
```
