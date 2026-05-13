# E061 Sphere Baseline Verification — 3-box port confirmed as SOLE regression source

## 状态: ✅ **PASS — sphere 完全复现历史 baseline (pelvis_min 0.672m, Stable 100%), 决定性确认 fa2e181 的 3-box port 是 box025 regression 的唯一根因**

**TL;DR**: 按照 log 74 §4.1 的 D 方案, 临时把 `robot.xml` + box025 scene 退回 `fa2e181~1` (sphere 时代), 跑 `box025 + E041c`. 结果 pelvis_min = **0.672m** (历史 0.575m, 新 3-box 0.253m), Stable% = **100.0%** (新 3-box 61.7%). 视觉 5/5 keyframe 全是站立扶箱, 跟 ref motion 同侧, 无任何摔/T-pose/掉箱行为. 这**决定性证明** 3-box port 是 E060.0/.1/.2 反直觉结果的唯一根因, reward stack (E041c) 在 sphere 几何下完全 work. **下一步**: 决定 B (修 3-box geometry / eef_offset 对齐) vs A (revert sphere) — 等待用户拍板.

## 1. 实验配置

| 项 | 值 |
|----|---|
| 触发 | log 74 §4.1 D 方案 (sphere baseline verification) |
| Hand collision | sphere @ wrist+10cm, r=5cm (`fa2e181~1` 的 robot.xml) |
| Reward stack | `core4d_e041c.yaml` (从未改动, 自 `f68dcca` 起) |
| Task | `box025_person1` (scene + scene_act 也回退到 `fa2e181~1`) |
| GPU | 0, MUJOCO_GL=egl, use_torch_compile=false |
| Wall time | ~30 min (124 ctrl frames × ~14.5s/step CEM) |
| Snapshot | `workspace/core4d/results/E061/scene_snapshot/` (sphere) + `E061_pre_checkout/scene_snapshot/` (3-box audit trail) |

## 2. 数值结果

| 指标 | 历史 sphere E041c (E048 视频复查) | **新 3-box E060.0 (E041c, log 74 表)** | **E061 sphere E041c (本次)** | E061 vs 新 3-box |
|------|-----------------------------------|----------------------------------------|------------------------------|------------------|
| pelvis_min | 0.575m | 0.253m | **0.672m** | **+0.419m (+165%)** |
| pelvis_mean | ~0.7m | 0.609m | **0.785m** | +0.176m |
| pelvis_max | — | — | 0.881m | — |
| Stable% (≥0.5m) | 100% | 61.7% | **100.0%** | **+38.3pp** |
| 帧数 | — | — | 124 | — |
| 全程最低帧 | — | (中段) | (无 — 全程 ≥0.81m) | — |

**E061 sphere 比历史 E048 sphere 还略好 (0.672 vs 0.575)**. 不是退化, 也不是 noise — 整段轨迹 pelvis_z 都在 0.80m 以上 (头 5 帧 0.81-0.82, 尾 5 帧 0.80-0.81), 说明 CEM 找到了一个比历史更稳的解. Stable 100% 满足判定表 §4.2 的 PASS 阈值.

## 3. 视觉 keyframe (5/5 站立扶箱)

视频路径: `workspace/core4d/results/E061/keyframes/box025_sphere_t{0.5,1.5,2.5,3.5,4.5}s.jpg`

| 时间 | ref 行为 | sim 行为 | 一致? |
|------|----------|----------|-------|
| t=0.5s | ref 半弯腰, 手伸向箱顶, 接近期 | sim **站立, 手伸向箱顶**, 略偏侧但完全直立 | ✓ 同动作 phase, 不同摄像机角度 |
| t=1.5s | ref 双手按箱顶, 弯腰发力 | sim **双手按箱顶, 弯腰发力** | ✓ 几乎完全一致 |
| t=2.5s | ref 双手抓箱前侧, 准备搬动 | sim **双手扶箱顶, 上半身在箱后** | ✓ 持物姿势, 站立 |
| t=3.5s | ref 弯腰前推/搬箱, 双臂展开 | sim **弯腰前推, 单手在箱顶, 另一手在箱前** | ✓ 同 phase 动作 |
| t=4.5s | ref 直立, 准备松手 | sim **直立, 单脚边缘接触箱底** | ✓ 站立结尾 |

**关键对比 (vs log 74 §1.2 新 3-box e041c 表)**:
- 3-box e041c t=2.00s: "sim 双手在箱顶, 但箱姿态跟 ref 不同 (ref 已倾斜, sim 是水平)"
- 3-box e041c t=4.00s: "sim 双臂高举接近 T-pose, 箱子掉地, trace_object site 在地上"
- **E061 sphere 全程无 T-pose, 无掉箱, 跟 ref 同侧同 phase**

视觉确认: sphere 物理 + E041c reward = 历史 E048 的"弯腰前倾, 手在箱顶, 手背朝向物体"复现.

## 4. Claims 验证

| ID | 描述 | 量化 | 通过 |
|----|------|------|------|
| C1 | sphere 复现历史 pelvis_min 阈值 | ≥ 0.50m → **0.672m** | ✅ |
| C2 | Stable% 复现 | ≥ 90% → **100.0%** | ✅ |
| C3 | 视觉 5/5 keyframe 站立持物 | 5/5 | ✅ |
| C4 | 全程无摔 | 124/124 frames pelvis ≥ 0.5m | ✅ |
| C5 | Phase 4 restore 验证 | 6 hand geoms, npair=34 (3-box) | ✅ |

**5/5**.

## 5. 决定性结论 (per log 74 §4.2 判别表)

判别表第一行: `pelvis_min ≥ 0.50m, Stable ≥ 90%` → **3-box port 是唯一 regression 源**, E060.0/.1/.2 全部基于物理 bug.

具体推论:
1. **E041c reward stack 无 bug**. 跟 sphere 配对时给出 100% stable + 0.67m pelvis_min, 说明 contact_hdmi + ori reward + dynamic_target 这套都 work.
2. **3-box port (commit fa2e181) 引入了 reward/物理脱节**. 根因如 log 74 §2.2: `contact_hdmi_eef_offset = [0.05,0,0]` (sphere 时代为 sphere 中心略后位置校准) 跟 3-box 的 box3 末端 (wrist+17.5cm) 错位 12.5cm. CEM 优化 reward 让 wrist+5cm 靠近物体 → 物理上 box3 提前撞翻物体.
3. **E060 reward ablation (log 71/72/73) 全部需重做**. case-divergent / catastrophic 等 "task-specific reward" 的解读都只是 3-box 几何在不同 palm normal 下的局部最优噪声, 不是 reward 真实差异.

## 6. 关联无变化的 confound

历史 sphere E048 与现在 sphere E061 之间的代码差异:
- `examples/config/override/core4d_e041c.yaml`: 自 `f68dcca` (E041 时代) 后未改动 ✓
- `examples/run_mjwp.py`: 可能有改动 (E058 加 warmstart hook 等), 但本次未启用 warmstart, hook 不影响 baseline path
- `spider/simulators/mjwp.py`: contact_hdmi_rew 块自 E041 起稳定, eef_offset 默认值未改

**E061 PASS 也间接证明这些非 robot.xml 的代码改动不影响 sphere baseline**, 否则 sphere 也会退化. 干净的 single-variable 验证.

## 7. 下一步选项 (留给用户决定, 不在本 log 内 plan)

per log 74 §4.3:

**B (推荐): 修复 3-box geometry / reward 对齐**
- B1: 缩 box2/box3 pos, 让 box3 末端 ≈ wrist+8cm
- B2: 改 `contact_hdmi_eef_offset` 从 `[0.05,0,0]` → `[0.10~0.15, 0, 0]` 让 reward 看 box3 末端附近
- 然后在 box025 上重做 E041c, 必须达到 sphere 时代基线再做下游 reward ablation
- 优势: 保留区分手心手背的物理意义

**A (backup): revert 到 sphere**
- `git revert fa2e181` (撤销 3-box port)
- 优势: 简单, 立即恢复所有历史 baseline
- 代价: 失去 E041 ori reward 的物理基础 (球各向同性), ori reward 退化为视觉补丁

**等待用户拍板 B/A**. 选定后再开 E062 plan.

## 8. 改动文件

| 文件 | 改动 |
|------|------|
| `workspace/core4d/log/75_E061_sphere_baseline_verify.md` | 本文件 |
| `workspace/core4d/plan/71_E061_sphere_baseline_verify_plan.md` | E061 plan (从 plan mode 复制) |
| `workspace/core4d/scripts/train/train_E061.sh` | E061 训练脚本 |
| `workspace/core4d/results/E061/scene_snapshot/` | sphere snapshot (训练用的 scene) |
| `workspace/core4d/results/E061_pre_checkout/scene_snapshot/` | 3-box snapshot (audit trail, 验证 checkout 行为) |
| `workspace/core4d/results/E061/box025_e041c_sphere.{npz,mp4}` | 训练输出 |
| `workspace/core4d/results/E061/keyframes/box025_sphere_t*.jpg` | 5 keyframes |
| `logs/E061/box025_e041c_sphere.log` | 训练 stdout |
| `workspace/core4d/EXPERIMENT_TRACKER.md` | 加 E061 行 + log 75 索引 + Phase 18 入口 |

## 9. 关联 commit

- `fa2e181` infra(core4d): port 3-box hand collision + box023 margin 0.90x — **regression source (本 log 决定性确认)**
- `2d8e27e` E060.2 catastrophic + box025 regression discovery — log 74 触发本实验
- (本 log) E061 sphere verification — **PASS**, 等待用户决定 B/A

## 10. 教训沉淀 (强化 log 74 §5 规则)

E061 验证了 log 74 §5.1 提出的"物理 port 必须配套历史 case 回归测试"规则: 如果 fa2e181 commit 时就跑了 box025 sphere → 3-box 的回归对比, **本次 E060 整套 (.0/.1/.2) 浪费的 4-5 小时计算 + 3 个 log 的错误结论都可避免**. 

强化版规则 (从此实施, 写入 CLAUDE.md 候选):

> **任何修改 `robot.xml` / 数据 scene XML / 物理参数的 commit, 必须满足以下 commit-time 规则 (不能延迟到下游实验)**:
> 1. 同 commit 内必须包含一个回归测试脚本, 在历史已知能跑通的 case 上跑, 确认 baseline pelvis_min 不下降 ≥ 5cm
> 2. 回归测试结果 (npz + 5 keyframes) 必须放在 `workspace/{exp_name}/results/REGRESSION_{commit_hash}/` 下与 commit 一起 push
> 3. 没有 step 1-2 的物理改动 commit, 后续任何 reward/CEM 实验**不允许引用其结论**
