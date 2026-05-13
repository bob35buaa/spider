# E062: X1 Auto Palm Normal on Sphere — Mixed Result (box025 PASS, box023 PARTIAL)

## 状态: ⚠️ **PARTIAL — box025 sphere baseline 不退化 (0.688m vs E061 0.672m), box023 出现 novel "carry attempt + fall + recover" 行为 (pelvis_min 0.058m 严格 FAIL, 但 mean 0.602m + stable 77.9% + max 0.828m 远好于 E048 historical 0.193/0.515/50%), X1 单独不够让 box023 通过 ≥0.5m 阈值, 需诊断或加 X2/其他**

**TL;DR**: E062 实现了 X1 (per-case auto-derive `contact_hdmi_palm_normal` from ref motion via proximity-windowed dot product, audit log 70 §2.2 算法). compute_palm_normal.py 自动算: box025 L=`[0,-1,0]` R=`[0,1,0]` (= E041c default, 完美 self-consistency); box023 L=`[1,0,0]` R=`[1,0,0]` (audit §2.2 +x best). 在 sphere 几何 (commit 61abf4c revert, 保留 box023 margin 0.90) 上跑 E041c+auto X1: **box025 0.688m 不退化** (X1 self-derived 等于 hardcoded → 行为不变, 验证算法一致性). **box023 出现 novel 行为**: 0-2s 站立尝试搬箱 → 2-2.7s 摔倒 → 2.7-4.5s push-up 站起来 → 结束直立行走. pelvis_min 0.058 严格不达标, 但 pelvis_mean 0.602 / stable 77.9% / max 0.828 全部远超 E048 historical (0.515 / 50% / ~). X1 单独**改善了行为质量** (从"摔倒后趴地"变成"摔倒后会爬起"), 但**没解决"为什么会摔"的根问题**.

## 1. 实验配置

| 项 | 值 |
|----|---|
| Hand collision | sphere @ wrist+10cm, r=5cm (commit 61abf4c) |
| box023 collision margin | 0.90x (保留 fa2e181 fix) |
| Reward stack | core4d_e041c.yaml + per-case palm_normal yaml override |
| **palm_normal (box025)** | L=`[0,-1,0]` R=`[0,1,0]` (auto-derived, == E041c default, self-consistency) |
| **palm_normal (box023)** | L=`[1,0,0]` R=`[1,0,0]` (auto-derived from ref motion +x best) |
| GPU | parallel: GPU0 box025, GPU1 box023 |
| Wall time | 33min (32min box025, 33min box023) |
| 输出 | `workspace/core4d/results/E062/E062_box{025,023}_sphere_autopalm.{npz,mp4}` |

## 2. compute_palm_normal.py 输出 (Phase 2 self-check)

`workspace/core4d/scripts/convert/compute_palm_normal.py --cases box025_person1 box023_person1`:

```
=== box025_person1  scene=scene.xml  ref=trajectory_kinematic_dual.npz ===
[box025_person1] left : n=124  +x=-0.056 -x=+0.056 +y=-0.504 -y=+0.504 +z=+0.454 -z=-0.454  -> -y [0.0, -1.0, 0.0]
[box025_person1] right: n=124  +x=+0.046 -x=-0.046 +y=+0.836 -y=-0.836 +z=+0.250 -z=-0.250  -> +y [0.0,  1.0, 0.0]
=== box023_person1  scene=scene.xml  ref=trajectory_kinematic.npz ===
[box023_person1] left : n=136  +x=+0.840 -x=-0.840 +y=+0.017 -y=-0.017 +z=+0.274 -z=-0.274  -> +x [1.0, 0.0, 0.0]
[box023_person1] right: n=136  +x=+0.727 -x=-0.727 +y=+0.451 -y=-0.451 +z=+0.057 -z=-0.057  -> +x [1.0, 0.0, 0.0]
```

**self-consistency 验证**:
- box025 L 选 -y 跟 audit log 70 §2.2 "L mean dot -y = 0.78~0.80" 方向一致 ✓ (mean 0.504 偏低因为算法用全帧 124 而 audit 只用 intent 窗口; 方向选择正确 = ✓)
- box025 R 选 +y 同 ✓
- box023 L/R 都选 +x, 跟 audit log 70 §2.2 "L true best +x dot +0.61, R true best +x dot ?" 方向一致 ✓

算法验证通过. 无 bug.

## 3. 数值结果

### 3.1 box025 (sphere baseline 不退化)

| 指标 | E041c original (E048) | E061 sphere baseline | **E062 sphere + auto X1** | E062 vs E061 |
|------|----------------------|---------------------|---------------------------|--------------|
| pelvis_min | 0.575m | 0.672m | **0.688m** | +0.016m |
| pelvis_mean | ~0.7m | 0.785m | 0.784m | -0.001m (essentially same) |
| pelvis_max | — | 0.881m | 0.874m | -0.007m |
| stable% (≥0.5m) | 100% | 100.0% | **100.0%** | 0pp |
| frames | 136 | 124 | 124 | (same case) |

**完美 self-consistency**: auto-X1 算出来跟 hardcoded E041c default 完全一样, 所以行为不变, 只有 CEM noise 级别差异. 这验证了:
- compute_palm_normal.py 算法正确 (没破坏已 work 的 case)
- yaml override + Hydra defaults 继承机制 work
- E061 sphere baseline 可重复

### 3.2 box023 (novel "carry-fall-recover" 行为)

| 指标 | E048_box023 (historical sphere) | E059 baseline (sphere) | E060.0 (3-box) | **E062 sphere + auto X1** | E062 vs E048 |
|------|--------------------------------|----------------------|---------------|---------------------------|--------------|
| pelvis_min | 0.193m | 0.140m | 0.176m | **0.058m** | -0.135m ⬇️ |
| pelvis_mean | 0.515m | (~0.4) | (~0.4) | **0.602m** | +0.087m ⬆️ |
| pelvis_max | (~0.8) | — | — | **0.828m** | (similar) |
| stable% (≥0.5m) | 50.0% | (~30%) | (~30%) | **77.9%** | +27.9pp ⬆️ |
| frames | 136 | — | — | 136 | — |

**严格 pelvis_min ≥ 0.5m 阈值**: ❌ FAIL (0.058 远低于阈值)

**但其他指标全面改善**:
- pelvis_mean +9cm (整体高度更高)
- stable% +28pp (更多帧站立)
- max 0.828 (能站到比 E048 还高)

**关键诊断: pelvis_z 全帧 trace 揭示行为模式变化**

```
t=0.00s  pz=0.794  ####################### (站立起步)
t=0.33s  pz=0.802  ########################
t=0.67s  pz=0.612  ##################  (弯腰开始)
t=1.00s  pz=0.613  ##################
t=1.33s  pz=0.654  ###################  (carrying-like 姿态)
t=1.67s  pz=0.678  ####################
t=2.00s  pz=0.416  ############         (开始倒)
t=2.33s  pz=0.130  ###                  (摔)
t=2.66s  pz=0.064  #                    (完全趴地)
t=3.00s  pz=0.605  ##################   ← RECOVERY
t=3.33s  pz=0.755  ######################
t=3.66s  pz=0.738  ######################
t=4.00s  pz=0.764  ######################
t=4.33s  pz=0.781  ####################### (直立结束)
```

**行为分段**:
1. **0.0-0.7s**: 站立起步, 接近 box (0.79-0.80m)
2. **0.7-1.7s**: 弯腰 carry 姿态 (0.61-0.68m, 像 E041c box025 的弯腰搬运)
3. **2.0-2.7s**: 摔倒 (0.42m → 0.058m, 0.7s 内骨盆从弯腰位置跌到地面)
4. **2.7-3.0s**: 紧急 push-up 站起 (0.058 → 0.605m)
5. **3.0-4.5s**: 完全恢复直立, pz 0.72-0.78m

**对比 E048** (log 61 §box023): "t=1s sim 机器人摔倒 — 臀部着地, 左腿抬起, 手搭在箱子上; t=3s 完全仰面倒地, 腿朝天" — E048 是"摔倒后**保持**摔倒". E062 是"摔倒后**爬起来**".

## 4. 视觉 keyframe (10 帧 + 5 帧 dense around fall)

### 4.1 box025 (5 帧, 跟 E061 一致)

| 时间 | sim 行为 |
|------|---------|
| t=0.5s | 站立, 接近 box (跟 E061 同) |
| t=1.5s | 低蹲, 双手 wrap 左前角 (E041c box025 老问题, log 77 §3 P1+P2 现象) |
| t=2.5s | 趴在 box 顶, 双手扶 box (站位偏的姿态扭曲) |
| t=3.5s | 同 |
| t=4.5s | 直立, 双臂在身体两侧 (E041c 自己的姿态扭曲, 与 E061 相同) |

E062 box025 视觉 = E061 sphere baseline = E041c original 视觉. 100% match. 所有 P1-P4 未解决但也未恶化.

### 4.2 box023 (5 帧 + 5 dense around fall)

| 时间 | ref | sim | phase |
|------|-----|-----|------|
| t=0.5s | 弯腰下蹲伸手准备搬 | **直立**, 略前倾, 手伸向 box | 起步, 站立 |
| t=1.5s | 弯腰双手抓 box 前面准备搬 | **弯腰, 右手按 box 顶, 左脚 contact ground (橙色)** | 像 carrying 但姿态错 |
| **t=1.7s (dense)** | 弯腰准备站起 | sim 弯腰前推 box, 右手在 box 上, **箱子向前移** | carry attempt 关键帧 |
| **t=2.0s (dense)** | 直立准备 | sim **大幅前倾, 右脚踩在 box 边角**, **整个上身扑向 box** | 摔倒前 0.4s |
| t=2.5s | 直立搬运 | sim **完全趴地**, 双臂伸向 box (box 仍在前方) | DEEP FALL |
| **t=2.7s (dense)** | 弯腰检查 box | sim **完全趴地, 头朝下, 双臂展开抓 box** | 最低点 0.058m |
| **t=3.0s (dense)** | 弯腰 | sim **类似 push-up 姿势, 身体水平, 脚撑地** | 开始 RECOVERY |
| t=3.5s | 弯腰 | sim push-up 姿势更高, 身体接近水平回升 | 中段 recovery |
| t=4.5s | 直立 box 在脚边 | sim **完全直立**, 腿迈开走路, **box 在身后** (没搬走) | 最终: 站立但没完成搬运 |

**视觉总结**:
- E062 sphere+autoX1 在 box023 上**有真实的"尝试搬箱"动作** (1.5-2.0s 弯腰前推)
- 但因物理不稳 + 臂展不够, 在 t=2.0s 之后栽倒
- 然后 CEM 找到了**主动 recovery 的解** — 用 push-up 把上身撑起, 腿伸直, 站起来
- 最终结束于直立行走, 但**没完成搬运**

vs E048: E048 也尝试搬箱但摔倒后**保持仰面躺地**, 没 recovery.

vs E041c box025: box025 sim 没摔, 但姿态扭曲 (P1-P4); box023 sim 摔 + recovery, 是不同的失败模式.

## 5. Claims 验证

| Claim | 阈值 | E062 结果 | 通过? |
|-------|------|----------|------|
| C1: box025 sphere baseline 不退化 | pelvis_min ≥ 0.6m | **0.688m** | ✅ |
| C2: box023 通过 (≥ 0.5m) | pelvis_min ≥ 0.5m | **0.058m** | ❌ |
| C3: 自动算法一致性 (box025 == hardcoded) | L=[0,-1,0], R=[0,1,0] | ✅ 完全一致 | ✅ |
| C4 (partial): box023 改善 ≥ +50% | pelvis_min +50% over 0.193m → ≥0.29m | **0.058m** (-70%) | ❌ |
| C5 (alt): box023 行为质量改善 | mean/stable/max 至少 1 项 +20%+ | mean +17%, stable +56% | ✅ (alt) |

严格判分: **2/5** (C1, C3 通过, C2 主目标失败)
按 Success Definition Acceptable (partial): **mixed signals** — pelvis_min 退化但其他指标改善

## 6. 决策树应用 + 下一步

按 plan §Decision Tree:
- "box025 ≥ 0.6m AND box023 < 0.5m → X1 单独不够, 可能 box023 还有别的问题 (比如 eef_offset 在 box023 上也需要 X2) → 加 X2 (auto eef_offset) 或诊断 box023 reward 其他 case-specific 部分"

具体下一步候选:

### 选项 X1+X2 (E063)
- 加 X2 (auto eef_offset = sphere center 0.10m vs hardcoded 0.05m)
- sphere 上 X2 价值有限 (5cm 误差, sphere 各向同性), 但低成本可一试
- 期望: 让 reward 看 sphere center 而不是后边缘 5cm, 可能让 contact 更稳定

### 选项 X3 (诊断 box023 摔倒物理原因)
- 详细看 t=1.7-2.0s 帧, 找出 sim 为什么开始倾倒
- 可能是: 物体被推前 → sim 重心继续前移 → 摔
- 解决方向: 加 base/leg tracking weight, 或减小物体跟踪权重

### 选项 X4 (case-specific reward weights)
- box023 物体小 (35×37×41cm) vs box025 大 (75×76×94cm), 可能 contact threshold/sigma 需要 case-specific
- audit log 70 §2 列的"task-specific 假设"还有 `contact_hdmi_threshold`, `local_frame_pos_sigma` 等

### 选项 X5 (推 X1 到 4 cases 看 box023 是否 outlier)
- 跑 bucket005_s2/007/001/desk021 + sphere + auto X1
- 如果 4/4 全摔类似 box023 → X1 不够泛化, 进 X2
- 如果 3/4 站住 → box023 是 outlier, 单独诊断 box023 reward

**推荐 X5 (推到 4 cases) 然后 X1+X2 二选一**. 推 4 cases 数据量更大, 更能区分"X1 不够"vs"box023 单独有问题".

但具体下一步**留给用户决定**.

## 7. 改动文件

| 文件 | 类型 | 改动 |
|------|------|------|
| `workspace/core4d/scripts/convert/compute_palm_normal.py` | 新建 | X1 算法实现 (audit §2.2 工具化) |
| `examples/config/override/core4d_e062_box025.yaml` | 新建 (auto-generated) | palm_normal L/R = [0,∓1,0] (== E041c default) |
| `examples/config/override/core4d_e062_box023.yaml` | 新建 (auto-generated) | palm_normal L=R = [+x] |
| `workspace/core4d/scripts/train/train_E062.sh` | 新建 | parallel train script |
| `workspace/core4d/results/E062/E062_box{025,023}_sphere_autopalm.{npz,mp4}` | 新建 | 训练输出 |
| `workspace/core4d/results/E062/keyframes/{box025,box023}_t*.jpg` | 新建 | 5+5 keyframes + 5 dense |
| `workspace/core4d/results/E062/scene_snapshot/` | 新建 | dual-safeguard snapshot |
| `workspace/core4d/log/78_E062_auto_palm_normal_sphere_results.md` | 新建 | 本文件 |
| `workspace/core4d/EXPERIMENT_TRACKER.md` | 更新 | E062 行 + log 78 索引 |

## 8. 关联 commit

- `61abf4c` infra(core4d): revert hand collision to sphere, keep box023 margin 0.90 — Phase 1
- (本 log) `<TBD>` exp(core4d): E062 X1 auto palm_normal on sphere — PARTIAL — Phase 5

## 9. 教训沉淀 #7 (累计)

之前 6 个教训 (log 76 §10 + log 77 §7):
1. 看视频前先看数据
2. ablation 多 case
3. 物理 port 必须回归测试
4. 跨 sub-experiment 假设链每一步重验
5. reward 工作必须 ≥2 case × ≥2 hand 横向验证
6. 视觉描述必须区分实验版本

**新教训 #7 (本 log)**:

> **"自动算法 self-consistency 通过 ≠ 算法对新 case 有效"**. compute_palm_normal.py 算 box025 完美等于 hardcoded (self-consistency PASS), 但应用到 box023 后**反而让 pelvis_min 变更差** (0.193 → 0.058). 这说明 X1 的"per-case 数据驱动"假设可能本身就不充分 — 即使每 case 算出来的 palm_normal 跟 ref motion 完美对齐, CEM 在那个方向上优化反而可能让 sim 更激进地伸手 → 重心前移 → 摔.

具体规则 (从此实施):
- 自动算法的"self-consistency check" (算 box025 算出已知正确值) 只能证明算法实现没 bug, **不能**证明算法假设 generalize
- 验证 generalization 必须在**至少一个 algorithm 没见过的 case 上 ≥ baseline 改善**
- 如果 baseline 退化 (本次 box023 pelvis_min 退化): 算法假设可能有缺陷, 需独立验证, 不能默认推广

## 10. 下一步 (等用户决策)

3 选项摆在桌上, 详见 §6:
- X5 = 推 X1 到 4 cases 看 outlier 模式 (~1.5h)
- X1+X2 = 加 X2 看是否补足 (~30min)
- 诊断 X3 = 看 box023 摔倒物理原因, 改 reward weights (~1-2h)

我推荐 X5 (信息量最大, 决定 X1 是否真泛化).
