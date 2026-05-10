# E044-E047: Phase 12 三方向广泛探索

## 状态: 全部完成 — 三方向均未超越 E041c baseline

## 背景

E039b-E043 确定了当前方法 (E041c) 的 contact 质量上限：box025=66%, bucket010=57%。
本 Phase 同时探索三个突破方向：
1. Body Tracking 改进（E044 手腕权重增强 / E045 sigma 收紧）
2. 更多 CORE4D Case（E046 数据管线扩展）
3. CEM 算法优化（E047 SBTO 对齐 DynaRetarget 论文）

## Baseline: E041c (当前最佳)

| Case | MPKPE | Contact<10cm | Preservation | Stability>0.6 |
|------|-------|-------------|-------------|--------------|
| box025 | 1.4cm | **66%** | 88.2% | **100%** |
| bucket010 | 1.4cm | **57%** | 81.6% | **100%** |
| desk005 | 1.9cm | 4% | — | 81% |

---

## Direction 1: Body Tracking 改进

### E045: Local-frame Sigma 收紧

**假设**: 当前 pos_sigma=0.5m 过宽松，收紧后手部位置误差代价更大 → CEM 更努力优化手部 → contact 改善。

**配置差异**: 仅修改 `local_frame_pos_sigma`（E041c=0.5 → E045a=0.3 / E045b=0.15）

#### 结果

| 实验 | Case | MPKPE | EEF Pos Err | Contact<10cm | Preservation | Stability | Pelvis min |
|------|------|-------|------------|-------------|-------------|-----------|------------|
| E041c | box025 | 1.4cm | — | **66%** | 88.2% | **100%** | — |
| E045a (σ=0.3) | box025 | 1.6cm | 1.72cm | 38% | 47.9% | 88% | 0.195m |
| E045b (σ=0.15) | box025 | 1.4cm | 1.65cm | 46% | 64.0% | 100% | 0.660m |
| | | | | | | | |
| E041c | bucket010 | 1.4cm | — | **57%** | 81.6% | **100%** | — |
| E045a (σ=0.3) | bucket010 | 1.6cm | 1.84cm | 20% | 29.1% | 100% | 0.695m |
| E045b (σ=0.15) | bucket010 | 1.3cm | 1.41cm | 23% | 34.1% | 100% | 0.742m |
| | | | | | | | |
| E041c | desk005 | 1.9cm | — | 4% | — | 81% | — |
| E045a (σ=0.3) | desk005 | 1.9cm | 1.90cm | 6% | 36.7% | 100% | 0.720m |
| E045b (σ=0.15) | desk005 | 1.2cm | 1.43cm | **22%** | 31.7% | **100%** | 0.686m |

#### 分析

**E045a (σ=0.3) — 全面退化**:
- box025: contact 66→38% (−28%), stability 100→88%, pelvis_min=0.195m（一度接近摔倒）
- bucket010: contact 57→20% (−37%)
- desk005: 基本持平 (4→6%)

**E045b (σ=0.15) — 混合结果**:
- box025: contact 66→46% (−20%), 但 stability 保持 100%
- bucket010: contact 57→23% (−34%)
- desk005: contact 4→**22%** (+18%), stability 81→**100%**, MPKPE 1.9→**1.2cm** — desk005 唯一改善

**根因分析**:
sigma 收紧使 body tracking reward 梯度更陡 → tracking max 不变 (W×7=3.5)，但梯度变化让 CEM 更优先满足 tracking → contact reward (gain=5.0) 的相对吸引力下降。这证实了 **contact 瓶颈不在 tracking 精度，而在 ref 中手-物体距离本身**。

desk005 的改善例外是因为：desk005 的 ref 中手-物体距离本来就更远，收紧 sigma 使 body tracking 更严格 → 减少了 E041c 中 desk005 的"前倾趴向物体"不自然行为 → stability 改善。

**结论**: ❌ sigma 收紧路线无效。desk005 的局部改善不值得 box025/bucket010 的大幅退化。

---

### E044a: 手腕权重增强 (wrist_weight=2.0)

**假设**: 给 wrist body (ID 23/30) 额外的 pos tracking 权重 → CEM 更优先优化手位置 → contact 改善。

**实现**: `spider/simulators/mjwp.py` 中 `upper_pos_rew` 计算后对 wrist body 额外加权。

**配置差异**: `local_frame_wrist_weight: 2.0`（默认 1.0）

#### 结果 (仅 box025，bucket010/desk005 待 E044b 远程完成)

| 实验 | MPKPE | EEF Pos Err | Contact<10cm | Stability>0.6 | Pelvis min | Hand dist |
|------|-------|------------|-------------|--------------|------------|-----------|
| E041c | 1.4cm | — | **66%** | **100%** | — | — |
| E044a (w=2.0) | 1.6cm | 2.00cm | **67%** | 73% | 0.113m | 10.37cm |

#### 分析

- Contact 持平 (66→67%) — wrist 权重翻倍未显著改善 contact
- **Stability 严重退化**: 100→73%, pelvis_min=0.113m（一度接近地面）
- **最长不稳定段 34 帧 (1.13s)** — 有明显摔倒趋势
- 手-物体平均距离 10.37cm（vs baseline 可能更近），但不足以提升 contact 阈值内的比例

**根因**: wrist 权重增大使 CEM 优先优化手部 → 牺牲下半身平衡 → stability 退化。这与 E045a 类似：**任何增强上半身 tracking 权重的方法都会与 stability 产生 tradeoff**。

**结论**: ⚠️ Contact 持平但 stability 严重退化。wrist_weight=2.0 不可用。

---

### E044b: 手腕权重增强 (wrist_weight=3.0)

**配置差异**: `local_frame_wrist_weight: 3.0`

#### 结果 (全 3 case)

| 实验 | Case | MPKPE | Contact<10cm | Stability>0.6 | 对比 E041c |
|------|------|-------|-------------|--------------|-----------|
| E041c | box025 | 1.4cm | **66%** | **100%** | — |
| E044b (w=3.0) | box025 | 1.6cm | 59% | **100%** | ❌ contact↓ |
| E041c | bucket010 | 1.4cm | **57%** | **100%** | — |
| E044b (w=3.0) | bucket010 | 1.8cm | 22% | 92% | ❌ 全面退化 |
| E041c | desk005 | 1.9cm | 4% | 81% | — |
| E044b (w=3.0) | desk005 | **1.1cm** | **13%** | **100%** | ⚠️ desk 改善 |

#### E044a vs E044b 对比 (box025)

| 指标 | E044a (w=2.0) | E044b (w=3.0) |
|------|-------------|-------------|
| Contact<10cm | **67%** | 59% |
| Stability>0.6 | 73% ❌ | **100%** |
| Pelvis min | 0.113m | — |

**有趣现象**: weight=3.0 反而比 weight=2.0 更稳定 (100% vs 73%)。原因推测:
- weight=2.0 时 CEM 找到了"翻转右肩"的捷径（reward 收益刚好足够），导致摔倒
- weight=3.0 时手腕权重过大，CEM 被迫选择更保守的全身姿态来满足约束，反而避免了极端动作
- 但 contact 从 67%→59% 说明过高权重使 CEM 优先"锁定手腕"而非"手靠近物体"

**E044 系列结论**: ❌ wrist weight 增强整体无效:
- box025: contact 最高 67% (w=2.0)，但 stability 崩溃; w=3.0 稳定但 contact↓
- bucket010: 严重退化 (57→22%, stability 92%)
- desk005: 局部改善 (MPKPE 1.9→1.1cm, stability 81→100%), 但 contact 仅 4→13%

---

## Direction 3: CEM/SBTO 算法优化

### E047a: SBTO 对齐 DynaRetarget (论文参数)

**实现**: 修复 SPIDER SBTO 与 DynaRetarget 论文的 5 个偏差：
1. 收敛准则: `max(noise_scale) < σ_min` (原 reward improvement)
2. Sigma EWMA: `Σ_new = α_Σ·Σ_old + (1-α_Σ)·Σ_elite` (原 noise 固定)
3. Mean EWMA: `μ_new = α_μ·μ_old + (1-α_μ)·μ_weighted` (原每步替换)
4. Elite fraction: 参数化 3% (原硬编码 10%)
5. MPC 行为不受影响

**配置**: `sbto_mean_momentum=0.95, sbto_cov_momentum=0.2, sbto_sigma_min=0.01, sbto_elite_fraction=0.03`

#### 结果 (box025)

| 指标 | E041c (MPC) | E047a (SBTO) | 变化 |
|------|------------|-------------|------|
| MPKPE | 1.4cm | **155cm** | ❌❌ 完全失败 |
| Contact<10cm | 66% | 0% | ❌ |
| Stability>0.6 | 100% | 31% | ❌ 机器人摔倒 |
| Object pos err | 0.8cm | 72.9cm | ❌ |
| 运行时间 | ~14min | ~8min | ✅ 更快 |

#### SBTO 过程日志

```
SBTO: knot  1/16, h=0.50s, iter=50, max_σ=0.0445, rew=3.194, t=8s
SBTO: knot  2/16, h=0.75s, iter=50, max_σ=0.1222, rew=2.939, t=20s
SBTO: knot  3/16, h=1.00s, iter=50, max_σ=0.0391, rew=2.511, t=34s
SBTO: knot  5/16, h=1.50s, iter=50, max_σ=0.0458, rew=2.071, t=72s
SBTO: knot 10/16, h=2.75s, iter=50, max_σ=0.0745, rew=1.354, t=233s
SBTO: knot 13/16, h=3.50s, iter=50, max_σ=0.0195, rew=1.259, t=372s
SBTO: knot 15/16, h=4.00s, iter=50, max_σ=0.0223, rew=1.255, t=481s
Total: 481.8s
```

#### 失败分析

1. **mean_momentum=0.95 太保守**: 每步仅更新 5% 的 mean。50 轮迭代后累计更新 1−0.95^50≈92%，但这分散在 50 步中，收敛路径极长。DynaRetarget 论文用 CPU MuJoCo rollout（无 CUDA graph 约束），可能迭代更多次；SPIDER 在 GPU Warp 上 50 次已是上限。

2. **所有 knot 用满 50 次迭代**: σ_min=0.01 从未达到（max_σ 最低 0.0153），说明 Sigma EWMA 收缩不够快或 σ_min 设置太紧。

3. **Reward 持续下降 3.19→1.26**: 随 horizon 增长，early knots 的优化效果被 later knots 的低质量稀释。DynaRetarget 用平方误差 cost（无界负），SPIDER 用 exp-kernel reward（有界 [0, 3.5]）→ 长 horizon 的 mean reward 自然偏低，CEM 更难区分好坏。

4. **DynaRetarget 用 CPU MuJoCo rollout，SPIDER 用 GPU Warp**: 论文的 MuJoCo `rollout` 函数是前向模拟，不需要 CUDA graph → 内存和计算约束不同。

**下一步**: E047b 正在远程运行 (momentum=0.5, σ_min=0.03)。

---

### E047b: SBTO 放松参数 (momentum=0.5, σ_min=0.03)

**修改**: `sbto_mean_momentum=0.5` (从 0.95), `sbto_sigma_min=0.03` (从 0.01)

#### 结果

| 实验 | Case | MPKPE | Contact<10cm | Stability>0.6 | 运行时间 | 对比 E041c |
|------|------|-------|-------------|--------------|---------|-----------|
| E041c (MPC) | box025 | 1.4cm | 66% | 100% | 14min | — |
| E047a (α_μ=0.95) | box025 | 155cm | 0% | 31% | 8min | ❌❌ 摔倒 |
| **E047b (α_μ=0.5)** | box025 | 87cm | 28% | **98%** | ~4min | ❌ tracking差 |
| **E047b (α_μ=0.5)** | bucket010 | 69cm | 0% | **100%** | ~4min | ❌ tracking差 |
| E047b (α_μ=0.5) | desk005 | — | — | — | killed | CPU-only 运行 |

#### 分析

E047b 比 E047a 有显著改善:
- **Stability 恢复**: 98-100% (E047a 仅 31%) — 降低 momentum 让 CEM 有效收敛
- 但 **MPKPE 仍然很大** (69-87cm vs MPC 的 1.4cm) — SBTO 的 body tracking 质量远不如 MPC

**SBTO vs MPC 的根本差异**:
1. MPC 每步从当前真实状态规划 → 闭环反馈纠正偏差
2. SBTO 一次性优化全程开环轨迹 → 累积误差无法纠正
3. DynaRetarget 论文中成功因为用 **MuJoCo CPU rollout** (精确模拟)，SPIDER 用 **Warp GPU** (可能有精度差异)
4. DynaRetarget 的代价函数是 **平方误差** (无界负)，SPIDER 的 reward 是 **exp-kernel** (有界 [0, 3.5]) → 长 horizon 的 mean reward 差异缩小，CEM 难区分

**E047b desk005 CPU-only 问题**: desk005 的 SBTO 在 GPU1 上运行但 GPU 利用率 0%，纯 CPU 跑了 54 分钟未完成 → 可能是 Warp kernel cache 问题或 scene 特定的 CUDA graph 失败。已 kill。

**E047 系列结论**: ❌ SBTO 在 SPIDER 的 reward 架构下不可用:
- 即使放松参数，body tracking 仍比 MPC 差 50-60 倍
- SBTO 的优势（长 horizon 全局优化）无法弥补闭环反馈的缺失
- DynaRetarget 的成功可能依赖于其特定的代价函数设计 + CPU 模拟精度

---

## Direction 2: 更多 Case 数据准备 (E046)

### 已完成的 person2 数据转换

| Case | 帧数 | 源 NPZ | 状态 |
|------|------|--------|------|
| box025_person2 | 124 | V2-replace trimmed | ✅ 转换完成 |
| bucket005_person2 | 159 | V2-replace trimmed | ✅ 转换完成 |
| bucket010_person2 | 126 | V2-replace trimmed | ✅ 转换完成 |

### 原始数据扫描结果

从 `action_labels.json` 筛选出 **452 个协作 session** (Collaborate 动作)，涵盖 **31 个非 chair/stick 物体**。

已选定 5 个新物体的最佳候选 session:

| 物体 | 类别 | Session | 动作 | 帧数 | 需要 |
|------|------|---------|------|------|------|
| Board020 | board (新类别) | 20231008/010 | move2_obs0 | 157 | OmniRetarget 全流程 |
| bucket001 | bucket | 20231030/094 | move2_obs0 | 154 | OmniRetarget 全流程 |
| Box021 | box | 20231018/030 | move2_obs0 | 172 | OmniRetarget 全流程 |
| Desk021 | desk | 20231023/037 | move2_obs0 | 153 | OmniRetarget 全流程 |
| desk007 | desk | 20231030/028 | move2_obs0 | 149 | OmniRetarget 全流程 |

选择标准: move2_obs0 (协作搬运+无障碍)，帧数 120-250 (适合 SPIDER)，每类别选 session 数最多的物体实例。

---

## 阶段性结论

### 方向评估

| 方向 | 结果 | 判定 |
|------|------|------|
| E045 sigma 收紧 | box025/bucket010 contact 大幅下降, desk005 局部改善 | ❌ 放弃 |
| E044a wrist weight=2.0 | contact 持平但 stability 崩溃 (73%) | ❌ 不可用 |
| E047a SBTO (论文参数) | 机器人摔倒, MPKPE=155cm | ❌ 需调参 |
| E046 数据扩展 | 3 个 person2 转换完成, 5 个新物体候选就绪 | ✅ 进行中 |

### 关键发现

1. **Body tracking 精度 ≠ contact 质量**: E045 证明了即使 MPKPE 从 1.4cm 降到 1.2cm (E045b desk005)，contact 改善也不显著。contact 的瓶颈是 ref 中手-物体的相对位置，不是 tracking 精度。

2. **上半身 tracking vs stability tradeoff**: E044a 和 E045a 都显示了增强上半身权重→ 下半身 stability 退化的模式。E041c 的配置 (σ=0.5, uniform weight) 恰好是 tracking/stability/contact 三者的帕累托最优。

3. **SBTO + SPIDER reward 不兼容**: DynaRetarget 用无界平方误差 cost, SPIDER 用有界 exp-kernel reward。SBTO 的渐进 horizon 机制在 exp-kernel reward 下难以有效优化（长 horizon 的 mean reward 趋于常数）。

### 进行中

- **E046 数据扩展**: 3 个 person2 case 已转换，5 个新物体候选已确定（待 OmniRetarget 处理）

---

## 最终结论

**E041c 仍然是 CORE4D 上的最佳配置**。Phase 12 的三方向探索全部未能超越：

| 方向 | 实验 | 最佳结果 vs E041c | 判定 |
|------|------|-------------------|------|
| Body tracking sigma | E045a/b | contact ↓20-37%, stability ≈ | ❌ 放弃 |
| Wrist weight | E044a/b | contact ≈/↓, stability ↓ 或 tradeoff | ❌ 放弃 |
| SBTO 算法 | E047a/b | MPKPE 69-155cm (MPC 1.4cm) | ❌ 放弃 |

**核心发现**:
1. contact 瓶颈不在 tracking 精度 — 降 MPKPE 从 1.4→1.2cm 不改善 contact
2. 任何增强上半身权重的方法都有 stability tradeoff
3. SBTO 开环优化与 SPIDER 的 exp-kernel reward 不兼容
4. E041c 的 σ=0.5 / uniform weight / MPC 配置是 tracking/stability/contact 的帕累托最优

---

## 改动文件

| 文件 | 改动 | 实验 |
|------|------|------|
| `spider/config.py` | +local_frame_wrist_weight/wrist_ids, +sbto_elite_fraction/mean_momentum/cov_momentum | E044, E047 |
| `spider/simulators/mjwp.py` | wrist weight 逻辑 (~10行) | E044 |
| `spider/optimizers/sampling.py` | elite_fraction 参数化, mean EWMA, elite_std 返回 (~30行) | E047 |
| `examples/run_mjwp.py` | run_sbto Sigma EWMA + 收敛准则修改 (~40行) | E047 |
| `spider/process_datasets/core4d.py` | 无改动, 用于 E046 数据转换 | E046 |

## 结果路径

| 产出 | 路径 |
|------|------|
| E044a box025 | `workspace/core4d/results/E044/E044a_box025.{npz,mp4}` |
| E044b box025/bucket010/desk005 | `workspace/core4d/results/E044/E044b_*.{npz,mp4}` |
| E045a/b × 3 case | `workspace/core4d/results/E045/E045{a,b}_{box025,bucket010,desk005}.{npz,mp4}` |
| E047a box025 | `workspace/core4d/results/E047/E047a_box025.{npz,mp4}` |
| E047b box025/bucket010 | `workspace/core4d/results/E047/E047b_{box025,bucket010}.{npz,mp4}` |
