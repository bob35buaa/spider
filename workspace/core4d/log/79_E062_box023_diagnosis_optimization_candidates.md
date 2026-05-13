# E062 box023 深度诊断 + 优化候选 (Tier 1/2/3)

## 状态: 🔬 **DIAGNOSIS — sim 实际完成了 task (final object pos 仅 9.8cm 偏离 ref), 但在 carry→place 转换 (t=2.0-2.7s) 摔倒, 然后 active recovery 站起. 摔倒元凶是 task_obj_rew 的强力 forward 牵引 + 缺 stability_penalty + 重心追物超越脚 → 优化方向: Tier 1 = re-enable stability_penalty + 减小 task_obj 权重**

**TL;DR**: 用户决定不看 box025 (数值不退化, 视觉差异是 CEM 噪声), 专注 box023. 提取 ref vs sim 的物体轨迹 + pelvis_z 全帧对比发现: (a) ref 0-1.7s 弯腰提箱 → 1.7-2.0s 直立搬走 → 2.0-2.5s 弯腰放下 → 2.5-4.5s 站起完成. ref pelvis_z 全程 0.70-0.81m. (b) sim 0-1.7s 跟着 ref 提箱 (滞后 0.2m) → 1.7-2.0s 试图跟上 ref 直立但**已经倾斜** (pelvis 0.68→0.42) → 2.0-2.7s **摔倒** (pelvis 0.42→0.06m), 物体被沿轨迹推到 (0.37, -0.14, 0.16) 距 ref 仅 6cm → 2.7-4.5s push-up 站起. **sim 完成了 task** (final object 距 ref 只 9.8cm), **但代价是中段摔倒**. 摔的物理原因: ref 在 t=2.0-2.5s 边走边放箱 (forward velocity 0.55m/0.5s), sim 为追物体被 task_obj_rew 强力牵引向前 → 重心超出脚底 → 栽倒. **优化候选 Tier 1**: re-enable stability_penalty (E041c 当前 0.0) + 减小 task_obj_pos/rot_rew_scale (1.0→0.5).

## 1. 用户决定: 聚焦 box023, 跳过 box025

> 那就不看 box025 了,数值 ok,那证明我们刚刚的改动不太影响 box025

box025 E062 vs E061: pelvis_min 0.688 vs 0.672 (+1.6cm), reward bytewise 等于 hardcoded, 视觉差异是 CEM 不同 random seed 的局部最优. 不再分析 box025. 跳过 X5 推 4 cases 之前, 先把 box023 解决.

## 2. ref vs sim 物体轨迹 + pelvis_z 全帧对比

| 时间 | ref obj xyz | sim obj xyz | obj 偏差 | ref pz | sim pz | sim phase |
|------|------------|------------|---------|--------|--------|-----------|
| 0.00s | (-0.83, -1.18, 0.14) | (-0.83, -1.18, 0.15) | 0.007m | 0.79 | 0.79 | 站立起步 |
| 0.33s | (-0.83, -1.18, 0.14) | (-0.83, -1.18, 0.16) | 0.016m | 0.80 | 0.80 | 接近 |
| 0.67s | (-0.83, -1.18, 0.14) | (-0.83, -1.18, 0.16) | 0.016m | 0.63 | 0.61 | 弯腰开始 |
| 1.00s | (-0.76, -1.12, **0.40**) | (-0.82, -1.16, **0.20**) | 0.216m | 0.73 | 0.61 | **ref 提箱, sim 滞后** |
| 1.33s | (-0.51, -0.86, 0.64) | (-0.67, -0.92, 0.42) | 0.270m | 0.77 | 0.65 | sim 持续滞后 |
| 1.67s | (-0.13, -0.55, **0.66**) | (-0.36, -0.72, **0.54**) | 0.312m | 0.78 | 0.68 | **box 已抬到 chest, sim 半抬** |
| 2.00s | (+0.24, -0.30, 0.54) | (+0.03, -0.40, 0.48) | 0.245m | 0.79 | **0.42** | ⭐ **sim 开始倒** |
| 2.33s | (+0.42, -0.18, 0.23) | (+0.34, -0.23, 0.27) | 0.102m | 0.73 | **0.13** | ⭐ **sim 摔** |
| 2.66s | (+0.46, -0.16, 0.15) | (+0.37, -0.14, 0.16) | 0.097m | 0.70 | **0.06** | ⭐ 最低点 (趴地) |
| 3.00s | (+0.46, -0.16, 0.15) | (+0.40, -0.15, 0.16) | 0.063m | 0.81 | 0.60 | push-up 起 |
| 3.33s | (+0.46, -0.16, 0.15) | (+0.43, -0.15, 0.16) | 0.036m | 0.81 | 0.76 | 站立 |
| 3.66s | (+0.46, -0.16, 0.15) | (+0.44, -0.16, 0.16) | 0.026m | 0.81 | 0.74 | 直立 |
| 4.00s | (+0.46, -0.16, 0.15) | (+0.44, -0.16, 0.16) | 0.020m | 0.81 | 0.76 | 直立 |
| 4.33s | (+0.46, -0.16, 0.15) | (+0.45, -0.15, 0.16) | 0.018m | 0.79 | 0.78 | 直立 |

**sim 视频日志报告: "Final object tracking error: pos=0.0981, quat=0.5035"** — 终态 box 离 ref 只 9.8cm.

## 3. 关键诊断

### 3.1 sim 实际"完成了"任务

- 起始位置 (-0.83, -1.18, 0.14) → 终止位置 (+0.45, -0.15, 0.16): box 沿对角线移动 ~1.6m
- sim 的物体终止位置只比 ref 偏 9.8cm
- task_obj_pos_rew 是 PENALTY (mean -1.75, min -9.27): sim 一直在被惩罚因为 box lag, 大部分被减小
- 摔倒后 (t≥3.0s) sim 的 box 跟踪已经 < 7cm 偏差

**这跟 E048 historical 截然不同**: E048 box023 sim 摔倒后**保持仰面躺地物体不跟随**. E062 sim 摔倒后**站起来, box 也跟到位**.

### 3.2 摔倒物理: forward 牵引超过脚底支撑

ref motion 在 t=1.67s → 2.50s 内**走 0.59m + 放下** (ref obj x: -0.13 → +0.45, ref pz: 0.78→0.70 即弯腰放). 这是 **ref person 边走边放, 重心同步向前移动**.

sim 在追 ref:
- t=1.67s: sim obj x=-0.36 (滞后 ref 0.23m)
- t=2.00s: sim obj x=+0.03 (赶上, 走了 0.39m 0.33s 内, 速度 1.18 m/s)
- t=2.00-2.50s: sim obj 继续 +0.39 (但 sim pelvis 已 0.42→0.13)

**hypothesis**: sim 为了追 task_obj_rew 急速向前移动手 + 上身, **重心快速前移**. ref 的人能边走边平衡 (双脚轮换), 但 sim 因为 G1 短臂展 + 平衡不熟练, 没能跟脚同步 → 上身先到位, 脚未到位 → 摔.

### 3.3 reward 配置审查 (E041c, 当前)

| Param | 值 | 作用 | 对 box023 摔倒的影响 |
|-------|-----|------|----------------------|
| `task_obj_pos_rew_scale` | 1.0 | 物体位置跟踪 | **HIGH** — 强 penalty 让 sim 急追物体 |
| `task_obj_rot_rew_scale` | 1.0 | 物体朝向跟踪 | 中 — 加剧追逐 |
| `contact_hdmi_gain` | 5.0 | 手到目标点 | **HIGH** — 强力把手拉向 box, 上身被带 |
| `local_frame_root_sigma` | 0.5 | root 跟踪松紧 | 偏松 (0.5m), 允许 root drift 过大 |
| `local_frame_w_track` | 0.5 | local frame 整体权重 | 中 |
| `stability_penalty_scale` | **0.0** | 摔倒惩罚 | **DISABLED** — sim 摔到 0.06m 没有任何额外惩罚 |
| `stability_penalty_threshold` | 0.55 | 摔倒阈值 | 配置存在但未使用 |
| `contact_hdmi_threshold` | 0.15 | 接触有效距离 | 中 |

**关键发现**: `stability_penalty_scale = 0.0`. 这个 reward 在 E034 时代被引入 (惩罚 pelvis_z < threshold 的帧), 在 E036 时被关闭 (因为 contact reward 还没成熟, 关掉避免冲突). E041 系列一直没重启它. **sim 摔到 0.058m 没有任何 penalty 信号告诉 CEM "这是个坏解"**, 反而 task_obj_rew 仍能给正信号 (sim 摔倒时手仍在 box 上).

### 3.4 视觉证据 (5 dense 帧)

| 时间 | sim 行为 | 解读 |
|------|---------|------|
| t=1.7s | 弯腰前推 box, 右手在 box 上, box 在 sim 前方 | carry attempt, 已经身体倾斜 |
| t=2.0s | **大幅前倾, 右脚踩在 box 边角, 整个上身扑向 box** | 失去平衡的临界帧 |
| t=2.3s | 完全趴地, 双臂伸向 box | DEEP FALL phase |
| t=2.7s | 完全趴地头朝下, 双臂展开抓 box (pz 0.058m) | 最低点 |
| t=3.0s | push-up 起身 (类似俯卧撑姿势, 身体水平脚撑地) | active recovery 开始 |

t=2.0s 是关键: sim 把右脚踩在 box 边角, 这是身体重心前移到无支撑区域的标志. 0.3s 后栽倒.

## 4. 优化候选 (按工程量 + 预期影响排序)

### Tier 1 (低成本, 高预期影响)

**T1-A: re-enable stability_penalty**
```yaml
stability_penalty_scale: 1.0   # 当前 0.0, 改回 E034 默认
stability_penalty_threshold: 0.55  # 已配置, 不变
```
- 直接惩罚 pelvis_z < 0.55m 的帧
- E062 sim 在 t=2.0-2.7s 7-8 帧 pz < 0.55, 这些帧每帧 -1.0 reward → CEM 强烈避免摔倒解
- **预期**: 摔倒被压制, sim 可能选择不那么激进追物体, 接受 task_obj 偏差以保平衡
- 风险: 若 stability 权重太大, sim 可能完全不弯腰 → 完不成 task

**T1-B: 减小 task_obj 权重**
```yaml
task_obj_pos_rew_scale: 0.5    # 当前 1.0
task_obj_rot_rew_scale: 0.5    # 当前 1.0
```
- 减弱"急追物体"压力
- sim 允许 box lag 更多, 换取自身平衡
- **预期**: sim 减速 carry, 更注重 footwork, 可能完成时 box 偏差从 9.8cm 增到 15-20cm 但全程不摔
- 风险: 若过分减小, sim 可能放弃 carry 任务

**T1 推荐: T1-A + T1-B 组合 single yaml**, 在 E063 跑一次

### Tier 2 (中成本, 中预期影响)

**T2-A: 增加 root tracking 严格度**
```yaml
local_frame_root_sigma: 0.3    # 当前 0.5, 减小让 root tracking 更严
```
- 缩小 sigma → root 偏差 (sim pz vs ref pz) 更敏感
- 若 sim 试图跟 ref 同样 forward 速度但 sigma 紧, 偏差大就被惩罚
- **预期**: 更接近 ref 站姿, 但可能干扰自由探索 footwork

**T2-B: 减小 contact_hdmi_gain**
```yaml
contact_hdmi_gain: 3.0    # 当前 5.0, 减小手到目标的拉力
```
- 弱 contact reward → sim 不那么急切伸手追 box
- **预期**: 站姿更稳, 但 contact% 可能下降

### Tier 3 (高成本, 不确定)

- **T3-A**: 加 walking phase reward (鼓励双脚交替) — 改 reward 代码, 复杂
- **T3-B**: object mass / inertia 调整 (减轻 box023 让 sim 更易追) — 数据层改, 但不是 reward 问题
- **T3-C**: ref motion smoothing (降低 ref 的 forward 速度) — 改 ref 数据, 失去 fidelity
- **T3-D**: warmstart hook (sim 跳过 0-1s 接近阶段) — E058 已试过, 不能修平衡问题

### 不做的方向 (defer)

- **X2 (auto eef_offset)**: 在 sphere 上是 5cm 误差, 跟 box023 摔倒的 forward-balance 问题不直接相关, 优先级低
- **3-box 重新评估**: log 77 已决策放弃, 不再考虑
- **修 box025 视觉扭曲 (P1-P4)**: log 77 决策暂不修, 不变
- **推 X1 到 4 cases (X5)**: 推迟. box023 通过后再推, 否则推过去都摔意义不大

## 5. 推荐下一步: E063 = Tier 1 (T1-A + T1-B)

**单一 yaml**: `examples/config/override/core4d_e063_box023.yaml`
```yaml
defaults:
  - core4d_e062_box023   # 继承 E062 (E041c + auto X1 palm_normal)
  - _self_

# E063 Tier 1: stability + reduce task_obj
stability_penalty_scale: 1.0           # E034 default, was 0.0 in E041c
task_obj_pos_rew_scale: 0.5            # 50% reduction from E041c
task_obj_rot_rew_scale: 0.5            # 50% reduction from E041c
```

**训练**: 在 box023 上跑一次 (~30min), 对比 E062 box023.

**Pass criteria**:
- C1: pelvis_min ≥ 0.5m (E062 0.058 → ≥ 0.5)
- C2: pelvis_mean ≥ 0.5m (E062 0.602 → 维持或更好)
- C3: stable% ≥ 80% (E062 77.9% → 类似或更好)
- C4: final object error ≤ 20cm (E062 9.8cm, 允许放宽到 20cm 因为 task_obj 减权重)
- C5: 视觉无 mid 摔倒 phase (5 dense frames around t=2.0-2.7 都站立)

**Decision tree**:
- C1+C2+C3 PASS, C4 ≤ 20cm: T1 成功, X1+T1 是泛化方案, 推 X5 (4 cases) 验证
- C1 PASS but C4 > 30cm: stability_penalty 过强压制了 task, 需调低 stability_penalty_scale (0.5 而非 1.0)
- C1 FAIL (仍摔): T1 不够, 进 T2 (root sigma + contact gain)
- C5 视觉发现新问题: 单独诊断

## 6. 改动文件 (本 log)

| 文件 | 改动 |
|------|------|
| `workspace/core4d/log/79_E062_box023_diagnosis_optimization_candidates.md` | 本文件 |
| `workspace/core4d/EXPERIMENT_TRACKER.md` | 加 diagnosis 行 + log 79 索引 |

(E063 yaml + train script 留到执行 plan 时新建.)

## 7. 教训 #8 (累计)

之前 7 个教训 (log 78 §9):
1. 看视频前先看数据
2. ablation 多 case
3. 物理 port 必须回归测试
4. 跨 sub-experiment 假设链每一步重验
5. reward 工作必须 ≥2 case × ≥2 hand 横向验证
6. 视觉描述必须区分实验版本
7. 自动算法 self-consistency 通过 ≠ 对新 case 有效

**新教训 #8 (本 log)**:

> **诊断 reward 问题必须看 ref vs sim 的物体轨迹 + pelvis_z + reward 组件全帧对比**, 不能只看 pelvis_min 一个数值. E062 box023 pelvis_min 0.058 看起来 catastrophic, 但实际 sim 完成了 task (object 终态 9.8cm), 摔倒只是 mid 阶段的 transient (4-7 帧 / 136 帧). 之前 E060.0/.1/.2 reward ablation 只看 pelvis_min/stable% 没看 ref vs sim 物体轨迹, 错过了 sim 是否真的在做 task vs 完全脱节这个关键区分.

具体规则:
- 任何 reward ablation log **必须**包含 ref vs sim 的物体 xyz 全帧对比 + 终态偏差数值
- 任何 "sim 失败"声明 **必须**先看 sim 是否在做 task (object tracking error) vs 完全脱节
- pelvis_min < 阈值 ≠ "失败", 可能是 transient 摔倒后 recovery; 看 stable% 和 mean 更重要

## 8. 关联

- log 76 §5: X1+X2 reward 泛化方向 — 本 log 是 X1 单跑后的诊断
- log 77 §6 P1-P4: E041c box025 视觉问题 — 不同于 box023 的 task-completion-with-fall 问题
- log 78 §6: E062 mixed result, 列出 X5/X1+X2/X3 三个候选 — 本 log 优先选 X3 (诊断 + 调 reward weights)
- audit log 70 §1.2: stability_penalty 历史 ("E036 关闭, 之后没重启") — 本 log 提议重启
