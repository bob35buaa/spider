# E085 Plan: Box021 exit gate and seed routes after E084

日期：2026-05-28

## Context

E082/E083/E084 连续说明：

- 只加腿/脚-箱 collision pair：不能解决 Box021，仍倒伏/推箱；
- 再加 upper-body-object collision pair：从深穿箱变成趴箱/压箱；
- 再加 safety / upright / semantic lift reward：三组 main gate 全失败。

E084 最关键现象：

| 组 | 结果 | 机制 |
|---|---|---|
| A safety | pelvis 稳、手不撑地，但箱子不抬，上身仍压箱 | penalty 抑制部分坏动作，但未产生承重 lift |
| B upright | object err/bottom gap 改善，但接触掉到 `9.3%`，箱体翻倒 | 姿态约束与接触搜索冲突 |
| C semantic | 类似 A，contact 高但 object-floor `96.9%` | soft lift reward 不足以改变局部最优 |

因此 E085 不继续做同类 reward 小调参，而是设为 exit-gate 计划：先判断 Box021 main 是否有可行动力学 seed；若没有，停止把它作为 CEM 正例；若有，再进入更强的 staged / hard-gate CEM。

## 实验范围

先只做：

```text
main:  d003_box021_20231018_029_p2
guard: box023_person2
```

不扩展到另外两个 Box021 main，除非 E085B 或 E085C 产生明确可用 seed。

## Group A: feasibility / seed audit

### Hypothesis

E084 失败可能不是 reward scale 问题，而是输入 seed/ref 本身对单 G1 不可行，或 contact/lift window 中手-箱、上身-箱、身体稳定三者互相冲突。

### 任务

对以下轨迹统一 replay + 诊断：

- source kinematic ref；
- E018/E018b 或已有 OmniRetarget baseline，如果本地存在；
- E083/E084A/E084B/E084C outputs；
- `box023_p2` positive guard 作为对照。

计算：

- hand-object adjusted SDF、hand contact mask 命中率；
- upperbody-object adjusted SDF；
- leg/foot-object interference；
- hand-floor clearance；
- object bottom vs ref、object floor-contact、object tilt；
- pelvis height、torso vertical、support foot contact；
- lift window 内手是否位于能施力的 face/edge，而不是箱体顶部/背侧。

### 成功标准

- 产出每条轨迹的可行性标签：`feasible_seed` / `bad_contact_geometry` / `requires_external_support` / `retarget_pose_bad`；
- 明确 `20231018_029_p2` 是数据/seed 问题还是 CEM 问题；
- 若没有任何 seed 同时满足 upperbody safe + hand semantic + object lift，则不进入 hard CEM。

### 计划产物

```text
workspace/core4d/results/E085/seed_audit/
workspace/core4d/results/E085/seed_audit/comparison.csv
workspace/core4d/results/E085/seed_audit/contact_sheets/
workspace/core4d/log/107_E085_seed_audit_results.md
```

## Group B: kinematic/support seed route

### Hypothesis

CEM 从当前 ref/control 出发找不到承重接触，但如果先给它一个“箱体由外部支撑、机器人姿态合理”的 seed，后续 RL 或 constrained CEM 可能可用。这个 seed 不要求完全物理搬箱，但必须避免趴箱、翻箱、手撑地。

### 两个变体

| Variant | 设计 | 目的 |
|---|---|---|
| E085B1 object-kinematic seed | 物体按 ref/filtered ref 运动，机器人只优化姿态和手部语义接触 | 验证机器人姿态和手位是否可行 |
| E085B2 COLA-style support-body seed | support body 与物体用 6-DoF joint / connector 支撑，机器人不负责全部物体重量 | 更接近协作搬运，生成可交给 RL 的 support seed |

### 关键约束

- object/support 轨迹可由 ref 低通生成，禁止 anchor 手工硬编码；
- hand contact 只允许 hand geoms 作为正语义；
- upperbody/leg/floor collision 仍作为硬失败指标；
- 输出必须可 replay、可导出、可和 E084 指标对齐。

### 成功标准

- main visual 不再蹲抱/压箱；
- upperbody penetration `<5%`；
- hand-floor `<5%`；
- object bottom gap `>= -5cm`；
- sim hand contact `>=50%`；
- `box023_p2` guard 不退化。

### 计划产物

```text
workspace/core4d/results/E085/support_seed/
workspace/core4d/scripts/E085/
workspace/core4d/scripts/train/train_E085.sh
workspace/core4d/scripts/eval/eval_E085.py
```

## Group C: hard-gate staged CEM

### Hypothesis

如果 B 能给出合理 seed，当前问题才值得继续用 CEM 解。此时需要把 E084 的 soft reward 改成 staged hard gate，而不是继续调 scale。

### 设计

两阶段：

1. Pre-lift / approach stage：只要求站稳、双手到语义接触面、禁止 upperbody/leg/floor 错误接触；
2. Lift / hold stage：启用 object bottom hard gate、object tilt hard gate、hand-only contact gate。

Reward / gate 形式：

- contact reward 只有在 upperbody safe + hand-floor safe + object tilt safe 时生效；
- object bottom 低于 ref bottom - margin 时直接大负 reward；
- object tilt 超阈值直接大负 reward；
- upperbody/leg penetration 使用 min-SDF hard hinge，不再只做小 scale shaping；
- 若 contact 归零，记录为“CEM 找不到可行解”，不继续放宽成趴箱解。

### 成功标准

- main 通过 E084 gate；
- 视觉不是压箱/翻箱/手撑地；
- 若 main 失败但 contact 归零，应视为“正确排除错误局部解”，然后停止 Box021 正例路线。

## 决策树

| 结果 | 下一步 |
|---|---|
| A 判定 source/ref/Omni seed 本身不可行 | 停止 Box021 正例；只保留为 negative/diagnostic case |
| A 找到可行 kinematic seed，但 B 失败 | seed 转换/支撑模型有问题，优先修 B，不跑 C |
| B 成功、C 失败且 contact 归零 | 说明 CEM 无法从该 seed 发现承重策略，转 RL/support policy，不再 CEM 小调参 |
| B 成功、C 成功 | E086 扩展到另外两个 Box021 main + `box023_p2` guard |
| B/C 在 main 过关但 guard 退化 | 回退该组，不推广 |

## 风险

| 风险 | 处理 |
|---|---|
| 本地找不到 E018/OmniRetarget baseline | Group A 先用现有 ref/E083/E084 outputs，缺失项标记为 unavailable |
| support seed 又变成 connect 假象 | 日志必须区分“seed 可用”与“真实物理搬箱”，不能当作 CEM/RL 成功 |
| hard gate 让 CEM 完全不接触 | 这是有价值的负结果，说明当前 seed 不可行动力学接触 |
| 评估再次误判 | 每个阶段都强制 contact sheet + subagent high 视觉复核 |

## 预期命令

```bash
# Group A: seed audit
bash workspace/core4d/scripts/run_E085_seed_audit.sh

# Group B/C: only after Group A supports feasibility
bash workspace/core4d/scripts/run_E085_preprocess.sh
bash workspace/core4d/scripts/train/train_E085.sh local 0
bash workspace/core4d/scripts/run_E085_remote.sh
bash workspace/core4d/scripts/pull_E085_remote_results.sh
```

这些脚本尚未实现；E085 第一阶段应先实现 audit，不直接启动新的 full CEM。
