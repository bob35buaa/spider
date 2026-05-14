# E074+ 总实验计划: E071/E073 之后的 post-2s hold/contact 修复路线

## Context

用户提出两个需要先澄清的问题：

1. **HDMI workflow 为什么可以 work，和当前 MJWP/E073 还有什么差异？**
2. **历史 contact/reward 改动是否因为 E071 修复前的 ctrl mapping bug 而污染；哪些合理改动可以纳入下一步实验？**

本计划基于：

- E070/E071: 修复 scene_act ctrl mapping，early drift 消失。
- E072: 定位 post-2s failure order 是 hold/contact 先失效，摔倒是后果。
- E073: 修正 dynamic target 的 `eef_offset` 口径，稳定性明显改善，但 f130 后仍脱手，f145 箱落地。
- subagent Halley: 复盘 HDMI workflow 差异。
- subagent Euler: 复盘 E037-E067 contact/reward 历史改动。

## 结论重置

### 1. HDMI workflow 的旧解释要改写

旧 R4/log87 里认为 HDMI work、MJWP fail 的主因可能是 init pose bug / pre-contact body tracking。E070/E071 后，这个解释已经不完整：

- 旧 MJWP 的 pre-contact lunge 很大程度来自 `qpos_ref[:, :nu]` 被误当 robot actuator ctrl。
- E071 修复后，不加 foot tracking、不改 reward，box023 early yaw 降到 `0.574/1.075 deg`，B1 pre-contact max foot z 降到 `0.069m`。
- 因此 HDMI 与当前 E073 的主要差异已经转移到 **post-2s hold/contact 和 CEM 控制偏离**，而不是 0-2s body tracking。

### 2. E060-E067 大部分归因被污染

这些实验都在 E071 修复前解释 early lunge / pre-contact failure：

| 实验 | 当时改动 | 现在判断 |
|------|----------|----------|
| E060 | 3-box / ori ablation / case-correct palm | 被 3-box regression + ctrl mapping bug 双重污染，不作为 E074 依据 |
| E062 | auto palm normal | 代码/工具可留，但“box023 pre-contact failure”结论需重跑才算 |
| E063/E064 | stability/task_obj/root_sigma/contact_gain 调参 | 当时针对错误 failure mode，不纳入下一步 |
| E065 | task_obj L2/drop/exp | 不应作为 post2 hold 的首要方向 |
| E066 | HDMI soft actuator | 会破坏 object tracking，不优先 |
| E067 | HDMI narrow body partition | 明确 catastrophic，不再重复 |

可信保留：

- E039b 的 config/body id 解析 bug 修复。
- E040 dynamic per-frame target 比 fixed target 更合理。
- E041c additive orientation reward 作为 legacy base 可用，但不能再视为泛化最优。
- E071 ctrl mapping fix 是硬前提。
- E073 `contact_hdmi_target_uses_eef_offset=true` 是可信小修，应作为下一步 base。

## 当前事实

E073 相对 E071/E072：

| 指标 | E071/E072 | E073 | 判断 |
|------|----------:|-----:|------|
| yaw err t=0.017/0.033 | 0.574/1.075 deg | 0.574/1.075 deg | early drift 未回归 |
| B1 pre-contact max foot z | 0.069m | 0.080m | 仍合格 |
| first obj_err >25cm | frame100 | frame100 | 未解决 |
| first sim zero contact | frame100 | frame108 | 小幅改善 |
| post2 sim contact frames | 44.4% | 49.4% | 小幅改善 |
| post2 obj_err max | 0.308m | 0.293m | 小幅改善 |
| first pelvis_z <45cm | frame166 | none | 摔倒消失 |
| 视觉 | f130 后脱手，f145 箱落地 | 同样未持箱 | hold 未解决 |

关键解释：

- E073 不是失败在 0-2s init/body tracking。
- E073 不是简单 stability failure；它已经“不倒”，但仍“没拿住”。
- 当前一阶问题是 **frame100-145 hand-object contact 不连续**。

## 总体策略

下一步不要再做大范围 reward sweep。采用“最小变量、同一评估口径”的分层实验：

1. 先验证 **CEM robot ctrl 偏离 ref 是否导致 hold 断裂**。
2. 再验证 **显式 hold/contact continuity reward 是否能补上 contact**。
3. 若两者单独有效，再组合。
4. 若仍失败，再转向 contact geometry / data grasp 诊断。

## 统一评估口径

所有 E074+ 实验必须输出同一套指标，避免再被单一 KPI 误导：

| 指标 | 目的 |
|------|------|
| yaw err t=0.017/0.033 | early drift regression guard |
| B1 pre-contact max foot z | 0-2s body tracking guard |
| first obj_err >25cm frame/time | 是否还在 2.0s 立即丢物体 |
| first sim zero contact frame/time | contact 首次断裂时间 |
| frame100-145 sim/ref contact frames | 主窗口 hold 质量 |
| post2 sim/ref contact frames | 整体 hold 质量 |
| frame100/115/130/145/160 obj_err | 明确定位 f130/f145 |
| post2 pelvis_z min | 稳定性 guard |
| robot ctrl Linf/L2 vs ctrl_ref | 验证 trust-region 是否生效 |
| object ctrl Linf vs ctrl_ref | 排除 object ctrl mapping 回归 |
| keyframes f100/f115/f130/f145/f160/f166/f180 | 必须视觉复核 |

成功不能只看 pelvis_z。主标准必须包含：**f130/f145 视觉上仍是持箱/托箱，而不是箱已落地**。

## E074: Robot Ctrl Trust-Region Guard

### 目标

验证 post-2s hold 失败是否来自 CEM 在 frame100 后允许 robot actuator ctrl 快速偏离 ref。

### Hypothesis

E073 中 `robot_ctrl_linf` 在 post2 达到 `0.778`，E072 中 frame109 后已经超过 `0.5`。如果加入 robot ctrl trust-region，应该：

- 减少 frame100-145 的断触；
- 推迟 first zero contact；
- 降低 post2 obj_err max；
- 不以摔倒换 contact。

### 改动

在 `spider/simulators/mjwp.py::get_reward()` 新增可选 reward：

```python
ctrl_ref_guard_rew = -scale * mean(huber(ctrl_sim_robot - ctrl_ref_robot))
```

建议字段：

```python
ctrl_ref_guard_scale: float = 0.0
ctrl_ref_guard_robot_only: bool = True
ctrl_ref_guard_sigma: float = 0.25
ctrl_ref_guard_start_eval_time: float = 1.8
ctrl_ref_guard_end_eval_time: float = 3.0
```

实现注意：

- 只比较 robot actuator 维度，不惩罚 object actuator 末 6 维。
- 初版使用 reward penalty，不直接 clamp；clamp 作为后续 E074b。
- gate 时间窗口覆盖 ref 仍应接触的 1.8-3.0s。
- 必须在 info 中输出 `ctrl_ref_guard_rew`。

### 变体

| Run | Base | 单一改动 | 目的 |
|-----|------|----------|------|
| E074A | E073 | `ctrl_ref_guard_scale=0.5` | 温和 trust-region |
| E074B | E073 | `ctrl_ref_guard_scale=1.0` | 较强 trust-region |

如果只能先跑一个，推荐 E074A。

### 成功标准

| 指标 | E073 | E074 目标 |
|------|-----:|----------:|
| yaw 0.017/0.033 | 0.574/1.075 deg | <2 deg |
| B1 foot z | 0.080m | <=0.10m |
| first zero contact | frame108 | > frame115，最好 > frame130 |
| frame100-145 contact | 45.7% | >=65% |
| post2 contact | 49.4% | >=60% |
| post2 obj_err max | 0.293m | <0.25m |
| post2 pelvis_z min | 0.663m | >=0.55m |
| 视觉 | f145 箱落地 | f130/f145 仍持箱/托箱 |

### 决策

| 结果 | 解读 | 下一步 |
|------|------|--------|
| contact/object 明显改善 | ctrl 偏离是主因 | E075 组合 hold continuity |
| pelvis 稳但 contact 不变 | ctrl guard 不够，缺直接 contact 约束 | E074C hold continuity |
| object tracking 变差但 contact 稳 | trust-region 过强 | 降 scale 或改 soft clamp |
| early drift 回归 | guard 实现错误或干扰 ref control | 回滚 E074 实现 |

## E074C: Hold/Contact Continuity Reward

### 目标

验证显式“ref 仍接触时，sim 至少保持单手真实接触/近场”的约束是否能解决 f130 后脱手。

### Hypothesis

E073 的 target 几何已修，但 contact reward 仍是“contact point 追 ref target point”，不是“真实 contact 持续”。CEM 可以让手在目标附近但不产生稳定接触。

### 改动

新增可选 reward 或诊断型 reward：

```python
hold_contact_rew = scale * ref_contact_mask * exp(-sim_min_hand_sdf / sigma)
```

建议字段：

```python
hold_contact_rew_scale: float = 0.0
hold_contact_sigma: float = 0.05
hold_contact_start_eval_time: float = 1.8
hold_contact_end_eval_time: float = 3.0
hold_contact_require_ref_contact: bool = True
```

实现方式：

- 初版不依赖 MuJoCo contact count 的不可微/离散信号，而用 SDF/min distance surrogate。
- 评估仍必须用真实 MuJoCo hand-object contact count 判断是否成功。
- 不直接加 stability_penalty，避免“站稳但不拿箱”的假成功。

### 变体

| Run | Base | 单一改动 |
|-----|------|----------|
| E074C | E073 | `hold_contact_rew_scale=2.0` |
| E074D | E073 | `hold_contact_rew_scale=5.0` |

如果 E074A 与 E074C 都要跑，可并行两 GPU。

### 成功标准

同 E074A，但更看重：

- frame100-145 sim contact >=65%
- first zero contact > frame130
- f130/f145 手不能只“压在箱上方”，要仍在持箱/托箱关系中

## E075: 组合验证

只在 E074A/B 或 E074C/D 至少一个方向有正信号后执行。

| Run | Base | 组合 |
|-----|------|------|
| E075A | best(E074A/B/C/D) | ctrl trust-region + hold continuity |
| E075B | E075A | mild stability guard，仅作 safety，不作为主驱动 |

E075 目标：

- first zero contact > frame130
- post2 sim contact >=65%
- post2 obj_err max <0.25m
- f130/f145 视觉可接受
- pelvis_z min >=0.55m

## E076: Contact Geometry / Data Grasp 诊断

如果 E074/E075 仍失败，需要停止调 reward，转向几何/数据：

### 诊断问题

1. CORE4D box023 的 ref grasp 是否物理上能由当前 G1 sphere hand 稳定托住？
2. E073 f130-f145 时 sim 手与 ref contact face 是否一致？
3. 左手/右手实际接触面是否和 E056/E057 诊断一致？
4. 当前 `contact_hdmi_eef_offset=[0.05,0,0]` 是否仍是最佳 sim contact point，还是只是 target 修了一半？

### 可能实验

| Run | 改动 | 目的 |
|-----|------|------|
| E076A | replay/ref-only contact face diagnosis | 不训练，只诊断 E073 f100-160 face/contact |
| E076B | auto palm normal on E073 base | 重新验证 E062 auto palm 在 E071 后是否有用 |
| E076C | eef_offset sweep `[0.03,0.05,0.08,0.10]` | 找手球/目标点最一致的 offset |
| E076D | sphere vs 3-box micro-regression | 只做短窗口，不直接大训练 |

注意：E076 之前不要直接重启 3-box port。历史已经证明 3-box 会造成严重 regression，除非先有短窗口 replay 证据。

## Regression Plan

在 box023 主问题有明显改善前，不跑大矩阵。达到 E075 阈值后，再跑：

| Case | 目的 |
|------|------|
| box023 | 主目标 |
| box025 | legacy regression guard |
| bucket005_s2 或 bucket010 | 非 box / bucket grasp 泛化 |
| desk005 | 稳定性与前倾失败 guard |

每个 regression 必须沿用同一 eval 脚本输出 contact/object/pelvis/ctrl/keyframes。

## 不建议重复的方向

| 方向 | 原因 |
|------|------|
| E065 task_obj drop/exp | E071 后 early failure 已修；这不是当前 post2 主问题 |
| E066 soft actuator | 会显著破坏 object tracking |
| E067 narrow body partition | catastrophic handstand，明确不再重复 |
| E063/E064 stability/root_sigma/contact_gain sweep | 针对错误 failure mode 调参 |
| wrist freeze | 历史上 contact 下降 |
| wrist weight / sigma tighten | 容易破坏 stability/contact tradeoff |
| SBTO | 开环 tracking 失败，不适合当前局部 hold 修复 |
| 先跑 box025 regression | 当前主失败面是 box023 post2 hold，先修主问题 |

## 文件与执行规划

### E074 需要新增/修改

| 类型 | 路径 |
|------|------|
| config | `spider/config.py` 增加 ctrl guard / hold reward 字段 |
| reward | `spider/simulators/mjwp.py` 增加可选 reward 项 |
| yaml | `examples/config/override/core4d_e074a_box023.yaml` 等 |
| train | `workspace/core4d/scripts/train/train_E074.sh` |
| eval | `workspace/core4d/scripts/eval/eval_E074.py` |
| plan | `workspace/core4d/plan/80_E074_ctrl_guard_hold_reward_plan.md` |
| log | `workspace/core4d/log/94_E074_*.md` |

### 建议先执行的最小批次

如果用户批准，第一批只做：

1. E074A: E073 + robot ctrl trust-region scale 0.5
2. E074C: E073 + hold continuity reward scale 2.0

两个方向互斥、可并行，能最快判断“控制偏离” vs “直接 contact 约束”哪个更有效。

## 当前审核问题

请用户确认优先方向：

1. 是否同意 E074 第一批做 **E074A ctrl trust-region** + **E074C hold continuity** 两个单变量实验？
2. 是否接受暂缓 E065/E066/E067 那些历史 reward/dynamics 改动，不纳入第一批？
3. 如果只能先跑一个，我建议先跑 E074A，因为它改动小、解释力强，并且直接对应 E072/E073 的 ctrl 偏离证据。

