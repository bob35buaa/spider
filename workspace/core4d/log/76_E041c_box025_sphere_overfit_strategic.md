# Strategic Finding: E041c reward stack 完全是 box025 sphere 的过拟合 — A/B 决策框架失效, 转向 X1+X2 reward 泛化方向

## 状态: 🚨 **STRATEGIC INSIGHT — 用户基于 4-cell 矩阵分析发现 E041c 没有任何泛化性, A/B 都是错误方向, 必须改为修 reward 让其 case-agnostic / geometry-agnostic**

**TL;DR**: 用户提示查看 `workspace/core4d/results/E041/E041c_box023_collision_fixed.npz` (= E048_box023, sphere + E041c + box023), 量化 pelvis_min 0.193m / Stable 50% 仍然摔. 加上 E061 (sphere + E041c + box025 ✅ 0.672m), E060.0 (3-box + E041c + box023 ❌ 0.176m), 远程 (3-box + E041c + box025 ❌ 0.253m), 4 个 cell 中**只有 box025+sphere 这一个组合 work**. 用户 insight: "box025 work 是 task-specific 设计或偶然, 完全一样的方法换 box023 就不行了". 重新审视 reward 参数: `palm_normal=[0,∓1,0]` 和 `eef_offset=[0.05,0,0]` 都是 box025+sphere 的 hand-tuned 常数 — **任一变量改变 → reward prior 失效 → 摔**. 因此 A (revert sphere) / B (修 eef_offset) 都只是"让 box025 重新 work", 不解决泛化问题, 推下一个 case 同样会摔. 转向 **X1 (per-case auto palm_normal) + X2 (per-hand auto eef_offset)** 让 reward 数据驱动而非常数 hardcoded.

## 1. 触发: 不需要跑 E062, 数据已经存在

之前 (E061 完成后) 我以为需要跑 sphere + E041c + box023 (E062) 来确认 box023 sphere 下是否能 work. 用户指出:

> `workspace/core4d/results/E041/E041c_box023_collision_fixed.mp4` 这个就是 box023+sphere+E041c 的结果吧, 这个修复了 collision margin 的

按 log 61 §"E048 与 E041c 的关系": `E041c_*_collision_fixed` 是 **E048 对应文件的 symlink/副本**. E048 时代是 sphere hand (早于 fa2e181), 用 `+override=core4d_e041c` 完全相同配置, 唯一区别是 E048 用 `fix_collision_boxes.py` 修过 box size (碰撞盒大小, 不是 margin).

本次重新量化 `E041c_box023_collision_fixed.npz`:

| 指标 | 数值 | 解读 |
|------|------|------|
| frames | 136 | 完整轨迹 |
| pelvis_min | **0.193m** | ❌ 远低于 0.5m 站立阈值 |
| pelvis_mean | 0.515m | 平均勉强站立 |
| stable% (≥0.5m) | **50.0%** | ❌ 半程倒地 |
| pz 头 5 帧 | 0.79m | 站立起步 |
| pz 尾 5 帧 | 0.43m | 蹲/摔结束 |

视觉 (log 61 §box023): "t=1s sim 机器人摔倒 — 臀部着地, 左腿抬起, 手搭在箱子上; t=3s 完全仰面倒地, 腿朝天".

## 2. 4-cell 矩阵: E041c reward 在哪些组合下 work

| Cell | Hand | Case | `palm_normal=[0,∓1,0]` | `eef_offset=[0.05,0,0]` | pelvis_min | 状态 |
|------|------|------|------------------------|-------------------------|-----------|------|
| (1) | sphere | box025 | ✅ 对 (audit §2.2 mean dot 0.78-0.80) | ✅ 对 (sphere center @ wrist+10cm, eef_offset 落在 sphere 内) | **0.575m / 0.672m** | ✅ **唯一 work** |
| (2) | 3-box | box025 | ✅ 对 | ❌ 错位 12.5cm (box3 末端 @ wrist+17.5cm, eef_offset 看 wrist+5cm) | 0.253m | ❌ |
| (3) | sphere | box023 | ❌ 错 (audit §2.2 L mean dot 0.04, 噪声; R mean dot +0.51 vs +x 真最优 +0.61) | ✅ 对 | **0.193m** | ❌ |
| (4) | 3-box | box023 | ❌ 错 | ❌ 错位 | 0.176m | ❌ |

**4 cells, 1 PASS**. PASS 的唯一条件: 两个 reward 常数都恰好匹配该 cell.

## 3. 用户的 strategic insight (引用)

> 这样看方向很清晰了呀,现在就不是 3-box 和 sphere 的问题吧,很显然是 box025 work 是一些 task-specific 的设计或者偶然, 完全一样的方法换成 box023 就不行了

正面验证: `palm_normal_left=[0.0,-1.0,0.0]` 和 `palm_normal_right=[0.0,+1.0,0.0]` 是 hardcoded 常数. audit log 70 §2.2 的点积分析早已发现这是 **box025 motion 的 fingerprint** — wrist 局部坐标系下哪个轴指物体, 因 case 不同而不同. 这两个值在 box025 上恰好正确 (CEM 探索方向被正确 prior 引导), 在 box023 上变成噪声 (CEM 失去方向).

类似地, `eef_offset=[0.05,0,0]` 是 **sphere geometry 的 fingerprint** — sphere center 在 wrist+10cm, eef_offset 5cm 落在 sphere 内部. 3-box 时代 box3 末端伸到 wrist+17.5cm, 这个 5cm 常数失效.

**E041c reward stack = box025 sphere 上手工调出来的两个常数, 不是 reward 设计**. 所以它在 box025 sphere 上 100% work, 任何变量改变都 break.

## 4. A/B 决策框架失效

之前 (E061 PASS 后) 准备的 A/B 决策:

| 选项 | 操作 | 期望效果 |
|------|------|---------|
| A (revert sphere) | `git revert fa2e181` | box025 重新 work |
| B (修 eef_offset) | 改 `[0.05,0,0]` → `[0.10,0,0]` | box025 在 3-box 下重新 work |

**两个都只解决 cell (1) 或 (2), 不动 cell (3)/(4)**. 推到 E054 标的的另外 4 个 B+C case (bucket001/005_s2/007, desk021), 同样会摔, 因为它们的 palm_normal 和 box025 的也不一样.

A/B 选择本质是**"让 box025 work 的方式"之争**, 不是**"让多 case 都 work"之争**. 这是错误的问题框架.

## 5. 修正方向: X1 + X2 让 reward 泛化

把 E041c 的两个 hand-tuned 常数都改为**数据驱动**:

### X2: per-hand auto-derived `eef_offset` (geometry-agnostic)

**思路**: 从 `robot.xml` 的 hand collision geometry 自动算出 eef_offset, 让它代表 hand 实际碰撞质心 (sphere → 球心; 3-box → 三个 box 的 mass-weighted centroid).

**实现**:
- 在 `mjwp.py` reward 块 init 时, 读取所有 `lh*` / `rh*` 的 `geom_pos` 和 `geom_size`
- 计算 mass-weighted (或 simply geometric) centroid → 设为 `eef_offset`
- 替代 hardcoded 配置值, 或作为 default

**自动算的预期值**:
- sphere: centroid = (0.10, 0, 0) — 跟 sphere center 一致, sphere 时代行为不变
- 3-box: 三个 box 的 mass-weighted centroid ≈ (0.085, 0, 0) 或 mean-centroid ≈ (0.08, 0, 0) — 比 sphere 时代略前, 看 box2 中心附近, 接触面附近

**验证**: 跑 box025 + 3-box + E041c (auto eef_offset) → 期望 pelvis_min 回到 ~0.6m

**工作量**: 30min 改代码 + 30min 验证 = 1h

### X1: per-case auto-derived `palm_normal` (case-agnostic)

**思路**: 用每个 case 的 ref motion 自动算出哪个 wrist 局部轴最常指向 object center, 把它当作该 case 的 palm_normal. 替代 hardcoded `[0,∓1,0]`.

**实现**:
- 写 `compute_palm_normal_from_ref.py`: 给 ref motion + intent 窗口 + wrist xpos/xquat + object center
  - 对 6 个候选 (`±x`, `±y`, `±z`) 各算 wrist→object 单位向量在该轴上的投影 mean
  - 选 max → 这就是该 case 该手的 palm_normal
- 自动注入 case yaml: `examples/config/override/core4d_e041c_<case>.yaml` 每 case 一个
- 或者改 reward 块: init 时直接从 ref 读, 不需要 yaml

**这跟 E060.2 (case-correct +x) 区别**:
- E060.2 当时只在 box023 上用 `[+x, 0, 0]`, 而且是 hardcoded 不是数据驱动
- E060.2 catastrophic 失败 (stable 5%) 在 3-box 物理 bug 叠加下, 不是 X1 思路本身的失败
- X1 是先修 X2 (3-box 物理对齐), 再做 per-case palm_normal — 不同的对照基础

**audit log §2.2 已经写过这个分析方法**, 现在只是把它从 "审查工具" 升级为 "训练时数据驱动 prior"

**工作量**: 1-2h 实现 + 验证

### 组合验证流程

1. X2 实现 → 跑 **box025 + 3-box + E041c (auto eef_offset)** → 期望 ~0.6m (类似 sphere baseline)
2. 同时跑 **box025 + sphere + E041c (auto eef_offset)** → 期望 ~0.6m (sphere 行为不变, 因为 auto centroid = (0.10,0,0))
3. 通过 → 添加 X1 → 跑 **box023 + 3-box + E041c (auto eef_offset + auto palm_normal)** → 期望站住
4. 通过 → 推广到其他 4 个 B+C case (bucket001/005_s2/007, desk021)
5. 5/6 case 通过 → reward 泛化性证明, 进 RL training (Path C)

## 6. 这个发现改变了 EXPERIMENT_TRACKER 多个早期结论的解读

| 早期结论 | 当时解读 | 修正后解读 |
|---------|---------|-----------|
| E041c "成功" | reward 设计正确 | **box025 sphere 上恰好两个常数都对, 不是 reward 设计** |
| E048 baseline 复用 E041c | 推广到其他 case 验证 | 只在 box025 上工作, box023/desk005 全摔 (log 61 视觉复查已记录但未拔高到这个高度) |
| E054 standardized 6 case (B+C) | 等待 reward 改进 | reward 改进 ≠ 调常数, 必须做泛化, 否则 6 case 全摔 |
| E060.0/.1/.2 reward ablation | reward task-specific divergence | 实际是 reward 在不同 case 上的偶然不同失败模式, 没有信息量 |

## 7. 下一步 (X2+X1 实施 plan, 单独写, 见 plan/72_*)

按用户决定 X2+X1 方向, 下一步进入 plan mode 写 E062/E063 实施 plan:

- **E062 = X2** (auto eef_offset, ~1h)
- **E063 = X1** (auto palm_normal, ~2h)
- 验证 case: box025 + box023 + 至少 1 个 bucket case
- 通过 → 推 4 cases 余下

如果 X2 单独通过 box025 但 X1+X2 不通过 box023, 再考虑 X3 (重写 contact reward 不用 palm_normal 假设).

## 8. 改动文件

| 文件 | 改动 |
|------|------|
| `workspace/core4d/log/76_E041c_box025_sphere_overfit_strategic.md` | 本文件 |
| `workspace/core4d/EXPERIMENT_TRACKER.md` | 加 Phase 18 strategic-finding 行 + log 76 索引 |

## 9. 关联

- log 70 §2.2 (audit): palm_normal 数值分析 — 早已诊断 box023 L 是噪声, 当时未拔高到"reward 完全无泛化"
- log 74: box025 3-box regression 发现 — 揭示 eef_offset 跟 sphere 绑定
- log 75: E061 sphere verify PASS — 决定性证明 3-box 是 box025 regression 源
- 本 log: 综合 E041c_box023_collision_fixed (已存在数据) + E061 + E060.0, **决定 reward 没泛化 = 真正问题**

## 10. 教训沉淀 #5 (累计)

之前 4 个 over-optimism 教训 (log 74 §5.3) 都是"看视频前先看数据 / ablation 多 case / 物理 port 必须回归测试 / 跨 sub-experiment 假设链每一步重验". 本次第 5 个:

> **判断"reward 设计成功"必须基于多 case 多 hand 的横向验证, 不能基于单 case 调通**

具体规则: 任何声称"reward 设计 X 工作"的 commit, 必须包含 ≥2 case × ≥2 hand 的 baseline 数据. 仅在 box025+sphere 上 work 的 reward 不能称为"工作的 reward", 只能称为"box025+sphere 上调通的常数". 这条规则在以后 reward 实验中强制实施.
