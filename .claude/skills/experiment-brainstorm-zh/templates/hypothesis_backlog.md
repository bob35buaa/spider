# 假设库 — {exp_name}

> 所有研究假设的持续维护列表。由 `experiment-brainstorm-zh` 写入，由 `experiment-planning-zh` 消费。

## 状态说明

| 图标 | 含义 |
|------|------|
| 💡 | 想法：初步假设，待评估 |
| 📋 | 已计划：已转为实验 plan（关联 Exx） |
| 🔬 | 运行中：实验正在进行 |
| ✅ | 完成：实验结束，结论已记录 |
| ❌ | 放弃：经过分析决定不做，原因已记录 |

---

## 假设总览

| # | 假设简述 | 类别 | 可证伪标准 | 优先级 | 来源 | 状态 | 关联实验 |
|---|---------|------|-----------|------|------|------|--------|
| H001 | （示例）增大 object mass friction 后 freejoint 搬运可行 | Physics | E003 sweep 中至少一个 variant obj_mean < 0.3m | P0 | E002 结论 | 📋→E003 | E003 |

---

## 假设详情

### H001 — （示例）标题

**背景**：E002 full CEM 显示真 freejoint 下 obj_mean/max 为 0.7/1.4m，比 E081 actuator-guided 显著退化。需判断是物理参数不可行还是 reward/optimizer 不足。

**假设声明**：当 object mass 从默认值下调至 0.5-2kg 范围，同时将 friction 系数提升至 1.5+，freejoint 搬运的 obj_mean 可降至 0.3m 以下，说明原参数导致物理上难以推动。

**类别**：Physics

**可证伪标准**：
- 支持假设：任意 sweep variant 的 obj_mean < 0.3m
- 否定假设：所有 variant 的 obj_mean 仍 ≥ 0.5m

**最小验证实验**：2 cases × 500 iter × 2 GPU ≈ 10 min；4 variants 分布在两张卡

**估算算力**：远程 2×RTX6000 Ada，约 10-15 min

**预期结论**（两种）：
1. 若参数调整后 obj_mean < 0.3m → 确认 Physics 是瓶颈 → 下一步优化物理参数
2. 若参数调整后仍 ≥ 0.5m → 确认 Reward/Optimizer 是瓶颈 → 下一步做 reward 消融

**打分**：预期影响 3，可行性 3，综合 9（P0）

---

## 已放弃假设存档

| # | 假设简述 | 放弃原因 | 放弃时间 | 记录 session |
|---|---------|---------|---------|-------------|
| （暂无） | | | | |

---

*由 `experiment-brainstorm-zh` 维护。每次头脑风暴 session 后更新。*
