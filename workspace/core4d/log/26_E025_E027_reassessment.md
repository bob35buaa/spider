# E025-E027 修正评估 (基于用户反馈 + 严格验证)

## 用户反馈要点

1. **bucket010**: 参考中物体在 0-1s 被 partner 侧向移动 (xy=0.32m, lift=13cm), 我们的 sim 中物体静止不动 → 机器人碰到的是一个不该在那里的静止物体
2. **desk005**: 同样未成功搬运, 只是推/碰
3. **Claims 口径过松**: "hand contact" ≠ "cooperative carrying retargeting"

## 修正后的 Claims 验证

### E025-f (bucket010) — 修正: FAIL

| 原 Claim | 原判定 | 修正判定 | 原因 |
|----------|--------|---------|------|
| C1: hand <3cm 持续≥30帧 | PASS | **N/A (指标无意义)** | 是推倒后的碰触, 非搬运中的持握 |
| C2: 手伸向物体 | PASS | **PARTIAL** | 手确实接近, 但动作是推/碰不是抓 |
| C3: lift 因果 | PASS | **FAIL** | lift 来自推倒/倾斜, 非沿 ref 轨迹搬运 |
| C4: 稳定 | PASS | PASS | — |
| C5: 优于 E024 | PASS | PASS | 手确实更近了 |

**失败根因**:
- 参考中物体在 0.77s 开始被 partner 抬起 (到 t=1.43s 高 19cm), 同时横向移动 0.32m
- 我们的 partner_force 只给向上恒力, 不复现 partner 的横向搬运动作
- 物体停在原地 → 机器人按 body tracking 做姿态 → 碰巧碰到静止物体 → "contact" 产生
- 视频证实: robot 推倒 bucket, 后续"接触"是 bucket 倒在机器人腿旁

### E026-b (bucket010) — 修正: FAIL

同上。虽然 bucket 没倒 (比 E025-f 好), 但本质仍是:
- 物体应该在 1.4s 时离地 19cm+横移 36cm, 实际原地不动
- 机器人在做 body tracking 时碰到了不该在那里的静止物体
- 非"沿参考轨迹的协作搬运重定向"

### E027-c (desk005) — 修正: FAIL

- desk 被推起 11.7cm, 但参考中 desk 也是被 partner 侧协作移动的
- 机器人推动 desk ≠ 协作搬运重定向
- 视频中看到的是"弯腰时身体碰到 desk", 非"有意识搬运"

## 技术贡献 (仍然有效)

尽管 E025-E027 **不构成成功的接触重定向**, 以下技术贡献仍然有效:

1. **hand_approach_rew 确实让 CEM 产生了手-物体接近** — 相比 E024 (手从不碰物体) 有改善
2. **data_path override 机制** — 允许使用 anchored 轨迹
3. **表面距离近似** — 使用 geom half-extents 有效
4. **4096 samples + 24 iter** — 产出更稳定结果

## Phase 6 真实结论

**hand approach reward 的局限**:
- 能让手接近物体 (距离从 0.30m → 0.00m), 但不能产生"按参考轨迹搬运物体"的行为
- CEM 仍然以 body tracking 为主, approach reward 只是让"碰巧碰到"的概率增加
- 根本问题未解: 参考中物体的运动是 partner 驱动的, 我们没有建模 partner 的搬运贡献

## 每个 Case 的特殊性分析

| Case | 物体 | 参考中 partner 作用 | G1 运动学限制 | 当前状态 |
|------|------|-------------------|-------------|---------|
| box025 | 大箱 0.61×0.61×0.89m | 对面对夹, 双人分担重量+夹持 | 臂展 < 物体宽度 | 不可解 (几何限制) |
| bucket010 | 中桶 0.40×0.74×0.40m | 对面持握, partner 驱动横向移动 | 手能到但 partner 动态缺失 | approach 让手碰到, 但物体不沿 ref 移动 |
| desk005 | 桌 0.40×0.74×0.80m | 协作抬起+移动 | 类似 bucket010 | 同上 |
| chair022 | 椅子 0.57×0.86×0.53m | 协作搬运 | 碰撞推飞 (E020) | 未在 Phase 6 验证 |

## 核心问题重新定义

**不是 "CEM 能否产生接触"** (E025 证明能), 而是:
> "如何让物体沿参考轨迹运动, 同时机器人产生合理的接触力?"

这需要:
1. 物体运动: 要么由 spring/force 驱动 (模拟 partner), 要么由 CEM 产生
2. 机器人接触: hand_approach 能做到
3. 两者协调: 机器人的接触力方向与物体运动方向一致

当前缺失: **Partner 的横向搬运动作** — partner_force 只给竖直力, 不给横向力。
