# HYPOTHESIS_BACKLOG — core4d_collab_retarget

> 持续维护的假设库。所有假设（含状态），按优先级排序。

---

## Active

### H003: Partner Proxy 3D 动态力使单人协作搬运可行

- **状态：** 🟢 SELECTED — 待设计 E004 实验
- **类别：** Physics + Algorithm
- **声明：** 引入 3D 弹簧力将物体拉向 ref 轨迹，模拟 partner 对物体的搬运贡献。机器人通过真实接触力贡献剩余部分。与 sim2real 终端目标对齐（另一端是真人）
- **可证伪标准：** box025_p2 main, obj_mean < 0.30m 且 pelvis_min > 0.50m → 支持; obj_mean > 0.50m → 失败
- **最小验证实验：** 1 case (box025_p2), 200 iter, ~15min × 1GPU
- **基础设施：** Mode D (`partner_force_spring_kp`) 已有 3D 弹簧力 + 重力补偿代码
- **关键设计决策：**
  - 力方向：沿 (ref_pos - sim_pos) 方向的恢复力，还是沿 ref 速度方向的前馈力？
  - 力大小：需覆盖真人合理范围，支持 domain randomization
  - 是否同时激活 robot 端的 contact reward？
- **来源：** E009 (单人几何不可行), Harmanoid (partner 力不可忽略), 用户 sim2real 需求
- **创建日期：** 2026-05-17

---

### H001: 物理参数是 freejoint 搬运的主要瓶颈 — ❌ 已证伪

- **状态：** ❌ FALSIFIED by E003
- **类别：** Physics
- **声明：** 5kg 过重 + 低摩擦是 E002 失败主因。1kg + friction 2.0-4.0 可解
- **可证伪标准：** E003 guard (box023, 1kg) obj_mean ≥ 0.40m → 假设失败
- **E003 结果：**
  - box025_m1_f4（最优 main）: obj_mean 0.400m, 仍 floor-supported
  - box023_m1: obj_mean 0.831m, 腿/箱干涉 11.3%（比 E002 更差）
  - box023_m1_f4: obj_mean 0.947m, 腿/箱干涉 10.0%
  - **guard 全部 obj_mean > 0.80m → 假设决定性失败**
- **结论：** 被动物理参数调整不是主要瓶颈。需要 explicit virtual support / contact constraint
- **来源：** E002 — 手部 89% 接近但无法搬运
- **创建日期：** 2026-05-17 | **证伪日期：** 2026-05-17

---

## Backlog (推迟)

### H002: 软虚拟抓取约束

- **状态：** ⏸️ DEFERRED — H003 的降级方案
- **类别：** Algorithm
- **声明：** 手距 <5cm 时激活软弹簧约束维持接触
- **可证伪标准：** box023, obj_mean < 0.20m 且 pelvis_min > 0.50m
- **推迟原因：** 先验证 H003 proxy 力是否足够；H002 可作为叠加方案
- **创建日期：** 2026-05-17

### H004: DynaRetarget SBTO 渐进 horizon

- **状态：** ⏸️ DEFERRED — 实现复杂
- **类别：** Optimizer
- **声明：** 正确实现 SBTO 可提升搬运成功率 >30%
- **可证伪标准：** box023 (freejoint, 5kg), obj_mean < 0.40m
- **推迟原因：** 需修改核心 sampling.py; E047 有失败先例; 优先验证力模型路径
- **创建日期：** 2026-05-17

### H005: 双机器人协作重定向 + 策略蒸馏

- **状态：** ⏸️ DEFERRED — 长期方案
- **类别：** Algorithm
- **声明：** 两个 G1 用 Harmanoid 交互奖励同时重定向，固定 partner 后训 ego RL
- **可证伪标准：** 双机器人 obj_mean < 0.25m 且两个 pelvis_min > 0.50m
- **推迟原因：** E016-E031 双机器人经验全失败; 代码量大; 先验证更轻量方案
- **创建日期：** 2026-05-17

---

## Changelog

| 日期 | 变更 |
|------|------|
| 2026-05-17 | 初始化: 5 个假设, H003 选定, H001 在测 |
| 2026-05-17 | H001 证伪: E003 full CEM 完成, guard obj_mean > 0.80m. E004 plan 已存在且对齐 H003 |
