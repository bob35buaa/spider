# Brainstorm Session: 2026-05-17

## 研究问题

> 纯 freejoint 物体控制失败（E002, obj_err 0.7-0.8m），而执行器引导非物理真实（E081, scene_act Mode B）。在「单人机器人重定向 → RL 训练 → 与真人合作 sim2real」pipeline 下，最小且最有效的干预方案是什么？

## 上下文概括

- **已知：** E081 的部分成功完全依赖 scene_act 物体执行器引导（6 PD actuators, gain 10→0）；E002 确认 freejoint 失败但手部接近度 89%/72%；CEM 硬上限 ~64-66% 接触率；HDMI 在相同数据上 100× 更好
- **失败的方案：** connect constraint (假悬挂), mocap partner (无限刚度), 垂直重力补偿 (CEM 不主动接近), SBTO E047 (α_μ 过高), hand_approach_rew (吞噬其他目标)
- **开放问题：** ~~E003 (1kg freejoint) 远端运行中~~ **E003 已完成且 H001 已证伪**; partner 建模方案未定; SBTO 是否可行

## 使用的框架

1. **失败分解** — Physics/Reward/Optimizer/Data/Algorithm 逐层排查
2. **文献缺口** — DynaRetarget SBTO + Harmanoid interaction reward + contact-aware retargeting
3. **可行性阶梯** — 从已有基础设施出发，最小代价优先
4. **双假设检验** — H001 vs H003 区分物理参数 vs partner 建模

---

## 全部候选假设

### H001：物理参数是 freejoint 搬运的主要瓶颈

- **类别：** Physics
- **具体声明：** E002 失败主因是 5kg 过重 + 球形手摩擦不足。1kg + friction 2.0-4.0 后 CEM 能实现 freejoint 搬运
- **可证伪标准：** E003 guard (box023, 1kg) obj_mean ≥ 0.40m → 假设失败; obj_mean < 0.25m → 假设支持
- **最小验证实验：** 已在跑 (E003, 4 variants on remote GPU)
- **来源：** E002 — 手部 89% 接近但无法搬运

### H002：软虚拟抓取约束是 freejoint 与 GT 引导的最优折中

- **类别：** Algorithm
- **具体声明：** 手距 <5cm 时激活软弹簧约束，CEM 负责接近、弹簧负责维持接触。比 connect (假悬挂) 更物理真实，比纯 freejoint (失败) 更可行
- **可证伪标准：** box023 guard, obj_mean < 0.20m 且 pelvis_min > 0.50m → 支持; obj_mean > 0.40m → 失败
- **最小验证实验：** 1 case, 200 iter, ~15min × 1GPU。利用 scene_weld.xml + mjwp_eq.py 基础设施
- **来源：** E081 Mode B 成功 vs E002 Mode A 失败的对比

### H003：Partner Proxy 3D 动态力使单人协作搬运可行 ★ 选定

- **类别：** Physics + Algorithm
- **具体声明：** 之前 partner force (E009/E024/E025) 仅垂直重力补偿。引入 3D 弹簧力（将物体拉向 ref 轨迹），模拟 partner 对物体的搬运贡献，机器人通过真实接触力贡献剩余部分
- **可证伪标准：** box025 main, obj_mean < 0.30m 且 pelvis_min > 0.50m → 支持; obj_mean > 0.50m → 失败
- **最小验证实验：** 1 case (box025_p2), 200 iter, ~15min × 1GPU。基于 Mode D 代码 + 3D 弹簧方向
- **来源：** (1) E009: 单人 box025 几何不可行; (2) Harmanoid: partner 力不可忽略; (3) sim2real 场景: 另一端是真人
- **选定理由：** 与 sim2real 终端目标完全对齐; 复用已有 Mode D 代码; 信息价值最高（失败则排除所有 force shaping 方案）

### H004：DynaRetarget SBTO 解决 CEM 短视问题

- **类别：** Optimizer
- **具体声明：** E047 SBTO 失败因 α_μ=0.95 + 固定迭代。正确 SBTO（渐进 horizon + sigma_min 收敛）可提升搬运成功率到 >30%
- **可证伪标准：** box023 guard (freejoint, 5kg), SBTO obj_mean < 0.40m → 支持; > 0.60m → 失败
- **最小验证实验：** 1 case, sigma_min sweep, ~30min × 1GPU。需改 sampling.py
- **来源：** DynaRetarget 74.6% vs 37.9%

### H005：双机器人协作重定向 + 策略蒸馏

- **类别：** Algorithm
- **具体声明：** 两个 G1 同时重定向，用 Harmanoid 交互奖励。固定 partner 策略后单独训练 ego RL 策略
- **可证伪标准：** 双机器人 obj_mean < 0.25m 且两个 pelvis_min > 0.50m → 重定向可行
- **最小验证实验：** 1 case, dual-robot CEM, 200 iter, ~30min × 1GPU
- **来源：** E009/E054 大物体单人不可行; Harmanoid shared policy

---

## 优先级排序

| # | 假设 | 类别 | 影响(1-3) | 可行性(1-3) | 综合 | 实验代价 |
|---|------|------|----------|-----------|------|---------|
| H001 | 1kg freejoint | Physics | 2 | 3 | 6 | 已在跑 |
| **H003** | **Partner Proxy 3D** | **Phys+Algo** | **3** | **3** | **9** | **1case, 200iter** |
| H002 | 软虚拟抓取 | Algorithm | 2 | 2 | 4 | 1case, 200iter |
| H004 | SBTO | Optimizer | 3 | 1 | 3 | 需改核心代码 |
| H005 | 双机器人+蒸馏 | Algorithm | 3 | 1 | 3 | 大量新代码 |

## 决策树

```
E003 结果（门控）→ ❌ 1kg 也失败（guard obj_mean > 0.80m）
└─ H003 (已选，E004 plan 已存在)
     ├─ H003 成功 → 进入 RL pipeline
     └─ H003 不够 → 叠加 H002 (virtual grasp)
                     └─ 再不够 → H004 (SBTO) 或 H005 (双机器人)
```

## 否决记录

| 假设 | 否决/推迟原因 |
|------|-------------|
| H001 | ❌ 已证伪 (E003)。guard obj_mean > 0.80m，被动物理参数不是主要瓶颈 |
| H002 | 推迟。virtual grasp 是 H003 的降级方案，先验证 proxy 力是否足够 |
| H004 | 推迟。需修改核心 sampling.py，实现复杂且 E047 有失败先例 |
| H005 | 推迟。长期最优但短期代码量大，且 E016-E031 双机器人经验全部失败 |

## 选定方向

**H003: Partner Proxy 3D 动态力** → 下一步进入 `experiment-planning-zh` 设计 E004 实验
