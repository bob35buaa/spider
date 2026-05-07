# E027d2: 视频深度分析 + 迭代计划

## 视频逐帧分析

### box025 (★★★ 最佳)
- **0% (0s)**: ref/sim 一致，机器人站在箱子后方，双手在箱子上方
- **20% (1s)**: 机器人弯腰俯身向箱子，ref/sim 上半身姿态匹配良好。sim 的手接近箱面但有轻微偏差
- **40% (2s)**: 机器人推着箱子向前走，ref 双手贴箱面，sim 双手也在箱面附近，**整体最像搬运的时刻**
- **60% (3s)**: 继续推动，ref 手在箱顶，sim 手在箱侧偏后。sim 下肢膝盖弯曲角度偏大（蹲着走）
- **80% (4s)**: ref 弯腰推箱，sim 身体前倾但**脚步不稳、膝盖过度弯曲**
- **100% (5s)**: 和起始姿态类似，但 sim 右脚有拖地/不稳

**核心观察**:
1. 手-物接触: contact(<10cm)=65%, 尤其 20-60% 区间持续接触
2. 下肢抖动: sim 膝盖弯曲偏大、脚步移动不够流畅（CEM 在行走优化上有局限）
3. **关键问题: 物体由 PD actuator 驱动（invisible force），不是机器人物理推动的**。即使手接触了物体，接触力不是物体运动的原因

### desk005 (★★ 中间有完全摔倒)
- **0%**: ref/sim 匹配，机器人在桌旁
- **20% (0.9s)**: ref 迈步走路 OK，sim 也在走但姿态轻微偏差
- **40% (1.9s)**: **sim 完全摔倒** — 面朝下趴在地上，ref 还在正常走路。这是 87% stability 的来源
- **60% (2.8s)**: **sim 恢复站立**，桌子继续移动（PD actuator 不受机器人摔倒影响）
- **80% (3.7s)**: sim 弯腰向前，不太稳定但还在走。与 ref 有明显姿态偏差
- **100% (4.6s)**: sim/ref 都在桌旁，sim 姿态恢复但脚步不自然

**核心观察**:
1. 40% 处完全摔倒是致命缺陷 — 即使恢复了，轨迹质量严重受损
2. 手从未接触桌子 (contact<10cm = 9%)
3. 桌子移动完全靠 PD actuator，与机器人动作无因果关系

### bucket010 (★★ 部分合理)
- **0%**: 机器人在桶旁，面对面。桶在左前方
- **20%**: ref 身体向桶倾斜、手伸向桶；sim 也向桶方向但**手没碰到桶** (距离更远)
- **40%**: ref 双手在桶上/旁搬运姿态；sim 手接近但明显不如 ref 紧密
- **60%**: ref/sim 都在桶旁行走，sim 比 ref 站得更远
- **80%**: ref 弯腰手在桶顶；sim 也弯腰但手距桶有间距
- **100%**: ref/sim 都走到了终点，姿态相似

**核心观察**:
1. contact(<10cm)=36% — 间歇性接触，远不如 box025
2. 机器人整体姿态跟踪还行 (100% stable)
3. 但手-物距离 mean=0.178m — 多数时间手在物体表面 10-20cm 外

### chair022 (★ 失败)
- 机器人不稳定 (81% stable)，椅子旋转严重 (quat_err=1.58)
- 不再详细分析

---

## 深度诊断: 当前方案的本质与限制

### 当前方案本质
```
contact_guidance (guidance_decay_ratio=1.0) + CEM body tracking
= Object PD actuator (invisible force) + Robot body motion optimization
= 两个独立系统: 物体自动移动 + 机器人尽量跟踪参考姿态
```

**物体移动的驱动力不是机器人**，是 PD actuator 的 invisible force。机器人的手有时碰到物体（box025 65%），但这是 body tracking 的副产品，不是因果关系。如果去掉 PD actuator，物体不会移动。

### 与真正 loco-manipulation 的差距
1. **因果性**: 真正搬运 = 机器人施力 → 物体移动。当前 = 物体自动移动 + 机器人跟随
2. **接触质量**: 65% 的"接近"并非持续的承重接触，只是空间上靠近
3. **下肢稳定性**: CEM 对行走优化有限，腿部抖动/过度弯曲
4. **物理一致性**: 物体被 invisible force 驱动，不符合真实物理

### 可改进空间（不改变方案本质的前提下）
1. **增加 hand_approach_rew**: 强制手靠近物体表面 → 提高接触比例
2. **增加 task_body_rew**: 手/脚等关键 body 的 task-space tracking → 改善姿态质量
3. **提高 base_pos_rew**: 更强的身体稳定性 → 减少摔倒
4. **降低 object PD gains**: 减弱 invisible force → 物体移动更多依赖碰撞（但会降低 obj tracking）
5. **HDMI physics_dt=0.002**: 更小步长 → PD 更稳定 + 碰撞更准确

---

## 8 小时迭代计划

### 估算运行时间
- E027d2 (32iter × 1024samp): ~5-6min per case
- 4x 快速测试 (8iter × 256samp): ~1.5min per case = 6min total
- 3x 完整测试 (32iter × 1024samp, 跳过 chair022): ~18min total

### Round 1: E032a — Hand Approach + Task Body Reward (1.5h)
**目标**: 提高手-物接触比例 + 改善关键 body 姿态

**改动**:
- `hand_approach_rew_scale: 3.0` (从 E025 验证有效)
- `task_body_rew_scale: 5.0` + task body names (hands, feet)
- `base_pos_rew_scale: 8.0` (从 5.0 提高)

**预期**: box025 contact(<10cm) 从 65% → 80%+, desk005 不再摔倒

**验证**: 3 case × 完整设置 = ~18min
**分析 + 调参**: ~30min
**Risk**: hand_approach 可能和 body tracking 冲突（手被拉向物体 vs 跟踪参考手位）

### Round 2: E032b — Object PD Gain Sweep (1.5h)
**目标**: 找到 "invisible force 最弱但物体仍能跟踪" 的 sweet spot

**改动**: kp_pos ∈ {100, 200, 500}, kp_rot ∈ {10, 50}

**逻辑**: 当 PD gains 降低时，物体更多依赖碰撞力移动。如果 hand_approach 同时激活，机器人手会靠近物体 → 碰撞力有可能部分替代 PD force → 更物理真实的搬运

**验证**: 3 gain configs × box025 (最佳case) = ~18min
**分析**: ~20min

### Round 3: E032c — HDMI Physics + 最佳配置 (1.5h)
**目标**: 验证 physics_dt=0.002 是否改善碰撞精度和步态稳定性

**改动**: 
- `physics_dt=0.002, sim_dt=0.02` (HDMI style)
- 用 Round 1/2 的最佳配置

**注意**: 运行时间会增加 ~4-8x (decimation)。可能需要减少 iterations
**验证**: box025 + desk005 = ~40min (慢速)
**分析**: ~20min

### Round 4: E032d — 3 Case 最终评估 (1.5h)  
**目标**: 用最佳配置跑 box025/desk005/bucket010 完整评估

**产出**: 3 case 视频 + 指标表 + 与 E027d2 baseline 对比

### Round 5: E032e — 降低 PD + 增加碰撞引导 (1h, 可选)
**如果 Round 2 显示低 PD + hand_approach 有效**:
- 进一步降低 PD gains → 物体几乎完全由碰撞驱动
- 增加 contact_rew_scale → 奖励真正的接触力

### 备选: E033 — Connect 约束 + Locomotion (1h, 如果 E032 全部失败)
- 从 E017/E018 的 connect 约束方案出发
- 结合 E027d2 的 body-frame fix + 非 anchored 轨迹
- 用 soft connect 将手焊接到物体 → CEM 只优化行走

---

## 成功标准

| 指标 | E027d2 baseline | 目标 |
|------|----------------|------|
| box025 stable | 100% | 100% |
| box025 contact(<10cm) | 65% | **80%+** |
| desk005 stable | 87% | **95%+** |
| desk005 contact(<10cm) | 9% | **40%+** |
| bucket010 contact(<10cm) | 36% | **60%+** |
| pelvis 抖动 | 明显 | 显著减少 |
