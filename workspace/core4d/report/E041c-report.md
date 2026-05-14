# E041c 是怎么炼成的 — Phase 9-12 (E032 → E047) 完整复盘

> 范围：E032 - E047 (16 主实验, 30+ 子变体)
> 目标：把 CORE4D 上 hand-object 接触从 36-65% 提升到稳定 60%+, 同时不摔倒、不破坏 body tracking
> 终点：**E041c (additive ori, w=0.3) — box025 contact=66%, bucket010=57%, MPKPE=1.4cm, stability=100%**
> 注：本报告与 `STAGE_REPORT_E001-E067.md` §2.3 互补 — STAGE_REPORT 是一行结论, 本文是过程账

---

## 0. TL;DR (E041c 是 CEM 数值最优, 但不是视觉最优)

经过 16 实验 4 个突破口, 最终落在 E041c 的原因不是它"完美", 而是它在四个互相打架的目标 (body tracking / stability / contact / 不出畸形姿态) 间是当前 CEM 框架下能达到的**帕累托最优**:

| 维度 | E041c 数字 | 是否合格 | 备注 |
|---|---|---|---|
| Body tracking (MPKPE) | **1.4cm** | ✅ 远超合格线 15cm | 来自 E036 突破 (关 hand_approach + 开 object tracking) |
| Stability (>0.6m) | **100%** (box/bucket) | ✅ | 来自 E035 local-frame body tracking 移植 |
| Object tracking | **0.8cm** | ✅ 超过 HDMI 5.4cm | 来自 E036 contact_guidance + object PD |
| Contact <10cm | **66% / 57%** | ⚠️ 未到 80% | E041c 是 16 实验中最高, ref 数据上限 ~69-76% |
| 视觉自然度 | ❌ box025 趴箱 / bucket010 推非搬 / desk005 丢桌走 | ❌ | 数值漂亮但 CEM 找的是局部最优 |

**真正的瓶颈不是 reward 设计**: E037 系列证明 `contact_mask_rew` 在 desk005/bucket010 上完全无效, 因为 ref 中手离物体本身就 >10cm; 提升 gain 又会破坏 body tracking. 数字到 66% 已经触顶.

---

## 1. 起点 — 为什么 E032 不是终点

E031 之前的双机器人 connect 路线被推翻 (见 STAGE_REPORT §2.1). 单机器人 + hand_approach reward (当时的标准做法) 在 box025/bucket010/desk005 上的 baseline:

| Case | E027d2 baseline | 视觉 |
|---|---|---|
| box025 | contact 65% / stable 100% | 站立推箱, 接触自然 |
| desk005 | contact 9% / stable 87% | t=0.4s 完全摔倒 ❌ |
| bucket010 | contact 36% / stable 100% | 手没碰桶 |

→ **三个核心问题同时存在**: desk005 摔倒 / bucket010 接触不够 / box025 也不算稳. 选 box025 + desk005 + bucket010 三件套作为 sweep 的固定 case.

---

## 2. Phase 9 (E032 - E033) — Reward weight sweep, 发现 contact-stability 是结构性 tradeoff

### E032a: hand_approach + base_pos sweep

**改动**: `hand_approach_rew_scale` 从 0 → 3.0, `base_pos_rew_scale` 从 5 → 10/15/25.

**32×1024 full-test 矩阵**:
| 配置 | box stable | box contact | desk stable | desk contact | bucket stable | bucket contact |
|---|---|---|---|---|---|---|
| E027d2 (HA=0) | **100%** | 65% | 87% | 9% | **100%** | 36% |
| **E032a (HA=3, base=5)** | 71% | **81%** | 79% | **83%** | **100%** | **63%** |
| E032a (HA=3, base=10) | **100%** | 57% | 75% | **82%** | — | — |

**用户视频判断 (诚实评估)**:
- box025: **变差** — sim 趴/蹲在箱后, hand_approach 把上半身拉向箱面, "接触 81% 是数值假象"
- desk005: **变好** — 仍前倾但没完全摔倒, 后半段 sim 走在桌旁手在桌面 (HA=3 σ=5)
- bucket010: 接触时间增加但身体偏离桶更远

**第一次教训** (lesson #1): hand_approach 靠"近物体"就给奖励, CEM 的最优解是"全身贴上去", 数值好看视觉糟糕.

### E033: σ sweep + CEM budget sweep

**desk005 stability 专项**: hand_approach σ 从 5 → 1.0/2.0/3.0, 同时试 sample 2048/iter 48/horizon 1.6:

| 配置 | stable_carry | <15cm | <10cm |
|---|---|---|---|
| **σ=1.0** ★ | **93.9%** | **91.4%** | 13.6% |
| σ=2.0 | 82.3% | 69.7% | 48% |
| σ=3.0 | 83.8% | 87.4% | 79.3% |
| σ=5.0 (E032a) | 75.8% | 83.8% | 81.8% |
| 2048 sample × σ=2 | 83.3% | 83.8% | 57.1% |
| 48 iter × σ=3 | **31.8%** | 70.2% | 66.2% |
| horizon=1.2 | 16.2% | 93.4% | 74.7% |
| horizon=1.6 | **11.6%** | 73.2% | 63.6% |

**两个核心发现**:

1. **σ 控制 stability-contact tradeoff 的位置**: σ↑ 手越近物体, 但越不稳定. 不存在"又贴又稳"的 sweet spot.
2. **更多 CEM budget 反而恶化 stability**: sample/iter/horizon 加大 → CEM 找到"更贪婪 (更近但更危险)"的方案. 反直觉但稳定复现.
3. **冷启动假设证伪**: 加 warmup_steps (前 1s 跳过 CEM 用 ref ctrl), 反而 stable 80.6% < no-warmup 95% — 问题不在冷启动, 是 CEM + hand_approach 的结构性缺陷.

→ **lesson #2**: 在 hand_approach reward 框架下, "更多算力 = 更糟结果". 必须从根本上改 reward 结构, 不是调参.

---

## 3. Phase 10 (E034 - E036) — HDMI 风格三阶段大改造, 一击突破

这是整个 Phase 9-12 最关键的三步. E034/E035/E036 不是独立实验, 而是**逐步移植 HDMI workflow 的三个核心 trick**, 每步解决一个具体问题.

### E034: HDMI bounded qpos + stability_penalty (反应式补救)

**移植 trick #1**: HDMI 用 `stability_penalty = -scale * relu(threshold - pelvis_z)` 在 pelvis 低于 0.55m 时给负奖励.

**desk005 数字**:
| 指标 | E032a | E034d (加 stability_penalty) |
|---|---|---|
| pelvis_z min | 0.223m (倒地) | 0.552m (前倾未倒) |
| 最长不稳定段 | 52帧 (0.87s) | 7帧 (0.12s) ↓86% |
| stable >0.55m | 79.3% | 100% |
| <10cm contact | 82.8% | 79.3% |

**视频复查 (诚实)**: desk005 t=1.04s sim 几乎水平, pelvis_z=0.552 刚过阈值 — 数字 "100% stable" 视觉上仍是"近乎摔倒". 4 个 case (chair022 加入) 严格评估有 2 个有近摔事件.

**lesson #3**: stability_penalty 是 **reactive** (pelvis 已低才触发), 不是 **preventive**. CEM horizon 0.8s 看不到 0.8s 后会摔. 必须在 reward 设计上让 CEM **预防**.

### E035: Local-frame body tracking (HDMI 不摔的真正核心)

**移植 trick #2 (HDMI 真正的杀手锏)**: body tracking error 在 pelvis yaw-only 坐标系中算, 不在 world frame.

```python
# 关键: 把 pelvis 旋转出来再算 body-to-pelvis 相对位置
yaw_quat = extract_yaw_only(pelvis_quat)
local_pos = quat_inv(yaw_quat) @ (body_pos - pelvis_pos)
err = ‖local_pos - local_pos_ref‖
```

物理意义: CEM 可以选择"pelvis 位置/yaw 稍偏 ref 但保持平衡". world-frame tracking 强行让 pelvis 跟 ref → 必摔.

**结果**:
| Case | MPKPE | Stability | Contact<10cm | 视觉 |
|---|---|---|---|---|
| desk005 | **47.92cm** ❌ | **100%** ✅ | 94.8% (但 ref 决定的) | 不摔了, 但跟 ref 完全不像 |
| box025 | 26.17cm | 100% | 40.3% | 同上 |
| bucket010 | 36.67cm | 100% | 44.0% | 同上 |

**关键诊断 (lesson #4)**: stability 解决了 (3/3 100% 不摔), 但 body tracking 直接崩 (MPKPE 47cm). 算 reward 上限:
- local_frame_rew (body tracking): max = 0.5 × 7 = 3.5
- hand_approach_rew: max = 5.0 (gain × exp(0))
- → CEM 把预算花在 hand_approach, body tracking 被忽略

→ HDMI workflow 之所以不摔, 不是因为加了 stability_penalty (E034), 而是因为同时**关掉了 hand_approach + 用了 local-frame tracking**.

### E036: 一刀关掉 hand_approach + 开 object tracking — 突破时刻 ★★★

**3 个改动**:
1. `hand_approach_rew_scale: 5.0 → 0.0` (关掉)
2. `stability_penalty_scale: 30.0 → 0.0` (不需要了)
3. `task_obj_pos_rew_scale: 0.0 → 1.0` + `task_obj_rot_rew_scale: 0.0 → 1.0` (开 object 跟踪)

**E036 vs E035 vs HDMI**:
| 指标 | HDMI R013 | DynaRetarget paper | **E036 box025** | E035 box025 | 改善倍数 |
|---|---|---|---|---|---|
| MPKPE | 7.72 cm | 3.57 cm | **1.43 cm** | 26.17 cm | **18×** |
| Joint Err | 3.22° | — | **0.85°** | 11.81° | **14×** |
| Root Pos | 7.17 cm | — | **1.06 cm** | 21.93 cm | **21×** |
| Obj Pos | 5.39 cm | 8.81 cm | **0.87 cm** | 16.80 cm | **19×** |
| Obj Ori | 4.28° | 6.3° | **0.40°** | 10.74° | **27×** |
| Stability >0.6 | 84.8% | — | **100%** | 100% | = |
| Contact<10cm | — | — | **55.6%** | 40.3% | +15% |

3/3 case 全面 18-50× 改善, MPKPE 从 26-48cm 降到 1.3-1.4cm — 跨数量级突破.

**为什么这么神?** 三件事同时发生:
1. CEM 唯一正向 reward 来自 body tracking → 100% 预算花在 tracking
2. `task_obj_rew = -err²` 是负向惩罚, 不与 tracking 抢上限 (本质是 cost, 不是 reward)
3. `contact_guidance: true` 让 object 用 PD 沿 ref 走, sim 不需要"主动搬", 也避免了"为接触而牺牲平衡"的 incentive

**新问题** — Contact<10cm 暴跌:
| Case | E035 | E036 | 变化 |
|---|---|---|---|
| desk005 | 94.8% | **6.9%** | -88% ❌ |
| box025 | 40.3% | 55.6% | +15% ✅ |
| bucket010 | 44.0% | **2.4%** | -42% ❌ |

→ 因为没有 reward 激励手伸向物体. box025 不掉因为 ref 中手本身就贴箱; desk005/bucket010 ref 中手离物体远, body tracking 把手送到"ref 位置"但那不是"物体表面".

**lesson #5 (整个 Phase 9-12 最核心的发现)**: 一个 scale 过大的 reward term (E035 时 hand_approach max=5 > tracking max=3.5) 可以摧毁所有其他目标. 关掉它 → 一切 18-50× 改善.

---

## 4. Phase 11 (E037 - E041) — Contact recovery 5 连发, 找到 E041c

E036 解决了 body tracking + stability + object tracking, 留下 contact 这个孤儿. Phase 11 全部聚焦在 contact recovery.

### E037: Mask-gated contact reward 第一发 — gain=2 太弱

**HDMI 风格**: `rew = mask × gain × exp(-dist/σ) + (1-mask) × const_baseline`

**E037 用 baseline=gain=2.0**: 致命 bug — 接触帧 max reward = 非接触帧 baseline → CEM 在接触帧能拿到的 ≤ 不接触时, 等于在**惩罚** contact. Contact<10cm: desk005 6.9%→10.3% (微改善 +3%), box025 55.6%→51.6% (退化), bucket010 2.4%→4.8%.

**E037b**: 修 baseline → 用 `mask × gain × exp(...)` (no constant). Contact 仍无改善 (desk 6.9, box 49.2, bucket 3.2). 真正的根因是 **gain=2.0 在 sigma=0.3 下信号太弱** vs body tracking max=3.5.

### E037c sweep: 找到 gain=3.5 σ=0.15 baseline=0 (S3)

S3 在 box025 上 55.6% → 57.3% (+1.7%), desk005/bucket010 完全无效.

**lesson #6** (重要): Contact mask reward 的根本局限 — 它只能在 "ref 中手已在物体附近" 的帧给信号, 但 desk005/bucket010 的 ref 中手离物体本身就 >10cm. **Contact<10cm 差不是因为 reward 弱, 是因为 ref 决定的**.

### E038: physics_dt=0.002 (HDMI alignment) — 4.3× 慢, 0 收益

| 指标 | E036 | S3 | **E038 (dt=0.002)** |
|---|---|---|---|
| MPKPE | 1.43 | 1.43 | 1.32 |
| Contact<10cm | 55.6% | 57.3% | **46.0%** ❌ |
| Total Time | 371s | 370s | **1609s** (4.3×) |

physics 更精细 → CEM 更保守 → 手不敢靠近. 完全无价值.

### E039: HDMI-aligned predefined target + per-EEF — 算法瓶颈"确认"

完全对齐 HDMI 的 contact reward 设计:
- `contact_target_offset` in obj local frame (per-task YAML)
- `target_world = obj_pos + quat_apply(obj_quat, offset)`
- `contact_point = eef_pos + quat_apply(eef_quat, [0.05, 0, 0])` (palm center)
- per-EEF (左右手分别), mean
- gain=5.0, sigma=0.3m (HDMI 默认)

box025 52% (退化), bucket010 13% (5× 改善), desk005 8%. 当时结论: "**算法瓶颈 = CEM**, RL 才能解决". **错的**.

### E039b: Smoking gun — Contact reward 之前 E036/E037/E038/E039 全都从未执行过 ⭐

**Bug**: `spider/config.py:651` 解析条件:
```python
if config.hand_approach_rew_scale > 0.0 and config.simulator == "mjwp":
    resolved_ids = [...]
    config.hand_approach_body_ids = resolved_ids   # 只有这里设置
```

E036 起 `hand_approach_rew_scale=0.0` → `body_ids=[]` → E037/E039 的 `if config.hand_approach_body_ids:` 永远 False → **整个 Phase 11 contact reward 从未执行过!** ~2 周工作建立在 contact reward 没生效的基础上.

**修复**: 扩展条件 + rotated SDF mask (替代 axis-aligned).

**E039b 结果 (修复后首次正确执行)**:
| Case | E036 | E039 (bug) | **E039b (fixed)** | Ref 上限 | Stability |
|---|---|---|---|---|---|
| box025 | 56% | 52% | **85%** ✅ | 69% | 100% |
| desk005 | 7% | 8% | **81%** ✅ | 0% | **20%** ❌ |
| bucket010 | 2.4% | 13% | **76%** ✅ | 76% | 100% |

box025/bucket010 直接达到 / 接近 ref 上限. desk005 contact 81% 但摔倒 (gain=5 把手拉向桌面下横梁 → 前倾摔).

**lesson #7**: Config 解析 bug 让我们花了 2 周得出错误的"CEM 算法瓶颈"结论. 任何"reward 不 work" 的结论必须先 verify reward 真的执行了.

**E039c (threshold 0.30 → 0.15)**: 修 mask 激活率, box025 82%, bucket010 68%, desk005 91% (但 stable 17%). 但视频暴露**新问题**: box025 t≈1.5-2.5s **手粘连物体** + 手腕反关节. 因为 `contact_target_offset` 是固定点, 但 CORE4D 中人围着物体活动, 手在物体表面位置随时变 → CEM 强行把手粘在固定点, 手腕扭曲, 身体后仰.

→ 固定 target 不适合 CORE4D, 必须用 dynamic target.

### E040: Dynamic per-frame target — 解决粘连, 引入手背接触

**改动**: 每帧从 ref FK 提取手相对物体的实际位置作为该帧的 target:
```python
target_offset[t, ei] = rot_inv(obj_quat[t]) @ (hand_pos_ref[t] - obj_pos[t])
```
本质上变成"在物体坐标系下的 EEF tracking".

**结果**:
| Case | E036 | E039b (固定) | **E040 (动态)** | Ref 上限 |
|---|---|---|---|---|
| box025 | 56% | 85% | 64% | 69% |
| bucket010 | 2% | 76% | 66% | 76% |
| desk005 | 7% | 81% (摔) | 4% | 0% |

数字看起来"退化", 但稳定性恢复 (desk005 100%), 手粘连消失.

**视频复查**: 仍有严重问题 — box025 t≈2s sim 弯腰趴向箱顶, **手背朝箱子**. position-only reward 让 CEM 旋转手腕去满足距离, 不管手掌方向. **lesson #8**: 距离指标 (Contact<10cm 64%, Preservation 88.8%) 高不代表接触自然 — 88.8% 的"接触"很多是手背接触, 数字漂亮行为不合理.

### E041: Hand orientation reward — 三种模式, additive(0.3)=E041c 胜出 ★

**Palm normal 分析** (用 box025 ref t=30-90 帧 wrist 旋转矩阵 vs wrist→object 方向 dot product):
- Left wrist: -y 轴一致指向物体 (dot −0.61~−0.87)
- Right wrist: +y 轴一致指向物体 (dot +0.67~+0.96)

→ `palm_normal_left=[0,-1,0]`, `palm_normal_right=[0,+1,0]`

**三种 mode**:
```python
# Mode 1: multiply (E041, E041b) — 严格门控
pos_rew = pos_rew * clamp(dot(palm, dir), min=0)

# Mode 2: additive (E041c, w=0.3) — 柔性加权
pos_rew = (1-w) * pos_rew + w * ori_rew

# Mode 3: near_field (E041d) — 仅近距离约束
near = (dist < 0.15).float()
pos_rew = pos_rew * (1 - near + near * ori_rew)
```

**Sweep 结果 (box025)**:
| 变体 | Mode | Gain | Contact<10cm | Stability | MPKPE | Preservation |
|---|---|---|---|---|---|---|
| E036 (no contact) | - | - | 56% | 100% | 1.4cm | - |
| E040 (pos-only) | - | 5 | 64% | 100% | 1.3cm | 88.8% |
| E041 (multiply) | × | 5 | 62% | 100% | 1.6cm | 86.5% |
| E041b (multiply) | × | 3 | 52% | 85% ❌ | 1.5cm | 74.4% |
| **E041c (additive)** ★ | + | 5 | **66%** | **100%** | **1.4cm** | **88.2%** |
| E041d (near_field) | gated | 5 | 58% | 98% | 1.5cm | 73.5% |

**bucket010**:
| 变体 | Contact<10cm | Stability | MPKPE | Preservation |
|---|---|---|---|---|
| E040 | 66% | 100% | 1.2cm | 95.4% |
| E041 (multiply) | 56% | 90% ❌ | 1.8cm | 80.5% |
| **E041c (additive)** | 57% | **100%** | **1.4cm** | **81.6%** |

**为什么 additive 赢, multiply 输**:
| 模式 | 对 CEM 压力 | 效果 |
|---|---|---|
| additive(0.3) (`0.7×pos + 0.3×ori`) | 低 (ori 只 30%) | ★★★ 保持 contact, 软引导 |
| near_field (`dist<15cm 时 pos×ori`) | 中 | ★★ stability OK, contact 偏低 |
| multiply (`pos × ori`) | 高 (必须同时满足) | ★ 全面退化 |
| multiply + gain=3 | 高 + 信号弱 | ✗ 完全崩 |

multiply 是逻辑"与" — CEM 必须同时让位置准 + 方向准 → 在 1024 sample × 32 iter 的有限 budget 下找不到. additive 是软"或" — 给 CEM 留余地用 70% 位置 + 30% 方向 trade-off.

**E041c 的视频** (诚实):
- box025 t=2.0s: 弯腰程度比 E040 轻, 手在箱面附近 — **比 E040 自然**, 但仍有前倾
- bucket010 t=1.7s: 弯腰手在桶顶 — 姿态合理
- 但 desk005 仍是 4% contact (没改 desk005 因为 ref 没接触, reward 救不了)

**重大概念修正 (lesson #9, 推翻之前 E039 的"算法瓶颈"结论)**:

核实 `examples/run_hdmi.py` 发现 — **HDMI workflow 在 SPIDER 中也是 CEM** (sampling-based MPC), 用完全相同的 `make_optimize_fn`. 而且 HDMI 的 `rew_contact` 也是 **position-only 无 orientation** (hdmi.py:1108). 但 HDMI 在 move_suitcase 任务上有自然手掌接触.

| | HDMI move_suitcase | CORE4D box025 |
|---|---|---|
| 接触目标 | 固定把手 | 平坦表面 |
| Wrist 自由度 | **零噪声** (run_hdmi.py:112-118) | 正常 CEM 采样 |
| 物体几何 | 凸出把手 → 引导手掌 | 平面 → 无引导 |
| Ref 动作 | 抓把手行走 → 手腕固定 | 推/搬 → 手腕变化 |

→ **HDMI 的"秘密"不是 RL, 是 (1) wrist 零噪声 + (2) 几何引导**. 启发 E042.

---

## 5. Phase 12 (E042 - E047) — 4 个突破方向, 全部失败, 锁定 E041c

E041c 后, 还有 4 条候选路径要 verify 是否能突破 66%:

### E042: Wrist freeze (HDMI 真正秘密) — CORE4D 上有害

**改动**: `zero_noise_joint_keywords: [wrist_roll, wrist_pitch, wrist_yaw]` (6 个关节).

**结果对 box025**:
| 变体 | E040 | E041c | E042a (E040+freeze) | E042b (E041c+freeze) | E042c (no contact+freeze) |
|---|---|---|---|---|---|
| Contact<10cm | 64% | 66% | 64% | 64% | 48% |
| Preservation | 88.8% | 88.2% | 83.2% | 84.0% | 58.8% |

**对 bucket010 — 严重退化**:
| 变体 | E040 | E041c | E042a | E042b |
|---|---|---|---|---|
| Contact<10cm | 66% | 57% | **48%** ↓18% | **34%** ↓23% |

**lesson #10**: HDMI 的 wrist freeze 在 move_suitcase 上 work 因为 ref 已经精确在把手位置, freeze 只需保持. CORE4D body tracking 有 1-5cm 误差, **CEM 需要微调手腕来补偿** — wrist freeze 阻止了这个补偿.

→ Wrist freeze 路线封死.

### E043: 原始 OmniRetarget Phase 3 ref 对比 — Phase 4 (松弛) 仍是最优

| Case | E040 (Phase 4) | **E043 (Phase 3 原始)** |
|---|---|---|
| box025 contact | **64%** | 52% (-12%) ❌ |
| bucket010 contact | **66%** | 57% (-9%) ❌ |
| desk005 stability | 81% | **100%** ✅ |
| desk005 MPKPE | 1.9cm | **1.0cm** ✅ |

Phase 3 原始 (无穿透松弛) 反而 box/bucket 更差, 只有 desk005 改善. 原因猜测: Phase 3 约束更严 → 姿态更僵硬, 手部位置受限; Phase 4 松弛允许手深入物体表面 → dynamic target 更精确.

→ **Phase 4 ref 仍是默认**. 数据源路线封死.

### E044: 手腕 tracking 权重增强 — stability 崩

| 实验 | wrist_weight | box025 contact | box025 stability | pelvis_min |
|---|---|---|---|---|
| E041c | 1.0 | 66% | 100% | — |
| E044a | **2.0** | 67% (+1) | **73%** ❌ | 0.113m |
| E044b | **3.0** | 59% (-7) | 100% | — |

w=2 时 contact 微改善 (+1) 但 stability 崩到 73%; w=3 时 stability 恢复但 contact 反而退化 (CEM 锁手腕导致全身保守).

→ **lesson #11**: 任何增强上半身 tracking 权重的方法都会与 stability 产生 tradeoff. wrist 权重路线封死.

### E045: Local-frame sigma 收紧 — 全面退化

| 实验 | local_frame_pos_sigma | box025 contact | bucket010 contact | desk005 |
|---|---|---|---|---|
| E041c | 0.5 | **66%** | **57%** | 4% |
| E045a | 0.3 | 38% (-28) | 20% (-37) | 6% |
| E045b | 0.15 | 46% (-20) | 23% (-34) | **22% (+18) 改善** |

sigma↓ → tracking 梯度更陡 → CEM 优先满足 tracking → contact reward 相对吸引力下降. desk005 唯一改善是因 desk005 ref 中手离物体本身就远, 收紧 sigma 抑制了 E041c 中"前倾趴向桌"的不自然行为.

→ **lesson #12**: contact 瓶颈不在 tracking 精度. 把 MPKPE 从 1.4 → 1.2cm 不改善 contact. **瓶颈在 ref 本身**.

### E047: SBTO 开环优化 (DynaRetarget paper 方法) — 完全摔

| 实验 | params | box025 MPKPE | box025 Contact | box025 Stability |
|---|---|---|---|---|
| E041c (MPC) | — | **1.4cm** | 66% | 100% |
| E047a | α_μ=0.95, σ_min=0.01 | **155cm** ❌❌ | 0% | 31% |
| E047b | α_μ=0.5, σ_min=0.03 | **87cm** ❌ | 28% | 98% |

E047 移植 DynaRetarget 论文的 SBTO 5 个修复 (收敛准则 / Sigma EWMA / Mean EWMA / elite_fraction / MPC 行为). 即使放松参数 (E047b), MPKPE 仍 50-100× 差于 MPC.

**根因**:
1. MPC 闭环反馈纠正偏差; SBTO 一次性优化全程, 累积误差无法纠正
2. DynaRetarget 用 **平方误差 cost** (无界负), SPIDER 用 **exp-kernel reward** (有界 [0, 3.5]) → 长 horizon mean reward 趋于常数, CEM 难区分好坏

→ **lesson #13**: SBTO 与 SPIDER 的 exp-kernel reward 不兼容. 算法路线封死.

---

## 6. 最终配置 — E041c 完整 yaml

```yaml
# examples/config/override/core4d_e041c.yaml
# (基于 E036 配置 + dynamic target + additive orientation)

# ── E036 突破: 关 hand_approach + 开 object tracking ──
hand_approach_rew_scale: 0.0
stability_penalty_scale: 0.0
task_obj_pos_rew_scale: 1.0
task_obj_rot_rew_scale: 1.0

# ── E035 移植: local-frame body tracking ──
use_local_frame_reward: true
local_frame_w_track: 0.5
local_frame_pos_sigma: 0.5      # E045 证明收紧无益
local_frame_wrist_weight: 1.0   # E044 证明增强无益

# ── E040 突破: dynamic per-frame target ──
contact_hdmi_dynamic_target: true
contact_hdmi_gain: 5.0
contact_hdmi_sigma: 0.3

# ── E041c 核心: additive orientation reward ──
contact_hdmi_ori_weight: 0.3    # multiply / near_field 都输给 additive
contact_hdmi_ori_mode: additive
palm_normal_left: [0, -1, 0]    # G1 wrist y 轴
palm_normal_right: [0, +1, 0]

# ── E036 接力: object PD ──
contact_guidance: true

# ── E039b bug fix: config 解析必须有 ──
# (在 spider/config.py:651 已修, contact_hdmi_gain>0 触发 body_ids 解析)
```

---

## 7. E041c 的真实定位 (诚实评估)

| 维度 | 数字 | 视频是否合格 |
|---|---|---|
| MPKPE 1.4cm | 数字超 HDMI 5×, DynaRetarget 2.5× | ✅ 但部分来自 contact_guidance PD 拖物体, 不全是 sim 主动跟踪 |
| Stability 100% | box/bucket OK, desk005 81% | ⚠️ desk005 仍有近摔事件 |
| Contact 66%/57% | 接近 ref 上限 (69%/76%) | ❌ 视频显示 box025 趴箱, bucket010 推非搬, **手没有真实抓握** |
| Object Pos 0.8cm | 远超 HDMI 5.4cm | ❌ object 由 PD 驱动, 不是 sim 接触力搬的 |

**STAGE_REPORT §0 已明确**: E041c 是"CEM 数值最佳, 但视频显示 box025 趴箱、desk005 丢桌走、bucket010 推非搬". 它解决了从 E001 起的 stability/body tracking/object tracking 三个问题, 但**没有解决"产生真实物理接触力"** — 这个是 Phase 13+ (E048-E067) 才被定位为 CORE4D + MJWP CEM 框架的根本限制.

E060 系列试图 box023 + bucket005 上"复用 E041c 配置" 时发现: **E041c 是 box025 + sphere hand collision 的过拟合配置, 没有 case 泛化性**. log 76/77 把战略转向 X1+X2 reward 泛化方向. 这是 E054-E067 的故事 (见 STAGE_REPORT §2.5/2.6).

---

## 8. 整个 Phase 9-12 的 13 条 lessons

| # | Lesson | 来源 |
|---|---|---|
| 1 | hand_approach reward 让 CEM"全身贴物体", 数字 81% 视觉趴箱 | E032a 视频 |
| 2 | hand_approach 框架下"更多 CEM budget = 更糟结果" (反直觉但稳定复现) | E033 sample/iter/horizon sweep |
| 3 | Stability_penalty 是 reactive 不是 preventive, 解决不了根本 | E034d 视频 t=1.04s 几乎水平 |
| 4 | HDMI 真正不摔的 trick 不是 penalty, 是 local-frame body tracking | E035 |
| 5 | **一个 scale 过大的 reward term 摧毁所有其他目标** (Phase 9-12 最核心) | E036 一刀关 hand_approach 后 18-50× 改善 |
| 6 | Contact mask reward 治不了 desk005/bucket010 因 ref 中手本身离物体 >10cm | E037c sweep |
| 7 | Config 解析 bug 让我们花了 2 周得出错误的"算法瓶颈"结论 | E039b smoking gun |
| 8 | 距离指标高 (88% Preservation) ≠ 接触自然, 88% 很多是手背接触 | E040 视频 |
| 9 | HDMI 的"秘密"不是 RL 是 wrist 零噪声 + 几何引导, HDMI 也是 CEM | E041 + run_hdmi.py 核实 |
| 10 | Wrist freeze 在 CORE4D 上有害, 因 body tracking 误差需 CEM 补偿 | E042 |
| 11 | 任何增强上半身 tracking 权重的方法都会与 stability 产生 tradeoff | E044 |
| 12 | Contact 瓶颈不在 tracking 精度 (MPKPE 1.4 → 1.2 不改善 contact) | E045 |
| 13 | SBTO 与 SPIDER 的 exp-kernel reward 不兼容 | E047 |

---

## 9. 索引 — 详细 log 位置

| 实验 | Log | 关键产出 |
|---|---|---|
| E032a | log/36 | sweep 矩阵 + 用户视频判断 |
| E033 | log/38 | σ sweep + warmup 证伪 |
| E034 | log/40 | stability_penalty 严格评估 |
| E035 | log/41 | local-frame tracking 移植 + MPKPE 48cm 诊断 |
| **E036** ★ | log/42 | 突破时刻 — 关 hand_approach 后 18-50× 改善 |
| E037 | log/43 | mask gate 第一发 + baseline bug |
| E037c | log/45 | gain/sigma sweep S3 |
| E038 | log/46 | physics_dt=0.002 — 4.3× 慢 0 收益 |
| E039 | log/47 | HDMI 完全对齐 |
| **E039b** ★ | log/48 | Smoking gun — 之前 4 个实验 contact reward 没执行 |
| E040 | log/49 | dynamic target — 解决粘连引入手背接触 |
| **E041 (E041c)** ★ | log/50 | 三 mode sweep, additive(0.3) 胜出 |
| E042 | log/51 | wrist freeze 失败 |
| E043 | log/52 | 原始 ref 对比 |
| E044/E045/E047 | log/53 | 4 突破方向全失败 |

→ Phase 9-12 后续: E048+ 是 collision box / HDMI 整体对照 (见 STAGE_REPORT §2.4); E054+ 是 box023/bucket005 hand-snap warmstart (见 §2.5); E062+ 是 init pose bug 定位 (见 §2.6).
