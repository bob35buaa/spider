# E048-E049 修正: HDMI 评估指标严重失真 + 真实对比

## 状态: 🔴 重大修正 (2026-05-11)

## 发现的评估 Bug

**trajectory_hdmi.npz 的双通道不是 sim vs ground-truth ref**:
- Channel 0 = sim (物理仿真)
- Channel 1 = HDMI 内部 ref (随仿真漂移, 非真实 kinematic ref!)

HDMI 内部 ref 与 kinematic ground truth 的偏差:
- box023: t=2s 偏移 45°, t=3s 偏移 422°
- box025: t=2s 偏移 45°, t=4s 偏移 108°

**之前报的 MPKPE=0.3-0.7cm / Contact=93-99% 全部是虚假指标。**

---

## 正确评估 (sim vs kinematic ground truth)

### box023 (小箱子, 34×35×39cm)

| 指标 | HDMI (真实) | E041c | 谁更好 |
|------|------------|-------|--------|
| Joint Angle Error | **7.3°** | 19.8° | HDMI 2.7× |
| Pelvis Pos Error | **17.2cm** | 39.5cm | HDMI 2.3× |
| Object Pos Error | 24.4cm | **13.9cm** | E041c 1.8× |
| Object Rot Error | 66.4° | **27.0°** | E041c 2.5× |
| Stability>0.6m | **100%** | 38% | HDMI 显著 |

### box025 (大箱子, 75×76×94cm)

| 指标 | HDMI (真实) | E041c | 谁更好 |
|------|------------|-------|--------|
| Joint Angle Error | **5.3°** | 21.5° | HDMI 4× |
| Pelvis Pos Error | 78.7cm | **24.8cm** | E041c 3.2× |
| Object Pos Error | 90.9cm | **15.5cm** | E041c 5.9× |
| Object Rot Error | 53.2° | **6.6°** | E041c 8× |
| Stability>0.6m | **100%** | 98% | 持平 |

---

## 修正后的结论

### 1. HDMI ≠ 全面优于 E041c, 而是各有所长

| 维度 | HDMI 优势 | E041c 优势 |
|------|----------|-----------|
| 机器人 body tracking | ✅ Joint err 5-7° (好 3-4x) | |
| 机器人稳定性 | ✅ 100% (不摔倒) | |
| 物体位置追踪 | | ✅ 14-16cm (好 2-6x) |
| 物体方向追踪 | | ✅ 7-27° (好 2-8x) |
| 全局位置 (pelvis) | 小箱子好 | 大箱子好 |

### 2. HDMI 的物体 PD 控制器严重失效

HDMI 的 contact guidance 使用 PD actuator 控制物体, 但:
- 物体位置漂移 24-91cm (vs E041c 的 14-16cm)
- 物体旋转漂移 53-66° (vs E041c 的 7-27°)
- 物体越大失效越严重 (box025 比 box023 差 3.7x)

根因: HDMI 的 object PD gains (kp=20, kd=20) 可能不足以对抗碰撞力, 或者 gain schedule (guidance_decay) 太激进。

### 3. "算法是瓶颈" 的结论需要修正

之前说 "HDMI 碾压 E041c, 算法是瓶颈" — **错误**。正确结论:
- 两种方法在不同维度各有优势
- HDMI 在 body tracking + stability 上更好 (得益于 Isaac PD gains + wrist damping)
- E041c 在 object tracking 上更好 (得益于 stronger contact guidance gains)
- 物体交互的核心问题 (box025 单人无法搬) 两者都未解决

### 4. 可视化 Bug 总结

render_trajectory_video.py 存在多个 bug:
1. euler convention 用了 YXZ (CORE4D scene_act 的) 而非 XYZ (HDMI 的)
2. `flatten_mjwp` 取 `qpos[:, -1, :]` = channel 1 (内部 ref), 不是 sim
3. 时间对齐: kin (542 frames) vs phys (250 frames) 线性 resample 导致时间错位

---

## 被撤回的错误结论

| 之前的结论 | 修正为 |
|-----------|--------|
| HDMI box023 Contact=93% | ❌ 虚假 (sim vs drifted ref) |
| HDMI box025 Contact=99% | ❌ 虚假 (sim vs drifted ref) |
| HDMI MPKPE=0.3-0.7cm | ❌ 虚假 (应为 17-79cm pelvis err) |
| "HDMI 碾压 E041c" | ❌ 各有所长, 不是碾压 |
| "算法是瓶颈, 非数据" | ⚠️ 需更审慎: 两种算法都有严重缺陷 |

---

## 结果路径 (不变)

视频 v5 (正确渲染): `workspace/core4d/results/E049/E049e_hdmi_box025_comparison_v5.mp4`
