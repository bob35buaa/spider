# E027d2: Contact Guidance — Body-Frame Fix + Commit Gain Restore

## 状态: 突破性进展 — 4/4 case 物体移动 + 机器人行走

## 核心发现

### Bug 1: Slide 关节坐标系错误（根本原因）
- slide_pos 计算使用 `world_pos - body_pos`（世界坐标偏移）
- 但 slide joints 在 **body frame** 中操作
- desk005 body_quat ≈ 90° rotation：body Y → world Z, body Z → world -Y
- 修复：`slide_pos = R_body.inv() @ (world_pos - body_pos)`
- 修复后 desk005 obj tracking error: **0.77 → 0.15**

### Bug 2: Commit Phase Gain 归零
- CEM 最后一次迭代 `residual_gain_ratio=0` → gains=0
- Commit step 调用 `step_env` 时 PD actuator 无力
- 修复：commit loop 前 `load_env_params` 恢复初始 gains

### Bug 3: Contact Guidance 加载预转换数据
- `contact_guidance=true` 时自动加载 `trajectory_kinematic_act.npz`
- 该文件用旧的（world-frame）转换生成
- 修复：始终加载 `trajectory_kinematic.npz` + 运行时转换

## 结果

| Case | stable% | obj_disp(m) | pelvis_disp(m) | pos_err | quat_err | 视觉 |
|------|---------|-------------|----------------|---------|----------|------|
| box025 | **100%** | 1.46 | 1.54 | **0.131** | 0.110 | ★★★ 机器人走路+箱子跟随 |
| desk005 | **87%** | 1.54 | 1.52 | **0.152** | 0.065 | ★★★ 桌子完整移动（中间摔一次恢复） |
| bucket010 | **100%** | 0.92 | 1.24 | **0.111** | 0.249 | ★★ 桶移动但旋转偏大 |
| chair022 | **81%** | 0.76 | 1.07 | **0.156** | 1.581 | ★ 椅子移动但大幅旋转 |

## 对比 E027b (Object PD Override, anchored)

| Case | E027b pos_err | E027d2 pos_err | E027b stable | E027d2 stable | 差异 |
|------|--------------|----------------|--------------|---------------|------|
| desk005 | 0.100 | **0.152** | 100% | 87% | 略差但机器人在走路! |
| box025 | 0.101 | **0.131** | 100% | **100%** | 相当 |
| bucket010 | 0.547 | **0.111** | 100% | **100%** | **大幅改善** |
| chair022 | 0.559 | **0.156** | 100% | 81% | pos改善，rot仍差 |

**E027d2 的关键进步**: 机器人不再原地站立——它实际在行走并跟随物体。E027b 用 anchored 轨迹（原地），E027d2 用原始轨迹（locomotion）。

## 配置

```yaml
contact_guidance: true
guidance_decay_ratio: 1.0  # 无衰减（持续 PD）
residual_gain_ratio: 1.0  # commit 时保留完整 gains
init_pos_actuator_gain: 500
init_rot_actuator_gain: 50
base_pos_rew_scale: 5.0  # 强身体跟踪
base_rot_rew_scale: 3.0
physics_dt: 0  # 使用默认
```

## 代码改动

| 文件 | 改动 |
|------|------|
| `examples/run_mjwp.py` | 1. slide_pos body-frame fix (line ~367) |
| | 2. ctrl_ref object channels = converted values (line ~400) |
| | 3. commit-phase gain restore (line ~732) |
| | 4. conversion triggers for contact_guidance too (line ~352) |
| `spider/config.py` | 始终加载 freejoint trajectory（不再特殊处理 _act.npz） |

## 结果路径

| 产出 | 路径 |
|------|------|
| desk005 (best) | `workspace/core4d/results/E027d2_desk005_strongbody.{npz,mp4}` |
| box025 | `workspace/core4d/results/E027d2_box025_contact_guidance.{npz,mp4}` |
| bucket010 | `workspace/core4d/results/E027d2_bucket010_contact_guidance.{npz,mp4}` |
| chair022 | `workspace/core4d/results/E027d2_chair022_contact_guidance.{npz,mp4}` |

## 下一步

1. 视频验证所有 4 case（特别是 chair022 的大旋转问题）
2. 提高 chair022 的 rot gains 或使用 per-case euler convention 调优（暂时不考虑死磕chair022，先把其他3个case做好）
3. 导出 hybrid 轨迹到 Holosoma RL（暂时也不考虑）
4. 尝试 HDMI physics_dt=0.002 看是否进一步改善
