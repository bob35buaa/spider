# E035: Local-Frame Body Tracking (HDMI Core Design)

## Context

E034 证明 stability_penalty 是 reactive 的（pelvis 已经低才触发），无法预防摔倒。
HDMI 不摔的**真正核心**是 local-frame body tracking：

- 在 pelvis yaw-only 坐标系中计算 body tracking error
- 即使 pelvis 全局位移/旋转偏离 ref，只要身体**相对姿态正确**，tracking reward 依然高
- CEM 可以选择"pelvis 稍偏但保持平衡"的方案，而不会因全局偏移被 qpos_rew 惩罚

当前 MJWP 的 qpos L2 reward：pelvis 位移误差和关节角在同一个 L2 norm，CEM 为了减少这个 penalty 会做危险的动作把 pelvis 拉回 ref 位置。

## Claims

1. **C1**: Local-frame tracking 消除 desk005 t≈1s 的严重前倾 — pelvis_z >0.65m 全程 (>0.70m ≥90%)
2. **C2**: 4/4 cases 无完全摔倒事件 (pelvis_z min > 0.50m)
3. **C3**: desk005 contact<10cm ≥ 70% (不降低)

## 方案

**替换 qpos tracking 为 HDMI 风格的 body-space tracking**，保留 stability_penalty 作为保底。

### 新 reward 结构:
```
reward = W_TRACK * (
    local_upper_pos +    # 上半身 body pos (local frame, σ=0.5)
    local_upper_ori +    # 上半身 body ori (local frame, σ=1.0)
    local_lower_pos +    # 下半身 body pos (local frame, σ=0.5)
    local_lower_ori +    # 下半身 body ori (local frame, σ=1.0)
    root_pos +           # pelvis global pos (σ=0.5)
    root_ori +           # pelvis global ori (σ=0.5)
    joint_rew            # 关节角 tracking (σ=0.25)
) + hand_approach_rew + stability_penalty
```

### G1 body 分组:
- **root**: pelvis (id=1)
- **lower_body**: hip/knee/ankle (ids=2-13)
- **upper_body**: waist/shoulder/elbow/wrist (ids=14-30)

### 改动文件

| 文件 | 改动 |
|------|------|
| `spider/simulators/mjwp.py` | 添加 `_yaw_quat`, `_quat_apply_inverse`, `_pos_tracking_local`, `_ori_tracking_local` helpers; 新 `get_reward_local_frame()` 函数 |
| `spider/config.py` | 添加 `use_local_frame_reward: bool`, `local_frame_upper_ids`, `local_frame_lower_ids`, sigma 参数 |
| `examples/run_mjwp.py` | 预计算 ref body xpos + xquat (所有 body, 每帧) |
| `examples/config/override/core4d_e035.yaml` | 新配置 |

### 预计算 ref body xpos/xquat

在 run_mjwp.py 中，对每帧做 mj_kinematics 获取所有 body 的 xpos 和 xquat。存为 ref_data 第8个 tuple 元素。

### ref_data tuple (8-tuple):
```
(qpos_ref, qvel_ref, ctrl_ref, contact, contact_pos, body_xpos_ref, approach_mask, body_xquat_ref)
```

其中 `body_xpos_ref` 改为 **(T, nbody, 3)** 全 body（目前只存 task_body_ids 对应的），`body_xquat_ref` 为新增 **(T, nbody, 4)**。

## 训练命令

```bash
MUJOCO_GL=egl uv run examples/run_mjwp.py +override=core4d_e035 task=desk005_person2
```

## 验证

1. 跑 desk005，用 `eval_e034_rigorous.py` 评估
2. 视频保存到 `workspace/core4d/results/E035/`
3. 跑 box025/bucket010/chair022
4. 所有视频+指标写入 log
