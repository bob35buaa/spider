# E034a: HDMI-Style Reward (Contact Mask + Bounded Qpos)

## Context

E033 + HDMI分析发现 MJWP 机器人摔倒的3个根因：
1. hand_approach 全程激活 → 不该前倾时前倾
2. Global qpos L2 无下界 → 正的 HA reward 覆盖负的 qpos reward
3. CEM 选"姿态差但手近"方案

## Claims

1. **C1**: Contact mask 消除不该前倾时的前倾 — stable ≥ 90% on desk005 (vs E032a 79%)
2. **C2**: Bounded reward 防止 HA 覆盖 qpos — reward 方差下降
3. **C3**: Contact quality 不降低 — contact<15cm ≥ 80% on desk005

## 改动

| 文件 | 改动 |
|------|------|
| `spider/config.py` | +4 fields: hand_approach_contact_threshold, use_bounded_qpos_reward, qpos_reward_sigma, qpos_reward_scale |
| `spider/simulators/mjwp.py` | get_reward: bounded exp qpos + contact mask gate on hand_approach |
| `examples/run_mjwp.py` | 预计算 approach_mask (ref hand-obj dist), 7-tuple ref_data |
| `examples/config/override/core4d_e034a.yaml` | 新配置 |

## 训练命令

```bash
MUJOCO_GL=egl uv run examples/run_mjwp.py +override=core4d_e034a task=desk005_person2
```

## 成功标准

- desk005 stable ≥ 90%
- desk005 contact<15cm ≥ 80%
- 视频无异常摔倒
