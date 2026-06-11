# E155 — hand_support_rew 放手平滑过渡实验

## 0. 背景

E154 确认 gate 代码正确无 bug。release_false 高是因为 b1 (scale=3.0) 在搬运段把手压到 SDF≈0，
放手段 reward 突降为 0 后，qpos_rew 对手位置的梯度仅 ~0.2/step，无法快速拉开手。

## 1. Claims

- **C1**: 至少 1 个方案使 3-case 平均 release_false ≤ 0.15（vs E153 的 0.39）
- **C2**: inmaskC 不下降超过 0.10（vs E153 的 0.81）
- **C3**: success_tracked 保持 3/3

## 2. 方法

| 方法 | 代号 | 描述 |
|------|------|------|
| A1 ramp-5 | `ramp5` | mask 5帧线性渐退(中点在边界) |
| A2 ramp-10 | `ramp10` | mask 10帧线性渐退(中点在边界) |
| C decay | `decay` | 末15%帧 scale 线性 3.0→0 |
| D neutral | `neutral` | gate=0时给1.0中性baseline(学hdmi) |

全部使用 gateA_b1 (sdf=-0.010, viol=0.100)，3 case 本机串行。

## 3. 改动文件

- `spider/config.py`: +4 字段
- `examples/run_mjwp.py`: mask union + ramp 后处理
- `spider/simulators/mjwp.py`: decay + neutral baseline

## 4. 训练命令

```bash
# Smoke
bash workspace/core4d/scripts/E155/smoke_all.sh

# Full
bash workspace/core4d/scripts/E155/run_all_local.sh
```

## 5. 结果路径

| 类型 | 路径 |
|------|------|
| 计划 | workspace/core4d/plan/164_E155_release_smooth_transition_plan.md |
| CEM 输出 | workspace/core4d/results/E155/cem/{smoke,full}/ |
| 评测 | workspace/core4d/results/E155/eval/ |
| 脚本 | workspace/core4d/scripts/E155/ |
