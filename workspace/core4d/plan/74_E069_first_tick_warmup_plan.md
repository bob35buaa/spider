# E069 实验计划: First-tick ref-control warmup 验证

## Context

E068 修正了 log 87 的根因判断：

- `qpos_ref[0] + mj_forward` 与 ref 完全对齐。
- init `mj_step` 只造成 0.22 deg yaw drift。
- `mjwarp.put_data` 不放大 init drift。
- 真实 E062/E063/E067 在 first committed step 后快速漂移：t=0.017s yaw err=12 deg, t=0.033s yaw err=22 deg。
- 第一帧 committed robot ctrl 相对 `ctrl_ref` 最大偏差约 1.56 rad，object ctrl 仅约 0.01。

因此当前最强假设不是 init pose bug，而是 **第一个 MPC tick 没有 ref-control warmup，CEM 在初始站姿未 settle 前立即提交极端 robot ctrl**。

### 根因分析

`examples/run_mjwp.py` 已有 `warmup_steps` 逻辑：

```python
if warmup_ctrl_steps > 0 and ctrl_step_idx < warmup_ctrl_steps:
    ctrls = ctrl_ref[sim_step : sim_step + config.horizon_steps]
else:
    ctrls, infos = optimize(...)
```

这正好可用于无代码验证：让前 0.2s / 0.5s 使用 ref ctrl，不让 CEM 在 t=0 直接覆盖控制。

### 关键 insight

如果 warmup 后 early yaw drift 消失，说明 first-tick CEM override 是主因；后续 fix 应该是 trust region / delta clamp / warm-start schedule，而不是继续调 reward weight。

## Claims

| Claim | 最低证据 |
|-------|---------|
| C1: warmup 能压住 early yaw drift | E069-W02 或 W05 在 t=0.017/0.033s yaw err < 5 deg (E062=12/22 deg) |
| C2: warmup 能压住 pre-contact foot lift | B1 max foot z [0,2s] <= 0.10m (E063=0.481m) |
| C3: box023 body stability 改善 | pelvis_min_intent >= 0.50m 或至少比 E063 0.192m 提升 > 20cm |
| C4: first ctrl 不再极端偏离 ref | warmup 窗口内 robot ctrl max diff <= 0.05 rad |
| C5: 视频实际姿态改善 | t=0.4/0.6/0.8s 不再出现快速侧转、lunge 或 handstand |

## 改动

### 1. 新增两个 yaml 变体

**文件**: `examples/config/override/core4d_e069w02_box023.yaml`

```yaml
defaults:
  - core4d_e062_box023
  - _self_
warmup_steps: 0.20
```

**文件**: `examples/config/override/core4d_e069w05_box023.yaml`

```yaml
defaults:
  - core4d_e062_box023
  - _self_
warmup_steps: 0.50
```

### 2. 训练脚本

**文件**: `workspace/core4d/scripts/train/train_E069.sh`

运行两个 box023-only 变体：

```bash
bash workspace/core4d/scripts/train/train_E069.sh parallel 0 1
```

### 3. 评估脚本

**文件**: `workspace/core4d/scripts/eval/eval_E069.py`

输出：

- `workspace/core4d/results/E069/eval_summary.csv`
- early yaw err
- first ctrl delta
- B1/B2 foot lift metrics
- pelvis_min / pelvis_min_intent

## 需要修改的文件

| # | 文件 | 改动 |
|---|------|------|
| 1 | `examples/config/override/core4d_e069w02_box023.yaml` | 新建 W02 warmup 变体 |
| 2 | `examples/config/override/core4d_e069w05_box023.yaml` | 新建 W05 warmup 变体 |
| 3 | `workspace/core4d/scripts/train/train_E069.sh` | 新建训练入口 |
| 4 | `workspace/core4d/scripts/eval/eval_E069.py` | 新建评估脚本 |
| 5 | `workspace/core4d/log/89_E069_first_tick_warmup_results.md` | 实验完成后记录结果 |
| 6 | `workspace/core4d/progress.md` | 更新执行状态 |

## 训练命令

```bash
bash workspace/core4d/scripts/train/train_E069.sh parallel 0 1
```

若只跑一个：

```bash
bash workspace/core4d/scripts/train/train_E069.sh single_w02 0
bash workspace/core4d/scripts/train/train_E069.sh single_w05 0
```

## 成功标准

| 指标 | 前次 E062/E063 | 本次目标 |
|------|----------------|----------|
| t=0.017s yaw err | 12.34 deg | **< 5 deg** |
| t=0.033s yaw err | 22.00 deg | **< 5 deg** |
| robot ctrl max diff during warmup | 1.56 rad | **<= 0.05 rad** |
| B1 max foot z [0,2s] | 0.481m | **<= 0.10m** |
| pelvis_min_intent | 0.192m | **>= 0.50m** |
| 视频 | lunge / handstand | **不出现 early lunge / handstand** |

## Decision Tree

| 结果 | 解读 | 下一步 |
|------|------|--------|
| W02/W05 early yaw + B1 通过 | first-tick CEM override 是主因 | E070: 实现 CEM trust region / ctrl delta clamp |
| early yaw 通过但 warmup 后立即 lunge | t=0 修好了，但 CEM 后续仍离 ref 太远 | E070: 每个 MPC tick 加 delta clamp，不只是 warmup |
| W05 通过但 W02 不通过 | 需要 settle window | 调 warmup/ramp schedule |
| W02/W05 都失败 | warmup 不是主因 | 回到 actuator force / contact impulse trace |

## 停止条件

- 不继续调 reward weight。
- 不先改 `spider/` 主代码。
- 若两个 warmup 变体都失败，先写 log 并重新诊断，不继续盲跑。
