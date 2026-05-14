# E071 实验计划: 修复 scene_act ctrl mapping 并验证 box023 early drift

## Context

E070 已定位 E069 early drift 的直接根因：

- CPU MuJoCo 与 MJWarp 在同一 ctrl 下完全一致，不是 MJWarp physics mismatch。
- 当前 `run_mjwp.py` 的 `qpos_ctrl` 口径精确复现 E069:
  - t=0.017/0.033 yaw err = 12.403 / 22.156 deg
  - vs E069 qpos max diff ≈ 0
- 正确 `orig_ctrl` 口径基本消除 early drift:
  - t=0.017/0.033 yaw err = 0.574 / 1.075 deg

根因代码：

```python
if config.contact_guidance and ctrl_ref.shape[1] != config.nu and qpos_ref.shape[1] >= config.nu:
    ctrl_ref = qpos_ref[:, : config.nu]
```

在 G1 humanoid_object scene_act 中，`qpos_ref[:, :35]` 包含 floating base pos(3) + quat(4) + robot joints 前 28 维，不是 35-dim actuator ctrl。正确 ctrl 应该是：

- robot ctrl 前 29 维 = 原始 `ctrl_ref`
- object ctrl 后 6 维 = scene_act object slide/euler target

## Claims

| Claim | 最低证据 |
|-------|---------|
| C1: 修复后 warmup ref-control early yaw 消失 | t=0.017/0.033 yaw err < 2 deg |
| C2: pre-contact foot lift 明显下降 | B1 max foot z [0,2s] <= 0.10m |
| C3: run_mjwp 保存结果与 E070 orig_ctrl parity 一致 | 前 2 substeps yaw 接近 0.574/1.075 deg |
| C4: 不破坏结果保存 | `.npz`、`.mp4`、`eval_summary.csv` 三件齐 |
| C5: 视频姿态改善 | t=0.2/0.6/0.8s 不再出现初始转身/单脚 lunge |

## 改动

### 1. 修复 `examples/run_mjwp.py`

删除 qpos-as-ctrl fallback：

```python
ctrl_ref = qpos_ref[:, : config.nu]
```

保留原始 `ctrl_ref`，让后续 scene_act conversion 负责把 29-dim robot ctrl padding 到 35-dim 并填入 object 6DOF ctrl。

### 2. 新增配置

**文件**: `examples/config/override/core4d_e071w02_box023.yaml`

```yaml
defaults:
  - core4d_e062_box023
  - _self_

warmup_steps: 0.20
```

### 3. 新增训练/评估脚本

**文件**: `workspace/core4d/scripts/train/train_E071.sh`

命令：

```bash
bash workspace/core4d/scripts/train/train_E071.sh 0
```

输出：

- `workspace/core4d/results/E071/E071W02_box023.npz`
- `workspace/core4d/results/E071/E071W02_box023.mp4`
- `workspace/core4d/results/E071/eval_summary.csv`
- `logs/E071/E071W02_box023.log`

评估可复用 E069 逻辑，但 ctrl_ref 口径必须改为 orig_ctrl。

## 成功标准

| 指标 | E069-W02 | E071 目标 |
|------|----------|-----------|
| t=0.017 yaw err | 12.40 deg | < 2 deg |
| t=0.033 yaw err | 22.16 deg | < 2 deg |
| B1 max foot z [0,2s] | 0.222m | <= 0.10m |
| qpos vs E070 orig parity | N/A | 前 2 substeps yaw 接近 0.574/1.075 deg |
| 视频 | 0.2s 已转身/单脚 | 不出现初始转身/单脚 lunge |

## Decision Tree

| 结果 | 解读 | 下一步 |
|------|------|--------|
| early yaw + foot lift 通过 | ctrl mapping 是主因 | 继续评估后续 CEM 是否仍在 contact 阶段失败 |
| early yaw 通过但 warmup 后 lunge | 初始 bug 修复，后续仍需 CEM trust-region / stability prior | E072 做 CEM delta clamp |
| early yaw 不通过 | run_mjwp 仍有其他 ctrl/state sync 口径不一致 | 用 E070 trace 对 run_mjwp 主循环逐行对齐 |
| 保存失败 | 修 eval/save，不改动力学结论 | 先修保存 |

## 停止条件

- 不再调 reward weight。
- 不再改 MJWarp physics。
- 不再延长 warmup 试错。
- 如果 early yaw 未修复，先回到 trace 对齐，不跑长实验。
