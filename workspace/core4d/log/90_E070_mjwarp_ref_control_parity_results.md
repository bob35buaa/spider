# E070 Results — MJWarp ref-control parity: 根因是 qpos-as-ctrl 映射错误

**日期**: 2026-05-14
**实验域 (exp_name)**: `core4d`
**对应 Plan**: `workspace/core4d/plan/75_E070_mjwarp_ref_control_parity_plan.md`
**前置**: `workspace/core4d/log/89_E069_first_tick_warmup_results.md`

## 1. 背景

E069 证明 warmup 内实际提交的 ctrl 与 `ctrl_ref` 完全一致，但 early yaw drift 仍是 12/22 deg。因此 E070 要回答：

> 同一 ref-control commit 下，MuJoCo CPU `mj_step` 和 MJWarp `step_env` 是否存在动力学不一致？

实际脚本进一步增加两个控制变量：

- `zero_gains`: object actuator gains 保持 0。
- `restored_gains`: 按 `run_mjwp.py` commit 阶段恢复 object actuator gains。
- `qpos_ctrl`: 当前 `run_mjwp.py` 口径，即 `ctrl_ref = qpos_ref[:, :config.nu]` 后再 scene_act 转换。
- `orig_ctrl`: 原始 29-dim robot ctrl + scene_act object 6DOF ctrl。

## 2. 结果路径

| 类型 | 路径 |
|------|------|
| CSV | `workspace/core4d/results/E070/ref_control_parity.csv` |
| JSON | `workspace/core4d/results/E070/ref_control_parity.json` |
| 运行日志 | `logs/E070/ref_control_parity.log` |
| 诊断脚本 | `workspace/core4d/scripts/debug/diagnose_E070_ref_control_parity.py` |
| 入口脚本 | `workspace/core4d/scripts/train/train_E070.sh` |

## 3. 核心结果

### 3.1 qpos_ctrl 精确复现 E069

| 路径 | t=0.017 yaw err | t=0.033 yaw err | qvel t=0.017 | vs E069 qpos |
|------|------------------|------------------|---------------|--------------|
| `qpos_ctrl_cpu_zero_gains` | 12.403 deg | 22.156 deg | 20.58 | ~0 |
| `qpos_ctrl_cpu_restored_gains` | 12.403 deg | 22.156 deg | 20.58 | ~0 |
| `qpos_ctrl_mjwarp_zero_gains` | 12.403 deg | 22.156 deg | 20.58 | ~0 |
| `qpos_ctrl_mjwarp_restored_gains` | 12.403 deg | 22.156 deg | 20.58 | ~0 |

结论：

- CPU MuJoCo 和 MJWarp 结果一致。
- object actuator gains 是否恢复几乎不影响结果。
- `qpos_ctrl` 路径与 E069 保存的 `.npz` 精确一致，因此 E069 的 drift 可以在纯 CPU MuJoCo 中复现。

### 3.2 orig_ctrl 基本消除 early drift

| 路径 | t=0.017 yaw err | t=0.033 yaw err | qvel t=0.017 | 脚高 |
|------|------------------|------------------|---------------|------|
| `orig_ctrl_cpu_zero_gains` | 0.574 deg | 1.075 deg | 0.475 | L=0.037m / R=0.042m |
| `orig_ctrl_cpu_restored_gains` | 0.574 deg | 1.075 deg | 0.475 | L=0.037m / R=0.042m |
| `orig_ctrl_mjwarp_zero_gains` | 0.574 deg | 1.075 deg | 0.475 | L=0.037m / R=0.042m |
| `orig_ctrl_mjwarp_restored_gains` | 0.574 deg | 1.075 deg | 0.475 | L=0.037m / R=0.042m |

12 substeps 后 `orig_ctrl` yaw err 约 2.67 deg，仍远低于 `qpos_ctrl` 的 64.21 deg。脚高保持在约 3.5-4.2cm，没有进入 E069 的单脚/lunge basin。

## 4. 根因

`examples/run_mjwp.py` 当前逻辑：

```python
if config.contact_guidance and ctrl_ref.shape[1] != config.nu and qpos_ref.shape[1] >= config.nu:
    ctrl_ref = qpos_ref[:, : config.nu]
```

在 G1 humanoid_object scene_act 中：

- 原始 `ctrl_ref` 是 29-dim robot actuator target。
- `config.nu` 是 35 = 29 robot actuators + 6 object actuators。
- `qpos_ref[:, :35]` 不是 35-dim actuator ctrl；它包含 floating base pos(3) + quat(4) + robot joints 的前 28 维。

因此 qpos-as-ctrl 把 floating-base qpos 前 7 维错误喂给 robot actuators。E068 看到的 first committed ctrl 相对原始 `ctrl_ref` 有 1.56 rad 偏差，本质不是 CEM override，而是这个错误 ctrl_ref 映射。

正确映射应是：

- robot ctrl 前 29 维保留原始 `ctrl_ref`。
- object ctrl 后 6 维使用 scene_act 转换得到的 object slide/euler target。

E070 的 `orig_ctrl` 对照正是这个映射，并且 early drift 基本消失。

## 5. Claims 验证

| Claim | 结果 |
|-------|------|
| C1: parity 脚本可复现 E069 early yaw | **通过** — qpos_ctrl CPU/MJWarp 均复现 12.403/22.156 deg，且 vs E069 qpos ≈ 0 |
| C2: CPU MuJoCo ref-control 不复现 drift | **修正后通过** — 使用正确 orig_ctrl 后 CPU yaw err 0.574/1.075 deg |
| C3: 差异能定位到物理通道 | **通过** — 差异不在 physics，而在 ctrl_ref preprocessing |
| C4: 结果可直接指导 E071 | **通过** — 修 `run_mjwp.py` scene_act ctrl mapping，然后重跑 box023 |

## 6. 结论修正

E068/E069 的推断需要再次修正：

1. log 87 的现象成立：MJWP early yaw drift 真实存在。
2. E068 的 "first CEM override" 判断不完整：所谓 override 来自错误 `ctrl_ref` 口径，而不是 CEM 本身。
3. E069 的 warmup 失败是必然的：warmup 强制提交的 `ctrl_ref` 已经是错误控制。
4. E070 定位到直接根因：**contact_guidance 下用 `qpos_ref[:, :nu]` 补 ctrl 维度是错误的**。

## 7. 下一步

E071: **修复 scene_act ctrl mapping 并重跑 box023**。

建议最小实验：

- 修改 `run_mjwp.py`：删除 qpos-as-ctrl 分支，改为保留原始 29-dim robot ctrl，只填入 object 6DOF ctrl。
- 新建 `core4d_e071_box023.yaml` 继承 E062 或 E069-W02。
- 首跑 warmup=0.20，成功标准：
  - t=0.017/0.033 yaw err < 2 deg。
  - B1 max foot z [0,2s] <= 0.10m。
  - 视频 t=0.2/0.6/0.8s 不再出现初始转身/单脚 lunge。

## 8. 改动文件

| 类型 | 路径 |
|------|------|
| 诊断脚本 | `workspace/core4d/scripts/debug/diagnose_E070_ref_control_parity.py` |
| 入口脚本 | `workspace/core4d/scripts/train/train_E070.sh` |
| 结果 | `workspace/core4d/results/E070/ref_control_parity.{csv,json}` |
| 日志 | `workspace/core4d/log/90_E070_mjwarp_ref_control_parity_results.md` |
