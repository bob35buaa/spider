# E070 实验计划: MJWarp ref-control commit parity 诊断

## Context

E069 推翻了 E068 的 first CEM override 假设：

- W02/W05 warmup 窗口内实际提交的 robot/object ctrl diff 都是 0。
- 但 t=0.017/0.033s yaw err 仍是 12.40/22.16 deg，和旧结果几乎相同。
- E068 CPU 诊断中，`mj_step(qpos_ref[0], qvel_ref[0], ctrl_ref[0])` 只造成 0.22 deg yaw drift。

因此新的最强假设是：

> 同样的 ref-control commit，在 MuJoCo CPU `mj_step` 和 MJWarp `step_env` 中产生了不同动力学响应；MJWarp 路径可能在 actuator、contact、state sync 或 scene_act 参数上与 CPU 路径不一致。

## Claims

| Claim | 最低证据 |
|-------|---------|
| C1: parity 脚本可复现 E069 early yaw | MJWarp `step_env(ctrl_ref)` 在前 2 substeps 复现约 12/22 deg yaw err |
| C2: CPU MuJoCo ref-control 不复现 drift | CPU `mj_step(ctrl_ref)` 前 2 substeps yaw err 仍 < 1 deg |
| C3: 差异能定位到物理通道 | dump 中至少能判断差异来自 qvel/actuator force/contact/object reaction 中的一类 |
| C4: 结果可直接指导 E071 | 明确下一步是修 MJWarp sync/actuator/contact 之一，而不是继续 reward/trust-region |

## 改动

### 1. 新增诊断脚本

**文件**: `workspace/core4d/scripts/debug/diagnose_E070_ref_control_parity.py`

功能：

- 构建 `core4d_e069w02_box023` config，但不跑 CEM。
- 加载并 mirror `run_mjwp.py` 的 ref 处理：
  - contact guidance 下 `ctrl_ref = qpos_ref[:, :nu]`
  - scene_act freejoint→slide/euler 转换
- 初始化同一 `qpos_ref[0] / qvel_ref[0]`。
- 跑三条路径：
  - `cpu_forward`: 只 `mj_forward`
  - `cpu_ref_step`: MuJoCo `mj_step` 连续 12 substeps，ctrl=`ctrl_ref[t]`
  - `mjwarp_ref_step`: `setup_env` + `step_env` 连续 12 substeps，ctrl=`ctrl_ref[t]`
- 每 substep 记录：
  - pelvis yaw / quat / qpos max diff vs ref
  - left/right foot z
  - qvel norm / qvel diff
  - ctrl max diff
  - contact count / normal force proxy if available
  - actuator force if available from MuJoCo / MJWarp tensors

### 2. 新增入口脚本

**文件**: `workspace/core4d/scripts/train/train_E070.sh`

命令：

```bash
bash workspace/core4d/scripts/train/train_E070.sh
```

输出：

- `workspace/core4d/results/E070/ref_control_parity.csv`
- `workspace/core4d/results/E070/ref_control_parity.json`
- `logs/E070/ref_control_parity.log`

## 成功标准

| 指标 | 目标 |
|------|------|
| CPU t=0.017/0.033 yaw err | < 1 deg |
| MJWarp t=0.017/0.033 yaw err | 若复现 E069, 约 12/22 deg |
| ctrl diff | 两条路径都为 0 |
| 输出完整性 | csv/json/log 三件齐 |

## Decision Tree

| 结果 | 解读 | 下一步 |
|------|------|--------|
| CPU 不漂, MJWarp 漂 | MJWarp commit path bug/mismatch | E071 修 sync/actuator/contact 中被定位的通道 |
| CPU 和 MJWarp 都漂 | E068 CPU 单步诊断口径不完整 | 扩展 CPU 连续 12 step + scene_act actuator force trace |
| MJWarp 不漂但 E069 漂 | run_mjwp 主循环在 `step_env` 外还有状态更新/同步副作用 | trace `sync_env` / receding horizon update / render-time qpos |
| 两者都不漂 | eval 的 qpos 对齐/保存 timing 有误 | 修 evaluation alignment |

## 停止条件

- 不跑新的 reward ablation。
- 不跑 trust-region / delta-clamp。
- 不继续调 `warmup_steps`。
- parity 根因未定位前，不改 CEM optimizer。
