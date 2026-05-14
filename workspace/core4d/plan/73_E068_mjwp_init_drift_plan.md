# E068 实验计划: MJWP init drift 诊断与最小修复

## Context

E065-E067 连续三轮 ablation 分别排查了 task_obj reward form、object actuator stiffness、body partition，但 box023 的 pre-contact lunge / handstand 仍未消失。最新诊断 `workspace/core4d/log/87_R4_HDMI_diagnosis_init_pose_bug.md` 将问题从 reward/dynamics/solver 调参收敛到更底层的初始化差异：

- HDMI box023 能在 pre-contact 阶段保持双脚接地，B1 max foot z = 0.066m。
- MJWP box023 在 sim 第一帧 pelvis yaw 已偏离 ref 约 22 deg，之后继续漂移。
- 把 ref qpos[0] 直接 assign 到 `scene_act.xml` 后 `mj_forward` 可对齐 ref，说明 ref 本身和 XML 坐标系不是根因。
- 代码审查发现 `spider/simulators/mjwp.py::setup_env()` 在写入 `qpos_ref[0] / qvel_ref[0] / ctrl_ref[0]` 后立刻 `mujoco.mj_step()`，随后才 `mjwarp.put_data()`；`examples/run_mjwp.py` 的 viewer-side init 也同样 `mj_step()`。

### 根因假设

当前最强假设是：MJWP 初始化阶段的 `mj_step()` 已经把环境推进到 post-step 状态，强 actuator / contact / object control 在第 0 帧就引入 yaw drift；后续 CEM 看到的是偏离 ref 的初态，于是通过 lunge / handstand / superman 等姿态代偿。

### 关键 insight

先不要继续调 reward。必须先把 init state 隔离出来，比较 `mj_forward`、`mj_step`、当前 `setup_env()`、以及去掉 init step 后的 Warp state。若 `mj_forward` 版本能消除 t=0 yaw drift，再做最小修复验证。

## Claims

| Claim | 最低证据 |
|-------|---------|
| C1: 当前 MJWP init drift 可复现 | 诊断脚本输出 current setup_env / mj_step 路径的 pelvis yaw drift 接近 log 87 的 20 deg 级别，或至少显著大于 `mj_forward` 路径 |
| C2: `mj_forward` init 与 ref 对齐 | CPU `qpos_ref[0] + mj_forward` pelvis yaw error < 2 deg，quat angle error < 2 deg |
| C3: drift 来源在 init step / put_data 前后可定位 | 输出 `cpu_forward`、`cpu_step`、`warp_after_setup` 三条路径的 qpos/yaw/foot/object 差异表，并保存 CSV/JSON |
| C4: 最小修复候选明确 | 如果 `cpu_step` 或 current setup_env 复现 drift，则提出 `mjwp_init_mode=forward` 或等价修复；如果不复现，则转向 put_data / first committed step trace |
| C5: 修复不破坏 box025 baseline | 若进入修复验证，box025 pelvis_min >= 0.60m，且视频无明显新退化 |
| C6: box023 pre-contact lunge 显著改善 | 若进入修复验证，box023 B1 max foot z [0,2s] <= 0.10m，pelvis_min_intent >= 0.50m，无 handstand/superman |

## 改动

### 1. 固化 E068 init drift 诊断脚本

**文件**: `workspace/core4d/scripts/debug/diagnose_E068_init_drift.py`

功能：
- 加载 `core4d_e062_box023` 配置与 box023 ref。
- 构建 `scene_act.xml` model。
- 比较以下 init 路径：
  - `cpu_forward`: assign qpos/qvel/ctrl 后 `mj_forward`
  - `cpu_step`: assign qpos/qvel/ctrl 后 `mj_step`
  - `warp_after_setup`: 调用当前 `setup_env()` 后读取 `env.data_wp.qpos[0]`
  - 可选 `warp_forward_manual`: 手动 `mj_forward` 后 `mjwarp.put_data`
- 输出 pelvis position/quaternion/euler/yaw error、foot z、object pose、qpos max diff。
- 保存到 `workspace/core4d/results/E068/init_drift_diagnosis.{json,csv}`。

### 2. 若 C1-C4 指向 init step，添加最小修复开关

**文件**: `spider/config.py`

新增：

```python
mjwp_init_mode: str = "step"  # "step" legacy; "forward" keeps exact ref pose before put_data
```

**文件**: `spider/simulators/mjwp.py`

在 `setup_env()` 中：

```python
if config.mjwp_init_mode == "forward":
    mujoco.mj_forward(model_cpu, data_cpu)
else:
    mujoco.mj_step(model_cpu, data_cpu)
```

**文件**: `examples/run_mjwp.py`

viewer-side `mj_data` 初始化使用同一 config 逻辑，避免渲染/保存 ref 对比混淆。

### 3. 验证脚本

**文件**: `workspace/core4d/scripts/train/train_E068.sh`

第一阶段只跑诊断：

```bash
bash workspace/core4d/scripts/train/train_E068.sh diagnose
```

第二阶段在修复后跑 box023 + box025：

```bash
bash workspace/core4d/scripts/train/train_E068.sh verify 0 1
```

## 需要修改的文件

| # | 文件 | 改动 |
|---|------|------|
| 1 | `workspace/core4d/scripts/debug/diagnose_E068_init_drift.py` | 新建 init drift 诊断脚本 |
| 2 | `workspace/core4d/scripts/train/train_E068.sh` | 新建诊断/验证入口 |
| 3 | `examples/config/override/core4d_e068_box023.yaml` | 若修复成立，继承 E062 box023 并启用 init forward |
| 4 | `examples/config/override/core4d_e068_box025.yaml` | 若修复成立，继承 E062 box025 并启用 init forward |
| 5 | `spider/config.py` | 若诊断确认，新增 `mjwp_init_mode` |
| 6 | `spider/simulators/mjwp.py` | 若诊断确认，支持 forward init |
| 7 | `examples/run_mjwp.py` | 若诊断确认，同步 viewer-side init |
| 8 | `workspace/core4d/log/88_E068_mjwp_init_drift_results.md` | 记录诊断、验证、可视化观察和结论 |
| 9 | `workspace/core4d/progress.md` | 持续记录本次执行状态 |

## 训练命令

诊断阶段不训练：

```bash
bash workspace/core4d/scripts/train/train_E068.sh diagnose
```

若诊断确认 init step 是 drift 来源，再运行修复验证：

```bash
bash workspace/core4d/scripts/train/train_E068.sh verify 0 1
```

## 成功标准

| 指标 | 前次 (R4/log87) | 本次目标 (E068) |
|------|----------------|----------------|
| Init yaw error: CPU forward | ~0 deg 已 inline 验证 | **< 2 deg** |
| Init yaw error: current setup_env / CPU step | MJWP sim t=0 约 +22 deg | **复现或定位到 first-step drift 来源** |
| box023 B1 max foot z [0,2s] | E062 0.481m, E065-A 0.305m, E067-N 1.30m | **<= 0.10m** |
| box023 pelvis_min_intent | E063 0.19m | **>= 0.50m** |
| box025 regression | E062/E041c sphere baseline可站 | **pelvis_min >= 0.60m, 视频无明显新退化** |
| 可视化 | log 87 静态诊断 | **保存 box023/box025 mp4 + keyframes，实际观察不可留空** |

## 停止条件

- 如果 `cpu_forward` 与 `cpu_step` 都不产生 drift，且 `warp_after_setup` 也不 drift，则 E068 不改 init，转向 first committed `step_env()` trace。
- 如果 `mjwp_init_mode=forward` 修复 init 但 box023 仍 lunge，则停止 reward 调参，下一轮专门分析 first committed control / contact force / actuator force trace。
- 如果 box025 regression 明显，修复不得作为默认行为，只作为 case-local override 继续诊断。
