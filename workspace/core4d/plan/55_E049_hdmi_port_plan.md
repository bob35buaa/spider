# Phase 14: E049 HDMI 优化移植 + HDMI 泛化验证

## Context

E048 证明 **算法是瓶颈**: HDMI 在 box023 上 MPKPE=0.7cm/Contact=93%/Stability=100%, 而 E041c 仅 38% Stability (摔倒)。

HDMI 的三个关键优化:
1. PD 增益: 从 Isaac Lab 导入 → MJWP 已有 `apply_holosoma_pd` 配置
2. 手腕阻尼: `dof_damping=5.0` → 需 ~5 行新代码
3. 手腕噪声归零: → MJWP 已有 `zero_noise_joint_keywords` 配置

另发现 eval_comprehensive.py 有 hardcoded half_ext_map bug, 导致新 case 评估错误。

## 实验矩阵

| 实验 | 内容 | GPU |
|------|------|-----|
| Fix | eval_comprehensive.py half_ext_map → 动态读取 scene XML | 无 |
| Fix | mjwp.py + config.py 添加 wrist_dof_damping | 无 |
| E049a | box023 E041c + HDMI三优化 (PD+damping+noise) | 远程 GPU0 |
| E049b | box025 E041c + HDMI三优化 | 远程 GPU0 |
| E049c | bucket010 E041c + HDMI三优化 | 远程 GPU1 |
| E049d | desk005 E041c + HDMI三优化 | 远程 GPU1 |
| E049e | box025 HDMI workflow (跨算法验证) | 本地 |

## Claims

- C1: HDMI 三优化使 box023 Stability 从 38% → ≥90%
- C2: HDMI 三优化使 box025 Contact 从 54% → ≥60%
- C3: HDMI workflow 在 box025 上也优于 E041c

## 实现

### 1. 修复 eval_comprehensive.py (前置)

替换 hardcoded `half_ext_map` → 从 scene XML collision geom 动态读取:
```python
# 找到 object body 的 collision geom size
for gi in range(model.ngeom):
    if "object_collision" in (mujoco.mj_id2name(...) or ""):
        obj_half = model.geom_size[gi].copy()
```

### 2. 添加 wrist_dof_damping (前置)

**config.py**: 添加 `apply_wrist_dof_damping: bool = False` 和 `wrist_dof_damping: float = 5.0`

**mjwp.py setup_mj_model**: 在 apply_holosoma_pd 后添加:
```python
if getattr(config, "apply_wrist_dof_damping", False):
    for ji in range(model_cpu.njnt):
        jname = mujoco.mj_id2name(model_cpu, mjtObj.mjOBJ_JOINT, ji)
        if jname and "wrist" in jname:
            model_cpu.dof_damping[model_cpu.jnt_dofadr[ji]] = config.wrist_dof_damping
```

### 3. 新 config: core4d_e049.yaml

基于 E041c, 添加三个 HDMI 优化:
```yaml
apply_holosoma_pd: true
apply_wrist_dof_damping: true
wrist_dof_damping: 5.0
zero_noise_joint_keywords: ["wrist_roll", "wrist_pitch", "wrist_yaw"]
```

### 4. 远程并行 (4 runs)

```
GPU0: box023 E049a → box025 E049b    (~12min)
GPU1: bucket010 E049c → desk005 E049d (~12min)
```

### 5. 本地 HDMI box025

convert_core4d_to_hdmi.py --case box025_person1 → run_hdmi.py

## 关键文件

| 文件 | 改动 |
|------|------|
| `spider/config.py` | +apply_wrist_dof_damping, +wrist_dof_damping |
| `spider/simulators/mjwp.py` setup_mj_model | +wrist dof_damping 逻辑 (~5行) |
| `workspace/core4d/scripts/eval/eval_comprehensive.py` | 修复 half_ext_map |
| `examples/config/override/core4d_e049.yaml` | 新建 — E041c + 3 HDMI 优化 |
| `workspace/core4d/scripts/run_E049_remote.sh` | 新建 |
