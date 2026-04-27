# R011 计划：深度分析 R008 vs R010b 退步 + 优化

## Context

R008 是目前最佳结果 (rew=5.63, tracking=2.52, obj_track=3.10)，但有已知 bug：suitcase slide joint 初始位置偏移 0.4m。R010b 修复了这个 bug 后，指标反而退步 (rew=5.37, tracking=2.20, robot 在 t=4s 摔倒)。

需要理解退步根因并修复，使 **正确位置 + 正确代码** 下达到 R008 或更好的指标。

### 深度根因分析

通过对比 R008 与 R010b 的 NPZ 轨迹数据和代码变更，发现 **三个独立回归因素**：

#### 根因 1: contact guidance 增益 bug — pos/rot 不区分 (最严重)

`run_hdmi.py:125-132` 只向 `env_params` 传入标量 `kp = pos_kp * scale`，而 `load_env_params()` (`hdmi.py:1371`) 将同一个 kp 值应用到所有 6 个 object actuator (3 pos + 3 rot)。

**问题**: pos actuator 需要 kp=10.0, rot actuator 需要 kp=0.3。当前所有 6 个都用 kp=10.0 — **rot 增益高了 33x**!

**对比 `run_mjwp.py:297-318`** (正确实现)：它按 actuator 名称分别设置 pos/rot gain，生成 per-actuator array `[10, 10, 10, 0.3, 0.3, 0.3]`。

这导致 suitcase rotation 被过度约束，CEM 无法自然地旋转 suitcase。R008 侥幸因为 0.4m 偏移让几何上更容易，掩盖了这个 bug。

#### 根因 2: 腕关节噪声归零 (R009 引入, 中等影响)

`run_hdmi.py:109-117` 将 6 个 wrist actuator 的 CEM noise 设为 0。

**问题**: `get_reward()` 的 upper body tracking 包含 `wrist_yaw_link` body (`hdmi.py:1017-1021`)。wrist noise=0 意味着 CEM 无法优化这些 body 的位置。轨迹数据显示 R010b 的 Q4 tracking 从 R008 的 2.27 暴跌到 0.97。

**折中方案**: 不完全归零，而是降低到 joint_noise_scale 的 30% (0.015 rad ≈ 0.86°)。保留一定搜索能力，同时避免 R008 的反关节问题。

#### 根因 3: 正确 suitcase 位置物理上更难 (不可避免)

R008 的 suitcase 错误地近了 0.4m (pelvis→suitcase 0.95m vs 正确 1.06m)。在 t=3.5s 机器人弯腰最低点时，R008 因为 suitcase 近可以借力恢复，R010b 够不到。

**应对**: 增大 pos guidance 增益 (kp=10→20) 并减慢衰减 (ratio=0.8→0.85)，让 guidance 更有力地拉动 suitcase 到机器人手边。

### R008 → R010b 之间的所有代码变更

| 变更 | 影响 | R011 处理 |
|------|------|----------|
| suitcase slide offset 修复 (body_default_pos) | 必须保留 (正确性) | 保留 |
| scipy euler 替换 _quat_to_euler | 必须保留 (正确性) | 保留 |
| wrist noise 完全归零 | 退步因素 | 改为降低到 30% |
| output_dir 修复 | 必须保留 | 保留 |
| freejoint ref 渲染 | 必须保留 | 保留 |
| tqdm 进度条 | 不影响指标 | 保留 |

## Claims

1. **per-actuator gain 修复后 obj_track > 3.0** (R008=3.10, R010b=3.17, HF=3.71)
2. **wrist noise 30% 恢复后 tracking > 2.3** (R008=2.52, R010b=2.20)
3. **增强 pos guidance (kp=20, decay=0.85) 后 robot 在 t=4s 站立**
4. **rew_mean > 5.5** (所有修复后综合效果)

## 改动

| # | 文件 | 改动 | 说明 |
|---|------|------|------|
| 1 | `examples/run_hdmi.py` L120-133 | per-actuator kp/kd array 替换标量 | 对齐 run_mjwp.py 的 contact guidance 实现 |
| 2 | `spider/simulators/hdmi.py` L1357-1389 | `load_env_params` 接收 per-actuator array | 支持 pos/rot 不同 gain |
| 3 | `examples/run_hdmi.py` L109-117 | wrist noise *= 0.3 替换 *= 0.0 | 恢复部分搜索能力 |
| 4 | `examples/config/hdmi.yaml` | init_pos_actuator_gain=20, guidance_decay_ratio=0.85 | 增强 guidance 力度 |

### Fix 1: per-actuator gain schedule (run_hdmi.py)

对齐 `run_mjwp.py:297-318` 的实现，按 actuator 名称区分 pos/rot:

```python
# Build per-actuator base gains (pos vs rot)
base_kp = []
base_kd = []
for aid in obj_act_ids:
    aname = mujoco.mj_id2name(env.model_cpu, mujoco.mjtObj.mjOBJ_ACTUATOR, aid)
    if aname and "_rot_" in aname:
        base_kp.append(rot_kp)
        base_kd.append(rot_kd)
    else:
        base_kp.append(pos_kp)
        base_kd.append(pos_kd)
base_kp = np.array(base_kp, dtype=np.float32)
base_kd = np.array(base_kd, dtype=np.float32)

for i in range(config.max_num_iterations):
    scale = decay ** i
    if i == config.max_num_iterations - 1:
        scale = 0.0
    env_params = [{"kp": base_kp * scale, "kd": base_kd * scale}] * config.num_dr
    env_params_list.append(env_params)
```

### Fix 2: load_env_params 支持 per-actuator array (hdmi.py)

当前 `hdmi.py:1371-1374` 用标量赋值。改为支持 array:

```python
kp = env_param.get("kp", 0.0)
kd = env_param.get("kd", 0.0)
kp_arr = np.atleast_1d(np.asarray(kp, dtype=np.float32))
kd_arr = np.atleast_1d(np.asarray(kd, dtype=np.float32))

# If scalar, broadcast to all actuators
if kp_arr.size == 1:
    kp_arr = np.full(len(object_actuator_ids), kp_arr.item(), dtype=np.float32)
if kd_arr.size == 1:
    kd_arr = np.full(len(object_actuator_ids), kd_arr.item(), dtype=np.float32)

for i, aid in enumerate(object_actuator_ids):
    env.model_cpu.actuator_gainprm[aid, 0] = kp_arr[i]
    env.model_cpu.actuator_biasprm[aid, 1] = -kp_arr[i]
    env.model_cpu.actuator_biasprm[aid, 2] = -kd_arr[i]
```

### Fix 3: wrist noise 降低而非归零 (run_hdmi.py)

```python
# Reduce (not zero) noise on wrist joints — they affect wrist_yaw body tracking
wrist_keywords = ["wrist_roll", "wrist_pitch", "wrist_yaw"]
for ai in range(env.model_cpu.nu):
    aname = mujoco.mj_id2name(env.model_cpu, mujoco.mjtObj.mjOBJ_ACTUATOR, ai)
    if aname and any(w in aname for w in wrist_keywords):
        config.noise_scale[:, :, ai] *= 0.3  # 30% of normal noise
        loguru.logger.info(f"Reduced noise for wrist actuator {ai}: {aname}")
```

### Fix 4: hdmi.yaml 增强 guidance

```yaml
init_pos_actuator_gain: 20.0   # was 10.0 — stronger pull for correct distance
init_pos_actuator_bias: 20.0   # was 10.0
guidance_decay_ratio: 0.85     # was 0.8 — slower decay, longer guidance
```

## 实验方案

### R011a: 仅修复 per-actuator gain bug (隔离验证)

只应用 Fix 1 + Fix 2，其他参数不变。验证 obj_track 是否改善。

```bash
uv run examples/run_hdmi.py data_id=1 max_sim_steps=250 \
    output_dir=workspace/hdmi_reproduce/results/R011a
```

### R011b: Fix 1 + Fix 2 + Fix 3 (wrist noise 30%)

在 R011a 基础上加 wrist noise 恢复。验证 tracking 是否改善。

### R011c: 全部修复 (Fix 1-4)

全部改动 + guidance 增强。最终结果。

如果 R011a 已经接近目标可以跳过 b/c；如果 R011a 不够再逐步加。

## 成功标准

| 指标 | R008 (bug) | R010b (正确) | R011 目标 | HF 参考 |
|------|-----------|-------------|----------|--------|
| rew_mean | 5.63 | 5.37 | **> 5.5** | 5.90 |
| tracking | 2.52 | 2.20 | **> 2.3** | 2.19 |
| obj_track | 3.10 | 3.17 | **> 3.0** | 3.71 |
| t=4s pelvis_z | 0.80m (站) | 0.60m (摔) | **> 0.7m** | — |
| runtime | ~2000s | ~2000s | < 2000s | — |

## 验证步骤

1. 实现 Fix 1+2 → 运行 R011a (250 steps) → 检查 obj_track, 抽帧 t=3s,4s
2. 如 obj_track < 3.0 → 加 Fix 3+4 → 运行 R011b/c → 比较
3. ffmpeg 抽帧验证 t=2s, 3s, 4s: robot 是否站立、suitcase 是否跟随
4. 更新 EXPERIMENT_TRACKER.md 和 progress.md

## 关键文件

- `examples/run_hdmi.py` (L89-138: contact guidance setup)
- `spider/simulators/hdmi.py` (L1357-1389: load_env_params)
- `examples/config/hdmi.yaml` (guidance params)
- `workspace/hdmi_reproduce/EXPERIMENT_TRACKER.md`
- `workspace/hdmi_reproduce/progress.md`
