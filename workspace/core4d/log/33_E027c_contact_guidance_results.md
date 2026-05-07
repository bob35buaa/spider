# E027c: Contact Guidance (OMOMO Paper Approach) for CORE4D

## 状态: 失败 — 机器人走路但物体不跟随

## 核心思路

E027b 用 anchored 轨迹 + 固定 PD override 实现了原地搬运姿态。用户指出 SPIDER 论文的 OMOMO/HDMI 任务能做 locomotion + manipulation（边走边搬），不应该需要 anchor。

E027c 尝试用 SPIDER 论文的 **contact_guidance** 方案（`guidance_decay_ratio=0.8`, PD gains 逐迭代衰减）替代 E027b 的固定 PD override。

## 运行命令

```bash
# Step 1: 生成 trajectory_kinematic_act.npz (freejoint → 6DOF euler, 含 ctrl object 通道)
# (inline script, 见 progress.md)

# Step 2: 运行
MUJOCO_GL=egl uv run examples/run_mjwp.py +override=core4d_e027c task=desk005_person2
```

## 结果

| Metric | 值 |
|--------|---|
| pelvis XY displacement | **1.607m** (机器人在走路!) |
| pelvis stable | **100%** |
| object displacement | **~0m** (物体没动) |

## 结论
1. SPIDER 论文的 locomotion+manipulation 成功案例（move_suitcase, move_largebox）用的是 HDMI simulator，不是 MJWP
2. MJWP 的 contact_guidance 配置 (humanoid_object_act.yaml) 用的是 dataset_name: omomo，但实际上 OMOMO 没有 MJWP 格式的场景文件——这个配置可能从未被成功运行过
3. CORE4D 用 MJWP + contact_guidance 时，gains 更新应该是有效的，但 CEM 短 horizon + 行走位移太大导致优化不够
4. 要让 CORE4D 也能走路+搬运，正确方向是：用 HDMI simulator 而非 MJWP，或者把 HDMI 的场景适配机制移植到 MJWP

## 关键发现

### 1. SPIDER 论文 loco-manipulation 用的是 HDMI simulator，不是 MJWP

- HDMI move_suitcase (R013) 的成功视频来自 `simulator: hdmi` (spider/simulators/hdmi.py)
- HDMI simulator 有专门的 `_make_contact_guidance_model()` 动态生成 scene_act
- MJWP 的 `humanoid_object_act.yaml` 写着 `dataset_name: omomo`，但 OMOMO 没有 MJWP 格式场景——这个配置可能从未在 MJWP 上成功运行过

### 2. HDMI vs CORE4D 对比

| | HDMI move_suitcase | CORE4D desk005 |
|--|---|---|
| simulator | **hdmi** | mjwp |
| pelvis XY | 0.54m | 1.59m |
| object mass | 2.0 kg | 5.0 kg |
| 单人/双人 | 单人 | 双人协作 |
| 结果 | 成功 (R013) | 机器人走但物体不动 |

### 3. contact_guidance 在 MJWP 下的问题

- CEM ctrl 初始值来自 `ctrl_ref`——需要 object actuator 通道包含正确的 ref qpos 值
- 原始 `trajectory_kinematic_act.npz` 的 ctrl object 通道为 0 → CEM 从 0 开始优化 → 物体不动
- 修复 ctrl 后仍然不动——说明 `load_env_params` 的 gain 更新可能在 MJWP 的 CUDA graph 中不生效

### 4. 下一步方向

1. **使用 HDMI simulator** 替代 MJWP 跑 CORE4D——需要适配 HDMI 的场景格式
2. **验证 MJWP load_env_params gain 更新**——添加 debug 验证 gains 是否在 Warp step 中被使用
3. **增大 contact_guidance 初始 gains**——CORE4D 物体更重，init_pos_actuator_gain=10 可能不够

## 配置

| 文件 | 路径 |
|------|------|
| Config | `examples/config/override/core4d_e027c.yaml` |
| 视频 | `workspace/core4d/results/E027c_desk005_fixed.mp4` |
| 帧截图 | `workspace/core4d/results/E027b_frames/desk005_e027c_fixed_t*.png` |
| HDMI R013 对比 | `workspace/core4d/results/E027b_frames/hdmi_R013_t*.png` |
