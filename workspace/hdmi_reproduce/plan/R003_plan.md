# R003 实验计划：基于 main 分支复现 HDMI 工作流

## Context

R002 在 feat/hdmi-uv-env 分支上完成，pelvis Z 稳定 (0.789m) 但物体跟踪发散 (qpos_dist=1.81m)。

**根因**：feat/hdmi-uv-env 分支绕过了 HDMI 的完整环境 API，自己实现了简化的 qpos L2 奖励和手动 PD 控制。而 main 分支的代码通过 `env.reward_groups` 使用 HDMI 原生的指数核 body tracking 奖励，通过 `env.apply_action()` 使用 HDMI 的完整 action pipeline。

**关键 insight**：HF 参考结果是用 main 分支 + MuJoCo 后端生成的（代码中使用 `env.sim.mj_model`, `env.sim.wp_data` 等 MuJoCo 特有 API）。HDMI 默认 backend 是 "isaac"，需要显式设置 `set_backend("mujoco")`。

## Claims

| Claim | 最低证据 |
|-------|---------|
| main 分支代码能成功运行完整 HDMI 工作流 | 250 ctrl steps 无报错完成 |
| rew_mean 与 HF 参考同量级 | rew_mean 在 4.0-7.0 范围（HF: ~5.4） |
| tracking_mean 与 HF 参考接近 | tracking_mean 在 2.0-3.0（HF: ~2.55） |
| object_tracking_mean 不发散 | > 2.0 throughout（HF: ~2.85） |
| kinematic npz 与 HF 一致 | diff = 0 |
| 对比视频显示相似运动质量 | 渲染侧对比视频中机器人完成搬运动作 |

## 改动

### Phase 1: 环境准备

1. 从 origin/main 创建新分支 `feat/hdmi-reproduce-v2`
2. 确认 HDMI 仍在 `feat/spider-mujoco-v2` 分支（提供 MuJoCo backend 支持）
3. 在 main 分支的 `spider/simulators/hdmi.py` 的 `setup_env()` 中添加 `active_adaptation.set_backend("mujoco")`
4. 确认环境依赖可用（torch 2.11.0+cu130, mujoco_warp, warp, tensordict, torchrl）

### Phase 2: 适配与测试

可能需要的微调（main 分支代码可能假设某些条件）：
- 确认 `env.sim.wp_data` 在 MuJoCo backend 下可用
- 确认 `env.reward_groups` 正确初始化（需要 tracking + object_tracking 奖励组）
- 确认 `env.action_spec`, `env.num_envs`, `env.max_episode_length` 等属性存在
- 确认 `env.command_manager.t` 等状态保存/恢复可用
- output_dir 指向 `workspace/hdmi_reproduce/results/R003`

### Phase 3: 短测试验证

```bash
.venv/bin/python3 examples/run_hdmi.py \
  task=move_suitcase +data_id=1 \
  joint_noise_scale=0.2 knot_dt=0.2 ctrl_dt=0.04 horizon=0.8 \
  viewer="none" save_video=false max_sim_steps=20
```

验证：
- reward 是否为正值 (~5.0)
- opt_steps 是否合理 (1-4)
- 无运行时错误

### Phase 4: 完整运行

```bash
.venv/bin/python3 examples/run_hdmi.py \
  task=move_suitcase +data_id=1 \
  joint_noise_scale=0.2 knot_dt=0.2 ctrl_dt=0.04 horizon=0.8 \
  viewer="none" save_video=false max_sim_steps=-1
```

### Phase 5: 对比验证

1. `trajectory_kinematic.npz` vs HF 原版（逐元素 diff）
2. `trajectory_hdmi.npz` vs HF 原版（pelvis Z, rew_mean, tracking_mean, object_tracking_mean）
3. 离线渲染对比视频

## 需要修改的文件

| # | 文件 | 改动 |
|---|------|------|
| 1 | `spider/simulators/hdmi.py` | 添加 `set_backend("mujoco")` |
| 2 | `examples/run_hdmi.py` | 可能需要 output_dir 调整 |

## 成功标准

| 指标 | R002 (feat/hdmi-uv-env) | R003 目标 | HF 参考 |
|------|--------------------------|----------|---------|
| rew_mean | -0.3 | **4.0-7.0** | 5.4 |
| tracking_mean | N/A | **2.0-3.0** | 2.55 |
| object_tracking_mean | N/A | **> 2.0** | 2.85 |
| pelvis Z (mean) | 0.789m | **0.65-0.85m** | ~0.75m |
| 完整轨迹 | 5 ctrl steps | **250 ctrl steps** | 250 |

## 风险评估

| 风险 | 缓解 |
|------|------|
| HDMI MuJoCo backend API 不完整 | HDMI feat/spider-mujoco-v2 已测试过，问题不大 |
| torchrl/tensordict 版本不兼容 | 如报错则调整版本 |
| env.reward_groups 在 MuJoCo backend 下可能缺少某些数据 | 逐步调试，必要时 stub |