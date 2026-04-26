# R005 Progress Log

## 2026-04-26 Session — MuJoCo Warp GPU 重写

### 环境
- 分支: feat/hdmi-reproduce-v2
- GPU: RTX 5090 (Blackwell sm_120)
- data_id=0
- mujoco_warp 3.7.0.1

### 目标
将 HDMI 后端从 CPU MuJoCo (N 个 MjData + thread pool) 重写为 MuJoCo Warp GPU 架构，对齐 MJWP 工作流。

### 计划
参考: `workspace/hdmi_reproduce/plan/R005_plan.md`

### 进度

| 时间 | 操作 | 结果 |
|------|------|------|
| 开始 | 读取 R005 计划、现有代码 | 理解完整架构差异 |
| Phase 1 | 重写 hdmi.py 为 MuJoCo Warp GPU | HDMIEnv dataclass, 复用 mjwp.py 模式 |
| Phase 1 | 修改 run_hdmi.py | 适配 HDMIEnv，从 warp 读 qpos/qvel |
| Phase 1 | 更新 hdmi.yaml | 添加 nconmax_per_env=200, njmax_per_env=600 |
| Phase 1 | 更新 test_hdmi_fast.py | N=64 快速测试 |
| Phase 2 | 第一次运行 — 名称映射错误 | rew=2.39 (低) — scene XML 用 robot/ 前缀 |
| Phase 2 | 修复: 添加 _find_in_scene() 前缀搜索 | 所有 body/joint 查找加 robot/suitcase/ 前缀 |
| Phase 2 | 第二次运行 — 通过! | rew=5.14 (HF: 5.40), 2.2ms/step |

### Phase 2 验证结果 (N=64, open-loop)

| 指标 | R005 (step 1) | R005 (step 10) | HF 参考 | R003 open-loop |
|------|--------------|----------------|---------|----------------|
| rew | **5.14** | **4.63** | 5.40 | 5.22 |
| tracking | **2.15** | **1.78** | 2.22 | 2.22 |
| obj_track | **3.00** | **2.85** | 2.87 | 3.00 |
| step perf | **2.2ms** | — | — | 5.4ms (N=8) |

### 关键改动

1. **HDMIEnv dataclass**: 取代 monkey-patched SimpleEnv
2. **ctrl = joint position targets (rad)**: 不再是 normalized actions，直接写入 MuJoCo ctrl
3. **GPU 全状态 save/load**: 复用 mjwp.py 的 _copy_state (30+ fields)
4. **零拷贝 reward**: xpos/xquat 直接从 warp 读取，不经过 numpy
5. **_find_in_scene()**: 处理 scene XML 的 robot//suitcase/ 名称前缀

### 遇到的错误

| 错误 | 尝试次数 | 解决方案 |
|------|---------|---------|
| 名称前缀不匹配 (robot/, suitcase/) | 1 | 添加 _find_in_scene() 通用前缀搜索 |
| GLFW segfault at cleanup | — | 已知问题，不影响功能 |

### Phase 3 完整 MPC 运行 (N=1024, max_iter=32, max_sim_steps=250)

| 指标 | R005 | R003 | HF 参考 | 目标 |
|------|------|------|---------|------|
| rew_mean | **2.65** | 1.93 | 5.72 | > 4.0 ❌ |
| tracking | **0.77** | 0.59 | 2.11 | > 1.5 ❌ |
| obj_track | **1.88** | 1.33 | 3.61 | > 2.0 ❌ |
| step perf | **2ms** | 90ms | — | < 5ms ✅ |
| total time | **16.4s** | ~150s | — | — ✅ |
| opt_steps | 1.00 | 1.00 | 2.5 | — ❌ |

GPU pipeline 完整可用，性能达标。但 CEM 仍然早停，reward 未达目标。

### R006 开始 (2026-04-26 续)

计划: `workspace/hdmi_reproduce/plan/R006_plan.md`

根因分析:
1. 噪声 0.2 rad 比 HF 等效噪声 (0.014~0.11 rad) 大 2~14x
2. improvement_threshold=0.02 太高 (HF mean improvement=0.011)
3. data_id 应为 1 (HF 参考数据)

改动: noise=0.05, threshold=0.005, check_steps=3, temp=0.3, data_id=1

### R006 结果

| 指标 | R006 (无decimation) | R006b (decimation=10) | HF (data_id=1) |
|------|-------|-------|------|
| rew_mean | 2.71 | 2.71 | 5.90 |
| tracking | 0.83 | 0.83 | 2.19 |
| obj_track | 1.88 | 1.88 | 3.71 |
| opt_steps | 3.58 | 3.18 | 3.12 |

opt_steps 修复成功 (1→3.18)，但 reward 没有实质改善。
加了 decimation (physics_dt=0.002, 10 sub-steps) 后指标几乎不变。
obj_track Q4=0.17 崩溃模式不变 — 非 CEM/decimation 问题。
视频待检查：机器人是否还摔倒？

### R006c: 修复 actuator gains (根因！)

Scene XML 的 PD 增益比 HDMI 弱 2~5x (hip_pitch: Kp=40 vs HDMI=200)。
在 setup_env 中用 HDMI 的 stiffness/damping 覆盖 scene XML actuator gains。

| 指标 | R006b | **R006c** | HF (data_id=1) | 目标 |
|------|-------|-----------|------|------|
| rew_mean | 2.71 | **4.13** | 5.90 | > 4.0 ✅ |
| tracking | 0.83 | **2.25** | 2.19 | > 1.5 ✅ |
| obj_track | 1.88 | **1.88** | 3.71 | > 2.5 ❌ |
| opt_steps | 3.18 | **4.84** | 3.12 | > 1.5 ✅ |

tracking 直接翻 3x！增益修复是最关键的改动。
剩余差距: obj_track Q4=0.17 (suitcase 后半程丢失)

---

## R003 历史记录 (2026-04-25)

### 关键发现
1. Warp batch stepping 在 RTX 5090 上极慢: 285ms/step (vs CPU MuJoCo 0.03ms/step)
2. Open-loop rew=5.22 (HF: 5.40) — 非常接近
3. MPC 优化器效果差: rew_mean=1.93 (HF: 5.72), opt_steps=1.0 (始终早停)
4. 原因: CPU 版本 save/load 不完整 + num_samples=8 太少
