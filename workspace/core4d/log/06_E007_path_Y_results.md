# E007: Path Y 物理参数对齐 — 结果

## 状态: 部分失败（关键 trade-off 发现）

## 实验结果

| Run | 改动 | obj_pos_err | obj_z_max | pelvis_z_min | 物体抬起? |
|-----|------|------------|-----------|--------------|----------|
| E006f baseline | 默认 (sim_dt=0.0167, XML kp=500) | 0.324 | 0.610 | 0.584 | 是! |
| **E007a** | physics_dt=0.005 + decimation=4 | **0.802** | 0.313 | 0.715 | 几乎否 |
| **E007b** | E007a + Holosoma PD (kp=14-99) | **0.843** | 0.305 | 0.676 | 否 |

## 关键发现：物理参数对齐 vs 重定向能力的根本矛盾

### 发现 1: physics_dt 改变本身就破坏 E006 的接触机制

E007a 仅把 sim_dt 从 0.0167→0.02 + 增加 decimation=4，**保留 E006 所有 reward 和 XML 设置不变**，物体抬起从 0.610m 跌回 0.313m。

**根因分析**：
- E006 的成功依赖于 ctrl_steps 和接触点的精确对齐
- 改变 sim_dt 改变了 control 频率（30Hz → 50Hz），CEM 的 control horizon 不变但实际控制密度变了
- 接触发生在 1-2 步内的瞬时事件，control 频率变化导致接触时机错位

### 发现 2: Holosoma G1 PD 增益对重定向任务过弱

E007b 用 Holosoma 配置（hip_pitch kp=40, knee kp=99 等），物体完全不动（z=0.305 = 落地高度）。

**根因分析**：
- Holosoma PD 是为 RL 控制下的**局部位置跟踪**设计的（kp=14-99）
- 重定向 CEM 需要执行器**直接驱动**机器人完成"把手伸到箱子+抬起"的大幅动作
- 数量级差距：XML 默认 kp=500，Holosoma kp 平均 ~40 → **约 12 倍弱**
- 弱 PD 下机器人的"推力"不够，即使 CEM 找到正确控制信号，物理上也执行不出来

### 发现 3: 物理参数对齐是 sim2real 的伪命题（在重定向阶段）

**初始假设**: 重定向 sim 与 RL sim 物理一致 → sim2real 链条更平滑

**实际情况**: 重定向和 RL 是**两个不同的优化目标**：
- 重定向优化：找一条物理可行的关节轨迹（需要强 PD 来执行大动作）
- RL 优化：训练 policy 生成 small action correction（需要弱 PD 来允许 RL 微调）
- 两者在 PD 增益上**天然冲突**

**真正的 sim2real 链条**：
```
重定向 (kinematic ref)  → 强 PD 物理验证 → 输出 qpos 轨迹
                                              ↓
                                   作为 reference motion 喂给 RL
                                              ↓
                                  RL (弱 PD + residual action) 学小幅纠偏
                                              ↓
                                          真机部署
```

reference motion 在 RL 阶段会被 **重新执行**——RL 用自己的 PD 跑一遍，policy 输出 residual action 来弥补 PD 差异。**重定向阶段的 PD 不需要和 RL 阶段一致**。

## Claims 验证

| Claim | 结果 | 通过? |
|-------|------|------|
| C1: physics_dt 不破坏 E006 物体抬起 | obj_z_max 0.610→0.313 | **❌** |
| C2: PD 顺序映射不引入回归 | pelvis 稳定但物体不动 | 部分 |
| C3: local contact reward 改善持续性 | 未实施（前 2 步已暴露根本问题） | N/A |
| C4: Holosoma 导出兼容 | 未测试 | N/A |
| C5: obj_pos_err 不劣化 > 10% | 0.32→0.80+ (劣化 150%) | **❌** |

## 修正的工程判断

### 路径 Y 的部分仍有价值

✓ **physics_dt + decimation 基础设施** — 代码已加好，未来其他场景需要时可以打开
✓ **PD 顺序映射工具 (apply_holosoma_g1_pd)** — 可用于 RL 训练前的 motion playback 验证
✓ **配置 sentinel 设计** — 默认 OFF，不破坏任何已有实验

### 不应该做的事（修正之前的判断）

✗ 把 Holosoma PD 用于 SPIDER 重定向阶段
✗ 把 Holosoma physics_dt 强加到 SPIDER 重定向上
✗ 期望"重定向 sim = RL sim"消除 sim2real gap

### 真正应该做的事

1. **重定向阶段保留 E006 配置**（强 PD + sim_dt=0.0167）— 为了执行大动作
2. **导出阶段做 PD-aware reference smoothing**：把 SPIDER 输出的 qpos 轨迹**再次执行**一遍，用 Holosoma PD 跑物理仿真，得到 RL 训练时机器人**实际能跟到**的轨迹（去掉 PD 差异引入的高频细节）
3. **RL 训练阶段**用 HDMI 风格 reward（local contact + Lost-Contact termination）来弥补 reference 中的接触不完美

## 结果路径

| 产出 | 路径 |
|------|------|
| 计划文档 | `workspace/core4d/plan/06_E007_path_Y_physics_alignment_plan.md` |
| 配置 (E007a) | `examples/config/override/core4d_box025_holosoma_physics.yaml` |
| 配置 (E007b) | `examples/config/override/core4d_box025_holosoma_pd.yaml` |
| 代码改动 | `spider/config.py`, `spider/simulators/mjwp.py`, `spider/mujoco_utils.py` |
| 轨迹 (E007b 最后) | `example_datasets/processed/core4d/unitree_g1/humanoid_object/box025_person1/0/trajectory_mjwp.npz` |

## 可视化观察

未生成视频（pos err 显著退化，物体未抬起，不需要可视化确认）。

## 运行命令

```bash
# E007a: physics_dt + decimation only
uv run examples/run_mjwp.py +override=core4d_box025_holosoma_physics \
    task=box025_person1 data_id=0 viewer=none

# E007b: + Holosoma PD gains
uv run examples/run_mjwp.py +override=core4d_box025_holosoma_pd \
    task=box025_person1 data_id=0 viewer=none

# 复现 E006f baseline (recommended baseline going forward)
uv run examples/run_mjwp.py +override=core4d_box025_forearm \
    task=box025_person1 data_id=0 viewer=none
```

## 下一步：方向调整

放弃**纯路径 Y**思路（"sim 物理参数对齐"），改走**两阶段策略**：

**Phase A**: 保留 E006f 重定向配置，导出到 Holosoma RL 数据格式（E005 已有适配）

**Phase B**: 在 Holosoma RL 训练 stack 内：
1. 加 HDMI 风格的 local contact reward + Lost-Contact termination
2. 用 PD-aware motion playback 做 reference smoothing
3. 跑 baseline 训练 vs 含 HDMI reward 的训练对比

**Phase B 是 Layer 2 的工作，不在 spider 项目内做**。

## 复盘：为什么计划 vs 现实差距大

实验计划假设"物理参数一致是好事"，但**没有考虑到**：
- 重定向和 RL 的 PD 数量级天然不同
- 改 sim_dt 会破坏接触发生的精确时机
- Sim 一致性 ≠ Reference 质量

真正的 sim2real 链条**不要求**重定向 sim 与 RL sim 在 PD/dt 上一致——只要求 reference 轨迹本身物理合规。后者已经由 SPIDER 在 E006 提供。

**教训**: "看起来更专业"的工程对齐不一定带来收益，需要先理解为什么两阶段在工程上故意分离。
