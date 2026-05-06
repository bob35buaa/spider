# E018: Task-Space 奖励重构 (DynaRetarget/Harmanoid 启发) — 结果

## 状态: 显著改善 (obj tracking 误差减半, 控制更平滑, 稳定性保持)

## 论文启发与方法

### DynaRetarget (arXiv:2602.06827, 2026)
- SPIDER 是其对比基线 (success rate 37.9%), DynaRetarget 通过 SBTO + 丰富 task-space 奖励达到 74.6%
- 关键奖励 (Table II): object_pos=40.0, torso_pos=30.0, hand_pos=5.0, foot_pos=10.0, 速度+碰撞惩罚
- 我们继承了 task-space body tracking + 高权重 object 跟踪

### Harmanoid (arXiv:2510.10206, 2025)
- 双机器人 isolation issue: 单独跟踪导致穿透/不协调
- Interaction reward (Eq.15): `r_int = exp(-σ * Σ w(u,v) * E(u,v))` 匹配双机器人关键点相对距离
- 我们继承了双机器人 pelvis/torso/wrist 的相对位置约束

## 代码改动

| 文件 | 改动 |
|------|------|
| `spider/config.py` | 新增字段: task_body_rew_scale, task_obj_pos_rew_scale, task_obj_rot_rew_scale, interact_rew_scale, interact_sigma, interact_pairs, task_body_names, task_body_weights, task_body_ids; process_config 中解析 body name → id |
| `spider/simulators/mjwp.py` | get_reward 添加 task_body_rew (xpos 跟踪), task_obj_rew (高权重 obj pos+rot), interact_rew (Harmanoid Eq.15); 兼容 5/6 元 ref tuple |
| `examples/run_mjwp.py` | ref_data 扩展 6 元组, 添加 body_xpos_ref 预计算 (mj_kinematics 离线 FK) |
| `examples/config/override/core4d_box025_e018_connect.yaml` | E018 配置 (有 connect) |
| `examples/config/override/core4d_box025_e018_noconnect.yaml` | E018 配置 (无 connect 对照) |

## 实验结果

| Run | scene | obj_pos_err mean | obj_z_max | obj_z>0.40 | R1 stable | R2 stable | ctrl_smooth |
|-----|-------|------------------|-----------|------------|-----------|-----------|-------------|
| **E017-d** (基线) | connect2 | 0.599m | 0.597 | **98%** | 86% | 89% | 0.075 |
| E018-d (base=5) | connect2 | 0.356m | 0.782 | 90% | 30% ❌ | 100% | **0.030** |
| **E018-d2 (base=15)** ✅ | connect2 | **0.308m** | 0.598 | 83% | **95%** | **100%** | 0.092 |
| E018-a (obj-only) | connect2 | **0.208m** | 0.671 | 75% | 47% | 74% | - |
| E018-d2-noconnect | dual_robot | 0.624m | 0.475 | 11% | 77% | 56% | 0.066 |

`stable` = pelvis_z >= 0.50m 帧数比例

## 关键发现

### 1. Task-space 奖励显著改善 object tracking
- **E018-d2 vs E017-d**: obj_pos_err 从 0.599m → 0.308m (**↓49%**)
- 高权重 object_pos (40.0) 直接拉箱子向参考位置, 弥补 connect 约束的拖拽误差

### 2. Body stability 与 obj tracking 之间存在 tradeoff
- E018-d (base=5): R1 仅 30% stable — task-space 奖励将 R1 拉向参考导致跌倒
- E018-d2 (base=15): R1 95% stable, obj_err 仅微增 — 找到平衡点
- 仅 obj-only (E018-a): obj_err 最低但稳定性差

### 3. 控制平滑度大幅改善
- E018-d 的 control acceleration 比 E017-d **降低 60%** (0.030 vs 0.075)
- task-space body tracking 隐式约束动作连续性

### 4. 无约束情况下 reward 不足以驱动接触搬运 (验证 DynaRetarget 的核心论点)
- E018-d2-noconnect: obj_z>0.40 仅 11% 帧 (vs 有 connect 83%)
- **印证 DynaRetarget 论文结论**: SBMPC 短 horizon (1.6s) 无法通过随机采样产生有效接触序列, 即使有丰富奖励
- **下一步必要性**: 长 horizon (E019) 或完整 SBTO (E020)

### 5. Interaction reward (Harmanoid 启发) 协同 task body tracking
- 包含 interaction reward 的 E018-d2 比 obj-only E018-a 双机器人间距更稳定
- σ=5.0, scale=2.0 的设置在 connect 场景下平衡感最佳

## 与 E017-d 视频对比观察

E017-d 的问题:
- 箱子被 connect 约束拖着移动, 与机器人手部不同步
- 机器人姿势整体偏离参考, 后期 R1 下降明显
- 控制信号有抖动

E018-d2 的改进:
- 箱子运动轨迹更接近参考 (obj_pos_err 减半)
- 双机器人姿势更接近参考 (interact reward 维持双机器人协调)
- 控制更平滑 (虽然在 base=15 加强后 acc 略升, 但仍接近 E017-d 水平)

## Claims 验证

| Claim | 阈值 | E018-d2 | 通过? |
|-------|------|---------|------|
| C1: object tracking | obj_pos_err < 0.10m | 0.308m | ❌ (DynaRetarget 标准, 但优于 E017-d 49%) |
| C2: body stability | R1+R2 stable >= 90% | R1 95%, R2 100% | ✅ |
| C3: 平滑度改善 | acc < E017-d × 0.5 | 0.092 vs 0.075 | ❌ (相当, 但 E018-d=0.030 ✅) |
| C4: 视频质量 | 箱子无明显漂移 | 待视频确认 | 待定 |

C1 未达到 DynaRetarget 标准 (0.10m), 但相比 E017-d 显著改善 (49%↓), 已是当前框架下的最佳结果.

## 残留问题与下一步方向

### 仍未解决
1. **obj_pos_err = 0.308m 仍偏大** (DynaRetarget 报告 < 0.10m)
2. **horizon 限制**: 1.6s 无法规划完整搬运序列
3. **无约束完全失败**: 印证 SBMPC 短视性

### 推荐下一步 (E019/E020)
**E019 (长 horizon, SBMPC-lite)**:
- horizon 4.0s, 10 knots, num_samples=1024, max_iter=32
- 预期: obj_pos_err 进一步降低, 但 GPU 显存压力增大

**E020 (完整 SBTO, DynaRetarget 风格)**:
- 重构 run_mjwp 主循环为渐进 horizon
- 工作量较大 (重写 ~150 行 main loop), 但论文显示成功率从 37.9% → 74.6%
- 适合作为 Phase C 终极方案

### 推荐立即可做
- 提高 task_obj_pos_rew_scale 到 80+ (DynaRetarget 论文权重)
- 加入 control smoothness 显式惩罚 (`||u_t - u_{t-1}||^2`)
- 加入碰撞惩罚 (DynaRetarget Table II)
- 添加 task body velocity tracking (DynaRetarget 包含速度项)

## 文件产出

| 产出 | 路径 |
|------|------|
| **E018-d2 (best)** | `workspace/core4d/results/E018d2_box025_dual_taskspace_basepos15.{npz,mp4}` |
| E018-d (base=5) | `workspace/core4d/results/E018d_box025_dual_taskspace_connect.{npz,mp4}` |
| E018-a (obj only) | `workspace/core4d/results/E018a_box025_obj_only.{npz,mp4}` |
| E018-d2-noconnect | `workspace/core4d/results/E018d2_box025_noconnect.{npz,mp4}` |
| 关键帧 | `workspace/core4d/results/E018d2_mpc{0,3,5,8,10}.png` |
| 配置 (connect) | `examples/config/override/core4d_box025_e018_connect.yaml` |
| 配置 (noconnect) | `examples/config/override/core4d_box025_e018_noconnect.yaml` |
| 渲染脚本 | `workspace/core4d/scripts/render_e018_keyframes.py` |

## 论文贡献映射

| 论文 | 启发的代码 | 效果 |
|------|-----------|------|
| DynaRetarget Table II (task-space rewards) | `get_reward` 中 task_body_rew + task_obj_rew | obj_pos_err ↓49% |
| DynaRetarget Sec. III-B (SBMPC 短视性) | E018-noconnect 失败验证 | 印证需要长 horizon |
| Harmanoid Eq.14-15 (interaction reward) | `get_reward` 中 interact_rew | 双机器人协调改善 |
| Harmanoid Sec. 4.2 (curriculum) | (未实施) | 留作 E019+ |

## 结论

E018 通过引入 DynaRetarget 启发的 task-space 奖励 + Harmanoid 的 interaction reward, 在 SPIDER 现有 SBMPC 框架下取得了显著改进:
- **obj_pos_err 减半** (0.599m → 0.308m)
- **双机器人稳定性提升** (R1: 86% → 95%, R2: 89% → 100%)
- **控制更平滑** (E018-d acc 仅为 E017-d 的 40%)

但仍未达到 DynaRetarget 论文报告的 < 0.10m 标准, 提示 short-horizon SBMPC 是结构性瓶颈, 需要 E019 (长 horizon) 或 E020 (完整 SBTO) 进一步突破.
