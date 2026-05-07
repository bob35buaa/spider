# E031: 双机器人 Connect 约束泛化 — 结果

## 状态: 失败 — 4 case 均不可用

## 运行命令

```bash
# Step 1: 生成双机器人数据 (轨迹 + scene)
python workspace/core4d/scripts/convert/generate_dual_data.py
python workspace/core4d/scripts/convert/generate_dual_connect_all.py

# Step 2: 修复 Gibbs sampling (examples/run_mjwp.py line 540)
# 添加 "dual_humanoid_object" 到 gibbs_enabled 条件

# Step 3: 运行 4 case
for TASK in box025_person1 bucket010_person1 chair022_person1 desk005_person2; do
    MUJOCO_GL=egl python examples/run_mjwp.py +override=core4d_e031_dual task=$TASK
done
```

## 数量结果

| Case | obj_pos_err | obj_quat_err | R1 stable% | R2 stable% | 时间 |
|------|-----------|------------|-----------|-----------|------|
| box025 | 0.442 | 0.941 | 73% | 100% | 262s |
| bucket010 | 0.466 | 1.321 | 73% | 73% | 258s |
| chair022 | 0.618 | 1.061 | 36% | 27% | 257s |
| desk005 | 0.262 | 0.655 | 100% | 100% | 222s |

对比 E018-d2 (仅 box025, 无 Gibbs): pos=0.308, R1=95%, R2=100%

## 视频观察 — 全部失败

### box025
- **f0 (0%)**: 两机器人站箱两侧, sim 姿态基本匹配 ref ✓
- **f2 (50%)**: **灾难** — 两机器人被箱子拖着走, 双方严重倾斜, R1 几乎摔倒, 箱子翻转悬空

### bucket010
- **f0 (0%)**: sim 中只看到一个机器人 + 桶, R2 不可见(可能在桶后面)
- **f2 (50%)**: 桶**横倒**, 两机器人弯腰蹲下被 connect 约束拽着

### chair022
- **f2 (50%)**: **完全崩溃** — 两机器人趴地, 椅子翻倒悬空

### desk005 (数据最好)
- **f0 (0%)**: 两机器人站桌两侧, sim 与 ref 几乎完美匹配 ✓
- **f2 (50%)**: **桌子翻倒**, 两机器人被拽着倾斜, 混乱

## 根因分析

### 问题 1: 参考轨迹包含行走 (walking)

所有 case 的 ref 都包含行走位移 (1-1.6m)。CEM 短 horizon (1.6s) + 58 维空间 → 无法跟上行走:
- 机器人原地不动, 但 connect 约束把手绑在跟随 ref 运动的物体上
- 物体沿 ref 移动 → connect 拽手 → 拽倒机器人

**E018 成功的原因**: box025 的 E018 也有行走, 但 E018 **没有 Gibbs**, 且场景数据是之前精心调试的。
更重要的是 E018 用的 connect 锚点是**手动校准的** (从 contact_pos 精确计算), 而 E031 用的是自动生成的新数据。

### 问题 2: 自动生成的 dual 轨迹质量差

`generate_dual_data.py` 简单拼接 person1 + person2 的单人 retarget 结果:
- 两个 person 的 retarget 是**独立做的**, 没有保证双人间的协调性
- Object qpos 从 person1 取, 但 person2 的手位置可能与 person1 的物体状态不一致
- desk005 只有 person2, person1 用了"偏移站立"伪数据 — 完全不合理

### 问题 3: Gibbs 交替优化在 connect 约束下有害

Gibbs 先优化 R1 (固定 R2), 再优化 R2 (固定 R1):
- R1 被优化到"拉物体向 R1 方向"的姿态
- 然后 R2 被优化到"拉物体向 R2 方向"的姿态
- 两者**对抗** → 物体被拉扯翻转 → 双方被拽倒

E018 **没有 Gibbs** 时 R1=95%/R2=100% stable, E031 **有 Gibbs** 时 R1=73%/R2=100%。
Gibbs 在 connect 约束场景下反而有害。

### 问题 4: Connect 锚点位置不准确

新 case 的锚点通过 `compute_connect_anchors` 自动计算, 但:
- bucket010 是圆柱, connect 在圆面上 — 旋转自由度约束不自然
- chair022 的"接触点"在椅背/椅腿上, 很难用 connect 建模
- desk005 的锚点在桌面边缘 — 不是自然的抬桌位置

### 根本问题: SPIDER CEM 不适合双人协作搬运

1. **58 维空间太大**: 即使 4096 samples, 对 58 维联合优化不够
2. **短 horizon (1.6s)**: 无法规划协作搬运的时序动作
3. **行走 + 搬运**: CEM 无法同时处理 locomotion + manipulation
4. **Connect 约束是 hack**: 将手焊接到物体表面绕过了接触发现问题, 但引入了新的稳定性问题(约束力拽倒机器人)

## 结论

> **双机器人 CEM 在 CORE4D 协作搬运场景上不可行。**
>
> 即使用了 connect 约束 + task-space rewards + Gibbs, 在包含行走的参考轨迹上:
> - CEM 短 horizon 无法跟上行走位移
> - Connect 约束力拽倒机器人 (而非辅助)
> - 58 维空间采样效率极低
>
> E018 在 box025 上的"成功"是高度参数敏感的 (特定锚点 + 特定 base_pos_rew 权重),
> 不能泛化到其他 case。

## 对比论文 OMOMO 成功案例

SPIDER 在 OMOMO 上成功因为:
- **单人搬运**: 29 维 (非 58 维)
- **原地操作**: OMOMO 是原地拿起放下 (无行走)
- **小物体**: 单手可搬, 不需要协作

CORE4D 的结构性差异:
- 双人协作 → 需要两个 agent 协调
- 包含行走 → locomotion + manipulation
- 大/重物体 → 需要双人力学配合

## 下一步建议

1. **放弃 CEM 双机器人方案**: 结构性不可行
2. **回到单机器人 + mocap partner**: E013 方案, 一个 robot 物理 + 另一个 kinematic (mocap)
3. **考虑 SBTO (长 horizon)**: DynaRetarget 风格的全序列优化 (但工作量大)
4. **接受 SPIDER 的定位**: SPIDER 适合 dexterous hand + 单人原地操作, 不适合 humanoid loco-manipulation collaboration

## 结果路径

| 产出 | 路径 |
|------|------|
| 视频 | `workspace/core4d/results/E031_dual/{case}.mp4` |
| NPZ | `workspace/core4d/results/E031_dual/{case}.npz` |
| 帧截图 | `workspace/core4d/results/E031_dual/{case}_f{0,1,2,3}.png` |
| Dual 数据生成 | `workspace/core4d/scripts/convert/generate_dual_data.py` |
| Connect 生成 | `workspace/core4d/scripts/convert/generate_dual_connect_all.py` |
| 配置 | `examples/config/override/core4d_e031_dual.yaml` |
| Plan | `workspace/core4d/plan/29_E031_dual_robot_generalization_plan.md` |

## 代码改动

| 文件 | 改动 |
|------|------|
| `examples/run_mjwp.py` | Gibbs 条件修复 (line 540): 添加 `dual_humanoid_object` |
| `workspace/core4d/scripts/convert/generate_dual_data.py` | 新: 生成双机器人轨迹 |
| `workspace/core4d/scripts/convert/generate_dual_connect_all.py` | 新: 泛化 connect 锚点 |
