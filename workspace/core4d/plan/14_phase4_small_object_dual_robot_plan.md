# Phase 4 实验计划: 小物体验证 + 双机器人重定向

## Context

Phase 1-3 (E001-E014) 在 box025 (61×61×89cm) 上完成了完整的重定向探索:
- **结论**: CEM-MPC 无法可靠物理搬运 box025 (太大, 需双人协作)
- **最佳结果**: E013-r7 (obj_z=0.477, pelvis=0.712, 40% C1+C2 通过率)
- **技术贡献**: wp.to_torch 共享内存 intra-rollout mocap, partner 碰撞体分析

Phase 4 有两个方向:
1. **E015**: 用更小物体 (bucket005, max=0.41m) 验证改进是否有效 — 如果对 G1 可行，说明改进有价值
2. **E016**: 双机器人交替优化 — box025 双人协作的正确建模

## E015: Bucket005 小物体验证

### 动机

box025 (0.61m) 超出 G1 臂展 (0.5m), 单人物理搬运不可解。
bucket005 (max 0.41m < 0.5m) 理论上单人可抓取。
→ 用 bucket005 验证 Phase 2-3 的技术改进是否对可行物体有效。

### 物体特征

| 属性 | bucket005 | box025 (对比) |
|------|-----------|--------------|
| 尺寸 | 0.295 × 0.410 × 0.286 m | 0.611 × 0.610 × 0.893 m |
| 最大维度 | 0.41m | 0.89m |
| G1 可抓? | 是 (< 0.5m) | 否 (> 0.5m) |
| 序列长度 | 160帧 (5.33s @ 30fps) | 124帧 (4.13s) |
| 初始 obj_z | 0.126m | 0.310m |

### 数据管线

1. 生成 scene.xml: 基于 box025 scene.xml 模板, 替换:
   - 物体 mesh → bucket005_m.obj
   - 碰撞几何 → 圆柱近似 (cylinder, r≈0.20, half-h≈0.14)
   - 物体初始位置 → 从 holosoma qpos 提取 (0, 36:39)
   - 物体质量 → 合理估计 (2-3kg, bucket 比 box 轻)
   - 物体惯性 → 圆柱公式
2. 转换 trajectory: `core4d.py --task bucket005_person1`
3. 验证 nq=43, 视频检查

### Claims (成功标准)

| Claim | 阈值 | 说明 |
|-------|------|------|
| C1: 管线通过 | 无 crash | scene.xml + 转换 + MJWP 运行 |
| C2: obj 离地 | obj_z_max > ref * 0.5 | 物体有明显抬升 |
| C3: 身体稳定 | pelvis_min ≥ 0.50m | G1 不摔倒 |
| C4: 视频确认 | 手触碰物体 | 视觉验证物理接触 |
| C5: 改进有效 | baseline < best config | Phase 2/3 技术改进在小物体上也有效 |

### 实验矩阵

| Run | 配置 | 场景 | 说明 |
|-----|------|------|------|
| E015-a | baseline (无引导) | scene.xml | 纯 body retargeting |
| E015-b | E012 best (kinobj) | scene_connect.xml | 运动学物体 |
| E015-c | E013-r7 best | scene_mocap_partner.xml | Intra-rollout + 高 contact |

### 产出

- `workspace/core4d/results/E015_bucket005_p1_baseline.npz/mp4`
- `workspace/core4d/results/E015_bucket005_p1_kinobj.npz/mp4`
- `workspace/core4d/results/E015_bucket005_p1_best.npz/mp4`

---

## E016: 双机器人重定向 (Box025)

### 动机

box025 是双人协作任务。E001-E014 仅优化 person1, person2 用 mocap 近似。
正确做法: 两个 G1 机器人同时优化, 共享一个物体。

### 架构设计

#### Scene XML (scene_dual_robot.xml)

```
worldbody
  ├── robot1 (pelvis_1): G1 freejoint + 29 joints
  ├── robot2 (pelvis_2): G1 freejoint + 29 joints  
  └── object: freejoint (shared)
  
nq = 7 (robot1 base) + 29 (robot1 joints) + 7 (robot2 base) + 29 (robot2 joints) + 7 (object) = 79
nu = 29 (robot1) + 29 (robot2) = 58
```

#### Embodiment Type

新增 `embodiment_type = "dual_humanoid_object"`:
- qpos 布局: [robot1_base(7) + robot1_joints(29) + robot2_base(7) + robot2_joints(29) + object(7)]
- ctrl 布局: [robot1_ctrl(29) + robot2_ctrl(29)]
- Gibbs 交替: 固定 robot2 优化 robot1, 再反过来

#### 代码改动评估

| 文件 | 改动 | 复杂度 |
|------|------|--------|
| `spider/config.py` | 新增 dual_humanoid_object type | 低 |
| `examples/run_mjwp.py` | 扩展 Gibbs sampling 为双 humanoid | 中 |
| `spider/simulators/mjwp.py` | `_diff_qpos`/`get_terminate`/reward 适配 79-DOF | 高 |
| Scene XML | 双 G1 + 共享 object | 中 |
| `spider/process_datasets/core4d.py` | 合并 person1+person2 轨迹 | 中 |

#### 参考轨迹合并

从两个 holosoma 文件:
- person1: qpos(124, 43) → 取 robot1_base(7) + robot1_joints(29)
- person2: qpos(124, 43) → 取 robot2_base(7) + robot2_joints(29)
- object: 取 person1 的 object freejoint (两人数据中 object 相同)
→ merged qpos(124, 79)

#### Gibbs 交替优化

```
for each MPC step:
    1. noise_scale[robot2_ids] = 0  →  optimize robot1
    2. noise_scale[robot1_ids] = 0  →  optimize robot2
    3. restore full noise_scale
```

与现有 bimanual Gibbs 完全同构, 只是 index split 不同。

### Claims

| Claim | 阈值 | 说明 |
|-------|------|------|
| C1: 场景加载 | nq=79, nu=58 | 双机器人 scene 正确 |
| C2: 两机器人都稳定 | pelvis1/2_min ≥ 0.50m | 都不摔倒 |
| C3: 物体改善 | obj_z_max > E013-r7 (0.477) | 双人比单人+mocap 更好 |
| C4: 视频确认 | 两机器人协作抬箱 | 视觉验证 |

### 分支策略

```
feat/hdmi-reproduce-v2 (当前, E001-E014 + E015)
  └── feat/dual-robot-retarget (E016, 从当前分支切出)
```

---

## 实施顺序

```
E015 小物体验证 (当前分支)
  ├── 1. 生成 bucket005 scene.xml 
  ├── 2. 转换 holosoma → SPIDER trajectory
  ├── 3. 运行 baseline retargeting
  ├── 4. 运行 kinobj + mocap_partner 配置
  ├── 5. 可视化分析
  └── 6. 记录日志

E016 双机器人 (新分支)
  ├── 1. 创建分支 feat/dual-robot-retarget
  ├── 2. 创建 scene_dual_robot.xml
  ├── 3. 合并 person1+person2 参考轨迹
  ├── 4. 适配 config.py + mjwp.py
  ├── 5. 扩展 Gibbs sampling
  ├── 6. 运行 + 可视化
  └── 7. 记录日志
```
