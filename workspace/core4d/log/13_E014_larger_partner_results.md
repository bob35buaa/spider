# E014: 增大 Partner 碰撞体 — 结果

## 状态: 失败 (增大碰撞体反而恶化)

## 核心发现

1. **原始 partner 胶囊 (r=0.04m) 从未触碰箱子** — 手腕距箱面 0.175m, 胶囊仅延伸 0.12m
2. **增大胶囊/box geom 后物理更混乱** — mocap 无限刚性 → 穿透箱子 → 大力弹射
3. **带偏移的可达胶囊 (pos=-0.12) 确认了接触** — 但接触力太大太不可控

## 实验矩阵

| Run | 场景 | partner geom | intra | reward | obj_z_max | pelvis_min |
|-----|------|-------------|-------|--------|-----------|------------|
| E014-a | scene_mocap_partner_large | box 15×15×5cm | yes | r7 | 0.393 | **0.697** |
| E014-b | scene_mocap_partner_large | box 15×15×5cm | no | E011 | 0.354 | 0.091 |
| E014-c | scene_mocap_partner_reach | capsule r=0.06 + offset | yes | r7 | 0.368 | 0.141 |
| E014-d | scene_mocap_partner_reach | capsule r=0.06 + offset | no | E011 | 0.315 | 0.087 |
| **E013-ctrl** | scene_mocap_partner (原始) | capsule r=0.04 | no | E011 | **0.463** | **0.687** |
| **E013-r7** | scene_mocap_partner (原始) | capsule r=0.04 | yes | r7 | **0.477** | **0.712** |

**结论**: 增大 partner 碰撞体 → 物理更不稳定 → 所有指标劣化。原始小胶囊反而最好（因为很少产生接触力）。

## 根因分析: 为什么大 partner 更差

| 问题 | 原因 |
|------|------|
| Mocap body 刚性无穷大 | 无论施加多大力, 位置不变 → 像刚性墙壁 |
| 穿透时产生极大力 | MuJoCo solver 在穿透时产生恢复力 → 箱子被弹飞 |
| 力方向不可控 | 穿透方向取决于碰撞法线 → 可能向任意方向推 |
| CEM 无法预测 | 非线性接触力 → rollout 内混沌 → CEM 搜索失效 |

## 可视化观察

### E014-c (reach, intra, r7)

视频未单独分析 — qpos 已明确显示 pelvis 在 step 7 崩溃 (z=0.144), 箱子从未离地。
Partner capsule 可见穿入箱子侧面，但没有产生有效的向上支撑力。

## Phase 3 综合结论

经过 E013 (9 个 reward 变体 + 技术修复) 和 E014 (3 种 partner 碰撞体) 的完整探索:

### ✓ 技术贡献

1. **wp.to_torch 共享内存** 可用于 CUDA graph launch 前更新 mocap — 通用能力
2. **Intra-rollout mocap + 高 contact_rew** 偶尔打破 C1/C2 trade-off
3. **Partner capsule 距箱面 0.175m** — 原因是数据使用手腕坐标而非指尖

### ✗ 物理搬运仍不可靠

| 最佳单次 | obj_z=0.477, pelvis=0.712 (E013-r7, 40% 通过率) |
| 根本原因 | CEM 0.4s horizon 不足以规划协作搬运序列 |
| Mocap 限制 | 太小→无接触, 太大→混沌弹射 |

### → 最终推荐: 分层策略

| 层 | 方法 | 产出 |
|----|------|------|
| Layer 1 (SPIDER) | **Body-only retargeting** | 物理合规的 G1 身体运动 (pelvis_err<0.1m) |
| Layer 1 (SPIDER) | **运动学物体** | 正确的搬运轨迹 (来自参考) |
| Layer 1 (SPIDER) | **Partner 双手** | Person2 手部世界坐标 (供 RL) |
| Layer 2 (Holosoma) | **RL + interaction reward** | 物体交互 + Lost-Contact 终止 |

## 结果路径

| 产出 | 路径 |
|------|------|
| E014 配置 | `examples/config/override/core4d_box025_e014.yaml` |
| 大 partner 场景 | `scene_mocap_partner_large.xml`, `scene_mocap_partner_reach.xml` |
| body-only 最佳 | `workspace/core4d/results/E014_box025_p1_bodyonly_scene.npz` |
| body-only 视频 | `workspace/core4d/results/E014_box025_p1_bodyonly_scene.mp4` |

## 运行命令

```bash
# E014-c: reach partner
uv run examples/run_mjwp.py +override=core4d_box025_e014 \
    scene_name=scene_mocap_partner_reach \
    task=box025_person1 data_id=0 viewer=none
```
