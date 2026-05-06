# CORE4D 双机器人协作搬运重定向 — 实验总结

> 基于 SPIDER 框架，将 CORE4D 双人协作搬箱数据重定向到双 G1 机器人；2026-04-30 至 2026-05-06，共 18 个实验 (E001–E018)。

---

## 1. 目标与问题

将 CORE4D `box025` 双人对夹搬箱序列 (124 帧 / 30 fps) 重定向为可被 Holosoma RL 直接消费的双 G1 物理轨迹，要求**箱子持续离地 + 双机器人 pelvis 稳定 ≥ 0.5 m**。

---

## 2. 实验路线 (Phase 0 → Phase 4)

| Phase | 实验 | 关键结论 |
|-------|------|----------|
| **P0 数据管线** | E001 | holosoma → SPIDER NPZ + scene XML 全通 |
| **P1 单人物理搬运** | E002–E009 | 全部失败 (obj_z ≤ 0.31 m)。**根因**: G1 单人臂展 0.5 m < box 长 0.61 m，几何不可解；非力大小问题 |
| **P2 单人 + Mocap Partner** | E010–E012 | E010 运动学可行；E011 partner 提供支撑但架构受限；**E012 导出 hybrid 轨迹 (pelvis_err = 0.083 m) 用作 RL 备选** |
| **P3 修复 + 高方差** | E013–E014 | Intra-rollout mocap 技术修复成功但 CEM 高方差；partner capsule gap=0.175 m 导致 mocap 太刚性失效 |
| **P4 双机器人** | E015–E018 | 路线突破 |

---

## 3. 关键突破 (E016 → E017 → E018)

| Run | 方法 | obj_pos_err | obj_z_max | obj_z>0.40 帧占比 | R1 / R2 stable |
|-----|------|-------------|-----------|-------------------|----------------|
| E016-a | 双 G1 Gibbs CEM | — | 0.488 (92% ref) | 瞬间 | 100% / 100% (推/翻) |
| E017-d | + **Soft 2-connect 焊接约束** | 0.599 m | 0.597 (112% ref) | **98% (198 连续帧)** | 86% / 89% |
| **E018-d2** ✅ | + **Task-space 奖励 (DynaRetarget) + Interaction reward (Harmanoid)** | **0.308 m (↓49%)** | 0.598 | 83% | **95% / 100%** |

**E018-d2 = 最终最佳配置** (`base_pos=15`, soft 2-connect, task body+obj+interact)。

---

## 4. 核心洞察

1. **Connect 约束是 SBMPC 的物理触手**：CEM 短 horizon (1.6 s) 无法通过随机采样产生有效接触力 — 用 weld/connect 把手"焊"到箱面，CEM 只优化身体姿态，箱子自动跟随。 验证：无 connect 时 obj_z>0.40 仅占 11%。
2. **Task-space 奖励 (obj_pos=40, torso=30, hand=5, foot=10) 显著提升 tracking** — obj 误差减半，控制平滑度↑60%，无需重写 optimizer。
3. **双机器人需要 Interaction reward (Harmanoid Eq.15)** — 单独跟踪导致穿透；relative pose 约束维持双臂协作几何。
4. **Stability ↔ Tracking tradeoff**：obj-only 拉爆 pelvis；`base_pos=15` 是平衡点。
5. **必须 qpos+视频双验证**：E006/E008 教训 — reward 字段不可信。

---

## 5. 产出文件

- 最佳轨迹：`results/E018d2_box025_dual_taskspace_basepos15.{npz,mp4}`
- 关键帧对比：`results/E018d2_mpc{0,3,5,8,10}.png`
- 早期 hybrid baseline：`results/box025_person1_spider_holosoma_w_partner.npz`
- 代码主分支：`feat/dual-robot-retarget` (config.py / mjwp.py / run_mjwp.py 三处主要改动)

---

## 6. 下一步

- **E019** 长 horizon SBMPC (2 s → 4 s) — 验证短视是否仍是瓶颈
- **E020** 完整 SBTO (DynaRetarget pipeline) — 当前 SBMPC 上限验证
- 将 E018-d2 喂给 Holosoma RL，做物理交互层 finetune
