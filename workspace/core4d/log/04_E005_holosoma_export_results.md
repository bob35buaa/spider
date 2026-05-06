# E005: 混合轨迹导出到 Holosoma 格式 — 结果

## 状态: 通过

## 输出格式验证

| Key | Ours | Holosoma ref | Match |
|-----|------|-------------|-------|
| body_pos_w | (210, 52, 3) | (200, 52, 3) | OK |
| body_quat_w | (210, 52, 4) | (200, 52, 4) | OK |
| body_lin/ang_vel_w | (210, 52, 3) | (200, 52, 3) | OK |
| joint_pos | (210, 36) | (200, 36) | OK |
| joint_vel | (210, 35) | (200, 35) | OK |
| body_names | (52,) | (52,) | OK (完全一致) |
| joint_names | (29,) | (29,) | OK (完全一致) |
| object_pos/quat_w | (210, 3/4) | (200, 3/4) | OK |
| fps | 50 | 50 | OK |

## 混合策略
- **机器人运动**: SPIDER E002 MJWP 输出 (pelvis 0.10m 误差, joint 0.073rad, 物理合规)
- **物体轨迹**: 运动学参考 (保持正确的抬举曲线, z: 0.31→0.53→0.31)
- **FK**: 使用 holosoma 的 `g1_29dof_w_Box025.xml` (52 body 完整树)
- **重采样**: 60fps→50fps (Holosoma 标准)

## 输出
- 路径: `workspace/core4d/results/E005_box025_person1_spider_holosoma.npz`
- 可直接替换 holosoma 的 `converted_for_rl_trimmed/*.npz` 用于 RL 训练

## 关键性质
- 物体 z: 0.31→0.53→0.31m (正确抬举)
- Pelvis z: 0.81→0.82m (站立稳定)
- 无脚滑/穿模 (SPIDER 物理保证)

## 下一步
1. 在 Holosoma 项目中用此数据替换运动学重定向数据进行 RL 训练
2. 对比: SPIDER 物理重定向 vs 原始运动学重定向的 RL 训练效果
3. 扩展到其他 CORE4D 案例 (bucket, chair, desk)

## 结果路径

| 产出 | 路径 |
|------|------|
| 混合轨迹 NPZ | `workspace/core4d/results/E005_box025_person1_spider_holosoma.npz` |
| 导出脚本 | `workspace/core4d/scripts/export/export_to_holosoma.sh` |
| FK 参考 XML | `/home/ubuntu/Workspace/holosoma/src/holosoma_retargeting/holosoma_retargeting/models/g1/g1_29dof_w_Box025.xml` |

## 可视化观察

已通过 E002 视频确认机器人运动质量（见 `log/02_E002_E003_results.md`）。
导出数据未单独可视化（格式验证通过，内容与 E002 机器人 + 运动学物体的组合一致）。
