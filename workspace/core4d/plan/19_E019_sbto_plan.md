# E019: SBTO (DynaRetarget 核心算法) 实验计划

## 实施状态: ✅ 代码完成, 运行中

## 已完成
- [x] `spider/config.py`: 新增 `use_sbto`, `sbto_sigma_min`, `sbto_max_iter_per_knot`, `sbto_knot_dt`
- [x] `examples/run_mjwp.py`: 新增 `run_sbto()` 函数; `main()` 中 `if config.use_sbto:` 分支
- [x] `examples/config/override/core4d_box025_e019.yaml`: E019 配置
- [x] MPC 回归测试通过
- [x] SBTO 功能测试通过 (64 samples, 2 knots)
- [ ] E019-a 全量运行 (进行中...)

## 实验矩阵

| Run | 配置 | 状态 |
|-----|------|------|
| E019-a | SBTO, knot_dt=0.25, connect2, base_pos=15, pos_rew=5 | 运行中 |
| E019-b | E019-a + task_obj_pos=40, task_body_rew=1 | 待定 |
| E019-c | E019-a 但 scene_dual_robot (无约束) | 待定 |

## 预期运行时间
- total_steps=248 (4.1s 轨迹)
- sbto_knot_dt=0.25 → total_knots=16
- 每个 knot: ~30 iter × 1024 samples × (active_steps) rollout
- 总计估计: 15-30 min

## 评测方法
完成后用 `workspace/core4d/scripts/eval/eval_metrics.py` 全面对比
