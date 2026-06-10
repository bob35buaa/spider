# Core4D Experiment Progress

## Archive Index

| Archive | Experiments | Period |
|---------|-------------|--------|
| [E098-E108 Data Foundation](progress_archive/E098_E108_data_foundation.md) | E098-E108 | 2026-05-31 — 2026-06-02 |
| [E109-E124 Contact Recovery](progress_archive/E109_E124_contact_recovery.md) | E109-E124 | 2026-06-02 — 2026-06-03 |
| [E125-E145 RL Bridge & Nonbox](progress_archive/E125_E145_rl_bridge_and_nonbox.md) | E125-E145 | 2026-06-03 — 2026-06-05 |
| [E147-E152 Collision Geometry](progress_archive/E147_E152_collision_geometry.md) | E147-E152 | 2026-06-08 — 2026-06-10 |

Full original backup: [progress_archive/E098_E152_full_backup.md](progress_archive/E098_E152_full_backup.md)

---

## Active: E152 — Hand Gate Physics (2026-06-10)

- [x] 恢复 E152 计划：当前目标是 `workspace/core4d/plan/160_E152_axis1_hand_object_physics_gate_plan.md`。用户指出 `box004` 三个方法接触箱子前手碰地，需要在 E152 evaluator 显式报告手-地接触/穿透。
- [x] 代码定位：CEM gate 采样逻辑在 `spider/optimizers/sampling.py` 和 `spider/optimizers/sampling_fast.py`；现有 `cem_safety_gate` 只支持一组全局 `min_sdf/max_violation_pct`，E152 手 gate 若要独立阈值，需要新增 `cem_hand_gate_*` 信息并在 sampling 中与 body gate 取交集。
- [x] 实现策略：新增默认关闭的 `cem_hand_gate_enabled`、`geom_names/ids`、`min_sdf_m`、`max_violation_pct`；`mjwp.py` 用 E151 mesh SDF helper 计算 `lh/rh` 到 `object_collision` 的 hand gate 指标，同时保留/合并旧 `cem_gate_*` 诊断。
- [x] 已实现核心 gate 增量：`config.py` 增加 `cem_hand_gate_*` 字段和 geom resolver；`mjwp.py` 输出 aggregate/body/hand 三套 gate 指标；`sampling.py` 与 `sampling_fast.py` 支持 body/hand 独立阈值并输出 `cem_hand_gate_valid_frac`、`cem_body_gate_valid_frac`。保留 terminal hard gate 合并到 body gate，避免旧 terminal carry gate 语义被新 helper 绕过。
- [x] 已补手-地/手-物体物理指标：`eval_E147_rubber_hand_collision.py` 现在输出 hand-object `con.dist` mean/min/`frac_lt_neg5mm`/frame frac，以及 hand-floor physics contact、hand-floor `con.dist` 分布和 deep-frame frac；`eval_E151_route_b_hand_surface_contact.py` 已把这些字段纳入 summary/delta。相关 `py_compile` 通过。
- [x] E152 固定入口已落地并通过 preflight：新增 manifest builder、hand gate preflight、train/local/remote/pull/eval 脚本。默认 hand gate 阈值为 `cem_hand_gate_min_sdf_m=-0.010`、`cem_hand_gate_max_violation_pct=0.05`；manifest 为 12 rows，复用 6 rows，新跑 6 rows，split 为 local/remote0/remote1 各 2 rows。
- [x] 按用户反馈复评 E151 `box004_083_p2` 手-地指标：四个方法均有短时手-地接触/穿地。baseline `hand_floor_penetration_frac=3.81%`、min `-1.60cm`；b2_sup `6.67%`、min `-2.31cm`；b2_tip `3.81%`、min `-1.40cm`；b1_mesh `3.81%`、min `-2.32cm`。说明视频里"接触箱子前手碰地"是真实物理/几何问题，不是渲染错觉；E152 evaluator 已把该指标纳入。
- [x] E152 smoke：本地单跑 `E152_box004_083_p2_gateA` smoke 通过，`CEM hand gate: 2 geoms resolved`，`cem_gate_valid_frac≈0.943`、fallback `0`。smoke 低迭代接触不足，不作为效果结论，只作为 gate plumbing/健康度检查。
- [ ] E152 full 运行中：本地 tmux `E152_local_full_115333` 已完成 `box021 gateA` 并进入 `box021 gateA_b1`；远端 tmux `E152_full_retry_115537` 正在跑 `box004 gateA` 与 `box023 gateA`。`box021 gateA` gate 健康度：`cem_gate_valid_frac mean=0.838`、fallback `0`、`cem_hand_gate_min_sdf_min=-0.0064m`，未触发 full gate 塌缩。
