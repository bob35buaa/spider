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
- [x] 2026-06-10 E152 完成度盘点 + eval 重算 + 写 log 192：远端 4 行（box004/box023 的 gateA/gateA_b1）已于 15:12 回收齐全；**box021_029_p2 本地 split 中断**——`gateA_outdir_full/` 仅有 `config_act.yaml` 无 trajectory、`gateA_b1` 无任何产出，`tmux ls` 已无 `E152_local_full_115333`、无 E152 进程在跑。原 `eval/full`（11:48）是 0/6 完成时的陈旧版。用 `eval_E152_*.py full --allow-missing` 重算得 `method_rows=10/delta_rows=4/missing=2`。结果(n=2)：gateA vs baseline 深穿透 `con<−5mm` mean −0.189、几何穿透 −0.037（未达 −0.10）、near-5cm −0.013、物理接触 −0.031、0 fall；gateA_b1 vs b1 near-5cm +0.029、几何穿透 −0.076、深穿透 −0.102、`success_pen_down_contact_keep` 2/2 pass。gate 健康 valid 0.77–0.91/fallback ≤0.096（box023 gateA_b1 逼近 R1 边缘）。视觉 f55 两 case 无 fall/穿模/悬浮、箱体竖直。写 `log/192_E152_axis1_hand_object_physics_gate_results.md`，TRACKER 加 E152 行（⏳ 4/6 done）。**未 commit**（box021 未完成）。GPU0 空闲，补跑：`bash workspace/core4d/scripts/run_E152_local.sh full`（local-gpu0 split=box021 两行）。
- [x] 2026-06-10 E152 box021 并行补跑完成 + 全 3-case eval：新增 `scripts/run_E152_box021_recover.sh`，用 `train ... single` 定向跑两行（不动 box004/box023 已完成结果）——本地 GPU0 跑 `box021_029_p2_gateA`、远程 RTX 6000 Ada GPU1 跑 `box021_029_p2_gateA_b1`，并行约 13min，远程已 pull。最终 eval `method_rows=12/delta_rows=6/missing=0`。结果(n=3)：gateA vs baseline 深穿透 mean −0.206（box021 −0.24/box023 −0.31）、near-5cm −0.009、几何穿透 −0.043、physC −0.038、不摔 3/3、success 0/3；gateA_b1 vs b1 near-5cm +0.019、深穿透 −0.168、success 2/3，**box021 physC +0.027 反升且深穿透 −0.298、不塌陷**。视觉 box021 f55/f85 屈身抱箱、箱竖直、无趴箱/无 fall。log 192 改完成态、TRACKER 改 ✅。准备 commit。
- [x] 2026-06-10 E153 脚手架就位 + 本地/远程 smoke 验证通过（plan `plan/161_E153_gate_threshold_sweep_plan.md`）：**Stage0 gate 解耦**——config.py 加 `cem_{safety,hand}_gate_hard_floor_m`（默认 nan→floor=min_sdf_m，E088-E152 不变）；sampling.py `_compute_sample_gate_info.add_gate` 加 hard_floor 项 `valid=(min_sdf≥floor)&(viol_pct≤max_viol)`（sampling_fast/mjwp 经 import 自动覆盖）。C1 单元验证：(a) nan==legacy `[T,F,F]`；(b) hard_floor=−0.02 时旧逻辑毙掉的样本(最深−0.015,viol0.025≤0.05)被放行→max_violation 激活。修一处 Hydra 坑：hard_floor 不在 base/override yaml struct，CLI 改用 `+cem_hand_gate_hard_floor_m=`（min_sdf/max_viol 经 E152 override 已在 struct,用 `key=`）。脚本 `scripts/E153/{_sweep_lib,run_case_box021,run_case_box004,run_case_box023}.sh`（每 case 6 组 3×2 网格,base override 复用 E152 gateA_b1,gate 参数 CLI 覆盖,hard_floor 固定 −0.02,支持 skip 续跑,训练前 snapshot）+ `run_E153_remote.sh`/`pull_E153_remote_results.sh`。**本地 smoke** box021 2 组通过（gate 2 geoms、hard_floor 写入 config_act、trajectory/video/exit0,仅 EGL 退出噪声）。**远程 smoke** box004(GPU0)+box023(GPU1) 各首条 trajectory 干净（gate 解析、obj_err pos=0.0895、仅无害 EGL `EGL_NOT_INITIALIZED` 退出噪声),验证后 kill session 释放 GPU；pull 回收 4 npz/4 mp4 round-trip OK。full 待用户运行。
