# CORE4D 协作重定向探索实验跟踪器

分支：`exp/core4d-collab-retarget`

新方向编号从 `E001` 开始。`workspace/core4d` 的 E081 仅作为 baseline 引用，不继承为本工作区实验编号。

## 实验总览

| Run | 日期 | Phase | 描述 | 状态 |
|-----|------|-------|------|------|
| E001 | 2026-05-17 | Exploration | 基于 core4d E081 baseline，审计 object/freejoint 控制口径、E081 eval+可视化验收口径，并深读 SPIDER / DynaRetarget / 双人交互控制论文。结论：active E079-E081 是 `scene_act` actuator-guided object，不是真 freejoint 物理搬运；E002 应优先做 freejoint-legobj 对照 | ✅ 详见 log 01 |
| E002 | 2026-05-17 | Freejoint Audit | **真 freejoint + leg-object collision 对照**: 新建 `box025_person2_freejoint_legobj` / `box023_person2_freejoint_legobj`，关闭 `scene_act/contact_guidance/object actuator`，并修复/防护 E070 同类 qpos-as-ctrl 隐患；full CEM 显示 main/guard 都无法通过真实接触跟踪 object ref | ✅ 详见 log 02 |
| E003 | 2026-05-17 | Physics Feasibility | **true-freejoint 物理参数可行性 sweep**: 在 E002 基础上改 object mass/friction。`box025_m1_f4` 从 E002 `0.703m` 改善到 `0.400m` case-window obj mean，但仍失败且 floor contact `76.9%`；`box023` guard 未恢复并出现 `10-11%` 腿/箱干涉/摔倒 | ✅ 详见 log 03 |
| E004 | 2026-05-17 | Virtual Partner | **true-freejoint 虚拟协作者支持计划**: 使用现有 `partner_force_*` 外力路径在 freejoint object 上施加重力补偿/弹簧/旋转支持，测试是否能表达 sim2real 中真实人类支撑另一侧的因素，同时保持 `nu=29`、no object actuator | 📋 计划中 |

## Baseline

| 来源 | 关键结论 | 后续验收要求 |
|------|----------|--------------|
| `workspace/core4d` E081 | 腿/脚-箱碰撞是必要物理修正；`box025_p2` 腿/箱 interference `28.9% -> 7.5%`，但 object lift/floor-contact 未改善；`box023_p2` guard 基本不破坏 | 新实验至少对齐 E081 的 eval 指标与视频/关键帧观察；work 的定义是大部分量化指标和可视化效果 `>= E081 baseline`，或提供 baseline 不具备的新功能 |

## 关键指标演进

| Run | 主要指标 | 结论 |
|-----|----------|------|
| E001 | E081 main `box025_p2_legobj`: obj mean/max `0.143/0.271m`, hand contact `89.0%`, leg intf `7.5%`, floor contact `59.5%`, bottom mean `-0.075m`; guard `box023_p2_legobj` strict proxy True | E081 当前成绩依赖 `scene_act` object actuator guidance；下一步需真 freejoint audit |
| E002 setup | E002 freejoint model `nq/nv/nu=43/41/29`，`contact_guidance=false`，object actuator ids empty；E081 对照 raw `ctrl=29` -> scene_act `ctrl=35` | 当前代码不会在 E002 freejoint 路径复现 E070 qpos-as-ctrl bug；新增维度断言防止后续静默错跑 |
| E002 full | main obj mean/max `0.703/1.356m`, hand contact `89.6%`, leg intf `0.0%`, floor contact `85.5%`; guard obj mean/max `0.830/1.488m`, hand contact `72.0%`, floor contact `88.7%` | 当前单机器人 CEM reward 能保持手部接触，但不能靠真实接触运输 object；E081 的 guard 成功也依赖 actuator-guided object |
| E003 full | main best `box025_m1_f4`: obj mean/max `0.400/0.795m`, hand contact `93.6%`, leg intf `1.7%`, floor contact `76.9%`; guard best remains failed: `box023_m1` obj `0.831/1.512m`, `box023_m1_f4` obj `0.947/1.678m`, leg intf `10-11%` | 被动 mass/friction 不是主要瓶颈；需要 explicit virtual collaborator/support/contact constraint 或 dual-agent support |

## Plans

- E001: `workspace/core4d_collab_retarget/plan/01_E001_baseline_literature_code_audit_plan.md`
- E002: `workspace/core4d_collab_retarget/plan/02_E002_freejoint_legobj_control_audit_plan.md`
- E003: `workspace/core4d_collab_retarget/plan/03_E003_freejoint_physics_feasibility_sweep_plan.md`
- E004: `workspace/core4d_collab_retarget/plan/04_E004_freejoint_virtual_partner_support_plan.md`

## Logs

- E001: `workspace/core4d_collab_retarget/log/01_E001_baseline_literature_code_audit_results.md`
- E002: `workspace/core4d_collab_retarget/log/02_E002_freejoint_legobj_control_audit_results.md`
- E003: `workspace/core4d_collab_retarget/log/03_E003_freejoint_physics_feasibility_sweep_results.md`

## Git

- Baseline merge: `feat/dual-robot-retarget -> main` 已 fast-forward 到 `ae8b342`。
- 当前实验分支：`exp/core4d-collab-retarget`。
