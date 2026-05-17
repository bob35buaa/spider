# CORE4D 协作重定向探索实验跟踪器

分支：`exp/core4d-collab-retarget`

新方向编号从 `E001` 开始。`workspace/core4d` 的 E081 仅作为 baseline 引用，不继承为本工作区实验编号。

## 实验总览

| Run | 日期 | Phase | 描述 | 状态 |
|-----|------|-------|------|------|
| E001 | 2026-05-17 | Exploration | 基于 core4d E081 baseline，审计 object/freejoint 控制口径、E081 eval+可视化验收口径，并深读 SPIDER / DynaRetarget / 双人交互控制论文。结论：active E079-E081 是 `scene_act` actuator-guided object，不是真 freejoint 物理搬运；E002 应优先做 freejoint-legobj 对照 | ✅ 详见 log 01 |
| E002 | 2026-05-17 | Freejoint Audit | **真 freejoint + leg-object collision 对照**: 新建 `box025_person2_freejoint_legobj` / `box023_person2_freejoint_legobj`，关闭 `scene_act/contact_guidance/object actuator`，并修复/防护 E070 同类 qpos-as-ctrl 隐患；full CEM 显示 main/guard 都无法通过真实接触跟踪 object ref | ✅ 详见 log 02 |
| E003 | 2026-05-17 | Physics Feasibility | **true-freejoint 物理参数可行性 sweep**: 在 E002 基础上改 object mass/friction。`box025_m1_f4` 从 E002 `0.703m` 改善到 `0.400m` case-window obj mean，但仍失败且 floor contact `76.9%`；`box023` guard 未恢复并出现 `10-11%` 腿/箱干涉/摔倒 | ✅ 详见 log 03 |
| E004 | 2026-05-17 | Virtual Partner | **true-freejoint 大规模虚拟协作者支持 sweep**: COM-level gravity/spring/hold-contact 9 组完成；全部保持 `nu=29`、`nq_obj=7`、no object actuator，但 main `box025` 全部失败，最好仍约 `0.66/1.29m` control、spring 约 `0.77/1.55m`；guard 2/3 stable，但无 transport | ✅ 详见 log 04 |
| E005 | 2026-05-17 | Partner Force Fix | **partner-force ref_dt 显式化 + support-site 几何对照**: `_apply_partner_force` 不再硬编码 30Hz；box025 显式用 task 30Hz，box023 显式用 50Hz default，再比较 object-local off-COM support-site 等效 wrench | ✅ main full 完成，guard support-site 远程卡住，详见 log 05 |
| E006 | 2026-05-18 | COLA Proxy | **COLA-style support-body proxy 迁移**: 保持 object true-freejoint 与 `nu=29`，不新增 freejoint body；在 MJWarp step 内用独立 support proxy trajectory + soft connector wrench 传力，并记录 partner effort / connector gap。7/7 parity ok，但 main 无 useful，best `box025_ypos_k20` 仍 `0.704/1.410m`、floor `91.3%` | ✅ 详见 log 06 |
| E007 | 2026-05-18 | Timebase + E081 Eval | **support proxy 时间基准修正 + E081 对齐验收**: 修正 `support_proxy_ref_dt<=0` 默认使用插值后 `sim_dt`；第一轮复跑 dt-corrected support proxy，并把 eval 从 E005 support-site 改为与 E081 `box025/box023_p2_legobj` 对比 | 🟡 Plan/implementation 中 |

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
| E004 full | main: `g05` obj `0.660/1.288m`, springs/hold variants all around `0.768-0.771/1.549-1.577m`; `s40_hc` lowers floor contact to `68.2%` but raises leg intf to `15.6%`; guard `s10/s20` stable but object mean `0.81m`, `s10_hc` falls | Post-hoc 发现 partner-force ref index 硬编码 30Hz；box025 task 本身是 30Hz，main 结论保留；box023 guard 需 E005 显式 50Hz 复查 |
| E005 full | 7 个 full 结果：main 最好 `box025_com_s20` obj `0.678/1.325m`, floor `71.7%`; support-site `yneg/ypos` 均无 useful proxy 且 floor `94-97%`; guard `box023_com_s10` obj `0.821/1.448m`, stable but no transport | ref_dt 修正必要但不是主因；直接 object COM/support-site 外力路线到达上限，下一步转向 COLA-style dynamic support-body proxy |
| E006 full | 7 个 full 结果：`num_freejoint_parity_ok=7`、`num_support_proxy_metrics_present=7`、`num_main_useful_proxy=0`、`num_main_beats_E005_support_site_proxy=0`；main best `box025_ypos_k20` obj `0.704/1.410m`、floor `91.3%`；guard `xneg` tracking/floor 好但摔倒，`xpos` stable but no transport | support proxy 工程接入成功，但 force-only proxy 仍不能形成双端搬运；下一步转向 XML-level contact pad/soft constraint 或强化 robot-side 支撑 |
| E007 plan | E081 baseline main `box025_p2_legobj`: obj `0.143/0.271m`、hand `89.0%`、leg intf `7.5%`、floor `59.5%`；guard `box023_p2_legobj`: obj `0.164/0.317m`、floor `34.7%` | E007 不再以 E005 为验收基线；先验证 dt-corrected proxy 能否完整走参考平移，再决定是否进入 contact-pad / robot-side reward |

## Plans

- E001: `workspace/core4d_collab_retarget/plan/01_E001_baseline_literature_code_audit_plan.md`
- E002: `workspace/core4d_collab_retarget/plan/02_E002_freejoint_legobj_control_audit_plan.md`
- E003: `workspace/core4d_collab_retarget/plan/03_E003_freejoint_physics_feasibility_sweep_plan.md`
- E004: `workspace/core4d_collab_retarget/plan/04_E004_freejoint_virtual_partner_support_plan.md`
- E005: `workspace/core4d_collab_retarget/plan/05_E005_partner_force_timing_support_site_plan.md`
- E006: `workspace/core4d_collab_retarget/plan/06_E006_cola_support_body_proxy_plan.md`
- E007: `workspace/core4d_collab_retarget/plan/07_E007_support_proxy_timebase_e081_plan.md`

## Logs

- E001: `workspace/core4d_collab_retarget/log/01_E001_baseline_literature_code_audit_results.md`
- E002: `workspace/core4d_collab_retarget/log/02_E002_freejoint_legobj_control_audit_results.md`
- E003: `workspace/core4d_collab_retarget/log/03_E003_freejoint_physics_feasibility_sweep_results.md`
- E004: `workspace/core4d_collab_retarget/log/04_E004_freejoint_virtual_partner_support_results.md`
- E005: `workspace/core4d_collab_retarget/log/05_E005_partner_force_timing_support_site_results.md`
- E006: `workspace/core4d_collab_retarget/log/06_E006_cola_support_body_proxy_results.md`

## Git

- Baseline merge: `feat/dual-robot-retarget -> main` 已 fast-forward 到 `ae8b342`。
- 当前实验分支：`exp/core4d-collab-retarget`。
