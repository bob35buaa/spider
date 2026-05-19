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
| E007 | 2026-05-18 | Timebase + E081 Eval | **support proxy 时间基准修正 + E081 对齐验收**: 修正 `support_proxy_ref_dt<=0` 默认使用插值后 `sim_dt`；首个 full 证明 dt 修正生效，但 `support_proxy_max_xy_speed=0.8` 成为新限速瓶颈，proxy 只走 `69%`、object 只走 `29%` | ⚠️ partial，详见 log 07 |
| E008 | 2026-05-18 | Proxy Speed Gate | **support proxy 高/不限速与 E081 搬运验收**: 5/5 main proxy gate 通过；best `ypos_k20_vmax2` 达到 object xy ratio `0.724`、rot `13.6deg`、obj `0.363/0.684m`，明显优于 E007 但仍未达 E081 transport gate | ✅ 详见 log 08 |
| E009 | 2026-05-18 | Contact Closure | **E008 best + robot-side hold-contact 闭环**: 6 个 full 结果完成；4/4 main proxy gate 通过，但 `num_main_reaches_E081_transport_proxy=0`、`num_main_improves_E008_best=0`。best-ish `hc05` obj `0.395/0.726m`、hand `76.9%`、floor `62.4%`、rot `18.2deg`，仍差于 E008/E081；强 HC 引入旋转/腿干涉 | ✅ 详见 log 09 |
| E010 | 2026-05-18 | Contact Pad | **mocap contact pad 虚拟协作者支持**: 7 个 full 结果完成；5/5 main proxy gate 通过且 true-freejoint parity ok，但 `num_main_reaches_E081_transport_proxy=0`、`num_main_improves_E008_best=0`。contact pad 保持较高手接触，却未把物体带离高 floor-contact/低 xy transport 失败区 | ✅ 详见 log 10 |
| E011 | 2026-05-18 | Soft Tether Diagnostic | **soft object tether 诊断 E081 所需外部 coupling**: 9 个 full 结果完成；COM spring 能压低 E006 的旋转捷径并恢复平移，best `box025_com_k100` 达 xy `0.905`、rot `3.5deg`，但 obj 仍 `0.340/0.673m`，7/7 main 都未达 E081 transport gate | ✅ 详见 log 11 |
| E012 | 2026-05-18 | Dual Point Closure | **dual-point partner pose closure 诊断**: 8 个 full 结果完成；8/8 true-freejoint parity ok，但 main `0/6` 达到 E081 transport，`pose_closure_helped=0`，best-by-error `k150` obj `0.306/0.606m` 伴随 rot `97.8deg`；2/2 guard 不稳定 | ✅ 详见 log 12 |
| E013 | 2026-05-19 | Object Oracle | **true-freejoint object oracle**: 2 条 full 完成；2/2 oracle config ok，main obj `0.011/0.038m`、xy `1.000`、rot `2.0deg`，guard obj `0.017/0.064m` 且 stable。E013 证明 true-freejoint oracle 不限制 E081 级 object target，E014 soft target 已生成 | ✅ 详见 log 13 |
| E014 | 2026-05-19 | COLA-B Soft Weld | **kinematic support body + soft weld/equality 位置约束**: 6 条 full 完成；6/6 true-freejoint parity ok、non-COM anchor ok、no direct wrench，4/4 main 过 E013 soft target，2/2 guard stable。best main `t02` obj `0.056/0.087m`、hand `86.7%`、floor `51.4%`、leg `0%` | ✅ 详见 log 14 |
| E015 | 2026-05-19 | COLA A+B Dynamic Support | **dynamic support body + PD command**: 4 条 full 完成；4/4 dynamic scene/parity/object-last/no-direct-wrench ok，但 0/3 main 过 soft target，0/3 effort reasonable，2/3 main numerical instability。default main obj `0.311/0.418m`、rot `50.3deg`、support target lag `0.127/0.195m`，force/torque 打 clamp；guard stable 但 hand 不过门 | ✅ 详见 log 15 |
| E016 | 2026-05-19 | Paper Metrics + 13-case Gen | **E014 paper-aligned metrics + 13-case quick 泛化**: 指标对齐 SPIDER/DynaRetarget/OmniRetarget/Holosoma；13/13 config ok、13/13 SPIDER/Dyna object success、13/13 transport success，mean Epos `0.050m`、Erot `4.1deg`、progress `1.001`；但 contact preservation ok `3/13`、deep penetration ok `9/13`、generalization pass `0/13`；corrected front-camera 可视化确认 E016 继承 E014 weld 结构但 anchor 为 mask 派生，失败源为 contact gap/leg shortcut/artifact | ✅ 详见 log 16 |
| E017 | 2026-05-19 | Anchor Audit + Selection | **E014 GT anchor 对齐 + E016/E017 anchor 归因**: 复查确认 E016/E017 auto anchor 输入是 `selected_person_contact_mask`，不是显式 partner-side；audit 已新增 counterpart-person 弱证据通道，并补齐 7 个 anchor-position videos。`box023_p2` 为明确 E016 face 错、`box025_p2` 为可能高度偏差；6 个 validation quick 中 `box025_p2` auto z-corrected 与 E014 seed 均 pass，`box025_p1/bucket005_s2_p2` 不支持 anchor 主因，`box023_p2` 需 E014 同等 `opt_steps=32` 控制 | ✅ 详见 log 17 |
| E018 | 2026-05-19 | Canonical Support Proxy | **E014 proxy anchor canonicalization GT gate**: 先跑 `box023_p2/box025_p2` 两个 E014 GT case；canonical rule 为 face center + `0.62*half_z`。2/2 GT anchor dist `<1.2cm`、2/2 config ok、2/2 soft target pass、2/2 SPIDER/Dyna/transport success。object tracking 几乎复现 E014 t02：box023 `0.0425/0.0793m`，box025 `0.0562/0.0871m`；box023 仍有 paper contact-preservation gap `32.1%`，说明 anchor gate 通过但 robot-side artifact 未完全解决 | ✅ 详见 log 18 |
| E018b | 2026-05-19 | 13-case Generalization | **canonical support proxy 13-case full 泛化 + 在线视频**: 本地 1 卡 + 远程 2 卡跑完 13/13 full，13/13 root NPZ 与 13/13 online MP4。13/13 config/canonical anchor/SPIDER/Dyna/transport success，2/2 GT anchor gate pass；mean Epos `0.054m`、Erot `5.22deg`。复核在线视频后新增 robot fall gate：`box021_p1/p2` 与 `bucket001_p1/p2` 4/13 明确摔倒，upright ok `9/13`，严格 generalization 仅 `1/13`；anchor/object-side 已成立，下一步应转 robot-side stability/contact/artifact control | ✅ 详见 log 19 |
| E020 | 2026-05-20 | Failure Attribution Audit | **E018b 13-case 失败归因审计**: 按 E076 分层证据扩展为 S1-S6 protocol，生成 13/13 root-cause CSV、13/13 attribution panel、13/13 keyframe triplet 与 scene snapshot。归因分布：`algo_stability=4`、`algo_contact=4`、`retarget_kinematic=2`、`contact_mask=1`、`raw_data=1`、`pass=1`；下一步明确为 E021 mask/ref 修复、E022 stability/leg collision、E023 contact closure、E024 multi-agent/data filter | ✅ 详见 log 20 |

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
| E007 partial | `E007_box025_p2_yneg_k20_simdt`: obj `0.625/1.205m`、hand `61.3%`、floor `93.1%`、object xy ratio `0.291`、rot `41.6deg`、proxy xy ratio `0.693`、E081 majority `1/6` | dt fix 生效但 max-speed clamp 截断 proxy；E008 需先测试高/不限速 proxy |
| E008 full | 6 个 full 结果：5 main + 1 guard，`num_main_proxy_support_tracking_ok=5`，`num_main_reaches_E081_transport_proxy=0`；best `E008_box025_p2_ypos_k20_vmax2` obj `0.363/0.684m`、hand `78.0%`、floor `64.7%`、leg intf `2.9%`、xy ratio `0.724` | 限速修正显著改善 transport，但剩余瓶颈是 robot hand/support 闭环；E009 应基于 best ypost 方向做 hold-contact/contact-pad，而非继续扫 speed/kp |
| E009 full | 6 个 full 结果：4 main + 2 guard，`num_main_proxy_support_tracking_ok=4`、`num_main_reaches_E081_transport_proxy=0`、`num_main_improves_E008_best=0`；`hc2` obj mean 略低到 `0.344m` 但 hand `66.5%`、rot `38.5deg`，`vmax0_hc1` xy `0.789` 但 hand `55.5%`、rot `38.6deg` | hold-contact reward 不能闭合 robot-side 支撑；E010 转结构性 contact pad / soft constraint，而不是继续加 reward |
| E010 full | 7 个 full 结果：5 main + 2 guard，`num_main_proxy_support_tracking_ok=5`、`num_main_reaches_E081_transport_proxy=0`、`num_main_improves_E008_best=0`；best-ish `pad10_vmax2_hc05` obj `0.680/1.295m`、hand `83.8%`、floor `83.8%`、xy `0.276`，`pad16` xy 最高也只有 `0.326` | 单个 mocap contact pad 不能替代 direct wrench；它减少旋转捷径但也几乎不传递运输。E011 应做 soft equality/weld diagnostic，量化离 E081 还差多少 partner coupling |
| E011 full | 9 个 full 结果：7 main + 2 guard，`num_freejoint_parity_ok=9`、`num_partner_force_metrics_present=9`、`num_main_reaches_E081_transport=0`、`num_main_improves_E008_best=1`；best main `box025_com_k100` obj `0.340/0.673m`、hand `86.1%`、floor `61.3%`、xy `0.905`、rot `3.5deg`；`g1` xy `0.971`/floor `22.5%` 但 hand `75.1%` | COM spring 证明 E006 的“只旋转”来自 off-COM wrench + 弱闭环；外部 coupling 能恢复平移但不能达到 E081 精度。下一步转向双点/双手 partner constraint 或 robot-side hand/support pose shaping |
| E012 full | 8 个 full 结果：6 main + 2 guard，`num_dual_points_config_ok=8`、`num_freejoint_parity_ok=8`、`num_main_reaches_E081_transport=0`、`num_main_pose_closure_helped=0`、`num_guard_stable=0`；diagnostic=`rotation_shortcut:4`、`insufficient_coupling:2`、`guard_unstable:2`；main best-by-error `k150` obj `0.306/0.606m`、xy `0.980`、floor `59.5%`，但 rot `97.8deg`、torque max `30Nm` | dual-point partner force 没有修好 E011 精度缺口，反而重新打开 off-COM torque shortcut；下一步应限制 partner 力矩通道或转 robot-side hand/support pose shaping |
| E013 full | 2 个 full 结果：`num_freejoint_oracle_config_ok=2`、`num_near_e081_obj_oracle=2`、`num_guard_stable=1`；main obj `0.011/0.038m`、hand `78.6%`、floor `57.8%`、leg `0.0%`、xy `1.000`；guard obj `0.017/0.064m`、hand `72.7%`、floor `34.7%`、leg `0.0%` | Oracle 证明 E081 object target 在 true-freejoint data/eval 下可达，后续软 target main 为 obj `<=0.193/0.371m`、hand `>=73.6%`、floor `<=69.5%`；E014 应进入 COLA-B 位置约束 |
| E014 full | 6 个 full 结果：`num_freejoint_parity_ok=6`、`num_anchor_not_com_oracle=6`、`num_no_direct_wrench=6`、`num_main_soft_target_pass=4`、`num_guard_stable=2`；main obj `0.056-0.082 / 0.085-0.142m`、hand `85.5-90.8%`、floor `49.7-51.4%`、leg `0%`、xy `0.995-0.999`、rot `1.6-2.2deg` | COLA-B 位置约束成立；E014b 不触发。下一步进入 E015 dynamic support + PD 或 pipeline 对接验证，而不是 E016 回退 |
| E015 setup | 4-step smoke：4/4 `freejoint_parity_ok`、4/4 `support_dynamic_scene_ok`、4/4 `object_last_ok`、4/4 `no_direct_wrench`、4/4 `pd_metrics_present`；scene/data 独立检查 `nq/nv/nu=49/47/29`，support q/d `36/35`，object q/d `42/41` | E015 工程 wiring 已通过；下一步 setup commit + full，本地跑 default main，远程跑剩余 main/guard |
| E015 full | 4 个 full 结果：`num_freejoint_parity_ok=4`、`num_support_dynamic_scene_ok=4`、`num_main_soft_target_pass=0`、`num_main_effort_reasonable=0`、`num_numerical_instability=2`、`num_guard_stable=1`；default main obj `0.311/0.418m`、rot `50.3deg`、target gap `0.127/0.195m`、force max `250N` | E015 差于 E014；失败模式是 dynamic support lag + PD saturation + 数值不稳。触发 E015b effort/PD tuning 分析，不跳 E016 |
| E016 quick | 13 个 quick 泛化结果：`num_config_ok=13`、`num_paper_spider_success=13`、`num_paper_dynaretarget_success=13`、`num_transport_success=13`、`num_contact_preservation_ok=3`、`num_deep_penetration_ok=9`、mean Epos `0.050m`、Erot `4.08deg`、contact preservation `39.6%` | E014 B-only object-side 泛化成立，但完整 retargeting 未通过；主要缺口是 robot-side contact preservation、deep penetration 与 leg/floor artifact |
| E017 validation + side audit | 6 个 quick 验证：`num_config_ok=6`、`num_paper_spider_success=6`、`num_transport_success=6`、`num_generalization_pass=2`；side audit: auto anchor source side=`selected_person_contact_mask`，selected/partner face same/different/opposed/missing=`8/3/1/1`，GT 与 counterpart dominant face `2/2` 不同 | anchor 归因缩小到：`box025_p2` 为已验证 anchor 高度问题；`box023_p2` 是明确 anchor face 错但还需 full-budget 控制；counterpart mask 只能作无 GT 弱证据，不能覆盖 E014 GT |
| E018 GT gate | 2 个 full 结果：`num_config_ok=2`、`num_gt_anchor_pass=2`、`num_soft_target_pass=2`、`num_gt_gate_pass=2`、`num_paper_spider_success=2`、`num_paper_dynaretarget_success=2`、`num_transport_success=2`；mean Epos `0.049m`、Erot `2.10deg`；`box023_p2` anchor dist `1.17cm`，`box025_p2` `0.94cm` | canonical support proxy 复现 E014 手工 anchor 语义和 object-side 效果；可以进入 E018b 10+ case 泛化，但需继续分离 robot-side contact/artifact gate |
| E018b full | 13 个 full 结果：`num_config_ok=13`、`num_canonical_anchor_pass=13`、`num_gt_gate_pass=2/2`、`num_paper_spider_success=13`、`num_paper_dynaretarget_success=13`、`num_transport_success=13`；mean Epos `0.054m`、Erot `5.22deg`、contact preservation ok `5/13`、deep penetration ok `7/13`、robot fall detected `4/13`、visual stability ok `9/13`、strict generalization `1/13` | canonical support proxy 的 anchor/object-side 泛化成立，但 object-only success 不能代表完整成功；完整 retargeting 仍被 robot-side stability、contact preservation、deep penetration、leg/floor artifact 限制。下一步不继续调 anchor，而应做 robot-side stability/contact/artifact control |
| E020 audit | 13 个 case 均完成 S1-S6：root causes=`algo_stability:4, algo_contact:4, retarget_kinematic:2, contact_mask:1, raw_data:1, pass:1`；13/13 attribution panels；S3 显示当前 processed `contact` 字段相对 raw 3cm mask 是系统性 all-on overclaim；S2 标出 `box025_p1/bucket007_p2` ref leg/object interference 高 | E018b 后续应拆线推进：E021 修 mask/ref geometry，E022 修 fall/stability，E023 修 contact/collision artifact，E024 判断 partner-heavy case 是否需 multi-agent 或数据过滤 |

## Plans

- E001: `workspace/core4d_collab_retarget/plan/01_E001_baseline_literature_code_audit_plan.md`
- E002: `workspace/core4d_collab_retarget/plan/02_E002_freejoint_legobj_control_audit_plan.md`
- E003: `workspace/core4d_collab_retarget/plan/03_E003_freejoint_physics_feasibility_sweep_plan.md`
- E004: `workspace/core4d_collab_retarget/plan/04_E004_freejoint_virtual_partner_support_plan.md`
- E005: `workspace/core4d_collab_retarget/plan/05_E005_partner_force_timing_support_site_plan.md`
- E006: `workspace/core4d_collab_retarget/plan/06_E006_cola_support_body_proxy_plan.md`
- E007: `workspace/core4d_collab_retarget/plan/07_E007_support_proxy_timebase_e081_plan.md`
- E008: `workspace/core4d_collab_retarget/plan/08_E008_support_proxy_speed_unclamped_e081_plan.md`
- E009: `workspace/core4d_collab_retarget/plan/09_E009_ypos_hold_contact_closure_e081_plan.md`
- E010: `workspace/core4d_collab_retarget/plan/10_E010_mocap_contact_pad_support_e081_plan.md`
- E011: `workspace/core4d_collab_retarget/plan/11_E011_soft_object_tether_diagnostic_e081_plan.md`
- E012: `workspace/core4d_collab_retarget/plan/12_E012_dual_point_partner_pose_closure_plan.md`
- E013: `workspace/core4d_collab_retarget/plan/13_E013_true_freejoint_object_oracle_plan.md`
- E014: `workspace/core4d_collab_retarget/plan/14_E014_cola_b_kinematic_weld_plan.md`
- E015: `workspace/core4d_collab_retarget/plan/15_E015_cola_ab_dynamic_support_pd_plan.md`
- E016: `workspace/core4d_collab_retarget/plan/16_E016_e014_paper_metrics_generalization_plan.md`
- E017: `workspace/core4d_collab_retarget/plan/17_E017_anchor_audit_selection_plan.md`
- E018: `workspace/core4d_collab_retarget/plan/18_E018_canonical_support_proxy_anchor_plan.md`
- E018b: `workspace/core4d_collab_retarget/plan/19_E018b_canonical_support_proxy_13case_plan.md`
- E019: `workspace/core4d_collab_retarget/plan/20_E019_unified_eval_framework_plan.md`
- E020: `workspace/core4d_collab_retarget/plan/21_E020_failure_attribution_audit_plan.md`

## Logs

- E001: `workspace/core4d_collab_retarget/log/01_E001_baseline_literature_code_audit_results.md`
- E002: `workspace/core4d_collab_retarget/log/02_E002_freejoint_legobj_control_audit_results.md`
- E003: `workspace/core4d_collab_retarget/log/03_E003_freejoint_physics_feasibility_sweep_results.md`
- E004: `workspace/core4d_collab_retarget/log/04_E004_freejoint_virtual_partner_support_results.md`
- E005: `workspace/core4d_collab_retarget/log/05_E005_partner_force_timing_support_site_results.md`
- E006: `workspace/core4d_collab_retarget/log/06_E006_cola_support_body_proxy_results.md`
- E007: `workspace/core4d_collab_retarget/log/07_E007_support_proxy_timebase_e081_partial_results.md`
- E008: `workspace/core4d_collab_retarget/log/08_E008_support_proxy_speed_unclamped_e081_results.md`
- E009: `workspace/core4d_collab_retarget/log/09_E009_ypos_hold_contact_closure_e081_results.md`
- E010: `workspace/core4d_collab_retarget/log/10_E010_mocap_contact_pad_support_e081_results.md`
- E011: `workspace/core4d_collab_retarget/log/11_E011_soft_object_tether_diagnostic_e081_results.md`
- E012: `workspace/core4d_collab_retarget/log/12_E012_dual_point_partner_pose_closure_results.md`
- E013: `workspace/core4d_collab_retarget/log/13_E013_true_freejoint_object_oracle_results.md`
- E014: `workspace/core4d_collab_retarget/log/14_E014_cola_b_kinematic_weld_results.md`
- E015: `workspace/core4d_collab_retarget/log/15_E015_cola_ab_dynamic_support_pd_results.md`
- E016: `workspace/core4d_collab_retarget/log/16_E016_e014_paper_metrics_generalization_results.md`
- E017: `workspace/core4d_collab_retarget/log/17_E017_anchor_audit_selection_results.md`
- E018: `workspace/core4d_collab_retarget/log/18_E018_canonical_support_proxy_anchor_results.md`
- E018b: `workspace/core4d_collab_retarget/log/19_E018b_canonical_support_proxy_13case_results.md`
- E020: `workspace/core4d_collab_retarget/log/20_E020_failure_attribution_audit_results.md`

## Git

- Baseline merge: `feat/dual-robot-retarget -> main` 已 fast-forward 到 `ae8b342`。
- 当前实验分支：`exp/core4d-collab-retarget`。
