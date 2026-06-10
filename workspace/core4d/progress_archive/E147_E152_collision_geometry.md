# E147 Progress — 2026-06-08

## E151 route-B hand surface contact reward

- [x] 2026-06-09 E151 上下文恢复：按 `experiment-planning-zh` 读取 `plan/159_E151_route_b_hand_surface_contact_reward_plan.md`、remote execution 指南、tracker/progress 与 E150 结果；确认本轮是 reward 轴路线B，核心矩阵为 3 case × {B2-sup, B2-tip, B1}，baseline 复用 E147/E148 rubber，必要时再按判据跑 B1+B2。当前 tracker/progress 尚无 E151 结果记录，工作树仅有用户 `.vscode/settings.json` 修改。
- [x] 2026-06-09 E151 初步代码审计：`spider/simulators/mjwp.py` 已有 E120/E121 的 `hand_support_*` reward plumbing 和 info 输出，但 `_geom_box_sdf_min` 对 mesh 仍退化为中心/半径口径；E147 evaluator 已有 rubber mesh 顶点 SDF 评测口径。已将 `_geom_box_sdf_min` 改为 primitive 走旧路径、mesh 走固定 64 顶点采样 SDF 的纯增量分支，待自检脚本验证。
- [x] 2026-06-09 E151 固定入口初版落地：新增 `scripts/E151/build_route_b_manifest.py`（含 box021 clean target 重生成/差异阈值 5mm 选择、12 行 manifest、E151 overlay override、scene snapshot）、`check_mesh_sdf_and_targets.py`（mesh SDF vs E147 evaluator、primitive 回归、target shape/finite）、train/local/remote/pull/eval 固定脚本。发现 rubber mesh 实际约 2.2 万顶点，64 点可能与 E147 `MESH_SAMPLE_COUNT=800` 评测口径不一致；已将 runtime `MESH_SDF_SAMPLE_COUNT` 调到 800，优先保证 reward/评测一致，B1 smoke 后再看 step time。
- [x] 2026-06-09 E151 preflight 通过：manifest 12 rows = 3 baseline reuse + 9 to_run，split 为 local-gpu0/remote-gpu0/remote-gpu1 各 3；mesh SDF parity 9 帧点 vs E147 evaluator `max_abs_diff=0.0m`，primitive regression `max_abs_diff=0.0m`，target validation 6/6 pass。box021 clean target 校验显示旧 target 不可直接复用：B2-sup old-vs-clean max diff 0.3280m、B2-tip 0.0692m，均超过 5mm，E151 manifest 已选择 clean 重生成 target。pre-run eval `method_rows=3/delta_rows=0/missing=9`，baseline 评估路径可用。
- [x] 2026-06-09 E151 smoke 完成：本地 3/3 + 远端 6/6 自然结束并回收，`pull_E151_remote_results.sh smoke` 后本地 root NPZ/MP4/outdir 为 9/9/9。发现 evaluator 只读 manifest full 路径，已修 `eval_E151_route_b_hand_surface_contact.py` 按 `stage` 重写 E151 run rows 的 `result_npz/outdir_npz/video`，保留 baseline 复用路径；`py_compile`/`bash -n` 通过。smoke eval 结果 `method_rows=12/delta_rows=9/missing=0/visual_sheets=9`。B1 机制信号存在：`hand_support_rew_mean_mean` 约 0.115/1.15e-10/0.072，`hand_support_score_mean_mean` 约 0.052/0.00012/0.030；低迭代 smoke 的 to_run rows 均 `fall_flag=true`，仅作为执行与 reward plumbing gate，不作为成败结论。B1 smoke step time：local 5090 约 3.1s/tick，remote A6000 约 1.2-1.4s/tick，可接受，进入 full CEM。
- [x] 2026-06-09 E151 full 已启动：本地 tmux `E151_local_full_232329` 跑 `local-gpu0` 3 条 box021，远端 tmux `E151_full_232329` 跑 `remote-gpu0` 3 条 box004 与 `remote-gpu1` 3 条 box023。启动检查确认 split/task 正确，首条 full 均进入 CEM；初始 full plan time 约 local 24s/tick、remote 9-11s/tick。
- [x] 2026-06-10 E151 full 远端完成、本地最终 B1 运行中：local `E151_box021_029_p2_b2_tip` 于 00:19:36 完成并进入 `b1_mesh`，当前约 `62/150`，本地 root NPZ 2/3。remote tmux `E151_full_232329` 已自然退出，远端 root NPZ 6/6（box004 与 box023 的 B2-sup/B2-tip/B1-mesh 均生成）。
- [x] 2026-06-10 E151 full 全部完成：本地最终 `E151_box021_029_p2_b1_mesh` 于 00:48:47 完成，tmux 自然退出；本地 full 目录 root NPZ 9/9、MP4 9/9，远端结果已回收。
- [x] 2026-06-10 E151 full eval/视觉/log/tracker 完成：`eval_E151_route_b_hand_surface_contact.sh full` 输出 `method_rows=12/delta_rows=9/missing=0/visual_sheets=9`。结果为 near-contact 提升但 penetration 同步上升：B2-sup 5cm +0.0963 / hand_pen +0.1771，B2-tip +0.0987 / +0.2694，B1-mesh +0.0789 / +0.2074；三方法 success 均 0/3。B1 机制信号非零（`hand_support_rew_mean_mean` 约 2.49/1.89/1.61），但视觉帧显示主要是手面/掌面压入 box，而不是 clean surface contact。B1+B2 预设触发条件未满足，未启动叠加。已写 `workspace/core4d/log/191_E151_route_b_hand_surface_contact_reward_results.md`，更新 `EXPERIMENT_TRACKER.md` E151 行。
- [x] 2026-06-10 按用户要求将 E151 实验日志 `workspace/core4d/log/191_E151_route_b_hand_surface_contact_reward_results.md` 从英文改写为中文；指标、路径、结论与原记录保持一致，tracker/progress 已本来是中文。

## E150 contact anchor eef_offset sweep

- [x] 2026-06-09 E150 上下文恢复：按 `experiment-planning-zh` 读取计划 `workspace/core4d/plan/158_E150_contact_anchor_eef_offset_sweep_plan.md`、remote execution 指南、tracker/progress。确认本轮是路线A纯 config sweep，不改 SPIDER reward/算法；benchmark 为 E149 relaxed8，0.05 baseline 复用 E148/E147 rubber，新增 0.08/0.11 共 16 个 CEM runs，远程 A6000 两卡并行。
- [x] 2026-06-09 E150 固定入口初版落地：新增 `scripts/E150/build_eef_offset_sweep_manifest.py`、`scripts/train/train_E150_eef_offset_sweep.sh`、`scripts/run_E150_remote.sh`、`scripts/pull_E150_remote_results.sh`、`scripts/eval/eval_E150_eef_offset_sweep.py/.sh`。manifest build 通过：24 rows = 8 `reuse_e148` baseline + 16 `to_run`；anchors off05/off08/off11 各 8；split remote-gpu0/off08=8、remote-gpu1/off11=8。
- [x] 2026-06-09 E150 static/pre-run eval 通过：`py_compile`、shell `bash -n`、`git diff --check` 通过；`eval_E150_eef_offset_sweep.sh full --allow-missing` 成功评估 8 条 off05 baseline，输出 `method_rows=8/delta_rows=0/missing=16`，证明 evaluator 能按 row 动态设置 `EEF_OFFSET`，当前等待 off08/off11 新 CEM 结果。
- [x] 2026-06-09 E150 remote smoke 首轮诊断：远端 `E150_smoke_174912` 自然退出但只产出 2/16 root NPZ/MP4；已确认 `box021_035_p2` 的 off08/off11 smoke 成功且 `config_act.yaml` 分别写入 `contact_hdmi_eef_offset=[0.08,0,0]` / `[0.11,0,0]`。失败点是第二组 `box023_person2`：E148 manifest 的 E143 task 为 `box023_person2_legobj_e026_e081`，但复用的 E147 rubber override/sidecar 实际在 `box023_person2_legobj`，训练命令覆盖 task 后找不到 `scene_act_E147_rubber_hull.xml`。已修 `build_eef_offset_sweep_manifest.py`：E150 运行侧 `derived_task/target_scene/trajectory/base_scene_act` 从 `rubber_scene_act` 所在 task 目录派生，保证 off08/off11 与 off05 rubber baseline 同 task/scene。
- [x] 2026-06-09 E150 git sync + full 启动：已提交并推送 `a6d5564 exp(core4d): add E150 eef offset sweep runner`。`run_E150_remote.sh full` 远端 gate 通过，tmux `E150_full_181630` 已启动；GPU0 跑 off08、GPU1 跑 off11，各 8 条。
- [x] 2026-06-09 E150 remote full 完成：远端 full 自然结束，tmux `E150_full_181630` 已消失；远端 artifact audit 为 16/16 root NPZ、16/16 MP4、16/16 outdir trajectory、16 logs，GPU0/GPU1 回到空闲。
- [x] 2026-06-09 E150 full 结果回收：`pull_E150_remote_results.sh full` 后本地 artifact audit 为 16/16 root NPZ、16/16 MP4、16/16 outdir trajectory、16/16 config、16 logs；manifest 中 16 个 `to_run` 的 `result_npz/outdir_npz/video` 全存在。config 抽查确认 off08/off11 各 8 个写入 `contact_hdmi_eef_offset=[0.08/0.11,0,0]`。
- [x] 2026-06-09 E150 full eval/xlsx/视觉完成：`eval_E150_eef_offset_sweep.sh full` 输出 `method_rows=24/delta_rows=16/missing=0`；XLSX recalc `total_errors=0/total_formulas=56`。offset 均值：off05 5cm=0.6592、hand_pen=0.2777；off08 5cm=0.6613、hand_pen=0.2846；off11 5cm=0.6605、hand_pen=0.3123。delta：off08 5cm +0.0020、hand_pen +0.0069；off11 5cm +0.0012、hand_pen +0.0346；success cases 0/16，事前成功判据未满足。视觉抽查 `visual_inspection/*_off05_off08_off11.jpg`：off08/off11 没有稳定更贴物体的手面，off11 多数只是更靠近/压入边缘，和 physics contact 上升但 penetration 上升一致。
- [x] 2026-06-09 E150 结果记录完成：新增 `workspace/core4d/log/190_E150_contact_anchor_eef_offset_sweep_results.md`，更新 `EXPERIMENT_TRACKER.md` E150 行。结论：route A 单点 eef_offset 前移不成立；off11 的物理接触提升伴随穿透提升，不是 clean contact-quality win。

## E148 e143 24-case rubber hand extension

- [x] 2026-06-09 E149 clean benchmark 启动：根据用户提醒重新读取 E143 标注表和 failure report；确认 E143 已建议不要用全部 24 case 做主 claim，而是用 clean6 和 relaxed8。新增 eval-only 计划和固定 eval 脚本；不启动 CEM/远程/RL。
- [x] 2026-06-09 E149 eval 首次运行修复：首次运行发现 E148 case comparison 的 `fall_*` 字段是 `true/false` 字符串而不是 `0/1`，已修复。
- [x] 2026-06-09 E149 clean benchmark eval 完成：输出 `workspace/core4d/results/E149/e143_clean_rubber_benchmark/`。clean6 rubber-sphere：5cm +0.0079、10cm +0.0054、手物穿透 -0.2293、腿穿透 +0.0507、手物物理接触 -0.1980；relaxed8：5cm +0.0144、10cm +0.0093、手物穿透 -0.1858、腿穿透 +0.0389、手物物理接触 -0.1565。

- [x] 2026-06-09 E148 计划写入本地：新增 `workspace/core4d/plan/156_E148_e143_24case_rubber_hand_collision_plan.md`。计划将 E147 rubber_hull 扩展到 E143 24-case workset。
- [x] 2026-06-09 E148 manifest/固定入口初版落地：新增相关脚本。manifest build 通过：24 rows = 8 `reuse_e147` + 16 `to_run`；remote split 为 GPU0/GPU1 各 8。
- [x] 2026-06-09 E148 evaluator pre-run smoke 通过：`eval_E148_e143_rubber_hand_collision.sh full --allow-missing-rubber` 成功重算 24 条 Omni/sphere 与 8 条 E147 reused rubber。
- [x] 2026-06-09 E148 remote full 已启动：`run_E148_remote.sh full` 远端 static/list gate 通过，tmux session `E148_full_020443` 已启动；GPU0/GPU1 各 8 条。
- [x] 2026-06-09 E148 remote full 完成并回收：tmux `E148_full_020443` 自然结束；`pull_E148_remote_results.sh full` 后本地新 E148 artifacts 为 16/16 root NPZ、16/16 MP4、16/16 outdir trajectory、16 logs。
- [x] 2026-06-09 E148 full eval + xlsx 完成：`eval_E148_e143_rubber_hand_collision.sh full` 输出 `method_rows=72/case_rows=24/missing_rubber=0`；LibreOffice recalc `total_errors=0/total_formulas=1476`。
- [x] 2026-06-09 E148 结果记录完成：结论：rubber hand Spider 相对 sphere Spider 在 24case 上 5cm +0.0090、10cm +0.0010、手物穿透 -0.0336、腿穿透 +0.0053；filtered 23case 为 5cm +0.0216、10cm +0.0114、手物穿透 -0.0299、腿穿透 +0.0052，仍未替代 sphere 默认 baseline。

## Rubber hand collision CEM A/B

- [x] 按 `experiment-planning-zh` 恢复上下文：读取 E147 计划、EXPERIMENT_TRACKER、progress、remote execution 指南。
- [x] 当前理解：E147 不是一次性 scene hack，而是把 `hand_collision_variant_id` 作为 v3 第三个正交轴纳入 schema/registry/handoff/docs/skill。
- [x] 关键已定：rubber 使用单凸包 `maxhullvert=64`；patched scene 旁路写到 E147 结果区，CEM 通过 `scene_name=` 指向，不覆盖 processed source scene。
- [x] 修复 `patch_hand_collision.py` import 语法错误，并修正 `update_case_state_registry.py` 中 visual/target/downstream merge 对 4D registry key 的 lookup。
- [x] 新增 `scripts/E147/build_rubber_hand_collision_manifest.py`，生成 10-case `variants.tsv`、10 个 override、10 个 `scene_act_E147_rubber_hull.xml` sidecar 和 scene snapshot；MuJoCo 验证 `lh/rh` 均为 mesh 且 rbound≈0.0998m。
- [x] 新增固定运行入口 `scripts/train/train_E147_rubber_hand_collision.sh`、`scripts/run_E147_remote.sh`、`scripts/pull_E147_remote_results.sh`。
- [x] 本地 smoke 通过：`E147_d003_box021_20231011_035_p1_rubber_hull` 完成；产出 root npz/mp4/outdir trajectory。
- [x] 启动远端 full：`run_E147_remote.sh full` 同步 artifacts 后启动 tmux `E147_full_rubber_231132`；GPU0/GPU1 各 5 case 串行。修复两个运行侧问题：远端无裸 `python`；远端不需要旧 sphere npz。
- [x] 新增 E147 mesh-aware evaluator `scripts/eval/eval_E147_rubber_hand_collision.py/.sh`；smoke eval 通过。
- [x] 完成 v3 文档/skill 更新：新增 `docs/data_construction_v3/15_hand_collision_variants.md`，`run_release_checks.sh --no-smoke` 通过，65/65 checks pass。
- [x] 2026-06-09 00:54 CST 远端 full 完成并回收：remote full root `npz=10/videos=10/outdirs=10`；tmux 自然结束；本地 `pull_E147_remote_results.sh full` 回收后 artifact audit 为 10/10 root npz、10/10 mp4、10/10 outdir trajectory/config、10/10 logs、100 keyframes。
- [x] 2026-06-09 01:00 CST full eval + S6 完成：A/B primary counts 为 pass 2 / stable-but-deep-not-improved 6 / fail 2；S6 `cem_status=pass 3/fail 7`。
- [x] 2026-06-09 结果记录完成：结论：`rubber_hull` 显著降低 hand penetration/deep penetration，但 physics contact 下降且未救回 fail case；保留为 v3 CEM variant，不直接替代默认 `sphere5cm`。
- [x] 2026-06-09 E147 OmniRetarget / sphere Spider / rubber hand Spider 对比表补齐：新增 xlsx 和逐 case TSV；LibreOffice 重算公式 `total_errors=0`、`total_formulas=245`。

## E152 — hand gate physics (active)

- [x] 2026-06-10 E152 续跑恢复：当前目标是 `workspace/core4d/plan/160_E152_axis1_hand_object_physics_gate_plan.md`。用户指出 `box004` 三个方法接触箱子前手碰地，需要在 E152 evaluator 显式报告手-地接触/穿透。
- [x] 代码定位：CEM gate 采样逻辑在 `spider/optimizers/sampling.py` 和 `spider/optimizers/sampling_fast.py`；现有 `cem_safety_gate` 只支持一组全局 `min_sdf/max_violation_pct`，E152 手 gate 若要独立阈值，需要新增 `cem_hand_gate_*` 信息并在 sampling 中与 body gate 取交集。
- [x] 已实现核心 gate 增量：`config.py` 增加 `cem_hand_gate_*` 字段和 geom resolver；`mjwp.py` 输出 aggregate/body/hand 三套 gate 指标；`sampling.py` 与 `sampling_fast.py` 支持 body/hand 独立阈值并输出 `cem_hand_gate_valid_frac`、`cem_body_gate_valid_frac`。
- [x] 已补手-地/手-物体物理指标：evaluator 现在输出 hand-object `con.dist` mean/min/`frac_lt_neg5mm`/frame frac，以及 hand-floor physics contact、hand-floor `con.dist` 分布和 deep-frame frac。
- [x] E152 固定入口已落地并通过 preflight：manifest 为 12 rows，复用 6 rows，新跑 6 rows，split 为 local/remote0/remote1 各 2 rows。默认 hand gate 阈值为 `cem_hand_gate_min_sdf_m=-0.010`、`cem_hand_gate_max_violation_pct=0.05`。
- [x] E151 `box004_083_p2` 手-地指标复评：四个方法均有短时手-地接触/穿地。baseline `hand_floor_penetration_frac=3.81%`；b2_sup `6.67%`；b2_tip `3.81%`；b1_mesh `3.81%`。
- [x] E152 smoke 通过：本地单跑 `E152_box004_083_p2_gateA` smoke，`CEM hand gate: 2 geoms resolved`，`cem_gate_valid_frac≈0.943`、fallback `0`。
- [ ] E152 full 运行中：本地 tmux `E152_local_full_115333` 已完成 `box021 gateA` 并进入 `box021 gateA_b1`；远端 tmux `E152_full_retry_115537` 正在跑 `box004 gateA` 与 `box023 gateA`。`box021 gateA` gate 健康度：`cem_gate_valid_frac mean=0.838`、fallback `0`、`cem_hand_gate_min_sdf_min=-0.0064m`，未触发 full gate 塌缩。
