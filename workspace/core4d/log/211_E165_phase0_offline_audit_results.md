# E165 Phase 0 离线审计结果（A/C/E1）

日期：2026-06-18
Plan：[plan/176_E165_rl_safe_eval_levers_plan.md](../plan/176_E165_rl_safe_eval_levers_plan.md)
环境：hssim（`/mnt/public/usr/yancilin/work_dir/.holosoma_deps/miniconda3/envs/hssim/bin/python`），纯离线、无训练、无 Isaac（复用已有 probe CSV）。
结果路径：`workspace/core4d/results/E165/{A_box004_contact_audit, C_box023_penetration, E1_onrails_probe}/`

## 目的

Phase 0 三个训练-free 审计，验证 plan 的 Claim C-A / C-C / C-E1，为 Phase 2/3 修复定向。

## 运行命令

```bash
PY=/mnt/public/usr/yancilin/work_dir/.holosoma_deps/miniconda3/envs/hssim/bin/python
$PY workspace/core4d/scripts/eval/runners/eval_E165_box004_contact_audit.py --case box004   # A (+box021/box023 对照)
$PY workspace/core4d/scripts/eval/runners/eval_E165_box023_penetration_trace.py             # C
$PY workspace/core4d/scripts/eval/runners/eval_E165_isaac_onrails_probe.py                  # E1
$PY workspace/core4d/scripts/eval/reports/gen_E165_offline_plots.py                         # 可视化
```

## Claims 验证

| Claim | 预设判据 | 实测 | 结论 |
|---|---|---|---|
| **C-A** box004 标签虚高 | 标签接触帧 hand↔box 距离中位数 > 0.08m **且** Isaac recall < 0.2 | 距离中位数 **0.054m**（< 0.08，**距离判据不成立**）；但 Isaac filtered recall **0.058**（≪0.2）| ✅ **坐实，但机制需修正**（见下）|
| **C-C** box023 自碰撞来源 | spider & omni frame0-10 min(手↔同侧髋) 都 <0.10 → 继承 | spider **0.069m**、omni **0.073m**，两者皆 <0.10 | ✅ **INHERITED（源/姿态继承，非 CEM 引入）** |
| **C-E1** 三标量分病 | recall/phantom/init-net 联合把三 case 分开 | 三 case mode 互异（清洁/虚高/自碰撞）；但 recall 单序≠下游序 | ✅ **三标量联合成立**（单标量否定）|

## 结果详情

### E165-A：box004 接触标签 = "近而未触"，距离判据被证伪、recall 判据坐实

| case | 标签接触帧 wrist↔box 距离中位数 | Isaac filtered recall(either) | mean 接触力 | 下游 staggered |
|---|---|---|---|---|
| **box004** | 0.054m | **0.058** | **1.08N** | 0.48 |
| box021（对照）| 0.050m | **0.615** | 21.4N | 0.67 |

**关键修正（诚实记录）**：plan 原判据"距离中位数 > proxy 半径"**不成立**——box004 标签接触帧 wrist 离箱仅 ~5cm，而 box021 对照**也是 ~5cm**。**几何中心距对 box004/box021 无判别力**。真正的 fiction 判据是消费端 **Isaac filtered recall**：box004=0.058 vs box021=0.615（10×差）。即标签说接触约 100 帧，rubber-hand 实际只在 ~6 帧触箱。

机制 = **源接触未被重定向复现（embodiment gap）**：box004 是 fingertip/edge 抓姿，wrist 近但 rubber-hand proxy 不咬合 → 源人体接触在 g1 上没复现。

**标签来源已核实（纠正一处早期误判）**：E163N 标签**不是**距离阈值产物，而是 `convert_core4d_e163_manifest_to_sugar.py` 从**源人体接触 mask** 读取（gap-fill+resample，脚本 72-111 行）。证据：5 个 case 的 stored label **全是连续区间**（box004=[45,147]、box023=[28,135]…），而距离 heuristic 输出非连续（实测 match 仅 0.91，非 1.0）。**因此标签是忠实的源接触**；"虚高"指的是 RL/物理视角下"标签要求接触、重定向却没复现"，**根因在 retarget 手部贴合，不在标签生成**。

**可视化（A_distance_label_recall.png，已亲验）**：box004 上图——橙色 label 带覆盖 frame45-147，绿色 Isaac 真接触只两条窄缝(~128/~143-147)，蓝色距离线全程 ~5cm 在 8cm 线下却几乎无真接触。box021 下图——橙 label 与绿 Isaac 接触带大面积重叠，距离同样 ~5cm。**距离相同、接触有无相反**，一图定论。

### E165-C：box023 手贴髋自碰撞 = 源/姿态继承，**非 spider CEM 引入**（决定性回答）

| 来源 | frame0 L手↔L髋 | frame0 R手↔R髋 | f0-10 min | wrist_z | 手↔箱 | <0.10? |
|---|---|---|---|---|---|---|
| **spider** (R158) | 0.075m | 0.069m | **0.069m** | 0.674m | 0.43m | ✅ |
| **omni** (SamePersonOmniRT) | 0.081m | 0.073m | **0.073m** | 0.675m | 0.43m | ✅ |

spider 与 omni frame0-10 手↔髋距离**几乎一致**（0.069 vs 0.073m），都远小于 0.10m → **两条 pipeline 都把手垂在髋边**。wrist_z≈0.674（髋高）、手↔箱≈0.43m（远离箱），完全复现文档描述。

**结论**：box023 的手贴髋自碰撞**继承自源人体姿态（站立/预抓取手垂胯边）**，OmniRetarget 原始数据同样存在，**不是 spider CEM 的问题**。修复应落在：上游源姿态处理 / g1 rubber-hand proxy 几何（半径/collision filter），而非改 CEM。

**可视化（C_hand_hip_spider_vs_omni.png，已亲验）**：spider(蓝)/omni(橙)四条曲线 frame0-9 全程在 10cm 线下；omni 略高(7.3-9.1cm)但仍贴髋。

### E165-E1：on-rails 三标量——单标量不预测下游，三标量联合分病

| case | filtered recall | phantom force rate | max init net force | mode | staggered |
|---|---|---|---|---|---|
| box021 | 0.615 | 0.00 | 0N | clean_transfer | 0.67 |
| box004 | 0.058 | 0.197 | 0N | label_fiction | 0.48 |
| box023 | 0.667 | 0.614 | **2459N** | init_penetration/self-collision | 0.00 |

**关键洞察**：recall 单序（box023>box021>box004）≠下游序（box021>box004>box023）——**box023 recall 最高(0.667)却 0/64**，因为它败在自碰撞而非接触缺失。证明**必须三标量联合、不可合成单一 gap**（与文档 §3"三种相反接触病"一致）：
- box021：高 recall + 0 phantom + 0 init → 干净，预言成功
- box004：低 recall(0.058) → 标签虚高，预言接触质量问题
- box023：高 init-net(2459N) + 高 phantom → 自碰撞穿透，预言结构性失败

**可视化（E1_three_scalars.png，已亲验）**：三柱图清晰——recall 中 box004 独低；phantom 中 box023 独高；init-net 中 box023(2459N) 鹤立、box021/box004≈0。

## 发现的 Bug / 工程隐患

> **更正（2026-06-18 同日）**：初稿曾把两个"bug"记到 `convert_holosoma_export_to_sugar.py`（BOX021 半尺寸硬编码 + 8cm wrist 阈值）。经核实，E163N case 的标签**不是**该脚本生成的（用的是 `convert_core4d_e163_manifest_to_sugar.py` 的源 mask 路径），**两个 bug 对本实验不适用，已撤回**。教训：定位前先确证标签 provenance。

| # | 项 | 位置 | 性质 |
|---|---|---|---|
| 1（撤回）| `heuristic_contact` 硬编码 BOX021_HALF_EXTENTS | `convert_holosoma_export_to_sugar.py:63,121` | 是该脚本的潜在隐患，但**未用于 E163N**，与本实验无关；仅作记录 |
| 2（撤回）| 8cm wrist 中心距阈值 | 同上 | 同上，未用于 E163N |

**真正的工程指向（替代上面撤回的 bug）**：box004 的源接触在 g1 上**未被重定向复现**（recall 0.058）。这不是标签/预处理 bug，而是 **retarget 手部贴合不足 + rubber-hand proxy 容错低**（fingertip/edge 抓姿）。属上游 SPIDER + SUGAR proxy 范畴，Phase 2/上游处理。

## Claims 总结 → 对杠杆的指向

- **杠杆1（on-rails 探针）验证有效**：E1 三标量 + A 的 recall 对照共同证明——廉价、训练-free、且能在 RL 前分出三种病。**recall 是接触质量的真判别量，几何距离不是**。建议 preflight 闸报这三个标量（已成型为 `eval_E165_isaac_onrails_probe.py`）。
- **杠杆3（RL 抬升 reward）**：本 Phase 未直接测 height（属 Phase 2），但 box004 的 recall 0.058 + mean 力 1.08N 说明"低位推/未咬合"在 on-rails 层面已可见。
- **box023 修复方向确定**：自碰撞继承自源姿态 → 改 proxy 几何 / collision filter / 预抓取窗口不计，**不动 CEM**（Phase 2 F）。

## 下一步

1. Phase 1 文档（已完成纠错）可补：把 A 的"距离无判别力、recall 才判别"和 C 的"INHERITED"结论回填到 `E163_RL_DEEP_INSIGHTS_CN.md` §3/§4（量化数字）。
2. Phase 2（远程 GPU）：F box023 proxy 修复 + B height reward；box004 方向 = 上游 SPIDER 手部贴合 / SUGAR rubber-hand proxy 容错（非标签/预处理）。
3. 可选：A 用 rubber-hand 位置（需 Isaac 重跑 probe 存 body_pos）复核，把 recall 口径从 wrist 提升到 proxy。
