# E089 Plan — G1-Feasibility Gate 验证（A 路 + B 路并行）

日期：2026-05-28
对应诊断：`workspace/exp_diagnostic/diagnostic_report.md`、`workspace/exp_diagnostic/data_filter_recommendation.md`
对应 TRACKER：E088 之后的 Phase 19 入口

## Context

E082-E088 已穷尽 D003 box021 (`20231018_029_p2` 等 3 case) 上的 reward/mass/hard-gate 调参，全 0 通过。`workspace/exp_diagnostic` 诊断定位主因是 **G1 IK retarget 出的双手目标在 box 上的几何对 G1 单人不可达**（box021_18029_p2 R wrist 33% 帧在 box 内、L_top_face_frac=0、pelvis 被强制蹲到 0.58m）。

诊断同时设计了 5 条 **G1-Feasibility Gate**（`workspace/exp_diagnostic/scripts/g1_feasibility_gate.py`），在 12 个已知 case 上完成 calibration：所有 E082-E088 失败 case 被拒绝，box023_p2 / box025_p2 通过，且**意外发现 `box021_person1`（pre-D003 老 OmniRetarget）通过新 gate** —— 中等尺寸 + G1 可行 + 从未跑过 SPIDER 动力学。

本 E089 同时验证两路：
- **A 路**：用现成 `box021_person1` 数据跑 SPIDER 动力学，验证"gate pass = SPIDER 可成功"假设；零额外 retarget。
- **B 路**：在 holosoma 重做 D003 box021 OmniRetarget，加 "双手目标投到 box 顶面 +5cm" 约束，把 13 个 case-person 中能 pass gate 的样本重生成；再回 SPIDER 上跑 1-2 个 top case。

## Claims

| ID | 描述 | 量化判据 |
|---|---|---|
| C1 (A) | `box021_person1` SPIDER smoke 阶段 head pen ≤ 30%（远低于 box021 D003 case 平均 80%+），证明 gate-pass 与 dynamic-feasibility 在 medium box 上确实正相关 | smoke 24-step `head_pen_frac ≤ 0.30` |
| C2 (A) | `box021_person1` full CEM 同时满足 contact ≥ 60% + pelvis_min ≥ 0.55m + head_pen ≤ 30%，第一次在 medium box (vol 0.06 m³) 上拿到可用 dynamic-retarget 结果 | full CEM 后 `contact_frac ≥ 0.60 AND pelvis_z_min ≥ 0.55 AND head_pen_frac ≤ 0.30` |
| C3 (B) | holosoma OmniRetarget 加 top-face 约束后，重做 D003 box021 13 case-person 中至少 5 个能 pass G1-Feasibility gate（vs 旧版 0/13）| `g1_feasibility_pass_rate ≥ 5/13` |
| C4 (B) | B 路 top-2 case 在 SPIDER 24-step smoke 上达到 head_pen ≤ 30%，证明改 OmniRetarget 约束确实改善了 dynamic feasibility | 2 个 case 中至少 1 个 `head_pen_frac ≤ 0.30` |

如果 C1+C2 双通过：直接产出第一条 medium-box dynamic-retarget seed。
如果 C3+C4 双通过：证明 OmniRetarget 约束修复是 D003 box021 整批的解药。
如果都失败：进入诊断 Phase B3（H2 binding，承认 box021 单 G1 不可行，转 dual-G1）。

## A 路实现（spider 本地）

### A1. 配置生成
脚本：`workspace/core4d/scripts/E089/generate_E089A_override.py`
- 复用 E081 leg-object collision pair（box021_person1 复用其老 scene_act_meta，需先 derive 一个 `box021_person1_legobj_e089` task）
- override yaml: `examples/config/override/core4d_E089A_box021_person1_legobj.yaml`
  - 基础：E085 reward stack（raw target external，但 A 路用 ref_fk dynamic target，不依赖 box021 D003 raw target，因为 box021_person1 老数据没生成 raw target npz）
  - `contact_hdmi_target_source=ref_fk`
  - `contact_hdmi_target_uses_eef_offset=true`（E073 修复版本）
  - `cem_safety_gate_*`：复用 E088 hard gate
  - `object_clearance_*`：复用 E088 绝对 clearance
  - leg-object collision pairs（E081）
  - upper-body-object collision pairs（E083）
  - 与 E088 的 ABC 不同，A 路不做权重 sweep，只跑 1 个变体

### A2. Snapshot + smoke
- `bash workspace/core4d/scripts/convert/snapshot_scenes.sh E089 box021_person1_legobj_e089`
- 4-step smoke：验证 scene 加载、ref FK target 不在 box 内、reward 不报错
- 期望：smoke 跑通且 first-tick `head_pen_frac` 已经低于 box021 D003 同期值

### A3. Full CEM
- `bash workspace/core4d/scripts/train/train_E089.sh local 0`
- 1 GPU 本地约 30-40 min（E088 体量参考）
- 写到 `workspace/core4d/results/E089/A/`

### A4. Eval
- `workspace/core4d/scripts/eval/eval_E089.py --variant A`
- 指标：contact_frac、pelvis_z_min、head/upper_pen_frac、hand_floor_frac、object clearance mean、IK FK in-box rate (sim)
- 视频 + keyframe sheet

## B 路实现（holosoma 远端，subagent 执行）

### B1. 探索 holosoma OmniRetarget 代码（subagent 调研）
- 找到 D003 production 中 OmniRetarget 调用入口、IK 目标生成位置
- 找到"hand target 位置"的来源（人手 mocap → robot wrist target 转换的代码）
- 评估两种插入约束的方式：
  - B-1：在 IK target 生成时，若 hand 在 box 周围 N cm 内，把 target 投到最近 +z 顶面 + 5cm 向上
  - B-2：post-IK 阶段，对 wrist 6-DoF target 做二次 IK，加 box top-face 投影 constraint

### B2. 实现选定方案 + 单 case 验证（subagent 实现）
- 实现约束（首选 B-1，简单且语义清晰）
- 单 case 验证：用 `d003_box021_20231018_029_p2` 重做，比较新旧 trajectory_kinematic.npz
- 用 G1-Feasibility gate 评估单 case：是否 pass

### B3. 13 case 批量重做 + gate 评估（subagent 自动化）
- 重做 D003 box021 13 case-person
- 用 `workspace/exp_diagnostic/scripts/g1_feasibility_gate.py` 批量评估
- 输出 `workspace/exp_diagnostic/findings/08_B_path_gate_results.json`

### B4. Top-2 case 转 SPIDER（spider 本地, 等 B3 完成）
- 同步重做后的 trajectory_kinematic.npz 到 spider 本地
- 用 E089A 同款 override（仅替换 task）跑 24-step smoke
- 报告 head_pen_frac 等关键指标

## 并行调度

| Track | 资源 | 启动时机 | 预计耗时 |
|---|---|---|---|
| A1-A2 (override + snapshot + smoke) | local CPU | 立即 | 5 min |
| A3 (full CEM) | local GPU0 | A2 完成后 | 30-40 min |
| A4 (eval + video) | local GPU0 | A3 完成后 | 5 min |
| B1-B3 (subagent in holosoma) | 调研 + 实现，远端 D003 重跑 | 立即（与 A 并行） | 60-120 min |
| B4 (spider smoke) | local GPU0 | B3 完成且 A3/A4 已结束 | 10 min |

总耗时约 1-2 小时 wall clock。

## 改动文件

| 类型 | 路径 | 状态 |
|---|---|---|
| 计划 | `workspace/core4d/plan/95_E089_g1_feasibility_AB_validation_plan.md` | 本文档 |
| A 路 override generator | `workspace/core4d/scripts/E089/generate_E089A_override.py` | 待创建 |
| A 路 derived task generator | `workspace/core4d/scripts/E089/derive_box021_person1_legobj.py` | 待创建 |
| A 路 override yaml | `examples/config/override/core4d_E089A_box021_person1_legobj.yaml` | 待生成 |
| A 路 train script | `workspace/core4d/scripts/train/train_E089.sh` | 待创建 |
| A 路 eval | `workspace/core4d/scripts/eval/eval_E089.py` | 待创建 |
| Snapshot | `workspace/core4d/results/E089/scene_snapshot/` | 待生成 |
| B 路 调研报告 | `workspace/exp_diagnostic/findings/08_B_path_omniretarget_audit.md` | subagent 写 |
| B 路 代码改动 | holosoma 内对应 OmniRetarget 文件 | subagent 改 |
| B 路 gate 评估 | `workspace/exp_diagnostic/findings/08_B_path_gate_results.json` | subagent 写 |
| 结果 | `workspace/core4d/log/111_E089_g1_feasibility_AB_results.md` | 待生成 |

## 不做什么

- 不做 reward weight sweep（E060-E088 已穷尽）
- 不动 mass / 不动 hard gate 阈值（E087/E088 已穷尽）
- 不做 dual-G1 / Mocap partner（C 路；只有 A+B 都失败才考虑）
- A 路不重新生成 raw contact target npz（用 ref_fk 即可，box021_person1 老 scene 没必要再做 raw）
- B 路不修 SPIDER 端 reward（约束修在 OmniRetarget 上游）

## 风险与回退

| 风险 | 应对 |
|---|---|
| B 路 OmniRetarget 代码复杂、subagent 一次性改不对 | subagent 先做调研 + 1 case 验证；只有验证通过才批量；否则停下报告 |
| A 路 box021_person1 老 scene 与 E085+ reward 不兼容（例如 mask 路径） | 简化 override，关 raw contact mask；仅依赖 ref_fk target |
| A 路 smoke 失败 | 立刻读 stderr；不超过 3 次重试，转 B 路单独推进 |
| B 路 13 case 重做时间超 1 小时 | subagent 先做 3 case 验证再批量；最差只批 3 |
| C1 通过但 C2 失败 | smoke 数值好但 full CEM 翻车 → 进一步 reward 微调；不是 dead-end |
| C1+C2 双失败 | gate 假阳性，需要补充 calibration 维度（例如加 IK joint limit feasibility）|
| C3 通过但 C4 失败 | OmniRetarget 修复后 gate 通过但动力学仍坏 → 说明 gate 不充分；进入 dual-G1 路 |
