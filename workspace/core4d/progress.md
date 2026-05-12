# E054 Progress — 2026-05-12

## 当前状态: ✅ 完成

## 完成步骤

- [x] 整理 E001-E053 阶段总结 → `report/STAGE_REPORT_E001-E053.md` + `log/63_E001_E053_stage_summary.md`
- [x] 写 Phase 17 路线图 (Path A/B/C/D) → `report/NEXT_PLAN_E054+.md` + `plan/63_phase17_post_E053_roadmap_plan.md`
- [x] 创建 E054 plan → `plan/64_E054_case_tier_analysis_plan.md`
- [x] 实现 case_tier_analysis.py (~330 行)
- [x] 第一次 run → 发现 intent detection 用绝对手速度过严 (B+C 仅 1 个)
- [x] 诊断: G1 retarget 后手速度 0.3-0.7 m/s (行走), 远超 0.30 m/s 阈值
- [x] 修正: 改用 **手-物体相对速度** + 5 帧滑窗平滑去 finite-diff 噪声
- [x] 第二次 run → B+C=3, C-only=3, dual-robot=2, drop=13
- [x] 可视化目检 (intent_detection_verify.png) — 3 个 B+C case 时序图
- [x] 写 log → `log/64_E054_case_tier_analysis_results.md`
- [x] 更新 EXPERIMENT_TRACKER.md (E054 行 + log/plan/script 引用)

## Claims 验证

| ID | 标准 | 实际 | 通过 |
|---|---|---|---|
| C1 | box023 → Tier 1 | dim=0.39m, pf=0.00, tier=1 | ✅ |
| C2 | box025 → Tier 3 | dim=0.89m, tier=3 | ✅ |
| C3 | ≥3 B-friendly | 3 (box023, bucket005_s2, desk021) | ✅ |
| C4 | ≥1 C-only | 3 (box021, bucket001, bucket007) | ✅ |
| C5 | csv 可复现 | 22 列 21 行完整 | ✅ |

**5/5 通过。**

## 关键发现

1. **box023 之前误归类**: 实际 dim=0.39m 是 Tier 1 (单臂可达), 之前与 box025 同处理浪费 5+ 实验
2. **3 个 B+C case 的 hand-obj 最近距离 19-28 cm** — 实测验证用户的判断: G1 retarget 后手与物体之间总有 20-28cm 几何间隙, mocap 中"接触"在 G1 几何上从未真正发生
3. **Path B 的 IK-snap 任务清晰**: 主动消除这 20-28cm 间隙
4. **Intent detection 设计教训**: 必须用相对速度 (搬运 ≈ 0) 而非绝对速度 (locomotion 主导)

## Path B 首批验证 case 排序

1. **box023_person1** (Tier 1, 53.2cm 抬升, 2 windows, dim 0.39m) ★★★ — 推荐 E055 起点
2. bucket005_s2_person1 (Tier 1, 40.6cm, 1 long window, dim 0.41m) ★★★
3. desk021_person1 (Tier 2, 42.4cm, 1 window, dim 0.51m) ★★

## 下一步

- E055 (Path B): box023_person1 上做 hand-snap warmstart + IK 投影到物体表面
- E056 (Path C): bucket001_person1 上做 force-closure reward (intent=0 但 dim 满足)

## 改动文件

| 类型 | 路径 |
|---|---|
| 新建脚本 | `workspace/core4d/scripts/analyze/case_tier_analysis.py` |
| 输出 csv | `workspace/core4d/results/E054/case_tier_classification.csv` |
| 输出 summary | `workspace/core4d/results/E054/E054_case_tier_summary.md` |
| 输出图 | `workspace/core4d/results/E054/intent_detection_verify.png` |
| 新建 plan | `workspace/core4d/plan/64_E054_case_tier_analysis_plan.md` |
| 新建 log | `workspace/core4d/log/64_E054_case_tier_analysis_results.md` |
| 阶段报告 | `workspace/core4d/report/STAGE_REPORT_E001-E053.md` |
| 路线图 | `workspace/core4d/report/NEXT_PLAN_E054+.md` |
| Phase 17 plan | `workspace/core4d/plan/63_phase17_post_E053_roadmap_plan.md` |
| 阶段 log | `workspace/core4d/log/63_E001_E053_stage_summary.md` |
| EXPERIMENT_TRACKER | 添加 E054 行 + 文件引用 |
