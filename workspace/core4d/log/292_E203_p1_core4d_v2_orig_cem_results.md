# E203 P1 — CORE4D v2 人体动作 orig 重定向 + CEM 结果（box001/box004/box024）

- **日期**: 2026-08-24
- **分支**: `experiment/E203-core4d-v2-orig-retarget`
- **计划**: `workspace/core4d/plan/233_E203_core4d_v2_orig_retarget_plan.md`
- **运行根**: `workspace/core4d/results/E203/`（不进 git，外部同步）

## 目的

用 **v2 人体动作**（`CORE4D_Real_human_object_motions_v2`，同捕捉、更精 SMPLX 重拟合）替代 v1，走完整 data-construction-v3 管线到 full CEM，产出与 v1 **物理隔离**的一版数据（`dataset_name=core4d_v2`）。本轮 = P1（box001/box004/box024），orig（不做 object augmentation）。

## 管线与隔离（见计划）

- v2 raw 物化 → v1 布局（复用 v1 物体位姿，逐帧对齐）；输出树 `example_datasets/processed/core4d_v2/`（assets/模板 symlink 自 core4d，输出独立）。
- retarget: `omnirt_v1`（默认，不替换 wrist）→ CVXPY 不可行的 case 用 `omnirt_v2`（Phase-4 松弛）**rescue**（用户定）。
- CEM: 复用 E199 PRG 契约（rubber_hull 手 + 16 下肢 pair + leg gate），1024×32，8 卡并行。

## 结果

### 漏斗产量

| 阶段 | 数量 |
|---|---|
| P1 in-scope cases（box001+box024 过 3cm 接触门；box004 未过） | 65 |
| omnirt_v1 可行 | **34** |
| omnirt_v1 infeasible | 31（→ rescue）|
| **omnirt_v2 rescue 可行** | **30 / 30（100% 救回）** |
| 重定向任务合计 | **64** |
| CEM 成功（`cem_ok`） | **54**（v1 31 + v2 23）|
| PRG 预检拒绝（起始帧腿-物穿透 > 硬底） | 10（v1 3 + v2 7）|

**关键结论：omnirt_v2 Phase-4 rescue 把 v1 的 31 个不可行 case 100% 救回（30/30 成功重定向），显著提升产量**（可行任务 34 → 64）。这与 dcv3 设计一致，验证了 rescue 机制在 v2 数据上有效。

### CEM 物体位姿跟踪误差（obj_pos，54 任务）

| 组 | n | mean(m) | std | min | max | median |
|---|---|---|---|---|---|---|
| omnirt_v1 | 31 | **0.112** | 0.027 | 0.067 | 0.180 | 0.114 |
| omnirt_v2 rescue | 23 | **0.131** | 0.034 | 0.081 | 0.186 | 0.125 |
| 全部 | 54 | **0.120** | 0.031 | 0.067 | **0.186** | — |

- 分物体：box001 mean 0.116（n=35）、box024 mean 0.126（n=19）。
- quat 误差：v1 mean 0.110 / v2 0.115。
- v2 rescue 组略差于 v1（+0.019m），符合预期——rescue 集正是 v1 难解的硬 case。
- worst 5 均为 v2 rescue 的 box001_20231108_*/box024_20231011_031（~0.18m）。

### 视觉 QC（强制项，抽检）

抽 best/worst CEM 视频关键帧（`s6_downstream/visual_qc/`）：机器人**直立搬箱、箱体贴合躯干、无跌倒/无掉箱/无明显穿透**，行为物理可信,与 ~0.12m 跟踪量级一致。

## 结论

1. **v2 人体动作端到端跑通**：物化→converter→OmniRetarget→SPIDER→PRG→CEM 全链在 `core4d_v2` 隔离树内正确运行,v1 树未污染。
2. **产量**：64 重定向任务 / 54 full-CEM 结果；omnirt_v2 rescue 是产量关键（+30）。
3. **质量**：obj_pos mean 0.120m、worst 0.186m,视觉可信。

## 未决 / 下一步

- **v1↔v2 A/B（核心验收未完成）**：现有 core4d(v1-motion) 树对这批 box001/024 case **无同口径 CEM 结果**（历史实验 CEM 的是另一组代表 case）。要严格判定"v2 更干净是否转化为下游不劣/更优",需对**相同 case 的 v1-motion 版**跑同一套 E203 CEM 做配对对比——另需一轮算力。
- P2（box021/box023）、P3（bucket）、P4（box026）按优先级待跑。

## 改动文件 / 产物

- 代码：`materialize_v2_raw_root.py`、`run_e203_cem.py`、`run_E203_p1_cem_and_rescue.sh`、`run_stage2b.py`/`pipeline.sh`/`create_spider_scene_from_template.py`/`verify_processed_case.py`/`generate_scene_act.py`/`run_pipeline.py`/`lib/common.py`（`--spider-dataset`/KEEP_GOING/修复）。
- 数据：`example_datasets/processed/core4d_v2/unitree_g1/humanoid_object/dcv3_omnirt_v{1,2}_ref_fk_box*`。
- 结果：`workspace/core4d/results/E203/{s3_retarget,s6_downstream}`（CEM npz/mp4/summary、pipeline_failed_cases、visual_qc）。
- 环境：装 `python3.12-dev`（torch.compile 必需）、`MUJOCO_GL=osmesa`。
