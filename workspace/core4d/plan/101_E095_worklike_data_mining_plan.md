# E095 Plan: worklike medium-box data mining after Box026 failures

日期：2026-05-29

## Context

E092/E094 给出一个清楚分界：

- `box004_083_p2` full CEM 可 work；E094 `adaptive_support` 只改了 `2/210` 个 hand-frame，说明它本身已经属于 G1 可容忍的 positive pattern。
- `Box026_039_p2` 和 `Box026_135_p2` raw contact 很强，但 D005b 已经 reject；E092/E094 full CEM 均失败，失败模式是 large reach / wrong-face / inside-risk / low-hip posture。
- 因此下一步不应继续扩大 Box026 同类 case，而应回到 `data_construction_v2`，优先挖和 box004/box023 更接近的候选。

## Claims

| Claim | 验证方式 |
|---|---|
| C1: `data_construction_v2` 里还有未跑的 box004-like 候选 | 从 D001/D002 全量 inventory 重建 candidate bank，不能只看 E091 硬编码的 3 个对象 |
| C2: 大箱失败经验能转成筛选规则 | 对大体积、长边、low-support/inside 历史 reject 做 feature-based 延后；不按 object key 直接加权/降权 |
| C3: 至少生成一批可直接进入 Stage2b 的 first-batch worklike case | 输出 `cases_e095_box004_priority_pipeline.tsv`（历史文件名），先跑不带 fingertip hack 的 OmniRetarget/SPIDER 预处理 |
| C4: 备选候选需要分层而不是混跑 | worklike priority、target/posture gate、missing raw-contact long-edge review、large-reach dynamics holdout 分开记录 |

## Scoring Rules

优先级从高到低：

1. `box004` / box023-like 尺寸：体积接近 box004，长边不过大，无 Box026 reach risk。
2. D002 raw-contact pass，特别是 both-hand active 高、longest-run 高。
3. source scene template 是否存在只作为 pipeline readiness，不进入 worklike score；缺失时应补齐，而不是降低候选质量分。
4. recent failures 不进入数值 score，只进入实验队列/风险标签：
   - `large_reach_dynamics_holdout`：max edge / volume ratio 明显偏大，且已有大箱 full CEM 失败经验，必须走 separate repair route。
   - `target_posture_gate_review`：medium-large 几何或 aspect 较高，即使 raw-contact pass，也先做 target/posture gate，不进入第一批。
   - `missing_raw_contact_long_edge_review`：缺 D002 raw-contact 且长边偏大，先补 raw-contact/reach review。

备注：`score` 只包含几何尺寸和 raw-contact 强度；`source_scene_exists` 不加分，object key 不做数值加权/降权。object key 只保留在表里做 traceability；执行队列的 tier 必须由 feature/risk route 解释。

## Execution

1. 新增并运行 `workspace/core4d/scripts/E095/mine_worklike_candidates.py`。
2. 输出到：
   - `workspace/core4d/results/E095/worklike_candidate_mining/`
   - `/home/ubuntu/Workspace/holosoma/workspace/v3/data_construction_v2/results/e095_worklike_candidates/`
   - `/home/ubuntu/Workspace/holosoma/workspace/v3/data_construction_v2/inputs/cases_e095_box004_priority_pipeline.tsv`
3. 为 `box004_person1` 补 source scene template。
4. 跑第一批 first-batch worklike Stage2b：
   - `e091_box004_20231003_2_083_p1`
   - `e091_box004_20231003_2_082_p1`
   - `e091_box004_20231003_2_082_p2`
5. 跑 OmniRetarget 可视化和 verify summary；只把预处理/可视化结果记为 E095，不在本轮直接跑 CEM full。

## Success Criteria

| 项 | 标准 |
|---|---|
| candidate bank | TSV/JSON/MD 均生成，包含 all raw-contact-pass boxes 和 review rows |
| first-batch case file | 只启用 worklike priority 新 case，不启用 large-reach holdout |
| template | `box004_person1/scene.xml` 可被 MuJoCo load |
| Stage2b | 新 case 至少有 retargeted/trimmed NPZ；SPIDER `verify_summary.json` 通过 |
| visual | OmniRetarget keyframes/timeline/mp4 非空 |

## Stop Rules

- 如果 first-batch retarget 出现 CVXPY infeasible，先记录，不扩大到 target/posture gate 或 missing raw-contact review。
- 如果 first-batch worklike 全部预处理通过，下一轮再选 2-3 条跑 SPIDER full CEM。
- large-reach dynamics holdout 不进入第一批；除非后续专门做 posture/target repair。
