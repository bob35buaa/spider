# E106：Box026 30-candidate clean ref-FK batch 结果

计划：`workspace/core4d/plan/113_E106_box026_30candidate_ref_fk_batch_plan.md`

评测输出：
- `workspace/core4d/results/E106/cem/full/full_eval_summary.md`
- `workspace/core4d/results/E106/cem/full/full_eval_summary.csv`
- `workspace/core4d/results/E106/cem/full/full_eval_summary.json`
- `workspace/core4d/results/E106/cem/full/preprocess_rejects.csv`

## 1. 实验设置

E106 承接 E104 3cm Box026 候选，固定单一路线 `ref_fk_clean`，目标是用 E103 clean Box026 template 后的真实 batch CEM 重新筛选可进入后续 RL 的 case。

本轮按用户要求执行：
- 30 条 Box026 candidate 冻结；
- 2 条在 Phase0 OmniRetarget 预处理阶段明确失败；
- 28 条 clean derived task 通过 scene/qpos/pre-CEM MuJoCo visual gate；
- 本地 GPU0 跑 9 条，远程 GPU0 跑 9 条，远程 GPU1 跑 10 条；
- 所有 CEM 先完成，不提前 eval；
- 远程结果回收后统一 eval。

## 2. 数据就绪情况

30 条 candidate 中：
- `28/30` 成功补齐 Holosoma/OmniRetarget `retargeted + trimmed`、SPIDER `trajectory_kinematic.npz`、clean derived task；
- `2/30` 为上游预处理失败，不是 CEM 失败：

| source task | 变体 | 阶段 | 原因 |
|---|---|---|---|
| `e091_box026_20231020_137_p2` | `E106B08_box026_20231020_137_p2_ref_fk_clean` | OmniRetarget | `RuntimeError: CVXPY solve failed: infeasible` |
| `e091_box026_20231018_043_p2` | `E106B16_box026_20231018_043_p2_ref_fk_clean` | OmniRetarget | `RuntimeError: CVXPY solve failed: infeasible` |

所有 28 条 runnable case 的 pre-CEM MuJoCo replay 均已生成，并经 medium subagent 审查为 `PASS_WITH_NOTES`。

## 3. 执行过程

最终执行状态：

| 项目 | 数量 |
|---|---:|
| frozen candidates | 30 |
| preprocess rejects | 2 |
| runnable CEM variants | 28 |
| local GPU0 completed | 9 |
| remote GPU0 completed | 9 |
| remote GPU1 completed | 10 |
| pulled root NPZ | 28 |
| pulled MP4 | 28 |
| unified eval | done |

关键时间点：
- 2026-06-01 06:07 启动本地/远程 full CEM；
- 2026-06-01 10:31 本地 9/9 完成；
- 2026-06-01 11:48 远程最后一条 `E106B20` 完成；
- 2026-06-01 12:07 hardened wait monitor 自动 pull，并统一 eval。

## 4. 结果汇总

E106 的上半身/回放口径沿用 E105；lower-body strict 沿用 E026/E081 leg-box SDF proxy，要求 `leg_box_interference_frac <= 5%`。

| 指标 | 结果 |
|---|---:|
| upper-body WORK | 15/28 |
| upper-body FAIL | 13/28 |
| replay gate pass | 19/28 |
| lower-body strict pass | 7/28 |
| RL strict positive | 4/28 |
| upper WORK but lower strict FAIL | 11/28 |
| upper FAIL but lower strict pass | 3/28 |

Strict RL-ready positives：

| 变体 | source task | 接触率 | 物体均值误差 | pelvis 最低 | 腿部干涉 |
|---|---|---:|---:|---:|---:|
| `E106B05_box026_20231020_134_p1_ref_fk_clean` | `e091_box026_20231020_134_p1` | 32.4% | 0.0049m | 0.709m | 0.0% |
| `E106B15_box026_20231023_137_p1_ref_fk_clean` | `e091_box026_20231023_137_p1` | 78.9% | 0.0099m | 0.716m | 0.0% |
| `E106B22_box026_20231020_135_p1_ref_fk_clean` | `e091_box026_20231020_135_p1` | 46.3% | 0.0095m | 0.686m | 0.0% |
| `E106B27_box026_20231020_141_p1_ref_fk_clean` | `e091_box026_20231020_141_p1` | 30.6% | 0.0091m | 0.710m | 0.0% |

## 5. 结果解释

E106 的核心结论不是 "Box026 全部不可用"，而是：

1. E103 template 修复后，Box026 batch 里确实能找到 clean-scene ref-FK CEM positive。
2. 只看 upper-body WORK 会过于乐观：15 条 upper WORK 中有 11 条被 lower-body strict 拦下，主要是腿/脚/大腿与箱体干涉。
3. 下游 RL positive 应先采用 4 条 strict positive，而不是把 15 条 upper WORK 全部放入。
4. 两条历史 legacy-risk case 仍不 strict：
   - `E106B01` upper WORK，但 leg interference `26.8%`；
   - `E106B02` upper WORK，但 leg interference `17.1%`。
   因此 E105/E106 共同说明旧 Box026 失败不能再作为 polluted-template 算法证据，但这两条也仍不是 RL-ready positive。

## 6. 下一步

建议下一步只把 4 条 strict positive 进入 Holosoma/RL handoff，并把其余 upper WORK but lower FAIL 作为失败模式分析样本，重点看是否能通过 source selection、lower-body clearance reward、或 contact/pose gate 在数据阶段提前过滤。
