# E096 结果：box004 三条候选的 contact semantics + full CEM

日期：2026-05-29

对应计划：`workspace/core4d/plan/102_E096_box004_three_case_semantics_full_cem_plan.md`

## 1. 目标和结论

E096 承接 E095 的 first-batch box004 候选，目标是先用 E093/E094 口径做接触语义诊断，再对真正 preprocess-ready 的 case 跑 full CEM。

本轮结论：

- `box004_083_p1` 和 `box004_082_p1` 都完成 full CEM，并达到 `WORK`。
- `box004_082_p2` 不是 CEM fail，而是 OmniRetarget / preprocess blocker；no-fingertip 与 fingertip retry 都在 CVXPY solve 阶段 infeasible，因此没有 SPIDER trajectory，不能进入 CEM/RL。
- 两条 WORK case 的 contact semantics 与 E092/E094 的 box004 positive pattern 一致：`inside=0%`、support 可通过 adaptive-support projection 修到接近 `100%`，但实际 full CEM 仍使用 `ref_fk wrist5cm` target。
- high subagent 视觉复核支持 P1/P2 的 `WORK` 判定：未见倒地、趴箱、头/上身/手接地或明显穿箱。

## 2. Case Readiness

| id | source task | person | split | readiness | decision |
|---|---|---:|---|---|---|
| P1 | `e091_box004_20231003_2_083_p1` | 0 | local GPU0 | scene / trajectory / mask ready | run full CEM |
| P2 | `e091_box004_20231003_2_082_p1` | 0 | remote GPU0 | scene / trajectory / mask ready | run full CEM |
| P3 | `e091_box004_20231003_2_082_p2` | 1 | remote GPU1 | no scene / trajectory / mask after OmniRetarget infeasible | preprocess blocked |

P3 retry：

| setting | result |
|---|---|
| retry task | `e096_box004_20231003_2_082_p2_fingertip` |
| setting | `REPLACE_WRIST_WITH_FINGERTIP=1` |
| result | `RuntimeError: CVXPY solve failed: infeasible` around frame `81/139` |
| log | `/home/ubuntu/Workspace/holosoma/workspace/v3/data_construction_v2/logs/stage2b_medium_20260529_163812.log` |

所以 P3 保持 `preprocess_blocked`，没有 full CEM 指标，也不进入 Holosoma RL。

## 3. Contact Semantics

Contact geometry summary：

| case | hand | wrist5->raw mean | wrist5 inside | wrist5 support | sphere p90 | handbox p90 | 3-box p90 |
|---|---|---:|---:|---:|---:|---:|---:|
| `box004_083_p1` | L | 0.263 | 0.000 | 1.000 | 0.237 | 0.222 | 0.245 |
| `box004_083_p1` | R | 0.292 | 0.000 | 0.510 | 0.225 | 0.184 | 0.191 |
| `box004_082_p1` | L | 0.236 | 0.000 | 0.881 | 0.230 | 0.210 | 0.229 |
| `box004_082_p1` | R | 0.272 | 0.000 | 0.477 | 0.208 | 0.199 | 0.205 |

读法：

- `wrist_yaw_link + 5cm` 到 raw contact 的偏差仍是 `23-29cm` 量级，和 E093 结论一致：它不是精确 raw-contact proxy。
- 但这两条 case 的 `inside=0%`，且 support 不属于 Box026 那种大面积 wrong-face / inside 组合。
- 对 box004 pattern，偏差仍在 G1 reach 和 CEM 可处理范围内；这是它容易 work 的关键差异。

Adaptive-support projection 诊断：

| case | hand | old support | patch support | reward inside | reward delta p90 | gate |
|---|---|---:|---:|---:|---:|---|
| `box004_083_p1` | left | 100.0% | 100.0% | 0.0% | 0.000m | PASS |
| `box004_083_p1` | right | 51.0% | 100.0% | 0.0% | 0.000m | PASS |
| `box004_082_p1` | left | 88.1% | 99.1% | 0.0% | 0.000m | PASS |
| `box004_082_p1` | right | 47.7% | 99.1% | 0.0% | 0.000m | PASS |

注意：本轮 full CEM 没有使用 adaptive-support external target；projection 只是语义诊断，用来确认这两条 case 属于低风险 box004 pattern。

## 4. Full CEM 设置

执行脚本：

```bash
bash workspace/core4d/scripts/train/train_E096_box004_cem.sh local full 0
bash workspace/core4d/scripts/run_E096_remote.sh full
bash workspace/core4d/scripts/pull_E096_remote_results.sh full
bash workspace/core4d/scripts/train/train_E096_box004_cem.sh eval full 0
```

任务构造：

- P1/P2 派生 task 都加入 E083-style leg/foot-object 和 upper-body-object collision pairs，`npair=49`。
- object mass 都是 `5.0kg`。
- CEM target 仍是 `ref_fk + wrist5cm`，projection 结果没有自动接入 reward。
- remote GPU1 没有运行 P3 CEM；日志中明确记录 `No E096 variants for split=remote-gpu1; nothing to run.`，原因是 P3 preprocess blocked。

## 5. Full CEM 指标

| variant | case | T | contact | obj_mean | obj_max | pelvis | head | upper | LH floor | RH floor | status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `E096P1_box004_083_p1_cem` | `box004_083_p1` | 102 | 56.9% | 0.007m | 0.022m | 0.639m | 0.0% | 0.0% | 0.0% | 0.0% | WORK |
| `E096P2_box004_082_p1_cem` | `box004_082_p1` | 109 | 54.1% | 0.011m | 0.034m | 0.643m | 0.0% | 0.0% | 0.0% | 0.0% | WORK |

两个 case 都满足当前 full-CEM `WORK` 口径：

- object tracking 在厘米级；
- pelvis 保持约 `0.64m`，没有低髋/倒地；
- head / upper body / hand-floor 均为 `0%`。

## 6. Visual Review

高思考子代理复核结论：

- P1/P2 的 `WORK` 结论成立。
- CEM 视频未见明显倒地、趴箱、头/上身/手接地，和量化指标一致。
- 弯腰阶段较深，但更像取箱/扶箱动作，不是上身压箱。
- 未看到明显穿箱；手-箱接触主要在箱体上表面/侧面附近。
- geometry/projection marker 基本落在箱体上表面或侧边附近，未见 target 跑到脚、地面、头部或远离箱体。
- P2 搬箱语义更清楚；P1 画面较小但整体仍像 positive。

Review 文件：

`workspace/core4d/results/E096/visual_review/high_subagent_review.md`

## 7. 结果路径

| 内容 | 路径 |
|---|---|
| case manifest | `workspace/core4d/results/E096/contact_semantics/case_manifest.tsv` |
| contact geometry summary | `workspace/core4d/results/E096/contact_semantics/geometry_summary.md` |
| contact geometry videos | `workspace/core4d/results/E096/contact_semantics/visuals/mujoco/videos/` |
| adaptive-support projection summary | `workspace/core4d/results/E096/adaptive_support_projection/projection_summary.md` |
| adaptive-support projection videos | `workspace/core4d/results/E096/adaptive_support_projection/visuals/mujoco/videos/` |
| P3 retry summary | `workspace/core4d/results/E096/preprocess_retry/retry_summary.md` |
| CEM task metadata | `workspace/core4d/results/E096/cem/task_build_meta.json` |
| preprocess blockers | `workspace/core4d/results/E096/cem/preprocess_blockers.json` |
| full CEM summary | `workspace/core4d/results/E096/cem/full/full_eval_summary.md` |
| full CEM NPZ/MP4 | `workspace/core4d/results/E096/cem/full/` |
| full CEM keyframes | `workspace/core4d/results/E096/cem/full/keyframes/` |
| visual review sheets | `workspace/core4d/results/E096/visual_review/frame_sheets/` |
| high visual review | `workspace/core4d/results/E096/visual_review/high_subagent_review.md` |
| train/eval logs | `logs/E096/cem/full/` |
| remote logs | `logs/E096/remote/` |

## 8. Claims 验证

| Claim | 结果 | 证据 |
|---|---|---|
| C1: 新 box004 p1 cases 与 known WORK `083_p2` 同 pattern | PASS | `inside=0%`，projection gate PASS，full CEM 两条 WORK |
| C2: `083_p1` 和 `082_p1` 可完成 SPIDER full CEM | PASS | P1/P2 full summary 都是 WORK |
| C3: `082_p2` 的 full CEM 可行性由 preprocess 决定 | PASS | no-fingertip 与 fingertip retry 均 OmniRetarget CVXPY infeasible，无 trajectory |
| C4: 三卡并行不干扰现有 RL 进程 | PASS | 只启动 E096 新 tmux/remote wrapper；未 stop/kill 既有 RL |
| C5: 可视化足够支持结论 | PASS | contact/projection/CEM videos、frame sheets 和 high review 已保存 |

## 9. 决策和下一步

可进入下一阶段 Holosoma RL 候选：

1. `E096P1_box004_083_p1_cem`
2. `E096P2_box004_082_p1_cem`

同时保留已有 known-WORK control：

- `E092D1_box004_083_p2_dyn` / `E094P1_box004_083_p2_hbproj`

不进入下一阶段：

- `box004_082_p2`：preprocess blocked，无 SPIDER trajectory。

建议下一步：

1. 把 `083_p2`、`083_p1`、`082_p1` 作为 box004 positive set，进入 Holosoma RL 输入准备。
2. RL 输入分清楚来源：`SPIDER full-CEM WORK` 与 direct `OmniRetarget` baseline，避免把 preprocess reject 混入。
3. 对后续候选继续沿 E095 feature-based route 执行：先 target/posture gate，再 full CEM；不要按 object key 直接加权或降权。
