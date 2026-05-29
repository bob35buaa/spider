# E096b 结果：box004 mask-on full CEM rerun

日期：2026-05-29

对应计划：`workspace/core4d/plan/103_E096b_box004_mask_on_full_cem_plan.md`

## 1. 目标和结论

E096b 是对 E096 的最小 rerun：E096 P1/P2 虽然 full CEM 都是 `WORK`，但复核发现当时为了避免继承 box021 旧 mask，把 `contact_hdmi_mask_source/path` 清空，实际使用的是 run-time rotated-SDF fallback mask，而不是 CORE4D raw 3cm mask。

本轮只补上正确的 per-case `core4d_3cm` contact mask，其他 CEM/safety/scene 设置保持 E096 口径：

- P1 `e091_box004_20231003_2_083_p1`：`WORK`
- P2 `e091_box004_20231003_2_082_p1`：`WORK`
- P3 `e091_box004_20231003_2_082_p2`：仍沿用 E096 的 preprocess blocked 结论，本轮不重跑

结论：E096 的 box004 WORK 结论对 raw 3cm mask 接入是稳健的。以后引用 mask 配置正确的结果时，应优先引用 E096b。

## 2. 执行设置

执行方式：

```bash
bash workspace/core4d/scripts/train/train_E096b_mask_cem.sh local full 0
bash workspace/core4d/scripts/run_E096b_remote.sh full
bash workspace/core4d/scripts/pull_E096b_remote_results.sh full
bash workspace/core4d/scripts/train/train_E096b_mask_cem.sh eval full 0
```

资源分配：

| split | GPU | variant |
|---|---|---|
| local | GPU0 | `E096bP1_box004_083_p1_mask_cem` |
| remote | GPU0 | `E096bP2_box004_082_p1_mask_cem` |

本轮没有使用 remote GPU1，也没有 stop/kill 既有 RL 进程。

配置保持项：

- base override 仍继承 `core4d_E089A_box021_person1_upperobj`
- dynamic contact target 仍为 `ref_fk + wrist_yaw_link + 5cm EEF offset`
- 派生 task 仍使用 E083-style leg/foot-object + upper-body-object pairs，`npair=49`
- object mass 仍为 `5kg`
- full CEM 仍为 `32` iter，`video_camera=auto`

唯一关键变化：

| item | E096 | E096b |
|---|---|---|
| contact mask source | blank -> rotated-SDF fallback | `core4d_3cm` |
| contact mask path | blank | per-case `raw_contact_mask_3cm.npz` |

## 3. Mask Loading Evidence

CEM 日志确认两条 case 都加载了 CORE4D 3cm per-EEF mask：

| variant | mask path | loaded stats |
|---|---|---|
| `E096bP1_box004_083_p1_mask_cem` | `workspace/core4d/results/E096b/contact_masks/e091_box004_20231003_2_083_p1/raw_contact_mask_3cm.npz` | `len 170->254`, active L/R `61.8%/61.0%` |
| `E096bP2_box004_082_p1_mask_cem` | `workspace/core4d/results/E096b/contact_masks/e091_box004_20231003_2_082_p1/raw_contact_mask_3cm.npz` | `len 182->268`, active L/R `55.2%/56.0%` |

对应日志：

- `logs/E096b/cem/full/E096bP1_box004_083_p1_mask_cem_full.log`
- `logs/E096b/cem/full/E096bP2_box004_082_p1_mask_cem_full.log`

## 4. Full CEM 指标

| variant | case | T | contact | obj_mean | obj_max | pelvis | head | upper | LH floor | RH floor | status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `E096bP1_box004_083_p1_mask_cem` | `box004_083_p1` | 102 | 55.9% | 0.007m | 0.018m | 0.639m | 0.0% | 0.0% | 0.0% | 0.0% | WORK |
| `E096bP2_box004_082_p1_mask_cem` | `box004_082_p1` | 109 | 54.1% | 0.011m | 0.034m | 0.642m | 0.0% | 0.0% | 0.0% | 0.0% | WORK |

Run-time summary：

| variant | total time | final object tracking |
|---|---:|---|
| `E096bP1_box004_083_p1_mask_cem` | `900.565s` | pos `0.1278`, quat `0.1148` |
| `E096bP2_box004_082_p1_mask_cem` | `2723.981s` | pos `0.1356`, quat `0.3053` |

## 5. 与 E096 对比

| case | metric | E096 | E096b | delta |
|---|---|---:|---:|---:|
| `box004_083_p1` | contact | 56.9% | 55.9% | -1.0pp |
| `box004_083_p1` | obj_mean | 0.007m | 0.007m | ~0 |
| `box004_083_p1` | obj_max | 0.022m | 0.018m | -0.004m |
| `box004_083_p1` | pelvis | 0.639m | 0.639m | ~0 |
| `box004_083_p1` | safety | all 0% | all 0% | unchanged |
| `box004_082_p1` | contact | 54.1% | 54.1% | 0.0pp |
| `box004_082_p1` | obj_mean | 0.011m | 0.011m | ~0 |
| `box004_082_p1` | obj_max | 0.034m | 0.034m | ~0 |
| `box004_082_p1` | pelvis | 0.643m | 0.642m | -0.001m |
| `box004_082_p1` | safety | all 0% | all 0% | unchanged |

读法：

- P1/P2 的 status 都保持 `WORK`。
- raw 3cm mask 接入没有把 box004 positive pattern 改成失败，也没有引入 head/upper/hand-floor 安全问题。
- P1 contact 小降 `1.0pp`，但 object max error 反而略好；P2 几乎完全不变。

## 6. 可视化复核

本轮保存了 full CEM MP4、keyframes 和 frame sheets，并做了本地帧级视觉复核：

- `workspace/core4d/results/E096b/visual_review/frame_sheets/cem_p1_sheet.jpg`
- `workspace/core4d/results/E096b/visual_review/frame_sheets/cem_p2_sheet.jpg`
- `workspace/core4d/results/E096b/visual_review/manual_visual_review.md`

视觉结论：

- P1/P2 均支持 `WORK`。
- 未见明显倒地、趴箱、头/上身压箱、手撑地或明显穿箱。
- P2 后段箱体姿态变化比 P1 更明显，但没有违背 object error 和 safety 指标；仍应作为 positive case。

## 7. 结果路径

| 内容 | 路径 |
|---|---|
| E096b plan | `workspace/core4d/plan/103_E096b_box004_mask_on_full_cem_plan.md` |
| builder/scripts | `workspace/core4d/scripts/E096b/`, `workspace/core4d/scripts/train/train_E096b_mask_cem.sh`, `workspace/core4d/scripts/run_E096b_remote.sh` |
| mask copies | `workspace/core4d/results/E096b/contact_masks/` |
| overrides | `examples/config/override/core4d_E096bP1_box004_083_p1_mask_cem.yaml`, `examples/config/override/core4d_E096bP2_box004_082_p1_mask_cem.yaml` |
| full CEM summary | `workspace/core4d/results/E096b/cem/full/full_eval_summary.md` |
| full CEM JSON/CSV | `workspace/core4d/results/E096b/cem/full/full_eval_summary.json`, `workspace/core4d/results/E096b/cem/full/full_eval_summary.csv` |
| full CEM NPZ/MP4 | `workspace/core4d/results/E096b/cem/full/` |
| full CEM keyframes | `workspace/core4d/results/E096b/cem/full/keyframes/` |
| frame sheets | `workspace/core4d/results/E096b/visual_review/frame_sheets/` |
| visual review | `workspace/core4d/results/E096b/visual_review/manual_visual_review.md` |
| train/eval logs | `logs/E096b/cem/full/` |
| remote launch log | `logs/E096b/remote/remote_gpu0_full.log` |

## 8. Claims 验证

| Claim | 结果 | 证据 |
|---|---|---|
| C1: 只改变 mask source，不改变 CEM/safety/scene 主配置 | PASS | override 和派生 task 沿用 E096 设置，只补 `contact_hdmi_mask_source/path` |
| C2: P1/P2 实际加载 CORE4D 3cm per-EEF mask | PASS | 两条 CEM log 均出现 `E078 core4d_3cm per-EEF mask` |
| C3: mask-on full CEM 仍能达到 WORK 或暴露差异 | PASS | P1/P2 均 `WORK`，指标基本等价 E096 |
| C4: 并行执行不干扰已有 RL | PASS | local GPU0 + remote GPU0；未停止既有 RL |
| C5: 可视化支撑结论 | PASS | MP4/keyframes/frame sheets/manual review 已保存 |

## 9. 决策和下一步

E096b P1/P2 都进入 box004 positive set：

1. `E096bP1_box004_083_p1_mask_cem`
2. `E096bP2_box004_082_p1_mask_cem`

连同已有 known-WORK control：

- `E092D1_box004_083_p2_dyn` / `E094P1_box004_083_p2_hbproj`

建议下一步：

1. Holosoma RL 输入准备优先使用 E096b 的 P1/P2，因为它们是 mask-on 口径。
2. 对 direct OmniRetarget baseline 继续保留同三条 case 的对照，但不能把 P3 preprocess blocked 混入 SPIDER-CEM positive set。
3. 后续找更多数据时继续使用 E095 的 feature-based route：raw contact / reach / inside / support / CEM posture gate，而不是按 object key 固定加权。
