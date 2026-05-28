# E091 Results: Holosoma data_construction_v2 medium-box pipeline

日期：2026-05-29

关联计划：`workspace/core4d/plan/97_E091_holosoma_data_construction_v2_medium_boxes_plan.md`

目标：参考 `workspace/exp_diagnostic/data_filter_recommendation.md`，在 Holosoma `workspace/v3/data_construction_v2` 中寻找尺寸介于 Box023 与 Box025 之间的新物体，把数据构建链路跑通；不把本轮目标转成继续优化 CEM/reward。

## 结论

E091 已把 v2 数据链路跑到 top-bank 和 minimal SPIDER smoke。

当前可用 seed 是 `e091_box004_20231003_2_083_p2`：

- 物体：`box004`，体积约 `0.041 m^3`，略大于 Box023，显著小于 Box025。
- Stage2b：no-fingertip 全链路完成，trim 后 `105` 帧，SPIDER trajectory `(105,43)`，`scene_act nq/nv/nu=42/41/35`。
- D005b：PASS，inside `0/0%`，signed distance `+18.5/+14.1cm`，support-face either `42.9%`，pelvis min `0.679m`。
- Visual QC：high-reasoning subagent 判定 visually credible PASS，可作为 seed positive。
- Minimal smoke：运行完成，collision/object tracking 侧通过，但 pelvis collapse 严重；因此 smoke 记为 REVIEW/动态后续问题，而不是 final pass。

Box026 no-fingertip 同配置暂停扩量：

- `039_p2` Stage2b pass 但 D005b reject，support-face either `19.5% < 30%`。
- `040_p2` retarget CVXPY infeasible。
- `135_p2` Stage2b pass 但 D005b reject，right wrist inside `12.2% > 10%`。

## 结果表

### D005b / top bank

| rank | task | object | decision | D005b | key metric |
|---:|---|---|---|---|---|
| 1 | `e091_box004_20231003_2_083_p2` | box004 | top_candidate | PASS | inside max `0.0%`, support `42.9%`, pelvis `0.679m` |
| 2 | `e091_box026_20231020_135_p2` | Box026 | review_only | REJECT | right inside `12.2%` |
| 3 | `e091_box026_20231018_039_p2` | Box026 | review_only | REJECT | support `19.5%` |

### Minimal smoke

| variant | T | collision pass | obj mean | pelvis min | head | upper | hand-floor | final |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `E091S1_box004_20231003_2_083_p2` | 105 | true | `0.006m` | `0.079m` | `0.0%` | `0.0%` | `0.0/1.0%` | REVIEW / pelvis collapse |

## Claims 验证

| Claim | 结果 |
|---|---|
| C1: 旧 D001-D003 足够支持 C 路启动 | 通过。v2 manifest 生成 `80` 行，Box026/box004/Box022 分层明确；Stage2b backlog 与 ready rows 已固化。 |
| C2: D005b 能在动态训练前拦住几何不可行 case | 通过。Box026 两条 near-reject 被 gate 拦下；box004 通过。 |
| C3: medium box 不默认使用 Box025 fingertip reach hack | 通过。E091 Stage2b 默认 `REPLACE_WRIST_WITH_FINGERTIP=0`，Box025 hack 保持为条件 variant。 |
| C4: 目标是候选数据而不是继续调 reward | 通过。E091 停在 data/top-bank/smoke 证据；pelvis collapse 记录为 dynamics follow-up，不回滚数据筛选。 |
| C5: 可视化支撑人工判断 | 通过。raw contact、D005b overlay/timeline/keyframes、smoke keyframes 均生成并经 high subagent 复核。 |

## 关键产物

Holosoma v2:

- `inputs/medium_box_manifest.tsv`
- `inputs/cases_stage2b_box004_control_pipeline.tsv`
- `results/omniretarget_visuals/omniretarget_visual_manifest.tsv`
- `results/d005b_g1_feasibility/d005b_summary.tsv`
- `results/top_bank/top_medium_box_bank_manifest.tsv`
- `results/visual_qc/summary.md`
- `reports/high_subagent_d005b_review.md`

SPIDER:

- `workspace/core4d/scripts/E091/`
- `workspace/core4d/scripts/train/train_E091_smoke.sh`
- `workspace/core4d/scripts/eval/eval_E091.py`
- `workspace/core4d/results/E091/smoke/smoke_eval_summary.json`
- `workspace/core4d/results/E091/smoke/E091S1_box004_20231003_2_083_p2_smoke.mp4`

## OmniRetarget 可视化补充

用户指出不能只看 SPIDER/D005b 的可视化，OmniRetarget retargeted 结果本身也需要可视化。已补 `workspace/core4d/scripts/E091/make_omniretarget_visuals.py`，直接读取 Holosoma v2 Stage2b 的 `retargeted/*.npz` 和 `trimmed/*.npz`，用对应 source scene 渲染。

输出：

- `visualizations/omniretarget/*_retargeted_keyframes.png`
- `visualizations/omniretarget/*_trimmed_keyframes.png`
- `visualizations/omniretarget/*_omniretarget_timeline.png`
- `visualizations/omniretarget/*_retargeted.mp4`
- `results/omniretarget_visuals/summary.md`
- `results/omniretarget_visuals/omniretarget_visual_manifest.tsv`

覆盖情况：4 个 Stage2b 目录中，3 个成功 retarget 的 case 已生成可视化；`e091_box026_20231018_040_p2` 因 CVXPY infeasible 无 retargeted NPZ，manifest 标记为 `missing_retargeted_npz`。PNG 非空检查 `9/9` 通过，MP4 `3/3` 存在。

## High subagent 复核

D005b 复核结论：

- `box004` visually credible PASS；可进 top bank / minimal smoke。
- Box026 两条 reject 与 overlay/timeline 一致；不建议扩大同配置。

Smoke 复核结论：

- 判定为 REVIEW。
- 头/上身穿箱和手撑地不是主因，物体 tracking 稳定。
- 后段 pelvis/hip 明显塌陷，人体姿态不可信；这是一条 dynamics follow-up，而不是数据管线 blocker。

## 下一步

1. 不继续在 E091 内优化 CEM/reward。
2. 若下一轮要让 box004 smoke/full pass，应单独开姿态/站立约束实验，重点处理 pelvis collapse。
3. Box026 只保留有限 H2 variant：support-face-preIK / exterior projection；不要扩大 no-fingertip 同配置。
4. 若需要更多 medium seed，可补 `box004_person1` template 后跑第二条 box004 control，但不阻塞当前“找到一个 medium object 并跑通 v2 链路”的结论。
