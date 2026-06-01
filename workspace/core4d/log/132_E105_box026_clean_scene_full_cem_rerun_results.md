# E105：Box026 clean-scene full CEM 重跑结果

日期：2026-06-01
计划：`workspace/core4d/plan/112_E105_box026_clean_scene_full_cem_rerun_plan.md`
上游：E103 scene rebuild / target regeneration；E104 D002 remine

## 目标

E103 发现旧 Box026 派生 scene 继承了 Box021 robot inertial 污染，导致历史 Box026 full CEM 结论不可信。E105 从 E103 clean target 重新构造 clean derived task，重跑历史已跑过的 Box026 full CEM，并补充 E101-style fingertip target ablation。

## 数据与 pre-CEM gate

| 项目 | 结果 |
|---|---|
| clean derived tasks | `e091_box026_20231018_039_p2_e105_clean`, `e091_box026_20231020_135_p2_e105_clean` |
| clean preflight | `workspace/core4d/results/E105/clean_scene_preflight.tsv` PASS；旧 `_e092_dyn/_e092_omni` 标为 `INVALIDATED_OLD_DERIVED` |
| target sources | adaptive/fingertip targets 均重新写入 `workspace/core4d/results/E105/` |
| pre-CEM visual gate | 6/6 MuJoCo replay + target sheet generated |
| medium subagent review | 6/6 `PASS_WITH_NOTES` |

## 运行

按本地 1 卡 + 远程 2 卡执行。由于 R1 比另外两条 wave1 慢，后半程改为 sliding window，保持最多 3 条 E105 CEM 同时跑。

| 变体 | 路线 | 位置 | 结果 |
|---|---|---|---|
| `E105R1_box026_039_p2_ref_fk_clean` | ref-fk | remote GPU0 | complete |
| `E105R2_box026_135_p2_ref_fk_clean` | ref-fk | remote GPU1 | complete |
| `E105A1_box026_039_p2_adaptive_clean` | adaptive | local GPU0 | complete |
| `E105A2_box026_135_p2_adaptive_clean` | adaptive | remote GPU0 | complete |
| `E105F1_box026_039_p2_fingertip_clean` | fingertip | local GPU0 | complete |
| `E105F2_box026_135_p2_fingertip_clean` | fingertip | remote GPU1 | complete |

Notes:

- 远程第一次 wave1 启动暴露两个同步问题：缺 E089 override 依赖链、缺 `box026_m.obj`，均已补齐后重启。
- 远程 eval 缺 `eval_E090.py`，已改为远程只跑 CEM、最终本地统一 eval；不影响已落盘的 CEM 结果。
- 本地 F1 第一次 `nohup` 启动早退且 CEM log 为空；用 `single` 模式重启后正常完成。

## 量化结果

统一评估路径：

- `workspace/core4d/results/E105/cem/full/full_eval_summary.csv`
- `workspace/core4d/results/E105/cem/full/full_eval_summary.md`
- `workspace/core4d/results/E105/comparison/box026_clean_vs_old_comparison.md`

| variant | route | T | contact | obj mean/max | pelvis min | safety | replay gate | status |
|---|---|---:|---:|---:|---:|---|---|---|
| `E105R1_box026_039_p2_ref_fk_clean` | ref-fk | 123 | 76.4% | 0.010 / 0.032m | 0.704m | 0% head/upper/floor | PASS: lie 24.4% | WORK |
| `E105R2_box026_135_p2_ref_fk_clean` | ref-fk | 82 | 62.2% | 0.009 / 0.028m | 0.655m | 0% head/upper/floor | PASS: lie 0.0% | WORK |
| `E105A1_box026_039_p2_adaptive_clean` | adaptive | 123 | 78.0% | 0.010 / 0.045m | 0.703m | 0% head/upper/floor | PASS: lie 11.4% | WORK |
| `E105A2_box026_135_p2_adaptive_clean` | adaptive | 82 | 30.5% | 0.009 / 0.034m | 0.672m | 0% head/upper/floor | FAIL: lie 30.5% >= 30% | FAIL |
| `E105F1_box026_039_p2_fingertip_clean` | fingertip | 123 | 82.1% | 0.010 / 0.032m | 0.709m | 0% head/upper/floor | FAIL: lie 30.1% >= 30% | FAIL |
| `E105F2_box026_135_p2_fingertip_clean` | fingertip | 82 | 61.0% | 0.009 / 0.029m | 0.656m | 0% head/upper/floor | PASS: lie 0.0% | WORK |

补充 E026/E081 lower-body strict proxy 后的腿/脚-箱体干涉：

| variant | lower-body strict | leg interference | leg contact | min leg SDF | argmin geom |
|---|---|---:|---:|---:|---|
| `E105R1_box026_039_p2_ref_fk_clean` | FAIL | 25.2% | 25.2% | -0.010m | `rf2` |
| `E105R2_box026_135_p2_ref_fk_clean` | FAIL | 15.9% | 15.9% | -0.007m | `lf3` |
| `E105A1_box026_039_p2_adaptive_clean` | FAIL | 60.2% | 60.2% | -0.026m | `left_shin_collision` |
| `E105A2_box026_135_p2_adaptive_clean` | FAIL | 9.8% | 9.8% | -0.012m | `right_shin_collision` |
| `E105F1_box026_039_p2_fingertip_clean` | FAIL | 27.6% | 27.6% | -0.014m | `rf2` |
| `E105F2_box026_135_p2_fingertip_clean` | FAIL | 18.3% | 18.3% | -0.009m | `lf3` |

口径说明：`leg interference` 继承 E026/E081 的 `leg_box_sdf_min_m < 0` 帧比例；strict proxy 阈值为 `<=5%`。它不是 GT 标签，而是 MuJoCo collision geometry 到 object box 的 SDF/接触代理，用于防止靠腿/脚顶箱或穿箱 shortcut。

Historical comparison:

| old | new | old status | new status | key delta |
|---|---|---|---|---|
| `E092D2_box026_039_p2_dyn` | `E105R1` | FAIL | WORK | pelvis `0.083 -> 0.704m`, contact `33.3 -> 76.4%` |
| `E092D3_box026_135_p2_dyn` | `E105R2` | FAIL | WORK | pelvis `0.177 -> 0.655m`, RH floor issue removed |
| `E094P2_box026_039_p2_hbproj` | `E105A1` | FAIL | WORK | pelvis `0.440 -> 0.703m`, contact remains high |
| `E094P3_box026_135_p2_hbproj` | `E105A2` | FAIL | FAIL | old low-pelvis/RH-floor mode removed (`0.171 -> 0.672m`), but strict replay gate still fails by body-on-box `30.5% >= 30%` |

## 可视化

Visual package:

- `workspace/core4d/results/E105/visuals/box026_clean_rerun/REVIEW.md`
- `workspace/core4d/results/E105/visuals/box026_clean_rerun/sheets/`
- `workspace/core4d/results/E105/visuals/box026_clean_rerun/timelines/`

实际观察：

- 6 张 old-vs-new sheet 和 6 张 timeline PNG 均非空；早/中/晚帧能看到机器人和 Box026 物体，视频不是空渲染或静止帧。
- `039_p2` 的 ref-fk/adaptive clean rerun 姿态稳定并通过 replay gate；fingertip route contact 最高，但 `lie_on_box_frac=30.1%`，按 E098 `>=30%` 阈值严格判 FAIL。
- `135_p2` 的 ref-fk/fingertip contact 明显稳定并通过 replay gate；adaptive route contact 只有 30.5%，且 `lie_on_box_frac=30.5%`，严格判 FAIL。
- lower-body strict proxy 发现 6/6 都有超过 5% 的腿/脚-箱体干涉；其中 `E105A1` 最严重，`left_shin_collision` 最小 SDF `-2.6cm`，60.2% 帧干涉。
- 历史 old-vs-new sheet 显示 clean rerun 不再出现旧结果中的低髋/贴地/右手地面风险模式；剩余失败转为边界 body-on-box gate，而不是 E092/E094 的旧失败模式。

## 验证声明复核

| claim | status | evidence |
|---|---|---|
| C1 clean-scene gate enforced | PASS | clean preflight + pre-CEM medium review 6/6 PASS_WITH_NOTES |
| C2 historical full-CEM variants exactly aligned | PASS | R1/R2 对齐 E092D2/D3；A1/A2 对齐 E094P2/P3 |
| C3 all E105 variants complete | PASS | 6/6 root NPZ, outdir NPZ, MP4, keyframes, eval |
| C3b pre-run MuJoCo visual + medium subagent | PASS | `pre_cem_visual_review/{variant}/REVIEW.md` 6/6 |
| C4 numerical comparison answers old conclusion | PASS / mixed | 3/4 historical old FAIL -> clean WORK；`E105A2` 旧低髋/右手贴地模式被修复，但严格 replay gate 因 `lie_on_box_frac=30.5%` 仍 FAIL |
| C5 visual comparison sufficient | PASS | sheets/timelines/REVIEW generated and nonblank |
| C6 lower-body strict proxy added | FAIL for RL-ready | E026/E081 leg interference proxy 6/6 >5%，因此 0/6 lower-body strict |

## 结论

E105 明确推翻了旧 Box026 full CEM 的低髋/贴地失败解释：在 E103 clean scene 下，历史 4 条 primary rerun 中 3 条在 upper-body/replay 口径变为 WORK，剩余 `E105A2` 的旧低髋/RH-floor 失败模式也消失，但因 `lie_on_box_frac=30.5%` 触发严格 replay gate，不能算 WORK。因此旧 Box026 FAIL 不能继续作为算法失败证据。

但补上 E026/E081 lower-body strict proxy 后，E105 不能作为 RL-ready positive：6/6 都有超过 5% 的腿/脚-箱体干涉。E101-style fingertip target 的结果也不是无条件更好：`135_p2` 在 upper-body/replay 口径为 WORK，`039_p2` 虽 contact 最高但因 `lie_on_box_frac=30.1%` 严格 FAIL；两者 lower-body strict 都 FAIL。后续如果扩到 E104 Box026 候选，应以 clean E105 数据链为前提，同时把 lower-body/object interference 作为硬指标纳入选 case 和 route 评估。
