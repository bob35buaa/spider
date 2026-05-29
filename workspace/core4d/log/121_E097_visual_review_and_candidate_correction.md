# E097 Addendum: visual review and candidate correction

日期：2026-05-29

对应计划：`workspace/core4d/plan/104_E097_feature_based_data_construction_v2_plan.md`
对应原日志：`workspace/core4d/log/120_E097_feature_based_data_construction_v2_results.md`

## 1. 目的

用户要求重新可视化 E097 当前列出的 Box021 case。执行过程中发现 E097 原先把这些 case 标成 `unverified candidate` 不够严谨：其中 5 条已经有 legacy D003/D004 retarget/visual QC 结果，另 1 条 `028_p1` 有 legacy D003 OmniRetarget CVXPY infeasible 记录。

本 addendum 做两件事：

1. 重新生成一套 E097 专属可视化，方便统一复查。
2. 修正 E097 miner，把这些 legacy case 从“新候选”队列移出。

## 2. 产出路径

| 内容 | 路径 |
|---|---|
| 可视化脚本 | `workspace/core4d/scripts/E097/make_e097_visuals.py` |
| 可视化 root | `workspace/core4d/results/E097/visual_review/` |
| 本次可视化输入 TSV | `workspace/core4d/results/E097/visual_review/visual_input_cases.tsv` |
| manifest | `workspace/core4d/results/E097/visual_review/visual_manifest.tsv` |
| raw-contact 图 | `workspace/core4d/results/E097/visual_review/raw_contact/` |
| MuJoCo keyframes | `workspace/core4d/results/E097/visual_review/mujoco_keyframes/` |
| MuJoCo videos | `workspace/core4d/results/E097/visual_review/mujoco_videos/` |
| retarget timelines | `workspace/core4d/results/E097/visual_review/timelines/` |
| video QC frames | `workspace/core4d/results/E097/visual_review/video_qc/` |

复现命令：

```bash
python workspace/core4d/scripts/E097/make_e097_visuals.py \
  --input workspace/core4d/results/E097/visual_review/visual_input_cases.tsv
```

注意：E097 miner 已修正，当前默认 pipeline TSV 已为空；因此复现本次 legacy visual review 时必须显式传入 `visual_input_cases.tsv`。

## 3. 可视化结果

| target | legacy status | raw contact | MuJoCo retarget/trimmed | 说明 |
|---|---|---|---|---|
| `e091_box021_20231018_028_p1` | `d003_infeasible_cvxpy` | OK | 无 | legacy D003 只有 converted/raw，缺 retargeted/trimmed NPZ |
| `e091_box021_20231018_028_p2` | `d004_visual_reject_fall_prone` | OK | OK | 已生成 retargeted + trimmed mp4/keyframes |
| `e091_box021_20231020_020_p2` | `d004_visual_pass` | OK | OK | 已生成 retargeted + trimmed mp4/keyframes |
| `e091_box021_20231011_035_p1` | `d004_visual_pass` | OK | OK | 已生成 retargeted + trimmed mp4/keyframes |
| `e091_box021_20231018_030_p2` | `d004_visual_pass_check_shortcut` | OK | OK | 已生成 retargeted + trimmed mp4/keyframes |
| `e091_box021_20231020_019_p2` | `d004_visual_review` | OK | OK | 已生成 retargeted + trimmed mp4/keyframes |

数量校验：

- manifest rows: `6`
- raw-contact PNG: `6/6`
- MuJoCo videos: `10/10` 可解码，分辨率 `1280x432`，fps `15`
- `028_p1` 无 MuJoCo video 是数据状态问题，不是渲染失败

## 4. 实际观察

MuJoCo panels 左半边使用 auto-framed full-body free camera，右半边保留 legacy `track2` 视角作对照。抽查 `e091_box021_20231020_020_p2` keyframe sheet 和 video frame 可见完整机器人与箱子，旧的“只看到上半身”问题在左半边 auto view 已规避。

抽取的 video QC frames：

- `workspace/core4d/results/E097/visual_review/video_qc/e091_box021_20231020_020_p2_retargeted_t04.jpg`
- `workspace/core4d/results/E097/visual_review/video_qc/e091_box021_20231018_028_p2_retargeted_t03.jpg`

视觉上这些是 legacy OmniRetarget/reference 可视化，不是 SPIDER full CEM 或 Holosoma RL 成功视频。部分帧仍呈现明显弯腰/趴箱倾向，因此不能把 D004 pass/review 直接等同于 “可进 RL positive”。

## 5. E097 miner 修正

更新文件：

- `workspace/core4d/scripts/E097/mine_feature_based_candidates.py`

新增 legacy outcomes：

- `028_p1`: D003 OmniRetarget CVXPY infeasible
- `028_p2`: D004 visual reject / fall-prone
- `020_p2`, `035_p1`, `030_p2`, `019_p2`: D004 pass/review/check-shortcut
- disabled 的 `030_p1`, `029_p1`: D004 review

重新运行 miner 后：

| 指标 | 修正后 |
|---|---:|
| candidate/audit rows | 38 |
| excluded verified rows | 22 |
| unverified candidate rows | 0 |
| enabled next-batch rows | 0 |

当前 Holosoma v2 input queue 已变为空表：

- `/home/ubuntu/Workspace/holosoma/workspace/v3/data_construction_v2/inputs/cases_e097_feature_candidate_pipeline.tsv`

## 6. 结论

E097 原先 6 条 Box021 enabled rows 应降级为 legacy visual review / diagnosis material，不应继续当成“新发现的 unverified candidates”。

如果下一步目标是继续找 box004 之外的 worklike 新数据，当前更合理的动作是：

1. 对 Box022 的 6 条 `raw_contact_preflight_disabled` 先补 D002/raw-contact preflight。
2. 或者扩大 inventory 到当前 old D001/D002 medium-box pool 之外。
3. Box026 继续作为 large-reach holdout，不混入 worklike next batch。
