# E018b 结果：canonical support proxy 13-case generalization

日期：2026-05-19

## 初始目标

在 E018 两个 GT case 通过后，将 canonical support proxy 扩展到 E016 的 13 个 case，并使用本地 1 卡 + 远程 2 卡并行 full run。

本轮不再依赖离线 replay 视频；每个 case 使用 `run_mjwp.py video_output_path` 直接生成在线 rollout 视频。

## 结果路径

| 产物 | 路径 |
|------|------|
| Plan | `workspace/core4d_collab_retarget/plan/19_E018b_canonical_support_proxy_13case_plan.md` |
| Results | `workspace/core4d_collab_retarget/results/E018b/` |
| Online videos | `workspace/core4d_collab_retarget/results/E018b/online_video/` |
| Logs | `logs/core4d_collab_retarget/E018b/` |
| Manifest | `workspace/core4d_collab_retarget/results/E018b/manifest.tsv` |

## 执行状态

- [x] 实现 E018b assets / overrides / train / eval / online video index。
- [x] 运行 preprocess + smoke。
- [x] 3 卡 full run。
- [x] 回收远程结果并统一 eval。
- [x] 在线视频 sheet 观察。

## 量化汇总

评测脚本：`workspace/core4d_collab_retarget/scripts/eval/eval_E018b.py --all`

| 指标 | 结果 |
|------|------|
| full results | 13/13 |
| config ok | 13/13 |
| canonical anchor pass | 13/13 |
| GT anchor available/pass | 2/2 |
| GT gate pass | 2/2 |
| SPIDER object success | 13/13 |
| DynaRetarget object success | 13/13 |
| transport success | 13/13 |
| OmniRetarget contact preservation ok | 5/13 |
| deep penetration ok | 7/13 |
| robot upright ok | 9/13 |
| robot fall detected | 4/13 |
| visual stability ok | 9/13 |
| strict artifact ok | 1/13 |
| strict generalization pass | 1/13 |
| mean Epos / Erot | `0.0545m` / `5.22deg` |
| mean contact preservation 5cm | `54.35%` |
| mean deep penetration duration | `30.04%` |
| mean carry progress ratio | `1.004` |
| max GT anchor dist | `0.0117m` |

诊断分布：

| diagnostic | count |
|------------|-------|
| `paper_generalization_pass` | 1 |
| `robot_fall_visual_fail` | 4 |
| `contact_preservation_gap` | 5 |
| `artifact_failed` | 2 |
| `push_or_leg_shortcut` | 1 |

## Case 表

| Variant | diag | fall | pelvis min | Epos | Erot | contact 5cm | deep pen | leg intf | GT |
|---------|------|------|------------|------|------|-------------|----------|----------|----|
| `box021_p1` | `robot_fall_visual_fail` | yes | 0.126 | 0.091 | 11.4 | 71.7 | 53.1 | 35.2 | - |
| `box021_p2` | `robot_fall_visual_fail` | yes | 0.260 | 0.074 | 7.2 | 10.3 | 0.7 | 18.4 | - |
| `box023_p1` | `contact_preservation_gap` | no | 0.654 | 0.043 | 2.0 | 22.5 | 0.0 | 0.0 | - |
| `box023_p2` | `contact_preservation_gap` | no | 0.685 | 0.042 | 2.3 | 28.6 | 3.3 | 0.0 | pass |
| `box025_p1` | `contact_preservation_gap` | no | 0.754 | 0.072 | 5.1 | 66.1 | 5.1 | 12.9 | - |
| `box025_p2` | `paper_generalization_pass` | no | 0.760 | 0.056 | 1.9 | 86.9 | 0.0 | 0.0 | pass |
| `bucket001_p1` | `robot_fall_visual_fail` | yes | 0.145 | 0.033 | 2.7 | 0.0 | 0.0 | 0.0 | - |
| `bucket001_p2` | `robot_fall_visual_fail` | yes | 0.439 | 0.038 | 7.7 | 66.3 | 64.6 | 8.1 | - |
| `bucket005_s2_p1` | `push_or_leg_shortcut` | no | 0.688 | 0.039 | 4.4 | 97.6 | 88.2 | 23.7 | - |
| `bucket005_s2_p2` | `artifact_failed` | no | 0.645 | 0.059 | 8.2 | 95.9 | 74.4 | 9.9 | - |
| `bucket007_p1` | `artifact_failed` | no | 0.767 | 0.065 | 9.1 | 79.0 | 68.5 | 2.0 | - |
| `bucket007_p2` | `contact_preservation_gap` | no | 0.742 | 0.048 | 4.2 | 29.7 | 18.4 | 5.9 | - |
| `desk021_p1` | `contact_preservation_gap` | no | 0.722 | 0.048 | 1.7 | 52.0 | 14.3 | 0.5 | - |

## Claims 验证

| Claim | 结果 |
|-------|------|
| C1 13/13 canonical anchor 生成 | 通过：13/13 `E018b_canonical_anchor_pass` |
| C2 GT/template case 保持 E018 语义 | 通过：`box023_p2` dist `1.17cm`、`box025_p2` dist `0.94cm`，2/2 `E018b_gt_gate_pass` |
| C3 无 GT case 不使用 palm median point | 通过：13/13 `anchor_policy=canonical_face_center_upper`，face 来自 E017 audit 但 point 固定为 face center + upper band |
| C4 三卡并行执行 | 通过：本地 5 条，远端 GPU0 4 条，远端 GPU1 4 条 |
| C5 在线视频覆盖 13/13 | 通过：13/13 online MP4 + 13/13 sheet，均为 `1440x480 @ 50fps` |
| C6 paper-aligned 分层评测 | 通过：输出 SPIDER/DynaRetarget/OmniRetarget 风格指标和 diagnostic class；结论显示 object/transport pass、robot fall gate 与 robot-side artifact pass 明确分离 |

## 可视化观察

在线视频索引：`workspace/core4d_collab_retarget/results/E018b/online_video/online_video_eval.md`

实际观察：

- `box023_p2`：online 视频中箱体没有出现离线 replay 曾见的“飘走”；sim 和 ref 的 object trajectory 基本同步，GT anchor gate 通过。但 robot 接触保持仍差，contact preservation 只有 `28.6%`，所以 strict generalization 未通过。
- `box025_p2`：视觉最稳定，人物与大箱保持贴近，量化为唯一 `paper_generalization_pass`；contact preservation `86.9%`，deep penetration 与 leg interference 均为 `0.0%`。
- `box021_p1/p2` 与 `bucket001_p1/p2`：object tracking 指标过，但视频中 robot 明显坐倒/翻倒式姿态；复核后统一归类为 `robot_fall_visual_fail`。对应 pelvis minimum 分别为 `0.126m`、`0.260m`、`0.145m`、`0.439m`，均触发 `pelvis_z_min < 0.45m` 或首次低于 `45cm` 的 fall gate。
- `bucket005_s2_p1/p2` 与 `bucket007_p1`：bucket/腿脚附近出现明显不自然接触或穿插，deep penetration duration 高，符合 `artifact_failed` 或 `push_or_leg_shortcut`。
- `desk021_p1`：object tracking 好、腿干涉低，但手-物 contact preservation 仍未达 strict gate，诊断为 `contact_preservation_gap`。

## 结论

E018b 证明 canonical support proxy 的 anchor 生成与 object-side transport 泛化成立：13/13 通过 config、anchor、SPIDER object、DynaRetarget object 与 transport gate，且两个 E014 GT case 均通过 GT anchor gate。

但 object-side success 不能等同于完整 retargeting 成功。复核在线视频后，`box021_p1/p2` 与 `bucket001_p1/p2` 四个 case 明确摔倒，已新增 robot fall gate 并计为失败；严格 generalization 仍只有 1/13。下一步不应继续改 anchor point 本身，而应把问题转到 robot-side artifact control：接触保持、躯体稳定、腿/脚与物体碰撞惩罚或姿态约束。

## 遇到的错误

| 错误 | 尝试次数 | 解决方案 |
|------|---------|----------|
| 远端/本地在线渲染结束时 EGL destructor warning | 多次 | `trajectory_mjwp.npz` 和 MP4 已保存，不影响结果；评测以 root NPZ 和 online MP4 为准 |
| 运行中同步 `train_E018b.sh` 后，本地 bash 队列末尾出现 `unexpected EOF` | 1 | 产物已完整落盘；当前脚本 `bash -n` 正常。根因是运行中的 bash 按旧文件偏移读取被修改过的脚本；后续避免在队列运行中覆盖入口脚本 |
| 部分 derived task 缺 `task_info.json` | 多次 warning | 与 E016 同口径，使用默认 ref_dt；不影响 E018b anchor/config gate，但记录为复现注意事项 |
