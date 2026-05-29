# E098 Plan: 诊断基础设施（exp_diagnostic_v2 Stage 0）

日期：2026-05-30

## Context

本实验是 `workspace/exp_diagnostic_v2/diagnostic_v2_report.md` 整体计划的 Stage 0，对应整体计划文件 `/root/.cc-mirror/codewiz-cc/config/plans/eager-baking-boole.md`。

v2 报告核心结论：
- **B1**：`E017.audit_select_anchors.face_label` 等 5 个文件的 anchor 选择器只用 xy 二维 argmax，**完全屏蔽 ±z 面**。box021/box023 实测 8/9 真 top face 为 +z 的 hand-case 中 7 个被错判为 ±x/±y。E006-E028 的 weld 实验受 wrong-target 影响；E082-E097 阶段降级为「面统计污染」（让 anchor_face_review flag 失去诊断价值、让所有面统计被错读）。
- **B2/B3**：E028b `_project_to_face` 把 ±z 静默映射到 ±y；E029 D6 asset writer 把 ±z 默认回退 `+x`。当前因 B1 屏蔽 ±z 未触发，B1 修后立即变 wrong-target。
- **B4**：`spider/process_datasets/core4d.py:128` 写到 `trajectory_kinematic.npz` 的 `contact_pos` 是 IK FK palm site，不是 raw mocap；下游 E017-E028b 全部"raw contact"统计实际算的是 G1 wrist FK。v1 §3.2/§3.5 也踩了这个坑。
- **B5**：`anchor_face_review=true` flag 只是 informational，9/13 box021 D003 case 进入 E082-E088 full CEM 队列。
- **新失败模式（v1 H1 没预测到）**：E090 S1 full、E091 box004 smoke、E094 box026_039 都出现"safety penetration 全 0% 但 pelvis 塌 / 趴箱"。当前 head/upper/floor gate 测不到。

E098 目标：把上述 5 个 bug + 1 个新 gate 落地为代码 + 工具 + back-test 报告，作为后续 E099/E100/E101/E102 所有动作的"地基"。**不动 OmniRetarget 算法**，不跑训练。

## Claims

| ID | Claim | 验证方式 |
|---|---|---|
| **C1** | `historical_case_manifest.tsv` 列出 ~20 case，覆盖 box021 D003 6 个典型 + box021 person1 + box023/025/004/026 守门 + box022 候选；后续所有 stage 用同一份 | 行数 ≥ 18 ≤ 22；用户视觉签收 |
| **C2** | B1+B2+B3 修复后 face_label 输出全 3D（含 ±z）；box021 D003 6 个典型 case 的 anchor refit 把 `anchor_face_review=true` flag 全部清除 | 跑 face_utils 单元测试 6/6 PASS；跑 E017 anchor refit on 6 case，diff old vs new manifest，`anchor_face` 至少 4/6 变成 ±z |
| **C3** | B4 deprecation comment 在 `spider/process_datasets/core4d.py:128` 上方；v1 诊断报告 §3.2/§3.5 加 errata 引用 | grep 找到 comment；errata 块在 `workspace/exp_diagnostic/diagnostic_report.md` 末尾 |
| **C4** | B5 anchor_face_review=true 流程：在 face refit 流程中如果 manifest 读到该 flag 且未经过 refit 则 hard block；输出明确错误 | 写测试用 case：故意把 flag 设 true，调用读取函数，应 raise |
| **C5** | 新 CEM gate（pelvis_min_world_z < 0.40、pelvis_pitch > 60°、torso-on-box < 0.10 m）作为可选 detector 加入 `spider/simulators/mjwp.py` 或 eval 侧；至少能在已存档 rollout 上 replay 给出 boolean 输出 | 写 `workspace/core4d/scripts/E098/replay_gate.py`，能加载历史 rollout、按 case 输出三个 flag |
| **C6** | back-test：对 historical_case_manifest 中已有 rollout 存档的 case，跑新 gate replay，**E094 C2"趴箱"必须 FAIL，box004 E096b WORK 必须 PASS**；输出 `results/E098/backtest_gate_recall.tsv` | E094 C2 lie_on_box=True；E096b P1/P2 三个 gate 全 False |
| **C7** | 可视化：B1 修复前后 anchor 位置 3D 对比视频（6 个典型 box021 D003 case 各一段）；back-test 失败案例视频（至少 2 个） | mp4 文件存在且大小 > 100KB；用户视觉签收 |

## 改动文件

### spider 主仓库
- `workspace/core4d/scripts/E098/face_utils.py`（新增）— 统一 face 选择 helper，所有 caller 应 import 同一份。
- `workspace/core4d/scripts/E098/historical_case_manifest.tsv`（新增）— 历史 case manifest。
- `workspace/core4d/scripts/E098/test_face_utils.py`（新增）— 单元测试。
- `workspace/core4d/scripts/E098/replay_gate.py`（新增）— pelvis + lie-on-box gate replay 工具。
- `workspace/core4d/scripts/E098/run_anchor_refit_demo.py`（新增）— 在 6 个典型 case 上跑 anchor refit 前后对比。
- `workspace/core4d/scripts/E098/render_anchor_compare.py`（新增）— 3D 散点视频生成器。
- `workspace/core4d_collab_retarget/scripts/E017/audit_select_anchors.py` — `face_label`、`FACE_ORDER`、`_face_axis_sign`、`_clip_anchor`、`_snap_to_face` 改全 3D。
- `workspace/core4d_collab_retarget/scripts/E018b/generate_e018b_assets.py` — 同样修复。
- `workspace/core4d_collab_retarget/scripts/E020_audit/audit_common.py` — `_face_label` 改全 3D。
- `workspace/core4d_collab_retarget/scripts/E028b/build_e028b_manifest.py` — `_project_to_face` 修 B2。
- `workspace/core4d_collab_retarget/scripts/E029/generate_e029_d6_assets.py` — 修 B3 hard fallback。
- `spider/process_datasets/core4d.py` — 在 `contact_pos` 写入处加 deprecation comment（不改行为，避免破坏现有 npz）。
- `workspace/exp_diagnostic/diagnostic_report.md` — 末尾加 ERRATA 块，标注 §3.2/§3.5 的面解读受 B1/B4 污染。

### holosoma 仓库
本 E098 不涉及 holosoma 代码改动（face_label bug 全部在 spider 仓库的 core4d_collab_retarget 与 spider/process_datasets 中；OmniRetarget 本身无 face 概念）。Stage 1 (E099) 才进 holosoma。

## historical_case_manifest 范围（~20 case，box021 D003 取 6 典型）

| Row | case_name | box_family | last_known_outcome |
|---|---|---|---|
| 1 | `d003_box021_20231018_029_p2` | box021_d003 | v1 主战场，E082-E088 反复 FAIL；E089A 替换；H1 主验证 case |
| 2 | `d003_box021_20231011_035_p2` | box021_d003 | E090 S1 full：safety 全 0% 但 pelvis 0.134 m FAIL |
| 3 | `d003_box021_20231020_019_p1` | box021_d003 | E089B4 smoke pelvis 0.087-0.147 m |
| 4 | `d003_box021_20231018_030_p1` | box021_d003 | 02 §3 表 L+z 73% / R+z 67%，B1 强误判代表 |
| 5 | `d003_box021_20231020_020_p2` | box021_d003 | 02 §3 表 R+z 76%，最清晰 +z 主面 |
| 6 | `d003_box021_20231018_028_p2` | box021_d003 | E097 visual review D004 reject 代表 |
| 7 | `box021_person1` | box021_other | E089A 单 G1 WORK，H1 对照 |
| 8 | `box023_person1` | box023 | 守门 |
| 9 | `box023_person2` | box023 | 守门，多 case 反复用 |
| 10 | `box025_person1` | box025 | 守门 |
| 11 | `box025_person2` | box025 | 守门，侧抓 case，E090 反退化 |
| 12 | `e091_box004_20231003_2_083_p2` | box004 | E092/E094 WORK |
| 13 | `e091_box004_20231003_2_083_p1` | box004 | E096/E096b WORK |
| 14 | `e091_box004_20231003_2_082_p1` | box004 | E096/E096b WORK |
| 15 | `e091_box004_20231003_2_082_p2` | box004 | OmniRetarget CVXPY infeasible 代表 |
| 16 | `e091_box026_20231018_039_p2` | box026 | E094 C2 "趴箱" 关键证据 |
| 17 | `e091_box026_20231020_135_p2` | box026 | E094 C3 反退化代表 |
| 18 | `box026_person2` | box026 | 守门 / canonical |
| 19 | `box022_20231022_001_p2` | box022 | preflight_pending（v1 §6 一直没做）|
| 20 | `box022_20231022_127_p2` | box022 | preflight_pending |

Box022 行写入 manifest 但 `source_npz` 字段标 `pending_e099_or_e102`，下游消费时知道这两行只能进 mining/preflight stage，不能进 CEM。

## 成功标准

C1-C7 全部 PASS，且：
- 所有产出 commit 到 spider 仓库 + 推 origin。
- 写完整的 log 到 `workspace/core4d/log/122_E098_diagnostic_infrastructure_results.md`。
- 在 `workspace/core4d/EXPERIMENT_TRACKER.md` 追加一行。

## 不做的事

- 不跑任何 full CEM。
- 不动 OmniRetarget 代码（holosoma 仓库）。
- 不重新生成 trajectory_kinematic.npz（只在原数据上跑诊断 helper）。
- 不改 reward 配置 / override yaml。
- 不引入新 case 进入正式实验队列（E099 才做）。

## Pipeline

```
P1: 写 plan + historical_case_manifest.tsv  (本文档)
P2: 修 face_label (5 个 source 文件) + 写 face_utils.py + 单元测试
P3: B4 deprecation comment + v1 errata + B5 anchor_face_review hard block
P4: 写 pelvis + lie-on-box gate replay helper
P5: back-test (E094 C2 / E096b / E090 S1 / box023 守门) + 6 case anchor refit demo + 可视化
P6: 写 log + 更新 tracker + commit + push (spider 仓库；holosoma 本期无改动)
```
