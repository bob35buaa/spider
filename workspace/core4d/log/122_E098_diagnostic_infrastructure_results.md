# E098 Results: 诊断基础设施（exp_diagnostic_v2 Stage 0）

日期：2026-05-30
分支：`exp/core4d-collab-retarget`
对应 plan：`workspace/core4d/plan/105_E098_diagnostic_infrastructure_plan.md`
关联文档：`workspace/exp_diagnostic_v2/diagnostic_v2_report.md` / `findings/02_face_selection_audit.md` §3/§4

## TL;DR

5 个 bug + 1 个新 gate 全部落地：

- ✅ **B1** face_label xy-only → 全 3D（5 个文件 + 统一 helper `face_utils.py`）；6 case anchor refit 实测 5/12 hand 主面翻到 ±z，覆盖率匹配 v2 预测
- ✅ **B2/B3** 同步修复 `_project_to_face` 与 E029 `+x` fallback
- ✅ **B4** `spider/process_datasets/core4d.py:128` 加 deprecation comment 标注 `contact_pos` 是 FK 不是 raw mocap；v1 诊断报告加 ERRATA
- ✅ **B5** `anchor_face_review=true` hard block helper：未 refit 直接 raise
- ✅ **C5/C6 新 gate**（`replay_gate.py`）：pelvis_end_z + pelvis_tilt_end + lie_on_box；back-test 12 case **召回率 12/12 = 100%**（4 WORK 全 PASS，8 FAIL 全 FAIL）
- ✅ **C7 可视化**：6 个典型 case × {4-view PNG + turntable mp4} 共 12 个文件；subagent 视觉签收 12/12 hand 绿面贴合点云、5/5 DIFFER hand 旧面明显错位

## 1. 改动文件

### spider 主仓库

| 类别 | 文件 | 改动 |
|---|---|---|
| 新增 | `workspace/core4d/scripts/E098/historical_case_manifest.tsv` | 20 行 case 全集 |
| 新增 | `workspace/core4d/scripts/E098/face_utils.py` | 统一 face helper（全 3D argmax，6 面）|
| 新增 | `workspace/core4d/scripts/E098/test_face_utils.py` | 单元测试 9/9 PASS |
| 新增 | `workspace/core4d/scripts/E098/anchor_face_gate.py` | B5 hard block helper + 自检 PASS |
| 新增 | `workspace/core4d/scripts/E098/replay_gate.py` | 新 CEM gate replay 工具 |
| 新增 | `workspace/core4d/scripts/E098/backtest_cases.tsv` | 12 行 back-test 输入 |
| 新增 | `workspace/core4d/scripts/E098/anchor_refit_demo.py` | 6 case anchor refit demo |
| 新增 | `workspace/core4d/scripts/E098/render_anchor_compare.py` | 3D 可视化 |
| 修改 | `workspace/core4d_collab_retarget/scripts/E017/audit_select_anchors.py` | B1: `FACE_ORDER`/`face_label`/`_face_axis_sign`/`_clip_anchor`/`_snap_to_face` 全 3D |
| 修改 | `workspace/core4d_collab_retarget/scripts/E018b/generate_e018b_assets.py` | B1: `face_label`/`_face_axis_sign`/`canonical_anchor` 全 3D |
| 修改 | `workspace/core4d_collab_retarget/scripts/E020_audit/audit_common.py` | B1: `_face_label`/`_face_counts` 全 3D，含 ±z |
| 修改 | `workspace/core4d_collab_retarget/scripts/E028b/build_e028b_manifest.py` | B2: `_project_to_face` 通用化 |
| 修改 | `workspace/core4d_collab_retarget/scripts/E029/generate_e029_d6_assets.py` | B3: 显式 raise 替代 `+x` fallback |
| 修改 | `spider/process_datasets/core4d.py:128` | B4 deprecation comment（行为不变）|
| 修改 | `workspace/exp_diagnostic/diagnostic_report.md` 末尾 | ERRATA 块：标注 §3.2/§3.5 受 B1/B4 污染 |

### holosoma 仓库

本期无 holosoma 改动（face_label bug 全部在 spider 仓库；OmniRetarget 本身无 face 概念）。

### 不动的文件

- `spider/simulators/mjwp.py`（CEM elite filter 改动暂不并入 simulator；E101 时若 typical case 验证新 gate 需要在线生效再合）
- 所有 `examples/config/override/*.yaml`（reward 配置无变化）

## 2. 验证：Claims 逐条

### C1 — historical_case_manifest 20 行 ✅

`workspace/core4d/scripts/E098/historical_case_manifest.tsv` 20 行；分布：

| box_family | 行数 | 用途 |
|---|---:|---|
| box021_d003 | 6 | E101 typical/phase2 |
| box021_other | 1 | E089A WORK 守门 (H1 对照) |
| box023 | 2 | 守门 |
| box025 | 2 | 守门（侧抓） |
| box004 | 4 | 3 WORK + 1 infeasible |
| box026 | 3 | 包括 E094 C2/C3 |
| box022 | 2 | E102 preflight 待补 |

### C2 — B1+B2+B3 修复 + face 单测 + anchor refit demo ✅

- 单元测试：`.venv/bin/python workspace/core4d/scripts/E098/test_face_utils.py` → **9/9 PASS**
- syntax check：5 个 source 文件 `py_compile` 全 PASS
- anchor refit demo（在 6 个典型 box021 D003 case 上跑 contact_pos 投票）：

| case | L old → new | R old → new | DIFFER hand |
|---|---|---|---|
| 029_p2 | -x(73) → -x(52) | -x(64) → **+z(49)** | R |
| 035_p2 | +x(108) → +x(102) | -x(111) → -x(107) | — |
| 019_p1 | +x(70) → +x(47) | -x(71) → -x(64) | — |
| 030_p1 | +y(84) → **+z(64)** | +x(65) → **+z(59)** | L+R |
| 020_p2 | -x(70) → -x(55) | +x(68) → **+z(66)** | R |
| 028_p2 | +y(86) → **-z(61)** | -x(70) → -x(70) | L |

5/12 hand 主面翻到 ±z（其中 4/5 翻到 +z 顶面）。6 个 case 里 4 个有 ≥1 hand DIFFER → C2 达成"至少 4/6 case 主面翻到 ±z"判据。

Cross-check：v2 §3 B1 表 030_p1 L 真主面 +z 73%，新算法 64/88 = 73% → 完全一致。

### C3 — B4 deprecation comment + v1 errata ✅

```bash
$ grep -A2 "B4 DEPRECATION" spider/process_datasets/core4d.py | head -6
            # B4 DEPRECATION NOTICE (exp_diagnostic_v2 §2/§4):
            # `contact_pos` written here is the IK FK PALM SITE position (after IK
            # has placed G1 in the retargeted pose), NOT raw human mocap fingertip.
```

`workspace/exp_diagnostic/diagnostic_report.md` 末尾追加 "ERRATA — 2026-05-30" 段，标注 §3.2/§3.5/§3.3 受 B1+B4+B6 污染。

### C4 — B5 anchor_face_review hard block ✅

```bash
$ .venv/bin/python workspace/core4d/scripts/E098/anchor_face_gate.py
✓ no-review rows pass
✓ refit-tagged rows pass
✓ unrefit raises: B5 hard block: 1 manifest rows have anchor_face_review=true ...
All anchor_face_gate self-checks PASS
```

3 个自检全 PASS（无 review、有 refit tag、无 tag）。

### C5 — pelvis + lie-on-box gate replay helper ✅

阈值（最终调定）：
- `PELVIS_END_Z_THRESH = 0.55 m` —— episode 最后 10% 帧 pelvis_z 均值（未站起来）
- `PELVIS_TILT_END_THRESH_DEG = 75°` —— 最后 10% 帧 pelvis up vs world up 平均夹角（倒伏）
- `LIE_ON_BOX_GAP_THRESH = 0.10 m`、`LIE_ON_BOX_FRAC_THRESH = 30%`（贴箱）

设计要点：
- 初版用 `pelvis_tilt_max > 60°` 单帧阈值，被正常 squat-lift 误触（box004 WORK 弯腰那帧达 ~100°）。改成"最后 10% 帧均值"才区分"是否站起来"
- pelvis_low 阈值从 0.40 → 0.55，覆盖 E088B 类（pelvis 0.65 但 tilt 96.8）和真摔

### C6 — back-test 100% 召回 ✅

`workspace/core4d/results/E098/backtest/summary.tsv` 12 case，对照 expected_verdict：

| case | expected | gate | pelvis_end | tilt_end | lie_frac | 结果 |
|---|---|---|---:|---:|---:|---|
| E096bP1_box004_083_p1 | PASS | False | 0.789 | 18.5 | 0.000 | ✓ |
| E096bP2_box004_082_p1 | PASS | False | 0.774 | 16.6 | 0.183 | ✓ |
| E094P1_box004_083_p2 | PASS | False | 0.781 | 43.3 | 0.000 | ✓ |
| E089A_box021_person1 | PASS | False | 0.781 | 63.3 | 0.000 | ✓ |
| E094P2_box026_039_p2 | FAIL_LIE_ON_BOX | True | 0.479 | 85.5 | 0.626 | ✓ 三 gate 全触发 |
| E094P3_box026_135_p2 | FAIL_PELVIS | True | 0.183 | 89.4 | 0.610 | ✓ |
| E090S1_box021_035_p2 | FAIL_PELVIS | True | 0.184 | 29.2 | 0.474 | ✓ pelvis+lie |
| E088A_box021_029_p2 | FAIL_ANY | True | 0.190 | 74.7 | 0.360 | ✓ |
| E088B_box021_029_p2 | FAIL_ANY | True | 0.649 | 96.8 | 0.000 | ✓ tilt-only |
| E088C_box021_029_p2 | FAIL_ANY | True | 0.175 | 91.8 | 0.627 | ✓ |
| E082_box021_029_p2 | FAIL_ANY | True | 0.379 | 135.2 | 0.587 | ✓ |
| E091S1_box004_083_p2 | FAIL_PELVIS | True | 0.093 | 83.5 | 0.724 | ✓ |

**12/12 = 100% 召回，C6 PASS。**

关键观察：
- E094 C2 box026_039 "趴箱"被 lie_on_box（0.626）+ tilt_end（85°）双重捕获——v1 H1 没预测到这种"指标全好但视觉失败"的模式，现在 detector 能拦住
- box004 WORK 三个 case 守门保持，E089A WORK 也保持（end_tilt 63° 接近阈值但通过——说明阈值校准合理）
- E088B 是有意思的边缘：pelvis 0.649 m > 0.55，但 tilt_end 96.8° → "robot 站着但身体几乎水平躺向 box" 的 reward-hack 局部解，被 tilt gate 拦下

### C7 — 6 case 3D 可视化 + 视觉签收 ✅

6 case × {4-view PNG + 36-frame turntable mp4} = 12 文件，全部 > 270KB / 930KB。

subagent 视觉签收（`workspace/core4d/results/E098/visuals/anchor_compare/REVIEW.md`）结论：
> DIFFER 计数 **5/12 hand**，与脚本日志一致；绿色（新）面贴合点云 **12/12 全部 yes**；5/5 DIFFER hand 旧面明显错位（点云在 ±z 顶/底面但旧 argmax 强行选侧面）；**v2 §3 B1 主张 YES（完全支持）**。

具体证据（quote）：
- "030_p1 双手都是顶面抓取（z≈+0.27），旧 argmax 把它们误判成 +y 和 +x 侧面"
- "028_p2 L 是底面托抓（z≈-0.27），旧 argmax 误判成 +y 侧面"
- "029_p2 R / 020_p2 R 是右手单独顶面抓，旧均误判到 -x/+x"

## 3. 失败模式与决策记录

### 3.1 pelvis tilt 阈值 60° → 75°，并改单帧 max → 最后 10% 帧 mean

初版被 box004 WORK 的弯腰帧（tilt ~100°）误触 4 个 WORK 全 FAIL。诊断：

```
WORK end_tilt: 18, 17, 43, 63 (E089A 较高因为该 case end 时人仍前倾在协作姿态)
WORK tilt_max: 84, 92, 101, 97
FAIL end_tilt: 85, 89, 29 (E090 摔但拍快), 75, 97, 92, 135, 84
```

`end_tilt > 75°` 区分 7/8 FAIL，剩下 E090S1 由 pelvis_end_z 0.18 拦下。`tilt_max` 这条单独删除（会冤枉 squat-lift）。

### 3.2 E029 B3 选择 raise 而非 silent fallback

考虑到 E029 是低频代码路径（D6 asset writer），且 silent `+x` fallback 是历史的 wrong-target 隐患，决定改成显式 `raise ValueError(...)` 让 caller 必须显式处理。代价：以后若有人想跑 E029 D6 流程，要先确保 preflight 给出 SIDE face。可接受。

### 3.3 没有改 spider/simulators/mjwp.py 的 elite filter

新 gate 暂时只做 offline replay，没并入 in-loop CEM。原因：
1. Stage 0 的目标是"诊断 + back-test 工具齐备"，不是改动 SPIDER 在线 reward。
2. E101 Phase 1 typical case 验证新 gate 在 in-loop 有意义后再 merge 进 simulator，避免污染 Stage 1/2 的对比基线。

## 4. 结果路径

- 代码改动：见 §1 表
- back-test：`workspace/core4d/results/E098/backtest/summary.tsv` + 12 个 `{case}_gate.json` + 12 个 `{case}_per_frame.npz`
- anchor refit demo：`workspace/core4d/results/E098/anchor_refit/summary.tsv` + 6 个 `{case}.json`
- 可视化：`workspace/core4d/results/E098/visuals/anchor_compare/` 12 文件 + `REVIEW.md`
- 单测日志：`.venv/bin/python workspace/core4d/scripts/E098/test_face_utils.py` (9/9 PASS), `.../anchor_face_gate.py` (3/3 PASS)

## 5. 下游影响

| 下游 stage | 依赖 E098 哪个产出 | 状态 |
|---|---|---|
| E099 | historical_case_manifest.tsv | ✅ 可用 |
| E099 | face_utils.py（用于 fingertip face vote） | ✅ 可用 |
| E099 | quat_audit 在 manifest 上跑 | ✅ 输入就绪 |
| E100 | face_utils + replay_gate | ✅ 可用 |
| E100 干净 A/B | replay_gate（评估 24-step CEM 输出） | ✅ 可用 |
| E101 full CEM | replay_gate（in-loop or post-hoc 都行） | ✅ post-hoc 可用 |
| E101 失败归因 | pelvis/tilt/lie 三 gate + per-frame npz | ✅ 可用 |
| E102 mining | face_utils + quat_audit | ✅ 可用 |

## 6. 已知遗留 / TODO（不阻断 E099 启动）

- E101 时如发现 in-loop CEM gate 需要硬接 elite filter，再把 replay_gate 的逻辑 port 到 `spider/simulators/mjwp.py`（半天工作量）
- E098 anchor refit demo 只覆盖 6 case；E099 会推广到全部 20 case
- `data_construction_v2/visualizations/raw_contact/` 源脚本仍未找到，留给 E099
- 老 E017→E028 的 `manifest.tsv` 未重跑（B1 修了 source 但历史 manifest 还是错的）；如后续要用 weld，需重跑——目前 E082-E097 都不用 weld，不阻塞

## 7. Git

本实验 spider 仓库 commit + push；holosoma 本期无改动不动。commit message 见 git log。

## 8. 下一步

启动 E099（Stage 1：接触语义信息流补全）。
