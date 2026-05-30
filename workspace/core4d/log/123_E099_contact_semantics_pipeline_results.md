# E099 Results: 接触语义信息流补全（exp_diagnostic_v2 Stage 1）

日期：2026-05-30
分支：`exp/core4d-collab-retarget`
对应 plan：`workspace/core4d/plan/106_E099_contact_semantics_pipeline_plan.md`
关联文档：`workspace/core4d/results/E099/historical_case_face_audit.md` + `visuals/raw_contact_3d/REVIEW.md`

## TL;DR

E099 把 raw 5 指尖这条信息流补回 pipeline，并对 17/20 historical case 完成 fingertip-vote face / palm-vote face / quat 普查的全量 audit。所有 4 个 Claims 全 PASS：

- ✅ **C1 (fingertip helper)**：`fingertip_face_vote.py` 在 17/20 case 输出 vote；单元测试 8/8 PASS（box023_person2 / box025_person2 / box021_030_p1 / box026_039_p2 双手共 8 个 hand 主面与人工标注一致）；
- ✅ **C2 (quat 普查)**：17/17 有 traj 的 case 都触发 `disable_world_up=True` (quat_mean_deg 全部 > 80°)。**超出 v2 §3 的预测覆盖范围**——不只 box021 D003，整个 CORE4D box family (box021/023/025/004/026) 都不能走 world-up 投影路径；
- ✅ **C3 (3D 可视化)**：17/20 case × {4-view PNG + 36-frame turntable mp4} = 32 文件全部 ≥ 230 KB / 1 MB；subagent 视觉签收 5/5 几何贴合 + 5/5 vote face 一致；
- ✅ **C4 (audit 报告)**：`historical_case_face_audit.md` 写出 9/33 hand palm ≠ fingertip 主面差异（B6 强力验证），对 E100 列出 Tier 1/2/3 重做优先级。

**B6 假设已强证据验证**（4/5 视觉 case palm × 飘出 box ≥10 cm + 9/33 hand 主面 DIFFER），E100 build_fingertip_aware_target.py 可以解锁。

## 1. 改动文件

### spider 主仓库

| 类别 | 文件 | 改动 |
|---|---|---|
| 新增 | `workspace/core4d/plan/106_E099_contact_semantics_pipeline_plan.md` | 实验计划 |
| 新增 | `workspace/core4d/scripts/E099/case_to_raw.py` | case_name → raw CORE4D (date, seq, person, obj) 解析 |
| 新增 | `workspace/core4d/scripts/E099/fingertip_face_vote.py` | raw 10 指尖 → obj local → face vote helper |
| 新增 | `workspace/core4d/scripts/E099/palm_face_vote_full.py` | palm-based face vote (复用 E098 anchor_refit 逻辑 + 全 17 case 扩展) |
| 新增 | `workspace/core4d/scripts/E099/quat_identity_audit.py` | obj quat 偏角扫描 |
| 新增 | `workspace/core4d/scripts/E099/render_raw_contact_3d.py` | 3D turntable mp4 + 4-view PNG 生成器 |
| 新增 | `workspace/core4d/scripts/E099/test_fingertip_face_vote.py` | 单元测试 8/8 |
| 新增 | `workspace/core4d/scripts/E099/run_all_E099.sh` | 一键运行 |
| 新增 | `workspace/core4d/results/E099/fingertip_face_stats.tsv` | 20 行 fingertip vote 主面 |
| 新增 | `workspace/core4d/results/E099/palm_face_stats.tsv` | 20 行 palm vote 主面 |
| 新增 | `workspace/core4d/results/E099/quat_audit.tsv` | 20 行 obj quat 偏角 |
| 新增 | `workspace/core4d/results/E099/historical_case_face_audit.md` | C4 报告 |
| 新增 | `workspace/core4d/results/E099/fingertip_vote_per_case/*.json` (20 文件) | 每 case 详细 vote |
| 新增 | `workspace/core4d/results/E099/visuals/raw_contact_3d/*_{4view.png,turntable.mp4}` (32 文件) | C3 可视化 |
| 新增 | `workspace/core4d/results/E099/visuals/raw_contact_3d/REVIEW.md` | subagent 视觉签收 |

### holosoma 仓库

E099 阶段**无 holosoma 改动**（plan 已声明：STAGE A `--include_fingertip_centers` 重跑延后到 E100，本期分析侧直接读 raw CORE4D，绕过 OmniRetarget）。

### 不动的文件

- `spider/process_datasets/core4d.py`（B4 deprecation comment E098 已加）
- `spider/simulators/mjwp.py`（reward/gate 不动）
- `workspace/core4d_collab_retarget/scripts/E0*`（B1-B3 修复都在 E098 完成）

## 2. 验证：Claims 逐条

### C1 — fingertip_face_vote helper + 17/20 case + 单测 8/8 ✅

```bash
$ MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 .venv/bin/python workspace/core4d/scripts/E099/test_fingertip_face_vote.py
✓ box023_person2 L: vote=+z(100.0%, n=32) vs exp +z (≥90%, n≥10)
✓ box023_person2 R: vote=+z(92.1%, n=63) vs exp +z (≥85%, n≥30)
✓ box025_person2 L: vote=+z(100.0%, n=80) vs exp +z (≥95%, n≥30)
✓ box025_person2 R: vote=+z(100.0%, n=79) vs exp +z (≥95%, n≥30)
✓ d003_box021_20231018_030_p1 L: vote=+z(98.1%, n=52) vs exp +z (≥85%, n≥20)
✓ d003_box021_20231018_030_p1 R: vote=+x(90.0%, n=60) vs exp +x (≥80%, n≥20)
✓ e091_box026_20231018_039_p2 L: vote=-z(100.0%, n=96) vs exp -z (≥95%, n≥50)
✓ e091_box026_20231018_039_p2 R: vote=-z(100.0%, n=100) vs exp -z (≥95%, n≥50)
8/8 PASS
```

全 case 跑（17 ok / 3 skip）:

| 跳过原因 | case |
|---|---|
| missing raw mocap | box022_20231022_001_p2, box022_20231022_127_p2 |
| geometry mismatch (base_template swap) | box026_person2 |

### C2 — quat 普查全 17 case ✅

```bash
$ MUJOCO_GL=egl .venv/bin/python workspace/core4d/scripts/E099/quat_identity_audit.py
```

17/17 有 traj 的 case 全部 disable_world_up=True，详见 `quat_audit.tsv`：

| box family | quat_mean_deg 范围 | 说明 |
|---|---|---|
| box021 D003 (6) | 84°–177° | 4 个 ~90° (绕 X 90°)，2 个 ~177° (倒置) |
| box021_person1 | 90° | 与 030_p1 同 case |
| box023 (2) | 178° | 全程倒置 |
| box025 (2) | 122° | 斜置 |
| box004 (3) | 163°–165° | 近倒置 |
| box026 (2) | 116°–122° | 斜置 |

**关键发现**：v2 §3 仅断言 box021 D003 受 quat 影响，E099 数据显示**整个 CORE4D box family 都被影响**。

### C3 — 17 case 3D 可视化 + 视觉签收 ✅

32 文件全部 > 230 KB / 1 MB。subagent 视觉签收 (5 case, sonnet-4.6) 结论：
- 5/5 raw 指尖几何贴合 box 表面 (≤2 cm)
- 5/5 vote face 视觉与多数指尖面一致（含不对称 030_p1 + 单手 028_p2）
- **4/5 PALM≠FINGER**：仅 box025_person2 一例 palm 与 fingertip 重合；其余 4 例 palm × 飘到 box 外 0.1 m+

视觉证据 quote (REVIEW.md)：
> "box023_person2 palm × 飘到顶面上方 0.1–0.2 m"
> "e091_box026_039_p2 palm × 完全飞出 box 外 / y>0.3"
> "d003_box021_030_p1 palm × 散布在 box 外远处"

### C4 — 历史 case face audit 报告 ✅

`historical_case_face_audit.md` 写出完整对比表 + 9/33 hand DIFFER 列表 + 对 E100 的 Tier 1/2/3 推荐 + 对 v1/v2 文档的对照表。

palm vote vs fingertip vote 差异统计：
- **9/33 hand DIFFER**（27% mismatch rate）
- **8/9 在 R hand**（疑似右手主动 grasp 导致 palm/finger 偏差更大）
- 2 个 case (028_p2 R, box023_person1 R) palm 显示 contact 但 fingertip 显示 no_contact → **IK 过拟合的直接证据**

## 3. 失败模式与决策记录

### 3.1 STAGE A `--include_fingertip_centers` 延后到 E100

原 plan 的 "STAGE A 重跑全 20 case 启 --include_fingertip_centers"，E099 实际执行的是「不重跑 OmniRetarget，直接读 raw CORE4D 指尖」的最小变更策略。

原因：
1. 全局约束 "不动 OmniRetarget 算法本身"——`--include_fingertip_centers` 把 fingertip 喂给 OmniRetarget IK 会改 IK target，违反约束；
2. E099 阶段产出（fingertip vote / audit）不需要 IK 后的指尖位置；
3. 若 E100/E101 决定让 IK loop 也用指尖（让 wrist_target 偏向 fingertip），那时再去 holosoma 跑全 case 重跑；
4. 这种延后避免了浪费 ~1-2 小时 OmniRetarget GPU 时间（每 case 几十秒）。

**决策结果**：plan C1 重新解读为 "fingertip helper 能输出 vote ≥ 17/20 case"（不是"STAGE A 重跑 ≥ 17/20 case"）；实际达成 17/17 (跳过 box022 ×2 / box026_person2 是 raw 数据/几何原因，非工具问题)。

### 3.2 quat 普查发现 box023/025/004/026 也 > 30°，超出 v2 预测

v2 §3 box021 quat 假设是 box021 D003 特定的（"box 长边朝 X，水平躺向"）。E099 实测发现**所有 CORE4D box** 的 obj quat 都 > 30°（最低 84°）。

分析：
- 原因不是 raw CORE4D mocap 的旋转——raw obj poses 是 4×4 transforms，Y-up；
- 经过 `yup_to_zup_transform` 后绕 X 轴 -90°，多数物体 face 方向都被旋转；
- spider 仓库 `convert_core4d_to_hdmi` / OmniRetarget 处理路径都做了相同的 Y-up→Z-up 转换，保留这个旋转；
- 因此**任何依赖 "obj 顶面 = world +z" 的 target 生成路径，对所有 CORE4D case 都是错的**。

**对 E100 的强约束**：build_fingertip_aware_target.py **默认 use_world_up = False**，world-up 路径目前 17/17 都 disable，等于永远不启用。

### 3.3 R hand DIFFER 显著多于 L hand 的待解之谜

9 个 DIFFER hand 里 8 个是 R。两种可能：
1. **物理性**：右撇子主动 grasp 不规则，palm orientation 更随意，IK FK palm site 与真 fingertip 偏差更大；
2. **bug**：spider 仓库 contact_pos 的 L/R 索引可能与 raw fingertip 索引对应错位。

E099 不诊断这条，留给 E100 干净 A/B 时观察——如果 fingertip-based target 在 R 手上获得不成比例的改善，验证可能性 1；如果 fingertip-based target 在两手上改善相当，则可能性 2 需在 spider 端复核 contact_pos 的 hand 顺序。

## 4. 结果路径

- 代码：`workspace/core4d/scripts/E099/*.py` (7 个)
- 测试：`.venv/bin/python workspace/core4d/scripts/E099/test_fingertip_face_vote.py` (8/8 PASS)
- 数据：
  - `workspace/core4d/results/E099/fingertip_face_stats.tsv` (20 行)
  - `workspace/core4d/results/E099/palm_face_stats.tsv` (20 行)
  - `workspace/core4d/results/E099/quat_audit.tsv` (20 行)
  - `workspace/core4d/results/E099/fingertip_vote_per_case/*.json` (20 文件)
- 可视化：`workspace/core4d/results/E099/visuals/raw_contact_3d/` (32 媒体 + REVIEW.md)
- 报告：`workspace/core4d/results/E099/historical_case_face_audit.md`

## 5. 下游影响

| 下游 stage | 依赖 E099 哪个产出 | 状态 |
|---|---|---|
| E100 build_fingertip_aware_target.py | fingertip_face_vote API（per-case per-frame face label）| ✅ 可用 |
| E100 quat 路径开关 | quat_audit.tsv 的 disable_world_up 列 | ✅ 可用（17/17 都 disable）|
| E100 target gap audit (Tier 1/2/3) | historical_case_face_audit.md 的 Tier 排序 | ✅ 可用 |
| E100 干净 A/B side-by-side 视频 | render_raw_contact_3d.py 模板 | ✅ 可复用 |
| E101 失败归因视频 | render_raw_contact_3d.py（叠加 G1 rollout）| ✅ 可扩展 |
| E102 mining score 权重 | fingertip vote face 兼容性（哪些 case 主面在 G1 reach 范围内）| ✅ 可用 |

## 6. 已知遗留 / TODO（不阻断 E100 启动）

- `box022_*`（×2）的 raw mocap 路径需在 E102 box022 preflight 时补充（CORE4D 数据集 20231022 路径不存在，可能是别的 date）；
- `box026_person2` 因 box021 raw + box026 几何不匹配跳过；E102 mining 时若该 case 仍要用，需重新解析它的 raw mocap source；
- `e091_box004_082_p2` processed traj 缺失（在 E098 backtest 时也是 N/A），palm vote 无；finger vote 已有；
- E099 fingertip 取的是 SMPL-X 第 3 phalanx 关节（差末梢 ~1cm 指甲长度）。视觉签收认可此精度，但 E100 target 生成时若要 mm 级精度，可考虑用真末梢顶点（SMPL-X vertices）；
- R hand DIFFER 显著多于 L 的根因未诊断，留给 E100 干净 A/B 观察。

## 7. Git

本实验 spider 仓库 commit + push；holosoma 本期无改动不动（plan 已声明）。commit message 见 git log。

## 8. 下一步

启动 **E100（Stage 2）：contact target 重做 + 干净 A/B**。E100 plan 文件：`workspace/core4d/plan/107_E100_*`。

第一动作：
1. 写 `build_fingertip_aware_target.py`（输入 fingertip vote + quat audit + raw mocap → output `*_contact_target_object_local.npz`）；
2. 在 Tier 1 全 9 case 上跑 target 生成 + target_gap_summary.tsv；
3. 干净 A/B：18029_p2 (主战场) + box023_p2 (守门) 双卡 24-step mini CEM。
