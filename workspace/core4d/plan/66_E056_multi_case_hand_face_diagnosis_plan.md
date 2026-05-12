# E056 实验计划：多 case Hand-Face 诊断 (筛选 ref 握姿合理的 case)

> 前置阅读：
> - `workspace/core4d/log/65_E055_box023_hand_snap_results.md` §坐标系定义 + §box023_person1 hand-face 诊断
> - `workspace/core4d/log/64_E054_case_tier_analysis_results.md` (6 个 B+C case 列表 + dominant_hand)

## Context

### 前置结论 (E055)

E055 在 box023_person1 上做 snap, **数值通过 + 视觉对比通过**, 但用户视觉发现 + 数值确认了一个根本问题：

- **L 手全程在 +xz 面** (朝机器人面) — 法向力推开 box, 物理上无用
- **R 手全程在 -yz 面** (远侧) — 法向力拉回机器人, 物理上有用
- **结论**: box023 ref 是"相邻面错位握" (-yz + +xz), **不是对侧夹握**, 力学不闭合
- snap 方法只能消除 gap, **不能修正接触面错误**

### 根因分析

- E054 把 6 个 B+C case 选出来时, 只看了"几何 Tier 1/2 + dim_max + obj_z_amp + intent window 存在", **没有验证 ref 握姿是否物理合理**
- 在所有 case 上跑 snap 之前, 必须先做 hand-face 诊断, 选出"两手在对侧面" 的 case 做核心验证
- 否则后续 E058 (Path B-CEM) 会重蹈 E041c 覆辙: 数值改善 + 视频不像搬运

### 关键 insight

**用 face_distance_timeseries 看每个 case 的 L/R 各自"长时间贴 0" 的面**, 然后判断:

- **对侧夹握** (理想): L 在 +F 面, R 在 -F 面 (其中 F ∈ {yz, xz}) → 法向力相互抵消, 力学闭合
- **垂直夹握** (可接受): L 在 ±xy (顶/底), R 在 ±yz 或 ±xz → 一个支撑重力, 一个稳定方向
- **相邻面错位** (无用): L/R 在两个相邻面 (例如 -yz + +xz) → 力学不闭合 ← box023 这种
- **同面** (异常): L 和 R 在同一个面 → mocap 错位

## Claims

| ID | Claim | 最低证据 |
|---|---|---|
| C1 | 5 个新 case 都跑通 face_distance_timeseries 诊断 | 5 张 png + 1 个 summary csv 完整 |
| C2 | summary csv 标注每个 case 的 L_main_face / R_main_face / 握姿类型 | csv 含 `L_main_face, R_main_face, grasp_type` 字段, 5 行非空 |
| C3 | 至少 1 个 case 的 ref 握姿是"对侧夹握" 或 "垂直夹握" | 5 case 中 ≥1 个 grasp_type ∈ {对侧, 垂直} |
| C4 | 鉴定结果与视频核实一致 | 选 grasp_type=对侧 的 case 1 个 + grasp_type=错位 的 case 1 个, 各自从 visualization_kinematic.mp4 抽 3 帧确认 |
| C5 | 输出 E057 决策建议 | log 末尾给出"E057 应该: snap 在 case X / 加 face-prior" |

## 改动

### 1. 重构 `face_distance_timeseries.py` 为多 case 通用脚本

**文件**: `workspace/core4d/scripts/E056/multi_case_face_diagnosis.py` (~250 行)

**核心逻辑**:

```python
def diagnose_case(case_name: str) -> CaseDiagnosis:
    """对单个 case 跑 hand-face 诊断:
       1. load scene + traj
       2. 复用 E055 v3 detector 找 intent window
       3. 计算 L/R 在 6 面的 signed dist time series
       4. 找每个手 "main face" = 整个 intent 窗口 |signed_dist| 中位数最小的面
       5. 判定 grasp_type:
          - 对侧: L_main 与 R_main 是同一平面的两侧 (±yz 或 ±xz)
          - 垂直: 一个在 ±xy, 一个在 ±yz/±xz
          - 错位: 两个相邻面
          - 同面: L_main == R_main
       6. 返回所有数据 + 渲染 png
    """
    ...

def main() -> None:
    cases = ["box021_person1", "box023_person1", "bucket001_person1",
             "bucket005_s2_person1", "bucket007_person1", "desk021_person1"]
    diags = [diagnose_case(c) for c in cases]
    write_summary_csv(diags)
    write_summary_md(diags)
```

### 2. summary csv schema

`workspace/core4d/results/E056/case_grasp_type_summary.csv`:

```
case_name,
intent_window_start, intent_window_end, intent_window_length,
L_main_face, L_main_face_median_dist_cm, L_main_face_pct_below_3cm,
R_main_face, R_main_face_median_dist_cm, R_main_face_pct_below_3cm,
grasp_type,                       # 对侧 / 垂直 / 错位 / 同面 / 单手
grasp_valid,                      # True / False (= 对侧 ∨ 垂直)
dominant_hand,                    # 复用 E054
notes
```

### 3. summary md (人读用)

`workspace/core4d/results/E056/E056_summary.md`:

- 6 个 case 的 grasp_type 分类表
- 每个 case 的 main face + 距离统计
- 推荐 E057 起点 case (grasp_valid=True 中 obj_z_amp 最大者)
- 如全部 grasp_valid=False 的 fallback 方案

### 4. 视频核实 (C4)

复用 `extract_case_keyframes.sh` 抽:
- 1 个 grasp_valid=True case 的 3 帧 (intent 起 / 中 / 末) → 目检确认双手对称握
- 1 个 grasp_valid=False case 的 3 帧 → 目检确认 L/R 错位

输出: `workspace/core4d/results/E056/video_verify/{case}_t{stamp}.jpg`

### 5. 一键脚本

`workspace/core4d/scripts/run_E056_diagnosis.sh`:
1. 跑 multi_case_face_diagnosis.py → 5 png + csv + md
2. 抽视频关键帧 → 6 jpg (2 case × 3 帧)
3. 打印 summary 表

## 需要修改的文件

| # | 文件 | 改动 |
|---|------|------|
| 1 | `workspace/core4d/scripts/E056/multi_case_face_diagnosis.py` | 新建 — 多 case face dist 时间序列 + grasp_type 分类 |
| 2 | `workspace/core4d/scripts/run_E056_diagnosis.sh` | 新建 — 一键流水线 |
| 3 | `workspace/core4d/results/E056/case_grasp_type_summary.csv` | 新生成 |
| 4 | `workspace/core4d/results/E056/E056_summary.md` | 新生成 |
| 5 | `workspace/core4d/results/E056/{case}_face_dist.png` | 新生成 (×6) |
| 6 | `workspace/core4d/results/E056/video_verify/*.jpg` | 新生成 (×6) |
| 7 | `workspace/core4d/EXPERIMENT_TRACKER.md` | 添加 E056 行 + scripts 引用 |
| 8 | `workspace/core4d/log/66_E056_multi_case_diagnosis_results.md` | 实验完成后写 log |

**核心不动 spider/ 任何代码** — 只在 workspace/core4d/ 内做数据分析。

## Reward 权重

不适用 — 纯数据分析。

## 训练命令

```bash
bash workspace/core4d/scripts/run_E056_diagnosis.sh
```

预期 < 2 分钟 (6 case × ~10s 诊断 + 几秒 ffmpeg 抽帧)。

## 成功标准

| 指标 | 前次 (E055 box023 单 case) | 本次目标 (E056 6 case 全覆盖) |
|------|----------------------------|-------------------------------|
| 诊断 case 数 | 1 | **6** |
| 已知 grasp_type | box023 = 错位 | **6 个全部分类完成** |
| ≥1 case grasp_valid=True | (未知) | **≥1, 否则启动 face-prior fallback** |
| 视频核实 | 5 帧 | **6 帧 (对侧 + 错位 各 3 帧)** |
| E057 决策 | (未定) | **明确给出"snap on case X" 或 "加 face-prior"** |

## 风险与回退

| 风险 | 概率 | 应对 |
|------|------|------|
| 6 case 全部 grasp_valid=False | 中 | E057 改为 face-prior 实现 (强制 L/R 在对侧面 ±yz 或 ±xz), 不依赖 ref 握姿 |
| dominant_hand=L (bucket001) 没有 R 主面 | 必然 (E054 已知) | 单手 case grasp_type = "单手" 单独处理, 不参与 valid 判定 |
| Intent window 边界帧 face 漂移大 | 低 | "main face" 用 intent 窗口内 |signed_dist| 的**中位数**, 不是平均, 抗边界异常 |
| 物体不是 box 时 face 命名混乱 (bucket / desk) | 中 | 仍用 ±yz/±xz/±xy 命名, 统一基于 collision box AABB; bucket 圆筒会有视觉错位但 collision 仍是 box |
| 6 case 中 1-2 个 ref 握姿其实 borderline (例如 main face 不稳定) | 中 | summary 加 "main_face_stability" 列 (中位数 vs 标准差), 标记 borderline |

## 后续衔接 (E057+)

依据 E056 结果分两条路线:

- **路线 A (有合理 case)**: E057 = 在该 case 上重做 E055 snap (复用 hand_snap_ik.py, 改 case_name)
  - 验证 snap 方法在 ref 合理时能产出物理合理的 warmstart
  - 然后 E058 接 CEM
- **路线 B (全部不合理)**: E057 = 改 hand_snap_ik.py 加 `face_filter` 选项
  - 强制 snap 选**对侧面对** (L 选 +F, R 选 -F, F ∈ {yz, xz})
  - 在 box023 重做 snap 验证 face-prior 修正握法
  - 然后 E058 接 CEM
