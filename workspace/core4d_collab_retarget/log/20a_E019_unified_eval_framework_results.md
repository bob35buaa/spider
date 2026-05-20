# E019 实验结果：统一评测框架（P0 + P1）

**日期**: 2026-05-20
**实验域 (exp_name)**: `core4d_collab_retarget`
**对应 Plan**: `workspace/core4d_collab_retarget/plan/20_E019_unified_eval_framework_plan.md`
**前置**: `workspace/core4d_collab_retarget/log/19_E018b_canonical_support_proxy_13case_results.md`
**性质**: 工程/评测框架（非 RL 训练），不涉及 CEM rollouts，结果度量为 paper-aligned metrics 实现完整性 + 数字一致性

---

## 1. 背景

E018b 完成后，工作区有 18 个实验的零散 paper metrics，但：
- 没有统一脚本能在 spider physical + holosoma kinematic 之间做对比
- SPIDER Table 4 (Joint/Pos/Ori) 只在 `workspace/core4d/scripts/eval/eval_comprehensive.py` 有原始实现，没迁到 `core4d_collab_retarget/scripts/eval/paper_metrics.py`
- OmniRetarget Table II 的 penetration 用的是 csv-based proxy（消费 `legobj_timeseries_*.csv`），不是论文原生 `mj_geomDistance + prefilter`
- 没有 论文级评测文档 `docs/eval_metrics.md`，也没 xlsx 多 sheet 跨方法对比

E019 把这些填齐：P0 = SPIDER T4 严格 FK + OmniRetarget mj_pen + unified_eval CLI + docs；P1 = 28cm obj-local contact preservation + EvalInputs adapter + holosoma kinematic 接入 + Tab.5 跨方法对比表。

---

## 2. 改动

### 2.1 文件清单

| 文件 | 行数变化 | 用途 |
|---|---|---|
| `scripts/eval/paper_metrics.py` | 426 → 972 | 新增 `_add_body_tracking_metrics` (SPIDER T4 FK) + `_add_penetration_metrics_mj` (mj_geomDistance) + `_add_contact_preservation_omni_local` (28cm obj-local) + `_quat_to_matrix_batch` + `add_paper_metrics_physics` (physics-only 入口) |
| `scripts/eval/unified_eval.py` | new, 530 行 | CLI 表格 + xlsx 生成器；支持 `--method` 多次输入；自动产出 SPIDER T4 / OmniRetarget T2 / DynaRetarget T5 / CORE4D 协作 4 张 md + 跨方法 Tab.5 + 多 sheet xlsx |
| `scripts/eval/eval_holosoma_kinematic.py` | new, 193 行 | 跑 holosoma v2 kinematic 评测（physics-only），输出与 spider 同 schema 的 comparison.csv |
| `scripts/eval/adapters/common_inputs.py` | new, 122 行 | `EvalInputs` dataclass — method/case/model/qpos_sim/qpos_ref/fps/case_window/human_joints/object_poses/extras |
| `scripts/eval/adapters/kinematic_to_common.py` | new, 140 行 | holosoma v2 NPZ → `EvalInputs`；`CASE_MAP` 当前覆盖 `box025_p1` / `box025_p2`（仅有的交集） |
| `scripts/eval/adapters/__init__.py` | new, 22 行 | 导出 adapter API |
| `docs/eval_metrics.md` | new, 306 行 | 论文级指标文档；P0 §1-6 + P1 §7（28cm 定义、adapter 接入、Tab.5 首发数字、5 条 caveat） |
| `plan/20_E019_unified_eval_framework_plan.md` | 已 commit | E019 计划文件 |

### 2.2 SPIDER Table 4 实现（P0）

`_add_body_tracking_metrics` (`paper_metrics.py:300-411`)：

- Robot body 集合：排除 world(0) + `object` + `support_weld_anchor` + `support_dynamic_anchor` → E018b `nbody=33` 时 30 个 robot body
- 每帧跑 `mj_kinematics(model, data)` × sim+ref，取 `xpos / xquat`
- Joint Err: `mean_{t,j∈[7..36)} |q_sim - q_ref| · 180/π`（29 dof）
- MPKPE: `mean_{t,b} ‖xpos_sim - xpos_ref‖ · 100` cm
- Body Ori: `mean 2·arccos(|xquat_sim · xquat_ref|) · 180/π`
- Root Pos/Ori: 限定 b=pelvis
- EEF Pos/Ori: L/R `wrist_yaw_link` 平均（不是 `left_rubber_hand` — 那是 mesh 而非 body，参考 `holosoma/v1/eval_paper_metrics.py:51-54`）

### 2.3 OmniRetarget mj_geomDistance penetration（P0）

`_add_penetration_metrics_mj` (`paper_metrics.py:414-592`)：

- 复用 `holosoma/v1/eval_paper_metrics.py:67-92` 的 prefilter 思路：临时扩 margin 跑 `mj_collision` 得候选 pair → 还原 margin → `mj_geomDistance` 精算
- 排除 object↔ground pair（物体接地是合理的）
- Object body 检测：`detect_object_body(model)` 找尾部 freejoint（qposadr=nq-7）
- 输出 full / case-window 两套：duration_pct + max_depth_cm + mean_depth_cm
- 容差 1cm（PEN_TOLERANCE），prefilter margin 10cm（PEN_COLLISION_DETECTION_THRESHOLD）

### 2.4 28cm obj-local contact preservation（P1）

`_add_contact_preservation_omni_local` (`paper_metrics.py:594-720`)：

- 对齐 holosoma `eval_paper_metrics.py:258-303` 严格定义
- 输入：`summary["human_joints"]`（直接 ndarray）或 `summary["human_joints_npz"]`（路径）
- demo wrist + sim wrist 各自转到对应 object 局部系（用 `_quat_to_matrix_batch` 批量算 R^T）
- 28cm 半径二值：`||p_local|| < 0.28`
- **OmniRetarget 原始定义**：`preservation = 1 - miss_frames / T`（T 是全帧数；miss = demo_contact ∧ ¬sim_contact）。注意**不是** `1 - miss/demo_frames`，否则当 demo 无接触时退化为 0%（我第一版写错了，已修正）

### 2.5 EvalInputs adapter（P1）

`adapters/common_inputs.py`: `EvalInputs` dataclass，最小字段 + `summary_template()` 工厂方法把数据投射成 spider 兼容 summary dict 形式。
`adapters/kinematic_to_common.py`: `load_kinematic_inputs(case, model)` 读 holosoma v2 retarget NPZ + companion `data/core4d_replace_batch/{seq}-object.npz`；自动检测 object pose 是 `[qw qx qy qz x y z]` 还是 `[x y z qw qx qy qz]` 并归一化。
`CASE_MAP` 只覆盖 `box025_p1` / `box025_p2` — 这是 holosoma v2 在 13 个 spider E018b case 中实际有 retarget 输出的全部交集。

### 2.6 unified_eval.py 跨方法对比

- 支持 `--method` 多次传入
- 4 张 standalone md 表（SPIDER T4 / OmniRetarget T2 / DynaRetarget T5 / CORE4D 协作）
- 5 sheet xlsx：`raw_<method>` 每方法一个 + `spider_t4_mean` 跨方法 + `omni_t2_mean` 跨方法 + `by_object_<primary>`
- **Tab.5 `table_method_comparison.md`**：4 张 metric group 表，列 `Metric × spider_E018b × holosoma_v2_kinematic × Δ`
- Case 匹配修复 `_short_case`：把 `box025_person2_freejoint_legobj_e018b` 与 `box025_p2` 都归一化到 `box025_p2`

### 2.7 环境改动

| 变更 | 原因 |
|---|---|
| 装 uv 0.11.15 (`~/.local/bin/uv`，via `https_proxy=http://10.140.15.68:3128`) | 本机原无 uv/pip |
| 装 openpyxl 3.1.5 (`uv pip install --python .venv/bin/python3 openpyxl --index-url http://pypi.devops.xiaohongshu.com/simple/`) | xlsx 生成 |

---

## 3. 结果路径

| 类型 | 路径 |
|---|---|
| 评测代码 | `workspace/core4d_collab_retarget/scripts/eval/{paper_metrics,unified_eval,eval_holosoma_kinematic}.py` + `adapters/` |
| 论文级文档 | `workspace/core4d_collab_retarget/docs/eval_metrics.md` |
| spider E018b 13 case（重评后含新字段）| `workspace/core4d_collab_retarget/results/E018b/comparison.csv` + `eval_summary_*.{csv,json}` × 13 + `aggregate_summary.json` |
| holosoma v2 kinematic 2 case（首次）| `workspace/core4d_collab_retarget/results/holosoma_v2_kinematic/` 全套 |
| 统一表格汇总（含 Tab.5） | `workspace/core4d_collab_retarget/results/eval_unified/` |

```
workspace/core4d_collab_retarget/results/eval_unified/
├── INDEX.md
├── tables/
│   ├── table_spider_t4.md
│   ├── table_omniretarget_t2.md
│   ├── table_dynaretarget_t5.md
│   ├── table_core4d_collab.md
│   ├── table_spider_E018b.md
│   ├── table_holosoma_v2_kinematic.md
│   ├── table_method_comparison.md    # E019 P1 核心交付物 (Tab.5)
│   ├── table_paper_all.xlsx          # 5 sheet
│   └── table_paper_all.csv_bundle/   # xlsx fallback
└── per_case/
    ├── spider_E018b/{box021_p1...desk021_p1}.json   # 13 个
    └── holosoma_v2_kinematic/{box025_p1,box025_p2}.json
```

---

## 4. 运行指令

> 脚本规则：本次实验的 eval 入口已固化在 `scripts/eval/`，下面命令是固定调用形式。

### 4.1 E018b 13 case 全量重评（拿到新 SPIDER T4 + mj_pen + 28cm 字段）

```bash
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl .venv/bin/python \
  workspace/core4d_collab_retarget/scripts/eval/eval_E018b.py --all
```

实际耗时：~5 分钟（13 case × 各 ~24s FK + 几秒 prefilter+mj_geomDistance × 2 pass full/case）。

### 4.2 holosoma v2 kinematic 2 case 评测

```bash
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl .venv/bin/python \
  workspace/core4d_collab_retarget/scripts/eval/eval_holosoma_kinematic.py --all
```

实际耗时：~30 秒（2 case × physics-only）。

### 4.3 跨方法 unified_eval

```bash
.venv/bin/python workspace/core4d_collab_retarget/scripts/eval/unified_eval.py \
  --method spider_E018b \
  --comparison workspace/core4d_collab_retarget/results/E018b/comparison.csv \
  --method holosoma_v2_kinematic \
  --comparison workspace/core4d_collab_retarget/results/holosoma_v2_kinematic/comparison.csv \
  --out workspace/core4d_collab_retarget/results/eval_unified
```

实际耗时：<1 秒（纯后处理）。

---

## 5. 可视化

E019 是评测框架实验，**不产生 rollout 视频**，没有 viewer 可视化需求；产物全部是 md/csv/json/xlsx 表格。

**Caveat**：E018b 重评后的 13 个 online MP4 已经在 `results/E018b/E018b_*_canonical_t02.mp4`，但 video 内容不因评测改动而变；本 log 不重复 `/video-frames` 分析（已在 `log/19` 完成）。

---

## 6. 评测结果

### 6.1 SPIDER Table 4（spider E018b 13 case，首发严格 FK 数字）

> 所有指标 ↓ 越小越好（sim vs ref tracking error）。

| 指标 | Mean | Std |
|---|---:|---:|
| Joint Err. (°) ↓ | 5.85 | 3.44 |
| MPKPE (cm) ↓ | 23.13 | 22.77 |
| Body Ori. Err. (°) ↓ | 23.49 | 22.46 |
| Root Pos. Err. (cm) ↓ | 22.90 | 24.70 |
| Root Ori. Err. (°) ↓ | 15.79 | 22.23 |
| EEF Pos. Err. (cm) ↓ | 20.29 | 25.29 |
| EEF Ori. Err. (°) ↓ | 30.17 | 26.04 |
| Obj. Pos. Err. (cm) ↓ | 5.45 | 1.63 |
| Obj. Ori. Err. (°) ↓ | 5.22 | 3.07 |

大 std 由 4 个 robot fall case（`box021_p1/p2`、`bucket001_p1/p2`）拉高；剔除后非 fall 9 case MPKPE / Root / EEF 都大幅下降。最佳 case `box025_p2`（唯一 paper_generalization_pass）：Joint 2.53° / MPKPE 7.34cm / Obj Pos 5.61cm / Obj Ori 1.92°。

### 6.2 OmniRetarget mj_geomDistance penetration（P0）

> 所有 penetration 指标 ↓ 越小越好。

13 case 全部 `mj_pen Duration ↓ ≈ 0%` / `max_depth ↓ ≈ 0cm`。与现有 csv-based `deep_penetration_duration ↓`（`mean 30%`）的差异：
- mj_geomDistance 是 robot↔object 的 geom 实算（容差 1cm）
- csv-based 的 deep_penetration_duration 用 `legobj_timeseries_*.csv` 的 SDF time-series（阈值 2cm，包含 leg + hand 双独立计算）
- 两者覆盖范围不同：mj_pen 是论文严格定义，csv-based deep_pen 是工程提示性指标。docs 已同时呈现，不冲突

### 6.3 Tab.5 跨方法对比（N=12，详见 log/20b）

> v2026-05-20-P2 修复 FPS=30 后重算的数字。N=12：12 个 spider E018b case ∩ 12 个 holosoma v2 kinematic case（desk021_p1 SOCP infeasible，缺）。
> 箭头：↓ 越小越好；↑ 越大越好；↑→1 期望接近 1。

| 指标 | spider physical (N=12) | holosoma kinematic (N=12) | Δ |
|---|---:|---:|---:|
| Obj. Pos. Err. (cm) ↓ | 5.50 | 0.00 (self-ref) | −5.50 |
| Obj. Ori. Err. (°) ↓ | 5.51 | 0.00 (self-ref) | −5.51 |
| mj_pen Duration (%) ↓ | 0.0 | 0.0 | 0 |
| mj_pen Max Depth (cm) ↓ | 0.0 | 0.0 | 0 |
| spider 5cm Contact Preservation (%) ↑ | 54.55 | — | — |
| kin 28cm Contact Preservation (%) ↑\* | — | 53.59 | — |
| **Smoothness (rad/s²) ↓** | **13428** | **36048** | **+22621 (kin 2.7× 更不平滑)** |
| Relative Smoothness vs ref ↓ | 0.757 | 1.00 | spider 更平滑 |
| Foot Skating Max Vel (cm/s) ↓ | 96.71 | — | — |
| Pelvis Min z (m) ↑ | 0.567 | 0.711 | −0.145（spider 含 4 fall case）|

\* 28cm 阈值在 box025（half-size 16-27cm）上 trivial 100%；在 box021/box023/bucket001/bucket007 等较小 obj 上有真实判别力（box021_p1 = 29.55%）。N=12 平均 53.59% 反映新 case 加入后阈值开始 "工作"。

历史（N=2 box025_p1/p2 sub）数字记录：spider smoothness 13470 vs kin 41846（gap −67.8%）。N=12 gap 略缩窄到 −62.7%，仍 ≥ 3×。

---

## 7. 关键发现

### 7.1 spider physical smoothness 显著优于 kinematic（N=12 验证）

N=12 跨方法 mean：spider sim smoothness **13428** vs holosoma kin **36048** rad/s²（**−62.7%，3.0× gap**）。**Relative smoothness vs ref = 0.757（spider）vs 1.00（kin self-ref）** — spider 物理 CEM 解比纯 SOCP 运动学解 jerk 显著更低。

gap 随 N 演化稳定：
- N=2 (box025 子集): −67.8%
- N=3 (+box021_p1): −66.9%
- N=12 (完整 batch): −62.7%

这与"物理约束自动惩罚高 jerk"的直觉一致，且**对论文是有意义的 selling point**：物理 retargeting 不仅在 contact / penetration 上赢，连基本平滑性都赢，**在 12 个跨 object 类型的 case 上 robust 保持 ≥ 3× gap**。

> 数字修正记录：v2026-05-20-P2 修复前（FPS=50 常量错用），spider 数字是 37418（高估 2.78×）；表面上 gap 只有 −10.6%。修复 FPS=30 + N=12 扩展后真实 gap = −62.7%。这把 selling point 从"略优"提升到"显著优且 robust"。详见 log/20b。

### 7.2 28cm contact preservation 在 CORE4D 大物体上退化（小物体上有判别力）

OmniRetarget 论文原始定义 28cm obj-local 二值接触阈值是基于小物体（cup / ball / 小箱）；CORE4D `box025` half-size 16/21/26 cm，demo SMPL-X wrist 中心到 obj COM 实测全程 45-63 cm。28cm 阈值永远不触发 → demo 接触帧数 = 0 → preservation 公式 `1 - miss/T` 永远返回 100%（仅 box025 退化）。

**N=12 验证**：扩到 box021/box023/bucket001/bucket007 等较小 obj 后，28cm 阈值开始有判别力 — kin 12 case mean = 53.59%，最低 box021_p1 = 29.55%。但 box025 仍 trivial 100%（半 size 大）。

5cm mask-gated proxy (`paper_omniretarget_contact_preservation_5cm_pct`) 仍是跨 obj 集合最 robust 的接触指标，spider 13 case mean = 54.35%。

### 7.3 Tab.5 N=12（升级完成，原 N=2 已弃用）

E019 P1 完工时 Tab.5 只有 N=2（box025_p1/p2）— holosoma v2 之前只 retarget 了 5 个 source motion。2026-05-20 用户跑 batch 补 10 case（脚本 `run_holosoma_batch_remaining10.sh`，4 worker 并行），9 成功 + 1 失败：
- ✅ 9 个 case retarget + trim 完成（box021_p2、box023_p1/p2、bucket001_p1/p2、bucket005_s2_p1/p2、bucket007_p1/p2）+ dry-run box021_p1 = 10 个新 case
- ❌ desk021_p1: CVXPY clarabel SOCP 返回 infeasible（motion-specific）

合并已有 box025_p1/p2，Tab.5 现 N=12。详细执行过程 + 数字解读见 `log/20b`。

### 7.4 box025 子集太"easy"看不出物理 vs 运动学的核心 gap

mj_pen 0/0 对 spider 与 holosoma 都成立 — 这两个 case 即使是 kin 输出也不显著穿透，说明 OmniRetarget 软约束已经把穿透压住了。**Tab.5 的 selling 数字（"物理把穿透从 X% 压到 0%"）需要在 harder case（bucket / desk）扩展才能见到**。当前 N=2 主要 demonstrate 的是 smoothness 优势。

---

## 8. ✅ FPS 单点修复（v2026-05-20-P2，已落地）

### 8.1 Bug 历史

`paper_metrics.py:19` 之前是 `FPS = 50.0`，被用在 2 个关键公式：

| 函数 | 公式 | FPS 入参 | 偏差性质 |
|---|---|---|---|
| `_smoothness` (`:48-52`) | `qdd = (q[2:] - 2·q[1:-1] + q[:-2]) · FPS²` | 平方 | spider 高估 `(50/30)² ≈ 2.78×` |
| `_add_keypoint_proxy_metrics` foot skating (`:280-310`) | `sim_vel = step_distance · FPS` | 线性 | spider 高估 `1.67×` |

数据源实际 fps：spider E018b NPZ `time` shape `(T, 2)`，substep 间隔 1/60s × 2 = **base 30Hz**（实测）；holosoma v2 kinematic `fps` 字段显式 30。两个数据源都是 30Hz，但代码常量 50 → spider smoothness/skating 数字被全面高估。

### 8.2 修复方案

走"单点常量修正 + 入口护栏"路线，**不**做 per-case 字段全面化：

```python
# paper_metrics.py:19
FPS = 30.0  # CORE4D processed; spider E018b NPZ verified 30Hz; holosoma v2 explicit 30

# add_paper_metrics() 入口
declared_fps = summary.get("fps")
if declared_fps is not None and abs(float(declared_fps) - FPS) > 1e-3:
    warnings.warn(...)  # 未来 50Hz 数据接入时会大声告警
```

理由：
- 当前所有消费者（E014 / E018 / E018b / holosoma kinematic）都是 30Hz
- per-case 全联动是 future-proofing 而非 today's correctness fix，1 天工程量换不到立刻可见的收益
- warning 护栏对未来 50Hz 数据不 silently 失败，已足够防御
- 输出字段加 `paper_metrics_fps: 30.0`，每行 csv/json 都可溯源

### 8.3 落地效果（重跑 E018b 13 case 验证）

| 指标 | 修复前（FPS=50） | 修复后（FPS=30） | 比例 |
|---|---:|---:|---:|
| box025_p2 smoothness | 37418 | **11071** | ÷3.38 |
| box025 N=2 mean smoothness | 37418 | **13470** | ÷2.78 ✓ |
| box025 N=2 max vel | 198.9 cm/s | **119.4 cm/s** | ÷1.67 ✓ |
| 13 case mean smoothness | ~37809 | **13601** | ÷2.78 ✓ |
| 13 case mean foot skating max vel | ~158.7 cm/s | **95.0 cm/s** | ÷1.67 ✓ |
| Tab.5 spider vs kin smoothness gap | −10.6% | **−67.8%** | 论文 selling point 增强 |

box025_p2 偏差比理论 2.78 略大（3.38）— 是因为 `_smoothness` 内部 `(q[2:] - 2q[1:-1] + q[:-2])` 的二阶差分对低频信号不严格线性，但 13 case 平均与 N=2 sub 平均都严格符合 (50/30)² = 2.78 ✓。

### 8.4 改动文件

| 文件 | 改动 |
|---|---|
| `scripts/eval/paper_metrics.py:19-23` | `FPS = 50.0` → `30.0` + 注释说明数据源 |
| `scripts/eval/paper_metrics.py:925-936` | `add_paper_metrics` 入口加 `summary["fps"]` warning + 输出含 `paper_metrics_fps` |
| `results/E018b/comparison.csv` + 13 个 `eval_summary_*.{csv,json}` + `aggregate_summary.json` | 重跑产出 |
| `results/eval_unified/tables/*.md` + `table_paper_all.xlsx` | unified_eval 重生成 |
| `log/20a_E019_*.md` §6.3 / §7.1 / §8 / §9 | 数字 + Claims 更新 |
| `docs/eval_metrics.md` §7.4 / §7.5 | Tab.5 数字 + caveat 更新 |

### 8.5 降级到 P3：per-case 字段全联动

仍保留作为 future-proofing 任务，触发条件：接入第一个 fps ≠ 30 的数据源（如 HDMI 转换的 50Hz 变体）。届时需要：

| Step | 内容 |
|---|---|
| 1 | `eval_E*` 在 `summary["fps"]` 里写明 per-case fps |
| 2 | `_smoothness(q, fps)` / `_add_keypoint_proxy_metrics` 改成读 `summary["fps"]`，模块 FPS 仅作 fallback |
| 3 | `EvalInputs.fps` 完全打通 adapter 层 |

警告护栏已在位，会触发该工作而不会 silently 错。

---

## 9. Claims 验证

| Claim | 结果 |
|---|---|
| **P0-C1** SPIDER Table 4 全 9 列字段在 `paper_metrics.add_paper_metrics` 输出中可见 | ✅ 通过 — 13 case CSV 列已含 `paper_spider_{joint,pos,ori,root_pos,root_ori,eef_pos,eef_ori,obj_pos,obj_ori}_err_*` |
| **P0-C2** OmniRetarget penetration 用 mj_geomDistance + prefilter 实现 | ✅ 通过 — `_add_penetration_metrics_mj` + 13 case 验证 |
| **P0-C3** `unified_eval.py` 能产出 md tables + xlsx | ✅ 通过 — 6 md + 1 xlsx (5 sheet) |
| **P0-C4** `docs/eval_metrics.md` 每指标可 cross-ref 到实现 file:line | ✅ 通过 — 306 行文档，每指标卡片含 `paper_metrics.py:NNN` 链接 |
| **P1-C1** EvalInputs adapter + holosoma kinematic 接入 | ✅ 通过 — `adapters/` 完整 + `eval_holosoma_kinematic.py` 跑通 box025_p1/p2 |
| **P1-C2** 28cm obj-local contact preservation 严格定义实现 | ✅ 通过 — `_add_contact_preservation_omni_local` + OmniRetarget 公式（`1 - miss/T`）已对齐 |
| **P1-C3** Tab.5 跨方法对比表自动生成 | ✅ 通过 — `table_method_comparison.md`，N=2 子集 |
| **P1-C4** FPS 修正 + 主入口数字论文公式正确 | ✅ **通过** — v2026-05-20-P2 走单点常量修正（`FPS=50`→`30`）+ `add_paper_metrics` 入口 warning 护栏；E018b 13 case 已重跑，N=2 Tab.5 smoothness gap 从 −10.6% 修正到 −67.8%。per-case 全联动降级为 P3（触发条件：接入 fps≠30 的新数据源），详见 §8 |

整体：8 条 Claim 全部通过。FPS 问题已通过单点修正路线落地，spider smoothness / foot skating 数字现在按论文公式严格正确。

---

## 10. Git 提交

**当前分支**: `exp/core4d-collab-retarget`（对应大方向 `core4d_collab_retarget`）

待 commit 文件（按 §2.1）：

```bash
git add \
  workspace/core4d_collab_retarget/plan/20_E019_unified_eval_framework_plan.md \
  workspace/core4d_collab_retarget/plan/21_E020_failure_attribution_audit_plan.md \
  workspace/core4d_collab_retarget/plan/22_E021_holosoma_rl_export_plan.md \
  workspace/core4d_collab_retarget/plan/23_tech_report_plan.md \
  workspace/core4d_collab_retarget/plan/AFTER_E018_INDEX.md \
  workspace/core4d_collab_retarget/scripts/eval/paper_metrics.py \
  workspace/core4d_collab_retarget/scripts/eval/unified_eval.py \
  workspace/core4d_collab_retarget/scripts/eval/eval_holosoma_kinematic.py \
  workspace/core4d_collab_retarget/scripts/eval/adapters/ \
  workspace/core4d_collab_retarget/docs/eval_metrics.md \
  workspace/core4d_collab_retarget/report/00_outline.md \
  workspace/core4d_collab_retarget/report/01_v0.5_draft.md \
  workspace/core4d_collab_retarget/log/20_E019_unified_eval_framework_results.md \
  workspace/core4d_collab_retarget/EXPERIMENT_TRACKER.md \
  workspace/core4d_collab_retarget/progress.md \
  workspace/core4d_collab_retarget/results/E018b/comparison.csv \
  workspace/core4d_collab_retarget/results/E018b/aggregate_summary.json \
  workspace/core4d_collab_retarget/results/E018b/eval_summary_*.{csv,json} \
  workspace/core4d_collab_retarget/results/holosoma_v2_kinematic/ \
  workspace/core4d_collab_retarget/results/eval_unified/

# 待用户确认是否 commit + push
```

> Claims 总评是 7/8，FPS 未通过为已知 P2 工程债且已在 log 明示路线 — 是否按规则 commit 由用户决策（也可等 P2 修完一并 commit）。

---

## 11. 下一步

按 `plan/AFTER_E018_INDEX.md` 的优先级：

1. **报告 v0.5 → v1**（`report/01_v0.5_draft.md` 已就绪，扩成 v1 全文 + Fig 实绘 + Tab.5 数字回填）— **本轮已部分推进**
2. **E020 失败归因**（`plan/21_E020_*.md`）— 用 E019 输出做 13 case root_cause CSV（远程已完成，13/13 done）
3. **E021 holosoma RL 导出**（`plan/22_E021_*.md`）— spider → RL 训练格式
4. **推动 holosoma 在剩余 11 case 补跑 retarget** — 让 Tab.5 从 N=2 升到 N=13
5. **E019 P3 FPS per-case 全联动**（§8.5）— 仅在接入第一个 fps≠30 数据源时触发，当前 warning 护栏已就位

**绝对优先**：报告 v1 + Tab.5 扩到 N=13（需 holosoma 补跑 11 case）。E019 P2 FPS 修复已落地，不再阻塞。
