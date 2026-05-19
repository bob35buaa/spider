# E019 计划：统一评测框架（task_afterE018 §1）

> **状态**：草案，等用户确认范围后实现。
> 上游：`task_afterE018.md` §1 — 对齐 OmniRetarget Table 2 + SPIDER Table 4 + 运动学重定向指标，产出统一脚本 + Markdown/xlsx 表格 + 论文级 `docs/eval_metrics.md`。
> 上游分支：`exp/core4d-collab-retarget`，编号继 E018b 之后用 E019。

---

## 0. 关键前置事实（必读）

1. **E018b 13 NPZ + 13 MP4 + aggregate + comparison.csv 已就绪**（`results/E018b/`），无阻塞。
2. **spider qpos npz 字段第二维是 parallel env，不是 person**：现有 `eval_E018b.py:136` / `eval_E072.py:42` 已有 `flatten_time_major` adapter；新统一脚本必须复用同一规范化。
3. **现成 SPIDER Table 4 实现已在 `workspace/core4d/scripts/eval/eval_comprehensive.py:119`**：MPKPE / Joint / EEF / Root / Obj / Stability / Skating / Contact / Smoothness 都实现过，**不要重写，而是迁移并对齐到 paper_metrics.py 风格**。
4. **`workspace/core4d/docs/eval_metrics.md` 已存在框架**，新 `core4d_collab_retarget/docs/eval_metrics.md` 在它基础上做"双人协作 + freejoint object + canonical anchor"扩展，不重新发明。
5. **E081 不是 freejoint**：是 `scene_act` 6-DoF actuator-guided object，`nq_obj=6`；E014/E018/E018b 才是 true freejoint，`nq_obj=7`。报告/文档对比时必须说清。

---

## 1. 目标

A. **一套统一评测脚本** `scripts/eval/unified_eval.py`：
   - CLI：`--method {spider_E018b|spider_E014|holosoma_v2_kinematic|...} --cases all|filter --out results/eval_unified/...`
   - 输入屏蔽：通过 `adapters/{spider,kinematic}_to_common.py` 把异构 NPZ → 公共 `EvalInputs` dataclass
   - 输出：`per_case/{method}/{case}.json`、`tables/*.md`、`tables/*.xlsx`、`plots/*.png`
B. **完整指标对齐三个论文 / 一个内部**：SPIDER Table 4（Joint/Pos/Ori/Obj.Pos/Obj.Ori Err）、OmniRetarget Table II（Penetration / Foot Skating / Contact Preservation）、DynaRetarget Table V（smoothness/success）、core4d 自定义（pelvis stability、carry progress、deep penetration 2cm gate）。
C. **论文级文档** `docs/eval_metrics.md`：每指标 → 中英名 / 数学公式 / 输入字段 / 阈值方向 / 实现 `file:line` / 已知 caveat。
D. **跨方法对比表** `tables/table_all_methods.xlsx`（multi-sheet：raw / mean±std / by_object / failures）。

---

## 2. 指标全集（实现优先级）

### P0 必做：SPIDER Table 4 一对一对齐

| 新增函数 | 论文符号 | 公式 | 输入字段 | 备注 |
|---|---|---|---|---|
| `_add_body_tracking_metrics` | Joint Err. (deg) | `mean |q_sim - q_ref| · 180/π`, robot 29 dof | qpos[:,7:36] sim+ref | 论文方法不是 mpkpe_proxy，是 J=29 关节直接平均 |
| 同上 | Pos. Err. = MPKPE (cm) | `mean ‖xpos_sim^k - xpos_ref^k‖·100`, K=nbody−2 | FK xpos | 跑一遍 mj_kinematics，全身 body |
| 同上 | Ori. Err. (deg) | `mean 2·arccos(|xquat_sim^k · xquat_ref^k|)·180/π` | FK xquat | 同上，全身 body |
| `_add_root_metrics`（拆出） | Root Pos / Ori Err | pelvis 专门版 | xpos/xquat[pelvis] | 当前只有 pelvis 位置 |
| `_add_eef_metrics` | EEF Pos / Ori Err | L/R wrist 专门版 | xpos/xquat[wrist] | 缺 |
| Obj. Pos / Ori Err | **已实现** | `_add_object_tracking_metrics` | qpos[-7:] | ✓ |

### P0 必做：OmniRetarget Penetration 真实算法

当前 `_add_contact_and_penetration_metrics` 是消费预算好的 csv，不是真正跑 `mj_geomDistance`。

| 新增函数 | 公式 | 关键依赖 |
|---|---|---|
| `_add_penetration_metrics_mj` | Penetration Duration: `frac(t : ∃ pair, sdf < -0.01m)` ；Max Depth: `max_t max_pair (-sdf)·100` | 用 `holosoma/workspace/v1/scripts/eval_paper_metrics.py:67-92` 的 `_prefilter_collision_pairs` + `evaluate_penetration` 思路；对 robot↔object 与 robot↔ground 两类分别评 |

### P1 重要：跨方法对齐 + 数据适配器

| 新增 | 说明 |
|---|---|
| `adapters/spider_to_common.py` | `(T,2,43) → (T,43)` 取 env 0；提取 qpos_sim/qpos_ref/ctrl/sim_dt；从 hydra config 重建 ref（参考 `eval_E018b.py` 的 case 加载逻辑） |
| `adapters/kinematic_to_common.py` | 读 `holosoma/workspace/v2/data/core4d_replace_batch/*.npz`；qpos `(T,43)` 直接当 sim_qpos，缺 ref 时用 demo 自重排或 mark "ref unavailable" |
| `adapters/omniretarget_inputs.py` | 输出 `{qpos, human_joints, object_poses_wxyz_then_xyz, fps}` 给 `eval_paper_metrics.py`-style 函数 |
| Per-case `fps` 字段 | 把 `paper_metrics.FPS=50` 常量改成 per-case；spider 写 `1/sim_dt`，kinematic 写 30 |

### P1 重要：OmniRetarget Contact Preservation 28cm local-frame 版

| 新增 | 说明 |
|---|---|
| `_add_contact_preservation_omni` | obj-local 28cm 二值版（`eval_paper_metrics.py:242,258`），输入需要 demo 22-joint（adapter A1 从 SMPL-X 取） |

### P2 可选：增益分析

- 全身 jerk / EEF smoothness
- per-task / per-object 聚合（xlsx 多 sheet）
- 失败 case 自动归因 → 复用 E019 输出，由后续 E020（plan 21）消费

---

## 3. 目录结构

```
workspace/core4d_collab_retarget/
├── docs/
│   └── eval_metrics.md                  # 论文级文档（新）
├── scripts/eval/
│   ├── paper_metrics.py                 # 现有，按 P0/P1 扩展
│   ├── adapters/
│   │   ├── common_inputs.py             # EvalInputs dataclass + fps + person_idx + mask_path
│   │   ├── spider_to_common.py          # E014/E018/E018b NPZ → EvalInputs
│   │   ├── kinematic_to_common.py       # holosoma v2 → EvalInputs
│   │   └── omniretarget_inputs.py       # → eval_paper_metrics.py 期望格式
│   ├── unified_eval.py                  # 唯一入口
│   ├── aggregate.py                     # md + xlsx 聚合
│   └── plot_panels.py                   # 箱线 / 散点 / 雷达
└── results/eval_unified/
    ├── per_case/{method}/{case}.json
    ├── tables/
    │   ├── table_spider_t4.md / .xlsx
    │   ├── table_omniretarget_t2.md / .xlsx
    │   ├── table_dynaretarget_t5.md / .xlsx
    │   └── table_all_methods.xlsx       # multi-sheet
    └── plots/
```

`docs/eval_metrics.md` 章节：
1. 数据流总览（spider vs kinematic 字段图）
2. 每指标卡片（公式 + 实现 file:line + 阈值方向 + caveat）
3. 任务级二值成功（DynaRetarget / Transport）
4. 复现命令（含 adapter 示例）
5. 与论文口径已知差异

---

## 4. 实施步骤（验证驱动）

| Step | 内容 | Verify |
|---|---|---|
| 1 | 写 `adapters/common_inputs.py` + 两个 spider/kinematic 适配器 | 单测：在 E018 `box023_p2` NPZ 上 `dump_eval_inputs(...)` 字段完整、shape 正确 |
| 2 | 把 `eval_comprehensive.py` 的 SPIDER Table 4 函数迁到 `paper_metrics.py`，命名 `paper_spider_*` | E014 跑 6 case，aggregate `num_paper_spider_*` 全字段非空 |
| 3 | 实现 `_add_penetration_metrics_mj`，复用 holosoma `_prefilter_collision_pairs` + `evaluate_penetration` | E018 2 case 跑通；与现有 csv-based deep-pen 对比，差异 <10% |
| 4 | 实现 `unified_eval.py` CLI；先支持 `--method spider_E014`、`--method spider_E018` | `--method spider_E018 --cases all` 产出 `per_case/spider_E018/*.json` + `tables/table_spider_t4.md` |
| 5 | 实现 `--method holosoma_v2_kinematic` adapter；跑 `core4d_replace_batch/` 同名 case | aggregate 表里 spider vs kinematic 列同时出现 |
| 6 | 实现 28cm local-frame contact preservation（需 SMPL-X 22 关节，从 `example_datasets/processed/core4d/.../joints.npy` 或 source pkl 取） | E014 + holosoma kinematic 对照 |
| 7 | `aggregate.py` 产出 xlsx multi-sheet | 检查 xlsx 包含 raw/mean±std/by_object/failures 4 sheet |
| 8 | 写 `docs/eval_metrics.md` | 每指标都能 cross-ref 到 `paper_metrics.py` 的具体函数和 `file:line` |
| 9 | 用一致输入 rerun `eval_E014.py` / `eval_E018.py`，确保旧 aggregate 与新 `unified_eval.py` 数字一致（容忍 <1% 浮点差） | 双系统一致性 gate |
| 10 | 若 E018b 13 case 数据已恢复，全跑一次三方法对照表 | `tables/table_all_methods.xlsx` 含 spider_E014/spider_E018b/holosoma_v2 三列 |

---

## 5. 成功标准

- ✅ `paper_metrics.py` 输出包含 SPIDER Table 4 全 5 列 + OmniRetarget Table II 全 3 列 + DynaRetarget Table V smoothness/success + core4d 自定义指标
- ✅ `unified_eval.py --method` 至少支持 `spider_E014` / `spider_E018` / `spider_E018b`（若数据恢复）/ `holosoma_v2_kinematic`
- ✅ `tables/table_all_methods.xlsx` multi-sheet 跨方法可比
- ✅ `docs/eval_metrics.md` 每指标可 cross-ref 到实现 `file:line`
- ✅ 旧 eval 脚本仍可运行，且与 unified 输出对账 <1% 差异
- ✅ 在 `results/eval_unified/` 同时落盘 md + xlsx + json + plot

## 6. 已知风险

1. **SMPL-X 22 关节来源**（P1 OmniRetarget contact preservation 用）— `example_datasets/processed/core4d/.../joints.npy` 路径需确认；如无，需从 `raw/core4d/*/smplx.pkl` 现场 forward。本轮 P0 不阻塞。
2. **FPS 统一口径** — spider 60Hz、holosoma kinematic 30Hz、OmniRetarget eval 30Hz；建议 unified_eval 在 30Hz 评测面统一（向下降采样 spider），同时保留 spider 60Hz 原值供 smoothness 等高频指标用。
3. **是否要把 `eval_E*.py` 全替换为 `unified_eval --legacy E0NN`** — 倾向保留旧 per-experiment eval，但 import 共同函数。

## 7. 时间预算（粗估）

| Phase | 内容 | 时间 |
|---|---|---|
| A | Adapter + SPIDER Table 4 迁移 | 0.5 day |
| B | OmniRetarget penetration 算法实现 | 0.5 day |
| C | unified_eval + aggregate + xlsx | 0.5 day |
| D | docs/eval_metrics.md 撰写 | 0.5 day |
| E | E014 / E018 / E018b / kinematic 4 方法跑通 + 对账 | 0.5 day |
| **合计** | | **2.5 day** |
