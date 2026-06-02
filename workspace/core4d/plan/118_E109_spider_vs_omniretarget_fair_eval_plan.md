# E109 Plan: Spider vs OmniRetarget fair-eval 指标实验

日期：2026-06-02

关联输入：

- 调研文档：`workspace/core4d/report/spider_vs_omniretarget_eval/metric_research_and_protocol.md`
- 历史 13case：`workspace/core4d_collab_retarget/results/E026_full_eval/method_case_metrics.csv`
- 数据状态缓存：`workspace/core4d/data_construction_v3/existing_cases.tsv`
- 新近 Spider CEM：`workspace/core4d/results/E105`、`E106`、`E107`、`E108`
- 历史日志：`workspace/core4d_collab_retarget/log/26_E026_full_eval_results.md`、`workspace/core4d/log/132_E105_box026_clean_scene_full_cem_rerun_results.md`、`133_E106_box026_30candidate_ref_fk_batch_results.md`、`135_E107_box021_selected4_full_cem_results.md`、`138_E108_nonbox_cem_and_rl_handoff.md`

## 0. 定位

E109 是一个评测协议实验，不跑新的 CEM/RL。目标是把 Spider 与 OmniRetarget 的比较从“对各自 retarget reference 的 tracking 表”改成“尽量依赖 Core4D raw / 几何 / 物理 rollout 的公平指标表”。

本轮先落地可复用脚本和首版结果：

```text
workspace/core4d/scripts/eval_omni_vs_spider/
workspace/core4d/results/E109/fair_eval/
workspace/core4d/results/E109/unified_replay_eval/
workspace/core4d/log/139_E109_spider_vs_omniretarget_fair_eval_results.md
```

用户后续确认：主评测 case 集合固定为 `workspace/core4d/data_construction_v3/existing_cases.tsv` 中 `cem_status=pass` 且 `target_variant_id != adaptive` 的 Spider case。E026 与 E105-E108 summary 不再直接进入默认评测 case bank，只作为历史背景和复现实验输入。

用户进一步指出旧表缺失指标过多后，本轮主结果从“拼接历史 summary 的 proxy table”升级为“统一 MuJoCo replay + 历史对齐校验”：对 OmniRetarget 和 Spider CEM 都重新 replay，同一套代码重算 hand/body/leg/object 指标；历史 Spider summary 仅用于校验新 evaluator 是否复现已有指标。

## 1. Claims

| Claim | 验证方式 |
|---|---|
| C1：现有 E026 表不能直接作为 Spider vs OmniRetarget 的公平胜负表 | 脚本必须把 self-ref / method-ref 指标标记为 `not_fair_for_method_comparison`，并在 summary 中单独列出 |
| C2：默认评测 case bank 来自 `existing_cases.tsv` 中 `cem_status=pass` 且排除 `adaptive` route 的 Spider case | `unified_replay_eval.py` 输出统一 replay 表；E026/E105-E108 summary 只作为历史复现输入 |
| C3：所有依赖阈值的指标必须显式标出阈值，且至少 3 个尺度 | 首版统一输出 `3cm/5cm/8cm` 三档 contact/near-contact proxy；penetration/deep-penetration 也输出阈值字段 |
| C4：公平主表只把 raw/geometry/physics 指标用于 method comparison | `method_case_metrics.tsv` 区分 `fair_metric_scope` / `fair_metric_ready`，summary 不用 self-ref object/body tracking 判胜负 |
| C5：脚本可复用到未来任意 E### | CLI 支持显式输入 E026、existing cases、E105-E108 summary，也支持之后新增 `--cem-summary` |
| C6：补齐缺失指标前必须证明新 evaluator 口径正确 | `unified_replay_eval.py` 对历史已有 Spider 指标做逐项校验；已有指标 mismatch 必须为 0，或明确解释历史/新实现哪一方有 bug |

## 2. 指标口径

### 2.1 首版可直接落地

| 指标 | 阈值/尺度 | 来源 | 方向 |
|---|---|---|:-:|
| `hand_near_frac_{03cm,05cm,08cm}` | 3cm / 5cm / 8cm | `hand_box_sdf_min_m` timeseries 或 summary 近似 | ↑ |
| `hand_object_penetration_frac` | SDF < 0 | hand/object SDF | ↓ |
| `deep_penetration_frac_2cm` | SDF < -2cm | E026/E081-style deep penetration | ↓ |
| `leg_object_interference_frac` | leg SDF < 0 | E081/E105 lower-body proxy | ↓ |
| `leg_near_frac_{03cm,05cm,08cm}` | 3cm / 5cm / 8cm | leg SDF timeseries | ↓ |
| `pelvis_min_m` / `fall_flag` | pelvis < 0.45m | rollout qpos/eval summary | ↑ / ↓ |
| `obj_err_mean_m` / `obj_err_max_m` | method ref diagnostic only unless raw object pose is attached | CEM summary | ↓ |
| `cem_status` / `rl_status` | pass/fail/not_run | registry/evidence | ↑ |

### 2.2 需要 raw GT 后增强

| 指标 | 需要输入 |
|---|---|
| raw object SE(3) error | `smooth_objposes.npy` 或 raw GT manifest |
| raw contact precision/recall/F1 | S1 raw contact mask + method FK/SDF |
| foot skating | raw stance label 或统一 foot velocity-height rule |
| mocap semantic pose fidelity | raw SMPL-X joints/vertices + robot semantic keypoints |

## 3. 脚本设计

新增目录：

```text
workspace/core4d/scripts/eval_omni_vs_spider/
  README.md
  common.py
  build_case_bank.py
  compute_proxy_metrics.py
  summarize_fair_eval.py
  build_existing_cases_comparison.py
  unified_replay_eval.py
  run_E109_fair_eval.sh
  run_existing_cases_only_eval.sh
```

职责：

| 脚本 | 作用 |
|---|---|
| `build_case_bank.py` | 从 E026、E105-E108 summary、`existing_cases.tsv` 生成统一 case bank |
| `compute_proxy_metrics.py` | 读取 case bank 和 legobj/timeseries，输出多阈值 proxy 指标 |
| `summarize_fair_eval.py` | 聚合 method/case/object 维度，生成 `summary.md` |
| `build_existing_cases_comparison.py` | 生成 existing CEM pass case 的 OmniRetarget/Spider 配对表，主要用于发现覆盖缺口 |
| `unified_replay_eval.py` | 当前主入口；对 OmniRetarget 和 Spider CEM 统一 MuJoCo replay，并对历史已有 Spider 指标做一致性校验 |
| `run_E109_fair_eval.sh` | 固化 E109 默认命令 |

## 4. 输出

```text
workspace/core4d/results/E109/fair_eval/
  case_bank.tsv
  method_case_metrics.tsv
  method_summary.tsv
  object_summary.tsv
  threshold_summary.tsv
  summary.md
  warnings.tsv
```

当前主结果：

```text
workspace/core4d/results/E109/unified_replay_eval/
  unified_omni_vs_spider_comparison.md
  unified_case_comparison_full.md
  unified_method_metrics_full.md
  history_validation_full.md
  unified_omni_vs_spider_comparison.xlsx
  unified_case_comparison.tsv
  unified_method_metrics.tsv
  history_validation.tsv
  unified_method_summary.tsv
  run_summary.json
```

## 5. 成功标准

| 标准 | 阈值 |
|---|---|
| 脚本可运行 | `run_E109_fair_eval.sh` 退出码 0 |
| case bank 覆盖 | 至少包含 E026 13case 历史 rows 与 `existing_cases.tsv` 中 Spider CEM pass rows |
| 多阈值 | 输出中至少存在 3cm/5cm/8cm 三档指标列和 threshold summary |
| self-ref caveat | summary 明确列出 `obj_pos_cm=0` 等 self-ref 指标不可作为公平胜负 |
| 历史对齐 | `history_validation.tsv` 中已有历史指标全部 `PASS`，当前为 87/87 |
| 缺失补齐 | hand/body/leg/object replay 指标由 `unified_replay_eval.py` 对两方法统一补齐；Spider CEM 特有 `obj_err` 不给 OmniRetarget 硬填 |
| 可复用 | CLI help 可读，默认路径可覆盖 |

## 6. 本轮不做

- 不重跑 OmniRetarget。
- 不新增 CEM/RL。
- 不把 E026 的 self-ref object/body tracking 解释为 OmniRetarget 真实优势。
- 不在 raw object/contact GT 尚未接入时声称已经完成最终 paper-level fair comparison。
- 不把 Spider CEM `sim/ref` object error 硬迁移成 OmniRetarget 指标。

## 7. 复现命令

后续默认评测：

```bash
python3 workspace/core4d/scripts/eval_omni_vs_spider/unified_replay_eval.py
```

历史 summary 拼接复现：

```bash
bash workspace/core4d/scripts/eval_omni_vs_spider/run_existing_cases_only_eval.sh
```

E109 历史复现：

```bash
bash workspace/core4d/scripts/eval_omni_vs_spider/run_E109_fair_eval.sh
```

可扩展输入：

```bash
python3 workspace/core4d/scripts/eval_omni_vs_spider/build_case_bank.py \
  --out-dir workspace/core4d/results/E109/fair_eval \
  --existing-cases workspace/core4d/data_construction_v3/existing_cases.tsv \
  --e026-metrics workspace/core4d_collab_retarget/results/E026_full_eval/method_case_metrics.csv \
  --cem-summary workspace/core4d/results/E105/cem/full/full_eval_summary.csv \
  --cem-summary workspace/core4d/results/E106/cem/full/full_eval_summary.csv \
  --cem-summary workspace/core4d/results/E107/cem/full/full_eval_summary.csv \
  --cem-summary workspace/core4d/results/E108/s6_downstream/cem/full/E108_cem_eval_summary.csv
```
