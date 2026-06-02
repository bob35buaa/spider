# E109 Spider vs OmniRetarget 公平评测协议首版结果

日期：2026-06-02

## 不看细节先看这里

这个实验要解决的问题很简单：**以前的 Spider vs OmniRetarget 指标表里，有些“GT”其实来自 OmniRetarget 或各自方法的 reference，所以不能直接拿来证明 Spider 比 OmniRetarget 好。**

E109 没有重跑算法，做的是评测协议整理：

1. 把历史 E026、近期 Spider CEM 结果 E105-E108、以及 `existing_cases.tsv` 里已经验证过的 Spider CEM/RL case 放进同一个 `case_bank.tsv`。
2. 把 E026 这类历史 method-reference 指标标成 `diagnostic_only`，意思是“只能当背景，不能判胜负”。
3. 先输出一批不直接依赖 OmniRetarget GT 的 proxy 指标：手和物体距离、手/物体穿透、腿/物体干涉、pelvis/fall。
4. 所有接触距离指标都按 `3cm / 5cm / 8cm` 三档输出，避免“阈值不同但数值放一起比”的问题。

所以当前结论不是“Spider 已经赢了 OmniRetarget”，而是：

- 旧表不能直接用于公平比较；
- 我们已经有了一个可复用的公平评测入口；
- 下一步必须给 OmniRetarget 也补同口径 geometry proxy，并接入 Core4D raw mocap/object/contact GT，才能形成最终 paper-level 对比。

## 后续 case 集合更新

用户确认后续评测不再混入 E026 或 E105-E108 summary case。默认评测 case 只来自：

```text
workspace/core4d/data_construction_v3/existing_cases.tsv
```

并且只取：

```text
cem_status=pass
```

新的默认入口为：

```bash
bash workspace/core4d/scripts/eval_omni_vs_spider/run_existing_cases_only_eval.sh
```

E026 和 E105-E108 summary 只保留为历史背景、调试复现或协议说明，不再作为后续主评测 case bank。

## 完整配对对比表

根据用户进一步要求，新增完整的 OmniRetarget vs Spider 配对对比表。case 集合仍严格使用 `existing_cases.tsv` 中 `cem_status=pass` 的 12 条 Spider case。

生成命令：

```bash
python3 workspace/core4d/scripts/eval_omni_vs_spider/build_existing_cases_comparison.py
```

输出目录：

```text
workspace/core4d/results/E109/omni_vs_spider_existing_cases/
```

核心产物：

| 文件 | 说明 |
|---|---|
| `omni_vs_spider_comparison.md` | Markdown 摘要表：方法汇总 + 逐 case 关键指标 |
| `omni_vs_spider_case_metrics_full.md` | Markdown 完整逐 case 指标表，字段与 xlsx `逐case对比` sheet 一致 |
| `omni_vs_spider_comparison.xlsx` | xlsx 完整表，包含 `逐case对比`、`方法汇总`、`指标定义`、`覆盖问题` 四个 sheet |
| `omni_vs_spider_case_metrics.tsv` | 逐 case 全字段机器可读表 |
| `metric_definitions.tsv` | 指标方向、单位、来源与注意事项 |
| `coverage_warnings.tsv` | 覆盖问题；当前为空 |

覆盖验证：

| 项目 | 结果 |
|---|---:|
| existing CEM pass case | 12 |
| OmniRetarget 轨迹覆盖 | 12/12 |
| Spider CEM npz + summary 覆盖 | 12/12 |
| coverage warnings | 0 |

方法汇总：

| 方法 | case数 | ready | pelvis均值 | 跌倒case | root位移 | 物体XY位移 | 物体高度变化 | 物体误差 | 8cm接触 | 腿部干涉 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| OmniRetarget | 12 | 12 | 0.689350 | 0 | 1.349992 | 1.402908 | 0.435075 |  |  |  |
| Spider CEM | 12 | 12 | 0.669608 | 0 | 1.383725 |  |  | 0.021147 | 0.548565 | 0.006714 |

注意：OmniRetarget 侧目前是轨迹级指标；Spider CEM 侧的接触、物体误差、腿部干涉来自 CEM eval summary。OmniRetarget 还没有全量同口径 SDF 接触/穿透 summary，因此表中没有伪造 Omni 的 SDF 接触指标。

## 统一 replay 补齐缺失指标

用户指出上一版表格缺失指标太多，原因是上一版主要在拼接历史 summary：

- OmniRetarget 历史侧通常只有轨迹或 kinematic 输出，没有和 Spider CEM 同口径的 hand/body/leg SDF 接触、穿透、腿部干涉 summary。
- Spider 历史侧不同实验记录字段不完全一致；早期 box004 等 case 没有完整腿部/身体穿透列，E105-E108 又有各自 summary schema。
- 旧 `contact_frac_either` 字段名有误导性。代码复核后确认它实际是 `max(left_contact_frac, right_contact_frac)`，不是左右手 union/either。统一评测必须按这个历史口径校验，否则会出现假 mismatch。

为避免“缺什么就凭感觉补什么”，新增统一 replay evaluator：

```bash
python3 workspace/core4d/scripts/eval_omni_vs_spider/unified_replay_eval.py
```

它对 `existing_cases.tsv` 中 `cem_status=pass` 的 12 条 case，分别读取 OmniRetarget qpos 和 Spider CEM qpos，用对应 `scene.xml/scene_act.xml` 做 MuJoCo replay，然后在同一套实现里重算：

| 指标族 | 是否两方法都可补 | 说明 |
|---|---|---|
| pelvis/fall | 是 | replay 中直接读机器人 root/pelvis 高度 |
| EEF near `3cm/5cm/8cm` | 是 | 左右末端执行器到物体 mesh SDF 的近接触比例，按历史 `max(L,R)` 口径输出 |
| hand geom near/penetration/deep penetration | 是 | 手部 collision/body geom 到物体 SDF |
| leg near/penetration/physics contact | 是 | 腿部 geom 到物体 SDF 与 MuJoCo contact |
| body/head/upper penetration | 是 | 身体非手腿部位到物体 SDF |
| object xy/z motion | 是 | replay 中直接读物体 qpos |
| Spider object error | 仅 Spider CEM | CEM npz 有 `sim/ref` 两通道时可算；OmniRetarget 没有等价 CEM reference，因此不硬填 |

严谨性校验方式：

- 对历史 summary 中已经存在的 Spider 指标，逐项和新 replay 结果比对。
- 能对齐的字段包括 `pelvis_min_m`、`obj_err_mean_m/max_m`、`contact_frac_either`、`leg_box_interference_frac`、`leg_box_near_2cm_frac`、`hand_object_contact_physics_frac`、`head_pen_frac`、`upper_pen_frac`。
- 当前共有 `87` 个历史对齐检查项，`mismatch=0`。因此同一指标在历史有值的 case 上能复现，缺失 case 再由同一个 replay evaluator 补齐。

统一 replay 输出目录：

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

核心结果：

| 方法 | case | pelvis | fall | eef 3cm | eef 5cm | eef 8cm | hand pen | hand deep | leg pen | leg 2cm | body pen | obj xy | obj z | obj err |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| OmniRetarget | 12 | 0.689350 | 0 | 0.285386 | 0.433412 | 0.599325 | 0.510628 | 0.258689 | 0.015230 | 0.027801 | 0.000000 | 1.402912 | 0.435058 |  |
| Spider CEM | 12 | 0.667786 | 0 | 0.070173 | 0.336754 | 0.526711 | 0.414096 | 0.039659 | 0.026535 | 0.048418 | 0.000000 | 1.392898 | 0.363729 | 0.008389 |

因此，上一版表里“缺失很多”的结论现在应更新为：

- hand/body/leg/object 几何 replay 指标已能按统一口径补齐；
- Spider 历史已有字段已通过 `87/87` 对齐校验；
- OmniRetarget 的 `obj_err` 仍不补，因为那是 Spider CEM `sim/ref` 特有诊断指标，不是两方法共同指标。

## 目标

承接用户提出的 Spider vs OmniRetarget 指标对比问题：历史 Spider 重定向指标里有一部分 GT 来自 OmniRetarget 或各自 method reference，这类指标不能直接用于证明 Spider 相对 OmniRetarget 的增益。E109 的目标不是重跑 CEM/RL，而是把已有 E026、E105-E108、`existing_cases.tsv` 中的结果整理成一个可复用的公平评测入口，并明确哪些列只是诊断背景，哪些列可以作为 geometry/proxy 评测。

计划文件：

```text
workspace/core4d/plan/118_E109_spider_vs_omniretarget_fair_eval_plan.md
```

调研协议：

```text
workspace/core4d/report/spider_vs_omniretarget_eval/metric_research_and_protocol.md
```

## 实现

新增可复用脚本目录：

```text
workspace/core4d/scripts/eval_omni_vs_spider/
  README.md
  common.py
  build_case_bank.py
  compute_proxy_metrics.py
  summarize_fair_eval.py
  run_E109_fair_eval.sh
```

默认运行：

```bash
bash workspace/core4d/scripts/eval_omni_vs_spider/run_E109_fair_eval.sh
```

输出目录：

```text
workspace/core4d/results/E109/fair_eval/
  case_bank.tsv
  case_bank_summary.json
  method_case_metrics.tsv
  method_summary.tsv
  object_summary.tsv
  threshold_summary.tsv
  metric_summary.json
  warnings.tsv
  summary.md
```

## 数据覆盖

`case_bank.tsv` 共 `129` 行：

| 来源 | 行数 | 说明 |
|---|---:|---|
| E026 method metrics | 76 | 历史 OmniRetarget / Spider 指标，只作为诊断背景 |
| E105-E108 CEM summary | 41 | 新近 Spider CEM rollout summary |
| `existing_cases.tsv` CEM/RL pass | 12 | data_construction_v3 已验证 Spider CEM/RL seed bank |

按 method family：

| family | 行数 |
|---|---:|
| Spider | 117 |
| OmniRetarget | 12 |

注意：`existing_cases.tsv` 的 12 行不是纯 E105-E108，包含 E081/E092/E094/E106/E107/E108 以及少量历史/空 experiment 标记；它的语义是“已验证 Spider CEM/RL pass bank”。

## 指标口径

E109 当前只把以下列作为首版公平 proxy：

| 指标 | 阈值 | 方向 | 说明 |
|---|---|:-:|---|
| `hand_near_frac_03cm/05cm/08cm` | 3cm / 5cm / 8cm | ↑ | 手-物体近接触比例，阈值必须随列名一起引用 |
| `leg_near_frac_03cm/05cm/08cm` | 3cm / 5cm / 8cm | ↓ | 腿部接近物体比例，作为 lower-body interference proxy |
| `hand_object_penetration_frac` | SDF < 0 | ↓ | 手部穿透比例 |
| `deep_penetration_frac_2cm` | SDF < -2cm | ↓ | 深穿透比例 |
| `leg_object_interference_frac` | leg SDF < 0 | ↓ | 腿部与物体干涉比例 |
| `pelvis_min_m` / `fall_flag` | pelvis < 0.45m | ↑ / ↓ | 站立/跌倒 proxy |

`obj_err_mean_m` / `obj_err_max_m` 暂不用于公平胜负，因为当前多数来自 method reference 或 Spider CEM summary。最终公平表需要接入 Core4D raw object pose 后重算 raw object SE(3) error。

## 关键结果

`metric_summary.json`：

| 项目 | 数值 |
|---|---:|
| 总行数 | 129 |
| fair/proxy ready 行数 | 51 |
| diagnostic-only 行数 | 76 |
| metadata-only 行数 | 2 |
| near-contact 阈值 | 0.03 / 0.05 / 0.08m |
| deep penetration 阈值 | -0.02m |
| fall pelvis 阈值 | 0.45m |

`threshold_summary.tsv`：

| threshold | hand near mean | leg near mean | hand N | leg N |
|---|---:|---:|---:|---:|
| 3cm | 0.595072 | 0.366835 | 38 | 38 |
| 5cm | 0.619829 | 0.455717 | 38 | 38 |
| 8cm | 0.615924 | 0.585333 | 50 | 38 |

按 method 聚合的 Spider CEM proxy：

| method | N | ready | fall | pelvis min | hand 3cm | hand 5cm | hand 8cm | leg interference | deep pen 2cm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `spider_cem_e105` | 6 | 6 | 0 | 0.679394 | 0.726964 | 0.735772 | 0.747290 | 0.261518 | 0.174119 |
| `spider_cem_e106` | 28 | 28 | 3 | 0.645055 | 0.541505 | 0.571176 | 0.592302 | 0.152778 | 0.027059 |
| `spider_cem_e107` | 4 | 4 | 0 | 0.636136 | 0.772206 | 0.786482 | 0.813311 | 0.092298 | 0.000000 |
| `spider_cem_e108` | 3 | 3 | 0 | 0.655881 |  |  | 0.486859 | 0.106115 |  |
| `spider_cem_registry` | 12 | 10 | 0 | 0.673600 |  |  | 0.557129 | 0.006202 |  |

E026 历史行现在统一为 `fair_metric_scope=diagnostic_only`、`fair_metric_ready=False`。这避免了把 `omniretarget_kinematic` 的 self-ref / method-ref object/body tracking 误读成公平比较指标。

## Caveat

1. 当前是 geometry/proxy 评测，不是最终 raw-GT paper 表。
2. E026 的 OmniRetarget kinematic object/body tracking 是 self-ref 或 method-ref 口径，不能作为 Spider vs OmniRetarget 的胜负依据。
3. 当前 Spider CEM summary 中的物体 tracking 仍是 method reference 口径，不能替代 raw object pose 评测。
4. near-contact 是阈值指标，必须明确 `3cm/5cm/8cm`，不能混用不同阈值的数值。
5. E108 旧 summary 中残留 `/tmp` artifact path，`build_case_bank.py` 已对 `/tmp` 记录做持久路径归一，优先指向 `workspace/core4d/results/E108/s6_downstream/cem/full/`。
6. OmniRetarget kinematic 当前还缺同一套 geometry/timeseries proxy，因此 E109 不能宣称 Spider 相对 OmniRetarget 的最终增益，只能说明现有历史指标为什么不公平，并给出后续统一评测入口。

## 旁路审查

使用 medium subagent 做只读审查，结论如下：

- case bank 覆盖了 E026、E105-E108、`existing_cases.tsv` CEM/RL pass。
- 3cm/5cm/8cm 阈值输出明确，少于 3 个阈值时脚本会报错。
- 必须在正式 log 中强调 E026 self-ref/method-ref caveat、OmniRetarget 缺 geometry proxy、当前不是最终 raw-GT 表。

审查后修正：

- E026 历史行不再计入 `fair_metric_ready`；
- E108 `/tmp` artifact path 自动归一到 `results/E108` 持久目录；
- 补充 `object_summary.tsv`，按物体聚合 proxy 指标。

## 验证

已运行：

```bash
python3 -m py_compile workspace/core4d/scripts/eval_omni_vs_spider/*.py
bash -n workspace/core4d/scripts/eval_omni_vs_spider/run_E109_fair_eval.sh
bash workspace/core4d/scripts/eval_omni_vs_spider/run_E109_fair_eval.sh
```

运行结果：

```text
[build_case_bank] wrote 129 rows
[compute_proxy_metrics] wrote 129 rows
[summarize_fair_eval] wrote summary.md
```

## 下一步

1. 接入 Core4D raw object pose，重算 raw object SE(3) error。
2. 接入 S1 raw contact mask，输出 hand/body/leg contact precision、recall、F1。
3. 对 OmniRetarget kinematic 输出补同一套 geometry/timeseries proxy，形成真正 method-to-method 的同口径表。
4. 在 paper/report 里只使用 raw-GT 或 geometry/physics 同口径指标，历史 method-ref 表只放入 appendix 或诊断说明。

## 最新口径修正：排除 adaptive route

用户复核后确认，PPT 和主表不应混入 `target_variant_id=adaptive`。已修正 `unified_replay_eval.py` 和旧配对表脚本，默认 case 集合变为：

```text
cem_status=pass
target_variant_id != adaptive
```

因此统一 replay 主表从 12 条变为 11 条，全部为 `ref_fk`：

| 来源 run | case 数 | 说明 |
|---|---:|---|
| E081 | 1 | box023 legacy ref_fk |
| E092 | 1 | box004 ref_fk |
| E096b | 2 | box004 ref_fk |
| E106 | 4 | box026 ref_fk clean batch |
| E107 | 1 | box021 ref_fk clean |
| E108 | 2 | bucket004 nonbox ref_fk |

E105 已被考虑进 `existing_cases.tsv`，但当前没有 CEM pass：2 条 `adaptive` 和 2 条 `fingertip_aware` 均为 `cem_status=fail`，所以不会进入主评测。

很多 case_id 仍以 `e091_` 开头，是因为 `case_id` 继承了早期 data construction / raw case 命名，不代表这些结果来自 E091。真实实验来源应看 `cem_run_id` 和 `cem_result_npz`：例如 `e091_box026_*` 的 4 条 pass 实际来自 E106，`e091_box004_*` 的 ref_fk pass 来自 E092/E096b。

重跑后验证：

```text
num_cases=11
num_metric_rows=22
history_validation=81/81 PASS
target_variant_id={ref_fk}
```

## 新增 24-case work 扩展表

按用户要求，旧 11-case strict 表保持不动，另出一个扩展表：

```bash
python3 workspace/core4d/scripts/eval_omni_vs_spider/build_expanded_24_work_eval.py
```

输出目录：

```text
workspace/core4d/results/E109/expanded_24_work_cases/
  expanded_24_omni_vs_spider_work_eval.xlsx
  expanded_24_omni_vs_spider_work_eval.md
  expanded_24_case_comparison.tsv
  expanded_24_method_metrics.tsv
  expanded_24_history_validation.tsv
```

case 口径：

| 分层 | case 数 | 说明 |
|---|---:|---|
| `A_strict_main` | 11 | 原 `ref_fk strict` 主表 case |
| `B_upper_WORK_non_strict` | 13 | 去重后额外 `ref_fk upper/object WORK`，但 lower-body strict 未过 |

验证：

```text
num_cases=24
num_strict_main_cases=11
num_upper_work_non_strict_cases=13
history_validation=198/198 PASS
```

注意：这 13 条补充 case 用于汇报“Spider CEM 能找到更多 upper/object WORK，但数据质量仍受腿部干涉限制”，不应直接当作 RL-ready positive。
