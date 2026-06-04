# E113 Contact-Aware Expanded Workset Plan

日期：2026-06-03

总控计划：`workspace/core4d/plan/contact_improvement_plan.md`

前置结果：

- E110：`workspace/core4d/log/140_E110_contact_metric_audit_results.md`
- E111：`workspace/core4d/log/141_E111_contact_evidence_chain_results.md`
- E112：`workspace/core4d/log/142_E112_contact_aware_cem_ablation_results.md`
- E109 24-case work set：`workspace/core4d/results/E109/expanded_24_work_cases/`

## 1. Context

E112 Phase A 已证明 contact-aware CEM 不是单纯用穿透换接触率：

- box004：baseline contact `10.1%`，raw/hold 提到 `54.1/55.0%`，均 WORK；
- box021：baseline contact `8.5%`，raw/hold 提到 `77.5/77.5%`，均 WORK；
- box026：contact 有提升但仍 FAIL，raw_mask 伴随 pelvis min 降到 `0.585m`。

因此 E113 进入扩量阶段，但扩量必须区分：

1. **release-candidate expansion**：优先验证 E112 成功族群，即 box004/box021/bucket004 等已有 clean task + raw contact mask 的 case；
2. **box026 diagnostic holdout**：box026 不直接作为 contact-aware release candidate，应单独诊断 raw mask 时窗、surface target、approach corridor 和 posture schedule。

## 2. Claims

| Claim | 验证方式 |
|---|---|
| C1: `hold_band` 可在 E109 work-set 的非 box026 可执行子集上稳定提升 contact | 对 selected runnable cases 跑 baseline/hold 或 hold-only+baseline-cache，对比 `contact_frac_either`、physics contact、deep penetration |
| C2: `hold_band` 不显著增加 deep penetration 或 lower-body/object interference | full eval 中 `hand_geom_deep_penetration_2cm` 增幅 <= 3pp，leg interference 不超过 strict gate |
| C3: E113 manifest 能明确区分 runnable / missing_contact_mask / missing_task / box026_holdout | preflight TSV 和 summary JSON/MD 可复现 |
| C4: E113 输出可形成 per-case Pareto decision 和 E114 RL handoff 候选 | full eval summary 写 release candidate、upper-work-only、contact-good-lowerbody-fail、holdout list |

## 3. Case Scope

### 3.1 E109 24-case 结构

| object | count | E113 默认策略 |
|---|---:|---|
| box004 | 3 | 主扩展；E112 已通过，mask 可由 E096b/E104 raw proxy 提供 |
| box021 | 3 | 主扩展；E112 已通过，但 non-strict rows 要严查 leg interference |
| bucket004 | 2 | 次扩展；E108 有 nonbox CEM 与 mask，需 bucket-aware visual QC |
| box023 | 1 | legacy strict case；可做 hold-band smoke，但不直接进入 RL candidate |
| box026 | 15 | diagnostic holdout；不进入 release-candidate expansion |

### 3.2 Phase A runnable subset

先跑非 box026 的 6 个核心 case，控制 GPU 时间并覆盖严格/边界：

| case | object | reason |
|---|---|---|
| `e091_box004_20231003_2_082_p1` | box004 | E112 pass reference；用于 regression anchor |
| `e091_box004_20231003_2_083_p1` | box004 | E096b positive；检验另一个 person1 trajectory |
| `e091_box004_20231003_2_083_p2` | box004 | E092/E094 known work；检验 person2/raw-proxy conversion |
| `d003_box021_20231011_035_p1` | box021 | E112 pass reference；strict main |
| `d003_box021_20231011_035_p2` | box021 | upper WORK non-strict；lower-body risk |
| `d003_box021_20231018_029_p2` | box021 | upper WORK non-strict；lower-body risk |

默认变体：

- `hold_band`：E112 选出的默认 contact-aware 配置；
- `baseline_ref_fk`：仅当已有 baseline cache 不能直接进入 E113 evaluator 时重跑；否则读取 E109/E096b/E107/E092 baseline NPZ 作为 baseline。

`raw_mask_ref_fk` 只保留为 targeted diagnostic，不作为 Phase A 默认扩展，原因是 E112 中 `hold_band` 更稳健且 box021 raw_mask leg interference 更高。

### 3.3 Phase B expansion

Phase A 达标后扩展：

- bucket004 strict cases：`bucket004_20231002_022_p1`、`bucket004_20231003_1_012_p1`，但必须进行 bucket-aware visual review；
- box023 legacy strict：只做 comparison，不进 RL handoff；
- E109 filtered-20 中 box026 rows：进入单独 E113-box026-diagnostic，不混入 release-candidate summary。

## 4. Implementation

新增本地脚本：

| 脚本 | 作用 |
|---|---|
| `workspace/core4d/scripts/E113/build_expanded_contact_manifest.py` | 从 E109 24-case 表和本地资产生成 E113 variants/preflight/overrides |
| `workspace/core4d/scripts/train/train_E113_contact_aware_expanded.sh` | 本地/远端/full/smoke 统一 runner |
| `workspace/core4d/scripts/run_E113_remote.sh` | 远端 GPU0/GPU1 split runner |
| `workspace/core4d/scripts/pull_E113_remote_results.sh` | 回收远端结果和日志 |
| `workspace/core4d/scripts/eval/eval_E113_contact_aware_expanded.sh` | 固定评估入口 |
| `workspace/core4d/scripts/eval/eval_E113_contact_aware_expanded.py` | 聚合 E112 evaluator + baseline cache，输出 per-case Pareto decision |

实现原则：

- 复用 E112 的 task validation、mask validation、keyframe extraction 和 split runner pattern；
- baseline override 必须显式 `contact_hdmi_gain=0.0`；
- hold override 必须显式 `contact_hdmi_gain=5.0`、`hold_contact_rew_scale=1.0`；
- 所有 generated overrides 写入 `examples/config/override/core4d_E113_*.yaml`；
- `variants.tsv` 必须记录 `baseline_npz_path`、`baseline_source`、`mask_path`、`mask_kind`、`phase_scope`、`holdout_reason`；
- preflight 必须拒绝 task/mask/override 缺失，但 box026 holdout 可写 `preflight_status=holdout` 而不是 fail。

## 5. Full CEM Execution

当 Phase A preflight + smoke 通过后，按用户要求：

- 本地 GPU0 跑 split `local-gpu0`；
- 远端 `spider-remote` GPU0/GPU1 跑 split `remote-gpu0` / `remote-gpu1`；
- 不 kill 其他 GPU 进程；如剩余显存足够，直接叠加运行；
- 因 E113 脚本当前未提交，远端采用 `rsync` 同步脚本、override、preflight、mask、task asset 后运行；
- 完成后必须 `pull_E113_remote_results.sh full` 回收结果再本地评估。

建议 Phase A split：

| split | cases |
|---|---|
| local-gpu0 | box004 3 cases |
| remote-gpu0 | box021 035_p1 / 035_p2 |
| remote-gpu1 | box021 029_p2 |

## 6. Success Criteria

Phase A 成功：

- 6/6 `hold_band` full CEM 成功产出 NPZ/MP4；
- 与 baseline cache 比，mean contact 提升 >= 8pp；
- per-case deep penetration 不增加超过 3pp；
- pelvis/fall 不退化；
- lower-body strict fail 的 case 必须被标为 `contact_good_lowerbody_fail`，不能进入 RL-ready；
- 至少 3 条 case 达到 `phase_b_release_candidate=true` 或明确说明失败模式。

进入 Phase B 条件：

- box004 3/3 不退化；
- box021 strict case 仍 WORK；
- box021 non-strict case 若接触提升但 lower-body 失败，则保留 diagnostic，不进入 E114。

## 7. Outputs

```text
workspace/core4d/results/E113/
  preflight/
    phaseA_preflight.tsv
    phaseA_manifest_summary.{md,json}
  cem/smoke/
  cem/full/
    full_eval_summary.{md,csv,json}
    pareto_decisions.tsv
    release_candidates.tsv
    contact_good_lowerbody_fail.tsv
```

结果日志：

```text
workspace/core4d/log/143_E113_contact_aware_expanded_workset_results.md
```

## 8. Stop Conditions

- 如果 Phase A smoke 中任一 generated hold override 没有实际加载 contact mask，停止 full；
- 如果 hold_band 在 box004 退化 deep penetration > 3pp，停止扩展；
- 如果 box021 non-strict contact 提升但 leg interference 大幅增加，不扩大到 RL handoff；
- 如果 box026 diagnostic 需要 surface target/approach corridor 代码改动，不在 E113 主扩展中临时硬调 contact gain。
