# E111 Plan: data_construction_v3 contact evidence chain

日期：2026-06-02

关联总控计划：`workspace/core4d/plan/contact_improvement_plan.md`

前置实验：E110 `workspace/core4d/log/140_E110_contact_metric_audit_results.md`

## 0. Context

E110 证明 Spider CEM 在 E109 24-case work set 上主要把 OmniRetarget 的深穿透式接触变成浅穿透/近场间隙，但没有恢复稳定无穿透物理接触。E110 仍只消费 aggregate replay 指标，不能回答 raw-contact active frame 上的 precision/recall、per-hand run-length 和 contact target gap。

E111 的目标是补齐 `data_construction_v3` 的接触证据链，让后续 E112 contact-aware CEM ablation 有可靠输入和固定评测口径。

本轮不跑 full CEM，不使用 GPU/远程机器。

## 1. Claims

| Claim | 验证方式 |
|---|---|
| C1: S1 raw contact artifact 可作为训练/评测证据，而不只是 gate | raw-contact manifest 记录 `contact_mask_npz`、raw/trimmed frame count、active/run-length fractions |
| C2: S3/S5/S6 能追踪 contact artifact 来源 | Stage2b/handoff/downstream 相关 manifest 保留 `contact_mask_*` 与 contact target 字段，不用 `ref_fk` 冒充 contact-aware route |
| C3: S6 contact alignment evaluator 可以在缺 MuJoCo replay 时先做 raw-mask alignment baseline | 新增固定入口输出 PR/Recall/F1/IoU、per-hand run metrics 和 failure label |
| C4: QA 可发现 contact artifact 缺失或 manifest 指向坏路径 | `verify_reproducibility.py` 或 smoke fixture 检查 contact mask path 存在性 |
| C5: E111 产物能直接服务 E112 case/variant 选择 | 结果 summary 给出可用 case 数、mask coverage 和缺失原因 |

## 2. Implementation Scope

### 2.1 S1 raw contact artifact schema

检查并补齐：

```text
workspace/core4d/scripts/data_construction_v3/stages/s1_raw_contact/run_raw_contact.py
```

目标字段：

```text
contact_mask_npz
contact_label
contact_person_idx
raw_frame_count
trimmed_frame_count
raw_to_trimmed_mapping_status
left_active_frac
right_active_frac
both_active_frac
left_longest_run_frac
right_longest_run_frac
both_longest_run_frac
contact_target_status
```

若现阶段缺 trimmed mapping，只允许明确写 `missing_trimmed_mapping`，不能伪造 trimmed-frame mask。

### 2.2 S3/S5 manifest propagation

检查并补齐：

```text
stages/s3_retarget/run_stage2b.py
stages/s5_handoff/export_cem_overrides.py
stages/s5_handoff/export_handoff.py
state/update_case_state_registry.py
```

目标字段：

```text
contact_mask_npz
contact_mask_status
contact_mask_label
contact_mask_person_idx
contact_mask_time_axis
raw_contact_artifact_npz
contact_target_npz
contact_target_source
contact_target_frame
contact_target_time_axis
contact_route_diagnostic_ref
```

### 2.3 S6 contact alignment evaluator

新增：

```text
workspace/core4d/scripts/data_construction_v3/stages/s6_downstream/evaluate_contact_alignment.py
workspace/core4d/scripts/eval/eval_E111_contact_alignment_smoke.sh
```

第一版输入 `handoff_manifest.tsv` 或 `stage2b_manifest.tsv` + contact mask NPZ。若没有 method qpos/replay SDF，先输出 raw mask coverage 和 baseline alignment columns，并把 replay-dependent fields 标为 missing；不伪造物理接触。

输出：

```text
workspace/core4d/results/E111/contact_alignment_smoke/
  contact_alignment_metrics.tsv
  contact_alignment_summary.json
  contact_alignment_summary.md
```

### 2.4 QA/reproducibility

检查：

- contact mask path 存在且非空；
- NPZ 内至少有 `raw_contact_mask_3cm` / `raw_contact_mask_5cm` 之一；
- manifest 中 `contact_mask_npz` 指向坏路径时应 fail 或 warning；
- smoke fixture 至少覆盖 active frame 和 inactive frame。

## 3. Eval Command

固定本地评估脚本：

```bash
bash workspace/core4d/scripts/eval/eval_E111_contact_alignment_smoke.sh
bash workspace/core4d/scripts/eval/eval_E111_contact_alignment_chain_smoke.sh
```

该脚本只运行 CPU/Python 评测，不占 GPU，不触发远程多卡执行。

## 4. Success Criteria

| 标准 | 阈值 |
|---|---|
| 计划先行 | 本文件存在并记录 claims/scope/eval |
| S1 schema | raw-contact 输出 manifest 包含目标 contact fields 或显式 missing reason |
| S6 evaluator | 固定 eval 脚本 exit 0，输出 TSV/JSON/MD |
| 不伪造 replay 指标 | 缺 qpos/SDF 时 replay-dependent fields 为空，并记录 coverage/status |
| QA | `py_compile`、固定 eval、`git diff --check` 通过 |
| 实验记录 | 新增 E111 log，更新 tracker/progress |

## 5. 本轮不做

- 不跑 full CEM；E112 才需要本地 1 卡 + 远程 2 卡。
- 不改 CEM reward。
- 不把 raw-contact mask alignment 解释为物理接触成功。
- 不把 `/tmp` smoke 作为最终结果；正式 E111 输出放在 `workspace/core4d/results/E111/`。
