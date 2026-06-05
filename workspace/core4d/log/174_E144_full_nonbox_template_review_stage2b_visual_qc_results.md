# E144 全量 nonbox template review / Stage2b / visual QC gate 结果

## 目标

继续执行 `workspace/core4d/plan/153_E144_full_nonbox_raw_mask_ref_fk_cem_plan.md`，但不再使用错误的 E108 handoff wrapper。默认 CEM 方法仍为 `raw_mask_ref_fk`，本轮只推进到 CEM-ready gate；不启动 RL。

## Template review

新增脚本：

- `workspace/core4d/scripts/E144/build_nonbox_template_review.py`

输出：

- `workspace/core4d/results/E144/E144_full_nonbox_raw_contact/s2_templates/nonbox_template_review.tsv`
- `workspace/core4d/results/E144/E144_full_nonbox_raw_contact/s2_templates/nonbox_template_review_checks.tsv`
- `workspace/core4d/results/E144/E144_full_nonbox_raw_contact/s2_templates/nonbox_template_review_summary.json`

Review 决策：

| decision | rows |
|---|---:|
| approve_clean | 4 |
| needs_manual_edit | 17 |

批准的 source templates：

- `bucket003_person2`
- `bucket004_person1`
- `bucket004_person2`
- `bucket009_person2`

批准条件为：bucket 类、visual render pass、MuJoCo load OK、inertial clean、robot 29.632kg 污染 absent，且当前 XML object body 具备底面 + 四壁 collision proxy（`object_collision`、`object_collision_bucket_xneg/xpos/yneg/ypos`）。

未批准的模板包括 5 个仍为单 `object_collision` 的 bucket，以及所有 desk/chair complex/manual 或 not-rendered 模板。render pass 没有被当作 clean approval。

## Stage2b / S4 / S5

使用 review TSV 合并 registry 后执行 Stage2b：

| stage | result |
|---|---|
| S3 rows | 82 |
| S3 `stage2b_ready` | 11 |
| S3 `stage2b_template_manual_review_required` | 71 |
| S3 execute pass | 11 |
| S4 target gate pass | 11 |
| S4 target gate not_run | 71 |
| S4 visual render pass | 11 |
| S4 visual render skipped | 71 |
| S4 visual QC review | 11 |
| S4 visual QC not_run | 71 |
| S5 `HANDOFF_REVIEW_VISUAL_QC` | 11 |
| S5 `HANDOFF_PENDING_TEMPLATE_OR_VARIANT` | 11 |
| CEM-ready variants | 0 |

11 条 target gate pass cases：

- `bucket003_20231018_001_p2`
- `bucket003_20231018_002_p2`
- `bucket003_20231020_064_p2`
- `bucket003_20231020_065_p2`
- `bucket004_20231002_021_p1`
- `bucket004_20231002_021_p2`
- `bucket004_20231002_022_p1`
- `bucket004_20231002_022_p2`
- `bucket004_20231003_1_012_p1`
- `bucket004_20231003_1_012_p2`
- `bucket009_20231002_056_p2`

Visual QC render package：

- `workspace/core4d/results/E144/E144_full_nonbox_raw_contact/s4_gate_visual_qc/omnirt_v1/ref_fk/visual_qc_render/`

## 为什么没有启动 full CEM

E144 的 release gate 要求 `visual_qc_status=pass` 才能写入 `raw_mask_ref_fk` CEM-ready manifest。当前 11 条机器 target gate pass case 只有 render package，尚无人工/视觉审核 TSV；因此 `make_visual_qc.py` 保守输出 `review`，S5 输出 `HANDOFF_REVIEW_VISUAL_QC`。

因此：

- `workspace/core4d/results/E144/cem_ready/raw_mask_ref_fk_cem_ready.tsv` 没有真实 data rows；
- `workspace/core4d/scripts/E144/variants.tsv` 只有注释和表头；
- `train_E144_raw_mask_ref_fk_full_cem.sh list full 0 local` 不会列出可跑 case；
- 本轮没有启动本地/远程 full CEM，也没有启动 RL。

## 验证

已通过：

```bash
find workspace/core4d/scripts/data_construction_v3 workspace/core4d/scripts/E144 workspace/core4d/scripts/eval -name '*.py' -print0 | xargs -0 python3 -m py_compile
bash -n workspace/core4d/scripts/train/train_E144_raw_mask_ref_fk_full_cem.sh
bash -n workspace/core4d/scripts/run_E144_remote.sh
bash -n workspace/core4d/scripts/pull_E144_remote_results.sh
bash -n workspace/core4d/scripts/eval/eval_E144_raw_mask_ref_fk_full_cem.sh
git diff --check
```

## 下一步

1. 对 `visual_qc_render/cases/*/*_target_replay_sheet.png` 和 MP4 做人工 visual QC，写入 review TSV，将确实通过的 11 条标为 `visual_qc_status=pass`。
2. 对剩余 17 个 source templates 做几何编辑/审核；尤其是 desk/chair 不能用规则 box proxy 自动放行。
3. 重新运行 `make_visual_qc.py --review-tsv ...`、registry update、S5 handoff、E144 builder。只有 `variant_rows > 0` 后再启动本地 1 卡 + 远程 2 卡 full CEM。
