# E144 全量非 Box 到 Raw-Mask Ref-FK CEM-Ready / Full-CEM 计划

## Context

E108 已验证非 box bucket 候选挖掘、reviewable proxy template、review override、Stage2b/target gate/visual QC 与 S6 RL export join。E143 固定了 `raw_mask_ref_fk` CEM 方法，但只覆盖 24 个 box/bucket workset case。

本轮扩展到全量 nonbox inventory。当前 E108 inventory 中 nonbox 约 1456 个 case-person，类别包括 bucket、board、stick、desk、chair 和 unknown。所有 nonbox row 必须进入 registry/accounting 终态；只有通过尺寸、动作、motion、raw-contact、template review、Stage2b、target gate 和 visual QC 的子集进入 CEM-ready。

默认 CEM 方法改为 E143 风格 `raw_mask_ref_fk`：

- `contact_hdmi_target_source=ref_fk`
- `contact_hdmi_target_uses_eef_offset=true`
- `contact_hdmi_gain=5.0`
- `contact_hdmi_mask_source=core4d_3cm`
- `contact_hdmi_mask_time_axis=auto`
- 不使用 hold-band reward

S1 可用 5cm 作为候选筛选档，但 CEM override 默认消费 3cm raw mask。

## Phase 1: CEM-Ready

目标：不跑 CEM，只生成可复查的 nonbox accounting、template 创建审查、CEM-ready manifest 和 raw-mask ref-FK override。

固定入口：

```bash
python workspace/core4d/scripts/E144/build_full_nonbox_raw_mask_ref_fk_cem_ready.py
```

产物：

- `workspace/core4d/results/E144/nonbox_inventory_accounting.tsv`
- `workspace/core4d/results/E144/template_creation_audit.tsv`
- `workspace/core4d/results/E144/cem_ready/raw_mask_ref_fk_cem_ready.tsv`
- `workspace/core4d/results/E144/cem_ready/raw_mask_ref_fk_cem_ready_preflight.tsv`
- `workspace/core4d/scripts/E144/variants.tsv`
- `examples/config/override/core4d_E144_*_raw_mask_ref_fk.yaml`

Template 创建策略：

| 类别 | 创建策略 | 默认状态 |
|---|---|---|
| bucket | `bucket_wall_proxy_aabb`，底面 + 四侧壁 collision | `manual_review_required` |
| board/stick | `mesh_aabb_box_proxy` | `manual_review_required` |
| desk/chair | `manual_complex_shape`，不自动构建规则 proxy | `manual_review_required` |
| unknown/missing mesh | 只 accounting，不进入 Stage2b | reject/backlog |

只有 `nonbox_template_review.tsv` 中 `review_decision=approve_clean` 的 source template 可写为 `clean_reviewed`，再进入 Stage2b。未经 review 的 proxy 不能进入 CEM-ready。

## Phase 2: Full CEM Batch

目标：只消费 Phase 1 的 `variants.tsv`，本地 1 卡 + 远程 2 卡批量跑 full CEM。

固定入口：

```bash
python workspace/core4d/scripts/E144/build_full_nonbox_raw_mask_ref_fk_cem_ready.py
bash workspace/core4d/scripts/train/train_E144_raw_mask_ref_fk_full_cem.sh list full local
bash workspace/core4d/scripts/run_E144_remote.sh full
bash workspace/core4d/scripts/train/train_E144_raw_mask_ref_fk_full_cem.sh local full 0
bash workspace/core4d/scripts/pull_E144_remote_results.sh full
bash workspace/core4d/scripts/eval/eval_E144_raw_mask_ref_fk_full_cem.sh full
```

Split 规则：

- local GPU0: `split=local-gpu0`
- remote GPU0: `split=remote-gpu0`
- remote GPU1: `split=remote-gpu1`

Full CEM 结果写入：

- `workspace/core4d/results/E144/cem/full/`
- `logs/E144/cem/full/`
- `workspace/core4d/results/E144/eval/full/`

评估输出包括 CEM metrics、missing outputs、S6 evidence input 和 RL export candidate table。`RL_EXPORT_READY` 只表示可作为 RL 数据输入，不启动 RL。

## Success Criteria

- 全量 nonbox inventory row 都有 accounting 决策。
- desk/chair 不会被自动 proxy release。
- 未 review template 不进入 Stage2b/CEM-ready。
- 每个 CEM-ready row 都具备 `scene_act`、`trajectory`、`contact_mask`、override YAML 和 split。
- Full CEM 完成后，每个 row 要么有 NPZ/MP4/outdir trajectory，要么有明确 missing/failure log。
- S6 evidence 中区分 `DOWNSTREAM_CEM_PASS`、`DOWNSTREAM_CEM_FAIL`、`DOWNSTREAM_NOT_RUN`。
- 不启动 PPO/Holosoma RL，不生成训练 checkpoint。

## Validation

```bash
find workspace/core4d/scripts/data_construction_v3 workspace/core4d/scripts/E144 workspace/core4d/scripts/eval -name '*.py' -print0 | xargs -0 python3 -m py_compile
bash -n workspace/core4d/scripts/train/train_E144_raw_mask_ref_fk_full_cem.sh
bash -n workspace/core4d/scripts/run_E144_remote.sh
bash -n workspace/core4d/scripts/pull_E144_remote_results.sh
bash -n workspace/core4d/scripts/eval/eval_E144_raw_mask_ref_fk_full_cem.sh
git diff --check
```
