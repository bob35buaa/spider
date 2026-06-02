# E108 Phase 1-3 非 box 候选挖掘与 template review 结果

## 目标

按 `workspace/core4d/plan/117_E108_nonbox_to_rl_data_pipeline_plan.md` 启动 E108，先验证三件事：

1. v3 可以自动从原始 CORE4D 中筛出非 box 候选；
2. 非 box bucket-specific proxy template 可以生成、MuJoCo load、可视化；
3. 未 review 的 proxy template 不会自动进入 Stage2b，只有显式 review override 后才允许进入。

## 代码改动

| 模块 | 改动 |
|---|---|
| `run_raw_contact.py` | 新增 `selected-medium-nonbox` / `review-medium-nonbox` queue；summary 增加 `nonbox_category_counts` 和 `nonbox_candidate_counts` |
| `build_or_audit_templates.py` | 新增非 box proxy builder；bucket 使用 `bucket_wall_proxy_aabb`，board/stick 使用 AABB proxy；proxy 默认仍为 review |
| `update_case_state_registry.py` | 新增 `--from-template-review-tsv`；`review_decision=approve_clean` 写入 `template_status=clean_reviewed` |
| `run_stage2b.py` / `export_handoff.py` | 允许 `clean_reviewed` 进入 Stage2b/handoff；`run_stage2b.py` 支持 `--template-review-tsv` |
| 文档/计划 | 更新 S2/S3 schema、template policy、E108 计划，明确 bucket-specific collision 和 review override |

## 候选挖掘 smoke

命令：

```bash
workspace/core4d/scripts/data_construction_v3/orchestration/run_pipeline.py \
  --mode full-from-raw \
  --run-id E108_nonbox_candidate_mining_smoke \
  --run-root workspace/core4d/results/E108 \
  --core4d-raw-root /mnt/a0ccc676-9496-49f8-a861-f8a1797dec52/mocap_data/CORE4D/CORE4D_Real \
  --smplx-model-dir /mnt/a0ccc676-9496-49f8-a861-f8a1797dec52/mocap_data/smplx \
  --queue selected-medium-nonbox \
  --stage2b-contact-label 5cm \
  --max-case-persons 40 \
  --sample-count 500
```

结果：

| 阈值 | rows | pass | review | fail | 物体分布 |
|---|---:|---:|---:|---:|---|
| 3cm | 40 | 29 | 3 | 8 | bucket007=16, bucket004=8, bucket010=6, bucket003=6, bucket009=4 |
| 5cm | 40 | 34 | 1 | 5 | bucket007=16, bucket004=8, bucket010=6, bucket003=6, bucket009=4 |

5cm top rows 中多条 `target_left/right/both/partner_any = 1.0`，说明非 box bucket 候选数量充足，满足 Phase 1 “至少 20 个 5cm pass/review” 的成功标准。

输出：

```text
workspace/core4d/results/E108/s1_raw_contact/
workspace/core4d/results/E108/registries/case_state_registry_all_nonbox_candidates.tsv
```

整理前的完整 smoke run 已归档到 `workspace/core4d/results/E108/archive_legacy/E108_nonbox_candidate_mining_smoke/`。

## 未 review 不放行验证

同一 run 的 S2/S3 结果：

| 阶段 | 结果 |
|---|---|
| S2 required templates | 9 |
| S2 `template_status` | 9/9 `manual_review_required` |
| S3 rows | 34 |
| S3 decision | 34/34 `stage2b_template_manual_review_required` |
| S3 `pipeline_ready` | 0 |
| S5 handoff | 0 |

结论：非 box proxy/template 未经审查不会自动进入 Stage2b。

## bucket-specific proxy template smoke

使用临时 scene/asset root 生成 9 个 bucket proxy：

```text
workspace/core4d/results/E108/archive_legacy/template_proxy_apply_smoke/
```

结果：

| 检查 | 结果 |
|---|---|
| build status | 9/9 `review_required` |
| template status | 9/9 `manual_review_required` |
| collision policy | 9/9 `bucket_wall_proxy_aabb` |
| proxy template | 9/9 `True` |
| MuJoCo load | 9/9 `True` |
| template visual render | 9/9 `pass` |

high subagent 只读审查结论：`PASS`。审查确认 `template_backlog.tsv`、`template_build.tsv`、`template_audit.tsv` 和 `template_visual_manifest.tsv` 任务集合一致，视频/sheet 文件均存在，且没有自动 clean/release。

## review override smoke

构造一个 review TSV，仅 approve `bucket004_person1`：

```text
workspace/core4d/results/E108/archive_legacy/nonbox_template_review_smoke.tsv
```

然后 dry-run S3：

```text
workspace/core4d/results/E108/archive_legacy/stage2b_review_smoke/
```

结果：

| S3 decision | count |
|---|---:|
| `stage2b_ready` | 4 |
| `stage2b_template_manual_review_required` | 30 |

registry merge 后，只有 4 个 `bucket004/person1` raw-contact pass rows 变为 `STAGE2B_READY`：

```text
bucket004_20231002_021_p1
bucket004_20231002_022_p1
bucket004_20231003_1_012_p1
bucket004_20231003_1_013_p1
```

其它未 review 的 bucket source templates 仍被 `REJECT_TEMPLATE_AUDIT` 拦住。

## 验证

```bash
find workspace/core4d/scripts/data_construction_v3 -name '*.py' -print0 | xargs -0 python3 -m py_compile
git diff --check
workspace/core4d/scripts/data_construction_v3/orchestration/run_release_checks.sh \
  --run-root /tmp/core4d_dcv3_E108_release_check_phase1_3 \
  --no-smoke
```

结果：全部通过；release audit `62/62`。

## 当前结论

E108 Phase 1-3 已通过：

- 非 box bucket 候选足够；
- bucket-specific proxy template 能生成、load、render；
- proxy 默认不会污染下游；
- review override 能精确放行被 approve 的 source template。

下一步进入 Phase 4：选择 2-4 个低风险 bucket case，把对应 proxy template 落到真实 source scene root，经过 high subagent review 后执行 Stage2b 和 target gate。若多个 case 进入 full CEM，则按用户要求使用本地 1 卡 + 远程 2 卡并行。
