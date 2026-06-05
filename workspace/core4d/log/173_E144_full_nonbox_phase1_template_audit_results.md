# E144 全量 nonbox Phase 1 raw-contact/template audit 结果

## 目标

按 `workspace/core4d/plan/153_E144_full_nonbox_raw_mask_ref_fk_cem_plan.md` 推进 E144。用户明确要求：

- 全量跑 nonbox，不复用 E108 已 handoff 的 3 条作为替代；
- 默认 CEM 方法后续使用 `raw_mask_ref_fk`；
- 先看 nonbox 物体 template 如何创建，尤其是不规则 desk/chair；
- 分两阶段：Phase 1 到 CEM-ready，Phase 2 才用本地 1 卡 + 远程 2 卡跑 full CEM；
- 本轮不启动 RL。

## 修正

前一版 E144 builder 错误地从 `E108/s5_handoff/handoff_manifest.tsv` 取 CEM-ready rows，因此只得到 3 条 `bucket004`。该路径已废弃：

- `build_full_nonbox_raw_mask_ref_fk_cem_ready.py` 默认 source 改为 `workspace/core4d/results/E144/E144_full_nonbox_raw_contact`；
- 当前 `workspace/core4d/scripts/E144/variants.tsv` 由 E144 自己的 source run 生成；
- 错误的 E108-wrapper CEM 产物已移动到 `workspace/core4d/results/E144/archive_mistaken_e108_wrapper/`；
- 远程 E144 full tmux 已停止，本轮未继续跑错误 full CEM。

## Phase 1 命令

```bash
workspace/core4d/scripts/data_construction_v3/orchestration/run_pipeline.py \
  --mode full-from-raw \
  --run-id E144_full_nonbox_raw_contact \
  --run-root workspace/core4d/results/E144 \
  --core4d-raw-root /mnt/a0ccc676-9496-49f8-a861-f8a1797dec52/mocap_data/CORE4D/CORE4D_Real \
  --smplx-model-dir /mnt/a0ccc676-9496-49f8-a861-f8a1797dec52/mocap_data/smplx \
  --queue review-medium-nonbox \
  --stage2b-contact-label 5cm \
  --sample-count 500 \
  --apply-build-templates
```

随后运行：

```bash
python3 workspace/core4d/scripts/E144/build_full_nonbox_raw_mask_ref_fk_cem_ready.py
```

## 结果路径

| item | path |
|---|---|
| E144 source run | `workspace/core4d/results/E144/E144_full_nonbox_raw_contact/` |
| accounting | `workspace/core4d/results/E144/nonbox_inventory_accounting.tsv` |
| template creation audit | `workspace/core4d/results/E144/template_creation_audit.tsv` |
| raw-contact 5cm candidates | `workspace/core4d/results/E144/E144_full_nonbox_raw_contact/s1_raw_contact/raw_contact/raw_contact_candidates_5cm.tsv` |
| template backlog/review queue | `workspace/core4d/results/E144/E144_full_nonbox_raw_contact/s2_templates/template_backlog.tsv` |
| template visual review | `workspace/core4d/results/E144/E144_full_nonbox_raw_contact/s2_templates/template_visual_review/template_visual_manifest.tsv` |
| CEM-ready manifest | `workspace/core4d/results/E144/cem_ready/raw_mask_ref_fk_cem_ready.tsv` |
| variants | `workspace/core4d/scripts/E144/variants.tsv` |

## 关键结果

Full nonbox inventory accounting 覆盖 `1456` 个 nonbox case-person：

| category | rows |
|---|---:|
| desk | 408 |
| bucket | 394 |
| chair | 292 |
| board | 232 |
| stick | 114 |
| unknown | 16 |

`review-medium-nonbox` flow 候选为 `132` rows：

| category | rows |
|---|---:|
| desk | 56 |
| bucket | 46 |
| chair | 30 |

5cm raw-contact 结果：

| decision | rows |
|---|---:|
| raw_contact_pass | 82 |
| raw_contact_reject_motion | 32 |
| raw_contact_fail | 16 |
| raw_contact_review | 2 |

5cm pass 的类别分布：

| category | rows |
|---|---:|
| bucket | 33 |
| desk | 33 |
| chair | 16 |

Template 审查结果：

- required source templates: `21`
- `template_status=manual_review_required`: `21/21`
- rendered template visual rows: `11 pass`, `10 not_rendered`
- Stage2b manifest rows: `82`
- Stage2b decision: `82/82 stage2b_template_manual_review_required`
- S5 handoff rows: `0`
- CEM-ready variants: `0`

这说明全量 nonbox raw-contact 已完成，但没有任何 nonbox template 被自动 release。当前无法进入 Phase 2 full CEM，必须先对 `template_backlog.tsv` 中的 21 个 source template 做人工/high-review，并写入 `nonbox_template_review.tsv`。

## Template 创建观察

- bucket 不总是规则 proxy：部分已有 scene 被标为 `geometry_review` / `manual_complex_shape`，说明现有 collision 与 mesh AABB/期望 proxy 差异较大，不能自动放行。
- 新建 bucket proxy rows 使用 `nonbox_proxy_aabb_review` + `bucket_wall_proxy_aabb`，但仍为 `manual_review_required`。
- desk/chair 均属于复杂形状，当前策略是 `manual_complex_shape`，不自动构建规则 proxy。
- board/stick 在全量 inventory 中存在，但本轮 `target_medium + main_move_obs0 + motion_pass` flow 候选没有进入 raw-contact pass 队列；保留在 accounting 中。

## 验证

```bash
find workspace/core4d/scripts/data_construction_v3 workspace/core4d/scripts/E144 workspace/core4d/scripts/eval -name '*.py' -print0 | xargs -0 python3 -m py_compile
bash -n workspace/core4d/scripts/train/train_E144_raw_mask_ref_fk_full_cem.sh
bash -n workspace/core4d/scripts/run_E144_remote.sh
bash -n workspace/core4d/scripts/pull_E144_remote_results.sh
bash -n workspace/core4d/scripts/eval/eval_E144_raw_mask_ref_fk_full_cem.sh
bash workspace/core4d/scripts/train/train_E144_raw_mask_ref_fk_full_cem.sh list full 0 local
bash workspace/core4d/scripts/eval/eval_E144_raw_mask_ref_fk_full_cem.sh full --allow-missing
git diff --check
workspace/core4d/scripts/data_construction_v3/orchestration/run_release_checks.sh --run-root workspace/core4d/results/E144_release_check --no-smoke
```

结果：

- `py_compile`: pass
- shell syntax: pass
- empty CEM-ready manifest eval: pass, `manifest_rows=0`
- `git diff --check`: pass
- release check: pass, release audit `63/63`, smoke skipped by `--no-smoke`

## 下一步

1. 审查 `s2_templates/template_backlog.tsv` 和 `template_visual_review/template_visual_manifest.tsv` 中的 21 个 source templates。
2. 写 `workspace/core4d/results/E144/E144_full_nonbox_raw_contact/s2_templates/nonbox_template_review.tsv`，只对确实通过的 template 标 `approve_clean`。
3. 用该 review TSV resume/run Stage2b execute，生成 E144 自己的 target gate、visual QC 和 S5 handoff。
4. 只有 `raw_mask_ref_fk_cem_ready.tsv` 出现 rows 后，才启动 Phase 2 本地 1 卡 + 远程 2 卡 full CEM。
