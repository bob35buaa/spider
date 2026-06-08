# E145 全量非 Box 模板 Release 到 RL-Ready 计划

## 背景

E144 已经完成全量 non-box accounting，并把复杂非 box 的模板构建规范写入正式 v3 数据管线：

- bucket: `bucket_wall_proxy_aabb`
- desk: `desk_surface_voxel_multibox_proxy_draft`
- chair: `chair_surface_voxel_multibox_proxy_draft`

desk/chair 的 mesh/collision overlay 已完成多轮人工审查：语义 desk/chair proxy 被废弃，当前采用 tight surface voxel multi-box proxy；该方案已经写入 `build_or_audit_templates.py`、S2 通用 overlay renderer、pipeline docs 和 skill。

当前 E144 5cm raw-contact pass 分布：

| 类别 | rows |
|---|---:|
| bucket | 33 |
| desk | 33 |
| chair | 16 |
| board | 0 |
| stick | 0 |

因此 E145 只继续 bucket/desk/chair。board/stick 虽然存在于 inventory，但没有进入 E144 5cm raw-contact candidates/pass，也没有进入 S2 backlog，本轮不处理。

当前状态：

- 21 个 required source templates。
- 4 个 bucket template 已经 `approve_clean`。
- 17 个 draft template 仍是 `not_released`：5 bucket、8 desk、4 chair。
- E144 之前只跑了 11 条 bucket CEM，11/11 CEM fail，`RL_EXPORT_READY=0`。
- 没有启动过 RL。

## 总目标

把已经审查过的 bucket/desk/chair source templates 通过显式 review gate release，然后沿 v3 数据管线推进到 RL-ready gate：

```text
S2 template review -> S3 Stage2b -> S4 target gate + visual QC -> S5 handoff -> full CEM -> S6 evidence -> rl_export_input.tsv
```

硬边界：

- 不启动 PPO/Holosoma RL。
- 不生成 `.pt/.pth/.ckpt` checkpoint。
- 不把 `RL_EXPORT_READY` 说成 RL 成功。

## 两阶段安排

E145 分成两个阶段执行。

### Phase 1: 做到 CEM-Ready + 可视化

目标：完成模板 release、Stage2b、target gate、visual QC、S5 handoff 和 CEM-ready manifest；不跑 full CEM。

Phase 1 的终点是：

- `variants.tsv` / CEM-ready TSV 已生成；
- 每个 CEM-ready row 都有完整 preflight；
- 目标轨迹和 visual QC 已人工审查；
- 可以安全进入 Phase 2。

### Phase 2: 3 卡并行跑 Full CEM

目标：只消费 Phase 1 产出的 CEM-ready variants，用本地 1 卡 + 远程 2 卡并行跑 full CEM，随后写 S6 evidence 和 RL export gate。

Phase 2 的终点是：

- 每个 CEM-ready row 都有 CEM terminal state；
- 生成 `s6_downstream/rl_export/rl_export_input.tsv`；
- rows 要么 `RL_EXPORT_READY`，要么有具体 skip/fail reason；
- 不启动 RL。

## Claims

1. 已审查过的 bucket/desk/chair source templates 可以通过显式 TSV release，而不是绕过 v3 template gate。
2. 之前被 `stage2b_template_manual_review_required` 拦住的 desk/chair rows，应能在 release 后进入 Stage2b。
3. Phase 1 必须先给出 CEM-ready + visual QC 证据，避免直接把错误 template 推进 CEM。
4. Phase 2 的 RL export table 必须可复现且诚实：只有 template、Stage2b、target gate、visual QC、CEM 全部通过的 row 才能成为 `RL_EXPORT_READY`。

## Phase 1: CEM-Ready + 可视化

### 1.1 显式 Template Release

创建新的 review TSV，显式 release 17 个已审查 draft templates，同时保留 4 个已通过的 bucket templates。

输入：

- `workspace/core4d/results/E144/E144_full_nonbox_raw_contact/s2_templates/draft_proxy/draft_proxy_review_queue.tsv`
- `workspace/core4d/results/E144/E144_full_nonbox_raw_contact/s2_templates_mesh_collision_review/mesh_collision_review_manifest.tsv`
- `workspace/core4d/results/E144/E144_full_nonbox_raw_contact/s2_templates/nonbox_template_review.tsv`

输出：

- `workspace/core4d/results/E145/full_nonbox_to_rl_ready/s2_templates/nonbox_template_review.tsv`

预期 release 数：

| 类别 | source templates |
|---|---:|
| bucket | 9 |
| desk | 8 |
| chair | 4 |
| total | 21 |

规则：

- `nonbox_template_review.tsv` 是唯一 release 权威。
- 每个 `approve_clean` row 必须带 orbit review 和 mesh/collision review evidence path。
- `render pass` 不等于 approve；approve 的依据是 E144 logs 181-183 中完成的人工 visual review。

### 1.2 Registry Refresh

使用 v3 state updater 合并 template review，不能手工改 registry：

```bash
python workspace/core4d/scripts/data_construction_v3/state/update_case_state_registry.py \
  --registry-dir workspace/core4d/results/E145/full_nonbox_to_rl_ready/registries \
  --from-template-review-tsv workspace/core4d/results/E145/full_nonbox_to_rl_ready/s2_templates/nonbox_template_review.tsv \
  --evidence-root workspace/core4d/results/E145/full_nonbox_to_rl_ready/s2_templates \
  --source-ref E145_nonbox_template_release
```

### 1.3 Stage2b

对 E144 5cm pass 的 82 rows 重新跑 Stage2b：

- `retarget_variant_id=omnirt_v1`
- `target_variant_id=ref_fk`

预期输入：

| 类别 | raw-contact pass rows |
|---|---:|
| bucket | 33 |
| desk | 33 |
| chair | 16 |
| total | 82 |

硬 gate：

- 如果 release 后仍有 row 被 template status 拦住，暂停并查原因。
- 不允许未 release template 进入 Stage2b。

### 1.4 Target Gate + Visual QC

对 Stage2b pass rows 跑 target gate。

然后生成 visual QC package，并做人工审查。审查标准：

- 没有明显 object offset；
- 没有物体飞离或跳变；
- 没有 robot collapse；
- 没有严重 lower-body/object entanglement；
- 没有明显 template pose 错误。

输出：

- target gate manifest；
- visual render package；
- `visual_qc_review.tsv`；
- refreshed registry。

注意：

- visual render `pass` 只表示视频/图生成成功。
- 必须有人工 `visual_qc_review.tsv` 才能把 row 放到 S5 handoff。

### 1.5 S5 Handoff + CEM-Ready Manifest

从刷新后的 registry 构建 S5 handoff。

然后生成 E145 CEM-ready variants，默认 CEM 方法沿用 E143/E144：

```text
method_id=raw_mask_ref_fk
contact_hdmi_target_source=ref_fk
contact_hdmi_target_uses_eef_offset=true
contact_hdmi_gain=5.0
contact_hdmi_mask_source=core4d_3cm
contact_hdmi_mask_time_axis=auto
hold_band=false
```

每个 CEM-ready row 必须 preflight：

- `scene_act` 存在；
- trajectory 存在；
- raw contact mask 存在；
- override YAML 存在；
- object assets 本地存在；
- remote sync path 可推导。

Phase 1 成功标准：

- 82 条 E144 5cm pass rows 都有 Stage2b/target/visual/S5 terminal state。
- 21 个 source templates 都在 explicit template review TSV 中。
- CEM-ready TSV 和 `variants.tsv` 已生成。
- CEM-ready rows 的 preflight 100% pass。
- 可视化审查完成并记录。
- 没有启动 CEM。

## Phase 2: 3 卡 Full CEM + RL Export Gate

### 2.1 运行前检查

Phase 2 只消费 Phase 1 的 `variants.tsv`。启动前检查：

- `variants.tsv` row count 与 CEM-ready TSV 一致；
- split list 覆盖所有 rows；
- local/remote output root 不覆盖 E144；
- object assets 能同步到远程；
- 没有 E145 stale outputs 会被误 skip。

### 2.2 3 卡并行 Full CEM

执行方式：

- 本地 GPU0 跑一个 split；
- 远程 `spider-remote` GPU0/GPU1 跑两个 split。

脚本要求：

- `workspace/core4d/scripts/train/train_E145_full_nonbox_raw_mask_ref_fk.sh`
- `workspace/core4d/scripts/run_E145_remote.sh`
- `workspace/core4d/scripts/pull_E145_remote_results.sh`
- `workspace/core4d/scripts/eval/eval_E145_full_nonbox_raw_mask_ref_fk.sh`

输出路径：

- `workspace/core4d/results/E145/full_nonbox_to_rl_ready/cem/full/`
- `logs/E145/cem/full/`
- `workspace/core4d/results/E145/full_nonbox_to_rl_ready/eval/full/`

安全规则：

- 不覆盖 E144 CEM 输出。
- 不 kill 无关 GPU 进程。
- 如发现 stale outputs，只删除 E145-owned stale row。

### 2.3 Eval + S6 Evidence + RL Export

评估每条 CEM variant，写出：

- per-row CEM metrics；
- missing output table；
- CEM pass/fail decision；
- S6 downstream evidence input；
- S6 evidence registry；
- `s6_downstream/rl_export/rl_export_input.tsv`。

RL export decision 定义：

| decision | 含义 |
|---|---|
| `RL_EXPORT_READY` | CEM pass 且所有 RL 输入路径存在 |
| `SKIP_CEM_FAIL` | CEM 跑完但 strict gate fail |
| `SKIP_MISSING_CEM_ARTIFACT` | CEM 产物缺失 |
| `SKIP_NOT_HANDOFF_READY` | CEM 前被 gate 拦住 |
| `SKIP_TEMPLATE_OR_VISUAL_QC` | template/target/visual QC 未通过 |

Phase 2 成功标准：

- 所有 CEM-ready rows 都有 CEM terminal state。
- 每个 row 要么有完整 CEM artifacts，要么有明确 missing/failure reason。
- `rl_export_input.tsv` 存在。
- `RL_EXPORT_READY` rows 只包含真实可消费路径。
- 没有启动 RL，没有生成 checkpoint。

## 总体验证

Phase 1 结束前：

```bash
find workspace/core4d/scripts/data_construction_v3 workspace/core4d/scripts/E145 workspace/core4d/scripts/eval -name '*.py' -print0 | xargs -0 python3 -m py_compile
bash -n workspace/core4d/scripts/train/train_E145_full_nonbox_raw_mask_ref_fk.sh
bash -n workspace/core4d/scripts/run_E145_remote.sh
bash -n workspace/core4d/scripts/pull_E145_remote_results.sh
git diff --check
```

Phase 2 / S6 结束后：

```bash
python workspace/core4d/scripts/data_construction_v3/qa/verify_reproducibility.py \
  --run-root workspace/core4d/results/E145/full_nonbox_to_rl_ready

find workspace/core4d/results/E145/full_nonbox_to_rl_ready -type f \
  \( -name '*.pt' -o -name '*.pth' -o -name '*.ckpt' \) -print
```

checkpoint 搜索必须为空。

## Stop Conditions

Phase 1 停止条件：

- template release TSV 缺 evidence path；
- release 后仍有异常 template block；
- visual QC 存在 gross failure；
- CEM-ready preflight 不是 100% pass。

Phase 2 停止条件：

- split list 与 `variants.tsv` 不一致；
- remote sync 缺 assets/masks/tasks；
- CEM artifacts 大面积缺失；
- S6 export 无法解释每个 row 的 terminal state。

最终无论 `RL_EXPORT_READY` 是否为 0，都只报告 RL-ready gate，不进入 RL。
