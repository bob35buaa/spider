# E111 Contact Evidence Chain 结果

计划：`workspace/core4d/plan/120_E111_contact_evidence_chain_plan.md`

总控计划：`workspace/core4d/plan/contact_improvement_plan.md`

## 目标

E111 承接 E110：把 `data_construction_v3` 的 raw-contact 从 gate 证据升级为后续 E112 contact-aware CEM 可消费、可评测、可复现的 contact evidence chain。

本轮不跑 full CEM，不使用 GPU/远程机器。

## 结果路径

| 产物 | 路径 |
|---|---|
| S6 evaluator | `workspace/core4d/scripts/data_construction_v3/stages/s6_downstream/evaluate_contact_alignment.py` |
| 固定 eval | `workspace/core4d/scripts/eval/eval_E111_contact_alignment_smoke.sh` |
| chain eval | `workspace/core4d/scripts/eval/eval_E111_contact_alignment_chain_smoke.sh` |
| alignment smoke | `workspace/core4d/results/E111/contact_alignment_smoke/` |
| alignment chain smoke | `workspace/core4d/results/E111/contact_alignment_chain_smoke/` |
| contact chain smoke | `workspace/core4d/results/E111/contact_chain_smoke/E111_contact_chain_smoke/` |
| release check | `workspace/core4d/results/E111/release_check_no_smoke/` |

## 改动

- S1 `run_raw_contact.py`：
  - TSV 新增 `contact_mask_npz/contact_label/contact_person_idx/raw_frame_count/raw_to_trimmed_mapping_status/left/right/both active/longest_run/contact_target_status`。
  - `raw_contact_proxy.npz` 新增 `raw_contact_centroid_world_3cm/5cm`、`raw_contact_centroid_object_local_3cm/5cm` 和 `raw_to_trimmed_frame_index`。
  - 当前没有 Stage2b trimmed mapping 时显式写 `missing_trimmed_mapping`，不伪造 trimmed mask。
- S3/S4/S5：
  - `stage2b_manifest`、`target_gate_manifest`、`handoff_manifest`、registry 保留 `contact_mask_*`、`raw_contact_artifact_npz`、per-threshold raw artifact、`contact_target_*` 和 route diagnostic ref。
  - Stage2b 未执行或未生成 trimmed mask 时，`contact_mask_npz` 留空，`contact_mask_status=missing_trimmed_mask`，只在 `stage2b_contact_mask_npz_expected` 记录预期路径。
  - `export_cem_overrides.py` 在 handoff pass/review 且有真实 mask 时写入 `contact_hdmi_mask_source/path/person/time_axis`；坏 mask path 会使 override fail。
- S6：
  - 新增 mask-based contact alignment evaluator；只有显式提供 method mask 时才计算 PR/Recall/F1/IoU。
  - replay/SDF/physics-dependent fields 在 E111 smoke 中留空，不伪造物理接触。
- QA/doc：
  - `verify_reproducibility.py` 增加 S1 contact field 和 NPZ key 检查。
  - `03_manifest_schema.md` 补 E111 contact evidence 字段。

## 执行与验证

固定 E111 alignment smoke：

```bash
bash workspace/core4d/scripts/eval/eval_E111_contact_alignment_smoke.sh
bash workspace/core4d/scripts/eval/eval_E111_contact_alignment_chain_smoke.sh
```

结果：

| 项目 | 数值 |
|---|---:|
| rows | 1 |
| evaluated | 1 |
| mean raw-contact F1 | 0.857143 |

chain smoke 读取真实 `s5_handoff/handoff_manifest.tsv`：

| 项目 | 数值 |
|---|---:|
| rows | 4 |
| raw_contact_only | 4 |
| mean raw-contact F1 | 空；无 method mask，不伪造 |

E111 compact chain smoke：

```bash
python3 workspace/core4d/scripts/data_construction_v3/orchestration/run_pipeline.py \
  --mode full-from-raw \
  --run-id E111_contact_chain_smoke \
  --run-root workspace/core4d/results/E111/contact_chain_smoke \
  --core4d-raw-root /mnt/a0ccc676-9496-49f8-a861-f8a1797dec52/mocap_data/CORE4D/CORE4D_Real \
  --smplx-model-dir /home/ubuntu/Workspace/CORE4D-Instructions/dataset_utils/smplx \
  --holosoma-repo /home/ubuntu/Workspace/holosoma \
  --queue selected-medium-box \
  --max-sequences 1 \
  --sample-count 500 \
  --thresholds-m 0.03,0.05 \
  --stage2b-contact-label 3cm \
  --retarget-variant-id omnirt_v1 \
  --target-variant-id ref_fk
```

结果：

| 项目 | 数值 |
|---|---:|
| run status | pass |
| S1 selected case-person | 2 |
| 3cm pass | 2 |
| 5cm pass | 2 |
| S3 stage2b ready | 2 |
| S5 handoff rows | 4 |

实际字段检查：

- S1 candidates 已写 `contact_mask_npz` 和 run-length 字段；
- S1 NPZ 含 `raw_contact_mask_3cm/5cm`、world/object-local centroid 和 `raw_to_trimmed_frame_index=-1`；
- S3/S4/S5/registry 均保留 `raw_contact_artifact_npz` 和 `contact_mask_*` 字段；
- S3/S4 中 `contact_mask_npz` 在 Stage2b 未执行时为空；`stage2b_contact_mask_npz_expected` 只记录未来 execute 的预期输出；
- `raw_contact_artifact_npz` 表示 S1 raw artifact，`contact_mask_3cm_npz/contact_mask_5cm_npz` 同时保留两档阈值，避免覆盖。

静态与 release 检查：

```bash
find workspace/core4d/scripts/data_construction_v3 -name '*.py' -print0 | xargs -0 python3 -m py_compile
bash -n workspace/core4d/scripts/eval/eval_E111_contact_alignment_smoke.sh
workspace/core4d/scripts/data_construction_v3/orchestration/run_release_checks.sh --run-root workspace/core4d/results/E111/release_check_no_smoke --no-smoke
git diff --check -- ...
```

均通过。release audit：`63 passed / 0 failed`。

补充：直接运行 `verify_reproducibility.py` 检查 compact chain smoke 时，contact 检查没有报错，但该 run 缺少 S0 `config_resolved.json/run_manifest.json/git_state.json`，因此 verifier 按旧 S0 完整性规则返回 fail。这个问题不是 contact evidence 断点；后续可在 run_pipeline/init_workspace 侧统一补 S0 self-describing 文件。

## Claims 验证

| Claim | 结果 |
|---|---|
| C1: S1 raw contact artifact 可作为训练/评测证据 | 通过：TSV 写 path/active/run-length，NPZ 写 raw mask 与 centroid |
| C2: S3/S5/S6 能追踪 contact artifact 来源 | 通过：S3/S4/S5/registry 均传播 raw artifact；未生成 trimmed mask 时不写假 `contact_mask_npz` |
| C3: S6 evaluator 可做 raw-mask alignment baseline | 通过：fixture smoke 输出 PR/Recall/F1/IoU；chain smoke 输出 `raw_contact_only` |
| C4: QA 可发现 contact artifact 缺失 | 通过主要路径：`verify_reproducibility.py` 已检查 S1/S3/S4/S5/override 非空 contact path；compact run 的整体 verifier 仍被旧 S0 文件缺失挡住 |
| C5: E111 可服务 E112 | 通过：E112 可从 S5 handoff/override 获取 mask path、label、person/time axis |

## 结论

E111 已完成 contact evidence chain 的最小可用版本。后续 E112 可以基于 `contact_mask_*` 与 `raw_contact_artifact_npz` 设计 raw-mask/ref-FK、hold-band、SDF near-zero 等 CEM ablation。

E112 开始 full CEM 时再按用户要求使用本地 1 卡 + 远程 2 卡并行，并且不 kill 其他 GPU 进程。
