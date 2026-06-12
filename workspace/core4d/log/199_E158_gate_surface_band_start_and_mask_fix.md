# E158 — gateA + surfaceBand-A 启动与 mask 修正记录

> 计划：`workspace/core4d/plan/167_E158_gateA_surface_band_plan.md`
> 状态：Stage-A 诊断完成；full CEM 准备启动/排队
> 指标口径：`core4d-e154-physics-contact-v1`

## 0. 当前结论

E158 在正式跑 `gateA+surfaceBand-A` 前先修正了两个 contact mask 问题，否则 Stage-A 可视化和 masked metrics 会被污染。

## 1. Mask 修正

| case | 问题 | 修正 | 备份 |
|---|---|---|---|
| `box021_035_p1` | E143 canonical mask 误用旧 E079 `box021_person1`，`spider_contact_mask_3cm` 只有 88 帧，但 E107 clean/E156 轨迹为 129 帧 | 替换为 case-specific `workspace/core4d_collab_retarget/results/E029/d6/contact_masks/d003_box021_20231011_035_p1/raw_contact_mask_3cm.npz`，129 帧 | `workspace/core4d/results/E143/contact_masks/box021_035_p1/raw_contact_mask_3cm.bad_88frame_backup.npz` |
| `box021_035_p2` | target person union mask 在 frame `91..92` 有 2 帧孤立断口，导致 3s 附近出现假 release | 保守填补右手：`spider_contact_mask_3cm[91:93, person_idx=1, right_hand=1]=true`；union window 从 `17..90 + 93..119` 变为 `17..119` | `workspace/core4d/results/E143/contact_masks/box021_035_p2/raw_contact_mask_3cm.pre_e158_gapfill_backup.npz` |

E143 builder 已固化上述修正：`box021_035_p1` 优先使用 case-specific mask；`box021_035_p2` 会自动应用 frame `91..92` 的 manual fill 并写入 metadata。

## 2. 防呆更新

- `workspace/core4d/scripts/experiments/E158/diagnose_e156_surface_band.py`：取消 mask 补零/截断；mask/qpos 长度不一致直接报错。
- `workspace/core4d/scripts/eval/core/core_metrics.py`：masked contact 评测要求 `spider_contact_mask_3cm.shape[0] == qpos_frames`，避免静默取 `min(T)`。

## 3. 已重算

| 项目 | 结果 |
|---|---|
| clean8 mask/qpos 长度审计 | 8/8 OK |
| E158 Stage-A diagnostics | `timeseries_rows=2040`, `band_rows=54` |
| E156 full eval | `metric_rows=32`, `missing=0` |
| E158 allow-missing eval | `metric_rows=24`, `missing=6`，缺项仅为未跑的 6 条 `gateA+surfaceBand-A` |

`box021_035_p1` 修正后 mask window 为 frame `18..117`；`box021_035_p2` 修正后 frame `88..95` 全 active，3s 附近不再断开。

## 4. Full CEM 启动策略

当前本地 GPU0 和远程 GPU0/GPU1 都被 R154 downstream RL 占用。为避免污染正在训练的 RL，E158 full CEM 使用 queued 启动：

```bash
WAIT_FOR_GPU_IDLE=1 bash workspace/core4d/scripts/launch/active/run_E158_local.sh full
WAIT_FOR_GPU_IDLE=1 bash workspace/core4d/scripts/launch/active/run_E158_remote.sh full
```

默认等待阈值：

| 参数 | 值 |
|---|---:|
| `E158_GPU_IDLE_MAX_MEM_MB` | 3000 |
| `E158_GPU_IDLE_MAX_UTIL_PCT` | 20 |
| `E158_GPU_IDLE_POLL_SEC` | 120 |
| `E158_GPU_IDLE_STABLE_POLLS` | 2 |

资源分配仍按计划：

| 资源 | cases |
|---|---|
| local GPU0 | `box021_035_p1`, `box021_035_p2` |
| remote GPU0 | `box021_029_p2`, `box004_083_p1` |
| remote GPU1 | `box004_083_p2`, `box023_person2` |

启动状态（2026-06-12 20:15 CST）：

| 位置 | tmux session | 状态 |
|---|---|---|
| local GPU0 | `E158_local_full_queued_201531` | waiting for GPU idle |
| remote GPU0/GPU1 | `E158_full_queued_201506` | waiting for GPU idle |

监控命令：

```bash
tmux capture-pane -t E158_local_full_queued_201531 -p | tail -80
ssh spider-remote "tmux capture-pane -t E158_full_queued_201506 -p | tail -100"
```

用户随后要求不等待 GPU idle，直接与现有 R154/RL 任务叠加运行，但不 kill 其他程序。因此只停止 E158 queued session，未动其他训练；重新启动 overlap full：

| 位置 | tmux session | 首个任务 |
|---|---|---|
| local GPU0 | `E158_local_full_overlap_201921` | `E158_box021_035_p1_gateA_surfaceBandA` |
| remote GPU0/GPU1 | `E158_full_overlap_201904` | GPU0: `box021_029_p2`；GPU1: `box004_083_p2` |

监控命令：

```bash
tmux capture-pane -t E158_local_full_overlap_201921 -p | tail -80
ssh spider-remote "tmux capture-pane -t E158_full_overlap_201904 -p | tail -100"
```

## 5. 验证

```bash
bash -n workspace/core4d/scripts/launch/active/run_E158_local.sh
bash -n workspace/core4d/scripts/launch/active/run_E158_remote.sh
python -m py_compile workspace/core4d/scripts/experiments/E143/build_raw_mask_ref_fk_24case_manifest.py
python -m py_compile workspace/core4d/scripts/experiments/E158/diagnose_e156_surface_band.py
python -m py_compile workspace/core4d/scripts/eval/core/core_metrics.py
```

以上检查已通过。
