# E148 — E143 24-case rubber hand collision extension with E147 reuse

## Context

E143 已完成 `raw_mask_ref_fk` 24-case sweep，并输出同 case 的 `OmniRetarget / ref_fk / raw_mask_ref_fk` 对比。E147 已把 `hand_collision_variant_id=rubber_hull` 接入 v3 downstream/CEM 维度，并在 10 个历史 sphere5cm CEM case 上完成 rubber hand full CEM。

本实验把 E147 的 rubber hand collision 评估扩展到 E143 的 24-case workset：E143 `raw_mask_ref_fk` 作为 sphere Spider 基线，新增 rubber hand Spider 结果，并和 OmniRetarget 同口径对比。

关键复用事实：

- E147 rubber_hull 结果已写入 `workspace/core4d/results/E147/registries/case_state_registry.tsv` 和 `workspace/core4d/results/E147/s6_downstream/evidence/downstream_evidence_manifest.tsv`。
- E143 24-case 中有 8 条和 E147 完全重叠；经核对，重叠行的 base `scene_act.xml` hash 一致，可直接复用 E147 rubber outdir trajectory/MP4/metrics，不需要重跑。
- E148 因此只需要跑剩余 16 条新 rubber_hull full CEM。

E143 中复用 E147 的 8 条：

| case_id | 说明 |
|---|---|
| `box023_person2` | E143 task 名与 E147 不同，但 base scene hash 一致，sphere outdir trajectory 同源 |
| `bucket004_20231003_1_012_p1` | 复用；后续 filtered summary 排除该视觉失败 case |
| `d003_box021_20231011_035_p1` | 复用 |
| `d003_box021_20231011_035_p2` | 复用 |
| `e091_box004_20231003_2_082_p1` | 复用 |
| `e091_box004_20231003_2_083_p1` | 复用 |
| `e091_box026_20231020_134_p1` | 复用 |
| `e091_box026_20231020_134_p2` | 复用 |

## Claims

1. **Compute reuse claim**：E143 24-case 中 8 条已有 E147 rubber_hull 结果可安全复用；E148 runner 只启动 16 条新 full CEM，不重复计算已注册 case。
2. **Comparison completeness claim**：最终比较表必须覆盖 E143 24/24 cases，并且每个 case 都有 `OmniRetarget`、`sphere Spider(raw_mask_ref_fk)`、`rubber hand Spider` 三列指标。
3. **Metric clarity claim**：输出完整 24-case 和 filtered 23-case 两套平均；filtered 只排除 `bucket004_20231003_1_012_p1`，不从实验运行/逐 case 表删除。
4. **No algorithm change claim**：E148 只替换手部碰撞体为 `rubber_hull`，不改 CEM reward、优化算法、E143 sphere baseline 或 OmniRetarget 输入。

## Implementation Plan

### A. Manifest + scene/override

新增 `workspace/core4d/scripts/E148/build_rubber_hand_collision_manifest.py`。

输入：

- E143 source of truth：`workspace/core4d/scripts/E143/variants.tsv`
- E147 reuse evidence：
  - `workspace/core4d/scripts/E147/variants.tsv`
  - `workspace/core4d/results/E147/rubber_hand_collision/comparison/e147_omni_sphere_rubber_case_comparison.tsv`
  - `workspace/core4d/results/E147/registries/case_state_registry.tsv`
  - `workspace/core4d/results/E147/s6_downstream/evidence/downstream_evidence_manifest.tsv`

输出：

- `workspace/core4d/scripts/E148/variants.tsv`
- `workspace/core4d/results/E148/e143_24case_rubber_hand_collision/cases_manifest.tsv`
- `workspace/core4d/results/E148/e143_24case_rubber_hand_collision/hand_collision_adapter_manifest.tsv`
- `workspace/core4d/results/E148/e143_24case_rubber_hand_collision/scene_snapshot/manifest.txt`
- `examples/config/override/core4d_E148_*_rubber_hull.yaml`

Manifest 字段必须包含：

- E143 identity：`e143_variant`、`e109_case_id`、`case_id`、`object_key`、`derived_task`、`person_idx`
- baselines：`omni_qpos_path`、`omni_scene_xml`、`sphere_npz`、`sphere_outdir_npz`、`sphere_video`、`sphere_scene_act`
- rubber run/reuse：`run_status` (`reuse_e147` / `to_run`)、`reuse_source_exp`、`rubber_npz`、`rubber_outdir_npz`、`rubber_video`
- execution: `split`, `override`, `base_scene_act`, `rubber_scene_act`, `scene_name`, `object_asset`

Rules:

- `run_status=reuse_e147` iff E143 `e109_case_id` matches E147 `case_id`, E147 rubber outdir trajectory exists, and E143 sphere/base scene hash equals E147 base scene hash.
- `run_status=to_run` rows get a new E148 rubber sidecar scene using `patch_hand_collision.py` with `hand_collision_variant_id=rubber_hull`, `maxhullvert=64`, `scene_name=scene_act_E148_rubber_hull`.
- `to_run` rows split evenly as `remote-gpu0` / `remote-gpu1` with 8 rows per GPU.
- Existing E147 rows remain referenced in E148 manifest; do not copy/rename their result files.

### B. Runner / remote / pull

新增 fixed entry scripts:

- `workspace/core4d/scripts/train/train_E148_e143_rubber_hand_collision.sh`
- `workspace/core4d/scripts/run_E148_remote.sh`
- `workspace/core4d/scripts/pull_E148_remote_results.sh`

Runner behavior:

- `list full 0 remote-gpu0` and `list full 0 remote-gpu1` must list only `run_status=to_run` rows.
- `run_one` must skip `reuse_e147` rows.
- New outputs go to `workspace/core4d/results/E148/e143_24case_rubber_hand_collision/cem/full/`.
- Logs go to `logs/E148/cem/full/`.
- Keyframes are extracted for new E148 MP4s only; reused E147 rows keep their E147 video paths.

Remote behavior:

- Build manifest locally first.
- Sync E148 scripts, train script, E148 result sidecar scenes, E148 overrides, task dirs, object assets, and E143 contact masks to `spider-remote`.
- Remote tmux starts two background workers:
  - GPU0 runs 8 new rows.
  - GPU1 runs 8 new rows.
- Pull script only pulls E148 newly generated outputs/logs; reused E147 artifacts remain referenced in place.

### C. Evaluation + XLSX

新增:

- `workspace/core4d/scripts/eval/eval_E148_e143_rubber_hand_collision.py`
- `workspace/core4d/scripts/eval/eval_E148_e143_rubber_hand_collision.sh`

Evaluation inputs:

- E143 method metrics / raw rows for OmniRetarget and sphere Spider.
- E147 reused rubber trajectories for 8 rows.
- E148 newly run rubber trajectories for 16 rows.

Metrics:

- `5cm`
- `10cm`
- `手物穿透`
- `深穿透2cm`
- `腿穿透`
- `body穿透`
- `物体触地`
- `pelvis_min_m`
- `fall_flag`

Diff conventions:

- `rubber-sphere = rubber hand Spider - sphere Spider`
- `rubber-Omni = rubber hand Spider - OmniRetarget`
- Markdown summary also reports `OmniRetarget-rubber = OmniRetarget - rubber hand Spider` for quick interpretation.

Outputs:

- `workspace/core4d/results/E148/e143_24case_rubber_hand_collision/eval/full/e148_method_metrics.tsv`
- `workspace/core4d/results/E148/e143_24case_rubber_hand_collision/eval/full/e148_case_comparison.tsv`
- `workspace/core4d/results/E148/e143_24case_rubber_hand_collision/eval/full/e148_summary.md`
- `workspace/core4d/results/E148/e143_24case_rubber_hand_collision/comparison/E148_e143_omni_sphere_rubber_hand_comparison.xlsx`

Workbook sheets:

- `24case平均`
- `逐case对比`
- `filtered平均`
- `filtered逐case`

Filtered definition:

- Exclude only `bucket004_20231003_1_012_p1`.
- Reason: E147/E143 visual inspection shows late bucket drift, so its 5cm/10cm contact metric is dominated by an obvious failed trajectory.

## Execution Commands

Manifest/preflight:

```bash
.venv/bin/python workspace/core4d/scripts/E148/build_rubber_hand_collision_manifest.py
bash workspace/core4d/scripts/train/train_E148_e143_rubber_hand_collision.sh list full 0 remote-gpu0
bash workspace/core4d/scripts/train/train_E148_e143_rubber_hand_collision.sh list full 0 remote-gpu1
```

Remote full CEM:

```bash
bash workspace/core4d/scripts/run_E148_remote.sh full
```

Monitor:

```bash
ssh spider-remote "tmux capture-pane -t E148_full_* -p | tail -80"
```

Pull:

```bash
bash workspace/core4d/scripts/pull_E148_remote_results.sh full
```

Eval:

```bash
bash workspace/core4d/scripts/eval/eval_E148_e143_rubber_hand_collision.sh full
python /home/ubuntu/.codex/skills/xlsx/scripts/recalc.py \
  workspace/core4d/results/E148/e143_24case_rubber_hand_collision/comparison/E148_e143_omni_sphere_rubber_hand_comparison.xlsx 60
```

## Success Criteria

Manifest:

- 24 total rows.
- exactly 8 `reuse_e147`.
- exactly 16 `to_run`.
- reused rows have existing E147 rubber outdir trajectory, root NPZ, and MP4.
- to-run rows have valid override, base scene, rubber sidecar scene, trajectory, object asset, and contact mask.

Remote execution:

- Remote list counts are 8/8 for GPU0/GPU1.
- Only 16 new E148 rows are launched.
- 16/16 new root NPZ exist.
- 16/16 new MP4 exist.
- 16/16 new outdir `trajectory_mjwp_act.npz` exist.
- Remote tmux exits naturally; no E148 process remains.

Evaluation:

- Evaluator produces 24/24 rubber rows: 8 reused from E147 + 16 new E148.
- No missing OmniRetarget/sphere/rubber metrics.
- XLSX formula recalculation reports `total_errors=0`.
- Workbook readback confirms sheet row counts:
  - `24case平均`: 3 method rows, `case_count=24`
  - `逐case对比`: 24 case rows
  - `filtered平均`: 3 method rows, `case_count=23`
  - `filtered逐case`: 23 case rows

Logging:

- Create `workspace/core4d/log/188_E148_e143_24case_rubber_hand_collision_results.md`.
- Update `workspace/core4d/EXPERIMENT_TRACKER.md` after evaluation completes.
- Update `workspace/core4d/progress.md` after manifest, remote launch, pull, eval, and final log.

## Validation

Before remote launch:

```bash
.venv/bin/python -m py_compile \
  workspace/core4d/scripts/E148/build_rubber_hand_collision_manifest.py \
  workspace/core4d/scripts/eval/eval_E148_e143_rubber_hand_collision.py
bash -n \
  workspace/core4d/scripts/train/train_E148_e143_rubber_hand_collision.sh \
  workspace/core4d/scripts/run_E148_remote.sh \
  workspace/core4d/scripts/pull_E148_remote_results.sh \
  workspace/core4d/scripts/eval/eval_E148_e143_rubber_hand_collision.sh
git diff --check -- \
  workspace/core4d/plan/156_E148_e143_24case_rubber_hand_collision_plan.md \
  workspace/core4d/scripts/E148 \
  workspace/core4d/scripts/train/train_E148_e143_rubber_hand_collision.sh \
  workspace/core4d/scripts/run_E148_remote.sh \
  workspace/core4d/scripts/pull_E148_remote_results.sh \
  workspace/core4d/scripts/eval/eval_E148_e143_rubber_hand_collision.py \
  workspace/core4d/scripts/eval/eval_E148_e143_rubber_hand_collision.sh \
  workspace/core4d/progress.md
```

After eval:

```bash
.venv/bin/python - <<'PY'
import pandas as pd
p = "workspace/core4d/results/E148/e143_24case_rubber_hand_collision/comparison/E148_e143_omni_sphere_rubber_hand_comparison.xlsx"
xl = pd.ExcelFile(p)
print(xl.sheet_names)
for sheet in xl.sheet_names:
    df = pd.read_excel(p, sheet_name=sheet)
    print(sheet, df.shape)
    print(df.head().to_string(index=False))
PY
```

## Out of Scope

- Do not rerun E143 sphere/raw_mask_ref_fk.
- Do not rerun E147 overlap rows.
- Do not launch RL.
- Do not change CEM reward, optimizer, target route, OmniRetarget, or object collision templates.
- Do not replace default `sphere5cm`; `rubber_hull` remains an experimental hand-collision variant.
