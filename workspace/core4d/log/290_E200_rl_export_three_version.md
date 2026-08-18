# E200 三版 RL-Export 导出

## 目的

E199（PRG 物体平移增强）+ E200（noPRG / PRG+G1+A2 三臂对比）的 CEM rollout 人工审核完成后
（`results/E200/s6_downstream/eval/E200_three_arm_master-review.xlsx` sheet `by_rollout`），
按**每物体指定 arm** 从 `manual` / `manual_label` 列导出 3 版 RL-ready 双人运动数据，字段对齐历史
`paired_rl_export_input.tsv`（E198/E190 63 列 schema），必须带 partner 字段。

## 口径（arm × object）

| object | arm | orig 源 | trans0/1/2 源 |
|---|---|---|---|
| box001 | PRG+G1+A2 | E198 `cem/full_g1a2_box001/` + `box001_user_approved` paired | E200 `cem/prg_g1a2/` |
| box024 | PRG+G1+A2 | E198 `cem/full_g1a2/` + `box024_user_approved` paired | E200 `cem/prg_g1a2/` |
| box004 | PRG+G1+A2 | E198 `cem/full_g1a2/` + `box004_user_approved` paired | （0 aug USE） |
| box023 | noPRG | E179 CEM（E190 `box023_noPRG_user_approved` paired） | E200 `cem/noprg/` |
| box021 | PRG | E170 `cem/full/`（4 例 E169）+ E170 paired | E199 `cem/full/` |

三版筛选（`manual` 门）：
- **box001_only**：box001 PRG+G1+A2 `manual==USE` → **47**（10 orig + 37 aug）
- **5object_all**：各 object 指定 arm `manual==USE` → **109**
  （box001=47, box024=16, box004=3, box023=19, box021=24）
- **5object_box001clean**：其余四物体同上（62）；box001 取 `USE & manual_label==CLEAN`（33）
  + 2 例豁免 `box001_20231003_1_040_p2`（orig, trans1；USE 但 MINOR_ACCEPTABLE）= 35 → **97**

数值门在 `select_rl_export_cases.py` 内硬断言（总数 + orig/aug 子计数 + box001 v3=35），
运行通过：47 / 109 / 97 全部匹配。

## Partner 策略（用户决策）

- orig 各物体已有成品 paired manifest（E198/E190/E170），**整行复用**（含 partner）。
- aug 从未做过 partner。经确认采用**每人自己 human-relative 帧、同 `trans_k` label**：
  partner 用自己的 `--augmentation`（omnirt_v2）跑同 label；接触段因偏移指数衰减基本一致，
  接近段两人世界物体位姿不同（可接受）。
- partner 运动 = holosoma `trimmed/{holo}_{trans_k}.npz`（与历史 `partner_trimmed_npz` 同格式）。
  76 个 aug USE 中 **51 个 partner 已存在可复用**，**25 个（9 个 take）需生成**。
- 生成复用 E199 `build_augmented_tasks.run_upstream` + `fixed_window_trim`，partner meta 由
  source aug `task_info.json` 翻转 person 得到。

### 关键 bug（已修复）
`pipeline.sh` 用 `IFS=$'\t' read` 解析 case 文件，**tab 是 whitespace IFS 字符**，空的
`source_scene_task` 字段会被折叠，导致 `target_task` 右移成 `"auto"`（产出 `holosoma_auto`）。
E199 因 `source_scene_task` 始终非空未触发。修复：`partner_meta` 用 source 的 geometry-template
task 名填充 `source_scene_task`（`--skip-spider` 下不使用，仅防字段折叠）。

## 运行命令

```bash
bash workspace/core4d/scripts/train/train_E200_rl_export.sh          # 含 partner 生成
# 或分步：
.venv/bin/python workspace/core4d/scripts/experiments/E200/select_rl_export_cases.py \
  --review-xlsx .../E200_three_arm_master-review.xlsx --out-root .../rl_export
.venv/bin/python workspace/core4d/scripts/experiments/E200/generate_partner_aug.py \
  --selection-tsv .../5object_all/selection.tsv --out-index .../partner_aug_motion_index.tsv \
  --generate --max-workers 6
python3 workspace/core4d/scripts/experiments/E200/build_rl_export.py \
  --repo "$PWD" --rl-export-root .../rl_export --partner-index .../partner_aug_motion_index.tsv
```

## 改动文件

| 文件 | 动作 |
|---|---|
| `workspace/core4d/scripts/experiments/E200/select_rl_export_cases.py` | 新建 |
| `workspace/core4d/scripts/experiments/E200/generate_partner_aug.py` | 新建 |
| `workspace/core4d/scripts/experiments/E200/build_rl_export.py` | 新建 |
| `workspace/core4d/scripts/train/train_E200_rl_export.sh` | 新建 |
| `results/E200/s6_downstream/rl_export/{box001_only,5object_all,5object_box001clean}/` | 产物（不入 git） |
| `results/E200/s6_downstream/rl_export/partner_aug_motion_index.tsv` | partner 运动索引 |

## 结果

三版全部就绪，`paired_rl_export_decision` 全为 `RL_EXPORT_READY`：

| 版本 | 行数 | orig | aug |
|---|---|---|---|
| box001_only | 47 | 10 | 37 |
| 5object_all | 109 | 33 | 76 |
| 5object_box001clean | 97 | 32 | 65 |

- **schema**：三版表头与 E198 `paired_rl_export_input.tsv` **63 列逐列一致**（diff 空）。
- **文件存在性**：每行 `scene_act / trajectory / contact_mask / cem_result_npz /
  partner_trimmed_npz / partner_omniretarget_output_npz / partner_trim_window_json` 全部存在非空。
- **partner 来源**（5object_all）：`reuse_e199_partner_aug=55`、`generate_e200_partner_aug=21`
  （aug 合计 76）；orig：`reuse_e198_stage2b_partner=17`、`reuse_e168_stage2b_partner=9`、
  `reuse_e173_stage2b_partner=7`（合计 33）。
- **partner 生成**：9 个 take（box001×6, box021×1, box024×2）按 object 分组并行生成
  （组内串行以规避 `sync_generated_object_model` / `ensure_g1_object_xml` 写竞争），
  每 take `trans0/1/2` 全可行、`rot0/1` 不可行（本任务不使用 rot）。
- **C3 增强正确性抽检**（新生成 partner）：接近段偏移 ≈0.188–0.200m、终点收敛 0.017–0.041m，
  与 E199 source 侧（0.200m / ~0.027m）一致，证明"每人自己帧、同 label"生成语义正确。

### 两处历史数据修正（build 内 remap，未改源文件）
1. box021 orig（E170/E168 paired）路径为旧挂载绝对路径 `/.../spider_workdirs/core4d/results/...`，
   按 `core4d/results/` / `example_datasets/` marker 重定位到当前 repo。
2. E198 box001 两例 orig（`box001_20231003_2_039_p1`、`_2_041_p1`）源 manifest 存在
   `results/results/E198` 双 `results/` typo，remap 内折叠为单 `results/`（真实文件在单路径）。

### 遗留
- 修复前误产的空目录 `results/E199/data_preprocess/holosoma_auto/` 需手动删除（本会话权限禁止 `rm -rf`）。

## 结论

- 三版导出口径与审核表一致，数值门全部通过。
- orig 复用历史成品 paired 行（含验证过的 partner）；aug 为净新，partner 用同 label 每人帧生成。
- schema 与历史 63 列一致、带完整 partner 字段。
