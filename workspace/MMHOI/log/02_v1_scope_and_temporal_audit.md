# R002 / E002：`C_2/C_8` 范围修订与 30 Hz 审计

## 目标

1. 将当前范围收敛为 `C_2/C_8`（Moving heavy stuffs）。
2. 固定首批实验为 `C_2 + box`。
3. 从完整 release 和一个实际解压 case 判断 SMPL-X/object GT 是否为连续
   30 Hz。
4. 修订可复跑 inventory、数据统计和全局适配方案。

## 输入

```text
/mnt/a0ccc676-9496-49f8-a861-f8a1797dec52/mocap_data/MMHOI_release.zip
/mnt/a0ccc676-9496-49f8-a861-f8a1797dec52/mocap_data/MMHOI
paper/Kogashi 等 - 2025 - MMHOI Modeling Complex 3D Multi-Human Multi-Object Interactions.pdf
```

Git：

```text
base branch: experiment/E161-surface-release-ablation
base commit: 67cef0b84d81107128a082f36804caa16a15251c
work branch: experiment/MMHOI-data-adaptation
```

## 执行

### Inventory 修订

统计器新增：

- `PRIMARY_SCENARIOS = C_2/C_8`；
- `PILOT_SCENARIO = C_2`、`PILOT_OBJECT = box`；
- 主范围 scenario/capture/sample/action/object TSV；
- active-box sample 和 verb TSV；
- `capture_fps` 与 release 30skip 的路径级 temporal audit；
- whole-archive dense candidate extension 和 nested ZIP 审计。

命令：

```bash
python3 workspace/MMHOI/scripts/data_inventory/inventory_mmhoi.py \
  --archive /mnt/a0ccc676-9496-49f8-a861-f8a1797dec52/mocap_data/MMHOI_release.zip \
  --output-dir workspace/MMHOI/results/E002/s0_scope_inventory
```

### 临时解压

从完整 ZIP 临时解压：

```text
MMHOI/sequences/20240412_personA_personB/20240412__C_2__30skip
MMHOI/sequences/20240412_personA_personB/calibs
MMHOI/object/03_box.ply
```

临时根：

```text
/tmp/mmhoi_c2_box.1IzJoU
```

解压占用约 261 MB。该目录仅用于只读结构审计，审计完成后清理；持久证据写入
本日志和 E002 inventory。

## 实际结果

### 主范围

| 指标 | `C_2/C_8` |
|---|---:|
| Scenario captures | 24 |
| Sequence roots | 14 |
| Samples | 1,289 |
| Train / val / test / unspecified | 602 / 308 / 314 / 65 |
| Action rows / active rows | 7,700 / 5,502 |
| Raw / active verbs | 6 / 5 |
| Raw / active `(object, verb)` classes | 26 / 20 |
| Active object types | 6 |
| Cooperative samples | 1,004 |

### 首批 `C_2 + box`

| 指标 | 结果 |
|---|---:|
| `C_2` samples | 636 |
| Active-box samples | 460 |
| Captures / sequence roots | 12 / 12 |
| Train / val / test / unspecified | 223 / 79 / 116 / 42 |
| Box action rows / active rows | 1,272 / 833 |
| Box together rows | 730 |
| Box cooperative samples | 365 |

Box verb：

```text
move together   686 rows / 343 samples
no-interaction  439 rows / 263 samples
pick up          99 rows /  91 samples
stack             4 rows /   4 samples
stack together   44 rows /  22 samples
```

### 完整 archive temporal audit

```text
C_2/C_8 numeric frame folders:                 1,289
C_2/C_8 PARAM/action frame folders:            1,289
C_2/C_8 PARAM/two-person JSON frame folders:   1,289
C_2/C_8 final object mesh frame folders:       1,289
capture 内相邻 frame-id gap:                   {30: 1265}
30skip/30_skip captures:                       24 / 24
scenario 数字帧外文件:                         mask_exist_all.csv × 24
```

Whole archive 中未发现：

```text
.npy .npz .pkl .pickle
.mp4 .avi .mov .mkv
.bvh .c3d
```

唯一 nested ZIP 位于无关 `C_6`，uncompressed size 为 22 bytes，与 dense
trajectory 无关。

### 解压 case

数字帧：

```text
00361 00391 00421 00451 00481 00511
00541 00571 00601 00631 00661 00691
```

12 个 gap 全为 30。每个稀疏帧均有：

```text
PARAM/person1.json
PARAM/person2.json
final/box.ply
Mesh_SMPLH/person1.obj
Mesh_SMPLH/person2.obj
```

`mask_exist_all.csv` 为 48 行（12 frame ids × 4 cameras），没有额外 frame
id。数字帧目录外只有该 CSV；其余解压内容只有静态 box template 和
calibration。

Hash 审计：

```text
PARAM/person1.json  12 unique / 12
PARAM/person2.json  12 unique / 12
final/box.ply        3 unique / 12
```

`final/box.ply` 重复说明多个稀疏点的 box 位姿相同，不提供两个点之间的
30 Hz 运动。

该 case 的 12 个 sparse samples 中只有 `00541` 有 active box，且动作是
单人 `pick up`；定位为 layout probe，不选作双人搬箱 motion canary。

## 30 Hz 结论

```text
capture fps = 30
released annotation stride = 30 source frames
released SMPL-X/object GT fps ≈ 1
```

当前 release 没有发现连续 30 Hz SMPL-X 或 object GT。`PARAM/person*.json`
和 `final/<object>.ply` 是每个 30skip sample 的单帧产物，不是包含 T 维的
轨迹文件。

决策：

- 设置 `blocked_temporal_density`；
- 稀疏数据只做 representation/layout probe；
- 不把 1 Hz linear/SLERP 插值标成 GT；
- 找到官方 dense GT 或经独立 dense 证据验证的 reconstruction 前，不开始
  production S1/S3–S6。

## 产物

```text
workspace/MMHOI/data_stat.md
workspace/MMHOI/plan/03_v1_c2_c8_box_global_adaptation_plan.md
workspace/MMHOI/log/02_v1_scope_and_temporal_audit.md
workspace/MMHOI/results/E002/s0_scope_inventory/
workspace/MMHOI/scripts/data_inventory/inventory_mmhoi.py
workspace/MMHOI/scripts/data_inventory/tests/test_inventory_mmhoi.py
```

## 验证

- 10 个标准库 `unittest` 通过；
- `py_compile` 通过；
- 完整 ZIP 的 8,071 个 action CSV 重新解析成功；
- E002 summary、TSV 和本文关键数字交叉一致。

## 下一步

1. E003：确认是否能取得未 skip 的官方 SMPL-X/object GT，或独立的 30 Hz
   RGB-D/视频源。
2. E004：不等待 dense source，先对 `C_2+box` 的 released sparse samples
   做 human-world 与 box 6DoF 全量 gate。
3. 只有 E003/E004 同时通过后，才选第一个 dense `C_2+box` motion canary
   进入 S1–S4。
