# MMHOI `C_2/C_8`（Moving heavy stuffs）数据统计

> 审计日期：2026-07-25
>
> 当前主范围：仅 `C_2`、`C_8`
>
> 首批实验：`C_2` 中 `box` 至少有一条 active action 的样本
>
> 数据权威：本地完整发布包 `MMHOI_release.zip`

## 结论摘要

| 指标 | `C_2/C_8` 主范围 | 首批 `C_2 + active box` |
|---|---:|---:|
| Scenario captures | 24 | 12 |
| Sequence roots | 14 | 12 |
| Annotated sample folders | 1,289 | 460 |
| Train / val / test / unspecified | 602 / 308 / 314 / 65 | 223 / 79 / 116 / 42 |
| Person-object action rows | 7,700 | 3,680（所选帧内全部物体） |
| Active action rows | 5,502 | 2,966（所选帧内全部物体） |
| Active object types | 6 | 4（同帧可含其他物体） |
| 显式双人 `* together` samples | 1,004 | 365 |

主范围的 6 类物体为：

```text
box
stool
suitcase_large
suitcase_small
chair_wood
table_wood
```

最关键结论是：论文中的 **30 fps 是 Kinect 采集帧率**，不是当前公开
release 中 SMPL-X/object GT 的有效帧率。完整 ZIP 内 `C_2/C_8` 的 1,289
个 GT sample 相邻 frame id 全部间隔 30，目录明确带 `30skip/30_skip`；
按 30 fps 源时钟换算，公开 GT 约为 **1 Hz**。当前 release 中没有找到未
skip 的 30 Hz 人体或物体连续轨迹，因此 production 重定向仍被
`blocked_temporal_density` 阻塞。

## 统计口径

| 名称 | 定义 |
|---|---|
| Sequence root | 一次人员组合/采集日的 `MMHOI/sequences/<sequence>/` |
| Scenario capture | 一个 sequence 下的 `C_2` 或 `C_8` 场景目录 |
| Annotated sample | 一个带 `PARAM/action.csv` 的数字 frame folder |
| Action row | `PARAM/action.csv` 中一条 person-object-verb 标注 |
| Active action | verb 不等于 `no-interaction` |
| Cooperative sample | 同一物体、同一 `* together` verb 至少出现两名 person |
| 首批 box sample | `C_2` 中 box 至少有一条 active action row 的 sample |

注意：

- 这里的 sample 是发布包中的稀疏 annotation folder，不是论文约 60 万
  source frames。
- 首批的 460 个 box sample 不是一条可直接拼接的连续轨迹；它们分布在
  12 个 capture 中，并且仍是 30skip。后续必须从完整 capture 上构建连续
  window，不能把筛出的 active 帧首尾相接。
- 官方 split 只负责 train/val/test 分配；archive 中超出 split 声明或缺
  split key 的样本保留为 `unspecified`，不静默丢弃或并入训练。

## 场景与数量

| Code | 场景名 | Captures | Samples | Train | Val | Test | Unspecified | Active objects |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `C_2` | Moving heavy stuffs 1 | 12 | 636 | 304 | 132 | 135 | 65 | box, stool, suitcase_large, suitcase_small |
| `C_8` | Moving heavy stuffs 2 | 12 | 653 | 298 | 176 | 179 | 0 | chair_wood, table_wood |
| **合计** |  | **24** | **1,289** | **602** | **308** | **314** | **65** | **6 类** |

主范围涉及 14 个 sequence roots、12 个可从目录名恢复的参与者身份：

```text
20240412_personA_personB
20240412_personA_personC
20240417_personM_personB_finish
20240417_personM_personC_to3C_finish
20240418_personB_personC_noC1andC2_all_30_skip_start-end
20240418_personM_personC_from4C_all_30skip_start-end
20240423_personB_personC_and_C2_all__start-end
20240423_personM_personC_30_skip_all_start-end
20240424_personB_personD_all_start-end
20240426_personB_personG_all_30skip_start-end
20240426_personL_personC_all_30skip_start-end
20240508_personE_personD_all_30skip_start-end
20240508_personJ_personK_30_skip_start-end
20240510_personH_personI
```

身份集合为 `A/B/C/D/E/G/H/I/J/K/L/M`。

## 动作与物体

主范围有 6 个原始 verb 字符串，其中 5 个是 active：

| 场景 | Raw verbs |
|---|---|
| `C_2` | move together, no-interaction, pick up, stack, stack together |
| `C_8` | hold, move together, no-interaction, stack together |

按原始 `(object, verb)` 统计共有 26 类，其中 active 类 20。完整 class
明细见
[`primary_action_classes.tsv`](results/E002/s0_scope_inventory/primary_action_classes.tsv)。

| Object | Action rows | Active rows | `* together` rows | Samples | Scenario |
|---|---:|---:|---:|---:|---|
| box | 1,272 | 833 | 730 | 636 | `C_2` |
| stool | 1,272 | 810 | 694 | 636 | `C_2` |
| suitcase_large | 1,272 | 832 | 716 | 636 | `C_2` |
| suitcase_small | 1,272 | 772 | 660 | 636 | `C_2` |
| chair_wood | 1,306 | 1,262 | 1,240 | 653 | `C_8` |
| table_wood | 1,306 | 993 | 948 | 653 | `C_8` |

每个 sample 会同时列出场景中的多个物体，因此 object sample 数不能相加
得到总 sample 数。

## 首批 `C_2 + box`

首批筛选规则是：

```text
scenario == C_2
and exists(action.object == box and action.verb != no-interaction)
```

| 指标 | 数量 |
|---|---:|
| `C_2` 全部 samples | 636 |
| 含 active box 的 samples | 460 |
| 覆盖 captures / sequence roots | 12 / 12 |
| Train / val / test / unspecified | 223 / 79 / 116 / 42 |
| Box action rows / active rows | 1,272 / 833 |
| Box `* together` action rows | 730 |
| Box cooperative samples | 365 |

Box verb 分布：

| Verb | Action rows | Samples |
|---|---:|---:|
| move together | 686 | 343 |
| no-interaction | 439 | 263 |
| pick up | 99 | 91 |
| stack | 4 | 4 |
| stack together | 44 | 22 |

`no-interaction` 的 263 个 sample 与 active sample 可以重叠，例如一人
`pick up`、另一人 `no-interaction`，所以各 verb 的 sample 数不能直接求和。

首批机器清单见
[`pilot_c2_box_sample_inventory.tsv`](results/E002/s0_scope_inventory/pilot_c2_box_sample_inventory.tsv)。
这个清单用于定位 box 活动区间；真正的轨迹输入必须保留活动前后的完整连续
上下文，并在 dense 30 Hz source 上重新切 window。

## 30 Hz 连续轨迹审计

### 论文声明与 release 实数

[MMHOI 论文](../../paper/Kogashi%20等%20-%202025%20-%20MMHOI%20Modeling%20Complex%203D%20Multi-Human%20Multi-Object%20Interactions.pdf)
声明四台 Azure Kinect 以 30 fps 捕获，并描述了逐 frame annotation。该表述
说明采集系统的源时间基准，但不能单独证明公开 ZIP 带有逐源帧 GT。

对完整 `93,682,766,723` bytes ZIP 做 central-directory 和路径审计：

| `C_2/C_8` release 证据 | 数量 |
|---|---:|
| Scenario captures | 24 |
| 数字 frame folders / `PARAM/action.csv` | 1,289 / 1,289 |
| 同时含 `PARAM/person1.json` 与 `person2.json` 的 frame folders | 1,289 |
| 含 final object mesh 的 frame folders | 1,289 |
| Capture 内相邻 frame-id transitions | 1,265 |
| Gap distribution | `30: 1,265` |
| 明确带 `30skip/30_skip` 的 captures | 24 / 24 |
| 数字帧目录外的 scenario 文件 | 24 个 `mask_exist_all.csv` |

在整个 archive 中没有发现：

```text
.npy / .npz
.pkl / .pickle
.mp4 / .avi / .mov / .mkv
.bvh / .c3d
```

唯一内嵌 ZIP 位于无关的 `C_6` sample，未压缩大小仅 22 bytes，是空 ZIP，
不是 dense 轨迹容器。

因此当前证据链是：

```text
Kinect capture clock = 30 fps
release annotation stride = 30 source frames
released SMPL-X/object GT rate ≈ 30 / 30 = 1 Hz
```

### 临时解压 case 实证

为排除 central-directory 路径统计的误判，临时解压了：

```text
MMHOI/sequences/20240412_personA_personB/20240412__C_2__30skip
```

临时目录约 261 MB，包含 12 个数字帧：

```text
00361 00391 00421 00451 00481 00511
00541 00571 00601 00631 00661 00691
```

相邻 gap 全为 30。逐帧检查结果：

- 每个数字帧恰有一份 `PARAM/person1.json` 和 `person2.json`；
- 每个数字帧有一份 `final/box.ply`；
- 每个数字帧有两份 `Mesh_SMPLH/person*.obj`；
- `mask_exist_all.csv` 有 `12 × 4 = 48` 行，只引用上述 12 个 frame id；
- 数字帧目录外只有 `mask_exist_all.csv`；
- case 外只额外解压了静态 `object/03_box.ply` 和四相机 calibration；
- 没有任何视频、mocap、NumPy/Pickle 或独立 object-pose track。

`PARAM/person*.json` 是单帧记录，顶层字段为
`betas/betas_new/pose_22/j2d_22/j3d_22/pose_53/j2d_127/j3d_127`，没有
时间数组。12 份人体 JSON 各自不同；12 份 `final/box.ply` 只有 3 个不同
hash，说明 box 在多个稀疏标注点保持静止或重复位姿，但仍不提供两点之间的
30 Hz 轨迹。

这个 case 在 12 个稀疏帧里只有 `00541` 出现 active box，适合验证文件
结构，不适合作为首个协作搬箱 motion canary。

### 对适配链路的约束

- 不得把 `fps=30` 写入这些 30skip GT 并据此计算 qvel、接触持续时间或
  CEM/RL reference。
- 不得把 1 Hz 线性/SLERP 插值结果标记为 GT 或 production 默认。
- 稀疏 GT 仍可用于人体世界坐标、SMPL-X 参数、物体 template→final 6DoF
  和单帧接触的格式探针。
- S1 连续 contact、S3 OmniRetarget、S4 target gate 和 S5/S6 handoff 必须
  等待独立的 dense source，或等待时间重建 variant 通过独立 30 Hz 证据
  验证。

## 人体与物体表示

### 人体

`PARAM/person*.json` 的字段形状为：

```text
betas       (10,)
betas_new   (10,)
pose_22     (66,)
pose_53     (159,)
j2d_22      (44,)
j2d_127     (254,)
j3d_22      (66,)
j3d_127     (381,)
```

`j3d_127` 与 camera-0 person mesh 位于同一坐标。camera-0 person mesh 与
`final/person*.ply` 同拓扑，已有样例 Kabsch 残差约 `4e-8–5e-8 m`，可
恢复每帧人体到 final Y-up world 的刚体变换。`betas` 与 `betas_new` 的选择
仍需 SMPL-X 重建 V2V/overlay gate。

### 物体

发布包没有独立 object-pose JSON。静态 template
`object/<id>_<name>.ply` 与 `final/<name>.ply` 保持逐顶点对应；已抽样验证
刚体 Kabsch 的 scale≈1、RMS 为 `8.6e-9–6.8e-8 m`，因此可恢复每个**已
发布稀疏帧**的 6DoF。这个结论不补足帧间 30 Hz 轨迹。

## 数据源与 split 风险

统计权威：

```text
/mnt/a0ccc676-9496-49f8-a861-f8a1797dec52/mocap_data/MMHOI_release.zip
```

用户正在解压的目录：

```text
/mnt/a0ccc676-9496-49f8-a861-f8a1797dec52/mocap_data/MMHOI
```

解压目录可用于 case 级检查；在解压完成前，总量继续以完整 ZIP 为权威。

`C_2/C_8` 的 65 个 `unspecified` 均来自 `C_2`：

- `20240424_personB_personD_all_start-end/C_2` 多于 split 声明 5 帧；
- `20240510_personH_personI/C_2` 缺官方 split key，共 60 帧。

这些样本默认不进入训练或正式 test，只允许用于可视化、数据修复和无 split
依赖的诊断。

## 可复跑产物

命令：

```bash
python3 workspace/MMHOI/scripts/data_inventory/inventory_mmhoi.py \
  --archive /mnt/a0ccc676-9496-49f8-a861-f8a1797dec52/mocap_data/MMHOI_release.zip \
  --output-dir workspace/MMHOI/results/E002/s0_scope_inventory
```

核心输出：

- [`inventory_summary.json`](results/E002/s0_scope_inventory/inventory_summary.json)
- [`primary_sample_inventory.tsv`](results/E002/s0_scope_inventory/primary_sample_inventory.tsv)
- [`primary_scenario_inventory.tsv`](results/E002/s0_scope_inventory/primary_scenario_inventory.tsv)
- [`primary_samples_by_scenario_split.tsv`](results/E002/s0_scope_inventory/primary_samples_by_scenario_split.tsv)
- [`primary_action_classes.tsv`](results/E002/s0_scope_inventory/primary_action_classes.tsv)
- [`primary_objects.tsv`](results/E002/s0_scope_inventory/primary_objects.tsv)
- [`pilot_c2_box_sample_inventory.tsv`](results/E002/s0_scope_inventory/pilot_c2_box_sample_inventory.tsv)
- [`pilot_c2_box_action_classes.tsv`](results/E002/s0_scope_inventory/pilot_c2_box_action_classes.tsv)
- [`split_mismatches.tsv`](results/E002/s0_scope_inventory/split_mismatches.tsv)

## 当前可用性

| 能力 | 状态 |
|---|---|
| `C_2/C_8` 主范围 inventory | 已完成 |
| `C_2 + active box` 首批清单 | 已完成 |
| 稀疏人体 world transform 路径 | 已验证样例，待全量 gate |
| 稀疏物体 6DoF 恢复路径 | 已验证样例，待全量 gate |
| 公开 release 的 30 Hz SMPL-X GT | **不存在/未发现** |
| 公开 release 的 30 Hz object GT | **不存在/未发现** |
| 30 Hz production 轨迹 | **阻塞** |
| S5/S6 production handoff | **不得开始** |
