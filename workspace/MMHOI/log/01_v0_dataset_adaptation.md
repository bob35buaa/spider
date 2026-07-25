# R001 / E001：MMHOI 双人数据统计与适配设计

## 目标

1. 从 `experiment/E161-surface-release-ablation` 开新分支。
2. 初始化 `workspace/MMHOI/`。
3. 对 MMHOI Collaborative work 做可复跑的实际本地统计。
4. 调研 MMHOI → Core4D v3/OmniRetarget 的字段、坐标、物体和时间兼容性。
5. 输出阶段化全局适配方案，不把未执行的 S1–S6 写成已通过。

## 输入

```text
paper/Kogashi 等 - 2025 - MMHOI Modeling Complex 3D Multi-Human Multi-Object Interactions.pdf
/mnt/a0ccc676-9496-49f8-a861-f8a1797dec52/mocap_data/MMHOI
/mnt/a0ccc676-9496-49f8-a861-f8a1797dec52/mocap_data/MMHOI_release.zip
workspace/core4d/docs/data_construction_v3/
/home/ubuntu/Workspace/holosoma/workspace/pipeline/convert_core4d_to_omniretarget.py
```

Git：

```text
base branch: experiment/E161-surface-release-ablation
base commit: 67cef0b84d81107128a082f36804caa16a15251c
work branch: experiment/MMHOI-data-adaptation
Holosoma inspected commit: 04ca515d7030916cccc6966d04cb0b3b6577d18a
```

## 执行

Inventory：

```bash
python3 workspace/MMHOI/scripts/data_inventory/inventory_mmhoi.py \
  --archive /mnt/a0ccc676-9496-49f8-a861-f8a1797dec52/mocap_data/MMHOI_release.zip \
  --output-dir workspace/MMHOI/results/E001/s0_inventory
```

测试：

```bash
python3 -m unittest discover \
  -s workspace/MMHOI/scripts/data_inventory/tests \
  -p 'test_*.py' -v

python3 -m py_compile \
  workspace/MMHOI/scripts/data_inventory/inventory_mmhoi.py \
  workspace/MMHOI/scripts/data_inventory/tests/test_inventory_mmhoi.py
```

表示探针：

- 安全解析 binary/ascii PLY header 与 vertex payload；
- 对 template object mesh → per-frame final object mesh 做同序顶点 Kabsch；
- 对 camera-0 person mesh → final person mesh 做同序顶点 Kabsch；
- 读取 calibration JSON、`PARAM/person*.json` 与 Holosoma converter contract；
- 对每个 capture 的数字 frame folder 计算相邻 gap。

## 实际结果

### 数据量

| 指标 | 结果 |
|---|---:|
| Archive entries | 782,626 |
| 全数据 annotated samples | 8,071 |
| Collaborative work samples | 3,452 |
| 严格双人 samples | 2,821 |
| 严格双人 scenario captures | 48 |
| 三人 `C_9_r2` backlog | 631 |
| 严格双人 action rows / active rows | 16,892 / 9,347 |
| Active object types | 10 |
| 显式 together samples | 1,004 |
| Split unspecified | 86 |

严格双人 production scope 为 `C_2/C_8/C_9/C_10`。完整数字和口径见
[`../data_stat.md`](../data_stat.md)。

### 时间

严格双人共有 2,773 个 capture 内相邻间隔，全部为 30 个源 frame id；Collaborative work 全类别 3,397/3,397 也全部为 30。结合 30 fps 源采集，发布 GT 约为 1 Hz。

决策：设置 `blocked_temporal_density` 为 production hard blocker；稀疏数据仅允许格式/姿态诊断。

### 人体

`PARAM/person*.json` 有 `pose_53/pose_22/betas/betas_new/j3d_127/j3d_22`，但 j3d 位于 camera-0 坐标。样例 camera-0 person mesh 与 final person mesh 同拓扑，刚体配准 RMS 为 `3.9e-8/5.0e-8 m`，可将 `j3d_127` 映射到 final Y-up world。

决策：新增 MMHOI human-world adapter；`betas`/`betas_new` 先做 SMPL-X V2V 选择，不预设。

### 物体

10 类 Collaborative work 物体各抽一个 sample，template/final mesh vertex count 一致、scale≈1，刚体配准 RMS 为 `8.6e-9–6.8e-8 m`。

决策：从同序顶点恢复逐帧 6DoF，并全量审计残差、反射、scale 和 quaternion continuity；不需要依赖不存在的 object pose JSON。

### 下游 contract

现有 Holosoma converter 消费：

```text
joints(T,127,3) + betas(T,10)
-> global_joint_positions(T,22,3) + height

smooth_objposes(T,4,4)
-> object_poses(T,7), wxyz + xyz
```

MMHOI 可在世界坐标恢复后复用这个输出 contract，但 reader、时间 manifest、人体变换、物体配准和 multi-object window 必须新增。

## 失败与修复

| 失败 | 原因 | 处理 |
|---|---|---|
| `jq` 不可用 | 环境未安装 | 使用 Python 标准库/文本工具，不安装依赖 |
| `pytest` 不可用 | 环境未安装 | 测试改为标准库 `unittest` |
| 首次 inventory 遇到 split count overflow | archive 比 split 多 sample | archive authority，额外 sample 标 `unspecified` |
| 第二次 inventory 遇到 missing split key | `20240510_personH_personI/C_2` 无官方 key | 保留 60 个 sample，全部标 `unspecified` |
| 一次 PLY header 检查越过 `end_header` | binary payload 被打印 | 后续使用显式 header parser；未修改任何文件 |
| base Python 无 `trimesh` | 非必需依赖缺失 | 使用小型 PLY reader + NumPy Kabsch 完成只读探针 |

最终 10 个 `unittest` 和 `py_compile` 均通过。

## 产物

```text
workspace/MMHOI/data_stat.md
workspace/MMHOI/plan/01_v0_dataset_adaptation_plan.md
workspace/MMHOI/plan/02_v0_global_adaptation_plan.md
workspace/MMHOI/scripts/data_inventory/inventory_mmhoi.py
workspace/MMHOI/scripts/data_inventory/tests/test_inventory_mmhoi.py
workspace/MMHOI/results/E001/s0_inventory/
```

## 结论

R001/E001 已完成“统计 + 适配设计”。数据范围、人体/物体字段映射已有高置信路径；连续时间轴仍是明确的 P0 blocker。下一实验应为 E002 全量 representation audit 与 dense source availability 检查，而不是直接运行全量 OmniRetarget。
