# 07 故障排查

## 缺 raw path

症状：

```text
Missing CORE4D_RAW_ROOT
Missing human_object_motions
Missing object_models
```

处理：

- 设置 `CORE4D_RAW_ROOT=/abs/path/to/CORE4D_Real`；
- 确认 `human_object_motions` 和 `object_models` 存在；
- 不要改脚本硬编码路径。

## source scene missing

处理：

- 不要跳过；
- 不要直接把 case reject；
- 写入 template backlog；
- 生成 source template 并通过 audit 后再进入 S3。

## CVXPY infeasible

含义：

- Stage2b / OmniRetarget preprocess 失败；
- 不是 raw contact 失败；
- 不能进入 CEM；
- registry 记录为 `stage2b_status=omniretarget_infeasible`。

处理：

- 保留 converted/input evidence；
- 不伪造 retargeted/trimmed/trajectory；
- 可尝试其它 retarget variant，但必须分叉记录。

## scene load fail

检查：

- XML 路径；
- mesh 路径；
- object body 名称；
- `nq/nv/nu`；
- hand contact sites；
- inertial/collision extents。

## `qvel is not a file in the archive`

含义：

- target reconstruction 只复制了 `qpos`，没有保留完整 `trajectory_kinematic.npz` arrays。

处理：

- 重建时复制 `qpos/qvel/ctrl/contact/contact_pos`；
- 再生成 `scene_act.xml`；
- 重新 verify。

## `contact_pos` 语义混淆

`contact_pos` 是 G1 FK palm site，不是 raw mocap fingertip。

处理：

- raw fingertip 指标必须来自 raw CORE4D；
- FK palm 指标命名为 `fk_palm_site`；
- external target 指标命名为 `external_target`。

## 旧 results 混入

处理：

- 不从 legacy 目录隐式读取；
- 需要旧数据时显式 import；
- import 后写入 `source_type=legacy_import`；
- 不把旧 polluted-template label 当 hard label。
