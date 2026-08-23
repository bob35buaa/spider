# CORE4D 新数据分布统计 + 重定向管线适配性分析

- **日期**: 2026-08-23
- **作者**: 数据/管线调研
- **目的**: 评估两份新增数据 —— `CORE4D_Real_human_object_motions_v2`(简称 **v2**) 与 `CORE4D_Synthetic_V1`(简称 **Synthetic**) —— 相对已有 `CORE4D_Real`(简称 **v1**) 管线的分布差异与迁移成本，判断"能否快速搞一版新数据重定向"。
- **数据路径**:
  - v1: `.../CORE4D/CORE4D_Real/human_object_motions`
  - v2: `.../CORE4D/CORE4D_Real_human_object_motions_v2`
  - Syn: `.../CORE4D/CORE4D_Synthetic_V1/CORE4D_Synthetic_V1_extracted/CORE4D_Synthetic`

---

## 0. 结论先行 (TL;DR)

| 数据 | 规模 | 物体覆盖 | 格式与 v1 兼容性 | 迁移成本 | 建议 |
|------|------|---------|-----------------|---------|------|
| **v2** | 1007 case-dir / ≈1937 person 序列 (v1 的超集，+87 case) | **与 v1 同一批真实物体**(box/bucket/chair/desk/board/stick) | **result.npz 键结构与 v1 `arr_0` 完全一致**；仅加载路径不同，且**不含物体位姿**(需复用 v1) | **低**(≈20 行 loader 适配) | ✅ **可快速出一版**。先解压 40GB zip，改 converter 的加载分支即可，下游全链路零改动 |
| **Synthetic** | 2641 case | **仅 table(1682)+chair(959)**，603 个合成物体实例 | **格式不同**：人体只存 SMPLX**参数**(无 joints/vertices，需前向)、物体存 axis-angle+translation(非 4×4)、mesh 内联(无 obj_name/类别)、torch cuda tensor | **中-高**(需新 converter + 603 套新物体资产) | ⚠️ **不能直接快跑**。价值在 table/chair 多样性(现管线几乎没碰)，但要新写 SMPLX 前向 + 合成物体资产化 |

一句话:**v2 是"同物体、更精、可立即迁移"；Synthetic 是"新品类多样性、需新建管线分支"。**

---

## 1. 已有重定向管线(v1 漏斗)回顾

链路(核心代码位置):
```
CORE4D_Real v1 person{N}_poses.npz + smooth_objposes.npy
  └─(holosoma) convert_core4d_to_omniretarget.py   # Y-up→Z-up, joints[:22]+height, obj 4x4→quat_pos
      └─ OmniRetarget 运动学重定向 → qpos(T,43), fps
          └─(spider) spider/process_datasets/core4d.py  # FK 补 qvel/ctrl/contact → trajectory_kinematic.npz
              └─ examples/run_mjwp.py  MJWP 采样式 MPC / full-CEM 物理重定向
                  └─ E201 三级筛选漏斗(14 门宽/窄双口径 + VLM 初审)
                      └─ 人工终审 → RL export (paired source+partner)
                          └─ OmniRetarget 物体平移/旋转增强 (E199/E202)
```

**converter 的输入依赖(关键)**:
- `person{N}_poses.npz["arr_0"]` → dict，用到 **`joints`(T,127,3)** 与 **`betas`(T,10)**;取前 22 个 body joint,可选把 wrist(20/21) 替换为指尖中心,或追加指尖中心成 (T,24,3)。
- `smooth_objposes.npy`(T,4,4) → Y-up→Z-up → (T,7) `[qw,qx,qy,qz,x,y,z]`。
- `object_metadata.json` 的 `obj_name` + `object_models/{类别}/{name}_m.obj`(生成 URDF/资产)。
- SMPLX 模型目录(仅用于按 betas 估算身高)。

**v1 漏斗量级(已跑透的很窄)**:
- 源: 920 case-dir × 2 person ≈ **1840 序列**，**41 个真实物体 / 6 类**(见 §2)。
- spider 侧已建物体资产仅 **20 个**(box001/004/021-026、bucket001/003/004/005/007/009/010、chair022、desk001/005/007/021)。
- 实际深挖的几乎只有 **box + bucket**;chair/desk/board/stick 基本未做。
- RL export 历史产量(base case 级,非增强):E168 box021×13、E173×23、E178×13、E187×14、E190 38-case;经 OmniRetarget 增强放量到数百(E199 249 trans、E202 73 bucket)。
- **即 v1 的 6 大类里,真正被重定向+筛选落地的只有 box/bucket 两类的一小片。**

---

## 2. 三数据集分布统计

### 2.1 CORE4D_Real v1 (已有)
- 10 个日期文件夹,共 **920 个 case-dir**(每 case 有 person1+person2)。
- 各日期 case 数: `20231002:41, 20231003_1:47, 20231003_2:73, 20231008:70, 20231011:81, 20231018:113, 20231020:125, 20231023:139, 20231030:91, 20231108:140`。
- **41 个不同物体 / 6 类**(按 case-dir 计频次):
  - desk 204、bucket 197、box 192、chair 146、board 124、stick 57。
- 物体位姿: `smooth_objposes.npy` (T,4,4);人体: 每 person 一个 `arr_0` dict,**已预计算 joints(T,127,3)/vertices(T,10475,3)**。

### 2.2 CORE4D_Real_human_object_motions_v2 (新增①)
- 结构: `{date}/{seq}/person_{1,2}/result.npz`(+ 抽样的 SMPLX mesh/关节 ply,重定向不需要)。
- **1007 个 date/case-dir**(比 v1 多 87),**≈1937 个 person-`result.npz`**(batch1-4 分别 387/536/603/411)。
- 各日期 case 数: `20231002:59, 20231003_1:55, 20231003_2:82, 20231008:72, 20231011:118, 20231018:117, 20231020:137, 20231023:121, 20231030:101, 20231108:145`。
- **物体范围与 v1 完全相同**(同一批真实录制)。同 case 逐帧核实(`20231002/003` p1、`20231003_2/023`):**帧数完全一致**(234=234、128=128)、**betas 完全相同**(同一被试),差异仅在 SMPLX 位姿拟合被**重新优化**:transl max\|Δ\|≈2.2cm/mean 0.6cm、joints[:22] max\|Δ\|≈8.7cm/mean 0.9cm。→ **v2 = 同一批捕捉、同帧率/长度的更精 SMPLX 重拟合版**,非重新分段。
- ⚠️ **v2 的 case 目录里只有 person_1/person_2,没有物体位姿**。物体位姿需按 `date/seq` 回到 v1 的 `smooth_objposes.npy` + `object_metadata.json` 复用(命名可对齐)。
- ⚠️ **当前尚未解压**: 目录树是空占位,数据都在 `batch1-4.zip`(合计 ≈ 39.6 GB)。

### 2.3 CORE4D_Synthetic (新增②)
- **2641 个 case**,命名 `{id}_v{table|chair}{objid}`。
- **仅 2 类**: table 1682、chair 959;**603 个不同合成物体实例**(vtableN/vchairN)。
- 每 case: `human_poses.npy`、`object_poses.npy`、`object_mesh.obj`(内联)。
- 帧长(抽样 60 case): min 63 / max 300 / mean ≈ 161。
- 格式细节(与 v1/v2 均不同):
  - `human_poses.npy` = (T,) 的**逐帧 dict**{person1, person2},每个是 SMPLX **参数**(betas/expression/global_orient/transl/body_pose/left,right_hand_pose),shape 均 (1,·),**torch.float32 且在 cuda:0**;**无 joints/vertices**。
  - `object_poses.npy` = (T,) 的逐帧 dict{`rotation`(1,3) **axis-angle**, `translation`(1,3)},torch.float64@cuda。
  - `object_mesh.obj` 内联,**没有 obj_name 字符串/类别标签**,也不在 `object_models/` 里。

### 2.4 品类覆盖对照

| 类别 | v1 (case-dir) | v2 (≈同物体) | Synthetic | 现管线是否已深挖 |
|------|:---:|:---:|:---:|:---:|
| box  | 192 | ~同 | 0 | ✅ (主力) |
| bucket | 197 | ~同 | 0 | ✅ (主力) |
| desk/table | 204 | ~同 | **1682** | ❌ |
| chair | 146 | ~同 | **959** | ❌ |
| board | 124 | ~同 | 0 | ❌ |
| stick | 57 | ~同 | 0 | ❌ |

→ **Synthetic 的增量价值集中在 table/chair(现管线的空白区);v2 的增量价值是同物体上"更高质量人体 + 87 个额外 case"。**

---

## 3. 格式兼容性对照(迁移的技术核心)

| 维度 | v1 (基准) | v2 | Synthetic |
|------|-----------|----|-----------|
| 人体文件 | `person{N}_poses.npz["arr_0"]` | `person_{N}/result.npz` (直接加载,无 `arr_0`) | `human_poses.npy` 逐帧 dict |
| 人体内容 | **含 joints(T,127,3)+vertices** | **含 joints(T,127,3)+vertices**(键完全一致) | **仅 SMPLX 参数,无 joints** → 需前向 |
| 张量类型 | numpy | numpy | **torch@cuda**(需 `.cpu().numpy()`) |
| 布局 | T-major 堆叠 | T-major 堆叠(同 v1) | 逐帧 (1,·) → 需 stack |
| 物体位姿 | `smooth_objposes.npy` (T,4,4) | **无(复用 v1)** | `object_poses.npy` (T,) axis-angle+transl → 需转 4×4/quat |
| 物体网格 | `object_models/{类}/{name}_m.obj` | 同 v1 | **内联 `object_mesh.obj`,无命名/类别** |
| converter 改动 | — | **加载分支 ≈20 行** | **新 converter + 资产化管线** |

**converter 已为 v2 预留**: `convert_person(..., include_fingertip_centers=...)` 注释明确写 "for core4d_v2 format",且 holosoma 已有 `demo_data/core4d_v2/`、`workspace/v2/data/core4d_replace*` 的样例产物 —— 说明 v2 分支已被部分设计过,只差成规模跑通。

---

## 4. 适配性评估与迁移方案

### 4.1 v2 —— 低成本,建议立即做一版
**可行性: 高。下游(holosoma 重定向 → spider core4d.py → CEM → E201 漏斗 → 增强)零改动。**

步骤:
1. **解压** batch1-4.zip(≈40GB)到 case 目录(占位目录已在)。重定向只需 `result.npz`,可选择性只抽 `*/result.npz` 省空间。
2. **改 converter 加载分支**(`convert_core4d_to_omniretarget.py`):v2 从 `person_{N}/result.npz` 直接 `np.load` 取 dict(v1 是 `["arr_0"].item()`),其余逻辑(joints[:22]、betas→height、Y-up→Z-up)完全复用。
3. **物体位姿复用 v1**:按 `date/seq` 读取 v1 `smooth_objposes.npy` + `object_metadata.json`。已核实同 case **帧数逐帧对齐**,可直接一一对应,无需重采样(仅对 v2 新增的 ~87 个 v1 没有的 case 需确认 v1 是否有对应物体位姿)。
4. 用 v1 同一套 `object_models/` 资产,直接进 holosoma 重定向。

工作量: 半天级改动 + 解压/批处理机时。**产出**:同物体上更干净的一版重定向,可与 v1 A/B 对比人体运动质量,并扩充 ~87 个新 case。

✅ **时间对齐已核实**:同 case v1/v2 帧数一致(逐帧对应),物体位姿可直接复用,原先担心的对齐风险不成立。剩余小风险仅为 v2 新增 87 case 的物体位姿是否齐备。仍建议按 `.claude/rules/experiment.md` §5 在 1-2 个 case 上做 A/B 可视化,确认重拟合后的姿态质量确实优于 v1。

### 4.2 Synthetic —— 中高成本,价值在品类多样性
**可行性: 中-高。需新增 converter 分支 + 合成物体资产化;下游 CEM/漏斗可复用。**

需要新增的工作:
1. **人体 SMPLX 前向**:Synthetic 无 joints,须用 smplx 模型对逐帧参数做前向,得到 (T,127,3) joints(v1 converter 下游全靠 joints)。参数已是标准 SMPLX,前向是标准操作,可批量(注意 cuda→cpu、逐帧 dict → stack 成 T-major)。
2. **物体位姿转换**:axis-angle(T,3)+translation(T,3) → 4×4 或直接 → (T,7) quat_pos(复用 `transform_to_quat_pos`),再 Y-up→Z-up。**需确认 Synthetic 是否也是 Y-up**(CORE4D 真实数据是 Y-up;合成数据坐标系需在 1 个 case 上验证)。
3. **合成物体资产化(主要成本)**:603 个内联 `object_mesh.obj` 需走 `decompose_fast.py`(凸分解)→ `generate_xml.py`(scene XML/惯量/碰撞体),并建立命名(如 `vtable270`)。这是新品类进 MuJoCo 的固定开销,可脚本批量,但要按 `experiment.md` §7 做 scene XML 快照与 git 追踪。
4. 命名/元数据适配:用 `{id}_v{cat}{objid}` 作为 task 名,替代 v1 的 `{name}_person{N}`。

工作量: 数天级(新 converter + 资产批处理 + 抽样验证)。**建议先做 pilot**:选 5-10 个 table + 5-10 个 chair 打通端到端,验证合成人体/物体的物理可行性(合成运动可能有穿透/漂浮),再决定是否放量。

⚠️ **风险**:(a) 合成数据物理真实性未知,可能在 CEM/漏斗被大量筛掉;(b) 603 物体资产化是长尾开销;(c) 坐标系/单位需逐一验证。

---

## 5. 推荐执行顺序

1. **立即**: v2 pilot —— 挑 2 个已在 v1 跑过的 case(如 `20231002-003-bucket005`、`20231011-048-Box025`),用 v2 `result.npz` 重跑 converter→holosoma→spider,与 v1 结果**同 clip A/B 视频对比**,确认物体时间对齐无误。通过后按需放量(仍聚焦 box/bucket,或借机补 chair/desk)。
2. **其次**: Synthetic chair/table pilot —— 新 converter(SMPLX 前向 + axis-angle 物体)先在单 case 打通并可视化验证坐标系与物理合理性;跑通后再评估 603 物体资产化是否值得放量。
3. 两者都遵循 `experiment.md`:结构化输出路径、scene XML 快照 + git 追踪、强制视觉评估、明确 pilot 成功/失败量化门。

---

## 6. 复现命令(本报告统计所用)

```bash
PY=/mnt/ali-sh-1/usr/xiayibo/work_dir/embodied/spider/.venv/bin/python
# v1 分布: 遍历 human_object_motions/*/*/object_metadata.json 统计 obj_name
# v2 case 数: unzip -l batch{1..4}.zip | grep -oE '^[0-9]{8}[^/]*/[0-9]+/' | sort -u | wc -l
#            result.npz 计数: unzip -l batchN.zip | grep -c result.npz
# Synthetic: ls CORE4D_Synthetic | 正则 {id}_v{table|chair}{objid} 计数
# 格式: np.load(..., allow_pickle=True) 打印各 npz/npy 的键与 shape/dtype
```
（v2 探针文件已从 analysis 目录清理。）

---

## 附:关键事实核验清单

- [x] v2 `result.npz` 键 = v1 `arr_0` 键(vertices/joints/betas/expression/global_orient/transl/body_pose/left,right_hand_pose),布局同为 T-major。
- [x] v2 case 目录**无物体位姿**,需复用 v1。
- [x] v2 尚未解压(数据在 4 个 batch zip,≈39.6GB)。
- [x] Synthetic 人体**无 joints**,仅 SMPLX 参数(torch@cuda),物体为 axis-angle+translation,mesh 内联。
- [x] Synthetic 仅 table/chair;603 物体实例。
- [x] v1 `object_models/`:box 11 / bucket 10 / chair 12 / desk 10 / board 6 / stick 8。
- [x] converter 已为 core4d_v2 预留 `include_fingertip_centers` 分支 + holosoma 已有 v2 样例产物。
