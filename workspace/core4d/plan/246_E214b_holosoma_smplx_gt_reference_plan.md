# R301 (E214b) 实验计划：Holosoma 三指标改用 CORE4D SMPLX 人体 GT 作为参考

## Context

E214（R300, [log 303](../log/303_E214_core_method_four_ablation.md)）完成核心方法四消融后，
后续为对标 holosoma 补充了三个 holosoma 口径指标（foot sliding、penetration、contact precision），
移植自 `Opensource_projects/holosoma/.../evaluation/eval_retargeting.py`，实现在
`workspace/core4d/scripts/eval/runners/eval_E214_holosoma.py`。

### 根因分析

holosoma 原版这三个指标用 **SMPL/SMPLX 人体 demo 作为参考/GT**：
- foot sliding 的"触地相"来自**人体脚趾**逐帧 xy 速度阈值；
- contact precision 用**人体手**相对物体判断接触。

当前移植因手头只有机器人 kin_ref，暂用 **OmniRetarget 机器人 kin_ref（CEM 输入）** 当参考，
衡量的是"CEM 是否保持了 OmniRetarget 参考"，**不是**"vs 人体 GT"。这与 holosoma 原义不符，
论文对标口径需要人体 GT。

### 关键 insight

本项目管线为 **CORE4D SMPLX → OmniRetarget → SPIDER-CEM**，CORE4D raw 里就有 SMPLX 人体关节。
把 holosoma 三指标（仅这三个）的参考改回 **CORE4D SMPLX 人体 GT**，即对齐 holosoma 原义。
penetration 不依赖参考、保持不变。其余 SPIDER 指标与 E214 结论不动。

关键数据与事实（探索已确认）：
- Raw：`/mnt/ali-sh-1/usr/xiayibo/xyb_data_tidal_alsh/other-datasets/mocap_data/CORE4D/CORE4D_Real/human_object_motions/<date>/<seq>/`
  - `person{1,2}_poses.npz`→`arr_0` dict→`joints (T,127,3)` 原生 SMPLX 顺序（toe=10/11，wrist=20/21）。**50 例全有。**
  - `smooth_objposes.npy (T,4,4)` 物体世界位姿，与 joints 同 raw 时间轴。
- s3 converted 中间件（raw 的确定性变换，时间轴与 raw 1:1）：
  `.../s3_retarget/*/*/results/*/holosoma_*<case>/converted/<date>-<seq>-person#-<obj>_with_obj.npz`
  → `global_joint_positions (T,22,3)`（已在**场景系**的 SMPLX body 关节）、`object_poses (T,7)`、`height`。
  约 ≥31/50 存在（E206 覆盖较广）；此处手腕**可能被 fingertip 均值替换**。
- 裁剪是对未裁剪 retarget 的**连续切片**，但未裁剪 retarget 是 raw 重采样→ rollout(kin_ref)=100 帧 vs raw 116 帧，需帧映射。
- raw→场景是**每 case 相似变换**（旋转+平移+可能 height 缩放），非固定坐标轴翻转（仅轴置换残差 0.38m）。
- contact precision 的 `‖·‖` 旋转不变 → 手-物距离与坐标系无关；但 foot sliding 用水平(xy)速度，依赖上轴 → 必须在场景系算。

### 已确认决策（用户）
- 参考 = CORE4D **SMPLX 人体 GT**（raw），仅 holosoma 三指标改。
- contact precision 的"手" = **真实 SMPLX 手腕**（joint 20/21），未修改（不用被 fingertip 替换的 converted 手腕）。
- 优先使用 **s3 converted 中间件**（能用则用）。

## Claims

| Claim | 最低证据 |
|-------|---------|
| C1 SMPLX-GT 对齐正确 | 每 case 物体匹配残差 `align_residual_m` < ~2cm；有 converted 的 case，`T·raw 非手腕关节` vs converted 非手腕关节 mm 级吻合（报告最大残差）；覆盖率≥目标并如实列出 `NO_GT_ALIGN` |
| C2 指标语义正确切换 | contact precision（SMPLX-GT）能区分 A2（够物失败）显著低于 full；foot sliding 触地帧占比合理（脚有意义着地一段），非 0/非全 1 |
| C3 隔离性 | 既有 SPIDER 指标与非 holosoma 报告行**逐字不变**；仅 holosoma 三指标行改变；penetration 数值不变 |

## 改动

### 1. 新增 SMPLX-GT 参考构建器

**文件**: `workspace/core4d/scripts/eval/runners/smplx_reference.py`

`build_smplx_reference(case_id, kin_ref_path)` 返回与 **rollout/kin_ref 帧 1:1 对齐**的
`toe_scene (N,2,3)`、`wrist_scene (N,2,3)`、`obj_pos/obj_quat (N,…)`（场景系）、`align_residual_m`、`status`。

```python
# 步骤
# 1) case token -> raw 目录(<date>/<seq>, person=person_idx_from_case)；读 raw joints(127) + smooth_objposes
# 2) 求 raw->场景相似变换 T(R,t,s):
#    - 有 converted _with_obj.npz(与 raw 同 116 帧 1:1): 含尺度 Umeyama 拟合 raw->converted(物体平移/非手腕关节)，校验残差
#    - 手腕取自 raw 并经 T 变换 -> 未修改的真实 SMPLX 手腕(即使 converted 手腕被 fingertip 替换)
#    - converted 定位: 跨 results/E*/s3_retarget/*/*/results/*/holosoma_*/converted/ (+ holosoma workspace) 建 token 索引，缓存小 json
#    - 兜底(任何实验都无 converted): raw 物体轨迹 vs kin_ref 物体轨迹用变换不变运动特征匹配后 Umeyama；残差过大 -> status=NO_GT_ALIGN
# 3) 帧映射 raw->rollout: T·raw 物体轨迹 与 kin_ref[:,36:43] 最近-单调匹配，逐帧校验残差；切 toe/wrist/object -> N 帧
```

复用：`core_metrics.signed_point_box / object_collision_geoms / point_object_sdf / npz_qpos / person_idx_from_case`；kin_ref 路径解析复用 `gen_paper_results`。

### 2. 改写 `eval_E214_holosoma.py` 参考侧

**文件**: `workspace/core4d/scripts/eval/runners/eval_E214_holosoma.py`

- `_foot_sliding`：触地相改由 `toe_scene`（SMPLX 脚趾）逐帧 xy 速度 ≤ `STICK_THRESHOLD` 判定；机器人脚趾仍用 rollout 踝。保留扫阈值（>5/10/20 mm/帧 + 均值）。
- `_contact_precision`：demo 手 = `wrist_scene`（SMPLX 手腕）到**参考物体**表面（`ref_object_sdf`，盒在参考位姿）；robot 手 = rollout 手腕到 rollout 物体表面。保留扫阈值 {2,5,10cm} 与 holosoma miss 公式。
- `_penetration`：**不变**。
- 每条记录加 `smplx_gt_align_residual_m`、`smplx_gt_status`。

### 3. 报告标签

**文件**: `workspace/core4d/scripts/eval/reports/gen_E214_ablation_table.py`

holosoma 行标签补 "SMPLX GT" + 表注；短 key / `HOLOSOMA_FIELD` 不变。报告加一行覆盖率（多少 /50 通过对齐，列出 `NO_GT_ALIGN`）。

## 需要修改的文件

| # | 文件 | 改动 |
|---|------|------|
| 1 | `workspace/core4d/scripts/eval/runners/smplx_reference.py` | 新增：构建器 + converted token 索引缓存 + 含尺度 Umeyama + 帧匹配 + 残差校验 |
| 2 | `workspace/core4d/scripts/eval/runners/eval_E214_holosoma.py` | 参考侧改用 SMPLX-GT（foot sliding 触地相、contact precision demo 手/物）；penetration 保留；加 CORE4D-raw 根常量与审计字段 |
| 3 | `workspace/core4d/scripts/eval/reports/gen_E214_ablation_table.py` | holosoma 行标签补 "SMPLX GT" + 表注 + 覆盖率行；不改 key |

## 训练命令

无 CEM 重训（复用 E214 200 rollout + 7 box023 + 43 full）。仅重算 holosoma 指标：

```bash
.venv/bin/python workspace/core4d/scripts/eval/runners/eval_E214_holosoma.py --workers 8 --fresh
# 重生成报告(47/50-case, mean 版)
EX="box024_20231011_026_p1,bucket003_20231020_068_p1,box021_20231011_037_p2"
.venv/bin/python workspace/core4d/scripts/eval/reports/gen_E214_ablation_table.py --exclude-cases "$EX" --tag exclude_floor3
.venv/bin/python workspace/core4d/scripts/eval/reports/gen_E214_ablation_table.py --exclude-cases "$EX" --tag exclude_floor3 --mean-only
.venv/bin/python workspace/core4d/scripts/eval/reports/gen_E214_ablation_table.py
.venv/bin/python workspace/core4d/scripts/eval/reports/gen_E214_ablation_table.py --mean-only
```

## 成功标准

| 指标 | 前次（kin_ref 参考） | 本次目标（SMPLX-GT，R301） |
|------|----------------------|----------------------------|
| SMPLX-GT 对齐残差 | — | **每 case < ~2cm；覆盖率 /50 明确，NO_GT_ALIGN 列出** |
| converted 交叉核验（非手腕关节） | — | **mm 级吻合** |
| contact precision A2 vs full | 区分（kin_ref 口径） | **人体 GT 口径下 A2 仍显著低于 full** |
| 隔离性 | — | **SPIDER 指标与非 holosoma 行逐字不变；penetration 不变** |
| 全量前冒烟 | — | **2–3 case（box + bucket/desk）扫阈值单调、有限** |

## 复现性 / 边界
- 只读 CORE4D raw 与 s3 converted 中间件；不改任何 raw/scene/kin_ref。
- converted 索引缓存与 `e214_holosoma.jsonl` 均在 `workspace/core4d/results/E214/eval/`（results 不进 git）。
- 纯评估脚本、不跑物理仿真 → 依 rule 10b 例外，无需 scene 快照。
- 结论若 claims 通过：写 `log/` 记录 + 更新 tracker（R301 一行 + 本 plan 链接）。
