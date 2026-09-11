# E214b 实验结果：Holosoma 三指标改用 CORE4D SMPLX 人体 GT 作为参考

**日期**: 2026-09-11
**实验域 (exp_name)**: `core4d`
**对应 Plan**: [`plan/246_E214b_holosoma_smplx_gt_reference_plan.md`](../plan/246_E214b_holosoma_smplx_gt_reference_plan.md)
**前置**: [`log/303_E214_core_method_four_ablation.md`](303_E214_core_method_four_ablation.md)（R300，E214 四消融）

## 1. 背景

E214 后为对标 holosoma 补了三个 holosoma 口径指标（foot sliding / penetration / contact
precision），实现在 `eval_E214_holosoma.py`。holosoma 原版这三个指标以 **SMPL/SMPLX 人体 demo** 为
参考（触地相来自人体脚趾速度；contact 用人体手 vs 物体）。此前移植暂用 **OmniRetarget 机器人
kin_ref（CEM 输入）** 当参考，衡量的是"CEM 是否保持 OmniRetarget 参考"，而非"vs 人体 GT"，与 holosoma
原义不符。本实验（纯评估口径变更，**不重跑 CEM**）把这三指标里**依赖参考的两项**（foot sliding 触地相、
contact precision demo 手/物）改回 **CORE4D SMPLX 人体 GT**（场景系）；penetration 不依赖参考，保持不变。

## 2. R301 (E214b): SMPLX-GT 参考对齐 + 口径切换

**变化**:
- 新增 `smplx_reference.py`：把 CORE4D raw SMPLX 关节/物体（Y-up，人体尺度）对齐到 rollout 场景系
  （Z-up，机器人尺度）。**以刚体物体平移**做含尺度 Umeyama + 暴力单调线性时间映射初始化 + ICP 式单调 DTW
  精修（吸收 retarget 的非均匀时间重采样）。参考 toe=SMPLX 10/11、wrist=真实 SMPLX 20/21、object=人体驱动物体位姿。
- 改 `eval_E214_holosoma.py` 参考侧：foot sliding 触地相由 SMPLX 脚趾场景系 xy 速度判定；contact
  precision demo 手 = SMPLX 手腕到参考物体表面。penetration **逐字不变**。每条记录加
  `smplx_gt_align_residual_m` / `smplx_gt_status` 审计字段。
- 改 `gen_E214_ablation_table.py`：foot sliding / contact precision 行标签补 "SMPLX-GT" + 覆盖率表注；
  短 key / `HOLOSOMA_FIELD` 不变。

### 运行指令

对齐验证（纯分析，rule 10b 例外，无需 scene 快照）：
```bash
.venv/bin/python workspace/core4d/scripts/eval/runners/validate_smplx_reference.py
```
重算 holosoma 指标（复用 E214 250 rollout，SMPLX 参考按 case 构建并缓存）：
```bash
.venv/bin/python workspace/core4d/scripts/eval/runners/eval_E214_holosoma.py --workers 8 --fresh
```
重生成报告（47/50-case × mean/±std 两版）：
```bash
EX="box024_20231011_026_p1,bucket003_20231020_068_p1,box021_20231011_037_p2"
.venv/bin/python workspace/core4d/scripts/eval/reports/gen_E214_ablation_table.py --exclude-cases "$EX" --tag exclude_floor3
.venv/bin/python workspace/core4d/scripts/eval/reports/gen_E214_ablation_table.py --exclude-cases "$EX" --tag exclude_floor3 --mean-only
.venv/bin/python workspace/core4d/scripts/eval/reports/gen_E214_ablation_table.py
.venv/bin/python workspace/core4d/scripts/eval/reports/gen_E214_ablation_table.py --mean-only
```

### 结果路径

| 类型 | 本地路径 |
|------|---------|
| SMPLX-GT 参考缓存（每 case） | `workspace/core4d/results/E214/eval/smplx_ref/*.npz` |
| 对齐验证 TSV | `workspace/core4d/results/E214/eval/smplx_reference_alignment.tsv` |
| holosoma 指标缓存 | `workspace/core4d/results/E214/eval/e214_holosoma.jsonl`（250/250 ok） |
| 报告（50-case） | `workspace/core4d/results/E214/reports/e214_ablation_table{,_mean}.md` |
| 报告（47-case，剔 3 floor-effect） | `workspace/core4d/results/E214/reports/exclude_floor3/` |

### 对齐验证（C1 关键证据）

| 指标 | 值（50 case） |
|------|--------------|
| 物体拟合残差（拟合目标） | mean **1.39mm** / median **0.00** / p95 9.9 / max 25.7mm |
| 通过对齐 gate(30mm) | **50/50**，NO_GT_ALIGN=0 |
| human→scene 尺度 s | ∈ [0.716, 0.784]（人体 ~1.8m → 机器人 ~1.3m） |
| pelvis 交叉核验（独立 sanity） | median 54mm / max 186mm |

> 关键坑：加"物体旋转 rigid-points"会污染拟合——raw CORE4D 与 MuJoCo 场景的 object **body 系差一个
> 常量 mesh 规范旋转**，故只用**物体平移**做锚点；参考物体位姿直接取 rollout 物体位姿（拟合后二者 ~0mm 重合）。

### 可视化（对齐目检，C1 补强）

叠加图 `workspace/core4d/results/E214/eval/smplx_alignment_overlay.png`（box001_039 + bucket007_075，full）。

**实际观察**：
- **xy 轨迹**（场景系）：SMPLX 脚趾（实线）与机器人踝（虚线）落在**同一空间区域、走向平行**——未翻转/未错缩放/未错转；
  两者存在恒定 ~10–15cm 偏移，属解剖学正常（SMPLX 脚趾关节在踝前方，且机器人足几何不同），不影响指标
  （foot sliding 用各侧自身速度/触地相，contact precision 用手腕非脚趾）。
- **脚趾高度**（场景 z）：稳定在 ~0.07–0.12m（脚趾关节略高于地面 z=0），无任何帧到达机器人骨盆高度或为负 →
  Y-up→Z-up 旋转与尺度正确。

### Overall holosoma 行（mean，47-case，列=full/A1−surf_band/A2−contact_hdmi/A3−hard_gate/A4−soft_pen）

| 指标 | full | A1 | **A2** | A3 | A4 |
|------|------|----|----|----|----|
| contact precision@10cm (SMPLX-GT) ↑ | **57.4%** | 65.6% | **33.5%** | 55.3% | 56.4% |
| contact precision@5cm (SMPLX-GT) ↑  | 38.2% | 41.7% | **33.5%** | 39.0% | 38.9% |
| foot sliding frac >5mm/f (SMPLX-GT) ↓ | 59.7% | 57.3% | 52.5% | 61.3% | 60.5% |
| penetration frac@5mm (holosoma) ↓ | 13.6% | **19.4%** | 10.4% | 15.7% | 14.3% |
| penetration depth max (holosoma, m) ↓ | 0.014 | 0.016 | 0.014 | 0.017 | 0.016 |

- **A2（去 contact_hdmi，够不到物体）** 的 contact precision@10cm 57.4%→**33.5%**：人体 GT 口径下清晰区分。
- A1（去 surface_band）penetration frac@5mm 13.6%→19.4%（穿透变差），与 E214 结论一致；其 contact 反而略高
  （A1 失效模式是穿透而非够不到）。penetration 各列与切换前**逐字一致**（下方 C3）。

## N. Claims 验证

| Claim | 结果 |
|-------|------|
| C1 SMPLX-GT 对齐正确 | **通过** — 物体残差 median 0.0/max 25.7mm，50/50 过 gate；scale∈[0.716,0.784]；pelvis 交叉核验 median 54mm |
| C2 指标语义正确切换 | **通过** — contact precision@10cm full 57.4% ≫ A2 33.5%（人体 GT 口径下 A2 够物失败可辨）；foot sliding 触地相非 0/非全 1（脚趾有意义着地段） |
| C3 隔离性 | **通过** — penetration 数值与切换前逐字一致（box001_039 full 5/10/20mm=0.08/0.01/0.0 max=0.012416 完全相同）；仅 foot sliding/contact precision 两项参考侧改变；其它 SPIDER 指标缓存与非 holosoma 报告行未触碰 |

## N+1. Git 提交

**当前分支**: `experiment/E199-omniretarget-object-augmentation`（大方向 `core4d`）

```bash
git add workspace/core4d/scripts/eval/runners/smplx_reference.py \
        workspace/core4d/scripts/eval/runners/validate_smplx_reference.py \
        workspace/core4d/scripts/eval/runners/eval_E214_holosoma.py \
        workspace/core4d/scripts/eval/reports/gen_E214_ablation_table.py \
        workspace/core4d/plan/246_E214b_holosoma_smplx_gt_reference_plan.md \
        workspace/core4d/log/304_E214b_holosoma_smplx_gt_reference.md \
        workspace/core4d/results/E214/reports workspace/core4d/EXPERIMENT_TRACKER.md \
        workspace/core4d/progress.md
git commit -m "exp(core4d): R301 E214b — holosoma foot-sliding/contact 改用 CORE4D SMPLX 人体 GT 参考（对齐 50/50）"
git push origin experiment/E199-omniretarget-object-augmentation
```

## N+2. 下一步

- 口径已对齐 holosoma 原义，可用于论文 holosoma 对标表。若需更严格的机器人脚"贴地点"，可把 robot 侧
  foot 由 `ankle_roll_link` 换成脚底 sphere/toe 几何（当前用踝关节，绝对滑移偏高但跨消融相对关系稳定）。
- 25.7mm 的最差对齐 case（box021_037_p2，本就在 3 个 floor-effect 剔除名单内）如需更低残差，可对个别 case
  引入按段线性时间映射；当前全过 30mm gate，不阻塞。
