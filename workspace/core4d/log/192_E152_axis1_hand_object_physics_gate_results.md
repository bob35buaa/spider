# E152 — 轴1：手-物体物理穿透硬约束（CEM safety gate 纳入手 geom）结果

> 计划：`workspace/core4d/plan/160_E152_axis1_hand_object_physics_gate_plan.md`
> 状态：**完成（12/12 行，6 复用 + 6 新跑全部产出，3-case 全齐）**
> box021 的 gateA/gateA_b1 由本地 GPU0 + 远程 RTX 6000 Ada 并行补跑完成（`run_E152_box021_recover.sh`）。

## 0. 一句话结论

3 case（box004、box021、box023）全跑齐：
- **gateA（轴1 单独，reward 不变）**：把手-物体**深穿透** `con_dist<−5mm` 帧占比平均压低 **−0.206**（box021 −0.240、box023 −0.308），近场接触基本不掉（near-5cm mean −0.009，在 −0.02 容忍内，box021 持平 0.000），不摔 3/3、obj_err 略好。但几何穿透 frac 仅 −0.043（**未达计划 ≥−0.10 阈值**），物理接触 frac 小幅降（−0.038）。→ **部分成立**：物理深穿透强降且不杀近场接触，几何穿透降幅与物理接触保持未完全达标。
- **gateA_b1（reward 吸引 B1 + 物理 gate 叠加）**：near-5cm 保持/超过 B1（mean +0.019），深穿透平均 −0.168，`success_pen_down_contact_keep` **2/3 pass**（box004、box023）。**box021 单独看更关键**：physC **+0.027（接触反升）**、深穿透 **−0.298**、near-5cm 持平、不摔——正是整条研究线追求的"接触↑且穿透↓"，仅因 geom-SDF<0 frac +0.013 微升被二值 success 判否。→ **成立（2/3 二值；3/3 在深穿透上一致强降，box021 实现接触升+穿透降）**。
- **核心收获**：E151 单 reward 做不到的"接触保持/上升 + 物理深穿透下降"，在 gateA_b1 上实现；自 E113 起 blocked 的 box021 首次出现"接触不降反升且深穿透腰斩、不塌陷"的合法搬运结果。

## 1. 完成度（12/12）

矩阵 = 3 case × 4 方法 = 12 行；6 复用（baseline=E147/E148 rubber、b1=E151 b1_mesh），6 新跑（gateA/gateA_b1）。

| case | baseline(复用) | gateA(新) | b1(复用) | gateA_b1(新) |
|---|---|---|---|---|
| box004_083_p2 | ✅ | ✅ remote | ✅ | ✅ remote |
| box023_person2 | ✅ | ✅ remote | ✅ | ✅ remote |
| box021_029_p2 | ✅ | ✅ 本地GPU0补跑 | ✅ | ✅ 远程GPU1补跑 |

- 补跑：box021 两行原本地 split 中断（仅 config_act.yaml）。用 `run_E152_box021_recover.sh full` 本地 GPU0 跑 gateA、远程 RTX 6000 Ada GPU1 跑 gateA_b1，并行约 13min 完成，远程结果已 pull。
- 最终 eval：`method_rows=12 / delta_rows=6 / missing=0`。

## 2. 量化结果（per-case delta + 3-case mean/std/worst）

> 用 per-case delta 表（`eval/full/e152_delta_vs_reference.tsv`）。深穿透 `con_dist<−5mm` 是计划新增、最贴物理的穿透指标。

### 2.1 gateA vs baseline（轴1 单独）

| case | near-5cm Δ | 几何穿透 Δ | 物理接触 Δ | **深穿透<−5mm Δ** | leg穿透 Δ | fall | fallback |
|---|---:|---:|---:|---:|---:|:--:|---:|
| box004_083_p2 | −0.019 | −0.038 | −0.010 | −0.071 | −0.048 | no | 0 |
| box021_029_p2 | **0.000** | −0.053 | −0.053 | **−0.240** | +0.013 | no | 0 |
| box023_person2 | −0.007 | −0.037 | −0.051 | **−0.308** | −0.007 | no | 0.052 |
| **mean** | **−0.009** | **−0.043** | **−0.038** | **−0.206** | −0.014 | 0/3 | |
| **worst** | −0.019 | −0.037 | −0.053 | −0.071 | +0.013 | | |

### 2.2 gateA_b1 vs b1（叠加：reward 吸引 + 物理阻挡）

| case | near-5cm Δ | 几何穿透 Δ | 物理接触 Δ | 深穿透<−5mm Δ | leg穿透 Δ | success | fall | fallback |
|---|---:|---:|---:|---:|---:|:--:|:--:|---:|
| box004_083_p2 | **+0.057** | −0.057 | −0.048 | −0.013 | −0.019 | ✅ | no | 0 |
| box021_029_p2 | 0.000 | +0.013 | **+0.027** | **−0.298** | −0.027 | ❌ | no | 0 |
| box023_person2 | 0.000 | **−0.096** | −0.074 | −0.192 | 0.000 | ✅ | no | 0.096 |
| **mean** | **+0.019** | **−0.047** | **−0.032** | **−0.168** | −0.015 | **2/3** | 0/3 | |
| **worst** | 0.000 | +0.013 | −0.074 | −0.013 | 0.000 | | | |

### 2.3 Gate 健康度（R1：是否塌缩到 fallback）

| 行 | gate_valid | hand_gate_valid | fallback | hand_gate_min_sdf_min(m) |
|---|---:|---:|---:|---:|
| box004 gateA / gateA_b1 | 0.902 / 0.835 | 0.910 / 0.843 | 0.000 / 0.000 | −0.005 / −0.008 |
| box021 gateA / gateA_b1 | 高 / 高 | — | 0.000 / 0.000 | — |
| box023 gateA / gateA_b1 | 0.839 / 0.769 | 0.839 / 0.769 | 0.052 / **0.096** | −0.017 / −0.018 |

→ gate 全程有效未名存实亡（fallback ≤0.096）；box023 gateA_b1 逼近 R1 边缘（valid 0.769 + fallback 0.096，阈值偏紧开始咬接触，与 box023 physC −0.074 一致）。box004/box021 fallback=0，gate 干净生效。

## 3. Claims 验证（计划 §4，3-case 完整）

| Claim | 判据 | 结果(n=3) | 裁定 |
|---|---|---|---|
| **主判据** gateA vs baseline | 几何穿透 ≥−0.10 且 near-5cm ≥−0.02 且 不摔 且 obj_err 不恶化 | 几何穿透 −0.043（**未达 −0.10**）；near-5cm −0.009 ✓；不摔 3/3 ✓；obj_err −0.0003 ✓；**深穿透 con<−5mm −0.206（强，box021 −0.24/box023 −0.31）**；physC −0.038 | **部分成立**：物理深穿透强降且近场接触不掉，但几何穿透 frac 降幅低于计划阈值、物理接触小幅降；`success` 0/3（二值严判） |
| **叠加判据** gateA_b1 vs b1 | 穿透↓ 且 near-5cm 保持 B1 增益 | near-5cm +0.019（保持/超过）✓；几何穿透 −0.047 ✓；深穿透 −0.168 ✓；`success` **2/3**；**box021 physC +0.027 且深穿透 −0.298** | **成立**：3/3 深穿透强降、近场接触不掉；box021 接触不降反升 |
| 3-case mean+std+worst（禁 cherry-pick） | 全 3 case 报统计 | 已补齐（见 §2），含 worst 行 | **达成** |

## 4. 可视化观察（skill §9 强制）

`visual_inspection/{case}_f{25,55,85}_baseline_gate_b1.jpg`（4 方法 ref/sim 并排）：
- **box021_029_p2 f55/f85**（历史易"趴箱"塌陷 case）：4 方法下机器人均**屈身站立、双手压箱顶、箱体竖直立于地面，无倒地、无趴箱、无悬浮**，与 fall_flag=false 一致；gateA/gateA_b1 接触姿态保持（与 near-5cm 持平、physC 不降一致）。深穿透 −0.24/−0.30 的改善在缩略图不可辨，由 `con_dist` 量化。
- **box004 f55、box023 f55**：弯腰/正面抱箱，箱体竖直，无 fall/穿模/悬浮。
- 满足最低视觉校验：3 case × 无 fall、无 gross 穿透/悬浮、箱体姿态正常。

## 5. 改动文件 / 结果路径

| 类型 | 路径 |
|---|---|
| 计划 | `workspace/core4d/plan/160_E152_axis1_hand_object_physics_gate_plan.md` |
| 结果根 | `workspace/core4d/results/E152/axis1_hand_object_physics_gate/` |
| CEM 产物(full) | `cem/full/E152_*_{gateA,gateA_b1}.{npz,_full.mp4}` + `_outdir_full/trajectory_mjwp_act.npz`（12/12 齐全） |
| Eval | `eval/full/e152_{summary.md,method_metrics.tsv,delta_vs_reference.tsv,eval_summary.json}`（missing=0） |
| 视觉 | `visual_inspection/{case}_f{25,55,85}_baseline_gate_b1.jpg`（9 张） |
| Scene 快照 | `results/E152/.../scene_snapshot/{3 case}/ + manifest.txt`（git HEAD `0adcb55` + sha256） |
| 代码增量 | `cem_hand_gate_*`（config.py / mjwp.py / sampling.py / sampling_fast.py，默认关，纯增量） |
| 脚本 | `scripts/E152/`、`scripts/train/train_E152_*.sh`、`scripts/{run_E152_local,run_E152_remote,pull_E152_remote_results,run_E152_box021_recover}.sh`、`scripts/eval/eval_E152_*` |

## 6. 下一步

承接 E152 结论（gate 强降物理深穿透、gateA_b1 在 box021 实现接触升+穿透降）：
1. **阈值扫（轴1 内）**：box023 gateA_b1 逼近 R1 边缘、box021 gateA_b1 几何穿透微升 +0.013，说明 `min_sdf=−0.010` 在不同 case 偏紧/偏松不一。按 case 或统一扫 `min_sdf ∈ {−0.005,−0.010,−0.015}` × `max_violation ∈ {0.05,0.10}`，找"压深穿但不咬接触/不抬几何穿透"的甜点，目标让 gateA_b1 三 case 都过二值 success。
2. **轴1 杠杆 B（solref 硬化，计划 §8 留作单独 E）**：穿透真因含软接触 `solref=[0.008,1]`。gate 是采样层堵、solref 是接触层硬化，两者正交可叠加，下一 E 单独硬化 pair solref 看能否进一步压几何穿透 frac（gate 没压住的那部分）。
3. **接 RL/Holosoma**：box021 gateA_b1 已得到"接触保持/上升 + 深穿透腰斩 + 不塌陷"的干净搬运轨迹，是比 E144/E145 更优的 RL 参考数据候选；E140 指出 Holosoma Box021 R135/R138 仍 `missing_ref_mask_reward`，可先补 Box021 ref-mask reward config variant 再做 ref_object_contact 衔接。

## 7. 记录口径声明（experiment.md §5）

- 报 per-case + 3-case mean + worst，无 cherry-pick；主判据按计划原文 `hand_geom_penetration_frac ≥ −0.10` 严判为"未达标"，同时并列报告更贴物理的 `con_dist<−5mm` 深穿透（强降）；二值 `success_pen_down_contact_keep` 与连续指标并报，不挑有利者下结论。
- box021 gateA_b1 二值 success=false（geom-SDF<0 frac +0.013）但物理深穿透 −0.298、physC +0.027，已如实标注"二值否、物理指标正向"的差异。
