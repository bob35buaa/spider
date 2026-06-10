# E153 — CEM hand gate 阈值扫（min_sdf × max_violation，先解耦再扫）结果

> 计划：`workspace/core4d/plan/161_E153_gate_threshold_sweep_plan.md`
> 状态：**完成（Stage0 解耦 + 18 grid full CEM 全跑齐，eval missing=0）**
> 评测：`scripts/eval/eval_E153_gate_threshold_sweep.py`（遵循 SKILL §13，`from lib.core_metrics import ...`，无 importlib 动态加载）

## 0. 一句话结论

3 case × {min_sdf∈[−0.005,−0.010,−0.015]} × {max_viol∈[0.05,0.10]} = 18 组 gateA_b1 full CEM（vs b1 reward-only 参考）：
- **找到 3/3 strict 甜点 `(min_sdf=−0.010, max_viol=0.10)`**：深度感知 success 3/3、深穿透 `con<−5mm` 3-case mean **−0.172**、真穿透 pen2mm −0.107，**接触几乎零损失**（physC −0.017，box021 ±0），不摔、gate 健康（valid 0.85–0.92、fallback ≤0.002）。**这是 E152 单点 (−0.010,0.05) 做不到的（彼时 2/3，box021 fail）——解耦让 max_violation 生效 + 提到 0.10 救回 box021。**
- **min_sdf 是主导旋钮**，单调权衡清晰：越紧（−0.005）→ 深穿透猛降（deep −0.39、pen2mm −0.18）但接触掉得多（physC −0.04~−0.06）、gate 咬得狠（valid 0.70）；越松（−0.015）→ 接触保住但穿透压不住（box021 真穿透反升）。
- **C3 gate 健康修复**：E152 box023 fallback=0.096 的退化边缘，在 E153 全 18 组 fallback ≤0.014（推荐点 ≤0.002）——解耦 + violation 容忍解决了 fallback 塌缩。
- **全程 0 fall**（18/18）。

## 1. Stage0：gate 解耦（C1 已在脚手架验证）

`valid=(min_sdf≥min_sdf_m)&(viol_pct≤max_viol)` → `valid=(min_sdf≥hard_floor)&(viol_pct≤max_viol)`，`hard_floor` 默认 nan→=min_sdf_m（旧行为），E153 固定 −0.020 激活 max_violation。
- **C1(a) 回归**：单元测试 nan==legacy `[T,F,F]`，E088–E152 数值不变。
- **C1(b) 激活**：hard_floor=−0.02 时旧逻辑毙掉的样本（最深−0.015、viol 0.025≤0.05）被放行 → max_violation 真正生效。
- 改动：`spider/config.py`（+`cem_{safety,hand}_gate_hard_floor_m`）、`spider/optimizers/sampling.py`（`_compute_sample_gate_info.add_gate` 加 floor）；sampling_fast/mjwp 经 import 自动覆盖。

## 2. 完整 Ablation 表

### 2.1 3-case 聚合（gateA_b1 vs b1，每格 3 case 均值；穿透/接触 Δ 负=穿透降，physC 负=接触降）

| min_sdf | max_viol | succ2mm | 5cmΔ | **pen2mmΔ** | pen5mmΔ | physCΔ | **deep<−5mmΔ** | legΔ | objErrΔ | gate_valid | fallback |
|---:|---:|:--:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| −0.005 | 0.05 | **3/3** | +0.012 | −0.179 | −0.234 | −0.041 | −0.387 | −0.012 | −0.000 | 0.700 | 0.005 |
| −0.005 | 0.10 | **3/3** | +0.019 | −0.192 | −0.230 | −0.063 | −0.408 | −0.004 | −0.000 | 0.762 | 0.000 |
| −0.010 | 0.05 | 2/3 | +0.019 | −0.038 | −0.062 | −0.013 | −0.195 | −0.020 | −0.000 | 0.853 | 0.003 |
| **−0.010** | **0.10** | **3/3** | +0.000 | −0.107 | −0.054 | −0.017 | −0.172 | −0.015 | −0.000 | 0.895 | 0.001 |
| −0.015 | 0.05 | 2/3 | +0.019 | −0.055 | −0.036 | −0.024 | −0.101 | +0.005 | −0.000 | 0.930 | 0.000 |
| −0.015 | 0.10 | 2/3 | +0.017 | −0.011 | −0.021 | +0.001 | −0.130 | −0.012 | −0.000 | 0.933 | 0.000 |

### 2.2 per-case 明细（gateA_b1 vs b1，关键列）

| case | min_sdf | max_viol | 5cmΔ | pen2mmΔ | physCΔ | deep<−5mmΔ | legΔ | succ2mm | gate_valid | fallback |
|---|---:|---:|---:|---:|---:|---:|---:|:--:|---:|---:|
| box021 | −0.005 | 0.05 | +0.000 | −0.107 | −0.053 | −0.368 | −0.027 | ✅ | 0.701 | 0.001 |
| box004 | −0.005 | 0.05 | +0.029 | −0.238 | −0.019 | −0.327 | −0.010 | ✅ | 0.653 | 0.014 |
| box023 | −0.005 | 0.05 | +0.007 | −0.191 | −0.051 | −0.466 | +0.000 | ✅ | 0.746 | 0.001 |
| box021 | −0.005 | 0.10 | +0.000 | −0.187 | −0.080 | −0.375 | −0.040 | ✅ | 0.775 | 0.000 |
| box004 | −0.005 | 0.10 | +0.057 | −0.162 | −0.029 | −0.373 | +0.029 | ✅ | 0.726 | 0.001 |
| box023 | −0.005 | 0.10 | +0.000 | −0.228 | −0.081 | −0.477 | +0.000 | ✅ | 0.784 | 0.000 |
| box021 | −0.010 | 0.05 | +0.000 | **+0.067** | −0.013 | −0.126 | −0.040 | ❌ | 0.847 | 0.000 |
| box004 | −0.010 | 0.05 | +0.057 | −0.143 | +0.019 | −0.273 | −0.019 | ✅ | 0.816 | 0.009 |
| box023 | −0.010 | 0.05 | +0.000 | −0.037 | −0.044 | −0.185 | +0.000 | ✅ | 0.895 | 0.000 |
| **box021** | **−0.010** | **0.10** | +0.000 | **−0.013** | +0.000 | −0.145 | −0.027 | ✅ | 0.850 | 0.000 |
| **box004** | **−0.010** | **0.10** | +0.000 | −0.257 | −0.029 | −0.192 | −0.019 | ✅ | 0.918 | 0.002 |
| **box023** | **−0.010** | **0.10** | +0.000 | −0.051 | −0.022 | −0.181 | +0.000 | ✅ | 0.918 | 0.000 |
| box021 | −0.015 | 0.05 | +0.000 | **+0.067** | +0.000 | −0.144 | +0.013 | ❌ | 0.915 | 0.000 |
| box004 | −0.015 | 0.05 | +0.057 | −0.181 | −0.029 | −0.156 | −0.019 | ✅ | 0.929 | 0.000 |
| box023 | −0.015 | 0.05 | +0.000 | −0.051 | −0.044 | −0.004 | +0.022 | ✅ | 0.946 | 0.000 |
| box021 | −0.015 | 0.10 | +0.000 | **+0.080** | +0.053 | −0.144 | −0.040 | ❌ | 0.918 | 0.000 |
| box004 | −0.015 | 0.10 | +0.057 | −0.076 | +0.010 | −0.117 | −0.019 | ✅ | 0.931 | 0.001 |
| box023 | −0.015 | 0.10 | −0.007 | −0.037 | −0.059 | −0.128 | +0.000 | ✅ | 0.950 | 0.000 |

（全列见 `eval/full/e153_grid_delta_vs_b1.tsv`；3-case mean/std/worst 见 `e153_combo_summary.tsv`；逐 row 绝对指标见 `e153_method_metrics.tsv`。）

## 3. 分析

1. **min_sdf 是主导旋钮，权衡单调**：−0.005（紧）把深穿透 `con<−5mm` 3-case mean 压到 −0.39~−0.41、真穿透 pen2mm −0.18~−0.19，但代价是 physC −0.04~−0.06、gate_valid 跌到 0.70（30% 样本被剔）；−0.015（松）gate 几乎不咬（valid 0.93），接触全保但穿透压不住，**box021 真穿透 pen2mm 反升 +0.07~+0.08**。
2. **box021 是瓶颈**：box004/box023 在几乎所有组合都 succ2mm pass；box021 只在 gate 够紧时（−0.005 任意 max_viol，或 −0.010 且 max_viol=0.10）真穿透才下降。松 gate 下 box021 的手停在箱面 2~?mm 浅插，pen2mm 升。
3. **max_violation 解耦后确有效**：同 min_sdf=−0.010，max_viol 0.05→0.10 把 box021 从 pen2mm +0.067（fail）翻到 −0.013（pass），3-case succ 2/3→3/3。机理：放宽 violation 容忍让 CEM 在 box021 上找到"贴面接触"而非"悬停+偶插"的解。
4. **甜点 `(−0.010, 0.10)`**：唯一在"接触零损失"前提下达 3/3 的组合（physC −0.017、box021 ±0），深穿透 −0.172、gate 最健康（valid 0.85–0.92、fallback ≤0.002）。
5. **猛压档 `(−0.005, *)`**：若可接受接触下降（physC −0.04~−0.08），能把深穿透干到 −0.39~−0.48（box023 deep −0.48！），适合"物理合规优先"的下游。

## 4. Claims 验证

| Claim | 判据 | 结果 | 裁定 |
|---|---|---|---|
| **C1** 解耦正确性 | (a) nan==legacy；(b) hard_floor 深时 max_viol 生效 | 单元测试均 PASS（脚手架阶段） | **成立** |
| **C2** 存在 3/3 strict 设定 | 某 (min_sdf,max_viol) 使 gateA_b1 全 3 case succ2mm + 穿透不回升 + 接触不掉 | `(−0.010,0.10)`：3/3，深穿透 −0.172、physC −0.017、不摔；另 `(−0.005,*)` 也 3/3（更猛） | **成立** |
| **C3** gate 健康 | 该设定 fallback 低、valid 不塌 | 推荐点 fallback ≤0.002、valid 0.85–0.92；全 18 组 fallback ≤0.014（修掉 E152 box023 的 0.096） | **成立** |

## 5. 可视化观察（SKILL §9 强制）

推荐组合 `(−0.010,0.10)` 的 ref|sim 关键帧（`cem/full/keyframes/E153_{case}_gateA_b1_sdf010_v10/f55.jpg`）：
- **box021 f55**：sim 机器人屈身、双手压蓝箱(手部橙色接触高亮)、箱体竖直、双脚站立未塌陷，与 ref 姿态一致；无 fall/趴箱/悬浮。
- **box023 f55**：sim 机器人下蹲、双手在橙箱上沿、箱体竖直、站姿稳定。
- 与 fall=false（18/18）+ 指标一致，无穿模塌陷视觉伪影。

## 6. 严谨性声明（experiment.md §5）

- 报 3-case mean + per-case（含 worst），无 cherry-pick；主判据用深度感知 `succ2mm`（pen2mm Δ≤0 + near-5cm Δ≥−0.02 + obj_err Δ≤0.02 + 不摔），并列报 0mm/2mm/5mm 三档几何穿透 + 物理 `con<−5mm`。
- **单 seed 局限**：每组合 1 seed CEM，相邻组合 ±0.05 级差异含随机性。稳健结论是宏观趋势（min_sdf 单调权衡、−0.005 强压穿透/代价接触、fallback 全面修复、0 fall）；`(−0.010,0.10)` 的 box021 pass 余量薄（pen2mm −0.013 贴近 0 阈值、单 seed），如需稳健 3/3 可选 `(−0.005,*)`（box021 pen2mm −0.11~−0.19，余量大但 physC 掉）。多 seed 复核留作后续。

## 7. 结论与下一步

- **gate 阈值可调出 3/3 strict**：解耦 + `(−0.010,0.10)` 在接触保持下达成 3/3，或 `(−0.005,*)` 以接触换更强物理合规。轴1（hand gate）作为方法贡献到此闭环（E088 身体→E152 手→E153 调优）。
- **推荐默认**：接触保持优先 → `(−0.010,0.10)`；物理合规优先 → `(−0.005,0.10)`。
- 下一步候选：
  1. **多 seed 复核** `(−0.010,0.10)` 与 `(−0.005,0.10)`（各 case ≥3 seed），固化 box021 pass 的稳健性。
  2. **轴1 杠杆 B（solref 硬化）** 与 gate 叠加，压 gate 没堵住的剩余几何穿透（计划留作单独 E）。
  3. **接 RL**：把选定配置的干净轨迹（接触保持 + 深穿透腰斩 + 不摔）作为 Holosoma RL 参考数据；E140 指出 Box021 仍 `missing_ref_mask_reward`，先补 config variant。

## 8. 结果路径

| 类型 | 路径 |
|---|---|
| 计划 | `plan/161_E153_gate_threshold_sweep_plan.md` |
| CEM 产物 | `results/E153/gate_threshold_sweep/cem/full/E153_*_{sdf,v}*.{npz,_full.mp4}` + `_outdir_full/`（18/18） |
| Eval | `results/E153/gate_threshold_sweep/eval/full/e153_{method_metrics,grid_delta_vs_b1,combo_summary}.tsv` + `e153_eval_summary.json` |
| 关键帧 | `cem/full/keyframes/E153_{case}_gateA_b1_{combo}/f{25,55,85}.jpg` |
| 代码 | Stage0：`spider/config.py`、`spider/optimizers/sampling.py`；eval：`scripts/eval/eval_E153_gate_threshold_sweep.py` |
| 脚本 | `scripts/E153/{_sweep_lib,run_case_box021,run_case_box004,run_case_box023}.sh`、`run_E153_remote.sh`、`pull_E153_remote_results.sh` |
