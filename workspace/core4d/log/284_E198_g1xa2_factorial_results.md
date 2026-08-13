# E198 结果：G1×A2 完整 2×2 因子 —— 两干预非可加，主要相互纠正对方的回退

_Core4D · Phase 61 · 2026-08-13 · 计划 [plan226](../plan/226_E198_g1xa2_factorial_and_E192_a2_expansion_plan.md) · evaluator `core4d-e154-physics-contact-v1`_

## TL;DR

- **执行闭合**：103/103 Full CEM 完成（本地 8 卡 priority 队列，与他人 job 叠加共跑，0 失败）；四物体 4 臂（A0/G1/A2/G1+A2）共 236/236 用同一公共 evaluator 打分，0 error。
- **C3 baseline parity PASS**：复用的 box021/023 A0+G1（88 行）重打分与冻结 E194 表逐 case z 差 `0.000000 cm`，交互项非 confounded。
- **z-tracking 由 G1 独占，A2 不贡献**：`track_obj_z_abs_err_cm_mean` 的改善几乎全部来自 G1（如 box024 5.574→3.229），A2 单加几乎不动 z；G1+A2 ≈ G1。交互项多数 CI 含 0（近似可加），仅 box021 `INT=−0.230 [−0.442,−0.039]` 轻微协同。
- **核心发现（gate 迁移）——两干预相互纠正对方的回退**：
  - **A2 单用会损伤姿态/朝向门**（A0→A2 全 59 例）：`root_ori −13.6pp`（exact p=0.021）、`hand_ori −16.9pp`（p=0.006）、`root_pos −6.8pp`、`hand_pos −5.1pp`。在四物体全集上复现了 E192 的 A2 姿态代偿/collapse 机制。
  - **叠加 G1 能救回 A2 的损伤**（A2→G1+A2）：`root_ori +15.2pp`（p=0.004，9 例 F→P）、`lower_body +13.6pp`（p=0.039，10 例 F→P）、`hand_ori +11.9pp`、`hand_pos +10.2pp`。
  - **在 G1 上叠加 A2 基本中性**（G1→G1+A2）：多数门 |Δ|<4pp；`object_ori +8.5pp`（5 例 F→P，p=0.06）——即 A2 叠在 G1 上不再引发它单用时的姿态塌陷。
- **A2 的降穿透收益在 G1 之上不叠加**：box024 A2 单用把 3mm 手物穿透 `0.378→0.308`，但 G1+A2=`0.320`≈G1（`INT=+0.069 [−0.034,0.170]`）。
- **object_ori 的物体特异协同**：box023 `INT=−3.07 [−5.70,−0.78]`、box004 `INT=−3.51 [−8.70,−0.29]`——G1 单用抬高 box023 朝向误差（5.11→7.96°），但 G1+A2 降回 4.94°（A2 纠正了 G1 的朝向回退）。
- **判决 `FACTORIAL_CHARACTERIZED`**：两干预**非独立**、主要通过相互抵消各自的副作用而非叠加各自的收益来交互；G1+A2 在大多数指标上接近 G1-only。**不升级 A2 或 G1+A2**；A2 的 `INCONCLUSIVE_GATE_COLLAPSE` governance 不被推翻。

## 1. 设计与执行

完成 box004/box021/box023/box024 四物体各自的 2×2 因子（none=A0/PRG、G1=object gravcomp、A2=hand-gate 三字段、G1+A2）。G1、A2 定义与冻结不变量见 plan226。本轮只新增两组 GPU 运行：

| 实验 | 臂 | Cases | 结果目录 |
|---|---|---|---|
| E198 | G1+A2 | box024(9)+box004(6)+box021(28)+box023(16)=59 | `results/E198/s6_downstream/cem/full_g1a2/` |
| E192-ext | A2 | box021(28)+box023(16)=44 | `results/E192/s6_downstream/cem/full_a2_expansion/`（见 [log283](283_E192_a2_expansion_box021_box023_results.md)）|

历史 A0/G1/A2（box024/004）与 box021/023 的 A0/G1 复用既有 rollout，由本轮同一 `core4d-e154-physics-contact-v1` evaluator 重打分（单变量 C1、单 evaluator C3 均满足）。

执行：本地 8× GPU（0-7）统一 priority 队列，`PER_GPU_MEM_MIB=5000`、每卡 1 run，严格 P0(box024 G1+A2)→P1(A2)→P2(G1+A2)→P3(box004 G1+A2)；与 GPU0/2/4 上他人 job 叠加共跑，未 kill/抢占任何进程。canary 4/4 通过。

## 2. 交互项（`INT = M(G1+A2) − M(A2) − M(G1) + M(A0)`，paired bootstrap 95% CI）

主指标（cm/占比，越低越好；in-mask contact 越高越好）。完整六指标见 report。

### obj z |err| (cm)
| Object | n | A0 | G1 | A2 | G1+A2 | INT [95% CI] |
|---|---:|---:|---:|---:|---:|---:|
| box024 | 9 | 5.574 | 3.229 | 5.281 | 3.227 | 0.291 [−0.091, 0.635] |
| box021 | 28 | 6.237 | 4.938 | 6.429 | 4.901 | **−0.230 [−0.442, −0.039]** |
| box023 | 16 | 5.817 | 5.276 | 5.848 | 5.137 | −0.170 [−0.453, 0.164] |
| box004 | 6 | 4.892 | 4.483 | 5.094 | 4.540 | −0.145 [−0.451, 0.172] |

### hand 3mm penetration（越低越好）
| Object | n | A0 | G1 | A2 | G1+A2 | INT [95% CI] |
|---|---:|---:|---:|---:|---:|---:|
| box024 | 9 | 0.378 | 0.320 | 0.308 | 0.320 | 0.069 [−0.034, 0.170] |
| box021 | 28 | 0.176 | 0.192 | 0.195 | 0.212 | 0.001 [−0.037, 0.037] |
| box023 | 16 | 0.148 | 0.164 | 0.144 | 0.146 | −0.013 [−0.088, 0.052] |
| box004 | 6 | 0.147 | 0.098 | 0.107 | 0.114 | 0.055 [−0.001, 0.131] |

### obj ori err (deg)（越低越好）
| Object | n | A0 | G1 | A2 | G1+A2 | INT [95% CI] |
|---|---:|---:|---:|---:|---:|---:|
| box024 | 9 | 6.251 | 5.951 | 6.247 | 5.159 | −0.789 [−2.970, 1.406] |
| box021 | 28 | 6.091 | 5.873 | 6.261 | 5.948 | −0.095 [−0.600, 0.350] |
| box023 | 16 | 5.108 | 7.963 | 5.160 | 4.943 | **−3.072 [−5.698, −0.781]** |
| box004 | 6 | 11.534 | 11.325 | 15.179 | 11.465 | **−3.505 [−8.697, −0.286]** |

leg penetration / 3D pos / in-mask contact 的完整表见 `E198_G1xA2_factorial_report.md`；leg penetration 全部 CI 含 0（box024 `INT=−0.100 [−0.228,0.024]` 呈救援趋势但不显著）。

## 3. 12-gate 迁移（全 59 例，exact McNemar）

| Transition | 关键门 | Δpp | P→F / F→P | exact p |
|---|---|---:|---:|---:|
| A0→A2 | root_ori | −13.6 | 9 / 1 | **0.021** |
| A0→A2 | hand_ori | −16.9 | 11 / 1 | **0.006** |
| A0→A2 | root_pos | −6.8 | 5 / 1 | 0.219 |
| A0→G1 | lower_body | +13.6 | 4 / 12 | 0.077 |
| A0→G1 | hand_penetration | −10.2 | 9 / 3 | 0.146 |
| G1→G1+A2 | object_ori | +8.5 | 0 / 5 | 0.063 |
| A2→G1+A2 | root_ori | +15.2 | 0 / 9 | **0.004** |
| A2→G1+A2 | lower_body | +13.6 | 2 / 10 | **0.039** |
| A2→G1+A2 | hand_ori | +11.9 | 2 / 9 | 0.065 |

读法：A2 单用（A0→A2）显著砸 orientation/pose 门；把 G1 叠加到 A2 上（A2→G1+A2）把这些门大幅救回（大量 F→P）。这说明 G1+A2 的净行为主要由 G1 决定，A2 单用的姿态破坏被 G1 的下肢/支撑改善抵消。

## 4. Claims 判定

| Claim | 判定 | 证据 |
|---|---|---|
| C0 scope/provenance | PASS | 103 新 run + 4 物体 4 单元 authority 全可追溯；SHA parity 全过 |
| C1 单变量 intervention | PASS | 59 个 G1+A2 单变量 gravcomp 审计通过；A2 仅改 3 gate 字段；`A2_GATE==E192` |
| C2 execution/numeric closure | PASS | 103/103 Full；236/236 scored；error/non-finite/diverged=0 |
| C3 baseline parity | PASS | 复用 88 行重打分 vs 冻结表 z 差 `0.000000 cm` |
| C4 交互项估计 | PASS | 四物体 2×2 完整；六指标 INT+CI+四主效应逐物体报告 |
| C5 承重接触保留 | PASS（除 box024 边界） | G1+A2 in-mask contact vs A0：box021/004 ≥−0.05；box024 `0.308→0.373`（较 A0 +0.065）；box023 `0.473→0.458`（−0.015）|
| C6 物理安全无灾难 | PASS | 无新增 fall/non-finite/diverged |
| C7 gate migration 透明 | PASS | 4 transition × 12 门全报 P→F/F→P + McNemar |
| C8 device confound | PASS | 队列按空闲卡 round-robin，落卡 GPU id 逐 case 记录 |
| C9 证据闭合 | PASS | 数值/gate/interaction 全闭合；4-cell MP4 + viser 交互复核见 §6 |

## 5. 解释（非额外观测）

1. **z-tracking 是 G1 的单变量领域**：A2（hand-gate）不触及 object servo，故 z 上 INT≈0、G1+A2≈G1。这与机制预期一致。
2. **A2 单用的收益（降 box024 手物穿透）不能叠加到 G1 上**：G1+A2 的穿透≈G1，说明两者在“降穿透”上争夺同一自由度而非互补。
3. **主交互形态是“相互救援”而非“协同增益”**：A2 单用引入姿态/朝向/下肢代偿（复现 E192），G1 单用引入 object_ori 回退（复现 E194）；组合时 G1 救 A2 的姿态、A2 救 G1 的 box023/004 朝向，于是 G1+A2 的门通过构成比任一单臂更均衡，但并未在任一主指标上超越 G1 的最好表现。
4. 因此 **没有证据支持把 G1+A2 作为优于 G1 的新默认**；它更像“用 G1 主导 + A2 微调姿态门”，收益有限且物体特异。

## 6. 可视化（已完成）

CEM 队列以 `save_video=false` 跑（吞吐优先），离线渲染另做。初始 `osmesa`/`egl` 均失败
（本机缺 `libOSMesa`、EGL 无 NVIDIA PLATFORM_DEVICE）；**安装 `libosmesa6` 后 osmesa 软件渲染恢复**，
据此离线渲染 2×2 四单元视频。

**产物**：
- **4-cell MP4**（A0 左上 / G1 右上 / A2 左下 / G1+A2 右下，带 arm 标签与 12-gate pass 标记）：
  box024 P0 全 9 例 + box004 全 6 例，`results/E198/s6_downstream/render/full_factorial/E198_{case}_4cell.mp4`。
- **viser 交互复核**：`bash workspace/core4d/scripts/eval/wrappers/review_player.sh E198 --port 8080`
  已接入（review_index 增加 E198 arm-sweep 条目），**live qpos 回放全部 236 个 arm-case（4 臂×59），
  236/236 playable**，不依赖本机 GL。

**实际观察**（box024_026_p1 中段帧，`/tmp/e198_verify/mid.png`，代表 P0 强制极端 case）：
G1 与 G1+A2（右列）机器人把长箱托得明显更水平/更高，箱体下栽被修正；A0 与 A2（左列）箱体明显
前倾下沉——直观印证 G1 的重力下垂修复，且 G1+A2 的箱姿几乎与 G1 相同、A2 单独不改变下沉。这与
§2 “z 由 G1 独占、G1+A2≈G1” 的数值结论一致。逐例四阶段人审可在 viser 中对 A2→G1+A2 的
root_ori/lower_body F→P 救援 case 与 box023/004 object_ori 协同 case 继续展开。

## 7. 复现入口

```bash
# 1. 构建 103-row priority manifest + 快照 + 单变量审计
MUJOCO_GL=osmesa .venv/bin/python workspace/core4d/scripts/experiments/E198/build_g1a2_manifest.py --apply --snapshot
# 2. 本地 8 卡 priority 队列（canary 后 Full）
MODE=canary bash workspace/core4d/scripts/launch/active/run_E198_local_8gpu.sh
MODE=full GPUS=0,1,2,3,4,5,6,7 PER_GPU_MEM_MIB=5000 MAX_PER_GPU=1 bash workspace/core4d/scripts/launch/active/run_E198_local_8gpu.sh
# 3. 四臂 2×2 因子 eval + 报告
bash workspace/core4d/scripts/eval/wrappers/eval_E198_factorial.sh
PYTHONPATH=workspace/core4d/scripts .venv/bin/python workspace/core4d/scripts/eval/reports/gen_E198_factorial_report.py
```

## 8. 产物

- 评估目录：`workspace/core4d/results/E198/s6_downstream/eval/full_factorial/`
  - `e198_factorial_by_case.tsv` / `e198_factorial_by_object.tsv` / `e198_gate_migrations.tsv` / `e198_factorial_summary.json` / `e198_arm_cache.tsv`（236 行）
  - `E198_G1xA2_factorial_report.md`
- CEM 输出：`results/E198/s6_downstream/cem/full_g1a2/`（59）；`results/E192/s6_downstream/cem/full_a2_expansion/`（44）
- Manifest/快照：`results/E198/s6_downstream/manifests/e198_priority_*_manifest.tsv`；`results/E198/scene_snapshot/g1a2/`、`results/E192/scene_snapshot/a2_expansion/`

## 9. 下一步

1. 补 viser 强制视觉复核（§6），回填实际观察后将 C9 置 PASS。
2. 若继续，优先解释 box023/box004 的 object_ori 协同机制，而非把 G1+A2 当默认推广。
3. 不重复 A2 单参数 seed；不叠加更多干预后声称单机制归因（延续 E192 下一步纪律）。
