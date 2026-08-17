# log288 · E201 · 三级数据筛选漏斗 + VLM 初审（下游 RL 数据自动分层）

_Core4D · Phase 64 · Run **R289**（纯离线分析，无 CEM/GPU） · 计划 [plan231](../plan/231_E201_data_filter_funnel_plan.md) · 承接 [log287](287_E199_box_fullscale_translation_augmentation.md)（E199 249 aug + 83 orig） · 2026-08-17-18 · 分支 `experiment/E199-omniretarget-object-augmentation`_

## Purpose / 假设

E199（332 rollout）+ E200（≤498）产出大量重定向数据，人工逐条复核不可持续。**假设**：用「14 门 · 双口径（宽/窄）」三级漏斗可把大部分 rollout 自动分层（L1 弃 / L3 自动收），只把中间带 + 家族不一致的少数交人工，在**不牺牲数据质量（0 假收）**前提下显著压缩人工负担。附带验证 VLM（Qwen3-VL-235B / gemini-3.5-flash）能否进一步替人工审中间带。

## Parameters / 关键配置

**14 门 = 4 硬门（全层强制）+ 10 带门（宽/窄双阈值）**，阈值集中于 `funnel_config.py`（单一真源）。

- 硬门：`fall==false`、`body_z≤0.20`、`ankle_jerk_p95<1000`、`obj_speed_max<3.0`。
- 带门（窄 / 宽）：root_pos 20/21、root_ori 20/21、eef_pos 20/21、eef_ori 20/21、obj_pos 20/21、obj_ori 10/11（cm/deg，≤）；contact_in_mask 0.50/**0.40**（≥）；release3mm 0.30/0.60、hand_pen3mm 0.32/**0.55**、leg_pen 0.20/0.40（≤）。
- 用户决策：接触宽=0.40（修正草案 0.51 方向 bug）；窄 leg_pen/hand_pen 刻意松于 E178 canonical（下游 RL 容忍）；运动健康=全局硬门；tracking 宽带=+1（刻意收窄）；body_z 归硬门；hand_pen 宽 0.50→0.55（救边界假弃）。
- **三级判定**：硬门/宽门任一不过→L1 弃；过宽不过窄→L2 中间带（人工）；过窄→L3，再看**家族一致性**（同 case_id 的 orig+trans0/1/2）：**默认严格**（其余 3 臂全过窄→自动收，否则人工二审）。`--family-min-other-pass {all|2|1}` 可放松（默认 all）。

## Run command / 复现

```bash
# 分层 + 报告 + 待审队列
.venv/bin/python workspace/core4d/scripts/experiments/E201/classify_funnel.py --exp E199 --assert-monotonic
.venv/bin/python workspace/core4d/scripts/eval/reports/gen_E201_funnel_xlsx.py --exp E199
# VLM 初审（双模型，2 视角，HQ 帧）
ECODATA_PYTHON=/root/miniconda3/bin/python STAGES=render,build,call,parse \
  RENDER_PROC=48 IMAGE_SIZE=1024 \
  MODELS="Qwen3-VL-235B-A22B-Instruct,gemini-3.5-flash-huangxiaoshuang" \
  bash workspace/core4d/scripts/eval/wrappers/run_E201_vlm_review.sh --exp E199
```

## Results

### 1. E199 分层 sizing（默认严格，332 rollout）

| 层 | 数 | 占比 | 去向 |
|---|---|---|---|
| L1_reject | 182 | 54.8% | 自动弃 |
| L2_review | 49 | 14.8% | 人工复审 |
| L3_auto | 20 | 6.0% | **自动收 → 下游 RL** |
| L3_review | 81 | 24.4% | 人工二审（家族不一致） |

**自动决策 202/332 = 60.8%**，人工 130（39.2%）。aug 61.0% 自动 / orig 86.7%。逐物体：box021 最高（80% 自动），**box001 最低（37.5%，数据本身偏弱）**。

- **C0 不变量**：`narrow⟹wide` 对全 10 带门 × 332 行 **0 违反**。
- 修复 2 个正确性 bug：① orig 用 `_p{N}`、aug 用 `_person{N}` 导致家族被拆 → 加 `family_key` 规范化；② **orig 也做家族检查**（原本 orig 过窄直接自动收，4 个假收全是家族不一致的"指标干净但人工判废"orig）。

### 2. 漏斗准确度（vs 用户人工标注 `user_manual_review_filled.tsv`，69 带变体标注）

自动决策桶（L1_reject + L3_auto）共 27 条重叠：**准确率 92.6%（25/27），假收 0，假弃 2**（皆单门 wide 边界，人工判 MINOR/CLEAN 的良性产量损失）。中间带（L2+L3_review）人工实测 USE/DNU 约对半 → 漏斗把真正模糊的正确路由给人工。
> 注：严格 L3_auto 与标注集 0 重叠（标注偏难样本、家族多不一致），故 L3_auto 精度无标注直接佐证，依赖家族先验；后续应对 20 条 L3_auto 抽检。

### 3. VLM 初审（负结论）

在**已控制**画质（修 JPEG quality 8→95 块噪 + 关阴影去地面条纹 + 640×480→960×720 + image_size 1024）、**双互补视角**（azimuth 135°+225° 左右并排）、**扩展 prompt**（加 back_of_hand / inverted_joint）之后：

| 模型 | 一致率(68重叠) | 分布 | 主导误报 | 假收 |
|---|---|---|---|---|
| Qwen3-VL-235B | 57.4% | DNU118/USE12 | bad_grasp 118 | 5 |
| gemini-3.5-flash | 52.9% | DNU97/USE33 | floating71/bad_grasp71 | 11 |

**全判拒基线 = 61.8%（42/68）。两模型 HQ 均 < 基线**（单视角 Qwen 曾 60%）。集成也无用：两模型都判 USE(n=3)真 USE 仅 1/3；都判 DNU(n=48)真 DNU 仅 58%。新增门检出极少（back_of_hand Qwen 9/gemini 0，inverted_joint 均 0）。

**根因**：CORE4D 搬箱多为"抵身托举"而非干净手指抓握 → VLM 系统性误判 bad_grasp（118/130）；且 L2/L3 本就是最模糊件（人工自身对半开）。加视角/提画质/换模型/集成都治不了。

### 4. 家族门放松 sweep（用户探索 → 决定保持严格）

| 家族规则 | L3_auto | L3_review | 自动决策 | 人工 | 新增自动收假收率 |
|---|---|---|---|---|---|
| **严格(all，默认)** | 20 | 81 | 60.8% | 130 | 0/0 |
| ≥2/3 过 | 59 | 42 | 72.6% | 91 | **10/16 = 62%** |
| ≥1/3 过 | 81 | 20 | 79.2% | 69 | **16/23 = 70%** |

放松确实降人工，但"家族不一致的 L3-narrow"正是"指标侥幸过窄但动作坏"的那批——放松等于把 6~7 成坏数据放进 RL。**决策：保持严格（数据质量优先）**。

## Claims 验证

| Claim | 结论 |
|---|---|
| C0 不变量 | ✅ narrow⟹wide 0 违反（332 行）|
| C1 分层定量 | ✅ 332 闭合，逐物体 + aug/orig 分层表 |
| C2 自动决策 ≥60% | ✅ 60.8%（严格，0 假收前提下）|
| C3 家族逻辑 | ✅ 0 错；含 family_key 规范化 + orig 家族检查修复 + 顺序无关重构 |
| C4 视觉复核 | ⚠️ 已渲双视角 mp4/帧供人工；20 条 L3_auto 抽检待做 |
| C7/C9 VLM 初审 | ❌ 负结论：两模型均 < 全判拒基线，不具判别力，弃用 |

## Conclusion / 下一步

1. **漏斗上线**：默认严格家族门，自动决策 60.8%（L1 弃 55% + L3-auto 收 6%，0 假收），人工只需审 130（L2 49 + L3_review 81）——较过去 332 全人工降 60.8%。
2. **VLM 初审弃用**于中间带 USE/DNU 判别（已充分证实）。双视角 mp4/帧/verdict 仅作人工复审的参考展示。
3. **待办**：抽检 20 条 L3_auto 确认自动收质量；把 130 条待审导入 review_player 做人工复审；E200 双 arm CEM 跑完后套同 funnel（`--exp E200-*`）。

## 改动文件

| 文件 | 说明 |
|---|---|
| `scripts/experiments/E201/funnel_config.py` | 14 门单一真源 + `assert_monotonic` |
| `scripts/experiments/E201/classify_funnel.py` | 分类器 + 家族仲裁（`--family-min-other-pass`，默认 all）|
| `scripts/experiments/E201/{select_review_queue,render_frames_for_vlm,build_vlm_requests,parse_vlm_verdicts}.py` | VLM 管线（渲双视角HQ帧+mp4 / 建请求 / 解析校验）|
| `scripts/experiments/E201/vlm_review_prompt.txt` | VLM prompt（含 back_of_hand/inverted_joint）|
| `scripts/eval/reports/gen_E201_funnel_xlsx.py` | 漏斗 workbook |
| `scripts/eval/wrappers/run_E201_vlm_review.sh` | VLM 管线入口（API 黑盒、路径可配、多模型）|
| `scripts/experiments/E201/README.md` | 数据筛选 + 评测引导 |
| 结果 | `results/E201/funnel/{E199_funnel_rollout.tsv,E201_E199_funnel.xlsx}`、`results/E201/vlm_review/{frames,verdicts,out}/` |

> 纯离线分析，无物理仿真 → rule 10b scene 快照豁免。
