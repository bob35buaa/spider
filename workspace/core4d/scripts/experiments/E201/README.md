# E201 · 重定向数据筛选漏斗（下游 RL 数据分层）

面向**数据增强放量后**（E199/E200/…）的一站式数据筛选：把每条 CEM rollout 自动分到
`L1 弃 / L2 人工复审 / L3 自动收（进下游 RL）`，只把少数模糊件交人工，在**不牺牲数据质量**
前提下压缩人工负担。

- 计划：[../../../plan/231_E201_data_filter_funnel_plan.md](../../../plan/231_E201_data_filter_funnel_plan.md)
- 实验记录：[../../../log/288_E201_data_filter_funnel_and_vlm_prescreen.md](../../../log/288_E201_data_filter_funnel_and_vlm_prescreen.md)
- 阈值单一真源：`funnel_config.py`（**改阈值只改这里**，classifier/xlsx/doc 都引用它）

---

## 1. 三级漏斗 · 14 门

**4 硬门（全层强制，任一不过→直接 L1 弃）**：`fall==false`、`body_z≤0.20m`、`ankle_jerk_p95<1000`、`obj_speed_max<3.0`。

**10 带门（宽/窄双阈值）**：6 tracking（root/eef/obj 的 pos+ori）+ contact_in_mask / release3mm / hand_pen3mm / leg_pen。窄=严格口径、宽=宽松口径，**保证 `narrow⟹wide` 嵌套**（classifier 启动自检，0 违反才继续）。完整阈值见 `funnel_config.py`。

**判定顺序（逐 rollout）**：
```
硬门任一不过 或 任一带门连"宽"都不过        -> L1_reject（自动弃）
10 带门全过"宽"但 ≥1 门不过"窄"            -> L2_review（人工复审：中间带）
10 带门全过"窄"                            -> L3，再看家族一致性：
      同 case_id 的 {orig,trans0,trans1,trans2} 其余臂全过窄 -> L3_auto（自动收→RL）
      否则                                                  -> L3_review（人工二审）
```
- **自动决策** = L1_reject + L3_auto；**人工** = L2_review + L3_review。
- 家族键用 `family_key()` 把 orig 的 `_p{N}` 与 aug 的 `_person{N}` 规范化到同一 case，否则家族会被拆开。

## 2. 快速开始

```bash
# ① 分层（读 case_metrics，recompute body_z，输出逐条 TSV + 打印各层计数）
.venv/bin/python workspace/core4d/scripts/experiments/E201/classify_funnel.py --exp E199 --assert-monotonic
#   -> results/E201/funnel/E199_funnel_rollout.tsv

# ② 漏斗 workbook（detail: 14 门宽/窄值+PASS+layer；summary: 各层计数+自动率）
.venv/bin/python workspace/core4d/scripts/eval/reports/gen_E201_funnel_xlsx.py --exp E199
#   -> results/E201/funnel/E201_E199_funnel.xlsx

# ③ 待审队列（L2+L3_review）供人工/可视化
.venv/bin/python workspace/core4d/scripts/experiments/E201/select_review_queue.py --exp E199
#   -> results/E201/vlm_review/E199_review_queue.tsv
```

**E199 实测（默认严格）**：L1=182(54.8%) / L2=49 / L3_auto=20 / L3_review=81 → **自动决策 60.8%，人工 130**。
对用户人工标注校验：自动决策桶准确率 92.6%，**假收 0**。

## 3. 家族一致性规则（默认严格）

`classify_funnel.py --family-min-other-pass {all|2|1}`（默认 **all**）：

| 取值 | 含义 | E199 自动决策 | 人工 | 新增自动收假收率 |
|---|---|---|---|---|
| `all`（默认） | 其余 3 臂全过窄才自动收 | 60.8% | 130 | **0** |
| `2` | ≥2/3 其余臂过窄 | 72.6% | 91 | ~62% |
| `1` | ≥1/3 | 79.2% | 69 | ~70% |

**默认严格 = 数据质量优先。** 放松能降人工，但"家族不一致的 L3-narrow"正是"指标侥幸过窄但动作坏"的那批，放松会把大量坏数据放进 RL（见 log288 sweep）。除非下游 RL 明确容忍噪声，否则不要放松。

## 4. 扩展到新实验（E200 / 新增强批次）

1. 跑完该实验的 CEM + `eval.core.core_metrics` 打分，得到 `*_case_metrics.tsv`（须含 14 门所需字段 + `qpos_path`/`scene_act`/`qpos_frames`/`duration_s`）。
2. 在 `classify_funnel.py` 的 `EXPS`（和 `select_review_queue.py` 的 `CASE_METRICS`）里加一行 `"E200-prg_g1a2": (<case_metrics 路径>, <manifest 路径>)`。
3. 跑第 2 节的三条命令，`--exp E200-prg_g1a2`。阈值/判定完全复用，不改。

> body_z 未落盘时由 `gen_E199_fullscale_gate_xlsx.body_z_p95()`（MuJoCo CPU FK）recompute，与 review player / E194/E198 arm sweep 的 12-gate 字节一致。

## 5. VLM 初审：现状与结论（**弃用于判别，仅作参考**）

`run_E201_vlm_review.sh` 提供完整 VLM 初审管线（渲双视角 HQ 帧+mp4 → 建请求 → 调 ecodata API → 解析+校验）。
**但 E199 上已充分证实**：Qwen3-VL-235B 与 gemini-3.5-flash（含双视角 + HQ 画质 + 扩展 prompt + 集成）在漏斗中间带的判别力**均低于"全判拒"基线**——CORE4D 搬箱多为"抵身托举"非干净抓握，VLM 系统性误判 bad_grasp。

**结论**：不要用 VLM 输出做 USE/DO_NOT_USE 自动决策。它的产出（双视角 mp4/帧 + verdict）可作为人工复审时的**参考展示**（尤其 fall/penetration 这类不依赖深度的失败它偶尔命中）。管线保留供换模型/换任务时复用。

调用（API 当黑盒、路径不写死）：
```bash
ECODATA_PYTHON=/root/miniconda3/bin/python IMAGE_SIZE=1024 RENDER_PROC=48 \
  MODELS="Qwen3-VL-235B-A22B-Instruct,gemini-3.5-flash-huangxiaoshuang" \
  bash workspace/core4d/scripts/eval/wrappers/run_E201_vlm_review.sh --exp E199 \
  [--call-api-path <path/to/call_api_imitate_redaccel.py>]
```

## 6. 渲染说明（headless 无 NVIDIA EGL ICD）

本机只装 Mesa EGL → MuJoCo EGL 回退 **llvmpipe 软件渲染**（CPU-bound，不占 GPU）。因此 `render_frames_for_vlm.py`
用**分片并行**（N 独立进程，非 mp.Pool——EGL 不能跨 fork-pool 存活），192 核用 `RENDER_PROC=48`。
抽帧 JPEG 用 quality=95（早期误用 8/100 导致块噪）、关阴影（去地面条纹）、双视角左右拼图。

## 7. 文件清单

| 文件 | 作用 |
|---|---|
| `funnel_config.py` | **14 门阈值单一真源** + `assert_monotonic` |
| `classify_funnel.py` | 分类器（分层 + 家族仲裁）|
| `select_review_queue.py` | 导出 L2/L3_review 待审队列 |
| `render_frames_for_vlm.py` | 渲双视角 HQ 帧 + 全帧 mp4（分片并行）|
| `build_vlm_requests.py` / `parse_vlm_verdicts.py` | 建 VLM 请求 / 解析+校验 |
| `vlm_review_prompt.txt` | VLM prompt（单一真源）|
| `../../eval/reports/gen_E201_funnel_xlsx.py` | 漏斗 workbook |
| `../../eval/wrappers/run_E201_vlm_review.sh` | VLM 管线入口（多模型、API 黑盒）|
