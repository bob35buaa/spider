# plan231 · E201 · 三级数据筛选漏斗（14-gate funnel，下游 RL 数据自动分层）

_Core4D · Phase 64 · Run **R289**（纯离线分析，无 CEM/GPU） · 承接 [plan229](229_E199_box_fullscale_translation_augmentation_plan.md)（E199 249 aug）+ [plan230](230_E200_augmentation_prg_g1a2_and_noprg_arms_plan.md)（E200 ≤498 aug 双 arm） · 2026-08-17 · 分支 `experiment/E199-omniretarget-object-augmentation`（同方向延伸，不新建分支）· **状态：计划态，待批准**_

## Context / 目标

E199（249 aug + 83 orig）与 E200（≤498 aug，PRG+G1+A2 / noPRG 双 arm）产出**大量**重定向 rollout，人工逐条复核不可持续。E201 建一套 **config 驱动的三级筛选漏斗**：用 **14 门 · 双口径（宽/窄）** 把每条 rollout 自动分到 `L1 弃 / L2 人工复审 / L3 自动收（进下游 RL）`，**只把「中间带」+「家族不一致」的少数 case 交人工**，量化并压缩人工负荷。

- **E201 是纯离线分析/分类实验**：不跑 CEM、不占 GPU、不改 scene，只读现有 `case_metrics.tsv`（外加 body_z recompute）。→ **rule 10b scene 快照豁免**（纯分析脚本）。
- 输入 = E199/E200 的 case_metrics（每条 rollout 的全指标 + 已算好的 6-gate）；输出 = 逐条分层 TSV + 漏斗 xlsx + 待审队列（喂给 viser review player）。
- 同分支同方向（数据筛选是 core4d 增强放量的下游治理，非新数据集/新任务域）。

草案来源：`workspace/core4d/docs/数据筛选漏斗构建.md`（本计划定稿其阈值与判定逻辑）。

## 关键决策（用户 2026-08-17 确认）

1. **接触门宽口径 = ≥0.40**（窄 ≥0.50）。修正草案原 0.51 的方向错误（接触是 ≥ 下限门，宽松版须调低门槛，否则破坏「宽⊇窄」嵌套）。
2. **窄口径 leg_pen ≤0.20 / hand_pen3mm ≤0.32 是刻意放松**（比冻结的 E178 canonical `0.10 / 0.30` 松），因下游 RL 对这两项容忍度高。非笔误，log 须显式记录窄口径 ≠ E178 canonical。
3. **运动健康 = 全局硬门**（`ankle_jerk_p95 < 1000` 且 `obj_speed_max < 3.0`，任何层都不豁免）。
4. **接触门沿用标准 in_mask 接触率**（`hand_object_physics_contact_in_mask_frac`，与原口径一致，**不用** 3mm 版）。
5. **tracking 宽带 = +1**（刻意保持窄，避免中间带过大导致人工没省）。

## 14 门定义（4 硬门 + 10 带门）

字段全部在 `e199_fullscale_case_metrics.tsv` header 中确认存在；`body_z` 未落盘 → 复用 `gen_E199_fullscale_gate_xlsx.py` 的 `G.body_z_p95()` recompute（与 review player / E194/E198 arm sweep 的 12-gate 字节一致）。

### 硬门（4，无宽窄，全层强制；任一不过 → 直接 L1 弃，不进后续判定）

| 硬门 | 字段 | 判据 |
|---|---|---|
| fall | fall_flag | == false |
| body_z | body_z_err_p95_m（recompute） | ≤ 0.20 m |
| ankle_jerk | ankle_jerk_p95 | < 1000 |
| obj_speed | obj_speed_max | < 3.0 |

> **body_z / fall 归硬门的设计决策**：草案宽/窄表未给 body_z 赋带，且 body_z（米制）与「+1」（cm/deg）不同量纲；body_z 与 fall 均为二值安全/健康门，与运动健康同性质 → 归硬门（全层 `≤0.20` canonical）。**标记为待复核决策**（见「待确认」），C4 视觉复核会专门抽查 body_z 触发的 L1 弃。

### 带门（10，宽/窄双阈值）

| gate | 字段 | 方向 | 窄 | 宽 |
|---|---|---|---|---|
| root_pos | track_root_pos_err_cm_mean | ≤ | 20 | 21 |
| root_ori | track_root_ori_err_deg_mean | ≤ | 20 | 21 |
| eef_pos | track_eef_pos_err_cm_mean | ≤ | 20 | 21 |
| eef_ori | track_eef_ori_err_deg_mean | ≤ | 20 | 21 |
| obj_pos | track_obj_pos_err_cm_mean | ≤ | 20 | 21 |
| obj_ori | track_obj_ori_err_deg_mean | ≤ | 10 | 11 |
| contact | hand_object_physics_contact_in_mask_frac | ≥ | 0.50 | **0.40** |
| release | hand_object_release_false_contact_3mm_frac | ≤ | 0.30 | 0.60 |
| hand_pen | hand_object_physics_penetration_3mm_frame_frac | ≤ | 0.32 | 0.55 |
| leg_pen | leg_penetration_frac | ≤ | 0.20 | 0.40 |

**核心不变量（必须自检，C0）**：对每条 rollout 的每个带门，`narrow_pass ⟹ wide_pass`（宽口径集合 ⊇ 窄口径集合）。修正后接触 0.50→0.40 满足方向；其余带门宽阈值均比窄更松 → 天然嵌套。分类器启动时对全部行断言，0 违反才继续。

## 三级漏斗判定（逐 rollout）

```
1. 硬门任一不过 → L1 弃（auto-reject），停止判定
2. 硬门全过后，看 10 带门：
   a. 任一带门连「宽」都不过        → L1 弃（auto-reject）
   b. 10 带门全过「宽」但 ≥1 门不过「窄」 → L2 中间带
                                       → Agent 初审(参考,无效力) + 人工复审(拍板)
   c. 10 带门全过「窄」            → L3
        - 家族一致性检查（orig 与 aug 同规则）：
              同 case_id 全部存在臂({orig,trans0,trans1,trans2}) 全 ∈ L3(窄通过)
                  → L3-auto 自动收 → 下游 RL
              否则 → L3-review(Agent 复审 + 人工二审)
```

- **人工负荷** = `|L2| + |L3-review|`；**自动决策** = `|L1| + |L3-auto|`。
- **家族定义**：同 `case_id` 的 `{orig, trans0, trans1, trans2}`（orig 用 `_p{N}`、aug 用 `_person{N}` 后缀 → 用 `family_key()` 规范化到同一 key，否则家族会被拆开）。E199 有 4 个 case 无可行 aug（部分家族）→ 按实际存在臂判定，存在臂全过窄即视为一致，缺失臂在 note 标注。
- **orig 也做家族检查**（2026-08-17 校验后修正）：orig 过窄但家族不一致 → L3-review。校验发现 4 个假收全是「指标全 14 门干净通过但人工判废」的 orig，且家族均不一致；施加家族检查后本标注集假收 4→0。

## Claims

- **C0 分类器 + 不变量**：14 门双口径分类器实现，阈值集中于 `FunnelConfig`（硬门/带门/宽窄单一真源，不散落）；对 E199 332 行断言 `narrow⟹wide` 对全 10 带门 100% 成立。verify：`--assert-monotonic` 0 违反。
- **C1 分层定量（sizing）**：对 E199 332 行输出逐条分层 + 计数汇总（aug vs orig，逐物体 box001/004/021/023/024）；计数和 = 332。verify：summary 表闭合。
- **C2 人工负荷压缩**：报告自动决策 `|L1|+|L3-auto|` 占比。**目标 ≥60% 自动决策**（相对过去 100% 逐条人工的显著下降）；不达标则**报告 + 回带宽调参数（不静默上线）**。verify：占比数值 + 各层明细。
- **C3 家族逻辑正确**：逐例核对 ≥3 个家族——全 4 臂过窄→auto、混合家族→review、部分家族→按存在臂判定。verify：逐例列出。
- **C4 视觉复核（强制，rule 9）**：各层随机抽样（L1 弃 ≥3、L3-auto ≥3、L2/L3-review ≥3），用 viser / `/video-frames` 渲染关键帧核对——自动收无穿模/漂浮/跌倒，自动弃确为坏 case，专项抽查 body_z-触发的 L1 弃。观察写入 log，**不得留空**。
- **C5 可扩展 E200**：分类器读任意 `case_metrics.tsv`（E199/E200 两 arm）；E200 CEM 跑完即可复用同 funnel（阈值不变）。verify：干跑 E200 case_metrics（若已产出）或接口 + 单测占位。
- **C6 人工复审闭环**：`L2 + L3-review` 子集导出为 review_player 可加载清单（含 Agent 参考列占位），人工只看该子集。verify：review_player 载入 filtered 清单成功。

**判定**：C0 不变量 0 违反 + C1 计数闭合 + C2 自动决策 ≥60% + C4 视觉无系统性误判 → 漏斗上线，用于 E199/E200 下游 RL 数据交付；否则按 C2/C4 反馈调宽带或门集，不静默交付。

## 改动文件（全新增，隔离，不动 E198/E199/E200 既有脚本，rule 3）

| 文件 | 改动 |
|---|---|
| `scripts/experiments/E201/funnel_config.py` | `FunnelConfig`：`HARD_GATES`(4)、`BANDED_GATES`(10，每门 field/op/narrow/wide)、家族定义、层判定顺序。集中单一真源；`assert_monotonic()` helper（narrow⟹wide）。 |
| `scripts/experiments/E201/classify_funnel.py` | 核心分类器：读 case_metrics.tsv（+ manifest 供 body_z recompute，import 复用 `gen_E199_fullscale_gate_xlsx` 的 `body_z_p95`/`orig_paths`/`actor_index`）→ 每 rollout 算硬门 + 带门(宽/窄) → 分层 + 家族一致性 → 输出逐条 TSV(`layer` / 每口径 failed gates / `family_flag` / note)。启动断言 `narrow⟹wide`。 |
| `scripts/eval/reports/gen_E201_funnel_xlsx.py` | 漏斗 workbook：`detail` sheet(逐 rollout：14 门 宽/窄 值 + PASS/FAIL + `layer` + family) + `summary` sheet(各层计数 aug/orig/逐物体 + 自动决策占比)。复用 gen_E199 的 openpyxl 布局/配色/`freeze_panes`。 |
| `scripts/experiments/E201/export_review_queue.py` | 从分类结果导出 `L2 + L3-review` 子集为 review_player 清单（`case_id#arm` + `layer` + Agent 参考列占位）。 |
| `scripts/eval/review/review_index.py` | 新增按 E201 funnel 待审清单过滤的 source（隔离，不动 E199/E199P/E200 现有 key），只呈现待审子集。 |
| `scripts/eval/wrappers/eval_E201_funnel.sh` | 入口：classify → gen xlsx → export review queue，参数 `--exp E199 | E200-prg_g1a2 | E200-noprg`。 |
| `workspace/core4d/docs/数据筛选漏斗构建.md` | 定稿：补齐宽/窄阈值表（含 0.40 修正）、硬门/带门划分、body_z 归属决策、三级判定伪码、家族定义、单一真源指向 `FunnelConfig`。 |

结果落盘：`workspace/core4d/results/E201/funnel/`（`E199_funnel_rollout.tsv`、`E201_E199_funnel.xlsx`、`review_queue_E199.tsv`）。

> **Agent 初审组件**：具体实现见下方「**补充计划 A · L2/L3-review 的 VLM 全量初审**」（Qwen3-VL-235B via ecodata2 API，2fps 抽帧）。已从「仅占接口」升级为可执行方案。

## 执行命令（本轮不跑，占位；批准并实现后执行）

```bash
# 1) 分类 + 报告 + 导出待审队列（E199）
bash workspace/core4d/scripts/eval/wrappers/eval_E201_funnel.sh --exp E199
#   → results/E201/funnel/E199_funnel_rollout.tsv + E201_E199_funnel.xlsx + review_queue_E199.tsv

# 2) 不变量 + 计数自检
.venv/bin/python workspace/core4d/scripts/experiments/E201/classify_funnel.py --exp E199 --assert-monotonic

# 3) 视觉复核（各层抽样，载入 review_queue 子集）
bash workspace/core4d/scripts/eval/wrappers/review_player.sh E199 --port 8080

# 4) E200 跑完后复用同 funnel
bash workspace/core4d/scripts/eval/wrappers/eval_E201_funnel.sh --exp E200-prg_g1a2
```

## 成功标准（量化）

1. C0：`narrow⟹wide` 断言 0 违反（全 10 带门 × 332 行）。
2. C1：332 行分层计数闭合，逐物体 + aug/orig 分层表。
3. C2：自动决策 `L1+L3-auto` 占比 ≥60%，含中间带/家族-review 明细。
4. C3：≥3 家族逐例核对家族逻辑（含部分家族）。
5. C4：各层抽样 ≥3 视觉复核，观察入 log，自动收/自动弃无系统性误判。
6. C6：review_player 载入待审子集成功。
7. 全过 → 写 log288+ / 更新 TRACKER + progress；定稿 funnel doc；漏斗上线用于 E199/E200 数据交付。

## 风险 / 缓解

| 风险 | 缓解 |
|---|---|
| 中间带过大（人工没省） | tracking +1 已刻意收窄；C2 量化，不达 60% 则报告 + 调物理宽带，不静默上线。 |
| 家族一致性反抬人工 | C1 统计「家族不一致 aug」条数；若过多，评估放宽家族判据（如「多数臂过窄」而非全臂），log 记录。 |
| body_z 归硬门误弃好 case | 标为待复核决策；C4 专项抽查 body_z-触发的 L1 弃，确认确为坏 case，否则改带门。 |
| Agent 初审误导人工 | Agent 显式「参考，无效力」，人工最终拍板；本轮不实现，先占接口。 |
| orig omnirt_v1 vs aug v2 混淆（继承 E199） | funnel 只做单条门判定、不做 orig↔aug 因果对比 → confound 不影响分层；报告沿用 E199 标注。 |
| 阈值散落 / 不可复现 | 全阈值集中 `FunnelConfig`（rule 3）；classifier / xlsx / doc 单一真源，doc 只引用不复制数值。 |

## 已定（用户 2026-08-17 确认，三项均保持默认）

1. **body_z 归硬门**（≤0.20，全层强制，无宽/窄缓冲）。
2. **C2 目标 = 自动决策 ≥60%**（保持比例目标，不改绝对条数）。
3. **家族判据 = 「存在臂全过窄」即一致**，部分家族（缺 aug 臂）**不加**额外约束（不要求最少臂数）。

---

# 补充计划 A · L2/L3-review 的 VLM 全量初审（Qwen3-VL-235B · ecodata2 API）

_2026-08-17 追加 · 将 plan231 原「Agent 初审仅占接口」升级为具体实现。状态：计划态待批准。_

## 目标

对落入 **L2_review + L3_review** 的全部待审 rollout（E199 当前 = 49 + 81 = **130 条**）**全量调用 VLM 初审**：抽渲染帧按时间顺序喂给 `Qwen3-VL-235B-A22B-Instruct`，产出**结构化推荐（参考，无效力，人工最终拍板）**——给每条一个 `use_decision + quality_label + failure_taxonomy + 观察`。目的是给人工复审提供先验、排序、聚焦，减轻 130 条的逐条盲审负担。可无改动扩到 E200。

**定位不变**：VLM 只做「初审参考」，不改变漏斗的 L1/L3-auto 自动决策，也不替代人工对 L2/L3 的最终拍板（rule：Agent 无效力）。

## 调用栈（已核实）

- **API 入口**：`call_api_imitate_redaccel.py`（`ecodata2` 库已装）。**约束（用户 2026-08-17）**：① 该文件当**纯黑盒**——只产出输入 JSONL / 消费输出 JSONL，**绝不修改**（若确需改先征得用户同意）；② 其路径**不写死**——通过 wrapper 的 `--call-api-path` 或环境变量 `ECODATA_CALL_API`（默认可给当前已知路径作 fallback，但不硬编码进 py 逻辑）在用时指定。
- **模型**：`Qwen3-VL-235B-A22B-Instruct`（须在 `ecodata.authentication.apis.MODEL_INFO` 中；build 前先 `python -c "from ecodata.authentication.apis import MODEL_INFO; print('Qwen3-VL-235B-A22B-Instruct' in MODEL_INFO)"` 断言存在）。
- **请求格式**（每行一条 `request_info`）：
  - `images`: 该 rollout 的帧路径列表，**时间顺序**（2fps 抽帧）。
  - `messages`: `[{"role":"user","content": <prompt> + 与帧数等量的 " <image> " token（按序）}, {"role":"assistant","content":""}]`。末条 assistant 是 GT 占位（API 会剥离；messages 必须偶数条）。`<image>` token 按出现顺序 popleft 消费 `images`。
  - `metadata`: `{exp, case_id, variant, layer, family_flag, failed_gates, object_key, n_frames}`（API 原样透传到输出，便于回填）。
- **执行**：`python call_api_imitate_redaccel.py --data_path <jsonl> --output_dir results/E201/vlm_review/out/{exp} --model_name Qwen3-VL-235B-A22B-Instruct --num_proc 16 --image_size 512`。产 `out/{exp}/0/generate_predictions.jsonl`（`predict[0]`=模型输出）+ `0/error_requests.jsonl`。

## 管线（5 步，全新增脚本，隔离）

1. **选件** `select_review_queue.py`：从 `results/E201/funnel/{exp}_funnel_rollout.tsv` 取 `layer ∈ {L2_review, L3_review}` → 待审清单（含 qpos_path/scene_xml/layer/family_flag/wide_failed/narrow_failed）。join case_metrics 补 `qpos_frames`/`duration_s`。
2. **渲染 + 抽帧（5fps）+ 存全帧 mp4** `render_frames_for_vlm.py`：复用 `experiments/E199/render_qc.py` 的 `mujoco.Renderer` offscreen 渲染逻辑（同相机 640×480）。一次渲全帧 → 写 `video.mp4`（native fps，供人工肉眼复看）+ 按 native_fps=`qpos_frames/duration_s` 计步长 `round(native_fps/5)` 抽 **5fps** 帧（封顶 32，超则均匀下采样）存 `results/E201/vlm_review/frames/{exp}/{case_id}#{variant}/f000.jpg…` + `frames.json`（帧号/时间戳/mp4 路径）。**mp.Pool 多进程并发**（每 worker 独立 EGL 上下文，fork-safe；`--num-proc` 默认 8）。headless EGL。
3. **建请求** `build_vlm_requests.py`：每条 rollout 生成 1 行 request_info（上述格式）；prompt 从 `vlm_review_prompt.txt` 单一真源读取；`<image>` token 数 == 帧数。写 `results/E201/vlm_review/requests/{exp}.jsonl`。
4. **调 API**：上述命令，全量 130 条（E199）。
5. **解析回填** `parse_vlm_verdicts.py`：读 `generate_predictions.jsonl`，从 `predict[0]` 抽严格 JSON（容错：正则兜底 + 解析失败标 `vlm_parse_error`），按 metadata join → 写 `results/E201/vlm_review/verdicts/{exp}_vlm_verdicts.tsv`（`vlm_use / vlm_quality / vlm_failure / vlm_note`）+ 合并进 `review_queue_{exp}.tsv` 的 Agent 参考列，供 review_player 显示。

## Prompt 草案（`vlm_review_prompt.txt`，单一真源）

> 你是人形机器人「人-物操作」重定向数据的质检专家。下面按**时间先后顺序**给你 N 张渲染帧（每秒 2 帧），来自一段物理仿真回放：一台 Unitree G1 人形机器人搬运一个箱子/桶。请判断这段重定向轨迹是否**物理可信、可用于下游强化学习训练**。
>
> 请重点检查以下失败模式（尤其是数值指标难以捕捉的视觉问题）：
> 1. **穿透（penetration）**：手、手指、身体或腿穿进箱子/桶内部，或穿入地面。
> 2. **漂浮（floating）**：物体悬空却无手支撑；或机器人脚离地悬浮、无支撑地滑行。
> 3. **抖动/不自然（jitter/unnatural）**：肢体高频抖动、关节反关节或扭曲、姿态明显不自然。
> 4. **抓握/接触（grasp/contact）**：手是否真正贴合并抓住物体；接触点是否合理，还是"隔空搬运"。
> 5. **跌倒/失稳（fall/instability）**：机器人失去平衡、跌倒或明显踉跄。
> 6. **任务完成（task）**：物体是否被合理地拿起/搬运/放下，动作是否连贯完整。
>
> 判断原则：这是**初审参考**，最终由人工拍板；因此**宁可标出可疑，也不要漏报穿透/漂浮/抖动**。逐帧观察其变化（例如某帧开始手插入箱体、某帧物体突然弹飞）。
>
> 只输出一个 JSON 对象，不要输出任何其它文字：
> ```json
> {
>   "use_decision": "USE | DO_NOT_USE",
>   "quality_label": "CLEAN | MINOR_ACCEPTABLE | MAJOR_DEFECT | UNUSABLE",
>   "failure_taxonomy": ["penetration" | "floating" | "jitter" | "bad_grasp" | "fall" | "task_incomplete" | ...],
>   "worst_frames": [出问题最明显的帧序号列表],
>   "overall_note": "一句话总体判断依据（中文）"
> }
> ```
> 标签口径：CLEAN=无可见缺陷可直接用；MINOR_ACCEPTABLE=有轻微瑕疵但可用；MAJOR_DEFECT=有明显缺陷不建议用；UNUSABLE=严重失败完全不可用。use_decision 与 quality_label 需自洽（CLEAN/MINOR_ACCEPTABLE→USE；MAJOR_DEFECT/UNUSABLE→DO_NOT_USE）。

> 输出的 enum 词表**刻意对齐**用户手工标注（`manual_use_decision` = USE/DO_NOT_USE，`manual_quality_label` = CLEAN/MINOR_ACCEPTABLE/MAJOR_DEFECT/UNUSABLE），使 VLM↔人工可直接混淆矩阵对比。

## 校验（把 VLM 当分类器，用 `user_manual_review_filled.tsv` 做 ground truth）

- 69 条带变体人工标注里，修复后有 **42 条落在 L2/L3_review**（human-deferred）且已有人工 USE/DO_NOT_USE 标签 → 直接算 **VLM vs 人工** 的一致率 + 混淆矩阵。
- **重点指标**：VLM「假收」= VLM 判 USE 但人工 DO_NOT_USE（会误导人工放行坏数据）——这是 VLM 初审最该压低的错误；同时报一致率、假弃、quality_label 相关性。
- 因 VLM 无效力，**不设硬门**；校验产出作为「VLM 初审可信度」一等结论写入 log，指导是否/如何在人工侧展示 VLM 建议（如仅作排序，或高置信 DO_NOT_USE 置顶）。

## 补充改动文件（全新增，隔离）

| 文件 | 改动 |
|---|---|
| `scripts/experiments/E201/select_review_queue.py` | 从 funnel_rollout 取 L2/L3_review + join case_metrics（qpos_frames/duration_s）。 |
| `scripts/experiments/E201/render_frames_for_vlm.py` | 复用 render_qc 渲染，2fps 抽帧到 `vlm_review/frames/`，写帧 sidecar。 |
| `scripts/experiments/E201/build_vlm_requests.py` | 组 JSONL request_info（images 按序 + prompt + `<image>` token + metadata + 空 assistant）。 |
| `scripts/experiments/E201/vlm_review_prompt.txt` | 上述 prompt 单一真源。 |
| `scripts/experiments/E201/parse_vlm_verdicts.py` | 解析 predictions → verdict TSV + 回填 review_queue Agent 列 + 对 42 条标注跑校验混淆矩阵。 |
| `scripts/eval/wrappers/run_E201_vlm_review.sh` | 串 select→render→build→call_api→parse，参数 `--exp E199 | E200-*`，`--image-size`/`--num-proc`/`--fps` 可调。 |
| 结果落盘 | `results/E201/vlm_review/{frames,requests,out,verdicts}/{exp}/`。 |

## 补充 Claims

- **C7 全量初审闭合**：130 条 L2/L3_review 全部产出 VLM verdict（error/parse_fail 计数明示，不静默丢）；每条含 use/quality/failure/worst_frames/note。
- **C8 抽帧正确**：2fps 抽帧步长 = `round((qpos_frames/duration_s)/2)`，帧数与 `<image>` token 数一致；随机抽 ≥3 条肉眼核对帧顺序/内容正确。
- **C9 VLM 可信度**：对 42 条有人工标注的 L2/L3 rollout 报 VLM↔人工一致率 + 混淆矩阵，**重点报 VLM 假收数**；结论写 log 指导人工侧展示策略。

## 补充已定（用户 2026-08-17 确认）

1. **image_size = 512**（渲染 640×480，API resize 长边到 512）。
2. **帧数上限 = 32**（5fps 抽帧后若 >32，再均匀下采样到 32；即 >6.4s 的序列低于 5fps）。
3. ~~单视角~~ → **双视角**（用户 2026-08-17 改）：单视角 VLM 因 2D 深度歧义把 122/130 误判为 bad_grasp「隔空搬运」。改为每时刻渲两个相机（azimuth 135° + 225°，90° 互补），**左右并排拼成一张**（图片数不变 ≤32）。mp4 也用拼图。
4. **抽帧 5fps** + **保存全帧 mp4**（拼图）。
5. **渲染分片并行**（默认 24，HQ 重跑用 48 shards；EGL 无 NVIDIA ICD → llvmpipe 软件渲染，CPU-bound，192 核不占 GPU），API 侧 num_proc=16。
6. **画质修复（用户 2026-08-18）**：① 抽帧 JPEG quality 8→**95**（原 8/100 = 块噪，这才是"抽帧比视频糊"的真因，mp4 用 libx264 quality=7 正常故清晰）；② **关闭阴影**（默认光源 shadow map 混叠成地面条纹噪声）；③ 每视角分辨率 640×480→**960×720**（拼图 1920×720）；④ **image_size 512→1024**（渲染放大、API resize 保更多细节）。
7. **prompt 增补（用户 2026-08-18）**：新增两类失败——**手背接触箱子**（`back_of_hand`，用手背/腕背而非手掌抓握）、**反关节**（`inverted_joint`，关节反向弯折/超范围）。
8. **双模型同步（用户 2026-08-18）**：`MODELS` 逗号列表并行评审——`Qwen3-VL-235B-A22B-Instruct` + `gemini-3.5-flash-huangxiaoshuang`，per-model 输出 `out/{exp}/{model}/` + `verdicts/{exp}_{model}_vlm_verdicts.tsv`，各自对人工做校验。

## 下一步

- E200 双 arm CEM 跑完 → 同 funnel 分层 + 同 VLM 初审管线，产下游 RL 交付集（三 arm 各一份 auto-accept 清单）。
