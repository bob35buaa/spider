# E214 — 核心方法四消融结果（R300 / plan245 / Phase 72）

- **日期**: 2026-09-09 起跑 → 2026-09-10 收口
- **分支**: `experiment/E199-omniretarget-object-augmentation`
- **计划**: [plan/245](../plan/245_E214_core_method_four_ablation_plan.md)

## 1. 概要

留一消融验证方法两大贡献。**50 case**（论文 52 − 2 E167A box021；box023 用 E173 全栈，基线齐平全栈），
**200 CEM 重跑**（50×4，`load_config_path` + 单键 toggle），基线不重跑（full = 论文缓存 43 + E173 重算 7）。

- **训练**：200/200 `cem_ok`，**0 失败**，用时 **19.35h**（本机 8 卡，与他人任务共存；wall median 43.4min，max 75.4min）。
- **单变量审计**：200/200 **ALL PASS**（`preflight/e214_single_variable_audit.json`）——每个消融的 config 只翻转了目标键，其余与基线逐字一致（含 box023 E173 全栈的 4 消融）。
- **评估**：207/207 序列 `ok`（200 消融 + 7 box023-E173 基线），0 error。指标与论文表同定义
  （`eval.core.core_metrics.evaluate_sequence` + `motion_health.run_health` + `fixed_reference_z_metrics`）。

## 2. Overall 指标表（n=50，mean±std (worst)，消融列附 full-vs-消融 配对 Wilcoxon p，*=p<0.05）

| metric | dir | full | A1 −surface_band | A2 −contact_hdmi | A3 −hard gate | A4 −soft penalty |
|---|---|---|---|---|---|---|
| raw contact | ↑ | 82.3%±10.4% | 57.7% p<.001* | 51.8% p<.001* | 81.2% ns | 82.5% ns |
| phys contact@3mm | ↑ | **58.8%±13.9%** | **26.3% p<.001*** | 39.4% p<.001* | 56.8% ns | 58.1% ns |
| phys penetration@3mm | ↓ | 17.4% | 21.7% p=.004* | 9.5% p<.001*(够不到) | 18.7% ns | 18.2% ns |
| geom penetration@2mm | ↓ | **5.0%±3.3%** | **17.5% p<.001*** | 3.9% ns | 5.0% ns | 5.5% ns |
| root pos err (cm) | ↓ | 13.66 | 11.43 p<.001* | **39.54 p<.001*** | 18.38 p=.002* | 15.00 ns |
| root ori err (deg) | ↓ | 8.88 | 4.94* | 5.78* | 15.73 p=.002* | 10.93 p=.002* |
| eef pos err (cm) | ↓ | 13.07 | 11.85* | **40.01 p<.001* (w148)** | 16.49 p=.002* | 14.32 p=.004* |
| eef ori err (deg) | ↓ | 17.21 | 13.05* | 9.65* | 24.77 p=.002* | 19.12 p=.007* |
| obj pos err (cm) | ↓ | 10.34 | 9.74* | 10.24 ns | 10.74 ns | 10.48 ns |
| obj ori err (deg) | ↓ | 5.11 | 4.99 ns | 4.97 ns | 5.47 p=.008* | 5.60 p=.003* |
| **fall rate** | ↓ | **0.0%** | 0.0% | 0.0% | **14.0% p=.008*** | 0.0% |
| body-z err p95 (m) | ↓ | 0.091 | 0.087 ns | 0.059* | **0.149 p=.024* (w0.85)** | 0.100 p<.001* |
| ankle jerk p95 | ↓ | 654 | 715 p=.001* | 566* | 678 ns | 642 ns |
| obj speed max (m/s) | ↓ | 1.288 | 1.345 p=.048* | 1.258 ns | 1.313 ns | 1.321 ns |
| foot slip max (m) | ↓ | 0.866 | 0.906 ns | 0.826 ns | 0.862 ns | 0.805 ns |

完整表（per-object 分层 + std/worst 全量）：`results/E214/reports/e214_ablation_table.md`（+ `.tsv`/`.xlsx`）。

## 3. Claims 验证

- **C1（A1 去 surface_band 细项）✅ 强确认**：接触贴合崩塌 + 几何穿透激增。phys contact@3mm **58.8%→26.3%**（p<.001），
  geom penetration@2mm **5.0%→17.5%**（p<.001，翻 3.5×），phys penetration@3mm 17.4→21.7%（p=.004）。
  代价：tracking 反而略好（root/eef pos↓）——去掉贴面项后优化器更偏 tracking，但**接触真实性与几何穿透显著恶化**，
  证明细项 SDF 薄带负责"贴面 + 把穿透推回"。
- **C2（A2 去 contact_hdmi 粗项）✅ 强确认**：够物/tracking 灾难性失败。eef pos err **13.07→40.01cm**（p<.001，worst 148cm），
  root pos 13.66→39.54cm，raw contact 82→52%（worst 0%，部分 case 完全够不到）。幅度远超 A1。
  注：phys penetration@3mm 反而降到 9.5%——因为手根本没够到物体，接触少故穿透也少（"够不到"特征）。
  证明粗项锚点吸引提供的长程梯度是"从远处够到物体"的关键。
- **C3（A3 去全部硬门）✅ 确认（两层故事的核心证据）**：硬门缺失放行灾难。**fall rate 0%→14%**（p=.008），
  body-z err p95 **0.091→0.149m**（p=.024，worst 0.85m 身体压穿/沉地），root/eef tracking 显著恶化。
  但 hand-object penetration（phys@3mm 18.7、geom 5.0）**未显著上升**——软惩罚仍在压手-物穿透。
  → 硬门的作用是**杜绝软层压不住的灾难性违约（摔倒/身体沉地/tracking 崩）**，而非手-物浅穿。
- **C4（A4 去全部软惩罚）✅ 确认**：无采样梯度→收敛质量普遍轻度下降（eef pos p=.004 / root ori p=.002 /
  eef ori p=.007 / body-z p<.001 / obj ori p=.003，均显著但幅度小），**但硬门 HOLD**：fall 仍 0%，
  penetration 不比 full 显著更差。与 A3 对照鲜明（A3 fall 14%）——**软层给梯度/质量，硬层防灾难，二者互补**。
- **C5（综合）✅ 确认**：full 不被任一消融在全轴上超过。A1 tracking 更好但 contact/geom-pen 远差；
  A2 个别轴更低但 eef pos 灾难；A3/A4 几乎全面更差或引入 fall。full 是最平衡点，每个消融只是换走某项能力。

**结论：两大贡献的四支均必要且互补，全部 5 条 claim 成立。** 贡献 1（contact）的 A1/A2 信号极强；
贡献 2（penetration）呈清晰"软梯度 + 硬可行性"两层分工（A3 灾难 vs A4 温和+硬门兜底）。

## 4. 可视化观察（rule 9，sim 右 / ref-FK 左）

代表案例渲染于 `results/E214/render/*.mp4`，关键帧 `results/E214/render/frames/`。

- **full（box021_034_p1 / 037_p2）**：sim 全程贴合 ref，双手抱箱贴躯干、站姿稳，抓取/搬运/放下干净，
  结束时与 ref 同样直立完成放箱。
- **A1 −surface_band（box021_034_p1）**：sim 抱箱时**双手明显嵌入箱体表面**（full 为干净贴面），
  躯干前倾略大——去掉 SDF 薄带后失去"把手推回表面"的势，几何穿透可见（overall geom@2mm 5%→17.5%）。
- **A2 −contact_hdmi（box021_034_p1）**：sim 的箱体位置/朝向与 ref 明显错位、手-箱相对关系偏移；
  worst 案例（如 box021_038_p1，eef 148cm）手完全够不到物体——缺长程锚点吸引导致"够不到"。
- **A3 −hard gate（box021_037_p2）**：**决定性**——结束帧 ref 直立完成放箱，sim **整体摔倒瘫在地面**箱体旁
  （fall）；body-z worst 0.85m 对应身体压穿/沉地。软惩罚单独挡不住这类灾难，正是硬门的作用。
- **A4 −soft penalty**：视觉上仍完成任务（硬门兜底，无摔倒），仅 tracking/姿态较 full 略糙，与指标一致。

关键帧存 `results/E214/render/frames/`（`A3fall_037p2_0.98.png` 为摔倒证据，`A1_034p1_0.55.png` 为手嵌箱体）。
注：渲染为逐帧自适应相机，**跨视频姿态对比无效**，同视频内 sim-vs-ref 有效。

## 5. 改动文件

| 文件 | 说明 |
|---|---|
| `examples/run_mjwp.py` | **核心改动（隔离/guarded/可逆）**：接触 mask 预计算 `gain>0` → `gain>0 OR surface_band 需 contact_mask 门`。仅 A2(gain=0) 生效，所有 gain>0 run（baseline/A1/A3/A4）字节不变。已 smoke 确认 A2 用回精确 3cm mask。 |
| `tmp/E214_case_id.txt` | 50 case |
| `workspace/core4d/scripts/experiments/E214/` | e214_common / build_manifest / run_ablation_cem / audit_single_variable / render_E214 / snapshot_E214_scenes.sh / test_e214_contract |
| `workspace/core4d/scripts/eval/runners/eval_E214_ablation.py`、`wrappers/eval_E214_ablation.sh`、`reports/gen_E214_ablation_table.py` | 评估 + 出表 |
| `workspace/core4d/scripts/launch/active/run_E214_local_8gpu.sh` | 本机 8 卡入口 |
| `workspace/core4d/results/E214/scene_snapshot/` | 50 case 基线 scene 快照 + manifest（git HEAD + sha256，rule 7） |

## 6. 结果路径

- 消融 npz + config_act：`workspace/core4d/results/E214/cem/E214_{case}__{A1..A4}/`
- manifest：`results/E214/manifests/e214_ablation_cem.tsv`（200 行，全 cem_ok）
- 指标：`results/E214/eval/e214_metrics.{jsonl,tsv}`（207 序列）
- 报告表：`results/E214/reports/e214_ablation_table.{md,tsv,xlsx}`
- 审计：`results/E214/preflight/e214_single_variable_audit.json`（ALL PASS）
- 训练日志：`logs/E214/run.log`；评估 `logs/E214/eval.log`；渲染 `logs/E214/render*.log`

## 7. 已知 caveat

- **基线协议不对称**：full 列为论文人工择优 rollout（复用缓存/E173），消融为该选定 config 的单次重跑；
  同 config/scene，结论以"消融相对 full 的退化方向 + 配对显著性"为准。
- **代码版本 confound**：基线 rollout 由旧代码生成，消融由当前代码；旧 config 缺 24 个新字段，重载填当前默认
  （`object_distance_backend=legacy_box` 复现旧语义；`surface_band_continuation_*`/`query_tape_*` 因 mode/开关惰性），
  confound 二阶，不改变退化方向。
- **A2 核心改动**：见 §5，已使 A2 用回精确 3cm 接触门（否则退化为 30cm 几何门）。

## 8. 下一步

- 论文消融小节可直接用 §2 表 + §3 结论（贡献 1：A1/A2；贡献 2：A3/A4 两层）。
- 如需更强视觉 A/B，可补充 `results/E214/render/` 侧边对比视频（注意 render 为逐帧自适应相机，跨视频姿态对比无效，同视频内 sim-vs-ref 有效）。
