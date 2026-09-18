# E214c — Part-wise Penetration 阈值统一消融结果（R303 / plan248 / Phase 72）

- **日期**: 2026-09-18 18:47 起跑 → 2026-09-19 04:21 收口
- **分支**: `experiment/E199-omniretarget-object-augmentation`
- **计划**: [plan/248](../plan/248_E214c_partwise_threshold_ablation_plan.md)
- **前置**: [log/303 E214 四消融](303_E214_core_method_four_ablation.md)

## 1. 概要

补 E214 未覆盖的**贡献2「part-wise 实质」——分部位阈值本身**：把 body/hand/leg 三个 SDF 硬门的
阈值三参数（min_sdf + hard_floor + max_vio）统一为单一口径，软层保持不变（基线软惩罚本就部位统一
2cm/2.0，part-wise 差异只在硬门——设计阶段核查结论，用户确认）。

- **A5 uniform_hand**：body/leg 门 ← 该 case 自己的 hand 阈值（最宽松：θ−10mm，floor−15/−20mm，vio 5/10%；per-case 计算）。
- **A6 uniform_leg**：body/hand 门 ← leg 阈值（最严格：θ+5mm，floor−5mm，vio 2%；全 case 一致）。
- **训练**：100/100 `cem_ok`，**0 失败**。第二段 4.55h + 第一段 ~4.5h（23:22 inode 配额事故中断，见 §8）；wall median 41.8min，max 67.1min（本机 8 卡）。
- **单变量审计**：100/100 **ALL PASS**（`preflight/e214c_single_variable_audit.json`）——每臂只改 6 个目标键（2 门×3 参数，per-case 值与基线 source 门逐字一致），source 门未动，其余键与基线逐字一致。
- **评估**：150/150 序列 `ok`（full 现算 50 + A5 50 + A6 50），同一评估器（`eval.core.core_metrics` + `motion_health` + `fixed_reference_z_metrics`）。**full 现算与 E214 full 列（论文缓存）标准指标 mean±std 逐位一致**（82.3%±10.4% / 58.8%±13.9% / eef 13.07±2.84 …）——消除评估器漂移疑虑。

## 2. Overall 指标表（n=50，mean±std (worst)，附 full-vs-臂配对 Wilcoxon，*=p<0.05）

| metric | dir | full (part-wise) | A5 uniform-hand θ | A6 uniform-leg θ |
|---|---|---|---|---|
| raw contact | ↑ | 82.3%±10.4% | 82.4% p=.954 | 80.9% p=.424 |
| phys contact@3mm | ↑ | **58.8%±13.9%** | 58.2% p=.787 | 59.7% p=.604 |
| phys penetration@3mm | ↓ | 17.4%±9.9% | 18.0% p=.916 | 15.7% p=.251 |
| geom penetration@2mm (hand) | ↓ | 5.0%±3.3% | 6.3% *p=.0495 | 5.0% p=.815 |
| **LEG penetration frac** | ↓ | **1.5%±3.6% (w 15.4%)** | **4.5%±8.3% (w 33.0%) *p=.0013** | **5.5%±8.0% (w 37.4%) *p=7.4e-5** |
| leg-object contact frac | ↓ | 1.6%±3.6% | 4.5% *p=.0014 | 5.5% *p=9e-5 |
| BODY / UPPER-BODY penetration | ↓ | 0.0% | 0.0% | 0.0% |
| root pos err (cm) | ↓ | 13.66±3.37 | 14.38 p=.093 | 14.04 p=.490 |
| eef pos err (cm) | ↓ | 13.07±2.84 | 13.68 p=.059 | 13.52 p=.066 |
| obj pos err (cm) | ↓ | 10.34±2.59 | 10.59 p=.954 | 10.59 p=.169 |
| obj ori err (deg) | ↓ | 5.11±2.68 | 5.43 *p=.0076 | 5.26 p=.215 |
| **fall rate** | ↓ | **0.0%** | **0.0%** | **0.0%** |
| body-z err p95 (m) | ↓ | 0.091 | 0.095 p=.318 | 0.094 p=.333 |
| ankle jerk p95 | ↓ | 654 | 645 p=.924 | 630 p=.135 |
| foot slip max (m) | ↓ | 0.866 | 0.846 p=.372 | 0.844 p=.478 |

完整表（per-object 分层 + 全指标 + 6 臂合并表）：`results/E214c/reports/e214c_partwise_table.md`、`e214c_6arm_table.md`（+ `.tsv`/`.xlsx`）。

## 3. 核心发现

1. **腿-箱净空是分部位阈值的真正价值所在**：full 的 LEG penetration 1.5% 同时显著优于 A5 4.5%（×3，p=.0013）与 A6 5.5%（×3.7，p=7.4e-5）。两个统一方向都以腿穿箱为代价。
2. **A5（放宽腿/身门到手级）→ 腿穿透直接放行**：机制直白——θ 从 +5mm 放到 −10mm、vio 2%→10% 后，软层 2cm margin 压不住的残留腿-箱接触不再被硬门剔除。连带 obj ori 变差（5.11→5.43°，p=.0076：腿碰箱扰动箱体）；hand geom pen 边缘上升（5.0→6.3%，p=.0495，行为漂移二阶效应）。手接触对照轴不变（contact@3mm 58.8→58.2 ns，hand 门未动）——内部对照成立。
3. **A6（收紧手/身门到腿级）→ 手接触没有崩（C2 预期落空），但腿穿透反而显著上升**：手门 θ+5mm 与接触任务本质冲突（抱箱帧全部 φ<5mm）→ 候选几乎全违约 → `least_violation` 回退**常态化接管**精英选择。硬门是**相对筛选器**而非绝对约束：当"违约"成为所有候选的共性，阈值收紧不再淘汰接触行为，只降低精英质量。body 门 +5mm 同理使"箱体贴躯干"常态违约 → 抱箱姿态漂移、下肢代偿靠近箱体 → 腿穿箱（leg 门未动却 ×3.7）。eef/root tracking 边缘变差（p≈.06）。
4. **BODY/UPPER-BODY penetration 全 0、fall 全 0**：躯干在任何口径下都不穿箱（软层 robot_object 2cm 仍生效 + 躯干本就不主动近箱）；统一阈值不引发摔倒——与 A3（去全部硬门 → fall 14%）对照：**防灾难是"硬门存在性"的价值，腿部净空是"阈值分部位"的价值**，两层分工进一步细化。

## 4. Claims 验证

- **C1（A5 → 下肢/身体穿透失控）✅ 确认（腿轴）**：LEG penetration 1.5%→4.5%（×3，*p=.0013，worst 15.4%→33.0%），leg-object contact 同步（*p=.0014），obj ori 连带变差（*p=.0076）。手接触对照轴不变（hand 门未动，ns）✓。body 轴无分化（全 0，见 §3.4）——claim 的"身体"部分为空集，如实记录。
- **C2（A6 → 手接触崩塌）❌ 被证伪（诚实负结果）**：contact@3mm 58.8→59.7%（p=.604），raw contact ns，eef 仅边缘变差（p=.066）。**机制**：过严手门与任务本质冲突时全候选违约 → least_violation 回退常态化 → 阈值失去淘汰力（§3.3）。代价转移到意外轴：**腿穿透 ×3.7（p=7.4e-5）+ tracking 边缘退化**。负结果本身有论文价值：说明硬门阈值的"严格方向"存在有效性边界（由回退机制兜底，不会灾难化，但会行为漂移）。
- **C3（part-wise 必要性）✅ 修正后成立**：原表述"full 两轴同时优"仅腿轴成立（手轴三配置等价）。修正为：**任一口径的统一都以腿-箱净空显著恶化为代价（×3 / ×3.7，均 p<.005），只有分部位阈值能同时保住"手能贴"与"腿净空"**——统一方向的失败模式不同（A5 直接放行 vs A6 回退漂移）但殊途同归到腿穿透。part-wise 设计必要性成立。
- **C4（与 A3/A4 正交互补）✅ 确认**：A3（去全部硬门）→ fall 14% + body-z 0.149（灾难放行）；A4（去软惩罚）→ 温和退化；A5/A6（门在、阈值统一）→ fall 0% 但腿穿透 ×3–3.7。三组消融分离出硬门的两个独立价值维度：**存在性→防灾难**（A3），**分部位阈值→腿部净空**（A5/A6），补全贡献2 的消融矩阵。

**结论：分部位阈值必要（C1/C3/C4 成立），但其作用不对称——"腿/身严格"方向是真实约束（放宽即穿透放行），"手宽松"方向由 least-violation 回退兜底（收紧不破接触、但引发行为漂移与腿穿透）。C2 的预设机制（过严门→手够不到）被证伪，如实报告。**

## 5. 可视化观察（rule 9）

代表 case（按 leg_pen worst 选）：`box001_20231020_014_p1`（A5 33.0% / A6 37.4%，两臂全局最差）、`box021_20231011_034_p1`（抱箱搬运，A5 29.4% / A6 25.9% / full 14.0%）。视频 `results/E214c/render/*.mp4`，关键帧 `results/E214c/render/frames/`（sim 面板腿-箱区域 3× 裁剪放大 `zoom_*`）。渲染为逐帧自适应相机，**跨视频姿态对比无效，同视频内 sim-vs-ref 有效**；三臂同 case 同时间戳的 sim 面板裁剪仅作同场景几何关系参照。

**实际观察**：

- **full（box001_014_p1, t=2.3/3.4s）**：搬运与下蹲放箱相双腿全程在箱体体积外——箱前面板不切腿，放箱时躯干俯于箱顶、两腿分居箱体两侧外侧，姿态干净，与 ref 同构。
- **A5 uniform-hand（同 case t=2.3s）**：箱体前面板横切双腿大腿-膝部区域（full 中腿完整遮箱面，A5 中箱面遮腿中段）——腿已进入箱体体积，即 leg_pen 指标（33%）的几何实体。
- **A5 uniform-hand（t=3.4s）**：**决定性帧**——放箱相姿态崩塌：躯干近水平趴倒于箱顶面，右腿自箱顶面穿入、小腿/膝自箱右侧面穿出（腿贯穿箱体体积），左腿膝部亦埋入箱前面板；与 full 同相位的干净下蹲形成鲜明对照。无摔倒（fall 0% 一致），但姿态明显非物理合理。
- **A6 uniform-leg（t=2.3s）**：搬运相箱面板切大腿中段（介于 full 与 A5 之间）；**t=3.4s**：左小腿/踝段没入箱前面板（踝部橙色接触标记出现在箱面上），右腿在箱外——腿穿透可见（leg_pen 37.4%）。**手仍正常扶箱**（与 C2 证伪一致：接触未崩）。
- **box021_034_p1（抱箱搬运, t=2.0s）**：full 腿与箱侧有明显净空；A5 箱左面已贴/切右大腿-膝部（净空消失）。A6 视频渲染完成后同法复核（`render/E214c_box021_20231011_034_p1__A6_uniform_leg.mp4`）。
- 三臂均无摔倒、无悬空、无明显抖动；差异集中在**腿-箱几何关系与放箱相姿态**，与量化（LEG penetration ×3–3.7、obj ori A5 变差、eef 边缘变差）互相印证。

**结论（视觉）**：A5 的腿/身穿透与放箱相姿态崩塌肉眼可见（C1 视觉确认）；A6 手接触保持但左腿穿箱（C2 证伪 + 意外轴视觉确认）；full 两相位均干净。

## 6. 改动文件

| 文件 | 说明 |
|---|---|
| `tmp/E214c_case_id.txt` | 50 case（=E214，contract 断言相同） |
| `workspace/core4d/scripts/experiments/E214c/` | e214c_common（per-case toggle 引擎）/ build_manifest / run_ablation_cem / audit_single_variable / test_e214c_contract / snapshot_E214c_scenes.sh / smoke_E214c.sh / render_E214c.py / watchdog_keepalive_E214c.sh |
| `workspace/core4d/scripts/eval/runners/eval_E214c_ablation.py`、`wrappers/eval_E214c_ablation.sh`、`reports/gen_E214c_ablation_table.py` | 评估（full 现算+A5/A6，逐部位穿透指标）+ 出表（主表+6臂） |
| `workspace/core4d/scripts/launch/active/run_E214c_local_8gpu.sh` | 本机 8 卡入口（snapshot→contract→manifest→dispatch→audit） |
| `workspace/core4d/results/E214c/scene_snapshot/` | 50 case 基线 scene 快照 + manifest（git HEAD + sha256，rule 7） |
| **核心代码零改动** | `examples/run_mjwp.py`/`spider/config.py` 未动；仅翻既有 config 键（smoke+audit 验证） |

## 7. 结果路径

- 消融 npz + config_act：`workspace/core4d/results/E214c/cem/E214c_{case}__{A5_uniform_hand,A6_uniform_leg}/`
- manifest：`results/E214c/manifests/e214c_partwise_cem.tsv`（100 行全 cem_ok）
- 指标：`results/E214c/eval/e214c_metrics.{jsonl,tsv}`（150 序列）
- 报告表：`results/E214c/reports/e214c_partwise_table.{md,tsv,xlsx}` + `e214c_6arm_table.md`
- 审计：`results/E214c/preflight/e214c_single_variable_audit.json`（ALL PASS）
- 训练日志：`logs/E214c/run.log`（第一段+事故）、`run2.log`（第二段）、`eval.log`、`watchdog.log`、`smoke_*.log`
- 渲染：`results/E214c/render/`

## 8. 已知 caveat / 事故记录

1. **inode 配额事故（23:22）**：spider 子树 JuiceFS 目录 inode 配额打满（95%→顶，空间仅 1%）→ runner 写日志 `Errno 122` 崩溃 → 链式 keepalive 误触发抢卡。恢复：kill keepalive+孤儿、重置 7 条僵尸行、绕过 manifest 重建直接续跑（6 条已写盘任务经自检修复为 cem_ok）；用户清理后 inode 95%→19%；keepalive 链加 rc=0 守卫 + watchdog 双保险。**教训：长跑前查 `df -i`；收尾链必须带 rc 守卫。**
2. **基线协议**：full 列为基线 rollout **现算**（与 A5/A6 同评估器同代码），与 E214 论文缓存 full 逐位一致——本表无 E214 的"缓存异构"caveat；但 full rollout 本身仍是论文人工择优产物（与 A5/A6 单次重跑的协议不对称保留，结论以配对退化方向为准）。
3. **代码版本 confound**：基线 rollout 旧代码生成、A5/A6 当前代码（`object_distance_backend=legacy_box` 复现旧语义），二阶。
4. **per-case hand 阈值两变体**：A5 在 E198 组（box001/004/024，13 case）用 floor−15mm/vio5%，其余 37 case 用 floor−20mm/vio10%；per-case 自洽拷贝，audit 逐 case 验证。
5. **C2 证伪的机制归因**（回退常态化）基于门公式与结果模式的推断，未逐迭代 dump 精英合法率；如需坐实可加 gate 统计日志复跑单 case。

## 9. 下一步

- 论文消融小节：贡献2 的消融矩阵由 4 臂扩为 6 臂（`e214c_6arm_table.md`），part-wise 阈值必要性用腿穿透轴陈述；A6 负结果可写入"硬门有效性边界"讨论（回退兜底）。
- 可选机理实验：单 case 加 per-iteration gate 合法率统计，坐实 A6 回退常态化推断。
