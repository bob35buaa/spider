# CORE4D 当前进度

## 归档索引

- [E192 完整执行与收尾备份](progress_archive/E192_full_backup_20260810.md)
- [E189 完整执行记录](progress_archive/E189_full_backup_20260807.md)
- [E187 Full 至人工比较完整备份](progress_archive/E187_full_manual_comparison_20260803_full_backup.md)
- [E186 production P/R/G 完整备份](progress_archive/E186_production_prg_20260802_full_backup.md)
- 更早阶段见 `progress_archive/`。

## 当前：E195/E192 已收尾；E194 G1 扩展与 E193 计划态

### 2026-08-10 · E194 G1 扩展计划（box001/box023/box021）

- 2026-08-10 用户已批准按 plan222 执行：本机 RTX 5090 GPU0 + 远程
  RTX 6000 Ada GPU0/GPU1 三 worker 并行，完整目标仍为 72 条新增 G1 Full。
- 启动前外部状态：本机 GPU0 空闲（401 MiB、0%）；远程两卡各约 7–8 GiB、
  67–68% 利用率且约 40–41 GiB 空闲，存在既有任务。按用户要求采用叠加运行，
  不 kill、不抢占、不修改既有 tmux；预期远程吞吐会受现有负载影响。
- 远程工作树包含独立 collab-retarget 未提交改动；E194 expansion 将只同步精确
  allowlist 文件与 72-case runtime assets，不执行 `git pull/reset`，不覆盖远程用户改动。
- 用户要求保持实验编号 `E194`，将已验证有效的 G1 gravcomp 扩展到
  `box001`、`box023`、`box021` 的全部 case；当前仅编写计划，不启动执行。
- 已按 `experiment-planning-zh` 恢复 Tracker、最新计划/日志与当前 progress；
  已知 E194 原 G1 覆盖 box024 9 例和 box004 6 例，扩展计划需复用既有 G1
  配置并保留原结果，不重跑 G2/G3。
- 下一动作：核对 E194 原计划/结果、目标物体完整 case inventory、既有脚本与
  baseline 来源，然后创建新的递增编号 plan 文件并做文档一致性审计。
- 原计划与结果已复核：G1 的冻结变量为 `gravcomp=1`、平移 kp=500、旋转
  kp=50、E167A_zOnlyBody、PRG、ref_fk、rubber_hull、seed 0、`1024×32`；
  G2/G3 已 15/15 发散，因此本次只扩 G1，禁止顺带重跑硬伺服臂。
- 目标 Full inventory 已初步闭合为 box001 28 + box023 16 + box021 28 =
  **72 个新增 G1 case**；box001/box023 的 paired A0 来自 E173，box021 的
  PRG baseline authority 需兼容 E170 production 与 E169 reuse 两种 provenance。
- 新计划编号应为 `222`；保持实验 ID `E194`，并用 expansion 专属 manifest、
  输出命名与报告增量合并，避免覆盖原 15-case 三臂结果。
- 已逐项核对 72-case authority：box001/box023 分别从 E173 Full manifest 取
  28/16 条；box021 从 `scripts/experiments/E170/variants.tsv` 取 28 条 PRG
  canonical row（其中存在 E170 production 与 E169 reuse，必须逐行保留）。
- 原 E194 builder/launcher 把 `CASES=15`、`RUN_ARMS=G1/G2/G3`、Full=45
  硬编码，扩展实现不能原地覆盖这些合同；计划采用单独 expansion contract/
  manifest/launcher，并让评测按 `(source_exp, object, arm, case)` 合并旧新证据。
- 既有 G1 参照结果：box024/box004 的 position error 分别
  `13.642→11.533 cm`、`12.011→10.629 cm`，3mm contact 均未退化；扩展
  claims 将检验该收益能否跨大箱/小箱/中箱复现，同时把 z 绝对误差纳入必报。
- 输入完整性核验通过：三物体 72/72 row 均有 result、override、scene；retarget
  variant 为 box001 `v1/v2=21/7`、box023 `15/1`、box021 `25/3`；box021
  baseline provenance 为 E170 production 24 条 + E169 audited reuse 4 条。
- 既有 z 指标锚点：A0 box001 `4.8456 cm`、box023 `5.8169 cm`；旧 G1
  box024 `5.5738→3.2290 cm (-42.1%)`、box004 `4.8922→4.4825 cm
  (-8.4%)`。扩展成功标准不能假定统一降幅，需逐物体报告 paired noninferiority
  与 improvement，并保持 case macro aggregation。
- 按 skill 的远程并行规则，72 条 Full 的计划将固化本地/远程 worker manifest、
  tmux launch、pull 和 watcher；不同设备输出路径隔离，每 GPU 严格串行，启动前
  只读核验资源，不抢占或终止非本实验任务。
- `data-construction-v3-zh` 合同已纳入：扩展只追加 S6 evidence，不回写
  E168/E170/E173 的 S1-S5 事实；主键保留 retarget/target/collision variant，
  manifest 记录 source_exp、method、SHA、run/config provenance，正式产物仅落
  `results/E194/`，results 不入 git。
- 计划文档将按 `markdown-mermaid-writing` 使用一个带可访问性说明的简洁 Mermaid
  flowchart 表达 authority→sidecar audit→canary→Full→eval/visual→log/tracker，
  其余配置、claims 与 stop-loss 用表格呈现。
- Markdown/Mermaid 规范已完整复核：计划保持单一 H1、H2 单 emoji、代码块带语言、
  Mermaid 使用 `accTitle/accDescr`、snake_case ID、单向 LR 流程且不使用 inline
  style；内部实验事实使用相对链接，不伪造外部引用。
- 已创建 [plan222](plan/222_E194_G1_box001_box023_box021_expansion_plan.md)：
  冻结 72 个新增 G1 Full（28+16+28），采用 Stage0 authority/A0 audit → 9 条
  三 worker canary → 6 条 Full sentinel → 72 条 resume-safe Full → paired
  eval/全量渲染/mandatory visual review。
- Claims 已预注册：逐物体 z MAE 非劣界 `+0.50 cm`、3D position 非劣界
  `+1.00 cm`、3mm contact/penetration/leg 回退界 `0.05`，至少 2/3 物体明确
  改善才判统一 PASS；否则只给 object-specific mixed/regression 结论。
- Tracker 的 E194 同一行已追加扩展 plan222，未创建新 E 编号；当前只完成计划与
  状态记录，未创建实现脚本、未生成 sidecar、未启动 GPU、未提交或推送。
- 计划初审通过：必需章节齐全，单一 H1/H2 层级与 Mermaid accessibility 合同
  正确，Tracker 描述 65 字符（≤80），内部链接存在，`git diff --check` 通过；
  补齐了执行命令中 render launcher 对应的 planned-file 条目。
- 遇到的错误：首次补 planned-file 条目时把 progress hunk 误放在 plan 的
  `Update File` 段内，`apply_patch` 因找不到上下文退出且未写入；已改为两个明确
  `Update File` hunk 后成功，未重复失败操作。
- 最终审计的首版 H1 计数器未排除 fenced bash 内的 `#` 注释，因此误报
  `h1=6`；原文件可见 H1 仅标题一处。将改用 fenced-code-aware 计数器复核，
  该误报不修改计划内容或实验状态。
- fenced-code-aware 复核通过：H1=1、H2=10 且 H2 emoji 合同无异常、code
  fence 16 个成对、planned files=15、Tracker 描述 65 字符；最终
  `git diff --check` 通过，E194 实现目录无本轮变更。

### 2026-08-10 · E195 更紧 hand gate 计划

- 计划：[plan221](plan/221_E195_stricter_hand_gate_plan.md)
- 直接承接 E192 A2，以 E192 的 15 条 Full 作为逐 case paired baseline
- 唯一配置变化：hand gate 从 `-0.010 / 0.05 / -0.015` 收紧为
  `-0.008 / 0.05 / -0.012`；seed 0 与 `1024 × 32` 保持不变
- 固定调度：本机 GPU0 七例，远程 RTX 6000 Ada GPU0/1 各四例
- 采用叠加式启动，不等待空卡、不干预已有程序
- 2026-08-10 用户已批准按 plan221 推进完整实验
- 当前状态：实现前审计完成；E195 尚无脚本、manifest 或运行产物
- 已确认 E192 的 15/15 paired baseline、最终 7/4/4 分片和 Ada 远程入口均存在
- 工作区包含 E192 及公共模块的既有未提交改动；E195 将基于并保留这些状态增量实现
- E192 可复用链已盘点：manifest builder、queue runner、local/Ada/hybrid/pull、eval/report/render 入口齐全
- 公共 metrics 当前已有固定 10/15/20 mm 口径；E195 需按计划追加 8/12 mm，既有字段保持不变
- 实现方案已冻结：E195 manifest 直接继承 E192 Full 的输入与最终 worker 映射，只输出 E195 所需字段
- 评测将用公共 `eval.core.core_metrics` 分别重评 E192 A2 与 E195 A3，再生成 15 条 paired delta；不改写 E192 已完成报告
- 已实现 E195 common contract、15-row builder、resolved-config 审计和 Full queue runner
- 公共 metrics 已纯追加固定 8/12 mm 观察列，原有 10/15/20 mm 口径未改
- 已实现本机七例、远程 Ada 4/4、三卡总入口与只回收 E195 产物的 pull/状态合并入口
- 已实现公共 metrics 双臂重评、paired delta、C1–C7 报告、15 self + 15 paired 离线渲染入口
- 启动前静态检查通过：13 个 E195 文件可编译/解析，shell 入口语法通过
- manifest 已生成并审计通过：15 个唯一 case，worker `7/4/4`，gate `-0.008/0.05/-0.012`，预算 `1024×32`、seed 0
- 三个 worker 的 dry-run 命令均正确继承 E192 task/override，并只附加 E195 hand-gate 策略包
- 三卡 Full 已于 2026-08-10 02:54 叠加启动：本机 `E195_full_local_20260810_025400`，远程 `E195_full_ada6000_20260810_025404`
- 三条队列分别为 7/4/4；启动过程未等待空卡、未停止或调整其他程序
- 三 worker 首例均已进入 32-iteration 优化：local `027_p1`、Ada0 `026_p1`、Ada1 `026_p2`
- 首轮实测：本机约 25–30 分钟/case；远程叠加既有负载后约 60–65 分钟/case
- 固定 7/4/4 队列的当前瓶颈预计约 4–4.5 小时；不做动态重分配
- CPU evaluator 冒烟通过：E192 `026_p1` 成功产出新增 fixed8/fixed12、CEM valid/fallback 与 12 门结果
- 03:19 本机首例 `027_p1` 自然完成并通过 runner 校验：qpos finite、必要 diagnostics 齐全、A3 gate 值准确
- 03:42 本机第二例 `027_p2` 完成并通过同一校验；本机已进入第三例 `028_p1`
- 03:56 远程首例 `026_p1/p2` 同时完成并通过远端 runner 校验；Ada0/Ada1 已进入 `030_p1/031_p1`
- 当前完成 `4/15`、失败 `0`；已完成四例均 qpos finite、diagnostics 齐全且 A3 gate 值准确
- 本机实测约 24 分钟/case，远程首例约 61 分钟/case；预计整体仍在 06:50–07:10 闭合
- 04:09 本机第三例 `028_p1` 完成，状态 `run_complete_pending_eval`；第四例 `028_p2` 已自动开始
- 04:30 Ada1 第二例 `031_p1` 完成，第三例 `082_p2` 已自动开始；Ada0 第二例继续运行
- 04:35 本机第四例 `028_p2` 完成，第五例 `031_p2` 已自动开始
- 04:47 Ada0 第二例 `030_p1` 完成，第三例 `082_p1` 已自动开始；Ada1 第三例 `082_p2` 继续运行
- 04:48 本机第五例 `031_p2` 完成，第六例 `083_p1` 已自动开始
- 05:07 本机第六例 `083_p1` 完成，本机最后一例 `086_p2` 已自动开始
- 05:21 本机最后一例 `086_p2` 完成；local shard `7/7 run_complete_pending_eval`，本机 session 自然结束
- 05:27 Ada1 第三例 `082_p2` 完成，最后一例 `083_p2` 已自动开始
- 05:43 Ada0 第三例 `082_p1` 完成，最后一例 `086_p1` 已自动开始
- 06:21 Ada0/Ada1 最后两例 `086_p1/083_p2` 完成；local `7/7`、Ada0 `4/4`、Ada1 `4/4`
- Full 执行 `15/15 run_complete_pending_eval`、失败 `0`；三个执行 session 均自然结束
- 06:23 watcher 自动收尾完成：远程回收、canonical `15/15`、统一评测 `E192 15 + E195 15`、errors 0
- 离线渲染闭合：E195 self `15/15`、E192/E195 paired `15/15`
- 视觉复核前报告暂为 `INCOMPLETE`；数值预判 C1/C3/C5 FAIL、C4/C6 PASS，C2/C7 待视觉闭合
- 下一动作：读取 claims/deltas 确定全部 mandatory/migration case，用 video-frames 抽帧并逐例填写 visual review
- 06:32 已按 E192 的精确四阶段口径（总帧数 15%/40%/65%/90% 向上取整）完成 14 条必审 case、56 张 grasp/lift/carry/place 关键帧与 14 张四阶段拼图
- 首次直接执行 video-frames `frame.sh` 因文件无 executable 位在写帧前退出；改用 `bash frame.sh` 后一次完成，不改脚本权限或视频
- 视觉结论：box024 9/9 未见可确认的承重 hand-away；主要变化是箱体倾斜、身体朝向和 lower-body 代偿；box004 `083_p1` 下肢介入、`086_p2` 末段塌姿与安全回退一致
- `visual_review.tsv` 已落盘 14/14，均复核 grasp/lift/carry/place；box024 无 `confirmed_loss`
- 下一动作：重跑 E195 comparison 得到最终 C1–C7/decision，随后写 log272 并同步 Tracker/progress/INDEX
- 06:33 重跑 comparison 后最终判决闭合：C1/C3/C5 FAIL，C2/C4/C6/C7 PASS，decision=`SAFETY_REGRESSION`
- 关键结果：box024 3mm penetration `0.3082→0.3433`、仅 `3/9` 改善；fixed12 仅 `1/9` 改善；box004 3 个 case×gate PASS→FAIL；box024 max leg Δ=`+0.0764`
- 已写 [log272](log/272_E195_stricter_hand_gate_results.md)，明确不升级 A3、不继续 hand-gate-only 机械收紧，并分开记录观测事实、机制推断、限制与后续建议
- Tracker E195 已更新为最终 `SAFETY_REGRESSION` 并链接 log272/plan221；plan221 顶部状态同步为实验完成
- 下一动作：重建 log INDEX，并执行 15/15 artifacts、metrics/deltas、视频/视觉、格式与工作树最终审计
- 06:36 log INDEX 已重建，Phase 58 正确收录 log272；总日志数 271
- 最终必要审计 PASS：manifest/finite qpos/diagnostics `15/15`，metrics `30`、deltas `15`，self+paired 视频 `15+15`，visual `14/14`
- Claims 与最终判决一致：`C1F/C2P/C3F/C4P/C5F/C6P/C7P`，`SAFETY_REGRESSION`
- E195 Python 编译、shell 语法、JSON/TSV/Markdown 引用、Tracker 描述长度及 `git diff --check` 全部通过
- E195 实验闭合；未提交、未推送，保留用户现有工作树与 E192 相关改动
- 03:29 已启动持久化收尾 watcher `E195_full_finalize_20260810_032900`
- watcher 只等待 E195 session 自然结束，随后依次执行 pull、15/15 状态校验、eval、self/paired render 与报告；不补跑、不迁移
- 下一动作：持续监控自然完成，回收远程产物并合并 15-row 状态

### 2026-08-10 · E192 A2 Full 诊断收尾

- 计划：[plan218](plan/218_E192_gate_threshold_size_dependence_plan.md)
- Sentinel：[log269](log/269_E192_corrected_box_sdf_sentinel_results.md)
- Canary：[log270](log/270_E192_A2_Ada6000_canary_stop_loss_results.md)
- Full：[log271](log/271_E192_A2_full_diagnostic_results.md)

执行闭合：

- 用户跳过 A0 baseline-drift stop-loss，并 waiver canary gate-collapse stop-loss 后启动 A2 Full
- Full 使用本机 GPU0 + 远程 RTX 6000 Ada GPU0/1；未使用 A100
- 15/15 manifest 行为 `run_complete_pending_eval`；runtime-output validation 15/15 PASS
- 评测 `evaluated=15/15, errors=0`；离线渲染 `15 self + 15 paired`，视觉复核 8 条
- 最终 claims：C1/C3/C4/C6 FAIL，C2/C5/C7 PASS
- Full-only diagnostic：`THRESHOLD_POLICY_NOT_EFFECTIVE`
- Governance decision：`INCONCLUSIVE_GATE_COLLAPSE`
- 决策：不升级 A2 为 production 默认；canary 未改判 PASS，原因果链未恢复
- 最终一致性审计 PASS：artifacts 15/15、metrics 30 行、deltas 15 行、视频 30 条、
  visual review 8 行；Python/shell 语法、JSON/TSV、Tracker 描述长度、Markdown/
  Mermaid 静态合同与 `git diff --check` 均通过。首次文档审计脚本误把 bash code
  block 内的 `#` 注释计作额外 H1，修正为忽略 fenced code 后通过；非文档缺陷。

关键证据：

- box024 penetration `0.3776→0.3082`，但仅 5/9 改善，未达 `≤0.20`
- signed DiD `0.0295`，bootstrap 95% CI `[-0.0572, 0.1172]`
- box004 有 4 个 case×gate PASS→FAIL；box024 max leg Δ=`+0.3025`
- non-fallback hard-floor violation `0/1152`，但 fixed15 仅 4/9 case 改善
- Full C7 逐物体通过，但 box024 fallback `0.2987` 距上限 `0.3022` 仅 0.0034

### E193 抓取拓扑

- 状态：计划态，待批准
- 计划：[plan219](plan/219_E193_grasp_topology_plan.md)
- 推荐下一动作：先跑零 GPU 的 Stage 1 接触法向对置审计；不要与 E192 阈值或 E194 gravcomp 叠加

### E194 物体重力补偿

- 状态：已完成
- 结论：G1 gravcomp 安全有效；G2/G3 硬伺服发散，C6 未判
- 结果：[log268](log/268_E194_object_gravity_compensation_results.md)

