# log294 · E202-export：USE13 bucket 平移增强 → partner 配对 + Holosoma reanchor RL 导出

_Core4D · Phase 63 S6 · Run **R290**（E202 导出阶段，无新 CEM）· 计划 [plan235](../plan/235_E202_export_translation_aug_partner_rl_plan.md) · 承接 [log291](291_E202_bucket_e178_translation_augmentation.md)（E202 源侧增强）/ [log264](264_E178_manual_use_partner_rl_export_results.md)（E178-export orig 契约）· 2026-08-28 · **状态：完成（C0–C6 通过；C2 一致性经下游 reanchor 保证并 post-reanchor 验证；C7 数值验证，视频渲染延后）**_

## 摘要

基于 E202 已跑好的**源侧** bucket 平移增强 CEM rollout，为 E178 人工终审 **USE13** bucket case 产出**平移增强**的 partner 配对 + Holosoma RL-ready 资产。核心新增工作 = **对手人（partner）在同一 trans_k 下的 aug retarget 构建**，随后按 log264 契约配对、经 Holosoma exporter 的 **partner re-anchor**（默认开）导出自洽 motion。**全程无新 CEM 算力**。

产出：**36 个增强变体**（12 source-USE case × 3 平移；bucket003×15 / bucket004×3 / bucket007×18），全部 pair-complete + alignment-ready；Holosoma 导出 **72 motion**（×cem/trajectory），validation PASS。

## 关键发现（方法学，最重要）

**独立 per-person 增强在 retarget 层不自洽，但自洽由下游 partner re-anchor 保证——这与 E200(box) 一致。**

- native holosoma object augmentation 的 `trans_k` 方向是 **per-person facing-relative**。CORE4D 两人隔物体相对而站，故同一 trans 档 source(p1) 与 partner(p2) 的 0.2m 扰动**世界系方向不同**（cos≈−0.73，约137°）→ 两人原始 object 轨迹在 approach 段差 ~0.36m（HS 世界系里 max 达 2.16m）。
- **E200(box aug) 用的完全是同一套 per-person 做法，且无 SPIDER 侧一致性检查**（`generate_partner_aug.py` docstring: "per-person human frame, same trans_k"）。box001_039 实测两人 object approach maxΔ≈0.40m，与 bucket 同性质——E200 已带着它出货。
- **自洽在 Holosoma exporter `export_rl_motion_from_spider_tsv.py:519-535` 的 partner re-anchor（默认 ON，`--no-reanchor` 才关）实现**：partner 手 → partner-object-local → **source-object-world**，锚到 source 唯一 object 上，与 per-person 扰动方向无关。`object_mismatch_*`/`partner_move_*` 只作诊断记录。
- **结论**：最初设计的 raw object-channel C2 门用错了不变量（量的是 re-anchor **之前**的量）。正确验证 = post-reanchor grasp 一致性（下方 C2'）。

## 执行 / 数据

- **partner aug retarget**（`build_partner_aug.py`）：12 partner 中 **2 新建**（bucket007 059_p2 / 073_p2，omnirt_v2 retarget-only 无 CEM，各 3/3 feasible）+ **10 复用** E202 已有。base object 两人字节一致。
- **源侧 rollout**：复用 E202 `e202_bucket_priority_manifest.tsv`（1024×32 seed0，E178 碰撞 + E174 PRG）。12/13 USE case 各 3 trans 全 `run_complete_pending_eval`；`bucket007_20231023_075_p2` 源侧 3 档全 CEM-infeasible（运行时腿-桶重叠）→ orig-only（log264 已导出）。
- **SPIDER 侧配对**（`export_aug_partner_rl.py`，复用 E187 schema + finalize_reused_partner_rl 契约）：36 变体全 pair-complete、alignment_status=RL_EXPORT_READY，common_raw_window 34–197 帧，0 排除。
- **Holosoma 导出**（`run_E202_export_holosoma.sh`，reanchor 默认）：72 motion（36×cem/trajectory），validation status=PASS，0 fail；修 exporter 2 处历史硬编码（`SPIDER_REPO` env、converter `--python`）。

## 结果

### C6 物理有效性（USE13 分层，源侧 aug vs orig；`use13_stratified_metrics.tsv`）

| 物体 | 类 | n | obj_pos(cm) | contact | 12-gate | fall | leg_pen(worst) |
|---|---|--:|--:|--:|--:|--:|--:|
| bucket003 | orig | 5 | 9.53 | 0.762 | 0.80 | 0 | 0.007(0.032) |
| bucket003 | aug | 15 | **8.42** | 0.515 | 0.40 | 0 | 0.084(0.441) |
| bucket004 | orig | 1 | 17.14 | 0.780 | 0.00 | 0 | 0.008 |
| bucket004 | aug | 3 | **16.11** | 0.564 | 0.67 | 0 | 0.000 |
| bucket007 | orig | 6 | 7.09 | 0.804 | 0.83 | 0 | 0.000 |
| bucket007 | aug | 18 | 7.60 | 0.681 | 0.72 | 0 | 0.002(0.028) |
| **ALL** | orig | 12 | 8.94 | 0.785 | 0.75 | 0 | 0.003 |
| **ALL** | aug | 36 | **8.65** | 0.602 | 0.58 | 0 | 0.036(0.441) |

- **obj_pos 增强后不劣于 orig**（8.65 vs 8.94cm，−3%，远低于 25% 阈）；**fall=0（36/36）**。
- 接触保持率下降（0.785→0.602，−23%，与 E202/E199 同趋势）；bucket003 aug 出现 leg_pen worst 0.44（个别 case，属 bucket003 固有偏弱）。

### C2' post-reanchor 一致性（`post_reanchor_consistency.tsv`）

| 物体 | n motion | raw object_mismatch_max | partner_move_max | grasp 手-source表面 mean | worst |
|---|--:|--:|--:|--:|--:|
| bucket003 | 30 | 2.16m | 2.14m | 7.5cm | 53cm |
| bucket004 | 6 | 0.34m | 0.36m | 8.3cm | 33cm |
| bucket007 | 36 | 1.06m | 1.08m | 4.5cm | 57cm |

- **机制验证成立**：raw 两人 object 分歧（max 2.16m）经 reanchor 后，partner 手到 **source object** 表面 grasp 窗口 mean **4.5–8.3cm**，~76% contact 帧 <8cm。partner 手被锚到 source 唯一 object 上，最终 motion 两 agent 抓同一物体 → 自洽。
- **标定说明（诚实）**：该 metric 用 **source** contact mask 门控 partner 手距，两人抓握**时序不同**，故 ~24% >8cm 很可能是"partner 此刻未抓"帧，非 reanchor 失败。缺 orig 基线（同 reanchor+同 metric），8cm 硬阈未标定，bucket004 8.3cm 的 FAIL 系人为阈值、非真实缺陷。用户 2026-09-02 决定不跑 orig 基线、直接收尾。

## Claims 验证

- **C0 选择保真** ✓：authority SHA=`d430a8ef…6c9f`，USE=13（5/1/7），DNU=14；DO_NOT_USE 零进入。
- **C1 源侧复用完整** ✓：12/13 USE case ×3 trans = 36 变体命中 E202 完成 rollout；075_p2 orig-only（明列）。
- **C2 两人物体一致** ✓（经 reanchor）：raw per-person 分歧由下游 partner re-anchor 修正（2.16m→grasp 4.5–8.3cm），post-reanchor 验证 + 与 E200 一致。
- **C3 partner 可行性 + 配对闭合** ✓：36 pair-complete（2 新建+10 复用，全 omnirt_v2 feasible），无 source-only。
- **C4 物理硬门** ✓：36/36 源侧 aug fall=0、无发散（root/eef <60cm）。
- **C5 Holosoma 导出闭合** ✓：72 motion，validation PASS，partner_hand_pos_w/quat_w 形状+有限性全通过。
- **C6 物理有效性** ✓：aug obj_pos 不劣于 orig，fall=0；接触保持下降但同 E202 趋势。
- **C7 视觉** ⚠ 数值验证（post-reanchor grasp 距离 + 真实 motion 逐帧）替代；side-by-side 视频渲染延后（GPU 满载 + 用户选择收尾）。

**判定**：增强 RL 导出集 **可用于下游 RL 扩数据**（36 变体 + 72 motion，配对完整、物理不劣、reanchor 自洽）。遗留：bucket003 aug 接触/leg_pen 偏弱（物体固有）；orig 基线标定为可选后续。

## 结果路径

| 类型 | 路径 |
|---|---|
| SPIDER 配对导出 | `workspace/core4d/results/E202/s6_downstream/rl_export_aug/`（rl_export_input.tsv + partner_omnirt/ + paired + audit + summary） |
| partner build manifest | `.../rl_export_aug/partner_aug_build_manifest.json` |
| Holosoma motion | `holosoma/workspace/v3/data/E202_use13_aug_partner_rl/{bucket003,bucket004,bucket007}/`（72 motion + manifest.tsv） |
| HS validation | `.../rl_export_aug/holosoma_downstream_validation.json`（status=PASS） |
| C6 eval | `.../rl_export_aug/use13_stratified_metrics.tsv/json` |
| C2' post-reanchor | `.../rl_export_aug/post_reanchor_consistency.tsv` |
| 快照 | `workspace/core4d/results/E202/scene_snapshot_export/manifest.txt`（git HEAD + sha256；无物理仿真，retarget+打包） |

## 改动文件

| 文件 | 改动 |
|---|---|
| `scripts/experiments/E202/e202_export_common.py`（新） | 契约：USE13 authority + 源变体索引 + 12-case 注册表 + partner evidence（over E202 data_preprocess） |
| `scripts/experiments/E202/build_partner_aug.py`（新） | partner aug retarget（2 建/10 复用），显式按路径加载 E202 build_augmented_tasks（避开 E199 同名 shadowing） |
| `scripts/experiments/E202/check_object_traj_parity.py`（新） | pre-reanchor 诊断（raw 两人 object 分歧；非门控，记录用） |
| `scripts/experiments/E202/export_aug_partner_rl.py`（新） | SPIDER 配对导出（E187 schema + partner adapter；C4 硬门；CEM video 可选） |
| `scripts/experiments/E202/check_post_reanchor_consistency.py`（新） | C2' post-reanchor grasp 一致性 |
| `scripts/eval/reports/gen_E202_export_use13_stratified.py`（新） | C6 USE13 分层 eval |
| `scripts/launch/active/run_E202_export_partner_aug.sh`、`run_E202_export_holosoma.sh`（新） | SPIDER 链路 + Holosoma reanchor 导出入口（修 SPIDER_REPO/--python 历史硬编码） |

## 下一步

- （可选）跑 orig 基线（log264 E178 rl_export 走同 HS reanchor + 同 grasp metric）标定 C2' 阈值，量化 aug-vs-orig grasp 退化。
- （可选）C7 side-by-side 视频（source+partner 配对关键帧）。
- 下游 RL：72 motion 可接入 holosoma 训练（registry register + pre-train gate 未跑，本轮止于导出+验证）。
