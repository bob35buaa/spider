# E173 结果日志：box024 / box023 / box001 全流程筛选 + Full CEM（move-only · rubber_hull + E170 PRG 跨物体候选）

_Core4D Phase 36 · 2026-07-22 · 执行完成待用户终审 · machine recommendation = `PENDING_USER_REVIEW`_

计划：[plan/189_E173_box024_box023_box001_full_pipeline_plan.md](../plan/189_E173_box024_box023_box001_full_pipeline_plan.md)

---

## 1. 目的与假设

对 `box024`、`box023`、`box001` 三物体从 CORE4D raw authority 重跑 data_construction_v3 `S0–S6` 主链路（**move-only**），对通过 S5 gate 的 person-case 用**冻结的** E170/E171/E172 算法（`omnirt_v1→v2 rescue` + `ref_fk` + `rubber_hull` + `E170 PRG` 跨物体候选）跑 Full CEM。核心科学看点：**PRG 是否随物体尺寸退化**——E173 首次纳入大箱（box024/box001 ~0.25 m³、对角线 >1.1m），box023（0.036 m³）作小箱对照。**不做 reward/route sweep；本轮不导出 RL/partner；science yield 与执行完成解耦。**

## 2. 冻结配置（与 E172 字段级一致）

| 轴 | 值 |
|---|---|
| Retarget primary / rescue | `omnirt_v1` / `omnirt_v2`（仅 v1 `omniretarget_infeasible` 触发） |
| Target route | `ref_fk` |
| Base reward | `E167A_zOnlyBody`（object-agnostic，`core4d_E167_box004_082_p1_E167A`，沿用 E171 多物体先例） |
| Hand collision | `rubber_hull`（mesh-aware, maxhullvert=64） |
| Lower-body physics / penalty | 16 geoms ↔ object；scale `2.0`、margin `0.02m` |
| Candidate gate | min SDF `0.005m`、max viol `0.02`、hard floor `-0.005m`、fallback=`least_violation` |
| CEM budget | seed `0`、samples `1024`、opt steps `32`（canary `64/4`） |
| Method ID | `E173_E170PRG_crossObject_candidate_r1` |

执行环境：hsretargeting conda env（S3 retarget），tidal `CORE4D_Real` + `human_model_files/smplx`，本机 8×L20Y 跑 CEM（`MUJOCO_GL=osmesa`；释放已授权 GPU-filler `.cache/run.py`）。

## 3. Funnel（194 raw 全终态闭合，move-only）

| 阶段 | 计数 |
|---|---|
| raw expected / seen | 194 / 194（3 物体，97 seq） |
| by object (raw) | box024=46, box023=46, box001=102 |
| move → raw-contact / Stage0 reject | 86 / 108（106 high_risk join/leave/rot/pass/strike + 2 box001 motion_weak） |
| raw-contact pass (move-only, 3cm) | 56 |
| Stage2b v1 pass / infeasible | 42 / 14 |
| v2 rescued pass / dual-infeasible | **14 / 0** |
| target gate pass / visual QC pass | 56 / 56 |
| **PRG scene-contract reject** | **3**（全大箱：box001×2, box024×1） |
| **CEM eligible (S5_READY)** | **53** |
| Full CEM completed / terminal-failed | 53 / 0（missing=0，completion audit **PASS**） |
| **Numeric pass** | **26** |

终态闭合：`CEM_ELIGIBLE 53 + REJECT_PRG_SCENE_CONTRACT 3 + NOT_STAGE2B_ELIGIBLE 28 + REJECT_RAW_CONTACT 110 = 194`。raw_closure_pass=True，rescue_set_equals_v1_infeasible=True。

## 4. v1→v2 rescue（大箱上 Stage2b 收益极强）

- v1 infeasible 14 条：box001×9、box024×4、box023×1。**大箱 v1-infeasible 率显著更高**（box024 4/10=40%、box001 9/30=30% vs box023 1/16=6%）——大物体 OmniRetarget 更难解。
- v2 Phase4 rescue：**14/14 全部救回 pass，dual-infeasible=0**（远超 E172 的 1/1、E171 box026 的 0 净产出）。
- v2 下游：12 条进 Full CEM（2 条被 PRG scene-contract reject），**4/12 numeric pass**（含大箱 box001 v2 视觉干净的推箱结果）。v2 在大箱上有真实 Stage2b + 部分下游产出。

## 5. PRG scene contract（大箱下肢 trade-off）

56 条 S5-ready 中 **3 条被 PRG scene contract reject**，全部大箱（box001×2、box024×1），reason=`runtime_initial_overlap`：参考首 5 帧 lower-body/object 最小距离为负（-0.016 / -0.011 / -0.009m < hard floor -0.005m），即机器人腿在起始帧已与大箱重叠。box023（小箱）0 reject。**PRG 的下肢/物体 contact trade-off 随箱体尺寸放大**：E172 box004（小）0 reject、E171 box026（中）5 reject、E173 大箱 3 reject。

## 6. Full CEM 数值评测（26/53 numeric pass，清晰的尺寸依赖）

evaluator 直接 import `eval.core.core_metrics`，阈值 launch 前冻结。completion audit PASS（53=53+0，missing=0）。

| 物体 | 尺寸 | numeric pass | 主要失败模式 |
|---|---|---|---|
| **box023** | 小 0.036 m³ | **13/16 (81%)** | fall×1, lower_body×2（基本干净） |
| **box024** | 大 0.253 m³ | **3/9 (33%)** | hand_penetration×6, lower_body×2, body_z×1 |
| **box001** | 大 0.256 m³ | **10/28 (36%)** | lower_body×9, hand_penetration×7, release×3, fall×1, body_z×1, contact×1 |
| **合计** | — | **26/53 (49%)** | lower_body 13 + hand_penetration 13 主导 |

- by variant：v1 22/41（54%）、v2 4/12（33%）。
- **gate-health 0/53**（与 E170 0/28、E171 0/12、E172 0/6 一致，candidate gate 已知弱项，与质量判定分列）。
- **核心发现——PRG pass 率随物体尺寸单调退化**：
  | 物体 | 体积 m³ | PRG numeric pass |
  |---|---|---|
  | box004 (E172) | 0.041 | 5/6 = 83% |
  | box023 (E173) | 0.036 | 13/16 = 81% |
  | box021 (E170) | 0.059 | 18/28 = 64% |
  | box026 (E171) | 0.116 | 5/12 = 42% |
  | box024 (E173) | 0.253 | 3/9 = 33% |
  | box001 (E173) | 0.256 | 10/28 = 36% |

  小箱（≤0.06 m³）~80%，中箱 ~40-64%，大箱（~0.25 m³）~35%。大箱失败主因是 **hand_penetration（手陷入大平面）+ lower_body（腿贴大箱）**——大物体表面大、起始姿态更贴身，rubber_hull SDF 与 PRG 下肢惩罚都更难约束。

## 7. 视觉复核（§8.3 强制，codex_opus）

- **S4 kinematic replay**（launch 前）：逐帧核验 box024/box001（v1+v2）lean-over-push、box023 squat-lift-carry，全部干净，无穿透/浮空/爆姿；小箱可举、大箱只能推（尺寸驱动的合理差异）。存 `s4_gate_visual_qc/codex_review/`。
- **S6 CEM montage**（`s6_downstream/evidence/visual_qc/codex_cem_spotcheck/`）：
  - box001_039_p1（大箱 v1 **pass**）：直立推大箱、脚着地、全程稳定 → 视觉印证 numeric pass。
  - box001_040_p1（大箱 v1 **fail** lower_body）：推箱时腿压入大箱低位底面 → 视觉印证 lower_body 穿透。
  - box001_2_039_p1（大箱 **v2** pass）：直立推箱干净 → v2 大箱下游有效产出。
- **S6 CEM 全量渲染（`s6_downstream/render/full/`，2026-07-22 补全）**：**53/53 MP4 全部渲染完成，0 fail**（box001 28 + box023 16 + box024 9，171M）。此前仅 box001 12 条（上轮渲染器回退时中断）；用户要求全量可视化后，用 `run_E173_render_all.sh`（osmesa 纯 CPU，按 case round-robin 8-shard 并行，resume-safe）补齐全部 box023/box024 + 剩余 box001。box023 全部 squat-lift-carry 干净；box024 fail 集中在 hand_penetration（手陷大平面）。
- 视觉结论与数值一致：大箱失败集中在手陷入大平面 + 腿贴箱，非 reward hacking。

## 8. Claims 验证

| Claim | 结论 |
|---|---|
| C0 raw authority 完整 | ✅ 194/194，3 物体 97 seq，raw_closure_pass |
| C1 两档 raw contact 可复现 | ✅ 3cm/5cm 均有终态；move-only 过滤（84→56，丢 28 非move）显式记录 |
| C2 template 可信 | ✅ 6/6 clean（box024/box001 person2 补建 + box024 材质名 bug 修复），碰撞盒=mesh AABB，SHA snapshot |
| C3 route/variant 未污染 | ✅ target 恒 ref_fk；v1/v2 provenance 分离 |
| C4 v1→v2 fallback 完整 | ✅ 每 eligible 有 v1 终态；14 infeasible 全有 v2 终态；rescue_set==v1_infeasible |
| C5 S3–S5 无静默缺失 | ✅ 每 selected row 有 gate/visual/handoff 终态 |
| C6 Full CEM 覆盖精确 | ✅ 53=53+0，missing=0，completion audit PASS |
| C7 质量证据完整 | ✅ metrics+MP4+montage+Codex verification |
| C8 分层结论可解释 | ✅ 分 object/person/variant/failure taxonomy/尺寸 报告 |
| C9 历史比较不越界 | ✅ box023 pre-PRG 仅定性参考，不作基线/completion |
| C10 用户 authority 独立 | ✅ Codex 只写核验列，`manual_*` 待用户 |
| C11 结果可复现 | ✅ config/SHA/scene snapshot/registry/manifest/report 均在 `results/E173/` |
| C12 PRG 跨物体（含大箱）可解释 | ✅ 尺寸单调退化趋势坐实（小 ~80% → 大 ~35%），大箱 3 scene-reject + hand/lower_body 主导 fail，与 box021/026/004 对照 |

## 9. 结果分级与结论

- **执行完成**：`PIPELINE` 完整（authority/terminal state/Full completeness/证据齐全，audit PASS），非 `PIPELINE_INCOMPLETE`。
- **科学产出（待用户终审）**：三物体 = **26/53 numeric pass**，机器侧倾向 **PARTIAL_YIELD**（box023 小箱 13/16 强 positive；box024/box001 大箱 case-level positive 但 <40%）。
- **核心结论——PRG 跨物体随尺寸退化**：小箱（box004/box023 ~80%）PRG 稳；大箱（box024/box001 ~35%）显著退化，主因 hand_penetration（手陷大平面）+ lower_body（腿贴箱）+ 3 条起始重叠 scene-reject。**PRG 不宜作大箱默认**；小箱可保留 case-level positives。
- **v2 rescue**：大箱 Stage2b 14/14 有效（大箱 v1 更易 infeasible），下游 4/12 numeric pass（真实净产出）。
- gate-health 仍 0/53（已知弱项，需另设实验修复）。

## 10. 结果路径

- funnel/authority：`results/E173/registries/{pipeline_funnel.json,pipeline_authority.tsv}`
- completion audit：`results/E173/completion_audit/completion_audit.json`（pass）
- metrics：`results/E173/s6_downstream/eval/full/{e173_case_metrics.tsv,summary.json}`
- Codex 核验：`.../eval/full/codex_verification.tsv`；用户模板：`.../user_manual_review_template.tsv`
- CEM manifest/scenes：`.../manifests/cem_full_manifest.tsv`、`scene_snapshot/{source_templates,cem_sidecars}/`
- 报告：`.../eval/full/E173_report.md`；MP4：`.../render/full/`（**53/53 全渲染**，box001 28+box023 16+box024 9）；montage：`.../evidence/visual_qc/codex_cem_spotcheck/`
- 脚本：`scripts/experiments/E173/`、`scripts/launch/active/run_E173_*`（含 `run_E173_render_all.sh` 全量渲染 launcher）、`scripts/eval/{runners,wrappers,reports}/*E173*`

## 11. 下一步

1. **用户对 53 条 CEM-complete row 给出 fresh `USE/DO_NOT_USE`**（machine 固定 `PENDING_USER_REVIEW`，回填前不生成 RL-ready/partner export）。
2. 用户裁决后确定最终 yield 分级。
3. PRG 大箱退化已坐实——若要大箱可用需另设实验（如尺寸自适应 hand-collision margin / 下肢可行域 / gate-health 修复），不在 E173 范围。

## 12. 实现备注（fork 修正 + 执行事件，已 git 跟踪）

- `scripts/experiments/E173/*` 与 launch/eval 由 E172 fork；scope=3 物体，method ID=E173，eval 输出改 `e173_` 前缀。
- **move-only 过滤**：raw_contact 对全 candidate 算接触，`run_stage2b` 不 gate move；E172(box004) 未触发（非move自然接触fail），E173 box001 有非move 通过 3cm → 在 pipeline S1 后加 move-only 过滤（inventory_decision∈{pass,review}_to_raw_contact），84→56。
- **S2 模板重建**：box024/box001 person2 缺失、box023/box024 person1 geometry_review（碰撞盒偏 mesh AABB >8%）→ `--apply-build --overwrite-existing` 重建全部为 mesh_aabb clean；**box024 material bug**（builder 默认 base=box023_person1 含 `box023_material`，rename 正则找 `box_material` 不匹配 → geom 引用 box024_material 无定义 load fail）→ 直接改两 scene 材质名。6/6 clean，git add -f + snapshot。
- **CEM 并行化教训**：CEM(`use_torch_compile=false`, warp) 单 case ~35-45min。先试 48-wide 并发（1 case/GPU×6-7）→ GPU 过饱和(99%×8)每 case 慢 >3×，2h 仅 5/53 → 回退**8-wide 串行**(1/GPU, GPU~44%, ~40min/case 稳定)，~5h 完成 53。教训：Warp CEM GPU-bound，最优 1/GPU。
- S3 v1/v2 retarget 用 24/14-shard 并行（OmniRetarget CPU-bound，安全提速）。
- gen_report/eval runner 改为 3 物体 data-driven（避免 E172 单物体硬编码）。

## 14. 2026-07-26 box023 人工审查与 source+partner RL export

- 用户对 box023 的 16 条 CEM-complete row 完成人工分级：
  `CLEAN=2`、`MINOR_ACCEPTABLE=5`、`UNUSABLE=9`。按 box001/box024
  operational allowlist 口径，前两档共 7 条进入导出，9 条不可用严格排除。
- 标准 source 表位于
  `results/E173/s6_downstream/rl_export/box023_user_approved/rl_export_input.tsv`，
  `7/7 RL_EXPORT_READY`；人工 snapshot、source audit、S6 downstream
  evidence 和 SHA256 审计均在同一 canonical S6 目录。
- partner 使用同一 `(object, date, seq)` 的 opposite person。7 条均复用
  E173 已通过的 `omnirt_v1` Stage2b 产物，无需新跑 OmniRetarget 或 v2
  rescue；`partner_omnirt/paired_rl_export_input.tsv` 为
  `7/7 PAIR_COMPLETE + RL_EXPORT_READY`。
- 已通过 `update_case_state_registry.py` 记录 7 条
  `omnirt_v1/ref_fk/rubber_hull` S6 evidence；逐字段核对 S1-S5 状态与原
  `sphere5cm` base 一致，未用下游人工结论反写数据构建事实。
- 本次仅完成 RL-ready paired 输入打包，`rl_status=not_run`，不宣称 RL
  训练成功。
