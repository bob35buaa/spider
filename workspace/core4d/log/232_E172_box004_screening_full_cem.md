# E172 结果日志：Box004 全流程筛选 + Full CEM（move-only · rubber_hull + E170 PRG 跨物体候选）

_Core4D Phase 35 · 2026-07-22 · 执行完成待用户终审 · machine recommendation = `PENDING_USER_REVIEW`_

计划：[plan/188_E172_box004_full_pipeline_plan.md](../plan/188_E172_box004_full_pipeline_plan.md)

---

## 1. 目的与假设

对 `Box004` 从 CORE4D raw authority 重跑 data_construction_v3 `S0–S6` 主链路（**move-only**），对通过 S5 gate 的 person-case 用**冻结的** E170/E171 算法（`omnirt_v1→v2 rescue` + `ref_fk` + `rubber_hull` + `E170 PRG` 跨物体候选）跑 Full CEM。验证 PRG 是否在第三个物体（box004）上成立、v1→v2 rescue 的真实下游收益。**不做 reward/route sweep；本轮不导出 RL/partner；science yield 与执行完成解耦。**

## 2. 冻结配置（与 E171 字段级一致）

| 轴 | 值 |
|---|---|
| Retarget primary / rescue | `omnirt_v1` / `omnirt_v2`（仅 v1 `omniretarget_infeasible` 触发） |
| Target route | `ref_fk` |
| Base reward | `E167A_zOnlyBody`（`core4d_E167_box004_082_p1_E167A`） |
| Hand collision | `rubber_hull`（mesh-aware, maxhullvert=64） |
| Lower-body physics / penalty | 16 geoms ↔ object；scale `2.0`、margin `0.02m` |
| Candidate gate | min SDF `0.005m`、max viol `0.02`、hard floor `-0.005m`、fallback=`least_violation` |
| CEM budget | seed `0`、samples `1024`、opt steps `32`（canary `64/4`） |
| Method ID | `E172_E170PRG_crossObject_candidate_r1` |

执行环境：hsretargeting conda env（S3 retarget），tidal `CORE4D_Real` + `human_model_files/smplx`，本机 8×L20Y 跑 CEM（`MUJOCO_GL=osmesa`；先释放已授权的 GPU-filler `.cache/run.py`）。

## 3. Funnel（20 raw 全终态闭合，move-only）

| 阶段 | 计数 |
|---|---|
| raw expected / seen | 20 / 20（box004，10 seq） |
| move eligible / 非首选 Stage0 reject | 14 / 6（pass2×4: 055/089；strike×2: 116） |
| raw-contact pass（3cm=5cm） | 6（seq 082/083/086 的 p1/p2；其余 8 move fail raw contact） |
| Stage2b v1 pass / infeasible | 5 / 1 |
| v2 rescued pass / dual-infeasible | 1 / **0** |
| target gate pass / visual QC pass | 6 / 6 |
| PRG scene-contract reject | **0**（首帧 lower-body/object 距离 0.07–0.17m） |
| **CEM eligible (S5_READY)** | **6** |
| Full CEM completed / terminal-failed | 6 / 0（missing=0，completion audit **PASS**） |
| **Numeric pass** | **5** |

终态闭合：`CEM_ELIGIBLE=6 + REJECT_RAW_CONTACT=14 = 20`（14 = 6 Stage0 非首选 + 8 move raw-contact fail）。raw_closure_pass=True。

## 4. v1→v2 rescue

- v1 infeasible 1 条：`box004_082_p2`，**精确命中历史 prior**（E095/E167 partner path 曾 OmniRetarget CVXPY infeasible/missing）。
- v2 adapter canary（v1-pass row）通过；production rescue **1/1 救回 pass**，dual-infeasible=0。
- **082_p2 (v2) 进入 Full CEM 且 numeric pass + 视觉干净** → v2 在 box004 产生**真实下游 yield**（区别于 E171 box026 v2 的 0 净产出）。

## 5. PRG scene contract

box004 6 条 S5-ready **全部通过** PRG scene contract（rubber_hull + 16-pair sidecar），0 reject。首帧 lower-body/object 距离 0.07–0.17m，box004 搬箱起始姿态较直立，未触发 hard floor。**对比 E171 box026 的 5 reject（低位贴腿搬箱）—— PRG 的 contact trade-off 在 box004 数据层显著更轻。**

## 6. Full CEM 数值评测（5/6 numeric pass）

evaluator 直接 import `eval.core.core_metrics`，阈值 launch 前冻结。

| case | variant | numeric | body_z_p95 | release_false_3mm | leg_pen | fall | failure modes |
|---|---|---|---|---|---|---|---|
| 082_p1 | v1 | ✅ | 0.170 | 0.094 | 0.0 | ✗ | - |
| 083_p1 | v1 | ✅ | 0.133 | 0.083 | 0.0 | ✗ | - |
| 086_p1 | v1 | ✅ | 0.083 | 0.0 | 0.0 | ✗ | - |
| 083_p2 | v1 | ✅ | 0.159 | 0.056 | 0.0 | ✗ | - |
| 082_p2 | **v2** | ✅ | 0.161 | 0.0 | 0.065 | ✗ | - |
| 086_p2 | v1 | ❌ | 0.462 | 0.526 | 0.274 | ✓ | fall, body_z, release, lower_body |

- **gate-health 0/6**（082_p2 fallback 0.28、086_p2 fallback 0.61，其余 0；与 E170 0/28、E171 0/12 一致，candidate gate 已知弱项，与质量判定分列）。
- numeric failure 全部集中在 **086_p2 一条**（fall+body_z+release+lower_body 四模式同时触发）。

## 7. 视觉复核（§8.3 强制，codex_opus）

对 6 条 Full CEM MP4 抽 8 帧 montage，存 `s6_downstream/evidence/visual_qc/codex_cem_spotcheck/`，结论写入 `codex_verification.tsv`（仅 Codex 列，不动 `manual_*`）。S4 阶段亦对 6 条 target replay keyframe sheet 全部逐帧核验（见 `s4_gate_visual_qc/*/visual_qc/codex_visual_review.tsv`）。

- **5 条 numeric-pass 视觉全部干净**：upright 抬箱-搬运-放下，箱体中段离地，无穿透/fall/爆姿；082_p2(v2) 为箱体前抱搬运，干净。
- **086_p2 视觉印证数值 fail**：机器人在箱上深蹲跨立，后段塌陷/趴向箱体（fall + 腿部穿箱 + 丢箱），与 leg_pen 0.27 / release 0.53 / body_z 0.46 一致。该 case 在 S4 kinematic replay 即为**深蹲跨箱低位姿**，是 PRG 下肢/接触 trade-off 的可预期体现。
- PRG 的 leg/box trade-off 在 box004 仅命中 1 条（086_p2），远轻于 box026。

## 8. Claims 验证

| Claim | 结论 |
|---|---|
| C0 raw authority 完整 | ✅ 20/20，box004 10 seq，raw_closure_pass |
| C1 两档 raw contact 可复现 | ✅ 3cm/5cm 同集 6，其余有终态；6 非首选 Stage0 reject 保留 registry |
| C2 template 可信 | ✅ box004_person1/2 audit + SHA snapshot |
| C3 route/variant 未污染 | ✅ target 恒 ref_fk；v1/v2 provenance 分离 |
| C4 v1→v2 fallback 完整 | ✅ 每 eligible 有 v1 终态；1 infeasible 有 v2 终态；非 eligible v2 attempt=0 |
| C5 S3–S5 无静默缺失 | ✅ 每 selected row 有 gate/visual/handoff 终态 |
| C6 Full CEM 覆盖精确 | ✅ 6=6+0，missing=0，completion audit PASS |
| C7 质量证据完整 | ✅ metrics+MP4+montage+Codex verification 全覆盖 |
| C8 分层结论可解释 | ✅ 分 person/action/variant/failure taxonomy 报告 |
| C9 历史比较不越界 | ✅ E091/E167 仅同 case 历史参考 |
| C10 用户 authority 独立 | ✅ Codex 只写核验列，`manual_*` 待用户 |
| C11 结果可复现 | ✅ config/SHA/scene snapshot/registry/manifest/report 均在 `results/E172/` |
| C12 PRG 跨物体可解释 | ✅ box004 5/6 pass、0 scene reject、1 lower-body fail，与 box021/box026 对照 |

## 9. 结果分级与结论

- **执行完成**：`PIPELINE` 完整（authority/terminal state/Full completeness/证据齐全，audit PASS），非 `PIPELINE_INCOMPLETE`。
- **科学产出（待用户终审）**：box004 = **5/6 numeric pass、视觉干净** → 机器侧倾向 **PARTIAL_YIELD**（有 case-level positive，未覆盖全 move 分层，仅 3 seq 的 6 pc 进入下游）。
- **PRG 跨物体**：box004 上 lower-body 定向成立且 **contact trade-off 显著更轻**（0 scene reject vs box026 的 5；仅 1 条 lower-body numeric fail vs box026 的多条），但 gate-health 仍 0/6。三物体证据（box021 E170 18/28 operational、box026 E171 5/12、box004 E172 5/6）显示 PRG 在**起始姿态直立的搬箱 case 上更稳**；仍不足以自动升级为跨物体统一默认，需另设含跨对象对照 + gate-health 修复 + RL evidence 的实验。
- **v2 rescue**：box004 上 Stage2b 1/1 有效，**且下游 numeric pass（真实净产出）**。

## 10. 结果路径

- funnel/authority：`results/E172/registries/{pipeline_funnel.json,pipeline_authority.tsv}`
- completion audit：`results/E172/completion_audit/completion_audit.json`（pass）
- metrics：`results/E172/s6_downstream/eval/full/{e171_case_metrics.tsv,summary.json}`（文件名 `e171_` 前缀为 fork 残留，内容为 E172）
- Codex 核验：`.../eval/full/codex_verification.tsv`；用户模板：`.../user_manual_review_template.tsv`
- CEM manifest/scenes：`.../manifests/cem_full_manifest.tsv`、`scene_snapshot/cem_sidecars/`
- 报告：`.../eval/full/E172_report.md`；MP4：`.../render/full/`；montage：`.../evidence/visual_qc/codex_cem_spotcheck/`
- 脚本：`scripts/experiments/E172/`、`scripts/launch/active/run_E172_*`、`scripts/eval/{runners,wrappers,reports}/*E172*`

## 11. 下一步

1. **用户对 6 条 CEM-complete row 给出 fresh `USE/DO_NOT_USE`**（machine 固定 `PENDING_USER_REVIEW`，回填前不生成 RL-ready/partner export）。
2. 用户裁决后确定最终 yield 分级（PARTIAL_YIELD / OBJECT_YIELD）。
3. 若推进 PRG 跨物体默认化：另设含跨对象对照 + gate-health 修复 + RL evidence 的实验（不在 E172 范围）。

## 12. 实现备注（fork 修正，已 git 跟踪）

- `scripts/experiments/E172/*` 与 launch/eval 脚本由 E171 fork；scope=box004，method ID=E172。
- `build_pipeline_authority.py`：`in_cem` 判定由 `status=="READY_FOR_FULL"` 改为 `status not in {"", "preflight_blocked"}`，使 authority 在 CEM+eval 后（status=`run_complete_pending_eval`）仍正确计 cem_eligible（隔离、可逆修正）。
- `gen_E172_box004_report.py`：报告叙述改为 box004 单物体 move-only（原为 box022/box026 硬编码）。
- S5 修正：dcv3 override 需在 Hydra `examples/config/override/`（export 默认写到 s5 dir），已复制 6 个 dcv3 override 后重建 manifest → 假阳性 `missing_dcv3_override` 消除，PRG 真实 reject=0。
