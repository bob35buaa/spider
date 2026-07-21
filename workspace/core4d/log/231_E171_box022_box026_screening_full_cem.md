# E171 结果日志：Box022/Box026 全流程筛选 + Full CEM（rubber_hull + E170 PRG 跨物体候选）

_Core4D Phase 34 · 2026-07-21 · 执行完成待用户终审 · machine recommendation = `PENDING_USER_REVIEW`_

计划：[plan/187_E171_box022_box026_full_pipeline_plan.md](../plan/187_E171_box022_box026_full_pipeline_plan.md)

---

## 1. 目的与假设

对 `Box022`、`Box026` 从 CORE4D raw authority 重跑 data_construction_v3 `S0–S6` 主链路，并对真正通过 S5 gate 的 person-case 用**冻结的** `omnirt_v1→v2 rescue` + `ref_fk` + `rubber_hull` + `E170 PRG` 跨物体候选配置跑 Full CEM。验证 PRG（E170 在 Box021 证明的 lower-body 定向收益）是否跨物体成立，以及 v1→v2 rescue 的真实收益。**不做 reward/route sweep；science yield 与执行完成解耦（0 positive 亦为有效结果）。**

## 2. 冻结配置

| 轴 | 值 |
|---|---|
| Retarget primary / rescue | `omnirt_v1` / `omnirt_v2`（仅 v1 `omniretarget_infeasible` 触发） |
| Target route | `ref_fk` |
| Base reward | `E167A_zOnlyBody`（`core4d_E167_box004_082_p1_E167A`，与 E170 PRG 一致） |
| Hand collision | `rubber_hull`（mesh-aware，maxhullvert=64） |
| Lower-body physics / penalty | 16 geoms ↔ object；scale `2.0`、margin `0.02m` |
| Candidate gate | min SDF `0.005m`、max viol `0.02`、hard floor `-0.005m`、fallback=`least_violation` |
| CEM budget | seed `0`、samples `1024`、opt steps `32`（canary `64/4`） |
| Method ID | `E171_E170PRG_crossObject_candidate_r1` |

执行环境（本机）：hsretargeting conda env（S3 retarget），tidal `CORE4D_Real` + `human_model_files/smplx`，本机 8×GPU 跑 CEM（`MUJOCO_GL=osmesa`，本机无 nvidia EGL）。

## 3. Funnel（60 raw 全终态闭合）

| 阶段 | 计数 |
|---|---|
| raw expected / seen | 60 / 60（Box022=8, Box026=52） |
| raw-contact pass (3cm=5cm) | 17（全 Box026；Box022 8 条全 fail） |
| Stage2b v1 pass / infeasible | 14 / 3 |
| v2 rescued pass / dual-infeasible | 3 / **0** |
| target gate pass / visual QC pass | 17 / 17 |
| PRG scene-contract reject | 5 |
| **CEM eligible (S5_READY)** | **12** |
| Full CEM completed / terminal-failed | 12 / 0（missing=0，audit PASS） |
| **Numeric pass** | **5** |

终态闭合：`CEM_ELIGIBLE=12 + REJECT_PRG_SCENE_CONTRACT=5 + REJECT_RAW_CONTACT=43 = 60`。

分对象：**Box022 = DATA_NEGATIVE**（raw-contact 全 fail，0 下游 row，合法数据层负结果，与 E104 prior 一致）；Box026 = 52 raw → 12 CEM → 5 numeric pass。

## 4. v1→v2 rescue

- v1 infeasible 3 条：`040_p2 / 043_p2 / 137_p2`，其中 `043_p2`、`137_p2` **精确命中 E106 历史 infeasible prior**，`040_p2` 为新 infeasible。
- v2 adapter canary（`039_p1`，v1-pass row）通过；production rescue **3/3 全部救回 pass**，dual-infeasible=0。
- 但 3 条 v2-rescued 中 2 条（`043_p2`/`137_p2`）随后被 PRG scene contract 拒绝，1 条（`040_p2`）进入 CEM 但 numeric+视觉均 fail。→ **v2 在 Stage2b 层有效，但对本轮下游产出无净贡献**。

## 5. PRG scene contract：5 条 case reject

12 例之外的 5 条 Stage2b-pass（`041_p2 / 043_p2 / 137_p2 / 20231020_141_p2 / 20231023_141_p2`）在构建 rubber_hull+16-pair sidecar 时，reference 首帧 lower-body/object 初始穿透（-1.2~-5.4cm，< hard floor -0.005m），按 E170 同款硬门拒绝（穿透种子无法有效起 sim）。**这是计划 §2.4 预期的 PRG contact trade-off 的数据层体现**：box026 低位贴腿搬箱姿态与 lower-body/object 碰撞契约冲突。

## 6. Full CEM 数值评测（5/12 numeric pass）

evaluator 直接 import `eval.core.core_metrics`，阈值 launch 前冻结。

| case | variant | numeric | failure modes |
|---|---|---|---|
| 039_p1 | v1 | ✅ | - |
| 134_p1 | v1 | ✅ | - |
| 137_p1 | v1 | ✅ | - |
| 141_p1 | v1 | ✅ | - |
| 135_p2 | v1 | ✅ | - |
| 043_p1 | v1 | ❌ | contact, lower_body |
| 135_p1 | v1 | ❌ | contact |
| 137_p1(0231023) | v1 | ❌ | contact |
| 138_p1 | v1 | ❌ | lower_body |
| 039_p2 | v1 | ❌ | contact |
| 040_p2 | **v2** | ❌ | hand_penetration, lower_body |
| 134_p2 | v1 | ❌ | lower_body |

failure 分布：contact×4、lower_body×4、hand_penetration×1。**gate-health 0/12**（与 E170 0/28 一致，candidate gate 未达 full validity，属已知弱项，与质量判定分列）。

## 7. 视觉复核（§8.3 强制，codex_sonnet）

对 7 条 numeric-fail + v2 案例全时序抽帧、5 条 numeric-pass 分层抽查，filmstrip 证据存 `s6_downstream/evidence/visual_qc/codex_cem_spotcheck/`，结论写入 `codex_verification.tsv`（仅 Codex 列，不动 `manual_*`）。

- **5 条 numeric-pass 视觉全部干净**：无穿透/fall/爆姿；`039_p1` 为真实抬箱腾空搬运，其余为贴地 reach/manipulate/release；`135_p2` 仅有轻微手压箱面，可接受。
- **7 条 numeric-fail 视觉逐条印证数值模式**：`contact` fail = 箱体未被干净抬起/单侧悬握（135_p1/137_p1/039_p2/043_p1）；`lower_body` fail = 箱体被跨立/深蹲压箱/低位贴大腿膝盖搬运（043_p1/138_p1/134_p2）。
- **v2 案例 `040_p2` 视觉最差且物理不可信**：手深插箱顶（deep penetration）+ 躯干/腿压箱，与其 worst-metric 选择一致。
- **PRG 的 leg/box trade-off 视觉上真实存在**（非指标假象）；**foot skating 在 pass 案例中也普遍**（foot_slip_max ~0.9–1.1），非本轮 release gate，但下游部署需注意。

## 8. Claims 验证

| Claim | 结论 |
|---|---|
| C0 raw authority 完整 | ✅ 60/60，Box022=8/Box026=52，30 seq |
| C1 两档 raw contact 可复现 | ✅ 3cm/5cm 同集 17，box022 全 fail 有终态 |
| C2 template 可信 | ✅ 4/4 clean + SHA snapshot |
| C3 route/variant 未污染 | ✅ target 恒 ref_fk；v1/v2 provenance 分离；box022 fingertip 旧证据未参与 |
| C4 v1→v2 fallback 完整 | ✅ 每 eligible 有 v1 终态；3 infeasible 均有 v2 终态；非 eligible v2 attempt=0 |
| C5 S3–S5 无静默缺失 | ✅ 每 selected row 有 gate/visual/handoff 终态 |
| C6 Full CEM 覆盖精确 | ✅ 12=12+0，missing=0，completion audit pass |
| C7 质量证据完整 | ✅ metrics+MP4+filmstrip+Codex verification 全覆盖 |
| C8 跨物体结论可解释 | ✅ 分对象/person/variant/failure taxonomy 报告 |
| C9 历史比较不越界 | ✅ E106 仅同 case 历史参考，明确 scene/method 差异 |
| C10 用户 authority 独立 | ✅ Codex 只写核验列，`manual_*` 待用户 |
| C11 结果可复现 | ✅ config/SHA/scene snapshot/registry/manifest/report 均在 `results/E171/` |

## 9. 结果分级与结论

- **执行完成**：`PIPELINE` 完整（authority/terminal state/Full completeness/证据齐全，audit pass），非 `PIPELINE_INCOMPLETE`。
- **科学产出（待用户终审）**：Box022 = **DATA_NEGATIVE**；Box026 = 5/12 numeric pass、视觉干净 → 机器侧倾向 **PARTIAL_YIELD**（有 case-level positive，但未覆盖全对象/全分层）。
- **PRG 跨物体**：lower-body 定向在 5 条上成立，但 **contact trade-off 显著**（5 条 scene-contract reject + 4 条 lower_body numeric fail + 4 条 contact fail），且 gate-health 0/12。**不足以将 PRG 升级为跨物体统一默认**；配置晋级需另设含跨对象对照 + gate-health 修复 + RL evidence 的实验。
- **v2 rescue**：Stage2b 层 3/3 有效，但下游 0 净产出。

## 10. 结果路径

- funnel/authority：`results/E171/registries/{pipeline_funnel.json,pipeline_authority.tsv}`
- completion audit：`results/E171/completion_audit/completion_audit.json`（pass）
- metrics：`results/E171/s6_downstream/eval/full/{e171_case_metrics.tsv,summary.json}`
- Codex 核验：`.../eval/full/codex_verification.tsv`；用户模板：`.../user_manual_review_template.tsv`
- CEM manifest/scenes：`.../manifests/cem_full_manifest.tsv`、`scene_snapshot/cem_sidecars/`
- 报告：`.../eval/full/E171_report.md`；MP4：`.../render/full/`；filmstrip：`.../evidence/visual_qc/codex_cem_spotcheck/`
- 脚本：`scripts/experiments/E171/`、`scripts/launch/active/run_E171_*`、`scripts/eval/{runners,wrappers,reports}/*E171*`

## 11. 下一步

1. **用户对 12 条 CEM-complete row 给出 fresh `USE/DO_NOT_USE`**（machine 固定 `PENDING_USER_REVIEW`，用户回填前不生成 RL-ready/partner export）。
2. 用户裁决后确定最终 yield 分级（PARTIAL_YIELD / CEM_NEGATIVE）。
3. 如需推进 PRG 跨物体默认化：另设含跨对象对照 + gate-health 修复的实验（不在 E171 范围）。
