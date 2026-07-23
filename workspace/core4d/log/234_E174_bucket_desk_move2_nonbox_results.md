# E174 结果日志：Bucket / Desk 非 box 物体 move2 全流程 + Full CEM

_Core4D Phase 37 · 2026-07-23 · plan [190](../plan/190_E174_bucket_desk_move2_full_pipeline_plan.md) · machine recommendation = `PENDING_USER_REVIEW`_

## 0. 一句话结论

E170 PRG + rubber_hull 冻结算法**首次用于非 box 凹几何物体**（5 bucket 空心 + 2 desk 桌腿，move2-only）：Full CEM 39 条 **numeric pass 仅 5/39 = 13%**，远低于 E173 同尺寸凸 box 的尺寸先验（~60–80%）。**关键对照：bucket004(凹,0.045m³)=0/4 vs box004(凸,0.041m³,E172)=83%——同尺寸、凹→全败**。失败集中在两条**物体侧 collision_policy proxy 相关轴**：腿穿进空心 bucket 薄壁（leg_penetration 均值 0.165）、手够不到 ref 接触点（hand_contact_in_mask 均值 0.379）。这是「凹几何 proxy 惩罚」的第一份量化证据。倾向 `PARTIAL_YIELD`（bucket010 2/2、desk007 2/9 可用），待用户对 39 条终审。

## 1. 目的与假设

验证冻结的 E170 PRG（lower-body physics + soft penalty + candidate gate）+ rubber_hull（手侧凸包）+ omnirt v1→v2 + ref_fk 算法，在**非 box 凹几何**物体上是否成立。凹几何由**物体侧 `collision_policy` proxy** 表达（bucket=`bucket_wall_proxy_aabb` 底+四壁；desk=`desk_surface_voxel_multibox_proxy_draft` 表面 voxel），与 rubber_hull（手侧，正交）分离。**假设**：若非 box 只是尺寸效应，bucket~60–80%、desk~35%；若 proxy 失配引入额外接触/穿透误差，实际显著更低，差值 = 凹几何 proxy 惩罚。

## 2. 冻结配置（与 E170–E173 字段级一致）

| 轴 | 值 |
|---|---|
| Retarget | omnirt_v1 primary → omnirt_v2 rescue（仅 v1 `omniretarget_infeasible`） |
| Target route | ref_fk |
| Object collision policy | bucket=`bucket_wall_proxy_aabb`；desk=`desk_surface_voxel_multibox_proxy_draft`（target_cells=26）|
| Hand collision | rubber_hull（lh/rh mesh 凸包 maxhullvert=64，E174 sidecar，与物体正交）|
| Base reward | E167A_zOnlyBody（object-agnostic）|
| Lower-body | 16 geom pair，penalty scale 2.0 / margin 0.02m；candidate gate min-sdf 0.005 / max-viol 0.02 / floor -0.005 |
| CEM | seed 0，samples 1024，opt_steps 32（canary 64/4）|
| method id | `E174_E170PRG_nonbox_candidate_r1` |
| Source config | `E170_PRG_lowerbodyPhysics_softPenalty_candidateGate` |

**scope**：bucket004/009/010/007/003 + desk005/007；**move2-only**（move1→EXCLUDE，其余→Stage0）。

## 3. 运行命令（复现）

```bash
# S0–S5（S2 在非 box template review gate 处停，approve 后重跑续 S3–S5）
bash workspace/core4d/scripts/launch/active/run_E174_data_pipeline.sh
# S2 approve（object-only overlay 视觉核验后）
.venv/bin/python workspace/core4d/scripts/experiments/E174/build_nonbox_template_review.py \
  --out-dir .../s2_templates/review --approve <task> ... --reviewer codex
# S3 v1 并行 + v2 rescue（数据管线内）；S4 visual QC apply pass（--default-pass-status pass）
# S6：override 拷到 examples/config/override/ 后
.venv/bin/python workspace/core4d/scripts/experiments/E174/build_prg_cem_manifest.py
MODE=canary bash workspace/core4d/scripts/launch/active/run_E174_local_8gpu_cem.sh
MODE=full   bash workspace/core4d/scripts/launch/active/run_E174_local_8gpu_cem.sh
STAGE=full bash workspace/core4d/scripts/eval/wrappers/eval_E174_nonbox.sh
bash workspace/core4d/scripts/launch/active/run_E174_render_all.sh
```

## 4. Funnel（fresh authority）

fresh S1 inventory 为 authority：7 物体 **330 pc / 165 seq**（plan prior 240 低估，主因 bucket007=126pc/63seq 是 CORE4D 最常用桶）。

| 阶段 | 计数 |
|---|---|
| raw pc（authority）| **330**（closure pass，missing/pending=0）|
| move2 pc | 92 |
| move2 3cm raw-contact pass | 44 |
| Stage2b pass | **44**（v1 42 + v2 2；dual-infeasible=0）|
| target gate pass | 44 |
| visual QC pass（Codex）| 44 |
| S5 HANDOFF_READY | 44 |
| **CEM-ready（S5_READY_SET）** | **39**（5 PRG scene-contract reject = runtime_initial_overlap，均 p2 起始腿-物重叠，合法，同 E173）|
| Full CEM complete | **39 / 39**（missing=0，terminal_failed=0）|
| **numeric pass** | **5 / 39 = 13%** |

终态闭合：CEM_ELIGIBLE 39 + REJECT_PRG_SCENE_CONTRACT 5 + NOT_STAGE2B_ELIGIBLE 78（move1+nonmove 过 3cm 但被 move2-only 排除）+ REJECT_RAW_CONTACT 208 = 330 ✓。

## 5. 结果（numeric pass 分层）

| 物体 | 类别 | 体积 m³ | CEM-ready | numeric pass |
|---|---|---|---|---|
| bucket004 | bucket | 0.045 | 4 | **0/4** |
| bucket009 | bucket | 0.092 | 1 | 0/1 |
| bucket010 | bucket | 0.120 | 2 | **2/2** |
| bucket007 | bucket | 0.179 | 14 | **0/14** |
| bucket003 | bucket | 0.192 | 9 | 1/9 |
| desk005 | desk | 0.238 | 0 | —（stage2b 无 pass）|
| desk007 | desk | 0.241 | 9 | **2/9** |
| **合计** | | | **39** | **5/39=13%** |

- **分组**：bucket 3/30=10%，desk 2/9=22%。v1 5/38，v2 0/1。
- **失败模式**（fail 34 条）：contact×10、contact+lower_body×10、lower_body×10、hand_penetration×3、release+hand_pen+lower_body×1。
- **聚合指标**：hand_object_contact_in_mask 均值 **0.379**（低，接触建立不良）；leg_penetration_frac 均值 **0.165**（高，腿穿物）；leg_object_contact 均值 0.25。
- **gate-health 0/39**（PRG candidate gate 已知弱项，与 E170 0/28、E171 0/12、E172 0/6、E173 0/53 一致，与质量判定分列）。

## 6. 视觉核验（强制，Codex）

S2 object-only overlay（14 template，4 方位）：bucket wall_proxy 贴合 mesh AABB（圆桶角落 phantom 为 box-hull 固有）、desk voxel_proxy 贴合面板+框架**未桥接桌腿间隙**。S4 target replay（7 代表）全 clean。**S6 Full CEM keyframe（`render/codex_frames/`）**：
- **fail·leg_penetration**（bucket004_021_p1）：把 jerry-can 抱在大腿前搬运→腿夹进桶（真实穿透，与 leg_pen=0.15 一致）。
- **fail·contact**（bucket003_001_p1）：sim 手臂飘到头部附近、始终没抓到垃圾桶盖（wall proxy 顶部无碰撞面，CEM 找不到抓取策略；hand_contact_in_mask=0.00）。
- **pass**（bucket010_055_p1 hand 0.67 / desk007_032_p1 hand 0.86）：sim 紧贴 ref、手贴物、无穿透——真实干净 pass。

结论：**失败均为真实物理，非 reward hacking**，与数值一致。详见 `s6_downstream/eval/full/codex_s6_verification.md`、`s4_gate_visual_qc/codex_review/codex_s4_review.md`。

## 7. Claims 验证

| Claim | 结论 |
|---|---|
| C0 raw authority | ✅ 330/330 closure pass（fresh inventory 为准，prior 240 已更正）|
| C1 action scope | ✅ move2 92→production；move1+nonmove 过 3cm 者标 NOT_STAGE2B_ELIGIBLE（未入 S2–S6）|
| C2 两档 raw contact | ✅ 3cm/5cm 均算，5cm 未覆盖 3cm |
| C3 非 box template | ✅ 14 全 `manual_review_required`→object-only overlay review→`clean_reviewed`；bucket009_p1/desk005_p1 fresh proxy-build；无 `_e0**`/`_s2_` base |
| C4 route/variant | ✅ 全 ref_fk；v1/v2 provenance 分离 |
| C5 v1→v2 fallback | ✅ 42 v1 pass / 2 infeasible→2 v2 救回；dual-infeasible=0 |
| C6 S3–S5 无静默缺失 | ✅ 每 selected row 有终态 |
| C7 Full CEM 覆盖 | ✅ 39=39+0，missing=0 |
| C8 质量证据 | ✅ metrics+MP4×39+keyframe+Codex verification |
| C9 分层结论 | ✅ 按 object/category/variant/failure-mode 报告 |
| C10 凹几何 proxy 证据 | ✅ 量化：bucket004(凹)0% vs box004(凸)83% 同尺寸对照 + 两失败轴（leg_pen/contact-loss）|
| C11 历史比较不越界 | ✅ 仅 warning，无 `_e0**`/`_s2_` 继承 |
| C12 用户 authority | ⏳ Codex 只写核验列；待用户 39 条 USE/DO_NOT_USE |
| C13 复现 | ✅ config/git/SHA/snapshot/registry/commands/NPZ/metrics/video/audit 全在 results/E174/ |

## 8. 分析与结论

1. **凹几何 proxy 惩罚成立且很大**：同尺寸凸→凹（box004→bucket004）通过率 83%→0%。非 box 失败**不是尺寸效应**，而是物体侧 proxy 保真度问题：
   - **wall proxy（bucket）**：空心薄壁 → 抱物时腿穿进桶腔（leg_penetration）；顶部/内壁无碰撞面 → 抓桶盖/内壁的 ref 接触无法建立（contact loss）。
   - **voxel proxy（desk）**：桌面 voxel 覆盖处能建立接触（desk007 2/9，最好条目 hand 0.86），但桌腿/局部接触点仍易 miss。
2. **bucket010 2/2、desk007 2/9 是真实可用条目**：当 ref 接触落在 proxy 有碰撞面的位置（桶沿/桌面）且物体不抱近腿时，PRG+rubber_hull 仍能产出干净结果。
3. **PRG/rubber_hull 不宜作凹几何默认**：适用边界目前止于凸 box + 接触落在 proxy 面上的少数非 box 条目。改进方向是**物体 proxy（更贴合的凹碰撞体、顶面/内壁补面）**，非改 reward/hull——留后续实验。
4. **chair（更极端凹：靠背+细腿）预计更差**，E175 是否做取决于用户是否要继续非 box 方向。

## 9. Yield 分级与下一步

- 机器倾向 **`PARTIAL_YIELD`**（5 条 numeric pass + 视觉干净，逐物体 bucket010/desk007 有产出；bucket004/007/009 = `CEM_NEGATIVE`）。
- machine recommendation 固定 `PENDING_USER_REVIEW`；**待用户对 39 条 CEM-complete 给 USE/DO_NOT_USE**（模板 `s6_downstream/eval/full/user_manual_review_template.tsv`）。
- 未导出 RL/partner。是否进入 chair(E175) / 改 proxy 由用户决定。

## 10. 结果路径

```
workspace/core4d/results/E174/   （软链→ tidal FS: .../spider_workdirs/core4d/results/results/E174；不进 git）
├── s1_raw_contact/{inventory,raw_contact}/
├── s2_templates/{template_backlog.tsv, review/{nonbox_template_review.tsv, object_only_overlay/}}
├── s3_retarget/{omnirt_v1,omnirt_v2,rescue}/
├── s4_gate_visual_qc/{omnirt_v1,omnirt_v2}/ + codex_review/codex_s4_review.md
├── s5_handoff/{handoff_manifest.tsv, cem_overrides/}
├── scene_snapshot/cem_sidecars/   （44 rubber_hull+PRG sidecar）
├── s6_downstream/{preflight/scene_audit.tsv, manifests/{cem_canary,cem_full}_manifest.tsv,
│                  cem/{canary,full}/*.npz, render/full/*.mp4 (39), render/codex_frames/,
│                  eval/full/{e174_case_metrics.tsv, summary.json, e174_group_summary.tsv,
│                            codex_s6_verification.md, E174_report.md, user_manual_review_template.tsv}}
├── registries/{pipeline_authority.tsv, pipeline_funnel.json}
└── completion_audit/{completion_audit.json (pass), pipeline_authority_audit.json}
```

git（active XML）：14 template scene.xml+task_info.json + 7 object `_m.obj` asset `git add -f`（bucket004 asset 补拷、bucket009_p1/desk005_p1 material 名修复）。

## 11. 与 E170–E173 关系

E174 = 跨物体扩展第 5 批，**首次离开 box 拓扑**。算法冻结不变，物体换 5 bucket + 2 desk，动作收紧 move2-only，物体侧改用 dcv3 非 box proxy collision policy + 强制 object-only overlay review→clean_reviewed。**PRG 尺寸-通过率曲线新增凹几何点**：box004(凸0.041)83% ≫ bucket004(凹0.045)0%——凹几何 proxy 惩罚是独立于尺寸的新失败轴。E174 是把 PRG 从凸 box 推向凹几何的第一份（否定性为主）证据。
