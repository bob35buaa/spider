# E112 Contact-Aware CEM Ablation Plan

日期：2026-06-02

总控计划：`workspace/core4d/plan/contact_improvement_plan.md`

前置实验：

- E110 contact metric audit：`workspace/core4d/log/140_E110_contact_metric_audit_results.md`
- E111 contact evidence chain：`workspace/core4d/log/141_E111_contact_evidence_chain_results.md`
- E109 24-case Spider vs OmniRetarget：`workspace/core4d/results/E109/expanded_24_work_cases/`

## 1. Context

E109/E110 显示 Spider CEM 相比 OmniRetarget 明显降低 deep hand-object penetration，但同时降低 `hand_object_physics_contact` 和 3/5cm near-contact。E110 的主标签以 `penetration_removed_contact_not_recovered` 为主，说明当前 `ref_fk + safety` stack 更倾向于把压入式接触修成“安全但有间隙”的解。

E111 已补齐 raw-contact artifact、manifest 传播和 S6 contact alignment evaluator。E112 开始验证接触下降是否能通过 contact-aware CEM 在不放松 safety 的前提下补回。

## 2. Claims

| Claim | 验证方式 |
|---|---|
| C1: 只加入 raw 3cm contact mask 能把优化集中到真实接触时窗，减少无意义接触/穿透 | `raw_mask_ref_fk` vs `baseline_ref_fk` 的 contact recall/physics contact、deep penetration |
| C2: mask + hold-contact near-field band 能补回“无穿透但隔几厘米”的失败模式 | `hold_band` vs `raw_mask_ref_fk` 的 hand-object contact 和 0-5cm band |
| C3: 接触和穿透不是不可兼得；若方法正确，接触提升不应伴随 deep penetration 明显回升 | deep penetration 增量 `<= 3pp`，leg/body/fall strict 不退化 |
| C4: 若 Phase A 全失败，主要问题不在时窗/hold，而在 target geometry 或 hand/object surface target | 失败后不扩到 E113，转入 `sdf_zero_band/raw_surface_target/handbox_surface_target` 诊断 |

## 3. Scope

### Phase A: 3 cases x 3 variants

本轮先做最小 full-CEM 消融，覆盖中箱、小箱和 Box021 strict positive，不直接跑 surface-target 系列。

| Case | source task | derived task | person | baseline override source | contact mask source | 选择理由 |
|---|---|---|---:|---|---|---|
| box004 guard | `e091_box004_20231003_2_082_p1` | `e091_box004_20231003_2_082_p1_e096b_mask_cem` | 0 | E096b P2 | `workspace/core4d/results/E096b/contact_masks/e091_box004_20231003_2_082_p1/raw_contact_mask_3cm.npz` | 保护已有 box004 正例，检验 mask/hold 是否破坏 guard |
| box021 strict positive | `d003_box021_20231011_035_p1` | `d003_box021_20231011_035_p1_e107_clean` | 0 | E107C02 | `workspace/core4d/results/E079/contact_masks/box021_person1/raw_contact_mask_3cm.npz` or case-specific regenerated mask | E107 strict positive；E110 中接触/穿透 trade-off 代表 |
| box026 strict positive | `e091_box026_20231020_135_p1` | `e091_box026_20231020_135_p1_e106_clean` | 0 | E106B22 | E104 raw-contact proxy converted to MJWP-compatible mask | 大箱 strict positive，验证方案是否跨 object scale 有效 |

Phase A variants：

| Variant | 改动 | 预期 |
|---|---|---|
| `baseline_ref_fk` | 当前 ref-FK clean stack；无 contact mask、无 hold | 复现 E096b/E107/E106 baseline 口径，作为 per-case 对照 |
| `raw_mask_ref_fk` | baseline + `contact_hdmi_mask_source=core4d_3cm` | 接触 reward 只在 raw-contact active hand/frame 生效，减少错帧接触 |
| `hold_band` | `raw_mask_ref_fk` + `hold_contact_rew_scale=1.0` + `hold_contact_sigma=0.05` | 鼓励 raw-contact 时窗内维持浅近场贴合，同时不关闭 penetration/safety |

Split：

| Split | Runs |
|---|---|
| local-gpu0 | 3 x `baseline_ref_fk` |
| remote-gpu0 | 3 x `raw_mask_ref_fk` |
| remote-gpu1 | 3 x `hold_band` |

### Phase B: only if Phase A passes

若至少一个 contact-aware variant 达到成功标准，再扩展：

- 加入 `d003_box021_20231011_035_p2` 或 `d003_box021_20231018_029_p2` borderline；
- 加入 `bucket004_20231002_022_p1` nonbox；
- 跑 `sdf_zero_band/raw_surface_target/handbox_surface_target`。

## 4. Implementation

新增固定入口：

| 类型 | 路径 |
|---|---|
| Manifest/override builder | `workspace/core4d/scripts/E112/build_contact_aware_cem_manifest.py` |
| Variants table | `workspace/core4d/scripts/E112/variants.tsv` |
| Preflight report | `workspace/core4d/results/E112/preflight/phaseA_preflight.tsv` |
| Training runner | `workspace/core4d/scripts/train/train_E112_contact_aware_cem.sh` |
| Remote runner | `workspace/core4d/scripts/run_E112_remote.sh` |
| Remote pull | `workspace/core4d/scripts/pull_E112_remote_results.sh` |
| Evaluation runner | `workspace/core4d/scripts/eval/eval_E112_contact_aware_cem.sh` |

Builder 必须检查：

- source override 存在；
- derived task 的 `example_datasets/processed/<task>/scene.xml` 和 `scene_act.xml` 存在；
- mask NPZ 存在，且包含 `eval_contact_mask_3cm` 或 `spider_contact_mask_3cm`，shape 为 `(T, person, hand)`；
- `person_idx` 在 mask person 维度内；
- 每个 variant 对应 override YAML 写入 `examples/config/override/core4d_<variant>.yaml`；
- `hold_band` 不修改 deep penetration/safety 相关 penalty，只增加 hold band。

如果 E104 raw-contact proxy 只有 `raw_contact_mask_3cm`，builder 需要生成 E112 专用 MJWP-compatible mask，写到：

```text
workspace/core4d/results/E112/contact_masks/<case_id>/raw_contact_mask_3cm.npz
```

并记录来源和转换规则。不能直接让 CEM 读取不匹配 key 的 raw proxy。

## 5. Commands

Manifest/preflight：

```bash
.venv/bin/python workspace/core4d/scripts/E112/build_contact_aware_cem_manifest.py
```

Smoke：

```bash
bash workspace/core4d/scripts/train/train_E112_contact_aware_cem.sh local smoke 0
bash workspace/core4d/scripts/train/train_E112_contact_aware_cem.sh remote-gpu0 smoke 0
bash workspace/core4d/scripts/train/train_E112_contact_aware_cem.sh remote-gpu1 smoke 1
```

Full CEM：

```bash
bash workspace/core4d/scripts/train/train_E112_contact_aware_cem.sh local full 0
bash workspace/core4d/scripts/run_E112_remote.sh full
bash workspace/core4d/scripts/pull_E112_remote_results.sh full
bash workspace/core4d/scripts/eval/eval_E112_contact_aware_cem.sh full
```

Full CEM 启动条件：

- 本地 smoke 通过；
- remote 代码/配置/scene/mask 与本地一致；
- 不 kill 远程已有 GPU 进程；
- 若存在未提交本地改动导致 `git push` 不可同步，必须先选择可复现同步方式或暂停 full CEM。

## 6. Success Criteria

逐 case 相对 `baseline_ref_fk`：

- `raw_contact_recall` 或 `hand_object_physics_contact` 提升 `>= 8pp`；
- `hand_geom_deep_penetration_2cm` 增加 `<= 3pp`；
- pelvis/fall 不退化；
- leg penetration strict 不退化；
- object error 不超过原 baseline 可接受范围；
- MP4/keyframes 视觉复核无手背假接触、腿/身体补偿性穿透、物体明显漂移。

阶段通过：

- 至少 2/3 case 有一个 contact-aware variant 满足 per-case 成功标准；
- box004 guard 不出现明显 regression；
- 至少一个 variant 可作为 E113 candidate。

## 7. Failure Decisions

| 观察 | 决策 |
|---|---|
| `raw_mask_ref_fk` 接触提升但 deep penetration 回升 | 不提高 contact gain；进入 `sdf_zero_band` 或 penetration hinge |
| `raw_mask_ref_fk` 无提升，`hold_band` 有提升但腿/身体退化 | 加 lower-body clearance gate；该 case 不进 RL handoff |
| 两个 contact-aware variant 都无提升 | 诊断 target geometry reachability，标记 `contact_geometry_unreachable` 候选 |
| box004 guard regression | 停止扩展；先修 reward 权重/时窗或回退该 object |

## 8. Logging

结果 log 写到：

```text
workspace/core4d/log/142_E112_contact_aware_cem_ablation_results.md
```

必须记录：

- variants/preflight 路径；
- 本地和远程 tmux/session/log 路径；
- 每个 run 的 NPZ/MP4/keyframes；
- E111 S6 alignment 指标；
- E109/E110 同口径 replay 指标；
- Phase A 是否允许进入 Phase B/E113。
