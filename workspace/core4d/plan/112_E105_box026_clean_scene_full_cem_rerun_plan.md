# E105 计划：E103 template 修复后的 Box026 clean-scene full CEM 重跑

日期：2026-06-01
分支：`exp/core4d-collab-retarget`

上游：

- E092: `workspace/core4d/plan/98_E092_three_case_spider_dynamic_and_omniretarget_rl_plan.md`
- E094: `workspace/core4d/log/116_E094_g1_handbox_target_projection_results.md`
- E100/E101: `workspace/core4d/results/E100/fingertip_targets/`, `workspace/core4d/log/125_E101_box021_d003_rerun_with_new_target_results.md`
- E103: `workspace/core4d/log/128_E103_source_template_rebuild_results.md`, `workspace/core4d/log/130_E103_rebuilt_target_regeneration_results.md`
- E104: `workspace/core4d/log/131_E104_d002_multithreshold_remine_results.md`

## 背景

E103 发现并修复了 CORE4D source template 的重大数据 bug：旧 `box021`/`box026` 派生 scene 中 robot link inertial 被污染为同一个 `mass=29.632` / 同一惯量。E103 Phase 1/2 已从 clean `box023_person1` base 重建 `box026_person1/2` source templates；Phase 3 已重新生成并验证两个历史 Box026 target：

| target | E103 status | qpos | scene | scene_act |
|---|---|---|---|---|
| `e091_box026_20231018_039_p2` | regenerated_and_verified | `[123,43]`, matches SPIDER qpos | `nq=43,nv=41,nu=29` | `nq=42,nv=41,nu=35` |
| `e091_box026_20231020_135_p2` | regenerated_and_verified | `[82,43]`, matches SPIDER qpos | `nq=43,nv=41,nu=29` | `nq=42,nv=41,nu=35` |

本轮目标是重跑历史上真正跑过的 Box026 full CEM case，并补充 E101-style fingertip target ablation，形成 `ref_fk_clean / adaptive_clean / fingertip_clean` 三路线对比。旧结论因为 scene inertial 污染，不能再直接作为算法失败证据。

## 历史 case 对齐

必须对齐之前实验跑过的 Box026 full CEM：

| old experiment | old variant | source task | old route | old status | old key metrics | 可信度 |
|---|---|---|---|---|---|---|
| E092 | `E092D2_box026_039_p2_dyn` | `e091_box026_20231018_039_p2` | ref_fk / spider_dyn full | FAIL | contact 33.3%, obj mean 0.009m, pelvis 0.083m | invalidated by scene inertial bug |
| E092 | `E092D3_box026_135_p2_dyn` | `e091_box026_20231020_135_p2` | ref_fk / spider_dyn full | FAIL | contact 57.3%, obj mean 0.008m, pelvis 0.177m, RH floor 17.1% | invalidated by scene inertial bug |
| E094 | `E094P2_box026_039_p2_hbproj` | `e091_box026_20231018_039_p2` | adaptive-support external target full | FAIL | contact 80.5%, obj mean 0.002m, pelvis 0.440m | invalidated by scene inertial bug |
| E094 | `E094P3_box026_135_p2_hbproj` | `e091_box026_20231020_135_p2` | adaptive-support external target full | FAIL | contact 41.5%, obj mean 0.010m, pelvis 0.171m, RH floor 22.0% | invalidated by scene inertial bug |

明确不作为 primary full-CEM rerun：

- `e091_box026_20231018_040_p2`：E091 记录为 OmniRetarget CVXPY infeasible，没有旧 full CEM 对齐对象。
- E092 `rl_from_omni smoke`：不是 full CEM，本轮只在需要时作为附录对照，不进入 primary matrix。
- E104 新挖出的其它 Box026 candidates：先不扩量；E105 只重跑历史结论被污染影响的核心 case。

## E101 风格 fingertip 消融

E101 的方法不是 E092 `ref_fk`，也不是 E094 `adaptive-support`。它承接 E100 的 fingertip-aware external contact target：按 E099 fingertip vote face，把手部目标投到物体局部表面，再让 CEM 追踪 `contact_hdmi_target_source=external`。

历史上 E101 原计划 Phase 2 会推广到 Box026，但因为 box021 D003 Phase 1 stop-loss，没有真正跑到 Box026 full CEM。因此 E101-style 在 E105 中定位为 **secondary ablation**，不是 historical primary rerun。

E100 已存在两条 Box026 fingertip target：

| source task | E100 target status | face_changed_L/R | implication |
|---|---|---|---|
| `e091_box026_20231018_039_p2` | target exists | `False/False`, vote `-z/-z` | 不是换面修复，主要检验 fingertip target 约束本身 |
| `e091_box026_20231020_135_p2` | target exists | `False/False`, vote `-x/-x` | 不是换面修复，主要检验 fingertip target 约束本身 |

E105 不直接把 E100 NPZ 当作权威输入。计划在 E105 目录重新生成或重校验 fingertip targets，并写入 source scene path、trajectory hash、target shape、vote face metadata。

## 核心原则

E105 不复用旧 `_e092_dyn` / `_e092_omni` derived task directory。

预检查已确认：

| task | current status |
|---|---|
| `e091_box026_20231018_039_p2` | clean |
| `e091_box026_20231020_135_p2` | clean |
| `e091_box026_20231018_039_p2_e092_dyn` | polluted_robot_inertial |
| `e091_box026_20231020_135_p2_e092_dyn` | polluted_robot_inertial |
| `e091_box026_20231018_039_p2_e092_omni` | polluted_robot_inertial |
| `e091_box026_20231020_135_p2_e092_omni` | polluted_robot_inertial |

因此 E105 必须新建 clean derived tasks，例如：

- `e091_box026_20231018_039_p2_e105_clean`
- `e091_box026_20231020_135_p2_e105_clean`

这两个 derived tasks 从 E103 clean target task 复制，重新 patch E083-style leg/upper-body object collision pairs，重新生成 scene_act；任何 CEM run 都必须指向这些 clean derived tasks。

## 验证声明

### C1: clean-scene gate is enforced before any CEM

判据：

- `box026_person1/2`、两个 source target、两个 E105 derived task 都通过 inertial audit：
  - `status=clean`
  - `robot_polluted_mass_29_632=False`
  - `robot_inertial_unique_pairs > 1`
- MuJoCo load：
  - source scene `nq=43,nv=41,nu=29`
  - derived `scene_act` `nq=42,nv=41,nu=35`
- qpos：
  - derived `0/trajectory_kinematic.npz` 与 E103 verified qpos 一致；
  - T 分别为 123 / 82。
- 若任一 clean gate 失败，E105 停止，不启动 CEM。

### C2: historical full-CEM variants are exactly aligned

判据：

- 新 matrix 必须包含 4 个 primary rerun：
  - E092D2 equivalent: `039_p2 ref_fk`
  - E092D3 equivalent: `135_p2 ref_fk`
  - E094P2 equivalent: `039_p2 adaptive-support target`
  - E094P3 equivalent: `135_p2 adaptive-support target`
- 每个新 variant 记录 old variant name、source task、new clean task、config diff、old summary path。
- E094 adaptive-support target 必须重新生成到 E105 目录；不能直接引用旧 `results/E094/.../targets/*.npz` 作为权威输入。可以生成后和旧 NPZ 做 diff，用于说明是否只因 scene 变化而变。

### C3: all E105 variants complete on local + remote GPUs

判据：

- 6/6 variants 有：
  - root NPZ
  - outdir `trajectory_mjwp_act.npz`
  - MP4
  - keyframes
  - eval metrics JSON/CSV/MD
- 本地 1 卡 + 远程 2 卡，按 3-card waves 执行；每轮同时跑 3 个实验，6 个实验分 2 waves 完成。
- 远程执行必须通过脚本，不手工裸命令：
  - `workspace/core4d/scripts/run_E105_remote.sh`
  - `workspace/core4d/scripts/pull_E105_remote_results.sh`

### C3b: every CEM run has pre-run MuJoCo visual gate + medium subagent review

判据：

- 每个 variant 在 full CEM 前必须完成 pre-CEM visual package：
  - clean derived task kinematic replay MP4；
  - target overlay / marker replay MP4 或 keyframe sheet；
  - qpos/object/contact timeline summary；
  - config target route metadata。
- 每个 variant 必须有 medium subagent preflight review，输出到：

```text
workspace/core4d/results/E105/pre_cem_visual_review/{variant}/REVIEW.md
```

- medium subagent 判定必须是 `PASS` 或 `PASS_WITH_NOTES` 才能启动该 variant 的 full CEM。
- 若任一 variant 被判 `FAIL_PRE_CEM_VISUAL`，该 variant 不启动 CEM；先修 task/target/scene，再重新生成可视化并复审。

### C4: numerical comparison answers whether old Box026 conclusion changes

判据：

输出 `workspace/core4d/results/E105/comparison/box026_clean_vs_old_comparison.{csv,md}`，至少包含：

| metric group | fields |
|---|---|
| object | `obj_err_mean_m`, `obj_err_max_m`, object z/tilt if available |
| contact | `contact_frac_either`, L/R contact fractions |
| posture | `pelvis_min_m`, `pelvis_end_m`, pelvis/torso tilt if available |
| safety | head/upper penetration, LH/RH floor, lie-on-box/body-on-object gate |
| validity | old_scene_status, new_scene_status, target_clean, source_clean |
| decision | old_status, new_status, status_delta, interpretation |

比较维度：

- E092D2 old vs E105 ref_fk 039
- E092D3 old vs E105 ref_fk 135
- E094P2 old vs E105 adaptive 039
- E094P3 old vs E105 adaptive 135
- E105 ref_fk vs E105 adaptive vs E105 fingertip within each source task

### C5: visual comparison is sufficient for human review

判据：

- 每个 E105 run 生成 autocam video + keyframe sheet。
- 生成 side-by-side old vs new review pages：
  - 旧 E092/E094 视频帧
  - 新 E105 视频帧
  - metric deltas
- 输出：
  - `workspace/core4d/results/E105/visuals/box026_clean_rerun/REVIEW.md`
  - 每个 variant 的 contact/object/pelvis timeline PNG
- 若视频存在，必须用 video-frames 或离线 keyframes 写入 log 的“实际观察”；不能只写数值。

## 实验矩阵

| new variant | aligns old variant | source task | clean derived task | target route | split |
|---|---|---|---|---|---|
| `E105R1_box026_039_p2_ref_fk_clean` | `E092D2_box026_039_p2_dyn` | `e091_box026_20231018_039_p2` | `e091_box026_20231018_039_p2_e105_clean` | `ref_fk` | wave1 remote-gpu0 |
| `E105R2_box026_135_p2_ref_fk_clean` | `E092D3_box026_135_p2_dyn` | `e091_box026_20231020_135_p2` | `e091_box026_20231020_135_p2_e105_clean` | `ref_fk` | wave1 remote-gpu1 |
| `E105A1_box026_039_p2_adaptive_clean` | `E094P2_box026_039_p2_hbproj` | `e091_box026_20231018_039_p2` | `e091_box026_20231018_039_p2_e105_clean` | `adaptive-support external target` | wave1 local-gpu0 |
| `E105A2_box026_135_p2_adaptive_clean` | `E094P3_box026_135_p2_hbproj` | `e091_box026_20231020_135_p2` | `e091_box026_20231020_135_p2_e105_clean` | `adaptive-support external target` | wave2 remote-gpu0 |
| `E105F1_box026_039_p2_fingertip_clean` | E101-style secondary ablation | `e091_box026_20231018_039_p2` | `e091_box026_20231018_039_p2_e105_clean` | `E100/E101 fingertip external target` | wave2 local-gpu0 |
| `E105F2_box026_135_p2_fingertip_clean` | E101-style secondary ablation | `e091_box026_20231020_135_p2` | `e091_box026_20231020_135_p2_e105_clean` | `E100/E101 fingertip external target` | wave2 remote-gpu1 |

Rationale:

- Wave1 starts three high-signal historical primary routes: two ref_fk on remote, the most promising old near-pass (`039_p2 adaptive`) locally.
- Wave2 runs the remaining historical primary (`135_p2 adaptive`) plus the two E101-style fingertip ablations.
- This uses exactly 3 GPUs per wave: local GPU0 + remote GPU0 + remote GPU1.

## 实现计划

### Phase 0: historical manifest + clean gate

New files:

- `workspace/core4d/scripts/E105/build_box026_historical_manifest.py`
- `workspace/core4d/results/E105/box026_historical_full_cem_manifest.tsv`
- `workspace/core4d/results/E105/clean_scene_preflight.tsv`

Tasks:

1. Read old summaries:
   - `workspace/core4d/results/E092/spider_dyn/full/full_eval_summary.csv`
   - `workspace/core4d/results/E094/cem/full/full_eval_summary.csv`
2. Freeze old metrics into E105 comparison input.
3. Run inertial audit on:
   - source target tasks
   - old `_e092_*` tasks
   - new E105 tasks after generation
4. Explicitly mark old `_e092_*` as invalidated evidence, not reusable input.

### Phase 1: build clean E105 tasks and targets

New files:

- `workspace/core4d/scripts/E105/build_box026_clean_tasks.py`
- `workspace/core4d/scripts/E105/build_e105_adaptive_targets.py`
- `workspace/core4d/scripts/E105/build_e105_fingertip_targets.py`
- `workspace/core4d/scripts/E105/variants.tsv`

Behavior:

1. Copy from E103 clean source target tasks, not from `_e092_dyn`.
2. Patch leg/foot + upper-body object pairs using E083 helper.
3. Generate or verify `scene_act.xml`.
4. Rebuild adaptive-support external targets into:

```text
workspace/core4d/results/E105/adaptive_targets/
```

5. Rebuild or revalidate E100/E101-style fingertip targets into:

```text
workspace/core4d/results/E105/fingertip_targets/
```

6. Write overrides:

```text
examples/config/override/core4d_E105R1_box026_039_p2_ref_fk_clean.yaml
examples/config/override/core4d_E105R2_box026_135_p2_ref_fk_clean.yaml
examples/config/override/core4d_E105A1_box026_039_p2_adaptive_clean.yaml
examples/config/override/core4d_E105A2_box026_135_p2_adaptive_clean.yaml
examples/config/override/core4d_E105F1_box026_039_p2_fingertip_clean.yaml
examples/config/override/core4d_E105F2_box026_135_p2_fingertip_clean.yaml
```

Override rules:

- Ref-fk route mirrors E092D2/D3 reward/config but points to E105 clean task.
- Adaptive route mirrors E094P2/P3 reward/config but points to E105 clean task and E105 regenerated target NPZ.
- Fingertip route mirrors E101/E100 external target config, but points to E105 clean task and E105 regenerated/revalidated fingertip target NPZ.
- `video_camera=auto`.
- `+use_torch_compile=false`.

### Phase 2: pre-CEM MuJoCo visualization + medium subagent gate

New files:

- `workspace/core4d/scripts/E105/render_pre_cem_mujoco_replays.py`
- `workspace/core4d/results/E105/pre_cem_visuals/{variant}/`
- `workspace/core4d/results/E105/pre_cem_visual_review/{variant}/REVIEW.md`
- `workspace/core4d/results/E105/pre_cem_visual_gate.tsv`

For each of the 6 variants:

1. Render clean derived task kinematic replay in MuJoCo.
2. Render target overlay for the selected route:
   - `ref_fk`: ref body/object/contact replay.
   - `adaptive`: E105 adaptive target markers.
   - `fingertip`: E105 fingertip target markers.
3. Generate 8-12 keyframes and one summary sheet.
4. Spawn medium subagent to check:
   - correct Box026 scene/object and no obvious scale/pose mismatch;
   - robot/object replay is nonblank and temporally aligned;
   - target markers sit on intended object faces and follow the expected hand/object frames;
   - no old `_e092_*` polluted task path appears in config or metadata;
   - no target path points to stale E094/E100 result as the authoritative E105 input.
5. Record `PASS / PASS_WITH_NOTES / FAIL_PRE_CEM_VISUAL`.

Full CEM is blocked until the corresponding row is `PASS` or `PASS_WITH_NOTES`.

### Phase 3: local + remote full CEM

New scripts:

- `workspace/core4d/scripts/train/train_E105_box026_clean_full.sh`
- `workspace/core4d/scripts/run_E105_remote.sh`
- `workspace/core4d/scripts/pull_E105_remote_results.sh`

Output:

```text
workspace/core4d/results/E105/cem/full/
logs/E105/cem/full/
logs/E105/remote/
```

Execution split:

| wave | machine | GPU | variant |
|---|---|---:|---|
| wave1 | local | 0 | `E105A1_box026_039_p2_adaptive_clean` |
| wave1 | remote | 0 | `E105R1_box026_039_p2_ref_fk_clean` |
| wave1 | remote | 1 | `E105R2_box026_135_p2_ref_fk_clean` |
| wave2 | local | 0 | `E105F1_box026_039_p2_fingertip_clean` |
| wave2 | remote | 0 | `E105A2_box026_135_p2_adaptive_clean` |
| wave2 | remote | 1 | `E105F2_box026_135_p2_fingertip_clean` |

Planned commands after implementation:

```bash
python workspace/core4d/scripts/E105/build_box026_clean_tasks.py --force
python workspace/core4d/scripts/E105/build_e105_adaptive_targets.py --force
python workspace/core4d/scripts/E105/build_e105_fingertip_targets.py --force
python workspace/core4d/scripts/E105/render_pre_cem_mujoco_replays.py --all
# only after pre_cem_visual_gate.tsv passes:
bash workspace/core4d/scripts/train/train_E105_box026_clean_full.sh local wave1 0
bash workspace/core4d/scripts/run_E105_remote.sh wave1
bash workspace/core4d/scripts/pull_E105_remote_results.sh wave1
bash workspace/core4d/scripts/train/train_E105_box026_clean_full.sh local wave2 0
bash workspace/core4d/scripts/run_E105_remote.sh wave2
bash workspace/core4d/scripts/pull_E105_remote_results.sh wave2
```

Remote launch requires clean git sync:

```bash
git status --short
git push origin exp/core4d-collab-retarget
```

Do not upload `workspace/core4d/results/` to git.

### Phase 4: eval + visual review

New files:

- `workspace/core4d/scripts/eval/eval_E105_box026_clean_cem.py`
- `workspace/core4d/scripts/E105/render_box026_clean_comparison.py`
- `workspace/core4d/results/E105/cem/full/full_eval_summary.{json,csv,md}`
- `workspace/core4d/results/E105/comparison/box026_clean_vs_old_comparison.{csv,md}`
- `workspace/core4d/results/E105/visuals/box026_clean_rerun/REVIEW.md`

Evaluation gates:

| status | condition |
|---|---|
| `WORK` | `T>=80`, `obj_mean<=0.10m`, `obj_max<=0.30m`, `pelvis_min>=0.55m`, head/upper/LH/RH floor all `<=5%`, lie/body-on-box gate pass, visual pass |
| `REVIEW+` | object/contact good, `pelvis_min>=0.45m`, no severe fall/penetration, visual ambiguous |
| `FAIL_POSTURE` | object/contact acceptable but pelvis/fall/lie-on-box fails |
| `FAIL_OBJECT` | object error/tilt/transport fails |
| `FAIL_LOAD` | scene/rollout missing or MuJoCo load fails |

Visual requirements:

- Autocam MP4 for all 6 variants.
- 8-12 keyframes per variant.
- Old-vs-new sheet per aligned pair.
- High subagent visual review after videos are generated.
- Three-route sheet per source task: `ref_fk_clean / adaptive_clean / fingertip_clean`.

### Phase 5: log + decision

Create:

- `workspace/core4d/log/132_E105_box026_clean_scene_full_cem_rerun_results.md`

Update:

- `workspace/core4d/EXPERIMENT_TRACKER.md`
- `workspace/core4d/progress.md`

Decision cases:

| outcome | interpretation | next step |
|---|---|---|
| Box026 flips to WORK under clean scene | old template/inertial bug was a major causal blocker | expand E104 Box026 candidates with same clean pipeline |
| 039 improves but remains REVIEW+/FAIL_POSTURE | template bug contributed, but posture gate still needed | open E106 posture/upright hard gate on 039 |
| 135 still severe fail | sequence-specific reference/reach issue remains | keep 135 as negative stress-test, do not expand similar action without new gate |
| both still fail similarly | old conclusions qualitatively stand despite invalidated scenes | focus on posture/support algorithm, not data template |
| adaptive beats ref_fk clearly | E094 target repair remains useful after clean scene | use adaptive target for Box026 expansion |
| ref_fk beats adaptive | E094 projection was overfitting or introducing bad posture pressure | revisit target projection before expansion |
| fingertip beats both or ties adaptive | E101-style target is a simpler viable target source | consider fingertip target route for E104 Box026 candidate expansion |
| fingertip does not improve over ref_fk | fingertip face/target is not the main Box026 bottleneck | keep focus on posture/support or adaptive projection |

## 产物

| deliverable | path |
|---|---|
| plan | `workspace/core4d/plan/112_E105_box026_clean_scene_full_cem_rerun_plan.md` |
| variants | `workspace/core4d/scripts/E105/variants.tsv` |
| build scripts | `workspace/core4d/scripts/E105/*.py` |
| train script | `workspace/core4d/scripts/train/train_E105_box026_clean_full.sh` |
| remote/pull scripts | `workspace/core4d/scripts/run_E105_remote.sh`, `workspace/core4d/scripts/pull_E105_remote_results.sh` |
| eval script | `workspace/core4d/scripts/eval/eval_E105_box026_clean_cem.py` |
| numeric summary | `workspace/core4d/results/E105/cem/full/full_eval_summary.md` |
| old-vs-new comparison | `workspace/core4d/results/E105/comparison/box026_clean_vs_old_comparison.md` |
| visual review | `workspace/core4d/results/E105/visuals/box026_clean_rerun/REVIEW.md` |
| pre-CEM visual gate | `workspace/core4d/results/E105/pre_cem_visual_gate.tsv` |
| pre-CEM medium reviews | `workspace/core4d/results/E105/pre_cem_visual_review/{variant}/REVIEW.md` |
| final log | `workspace/core4d/log/132_E105_box026_clean_scene_full_cem_rerun_results.md` |

## 风险

| risk | mitigation |
|---|---|
| Accidentally reusing polluted `_e092_dyn` task | hard fail if task path contains `_e092_`; clean gate requires new E105 task names |
| Adaptive target generation silently reuses old scene | write target metadata with source scene path and SHA256; compare against E103 verify |
| Fingertip target silently reuses stale E100 result | regenerate/revalidate into `results/E105/fingertip_targets/` and require E105 metadata before CEM |
| Remote repo not synced | `run_E105_remote.sh` refuses dirty git and runs `git pull --ff-only` |
| Full CEM runtime too long | run exactly 3 variants per wave using local GPU0 + remote GPU0/GPU1 |
| Pre-CEM visualization looks wrong | block that variant before CEM; fix task/target and rerender, then rerun medium subagent review |
| Metrics say pass but video shows posture failure | visual gate is required for `WORK`; subagent review required before final log |
| Results directory is ignored by git | log must record paths and summary tables; do not force-add `results/` unless user explicitly asks |

## 预计时间

| step | estimate |
|---|---:|
| build/preflight scripts | 1.5-2 h |
| pre-CEM MuJoCo visualization + medium review | 0.5-1 h |
| local + remote wave1 | 2-4 h wall time |
| local + remote wave2 + pull | 2-4 h |
| eval/visual/comparison | 1-1.5 h |
| final log/tracker | 0.5 h |
