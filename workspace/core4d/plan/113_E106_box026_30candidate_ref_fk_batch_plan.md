# E106 计划：Box026 30-candidate clean ref-FK batch CEM

日期：2026-06-01
上游：E103 clean scene rebuild；E104 D002 multi-threshold remine；E105 clean-scene Box026 rerun

## 背景

E105 在 E103 clean scene 下推翻了旧 Box026 full-CEM 的低髋/贴地失败解释：历史 primary rerun 中 3/4 在 upper-body/replay 口径变为 WORK。但 E105 也暴露了新的硬指标缺口：6/6 的 E026/E081 lower-body strict proxy 都 FAIL，说明 Box026 不能因为 upper-body/replay 变好就直接进入 RL-ready。

E104 重新挖掘后给出 Box026 30 条 3cm candidate：

- 2 条 `candidate_legacy_risk_needs_visual`：`e091_box026_20231018_039_p2`、`e091_box026_20231020_135_p2`；
- 28 条 `candidate_executable`；
- 30 条均来自 clean `box026_person1/2` source scene；
- 但当前本地状态显示：只有 4 条已有 SPIDER task，其中只有 2 条已有 Holosoma/OmniRetarget `retargeted/trimmed` 输出。其余候选只是 raw-contact candidate，还不能直接跑 SPIDER CEM。

本轮目标不是调 reward，而是做一次大规模 case sweep：先把 E104 的 30 条 Box026 candidate 补齐到可运行 clean SPIDER task，然后用同一套 E105-clean ref-FK route 跑 full CEM。按用户要求，本地 1 卡 + 远程 2 卡并行，三张卡各约 10 个 case，同卡串行；第一阶段只跑 CEM，不做评测。等 30 条全部完成并拉回本地后，再统一评测和可视化。

## 范围

### Primary matrix

E106 primary route 只跑一条算法路线：

```text
ref_fk_clean = E105R-style clean derived task + core4d_E089A safety stack + ref FK target
```

理由：

- E105 中 `ref_fk_clean` 两条历史 Box026 都是 upper-body/replay WORK，是最适合大规模筛 case 的基线；
- adaptive/fingertip route 需要为每条候选额外构造 external target，容易把 data readiness 和算法路线混在一起；
- 用户要求的是 30+ candidate 大批量“先都跑一下”，3 卡各约 10 条；单路线正好形成 30 个 full-CEM run。

### Explicit non-goals

- 本阶段不跑 adaptive/fingertip route；
- 本阶段不把 CEM 结果直接判 RL-ready；
- CEM 跑完前不做中途逐条评测，不用单个 early positive 停止；
- 不 kill 已有 RL/CEM 进程，按用户要求直接叠加运行。

## 候选来源

权威输入：

```text
workspace/core4d/results/E104/v2_candidates_3cm_with_fingertip.tsv
```

选择规则：

1. `object_key == box026`
2. `route in {candidate_legacy_risk_needs_visual, candidate_executable}`
3. 按 E104 `rank` 升序固定顺序；
4. 输出 E106 manifest 后冻结，不再随 E104 文件变化隐式改变。

计划输出：

```text
workspace/core4d/scripts/E106/candidates.tsv
workspace/core4d/scripts/E106/variants.tsv
workspace/core4d/results/E106/manifest_summary.md
```

## Phase 0：数据就绪与缺失 OmniRetarget

由于 30 条候选并非全部已有 OmniRetarget/SPIDER 输出，E106 Phase 0 必须先做 readiness：

| gate | required evidence |
|---|---|
| source scene | `box026_person1/2/scene.xml` exists and inertial audit clean |
| Holosoma original output | `.../stage2b_medium/results/holosoma_{task}/retargeted/*_original.npz` |
| Holosoma trimmed output | `.../stage2b_medium/results/holosoma_{task}/trimmed/*_original.npz` |
| SPIDER task | `example_datasets/.../{task}/0/trajectory_kinematic.npz` |
| clean CEM task | `example_datasets/.../{task}_e106_clean/scene_act.xml` |

New script:

```text
workspace/core4d/scripts/E106/build_e106_box026_manifest.py
```

Responsibilities:

- read E104 3cm candidate TSV；
- write frozen 30-row candidate manifest；
- write data-construction case file for rows missing SPIDER task:

```text
/home/ubuntu/Workspace/holosoma/workspace/v3/data_construction_v2/inputs/cases_e106_box026_30candidate_pipeline.tsv
```

- write readiness table:

```text
workspace/core4d/results/E106/data_readiness.tsv
```

If `retargeted/trimmed/SPIDER` is missing, run the existing data preprocess pipeline before CEM:

```bash
CASE_FILE_REL=../holosoma/workspace/v3/data_construction_v2/inputs/cases_e106_box026_30candidate_pipeline.tsv \
RESULT_ROOT_REL=../holosoma/workspace/v3/data_construction_v2/results/stage2b_medium/results \
REPLACE_WRIST_WITH_FINGERTIP=0 \
bash workspace/core4d/scripts/E091/run_stage2b_medium_boxes.sh
```

`REPLACE_WRIST_WITH_FINGERTIP=0` follows the E091/E104 medium-box Box026 no-fingertip route; E106 is screening clean case dynamics, not testing fingertip replacement.

## Phase 1：Clean derived tasks 与 overrides

New script:

```text
workspace/core4d/scripts/E106/build_e106_clean_tasks.py
```

For each ready source task:

- copy source task to `{source_task}_e106_clean`；
- patch E083-style leg/foot + upper-body object collision pairs；
- validate:
  - `scene.xml`: `nq=43,nv=41,nu=29`
  - `scene_act.xml`: `nq=42,nv=41,nu=35`
  - qpos shape is `(T,43)`
  - robot inertial is not the old `mass=29.632` pollution pattern
  - required leg/upper pairs exist
- write `examples/config/override/core4d_{variant}.yaml` using E105R-style ref-FK route；
- write `workspace/core4d/scripts/E106/variants.tsv` with 30 rows and fixed split assignment.

Split assignment:

| split | GPU | ranks |
|---|---|---|
| `local-gpu0` | local GPU0 | 1-10 |
| `remote-gpu0` | remote GPU0 | 11-20 |
| `remote-gpu1` | remote GPU1 | 21-30 |

## Phase 2：Pre-CEM 可视化 gate

Before each CEM run, generate MuJoCo replay/visual evidence for the clean derived task. This inherits the user requirement from E105 planning: CEM should not start from an uninspected scene/template.

New script:

```text
workspace/core4d/scripts/E106/render_pre_cem_replays.py
```

Outputs:

```text
workspace/core4d/results/E106/pre_cem_visual_gate.tsv
workspace/core4d/results/E106/pre_cem_visual_review/{variant}/REVIEW.md
```

Default run script behavior:

- require `REVIEW.md` to contain `PASS` or `PASS_WITH_NOTES` before starting a variant；
- allow override only with `E106_REQUIRE_PRE_CEM_GATE=0` for emergency debugging, and record that in progress/log.

## Phase 3：Batch CEM 执行，暂不评估

Training script:

```text
workspace/core4d/scripts/train/train_E106_box026_candidate_batch.sh
```

Remote script:

```text
workspace/core4d/scripts/run_E106_remote.sh
```

Behavior:

- local and remote use the same train script;
- each GPU slot reads its split from `variants.tsv`;
- same GPU runs variants serially;
- remote GPU0 and GPU1 run in parallel under tmux;
- no eval is run during CEM;
- output paths are isolated per variant:

```text
workspace/core4d/results/E106/cem/full/{variant}.npz
workspace/core4d/results/E106/cem/full/{variant}_full.mp4
workspace/core4d/results/E106/cem/full/{variant}_outdir_full/
logs/E106/cem/full/{variant}_full.log
```

Launch commands:

```bash
# Local GPU0, serial ranks 1-10
E106_SKIP_EVAL=1 bash workspace/core4d/scripts/train/train_E106_box026_candidate_batch.sh local full 0

# Remote 2 GPUs, serial ranks 11-30
bash workspace/core4d/scripts/run_E106_remote.sh full
```

Important execution rule:

- Do not kill existing RL/CEM processes. If GPU memory is enough, stack this batch on top as requested.

## Phase 4：拉回结果并统一评估

Pull script:

```text
workspace/core4d/scripts/pull_E106_remote_results.sh
```

Evaluation is explicitly deferred until all expected outputs exist. Then run:

```bash
bash workspace/core4d/scripts/train/train_E106_box026_candidate_batch.sh eval full 0
```

Evaluation must include:

- E105 upper-body/replay gate metrics；
- E026/E081 lower-body strict proxy；
- per-case object/contact/posture/safety table；
- final CEM video/keyframe visual package；
- no RL-ready label unless lower-body strict passes.

Expected eval outputs:

```text
workspace/core4d/results/E106/cem/full/full_eval_summary.csv
workspace/core4d/results/E106/cem/full/full_eval_summary.md
workspace/core4d/results/E106/visuals/box026_30candidate_batch/REVIEW.md
```

## 验证声明与成功标准

| claim | success criterion |
|---|---|
| C1 candidate freeze | exactly 30 Box026 E104 3cm rows frozen in `candidates.tsv` |
| C2 data readiness | every CEM-launched variant has retargeted/trimmed/SPIDER/clean-task evidence |
| C3 clean-scene safety | no launched variant uses old `_e092_*` or polluted scene |
| C4 3-card batch script | local-gpu0 / remote-gpu0 / remote-gpu1 each has 10 serial variants |
| C5 no eval during run | run scripts do not call eval after each variant |
| C6 completion gate | before final eval, all 30 root NPZ + MP4 outputs exist or failures are explicitly recorded |
| C7 final eval quality | final summary includes lower-body strict proxy; candidate is not RL-ready unless that proxy passes |

## 风险表

| risk | mitigation |
|---|---|
| many candidates lack OmniRetarget output | Phase 0 writes pipeline case file and blocks CEM until SPIDER tasks exist |
| OmniRetarget CVXPY infeasible on some rows | keep failure in readiness/result table; do not silently drop row |
| remote does not have generated data | sync scripts/data explicitly before `run_E106_remote.sh`; use pull script after completion |
| pre-CEM visual review not scalable | generate automatic visual artifacts for all rows; only allow CEM after `PASS/PASS_WITH_NOTES` review file |
| lower-body shortcut repeats E105 | final eval must include E026/E081 leg-box proxy before any RL-ready conclusion |

## 产物

- `workspace/core4d/scripts/E106/build_e106_box026_manifest.py`
- `workspace/core4d/scripts/E106/build_e106_clean_tasks.py`
- `workspace/core4d/scripts/E106/render_pre_cem_replays.py`
- `workspace/core4d/scripts/E106/candidates.tsv`
- `workspace/core4d/scripts/E106/variants.tsv`
- `workspace/core4d/scripts/train/train_E106_box026_candidate_batch.sh`
- `workspace/core4d/scripts/run_E106_remote.sh`
- `workspace/core4d/scripts/pull_E106_remote_results.sh`
- `workspace/core4d/scripts/eval/eval_E106_box026_candidate_batch.py`
- `workspace/core4d/log/133_E106_box026_30candidate_ref_fk_batch_results.md` after run/eval
