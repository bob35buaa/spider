# R304 (E216) 实验计划：下游对应臂的 trans-only **0.1m** 物体增强放量

> Phase 74 · Run **R304** · 分支 `experiment/E199-omniretarget-object-augmentation`（同 augmentation 方向，不新建分支）
> 承接 [plan228(E199)](228_E199_omniretarget_object_augmentation_full_cem_plan.md) / [plan232(E202)](232_E202_bucket_e178_translation_augmentation_plan.md) / [plan243(E213)](243_E213_paired_export_selected_arm_aug_plan.md) / [plan247(E215)](247_E215_bucket_box_rot_augmentation_plan.md)
> 状态：**计划态，待用户批准**

## Context

此前的 trans 物体增强（E199 box / E202 bucket / E213 chair-desk）全部用原生 **0.2m** approach 位移。下游 RL（`exp/R018`）在采用这些增强数据时有一层预筛选（14-gate 漏斗 + Refiner/Tracker 双门），把上游合格的 case 进一步裁掉——例如 E199/E200/E202/E213 导出的候选池（CEM 104 / Omni 116）最终只正式选入 66。

下游 provenance（`tmp/R018-51_onward_augmented_test_upstream_provenance.md`）已确认：**每个物体采用的重定向臂是固定的、逐物体不同的**，且与我们上游的 arm 分派（`e215_common.py::ARM_GROUP_BY_OBJECT` + E213 `e213_source_arm_cem.tsv`）逐条吻合。

### 关键 insight

**更小的 approach 位移（0.2m → 0.1m）扰动更温和**，物体更不容易离开可达域、腿-物净空更宽、姿态更贴近 orig → 预期**上游可行率更高、14-gate 通过率更高、下游预筛后留存更多**。本实验在**每个物体各自正确的重定向臂**下，用 **0.1m、trans-only、前/左/右 3 方向** 重做一遍增强放量，case 取**上游该臂的全量**（理论上比下游预筛后的集合大），为下游提供一批"更易通过预筛"的增强数据，并量化 0.1m vs 0.2m 的质量/通过率差异。

### 与下游的口径对齐（已核实，非假设）

- **增强烘焙在上游 retarget 轨迹里，与下游 CEM arm 完全解耦**（E199/E200 的同一批轨迹挂不同臂跑 CEM 即为证）。故 0.1m 只需改上游 retarget 位移，下游各臂 scene 直接复用。
- 每物体臂来源：box021=PRG(E199) / box023=noPRG(E200) / box001·004·024=PRG+G1A2(E200) / bucket003·007=PRG(E202) / chair006·desk021=G1(E209) / desk007=G0.8·0.4·0.6 per-case(E211) / desk023=G0.6(E212)。

## Claims

| Claim | 最低证据 |
|-------|---------|
| **C1 可行性不劣于 0.2m** | 0.1m trans 上游 omnirt_v2 可行率 ≥ 对应臂 0.2m 的可行率（期望 ≥95%，趋近 100%，因扰动更小）；逐物体报告可行/总数 |
| **C2 增强正确（approach 幅度）** | 每可行变体 approach 段位移峰值 = 0.100m ± 0.01；endpoint 残差 ≤ 0.03m（指数衰减锚定仍成立） |
| **C3 数值不劣于同臂 orig（C4 门）** | 每 case obj_pos Huber-loss ≤ **+5cm** vs **同臂 orig** 基线；fall 不新增；leg/hand 穿透不劣于 orig；报 mean+std+worst |
| **C4 相对 0.2m 的通过率不降（核心动机）** | 同物体同臂下，0.1m 的 14-gate 宽/窄通过率 **≥** 0.2m（目标更高）；逐 case 配对比较（同臂同打分） |
| **C5 视觉无 artifact** | osmesa 代表案例 sim-vs-aug-ref 同视频：精确跟踪、无穿模/漂浮/jitter；fall 与 fall_flag 一致 |

## 已核实的 case 盘点（124 base case × 3 trans = 372 变体）

| 物体 | 重定向臂 | CEM scene（复用） | orig 基线来源（复用，零重跑） | base case 数 |
|------|----------|-------------------|-------------------------------|--------------|
| box021 | **PRG / A0** | `scene_act_E199_rubberHull_PRG` | E198 A0 arm | 28 |
| box023 | **noPRG (E167A)** | base scene（无碰撞对） | E190/E200 noPRG orig | 16 |
| box001 | **PRG+G1A2** | `scene_act_E199_rubberHull_PRG_gravcomp` + A2 gate | E198 G1A2 arm | 24 |
| box004 | **PRG+G1A2** | 同上 | E198 G1A2 arm | 6 |
| box024 | **PRG+G1A2** | 同上 | E198 G1A2 arm | 9 |
| bucket003 | **PRG** | `scene_act_E202_bucketAlignedTop_PRG` | E202 orig（E178 full CEM） | 9 |
| bucket007 | **PRG**（E202，非 E210 gravcomp） | `scene_act_E202_bucketAlignedTop_PRG` | E202 orig | 12 |
| chair006 | **G1**（gravcomp 1.0，E209） | E209 gravcomp sidecar | E213 选定臂 orig | 5 |
| desk007 | **G0.8×2 / G0.4 / G0.6** per-case（E211） | E211 gravcomp sidecar | E213 选定臂 orig | 4 |
| desk021 | **G1**（gravcomp 1.0，E209） | E209 gravcomp sidecar | E213 选定臂 orig | 7 |
| desk023 | **G0.6**（gravcomp 0.6，E212） | E212 gravcomp sidecar | E213 选定臂 orig | 4 |
| **合计** | | | | **124（box 83 / bucket 21 / chair-desk 20）** |

> **chair/desk 只取 E213 已知选定臂的 20 个 base case**（用户决定）：臂来自 `results/E213/s6_downstream/manifests/e213_source_arm_cem.tsv`（chair006 5 / desk007 4 / desk021 7 / desk023 4）。desk007 的 4 个 case 臂各异（028_p1/p2=G0.8、030_p2=G0.4、032_p2=G0.6），须逐 case dispatch。
> box 全量 83 case = E199 fullscale authority（`e199_fullscale_authority.tsv`，box001 24 / box004 6 / box021 28 / box023 16 / box024 9）；bucket 21 = E202 authority 中 bucket003 9 + bucket007 12（**不含 bucket004**，下游未用）。

## 改动

### 1.【核心】上游 trans 位移 0.2m → 0.1m（参数化 + git 追踪）

**文件**: `holosoma/.../holosoma_retargeting/examples/parallel_robot_retarget.py:117-119`

```python
# generate_augmentation_configs() 内，object_interaction 分支：
off = float(os.environ.get("OMNIRT_TRANS_OFFSET_M", "0.2"))   # E216 传 0.1
trans_configs = [[off,0,0], [0,off,0], [0,-off,0]]            # fwd / left / right
# rot 分支不触发（E216 trans-only，VARIANTS 不含 rot）
```

设计：env 参数化（默认 0.2 保持历史行为不变），E216 pipeline 传 `OMNIRT_TRANS_OFFSET_M=0.1`。改动隔离、可逆、落在 holosoma 分支由 git 追踪（符合 experiment.md §3）。

### 2. C2/C3 校验阈值随位移缩放

**文件**: `scripts/experiments/E216/e216_common.py`（新建；expected approach offset 0.2→**0.1**，endpoint 残差阈值不变）。仅影响断言，不影响生成。

### 3. 新建 `scripts/experiments/E216/`（复用 E199/E202/E213/E215 机制）

| 文件 | 职责 |
|------|------|
| `e216_common.py` | 124-case registry + 逐物体 `ARM_GROUP_BY_OBJECT`（复用 e215）+ chair/desk 从 `e213_source_arm_cem.tsv` 读 per-case 选定臂；`OMNIRT_TRANS_OFFSET_M=0.1`；trans-only VARIANTS；CEM 常量（1024×32 seed0，import e173/e215） |
| `build_augmented_tasks.py` | 上游 omnirt_v2 @0.1m retarget（warm-start `_original` IK，省 ~60%）→ trim → SPIDER task → 逐物体臂 CEM scene（复用既有 scene builder / gravcomp sidecar，按变体重生 scene_act）；C2/C3 pose-diff |
| `build_manifest.py` | per-variant base yaml + arm override + 分片优先级 manifest（蓝本 e215/e200） |
| `run_E216_cem.py` | 分片子集校验 + per-manifest flock + per-task 180min timeout（蓝本 e215 `run_e215_cem`） |
| `audit_single_variable.py` | gravcomp sidecar 单变量自检（仅 G1/G0.x 臂）+ scene sha pin |
| `test_e216_contract.py` | C0 契约：124 case≡registry、臂映射逐 case==权威表、trans-only、expected offset=0.1 |
| `snapshot_E216_scenes.sh` | rule 7 scene 快照（所有 case scene + manifest.txt git HEAD+sha256） |
| `render_c5.py` | osmesa sim-vs-aug-ref MP4 |

### 4. eval/launch wrapper

- `eval/runners/eval_E216_trans01.py`（import `eval.core.core_metrics`；E201 14-gate 宽/窄 + 穿透感知三元组 + clean-contact；C3 vs 同臂 orig、C4 vs 0.2m）
- `eval/wrappers/eval_E216_trans01.sh`
- `eval/reports/gen_E216_table.py`（0.1m vs 0.2m vs orig 三列对比表 + 逐物体可行率）
- `launch/active/run_E216_local_8gpu.sh` + `run_E216_remote.sh` + `pull_E216_remote_results.sh`

## 需要修改的文件

| # | 文件 | 改动 |
|---|------|------|
| 1 | `holosoma/.../examples/parallel_robot_retarget.py` | trans 位移参数化 0.2→env（E216=0.1）；holosoma 分支 git 追踪 |
| 2 | `scripts/experiments/E216/*`（8 文件，新建） | registry/build/manifest/cem/audit/contract/snapshot/render |
| 3 | `scripts/eval/{runners,wrappers,reports}/*E216*`（3 文件，新建） | 评估 + 对比表 |
| 4 | `scripts/launch/active/*E216*`（3 文件，新建） | 本机/远程 launch + pull |
| 5 | `results/E216/scene_snapshot/`（产物） | rule 7 快照 + manifest.txt |

**零 SPIDER 核心代码改动**（run_mjwp.py / sampling.py 不动）；所有臂 scene 复用既有文件，只换 0.1m 轨迹。

## Pipeline（6 阶段）

```
S0 冻结 registry（124 case + 逐物体臂 + chair/desk per-case 选定臂）   → verify: contract test ALL PASS
S1 上游 omnirt_v2 @0.1m retarget（warm-start _original）              → verify: 可行率 ≥95%，C2 approach=0.1m±0.01
S2 build SPIDER task + 逐物体臂 CEM scene（复用 scene/gravcomp）        → verify: scene sha pin，gravcomp 单变量 audit 过
S3 full CEM（1024×32 seed0，180min/task）                            → verify: cem_ok 100%，0 error/timeout
S4 eval：C3 vs 同臂 orig、C4 vs 0.2m（14-gate 宽/窄 + 穿透三元组）     → verify: C3 逐变体 ≥80%，C4 ≥ 0.2m
S5 visual C5：osmesa 代表案例 sim-vs-aug-ref                          → verify: 实际观察填入 log，无 artifact
```

> 快照（rule 7）在 S3 训练前执行：`bash scripts/convert/snapshot_scenes.sh E216 <124 case...>`（或 E216 专用 snapshot 脚本）。
> **RL-export（partner 配对 + Holosoma re-anchor）为可选后续 E216-export**，蓝本 E202-export/E213-export/E215，本计划不含（先交付 source 侧 CEM + 质量对比）。

## 总预算

| 项 | 规模 | 估算 |
|----|------|------|
| 上游 retarget（omnirt_v2 @0.1m） | 124 case × 3 trans = 372 pass，warm-start `_original` IK（省 ~60%） | 主要 CPU/IK，约数 GPU-h |
| **Full CEM** | **≤372 可行变体**（0.1m<0.2m，预期可行 ≥95%）× 1024×32 seed0，中位 ~47min/run | **~290 GPU-h** |
| eval + 可视化 | 372 打分 + osmesa 抽样渲染 | 数 GPU-h |
| **墙钟（并行）** | 8 卡 ≈ **37h**；24 卡（3 机，E215 模式）≈ **12–14h** | — |
| **合计** | **~372 CEM runs ≈ 290–300 GPU-hours** | — |

> 成本相对 0.2m 版本基本等价（同 case 数、同 CEM 预算）；增量仅上游一次 0.1m retarget（warm-start 后很轻）。CEM 与评估全部复用既有栈。

## 成功标准

| 指标 | 基线 | 本次目标（R304 / 0.1m） |
|------|------|--------------------------|
| 上游可行率 | 0.2m：box 100% / bucket 90% / chair-desk 视 case | **≥95%（逐物体 ≥ 对应臂 0.2m）** |
| CEM 完成 | — | **cem_ok 100%，0 error/timeout** |
| C3 obj_pos HL vs 同臂 orig | ≤ +5cm | **逐变体 ≤ +5cm，case 级通过 ≥80%** |
| C4 14-gate 窄通过率 vs 0.2m | 0.2m 同臂数值 | **≥ 0.2m（期望更高）** |
| fall / leg·hand 穿透 | 同臂 orig | **不新增 fall，不劣于 orig** |
| C5 视觉门 | — | **代表案例无穿模/漂浮/jitter** |

## 待确认 / 风险

1. **0.1m 可能进入"增强太弱"区间**：若 0.1m 的 obj yaw/pos 多样性对下游帮助甚微，数据价值下降——由 C4 对比量化（若 0.1m 与 orig 几乎无差异需在 log 诚实指出）。
2. **desk007 per-case 臂**（G0.8/G0.4/G0.6 各异）：build 时须逐 case dispatch，contract test 覆盖。
3. **bucket007 用 E202 PRG**（非 E215 的 E210 gravcomp）：与下游 trans 来源一致（用户决定）。
4. **holosoma 改动须在其仓库分支提交**并在本 log「改动文件」记录 commit，保证 0.1m 可复现。
