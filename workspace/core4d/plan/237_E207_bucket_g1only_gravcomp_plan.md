# plan237 · E207：bucket G1-only（object gravcomp 单变量）

_Core4D · Phase 67 · Run **R293** · 承接 [E178](../log/239_E178_local_5090_hybrid_rebalance.md) / [E194](../log/268_E194_object_gravity_compensation_results.md) / [plan234·E204E205](../plan/234_E204_E205_bucket_noprg_g1a2_arm_ablation_plan.md) · 2026-09-04 · 分支 `feat/E207-bucket-g1only-gravcomp` · **状态：计划已批准，执行中**_

---

## Context

本次 E178 full CEM 的 object z 诊断（新产出 `E178_object_z_diff_report.md`）发现一个**单向系统性偏差**，不是随机跟踪噪声：

- 27 case 中 **24/27 物体平均低于参考轨迹**，z bias 宏平均 **−1.820 cm**
- 欠抬升幅度与参考抬升幅度强相关：`pearson r(ref_z_range, z_bias) = −0.870`，斜率 −14.8 cm/m

**根因已在 E194 定位并有现成解**：CORE4D 的物体不是自由体，而是被 6 个 slide/hinge 关节 + `<position>` 执行器（`kp=500`）钉在参考轨迹上。该控制是 **P-only、无重力前馈**，稳态下垂 `sag = m·g/kp`。E194 的 **G1 arm = object body 加 `gravcomp="1"`**（`E194/e194_common.py::ARMS`），在质心施加 `+m·g` 抵消该偏置。这是唯一改动，不涉及 reward 权重 / CEM 配置 / 接触约束。

**为什么还要跑**：E205 已在 27 bucket case 上跑过 **G1A2**，但它相对 E178 同时改了**两处**（gravcomp + A2 hand-gate 收紧）。`gravcomp` 单独的贡献从未被隔离。E207 补上 **A0 + gravcomp** 这一格。

**目标**：在 9 个 case 上以严格单变量验证 gravcomp 是否消除 z 欠抬升，且不劣化 E201 14-gate。

---

## 已锁定的口径（用户 2026-09-04 确认 4 项）

| 项 | 决定 |
|---|---|
| 算法 | **E178 + 仅 G1**（保留 A0 hand-gate），相对 E178 严格单变量 |
| 主判定 | **沿用 E201 14-gate 漏斗**（与 E204/E205 同尺子）；z 指标作诊断项附带报告 |
| gravcomp 物理落差 | **接受**，视为纯重定向阶段的建模手段，不额外做真实重力 replay |
| GPU | 用户手动释放本机 8 卡后再跑 |

---

## Plan 阶段已实测的关键事实（非假设）

| # | 事实 | 证据 |
|---|---|---|
| **M1** | 9 个 case 全部存在于 E178 evaluated manifest，且全部属于 RL 导出 13 子集；分布 **bucket003=3 / bucket007=6 / bucket004=0** | 逐 case 比对 `evaluated_manifest_snapshot.tsv` |
| **M2** | `scene_act_E205_contactAlignedTop_gravcomp.xml` **恰好等于** `scene_act_E178_contactAlignedTop.xml` + object `gravcomp=1`，**9/9 通过** `e200_common.assert_gravcomp_diff()` | 计划期实跑该断言 |
| **M3** | E205 override 相对 E178 加 **4 个 key**：`scene_name`(G1) + 3 个 A2 hand-gate 字段（`max_violation_pct` 0.10→0.05、`hard_floor_m` −0.020→−0.015、`min_sdf_m` 不变）。→ **E207 只需 1 个 key** | 读 `core4d_E205_*_G1A2.yaml` + `E198/e198_common.py::A2_GATE` |
| **M4** | E205 G1A2 在这 9 case 上**已消除系统性欠抬升**：z bias **−1.379 → +0.622 cm**（+2.0 cm 上移）；但 z MAE 仅 2.563 → 2.198，且只有 5/9 改善 | 用现有 E205 npz 实算 |
| **M5** | 9 个 case 的 `scene_act_E178/E205` XML **均未被 git 追踪**（仅 E174/E188 的被追踪）——违反 rules §7 保障 1，E207 需补 | `git ls-files` |
| **M6** | 本机 8×L20Y，当前被一外部 8 卡任务占满（7.5h，每卡 37–56 GB / 81 GB）；E178 的远程 A100+SSH 链路本机不可用 | `nvidia-smi` / 读 `run_E178_remote_a100.sh` |

**M4 的读法（重要）**：gravcomp 修的是**下沉**（bias），不是**总跟踪误差**（MAE）。E207 的成功判据必须据此设定，不能要求 MAE 大幅下降。

---

## Arm 精确定义

| Arm | scene_act | leg PRG | hand-gate | 数据来源 |
|---|---|---|---|---|
| **PRG** (E178) | `scene_act_E178_contactAlignedTop` | penalty 2.0 + gate on | A0 (E163 默认) | 已有，复用 |
| **noPRG** (E204) | `scene_act_E204_contactAlignedTop_noPRG` | 关 | A0 | 已有，复用 |
| **G1A2** (E205) | `scene_act_E205_contactAlignedTop_gravcomp` | penalty 2.0 + gate on | **A2 收紧** | 已有，复用 |
| **G1only** (E207) | `scene_act_E205_contactAlignedTop_gravcomp` ← **复用（M2 已证等价）** | penalty 2.0 + gate on | A0 | **本次新跑 9 条** |

**两条单变量路径**（这是 E207 的科学价值）：
- `E178 → E207`：只差 gravcomp → **隔离 G1 的效应**
- `E207 → E205`：只差 hand-gate → **隔离 A2 的效应**（在 gravcomp=on 条件下）

不做 2×2 第四格（A2 + 无 gravcomp），故**不检验交互项**——显式声明为非目标。

**scene 复用决策**：不新建 `scene_act_E207_*.xml`。复用 E205 的 sidecar，好处是 E207-vs-E205 的 scene 逐字节相同，hand-gate 对比也成为严格单变量；代价是命名跨实验，由 P1 的断言 + manifest 显式记录消解。

---

## 执行阶段

### P0 · 分支 + 计划落盘 + git 保障
- 从当前 HEAD 建 `feat/E207-bucket-g1only-gravcomp`（当前在 `experiment/E199-...`）
- 写 `workspace/core4d/plan/237_E207_bucket_g1only_gravcomp_plan.md`（本文件内容）
- **rules §7 保障 1**：对 9 个 case `git add -f` 其 `scene.xml` / `scene_act_E178_contactAlignedTop.xml` / `scene_act_E205_contactAlignedTop_gravcomp.xml` / `scene_act_meta.json` / `task_info.json`（M5 缺口）
- `EXPERIMENT_TRACKER.md` 顶部插入 `| R293 | 2026-09-XX | Phase 67 | E207 bucket G1-only gravcomp 单变量 | 📝 计划态，待批准 | [plan237](...) |`

**退出检查**：`git ls-files` 对 9 case 各返回 ≥5 个文件；分支已切换。

### P1 · 契约模块 + scene 断言
新建 `workspace/core4d/scripts/experiments/E207/e207_common.py`，**照抄 `E204_E205/e204e205_common.py` 结构**（rule 13：不复制指标实现，只复用）：
- `REPO = Path(__file__).resolve().parents[5]`
- `CASES`：9 个 case_id 硬编码为白名单，并断言全部 ∈ `load_sources()` 的 27 行
- 复用 `e204e205_common.load_sources()` / `task_of()`（**注意坑**：`bucket007_20231020_055_p1` 是 omnirt_v2，虽不在本 9 case 内，仍按 source row 推导 task 目录，不假设 variant）
- `SCENE = "scene_act_E205_contactAlignedTop_gravcomp"`（复用）
- 复用 `E200/e200_common.assert_gravcomp_diff` 对 9 case 逐一断言

**退出检查**：`python -c "import e207_common; e207_common.audit()"` → 9/9 PASS，任一失败即中止。

### P2 · override 生成 + Hydra compose 审计
新建 `E207/build_overrides.py`，为 9 case 写：
```yaml
# @package _global_
defaults:
- core4d_E178_<case>_contactAlignedTop
- _self_
scene_name: scene_act_E205_contactAlignedTop_gravcomp
```
`--audit`：用 Hydra `compose()` 解析 E178 与 E207 两侧，逐 key diff，**只允许 `scene_name` 一个 key 不同**（照抄 `build_semantic_bucket_production.py:140-182` 的 allow-list diff）。额外断言 `cem_hand_gate_*` 三字段 == `E163_HAND_GATE`（证明未误带 A2）。

**退出检查**：9/9 override 写出；audit 报告 `diff_keys == {"scene_name"}`，否则 `non_axis_drift` 抛错。

### P3 · manifest
新建 `E207/build_manifest.py` → `results/E207/s6_downstream/manifests/g1only_full_manifest.tsv`（9 行）。
列沿用 E199 priority-queue 契约：`case_id, object_key, target_task, override_id/path/sha256, scene_act, scene_name, effective_scene_sha256, trajectory(+sha), contact_mask(+sha), cem_samples=1024, cem_opt_steps=32, cem_seed=0, execution_mode=production, result_npz, outdir_npz, config_act, video, log, status, failure_mode, gpu_id, updated_at, tier`。
CEM 预算与 E178/E204/E205 **完全一致**（1024×32, seed 0）。

**退出检查**：9 行；`trajectory_sha256` / `effective_scene_sha256` 与 E178 manifest 对应值逐 case 相等（证明参考与场景未漂移）。

### P4 · 场景快照（rules §7 保障 2）
```bash
bash workspace/core4d/scripts/convert/snapshot_scenes.sh E207 \
  dcv3_omnirt_v1_ref_fk_bucket003_20231018_001_p2 ... (9 个 task 目录名)
```
产出 `results/E207/scene_snapshot/` + `manifest.txt`（git HEAD + 每文件 sha256）。

**退出检查**：`manifest.txt` 含 9 个 case，且其中 `scene_act_E205_contactAlignedTop_gravcomp.xml` 的 sha256 == manifest 的 `effective_scene_sha256`。

### P5 · smoke（1 case × 64×4）
```bash
STAGE=smoke LIMIT=1 NUM_SAMPLES=64 MAX_ITERS=4 GPUS=0 \
  bash workspace/core4d/scripts/launch/active/run_E207_local_8gpu.sh
```
落到 `cem/smoke/` 不污染 full。回读 `config_act.yaml` 校验 `scene_name` 后缀 == gravcomp、`leg_object_penalty_scale == 2.0`、`cem_leg_gate_enabled == true`、`cem_hand_gate_max_violation_pct == 0.10`（A0！）、`init_pos_actuator_gain == 500`。

**退出检查**：1/1 产出 npz + 上述 5 项运行时校验全过。**这是唯一能证明"跑的确实是 G1only 而非 G1A2"的关卡，不可跳过。**

### P6 · full 9 条 × 8 卡
`scripts/launch/active/run_E207_local_8gpu.sh`（照抄 `run_E202_local_8gpu.sh`，30 行 thin wrapper）
→ 驱动 `E199/run_local_priority_queue.py`（按 `nvidia-smi` free-mem 派发、`--max-per-gpu 1`、**绝不 kill 他人进程**、状态原子写回 manifest、resume-safe）。

环境（本机必须，与 E178 远程不同）：`MUJOCO_GL=disable`、`TORCHDYNAMO_DISABLE=1`、`+use_torch_compile=false`、`save_video=false`、BLAS 单线程。**不要跑 `uv sync`**（共享 venv）。

9 case / 8 卡 ≈ 2 波。

**退出检查**：9/9 `run_complete_pending_eval`，0 fail/diverge。

### P7 · 评测（主：14-gate；辅：z 诊断）
1. **14-gate**：新建 `scripts/eval/runners/eval_E207_g1only.py`，**复用** `eval_E204E205_arm_ablation.py` 的 14-gate 实现，把 E207 作为第 4 个 arm 并入；输出 4-arm × 9-case 表。E178/E204/E205 三臂直接子集化已有结果，**同尺子、零重跑**。
2. **z 诊断**：复用本次新写的 `gen_E178_object_z_diff_report.py`（已支持 `--cases-from` / `--out-prefix` / `--compare-to`），对 E207 出一份 z 报告并与 E178/E205 并列。
3. 工作簿：`gen_E207_four_arm_workbook.py`（照抄 `gen_E204E205_three_arm_workbook.py`）。

**退出检查**：4 arm × 9 case 表齐全；E178 侧重算值与冻结 `e178_case_metrics.tsv` 一致。

### P8 · 视觉复核（rule 9 强制，不可留空）
- 渲染 E207 9 条视频
- viser 并排回放 **E178 vs E207**（复用 `E204_E205/viser_replay_arms.py`，已支持 arm 下拉）
- 用 `/video-frames` 对 **z MAE 最差 3 case + bias 改善最大 3 case** 抽关键帧（抓取 / 抬升峰值 / 放置）
- 重点看：gravcomp 后手是否变成"虚扶"（物体失重被托着走但手没真正夹紧）——这是 E194 C4 的核心风险，E204/E205 观察到接触 −0.013

**退出检查**：log 的「实际观察」写出具体描述（禁止"待补充"）。

---

## Claims 与量化成功标准

| ID | Claim | 数值门 |
|---|---|---|
| **C0** | 数据复用无漂移 | 9 case 的 `trajectory_sha256` 与 E178 逐一相等；0 条新 retarget |
| **C1** | 单变量合同 | Hydra compose diff == `{scene_name}`；`cem_hand_gate_*` == E163 默认；scene 9/9 过 `assert_gravcomp_diff` |
| **C2** | 执行闭合 | 9/9 cem_ok，0 fail/diverge |
| **C3** | **主判定 · 14-gate 不劣化** | E207 narrow_all ≥ E178 narrow_all − 1 case（9 case 尺度） |
| **C4** | **z 欠抬升被消除** | \|z bias 宏平均\| ≤ 0.8 cm（E178 基线 −1.379）**且** bias 绝对值下降的 case ≥ 7/9 |
| **C5** | 承重接触未塌 | `hand_object_physics_contact_3mm_in_mask_frac` 相对 E178 下降 ≤ 0.05（E204/E205 实测 −0.013） |
| **C6** | 视觉无新增失效 | P8 完成；无"虚扶"、无新增穿透/抖动 |
| **C7** | A2 效应被隔离 | 给出 E207→E205 的 paired delta 表（14-gate 逐门 + z + contact） |

**总判定**：
- **SUCCESS** = C0–C3 全过 **且** C4 达标 **且** C5、C6 无红线
- **PARTIAL SUCCESS** = C0–C3 过，C4 方向正确但未达门（如 \|bias\| 落在 0.8–1.2 cm）
- **FAIL** = C1 破（非单变量）或 C2 < 9/9 或 C3 劣化 ≥2 case

**显式不作为判据**：
- z **MAE** 的下降幅度（M4 已证 gravcomp 主要修 bias 不修 MAE，拿 MAE 判定会误杀）
- 2×2 交互项（第四格未跑）
- 真实重力下的可执行性（用户已确认接受该建模落差）

---

## 风险 / 缓解

| # | 风险 | 缓解 | 触发点 |
|---|---|---|---|
| 1 | 误跑成 G1A2（带了 A2 gate） | P5 回读 `config_act.yaml` 断言 `max_violation_pct == 0.10` | P5 |
| 2 | gravcomp 致手"虚扶"、接触塌 | C5 数值门 + P8 视觉专项看抓握 | P7/P8 |
| 3 | 8 卡未真正释放 / 被他人重新占用 | 用 E199 priority-queue（按 free-mem 派发，不抢占）；resume-safe 可中断续跑 | P6 |
| 4 | 9 case 无 bucket004（抬升最高、欠抬升最重的物体缺席） | 显式声明结论**不外推到 bucket004**；如需覆盖另开子实验 | 报告 |
| 5 | n=9 统计功效弱 | 用 paired 逐 case delta + 符号检验，不只报均值；报 mean+std+worst | P7 |
| 6 | 复用 E205 scene 造成命名混淆 | manifest 与 log 显式记录 `scene_name` 归属 + M2 断言 | P1 |

---

## 改动文件

**新建**
| 路径 | 作用 |
|---|---|
| `workspace/core4d/plan/237_E207_bucket_g1only_gravcomp_plan.md` | 本计划 |
| `workspace/core4d/scripts/experiments/E207/e207_common.py` | 契约 + 9 case 白名单 + scene 断言 |
| `workspace/core4d/scripts/experiments/E207/build_overrides.py` | 9 个 override + compose 审计 |
| `workspace/core4d/scripts/experiments/E207/build_manifest.py` | 9 行 manifest |
| `workspace/core4d/scripts/launch/active/run_E207_local_8gpu.sh` | 8 卡入口（thin wrapper） |
| `workspace/core4d/scripts/eval/runners/eval_E207_g1only.py` + `wrappers/eval_E207_g1only.sh` | 14-gate 第 4 arm |
| `workspace/core4d/scripts/eval/reports/gen_E207_four_arm_workbook.py` | 4-arm 工作簿 |
| `examples/config/override/core4d_E207_<case>_contactAlignedTop_G1only.yaml` × 9 | override |
| `workspace/core4d/results/E207/scene_snapshot/` | rules §7 保障 2 |
| `workspace/core4d/log/296_E207_bucket_g1only_gravcomp.md` | 跑完后写 |

**修改**
| 路径 | 改动 |
|---|---|
| `workspace/core4d/EXPERIMENT_TRACKER.md` | 顶部加 R293 一行 |
| `workspace/core4d/progress.md` | 按 rule 3 每 2 步更新 |
| `example_datasets/.../<9 case>/*.xml, *.json` | **仅 `git add -f`**，不改内容（M5 缺口） |

**SPIDER core（`spider/`、`examples/run_mjwp.py`）零改动** — E207 只换 scene sidecar 与 override。

---

## 验证（端到端）

```bash
cd /mnt/ali-sh-1/usr/xiayibo/work_dir/embodied/spider
export PATH="$PWD/.venv/bin:$PATH"

# P1-P3 契约（全部 fail-fast）
.venv/bin/python workspace/core4d/scripts/experiments/E207/e207_common.py --audit
.venv/bin/python workspace/core4d/scripts/experiments/E207/build_overrides.py --audit
.venv/bin/python workspace/core4d/scripts/experiments/E207/build_manifest.py

# P4 快照
bash workspace/core4d/scripts/convert/snapshot_scenes.sh E207 <9 个 task 目录名>

# P5 smoke（含运行时 config_act 断言）
STAGE=smoke LIMIT=1 NUM_SAMPLES=64 MAX_ITERS=4 GPUS=0 \
  bash workspace/core4d/scripts/launch/active/run_E207_local_8gpu.sh

# P6 full（等用户释放 GPU 后）
GPUS=0,1,2,3,4,5,6,7 PER_GPU_MEM_MIB=5000 MAX_PER_GPU=1 \
  bash workspace/core4d/scripts/launch/active/run_E207_local_8gpu.sh

# P7 评测
bash workspace/core4d/scripts/eval/wrappers/eval_E207_g1only.sh full --require-all
.venv/bin/python workspace/core4d/scripts/eval/reports/gen_E178_object_z_diff_report.py \
  --cases-from workspace/core4d/results/E207/s6_downstream/manifests/g1only_full_manifest.tsv \
  --out-dir workspace/core4d/results/E207/s6_downstream/eval/full \
  --out-prefix e207_object_z_diff --scope-title "E207 G1only：物体 z 高度与参考轨迹的差异" \
  --compare-to workspace/core4d/results/E178/s6_downstream/eval/full/e178_object_z_diff_by_case.tsv
.venv/bin/python workspace/core4d/scripts/eval/reports/gen_E207_four_arm_workbook.py

# P8 视觉
bash workspace/core4d/scripts/launch/active/run_E207_render_all.sh
.venv/bin/python workspace/core4d/scripts/experiments/E204_E205/viser_replay_arms.py --arms E178,E207
```

**Git**：Claims 全过后按阶段分次 commit，`exp(core4d): E207 P{N} — {一句话}`；最终 `exp(core4d): R293 E207 bucket G1-only — {结论}`。

---

## 下一步（本计划之外）

- 若 C4 成立：考虑把 gravcomp 纳入 bucket 的默认重定向配置，并回填 bucket004（本次缺席）
- 若 C5 出现接触塌陷：走 E194 C4 的备选（偏心施力 / partner 模型），而非继续加大 gravcomp
