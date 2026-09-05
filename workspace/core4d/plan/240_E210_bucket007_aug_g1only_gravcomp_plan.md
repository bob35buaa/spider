# plan240 · E210：E207 交付 6 例的 aug 变体 · PRG+G1（object gravcomp）

_Core4D · Phase 69 · Run **R296** · 承接 [E207/log296](../log/296_E207_bucket_g1only_gravcomp.md)（orig 侧 G1only + 6 例交付）/ [E202/log291](../log/291_E202_bucket_e178_translation_augmentation.md)（bucket aug 源侧）/ [E208/log297](../log/297_E208_deskchair_translation_augmentation.md)（F6 rot 修复、F7 有效位移）/ [E209/log298](../log/298_E209_desk_chair_prg_g1_gravcomp.md)（gravcomp 虚扶机制）· 2026-09-05 · 分支 `feat/E207-bucket-g1only-gravcomp` · **状态：计划态，待批准**_

---

## Context

E207 在 9 个 bucket case 上跑通 **PRG + G1（object `gravcomp=1`，A0 hand-gate）**，并把其中 6 个 bucket007 case 打包成 paired RL 导出集（`paired_rl_export_input.tsv`，6/6 `RL_EXPORT_READY`）。**但这 6 条全是 orig，数量不足以支撑下游 RL。**

E202 已经在 **E178 PRG arm（无 gravcomp）** 下为 bucket 类做过 OmniRetarget 平移增强并证明物理可信（73/81 可行、73/73 CEM 0 err、obj_pos −0.6%）。**其中恰好覆盖了本次 6 例里的 5 例，共 15 个 trans 变体，全部 `run_complete_pending_eval`。**

**E210 = 把 E207 的唯一干预（object gravcomp）加到这 15 个已建好的 aug 变体上，重跑 full CEM，用与 E207 同一把 14-gate 尺子评估。**

### 用户 2026-09-05 锁定的四项口径

| # | 决定 | 含义 |
|---|---|---|
| 1 | **复用 E202 aug 资产，只换 gravcomp scene** | 不重建 aug（不走 E208 的 5 变体/v1-first 路线）→ 15 条 trans 变体，omnirt_v2，E178 五段 proxy 原样 |
| 2 | **止于 full CEM + 14-gate + 视觉复核** | 不做 partner 增强、不做 Holosoma paired RL 导出 |
| 3 | **只做 trans，不做 rot**（2026-09-05 第二轮追加） | rot 档不构建、不评估。连带使 075_p2 的 rescue 失去唯一新变量 → **075_p2 直接排除** |
| 4 | **orig 配对基线 = E207 gravcomp orig（这 6 条）**，零新 CEM | E202 aug（无 gravcomp）作零成本诊断项，**不作判据** |
| 5 | **Claude 只负责 P0–P6（到 smoke 通过）+ 交付一条 8 卡全量命令**（第二轮追加） | P7 full CEM 由用户自行执行；P8/P9 待 full 跑完后另议 |

### 075_p2 排除的理由（第二轮口径变更的直接后果）

E202 对 `bucket007_20231023_075_p2` 三档的失败记录是 **`runtime_initial_overlap < 0`** —— 平移物体后**参考轨迹首帧**腿-桶几何重叠，被 PRG 运行时保护拦截。三点使它不可救：
1. 这是**参考层面的几何事实**，`gravcomp` 只改物体是否受重力，不改参考轨迹 → 拦截照旧；
2. 它**不是** E208 F6 那个 scipy 崩溃的受害者：变体顺序是 `original → trans_0/1/2 → rot_0/1`，崩溃发生在 `rot_0`，trans 三档在此之前已经跑完并各自独立判定；
3. 唯一的新变量本来是 rot 档（从未真正尝试过），**口径 3 已将其排除**。
→ 重跑 trans 只会原样复现同一拦截。**不做 rescue，直接排除，范围 5 case / 15 变体。**

---

## Plan 阶段已实测的关键事实（非假设，全部本机核对过）

| # | 事实 | 证据 |
|---|---|---|
| **M1** | 6 个交付 case 全是 **bucket007 / omnirt_v1 / ref_fk**，`paired_rl_export_input.tsv` sha256 = `9278cc0d…7c2c` | 读该 TSV |
| **M2** | 其中 **5 例在 E202 各有 3 个 trans 变体 = 15 条**，`e202_bucket_priority_manifest.tsv`（sha256 `b7b187ce…2a17`）中全为 `run_complete_pending_eval` | 逐 case 比对 |
| **M3** | **`bucket007_20231023_075_p2` 在 E202 下 3 档全不可行**，连 orig 行都没进 `e202_bucket_authority.tsv`（25/27 case）。同批被剔的还有 `bucket007_20231018_021_p2` | 逐 case 比对 E178(27) vs E202 authority(25) |
| **M4** | 15 个 aug task 目录实存，各含 `scene_act_E202_bucketAlignedTop_PRG.xml` + `scene_act_meta.json` + `task_info.json` | `ls example_datasets/.../dcv3_omnirt_v2_ref_fk_bucket007_*__aug_trans*/` |
| **M5** | `e200_common.assert_gravcomp_diff(base, sidecar)` 是通用的（只认 `object` body 的 `gravcomp` 0/absent→1），**可直接复用**；但同文件的 `build_gravcomp_sidecar()` **硬编码输出名** `scene_act_E199_rubberHull_PRG_gravcomp`，**不可复用**（E209 log F1 已登记该陷阱） | 读 `E200/e200_common.py:134-176` |
| **M6** | **本机 8 卡当前被 E208 P6 占用 7 张**（`run_mjwp.py +override=core4d_E208_*_aug_*_lowgeom_PRG`），E208 队列进度 **63 cem_ok / 8 running / 34 pending（共 105）**，按 ~45min/条 × 8 卡估算 **剩余 ≈3–4h** | `nvidia-smi` + `ps` + `e208_priority_manifest.tsv` |
| **M7** | 本机 **只有 `MUJOCO_GL=osmesa` 能渲染**（egl 抛 `EGLError`、glfw 无 `_mjr_context`），E207 F7 已登记 | log296 F7 |
| **M8** | E207 计划里写的 `eval/wrappers/eval_E207_g1only.sh` **实际从未创建**，评测脚本是直调的；workbook 实名 `gen_E207_three_arm_workbook.py`（非 four_arm） | `ls scripts/eval/{runners,wrappers,reports}/` |
| **M9** | `workspace/core4d/results` 是**指向外部盘的符号链接** → `scene_snapshot/` **进不了 git**（E199/E202/E206/E208 全受影响，log297 R11）。rule 7 保障 2 在本仓库只能以「manifest.txt 的 sha256 抄进 log」的形式满足；且 `Path.relative_to(REPO)` 在自定义 out-dir 下会抛 `ValueError`（E207 F8） | log297 R11 + log296 F8 |
| **M10** | SPIDER venv 里有个**残缺的 `torch` 命名空间 stub（无 `Tensor`）**，会经 array-api-compat 污染 `scipy.spatial.transform.Rotation`。任何用到 scipy 旋转的脚本必须先调 `_drop_broken_torch()`（`e202_common.py:118-130` / `e208_common.py:61-74`） | 读两处源码 |
| **M11** | `write_tsv` 把布尔序列化成**小写 `true/false`**，而 `gate_pass()` 产出 `str(bool)` 的 `True/False`。按字面量 `"True"` 计数会**全部得 0**（E208 F12-3 实际踩过）。必须用 `C.truth` | log297 F12 |

---

## 【前置风险，本计划的核心增量】E209 推翻了 E207 对 gravcomp 的两处读法

E209（log298）在 desk/chair 22 例上用**完全相同**的 gravcomp 干预，**主门 C3 被打破**。两条结论直接决定 E210 的判据设计：

1. **虚扶（phantom support）的检测器是 `eef_ori`，不是 `contact_in_mask`。**
   E209：narrow 13→10，退化几乎全在 `eef_ori`（narrow 失败 2→9，mean **+2.78°**）+ `root_ori` +1.81°，而 obj_pos/obj_ori/leg_pen **全部改善**、`contact_in_mask` 仅 **−0.016 完全无反应**。最坏例 obj_pos 反而更好（11.38→9.84），root_ori 9.7→**41.9°** —— 物体失重后 CEM 用姿态跑偏的机器人「托」着物体走。
   → **E207 的 C5（contact_in_mask 降幅 ≤0.05）作为虚扶门是失效的**，E207 拿它判「未塌陷」并通过，属于盯错指标。E210 **必须**把 `eef_ori` 升为一等判据。
   → E207 在 bucket 上已有同向信号：`eef_ori` 17.49→19.83°（**+2.34°**），且 9 例中唯一掉门的 `bucket007_20231023_073_p1` 就输在 eef_ori（19.83° vs 门 20°）。**该 case 在本次 6 例之内。**

2. **gravcomp 不是常量偏移，是收缩型修正。**
   E209 实测 `r(pre_bias, Δ) = −0.915`、OLS slope **0.408**（加性模型预测 0）、不动点 +3.50 cm。E207 得出「+1.945 cm 常量」是因其 9 例 pre-bias 只覆盖 `[−3.15, +0.24]` 且质量与 pre-bias 共线。
   → E210 **不预注册任何基于「常量偏移」的 z 数值门**；z 只作诊断项报告。

3. **额外风险（本实验特有）**：aug 把物体接近段平移 0.2 m，**改变了抓握几何**。虚扶在 aug 上可能比 orig 更容易触发（手离物体质心更远、力臂更长）。这正是 C5 要回答的。

### 并且：E208 F14 指出 aug 自身的代价也在「手」上

E208 在 desk/chair 上做同样的 aug（n=56/105 中途读数）：**pass-rate 掉 +0.179（门 0.15）✗，但 `obj_pos` 反而更好（HL +0.554 cm）✓**。16 个 orig-pass→aug-fail 的翻转里 **13 个是 `hand_penetration`**。且两个直觉解释都被证伪——与增强幅度无关（`r(Δhand_pen, 有效位移) = −0.029`）、也不是「orig 本来就压线」（最贴门的 desk007 翻转 0/14）。形态是**中位小（+0.042）、右尾重（p90 +0.190）**。

**叠加判断**：`hand_pen` 是 **aug 的主代价（E208 F14）**，同时也是 **gravcomp 的次代价（E207 实测 +0.035）**。E210 是两者首次叠加，`hand_pen` 很可能是本实验的主要掉门项，**必须与 `eef_ori` 并列进 C5**，不能只盯虚扶。

---

## Arm 精确定义与对照结构

| Arm | scene_act | retarget | 物体扰动 | 数据来源 |
|---|---|---|---|---|
| **A. orig × PRG** (E178) | `scene_act_E178_contactAlignedTop` | omnirt_v1 | 无 | 已有，复用（表格第四格） |
| **B. orig × PRG+G1** (E207) | `scene_act_E205_contactAlignedTop_gravcomp` | omnirt_v1 | 无 | 已有，复用 → **主基线** |
| **C. aug × PRG** (E202) | `scene_act_E202_bucketAlignedTop_PRG` | omnirt_v2 | trans0/1/2 | 已有，复用 → **零成本诊断** |
| **D. aug × PRG+G1** (E210) | `scene_act_E210_bucketAlignedTop_PRG_gravcomp` ← **新建** | omnirt_v2 | trans0/1/2 | **本次新跑 15 条** |

两条可用路径：

- **C → D**：只差 `gravcomp` → **严格单变量**，隔离「gravcomp 在 aug 变体上的效应」。**零额外算力**（C 已跑完）。这是 E210 最干净的科学产出。
- **B → D**：主判定路径，但**同时差两件事**：物体扰动 **和** retarget 变体（v1 vs v2）。这是 E202 时代就接受的 confound（碰撞体两侧完全相同，唯一变量是 retarget 变体）。**必须在 log 中显式标注，结论只声称「aug 数据物理不劣、可用于扩数据」，不声称严格 ablation。**

2×2 四格齐全（A/B/C 全部已有，只缺 D），**交互项可算**——这是 E207（只有 2 格）做不到的。

---

## 范围

| 项 | 值 |
|---|---|
| case | 6 个 bucket007（来源 = E207 `paired_rl_export_input.tsv`，sha256 pin `9278cc0d…7c2c`） |
| 有 aug 的 case | **5 个 → 15 条 trans 变体**（trans0/1/2 前/左/右 0.2 m） |
| 排除的 case | `bucket007_20231023_075_p2`（理由见上；契约里以显式断言登记，非静默丢失） |
| 新 CEM | **15 条** |
| CEM 预算 | **1024 × 32, seed 0**，与 E178/E202/E207 逐字段一致 |
| 复用 CEM | arm A/B/C 共 6 + 6 + 15 = 27 条，**零重跑** |

**产能预期**：15 条 aug × G1 + 6 条 orig × G1 = **21 条 gravcomp 轨迹**，相对 E207 的 6 条 **×3.5**。

---

## 执行阶段

### P0 · 编号 + 计划落盘 + 分支
- 编号：**E210 / plan240 / log299 / R296 / Phase 69**（E203–E209 已占；plan238 是 E208 的历史欠账，不占用）
- **不新建分支**（rule 11：同 exp_name=core4d、同数据集、同 bucket 主题，属 E207 的放量延伸）→ 沿用 `feat/E207-bucket-g1only-gravcomp`
- `EXPERIMENT_TRACKER.md` 顶部插 `| R296 | 2026-09-05 | Phase 69 | E210 bucket007 aug × PRG+G1 | 📝 计划态，待批准 | [plan240] |`
- **git 保障 1（rules §7）**：对 15 个 aug task 目录 `git add -f` 其 `scene.xml` / `scene_act_E202_bucketAlignedTop_PRG.xml` / `scene_act_E210_*_gravcomp.xml` / `scene_act_meta.json` / `task_info.json`

**退出检查**：`git ls-files` 对 15 个 aug 目录各返回 ≥5 个文件。

### P1 · 契约模块 `E210/e210_common.py`
照抄 `E207/e207_common.py` 结构（rule 13：复用指标实现，不复制）：
- `CASES`：6 个 case_id 硬编码白名单 + 断言 `paired_rl_export_input.tsv` sha256 == `9278cc0d…7c2c` 且恰好 6 行
- `load_e202_variants()`：从 `e202_bucket_priority_manifest.tsv`（sha256 pin `b7b187ce…2a17`）取这 6 个 case 的 aug 行，断言**命中 15 行且全为 `run_complete_pending_eval`**
- **显式断言 `075_p2` 命中 0 行**（M3 是已知缺口，不是静默丢失）——若某天 E202 被回填，这条断言会主动报错提醒复核
- `SCENE = "scene_act_E210_bucketAlignedTop_PRG_gravcomp"`
- CEM 预算从 `e202_common` 取，**不硬编码**
- **坑规避（E208 F1）**：不从 `e199_common.OMNIRT_V2_ENV` 照抄 env（它缺 `REPLACE_WRIST_WITH_FINGERTIP`）；本阶段不做 retarget，故只在 P6b rescue 时需要，届时对 `run_stage2b_omnirt_v2_ref_fk.sh` 的 env 字面量逐键 diff
- **坑规避（E207 F9 / E202）**：模块与脚本名一律带 `e210_` 前缀，避免 `sys.path` 同名自我遮蔽

**退出检查**：`python E210/e210_common.py` → `15/15 located, 075_p2 absent (expected), 2 authority sha OK`；任一失败即中止。

### P2 · gravcomp sidecar 生成 + 逐字节断言
新建 `E210/build_gravcomp_sidecars.py`：
- **不调用** `e200_common.build_gravcomp_sidecar()`（M5：硬编码输出名，E209 F1 陷阱）
- 自写 writer：解析 `scene_act_E202_bucketAlignedTop_PRG.xml` → 唯一 `object` body 置 `gravcomp="1"` → `ET.indent(space="  ")` → 写 `scene_act_E210_bucketAlignedTop_PRG_gravcomp.xml`
- 写完立刻 **复用 `e200_common.assert_gravcomp_diff`** 反向验证；幂等（已存在则只重验不重写）
- 额外断言：object_collision geom 数 == **5**、contact pair == **90**（bucket007 的 E178 五段 proxy），与 E202 的 C1 geom-parity 同口径

**退出检查**：15/15 `assert_gravcomp_diff` PASS + 15/15 geom(5)/pair(90) PASS。**反向测试**：手工构造一个多改了别的属性的 sidecar，确认守卫会抛（照 E207 P1 做法）。

### P3 · override 生成 + Hydra compose 审计
新建 `E210/build_overrides.py`，为 15 个变体写（**链到 E202 的 arm override，照 E207 链到 E178 override 的做法**——已核实 `core4d_E202_bucket007_20231003_2_021_p1_aug_trans0_PRG.yaml` 是自足的 flat override，链得上）：
```yaml
# @package _global_
# Auto-generated by E210/build_overrides.py.
# E210 aug×G1only = E202 aug PRG arm + object gravcomp. Single variable: scene_name.
defaults:
- core4d_E202_<case>_aug_<trans>_PRG
- _self_
scene_name: scene_act_E210_bucketAlignedTop_PRG_gravcomp
```
`--audit`：用 Hydra `compose()` 解析 E202 与 E210 两侧逐 key diff，**只允许 `scene_name` 一个 key 不同**。额外断言：
- `cem_hand_gate_max_violation_pct == 0.10` / `hard_floor_m == -0.020` / `min_sdf_m == -0.01`（**A0，证明未误带 A2**。注意 E202 override 本身不写 hand-gate 键，它们由 base task yaml + config 默认给出，所以这条查的是 compose 后的**有效值**）
- `leg_object_penalty_scale == 2.0`、`cem_leg_gate_enabled == true`、`object_collision_sdf_mode == "union"`、`object_collision_sdf_batch_groups == true`

**退出检查**：15/15 写出；`diff_keys == {"scene_name"}`，否则 `non_axis_drift` 抛错。

### P4 · manifest
新建 `E210/build_manifest.py` → `results/E210/s6_downstream/manifests/aug_g1only_full_manifest.tsv`（15 行）。
列沿用 E199 priority-queue 契约（含 `variant` 列——渲染器需要，E207 F 已登记）。

**退出检查**：15 行；`trajectory_sha256` / `contact_mask_sha256` 与 E202 manifest 对应值**逐条相等**（证明参考与掩码零漂移）；`effective_scene_sha256` 等于新 sidecar 实际 sha 且**不等于** E202 的（照 E207 F2 的修正口径，不重犯「要求相等」的笔误）。

### P5 · 场景快照（rules §7 保障 2）
```bash
bash workspace/core4d/scripts/convert/snapshot_scenes.sh E210 \
  dcv3_omnirt_v2_ref_fk_bucket007_20231003_2_021_p1__aug_trans0 ... （15 个 task 目录名）
```
**退出检查**：`manifest.txt` 含 15 个 task + git HEAD；其中 `scene_act_E210_*_gravcomp.xml` 的 sha256 == manifest 的 `effective_scene_sha256`。

> **M9 说明**：`results/` 是符号链接到外部盘，**快照目录本身进不了 git**。rule 7 保障 2 在本仓库只能以「把 `manifest.txt` 的 sha256 抄进 log」的形式满足；真正入 git 的是 P0 里 `git add -f` 的那 15 份 sidecar（保障 1）。这一点 E199/E202/E206/E208 全都一样，log 里要写明而不是假装快照被版本管理了。

### P6 · smoke（1 条 × 64×4）
```bash
STAGE=smoke LIMIT=1 NUM_SAMPLES=64 MAX_ITERS=4 GPUS=<空闲卡> \
  bash workspace/core4d/scripts/launch/active/run_E210_local_8gpu.sh
```
落到 `cem/smoke/`。回读 `config_act.yaml` 断言 7 项：`scene_name` 后缀 == `_gravcomp`、`cem_hand_gate_max_violation_pct == 0.10`（**A0**）、`hard_floor == -0.020`、`leg_object_penalty_scale == 2.0`、`cem_leg_gate_enabled == true`、`init_pos_actuator_gain == 500`、`init_rot_actuator_gain == 50`。

**退出检查**：1/1 npz + 7 项全过。**这是唯一能证明「跑的是 aug×G1only 而非 aug×PRG 或 aug×G1A2」的关卡，不可跳过。**

### ~~P6b · 075_p2 rescue~~ —— **已取消**（口径 3 的后果）

理由见上文「075_p2 排除的理由」。**零新 retarget**，本实验全程不碰 upstream holosoma 流水线 —— 连带规避了 E208 §5 的并发 retarget 事故、F15 不可复现、`--force` 污染 warm start 三类风险。契约层以显式断言登记该 case 的缺席（P1）。

---

## 交付边界（第二轮口径）

| 阶段 | 责任方 |
|---|---|
| P0–P6（契约 → sidecar → override → manifest → 快照 → **smoke 通过**） | **Claude 本轮完成** |
| P7 full CEM（15 条） | **用户自行执行**，Claude 只交付一条可直接粘贴的 8 卡命令 |
| P8 评测 / P9 视觉 | full 跑完后另议 |

### P7 · full CEM（15 条）— **交付命令，不由 Claude 执行**
```bash
GPUS=<按 M6 实际空闲> PER_GPU_MEM_MIB=5000 MAX_PER_GPU=1 \
  bash workspace/core4d/scripts/launch/active/run_E210_local_8gpu.sh
```
→ 驱动 `E199/run_local_priority_queue.py`（按 free-mem 派发、resume-safe、**绝不 kill 他人进程**）。环境：`MUJOCO_GL=disable`、`TORCHDYNAMO_DISABLE=1`、`+use_torch_compile=false`、`save_video=false`、BLAS 单线程。**不要跑 `uv sync`**（共享 venv）。

> **⚠ GPU 排期（M6）**：E208 P6 正在占 7/8 卡，剩 34/105 待跑（≈3–4h）。priority queue 的 free-mem 判据（5000 MiB）在 81 GB 卡上**对 E208 占用的卡同样会放行**（E208 每卡仅用 ~2.6 GB），`--max-per-gpu 1` 只约束队列自己的派发数，**不感知他人进程** → 会与 E208 抢核、双方都变慢。这条已经在本分支上发生过两次（E208 log297 §5 登记的 GPU 与 retarget 进程碰撞）。
>
> **默认处置：等 E208 P6 跑完再启 P7**（P0–P6 + P6b 全部是 CPU 侧，可立即做完，等待窗口不浪费）。若要提前启动，需**显式指定 E208 未占用的卡号**并把 `MAX_PER_GPU=1`，不要用 `GPUS=0,1,...,7`。

15 条 / 8 卡 ≈ 2 波 ≈ **1.5–2h**（按 E207 实测 9 条 1h10m 外推）。

**退出检查**：15/15（+rescue）`run_complete_pending_eval`，0 fail/diverge。

### P8 · 评测
1. **主表 · E201 14-gate 四格对照**：新建 `scripts/eval/runners/eval_E210_aug_g1only.py`，**复用** `eval_E204E205_arm_ablation.py` 的 14-gate 实现 + `E201/funnel_config.py`（rule 13，禁止 importlib 动态加载）。输出：
   - `four_cell_rollout.tsv`：A(orig×PRG) / B(orig×G1) / C(aug×PRG) / D(aug×G1) × 逐 rollout
   - **paired delta 三张**：`C→D`（隔离 gravcomp on aug，严格单变量）、`B→D`（主判定）、`A→B` vs `C→D`（**交互项**：gravcomp 在 orig 与 aug 上的效应是否一致）
2. **虚扶专项（C5，本计划新增）**：`eef_ori` / `root_ori` / `hand_pen` 逐 case delta + 符号检验，与 `contact_in_mask` 并列，**显式检验 E209 的「contact 不是虚扶检测器」在 bucket aug 上是否复现**
3. **有效位移分层（C6，E208 F7）**：对 15 条算 approach 段有效位移，分档 `≥0.18 / 0.10–0.18 / 0.05–0.10 / <0.05`，作为 delta 的分层变量。E202 bucket 全量实测 `min 0.074 / <0.18 占 33%`，**本 15 条的分布此前从未报过**
4. **z 诊断（附带，非判据）**：复用 `gen_E178_object_z_diff_report.py --cases-from`，报 aug 侧 z bias/MAE，与 E207 orig 并列。**不设 z 数值门**（E209 已证 E207 的常量模型不可外推）
5. 工作簿：`gen_E210_four_cell_workbook.py`（照抄 `gen_E207_three_arm_workbook.py`）
6. review TSV：`E210/build_e210_aug_review_tsv.py`（**带 e210 前缀**，规避 E207 F9 同名遮蔽），注册进 `review_index.py` 为 `E210AUG`

**三条实现纪律（都是别人踩过的）**：
- **门槛必须与 E207 逐字相同**（E208 F11：同一批 case 曾算出三个互不相等的通过率，根因是 `lower_body` 门 0.20 vs 0.10 的阈值差异，不是打分差异）。E210 全部 4 格用 **同一份 `E201/funnel_config.py`**，并输出一张 `gate_criterion_vs_E207` 交叉表逐条枚举分歧（应为空）。
- **布尔计数用 `C.truth`，不要比字面量 `"True"`**（M11 / E208 F12-3：`write_tsv` 写小写 `true`，按 `"True"` 数会全得 0 且不报错）。
- **任何用到 scipy 旋转的地方先调 `_drop_broken_torch()`**（M10）；自定义 `--out-dir` 下用 `rel_label()` 而非 `relative_to(REPO)`（M9 / E207 F8）。

**退出检查**：4 格 × 15 变体表齐全；A/B/C 三格重算值与各自冻结 TSV（`e178_case_metrics.tsv` / `e207_arm_case_metrics.tsv` / E202 `e202_aug_case_metrics.tsv`）**逐位一致**（证 evaluator 无漂移，照 E209 做法）；`gate_criterion_vs_E207` 分歧表为空。

### P9 · 视觉复核（rule 9 强制，不可留空）

> **先读 E208 F13：orig-vs-aug 的 A/B 视频比较在本仓库是无效的。**
> 两个叠加原因：① `spider/viewers/__init__.py:262-291 _auto_video_camera` **逐帧**用 sim+ref 身体位置的并集 bbox 重算 lookat/radius，机位跟着被移动的物体走 → 跨视频比较绝对位姿/屏幕位置/大小**没有意义**；② aug 视频的 ref pane 本身就是**增强后的**参考。
> **推论**：「增强到底发生了没有」**只能数值确认**（P4 的 sha 链 + C6 的有效位移），不能靠眼睛。人眼只判**质量**：穿模 / 漂浮 / 抖动 / 姿态 / 接触。

因此 P9 分两路，**只做安全的那一路 A/B**：

- 渲染 15 条：`run_E210_render_all.sh` → `E168/render_a100_cem_videos.py`，**固定 `MUJOCO_GL=osmesa`**（M7）。参考 pane 用 **aug 轨迹**（照 E208 `render_aug_results.py` 的处置）
- **✅ 安全 A/B = C↔D（aug×PRG vs aug×G1）**：两侧**参考轨迹逐字节相同**，机位差异只来自 sim 身体位置，可作 A/B。这恰好也是本实验唯一的严格单变量轴
- **❌ 不做 B↔D（orig×G1 vs aug×G1）的画面比较**：参考不同 → 触发 F13。该对照只走数值
- **抽帧对象（预先指定，避免事后挑）**：
  - `eef_ori` 退化最大的 2 条（**虚扶主嫌**，E209）
  - `hand_pen` 增幅最大的 2 条（**aug 主代价**，E208 F14）
  - `eef_ori`/`hand_pen` 都几乎不动的 1 条（阴性对照）
  - 有效位移 <0.10 m 的 1 条（F7：**它在画面上应当与 orig 几乎一样**——这是 F13 之外唯一能靠眼看出的位移相关结论）
  - 若 rescue 成功，其 rot 档 1 条（**rot 变体在 bucket 上从未被任何人看过**）
- **重点看**：① 是否出现 E209 描述的虚扶形态——机器人姿态跑偏却仍「托」着物体走；② aug 平移后手-桶接触点是否外移到桶沿、`hand_pen` 的右尾在画面上长什么样（E208 F14 的机制假说是「代理面对的是物体原始位姿」，bucket 的五段 proxy 与 desk/chair 的手编 lowgeom 不同，**这是首次能在另一种代理上检验该假说**）
- review feed：`E210AUG`，照 E208 的形状——`case_id` = orig case、`arm` = 单元格（`ORIG_PRG / ORIG_G1 / TRANS{0,1,2}_PRG / TRANS{0,1,2}_G1`，8 臂 × 5 case），让 orig 与它自己的变体并排

**退出检查**：log 的「实际观察」写出具体描述（禁止「待补充」）；且**必须显式声明哪些结论是数值来源、哪些是画面来源**（F13 教训）。

---

## Claims 与量化成功标准

| ID | Claim | 数值门 |
|---|---|---|
| **C0** | 数据复用无漂移 | 15 条 `trajectory_sha256`/`contact_mask_sha256` 与 E202 逐条相等；**0 条新 retarget**（本实验不碰 upstream）；两份 authority sha256 命中 pin |
| **C1** | 单变量合同 | compose diff == `{scene_name}`；hand-gate == A0；sidecar 15/15 过 `assert_gravcomp_diff`；geom(5)/pair(90) 15/15；运行时 config_act 7 项 |
| **C2** | 执行闭合 | 15/15 cem_ok，0 fail/diverge/non-finite |
| **C3** | **主判定 · aug×G1 不劣于 orig×G1** | D 的 narrow_all 通过率 ≥ B 的通过率 − 1 个 case 等价量（B 为 6 例尺度、D 为 15 变体尺度 → 按**比率**比：`rate(D) ≥ rate(B) − 0.15`）；且 **fall = 0 / 15**、无 root|eef 发散（>60 cm） |
| **C4** | **gravcomp 在 aug 上的效应被隔离（C→D）** | 给出 15 条 paired delta（14 门逐门 + obj_pos/obj_ori/contact/eef_ori/z）+ 符号检验；**报交互项**：`(D−C)` 与 `(B−A)` 的方向是否一致。**本条只要求「出结论」不设通过门**——它是发现型产出 |
| **C5a** | **虚扶未在 aug 上放大（源自 E209）** | `eef_ori_deg_mean` 的 `D−C` 增幅 **≤ `B−A` 的增幅 + 1.0°**（gravcomp 在 aug 上的姿态代价不显著大于它在 orig 上的代价）；`root_ori` 同向检查；**同时报 `contact_in_mask` 以复核 E209「contact 不是虚扶检测器」的方法学结论在 bucket aug 上是否复现** |
| **C5b** | **手穿透的双重代价未失控（源自 E208 F14 + E207）** | `hand_object_physics_penetration_3mm_frame_frac`：报 `D−C`（gravcomp 代价）、`D−B`（aug 代价）、以及**是否可加**（`(D−C)+(C−A)` vs `D−A`）。门：`hand_penetration` 门的 narrow 通过率 `rate(D) ≥ rate(B) − 0.20`；**并报右尾 p75/p90/max**（E208 F14 的形态是中位小、右尾重，只看均值会漏） |
| **C6** | 有效位移被量化并分层 | 15 条全部报有效位移；`<0.05 m` 的条目标记为「不算增强」并从 C3 主表剔除（**下限 `EFFECTIVE_AUG_FLOOR_M=0.05`，不追溯否定 E202 已交付数据**）；delta 按位移档分层 |
| **C7** | 075_p2 的缺席是显式登记而非静默丢失 | P1 契约断言它在 E202 manifest 命中 0 行；log 写明排除理由（`runtime_initial_overlap`，参考层几何事实）。**若某天 E202 被回填，该断言会主动报错提醒复核** |
| **C8** | 视觉无新增失效 | P9 完成；三路并排；无穿地/抖动/发散；虚扶若出现须**指出具体帧与形态**，不得只写「未见异常」 |

**总判定**：
- **SUCCESS** = C0–C3 全过 **且** C5a/C5b 达标 **且** C6/C7/C8 无红线（C4 为发现型，不参与判定）
- **PARTIAL SUCCESS** = C0–C2 过、C3 过，但 C5a 或 C5b 破（姿态或手穿透代价显著放大）→ 数据仍可交付但须在交付说明中标注具体退化维度与受影响条目
- **FAIL** = C1 破（非单变量）或 C2 < 15/15 或 C3 破（aug×G1 显著劣于 orig×G1 / 出现 fall）

**关于「预注册就不改」**：E207 C4 的计数子句、E208 C4 的 pass-rate 门都在中途显得「不该那么定」，两次都选择**不回溯改门、如实记 PARTIAL/FAIL 并解释成因**。E210 沿用该纪律——上面的门若在跑完后显得过紧/过松，写进 log 分析，不改判据。

**显式不作为判据**：
- z bias/MAE 的任何数值门（E209 已证 E207 的常量模型不可外推；z 只作诊断）
- B→D 的严格因果归因（含 v1/v2 retarget confound，只声称「不劣、可用」）
- 真实重力下的可执行性（E207 已由用户确认接受该建模落差）
- rot 变体的放量结论（只有 rescue 一例可能产出，n 太小）

---

## 风险 / 缓解

| # | 风险 | 缓解 | 触发点 |
|---|---|---|---|
| 1 | **与 E208 P6 抢 GPU**（M6；本分支已发生 2 次） | 默认等 E208 跑完；若提前启动须显式指定空闲卡号，不用 `GPUS=0..7`。priority queue 的 free-mem 判据识别不了他人进程，这条**不能靠脚本自动兜住** | P7 |
| 2 | 误跑成 aug×PRG 或 aug×G1A2 | P6 smoke 回读 `config_act.yaml` 7 项断言 | P6 |
| 3 | **虚扶在 aug 上被放大**（E209 主结论 + aug 力臂更长） | C5a 数值门（eef_ori 为主，**非 contact**）+ P9 视觉专项；若破则判 PARTIAL 并在交付说明中标注 | P8/P9 |
| 3b | **手穿透双重代价叠加**（E208 F14：aug 主代价是 hand_pen；E207：gravcomp 也 +0.035） | C5b 独立门 + 报右尾 p75/p90/max（F14 形态是中位小右尾重）+ 报可加性检验；P9 抽 hand_pen 增幅最大 2 条看画面 | P8/P9 |
| 3c | **A/B 视频比较被机位陷阱污染**（E208 F13：`_auto_video_camera` 逐帧重算，且 aug 的 ref pane 是增强后的参考） | P9 只做 C↔D（参考轨迹逐字节相同）的画面 A/B；B↔D 只走数值；log 里逐条标注结论是数值来源还是画面来源 | P9 |
| 4 | `073_p1` 在 orig×G1 下已压线掉门（eef_ori 19.83° / 门 20°），aug 后大概率整体掉出 | 预期内，不视为新缺陷；C3 用**比率**而非绝对 case 数，且逐 case 报 | P8 |
| 5 | 复用 `e200_common.build_gravcomp_sidecar` 写错文件名 | M5/E209 F1 已登记，P2 明确自写 writer + 复用 `assert_gravcomp_diff` | P2 |
| 6 | 模块同名自我遮蔽（E207 F9、E202 也踩过） | 所有脚本/模块带 `e210_` 前缀；`load_e210_module` 按绝对路径加载 | P1 |
| 7 | 部分变体有效位移过小、实为「假增强」（E208 F7，E202 bucket 33% <0.18 m） | C6 强制量化 + `<0.05` 剔除 + 分层；不追溯否定 E202 | P8 |
| 8 | v1/v2 retarget confound 被误读为严格 ablation | log 与 workbook 每张 B→D 表头**强制标注** confound；C→D 才是严格单变量 | P8 |
| 9 | n=15（5 case × 3 档）统计功效弱，且档位在 case 内相关 | 用 paired 逐条 delta + 符号检验；**按 case 聚合后再报一次**（避免把 3 档当独立样本）；报 mean+std+worst | P8 |
| 10 | **门槛口径漂移导致通过率不可比**（E208 F11：同批 case 曾算出 3 个互不相等的通过率，根因是 `lower_body` 0.20 vs 0.10） | 4 格共用同一份 `E201/funnel_config.py`；输出 `gate_criterion_vs_E207` 分歧表，应为空 | P8 |
| 11 | **静默的 join/序列化 bug**（E208 F12：三个都不报错、不打日志——review sheet 键是 `{case}#ARM` 非裸 case_id；`--no-rescore` 只拷 KEY_METRICS 导致 release 门假失败；布尔小写 `true` vs `"True"` 计数全 0） | P8 三条实现纪律 + 对每张 join 后的表断言「非空且非全空列」 | P8 |
| 12 | scipy 旋转被残缺 torch stub 污染（M10）；`relative_to(REPO)` 在符号链接下抛错（M9/E207 F8） | 统一先调 `_drop_broken_torch()`；用 `rel_label()`；`write_tsv` 带 JuiceFS EIO 重试（`e199_common.py:239-249`） | P1–P8 |
| 13 | ~~rescue 与他人 retarget 进程竞争~~ | **已消除**：口径 3 取消 rescue，本实验零 upstream retarget | — |

---

## 改动文件

**新建**
| 路径 | 作用 |
|---|---|
| `workspace/core4d/plan/240_E210_bucket007_aug_g1only_gravcomp_plan.md` | 本计划 |
| `workspace/core4d/scripts/experiments/E210/e210_common.py` | 契约 + 6 case 白名单 + 2 份 authority sha pin + 15 变体定位 + 075_p2 缺口断言 |
| `.../E210/build_gravcomp_sidecars.py` | 15 个 gravcomp sidecar + `assert_gravcomp_diff` + geom/pair parity |
| `.../E210/build_overrides.py` | 15 个 override + compose 审计 |
| `.../E210/build_manifest.py` | 15 行 manifest |
| `.../E210/build_e210_aug_review_tsv.py` | review TSV（前缀防遮蔽） |
| `workspace/core4d/scripts/launch/active/run_E210_local_8gpu.sh` | CEM 队列入口（thin wrapper） |
| `workspace/core4d/scripts/launch/active/run_E210_render_all.sh` | 渲染入口（固定 osmesa） |
| `workspace/core4d/scripts/eval/runners/eval_E210_aug_g1only.py` + `wrappers/eval_E210_aug_g1only.sh` | 四格 14-gate + C5 虚扶专项 + C6 位移分层（**wrapper 这次真的创建**，M8） |
| `workspace/core4d/scripts/eval/reports/gen_E210_four_cell_workbook.py` | 四格工作簿 |
| `examples/config/override/core4d_E210_<case>_aug_<trans>_PRG_gravcomp.yaml` × 15 | override（链到 E202 同名 arm override） |
| `example_datasets/.../<15 aug task>/scene_act_E210_bucketAlignedTop_PRG_gravcomp.xml` | 新 sidecar（并 `git add -f`） |
| `workspace/core4d/results/E210/scene_snapshot/` | rules §7 保障 2 |
| `workspace/core4d/log/299_E210_bucket007_aug_g1only_gravcomp.md` | 跑完后写 |

**修改**
| 路径 | 改动 |
|---|---|
| `workspace/core4d/EXPERIMENT_TRACKER.md` | 顶部加 R296 一行 |
| `workspace/core4d/progress.md` | 按 rule 3 每 2 步更新 |
| `workspace/core4d/scripts/eval/review/review_index.py` | 注册 `E210AUG` review set |
| `example_datasets/.../<15 aug task>/*.xml, *.json` | 仅 `git add -f`，不改内容 |

**`spider/` core 与 `examples/run_mjwp.py` 零改动** —— E210 只加 scene sidecar 与 override。
**E202 / E207 / E208 / E209 的任何历史产物零覆盖**（只读 import）。

---

## 验证（端到端）

```bash
cd /mnt/ali-sh-1/usr/xiayibo/work_dir/embodied/spider
export PATH="$PWD/.venv/bin:$PATH"

# P1-P4 契约（全部 fail-fast）
.venv/bin/python workspace/core4d/scripts/experiments/E210/e210_common.py
.venv/bin/python workspace/core4d/scripts/experiments/E210/build_gravcomp_sidecars.py
.venv/bin/python workspace/core4d/scripts/experiments/E210/build_overrides.py --audit
.venv/bin/python workspace/core4d/scripts/experiments/E210/build_manifest.py

# P5 快照
bash workspace/core4d/scripts/convert/snapshot_scenes.sh E210 <15 个 aug task 目录名>

# P6 smoke（含运行时 config_act 7 项断言）
STAGE=smoke LIMIT=1 NUM_SAMPLES=64 MAX_ITERS=4 GPUS=<空闲卡> \
  bash workspace/core4d/scripts/launch/active/run_E210_local_8gpu.sh

# ---- 以下由用户执行 ----
# P7 full（⚠ 等 E208 P6 跑完，或显式指定空闲卡）
GPUS=0,1,2,3,4,5,6,7 PER_GPU_MEM_MIB=5000 MAX_PER_GPU=1 \
  bash workspace/core4d/scripts/launch/active/run_E210_local_8gpu.sh

# P8 评测
bash workspace/core4d/scripts/eval/wrappers/eval_E210_aug_g1only.sh full --require-all
.venv/bin/python workspace/core4d/scripts/eval/reports/gen_E210_four_cell_workbook.py

# P9 视觉
bash workspace/core4d/scripts/launch/active/run_E210_render_all.sh
bash workspace/core4d/scripts/eval/wrappers/review_player.sh E210AUG
```

**Git**：按阶段分次 commit，`exp(core4d): E210 P{N} — {一句话}`；最终 `exp(core4d): R296 E210 bucket007 aug × PRG+G1 — {结论}`。
**注意**（E207/E209 已踩）：本分支多会话并行，commit 必须**显式列路径**，禁止 `git commit -a`。

---

## 下一步（本计划之外）

- 若 C3/C5 通过：接 E202-export 口径做 partner 增强 + Holosoma paired RL 导出，把 15 条 aug 变成配对 motion（用户本轮明确不做，此处登记）
- 若 C5 破（虚扶在 aug 上放大）：按 E209 的建议试**部分补偿 `gravcomp ≈ 0.5`**，收缩型拟合已给出预测起点
- 14-gate 缺一条 **hand-object 相对姿态门**（E209 结论）——E210 的 C5 是它的临时替代，值得单独立项
- **bucket 类的 rot 档从未真正跑过**（E208 F6：E202 的 trans-only 是 scipy 崩溃导致的误判，非物理限制）。本轮按用户口径 3 不做；若日后要把 bucket aug 放量翻倍，rot0/rot1 是现成的一倍产能，且 E208 在 desk/chair 上实测可行率与 trans 持平（20/22、21/22）。届时 075_p2 也才有被救回的可能
