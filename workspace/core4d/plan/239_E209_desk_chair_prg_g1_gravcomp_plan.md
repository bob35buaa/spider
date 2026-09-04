# plan239 · E209：desk/chair PRG + G1（object gravcomp）单变量

_Core4D · Phase 68 · Run **R295** · 承接 [E206/plan236](workspace/core4d/plan/236_E206_desk_chair_move2_dcv3_noprg_prg_plan.md) + [E207/log296](workspace/core4d/log/296_E207_bucket_g1only_gravcomp.md) · 分支沿用 `feat/E207-bucket-g1only-gravcomp`（不新建）_

---

## Context

E206 在 desk/chair 上交付了 22 例人审 USE 的 PRG rollout（`results/E206/s6_downstream/rl_export/paired_rl_export_input.tsv`）。这批数据有一个**未被 E206 度量过**的系统性缺陷：物体持续低于参考轨迹。

我在计划期用 E178/E207 的同一函数（`gen_E178_object_z_diff_report.object_z_series`）对这 22 例实测：

| | E178 bucket (n=27) | E207 的 9 例子集 | **E206 desk/chair (n=22)** |
|---|---|---|---|
| z bias 宏平均 | −1.820 cm | −1.379 cm | **−2.517 cm** |
| 下沉 case 数 | 24/27 | — | **18/22** |
| r(参考抬升, bias) | −0.870 | — | **−0.738** |

**根因已由 E194/E207 定位且有现成解**：CORE4D 的 object 不是自由体，而是被 6 个 slide/hinge 关节 + `<position>` 执行器钉在参考轨迹上（运行时 `init_pos_actuator_gain=500`），P-only 无重力前馈，稳态下垂 `sag = m·g/kp`。**G1 干预 = object body 加 `gravcomp="1"`**，在质心施加 `+m·g`。这是唯一改动，不涉及 reward、CEM 配置、接触约束、几何。

**为什么值得跑，而不只是把 E207 复制一遍**：E207 的结论（gravcomp ≈ **+1.945 ± 0.711 cm 常量偏移**）存在两个它自己无法解开的混淆，而 E209 的数据结构恰好能解开：

1. **质量混淆**。E207 按质量分层后根本不是常量：`mass=2.0`(n=7) → **+1.691±0.533**，`mass=5.0`(n=2) → **+2.836±0.550**，`r(mass, delta)=+0.710`。而 **E209 的 22 例物体质量全部 = 5.000 kg**（逐 XML 核对，零方差），正好落在 E207 只有 2 个样本的高质量端。
2. **pre-bias 与质量共线**。E207 里 `mass=5` 的 2 例 pre-bias 均值 −2.08，`mass=2` 的 7 例是 −1.18 —— 「按 pre-bias 收缩」与「按质量加性偏移」两族模型在 E207 的数据上**结构性不可分**。E209 在**固定质量 5.0 kg** 下把 pre-bias 拉到 **−5.646 … +2.668**，直接切断该共线。

**目标**：在 22 例上以严格单变量验证 gravcomp，同时裁定其传递函数是「收缩型」还是「加性型」。

---

## 已确认的口径（用户 2026-09-05 决定 4 项）

| 项 | 决定 |
|---|---|
| 编号 | **E209**（E208 已被 desk/chair 平移增强占用，且共用同一批 22 case） |
| arm 格数 | **只跑 G1only = E206 PRG + gravcomp**，22 条；基线复用 E206 PRG rollout 零重跑 |
| 收尾 | 14-gate + z 诊断 **+** 渲染 22 条视频 **+** 22 例全量人审 |
| GPU | **本实验先跑**（8 卡当前全空闲，~2.1 h），E208 P6 随后 |
| 不做 | RL 重导出（`paired_rl_export`）—— 用户明确未选 |

**log 编号裁定**：`log/296` 已被 E207 落盘占用，而 E208 的 commit 也口头声明用 log296。裁定为 **E207=296（已存在，不动）/ E208=297 / E209=298**，并在 P0 顺手修 E208 progress 里的 log 引用。

---

## 计划期已实测的关键事实（非假设）

| # | 事实 | 证据 |
|---|---|---|
| **M1** | 22 例 z 基线：宏 bias **−2.517**、\|bias\| 均值 **3.241**、z MAE **4.460**、obj 3D 误差 **11.08 cm**、参考抬升 25.9 cm(9.8–44.1) | 复用 `object_z_series` 实跑 22/22 |
| **M2** | 分层：**S⁻**（bias<0，n=18）宏 **−3.519**；**S⁺**（bias≥0，n=4）宏 **+1.994**，S⁺ 恰为 chair006 的 4 例（+2.668/+2.268/+0.651/+2.386） | 同上 |
| **M3** | 22 例 object **mass 全部 = 5.000 kg** | 逐 `scene_act_E206_lowgeom_PRG.xml` 正则提取 |
| **M4** | E207 分层 delta：mass=2 → +1.691±0.533(n=7)；mass=5 → **+2.836±0.550**(n=2)；池化 +1.945±0.711；`r(mass,delta)=+0.710` | 用 E207 `e207_object_z_diff_by_case.tsv` + E207 scene 质量重算 |
| **M5** | E207 收缩拟合 `post = 1.0939 + 0.3827·pre`，r=0.705，**残差 sd = 0.377 cm** | 同上 OLS |
| **M6** | 22 例 14-gate 基线：`hard 22/22`、`wide 19/22`、**`narrow 13/22`**；narrow 失败构成 = hand_pen 5 / release 2 / root_pos+ori+eef_ori 1 / root_pos+eef_ori 1 | `e206_two_arm_rollout.tsv` 过滤 arm=prg ∩ 22 |
| **M7** | 其它基线：`leg_penetration_frac` 均值 0.0102、`track_obj_pos_err` 11.08 cm、`track_obj_ori` 均值 5.64 / **max 9.808**（narrow 门 10.0，唯一贴边者 = `desk021_20231011_014_p2`，也是唯一 omnirt_v2 case） | 同上 |
| **M8** | 人审基线 **USE 22/22**（CLEAN 11 / MINOR_ACCEPTABLE 11），`manual_failure_taxonomy` 全空 | `user_manual_review_filled.tsv` |
| **M9** | 22 例的 dcv3 task dir 在 git 里**一个文件都没有**（`.gitignore:206` 命中 `example_datasets/`）→ 违反 rules §7 保障 1 | `git ls-files` / `git check-ignore -v` |
| **M10** | `results/E206/scene_snapshot/` 快照的是 `<obj>_person<N>/` 源模板，**不含实际跑 CEM 的 `scene_act_E206_lowgeom_PRG.xml`**（E207 做对了）→ 保障 2 也有缺口 | 列快照目录 |
| **M11** | 8 卡 L20Y 全空闲（4 MiB / 0%）；E206 PRG 这 22 例 wall median **44.0 min**（mean 45.8 / max 70.5）→ 8 卡投影 **2.1 h** | `nvidia-smi` + `e206_cem_summary.json` |

---

## 精确定义

| Arm | scene_act | leg P/R/G | hand-gate | 来源 |
|---|---|---|---|---|
| **PRG**（基线） | `scene_act_E206_lowgeom_PRG` | 18N pair + penalty 2.0 + gate on | A0（E163 默认） | E206 已有 22 条，**零重跑** |
| **G1**（本次） | `scene_act_E209_lowgeom_PRG_gravcomp` | 同上，逐字节相同 | A0，同上 | 新跑 22 条 |

唯一差异 = object body `gravcomp` 缺失 → `"1"`。**不带 A2 hand-gate**（E207 的 C7 已证 A2 对 z 贡献可忽略）。

> **术语澄清（易错）**：PRG **不是** arm/gripper 变体，而是 lower-body 的 P（16 个 lower-body geom ↔ object 的 collision pair）/ R（`leg_object_penalty_*`）/ G（`cem_leg_gate_*`）三开关。E206 两 arm 的手部几何完全相同。G1 指 object gravcomp，与 `robot_type: unitree_g1` 无关。

---

## 预注册预测（P0 冻结进 `e209_common.py` 常量，跑完只做 RMSE 比较，不得事后换模型）

| 模型 | 形式 | 来源 |
|---|---|---|
| **M-shrink** | `post = 1.0939 + 0.3827·pre` | E207 9 例 OLS（M5） |
| **M-pooled** | `post = pre + 1.945` | E207 池化 delta（M4） |
| **M-mass** | `post = pre + 2.836` | E207 `mass=5.0` 单元，与 E209 质量匹配（M3/M4） |

对 22 例的预测：

| 模型 | 宏 bias(22) | mean\|bias\|(22) | 改善数 | 宏 S⁻ | 宏 S⁺ |
|---|---|---|---|---|---|
| M-shrink | **+0.131** | **0.680** | **20/22** | −0.253 | +1.857 |
| M-pooled | −0.572 | 2.154 | 17/22 | −1.574 | +3.939 |
| M-mass | +0.319 | 1.783 | 17/22 | −0.683 | +4.830 |

**判别 case（7 例，三模型间距 ≥2.1 cm，测量残差 sd 仅 0.377 cm）**：

| case | pre | M-shrink | M-pooled | M-mass | 分离度 |
|---|---:|---:|---:|---:|---:|
| `chair006_20231003_1_003_p1` | +2.668 | +2.115 | +4.613 | +5.504 | **9.0σ** |
| `chair006_20231011_076_p2` | +2.386 | +2.007 | +4.331 | +5.222 | 8.5σ |
| `chair006_20231003_1_005_p1` | +2.268 | +1.962 | +4.213 | +5.104 | 8.3σ |
| `desk021_20231008_005_p2` | −5.646 | −1.067 | −3.701 | −2.810 | 7.0σ |
| `chair005_20231030_043_p1` | −4.906 | −0.784 | −2.961 | −2.070 | 5.8σ |
| `chair006_20231003_2_011_p2` | +0.651 | +1.343 | +2.596 | +3.487 | 5.7σ |
| `desk021_20231008_005_p1` | −4.780 | −0.735 | −2.835 | −1.944 | 5.6σ |

**S⁺（chair006 4 例）的处理原则** —— 防止事后辩护：
1. 分层 `S⁺ = {case : E206_PRG_z_bias ≥ 0}` 在 **P0 由基线数据机械决定并 commit**，不是跑完看结果划线。`chair006_20231003_2_015_p1`(−0.726) 留在 S⁻，**不按物体名切**。
2. **C4b 是可失败的预注册硬门**，不是注脚。若 FAIL，结论必须写成「gravcomp 对基线 bias≥0 的 case 禁用」并进入后续 arm 准入条件。
3. **C4c 强制报告全 22 例**；只报 S⁻ 视为违反 rules §5「选择性报告」。

---

## 执行阶段

### P0 · 基线冻结 + git 欠账（无 GPU，~20 min）
- `git add -f` 22 个 dcv3 task dir 的 `scene.xml` / `scene_act.xml` / `scene_act_E206_lowgeom_{PRG,noPRG}.xml` / `scene_act_meta.json` / `task_info.json`（补 M9）
- 新建 `E209/e209_common.py`：`CASES`(22，硬编码 + 对 `paired_rl_export_input.tsv` 校验分布 `{desk021:7, chair006:5, desk007:5, desk023:4, chair005:1}`)、`SCENE`、`build_sidecar()`、`audit()`、**三模型系数 + S⁺/S⁻ 分层常量**
- 冻结三份基线 TSV 到 `results/E209/baseline/` 并记 sha256：z（22 行）/ 14-gate（22 行）/ 人审（22 行）

**退出检查**：`git ls-files` 覆盖 22 dir × ≥5 文件；基线回归值 z 宏 **−2.517**、neg **18/22**、narrow **13/22**、USE **22/22** 全部复现。

### P1 · scene sidecar + 单变量审计（无 GPU，~2 min）
`E209/build_scenes.py` → 22 个 `scene_act_E209_lowgeom_PRG_gravcomp.xml`。

**必须自写 writer（~12 行）**：`E200/e200_common.py:161` 的 `build_gravcomp_sidecar()` 把输出名硬编码成 `GRAVCOMP_SCENE`（= `scene_act_E199_rubberHull_PRG_gravcomp`），直接套用会在 desk 目录里写出一个名字撒谎的文件。**`assert_gravcomp_diff()`（`:134-149`）逐字复用，不改一行**（E207 的先例）。

**退出检查**：22/22 `assert_gravcomp_diff` PASS；每个 sidecar `grep -c gravcomp == 1`；MuJoCo 22/22 编译通过且 `ngeom`/`npair` 与基线逐 case 相等（gravcomp 不应改变任何几何 —— 单变量的第二重证明）。

### P2 · override + Hydra compose 审计（无 GPU，~3 min）
`E209/build_overrides.py` → `examples/config/override/core4d_E209_<case>_lowgeom_PRG_G1.yaml`，base = `core4d_E206_<case>_lowgeom_PRG`，仅覆盖 `scene_name`。

审计照 `E207/build_overrides.py:58-92`，**allow-list = `{"scene_name"}` 恰好 1 键**。注意：**不要沿用 E206 的 `ARM_DIFF_KEYS`(12 键)** —— 那是 noPRG↔PRG 的对比集，E209↔E206-PRG 两侧 leg 三件套完全一致。

正向断言：A0 hand-gate 4 键 == `e206_common.E163_HAND_GATE`；`leg_object_penalty_scale==2.0`；`cem_leg_gate_enabled is True`；`object_collision_sdf_mode=="union"`；**`contact_hdmi_mask_path` 两侧字符串相同**（eval 从各自 `config_act.yaml` 读掩码，不同则接触类指标不可比）。

### P3 · manifest（无 GPU，~2 min）
`E209/build_manifest.py`（照 `E207/build_manifest.py` 的 FIELDS 契约）→ 22 行，`tier=P0`、`arm=G1`、1024×32×seed0，含 `variant`/`preferred_pool` 两列（渲染器需要）。

**退出检查**：`trajectory_sha256` / `contact_mask_sha256` 与 E206 PRG 的 `config_act.yaml` 实际消费路径逐 case 一致；22 个 `effective_scene_sha256` 互不相同且都 ≠ E206 PRG scene。

### P4 · 场景快照（rules §7 保障 2，~1 min）
`bash workspace/core4d/scripts/convert/snapshot_scenes.sh E209 <22 个 dcv3 task dir>`。

**照 E207 快照 dcv3 task dir，不要照 E206 快照源模板**（M10）——这样一份快照同时含基线 `scene_act_E206_lowgeom_PRG.xml` 与处理组 sidecar，双臂覆盖。

### P5 · smoke（1 卡，~5 min）
`chair006_20231003_1_003_p1`（最强判别 case，9.0σ）× 64×4，落 `cem/smoke/`。

**退出检查（7 项运行时断言，不可跳过）**：回读 `config_act.yaml` 断言 `scene_name` 后缀 == `_gravcomp`、`cem_hand_gate_max_violation_pct==0.10`、`cem_hand_gate_hard_floor_m==-0.020`（**A0，不是 A2 的 0.05/−0.015**）、`leg_object_penalty_scale==2.0`、`cem_leg_gate_enabled==true`、`init_pos_actuator_gain==500.0`、`init_rot_actuator_gain==50.0`；与 E206 PRG 的 `config_act.yaml` 全键 diff **恰好 1 键**（`scene_name`，`output_dir` 除外）。**这是唯一能证明「跑的确实是 G1 而非 G1A2 / 而非基线」的关卡。**

### P6 · full CEM（8 卡，~2.1–2.5 h）
`E209/run_E209_cem.sh` → **`E199/run_local_priority_queue.py`**（manifest 队列），`--gpus 0..7 --per-gpu-mem-mib 5000 --max-per-gpu 1`。

**必须用 E199 版，不能用 E200 版**：`e200_common.py:93 TIER_RANK = {"P1": 1}`，manifest 里 `tier=P0` 会直接 KeyError；`e199_common.py:140` 才含 P0/P1/P2。

选队列而非 `E206/run_e206_cem.py` 进程池的理由：队列的 `input_failures()` 对 scene/trajectory/override/contact_mask 逐项比 sha256 后才发车（这正是 C0 的机械化证明），`output_failures()` 回读 `config_act.yaml` 断言 `scene_name` 一致；E206 进程池只看 returncode，且会硬性依赖 `admission_decision.json`。

环境：`MUJOCO_GL=disable`、`TORCHDYNAMO_DISABLE=1`、`+use_torch_compile=false`、`save_video=false`。**不要跑 `uv sync`**（共享 venv）。

**不重做吞吐探针**（gravcomp 是 body 标量属性，不改 geom/pair 数与 CEM 预算，E206 准入按构造转移），改为事后门 C8。

### P7 · 评测 + z 诊断 + 渲染（CPU，~40 min，三条并行）
1. **14-gate**：新写 `scripts/eval/runners/eval_E209_g1_gravcomp.py`。**不给 `eval_E206_arm_ablation.py` 加 arm** —— 它 5 处硬绑 `e206_common.ARMS`，而 `e206_common` 被 E208 直接 import（`e208_common.py:53`），改它会静默污染 E208 分支。照 E207 先例（`eval_E207_g1only.py:47`）静态 import 基线 runner 的**阈值表与分类器**（`GATE_FIELDS` / `PHYSICS_6` / `TRACKING_6` / `target_task_by_case` / `contact_mask_for` / `classify_funnel` / `classify_12gate` / `stats`），打分一律走 `eval.core.core_metrics`（rule 13）。
   **两 arm 都重新打分，绝不读 `e206_two_arm_rollout.tsv`** —— `eval_E206_arm_ablation.py:5-9` 专门修掉过「两 arm 不在同一 code path 下测量」这个 confound，不能请回来。
2. **z 诊断**：新写 `scripts/eval/reports/gen_E209_object_z_diff.py`，复用 `gen_E178_object_z_diff_report.object_z_series` / `sha256`；参考轨迹从 `paired_rl_export_input.tsv` 的 `stage2b_target_task` 拼（E206 无 `evaluated_manifest_snapshot.tsv`）。**必须输出三模型 RMSE 判别表**。
3. **渲染**：新写 `E209/render_cem_results.py`（~70 行，从 manifest 构 row，复用 `E168/render_a100_cem_videos.render_row`）。`E206/render_cem_results.py` 三处硬绑 `C.ARMS`/`SCENE_BY_ARM`，不可直接用。**`export MUJOCO_GL=osmesa`** —— E207 P8 实测 egl→EGLError、glfw→无 `_mjr_context`，本机只有 osmesa 可用。**只渲 22 条 E209**，E206 PRG 的 22 条 mp4 已全部存在。

然后 `E209/build_e209_arm_review_tsv.py`（**文件名不得叫 `build_arm_review_tsv.py`**，会 shadow 被 import 的 E206 同名模块）+ `review_index.py` 注册 `E209ARM`。

### P8 · 22 例全量人审 + 收口（人工，~1–2 h）
`bash workspace/core4d/scripts/eval/wrappers/review_player.sh E209ARM`，逐 case **A/B 对看 PRG mp4 vs G1 mp4**（E209 相对 E207 的实质优势：两 arm 都有视频）。

**新增强制审查项**：`manual_failure_taxonomy` 必须能记 **`object_floating`** —— gravcomp=1 让物体在 sim 里完全失重，「被托着走 / 虚扶 / 漂浮」是最可能的视觉反常，而 **14-gate 里没有任何一门看得见它**。这是 rules §5「视觉评估强制」在 E209 的具体落点。

**退出检查**：22/22 已审；每个 USE→DO_NOT_USE 降级必须填 taxonomy；产出 log298 + TRACKER R295 行 + Claims 结算表。

---

## Claims 与量化成功标准

基线均为计划期实测值。

| ID | Claim | 数值门 | 基线 |
|---|---|---|---|
| **C0** | 输入零漂移 | 22/22 `trajectory` + `contact_mask` sha256 相同；`contact_hdmi_mask_path` 字符串相同 | — |
| **C1** | 单变量·配置层 | Hydra compose 对称差 **恰好 `{scene_name}`**，22/22 | — |
| **C1b** | 单变量·场景层 | 22/22 `assert_gravcomp_diff` PASS ∧ `ngeom`/`npair` 逐 case 相等 | — |
| **C1c** | 单变量·运行时 | 22/22 `config_act.yaml` 全键 diff 仅 `scene_name`；A0 hand-gate 4 键 == E163 | — |
| **C2** | 执行闭合 | 22/22 完成，0 fail/diverge | E206 22/22 |
| **C3** | **主门 · 14-gate 不劣化** | `narrow_pass ≥ 12/22` ∧ `hard_pass == 22/22` | narrow **13/22**、hard 22/22 |
| **C4** | **z 修正 · S⁻(n=18)** | ① \|宏 bias\| ≤ **1.0 cm** ② ≥**16/18** case \|bias\| 下降 ③ \|post\|>1.5 cm 者 ≤2/18 | 宏 **−3.519**，最小 \|pre\|=0.726 |
| **C4b** | **z 危害上限 · S⁺(n=4)** | ① \|宏 bias\| ≤ 2×基线 = **3.99 cm** ② 无单例 \|post\| > 6.0 cm | 宏 **+1.994** |
| **C4c** | **全 22 例反挑拣（强制报告）** | \|宏 bias\| < 2.517 ∧ mean\|bias\| < 3.241 ∧ Wilcoxon 配对符号秩 p<0.05 | −2.517 / 3.241 |
| **C4d** | **传递函数判别（科学产出）** | 三模型对实测 post 的 RMSE：胜者比次者低 **≥2×** 且 RMSE ≤ 1.0 cm | 见预注册表 |
| **C5** | 承重接触不塌 | `contact_in_mask` 均值 ≥ 0.85 ∧ `leg_penetration_frac` 均值 ≤ 0.02 | 0.879 / **0.0102** |
| **C6** | 副作用天花板（E207 已知代价 release +0.072、hand_pen +0.035） | `release_false_contact` 均值 ≤ 0.18 ∧ `hand_pen_3mm` 均值 ≤ 0.26 | 0.081 / 0.211 |
| **C7** | 人审不降级 | 22/22 已审 ∧ **USE ≥ 18/22** ∧ 降级必填 taxonomy ∧ `object_floating` 单独计数 | **USE 22/22** |
| **C8** | 吞吐无回归（替代重做探针） | wall median ∈ 44.0 ×[0.75,1.25] = **[33.0, 55.0] min** | median **44.0** |
| **C9** | 复现性双保障 | 22 task dir × ≥5 文件已 `git add -f` ∧ 快照 manifest 含 HEAD + **双臂两个 scene** 的 sha256 | E206 快照缺 PRG scene（本次修） |

**C4 门线为何取 1.0 cm 而非 E207 的 0.8**：E209 的 S⁻ 基线（−3.519）比 E207 基线（−1.379）差 2.55×，套 0.8 等于要求 2.72 cm 修正，高于 E207 池化点估计 1.945。取 1.0 → 要求 ≥2.52 cm 修正，正好卡在 M-mass(2.836) 之下、M-pooled(1.945) 之上，**这条门因此能区分「按质量缩放」与「池化常量」两个世界**，不是走过场。0.8 作为 stretch 线并列报告。

**C4「≥16/18」是对 E207 C4 失败模式的定向修复**：E207 的「≥7/9」只拿到 6/9，原因是 3 例 `|pre|` 本就极小（0.243/0.362/1.014），常量偏移必然过冲 —— 那是判据设计缺陷而非干预失败。E209 的 S⁻ 最小 `|pre|`=0.726 且 17/18 的 `|pre|>1.5`，「改善」子句在 S⁻ 上是公平的；S⁺ 拆到 C4b 用「危害上限」单独评判。

**总判定**：
- **SUCCESS** = C0–C3 全过 ∧ C4 达标 ∧ C4b/C4c 无红线 ∧ C5/C6/C7 无红线
- **PARTIAL** = C0–C3 过，C4 方向正确未达门（\|宏 S⁻ bias\| 落 1.0–1.5 cm）
- **FAIL** = C1 系列破（非单变量）∨ C2 < 22/22 ∨ C3 narrow ≤ 11/22 ∨ C7 USE < 18/22

**显式不作为判据**：z **MAE** 降幅（E207 已证 gravcomp 修 bias 不修 MAE）；2×2 交互项（不跑 noPRG+G1）；真实重力下可执行性（沿用 plan237 的声明式接受）。

---

## 风险 / 缓解

| # | 风险 | 缓解 | 触发点 |
|---|---|---|---|
| R1 | 直接套 `build_gravcomp_sidecar` → 在 desk 目录写出名叫 `scene_act_E199_rubberHull_PRG_gravcomp.xml` 的文件 | 自写 12 行 writer，只复用 `assert_gravcomp_diff` | P1 |
| R2 | 用 `E200/run_local_priority_queue.py` → `TIER_RANK["P0"]` KeyError | 用 **E199 版** | P6 |
| R3 | 给 `e206_common.ARMS` 加第三 arm → 静默污染 E206 与 E208（后者 import 它） | 新写 runner，`e206_common` 只读不改 | P7 |
| R4 | eval 基线读 E206 TSV 而非重打分 → 复活已修掉的 code-path confound | 两 arm 同 code path 重打分 | P7 |
| R5 | 渲染用 egl/glfw → 卡死 | 脚本内 `export MUJOCO_GL=osmesa` + 入口断言 | P7 |
| R6 | `review_index.py:996` 的 arm-sweep 元组与 E208 行冲突 | 改派生式 `SOURCE_OVERRIDES[exp].get("arm_sweep") or exp=="E205"`，跑 `--check` 全绿；否则接受一次 rebase | P7 |
| R7 | `desk021_20231011_014_p2` 是唯一 omnirt_v2 case，走 v1 会静默用错轨迹 | 一律用 `target_task_by_case()`，manifest 逐行落 `target_task` 并 sha 校验 | P3/P7 |
| R8 | **gravcomp 致物体失重「漂浮/虚扶」，14-gate 全门看不见** | C7 强制 `object_floating` taxonomy + 22 例 A/B 对看 | P8 |
| R9 | chair006 大幅过冲后被事后说成「本来就不适用」 | 分层 P0 冻结 commit；C4b 可失败硬门；C4c 强制全量报告；三模型系数预注册 | P0/P8 |
| R10 | `obj_ori` 最大值 9.808 贴 narrow 门 10.0，单点可能翻转 | C3 门设 ≥12/22 留 1 例余量；逐 case 报告翻转明细 | P7 |
| R11 | 与 E208 抢 GPU / 文件冲突 | 本实验先跑（2.1 h 即腾空）；除 `review_index.py` + TRACKER 外全部新路径 | P6 |

---

## 改动文件

**新建 · 脚本（8）**：`E209/{e209_common.py, build_scenes.py, build_overrides.py, build_manifest.py, snapshot_E209_scenes.sh, run_E209_cem.sh, render_cem_results.py, build_e209_arm_review_tsv.py}`

**新建 · 评测（2）**：`scripts/eval/runners/eval_E209_g1_gravcomp.py`、`scripts/eval/reports/gen_E209_object_z_diff.py`

**新建 · 记录（2）**：`plan/239_E209_desk_chair_prg_g1_gravcomp_plan.md`、`log/298_E209_desk_chair_prg_g1_gravcomp.md`

**新建 · 数据**：22 个 `scene_act_E209_lowgeom_PRG_gravcomp.xml`（`git add -f`）；22 个 override；`results/E209/{baseline,scene_snapshot,s6_downstream/{manifests,cem/full,eval/two_arm,render/full}}/`；补 22 task dir × ~5 文件的 `git add -f`（M9 欠账）

**修改（3）**：`scripts/eval/review/review_index.py`（追加 `E209ARM` + `:996` 派生式）、`EXPERIMENT_TRACKER.md`（R295 行）、`progress.md`

**SPIDER core（`spider/`、`examples/run_mjwp.py`）零改动**；**E206/E207/E208 脚本零改动**。

---

## 验证（端到端）

```bash
cd /mnt/ali-sh-1/usr/xiayibo/work_dir/embodied/spider
E=workspace/core4d/scripts/experiments/E209

.venv/bin/python $E/e209_common.py --audit        # P0/P1: 22/22 契约 + gravcomp 断言
.venv/bin/python $E/build_overrides.py --audit    # P2: diff_keys == {"scene_name"}
.venv/bin/python $E/build_manifest.py             # P3: 22 行 + sha 对齐
bash $E/snapshot_E209_scenes.sh                   # P4: 双臂 scene 入快照
bash $E/run_E209_cem.sh smoke                     # P5: 7 项运行时断言
bash $E/run_E209_cem.sh                           # P6: 22/22, ~2.1h

.venv/bin/python workspace/core4d/scripts/eval/runners/eval_E209_g1_gravcomp.py
.venv/bin/python workspace/core4d/scripts/eval/reports/gen_E209_object_z_diff.py
MUJOCO_GL=osmesa .venv/bin/python $E/render_cem_results.py
.venv/bin/python $E/build_e209_arm_review_tsv.py
.venv/bin/python workspace/core4d/scripts/eval/review/review_index.py --check   # 全实验无 MISMATCH
bash workspace/core4d/scripts/eval/wrappers/review_player.sh E209ARM            # P8 人审
```
