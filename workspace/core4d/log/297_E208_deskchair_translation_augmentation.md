# log297 · E208：desk/chair 物体增强 → PRG Full CEM

**日期**: 2026-09-05（进行中，**P0–P5 完成，P6 已启动**）
**实验域**: `core4d`
**Run**: **R294**
**分支**: `feat/E207-bucket-g1only-gravcomp`（用户指定不新建分支；E208 全部改动都在新路径下，与并行推进的 E207/E209 零文件冲突）
**对应 Plan**: `/root/.cc-mirror/codewiz-cc/config/plans/e206-desk-chair-e207-aug-spider-agile-puddle.md`（**尚未落盘到 `workspace/core4d/plan/238_*.md`，见 §7 待办**）
**前置**: [log295 · E206](295_E206_desk_chair_move2_dcv3_noprg_prg.md) · [log286 · E199](286_E199_omniretarget_object_augmentation_results.md) · [log291 · E202](291_E202_bucket_e178_translation_augmentation.md)

> **编号说明**：用户原话是 "E207"，但 E207 已被 plan237（bucket G1-only gravcomp, R293）占用且当时正在跑，经用户确认改用 **E208 / plan238 / R294**。
>
> **log 号二次让位（2026-09-05）**：本日志初次落盘时写作 `log296`，但并行会话的 E207 早在 01:24 就已提交 `log/296_E207_bucket_g1only_gravcomp.md`（我 15:16 才提交，晚了 14 h），A9 因此报冲突。按 [plan239:40](../plan/239_E209_desk_chair_prg_g1_gravcomp_plan.md) 的裁定 **E207=296 / E208=297 / E209=298**，本日志改号为 **log297**。这是同分支多会话并行的第三次资源碰撞（前两次是 GPU 与 retarget 进程，见 §5）。

---

## 1. 背景与目标

E206 在 desk/chair 上打通 dcv3 全流程 + 手编 lowgeom 碰撞代理 + noPRG/PRG 双 arm，人审出 **22 例 USE**，交付 `paired_rl_export_input.tsv`。但 22 例不足以支撑 RL 训练。

box 类（E199，249 变体）与 bucket 类（E202，73 变体）已证明 OmniRetarget 的物体增强能把单条演示扩成多条且数值不劣于 orig。desk/chair 是唯一还没做增强的物体族。

**E208 = 把这条路线搬到 desk/chair 的 22 例上，跑正式 Full CEM，验证数值与人审都不劣于 orig。**

### 用户锁定的口径

| # | 决定 | 备注 |
|---|---|---|
| 1 | retarget **omnirt_v1 优先 + omnirt_v2 rescue** | orig 基线复用 E206 已有 22 条 PRG rollout，**不重跑 CEM** |
| 2 | **仅 PRG arm** | 与 E206 人审过的 22 例同 arm，直接可比 |
| 3 | 止于 aug 构建 + Full CEM + 评估 | 不做 partner 增强 / paired RL 导出 |
| 4 | 数值门 + orig delta + 抽样视觉 + **全量人审** | 同 E206 C6 口径 |
| 5 | **（中途追加）扩到 5 变体** | 见 §4 F6，原计划 trans-only |
| 6 | **（中途追加）排除 chair005** | 见 §4 F7 |
| 7 | **（中途追加）有效位移升为 C4 分层变量** | 见 §4 F7 |

---

## 2. 已完成（P0–P5 部分）

### P0 契约 + 离线自检 — **C0 通过 10/10**

`E208/e208_common.py`（registry / `OMNIRT_V{1,2}_ENV` / v1-aware `aug_task_name` / rescue 状态机 / PRG-only 命名 / `load_e208_module` / 48 列 FIELDS）+ `test_e208_contract.py`（A1–A10）。

| 断言 | 内容 | 结果 |
|---|---|---|
| A1 | registry 从 E206 `paired_rl_export_input.tsv` 派生恰好 22 行（sha256 `af17e209…1d5e` pin），per-object 计数 pin | pass |
| A2 | 22 个 base task 目录存在、带 task_info.json + scene.xml、21 v1 + 1 v2 | pass |
| A3 | `aug_task_name` v1-aware / 幂等 / 66 名唯一，且与 E199 版本行为确有差异 | pass |
| A4 | `object_name` 逐字透传（大小写敏感） | pass |
| A5 | `load_e208_module` 不会静默落到 E199 同名模块 | pass |
| A6 | PRG-only；`PRG_OVERRIDES`/`E163_HAND_GATE` 取值自 `e206_common` | pass |
| A7 | `OMNIRT_V*_ENV` 六键全显式 且与 E206 `run_stage2b_*.sh` 的 env 字面量逐键相等 | pass |
| A8 | CEM 预算取自 E206 `admission_decision.json` 的 frozen 块（1024×32×seed0，compile=false） | pass |
| A9 | E208 命名空间未被占用 且 E207 文件未被改动 | pass |
| A10 | 源模板仍带 E206 手编代理且全为 box（desk007=12/chair006=10/desk023=9/desk021=5/chair005=2） | pass |

产出 `results/E208/preflight/e208_contract_selfcheck.json`。

### P1 从 E206 逐字节播种 orig — **C1 通过 22/22**

`seed_from_e206.py` + `check_orig_seed_integrity.py`。43 个 case-variant（21 例 v1 × 2 root + 1 例源 v2 × 1 root），**215 硬链接 + 43 拷贝，增量 7 MB**。

R1–R4 全过：`retargeted/_original` 与 `trimmed/_original` sha256 == E206；`trim_window.json` 数值字段逐项相等；E206 的 3cm mask 内嵌 `trim_start` == 播种窗口，三条时间轴（raw/spider/eval）分别与未裁剪帧数、SPIDER `trajectory_kinematic.npz` 的 qpos 帧数、`帧数×50/30` 逐例对齐。

**为什么播种而不是重算**：F15 的两个已知发散例（`chair005_20231030_043_p1` worst |Δ|=1.096、`desk023_20231030_019_p1` 0.097）**都在这 22 例内**，且 chair005 是唯一的 chair005（n=1）。重算会让它的 aug-vs-orig delta 同时含增强效应与同量级 IK 噪声。播种是**构造性免疫，不是检验**。

### P2 探针 + 闸门

| 闸门 | 结果 |
|---|---|
| **G1 成本** | 每例 6 变体 279–384 s（中位 322 s），远低于 25 min 阈值 → rot 开销可忽略，无需 upstream opt-in 补丁 |
| **G2 产率** | 探针 15/15 trans 可行 |
| **G3 构建** | 探针 15/15 aug 任务建成，`yaw=0.00deg`、PRG pair = 18×N 逐例正确 |
| **G4 确定性** | **已做（事后补测，零算力）** —— 24/24 逐字节相同，`max\|Δqpos\| = 0.0`，见 F10 |

> 四个门此前只以散文形式记在本日志里，`p2_gate_decision.json` 从未写过 —— 而
> 计划规定 eval runner 要读 `G4.verdict` 决定 C4 判据粒度。已补
> `build_p2_gate_decision.py`，把四门**全部从 TSV 重新推导**（不转抄），
> 顺带纠出 G1 的一处口径错误（F9）。

### P3 + P4 放量 retarget — **C2 通过，c = 110/110（L0）**

pass1 104/110；pass2 v2 rescue 补齐 6 个真 CVXPY infeasible → **最终 110/110**，99 omnirt_v1 + 11 omnirt_v2（5 源侧 v2 + 6 rescue）。

| 变体 | pass1 可行 |
|---|---|
| trans0 | 20/22 |
| trans1 | 21/22 |
| trans2 | 22/22 |
| **rot0** | **20/22** |
| **rot1** | **21/22** |
| 合计 | 104/110 = 94.5% |

按物体：chair005 5/5、desk023 20/20、desk007 24/25、chair006 23/25、desk021 32/35。

`verify_retarget_artifacts.py`：**124/124** 通过 V1–V5（可加载 / 帧数与 `_original` 一致 / 宽度一致 / 全有限 / 物体通道确有位移）。这是应对 §5 并发事故的收口方式 —— 与其追查是哪个进程写的，不如直接证明产物可用。

### P5 建 aug 任务 / 场景 / override + 快照 + 冻结 — **C3 关闭**

`build_augmented_tasks.py`：**110/110 建成，0 错误**（105 有效 + 5 degenerate）。rot 变体 `yaw=45.00deg` 端到端确认旋转修复生效。

`build_arm_scenes.build_one` 靠 `target_task` 列落到 aug 目录，**E206 脚本零改动**复用成立。

**V5c 场景奇偶（`check_scene_parity_vs_e206.py`）—— 105/105 gated 通过**，chair005 的 5 条单独报告（也全过）。范围不止 `scene.xml`，而是整条 arm 链五个 XML：

| 检查 | 内容 |
|---|---|
| P1 | 掩码物体位姿后与 base **逐字节相等**。`scene_act_E206_lowgeom_PRG.xml`（CEM 真正加载的那份）的奇偶，是「E206 手部 rubber_hull 补丁与 18N pair 算术在 aug 目录里落对了」的最强单点证据 |
| P2 | 物体 body / 6 关节 / 6 执行器数目正确 |
| P3/P4 | 场景里的物体位姿 == **它所声称那条** trimmed npz 的第 0 帧。掩码比对本身看不出「拿错变体」，这条才看得出 |
| P5 | 从 XML 独立重导出位移，与 manifest 的 `approach_trans_offset_m_max` 相符到 **1e-4 m**（正是 XML 的打印精度） |
| P6 | 离线复现运行时那条 fail-closed 断言（见 F8），编译 210 个模型全过 |

有效位移分档：**full 80 / partial 15 / weak 10**。

**V5e override 审计（`build_aug_manifest.py`）—— 105/105 通过**。aug base yaml = E206 S5 base override **只换 `task:` 一行**，其余逐字节继承（reward、ref_fk 接触目标、3cm mask 路径、机位）；PRG override 展开 `e206_common.PRG_OVERRIDES` 的值，不重打字。compose 后与 E206 同 case 的 PRG config 做全键 diff，**实测差异集恰为 `{task}` 一个键** —— trajectory/scene/mask 路径都在运行时按 task 解析，不落进 composed config。据此把白名单从 8 键收紧到 `{task, output_dir}`：「差异 ⊆ 白名单」只在白名单够紧时才是个闸门。

**队列排序**改为 (object × variant) 轮转 + 轮内 (object+variant) 对角线。4 物体 × 5 变体 = 20 格，**前 20 行恰好覆盖全部 20 格各一次**，前 4 行既是 4 个不同物体又是 4 个不同变体。任意时刻中断得到的是均衡设计而非截断样本。（plan238 让 chair005 排每轮第一的规则随它被排除一并删除。）

**A2 重算（`recheck_admission.py`）**：不重做吞吐探针 —— V5c 已证场景只差物体初始位姿，E206 的拟合直接适用，再探一次就会有第二个预算权威。只重算队列算术：plan238 按 66 条（22×3）估，实际 **105 条** → 14 轮，乐观 **11.06 h**，最坏界 **41.35 h < 48 h 门**。`frozen` 块逐字继承，唯独 `queue_priority` 不继承并写明原因（E206 的 desk007 lifeline 服务于它的 C5a，E208 没有这条 claim；chair005 已排除）。

**规则 7 快照**：126 个目录（105 aug + 21 base），HEAD `d481cc7`，manifest sha256 `4c1882e0…9930`。
**冻结**：105 行，`(case_id, variant)` 集合 sha `41a31921…4570`，供 C7(d) 在 P6 后核对行集未变。

### P6 Full CEM — **已启动，进行中**

`run_e208_cem.py` + `scripts/launch/active/run_E208_local_8gpu.sh`，105 条，8 卡独占（启动时 E207/E209 队列已排空，8 卡全空）。

runner = E206 的准入硬门 + E199 的队列模型 + 两次并发事故换来的三样东西：
1. **flock 单实例锁** —— 这个 runner 每次状态变更都整表回写 manifest，第二个实例会静默把第一个的行改回去。`SingleInstance` 提到 `e208_common` 两个驱动共用。
2. **per-task 超时 180 min**（略高于 E206 实测 177.2 min 的 per-task bound）。超时先 SIGTERM、10 s 后 SIGKILL，并在标记状态前**显式 unlink 半写的 npz** —— 否则下次续跑会把它当成已完成，正好掩盖超时要暴露的问题。
3. **派发前校验冻结集**（C7d 提前到派发前而非只在 P8 事后查），收尾再校一次。

刻意**不在命令行传 `task=`**：V5e 审的是 composed override 并已证明 `task` 是唯一差异键，命令行再传一次就多出一个 V5e 管不到的权威；改为在 preflight 断言 override 自己的 `task:` 行与 manifest 行一致。

启动实测：8 卡各 ~50% 利用率、~2.5 GB 显存，`plan time ≈ 22 s/step`、`opt_steps=32`，日志确认加载了 E206 的 3cm mask（chair006 active L/R = 42.9%/48.3%，与已知 `blind3cm=0.429` 吻合）。

**首轮（8 条）结果：8/8 `cem_ok`，零失败**，输出校验（qpos 有限 + `config_act.scene_name` 相符）全过。wall 按物体分层明显：

| 物体 | wall_min |
|---|---|
| desk007 | ~29 |
| desk021 | ~43 |
| chair006 / desk023 | ~51–55 |

12 条完成时中位 **44.9 min**（优于 E206 的 47.4），4 物体 × 5 变体均已有代表 —— 轮转排序按设计生效。ETA 修正为 **≈ 04:00（约 11.7 h）**。

### P7 渲染 — 链路已验证

`render_aug_results.py` 首次实跑 2 条成功。

**视觉复核（规则 9）**，对象是 `desk007_20231030_028_p1__aug_rot1` —— 按 F6，**旋转增强在本仓库历史上从未产出过产物，因此也从未被人看过**：

- ref/sim 在 t = 0 / 1.2 / 1.8 / 3.1 s 四个时刻**全程贴合**，无漂浮、无穿模、无抖动。
- 接触点（橙色）显示手-桌接触在维持；桌-地、脚-地接触位置合理。
- 末帧双手抬到面部的姿态**在 ref 里同样存在** → 属源动作/重定向性质，不是 CEM 控制产物。留给人审记录，不在此下判断。

### P8 评估 — 脚本就位，已有早期信号

`eval/runners/eval_E208_aug.py` + wrapper。三把尺、预注册配对统计（HL / Wilcoxon / n<10 时精确符号检验 / McNemar）、五级分层（pooled / variant / object / object×variant / **offset band**）。

早期信号（n=9，**不作结论**，仅用于尽早发现系统性问题）：

| 指标 | 值 |
|---|---|
| `obj_pos` HL | **+0.061 cm**（门 +5.0） |
| band=full | **−0.348 cm**（aug 反而更好） |
| band=partial | +0.355 cm |
| 通过率 | orig 0.444 → aug 0.333（门 0.15） |

### P9 人审 feed — 就位

`build_aug_review_tsv.py`（**21 case × 6 臂** = orig + trans0/1/2 + rot0/1）、`build_review_coverage.py`（按分层查覆盖 + C5/C6）、`review_index.py` 加性注册 `E208AUG`。实测播放器索引加载 31 条 / 6 臂 / 4 物体 / 全部可播。

feed 形状**不按 plan238**：plan238 想让 `case_id = {orig}__aug_{variant}`，那样播放器里是 105 个互不相关的 case；改成 `case_id` 用 orig、`arm` 用变体，评审时 orig 与它自己的 5 个增强并排 —— 这才是人真正被要求做的判断，也是评审者读 E199/E200 的既有习惯。

---

## 3. 结果路径

| 类型 | 路径 |
|---|---|
| 契约自检 | `workspace/core4d/results/E208/preflight/e208_contract_selfcheck.json` |
| 播种完整性 | `.../preflight/orig_seed_integrity.{tsv,json}`、`orig_seed_report.json` |
| 产物校验 | `.../preflight/retarget_artifact_verify.{tsv,json}` |
| **G1–G4 门决策** | `.../preflight/p2_gate_decision.json` |
| **V5c 场景奇偶** | `.../s5_handoff/scene_parity_vs_e206.{tsv,json}` |
| **V5e override 审计** | `.../s5_handoff/aug_override_audit.json` |
| **优先级 manifest** | `.../s6_downstream/manifests/e208_priority_manifest.tsv`（105 行） |
| **冻结副本 / 冻结元数据** | `.../manifests/e208_priority_manifest.frozen.tsv`、`freeze.json` |
| **权威表（含 orig 配对）** | `.../manifests/e208_aug_authority.tsv`（126 行 = 105 aug + 21 reused_e206） |
| **准入决策** | `.../s6_downstream/cem/throughput/admission_decision.json` |
| **规则 7 快照** | `.../scene_snapshot/`（126 目录 + manifest.txt） |
| **CEM 输出** | `.../s6_downstream/cem/full/E208_{case}_aug_{variant}_PRG/` |
| **CEM 队列/单任务日志** | `logs/E208/queue_*.log`、`logs/E208/cem/full/{variant_id}.log` |
| aug override | `examples/config/override/core4d_{aug_task}.yaml` + `core4d_E208_{case}_aug_{v}_lowgeom_PRG.yaml`（各 105） |
| 可行性 | `.../data_preprocess/manifests/e208_aug_feasibility.tsv`（130 行 = 110 pass1 + 20 pass2） |
| aug 产物 | `.../data_preprocess/manifests/e208_aug_artifacts.tsv`（110 行） |
| case-file | `.../data_preprocess/case_files/cases_e208_{object}_{seq}_{person}.tsv`（22 个） |
| 隔离区 | `.../data_preprocess/quarantine/rot_npz_2026-09-05_concurrent_runners/`（24 个，留证） |
| retarget 中间产物 | `.../data_preprocess/omnirt_v{1,2}/holosoma_{base}/` |
| aug 任务目录 | `example_datasets/processed/core4d/unitree_g1/humanoid_object/*__aug_{trans0,trans1,trans2,rot0,rot1}`（110 个） |
| 运行日志 | `logs/E208/{p2_probe_pass1,p3_pass1_all,p3_pass1_rot_clean,p4_pass2_rescue,p5_build_tasks}.log` |

---

## 4. 发现（F1–F7）

### F1 · `e199_common.OMNIRT_V2_ENV` 缺一个键，照抄会静默偏离契约

它只有 **5 个键**，缺 `REPLACE_WRIST_WITH_FINGERTIP`。`pipeline.sh:28` 该键默认 **1**，而 E206 实录是 **0**。照抄 E199 会静默换成「指尖替代腕部」的 IK 目标、不报错、aug 与 orig 系统性错位。→ A7 改为对着 `run_stage2b_omnirt_v{1,2}_ref_fk.sh` 的 `env` 字面量逐键 diff，六键全显式。

### F2 · `object_name` 大小写陷阱比预估更宽

不只 desk021。实测 **`desk021 → "Desk021"`、`desk023 → "Desk023"` 都是大写**（合计 11/22 例），而 `parallel_robot_retarget.find_files` 用 `f"*{object_name}*.npz"` **大小写敏感**。任何 lowercase 归一化会静默匹配 0 文件。→ A4 强制逐字透传 `task_info.json`。

### F3 · 播种后 convert / trim 都不会跑，mask 因此可完全复用

`pipeline.sh` 三个跳过闸：`converted/{task}.npz` 存在跳 convert（:319）、`trimmed/{task}_original.npz` 存在跳 holosoma trim（:403）、`parallel_robot_retarget.py:300` 按输出文件短路。**推论**：`trim_start` 必然与 E206 相同 → E206 的 3cm mask 逐帧有效 → 驱动改用 `--skip-contact`，mask 完全不重算，override 继续指向 E206 那一份（单一权威，无第二份可漂移）。计划里原写的「播种 mask 副本」不需要。

另经源码确认 `initialize_robot_pose`（`robot_retarget.py:563-569`）在 augmentation 分支 `np.load(original_path)["qpos"]` —— warm-start **确实从磁盘读**，所以每个变体都从播种的 E206 轨迹热启动。这是 P1 设计成立的地基。

### F4 · E206 的 3cm mask 是自描述三时间轴结构

`raw_*`(=untrimmed_frames)、`spider_*`(=trimmed_frames，与 `trajectory_kinematic.npz` 的 qpos 逐帧对齐)、`eval_*`(=trimmed×50/30)，并内嵌 `trim_start / ref_fps / eval_fps / threshold_m`。最初只取 `list(keys)[0]` = raw 轴，帧数对不上 trimmed 让人误以为不对齐。→ R4 改为断言**内嵌 `trim_start` == 播种窗口**（这才是「mask 可复用」的充要条件）+ 三轴分别对齐 + `threshold_m==0.03`。

### F5 · E206 实录的 `HOLOSOMA_DIR` 在本机已不存在

`/root/Workspace/holosoma` 不存在，现行路径是 `/mnt/ali-sh-1/.../holosoma`。`CORE4D_REAL_ROOT` / `SMPLX_MODEL_DIR` 两处写法不同但 **realpath 相同**（同一份数据，无混淆）。因为 convert 与 contact 都被跳过，这些路径只用于 `need_dir` 校验，不影响结果 —— 但 A7 只 diff 6 个 retarget knob 而不 diff 路径，是正确的选择。

### F6 · **【重大】旋转增强从来没跑起来过，E199「系统性不可达」是误判**

- **根因**：holosoma `src/utils.py:346` 的 `R.from_euler("z", rotation_list)`，`rotation_list` 是 `(N,)`，而 **scipy 1.17.1 要求 `(N,1)`**，抛 `ValueError: Expected last dimension of 'angles'...`，**发生在任何 IK 求解之前**，且会终止整个文件的处理（所以 `rot_1` 连尝试都没有过）。
- **证据（E199 全量日志实测）**：`rot_0` 尝试 **97** 次、`rot_1` **0** 次、产出 rot npz **0** 个（对照 trans npz **582** 个）、全实验只有 **1** 条真正的 `[skip] ... infeasible`。
- **实际可行率（E208 修复后实测）**：rot0 20/22、rot1 21/22，与 trans 基本持平。
- **已修**：holosoma `9e544b1`（`rotation_list[:, None]`）。`E208/test_rotation_fix.py` **9/9** 验证语义：起动帧前保持满角、之后按 `rotation_tau=25` 指数衰减、位置不受影响、四元数保持单位模、`rotation_initial=0` 严格 no-op、rot_1 与 rot_0 镜像。
- **附带确认**：`rot_*` **不是纯旋转** —— 每档还带 0.2 m 侧移，且 yaw 用 tau=25 而位移用 tau=50（upstream 设计，未改）。
- **需修正的既有记录**（见 §7 待办）：E199 log286、E202 log291 的 trans-only 理由、`EXPERIMENT_TRACKER.md` 对应行、记忆文件 `omniretarget-object-augmentation`。

### F7 · **增强位移在裁剪窗口开启前就衰减掉了（E199/E202 也踩了但从未报过）**

增强扰动的是**接近段**，从 `object_moving_frame_idx` 起按 `translation_tau=50` 帧指数衰减；SPIDER 只拿到接触裁剪后的窗口，所以 **`trim_start` 越晚，SPIDER 看到的残余位移越小**。

横向实测：

| 实验 | n | min | <0.05 m | <0.10 m | <0.18 m |
|---|---:|---:|---:|---:|---:|
| E199 fullscale | 249 | 0.083 | 0 | 6 | 21 (8.4%) |
| E202 bucket | 73 | 0.074 | 0 | 6 | **24 (33%)** |
| E208 全量 | 110 | **0.023** | 5 | 15 | 30 |

E208 逐物体：

| 物体 | n | min | median | <0.05 |
|---|---:|---:|---:|---:|
| **chair005** | 5 | **0.0226** | 0.0226 | **5** |
| chair006 | 25 | 0.2000 | 0.2000 | 0 |
| desk007 | 25 | 0.1213 | 0.2000 | 0 |
| desk021 | 35 | 0.0935 | 0.2000 | 0 |
| desk023 | 20 | 0.1452 | 0.1905 | 0 |

**chair005 全军覆没**：`trim_start=113`（全场最晚），5 个变体全部只剩 2.3cm 位移，连 rot 档的 yaw 也只剩 **0.58°**（本应 45°，`rotation_tau=25` 衰减更快）。它的 5 条「增强」轨迹与 orig 几乎完全一样。

**结论**：
1. plan238 写的 `approach_trans_offset_m_max ∈ [0.18,0.22]` **作为硬门是错的** —— 会判掉 E202 已交付数据的 33%。
2. 改为：有效位移作为 **C3 一等输出**；只对「根本不算增强」设下限 `EFFECTIVE_AUG_FLOOR_M = 0.05`（低于 E199/E202 交付过的任何值，不追溯否定它们）。
3. **用户决定**：直接**排除 chair005**（105 条，分层从 5 物体降为 4）；有效位移**升为 C4 分层变量**（分档 ≥0.18 / 0.10–0.18 / <0.10），用来回答一个此前无人能答的问题：**增强幅度多大时才开始付出跟踪代价**。

### F8 · rot 变体会重新挑选 euler 约定 —— 不是 bug，但会让「只差物体位姿」的说法字面上不成立

V5c 首次运行时 **11 条 rot** 的 `scene_act*.xml` 在 line ~340 处失配（`scene.xml` 本身是过的）。差异是**物体三个转动关节的顺序**（base `rot_x,rot_z,rot_y` vs aug `rot_z,rot_y,rot_x`），执行器顺序随之变化。

- **机制**：`generate_scene_act.py:55-78` 的 `find_best_euler_convention` 会遍历 6 种 euler 约定，选中间轴角度最大值最小的那个（万向锁裕度）。rot 把物体转了 45°，物体的朝向轨迹变了，最优约定因此可能改变。
- **只发生在 rot**：实测 **rot 11/42，trans 0/63** —— 完全符合「只有改变物体朝向的变体才会重挑」。变更方向 `XZY→ZYX` 3、`ZYX→XZY` 4、`XYZ→ZYX` 1、`XYZ→XZY` 1、`YZX→XZY` 2。
- **为什么安全**，三条独立支撑：
  1. `examples/run_mjwp.py:583` 用 `resolve_scene_act_reference(config.model_path, _m_act)` **从任务目录自己的 `scene_act_meta.json`** 读约定，再据此做 quat→euler（`:620`）。每个 aug 目录有自己正确的 meta。
  2. `resolve_scene_act_reference` 是 **fail-closed** 的：meta 缺失/非法/**与编译后铰链轴序不符**都直接抛错，没有 euler fallback。
  3. **E206 自己交付的 22 个 base 本就横跨 4 种约定**（XZY 16 / ZYX 4 / XYZ 1 / YZX 1）—— 下游不可能假定单一约定，E208 没有引入任何新暴露。
- **强行统一反而更差**：被换掉的恰恰是离万向锁更近的那个约定。
- **处置**：把物体 DOF 顺序并入 V5c 的掩码区（**对顺序不敏感、对集合仍敏感**），改为新增 **P6 在离线复现那条运行时断言** —— 它在运行时会在 CEM 已经派到 GPU 之后才失败，用一次模型编译提前挡掉。210 个模型全过。

> 这条也说明「掩码比对」这类检查的设计要点：掩掉的东西必须**换成一条更强的检查**，否则就是把失败改成了沉默。

### F9 · 两处会让报表算出错数的口径问题（都在派发 CEM 前纠正）

**(a) `wall_s` 是整例耗时，被复制到该例 5 行上。** 一次 `pipeline.sh` 调用产出全部 5 个变体，driver 把同一个 `wall_s` 写到 5 行。按行求和会**多算 5 倍** —— G1 初版因此误报 fail（max 35.7 min > 25 min 门）。改为按例取值 + 断言例内取值一致，实测中位 **3.16 min/例**、最大 **7.14 min**，远在门下（G1 pass）。

**(b) `built` ≠ `adopted`，差 14 行。** `pass2-rescue` 在 v2 树里只有 `_original` 可短路，所以会**重算全部 5 个配置**；20 行标 `rescued_v2`，但真正被采纳的只有 pass1 失败的那 **6** 个。`built=1` 有 **124** 行，实际进入实验的是 **110**。任何人从 `built` 直接算 C2 或 rescue 产率都会拿到错的数。已加 `adopted` 列（派生规则与 `build_augmented_tasks.effective_variants` 同一条），并在 `p2_gate_decision.json` 里同时报两个数。

### F10 · **G4 实测：aug IK 跨进程逐字节确定 —— C4 可以逐 case 判**

事故当时**隔离而非删除**那 24 个 rot npz，在这里付了息：每一个都有干净单实例重跑的对应文件，两者来自同一份播种 `_original`、同一 case file、同一六键 env，但**出自两个不同进程** —— 正是探针 R6 要问的问题。

**结果：24/24 逐字节相同（sha256 相等），`max|Δqpos| = 0.0`。**

- **这个方向的结论是决定性的**：aug IK 跨进程可复现 → per-case 的 aug-vs-orig delta 不含 IK 噪声 → **C4 判据保持逐 case，不触发 plan238 R7b 的分布级降级**。
- **反方向不成立**，脚本与 JSON 里都写明了：若两者不同，无法区分「IK 不确定性」与「当时那次竞争」，只能给出确定性的**上界**。
- **附带结论**：那次并发事故**实际造成零数据损坏**，隔离的 24 个文件本身就是好的。当初「不追查是谁写的、改为证明产物可用」的收口方式（124/124 V1–V5）是对的，而「隔离而非删除」让这个证据在两天后还能用。

### F11 · 「同一批 22 例，三个互不相等的通过率」—— 已变成被测量的数字

三把尺给出三个数，都对，但放在一起会让人以为有矛盾：

| 口径 | 结果 |
|---|---|
| E206 人审 | **22/22 USE** |
| E206 发布的 **12 门** `numeric_release_pass` | **18/65** PRG 行 |
| 增强线的 **6 门**（E199/E202/E208） | **29/65**；本 registry 的 21 例中 **15/21** |

根因不是打分不一致，而是**阈值不同**：E206 的 `lower_body` 门在 `leg_penetration_frac ≈ 0.20` 处放行，增强线用 **0.10**（更严）。实测 21 例里恰好 **1 例**（`desk007_20231030_028_p2`，0.1017）因此判定不同。

C4 本身干净 —— 同一套 6 门阈值同时施加于 orig 与 aug。但差异必须显式记录：已加 `gate_criterion_vs_e206` 交叉核对，**自动列出**每一处判定分歧，而不是留给读者去猜为什么数字对不上。播放器 feed 里同时带两个数（`numeric_release_pass` = 12 门、`c4_all_gates_pass` = 6 门）。

### F12 · 三个不会报错的 join/序列化坑（都在 P9 抓到）

这三个共同点是：**不抛异常、不留日志，只让结论悄悄变错**。

1. **E206 人审表的主键是复合键 `{case_id}#PRG`**，不是裸 `case_id`。直接 join 得到**全空 prefill** —— 评审表看起来完全正常，只是每一行都少了「这条 demo 增强前被判成什么」。修后 orig USE 率 21/21 = 1.00，C5 门线落在 0.85（与计划一致）；修之前它是 0.00，会把 C5 判成必然失败。
2. **`--no-rescore-orig` 路径下 orig 行只拷了 `KEY_METRICS`**，缺 12 门视图需要的 `hand_object_release_false_contact_3mm_frac` 等列 → orig 被判了本不该有的 `release` 失败，人审的 side-by-side 对 orig 不公平。改为拷贝全部已发布数值列。
3. **`write_tsv` 把布尔序列化成小写 `true`/`false`**，而 `gate_pass()` 输出 `str(bool)` 的 `True`/`False`。按字面量 `"True"` 统计恒为 0 —— 一度报「12 门与 6 门都 0 通过」。改用 `C.truth`。

另有一个同类问题在命名层：评估器与评审 feed 原本都写 `e208_aug_case_metrics.tsv`，后跑的会静默弄坏另一个的消费者。按 E206 既有约定分工（eval 写 `*_rollout.tsv`、评审 feed 写 `*_case_metrics.tsv`）。

---

## 5. 事故：并发跑了两个 retarget 实例（两次）

### 经过

| 次 | 起因 |
|---|---|
| 1 | 等待循环写成**固定次数 sleep**（最多 9 min）而非等进程真正退出，误判 P3 已结束就启动了第二个实例 |
| 2 | 用 `pgrep -f ... \| head -1` 抓 PID，抓到的是 **bash 包装**而不是真 runner，于是又误判 |

两次都还叠加了一个坑：**`kill -0` 对僵尸进程也返回成功**。P3 结束后因 nohup 的父 shell 已退出而变成 `Z` 态，等待循环因此永远不退出。

### 后果

两实例在同物体的 `sync_generated_object_model`（`cp -a`）与 `ensure_g1_object_xml`（首写）上竞争 —— 正是 `run_upstream_retarget.py` 按物体分组要规避的那个 hazard。**分组只防进程内竞争，防不了进程间。**

### 损害范围（按 mtime 实证，非假设）

- `_original`：**未受影响**（上游按文件短路，从不重写）→ C1 成立。
- `*_trans_*`：**未受影响**（4 个探针 case 的 trans mtime 全在 00:51–01:21，早于第二实例的 01:40）→ 63/66 trans 成立。
- `*_rot_*`：来源不确定（01:31–01:56 跨越重叠窗口）。

### 处置

1. `quarantine_rot_npz.py` 把 24 个 rot npz **隔离而非删除**（留证；日后若测出 aug IK 不确定性可与干净重跑做 diff）。
2. 单实例重算 rot。
3. `verify_retarget_artifacts.py` —— **不依赖 provenance 的产物校验**，124/124 通过。这是本次事故的正确收口方式。
4. **加 `flock` 单实例锁**（`SingleInstance`），`run_upstream_retarget.py` 与 `build_augmented_tasks.py` 都接入，已验证第二实例被正确拒绝。用锁替代判断力。

### 教训

- 等待外部进程要检查**进程状态**（排除 `Z`），不能只看存在性；更不能用固定次数 sleep。
- `pgrep | head -1` 不可靠，会命中 shell 包装。
- 真正的解法不是「更小心」，而是让并发在机制上不可能 —— 加锁。

### 其它踩坑

- **P4 第一次疑似静默死亡**：日志只有表头、per-case 日志停在进度条 18% 且无 traceback。事后判明是**我抓错 PID 导致的误判**（进程其实还活着），并非真的被杀。
- **`kill` 被安全分类器拦截**：清理孤儿进程时 `kill` / `pkill` / `find -delete` 多次被拦，需要用户手工执行或改用 python 脚本（隔离而非删除，反而是更好的做法）。
- **并发实验**：同分支上 E207（R293）、E209（R295）由其它会话并行推进，`EXPERIMENT_TRACKER.md` 与 GPU 都是共享资源。E208 提交时只 stage 自己的路径，从未 stage TRACKER。
- **第三次资源碰撞：log 号**。并发的 E207 会话在 01:24 就提交了 `log/296`，我 15:16 才提交同号文件，A9 因此报冲突。按 plan239:40 的裁定让号到 **297**，并把 `LOG_SLOT`/`PLAN_SLOT` 提为契约常量、A9 改读常量（原先是测试里的硬编码 glob）。前两次碰撞是 GPU 与 retarget 进程，这次是编号空间 —— 同分支多会话并行的成本是持续的，不是一次性的。

---

## 6. Claims 现状

| # | 判据 | 状态 |
|---|---|---|
| C0 契约闭合 | A1–A10 全过 | ✅ **10/10** |
| C1 orig 播种完整性 | R1–R4 在 22/22 全过 | ✅ **22/22** |
| C2 aug 可行率 | `c ≥ 22` 且 5/5 物体 `c_obj > 0` | ✅ **110/110（L0）** |
| C3 aug 构建正确性 | 位移/yaw/pair 数/V5c/V5e | ✅ **关闭**：110/110 建成、pair 数与 yaw 逐例正确、**V5c 105/105**、**V5e 105/105** |
| C3b rot 可行率 | 报可行率 | ✅ 已升为一等结果（见 F6），不再是「免费副产品」 |
| C4 aug-vs-orig 数值不劣 | McNemar ≤0.15、obj_pos HL ≤ +5cm | 🟡 CEM 进行中；**判据粒度已定为逐 case**（G4 见 F10） |
| C5 人审 USE 率 | ≥0.85 | 🟡 feed 就位（21×6 臂，播放器索引已验证）；门线由 orig 21/21=1.00 定为 **0.85**；未开审 |
| C6 视觉无度量欺骗 | ≤0.20 | 🟡 计算脚本就位；抽样视觉已做 1 条 rot（无度量欺骗迹象） |
| C7 复现性 / 快照 | (a) 快照 (b) V6d (c) rescore-orig (d) 冻结行集 | 🟡 (a) 完成（126 目录）、(d) 已冻结且**派发前已校验一次**；(b)(c) 未做 |

**G1–G4 全部通过**（`preflight/p2_gate_decision.json`，从 TSV 派生非转抄）：G1 中位 3.16 min/例（门 25）、G2 采纳 110/110 → L0、G3 trans yaw 恰 0.0°/rot yaw 0.575–45.0°/pair 数逐例正确、G4 逐字节确定。

---

## 7. 未做 / 待办

### 进行中

- **P6 Full CEM**：105 条已在 8 卡跑，预计 ~12.8 h。收尾需核对：105 条 `cem_ok`、冻结行集 sha 未变（C7d）、`wall_min` 分布写进本 log。

### 后续阶段

- **P7** 链路已验证（2 条）。队列跑完后需 `--watch` 补齐 105 条 MP4。
- **P8** runner + wrapper 已就位并跑通。**仍缺**：`gen_E208_aug_workbook.py`（xlsx，n<3 强制 indicative）、`audit_runtime_config_vs_e206.py`（V6d）；以及队列跑完后开 `--rescore-orig` 的正式一趟（C7c）。
- **P9** feed / 覆盖率 / 播放器注册全部就位并验证。**仍缺**：105 条全量人审本身。
- **P10** 收口：本日志补完、`EXPERIMENT_TRACKER.md` 加行、`build_log_index.py`、plan238 落盘。

### 遗留的证据缺口

- ~~**G4 / 探针 R6**~~ —— **已补测**，见 F10：24/24 逐字节相同，C4 保持逐 case 判据。
- **R5 未做**：F15 两个发散例的确认性重跑量化（非门，但计划里承诺写进 log）。注意 F10 已从侧面给出相关证据 —— aug IK 跨进程逐字节确定，说明 F15 的发散不来自求解器本身的随机性。
- **plan238 未落盘**：计划仍在 `/root/.cc-mirror/.../plans/` 下，未按规范写入 `workspace/core4d/plan/238_*.md`。落盘时需订正 **6 处**已被实际执行推翻的内容：5 变体（非 trans-only）、排除 chair005（105 非 66）、offset 分层、`[0.18,0.22]` 硬门作废（F7）、V5c 需容纳 euler 约定重选（F8）、C4 粒度已由 G4 判定为逐 case。
- **规则 7 的一处结构性限制**：`workspace/core4d/results` 是指向外部存储的 symlink，快照**无法进 git**（R11，E199/E202/E206 同）。可核对的部分是把 manifest sha256 记进本 log（在 git 里）：`4c1882e0…9930`。这不是 E208 引入的问题，但也不应被当作「规则 7 已满足」。

### 需要修正的既有记录（因 F6）

| 位置 | 内容 |
|---|---|
| `log/286_E199_*.md` | 「rot 全不可达」「旋转档位系统性不可达」 |
| `log/291_E202_*.md` | 继承的 trans-only 理由 |
| `EXPERIMENT_TRACKER.md` | E199 行的同一表述 |
| 记忆 `omniretarget-object-augmentation` | 「±45° yaw still infeasible」 |

> 按规范「不修改已完成实验的 log」，修正应在本 log 中说明并在 TRACKER 加注，而非改写 log286/291 原文。

---

## 8. 改动文件

### 新建 `workspace/core4d/scripts/experiments/E208/`

| 文件 | 用途 |
|---|---|
| `e208_common.py` | 契约：registry / `OMNIRT_V{1,2}_ENV` / v1-aware `aug_task_name` / rescue 状态机 / PRG-only 命名 / `load_e208_module` / `BUILD_VARIANTS` / 比例化 L 阶梯 / 有效增强下限 |
| `test_e208_contract.py` | A1–A10 离线自检 |
| `seed_from_e206.py` | P1 播种 |
| `check_orig_seed_integrity.py` | R1–R4（`--probe-rerun` 预留 R5/R6） |
| `run_upstream_retarget.py` | pipeline.sh 驱动，`--pass pass1\|pass2-rescue`，按物体分组、sentinel、双信号可行性检测、`SingleInstance` 锁 |
| `build_augmented_tasks.py` | fixed_window_trim + SPIDER 任务 + `ARM.build_one` + pose_diff + `effective_aug` 判定 |
| `verify_retarget_artifacts.py` | 不依赖 provenance 的产物校验 V1–V5 |
| `quarantine_rot_npz.py` | 事故处置：隔离而非删除 |
| `test_rotation_fix.py` | holosoma 旋转修复的语义验证（须用 hsretargeting conda python 跑） |
| `check_scene_parity_vs_e206.py` | **V5c**：整条 arm 链五个 XML 的掩码奇偶 + 场景↔npz 位姿绑定 + 离线复现运行时 scene-act 契约 |
| `build_aug_manifest.py` | **V5e**：aug base yaml + PRG override + compose 差集审计 + 分层轮转队列 + 冻结 + `reused_e206` 权威表 |
| `recheck_admission.py` | 继承 E206 frozen，只重算 A2 队列算术（105 条） |
| `build_p2_gate_decision.py` | G1–G4 从 TSV 派生；G4/R6 用隔离区做零算力确定性实测 |
| `run_e208_cem.py` | Full CEM 队列：准入硬门 + 显存准入 + sha 预检 + 输出校验 + 超时 + 冻结集校验 + flock |
| `render_aug_results.py` | P7：读 manifest 只渲 `cem_ok`，`--watch` 与 P6 重叠，osmesa 不抢 GPU |

### 新建 launch

| 文件 | 用途 |
|---|---|
| `workspace/core4d/scripts/launch/active/run_E208_local_8gpu.sh` | P6 唯一入口（`DRY_RUN` / `GPUS` / `CASES` / `VARIANTS` / `LIMIT` / `E208_FORCE`） |

### 修改（跨仓）

| 仓 | 文件 | 改动 |
|---|---|---|
| holosoma | `src/holosoma_retargeting/holosoma_retargeting/src/utils.py:346` | `R.from_euler("z", rotation_list)` → `rotation_list[:, None]`（提交 `9e544b1`） |

### 明确零改动

`E206/{e206_common,build_arm_scenes,build_overrides,run_e206_cem,render_cem_results,build_arm_review_tsv}.py`、`E199/e199_common.py`、`E202/*`、`data_preprocess/pipeline.sh`、`eval/core/*`、`scripts/experiments/E207/*` 与 `E209` 相关文件（并行实验，绝不触碰）。

### 相关提交

| 提交 | 内容 |
|---|---|
| `b720ac2` | E208 P0 — 契约就绪，A1–A10 10/10 |
| `0138792` | E208 P1 — 逐字节播种 orig，C1 22/22 |
| `fb9928d` | E208 P2/P3 — 旋转修复 + 扩到 5 变体，trans 63/66 |
| `dfc1581` | E208 P3/P4 收口 — 110/110（L0），旋转可行率 ~93% |
| `9b954e1` | log 阶段性记录（P0–P5 部分） |
| `d481cc7` | P5 — V5c 场景奇偶 105/105 + log 让号 296→297 |
| `3dd5e8f` | P5 收口 — V5e 105/105、A2 重算、快照 + manifest 冻结，C3 关闭 |
| `c30ab89` | G4 确定性实测（24/24 逐字节相同），G1–G4 门文件从数据派生 |
| `7840258` | P6 — 105 条 Full CEM 队列已在 8 卡启动 |
| holosoma `9e544b1` | 旋转增强 scipy 形状修复 |
