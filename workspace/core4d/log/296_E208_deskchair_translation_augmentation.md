# log296 · E208：desk/chair 物体增强 → PRG Full CEM

**日期**: 2026-09-05（进行中，P0–P5 部分完成）
**实验域**: `core4d`
**Run**: **R294**
**分支**: `feat/E207-bucket-g1only-gravcomp`（用户指定不新建分支；E208 全部改动都在新路径下，与并行推进的 E207/E209 零文件冲突）
**对应 Plan**: `/root/.cc-mirror/codewiz-cc/config/plans/e206-desk-chair-e207-aug-spider-agile-puddle.md`（**尚未落盘到 `workspace/core4d/plan/238_*.md`，见 §7 待办**）
**前置**: [log295 · E206](295_E206_desk_chair_move2_dcv3_noprg_prg.md) · [log286 · E199](286_E199_omniretarget_object_augmentation_results.md) · [log291 · E202](291_E202_bucket_e178_translation_augmentation.md)

> **编号说明**：用户原话是 "E207"，但 E207 已被 plan237（bucket G1-only gravcomp, R293）占用且当时正在跑，经用户确认改用 **E208 / plan238 / log296 / R294**。

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
| **G4 确定性** | **未做**（见 §7） |

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

### P5 建 aug 任务（部分）

`build_augmented_tasks.py`：**110/110 建成，0 错误**（105 有效 + 5 degenerate）。rot 变体 `yaw=45.00deg` 端到端确认旋转修复生效。

`build_arm_scenes.build_one` 靠 `target_task` 列落到 aug 目录，**E206 脚本零改动**复用成立。

---

## 3. 结果路径

| 类型 | 路径 |
|---|---|
| 契约自检 | `workspace/core4d/results/E208/preflight/e208_contract_selfcheck.json` |
| 播种完整性 | `.../preflight/orig_seed_integrity.{tsv,json}`、`orig_seed_report.json` |
| 产物校验 | `.../preflight/retarget_artifact_verify.{tsv,json}` |
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

---

## 6. Claims 现状

| # | 判据 | 状态 |
|---|---|---|
| C0 契约闭合 | A1–A10 全过 | ✅ **10/10** |
| C1 orig 播种完整性 | R1–R4 在 22/22 全过 | ✅ **22/22** |
| C2 aug 可行率 | `c ≥ 22` 且 5/5 物体 `c_obj > 0` | ✅ **110/110（L0）** |
| C3 aug 构建正确性 | 位移/yaw/pair 数/V5c/V5e | 🟡 部分：110/110 建成、pair 数逐例正确、yaw 正确；**V5c/V5e 未做** |
| C3b rot 可行率 | 报可行率 | ✅ 已升为一等结果（见 F6），不再是「免费副产品」 |
| C4 aug-vs-orig 数值不劣 | McNemar ≤0.15、obj_pos HL ≤ +5cm | ⬜ 未开始（需 CEM） |
| C5 人审 USE 率 | ≥0.85 | ⬜ 未开始 |
| C6 视觉无度量欺骗 | ≤0.20 | ⬜ 未开始 |
| C7 复现性 / 快照 | 快照 / V6d / rescore-orig / 冻结 | ⬜ 未开始 |

---

## 7. 未做 / 待办

### 立即（P5 收尾，阻塞 CEM）

1. **把 chair005 排除与 offset 分档写进 `e208_common`** —— 这一步在写本日志时被打断，代码**尚未落地**：需要加 `EXCLUDED_OBJECT_KEYS = ("chair005",)`、`is_excluded()`、`OFFSET_BANDS`、`offset_band()`。
2. **`check_scene_parity_vs_e206.py`（V5c）** —— aug 的 `scene.xml` 剥掉 object `pos`/`quat` 后与 base 逐字相等。未写。
3. **`build_aug_manifest.py`（V5e）** —— aug base yaml + `core4d_E208_{case}_aug_{variant}_lowgeom_PRG.yaml` + Hydra compose 差集审计 + 优先级 manifest + `reused_e206` 行。未写。
4. **`recheck_admission.py`** —— 继承 E206 frozen，按 105 条重算 A2 队列算术。未写。
5. **规则 7 快照** —— `snapshot_scenes.sh E208 <105 aug tasks> <21 base tasks>`。未做。
6. **冻结 manifest** —— `e208_priority_manifest.frozen.tsv` + `freeze.json`。未做。

### 后续阶段

- **P6** `run_e208_cem.py` + `run_E208_local_8gpu.sh`：105 条 Full CEM。**⚠️ 与 E207/E209 共用 8 卡，需排队让路**；预计 105/8 = 14 轮 × 47.4 min ≈ 11 h（乐观）。
- **P7** `render_aug_results.py`：105 条 MP4。
- **P8** `eval_E208_aug.py` + wrapper + `gen_E208_aug_workbook.py` + `audit_runtime_config_vs_e206.py`（V6d）。C4 需加 **offset band 分层**。
- **P9** `build_aug_review_tsv.py` + `build_review_coverage.py` + `review_index.py` 注册 `E208AUG`；105 条全量人审。
- **P10** 收口：本日志补完、`EXPERIMENT_TRACKER.md` 加行、`build_log_index.py`。

### 遗留的证据缺口

- **G4 / 探针 R6 未做**：aug IK 自身的确定性从未测量。计划里它决定 C4 是逐 case 判还是只判分布级。**在 C4 之前必须补**，否则 per-case delta 没有误差棒。隔离区的 24 个 rot npz 正可用于这个 diff。
- **R5 未做**：F15 两个发散例的确认性重跑量化（非门，但计划里承诺写进 log）。
- **plan238 未落盘**：计划仍在 `/root/.cc-mirror/.../plans/` 下，未按规范写入 `workspace/core4d/plan/238_E208_deskchair_translation_augmentation_plan.md`，且计划内容已被 3 项用户决定修订（5 变体 / 排除 chair005 / offset 分层），落盘时需一并订正。

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
| holosoma `9e544b1` | 旋转增强 scipy 形状修复 |
