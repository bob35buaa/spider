# plan247 · E215：bucket + box 的 rot 版 object-augmentation 放量 + full CEM

_Core4D · 承接 E199(打通 aug)/ E202(bucket trans 放量)/ E208(修复 rot bug、desk/chair rot 打通)/ E210(bucket007 PRG+G1)_

## 0. 一句话

对 **7 个 bucket + 31 个 box 共 38 个 case**，用 E208 已修复的上游 rot 增强(`rot0/rot1` = ±45° yaw + 0.2m 侧移)生成 object-augmentation 变体 → SPIDER task → 各自臂的 CEM override → **full CEM + eval**。共 **38 × 2 = 76 条 rot aug full CEM**(不含 orig 复跑与 RL 导出)。按物体类分 5 组臂：bucket003 用 **E178 PRG**、bucket007 用 **E210 PRG+G1**、box021 用 **PRG(rubberHull)**、box023 用 **noPRG**、box001/004/024 用 **G1A2**。

## 1. Context / 背景

- 现有 object-augmentation 覆盖**只有平移(trans0/1/2)**。E199 曾记「rot 全不可达」，但 E208 查明那是上游 bug——holosoma `src/utils.py` 的 `R.from_euler("z", rotation_list)` 因 scipy 1.17.1 要求 `(N,1)` 而在**任何 IK 之前**抛错并终止整文件(E199 全量日志:rot_0 尝试 97 次、rot_1 0 次、产 rot npz 0)。**已修**(holosoma `9e544b1`，`rotation_list[:, None]`，见 `../holosoma/.../src/utils.py:354`)。E208 修复后 desk/chair 实测 rot0 20/22、rot1 21/22、CEM wall 与 trans 无法区分。**故本实验目标就是把已验证可行的 rot 增强放量到 bucket/box，补齐旋转方向的下游 RL 数据。**
- **rot 不是纯旋转**:每档叠 0.2m 侧移，yaw 用 `rotation_tau=25`、位移用 `translation_tau=50` 指数衰减回原轨迹(上游设计，逐字节沿用，不改数值)。
- 全链路已在 E199/E202/E208 打通并复用:上游 OmniRetarget(`omnirt_v2`)→ trim → SPIDER task(scene→trajectory→scene_act→臂 sidecar)→ CEM override → full CEM。本实验**主要是换 case 集 + 换臂 builder**，不新写核心逻辑。

## 2. 冻结不变量(与 E199/E202/E208 full-CEM 对齐)

| 项 | 冻结值 |
|---|---|
| CEM | `seed=0`、`num_samples=1024`、`max_num_iterations=32`、`+use_torch_compile=false` |
| retarget variant | `omnirt_v2/ref_fk`(Phase-4 松弛;aug task 目录统一 v2 标签，即使 base 是 v1) |
| 上游增强 config | holosoma 原生 `rot_0={trans[0,+0.2,0], yaw +π/4}`、`rot_1={trans[0,-0.2,0], yaw -π/4}`(`parallel_robot_retarget.py:103-135`)，逐字节沿用 |
| holosoma 仓库 | **`../holosoma/`**(含 rot 修复);**禁用** `Opensource_projects/holosoma/`(旧、未修)。经 `HOLOSOMA_REPO` env / `e173_common.py:48` 默认已指向 `../holosoma` |
| 接触掩码 | 复用各 case `_original` 的 `raw_contact_mask_3cm.npz`(时间维，对刚性平移/旋转不变) |
| 参考系 | 各 rot 变体参考 = 其自身增强后轨迹 FK(`contact_hdmi_target_source: ref_fk`) |
| evaluator | 公共 `eval.core.core_metrics`，`core4d-e154-physics-contact-v1` |

**单变量纪律**:同一 case 内 `orig`/`trans*`/`rot*` 唯一差异 = 上游增强 config;CEM 参数、臂、reward override、evaluator 逐字段相同。**例外(rot 特有，E208 F8)**:rot 把物体转 45°，`generate_scene_act.py::find_best_euler_convention` 可能为该 task 重挑 euler 约定(实测 E208 rot 11/42 重挑)→ scene_act 的三个转动关节顺序会变。这是预期，非 bug。单变量审计白名单对 rot 需**额外允许 euler 约定/hinge 顺序键**，reward/gate/CEM 仍须逐字节相同。

## 3. 执行范围(38 case × rot0/rot1，按臂分组)

### 3.1 臂→builder 映射(Part B 已定位)

| 组 | 物体 | 臂 | override token | scene builder / common | override 样例 |
|---|---|---|---|---|---|
| G1 | bucket003 | **PRG(E178 bucketAlignedTop)** | `_aug_rot{0,1}_PRG` | `E202/e202_common.py::build_prg_scene()`，scene `scene_act_E202_bucketAlignedTop_PRG` | `core4d_E202_bucket003_..._aug_trans0_PRG.yaml` |
| G2 | bucket007 | **PRG+G1(E210)** | `_aug_rot{0,1}_PRG_gravcomp` | E202 PRG scene + `E210/build_gravcomp_sidecars.py`(object `gravcomp` 0→1)+`assert_gravcomp_diff` | `core4d_E210_bucket007_..._aug_trans0_PRG_gravcomp.yaml` |
| G3 | box021 | **PRG(rubberHull)** | `_aug_rot{0,1}_PRG` | E199/E200 box PRG，scene `scene_act_E199_rubberHull_PRG` | `core4d_E199_box021_..._aug_trans0_PRG.yaml` |
| G4 | box023 | **noPRG(E167A)** | `_aug_rot{0,1}_noPRG` | `E200/e200_common.py` arm `noprg`，scene `scene_act_E199_rubberHull`(无 PRG 门) | `core4d_E189_box023_..._E167A_noPRG.yaml`(参照命名) |
| G5 | box001/004/024 | **G1A2(E198/E205)** | `_aug_rot{0,1}_G1A2` | `E198/e198_common.py` arm `G1A2`(g1_scene + `gravcomp=1.0`) | `core4d_E205_box001_..._G1A2.yaml`(参照命名) |

> 注:bucket 用 `bucketAlignedTop` 代理(E178/E202)，box 用 `rubberHull` 代理(E199/E200)——两类代理不同，勿混用。

### 3.2 case 清单与 base 状态(Part A)

- **Task 1 bucket(7)**:bucket003 → `20231018_001_p1`、`20231018_005_p2`、`20231020_068_p1`;bucket007 → `20231003_2_021_p1`、`20231023_073_p1`、`20231023_075_p1`、`20231023_075_p2`。**全部有 base 任务**(v1)，可直接建 rot。
- **Task 2 box(31)**:box001(6)、box021(11)、box004(3)、box024(4)、box023(7)。**29 个有 base**(v1/v2 混)，可直接建 rot。
- **⚠️ 2 个 box 无 base，需 Stage 0 先建**:`box021_20231011_035_p1`、`box021_20231018_029_p2`(仅有 E203 PRG override 指向不存在的 v1 base)。需先跑该 (case,person) 的 dcv3 s3 base 重定向(`omnirt_v2`)产出 `_original`，再建 rot。若 Stage 0 base 建不出(数据缺失/不可行)→ 该 case 登记排除并告知，不阻塞其余 36 个。

### 3.3 baseline(单变量对照)

rot 的对照 = **同 case、同臂的 orig(与 trans) CEM**。Preflight 审计每 case 同臂 orig rollout 是否已落盘(bucket003=E202/E178;bucket007=E210;box021=E199/E202;box023=E200 noPRG;box001/004/024=E200 G1A2)。**默认复用已有同臂 orig/trans 作 baseline**;仅当同臂 orig baseline 缺失时，补跑该 case 的 `orig×臂` CEM(预计集中在部分 bucket007 PRG+G1 与两个 Stage0 case)。缺失清单在 preflight 产出后**先报给用户**再决定补跑，不静默扩量。

## 4. 需要修改/新增的文件

以 **E208 管线为蓝本**(`workspace/core4d/scripts/experiments/E208/`)，新建 `E215/`:

| # | 文件 | 改动 |
|---|------|------|
| 1 | `workspace/core4d/scripts/experiments/E215/e215_common.py` | 契约:38-case registry(每行 pin base task + object class + 臂组)、`BUILD_VARIANTS=[rot0,rot1]`(复用 `E199.rot` 名映射 `e199_common.py:98`)、`RETARGET_VARIANT=omnirt_v2`、`HOLOSOMA_REPO=../holosoma`、按物体类分派臂 builder(import E202/E200/E198/E210 common) |
| 2 | `E215/run_upstream_retarget.py` | 复用 E208 版:对每 case 跑 `pipeline.sh RETARGET_AUGMENTATION=1 --skip-spider` + omnirt_v2 env，只产 rot0/rot1 npz(warm-start 需先有 `_original`);指向 bucket/box base task |
| 3 | `E215/build_augmented_tasks.py` | 复用 E208 版:trim rot npz 到 `_original` 窗口 → 建 SPIDER task(`{base}__aug_rot{k}`);**新增臂分派**:按 case 的臂组调对应 scene builder(G1 E202 bucketAlignedTop / G2 +gravcomp sidecar / G3 E199 rubberHull PRG / G4 noPRG / G5 G1A2);euler 约定 re-pick 交给 `generate_scene_act.py` |
| 4 | `E215/build_gravcomp_sidecars.py`(仅 G2) | 复用 `E210/build_gravcomp_sidecars.py`:对 bucket007 rot task 写 object `gravcomp` sidecar，过 `assert_gravcomp_diff` + 篡改反测 |
| 5 | `E215/build_aug_manifest.py` | 生成 76(+补跑) 行 full-CEM manifest(tier/override_id/task/scene/trajectory/mask/sha256/cem_*/arm/gravcomp);复用 E208 结构 |
| 6 | `examples/config/override/core4d_dcv3_omnirt_v2_ref_fk_{case}__aug_rot{0,1}.yaml` ×76 | 自动生成的 base task yaml(swap `task:`/mask 指回 orig mask)，`git add -f` |
| 7 | `examples/config/override/core4d_E215_{case}_aug_rot{0,1}_{PRG\|PRG_gravcomp\|noPRG\|G1A2}.yaml` ×76 | full-CEM override(复用对应臂 override，换 defaults+task) |
| 8 | `E215/test_rotation_fix.py` | 复用 E208 版:在 `hsretargeting` python 下验证 `../holosoma` 的 rot 修复语义(9/9)——**首步运行，确认用的是修复版仓库** |
| 9 | `E215/quarantine_rot_npz.py` | 复用 E208:并发跑 upstream 时隔离 provenance 不明的 rot npz(留证不删) |
| 10 | `workspace/core4d/scripts/train/train_E215.sh` | 数据构建入口:**首步 snapshot_scenes.sh** → (Stage0 建 2 个缺失 base)→ test_rotation_fix → upstream rot 重定向 → build tasks → override/manifest |
| 11 | `workspace/core4d/scripts/launch/active/run_E215_local_8gpu.sh` + `pull_E215_remote_results.sh` | 8 卡 priority 队列 + 远程 2 卡 pull(复用 `run_local_priority_queue.py`) |
| 12 | `E215/preflight_baseline_audit.py` | 逐 case 检查同臂 orig baseline rollout 存在性，产缺失清单 |
| 13 | `workspace/core4d/scripts/eval/wrappers/eval_E215_rot_augmentation.sh` + `runners/eval_E215_rot_augmentation.py` | 公共 `eval.core.core_metrics` 打分，按 (object, 臂, variant) 聚合;禁止 importlib 动态加载他实验 evaluator(规则 13) |
| 14 | `workspace/core4d/results/E215/scene_snapshot/` + `manifest.txt` | 76 条 scene XML + `scene_act_meta.json` 快照(git HEAD + sha256)，训练前生成(规则 7/10b) |

**优先复用生成器**:能用 `scripts/gen_experiment.py --exp-id E215` 出 train/run/pull/eval wrapper 骨架就用(规则 15)，manifest/build 逻辑手写。

## 5. 关键前置(reproducibility，规则 7/10b/13)

1. **rot 修复自检**:`test_rotation_fix.py` 9/9 通过，确认 `../holosoma` 是修复版(`src/utils.py:354` 含 `[:, None]`)。
2. **上游产物校验**:每 case rot0/rot1 npz 齐全;抽查物体接近段 yaw≈45°、终点 yaw 按 tau=25 衰减趋同 orig(注意 `trim_start` 过晚的 case yaw 会被衰减到很小——E208 chair005 只剩 0.58°，需在 preflight 标出**有效 yaw < 阈**的退化 case)。
3. **SHA parity + 单变量审计**:76 变体 trajectory/scene_act/override/mask 逐 case sha256 入 `e215_aug_authority.tsv`;`orig`↔`rot*` CEM config 全键 diff，rot 白名单 = `{task, output_dir, euler约定/hinge键}`，reward/gate/CEM 逐字节相同。
4. **G2 gravcomp sidecar 自测**:`assert_gravcomp_diff` + 篡改样本反向自测(同 E210)。
5. **scene 快照**:76 条 scene XML + meta 快照到 `results/E215/scene_snapshot/`。
6. **base 激活入 git**:所有新 rot task scene(`scene.xml`/`scene_act.xml`/`scene_act_meta.json`/`task_info.json`)`git add -f`;Stage0 的 2 个新 base 亦然。

## 6. Claims / 可验证声明

| Claim | 判据(量化，无宽松阈值) |
|---|---|
| **C0 链路** | 36 有 base 的 case(+Stage0 成功者)各产 rot0/rot1 SPIDER task + override;base yaml + arm override + manifest 全生成;SHA parity 全过 |
| **C1 单变量** | 同 case orig↔rot* 的 CEM/reward/gate/evaluator 逐字段相同(rot 白名单外 0 违例);审计脚本 0 违例 |
| **C2 执行闭合** | 全部 rot CEM 完成、打分成功;error/non-finite/diverged = 0 |
| **C3 上游 rot 正确性** | 每 rot 变体:接近段 yaw ≥30°(退化 case 单列)、终点 pose 相对 orig 偏差 ≤ 接近段偏差 20%(指数衰减锚定) |
| **C4 物理可信度** | rot 变体 6 项核心指标(body/obj-pos/obj-ori/obj-z/in-mask-contact/物理安全)**mean+std+worst** 相对同 case+同臂 orig 无显著劣化:obj-pos/ori mean 增幅 ≤25% 且无新增 fall/diverged。**报告全量分布，不 cherry-pick**;rot 的 obj-ori 误差抬升是旋转固有代价(E208:rot0 +1.1°/rot1 +2.7°)，单列不计入劣化判定但如实报告 |
| **C5 数据增益** | rot 变体使初始 obj **yaw 分布跨度**相对 orig+trans 显著扩大(定量 yaw spread) |
| **C6 可视化门(强制，规则 9)** | 每臂组至少渲染 orig + rot0/rot1 各 1 例 rollout 视频，`/video-frames` 抽 grasp/接触/终点关键帧，逐例填「实际观察」:穿模/漂浮/抖动/不自然姿态;**同视频内 sim-vs-ref 比对**(E210 F3:跨视频 auto-camera 机位不同不可比) |

## 7. 成功标准 & 判定

- **链路**:≥95% 有 base case 产出 rot0/rot1 task + full CEM 完成。
- **物理**:≥80% rot 变体满足 C4 且视觉无致命 artifact(穿模/漂浮/整体摔倒)→ rot 增强可用于下游扩数据。
- **诚实报告**:分臂组报 mean/std/worst + 通过率;退化(yaw 衰减过小)case 单列;G2(PRG+G1)延续 E210「PARTIAL」预期，逐 case 判可用性而非整批出片。
- 未达标 → 定位失败模式(哪个物体类/臂/yaw 档破坏物理)，收紧幅度或换臂再议。

## 8. 执行与调度(规则 8b)

- 本机 8 卡 priority 队列(`run_local_priority_queue.py`，共跑不抢占) + 远程 2 卡并行(`ssh spider-remote` + tmux)。tier:P0 每物体首个 rot0(先出可行性) → P1 其余 rot0 → P2 rot1。
- 每 2 次工具操作更新 `progress.md`(规则 3);错误入 progress「遇到的错误」表(规则 6)。
- 完成后:log(`log/{NN}_E215_bucket_box_rot_augmentation.md`)+ 更新 `EXPERIMENT_TRACKER.md`(R302 一行摘要 + log 链接)+ `build_log_index.py`。
- Claims 全过 → commit + push:`exp(core4d): R302 E215 bucket+box rot-aug full CEM — {结论}`。分支沿用当前 `experiment/E199-omniretarget-object-augmentation`(同 aug 大方向，无需新分支)。

## 9. 主要风险

| 风险 | 缓解 |
|---|---|
| bucket/box 比 desk/chair 更易因 45° yaw 出可达域/穿透 → rot 实际可行率低 | 幅度沿用 E208(用户定);preflight 报可行率;不可行 case 如实登记，不强行补 |
| G2(bucket007 PRG+G1)延续 E210 姿态崩溃/接触塌陷两形态代价 | eval 同时报 eef_ori 与 contact(E210 F2:单指标必漏一种);逐 case 判 |
| 2 个 box021 无 base | Stage0 先建;失败则排除并告知 |
| 并发跑 upstream 致 rot npz provenance 混淆 | `quarantine_rot_npz.py` 隔离留证;单实例重算 |
| euler 约定 re-pick 破坏「只差 task」审计 | rot 白名单显式允许 euler/hinge 键(§2 例外) |
