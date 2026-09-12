# E215 · bucket+box rotation object-augmentation 放量 + full CEM

_plan247 / R302 / Phase 73 · 2026-09-12 · 分支 `experiment/E199-omniretarget-object-augmentation`_

## 0. 一句话

把已修复的上游 **rot 增强**(rot0/rot1 = ±45° yaw + 0.2m 侧移)放量到 **7 bucket + 31 box = 38 case**,
按物体分 5 组臂建 SPIDER task + 各臂 CEM scene,3 机 24 卡跑 **70 条 rot full CEM**,与同 case 同臂 orig 配对评估。
**结论:rot 增强逐变体 89.7% 物理可信、位置追踪几乎零代价,可用于下游扩数据**;旋转固有的 obj-ori 抬升、
30 个 v1-orig retarget confound、6 个 yaw 衰减退化档、2 个不可行变体、2 个 mask 长度排除,均如实单列。

## 1. 目的 / 假设

E199/E202 只放量了平移增强,rot 记为"系统性不可行";E208 查明那是上游 scipy shape bug(修于 holosoma `9e544b1`),
rot 从未真正评估过。E208 在 desk/chair 上打通 rot。**E215 假设:rot 增强在 bucket/box 上同样可行且物理可信,
可补齐旋转方向的下游 RL 数据。** 蓝本 = E208 管线,主要换 case 集 + 5 组臂 builder。

## 2. 冻结不变量

- CEM:`seed=0 / num_samples=1024 / max_num_iterations=32 / use_torch_compile=false`。
- retarget:`omnirt_v2/ref_fk` 统一(与 E199/E202 baseline 逐键一致的 5 键 env;wrist=pipeline 默认 1)。
- 上游增强 config:holosoma 原生 `rot_0`(yaw+π/4, 侧移 [0,+0.2,0])、`rot_1`(−π/4, [0,−0.2,0]),逐字节沿用;
  yaw 用 rotation_tau=25、位移 translation_tau=50 指数衰减回原轨迹。
- holosoma 仓库 `../holosoma`(含 rot 修复,`src/utils.py:354` `rotation_list[:, None]`;自检 9/9 PASS)。
- 接触掩码:复用各 case orig 的 `raw_contact_mask_3cm.npz`(时间维,对刚性平移/旋转不变)。
- evaluator:公共 `eval.core.core_metrics`(`core4d-e154-physics-contact-v1`)。

## 3. case 集 & 臂分派

- **权威清单 = `tmp/paper_case_id.txt` 的 box*/bucket* 行**(计划 §3.2 只给数量,Part A 定了 38 但未转录 box case_id;
  已逐 case 核验 TASK_ROOT dcv3 dir + E199/E202 warm-start 树)。
- **base_variant:30 v1 + 6 v2 + 2 MISSING**。6 v2:box001_040_p2/039_p1/108_p1、box021_034_p2、box024_026_p2/027_p2
  (它们只有 v2 有 task_info.json,无歧义)。2 MISSING(无 base/warm-start):box021_035_p1、box021_029_p2 → 排除(未 Stage0)。
- **臂分派(按物体)**:bucket003=E202 PRG(bucketAlignedTop 90 pair);bucket007=E202 PRG+object gravcomp(E210 G1);
  box021=E199 PRG(rubberHull 16 pair);box023=E167A noPRG(rubberHull,无 leg pair/gate);
  box001/004/024=E199 PRG+gravcomp+A2 hand-gate(= E200 prg_g1a2)。box 臂统一走 E199.build_prg_scene + E200 分派。

## 4. 数据构建结果

| 阶段 | 结果 |
|---|---|
| seed warm-start | 36/36(额外 hardlink 已有 trans_* retarget → 上游只算 rot,省 ~60% IK,更贴合"只产 rot npz") |
| 上游 rot retarget(omnirt_v2) | **72/72 全可行(rot_ok, 100%)** — 远好于 plan 对 45° yaw 可达域的担忧 |
| build SPIDER task + arm scene | **70/72 built**(64 正常 + 6 退化-yaw)+ 2 不可行 |
| gravcomp sidecar(G2/G5) | 32 verified(单变量 `assert_gravcomp_diff` + 篡改自测过) |
| override + manifest | 70 行;**单变量 compose 审计 36/36 pass**(rot0-vs-rot1 严格 {task};9 个 box021 rot0-vs-trans0 sibling 亦过) |
| freeze | 70 行,set sha `84745707`,frozen manifest sha `35f02e9e` |

**2 不可行变体**(rot 后初始帧腿-物穿透超硬地板,scene 建不出;plan §9 可达域风险):
`box001_20231020_014_p1/rot1`(ref_first5 −0.0115)、`bucket007_20231023_075_p2/rot0`(−0.0482)。其 sibling 变体成功。

**6 退化-yaw**(trim_start 过晚,yaw 衰减到 <30°,C3 单列):bucket003_068_p1(**6.9°**)、box021_022_p1(21°)、
bucket007_2_021_p1(20.2°),各 rot0/rot1。

## 5. CEM(3 机 24 卡分片)

70 行按 ordinal round-robin 分 3 shard(24/23/23,tier/臂均衡),3 机各 8 卡(本机 shardA + 2 远程 shardB/C),
共享文件存储。`run_e215_cem` 支持 shard 子集校验(shard 行必须 ⊆ frozen 全集)+ per-manifest flock + per-task 180min timeout。
**结果:70/70 `run_complete_pending_eval`,0 失败。** 占卡脚本 `run.py` 跑前 kill、跑后挂回(本机)。

## 6. Eval(rot vs 同 case 同臂 orig,core_metrics 6 门)

orig 基线考古定位(用户决策):box→E198 four-arm cache(box_prg/box_noprg→A0、box_prg_g1a2→G1A2),
bucket→E178 contactAlignedTop。**36/36 定位**,但 **30 个 orig 是 v1 retarget**(与 v2 rot 二阶 confound),6 个干净 v2。

**68 pair(2 mask-length 打分失败,见下),分臂组(rot vs orig):**

| 臂组 | n(case) | obj_pos HL | obj_pos mean 增幅 | obj_ori mean 增幅 | new fall |
|---|---|---|---|---|---|
| box_prg | 18(9) | +0.150cm | +2% | +49% | 0 |
| box_prg_g1a2 | 24(12) | +0.218cm | +4% | +39% | 2 |
| box_noprg | 14(7) | +0.447cm | +4% | +61% | 1 |
| bucket_prg | 6(3) | **−2.133cm** | **−16%** | +53% | 0 |
| bucket_prg_gravcomp | 6(3) | +0.535cm | +7% | +35% | 0 |

- **C0 链路** ✅:retarget 72/72、build 70/72、CEM 70/70、eval 0 error(除 2 mask-length)。
- **C1 单变量** ✅:compose 审计 36/36 pass。
- **C2 执行闭合** ⚠️:CEM 70/70;eval 68/70,2 个 fallback case(box001_014_p1/rot0、bucket007_075_p2/rot1)因
  contact-mask 帧数与 qpos 不符打分失败(mask 取自 dcv3 base,长度与 fixed-window trim 后的 qpos 差 2/13 帧)→ 登记排除。
- **C3 上游 rot 正确性** ✅:64 变体 yaw≈45°;6 退化-yaw 单列。
- **C4 物理可信度**:**obj_pos 追踪代价极小**(臂组 +2~7%,bucket_prg −16% 反而更好),全 ≤25% 阈;
  **obj_ori +35~61% = 旋转 45° 固有代价**(plan 明确单列不计劣化);**逐变体 C4 通过 61/68 = 89.7%(≥80% 达标)**。
- **3 个 new fall**(box004_083_p2 rot0/rot1、box023_021_p2 rot1)**全落在 v1-orig confound case**,fall 差异混入
  v1→v2 retarget 变化,归因不纯,单列(非纯 rot 代价);box004_083_p2 两 rot 皆 fall,是需复核的可疑 case。
- **C5 数据增益** ✅(定性):rot 把初始 obj yaw 从 orig/trans 的 ~0° 扩到 ±45°(64/70 达 45°,median 45°),
  显著扩大旋转方向覆盖。
- **C6 视觉门** ✅(osmesa 软渲染 9 代表案例,sim vs aug-ref 同视频同机位,`render_c6.py`):
  - **健康 rot 变体**(box_prg box021_034_p1 / box_prg_g1a2-v2 box001_040_p2 / bucket_prg_gravcomp bucket007_073_p1):
    sim 精确跟踪 aug-ref,45° 旋转下正常抓取/搬运/按压,**无穿模、无漂浮、无 reward-hacking**。
  - **2 个 fall case 视觉确认真崩溃**(非度量欺骗,与 fall_flag 一致):box004_083_p2 rot0/rot1 上半身前倾扑到箱上(姿态崩溃);
    box023_021_p2 rot1 重度前倾/临界摔(noPRG 无腿保护)。两者均 v1-orig confound case。
  - **退化档** bucket003_068_p1:sim 与 ref 姿态几乎一致(半蹲桶上),物体几乎无转(6.9°≈orig 复制),物理正常——符合 C3 退化定性。
  - 视频:`results/E215/s6_downstream/render/c6/*.mp4`(gitignored)。

## 7. 判定

按 plan 成功标准"≥80% rot 变体满足 C4 且视觉无致命 artifact":**数值侧达标(逐变体 89.7%)+ 视觉侧通过**
(健康档无致命 artifact、sim 精确跟踪 ref;fall/退化档视觉与数值一致,无 reward-hacking)。臂组级聚合 verdict 因
3 个 confound-fall 显示 fail,**正确判据是逐变体 89.7%**。**rot 增强判定为可用于下游扩数据**。
遗留:box004_083_p2(两 rot 皆崩)复核、30 个 v1-confound case 若要干净 C4 需补 v2 orig。

## 8. 主要坑 / 修复

- **build_aug_manifest 空 list 序列化 bug**:空 list 写成 `key:`(YAML→null)而非 `key: []` → rot-vs-trans compose
  报 geom_ids diff。修为 `key: []` 后审计全过。
- **多机分片**:`run_e215_cem` 原用固定 lock + frozen 全集 sha 校验,三机共享存储会互撞;改为 per-manifest lock +
  shard 子集校验(shard ⊆ frozen)。
- **eval baseline 定位**:orig 基线不在 E199/E200 的 aug manifest(那里只有 trans/rot),而在更早的 E198/E178;
  且 E199 fullscale authority 用 person{N}、E198/E178 用 p{N}。考古到正确源后 36/36 定位。
- **2 mask-length 排除**:fallback case(无 trans sibling)的 dcv3-base mask 与 fixed-window qpos 帧数差 2 帧。

## 9. 关键文件

- 脚本:`workspace/core4d/scripts/experiments/E215/`(e215_common / seed_warmstart / run_upstream_retarget /
  build_augmented_tasks / build_gravcomp_sidecars / build_aug_manifest / run_e215_cem / split_manifest_shards /
  preflight_baseline_audit / quarantine_rot_npz / test_rotation_fix);
  `scripts/train/train_E215.sh`;`scripts/launch/active/run_E215_local_8gpu.sh` + `pull_E215_remote_results.sh`;
  `scripts/eval/{wrappers,runners}/eval_E215_rot_augmentation.*`;`split_manifest_shards.py`。
- override:`examples/config/override/core4d_dcv3_omnirt_v2_ref_fk_*__aug_rot{0,1}.yaml`(70 aug base)+
  `core4d_E215_*_aug_rot{0,1}_{PRG,PRG_gravcomp,G1A2}.yaml`(56 arm override;noPRG 无 arm yaml)。
- manifest/freeze:`results/E215/s6_downstream/manifests/e215_priority_manifest{,.frozen}.tsv` + `freeze.json` +
  3 shard;`s5_handoff/aug_override_audit.json`。
- **scene 快照(规则 7/10b)**:`results/E215/scene_snapshot/`(70 aug task 的 scene*.xml + meta + task_info +
  `manifest.txt` 含 git HEAD + sha256)。
- eval:`results/E215/s6_downstream/eval/aug/e215_rot_{rollout,deltas}.tsv` + `e215_rot_eval_summary.json`。
- preflight:`results/E215/preflight/{seed_report,gravcomp_sidecars,baseline_audit}.json`。

## 10. 下一步

1. box004_083_p2(两 rot 皆崩,v1-confound)复核;30 个 v1-confound case 若要干净 C4 需补 v2 orig CEM。
2. 达标后交下游 RL 导出(同 E202-export/E213 partner re-anchor)。
3. (可选)C6 补渲 orig 三联对照 + fall case 中段帧,定位崩溃时刻。
