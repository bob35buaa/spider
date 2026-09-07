# plan243 · E213：E212 paired-export cases → 选定臂 object-augmentation（trans+rot）

**exp_name**: `core4d` · **Run**: R299 · **分支**: `experiment/E199-omniretarget-object-augmentation`
**前置**: E208(log297, desk/chair 源侧 aug PRG CEM) · E212(plan242, 混合臂 paired 导出) · E206(源基线)
**对应 log**: log/301_E213_*.md（待收口）

## Context

E212 交付的 `results/E212/s6_downstream/rl_export/paired_rl_export_input.tsv`（21 source arm-case，
4 物体）里，每个 case 用了各自选定的 gravcomp 臂（G1/G0.8/G0.6/G0.4/PRG）。E208 已对这 21 个 source
做过 trans+rot 增强，但 **CEM 只跑了 PRG 臂**——与选定臂不符（除 desk007_034_p1 恰为 PRG）。
partner 侧从未增强。

**E213 = 让这 21 个配对 case 拿到「与 E212 选定臂一致」的增强数据**。用户锁定口径：
① source+partner 都做；② source 终点 = Full CEM（各自选定臂）；③ partner = 仅增强重定向（运动学，
不跑 CEM，供日后 paired 导出 re-anchor）；④ retarget 复用 E208 口径（v1+v2 rescue），source 直接复用
E208 已建 retarget+`__aug_*` 任务；⑤ 只新生成每 case 选定 g 的 gravcomp scene_act + 重跑 CEM。

## 每 case 选定臂（取自 TSV arm/arm_gravcomp/arm_scene_name）

| 臂 | g | scene_name | cases | source CEM 行 |
|---|---|---|---|---|
| G1 | 1.0 | scene_act_E209_lowgeom_PRG_gravcomp | chair006×5 + desk021×7 = 12 | 60 |
| G08 | 0.8 | scene_act_E211_lowgeom_PRG_gc08 | desk007_028 p1/p2 = 2 | 10 |
| G04 | 0.4 | scene_act_E211_lowgeom_PRG_gc04 | desk007_030_p2 = 1 | 5 |
| G06 | 0.6 | scene_act_E211/E212_lowgeom_PRG_gc06 | desk007_032_p2(E211) + desk023×4(E212) = 5 | 25 |
| PRG | 0.0 | scene_act_E206_lowgeom_PRG | desk007_034_p1 = 1 | **复用 E208（5 行不重跑）** |

⇒ **source 新增 CEM = 100 行**（20 例 × ~5 变体）；PRG 那例 5 行复用 E208。

## 实现（scripts/experiments/E213/）

- `e213_common.py` — 契约：从 E212 TSV 派生 21 case + 选定臂映射（sha pin `d71370fa…`）；
  从 E208 `e208_aug_artifacts.tsv`（status=built）取 105 aug 变体行；`build_selected_sidecar`
  （读 aug 目录 `scene_act_E206_lowgeom_PRG.xml`，物体 body 置 gravcomp=原选定场景读回的值，
  复用 `e212_common.assert_gravcomp_diff_value`）；partner gap（11 例）；4-shard 拆分。
- `build_source_arm_scenes.py` — 100 个选定臂 gravcomp sidecar（C1 单变量自测）。
- `build_source_overrides.py` — 100 个 E213 override（`defaults: [E208 PRG override, _self_]` + 换
  `scene_name`），Hydra compose 对称 diff == {scene_name}（C2）。
- `build_manifest.py` — 100 行 source CEM manifest + 4 shard（A/B/C/D 各 25，round-robin 按
  (arm,object,case,variant)，每 shard 跨全部臂）；sha pin scene/traj/mask/override。
- `run_source_cem.py` + `launch/active/run_E213_shard.sh <A|B|C|D>` — 每机一 shard，8 卡池，
  per-row scene_name，flock 锁，180min 超时，输出校验 config_act.scene_name==行 scene_name，写 host。
- `merge_shards.py` — 4 shard 归并（coverage/completion/identity/provenance）。
- `build_partner_aug.py` — Phase B：11 partner seed(converted+_original) → pipeline.sh aug
  (`--skip-spider --skip-contact` RETARGET_AUGMENTATION=1, v1+v2 rescue) → fixed-window trim。仅运动学。
- `snapshot_E213_scenes.sh` — 规则7：100 aug 任务目录 scene 快照（仅 shard A 机执行）。
- `test_e213_contract.py` — A1–A7 离线自检（**PASS**）。

## 32 卡调度（4×8，共享 /mnt+.venv）

- A = 本机（Claude 跑）；B/C/D = 3 台远端（用户各一条命令）。每机只读写自己 shard manifest（避免整表写竞争），
  结果写共享 /mnt，最后 `merge_shards.py` 归并。
- 100 行 / 32 卡 ≈ 4 轮 × ~44min ≈ **~3h wall**（单机 8 卡约 9.5h）。
- 命令：`bash workspace/core4d/scripts/launch/active/run_E213_shard.sh {A|B|C|D}`。

## 成功标准（预注册）

C0 契约 A1–A7 全过（**已过**）；C1 sidecar 100/100 单变量（**已过**）；C2 override diff=={scene_name}
（**抽样过**）；C3 source CEM 100/100 cem_ok + config_act.scene_name 匹配 + 零超时；
C4 aug-vs-orig（orig=选定臂原 rollout）obj_pos HL ≤+5cm，通过率下降按 case 级检验报告（mean+std+worst）；
C5 partner 11 例增强 retarget 可加载/帧数一致/物体通道有位移；C6 抽帧目视无度量欺骗；C7 快照+sha 入 log。

## 验证

契约自检 → smoke 1 行（scene=scene_act_E209_lowgeom_PRG_gravcomp，parity pass，3cm mask
L/R 42.9/48.3% ✓）→ 4 shard Full CEM → merge → eval（eval.core.core_metrics，配对 orig）→ 抽帧目视。

## 风险/边界

- desk007_034_p1（PRG）source CEM 复用 E208，不重跑。
- aug 变体按 E208 已交付可行集（v1 IK 缺口不追加 v2 重跑）。
- partner chair006_015_p2 是 direct_omnirt_partner_temp，seed 自 partner_omnirt_direct_v2 树，meta 构造。
- partner 仅运动学产物；paired RL 导出（exporter re-anchor）为后续阶段。
