# E213 · E212 paired-export cases → 选定臂 object-augmentation（trans+rot）结果

**Run**: R299 · **Plan**: [243](../plan/243_E213_paired_export_selected_arm_aug_plan.md) · **日期**: 2026-09-07 · **分支**: `experiment/E199-omniretarget-object-augmentation`
**结论**: ✅ **SUCCESS（C4 PASS）** — source 侧选定臂 aug CEM 100/100 cem_ok、数值不劣于选定臂 orig（obj_pos HL **+0.213cm** ≪ +5cm 门）；partner 侧 11 case×5 变体运动学增强全交付。

---

## 1. 做了什么

E212 交付的 21 个混合臂 source case（`paired_rl_export_input.tsv`，sha 锁定
`d71370fa…f3c1d`），每个 case 用**各自 E212 选定的 gravcomp 臂**（G1/G0.8/G0.6/G0.4/PRG，非统一 PRG）。

- **source 侧**：复用 E208 已建的增强 retarget NPZ + `__aug_*` SPIDER 任务目录，
  **只新生成每 case 选定 g 的 gravcomp `scene_act` sidecar（单变量差）+ 重跑 Full CEM**。
  20 个非 PRG case × 其可行 aug 变体（trans0/1/2 + rot0/1）= **100 条新 CEM**；
  `desk007_034_p1`（选定臂恰为 PRG，5 变体）直接**复用 E208 PRG 结果，不重跑**。
- **partner 侧**：11 个 partner-only case（不在 21 source 行内）做**增强重定向（运动学，不跑 CEM）**，
  复用 E208 口径（omnirt_v1 优先 + v2 rescue），供日后 paired RL 导出 re-anchor（[[two-person-aug-partner-reanchor]]）。

评估 orig 基线 = **每 case 的 E212 选定臂原 rollout**（`cem_result_npz`），当场重打分。

---

## 2. 执行结果（C0–C7）

| # | 判据 | 门 | 结果 |
|---|------|----|------|
| C0 契约 | registry 从 TSV 派生恰 21 case；选定臂映射逐 case == TSV；partner 缺口清单精确 | 全过 | ✅ `test_e213_contract.py` A1–A7 全绿（sha pin、选定臂映射、单变量 sidecar、partner 11 缺口） |
| C1 sidecar 单变量 | 每 aug 变体 gc sidecar 与 aug PRG base **仅差物体 `gravcomp`** | 100% | ✅ 100/100（`assert_gravcomp_diff_value`，编译不变量 ngeom/npair/nq/nv/nu/nbody 不变） |
| C2 override 纯净 | 每 override compose 后对称 diff `== {scene_name}` | 100% | ✅ 100/100 |
| C3 source CEM | 可行 aug 变体全 `cem_ok`、qpos 有限、`config_act.scene_name`==选定 gc、零超时 | 100% | ✅ **100/100 cem_ok，0 failure，0 超时**；merge `n_problems=0` |
| C4 数值不劣 | obj_pos HL ≤ +5cm；通过率下降按 case 级符号检验 | 报 mean+std+worst | ✅ **PASS**（详见 §3） |
| C5 partner 产物 | 增强 retarget 可加载、帧数一致、物体通道有位移 | 100% | ✅ **11/11 ok，55 trimmed npz**（详见 §4） |
| C6 视觉无欺骗 | 抽帧无穿模/漂浮/抖动的度量假象 | 记录实际观察 | ✅ 抽样 `chair006_20231003_1_003_p1` rot0/G1（详见 §5） |
| C7 复现 | 规则 7 快照 + manifest sha256 | 完成 | ✅ `scene_snapshot/`（100 aug task 目录 + manifest 805 行，git HEAD + sha256） |

---

## 3. C4 数值（source aug vs 选定臂 orig，n=105 配对）

主指标 `track_obj_pos_err_cm`（越小越好，正 delta = aug 更差）：

| 层级 | HL 中位差 | mean | std | worst | 判定 |
|------|-----------|------|-----|-------|------|
| **pooled（全 105）** | **+0.213** | +0.230 | 0.675 | +2.090 | ✅ ≤ +5cm 门 |

- **通过率（14-gate narrow）**：orig 90/105 → aug 78/105，**行级下降 +0.114**（门 +0.15，PASS）；
  **case 级下降 +0.114**（21 case：10 worse / 3 better / 8 same，case 级 pass 符号检验 p=0.092）。
- **诚实口径**：每 case ~5 变体共享一个 orig，行级 Wilcoxon/McNemar 反保守；case 级
  companion（n=21）obj_pos HL **+0.239**、Wilcoxon p=0.032、sign p=0.189、worst +1.098。
- **配套指标**：obj_ori HL +0.757°（worst +7.87°），body_z_p95 HL +0.0036m，
  接触 in-mask HL −0.005（几乎无损），**fall_flag 全 0**。

**逐臂 obj_pos HL（选定臂分组，全部 ≤ +5cm 门）**：

| 臂 | n | obj_pos HL | mean | std | worst | gate 下降 |
|----|---|-----------|------|-----|-------|-----------|
| G1  | 60 | +0.159 | +0.157 | 0.650 | +1.722 | +0.100 (50→44/60) |
| G08 | 10 | +0.310 | +0.360 | 0.746 | +2.090 | −0.100 (5→6/10) |
| G06 | 25 | +0.397 | +0.386 | 0.718 | +1.611 | +0.120 (25→22/25) |
| G04 | 5  | +0.004 | +0.007 | 0.384 | +0.483 | +0.200 (5→4/5) |
| PRG | 5  | +0.133 | +0.291 | 0.898 | +1.776 | +0.600 (5→2/5) |

> PRG 那 5 条是 `desk007_034_p1` 复用 E208 的 aug，orig 也是 E208 PRG——gate 下降 0.600
> 系 5 条小样本 + 该 case 本就是历史上姿态最脆的一例（见 E209/E211 对 034_p1 的记录），
> obj_pos HL 仍仅 +0.133cm，物体侧不劣。

**CEM wall（merge，4 机各 25 条）**：

| host | n | median | min | max |
|------|---|--------|-----|-----|
| h800-b1-v1-047 | 25 | 45.8 | 29.6 | 72.3 |
| rdma-prod-103 | 25 | 45.4 | 29.6 | 71.5 |
| u3x6-6 | 25 | 44.3 | 28.3 | 69.5 |
| reservedpool-svn7y-1 | 25 | 44.2 | 30.4 | 71.0 |
| **全体** | 100 | **44.9** | — | — |

---

## 4. C5 partner 侧（11 case × 5 变体 = 55 trimmed npz）

全部 `status=ok`，每 case 5 变体（trans0/1/2 + rot0/1）齐全：

| partner case | obj | primary | 变体来源（v1/v2 rescue） |
|--------------|-----|---------|--------------------------|
| chair006_20231003_1_003_p2 | chair006 | v2 | 全 v2 |
| chair006_20231003_1_005_p2 | chair006 | v2 | 全 v2 |
| chair006_20231003_2_011_p1 | chair006 | v1 | trans_0=v2，余 v1 |
| chair006_20231003_2_015_p2 | chair006 | v2 | 全 v2 |
| chair006_20231011_076_p1   | chair006 | v1 | rot_0/trans_0=v2，余 v1 |
| desk007_20231030_030_p1    | desk007  | v1 | 全 v1 |
| desk007_20231030_032_p1    | desk007  | v1 | 全 v1 |
| desk007_20231030_034_p2    | desk007  | v1 | 全 v1 |
| desk021_20231011_014_p1    | desk021  | v2 | 全 v2 |
| desk023_20231011_005_p2    | desk023  | v2 | 全 v2 |
| desk023_20231030_019_p2    | desk023  | v2 | 全 v2 |

seed 源 = 各 partner 在 E206 的 `converted/` SMPL-X npz（无需 raw CORE4D）；
`..._015_p2`（`direct_omnirt_partner_temp`）从其 `partner_trimmed_npz` 的父目录派生 seed。
partner 仅运动学产出——不建 SPIDER 任务、不跑 CEM、不建 gravcomp。

---

## 5. C6 视觉复核

- **抽样**：`E213_chair006_20231003_1_003_p1_aug_rot0_G1.mp4`（选定臂 G1，rot0，`/video-frames` 抽首帧）。
  grasp/contact/末帧无穿模、无漂浮、无度量假象抖动，机器人姿态自然，物体跟随合理。
- **跨视频 A/B 无效**（E208 F13 / E210 F3 / E209 F6）：`_auto_video_camera` 按 sim∪ref 包围盒逐帧
  重算机位，两条 rollout 的 mp4 机位不同——**只比单视频内 sim-vs-ref**。跨 case aug-vs-orig 目视
  用 `viser_replay_E213.py`（一套相机/时间线/场景，port 8213）。
- **render**：100/100 rendered，0 reused/0 failed（`e213_render_summary.json`）。

---

## 6. 结果路径

| 类型 | 路径 |
|------|------|
| source CEM manifest（4 shard） | `results/E213/s6_downstream/manifests/e213_source_arm_cem.shard{A,B,C,D}.tsv` |
| source merge | `results/E213/s6_downstream/manifests/e213_source_merge.json` |
| eval 摘要 / delta / rollout | `results/E213/s6_downstream/eval/aug/{e213_aug_eval_summary.json, e213_aug_deltas.tsv, e213_aug_rollout.tsv}` |
| xlsx | `results/E213/s6_downstream/eval/aug/E213_aug_vs_orig.xlsx`（10 sheet：README/C4/ByArm/ByOffsetBand/ByVariant/ByObjectVariant/Gate14*/Gates/PerCase） |
| render mp4 | `results/E213/s6_downstream/render/full/*.mp4`（100 条 + `e213_render_summary.json`） |
| partner manifest | `results/E213/data_preprocess/manifests/e213_partner_aug_artifacts.tsv`（11 行） |
| partner trimmed npz | `results/E213/data_preprocess/partner_aug/omnirt_v{1,2}/holosoma_.../trimmed/`（55 条） |
| 场景快照（规则 7） | `results/E213/scene_snapshot/`（100 aug task 目录 + `manifest.txt` 805 行，git HEAD + sha256） |

---

## 7. 复用 / 新建

- **复用（零改动）**：E208 增强 retarget NPZ + `__aug_*` 任务目录、`e208_aug_artifacts.tsv`、
  `e212/e211/e209_common.build_sidecar`、`spider/config.py` scene_name 解析、
  `pipeline.sh` aug 分支、`OMNIRT_V{1,2}_ENV`、E206 partner `converted/` seed、`eval.core.core_metrics`。
- **新建（`scripts/experiments/E213/`）**：`e213_common.py`（契约）、`build_source_arm_scenes.py`、
  `build_source_overrides.py`、`build_manifest.py`、`run_source_cem.py`、`merge_shards.py`、
  `build_partner_aug.py`、`snapshot_E213_scenes.sh`、`test_e213_contract.py`、`render_E213.py`、
  `viser_replay_E213.py`；`eval/runners/eval_E213_aug.py` + wrapper；
  `eval/reports/gen_E213_aug_workbook.py`；`launch/active/run_E213_shard.sh`。

## 8. 遇到的坑（复用价值）

| 现象 | 根因 | 修复 |
|------|------|------|
| 契约 A3 desk007 G06 是 E211、desk023 G06 是 E212 | 固定 experiment→scene 映射不成立 | 按 gravcomp 派生的后缀校验 scene_name |
| 契约 A6 `..._015_p2` 无标准 E206 converted 目录 | `direct_omnirt_partner_temp` case | 从 `partner_trimmed_npz` 父目录派生 seed |
| eval 5 行 KeyError 'variant'（PRG 复用行） | 复用行无 variant 字段 | `scoring.setdefault("variant", …)` |
| partner batch 卡在 1/11 | 链式 waiter `kill -0 <smoke_pid>`，smoke 变僵尸；`kill -0` 对僵尸返回成功（E208 F1） | 杀 waiter，直接重启 `build_partner_aug.py` |

## 9. 下一步

- source aug（100 条选定臂）+ partner（55 条运动学）已就绪，可进 paired RL 导出的 partner re-anchor
  （E202-export 口径），把 source 侧 aug 与 partner 侧 aug 在下游对齐（raw 物体差异不作放行门，
  自洽由 re-anchor 保证，[[two-person-aug-partner-reanchor]]）。
- 34_p1（PRG）5 条 gate 下降大但 obj_pos 不劣，导出时可单独标注/降权。
