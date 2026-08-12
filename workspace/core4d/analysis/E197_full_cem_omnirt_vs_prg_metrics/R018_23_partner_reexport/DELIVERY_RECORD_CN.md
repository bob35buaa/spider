# E197 → R018-23 OmniRetarget Partner 重定向再导出 · 交付记录

_交付方:SPIDER core4d data_construction · 需求:`R018-23_e197_omnirt_partner_reexport_requirements_CN.md` · 日期:2026-08-13 · SPIDER git:`4e643e7`_

## 1. 目的(Purpose)

R018-23 的 52 条 CORE4D partner-assist 轨迹在下游 `core4d_partner.bundle` 里因缺少
OmniRetarget `s3_retarget` 中间产物(`qpos/human_joints/trim_window/raw_contact_mask`)
而无法算出 `anchor_local` 与 `controller_thresholds`,导致全部 52 case 标定量为 0 占位、
partner-assist 失效。本次交付补齐这些中间件,使下游 bundle 可 52/52 跑通。

## 2. 关键发现:这是"装配"而非"重跑"

需求文档假设上游只有 E197 精简 `s6/rl_export`(世界系 wrist),需要重跑 OmniRetarget + 接触阶段。
**实读磁盘后发现:每个 case 原生、自洽的 `s3_retarget` 完整产物仍然存在**于来源实验树
(E170/E172/E173,box021 的接触在 E168 tree)。这些 npz 是当年 OmniRetarget 跑出来的原件:
`qpos(T,43) f64`(`[:,36:39]`=世界系物体位置、`[:,39:43]`=单位四元数 wxyz)、
`human_joints(T,22,3) f32`、`fps`、`cost`,以及场景级 `raw_contact_mask_3cm.npz`。

因此本交付是**纯装配**:定位 → 校验(fail-closed)→ 按 E173 布局镜像 → 重写稳定路径 → 出 SHA256 与配对清单。
**qpos 直接沿用来源原生 43 维**(与同一次 retarget 的 human_joints、contact 自洽),
**无需** `scene_act_to_free` 42→43 重转(那会把 E197 的 PRG-rubberHull 几何与 E173 的
human_joints 混在不同世界系,破坏 anchor 重投影)。

### 52 target × 2 person = 70 唯一 person-case 的来源
- **65** 个来自 method_metrics 指向的来源树(E170/E172/E173,含 E168 接触树)。
- **1** 个(`box024_20231011_030_p2`)不在 method_metrics 但有完整 E173 原生 dir(glob 兜底)。
- **4** 个 partner 主跑未 retarget,仅存在于 E197 `partner_omnirt_rerun`(当年 `--skip-contact`):
  `box021_20231018_029_p2 / _034_p1 / _035_p1`、`box023_20231020_039_p1`。
  其 `raw_contact_mask_3cm` **取自各自的 gate-pass 同场景兄弟 case**——因为该掩码是
  raw 人体+物体运动的**场景级**属性(轴序 [帧, person(2), hand(2)]),p1/p2 逐帧完全相同
  (实测 `box001_20231003_2_038` p1 vs p2 `identical=True`),且 rerun 的 untrimmed 帧数
  (134/102/125/134)与兄弟 raw-contact 帧数逐一相等。**没有伪造接触,也没有跳过 case。**

## 3. 交付内容(Deliverables)

**数据树**(位于 gitignored `/mnt` 上,经 `workspace/core4d/results` 软链可解析):
```
E197/s6_downstream/rl_export/partner_reexport_v1/
  release/omnirt_v1_ref_fk/
    holosoma_dcv3_omnirt_v1_ref_fk_<case_id>/       # 70 个
      retargeted/<task>_with_obj_original.npz        # untrimmed
      trimmed/<task>_with_obj_original.npz
      trim_window.json                               # 路径重写为 SPIDER_ROOT:: 稳定串
    contact_masks/dcv3_omnirt_v1_ref_fk_<case_id>/raw_contact_mask_3cm.npz
  manifests/
    source_pairing_manifest.tsv    # 52 行 target↔partner 配对 + 全路径 + SHA256
    file_sha256.tsv                # 280 个交付文件的 sha256+字节数
    provenance.tsv                 # 每 case 的真实来源实验/变体/slug/tree
    reexport_summary.json
    validation_report.json
```
**git 跟踪副本**:`workspace/core4d/analysis/E197_full_cem_omnirt_vs_prg_metrics/R018_23_partner_reexport/`
(5 个文本清单 + 本记录);重型 npz 保留在 `/mnt`,以 SHA256 保证可复现。

**`source_pairing_manifest.tsv` 列**(对齐下游 56 列 source-manifest 子集,可直接映射):
`case_id, scene_id, date, sequence, object_name, target_actor(_index), partner_actor(_index),
raw_frame_count, raw_fps(=30), canonical_fps(=50), raw_contact_key(=raw_contact_mask_3cm),
partner_contact_person_index, contact_left_index(=0), contact_right_index(=1),
contact_threshold_m(=0.03), source_pairing_case_id, spider_git_commit, upstream_version,
upstream_method_id, partner_status(=pass),
{target,partner}_{untrimmed,trimmed}_npz, {target,partner}_trim_window_json,
target_raw_contact_path, raw_contact_path(=partner 侧),以及以上每个路径的 `_sha256`。`
路径均为 `SPIDER_ROOT::workspace/core4d/results/E197/...` 稳定可解析串。

## 4. 复现命令(Run command)

```bash
cd /home/ubuntu/Workspace/spider
.venv/bin/python workspace/core4d/scripts/data_construction_v3/stages/s6_downstream/build_e197_partner_reexport.py --force
```
输入:`e197_omni_absolute_wide_gate_filter.tsv`(52 gate-pass)+ `e197_method_metrics.tsv`(来源路径)。

## 5. 验收(Result,对照需求 §5)

| 验收项 | 结果 |
|---|---|
| 完整性 52×{p1,p2} 齐全 | **70/70** person-case、每个 retarged/trimmed/trim_window/contact 齐全 |
| schema gate(qpos≥43、human_joints=(22,3)、四元数单位) | **70/70 PASS** |
| trim 逐帧证明 `trimmed==untrimmed[start:end]` | **70/70 PASS**(qpos+human_joints 精确相等) |
| 接触:persons/hands/threshold=0.03、`shape[0]==raw_frame_count` | **70/70 PASS**(bool `(raw,2,2)`) |
| target 与 partner 同场景 `raw_frame_count` 一致 | **52/52 PASS** |
| SHA256 全部匹配清单 | **52 行 × 8 路径独立校验 PASS**;280 文件已登记 |
| 交付 npz 与来源字节一致(运动未被改写) | **各 box 抽检 SHA 相同**(沿用已视觉审核过的 OmniRetarget 原件) |
| anchor 物理合理(手位贴近箱面) | 每 box 69–135 帧在 0.20m 近场内,min-dist 近/入表面 → 选帧非空、非全帧回退 |
| bundle `build_release` 52/52 | **下游步骤**(需下游 robot-state + recipe,见需求 §7);上游侧 bundle 所校验的全部门禁已在交付件上复算通过 |

独立校验脚本读取**交付树 + 清单**(非构建内部状态),52/52 PASS。

## 6. 口径与约定

- 变体:46 target 用 `omnirt_v1`、6 用 `omnirt_v2`(来源数据事实,逐 case 记于 `provenance.tsv`);
  交付树统一 slug `dcv3_omnirt_v1_ref_fk_<case_id>`,真实来源在 provenance 可追。
- object_name:`Box001/Box004/Box021/Box023/Box024`;box001 的 `date=20231003_1|_2`、`seq=NNN`,与下游既有 box001 source-manifest 一致。
- 30fps raw + trim 窗口 + raw 接触 全部交齐;canonical 50fps 由下游 bundle 重采样(未在上游做)。
- 物体几何(half-extents / near_threshold=0.20)为下游 recipe 常数,**上游未导出**(符合需求 §2 注)。

## 7. 说明 / 未做项

- 本交付**不跑物理仿真**,是对既有、已视觉审核的 OmniRetarget 原件的字节级重打包(仅重写 `trim_window.json` 路径字段),故 experiment.md §7 的 scene-XML 快照不适用;可复现性由 SHA256 清单保证。
- 4 个 rerun partner 的接触复用同场景兄弟(场景级、逐帧对齐),已在 `provenance.tsv` 的 `contact_source_case` 标注。
- 下游后续(新建 recipe + source-manifest → bundle 重建 52 release → resolver/anchor → 重封新版本)见需求 §7,非上游职责。
