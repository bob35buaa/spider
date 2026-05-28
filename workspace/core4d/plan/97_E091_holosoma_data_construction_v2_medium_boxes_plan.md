# E091 Plan: Holosoma data_construction_v2 for medium box discovery

日期：2026-05-28

关联输入：

- 诊断建议：`workspace/exp_diagnostic/data_filter_recommendation.md`
- E090 结果：`workspace/core4d/log/112_E090_h2_first_retarget_and_spider_smoke_results.md`
- Holosoma 旧链路：`/home/ubuntu/Workspace/holosoma/workspace/v3/data_construction`
- 新实验目录：`/home/ubuntu/Workspace/holosoma/workspace/v3/data_construction_v2`

## 0. 计划更新结论

本计划不再把 B 路「继续扩 Box021 D003 13 case」作为主线。E090 已经支持 H2 的核心判断：Box021 失败主要来自上游 retarget 后 G1 hand/wrist target 的几何语义错误；topface-preIK 能明显修复几何 safety，但 full CEM 仍出现 pelvis collapse。因此继续扩 Box021 D003 会把问题带回动力学局部解优化，不符合当前目标。

E091 主线改为 C 路：在 Holosoma `workspace/v3/data_construction_v2` 中，围绕尺寸介于 Box023 与 Box025 之间的新物体，补模板、跑 OmniRetarget/SPIDER、接入 D005b G1 feasibility gate，并用可视化与 high-reasoning 复核筛出下一批候选。

优先级：

1. **Box026**：52 seq，全量最多；D001 clean 12，D002 raw-contact pass 7；体积 `0.116 m^3`，约 `3.23x Box023 / 0.35x Box025`。主线优先。
2. **box004**：20 seq；D001 clean 4，D002 pass 4；体积 `0.041 m^3`，略大于 Box023。作为快速低风险补量。
3. **Box022**：8 seq；D001 只有 review，没有进入 clean D002。作为 stress-test / review 回收，不抢主线。
4. **Box021**：保留为 calibration/baseline，不再扩 D003 13 case；只在需要对照 D005b 或 retarget variant 时使用。

### 0.1 2026-05-29 动态更新

Phase 0/1 已完成，`data_construction_v2` 已生成 manifest、raw-contact 可视化、Box026/box004 source scene template，并完成 high-reasoning 视觉复核。Box026 top3 的 no-fingertip Stage2b/D005b 初筛结果如下：

| case | Stage2b | D005b 结论 | 关键原因 |
|---|---|---|---|
| `e091_box026_20231018_039_p2` | pass，trim `123` frames | reject / near-pass | inside 与 pelvis 均好，但 support-face frac 约 `20% < 30%` |
| `e091_box026_20231018_040_p2` | fail | 未进入 D005b | no-fingertip retarget 在约 `67/99` 帧 CVXPY infeasible；不重复同配置 |
| `e091_box026_20231020_135_p2` | pass，trim `82` frames | reject / near-pass | support-face frac 约 `62%`，但 right wrist inside `12.2% > 10%` |

更新后的执行策略：

1. **先把 box004 control 提前为下一步**：目标是尽快拿到 medium-object 的 positive pipeline，而不是在 Box026 上继续扩大同一配置扫描。
2. **暂停 Box026 no-fingertip top7 扩展**：当前两条成功预处理都是 near-reject，说明问题不是模板/目录，而是 hand target/support semantics 仍需有限修正。
3. **Box026 只保留有限 H2 variant**：若 box004 不能给出 D005b pass，再对 `039_p2` 和 `135_p2` 做 support-face-preIK / exterior projection variant；`040_p2` 不重复 no-fingertip。
4. **`--replace_wrist_with_fingertip` 仍不作为 medium-box 默认策略**：它可以作为 reach shadow/negative guard，但不是 Box026/box004 production 默认。

### 0.2 2026-05-29 结果更新

E091 已找到第一条 medium-object seed：`e091_box004_20231003_2_083_p2`。它在 Stage2b、D005b、visual QC 和 top bank 中通过，进入 minimal SPIDER smoke。smoke 已运行完成：collision/object tracking 侧通过（head/upper `0%`，object mean error `0.006m`），但 pelvis min `0.079m`，high-reasoning 视觉复核判定为 **REVIEW / dynamics pelvis collapse**。

因此本轮结论是：`data_construction_v2` 数据链路已经跑通到 smoke 并产出可复查证据；`box004` 是当前 top-bank seed positive；dynamic smoke 未 final pass 的原因是 CEM 姿态/站立问题，不回滚 D005b/top-bank 数据筛选结论，也不在 E091 内继续优化算法。

## 1. 术语澄清

**world-up top face** 指：在物体当前 world pose 下，把 box collision 的 6 个外法线转到世界坐标，选择 outward normal 与 MuJoCo/world `+Z` 夹角最小的那个面。它不是写死的 object local `+z` 面。Box021 有旋转，写死 local `+z` 会误判；E090 已把 gate 改成 world-up 支持面，并保留 `legacy_local_z` 作为 Box025 guard 兼容指标。

`--replace_wrist_with_fingertip` 的定位也更新：它原本是为 Box025 太大、G1 臂展不够而设计的 reach hack，不能默认推广到中等箱。E091 中：

- Box025 / reach-limited guard 可以保留 fingertip replacement；
- Box026 / box004 / Box021 这类 medium box 默认不能把它当作无条件策略；
- 若 raw support face 和 reach margin 显示确有 reach 不足，再作为条件 variant，而不是 production 默认。

## 2. Claims

| Claim | 可验证标准 |
|---|---|
| C1: 旧 D001-D003 已足够支持 C 路启动 | 能从旧结果复现 medium manifest：Box026 7 条 D002 pass、box004 4 条 D002 pass、Box022 review-only，并标清每条是否缺模板 |
| C2: D005b 能在动态训练前拦住 retarget 几何不可行 case | 每个 Stage2b pass case 都输出 `inside_frac / signed_dist / support_face_frac / pelvis_z_min / T`，D006 前必须有 D005b pass/reject |
| C3: 对 medium box，conditional no-fingertip/support-face retarget 比全局 fingertip 更合理 | Box026/box004 主跑 no-fingertip/support-face 语义；只在 reach guard 失败时补跑 fingertip shadow，不把 Box025 hack 直接推广 |
| C4: E091 的产出目标是候选数据，不是继续调 SPIDER reward | 成功标准先到 D005b + visual QC + top bank；只有 top1/top2 才进入 smoke，且 smoke 失败不回写为“数据无效” |
| C5: 可视化能支撑人工判断 | 每个进入 D005b/top bank 的 case 有 raw-contact、retarget timeline、object-local wrist/support overlay、短视频/帧图和 high-reasoning review |

## 3. 现有链路审计结论

旧 Holosoma `data_construction` 的正式 v2 状态：

- D001 全量评估 `1840` case-person，clean `32`，review `654`，discard `1154`。
- D001 clean 分布：`Box021=16`、`Box026=12`、`box004=4`。
- D002 clean raw-contact：`box004=4/4 pass`、`Box026=7/12 pass`、`Box021=15/16 pass`。
- D003 queue：`pipeline_ready_now=15`，全部是 Box021；`template_backlog=11`，包括 `box004=4` 和 `Box026=7`。
- Box022 处于 D001 review-only，不在 D002 clean 队列，应作为后续 review 回收。
- D004/D005 的 JSON/ledger 声称有 PNG/timeline 输出，但当前实际目录中主要能看到 mp4/metrics，部分 PNG 路径指向旧拼写或不存在。E091 必须把可视化重新作为显式产物验证，而不是沿用旧 summary 假设。

需要注意：旧文档/脚本中有 `data_constructon` 拼写残留，而实际目录是 `data_construction`。E091 新目录必须使用绝对路径和显式 `RESULT_ROOT/CASE_FILE`，不要直接复用旧脚本默认路径，避免写到不存在或错误目录。

底层 Stage2b/D003 入口可复用 SPIDER pipeline：

```text
convert_core4d_to_omniretarget.py
  -> robot_retarget.py
  -> trim_no_contact.py
  -> generate_core4d_contact_masks.py
  -> create_spider_scene_from_template.py
  -> spider/process_datasets/core4d.py
  -> scene_act
  -> verify
```

当前真正 blocker 不是命令入口，而是 Box026/box004/Box022 没有 SPIDER/MuJoCo `source_scene_task/scene.xml` 模板。模板制作和预检必须先行，参考：

```text
/home/ubuntu/Workspace/spider/workspace/core4d/data_preprocess/SCENE_TEMPLATE_GUIDE.md
```

## 4. 实验阶段

### Phase 0: data_construction_v2 初始化

新建目录结构：

```text
/home/ubuntu/Workspace/holosoma/workspace/v3/data_construction_v2/
  inputs/
  scripts/
  results/
  visualizations/
  logs/
  reports/
```

输入快照：

- 复制或引用旧 D001/D002/D003 summary；
- 生成 `inputs/medium_box_manifest.tsv/json`，字段至少包含 object、sequence、person、D001 decision、D002 raw-contact score、stage2 route、template status、object extents/volume。

Phase 0 成功标准：

- manifest 中 Box026 有 7 条 primary pass；
- box004 有 4 条 secondary pass；
- Box022 被标为 review-only，不混入 primary；
- 所有路径指向 `data_construction_v2`，不依赖旧 `data_constructon` 拼写。

### Phase 1: template / asset preflight

目标：让 Box026 和 box004 能进入 Holosoma stage2b / D003 production。

检查：

- object mesh：`box/box026_m.obj`、`box/box004_m.obj`、后续 `box/box022_m.obj`；
- SPIDER source scene template 是否存在；
- MuJoCo scene / object collision extents 是否与 inventory AABB 一致；
- scene 中 object mass 是否合理，避免 Box021 D003 曾出现的异常 `29.632kg` 类问题；
- `scene.xml`、`scene_act.xml`、URDF/MJCF 产物能被 MuJoCo load。

Phase 1 成功标准：

- Box026、box004 至少各有一个 source_scene_task 通过 `mj_loadXML`；
- template preflight 输出尺寸、质量、geom count、npair、qpos/nq/nv；
- 失败原因明确归类为 missing mesh / bad XML / mass mismatch / scene load error。

### Phase 2: Stage2b OmniRetarget + SPIDER production

主跑对象：

- Box026 D002 pass 7 条；
- box004 D002 pass 4 条；
- Box022 暂不主跑，除非 Box026/box004 无任何 D005b pass；若回收，先手动抽 `20231023/125`、`20231023/126`，并明确标为 review stress-test。

retarget variant 策略：

1. primary variant：medium-box no-fingertip 或 support-face-aware retarget，不默认使用 Box025 的 fingertip reach hack；
2. shadow guard：每个对象只挑 top1/top2 raw-contact case 跑 default/fingertip 作为负向/对照；
3. 若 Box026 因 reach margin 明确不足失败，再启用 conditional fingertip variant，而不是全量重跑。

每条 case 必须产出：

- converted NPZ；
- retargeted NPZ；
- trimmed NPZ；
- contact mask；
- SPIDER `scene.xml / scene_act.xml / trajectory_kinematic.npz`；
- verify summary；
- run log。

Phase 2 成功标准：

- Box026 top3 已完成阶段性初筛：2 条 preprocess pass，1 条 CVXPY infeasible；在没有 H2 variant 前不继续扩大同配置；
- box004 至少完成 2 条 preprocess pass，或明确记录失败原因；
- 不因单条 CVXPY infeasible 重复同配置。

### Phase 3: D005b G1 feasibility gate

把 `workspace/exp_diagnostic/scripts/g1_feasibility_gate.py` 集成到 v2：

- 输入：`trajectory_kinematic.npz` + `scene.xml`；
- 输出：`results/d005b_g1_feasibility/d005b_summary.tsv/json`；
- 每 case 生成 object-local support overlay 和 timeline。

核心指标：

| 指标 | 初始阈值 |
|---|---:|
| wrist/eef inside box frac | `<= 10%` |
| support signed distance mean | `>= 0.03m` |
| pelvis_z_min | `>= 0.60m` |
| wrist_below_pelvis_gap mean | `<= 0.30m` |
| trajectory length | `>= 80` frames |
| support face frac | `>= 30%` for at least one hand |

support face = `max(world_up_top_face_frac, legacy_local_z_face_frac)`；但 report 中必须同时列出两者，防止 local-axis 误判。

Phase 3 成功标准：

- 所有 Stage2b pass case 都有 D005b row；
- D005b reject 必须有 reason list；
- 至少生成 aggregate scatter：inside vs support、pelvis_min vs support、object size vs pass；
- high-reasoning subagent 对 top pass/reject 的 overlay 做复核，并写入 `reports/high_subagent_d005b_review.md`。

### Phase 4: visual QC and top bank

对 D005b pass 和 near-pass case 生成可视化：

- raw contact timeline；
- retarget root/torso/pelvis height；
- object trajectory raw vs retarget；
- object-local wrist/support face overlay；
- per-case frame sheet；
- short mp4/keyframes；
- aggregate dashboard。

E091 可视化必须有文件存在性和像素非空检查：

- raw contact：从 `raw_contact_proxy.npz` 生成 timeline PNG；
- retarget/trimmed：生成 diagnostic PNG、frame sheet 和 mp4；
- G1 D005b：生成 object-local wrist/support overlay PNG；
- aggregate：生成 inside-vs-support、pelvis-vs-support、size-vs-pass dashboard；
- high-reasoning review 必须引用实际存在的图片/视频路径。

Top bank scoring 新增 `g1_feasibility_score_25`：

- D005b pass：+25；
- 每个 reject reason：-7；
- inside box severe 或 pelvis severe 直接 hard reject；
- visual QC 可把自动 pass 降为 review，但不能把 severe D005b reject 升为 mainline。

Phase 4 成功标准：

- 产出 `top_medium_box_bank_manifest.tsv/json`；
- 明确 top1/top2/top3 及 reserve；
- 每个 top case 链接到 raw、retarget、D005b、visual evidence；
- 若无可用 case，给出是 asset/template 问题、retarget 问题，还是 G1 geometry 不可行。

### Phase 5: minimal SPIDER smoke only for selected cases

只有 Phase 4 top1/top2 进入 smoke。E091 不把目标设成 full CEM 调通。

Smoke 检查：

- startup 是否能 load；
- 64-env metrics；
- rollout video/keyframes；
- head/upper/lower/body/object collision；
- pelvis_z_min；
- object error；
- contact continuity。

Phase 5 成功标准：

- top1 smoke 不出现 E082-E088 式头胸穿箱或手撑地；
- 若 pelvis collapse，记录为动力学后续问题，不回滚 D005b 数据筛选结论；
- 只有 smoke metrics + video + high-reasoning review 都通过，才允许进入下一实验的 full CEM/训练计划。

## 5. 计划落地脚本

计划新增脚本路径：

```text
workspace/core4d/scripts/E091/
  build_medium_box_manifest.py
  make_data_construction_v2_dirs.sh
  run_template_preflight.sh
  run_stage2b_medium_boxes.sh
  check_d005b_g1_feasibility.py
  make_raw_contact_visuals.py
  make_medium_box_visual_qc.py
  make_top_medium_box_bank.py
```

计划固化命令：

```bash
bash workspace/core4d/scripts/E091/make_data_construction_v2_dirs.sh
python workspace/core4d/scripts/E091/build_medium_box_manifest.py
bash workspace/core4d/scripts/E091/run_template_preflight.sh --objects box026,box004
bash workspace/core4d/scripts/E091/run_stage2b_medium_boxes.sh --objects box026,box004 --variant medium_primary
python workspace/core4d/scripts/E091/check_d005b_g1_feasibility.py
python workspace/core4d/scripts/E091/make_raw_contact_visuals.py
python workspace/core4d/scripts/E091/make_medium_box_visual_qc.py
python workspace/core4d/scripts/E091/make_top_medium_box_bank.py
```

这些脚本在实现时必须写死输出根目录为：

```text
/home/ubuntu/Workspace/holosoma/workspace/v3/data_construction_v2
```

## 6. 不做事项

- 不继续把 Box021 D003 13 case 扩成主线；
- 不把 `--replace_wrist_with_fingertip` 全局删除，也不全局启用；
- 不把 Box025 topface-preIK 失败推广成所有对象失败；
- 不在 D005b/visual QC 之前启动 full CEM；
- 不用 startup pass 或单个数值指标声明 clean trainable。

## 7. 下一步执行顺序

1. 当前 E091 停止继续扩同配置扫数据；已有 top1 seed 和 smoke 证据足以完成本轮数据链路目标。
2. 下一轮若要推进 dynamics，应单独开 pelvis/upright/stand-stability 约束实验，而不是把 E091 变成 reward 调参。
3. 下一轮若要补更多数据，优先补 `box004_person1` template / 第二条 box004 control；Box026 只跑 bounded H2 variant，不重复 `040_p2` no-fingertip。
4. Box022 仍只作为 stress-test/review 回收。
