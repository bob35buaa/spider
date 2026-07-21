# E170 Box021 PRG 全量验证：终审前实验日志

_实验执行：2026-07-18；日志整理：2026-07-19；状态：pre-user review ready_

---

## 📋 一屏摘要

- **执行完整**：24 条 E170 new full 自然完成，4 条 E169 PRG full 按 SHA 复用；统一评测 `28/28`，无 missing/error
- **数值结果**：六项严格数值门全部通过 `12/28`；E168 `USE` 组为 `9/13`，E168 `DO_NOT_USE` 组为 `3/15`
- **主要告警**：lower-body `11` 条、raw contact `8` 条、body-z `2` 条、hand penetration `2` 条；fall 与 release 新失败均为 `0`
- **视觉结果**：Codex 已看 `28/28` 关键帧 sheet，并对 10 条重点 paired MP4 做 3 fps 全时序复核；未发现 E170 新增跌倒、踢飞箱体、真实踩箱支撑或爆姿
- **机制边界**：gate-health 为 `0/28`，因此 G 不能被称为健康的硬约束机制；这与轨迹是否可人工接受分列
- **最终裁决**：机器建议仍为 `PENDING_USER_REVIEW`；用户表 `0/28` 已审核，本文不代填 `manual_*`，也不提前宣布 strong/partial/fail

> 📌 **给终审者：** 建议先看下方 8 条优先 case，再按逐 case 表补完其余视频。XLSX 保留作完整审计底稿，不必作为主要阅读入口。

## 🎯 裁决边界

E170 同时保留两条互不覆盖的判断轨道：

| 轨道 | 定义 | 当前状态 |
| --- | --- | --- |
| **人工 operational use** | 用户最终 `manual_use_decision == USE` | 待用户 28/28 fresh 标签 |
| **严格 release usable** | 六项数值门全部通过，且用户判为 `USE` | 数值上限 `12/28`，仍待用户标签 |

数值 fail 是严格发布告警，不自动等价于视觉不可用。用户可以把视觉可接受的数值 fail 判为 operational `USE`；该决定不会删除告警，也不会把它改写成 strict release pass。

## 📊 数值核验

### 六项严格门

| 门 | 阈值 | 通过 | 未通过 |
| --- | ---: | ---: | ---: |
| **Fall** | `fall=false` | `28/28` | `0` |
| **Body-z** | p95 `<=0.20 m` | `26/28` | `2` |
| **Raw contact** | in-mask `>=0.50` | `20/28` | `8` |
| **Release false contact** | `<=0.30` | `28/28` | `0` |
| **Hand penetration** | 3 mm frame frac `<=0.30` | `26/28` | `2` |
| **Lower-body penetration** | frac `<=0.10` | `17/28` | `11` |
| **六门同时通过** | 全部满足 | `12/28` | `16` |

`20231020_020_p2` 没有 release window，因此 release 门按 `NOT_APPLICABLE_NO_RELEASE_WINDOW` 记通过；其余 `27/27` applicable case 均通过。失败模式允许重叠，不能把各行未通过数相加当成失败 case 总数。

### E168 标签分层

| E168 分层 | 数值通过 | 数值未通过 | 严格轨道含义 |
| --- | ---: | ---: | --- |
| **USE retention** | `9/13` | `4/13` | strict retention 上限为 `9/13`，低于强泛化门 `12/13` |
| **DO_NOT_USE recovery** | `3/15` | `12/15` | strict recovery 上限为 `3/15`，低于强泛化门 `9/15` |
| **Overall** | `12/28` | `16/28` | operational use 仍需人工逐条判断 |

数值层已足以说明“强泛化”所需的 strict 上限达不到，但不能代替用户决定哪些告警轨迹在 operational 口径下仍可用。

## 🔍 视觉核验

实际检查覆盖：

- `28/28` case × 5 个事件关键帧，共 `140/140` 条关键帧证据
- 10 条重点 paired MP4 的 3 fps 全时序 filmstrip：`038_p2`、`028_p1/p2`、`030_p1`、`031_p2`、`035_p2`、`019_p2`、`020_p2`、`022_p2`、`023_p2`
- 全部重点视频未见 E170 新增 fall、箱体踢飞、真实踩箱支撑或爆姿

最值得关注的视觉变化：

- `030_p2`：E168 有倒立/跌倒式异常；E170 保持地面支撑并恢复直立，但 raw contact 数值失败
- `033_p1/p2`：E168 有明显站箱或借箱支撑；E170 避开灾难性形态，但 contact/lower-body 数值仍失败
- `028_p2`：全批姿态最极端，存在大跨步和深倾；动作语义与基线一致，末段可恢复站立
- 剩余共性问题是短时深蹲、贴箱，以及手/腿与箱体的几何重叠；“未见灾难性事件”不等价于推荐 `USE`

## ⚠️ 优先人工审核

| Case | 原标签 | 为什么优先看 | 入口 |
| --- | --- | --- | --- |
| **`20231011_038_p2`** | USE | hand + lower-body 双 fail；深蹲和手/腿贴箱，retention 风险高 | [关键帧](../results/E170/s6_downstream/evidence/visual_qc/box021_20231011_038_p2_keyframes.jpg) · [对比视频](../results/E170/s6_downstream/render/full/paired/box021_20231011_038_p2_E168_vs_E170_PRG.mp4) |
| **`20231018_028_p2`** | DNU | body-z + contact + lower-body 三 fail；全批姿态最极端 | [关键帧](../results/E170/s6_downstream/evidence/visual_qc/box021_20231018_028_p2_keyframes.jpg) · [对比视频](../results/E170/s6_downstream/render/full/paired/box021_20231018_028_p2_E168_vs_E170_PRG.mp4) |
| **`20231018_030_p1`** | DNU | lower-body=`0.330`；深弯时短时腿/箱重叠 | [关键帧](../results/E170/s6_downstream/evidence/visual_qc/box021_20231018_030_p1_keyframes.jpg) · [对比视频](../results/E170/s6_downstream/render/full/paired/box021_20231018_030_p1_E168_vs_E170_PRG.mp4) |
| **`20231018_031_p2`** | DNU | contact=`0.327`、lower-body=`0.159`；蹲姿贴箱 | [关键帧](../results/E170/s6_downstream/evidence/visual_qc/box021_20231018_031_p2_keyframes.jpg) · [对比视频](../results/E170/s6_downstream/render/full/paired/box021_20231018_031_p2_E168_vs_E170_PRG.mp4) |
| **`20231018_033_p1`** | DNU | 数值 contact + lower-body fail，但明显消除了 E168 站箱/借箱 | [关键帧](../results/E170/s6_downstream/evidence/visual_qc/box021_20231018_033_p1_keyframes.jpg) · [对比视频](../results/E170/s6_downstream/render/full/paired/box021_20231018_033_p1_E168_vs_E170_PRG.mp4) |
| **`20231018_033_p2`** | DNU | 数值 contact + lower-body fail，但视觉灾难显著改善 | [关键帧](../results/E170/s6_downstream/evidence/visual_qc/box021_20231018_033_p2_keyframes.jpg) · [对比视频](../results/E170/s6_downstream/render/full/paired/box021_20231018_033_p2_E168_vs_E170_PRG.mp4) |
| **`20231018_035_p2`** | DNU | contact=`0.439`、lower-body=`0.351`，且箱体速度高 | [关键帧](../results/E170/s6_downstream/evidence/visual_qc/box021_20231018_035_p2_keyframes.jpg) · [对比视频](../results/E170/s6_downstream/render/full/paired/box021_20231018_035_p2_E168_vs_E170_PRG.mp4) |
| **`20231020_022_p2`** | USE | lower-body=`0.314`；原 USE 的 retention 风险 | [关键帧](../results/E170/s6_downstream/evidence/visual_qc/box021_20231020_022_p2_keyframes.jpg) · [对比视频](../results/E170/s6_downstream/render/full/paired/box021_20231020_022_p2_E168_vs_E170_PRG.mp4) |

## 📚 28 条逐 case 快速入口

<details>
<summary><strong>展开完整逐 case 审核表</strong></summary>

| Case | E168 | 数值 | Codex 视觉短评 | 审核入口 |
| --- | --- | --- | --- | --- |
| `20231011_034_p1` | USE | FAIL: lower | 短时小腿/箱重叠，仍保持地面支撑 | [帧](../results/E170/s6_downstream/evidence/visual_qc/box021_20231011_034_p1_keyframes.jpg) · [视频](../results/E170/s6_downstream/render/full/paired/box021_20231011_034_p1_E168_vs_E170_PRG.mp4) |
| `20231011_034_p2` | USE | PASS | 姿态稳定，无灾难性回归 | [帧](../results/E170/s6_downstream/evidence/visual_qc/box021_20231011_034_p2_keyframes.jpg) · [视频](../results/E170/s6_downstream/render/full/paired/box021_20231011_034_p2_E168_vs_E170_PRG.mp4) |
| `20231011_036_p1` | DNU | FAIL: lower | 短时小腿/箱重叠，无跌倒 | [帧](../results/E170/s6_downstream/evidence/visual_qc/box021_20231011_036_p1_keyframes.jpg) · [视频](../results/E170/s6_downstream/render/full/paired/box021_20231011_036_p1_E168_vs_E170_PRG.mp4) |
| `20231011_036_p2` | USE | PASS | 手部贴箱较深，整体稳定 | [帧](../results/E170/s6_downstream/evidence/visual_qc/box021_20231011_036_p2_keyframes.jpg) · [视频](../results/E170/s6_downstream/render/full/paired/box021_20231011_036_p2_E168_vs_E170_PRG.mp4) |
| `20231011_037_p1` | USE | PASS | 稳定支撑，搬运受控 | [帧](../results/E170/s6_downstream/evidence/visual_qc/box021_20231011_037_p1_keyframes.jpg) · [视频](../results/E170/s6_downstream/render/full/paired/box021_20231011_037_p1_E168_vs_E170_PRG.mp4) |
| `20231011_037_p2` | USE | PASS | 抬放动作稳定 | [帧](../results/E170/s6_downstream/evidence/visual_qc/box021_20231011_037_p2_keyframes.jpg) · [视频](../results/E170/s6_downstream/render/full/paired/box021_20231011_037_p2_E168_vs_E170_PRG.mp4) |
| `20231011_038_p1` | USE | PASS | 近箱操作，保持地面支撑 | [帧](../results/E170/s6_downstream/evidence/visual_qc/box021_20231011_038_p1_keyframes.jpg) · [视频](../results/E170/s6_downstream/render/full/paired/box021_20231011_038_p1_E168_vs_E170_PRG.mp4) |
| `20231011_038_p2` | USE | FAIL: hand, lower | 深蹲并有手/腿贴箱 | [帧](../results/E170/s6_downstream/evidence/visual_qc/box021_20231011_038_p2_keyframes.jpg) · [视频](../results/E170/s6_downstream/render/full/paired/box021_20231011_038_p2_E168_vs_E170_PRG.mp4) |
| `20231018_028_p1` | DNU | PASS | 快速抬放后恢复，无跌倒 | [帧](../results/E170/s6_downstream/evidence/visual_qc/box021_20231018_028_p1_keyframes.jpg) · [视频](../results/E170/s6_downstream/render/full/paired/box021_20231018_028_p1_E168_vs_E170_PRG.mp4) |
| `20231018_028_p2` | DNU | FAIL: z, contact, lower | 大跨步深倾，末段恢复 | [帧](../results/E170/s6_downstream/evidence/visual_qc/box021_20231018_028_p2_keyframes.jpg) · [视频](../results/E170/s6_downstream/render/full/paired/box021_20231018_028_p2_E168_vs_E170_PRG.mp4) |
| `20231018_029_p1` | DNU | FAIL: z | 保持直立和地面支撑 | [帧](../results/E170/s6_downstream/evidence/visual_qc/box021_20231018_029_p1_keyframes.jpg) · [视频](../results/E170/s6_downstream/render/full/paired/box021_20231018_029_p1_E168_vs_E170_PRG.mp4) |
| `20231018_030_p1` | DNU | FAIL: lower | 深弯时腿/箱短时重叠 | [帧](../results/E170/s6_downstream/evidence/visual_qc/box021_20231018_030_p1_keyframes.jpg) · [视频](../results/E170/s6_downstream/render/full/paired/box021_20231018_030_p1_E168_vs_E170_PRG.mp4) |
| `20231018_030_p2` | DNU | FAIL: contact | 消除 E168 倒立/跌倒式异常 | [帧](../results/E170/s6_downstream/evidence/visual_qc/box021_20231018_030_p2_keyframes.jpg) · [视频](../results/E170/s6_downstream/render/full/paired/box021_20231018_030_p2_E168_vs_E170_PRG.mp4) |
| `20231018_031_p2` | DNU | FAIL: contact, lower | 蹲姿贴箱，局部腿/箱重叠 | [帧](../results/E170/s6_downstream/evidence/visual_qc/box021_20231018_031_p2_keyframes.jpg) · [视频](../results/E170/s6_downstream/render/full/paired/box021_20231018_031_p2_E168_vs_E170_PRG.mp4) |
| `20231018_032_p1` | DNU | FAIL: hand | 手部接触较深，姿态稳定 | [帧](../results/E170/s6_downstream/evidence/visual_qc/box021_20231018_032_p1_keyframes.jpg) · [视频](../results/E170/s6_downstream/render/full/paired/box021_20231018_032_p1_E168_vs_E170_PRG.mp4) |
| `20231018_032_p2` | DNU | FAIL: contact | 长时近箱，未见失稳 | [帧](../results/E170/s6_downstream/evidence/visual_qc/box021_20231018_032_p2_keyframes.jpg) · [视频](../results/E170/s6_downstream/render/full/paired/box021_20231018_032_p2_E168_vs_E170_PRG.mp4) |
| `20231018_033_p1` | DNU | FAIL: contact, lower | 消除 E168 站箱/借箱形态 | [帧](../results/E170/s6_downstream/evidence/visual_qc/box021_20231018_033_p1_keyframes.jpg) · [视频](../results/E170/s6_downstream/render/full/paired/box021_20231018_033_p1_E168_vs_E170_PRG.mp4) |
| `20231018_033_p2` | DNU | FAIL: contact, lower | 灾难性支撑改善，蹲后恢复 | [帧](../results/E170/s6_downstream/evidence/visual_qc/box021_20231018_033_p2_keyframes.jpg) · [视频](../results/E170/s6_downstream/render/full/paired/box021_20231018_033_p2_E168_vs_E170_PRG.mp4) |
| `20231018_034_p2` | DNU | FAIL: contact | 操作稳定，手部接触不足 | [帧](../results/E170/s6_downstream/evidence/visual_qc/box021_20231018_034_p2_keyframes.jpg) · [视频](../results/E170/s6_downstream/render/full/paired/box021_20231018_034_p2_E168_vs_E170_PRG.mp4) |
| `20231018_035_p2` | DNU | FAIL: contact, lower | 快速搬箱并有腿/箱接近 | [帧](../results/E170/s6_downstream/evidence/visual_qc/box021_20231018_035_p2_keyframes.jpg) · [视频](../results/E170/s6_downstream/render/full/paired/box021_20231018_035_p2_E168_vs_E170_PRG.mp4) |
| `20231020_019_p1` | DNU | PASS | 搬运受控，无灾难性回归 | [帧](../results/E170/s6_downstream/evidence/visual_qc/box021_20231020_019_p1_keyframes.jpg) · [视频](../results/E170/s6_downstream/render/full/paired/box021_20231020_019_p1_E168_vs_E170_PRG.mp4) |
| `20231020_019_p2` | USE | FAIL: lower | 深蹲并有腿/箱短时重叠 | [帧](../results/E170/s6_downstream/evidence/visual_qc/box021_20231020_019_p2_keyframes.jpg) · [视频](../results/E170/s6_downstream/render/full/paired/box021_20231020_019_p2_E168_vs_E170_PRG.mp4) |
| `20231020_020_p1` | USE | PASS | 抬放和结束姿态稳定 | [帧](../results/E170/s6_downstream/evidence/visual_qc/box021_20231020_020_p1_keyframes.jpg) · [视频](../results/E170/s6_downstream/render/full/paired/box021_20231020_020_p1_E168_vs_E170_PRG.mp4) |
| `20231020_020_p2` | DNU | PASS | 深手部 SDF，但未见粗大穿箱 | [帧](../results/E170/s6_downstream/evidence/visual_qc/box021_20231020_020_p2_keyframes.jpg) · [视频](../results/E170/s6_downstream/render/full/paired/box021_20231020_020_p2_E168_vs_E170_PRG.mp4) |
| `20231020_022_p1` | USE | PASS | 全程地面支撑稳定 | [帧](../results/E170/s6_downstream/evidence/visual_qc/box021_20231020_022_p1_keyframes.jpg) · [视频](../results/E170/s6_downstream/render/full/paired/box021_20231020_022_p1_E168_vs_E170_PRG.mp4) |
| `20231020_022_p2` | USE | FAIL: lower | 深蹲并有腿/箱短时重叠 | [帧](../results/E170/s6_downstream/evidence/visual_qc/box021_20231020_022_p2_keyframes.jpg) · [视频](../results/E170/s6_downstream/render/full/paired/box021_20231020_022_p2_E168_vs_E170_PRG.mp4) |
| `20231020_023_p1` | USE | PASS | 搬运受控，结束姿态稳定 | [帧](../results/E170/s6_downstream/evidence/visual_qc/box021_20231020_023_p1_keyframes.jpg) · [视频](../results/E170/s6_downstream/render/full/paired/box021_20231020_023_p1_E168_vs_E170_PRG.mp4) |
| `20231020_023_p2` | USE | PASS | lower-body 临界通过，视觉稳定 | [帧](../results/E170/s6_downstream/evidence/visual_qc/box021_20231020_023_p2_keyframes.jpg) · [视频](../results/E170/s6_downstream/render/full/paired/box021_20231020_023_p2_E168_vs_E170_PRG.mp4) |

</details>

---

## ✅ Claims 终审前状态

| Claim | 当前证据状态 | 说明 |
| --- | --- | --- |
| **C0 / C1 provenance 与执行完整性** | 已支持 | `24 new + 4 reuse`，逐项 artifact/SHA/配置契约通过 |
| **C2 DNU recovery** | 数值上限不足；人工待定 | strict 最多 `3/15`；operational recovery 取决于用户标签 |
| **C3 USE retention** | 数值上限不足；人工待定 | strict 最多 `9/13`；operational retention 取决于用户标签 |
| **C4 量化质量完整** | 已支持 | 28 条核心指标、paired delta、worst ranking 全量核验 |
| **C5 视觉改善** | Codex 抽查完成；用户待定 | 未见新增灾难性事件，但完整 paired package 由用户终审 |
| **C6 gate 可诊断** | 诊断完整，机制健康不支持 | gate-health=`0/28`，不能宣称 G 是健康硬约束 |
| **C7 分层覆盖** | 已支持 | person、E168 标签、retarget、sequence 等分组表齐全 |
| **C8 可复现** | 已支持 | full/reuse SHA、配置、视频、评测表及 completion audit 齐全 |

## 🔗 审核材料与下一步

- [人工审核 TSV](../results/E170/s6_downstream/eval/full/user_manual_review_template.tsv) — 唯一允许填写 `manual_*` 的权威表
- [精简工作簿](../results/E170/s6_downstream/eval/full/E170_box021_prg_full_validation.xlsx) — 完整数值、公式和关键帧底稿
- [逐 case 指标](../results/E170/s6_downstream/eval/full/e170_case_metrics.tsv) · [Codex 核验](../results/E170/s6_downstream/eval/full/codex_verification.tsv)
- [分组汇总](../results/E170/s6_downstream/eval/full/e170_group_summary.tsv) · [paired delta](../results/E170/s6_downstream/eval/full/e170_paired_deltas.tsv) · [worst cases](../results/E170/s6_downstream/eval/full/e170_worst_cases.tsv)
- [pre-user 审计](../results/E170/s6_downstream/evidence/completion/review_package_audit_pre_user.json) — `14/14 pass`
- [实验计划](../plan/186_E170_box021_prg_full_validation_plan.md) — 冻结配置、双轨定义和最终裁决规则

下一步只有一项：用户完成 28/28 fresh 标签并保存权威 TSV。随后运行既有 `refresh_E170_review_package.sh final`，生成独立 final log；不重跑 A100 CEM，也不回写本日志的历史证据。
