# E213-export · 选定臂 object-aug 单元 → dcv3 paired RL export

**Run**: R299（E213 导出步）· **Plan**: [244](../plan/244_E213_export_selected_arm_aug_rl_plan.md) · **日期**: 2026-09-08
**结论**: ✅ **RL_EXPORT_READY** — 32 个人工挑选单元全部导出为 dcv3 paired RL 输入，partner 配对 32/32、对齐 32/32、0 排除、0 misalign。

---

## 1. 做了什么

把 `E213_export_aug_cases.xlsx` 里人工挑选的 **32 个 (case, variant) 单元（20 个 source case）**
导出为 **dcv3（`core4d_data_construction_v3.0`）paired RL 输入**（含 partner 信息），
输出到 `workspace/core4d/results/E213/s6_downstream/export/`。

- **source** = E213 选定臂 aug CEM rollout（各 case 各自臂 G1/G08/G06/G04；scene_act/trajectory/
  contact_mask/cem_result 来自 E213 merged manifest），静态 provenance 从 E212 seed 行继承。
- **partner** = 对手人**同一 variant** 的**运动学 aug retarget trimmed npz**（不跑 CEM），两来源：
  E213 partner_aug（partner-only）或 E208 source aug retarget（partner 本身是 source）。
- 两人物体一致性由**下游 Holosoma partner re-anchor** 保证 + post-reanchor 校验（[[two-person-aug-partner-reanchor]]）；
  本层只硬门 C4 物理 + 结构 pair 完整，产 paired **输入**（不在此跑 re-anchor，与 E202-export/E212 一致）。

复用：dcv3 `finalize_reused_partner_rl`（`build_partner_row/paired_row`）+ E206 exporter 的
`alignment_audit`/字段表（经 importlib 作库）+ E212 mixed-arm 口径 + E202-export aug-partner 语义。

## 2. Claims 验证（plan244 E0–E5）

| # | 判据 | 结果 |
|---|------|------|
| E0 契约 | xlsx/seed sha pin；32 单元 / 20 case；20 case ⊂ E212 21 case | ✅ 自检全过 |
| E1 source 可加载 | 32/32 scene_act/trajectory/contact_mask/cem_result 存在且 `arm_scene_name` 命中 scene 文件名 | ✅ 32/32 |
| E2 partner 解析 | 32/32 partner-aug trimmed（15 E213 partner_aug + 17 E208 source-aug）存在、变体名匹配 | ✅ 32/32，0 mismatch |
| E3 C4 物理门 | 逐单元 fall=0、root/eef 无 >60cm 发散 | ✅ **0 排除** |
| E4 paired 就绪 | pair_status=PAIR_COMPLETE + paired_rl_export_decision=RL_EXPORT_READY，无 misalign | ✅ **32/32**，共窗帧 46–150 全 ≥2 |
| E5 schema | schema_version==dcv3；paired 100 列 | ✅ 全 dcv3 |

> C4 narrow 逐门 verdict 一并写入（8 单元 narrow=false，物理 C4 仍过）；xlsx 人工挑选是唯一放行权威，
> narrow 记录不作门（`candidate_decision=USER_SELECTED_E213_AUG_XLSX`，非单一预注册门，同 E212 口径）。

## 3. 交付统计

- source 32 行 / partner 32 行 / paired 32 READY。物体：desk021 10、desk023 10、chair006 8、desk007 4。
- 臂：G1 18、G06 11、G08 2、G04 1；变体：trans2 12、trans0 8、trans1 7、rot1 3、rot0 2。
- partner 变体：omnirt_v1 19、omnirt_v2 13；partner 来源：E208 source-aug 17、E213 partner_aug 15。

## 4. 结果路径

`results/E213/s6_downstream/export/`：`rl_export_input.tsv/.json`（32 source）、
`partner_omnirt/rl_partner_omnirt_manifest.tsv/.json`（32 partner）、
`paired_rl_export_input.tsv/.json`（100 列，dcv3 消费）、`partner_resolution_audit.tsv`（32 全 READY）、
`rl_export_summary.json`、`validation_report.json`（7 checks 全 True）、`selection_snapshot.xlsx`。

## 5. 脚本（规则 8）

`scripts/experiments/E213/`：`e213_export_common.py`（契约：xlsx 权威 + 两来源 partner-aug 解析 +
source 对齐 trim 解析）、`export_selected_arm_aug_rl.py`（主驱动）、`test_e213_export_contract.py`（E0–E2/E5 自检，ALL PASS）。

## 6. 坑

| 现象 | 根因 | 修复 |
|------|------|------|
| source 对齐 trim_window 定位失败（v2 rescue 单元） | rescued 变体的 SPIDER `target_task` 标 `omnirt_v2`，但 E208 retarget 树的 holosoma 子目录仍用**主变体**名（v1），仅父目录换成 v2 | `source_aug_evidence` 改按 `(effective_variant 父目录, case_id)` glob，不再从 target_task 拼 base |

## 7. 下一步

paired 输入就绪，可交下游 Holosoma exporter 跑 partner re-anchor + post-reanchor 一致性校验（E202-export 口径）。
