# E213-export · 选定臂 object-aug 单元 → dcv3 paired RL export

**exp_name**: `core4d` · **Run**: R299（E213 同 run 的导出步）· **Plan NN**: 244 · **Log NN**: 302
**分支**: `experiment/E199-omniretarget-object-augmentation`（沿用）

---

## Context（为什么做）

E213 已交付 source 侧选定臂 aug CEM（100/100 cem_ok）+ partner 侧运动学增强（11×5=55 trimmed）。
用户在 `E213_export_aug_cases.xlsx` 里**人工挑选了 32 个 (case, variant) 单元（20 个 source case）**，
要把它们**导出为 dcv3 paired RL export（含 partner 信息）**到
`workspace/core4d/results/E213/s6_downstream/export/`。

这是 E202-export（平移增强 + partner re-anchor RL 导出）与 E212（混合臂 paired 导出）的**合体**：
- source = E213 选定臂 aug CEM rollout（每 case 各自臂 G1/G08/G06/G04）；
- partner = 对手人在**同一 variant** 下的**运动学 aug retarget trimmed npz**（不跑 CEM），
  两人物体一致性由**下游 Holosoma partner re-anchor** 保证并 post-reanchor 校验（[[two-person-aug-partner-reanchor]]）。

**本步只产出 paired 导出 INPUT**（与 E202-export/E212 一致），不在此运行 Holosoma re-anchor。

---

## 选择权威 + partner 拓扑（已核实）

- **选择权威** = xlsx `Sheet1`（32 行数据，列 `object/case_id/arm/variant`），sha 锁定。
  20 个 source case，全部 ⊂ E212 的 21 case（缺 desk007_034_p1 PRG，未被挑选）。
- **每单元 partner = 对手人同 variant 的运动学 aug trimmed**，两个来源（已核实 32/32 可解析，0 缺口）：
  - **partner-only（10 单元的对手）**：`results/E213/data_preprocess/partner_aug/{omnirt_vX}/holosoma_{base}/trimmed/{holo}_{trans_k|rot_k}.npz`
    （E213 partner_aug 产物，11 case 各 5 变体齐全；`e213_partner_aug_artifacts.tsv` 记 base/variant/provenance）。
  - **partner 本身也是 source（10 单元的对手）**：E208 source aug retarget 的 `trimmed_npz`
    （`e208_aug_artifacts.tsv` 按 (case, aug_variant) 索引，status=built）。
- variant 命名映射：xlsx `trans0/trans1/trans2/rot0/rot1` ↔ trimmed 文件 `trans_0/…/rot_1`。

---

## 设计（复用最大化）

复用 **dcv3 partner adapter**（`dcv3/stages/s6_downstream/finalize_reused_partner_rl.py`，经
E206 exporter 的 `PARTNER`）的 `build_partner_row / paired_row / alignment_audit / trim_window_path`
+ `PARTNER_FIELDS / PAIRED_EXTRA_FIELDS / SCHEMA_VERSION`（= `core4d_data_construction_v3.0` = dcv3）。
source 行走 **E212 mixed-arm 口径**（每 case 选定臂路径来自 E213 merged manifest，静态 provenance
从 E212 seed 行继承）；aug + partner 语义走 **E202-export 口径**。

### 新建 `scripts/experiments/E213/`

| 文件 | 职责 |
|---|---|
| `e213_export_common.py` | 契约：读 xlsx（sha pin）→ 32 单元；merged manifest 索引（selected-arm aug CEM）；两来源 partner-aug trimmed 解析器；E212 seed 行索引（静态 provenance）；person/flip 工具 |
| `export_selected_arm_aug_rl.py` | 主驱动：逐单元建 source 行 + partner 行 + alignment audit → 写 dcv3 paired 导出文件集 |
| `test_e213_export_contract.py` | 离线自检：xlsx sha、32 单元解析、20 case ⊂ 21、partner 变体 32/32 存在、变体命名映射、schema 字段齐全 |

### source 行（每单元）

- unit_id = `{case_id}__aug_{variant}_{arm}`。
- 路径来自 **E213 merged manifest**（`e213_source_arm_cem.tsv`，status=cem_ok）：`selected_scene_act` / `trajectory` /
  `contact_mask` / `outdir_npz`（= cem_result_npz）。
- 静态 provenance（object_name/date/seq/person/person_idx/target_scene/stage2b_*）从 **E212 seed 行**
  （该 case 在 `E212/.../paired_rl_export_input.tsv`）继承 byte-for-byte。
- arm 列（arm/arm_experiment/arm_gravcomp/arm_scene_name）来自 E212 seed；`arm_scene_name` 必须出现在
  `selected_scene_act` 文件名里（per-case scene guard，仿 E212）。
- 评审诚实：全部 `manual_use_decision=USER_SELECTED_E213_AUG_XLSX`（xlsx 人工挑选是唯一权威，
  非单一预注册门）；`visual_qc_status=xlsx_selected`。
- **C4 物理硬门**：从 E213 eval rollout（`e213_aug_rollout.tsv`）取该单元 `fall_flag`/root/eef 发散，
  fall>0 或 >60cm 发散→硬排除并登记（仿 E202；预期 0 排除，因 xlsx 已人工挑选）。gate 逐门verdict 一并写入。

### partner 行（每单元）

- partner_case = 对手人 flip；partner variant = 同 variant。
- 合成 Stage2b 形 evidence（`stage2b_status=pass`、`retarget_variant_id`=该 partner 的 v1/v2、
  `trimmed_npz`/`retargeted_npz`/`converted_npz`/`trim_window_json`/`target_task` 指向对应来源），
  调 `PARTNER.build_partner_row(...)`，`generation_mode=e213_selected_arm_aug_partner_kinematic_same_variant`。
- `alignment_audit` 复用 dcv3 共窗对齐（source rollout 帧数 vs partner npz 帧数 vs 各自 trim_window）。

### 输出（`results/E213/s6_downstream/export/`）

`rl_export_input.tsv/.json`（32 source 行）、`partner_omnirt/rl_partner_omnirt_manifest.tsv/.json`、
`paired_rl_export_input.tsv/.json`（dcv3 消费）、`partner_resolution_audit.tsv`、
`rl_export_summary.json`、`validation_report.json`、`selection_snapshot.xlsx`（xlsx 快照）。
所有 in-repo 绝对路径经 `to_repo_rel` 归一（results/ 是 /mnt symlink，仿 E212 `norm()`）。

---

## 成功标准（Claims，跑前预注册）

| # | 判据 | 门 |
|---|------|----|
| E0 契约 | xlsx sha pin；32 单元 / 20 case 解析；20 case ⊂ E212 21 case；变体命名映射正确 | 全过 |
| E1 source 可加载 | 32 单元 selected_scene_act/trajectory/contact_mask/cem_result 全部存在且 `arm_scene_name` 命中 scene 文件名 | 100% |
| E2 partner 解析 | 32 单元 partner-aug trimmed（两来源）全部存在、可加载、帧数>0 | 100% |
| E3 C4 物理门 | 逐单元 fall=0、root/eef 无 >60cm 发散（未过→硬排除并登记，禁止静默） | 报告，越门显式记 |
| E4 paired 就绪 | 每单元 `pair_status=PAIR_COMPLETE` 且 `paired_rl_export_decision=RL_EXPORT_READY`；无 misalignment | 100%（否则 PARTIAL 并列出） |
| E5 schema | 输出 schema_version==`core4d_data_construction_v3.0`(dcv3)，字段与 E212/E202-export 逐列一致 | 全过 |

> 两人物体一致性**不在本层门控**（per-person 增强物体通道本就分叉，见 E200/E202），
> 由下游 re-anchor 保证 + post-reanchor 校验；本层只硬门 C4 物理 + 结构 pair 完整。

---

## 运行命令（固化脚本，规则 8）

```bash
.venv/bin/python workspace/core4d/scripts/experiments/E213/test_e213_export_contract.py
.venv/bin/python workspace/core4d/scripts/experiments/E213/export_selected_arm_aug_rl.py --dry-run
.venv/bin/python workspace/core4d/scripts/experiments/E213/export_selected_arm_aug_rl.py
```

## 验证

1. 契约自检全绿；2. `--dry-run` 打印 32 单元 + 每单元 arm/partner 来源；3. 正式跑 → paired 32/32 READY、0 blocked、0 misalign；
4. 抽查 paired tsv 一行：source 选定臂 aug CEM + partner 同 variant 运动学 trimmed + 共窗对齐字段非空；5. summary/validation JSON 记 sha256。

## 风险 / 注意

- partner=source 的对手若某 variant 在 E208 不可行→该单元 partner 缺失→blocked（已核实 32/32 无此情况）。
- 两来源 retarget 变体可能不同（partner v1/v2 与 source 各自）→ 在 partner 行如实记 `partner_retarget_variant_id`。
- 不触碰 E202/E206/E208/E212 历史产物；只写 `results/E213/s6_downstream/export/`。
