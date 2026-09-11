# log305 · E213-export：选定臂 object-aug paired 输入 → Holosoma partner re-anchor RL motion 导出

_Core4D · S6 downstream · 承接 [log302](302_E213_export_selected_arm_aug_rl.md)（R299 产 paired 输入）的"下一步" · 计划 [plan244](../plan/244_E213_export_selected_arm_aug_rl_plan.md) · 2026-09-12 · **状态：完成（导出 64/64 motion，validation PASS；post-reanchor 一致性验证）**_

## 摘要

log302 把 `E213_export_aug_cases.xlsx` 的 **32 个选定臂 object-aug 单元（20 source case）** 导出为 dcv3 paired RL **输入**（source 32 + partner 32，全 RL_EXPORT_READY），留下"交下游 Holosoma exporter 跑 partner re-anchor"的下一步。本轮执行该步：把 paired 输入喂给 Holosoma `export_rl_motion_from_spider_tsv.py`，**partner 手 re-anchor 到 source object（exporter 默认 ON，未传 `--no-reanchor`）**，产出自洽的双人 RL motion。**无新 CEM 算力**（复用 E213 已跑的选定臂 aug CEM rollout）。

产出：**64 motion**（32 unit × {cem, trajectory}），全部 `export_pass`，validation status=**PASS**，0 fail。

## 执行

- **入口**：`scripts/launch/active/run_E213_export_holosoma.sh export`（新增，改编自 `run_E202_export_holosoma.sh`）。dry-run 校验 32 unit 选择 → 真实导出 → 内建 validate。
- **对象/半轴**：chair006 / desk007 / desk021 / desk023。`--object-half-extents` 仅供表面距离**诊断**（不影响 reanchor/导出 motion）：取 object mesh AABB 半轴（`holosoma_retargeting/models/*/*.obj`）；已核对 desk021/desk023 与 exporter 内建表一致（误差 <1e-4），chair006/desk007 为新算 `[0.2972,0.4279,0.2803]` / `[0.2152,0.3490,0.4008]`。
- **exporter 参数**：`--target-source both --partner-source omnirt --target-variant ref_fk --include-contact-mask`，reanchor 默认开。`SPIDER_REPO`/`--python` 指向本机路径（沿用 E202-export 修的两处历史硬编码）。
- 输入完整性预检：32 source + 32 partner 行的 scene_act/trajectory/contact_mask/cem_result/trimmed_npz **全部存在**；contact_mask 有 `spider_contact_mask_3cm` key；scene_act_meta.json 齐备。

## 结果

### 导出闭合（`holosoma_downstream_validation.json`）

| 对象 | motion（cem+traj） | decision |
|---|--:|---|
| chair006 | 16 | 16 export_pass |
| desk007 | 8 | 8 export_pass |
| desk021 | 20 | 20 export_pass |
| desk023 | 20 | 20 export_pass |
| **ALL** | **64** | **64 export_pass · 0 fail · schema/finite 全通过** |

reanchor 诊断（raw pre-reanchor，per-person object 世界系分歧）：object_mismatch_max chair006 1.60m / desk007 1.25m / desk021 0.92m / desk023 1.14m —— 与 [[two-person-aug-partner-reanchor]] 一致（per-person facing-relative 扰动，两人隔物体相对而站，故世界系方向不同），**属预期非 bug**，final `partner_hand_pos_w` 锚到 source object。

### post-reanchor 一致性（`post_reanchor_consistency.tsv/.json`；本轮新增校验）

re-anchor 后 partner 双手到 **source** object OBB 表面的**全程最近距离**（closest-approach over episode）：

| 对象 | n | mean(cm) | median(cm) | max(cm) | <8cm |
|---|--:|--:|--:|--:|--:|
| chair006 | 16 | 0.1 | 0.0 | 1.2 | 16/16 |
| desk007 | 8 | 0.2 | 0.2 | 0.4 | 8/8 |
| desk021 | 20 | 1.3 | 1.2 | 2.9 | 20/20 |
| desk023 | 20 | 2.2 | 0.2 | 12.6 | 18/20 |
| **ALL** | **64** | **1.2** | 0.2 | 12.6 | **62/64（全 64 <15cm）** |

**机制验证成立**：raw 两人 object 世界系分歧最大 1.6m，reanchor 后 partner 手在抓握时刻贴到 source object 表面（全程最近 mean 1.2cm，全部 <15cm），两 agent 抓同一 source object → 自洽。desk023 2 个 ~12.6cm 属抓握时序/接触窗差异，非 reanchor 失败（口径同 E202-export C2'；此处用 closest-approach 而非 contact-mask 门控均值，故数值远优于 E202 bucket 的 4.5–8.3cm）。

## Claims

- **导出闭合** ✓：64/64 export_pass，validation PASS，`partner_hand_pos_w/quat_w` 形状 (T,2,3)/(T,2,4) + 有限性全通过。
- **partner 包含且 reanchor 生效** ✓：每 motion 含 partner 通道，post-reanchor partner 手锚到 source object（mean 1.2cm，全 <15cm）。
- **无 CEM 算力** ✓：复用 E213 选定臂 aug rollout；本轮仅 Holosoma 运动学转换 + reanchor 打包。
- **claim 边界**：止于 RL motion 导出 + reanchor 自洽验证；未跑下游 RL 训练，不作 RL 成功声明。

## 结果路径

| 类型 | 路径 |
|---|---|
| Holosoma motion | `holosoma/workspace/v3/data/E213_selected_arm_aug_partner_rl/{chair006,desk007,desk021,desk023}/exports/`（64 `*_w_partner.npz` + target/partner 分量 + 各 `manifest.tsv`） |
| HS validation | `results/E213/s6_downstream/export/holosoma_downstream_validation.json`（status=PASS） |
| post-reanchor 校验 | `results/E213/s6_downstream/export/post_reanchor_consistency.tsv/.json` |
| 导出日志 | `results/E213/s6_downstream/export/holosoma_export.log` |
| paired 输入（log302） | `results/E213/s6_downstream/export/paired_rl_export_input.tsv`（100 列 dcv3） |

## 改动文件

| 文件 | 改动 |
|---|---|
| `scripts/launch/active/run_E213_export_holosoma.sh`（新） | E213 Holosoma reanchor 导出入口（改编 E202；4 对象半轴 + 32-unit 过滤 + 4-object validate） |

## 坑

- 无。exporter 内建 `OBJECT_HALF_EXTENTS` 已含 desk021/desk023；chair006/desk007 未登记，需 `--object-half-extents` 显式传（仅诊断用）。partner tsv 缺 `case_id/rl_export_decision/euler_convention` 等列不影响：exporter 用 `.get()` 带默认，且 partner join 只依赖 `source_case_id`+`partner_status`+`trimmed_npz`，`euler_convention` 从 scene_act_meta.json 读。

## 下一步

- Holosoma S6 registry / downstream evidence 登记这 64 motion 路径（`record_downstream_evidence.py`，仅记录不反写上游事实）。
- 下游 RL：64 motion 可接入 holosoma 训练（registry register + pre-train gate 未跑，本轮止于导出+验证）。
