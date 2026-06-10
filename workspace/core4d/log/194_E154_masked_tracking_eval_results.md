# E154 — Masked-contact(真实 3cm)+ body-tracking 评测修复 + 重评 E152/E153 结果

> 计划:`workspace/core4d/plan/162_E154_masked_tracking_eval_plan.md`
> 状态:**完成(纯评测方法学修订,复用已有轨迹,无重训)**
> 评测:`scripts/eval/lib/core_metrics.py`(共享)+ `eval_E152_*`/`eval_E153_*`(遵循 SKILL §13)
> **本 log 取代 logs 192/193 中基于全序列接触的结论**(192/193 正文不改,见 SKILL 安全边界)。

## 0. 一句话结论

用户指出的两个评测缺陷属实并已修复:(1) 接触指标过去按全序列统计、用了**退化的全 1 mask**,
会奖励"结尾不松手"的失败;(2) 缺机器人本体跟踪。改用**真实 3cm contact mask** + **对固定运动学
真值的 body tracking**后:
- **新增 tracking 门控 success**:box004 有 3 个 combo 因**结尾弯腰不起身**被正确判 fail(过去全 pass)。
  E153 推荐点 **`(−0.010,0.10)` 存活 3/3**;过去标榜"稳健 3/3"的 **`(−0.005,0.05)` 降级为 2/3**。
- **根因诊断**:`release_false`(放手窗口内仍接触)在**所有 run 包括 b1 参考**都偏高
  (0.19~0.51),证明是**训练侧 all-1 mask**(reward 全程奖励接触)而非 gate 的锅。放手普遍失败,
  作诊断不进门控。**这使 E152/E153"接触↑=好"的论断失效,需用真实 mask 重训(单独后续)。**

## 1. 根因:全 1 mask 同时污染训练与评测

- `spider/process_datasets/core4d.py` `contact_detection_mode` 默认 `"one"` → `trajectory_kinematic.npz`
  的 `contact` 字段 = **全 1**(注释"协作搬运默认始终接触")。
- `spider/io.py:load_data` 把 `contact` 同时读给 **reward 的 contact target**(`contact_ref_torch`)。
  ⇒ 优化器从训练起被奖励**全程保持接触**,"放手"从未被激励。
- 真实 3cm mask(`generate_core4d_contact_masks.py`,SMPL-X 手顶点对物体表面 <0.03)一直存在于
  `results/E143/contact_masks/<case>/raw_contact_mask_3cm.npz`,但未被训练/评测使用。
- 真实 mask 的放手段(person2):box004 尾 ~17 帧、box021 尾 4 帧、box023 尾 ~56 帧。
  `ref_contact_frac`(真实接触帧占比)box004=0.59、box021≈0.68、box023≈0.45 —— **远非全 1**。
- **mask 时间对齐验证**:b1 box004 手-物 SDF 接近段 0.245 → 搬运段 **−0.003** → 放手段 0.147,
  最小值正落在接触窗口,确认 mask 与轨迹逐帧对齐。

## 2. 新增指标(`core_metrics.evaluate_sequence`)

- **body tracking(对固定 kin 真值 `trajectory_kinematic.npz` 的 `qpos`,robot dof [0:36])**:
  `track_{root_pos,root_quat,joint,pelvis_z}_err_{mean,terminal}`(末段=最后15%帧)。复用
  `get_humanoid_tracking_err.py` 约定 + `spider.math.quat_sub`。
- **masked 接触(真实 3cm,union 双手)**:`ref_contact_frac`、`hand_object_physics_contact_in_mask_frac`、
  `hand_object_{false,approach,release}_false_contact_frac`、`hand_geom_penetration_{2mm,5mm}_in_mask_frac`。
- 向后兼容:不传 ref 时全 NaN;E147/E151 等旧评测照常运行(已验证 `missing=0`)。

## 3. 重评结果

### 3.1 E153 18-grid:tracking 门控 vs 旧 pen2mm 门控(3-case 聚合)

| min_sdf | max_viol | **succ_tracked** | succ_pen2mm(旧) | pz_term mean | pz_term worst | inmaskC | release_false | gate_valid | fallback |
|---:|---:|:--:|:--:|---:|---:|---:|---:|---:|---:|
| −0.005 | 0.05 | **2/3** | 3/3 | 0.045 | 0.101 | 0.813 | 0.187 | 0.700 | 0.005 |
| −0.005 | 0.10 | **3/3** | 3/3 | 0.028 | 0.048 | 0.750 | 0.510 | 0.762 | 0.000 |
| −0.010 | 0.05 | 1/3 | 2/3 | 0.038 | 0.081 | 0.846 | 0.335 | 0.853 | 0.003 |
| **−0.010** | **0.10** | **3/3** | 3/3 | 0.024 | 0.035 | 0.813 | 0.499 | 0.895 | 0.001 |
| −0.015 | 0.05 | 2/3 | 2/3 | 0.031 | 0.049 | 0.829 | 0.377 | 0.930 | 0.000 |
| −0.015 | 0.10 | 1/3 | 2/3 | 0.045 | 0.105 | 0.866 | 0.281 | 0.933 | 0.000 |

### 3.2 per-case 明细(18 行;tracked=false 原因分两类)

| case | combo | tracked | pz_term | inmaskC | release_false | fail 原因 |
|---|---|:--:|---:|---:|---:|---|
| box021 | sdf005_v05 | ✅ | 0.032 | 0.691 | 0.000 | — |
| box021 | sdf005_v10 | ✅ | 0.031 | 0.600 | 0.750 | — |
| box021 | sdf010_v05 | ❌ | 0.028 | 0.727 | 0.250 | pen2mm(起身OK) |
| box021 | sdf010_v10 | ✅ | 0.035 | 0.709 | 0.750 | — |
| box021 | sdf015_v05 | ❌ | 0.040 | 0.727 | 0.500 | pen2mm(起身OK) |
| box021 | sdf015_v10 | ❌ | 0.030 | 0.836 | 0.000 | pen2mm(起身OK) |
| box004 | sdf005_v05 | ❌ | **0.101** | 0.871 | 0.375 | **弯腰(tracking)** |
| box004 | sdf005_v10 | ✅ | 0.048 | 0.758 | 0.688 | — |
| box004 | sdf010_v05 | ❌ | **0.081** | 0.871 | 0.625 | **弯腰(tracking)** |
| box004 | sdf010_v10 | ✅ | 0.031 | 0.790 | 0.562 | — |
| box004 | sdf015_v05 | ✅ | 0.049 | 0.823 | 0.500 | — |
| box004 | sdf015_v10 | ❌ | **0.105** | 0.823 | 0.750 | **弯腰(tracking)** |
| box023 | (全 6) | ✅×6 | 0.002–0.005 | 0.88–0.94 | 0.09–0.19 | — |

- **box004 是 tracking 瓶颈**:3 个 combo 结尾弯腰不起身(pz_term 0.08~0.105)被判 fail——过去全 pass。
- box021 的 fail 全是 pen2mm 驱动(起身 pz_term<0.04 正常);box023 全过。

### 3.3 E152(gateA_b1 vs b1)

| method | succ(0mm,旧) | succ(2mm,旧) | **succ_tracked** | 备注 |
|---|:--:|:--:|:--:|---|
| gateA(vs baseline) | 0/3 | 0/3 | 0/3 | pen 门控本就严,未变 |
| gateA_b1(vs b1) | 2/3 | 3/3 | **3/3** | 该 run 3 case pz_term 0.004–0.047 全过 |

- **单 seed 脆弱性**:E152 box004 gateA_b1(min_sdf −0.010, max_viol 0.05)pz_term=**0.047**(过),
  但 E153 同配置 sdf010_v05 = **0.081**(不过)。同配置不同 seed,box004 起身在 0.05~0.08 边界抖动。

## 4. Claims 验证

| Claim | 结果 | 裁定 |
|---|---|---|
| C1 真实 mask≠全1 | ref_contact_frac 0.45–0.68;mask 时间对齐(SDF 搬运段最小) | **成立** |
| C2 指标揭露失败 | box004 弯腰 3 combo pz_term>0.08→tracked fail;视觉佐证 | **成立** |
| C3 重评裁定 | `(−0.010,0.10)` 存活 3/3;`(−0.005,0.05)` 降级 2/3 | **成立** |
| C4 放手普遍失败(诊断) | release_false 含 b1 都高(0.19–0.51)→训练侧 all-1 mask | **成立** |

## 5. 可视化(SKILL §9)

box004 结尾帧(`results/E154/visual/box004_*_terminal.jpg`,ref|sim):
- **sdf005_v05(tracked fail)**:ref 直立于箱旁,**sim 塌陷弯腰几乎扑在箱上**、手张开——起身失败一目了然。
- **sdf010_v10(tracked pass)**:sim 与 ref 同为近直立姿态,pelvis 高度吻合(pz_term 0.031);
  但 sim 手仍滞留箱面(release_false 0.562)——印证"起身但未真正放手"。

## 6. 严谨性 / 单 seed

- 报全 18 行 + per-case worst,无 cherry-pick;tracking 门控阈值 0.08 ≈ 2× 最差 b1(0.043),
  给参考级跟踪留余量。
- 单 seed:box004 起身在阈值边界(E152 0.047 vs E153 0.081 同配置),`success_tracked` 对 box004
  敏感,多 seed 复核留作后续。
- **release 不进门控**:它在 b1 也高,是训练 bug 的体现而非 gate 可调;硬门控会让全部(含 b1)fail、
  抹掉 combo 间区分(用户已确认仅用 tracking 门控)。

## 7. 结论与下一步

- **方法学**:评测必须用真实 3cm mask + body tracking。`success_tracked` 取代旧全序列接触 success。
- **对 E152/E153 的修正**:"接触↑"部分是 all-1 mask 下"不松手"的伪收益;真正可区分的行为指标是
  body tracking。`(−0.010,0.10)` 仍是最佳 gate 配置(tracking 3/3 + gate 健康)。
- **下一步(待用户决策)**:
  1. **训练侧根因修复(大)**:用真实 `spider_contact_mask_3cm` 替换 `io.py` 的接触 target 重训,
     让机器人被奖励在放手窗口松手。可能新开 E + 分支。
  2. box004 起身边界的多 seed 复核。
  3. 选定配置接 Holosoma RL 前,先确认放手行为(否则下游继承"不松手")。

## 8. 结果路径

| 类型 | 路径 |
|---|---|
| 计划 | `plan/162_E154_masked_tracking_eval_plan.md` |
| 代码 | `scripts/eval/lib/core_metrics.py`(track_*/mask 指标+helpers);`eval_E152_*`/`eval_E153_*`(门控) |
| E153 eval | `results/E153/gate_threshold_sweep/eval/full/e153_{grid_delta_vs_b1,combo_summary,method_metrics}.tsv` |
| E152 eval | `results/E152/axis1_hand_object_physics_gate/eval/full/e152_{delta_vs_reference,delta_summary,method_metrics}.tsv` |
| 视觉 | `results/E154/visual/box004_{sdf005_v05,sdf010_v10}_terminal.jpg` |
| 真实 mask | `results/E143/contact_masks/<case>/raw_contact_mask_3cm.npz`(key `spider_contact_mask_3cm`) |
