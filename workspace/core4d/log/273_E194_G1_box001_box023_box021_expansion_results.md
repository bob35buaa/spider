# E194 G1 扩展结果：72-case 重力补偿跨 box 泛化与 noPRG / PRG / G1 对比

_Core4D · Phase 57 expansion · 2026-08-11 · 计划 [plan222](../plan/222_E194_G1_box001_box023_box021_expansion_plan.md) · 原 E194 [log268](268_E194_object_gravity_compensation_results.md)_

## TL;DR

- **执行与证据闭合**：G1 Full 72/72 完成，local/Ada0/Ada1=`36/18/18`，失败 0；公共 evaluator 得到 noPRG/PRG/G1 各 72 行、144 个同 case 配对、1,728 个 gate migration、评估错误 0；G1 MP4 72/72。
- **G1 在三个物体上都有效**：相对 PRG，box001/box023/box021 的 z MAE 分别下降 `1.220/0.540/1.299 cm`，paired 95% CI 均低于 0；3D position 分别下降 `1.299/1.024/1.231 cm`。
- **三版本结论**：PRG 相对 noPRG 的 z 几乎持平（object macro `-0.032 cm`），但 3D position 三个物体均变差（`+0.451/+0.752/+1.038 cm`）。G1 不仅修复 PRG 的 z，下游 3D 也低于 noPRG：三个物体的 G1−noPRG z 为 `-1.148/-0.586/-1.422 cm`，3D 为 `-0.849/-0.272/-0.192 cm`。
- **12-gate 是结构性 mixed，而不是一致改善**：总 gate rate 为 `82.1%→83.6%→83.3%`，strict-12 为 `20.8%→26.4%→26.4%`；G1 显著改善 `lower_body`（PRG→G1 `69.4%→87.5%`, p=`0.002350`），但显著回退 `object_ori`（`98.6%→84.7%`, p=`0.001953`），且 box023 strict-12 `43.8%→18.8%`。
- **判决**：`PROMOTE_G1_WITH_CASE_LEVEL_EXCEPTIONS`。推广范围仅限本轮冻结的 `E167A_zOnlyBody + PRG + rubber_hull`、box001/box023/box021；保留三条数值例外，不宣称对所有物体或 RL 训练普适。

## 1. 实验问题与设计

原 E194 已在 box004/box024 证明 object body `gravcomp="1"` 能修复重力下垂，但样本仅 15 条。本轮保持实验 ID `E194`，把唯一新臂 G1 扩到：

| Object | Cases | PRG authority | noPRG authority | Retarget variants |
|---|---:|---|---|---|
| box001 | 28 | E173 Full | E189 Full | v1=21, v2=7 |
| box023 | 16 | E173 Full | E179 Full | v1=15, v2=1 |
| box021 | 28 | E170/E169 audited Full | E168 production | v1=25, v2=3 |

冻结设置：G1 仅增加 object `gravcomp=1`，`kp_pos/kp_rot=500/50`，PRG 开启，CEM `1024×32`、seed 0。72 个 case 按 `36/18/18` 分给本地 GPU、Ada6000 GPU0、Ada6000 GPU1；每个 profile 均覆盖三个物体。

noPRG 不是重新运行：它使用与 E194 authority 完全同 case、同 retarget variant 的历史 Full 产物，再由本轮同一个 public-core scorer 重新计分。E189 三条 `READY_FOR_FULL` 是 stale status；NPZ、resolved config、scene、trajectory、mask、video 均存在并通过逐文件审计。

## 2. Observed results（观测事实）

### 2.1 技术闭合

| 检查 | 结果 |
|---|---:|
| G1 Full terminal rows | 72/72 |
| local / Ada0 / Ada1 | 36 / 18 / 18 |
| execution failures | 0 |
| PRG authority parity | 72/72，z 容差 `1e-4 cm` |
| noPRG / PRG / G1 scored rows | 72 / 72 / 72 |
| same-case paired rows | 144（noPRG→PRG 72；PRG→G1 72） |
| gate migrations | 1,728（2×72×12） |
| eval errors / non-finite / diverged | 0 / 0 / 0 |
| G1 self videos | 72/72 |
| mandatory visual review | 36/36 cases；432 stage frames |
| XLSX validation | 11,128 formulas；LibreOffice formula errors=0；含72行PRG/G1数值失败模式对照 |
| E173 PRG manual review join | 28 unique box001 cases；paired rows 56；USE 38 / DO_NOT_USE 18；其余88行显式无E173记录 |

### 2.2 三臂主结果

主指标为 `track_obj_z_abs_err_cm_mean`，单位 cm，越低越好。`ΔPRG=noPRG→PRG`，`ΔG1=PRG→G1`。

| Object | n | noPRG z | PRG z | G1 z | ΔPRG z | ΔG1 z [paired 95% CI] | G1−noPRG z |
|---|---:|---:|---:|---:|---:|---:|---:|
| box001 | 28 | 4.773 | 4.846 | **3.626** | +0.072 | **−1.220** [−1.525, −0.865] | −1.148 |
| box023 | 16 | 5.863 | 5.817 | **5.276** | −0.046 | **−0.540** [−0.974, −0.131] | −0.586 |
| box021 | 28 | 6.360 | 6.237 | **4.938** | −0.124 | **−1.299** [−1.929, −0.734] | −1.422 |
| object macro | — | 5.665 | 5.633 | **4.613** | −0.032 | **−1.020** | −1.052 |

3D object position 显示 PRG 与 G1 的作用方向不同：

| Object | noPRG 3D | PRG 3D | G1 3D | PRG−noPRG | G1−PRG | G1−noPRG |
|---|---:|---:|---:|---:|---:|---:|
| box001 | 10.807 | 11.257 | **9.958** | +0.451 | **−1.299** | −0.849 |
| box023 | 12.310 | 13.062 | **12.038** | +0.752 | **−1.024** | −0.272 |
| box021 | 14.637 | 15.675 | **14.444** | +1.038 | **−1.231** | −0.192 |
| object macro | 12.584 | 13.331 | **12.147** | +0.747 | **−1.185** | −0.438 |

### 2.3 G1 安全/质量观测

| Object | Δ3mm contact | Δhand penetration | Δleg penetration | 12-gate pass-rate Δ | New falls |
|---|---:|---:|---:|---:|---:|
| box001 | +0.040 | −0.078 | −0.047 | +0.3 pp | 0 |
| box023 | −0.032 | +0.016 | −0.015 | −1.0 pp | 0 |
| box021 | −0.009 | +0.016 | −0.073 | −0.3 pp | 0 |

三个物体均满足预注册 C3–C7 阈值。profile 分层的 z delta 在九个 object×profile cell 中全部为负：box001 `−1.438/−1.354/−0.650`，box023 `−0.897/−0.219/−0.149`，box021 `−1.623/−0.771/−1.177 cm`（local/Ada0/Ada1）。

### 2.4 具体 12-gate 对比

以下均为同一批 72 case 的观测值。`P→F/F→P` 和双侧 exact McNemar `p` 使用 PRG→G1 的逐 case 配对结果。

| Gate | noPRG | PRG | G1 | G1−PRG | P→F / F→P | exact p |
|---|---:|---:|---:|---:|---:|---:|
| fall | 69/72 (95.8%) | 70/72 (97.2%) | 70/72 (97.2%) | +0.0 pp | 0 / 0 | 1.000000 |
| body_z | 67/72 (93.1%) | 69/72 (95.8%) | 69/72 (95.8%) | +0.0 pp | 1 / 1 | 1.000000 |
| contact | 65/72 (90.3%) | 63/72 (87.5%) | 62/72 (86.1%) | −1.4 pp | 3 / 2 | 1.000000 |
| release | 69/72 (95.8%) | 69/72 (95.8%) | 67/72 (93.1%) | −2.8 pp | 3 / 1 | 0.625000 |
| hand_penetration | 63/72 (87.5%) | 63/72 (87.5%) | 61/72 (84.7%) | −2.8 pp | 8 / 6 | 0.790527 |
| lower_body | 35/72 (48.6%) | 50/72 (69.4%) | 63/72 (87.5%) | **+18.1 pp** | 2 / 15 | **0.002350** |
| root_pos | 50/72 (69.4%) | 48/72 (66.7%) | 49/72 (68.1%) | +1.4 pp | 8 / 9 | 1.000000 |
| root_ori | 60/72 (83.3%) | 63/72 (87.5%) | 63/72 (87.5%) | +0.0 pp | 2 / 2 | 1.000000 |
| hand_pos | 56/72 (77.8%) | 52/72 (72.2%) | 51/72 (70.8%) | −1.4 pp | 6 / 5 | 1.000000 |
| hand_ori | 35/72 (48.6%) | 39/72 (54.2%) | 37/72 (51.4%) | −2.8 pp | 7 / 5 | 0.774414 |
| object_pos | 69/72 (95.8%) | 65/72 (90.3%) | 67/72 (93.1%) | +2.8 pp | 0 / 2 | 0.500000 |
| object_ori | 71/72 (98.6%) | 71/72 (98.6%) | 61/72 (84.7%) | **−13.9 pp** | 10 / 0 | **0.001953** |

`gate rate` 汇总每个 case 的 12 个 gate decision；`strict-12` 要求一个 case 的 12 门全部通过。

| Object | noPRG gate rate | PRG gate rate | G1 gate rate | noPRG strict-12 | PRG strict-12 | G1 strict-12 |
|---|---:|---:|---:|---:|---:|---:|
| box001 (n=28) | 290/336 (86.3%) | 289/336 (86.0%) | 290/336 (86.3%) | 6/28 (21.4%) | 5/28 (17.9%) | 8/28 (28.6%) |
| box023 (n=16) | 166/192 (86.5%) | 167/192 (87.0%) | 165/192 (85.9%) | 4/16 (25.0%) | 7/16 (43.8%) | 3/16 (18.8%) |
| box021 (n=28) | 253/336 (75.3%) | 266/336 (79.2%) | 265/336 (78.9%) | 5/28 (17.9%) | 7/28 (25.0%) | 8/28 (28.6%) |
| ALL (n=72) | 709/864 (82.1%) | 722/864 (83.6%) | 720/864 (83.3%) | 15/72 (20.8%) | 19/72 (26.4%) | 19/72 (26.4%) |

观测结论：G1 相对 PRG 的总 gate rate 仅 `−0.2 pp`，strict-12 总体不变，但这是 `lower_body` 的显著改善与 `object_ori` 的显著回退相互抵消。box023 strict-12 从 `7/16` 降到 `3/16`，因此不能用总体通过率替代逐 gate、逐物体审计。

### 2.5 Mandatory visual review

选集包含任一 12 gate PRG PASS→G1 FAIL 的全部 case，加上 `Δz>+1 cm`、`Δ3D>+2 cm`、每物体 PRG z 最坏/中位数样本和 profile 覆盖；并集 36 例（box001=15、box023=10、box021=11）。

- 36/36 的 G1 grasp/lift/carry/place 帧均保持站立，未见新增 fall、飞散、非有限跳变或灾难性脱手。
- 33 例包含至少一个单 gate PASS→FAIL；视频用于解释外观，不覆盖数值 gate。3 mm penetration 等细阈值不能由缩略帧反向判 PASS。
- `box001_20231020_014_p2`：G1 carry/place 箱体姿态明显不同；`Δz=+1.866 cm, Δ3D=+1.878 cm`。
- `box021_20231018_028_p1`：无粗粒度失败，但 `Δz=+1.089 cm`，保留 z 例外。
- `box021_20231018_028_p2`：G1 carry 持箱位置相对 PRG 偏移；`Δ3D=+3.043 cm`。
- 逐例非空观察与 contact sheet 路径见 `e194_three_arm_visual_review.tsv`；九张 atlas 覆盖全部 36 例。

### 2.6 预注册 claims

| Claim | Verdict | 证据 |
|---|---|---|
| C0 scope/provenance | PASS | 72 unique；28/16/28；variant/artifact parity 72/72 |
| C1 单变量 intervention | PASS | 72 sidecar audit；仅 object gravcomp；500/50 不变 |
| C2 execution/numeric closure | PASS | 72/72 Full；errors=0 |
| C3 z 泛化 | PASS | 三物体 Δz 均 <−0.5 cm，CI 上界均 <0 |
| C4 3D 不回退 | PASS | 三物体 Δ3D 均约 −1.0 至 −1.3 cm |
| C5 contact 保留 | PASS | 最差 box023 −0.032 > −0.05 |
| C6 physics safe | PASS | penetration/leg 增幅均 ≤0.05；new fall=0 |
| C7 aggregate 12-gate floor | PASS（有结构性 warning） | 最差物体总 gate rate −1.0 pp > −10 pp；但 object_ori 显著回退且 box023 strict-12 下降 |
| C8 device confound | PASS | 三 profile 覆盖三物体；9/9 z 方向一致 |
| C9 evidence closure | PASS | 72 MP4；36/36 review；report/workbook/TSV 已落盘 |

## 3. Interpretation（解释，不是额外观测）

1. **PRG 不是本组 tracking 改善的主因。** 与 noPRG 相比，PRG 的 z 只有 `−0.032 cm` object-macro 变化，CI/逐物体方向混合；3D 却三个物体一致变差。PRG 的价值更可能在下肢/接触约束，而不是 object tracking 本身。
2. **G1 是独立且跨物体稳定的杠杆。** 三个不同尺寸 box、三个 profile、全部 72 case 都给出一致的 object-level 改善；这符合“重力下垂由 `m·g/kp` 引入，gravcomp 去掉静态偏置”的机制预期。
3. **G1 不只是把 z 做好、牺牲 xy。** G1 对 PRG 的 3D 也在三个物体改善，并且最终低于 noPRG，因此没有观察到“用 xy 回退换 z”的总体模式。
4. **12-gate 总率掩盖了 gate 构成迁移。** G1 对 lower-body 的作用与预期一致，但 object orientation 出现统计显著回退；尤其 box023 strict-12 明显下降。由此只能说 aggregate floor 通过，不能说 G1 对所有 gate 一致更安全。
5. **均值推广不能抹掉逐例例外。** 三条 regression threshold 例外和 33 个 gate flip 仍需保留到 by-case 消费端；本结论支持 scoped default，不支持取消 gate 或跳过逐例审计。

## 4. 决策与后续

最终判决：`PROMOTE_G1_WITH_CASE_LEVEL_EXCEPTIONS`。

- 对 `E167A_zOnlyBody + PRG + rubber_hull` 的 box001/box023/box021 后续 CEM，默认启用 object `gravcomp=1`、保持 `kp_pos/kp_rot=500/50`。
- 保留三条明确数值例外与全部 gate migration；新增物体仍先做 sentinel，不外推为全物体规则。
- promotion 消费端必须保留 `object_ori` gate，并对 box023 使用逐 case strict-12 结果；不得只按 83.3% aggregate gate rate 放行。
- noPRG/PRG/G1 比较只证明 CEM/重定向指标，不等价于 RL policy 成功，也不修改 S1–S5 数据构建事实。
- 若继续优化，优先解释 box001 `014_p2` 和 box021 `028_p1/p2` 的姿态/路径异质性，而不是再次提高 actuator gain；E194 已证明 kp=2500 会发散。

## 5. 复现入口

```bash
# G1 72-case Full（local + Ada6000×2）
MODE=full bash workspace/core4d/scripts/launch/active/run_E194_G1_expansion_hybrid_3gpu.sh

# PRG↔G1 authority eval/report
MUJOCO_GL=egl .venv/bin/python workspace/core4d/scripts/eval/runners/eval_E194_G1_expansion.py full --require-all
.venv/bin/python workspace/core4d/scripts/eval/reports/gen_E194_G1_expansion_report.py

# noPRG / PRG / G1 same-case eval
MUJOCO_GL=egl .venv/bin/python workspace/core4d/scripts/eval/runners/eval_E194_three_arm_comparison.py

# mandatory frames/review + XLSX
.venv/bin/python workspace/core4d/scripts/eval/reports/extract_E194_three_arm_visual_review.py
.venv/bin/python workspace/core4d/scripts/eval/reports/write_E194_three_arm_visual_review.py
.venv/bin/python workspace/core4d/scripts/eval/reports/build_E194_three_arm_workbook.py
python /home/ubuntu/.codex/skills/xlsx/scripts/recalc.py \
  workspace/core4d/results/E194/s6_downstream/eval/full_g1_expansion/E194_noPRG_PRG_G1_comparison.xlsx

# E194/G1 72-case 3D viser review
bash workspace/core4d/scripts/eval/wrappers/review_player.sh E194 --port 8080

# Headless index/cardinality audit
bash workspace/core4d/scripts/eval/wrappers/review_player.sh E194 --check
```

## 6. 产物

- 评估目录：`workspace/core4d/results/E194/s6_downstream/eval/full_g1_expansion/`
- XLSX：`E194_noPRG_PRG_G1_comparison.xlsx`
- canonical G1 report：`E194_G1_box001_box023_box021_expansion_report.md`
- 三臂 TSV：`e194_three_arm_{case_metrics,paired_deltas,by_object,gate_migrations,12gate_by_object,12gate_overall}.tsv`
- noPRG authority：`e194_noprg_authority_audit.tsv`
- visual review：`e194_three_arm_visual_review.tsv` 与 `visual_review/{case_sheets,atlases}/`
- G1 MP4：`workspace/core4d/results/E194/s6_downstream/render/full_g1_expansion/`（72/72）
- 3D viser：`review_player.sh E194`，只索引 `full_g1_expansion` 的 `arm=G1` 72条
