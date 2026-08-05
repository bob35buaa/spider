# E188 实验报告：Bucket 2→5 kg 受控 Full CEM

_CORE4D Phase 51 · 2026-08-05 · single-experiment controlled comparison_

## Summary

- E188 将 E187 中 15 条真实 `2 kg` bucket case 改为 `5 kg`，并将 object `diaginertia` 同比放大 `2.5×`；其余 trajectory、contact mask、collision/grid、reward、`1024×32` CEM、seed 0 全部冻结。
- CEM、评测、最终单侧视频、paired 视频均 `15/15 PASS`；xlsx 11 sheets、1538 formulas，LibreOffice 重算后公式错误为 0。
- E188 numeric pass 仍为 `3/15`；lower-body pass 虽由 `4/15→8/15`，但 mean leg improvement 只有 `+0.00627`，非退化仅 `8/15`，未达到 C5。
- pooled 接触和手穿透分别改善 `+0.0130/+0.0256`，C6 通过；六项 tracking paired mean regression 均未超过 2 cm/2°，C7 通过；C8 因 pass 未达到 5/15 而失败。
- 最强的同设备 local-4 证据与跨设备 11 条方向冲突：local-4 leg improvement=`-0.05535`，95% CI=`[-0.08414,-0.01667]`，即 5kg 明显恶化；cross-device11 为 `+0.02868` 且 CI 跨0。因此不升级 5kg 默认质量，结论层级为 **T2 可解释负/中性结果**。

## 1. Experiment Motivation

本实验回答：E187 的 bucket 质量从 2kg 提高到 5kg 后，SPIDER/CEM 解是否系统性改变，并能否缓解 E187 的主要失败模式 lower-body penetration，同时不牺牲接触、手穿透和 tracking。

E187 22 条 bucket case 中只有15条原始质量为2kg；其余7条已为5kg，按用户修订不重复运行。因此 E188 是15条真实质量处理组，不包含 A/A 控制。

## 2. Experiment Setup

| 项目 | E187 baseline | E188 treatment |
|---|---|---|
| Case | 相同15条历史2kg case | 相同15条 |
| Object mass | 2.0 kg | 5.0 kg |
| Object inertia | E187原值 | `diaginertia×2.5` |
| Bucket | bucket003×2；bucket007×13 | 不变 |
| CEM | 1024 samples × 32 iterations | 不变 |
| Seed | 0 | 0 |
| Reward/grid/P/R/G | E187冻结生产配置 | 不变 |
| Query tape | off | off |
| 最终 E188 worker | — | local-0=7；A100-4=4；A100-5=4 |
| 因果证据分层 | E187 local-4=RTX5090 | 同4条仍为RTX5090 |
| 其余11条 | RTX6000 Ada | A100或RTX5090；mass/device混杂 |

执行入口：

```bash
bash workspace/core4d/scripts/eval/wrappers/eval_E188_vs_E187_full.sh run
bash workspace/core4d/scripts/eval/wrappers/render_E188_vs_E187_paired.sh run
.venv/bin/python workspace/core4d/scripts/eval/reports/gen_E188_vs_E187_xlsx.py
bash workspace/core4d/scripts/eval/wrappers/review_player.sh E188 --check
```

## 3. Core Algorithm or Method

本实验不改变 SPIDER 算法。唯一主动干预是 object rigid-body inertial：

```text
new_mass = 5.0 kg
new_diaginertia = old_diaginertia × (5.0 / 2.0)
```

Scene canonical diff 只允许 object inertial 的 `mass` 与 `diaginertia` 两个属性变化。15个scene均通过MuJoCo load，robot inertial、geometry、friction、pose与全部优化输入保持不变。

评测直接调用公共 `eval.core.core_metrics.evaluate_sequence`，使用与E187一致的6个物理门+6个tracking门。paired improvement统一定义为“正数更好”：contact=`E188−E187`，其余指标=`E187−E188`。

## 4. Metrics

| 指标 | 方向 | Gate/Claim用途 |
|---|---|---|
| 手接触 in-mask fraction | 越高越好 | gate≥0.50；C6 mean regression≤0.03 |
| 手穿透 3mm frame fraction | 越低越好 | gate≤0.30；C6 mean regression≤0.03 |
| 下肢穿透 fraction | 越低越好 | gate≤0.10；C5 pass≥7、mean improvement≥0.05、非退化≥10 |
| Root/hand position error | 越低越好，cm | gate≤20cm；C7 paired mean regression≤2cm |
| Root/hand orientation error | 越低越好，degree | gate≤20°；C7 paired mean regression≤2° |
| Object position/orientation error | 越低越好，cm/degree | gate≤20cm/10°；C7 paired mean regression≤2 |
| Numeric release pass | 12门全过 | C8 E188≥5/15、PASS→FAIL≤1 |

每个关键指标对15条 paired improvement 做10,000次case bootstrap，seed=0，报告95% percentile CI。

## 5. Results

### 5.1 Gate outcome

| 项目 | E187 | E188 | 结果 |
|---|---:|---:|---|
| Numeric pass | 3/15 | 3/15 | 无净提升 |
| Lower-body pass | 4/15 | 8/15 | 门附近case增多，但均值门未过 |
| PASS→PASS | — | 2 | — |
| FAIL→PASS | — | 1 | `bucket007_20231003_1_021_p2` |
| PASS→FAIL | — | 1 | `bucket007_20231020_059_p1` |
| FAIL→FAIL | — | 11 | — |

### 5.2 Pooled paired metrics

| Metric | E187 mean | E188 mean | Improvement | 95% CI | Nondegraded |
|---|---:|---:|---:|---:|---:|
| Hand contact | 0.73976 | 0.75277 | +0.01301 | [-0.02242, +0.05156] | 10/15 |
| Hand penetration | 0.22674 | 0.20113 | +0.02562 | [-0.00943, +0.06329] | 10/15 |
| Leg penetration | 0.15848 | 0.15221 | +0.00627 | [-0.03346, +0.05125] | 8/15 |

三项 pooled CI 均跨0；不能据此宣称稳定改善。

### 5.3 Device-scope split

| Scope | n | Contact improvement | Hand-pen improvement | Leg improvement | Leg 95% CI |
|---|---:|---:|---:|---:|---:|
| Same-device local-4 | 4 | -0.03383 | +0.06819 | **-0.05535** | **[-0.08414, -0.01667]** |
| Cross-device11 | 11 | +0.03004 | +0.01014 | +0.02868 | [-0.01910, +0.08009] |

local-4 是唯一 mass-only 的强配对证据，其下肢方向显著为负；cross-device11 的正向趋势同时包含 mass 与 device 变化，不能推翻 local-4。

### 5.4 Tracking

| Metric | E187 mean | E188 mean | Improvement（正为好） |
|---|---:|---:|---:|
| Root position cm | 17.1703 | 16.9765 | +0.1938 |
| Root orientation ° | 9.6985 | 8.8715 | +0.8271 |
| Hand position cm | 16.1721 | 16.5498 | -0.3777 |
| Hand orientation ° | 22.6287 | 22.0313 | +0.5974 |
| Object position cm | 9.8231 | 11.4293 | -1.6061 |
| Object orientation ° | 5.8030 | 7.6085 | -1.8055 |

点估计满足 C7 的2cm/2°非劣界限，但 object tracking 已接近界限，且对应 bootstrap CI 的坏侧超过2，不能称为明确改善。

### 5.5 Claims

| Claim | 状态 | 证据 |
|---|---|---|
| C0 Authority完整 | PASS | 15 unique；bucket003=2、bucket007=13、bucket004=0 |
| C1 质量变体正确 | PASS | mass=5kg、inertia×2.5，15/15 |
| C2 单变量合同 | PASS | XML/override/input SHA门通过 |
| C3 执行闭合 | PASS | CEM、artifact、final video均15/15 |
| C4 设备边界如实 | PASS | actual 7/4/4；same-device4/cross-device11单列 |
| C5 下肢改善 | **FAIL** | pass 4→8，但mean +0.00627、非退化8/15 |
| C6 接触/手穿透非劣 | PASS | pooled +0.0130/+0.0256 |
| C7 Tracking非劣 | PASS | 六项paired mean regression均≤2 |
| C8 总门改善 | **FAIL** | numeric pass 3→3；PASS→FAIL=1 |
| C9 可视化与报告 | PASS | xlsx、15 paired videos、实际观察完成 |

## 6. How to Read the Figures

Paired视频固定为左侧E187 2kg、右侧E188 5kg；每侧内部仍保留原CEM视频的reference/simulation视图。底部叠字给出case、device scope、E188 worker及contact/handPen/leg improvement，正数始终表示E188更好。

抽帧路径：`workspace/core4d/results/E188/s6_downstream/eval/full/visual_qc/`。

- `bucket007_20231020_055_p1`：E187末段机器人明显倒地，E188保持站立并继续持桶；对应contact +0.203、leg +0.108。E188仍有hand penetration -0.072且numeric FAIL，因此是“明显视觉收益伴随手穿透trade-off”。
- `bucket007_20231018_019_p2`：同设备local-4中，E188末段更深俯身贴桶，手臂/下肢姿态更拥挤；contact -0.123、leg -0.091，支持真实退化。
- `bucket003_20231018_003_p1`：左右整体姿态接近，leg近0；但contact -0.041、hand penetration -0.087，细粒度物理指标回退。
- `bucket007_20231003_1_021_p2`：E188末段更直立稳定，numeric FAIL→PASS、leg +0.173；因属cross-device11，只能称为5kg版本改善，不能单独归因于质量。

## 7. Interpretation

观察结果不支持“将bucket统一为5kg会系统性改善SPIDER重定向”。pooled gate count 的 lower-body 4→8 容易给出乐观印象，但paired mean、非退化数和同设备分层均不支持该解释。

最可能的解释是：质量变化确实改变了CEM可行解，但影响依赖case；部分跨设备case出现更稳定姿态，另一些同设备case发生接触与下肢trade-off。由于remote-11同时更换了设备，正向pooled趋势不能作为纯质量因果证据。

## 8. Conclusion and Discussion

E188 达到技术完整，但效果Claims只通过C6/C7，C5/C8失败。按plan212成功分层，本实验收口为 **T2 可解释负/中性结果**。

决策：

- 不把5kg升级为bucket默认质量；
- 保留E187质量配置作为当前基线；
- E188作为物体质量敏感性的完整负/混合证据归档；
- 不自动进入RL export或training。

## 9. Limitations and Caveats

- 单seed；没有独立seed置信度。
- 只覆盖15条原2kg bucket003/007，不含bucket004和7条原5kg控制。
- 11/15条存在E187 RTX6000 Ada→E188 A100/RTX5090设备混杂。
- 同设备质量证据只有4条；虽leg CI为负，但样本仍小。
- pooled bootstrap以case为单位，不能消除设备混杂。
- 视觉观察是代表case抽帧，不等同于15条完整人工终审。

## 10. Next Steps

1. 若仍需验证质量因果，另开同设备、多seed实验，只覆盖local-4或重新在同一GPU复跑扩展集。
2. 优先检查质量变化为何改善部分fall但恶化同设备lower-body/contact，可做动力学轨迹与CEM候选分布诊断；不要在E188内post-hoc改门。
3. 在用户明确授权前，不启动RL export/training。

## Reproducibility Notes

| Artifact | Path | SHA256 |
|---|---|---|
| Plan | `workspace/core4d/plan/212_E188_bucket_5kg_controlled_full_cem_plan.md` | 见git/worktree |
| Eval summary | `workspace/core4d/results/E188/s6_downstream/eval/full/summary.json` | `08c83ad609fa1a8f2987f5366978d46e86b4db76f58ec84c35f097a5909aac82` |
| Case metrics | `workspace/core4d/results/E188/s6_downstream/eval/full/e188_case_metrics.tsv` | `9db109ab50e26d8984903e6ca5f39f6759f03058c0a5e06dc21d13628f31eff9` |
| Paired deltas | `workspace/core4d/results/E188/s6_downstream/eval/full/e188_vs_e187_paired_deltas.tsv` | `7d95d527eaa11f2d96d1d08749ff741645864722198eb15020f82db83688ac85` |
| Workbook | `workspace/core4d/results/E188/s6_downstream/eval/full/E188_vs_E187_paired_evaluation.xlsx` | `c9d728c54a203871f773c41373869b0740d1e8b95395ae51e97d4e8237ded683` |
| Final video summary | `workspace/core4d/results/E188/s6_downstream/render/full/e188_videos/summary.json` | `7a317bc2b2560d66b9616480783a3242f7669acf592c5885c27cfda5b0ba59a3` |
| Paired video summary | `workspace/core4d/results/E188/s6_downstream/render/full/paired_e187_vs_e188/summary.json` | `7ea9b88ceeaaf4653ced318341f4d70366ca34b895ec3e869016e1718e16470d` |

E187 baseline metrics SHA=`77405ebd72a02c7135d3b28945ffcf65e1fa2c0e2394bbba687e3c699000cb06`。本轮未commit/push：效果Claims并非全部通过，按实验规则保留worktree供审阅。
