# E090 Results: H2-first retarget ablation and SPIDER smoke/full

日期：2026-05-28
计划：`workspace/core4d/plan/96_E090_original_omniretarget_ablation_plan.md`

## TL;DR

E090 已完成 H2-first canonical retarget、Box023/Box025 guard、SPIDER smoke，以及 smoke-pass case 的 full CEM。

结论：

- no-fingertip 只能部分支持 H2：它能把 Box021 wrist 从 box 内拉出来，但 support-face 不够，且 1/3 canonical case 求解 infeasible。
- topface-preIK 在 Box021 几何上明显更强：canonical 2/3 gate pass，剩余 1 个仅因 `T=78<80` reject。
- Box025 topface-preIK guard 失败（inside `48.1/51.9%`），所以 topface-preIK 不能全局应用；Box025 仍需要 reach-aware/fingertip 语义。
- SPIDER 中 topface-preIK 消除了 E082-E088 的头胸穿箱/手撑地主问题，但 full CEM 仍在 S1 上收敛到低 pelvis 局部解（`pelvis_min=0.134m`），未达到 full pass。

下一步：不扩展 D003 13 case；先做 S1-only 姿态约束验证（pelvis/upright elite gate 或等价 cost），验证能否在保留 topface-preIK safety 的同时把 pelvis 拉回 `>=0.55m`。

## 1. 产物路径

| 类型 | 路径 |
|---|---|
| Retarget manifest | `workspace/core4d/results/E090/variants.tsv` |
| Geometry summary | `workspace/core4d/results/E090/geometry/geometry_summary.csv` |
| Smoke task build | `workspace/core4d/scripts/E090/build_spider_tasks.py` |
| Smoke variants | `workspace/core4d/scripts/E090/variants_smoke.tsv` |
| Smoke train/eval | `workspace/core4d/scripts/train/train_E090_smoke.sh`, `workspace/core4d/scripts/eval/eval_E090.py` |
| Full train/eval | `workspace/core4d/scripts/train/train_E090_full.sh`, `workspace/core4d/scripts/eval/eval_E090.py --stage full` |
| Smoke results | `workspace/core4d/results/E090/smoke/smoke_eval_summary.csv` |
| Full results | `workspace/core4d/results/E090/full/full_eval_summary.csv` |
| Videos | `workspace/core4d/results/E090/smoke/*.mp4`, `workspace/core4d/results/E090/full/*.mp4` |
| Keyframes | `workspace/core4d/results/E090/smoke/keyframes/`, `workspace/core4d/results/E090/full/keyframes/` |

## 2. Geometry results

| Variant / task | Gate | 关键指标 | 解释 |
|---|---|---|---|
| `d003_box021_20231018_029_p2_nofing_e090` | reject | `inside=0/0%`, `support=17.3/14.7%`, `T=75` | no-fingertip 消除 inside，但未放到 support face |
| `d003_box021_20231011_035_p2_nofing_e090` | reject | `inside=0/0%`, `support=27.1/21.8%` | 接近阈值但仍 reject |
| `d003_box021_20231020_019_p1_nofing_e090` | infeasible | `CVXPY solve failed: infeasible` | 不重复该失败配置 |
| `d003_box021_20231018_029_p2_btop_preik_e090` | reject | `support=100/100%`, `T=78` | 几何修好但片段太短 |
| `d003_box021_20231011_035_p2_btop_preik_e090` | pass | `inside=0/0%`, `support=100/100%`, `T=135` | Smoke S1 |
| `d003_box021_20231020_019_p1_btop_preik_e090` | pass | `inside=4/4%`, `support=98/100%`, `T=101` | Smoke S2 |
| `box023_person2_btop_preik_e090` | pass | `inside=0/0%`, `support=100/100%` | Guard pass |
| `box025_person2_btop_preik_e090` | reject | `inside=48.1/51.9%`, signed distance mean `-1/-7mm` | Negative guard，禁止全局 topface-preIK |

## 3. SPIDER smoke/full

| Stage | Variant | Pass | contact | obj_mean | pelvis_min | head | upper | LH floor | RH floor |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| smoke | `E090S1_box021_20231011_035_p2_btop` | yes | 95.6% | 0.001m | 0.538m | 0.0% | 0.0% | 0.0% | 0.0% |
| smoke | `E090S2_box021_20231020_019_p1_btop` | no | 40.6% | 0.011m | 0.180m | 0.0% | 0.0% | 56.4% | 0.0% |
| full | `E090S1_box021_20231011_035_p2_btop` | no | 57.8% | 0.009m | 0.134m | 0.0% | 0.0% | 0.0% | 0.0% |

Full pass 阈值要求 safety 三项 `<=5%`、`obj_mean<=0.10m`、`pelvis_min>=0.55m`。S1 full 只因 pelvis collapse fail。

## 4. 可视化观察

- S1 smoke：双手能维持在箱体上表面/上沿附近，未见明显头胸穿箱或手撑地；但姿态仍偏弯腰，头部贴近箱面。
- S2 smoke：后段左手/身体落地并翻箱，和 `LH_floor=56.4%` 一致。
- S1 full：安全指标仍为 0%，但机器人明显趴低/跪低，头部贴近箱面；`pelvis_min=0.134m` 是真实姿态失败，不是统计误判。

## 5. Claims

| Claim | 结果 |
|---|---|
| C1 fingertip replacement 是关键变量 | 部分支持。no-fingertip 降低 inside，但没有稳定 support-face，且一条 infeasible。 |
| C2 no-fingertip 通过 gate 后 SPIDER 改善 | 未触发。no-fingertip 没有通过 canonical gate。 |
| C3 original OmniRetarget 是否优于 current | 暂未执行；topface-preIK 已解释主要几何问题，original 降为确认项。 |
| C4 no-fingertip/original fail 但 topface pass 指向 target semantic | 支持。Box021 topface-preIK 几何和 SPIDER safety 均显著改善。 |
| C5 Phase4 flags 风险 | 暂未执行；D003 production 默认未显式启用 Phase4 flags。 |
| C6 H2 修复不能破坏 Box025 reach guard | 触发。Box025 topface-preIK guard 失败，因此最终必须条件化。 |

## 6. 下一步

1. E090 Phase 4B：只对 S1 做姿态约束验证，候选为 pelvis/upright elite gate 或 pelvis `z<0.55m` penalty。
2. 若 Phase 4B 仍 pelvis fail，则不要扩展 D003 13 case，Box021 D003 单 G1 先降级。
3. Box025/大箱体路径保留 reach-aware/fingertip 语义；不要全局启用 topface-preIK。
