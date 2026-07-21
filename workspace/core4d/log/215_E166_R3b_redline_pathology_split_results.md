# E166-R3b 分病因红线重审结果

日期：2026-06-18
计划：`workspace/core4d/plan/178_E166_redline_replan_after_r3_fail_plan.md`
前置：`log/214_E166_redline_predictive_results.md`
产物：`workspace/core4d/results/E166/redline_predictive/`

## 结论

**R3b 部分支持，但仍不放行 GPU。**

分病因后，`box023_person2` 被 E165 on-rails 指标明确标记为 self-collision/init-penetration pathology（`max_init_net_force=2459.2N`）。在排除该 pathology 后，平滑/速度指标对剩余 6 个有标签 case 的预测力满足 `|rho|>=0.6`：

| 指标 | subgroup rho | 解释 |
|---|---:|---|
| `ankle_acc_max` | -0.783 | 踝部加速度越大，staggered 越差 |
| `trackbody_jerk_p95` | -0.725 | 被跟踪 body jerk 越大，staggered 越差 |
| `obj_speed_max` | -0.667 | 物体峰值速度越大，staggered 越差 |
| `foot_slip_max_m` | -0.638 | 支撑脚累计滑移越大，staggered 越差 |

这支持“脚/平滑是 contact-feasible 子集里的主因”这个修订假设。

但新的 Tier-1 多维 OR 红线仍未达到计划门槛：

| precision | recall | TP | FP | FN |
|---:|---:|---:|---:|---:|
| 0.667 | 1.000 | 4 | 2 | 0 |

因此当前红线会拦下所有失败 case，但也误拦 2 个非失败 case（含中等成功 case），不满足 `precision>=0.75`。按 plan 178，**仍不启动 E166 CEM/RL GPU 消融**。

## 关键产物

新增/更新：
- `redline_subgroup_correlations.tsv`
- `redline_pathology_split.tsv`
- `redline_summary.md`
- `redline_thresholds.json`

`redline_pathology_split.tsv` 的核心行：

| case | self_collision | low_contact | reason |
|---|---:|---:|---|
| box023_person2 | true | true | `onrails_init_net_force=2459.2N; low_isaac_both_and_iou` |
| box004_083_p2 | false | true | `low_isaac_both_and_iou` |
| box004_082_p1 | false | true | `low_isaac_both_and_iou` |
| box004_083_p1 | false | true | `low_isaac_both_and_iou` |

注意：box004 低接触不能简单当作排除条件，因为 `box004_083_p2` 在低 Isaac contact 下仍有 `staggered=0.484`。当前 R3b 只把 box023 self-collision 当作非平滑 pathology 隔离。

## Claim 验证

| Claim | 结果 | 证据 |
|---|---|---|
| C-R3b-1：分病因后平滑预测力成立 | ✅ 部分通过 | 排除 box023 后 `ankle_acc/jerk/objV/foot_slip` 的 `|rho|=0.638-0.783` |
| C-R3b-2：foot metric 更有效 | ⚠️ 部分支持 | 旧 `foot_slip_max_m` 在 subgroup `|rho|=0.638`；新增 0.5s drift/stance-z 指标仍弱 |
| C-R3b-3：Tier-1 可解释且不过拟合 | ❌ 未通过 | PR=`0.667/1.0`，误拦 2 个非失败 case |

## 运行命令

```bash
python -m py_compile workspace/core4d/scripts/eval/runners/eval_E166_redline_predictive.py
git diff --check -- workspace/core4d/scripts/eval/runners/eval_E166_redline_predictive.py
bash workspace/core4d/scripts/eval/wrappers/eval_E166_redline.sh
```

## 决策

不启动：
- `workspace/core4d/scripts/launch/active/run_E166_remote.sh`
- 9 条 E166 CEM
- 12 条 SUGAR RL

下一步若继续推进，应把红线从单 OR 规则改为 **two-stage classifier**：
1. `self_collision/contact_pathology` 单独分流；
2. 对 contact-feasible 子集用 `ankle_acc/jerk/objV/foot_slip` 建立 redline；
3. 用至少一个新增 labeled case 或留一验证减少 n=6 过拟合风险。
