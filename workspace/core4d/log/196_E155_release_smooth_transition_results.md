# E155 — release smooth transition full 结果

> 计划:`workspace/core4d/plan/164_E155_release_smooth_transition_plan.md`
> 状态:**完成(12/12 full CEM + E154+ 标准评测)**
> 指标标准:`core4d-e154-physics-contact-v1`
> 参考:`E153_gateA_b1_sdf010_v10`(`min_sdf=-0.010,max_viol=0.10,hard_floor=-0.020`)

## 0. 一句话结论

E155 的平滑放手改动有效。四个方案都保持 **tracking 3/3、0 fall**；按 E154 后固定的 3mm/5mm
干净物理接触口径，**`decay` 最好**:release_false_3mm/5mm 降到 **0.033/0.054**，同时
in-mask contact_3mm/5mm 为 **0.291/0.421**，不但没有掉，反而高于 E153 参考的 **0.233/0.419**。

旧的无阈值 contact 诊断也保留: E153 release_false=0.499，`decay`=0.183，未到原计划的 0.15；
但该旧口径会把轻微/穿透接触混在一起，后续以 3mm/5mm 指标为准。

## 1. 执行

补齐 E155 full 矩阵 3 case × 4 method = 12 run。

| 资源 | 任务 |
|---|---|
| 本地 GPU0 | `box004_083_p2:ramp5` |
| 远程 GPU0 | `box004_083_p2:ramp10` |
| 远程 GPU1 | `box023_person2:decay -> neutral` |

完整性检查:12/12 均有 root npz、full mp4、`trajectory_mjwp_act.npz`。

## 2. 3-case 聚合

| method | tracked | guarded_vs_ref | pz_worst | rel_false3 | rel_false5 | inmaskC3 | inmaskC5 | pen2mm | phys_pen3 | leg_pen | gate_valid | fallback |
|---|:--:|:--:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E153 ref | 3/3 | 0/3 | 0.035 | 0.296 | 0.412 | 0.233 | 0.419 | 0.364 | 0.374 | 0.022 | - | - |
| ramp5 | 3/3 | 1/3 | 0.043 | 0.102 | 0.150 | 0.179 | 0.365 | 0.440 | 0.424 | 0.029 | 0.905 | 0.000 |
| ramp10 | 3/3 | 1/3 | 0.047 | 0.083 | 0.123 | 0.158 | 0.352 | 0.446 | 0.434 | 0.044 | 0.902 | 0.000 |
| decay | 3/3 | 1/3 | 0.047 | **0.033** | **0.054** | **0.291** | 0.421 | 0.412 | 0.357 | 0.048 | 0.895 | 0.000 |
| neutral | 3/3 | 2/3 | 0.040 | 0.123 | 0.233 | 0.236 | **0.426** | **0.407** | 0.388 | 0.029 | **0.907** | 0.000 |

说明:
- `tracked` = no fall 且 terminal pelvis-z tracking ≤0.08m。
- `guarded_vs_ref` 是额外诊断:tracking 通过且相对 E153 ref 不增加 2mm 几何穿透、不掉 5cm near-contact、不恶化 obj_err。
  它不是 E155 的主 claim；E155 目标是放手 false contact 与 in-mask contact。
- `fallback` 中 `decay` 为 0.000198，表中四舍五入为 0.000。

## 3. per-case 关键信号

| case/method | tracked | rel_false3 | inmaskC3 | 备注 |
|---|:--:|---:|---:|---|
| box021 ramp5/ramp10 | ✅ | 0.000 / 0.000 | 0.145 / 0.127 | 放手干净，但 inmaskC3 低于 ref |
| box021 decay | ✅ | 0.000 | 0.382 | 该 case 最好，且 pen2mm 低于 ref |
| box021 neutral | ✅ | 0.000 | 0.327 | guarded_vs_ref 通过 |
| box004 ramp5/ramp10 | ✅ | 0.250 / 0.250 | 0.161 / 0.161 | box004 仍是 release_false 瓶颈 |
| box004 decay | ✅ | 0.062 | 0.306 | box004 显著优于 ramp/neutral |
| box004 neutral | ✅ | 0.312 | 0.226 | release_false 最差 |
| box023 ramp10 | ✅ | 0.000 | 0.185 | release 最干净 |
| box023 decay | ✅ | 0.037 | 0.185 | 稳定，通过 guarded_vs_ref |
| box023 neutral | ✅ | 0.056 | 0.154 | inmaskC3 低但 guarded_vs_ref 通过 |

## 4. Claims 验证

| Claim | 标准 | 结果 | 裁定 |
|---|---|---|---|
| C1 release_false 降低 | 新标准:3mm/5mm release_false ≤0.15 | `decay`=0.033/0.054；`ramp10`=0.083/0.123；`ramp5`=0.102/0.150 | **成立** |
| C2 inmaskC 不下降超过 0.10 | 新标准:3mm/5mm in-mask contact 相对 E153 不低于 −0.10 | `decay` +0.058/+0.003；`ramp10` −0.075/−0.067；`ramp5` −0.053/−0.054；均满足 | **成立** |
| C3 success_tracked 保持 3/3 | no fall + pz_term≤0.08m | 四个方法均 3/3，worst pz=0.040~0.047 | **成立** |

旧口径补充:无阈值 release_false 的最优为 `decay`=0.183，未达原计划 0.15；但该列已降级为诊断。

## 5. 结论与下一步

- 推荐配置:**`decay`**。它在四个方案里最直接解决放手 false-contact，并保持/提升 3mm/5mm in-mask contact。
- 备选:**`ramp10`**。release_false 也低，但 inmaskC3/5 下降更多，且 leg_pen 高于 ramp5/neutral。
- 风险:box004 仍是 release_false 瓶颈；`decay` 把 box004 rel_false3 从 E153 ref 的 0.562 降到 0.062，但不是 0。
- 后续若进入 RL/handoff，建议把 E155 的 `decay` 作为默认 hand_support release 策略，同时保留 E154 3mm/5mm 指标作为固定评测基准。

## 6. 结果路径

| 类型 | 路径 |
|---|---|
| CEM full | `workspace/core4d/results/E155/cem/full/` |
| 评测 | `workspace/core4d/results/E155/eval/full/e155_{method_metrics,delta_vs_e153_sdf010_v10,method_summary}.tsv` |
| 评测脚本 | `workspace/core4d/scripts/eval/runners/eval_E155_release_smooth_transition.py` |
| shell wrapper | `workspace/core4d/scripts/eval/wrappers/eval_E155_release_smooth_transition.sh` |
| 本地/远程 launch | `workspace/core4d/scripts/launch/active/run_E155_{local,remote}.sh` |
| remote pull | `workspace/core4d/scripts/launch/active/pull_E155_remote_results.sh` |
