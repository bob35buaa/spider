# E158 — gateA + surfaceBand-A clean6 结果

> 计划：`workspace/core4d/plan/167_E158_gateA_surface_band_plan.md`
> 启动与 mask 修正记录：`workspace/core4d/log/199_E158_gate_surface_band_start_and_mask_fix.md`
> 状态：**完成；不晋级 clean8 复核**

## 0. 一句话结论

`gateA+surfaceBand-A` 证明了 one-sided surface band 能显著提高 mask 内干净物理接触，并大幅降低几何穿透；但它破坏了整体运动稳定性，`success_tracked=1/6`、`fall=4/6`，且 release false contact 明显升高。因此 E158 不应作为当前默认方法，也不进入 clean8 复核。

## 1. 运行与产物

| 项目 | 结果 |
|---|---|
| full CEM | 6/6 complete |
| 运行方式 | local 2 条 + remote 4 条，按用户要求与 R154/RL 叠加运行 |
| eval | `metric_rows=30`, `methods=5`, `missing=0` |
| 指标口径 | `core4d-e154-physics-contact-v1` |
| result root | `workspace/core4d/results/E158/gate_surface_band/` |
| CEM outputs | `workspace/core4d/results/E158/gate_surface_band/cem/full/` |
| eval outputs | `workspace/core4d/results/E158/gate_surface_band/eval/full/` |
| xlsx | `workspace/core4d/results/E158/gate_surface_band/eval/full/E158_gate_surface_band_clean6.xlsx` |

XLSX sheets：`method_summary`、`per_case_metrics`、`delta_vs_omniretarget`、`delta_vs_baseline`、`delta_vs_gateA`、`summary`。

## 2. Method summary

| method | tracked | fall | inmaskC3 | physPen3 | geomPen2 | releaseF3 | trackPz |
|---|---:|---:|---:|---:|---:|---:|---:|
| `+gateA` | 6/6 | 0 | 0.147 | 0.188 | 0.194 | 0.000 | 0.011 |
| `E155_decay` | 6/6 | 0 | 0.346 | 0.314 | 0.386 | 0.043 | 0.025 |
| `gateA+surfaceBand-A` | 1/6 | 4 | 0.520 | 0.151 | 0.013 | 0.115 | 0.264 |

相对 `+gateA` 的均值变化：

| metric | delta | 判读 |
|---|---:|---|
| `inmaskC3` | +0.372 | 接触提升很强 |
| `geomPen2` | -0.181 | 几何穿透大幅下降 |
| `physPen3` | -0.037 | 均值下降，但 per-case 不稳 |
| `releaseF3` | +0.115 | 明显失败 |
| `success_tracked` | 1/6 vs 6/6 | 明显失败 |

## 3. Per-case gateA+surfaceBand-A

| case | tracked | fall | trackPz | inmaskC3 | physPen3 | geomPen2 | releaseF3 |
|---|---|---|---:|---:|---:|---:|---:|
| `box021_035_p1` | true | false | 0.026 | 0.590 | 0.248 | 0.016 | 0.000 |
| `box021_035_p2` | false | true | 0.529 | 0.544 | 0.150 | 0.008 | 0.000 |
| `box021_029_p2` | false | true | 0.554 | 0.436 | 0.187 | 0.000 | 0.000 |
| `box004_083_p1` | false | true | 0.355 | 0.429 | 0.147 | 0.000 | 0.333 |
| `box004_083_p2` | false | false | 0.096 | 0.581 | 0.143 | 0.048 | 0.188 |
| `box023_person2` | false | true | 0.024 | 0.538 | 0.029 | 0.007 | 0.167 |

失败模式：

- Box021 p2 / 029 / Box004 p1 / Box023 出现 fall 或明显 tracking 失败。
- Box004 p1、Box004 p2、Box023 出现 release false contact，说明 surface reward 仍有“贴住不放”的副作用。
- 几何穿透被压低，但 MuJoCo 物理接触穿透在 `box021_035_p1`、`box004_083_p1` 仍有 per-case 回升，说明 one-sided band 不是充分的物理稳定约束。

## 4. Claims

| Claim | 标准 | 结果 | 裁定 |
|---|---|---|---|
| C1 tracking | `success_tracked=6/6` | 1/6 | 失败 |
| C2 penetration guard | mean `physPen3<=+0.02`, `geomPen2<=+0.02` vs `+gateA` | mean 通过；per-case 不稳 | 部分成立 |
| C3 release guard | `releaseF3<=+0.03` vs `+gateA` | +0.115 | 失败 |
| C4 contact gain | `inmaskC3>=+0.05` vs `+gateA` | +0.372 | 成立 |

最终判定：`promote_to_clean8_probe=false`。

## 5. Mask 与诊断说明

正式评测前已修正两处 contact mask 问题：

- `box021_035_p1`：替换错误 88 帧旧 mask 为 129 帧 case-specific mask。
- `box021_035_p2`：填补 frame `91..92` 的 2 帧孤立断口，只补 target person 右手。

详见 `log/199_E158_gate_surface_band_start_and_mask_fix.md`。修正后 E156/E158 均已重算。

## 6. 验证命令

```bash
bash workspace/core4d/scripts/launch/active/pull_E158_remote_results.sh full
bash workspace/core4d/scripts/eval/wrappers/eval_E158_gate_surface_band.sh full
python -m py_compile workspace/core4d/scripts/eval/runners/eval_E158_gate_surface_band.py
git diff --check
```

验证结果：

- CEM root npz / video / outdir trajectory / config：6/6 complete。
- strict eval：`metric_rows=30`, `methods=5`, `delta_vs_gate=12`, `missing=0`。
- workbook：6 sheets，`method_summary` 5 method rows。

## 7. 下一步

E158 的方向不应继续“单纯加大接触 reward”。下一轮应把 surfaceBand 作为接触形状项，而不是主导项：

1. 降低 surfaceBand reward scale，并增加 terminal/release gate，避免 sticky release。
2. 增加 tracking/posture guard 或 CEM selection ranking，避免接触奖励把身体姿态拖垮。
3. 优先在 `box021_035_p1` 这类不 fall case 上做局部权重扫，再扩到 clean6。
