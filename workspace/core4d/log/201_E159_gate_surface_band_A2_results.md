# E159 — gateA + surfaceBand-A2 no-penalty clean6 结果

> 计划：`workspace/core4d/plan/168_E159_gate_surface_band_A2_no_penalty_plan.md`
> 状态：**完成；不晋级 clean8 复核**

## 0. 一句话结论

`gateA+surfaceBand-A2` 相比 E158 的 surfaceBand-A 明显更稳定：`success_tracked` 从 1/6 提升到 5/6，`fall` 从 4/6 降到 1/6，同时保持较强接触提升和较低穿透。但它仍未满足晋级标准：`box021_029_p2` fall/tracking fail，且 `box004_083_p2` 有 release false contact，clean6 不应 promote。

## 1. 运行与产物

| 项目 | 结果 |
|---|---|
| full CEM | 6/6 complete |
| 运行方式 | local 1 GPU + remote 2 GPU，按用户要求叠加运行，未等待 GPU idle，未 kill 其他程序 |
| local session | `E159_local_full_overlap_224419` |
| remote session | `E159_full_overlap_224419` |
| eval | `metric_rows=36`, `methods=6`, `delta_vs_gate=18`, `missing=0` |
| 指标口径 | `core4d-e154-physics-contact-v1` |
| result root | `workspace/core4d/results/E159/gate_surface_band_A2/` |
| CEM outputs | `workspace/core4d/results/E159/gate_surface_band_A2/cem/full/` |
| eval outputs | `workspace/core4d/results/E159/gate_surface_band_A2/eval/full/` |
| xlsx | `workspace/core4d/results/E159/gate_surface_band_A2/eval/full/E159_gate_surface_band_A2_clean6.xlsx` |

E159 A2 reward:

```text
reward_band = -0.001 <= sdf <= 0.030
surface_score = exp(-max(sdf, 0) / 0.015)
surface_reward = 1.5 * gate * surface_score * 1[reward_band]
surface_penalty = 0
```

## 2. Method summary

| method | tracked | fall | inmaskC3 | physPen3 | geomPen2 | releaseF3 | trackPz |
|---|---:|---:|---:|---:|---:|---:|---:|
| `+gateA` | 6/6 | 0 | 0.147 | 0.188 | 0.194 | 0.000 | 0.011 |
| `E155_decay` | 6/6 | 0 | 0.346 | 0.314 | 0.386 | 0.043 | 0.025 |
| `gateA+surfaceBand-A` | 1/6 | 4 | 0.520 | 0.151 | 0.013 | 0.115 | 0.264 |
| `gateA+surfaceBand-A2` | 5/6 | 1 | 0.494 | 0.130 | 0.021 | 0.042 | 0.094 |

相对 `+gateA` 的均值变化：

| metric | delta | 判读 |
|---|---:|---|
| `inmaskC3` | +0.346 | 接触提升强，接近 E158 A |
| `geomPen2` | -0.173 | 几何穿透大幅下降 |
| `physPen3` | -0.057 | 物理深穿透下降 |
| `releaseF3` | +0.042 | 超过 +0.03 门槛，失败 |
| `success_tracked` | 5/6 vs 6/6 | 仍有 1 条失败 |

## 3. Per-case gateA+surfaceBand-A2

| case | tracked | fall | trackPz | inmaskC3 delta | physPen3 delta | geomPen2 delta | releaseF3 delta |
|---|---|---|---:|---:|---:|---:|---:|
| `box021_035_p1` | true | false | 0.023 | +0.460 | +0.047 | -0.047 | +0.000 |
| `box021_035_p2` | true | false | 0.015 | +0.320 | -0.015 | -0.180 | +0.000 |
| `box021_029_p2` | false | true | 0.493 | +0.273 | -0.053 | -0.147 | +0.000 |
| `box004_083_p1` | true | false | 0.011 | +0.286 | +0.147 | -0.078 | +0.000 |
| `box004_083_p2` | true | false | 0.022 | +0.355 | -0.181 | -0.238 | +0.250 |
| `box023_person2` | true | false | 0.001 | +0.385 | -0.287 | -0.346 | +0.000 |

主要失败模式：

- `box021_029_p2`：`fall=true`，`track_pelvis_z_err_terminal_m=0.493`，导致 C1 失败。
- `box004_083_p2`：`releaseF3 delta=+0.250`，导致 release guard 失败。
- `box004_083_p1`：物理深穿透相对 `+gateA` 上升 `+0.147`，虽然 mean C4 通过，但 per-case 有风险。

## 4. Claims

| Claim | 标准 | 结果 | 裁定 |
|---|---|---|---|
| C1 tracking | `success_tracked=6/6`, `fall=0/6` | 5/6, fall 1/6 | 失败 |
| C2 release guard | mean `releaseF3 <= +0.03` vs `+gateA` | +0.042 | 失败 |
| C3 contact gain | mean `inmaskC3 >= +0.10` vs `+gateA` | +0.346 | 成立 |
| C4 penetration guard | mean `physPen3 <= +0.05`, `geomPen2 <= +0.02` vs `+gateA` | -0.057, -0.173 | 成立 |
| C5 improves over decay tradeoff | 接触接近/超过 decay 且穿透/release 更低 | 接触 0.494 > 0.346，穿透更低；release 0.042 接近 decay 0.043 | 部分成立 |

最终判定：`promote_to_clean8_probe=false`。

## 5. 解释

A2 验证了“取消 penalty + 降低 scale + 只奖励浅负到 30mm band”方向有效：它保留 E158 的接触和低几何穿透优势，同时明显减少 fall。但 A2 仍会在个别 case 把姿态推坏或造成 sticky release，说明 surface reward 仍能主导局部策略，不能直接升为默认方法。

## 6. 验证命令

```bash
python -m py_compile spider/config.py spider/simulators/mjwp.py workspace/core4d/scripts/experiments/E159/build_gate_surface_band_A2_manifest.py workspace/core4d/scripts/eval/runners/eval_E159_gate_surface_band_A2.py
bash -n workspace/core4d/scripts/launch/active/run_E159_local.sh
bash -n workspace/core4d/scripts/launch/active/run_E159_remote.sh
bash -n workspace/core4d/scripts/launch/active/pull_E159_remote_results.sh
bash workspace/core4d/scripts/launch/active/pull_E159_remote_results.sh full
bash workspace/core4d/scripts/eval/wrappers/eval_E159_gate_surface_band_A2.sh full
```

验证结果：

- Manifest/preflight：`rows=6`, `preflight_ok=true`, split 为 local 2 / remote0 2 / remote1 2。
- CEM artifact：root npz / video / outdir trajectory 6/6 complete。
- Strict eval：`metric_rows=36`, `methods=6`, `missing=0`。
- XLSX：6 sheets，`method_summary` 最优黑色加粗、次优下划线。

## 7. 下一步

不建议继续单纯增加 surface reward。下一轮应保留 A2 的无 penalty 结构，但加一个 release/tracking guard：

1. 对 `surface_band_rew` 加 release tail decay 或 contact-mask tail gate，优先解决 `box004_083_p2` sticky release。
2. 对 `box021_029_p2` 增加姿态/高度 guard，避免接触奖励把 pelvis tracking 拖坏。
3. 若继续做权重扫，优先 `scale=0.75/1.0`，而不是回到 penalty。
