# E163 — box004_082_p1 failure analysis

日期：2026-06-15

关联结果：`workspace/core4d/log/208_E163_narrow_surface_band_clean8_results.md`

## 1. 问题

E163 clean8 只有一个 case 失败：

```text
case = box004_082_p1
rubberhand raw contact = 0.6721
hard lower bound = 0.6221
E163 raw contact = 0.5738
delta = -0.0984
```

该 case tracking 正常、没有 fall、Table4 tracking 完整，因此失败只来自 raw contact hard gate。

## 2. 关键事实

同一 `core4d_3cm` mask 下，`box004_082_p1` 的 active mask 只有 61 帧：

```text
mask runs = 15..59, 61..76
active frames = 61 / 109
```

所以 raw contact hard gate 对帧数很敏感：

```text
0.05 * 61 = 3.05 frames
```

实际 E163 比 rubberhand 少 6 个 active-mask raw contact 帧，因此直接越过阈值。

Mask 诊断产物：

```text
workspace/core4d/results/E163/narrow_surface_band/diagnostics/box004_082_p1_mask_analysis/
```

| 文件 | 内容 |
|---|---|
| `mask_overview_raw_spider_eval.png` | raw/spider/eval 三个时间轴的 person/hand mask |
| `mask_vs_raw_contact_timeline.png` | spider mask 与各方法 MuJoCo raw contact pair 对齐 |
| `sdf_vs_raw_contact_timeline.png` | min hand-object SDF 与 raw contact ticks |
| `box004_082_p1_mask_contact_per_frame.csv` | 每帧 mask、raw contact、clean3/5、SDF、contact dist |
| `box004_082_p1_mask_contact_summary.csv` | 分段统计 |

Mask 文件结构：

```text
raw_contact_mask_3cm    = 139 frames, original mocap time axis
spider_contact_mask_3cm = 109 frames, SPIDER/CEM eval qpos time axis
eval_contact_mask_3cm   = 182 frames, 50Hz eval time axis
```

E163 run-time reward log 使用 `eval_contact_mask_3cm` resize 到 CEM reference length；strict eval 使用 `spider_contact_mask_3cm` 与 saved qpos 对齐。两者 active 区间一致，未发现 mask 错位。

## 3. 方法对比

| 方法 | raw contact | raw frames | clean3 | clean5 | deep3 | SDF mean in mask | min contact dist mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| `SPIDER+rubberhand` | 0.6721 | 41/61 | 0.0820 | 0.2131 | 0.5902 | -0.0002 | -0.0082 |
| `+gateA` | 0.6393 | 39/61 | 0.1967 | 0.2459 | 0.4426 | 0.0029 | -0.0058 |
| `E156_decay` | 0.6721 | 41/61 | 0.2131 | 0.2951 | 0.4590 | -0.0008 | -0.0057 |
| `E161 releaseDecay` | 0.6066 | 37/61 | 0.3279 | 0.4590 | 0.2787 | 0.0066 | -0.0034 |
| `E163 narrowSurfaceBand` | 0.5738 | 35/61 | 0.2951 | 0.4918 | 0.2787 | 0.0045 | -0.0032 |

Interpretation:

- rubberhand 的 raw contact 高，但大量来自 penetration：deep3=0.5902，mean contact dist=-8.2mm。
- E163 的 penetration 明显少，clean5 最高，但 raw contact 变少。
- E163 的失败不是“离物体很远”，而是把部分原本穿进去触发 contact pair 的帧变成了小正间隙/近接触。

## 4. 逐帧丢失

E163 相对 rubberhand 在 mask 内丢失 raw contact 的帧：

```text
36, 38, 39, 40, 41, 45, 46, 47, 51, 52, 53, 54, 62, 63, 75
```

其中 E163 在这些帧的 SDF 大多仍然很小：

```text
0.0004m, 0.0005m, 0.0006m, 0.0009m, 0.0012m, ...
0.0079m, 0.0104m, 0.0147m, 0.0151m, 0.0184m
```

也就是说这些帧大多是 near miss，而不是远离物体。

E163 在 mask 内无 raw contact 的 26 帧中：

```text
SDF < 3cm: 24/26
SDF < 5cm: 25/26
```

## 5. 手侧切换

`box004_082_p1` 还有明显接触构型变化。

mask1 `15..59`：

| 方法 | raw left | raw right | any |
|---|---:|---:|---:|
| `rubberhand` | 30/45 | 2/45 | 32/45 |
| `E161 releaseDecay` | 10/45 | 20/45 | 29/45 |
| `E163 narrowSurfaceBand` | 6/45 | 21/45 | 27/45 |

mask2 `61..76`：

| 方法 | raw left | raw right | any |
|---|---:|---:|---:|
| `rubberhand` | 6/16 | 5/16 | 9/16 |
| `E161 releaseDecay` | 1/16 | 7/16 | 8/16 |
| `E163 narrowSurfaceBand` | 1/16 | 7/16 | 8/16 |

E161/E163 的解从 rubberhand 的左手主导接触切到了右手主导接触。raw metric 是 any-hand，所以手侧切换本身不直接导致失败；但它说明优化找到了不同接触构型，并且该构型在 active mask 内少触发了若干 MuJoCo contact pair。

## 6. 根因判断

主要原因不是：

- artifact/config 错：E163 artifact/config check 8/8 通过，`core4d_3cm` mask 正确。
- tracking/fall：`success_tracked=true`、`fall=false`。
- 表格或 baseline missing：rubberhand baseline 存在。

更合理的根因：

```text
E161/E163 的 hand gate + surfaceBand stack 把解从“穿透式稳定接触”
推向“更少穿透的贴近/近接触”，同时在该 case 切换了接触手侧。
但 hard gate 使用的是 MuJoCo raw contact pair，而不是 SDF 近接触。
因此少量小正间隙帧不算 contact，raw contact 直接掉过阈值。
```

这也是为什么总体指标看起来更好：

```text
E163 clean5 contact 更高: 0.4918 vs rubberhand 0.2131
E163 physPen3 更低: 0.1651 vs rubberhand 0.3303
E163 geomPen2 更低: 0.0734 vs rubberhand 0.3303
```

但 RL-safe 口径仍判 fail，因为 RL 关心 raw contact。

## 7. 后续建议

不要把 E163 clean8 直接导出 RL-ready。

优先做一个小修复实验，只针对 `box004_082_p1` 及必要回归 case：

1. 在 CEM selection/rerank 加 raw-contact-in-mask surrogate，按当前 case mask 直接奖励 active mask 内 MuJoCo hand-object contact pair。
2. 或在 surfaceBand 上加轻微负 SDF bias，例如目标从 `sdf=0` 改为 `sdf=-0.5mm`，但必须限制 penetration，避免回到 rubberhand 的 deep penetration。
3. 或做 case-local rerank：在最终候选里优先选择 raw contact count 达标且 penetration 不超过 E163 的轨迹。

不建议单纯放宽 clean3/clean5 口径，因为这会绕开 E162 后确定的 RL raw-contact hard gate。

## 8. Reward 与 raw contact 定义澄清

当前 E163 reward 不硬性要求“双手同时 raw contact”。

`contact_hdmi_rew`：

```text
per_eef_rew = [left target reward, right target reward]
mask_eef = per-hand contact mask
contact_hdmi_rew = mean(per_eef_rew * mask_eef * gain + (1 - mask_eef))
```

因此它是 per-hand soft reward 后取平均。若某帧 L/R mask 都为 1，则两只手都靠近会更高；但没有 `left AND right` 的硬门，也没有要求 MuJoCo raw contact pair。

E163 本 case 配置：

```text
contact_hdmi_mask_source = core4d_3cm
contact_hdmi_mask_carry_union = false
surface_band_geom_names = ["lh", "rh"]
surface_band_gate_source = contact_mask
```

`surface_band_rew` 使用：

```text
surface_band_sdf = min_sdf(["lh", "rh"], object)
```

也就是两只手取最小 SDF；任一只手落在 band 里就能给 surfaceBand reward，不要求双手同时接触。

Eval 里的 `MuJoCo raw contact pair` 定义是：

```text
for each frame:
  mj_forward(model, data)
  for con in data.contact[:data.ncon]:
    pair = {con.geom1, con.geom2}
    if pair contains any object_collision* geom
       and the other geom is "lh" or "rh":
         raw hand-object contact = true
```

这个 raw contact 不检查 `con.dist` 阈值；只要 MuJoCo collision detector 生成了 hand-object contact pair 就算 true。`clean3/clean5` 才进一步要求该 frame 的最小 contact distance 不深于 `-3mm/-5mm`。
