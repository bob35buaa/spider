# E029 结果：COLA D6 support body phase-1 audit/preflight

日期：2026-05-27

## 目标

按 `plan/34_E029_cola_d6_support_body_redesign_plan.md` 启动 E029。phase-1 只做只读审计和 axis/contact preflight，分母固定为：

- `workspace/core4d_collab_retarget/results/E028/candidates.json`

## 新增脚本

```text
workspace/core4d_collab_retarget/scripts/E029/e029_common.py
workspace/core4d_collab_retarget/scripts/E029/audit_support_semantics.py
workspace/core4d_collab_retarget/scripts/E029/preflight_axis_contact.py
```

静态检查：

```bash
python -m py_compile \
  workspace/core4d_collab_retarget/scripts/E029/e029_common.py \
  workspace/core4d_collab_retarget/scripts/E029/audit_support_semantics.py \
  workspace/core4d_collab_retarget/scripts/E029/preflight_axis_contact.py

git diff --check -- workspace/core4d_collab_retarget/scripts/E029
```

结果：通过。

## 运行命令

```bash
cd /home/ubuntu/Workspace/spider
.venv/bin/python workspace/core4d_collab_retarget/scripts/E029/audit_support_semantics.py
.venv/bin/python workspace/core4d_collab_retarget/scripts/E029/preflight_axis_contact.py \
  --candidates workspace/core4d_collab_retarget/results/E028/candidates.json
```

## 产物

| 产物 | 路径 |
|---|---|
| current semantics report | `workspace/core4d_collab_retarget/results/E029/audit/current_support_semantics.md` |
| current semantics CSV | `workspace/core4d_collab_retarget/results/E029/audit/e028_candidate_modes.csv` |
| axis/contact report | `workspace/core4d_collab_retarget/results/E029/preflight/preflight_report.md` |
| axis/contact CSV | `workspace/core4d_collab_retarget/results/E029/preflight/axis_contact_summary.csv` |
| axis/contact panels | `workspace/core4d_collab_retarget/results/E029/preflight/*_axis_contact_panel.jpg` |

已检查 5 张 panel 均为 `1680x1260` RGB，文件非空。

## Semantics audit 结论

5/5 E028 candidates 都不是 COLA dynamic support body + D6：

| 指标 | 结果 |
|---|---:|
| candidates audited | 5/5 |
| `support_proxy_mode=mocap_pad` | 5/5 |
| `support_weld_anchor mocap=true` | 5/5 |
| dynamic support body present | 0/5 |
| compiled `nq/nv/nu` | 43/41/29 for 5/5 |
| `nmocap` | 1 for 5/5 |
| object last freejoint | 5/5 |
| direct object actuator | 0/5 |
| `support_proxy_force` max | 0 for 5/5 |

因此 E018/E028 当前准确语义是 kinematic mocap support anchor + equality weld scaffold。它不是 COLA 论文里的 support body 与 object 通过 6-DoF joint 连接、support command 经 support body dynamics 传力的路线。

E028 rollout 中 support weld residual 不是零：candidate mean gap `4.7-6.3cm`，最大 gap `11.2-32.7cm`。这说明即使 mocap weld scaffold 能给 object-side target，它也不是一个可解释的 dynamic reaction-force model。

## Axis/contact preflight 结论

最关键发现：Box021 这 5 条的 object-local 高度轴不是 `z`，而是 `y`。

| Variant | selected side | side frac/margin | old anchor face | anchor to selected-side centroid | height local axis | status |
|---|---|---:|---|---:|---|---|
| `20231011_034_p1` | `+x` | `0.434 / 0.028` | `+x` | `0.135m` | `y` (`0.992`) | `review_axis_before_d6` |
| `20231011_035_p1` | `+x` | `0.407 / 0.016` | `+x` | `0.141m` | `y` (`0.989`) | `review_axis_before_d6` |
| `20231011_035_p2` | `-x` | `0.402 / 0.019` | `-x` | `0.380m` | `y` (`0.990`) | `review_axis_before_d6` |
| `20231018_029_p2` | `-x` | `0.447 / 0.360` | `-x` | `0.452m` | `y` (`0.996`) | `review_axis_before_d6` |
| `20231020_019_p1` | `-x` | `0.327 / 0.071` | `-x` | `0.422m` | `y` (`0.990`) | `review_axis_before_d6` |

这解释了为什么旧 canonical anchor 对 D003 Box021 特别脆：E018 的 `canonical_z=0.62*half_z` 在这些 case 上不是“高度”，而是在局部水平/深度轴上偏移。旧 anchor 和 selected-side contact cloud 的距离在 3 个 p2/p1 case 上达到 `0.38-0.45m`，已经不是微调 anchor 可以解释的误差。

## 决策

1. 不继续沿 E028/E028b 的 fixed local-`z` single anchor 路线跑 full。
2. E029 下一步应进入 D6 support-body sanity，但 endpoint 初始化必须先做 axis remap：
   - height axis 使用 local `y`；
   - side axis 仍从 preflight dominant side 取 `+x/-x`，但 side confidence 低的 case 保留 caveat；
   - free axes 使用 selected-side robust centroid，而不是 `0.62*half_z` 和 free-axis center `0`。
3. 第一条 D6 sanity 建议只跑 `E028_d003_box021_20231018_029_p2_canonical_t02`，因为它是 5 条中 side margin 最清晰的一条；通过后再扩到 5 条。

## 当前停止点

phase-1 已完成；尚未生成 E029 D6 scene，也尚未跑 no-training D6 support load-path sanity。

