# E093 结果：`wrist+5cm` 与 raw contact 几何语义明显错位，handbox/3-box 只能缓解 proxy 形状误差

日期：2026-05-29

对应计划：`workspace/core4d/plan/99_E093_contact_target_geometry_audit_plan.md`

## 结论

E093 在 7 条已有 case 上同时量化了 raw contact、当前 reward target (`wrist_yaw_link + 5cm`)、G1 sphere、历史 HDMI 3-box、Holosoma handbox proxy。结果支持用户的判断：继续 CEM/RL 前必须先把 contact target 语义搞清楚。

核心结论：

1. **`wrist+5cm` 不是 raw contact 的小误差 proxy。** 14/14 个 per-hand summary 的 `wrist5->raw mean` 都超过 `20cm`；Box026/Box025 达到 `49-64cm`。
2. **sphere 几何确实让接触位置语义不准。** sphere surface 到 raw 的 p90 absolute gap 全部超过 `10cm`，Box026/Box025 达 `45-61cm`；球只能表达“附近有碰撞”，不能表达手掌/手指/支撑面的接触区域。
3. **handbox 通常比 sphere/3-box 更接近 raw，但不是根治。** handbox 是 14/14 行的 best proxy，但 p90 gap 仍普遍 >`10cm`，Box026/Box025 仍在 `0.53-0.59m` 量级。
4. **box004/box023 能 work 不是因为 target 语义准确，而是偏差仍可被小/中箱尺寸与可达性容忍。** 二者 `wrist+5cm` inside 为 0，整体还在外侧/上沿附近；CEM 可以找到可站立接触。
5. **D003 box021 / Box026 的失败更像错面 + inside/low-support + 大偏移的组合。** D003 R hand inside `33.3%`；Box026 039 support 仅 `19.5/15.4%`；Box026 135 R inside `12.2%` 且 offset `~0.63m`。

因此下一轮不应直接把 sphere radius、3-box 或 handbox 当作单变量替换后继续大跑。优先级应是修上游 contact target/face assignment：raw target 如何映射到 G1 可达的手掌/支撑面，而不是继续让 reward 追 `wrist+5cm`。

## 运行命令

```bash
python workspace/core4d/scripts/E093/build_contact_geometry_manifest.py --force
python workspace/core4d/scripts/E093/audit_contact_geometry.py \
  --manifest workspace/core4d/results/E093/contact_geometry/case_manifest.tsv \
  --sample-count 6000
python workspace/core4d/scripts/E093/render_contact_geometry_mujoco.py \
  --manifest workspace/core4d/results/E093/contact_geometry/case_manifest.tsv \
  --points workspace/core4d/results/E093/contact_geometry/per_frame_points.csv \
  --camera auto \
  --width 960 \
  --height 720 \
  --video-frames 48
```

说明：raw contact target 从 raw mesh/person vertices 重新生成。`--sample-count 6000` 是下限；脚本优先使用各 `audit_summary_3cm.json` 里的 sample_count，当前实际 sample_count 为已有 audit 的配置，最多 capped 到 `30000`。

MuJoCo 可视化在用户指出“只能看到机器人上半身”后已纠偏：旧脚本默认使用 scene XML 里的 `track2` 命名相机，画面确实会裁掉腿/脚。当前 `render_contact_geometry_mujoco.py` 默认改为 `--camera auto`，每个 case 先扫多帧 qpos、机器人 body、object collision box 和 contact marker 的世界坐标，再生成固定 full-body free camera，避免视频内相机抖动，同时完整保留机器人、脚部、箱子和接触 marker。

## 结果路径

| 类型 | 路径 |
|---|---|
| plan | `workspace/core4d/plan/99_E093_contact_target_geometry_audit_plan.md` |
| scripts | `workspace/core4d/scripts/E093/` |
| manifest | `workspace/core4d/results/E093/contact_geometry/case_manifest.tsv` |
| summary | `workspace/core4d/results/E093/contact_geometry/geometry_summary.{csv,json,md}` |
| per-frame | `workspace/core4d/results/E093/contact_geometry/per_frame_points.csv` |
| object-local visuals | `workspace/core4d/results/E093/contact_geometry/visuals/object_local/` |
| timeline visuals | `workspace/core4d/results/E093/contact_geometry/visuals/timeline/` |
| dashboard | `workspace/core4d/results/E093/contact_geometry/visuals/dashboard/` |
| MuJoCo keyframes/videos | `workspace/core4d/results/E093/contact_geometry/visuals/mujoco/` |
| MuJoCo video QC frames | `workspace/core4d/results/E093/contact_geometry/visuals/mujoco/video_qc/` |
| high review | `workspace/core4d/results/E093/contact_geometry/visual_review/high_subagent_review.md` |

可视化完整性：

| 检查 | 结果 |
|---|---:|
| manifest ready | `7/7` |
| summary rows | `14` |
| per-frame rows | `1466` |
| object-local/timeline/dashboard PNG | `16/16` nonblank |
| MuJoCo keyframe sheets | `7/7` nonblank, auto full-body camera |
| MuJoCo mp4 decode | `7/7` PASS, each `960x720`, `48` frames |
| MuJoCo video QC frames | `2/2` extracted and visually checked (`box023_p2`, `box026_039_p2`) |

## 关键指标

| case | role | wrist5->raw mean L/R | wrist inside L/R | wrist support L/R | raw support L/R | handbox p90 L/R | sphere p90 L/R |
|---|---|---:|---:|---:|---:|---:|---:|
| `box023_p2` | known guard | `0.282/0.282` | `0.0/0.0%` | `100.0/58.1%` | `100.0/30.8%` | `0.257/0.173` | `0.273/0.212` |
| `box025_p2` | large-box partial | `0.578/0.641` | `0.0/9.7%` | `100.0/100.0%` | `94.8/7.8%` | `0.545/0.531` | `0.582/0.585` |
| `box004_083_p2` | E092 C1 WORK | `0.283/0.212` | `0.0/0.0%` | `42.9/24.8%` | `0.0/100.0%` | `0.163/0.198` | `0.209/0.223` |
| `box021_person1` | old control | `0.217/0.295` | `0.0/4.5%` | `98.9/75.0%` | `24.1/91.7%` | `0.110/0.184` | `0.159/0.221` |
| `box021_d003_029_p2` | known fail | `0.315/0.259` | `0.0/33.3%` | `9.3/77.3%` | `68.6/96.4%` | `0.256/0.186` | `0.292/0.213` |
| `box026_039_p2` | E092 C2 FAIL | `0.635/0.486` | `0.8/0.0%` | `19.5/15.4%` | `0.0/0.0%` | `0.529/0.437` | `0.575/0.455` |
| `box026_135_p2` | E092 C3 FAIL | `0.603/0.634` | `0.0/12.2%` | `62.2/41.5%` | `0.0/0.0%` | `0.538/0.589` | `0.564/0.610` |

## 可视化观察

high subagent 已完成只读复核，报告见：

`workspace/core4d/results/E093/contact_geometry/visual_review/high_subagent_review.md`

要点：

- raw 和 `wrist+5cm` 大多不是同面/同区域。
- sphere 不是唯一问题，但确实把接触简化成球面距离，无法表达平面接触和朝向。
- handbox 在数值上最接近 raw，但仍无法解决 raw target 与 retargeted hand region 错面的问题。
- box004/box023 能 work 是因为偏差仍在可达外侧/上沿附近；D003 box021/Box026 失败来自 inside、低 support、错面和大 offset 的组合。

## Claims 验证

| Claim | 结果 | 证据 |
|---|---|---|
| C1: `wrist+5cm` 与 raw contact 的偏移在 fail case 显著大于 work/guard case | PARTIAL | Box026/Box025 明显更大 (`0.49-0.64m`)；D003 box021 与 box023/box004 同为 `0.21-0.32m`，但 D003 有 inside/low-support/错面，所以偏移大小不是唯一判据。 |
| C2: sphere 几何会让接触位置语义不准 | PASS | `sphere_surface_gap_p90_abs_m` 全部 >`10cm`，Box026/Box025 `45-61cm`。 |
| C3: 旧 3-box 失败可由 reward/geometry mismatch 解释，而不是“面接触无价值” | PASS/PARTIAL | 3-box 多数略好于 sphere，但 p90 gap 仍大；历史 `box3 tip wrist+17.5cm` 与 reward `wrist+5cm` 错位仍成立。 |
| C4: handbox proxy 在 Box023/box004 pattern 上比 sphere/reward 更接近 raw | PASS relative / FAIL absolute | handbox 是 14/14 行 best proxy；Box023/box004 p90 通常低于 sphere，但仍 `0.16-0.26m`，不能直接当作解决方案。 |
| C5: box004 和 box023 属于容易 work 的相近 pattern | PARTIAL | 二者均无 wrist inside，尺寸较小/可达，offset 比 Box026/Box025 小；但 raw/wrist 并非严格同面，work 来自容忍度而非语义完全正确。 |

## 下一步建议

1. **先做 target semantic repair，不直接 rerun CEM/RL。** 对 Box026/D003，应把 raw contact face 重新映射到 G1 可达的 handbox/palm support patch，而不是继续追 `wrist+5cm`。
2. **把 handbox 作为 reward/physics proxy 候选，但不能只换几何。** handbox 相对 sphere 更接近 raw，可作为下一轮几何，但必须配套 target face assignment；否则只是让错误区域的 proxy 更平滑。
3. **保留 box004/box023 作为 positive guards。** 任何 repair 必须保持这两类小/中箱的 no-inside、可达外侧/上沿模式。
4. **Box026 需要单独处理大物体错面问题。** `039_p2` raw 与 wrist 基本错面且 support 低；`135_p2` 有 right-inside 风险。它们不应直接进入 RL。
5. **补一个 E094 计划：G1-handbox-aware target projection。** 输入 raw surface target，输出 G1 wrist/handbox 可达的 object-local support patch；先只做 kinematic + visualization gate，再决定是否接 SPIDER CEM。
