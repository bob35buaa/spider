# E072 结果: box023 post-2s 失败是 hold/contact 先失效，随后才摔倒

## 实验目的

E071 修复了 scene_act ctrl mapping 的 init bug，box023 0-2s 姿态已经明显正常。但用户指出约 2s 后机器人没有拿住箱子，并在放下阶段摔倒。

E072 不重新跑 CEM，不改 reward/config，只对 E071 保存轨迹做 MuJoCo `mj_forward` replay，量化 post-2s 的 hand-object distance/contact、object tracking、pelvis stability 和 ctrl divergence。

## 本次改动

| 文件 | 说明 |
|------|------|
| `workspace/core4d/plan/77_E072_post2_hold_place_diagnosis_plan.md` | E072 计划 |
| `workspace/core4d/scripts/eval/eval_E072.py` | replay 诊断脚本 |
| `workspace/core4d/scripts/train/train_E072.sh` | analysis-only 入口脚本 + keyframe 抽取 |
| `workspace/core4d/results/E072/` | 诊断输出 |
| `logs/E072/eval_E072.log` | 执行日志 |

## 运行命令

```bash
bash workspace/core4d/scripts/train/train_E072.sh
```

说明：运行日志里有 `Warp CUDA error 100: no CUDA-capable device is detected`，这是 import warp 时的 CUDA 探测警告。E072 使用 CPU MuJoCo `mj_forward` replay，不需要 CUDA，脚本 exit 0，输出完整。

## 输出

| 路径 | 内容 |
|------|------|
| `workspace/core4d/results/E072/timeseries.csv` | 每帧 obj/pelvis/hand/contact/ctrl 时间线 |
| `workspace/core4d/results/E072/diagnosis_summary.json` | first failure order 与统计摘要 |
| `workspace/core4d/results/E072/contact_summary.csv` | post-2s ref vs sim contact 汇总 |
| `workspace/core4d/results/E072/plots/post2_failure_timeline.png` | failure timeline 图 |
| `workspace/core4d/results/E072/keyframes/f100.jpg` 等 | frame-index 关键帧 |

## 时间轴说明

E071 历史指标按 `frame / 50 FPS` 标注，例如 frame 100 = 2.00s。E071 `.npz` 内保存的 `time` 实际约为 60Hz，例如 frame 100 的 `npz_time_s=1.683s`。

E072 的 `timeseries.csv` 同时输出：

- `eval_time_s = frame / 50`
- `npz_time_s = npz["time"]`

本 log 沿用 E071 的 `eval_time_s`，便于和 log 91 对齐。

## 关键指标

| 指标 | frame | eval_time_s | npz_time_s | 数值 |
|------|------:|------------:|-----------:|------|
| first obj_err > 25cm | 100 | 2.00 | 1.683 | obj_err=0.308m |
| first sim hand-object zero contact | 100 | 2.00 | 1.683 | sim contact=0, ref contact=1 |
| first robot ctrl Linf > 0.5 | 109 | 2.18 | 1.833 | robot_ctrl_linf=0.531 |
| first sim min hand SDF > 10cm | 134 | 2.68 | 2.250 | sim_min_sdf=0.101m |
| first pelvis_z < 45cm | 166 | 3.32 | 2.783 | pelvis_z=0.428m |
| first ref zero contact | 165 | 3.30 | 2.767 | ref 正常进入放下/离手 |
| first ref min hand SDF > 10cm | 168 | 3.36 | 2.817 | ref 已完成放下趋势 |

Post-2s 窗口统计：

| 指标 | sim | ref |
|------|-----|-----|
| hand-object contact frames | 44.4% | 80.2% |
| min hand SDF mean | 5.97cm | 7.52cm |
| min hand SDF max | 11.72cm | 38.81cm |

其他：

- post2 obj_err max/mean = 0.308 / 0.165m
- post2 pelvis_z min = 0.207m
- post2 robot_ctrl_linf max = 0.912
- post2 object_ctrl_linf max = 0.010，object actuator ctrl 不是直接触发点

## 关键帧观察

| frame | eval_time_s | 观察 |
|------:|------------:|------|
| 100 | 2.00 | sim 仍站得住，但箱体已相对 ref 明显错位；手在箱附近但 replay contact=0，ref contact=1 |
| 115 | 2.30 | sim 双手仍伸向箱体，但箱体偏移继续存在，ref 有双手接触，sim 无 hand-object contact |
| 130 | 2.60 | ref 开始放箱，sim 上身/手臂压向箱体上方，右脚出现不自然后伸，contact 仍为 0 |
| 145 | 2.90 | sim 重新碰到箱体，但姿态已是前扑/用身体压箱，pelvis 降到 0.685m |
| 160 | 3.20 | sim 已坐/压到箱子上，pelvis 继续下降 |
| 166 | 3.32 | pelvis_z 低于 0.45m，进入明确摔倒状态 |
| 180 | 3.60 | sim 已倒在箱体周围，ref 则已站立完成放下 |

## Claims 验证

| Claim | 结果 | 证据 |
|-------|------|------|
| C1: 可复现并定位 first failure order | PASS | summary 给出 frame 100 object/contact fail，frame 166 pelvis fall |
| C2: 区分 hold/contact vs stability 先失败 | PASS | contact/object failure 比 pelvis fall 早 66 frames / 1.32 eval-s |
| C3: 区分 ref 本身可行还是 sim 偏离 | PASS | frame 100-130 ref contact=1-2，sim contact=0；ref 直到 frame 165 才正常离手 |
| C4: 检查 ctrl divergence 是否是直接触发点 | PARTIAL | robot ctrl Linf 在 frame 109 超 0.5，早于 hand SDF >10cm，但晚于 frame 100 zero-contact/object fail；object ctrl 不触发 |
| C5: 可视化证据闭环 | PASS | frame 100-180 keyframes 已抽取并复核 |

## 结论

E072 将 E071 的 post-2s 失败排序钉住了：

1. frame 100 / eval 2.00s 时，object error 已达到 30.8cm，sim hand-object contact 已为 0；同一帧 ref 仍有 hand-object contact。
2. pelvis 在 frame 100-160 仍高于 0.64m，明确摔倒发生在 frame 166 / eval 3.32s。
3. robot ctrl 在 frame 109 后开始明显偏 ref，但 object ctrl 始终只差约 0.01，不是 object actuator ctrl mapping 的新问题。

因此当前一阶问题不是 putdown stability 先崩，而是 **hold/contact 在 2s 附近已经没有真实接触约束**。后续 robot 前扑、坐到箱子上、摔倒是为追 object/body target 的二阶后果。

## 下一步

E073 不应优先做 pelvis stability weight 或 box025 regression。推荐优先做 hold/contact 约束验证：

1. 在 1.8-3.0s 增加 hand-object contact/hold consistency 约束，目标是让 sim 在 ref 仍接触时保持至少单手真实 contact。
2. 同时加 robot ctrl trust-region 作为 guard，限制 frame 109 后 robot ctrl Linf 级偏移。
3. 成功标准先看 frame 100-145：sim contact frames 从 0 提升到接近 ref，object error 不再 2.0s 即 >25cm；再看 pelvis 是否自然改善。
