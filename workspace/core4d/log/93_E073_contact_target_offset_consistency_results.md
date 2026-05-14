# E073 结果: dynamic contact target 的 eef_offset 口径修正改善稳定性，但未解决 post-2s hold

## 实验目的

E072 证明 E071 的 post-2s failure 顺序是：

1. frame100 / eval 2.00s: object error 已 >25cm，sim hand-object contact=0，而 ref 仍 contact=1。
2. frame166 / eval 3.32s: pelvis_z 才低于 45cm，摔倒是后果不是起点。

E073 验证一个几何口径假设：E040 dynamic contact target precompute 使用 ref wrist body origin，但 reward 端用 sim `wrist + contact_hdmi_eef_offset` 作为接触点。E073 将 target 也改为 ref `wrist + eef_offset`，使 reward 追踪口径一致。

## 本次改动

| 文件 | 说明 |
|------|------|
| `spider/config.py` | 新增 `contact_hdmi_target_uses_eef_offset: bool = False`，默认保持历史兼容 |
| `examples/run_mjwp.py` | dynamic contact target 可选使用 ref `wrist + eef_offset` |
| `examples/config/override/core4d_e073_box023.yaml` | 继承 E071，打开 `contact_hdmi_target_uses_eef_offset: true` |
| `workspace/core4d/scripts/train/train_E073.sh` | E073 训练、snapshot、eval、关键帧入口 |
| `workspace/core4d/scripts/eval/eval_E073.py` | E073 replay/eval 指标与关键图输出 |
| `workspace/core4d/results/E073/` | 训练与评估输出 |

E073 不改 reward 权重、不改 contact gain、不改 stability/task_obj，只修 contact target 几何口径。

## 运行命令

```bash
bash workspace/core4d/scripts/train/train_E073.sh 0
```

运行日志确认：

- GPU: RTX 5090 可见。
- `E040 dynamic target: shape=(322, 2, 3), uses_eef_offset=True`
- 训练与评估 exit 0。

## 关键指标

| 指标 | E071/E072 baseline | E073 | 结论 |
|------|-------------------:|-----:|------|
| yaw err t=0.017 / 0.033 | 0.574 / 1.075 deg | 0.574 / 1.075 deg | early drift 未回归 |
| B1 pre-contact max foot z | 0.069m | 0.080m | 小幅退化但仍 <=0.10m |
| first obj_err >25cm | frame100 / 2.00s | frame100 / 2.00s | 未解决 2s object error |
| post2 obj_err max | 0.308m | 0.293m | 小幅改善 1.5cm |
| first sim zero contact | frame100 / 2.00s | frame108 / 2.16s | contact 丢失被推迟 8 frames |
| post2 sim contact frames | 44.4% | 49.4% | 小幅改善 +4.9pp |
| post2 ref contact frames | 80.2% | 80.2% | ref 不变 |
| frame100-145 sim contact frames | E072 关键帧 f100/f115/f130 为 0 | 45.7% | 早段 contact 有改善但不连续 |
| post2 pelvis_z min | 0.207m | 0.663m | 摔倒消失 |
| first pelvis_z <45cm | frame166 / 3.32s | none | 稳定性明显改善 |
| post2 robot ctrl Linf max | 0.912 | 0.778 | 最大偏移降低 |

输出文件：

| 路径 | 内容 |
|------|------|
| `workspace/core4d/results/E073/eval_summary.json` | 指标摘要 |
| `workspace/core4d/results/E073/timeseries.csv` | 每帧 contact/object/pelvis/control 时间线 |
| `workspace/core4d/results/E073/plots/post2_failure_timeline.png` | timeline 图 |
| `workspace/core4d/results/E073/keyframes/` | frame100-180 关键帧 |

## 关键帧数值对照

| frame | eval_time_s | obj_err | sim contact | ref contact | pelvis_z | 观察 |
|------:|------------:|--------:|------------:|------------:|---------:|------|
| 100 | 2.00 | 0.293m | 1 | 1 | 0.767m | contact 比 E072 好，但 object error 已 >25cm |
| 108 | 2.16 | 0.279m | 0 | 2 | 0.789m | 首次 sim zero contact |
| 115 | 2.30 | 0.262m | 1 | 2 | 0.780m | 仍有单手 contact |
| 130 | 2.60 | 0.163m | 0 | 2 | 0.774m | contact 再次丢失 |
| 145 | 2.90 | 0.110m | 1 | 2 | 0.730m | 重新碰到箱，但视觉上已像落地/压箱 |
| 160 | 3.20 | 0.108m | 0 | 1 | 0.700m | 箱已在地面，robot 未倒 |
| 166 | 3.32 | 0.105m | 0 | 0 | 0.672m | E071 此时已摔，E073 仍站立 |
| 180 | 3.60 | 0.051m | 0 | 0 | 0.788m | 终段站立，但不是持箱 |

## Subagent 视觉复核

按用户要求，主线程没有直接 `view_image`；关键帧由 subagent Ampere 复核。

视觉结论：

- E073 在 2s 后仍没有稳定拿住箱子。
- f100/f115 手和箱还较近，箱子没有立刻飞走。
- f130 起手-箱接触明显变差，双手不再像抓/托住箱子。
- f145 箱子已经明显落到地面或接触地面，可视为脱手/掉落开始。
- f166/f168 机器人没有像 E071/E072 那样横向翻倒，仍保持弯腰站姿。
- 但后段出现脚/腿踩进或压到箱体的异常接触，不是自然放箱。

逐帧视觉摘要：

| frame | 视觉观察 |
|------:|----------|
| f100 | 箱子仍在手前方，手和箱距离较近，尚未明显脱手 |
| f115 | 箱子仍悬在前方，双手靠近箱体，接触还算连续 |
| f130 | 箱子开始下沉/偏离稳定抓握，手更像压在上方或擦过箱体 |
| f145 | 箱子已明显落到地面附近/地面上，基本不是稳定持箱 |
| f160 | 箱子在地面，机器人弯腰贴近，手没有可靠抓住箱子 |
| f166 | 机器人未摔倒，但腿/脚与箱体严重异常接触 |
| f168 | 异常接触持续，身体仍弯腰站立，没有 E071/E072 的翻倒 |
| f180 | 箱子仍在地面，机器人站在箱旁/箱上方，未持箱但未倒地 |

## Claims 验证

| Claim | 结果 | 证据 |
|-------|------|------|
| C1: dynamic target 口径修正生效 | PASS | 训练日志打印 `uses_eef_offset=True` |
| C2: 0-2s early drift 不回归 | PASS | yaw 0.574/1.075 deg，B1=0.080m |
| C3: frame100-145 hold/contact 改善 | PARTIAL | post2 contact 44.4%→49.4%，first zero contact 100→108，但仍大量断触 |
| C4: object error 不在 frame100 即失控 | FAIL | first obj_err >25cm 仍是 frame100 |
| C5: 不用摔倒换 contact | PASS | first pelvis_z<45cm 不再出现；视觉也未见 3.3s 翻倒 |

## 结论

E073 证明 dynamic contact target 的 eef_offset 口径不一致确实是一个有效问题，但不是 post-2s hold failure 的主因。

有效部分：

- target/reward 口径一致后，早期 contact 不再 frame100 立即全丢，first zero contact 推迟到 frame108。
- post2 contact 从 44.4% 提到 49.4%。
- 最大 object error 从 30.8cm 降到 29.3cm。
- 机器人 3.3s 左右不再摔倒，pelvis_z 全程保持 >=0.663m。

未解决部分：

- object error 仍在 frame100 / 2.00s 立即超过 25cm。
- f130 后视觉上已不是稳定抓/托箱，f145 箱子落地。
- contact frames 仍远低于 ref 的 80.2%。
- 后段稳定性改善有一部分来自“不倒但没拿住”，并伴随脚/腿与箱体异常接触。

因此 E073 应保留为默认关闭的几何修正，并作为下一轮实验的 base；但不能把它记为 hold 成功。

## 下一步建议

E074 不应继续单纯修 target 口径或只加 stability。当前更明确的问题是：CEM 在 2.0s 附近仍允许 robot ctrl 快速偏离 ref，导致手-箱接触不连续。

推荐 E074：

1. 继承 E073 的 `contact_hdmi_target_uses_eef_offset=true`。
2. 新增 robot ctrl trust-region guard，惩罚/限制 robot ctrl 相对 `ctrl_ref` 的大幅偏移，重点看 frame100-145 是否减少断触。
3. 成功标准不只看 pelvis_z，必须同时要求 f130/f145 视觉上仍是持箱/托箱，而不是箱已落地。
