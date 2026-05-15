# E078 结果: 3cm per-EEF contact mask + box023 person1/person2 CEM

## 状态

E078 已完成：

- E078A (`box023_person1`)：远程 GPU0 因其他进程占用严重变慢，已停止远程 run，改在本机 RTX 5090 完成。
- E078B (`box023_person2`)：远程 GPU1 完成，已 scp 回收。
- 本地统一评估已重跑，`comparison.csv` 包含 E078A/E078B。

结论简述：

- **E078A / person1 没有实质改善**。虽然 3cm per-EEF mask 接入成功，但 p1 f119-f125 右脚仍明显大步前跨，`right_hip_pitch` ctrl 偏差仍大，动作质量和 E075B 接近甚至部分 contact 指标更差。当前更像是 p1 原始/retarget 数据质量和接触边界问题，而不是 scalar mask 这一层能单独修好的 reward gating 问题。
- **E078B / person2 明显更接近可用质量**。person2 ref 接触本身更扎实，3cm mask 在 f115-f130 双手全开，sim 的右脚 step 和 ref 接近，ctrl 偏差小，整体动作稳定。该结果支持用户判断：person2 数据质量更好，动力学重定向更容易得到可用动作。

## 实验配置

| Run | Case | Base | Contact mask | 执行位置 |
|-----|------|------|--------------|----------|
| E078A | `box023_person1` | E075B config | E077 3cm `person_idx=0`, per-EEF | 本机 RTX 5090 |
| E078B | `box023_person2` | E075B-like p2 config | E077 3cm `person_idx=1`, per-EEF | 远程 GPU1 |

实现改动：

- `contact_hdmi_mask_source="core4d_3cm"` 从 `workspace/core4d/results/E077/contact_masks/box023/raw_contact_mask_3cm.npz` 读取 mask。
- `contact_hdmi` reward 支持 HDMI-style per-EEF `(T,2)` / `(N,2)` mask。
- 旧默认仍是 `rotated_sdf`，不影响未显式开启 3cm mask 的实验。

训练与评估脚本：

| 类型 | 路径 |
|------|------|
| train | `workspace/core4d/scripts/train/train_E078.sh` |
| eval | `workspace/core4d/scripts/eval/eval_E078.py` |
| remote | `workspace/core4d/scripts/run_E078_remote.sh` |
| pull | `workspace/core4d/scripts/pull_E078_remote_results.sh` |

## 结果路径

| 类型 | 路径 |
|------|------|
| E078A npz | `workspace/core4d/results/E078/E078A_box023_p1.npz` |
| E078A video | `workspace/core4d/results/E078/E078A_box023_p1.mp4` |
| E078B npz | `workspace/core4d/results/E078/E078B_box023_p2.npz` |
| E078B video | `workspace/core4d/results/E078/E078B_box023_p2.mp4` |
| comparison | `workspace/core4d/results/E078/comparison.csv` |
| summary | `workspace/core4d/results/E078/eval_summary_E078{A,B}.json` |
| timeseries | `workspace/core4d/results/E078/timeseries_E078{A,B}.csv` |
| keyframes | `workspace/core4d/results/E078/keyframes/E078{A,B}/` |
| plots | `workspace/core4d/results/E078/plots/` |
| logs | `logs/E078/` |

## 量化指标

### 与 E075B person1 对比

| 指标 | E075B p1 | E078A p1 3cm per-EEF | 判断 |
|------|---------:|---------------------:|------|
| yaw err 0.017/0.033 deg | 0.574 / 1.075 | 0.574 / 1.075 | early drift 不回归 |
| B1 pre-contact max foot z | 0.083m | 0.084m | 基本持平 |
| first obj_err >25cm | f100 | f100 | 未改善 |
| first zero contact | f110 | f114 | 略晚，但不解决核心问题 |
| frame100-145 contact | 76.1% | 69.6% | 变差 |
| post2 contact | 67.9% | 61.7% | 变差 |
| post2 obj_err max | 0.287m | 0.287m | 持平 |
| post2 obj_err mean | 0.158m | 0.160m | 持平/略差 |
| post2 pelvis_z min | 0.660m | 0.663m | 稳定 |
| first robot ctrl Linf >0.5 | f117 | f116 | 无改善 |
| post2 robot ctrl Linf max | 0.690 | 0.688 | 持平 |
| post2 min hand SDF mean | 0.057m | 0.081m | 变差 |
| post2 min hand SDF max | 0.194m | 0.429m | 明显变差 |

E078A 说明：3cm per-EEF mask 能跑通，但没有带来 p1 的质量提升。contact 相关指标反而低于 E075B，object tracking 只是持平，右腿 phase mismatch 仍在。

### E078A vs E078B

| 指标 | E078A p1 | E078B p2 | 判断 |
|------|---------:|---------:|------|
| yaw err 0.017/0.033 deg | 0.574 / 1.075 | 0.041 / 0.126 | p2 初始对齐更好 |
| B1 pre-contact max foot z | 0.084m | 0.081m | 都合格 |
| first obj_err >25cm | f100 | f100 | 两者都早期物体误差超过阈值 |
| frame100-145 contact | 69.6% | 78.3% | p2 更好 |
| post2 contact | 61.7% | 65.4% | p2 略好 |
| post2 obj_err max | 0.287m | 0.308m | p1 数值略好，但 p2 动作更稳定 |
| post2 obj_err mean | 0.160m | 0.156m | p2 略好 |
| post2 pelvis_z min | 0.663m | 0.700m | p2 更稳 |
| first robot ctrl Linf >0.5 | f116 | f140 | p2 大偏离明显更晚 |
| post2 robot ctrl Linf max | 0.688 | 0.617 | p2 更低 |
| f119-f125 right foot step sim/ref | 0.531 / 0.159m | 0.0219 / 0.0255m | p2 基本跟 ref，p1 严重偏离 |
| f120-f124 right hip pitch ctrl diff abs max | 0.554rad | 0.138rad | p2 明显更好 |

注意：E078A 和 E078B 不是严格同一 reference motion 的公平对比。E077 已确认 p1/p2 retarget 层存在 person-specific `smpl_scale` 导致的 object qpos 差异，因此不能直接把 p2 当作 p1 的替代结论。这里的含义是：**在单人 CEM 动力学重定向任务上，person2 的 ref/contact 数据质量明显更利于成功**。

## Contact Mask 复核

| Run | mask key | person_idx | global L/R active | f115-f130 L/R active |
|-----|----------|------------|-------------------|----------------------|
| E078A | `eval_contact_mask_3cm` | 0 | 46.3% / 45.2% | 100.0% / 87.5% |
| E078B | `eval_contact_mask_3cm` | 1 | 44.9% / 47.8% | 100.0% / 100.0% |

解释：

- E078A 的 p1 右手在 f115-f130 不是完全没有接触，但仍不是稳定强接触；E076/E077 已看到 p1 右手均距约 2.12cm，属于 3cm 阈值边界接触。
- E078B 的 p2 双手在同窗口是强接触，mask 和 ref 动作更一致。
- 因此 E078A 的失败不能再归因于“mask 没打开右手”这么简单；更可能是 p1 ref/retarget 的接触几何和动力学可执行性本身偏弱。

## 可视化观察

视频已生成并抽取关键帧：

- `workspace/core4d/results/E078/keyframes/E078A/f100..f180.jpg`
- `workspace/core4d/results/E078/keyframes/E078B/f100..f180.jpg`

实际观察与用户反馈一致：

- **E078A / p1**：f120 前后仍出现明显右腿相位偏差。ref 在准备弯腰放箱时，sim 右腿继续大步前跨，f119-f125 的右脚 XY 位移约为 ref 的 3.3 倍；动作质量没有比 E075B 明显改善。虽然最终没有摔倒，箱子和姿态仍不像一个可靠的放置动作。
- **E078B / p2**：f115-f130 的右脚移动和 ref 接近，没有 p1 那种突然大跨步；双手接触窗口更稳定，sim 动作更顺，整体更接近能用的质量。p2 的结果说明 CEM 和 reward stack 并非完全不能工作，输入 ref/contact 质量对结果影响很大。

## Claims 验证

| Claim | 结果 | 判定 |
|-------|------|------|
| C1: MJWP 支持 HDMI-style per-EEF mask | E078A/E078B 均成功读取 `(T,2)` 3cm mask 并完整运行 | PASS |
| C2: E077 3cm mask 可按 case/person 读取 | p1 使用 `person_idx=0`，p2 使用 `person_idx=1`；eval summary 记录 mask key/person/active pct | PASS |
| C3: p1 f115-f130 gait/putdown mismatch 减轻 | E078A 右脚 step 和 right hip pitch 偏差仍大，contact 低于 E075B | FAIL |
| C4: p2 单人 case 能完成 CEM sanity run | E078B 完整生成 npz/mp4/metrics，动作质量更接近可用 | PASS |
| C5: 不继续依赖 hand-crafted hold window | 本轮主变量是 contact source/per-EEF mask；未新增 case-specific window | PASS |

## 关键结论

1. **3cm per-EEF mask 是必要的接口修正，但不是 p1 的充分修复。**
   E078A 没有解决 person1 的 f120-f125 右腿前跨，说明错误不主要来自 scalar mask 把左右手混在一起，而更可能来自 p1 reference/retarget 的接触质量、姿态相位、或 G1 可执行性。

2. **person1 的问题更像数据质量问题。**
   E076/E077 看到 p1 右手属于边界接触，E078A 即使按 3cm 打开右手 reward，也没有得到更自然的动力学解。CEM 仍通过迈右腿/调整重心去满足任务，说明 reference 本身给出的物理约束不够可靠。

3. **person2 是更有希望的单人重定向方向。**
   person2 原始接触更扎实，E078B 的右脚 step、hip ctrl、pelvis 稳定性都明显更好。后续如果目标是得到可用动作，应优先沿 person2 做更完整评估。

4. **暂时不要把 p1+p2 直接合成双机器人。**
   E077 的 common-scale caveat 仍成立：p1/p2 retarget 层 object qpos 最大差约 6.2cm。双机器人同场景前必须解决 common-scale/common-world alignment。

## 下一步建议

短期建议：

- 把 E078B/person2 作为下一步单人 CEM 主线，先做更完整的视觉复核和 downstream 可用性检查。
- 对 E078A/person1 暂停 reward 调参，不再继续围绕 p1 的 contact window/weight 做局部修补。
- 做一个 p1 vs p2 的数据质量报告：raw hand-object distance、retarget hand-object SDF、right foot phase、object pose scale/alignment，明确哪些误差来自 raw、哪些来自 retarget。

中期建议：

- 若要做人机协作/双人搬箱，应先做 p1/p2 common-scale/common-world 对齐，再讨论双机器人同场景。
- 若以单机器人可用动作为目标，可以优先基于 person2 拓展到更多 CORE4D case，筛选出“ref 接触质量好”的序列，而不是在 p1 边界接触样本上继续 reward 调参。
