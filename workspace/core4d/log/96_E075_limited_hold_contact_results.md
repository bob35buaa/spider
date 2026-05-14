# E075 结果: ctrl guard + 限时弱化 hold_contact

## 状态

E075 已完成远程 2-GPU 并行训练、scp 回收、本地评估与 subagent 关键帧视觉复核。

结论简述：

- **E075B (`scale=1.0`, 1.8-2.5s)** 是当前最好的组合：contact 明显高于 E074A/E074C，object error 没有 E074C 的退化，robot ctrl 偏离更低，视觉上最终箱子落地且机器人站稳。
- **E075A (`scale=0.5`, 1.8-2.5s)** 数值上 hand SDF/contact 也改善，但稳定性失败：frame168 pelvis_z <45cm，视觉 f166-f180 摔倒/腿箱干涉明显。
- 限时 window 明显缓解了 E074C 的后段持续吸箱问题；但 E075B 仍不是最终成功，f145-f166 释放不够干净。

## 实验配置

| Run | Base | 变量 | 远程 GPU |
|-----|------|------|----------|
| E075B | `core4d_e074a_box023` | `hold_contact_rew_scale=1.0`, window 1.8-2.5s | GPU0 |
| E075A | `core4d_e074a_box023` | `hold_contact_rew_scale=0.5`, window 1.8-2.5s | GPU1 |

Base 继承链：

```text
core4d_e075{a,b}_box023
  -> core4d_e074a_box023
  -> core4d_e073_box023
  -> core4d_e071w02_box023
  -> core4d_e062_box023
  -> core4d_e041c
```

远程信息：

| 项 | 值 |
|----|----|
| Host | `spider-remote` |
| Repo | `/home/xiayb/pHRI_workspace/spider` |
| Training HEAD | `90d5b34` |
| Session | `tmux E075` |
| 启动时间 | 2026-05-14 21:57 |
| 完成/回收时间 | 2026-05-14 22:19 |

## 结果路径

| 类型 | 路径 |
|------|------|
| E075B npz | `workspace/core4d/results/E075/E075B_box023.npz` |
| E075B video | `workspace/core4d/results/E075/E075B_box023.mp4` |
| E075A npz | `workspace/core4d/results/E075/E075A_box023.npz` |
| E075A video | `workspace/core4d/results/E075/E075A_box023.mp4` |
| 对比表 | `workspace/core4d/results/E075/comparison.csv` |
| timeseries | `workspace/core4d/results/E075/timeseries_E075{A,B}.csv` |
| keyframes | `workspace/core4d/results/E075/keyframes/{E075A,E075B}/` |
| plots | `workspace/core4d/results/E075/plots/` |
| scene snapshot | `workspace/core4d/results/E075/scene_snapshot/` |
| logs | `logs/E075/` |

## 量化指标

| 指标 | E073 | E074A ctrl guard | E074C hold contact | E075B limited scale1.0 | E075A limited scale0.5 | 判断 |
|------|-----:|-----------------:|-------------------:|-----------------------:|-----------------------:|------|
| yaw err 0.017/0.033 deg | 0.574 / 1.075 | 0.574 / 1.075 | 0.574 / 1.075 | 0.574 / 1.075 | 0.574 / 1.075 | 无 early drift 回归 |
| B1 pre-contact max foot z | 0.080m | 0.083m | 0.085m | 0.083m | 0.085m | 合格 |
| first obj_err >25cm | f100 | f100 | f100 | f100 | f100 | 仍未解决 2.0s 物体偏离 |
| first zero contact | f108 | f110 | f101 | f110 | f111 | E075 都优于 E074C |
| frame100-145 contact | 45.7% | 54.3% | 63.0% | **76.1%** | 60.9% | E075B 最好 |
| post2 contact | 49.4% | 54.3% | 64.2% | **67.9%** | 61.7% | E075B 最好 |
| post2 obj_err max | 0.293m | 0.289m | 0.324m | **0.287m** | 0.290m | E075B 略优 E074A |
| post2 obj_err mean | 0.155m | 0.177m | 0.197m | 0.158m | **0.155m** | E075B/A 均优于 E074A/C |
| post2 pelvis_z min | 0.663m | 0.692m | 0.701m | 0.660m | **0.147m** | E075A 摔倒回归 |
| first robot ctrl Linf >0.5 | f101 | f122 | f114 | f117 | **f100** | E075B 保留 guard；E075A 不稳 |
| post2 robot ctrl Linf max | 0.778 | 0.740 | 0.897 | **0.690** | 0.662 | E075B/A 后段 max 较低 |
| post2 min hand SDF mean | 0.066m | 0.073m | 0.043m | 0.057m | 0.054m | E075 让手更贴近 |
| post2 min hand SDF max | 0.222m | 0.324m | 0.167m | 0.194m | **0.095m** | E075A 贴得更近但摔倒 |

## Claims 验证

### C1: 不破坏 early drift

| Run | 标准 | 结果 | 判定 |
|-----|------|------|------|
| E075B | yaw <2deg, B1<=0.10m | yaw 0.574/1.075, B1=0.083m | PASS |
| E075A | yaw <2deg, B1<=0.10m | yaw 0.574/1.075, B1=0.085m | PASS |

### C2: 保留 E074A 的 robot ctrl 安全性

| Run | 标准 | 结果 | 判定 |
|-----|------|------|------|
| E075B | first robot ctrl Linf>0.5 晚于 E073 f101，post2 max <=0.80 | f117, max=0.690 | PASS |
| E075A | 同上 | f100, max=0.662 | MIXED/FAIL |

### C3: 提升接触且避免 E074C 后段干涉

| Run | 标准 | 结果 | 判定 |
|-----|------|------|------|
| E075B | frame100-145 >=60%, post2>=60%, 视觉不贴腿/不摔 | 76.1%, 67.9%, f180 站稳分离 | PASS |
| E075A | 同上 | 60.9%, 61.7%, f166-f180 严重干涉/摔倒 | FAIL |

### C4: object tracking 不比 E074C 坏

| Run | 标准 | 结果 | 判定 |
|-----|------|------|------|
| E075B | post2 max <0.30m，mean <=0.177m | max=0.287m, mean=0.158m | PASS |
| E075A | 同上 | max=0.290m, mean=0.155m，但摔倒 | MIXED |

## 可视化复核

按用户要求，关键帧观察交给 subagent Erdos 完成。观察对象：

- `workspace/core4d/results/E075/keyframes/E075B/f100.jpg ... f180.jpg`
- `workspace/core4d/results/E075/keyframes/E075A/f100.jpg ... f180.jpg`

逐帧总结：

| 帧 | E075B | E075A |
|----|-------|-------|
| f100 | 箱子悬空，双手托住/持住；未落地；机器人未摔倒，无明显腿箱干涉 | 箱子悬空，双手托住/持住；未落地；机器人未摔倒 |
| f115 | 箱子仍悬空被双手持住；未落地；机器人未倒 | 箱子仍悬空被持住；未落地；机器人未倒 |
| f130 | 箱子悬空偏低，双手仍在控制；未落地；机器人未倒 | 箱子悬空且较高，双手仍扶/推；机器人未倒但身体后撤/不太跟参考 |
| f145 | 箱子接近/开始落地，手仍贴箱；机器人未倒；箱子未明显贴腿 | 箱子接近落地，手仍在箱顶/侧；机器人前倾明显，箱子靠近脚 |
| f160 | 箱子落地，双手还贴近/扶箱；机器人未倒；腿箱距离近但未明显压上去 | 箱子落地，双手仍压/扶；机器人前倾更重，有干涉风险 |
| f166 | 箱子落地，双手还在箱边/上方，释放不干净；机器人未倒；箱子略靠腿但可分离 | 明显失败：机器人上身/手臂压到箱子上，腿箱严重干涉，失去正常站姿 |
| f180 | 箱子落地并与机器人分离；机器人站立未摔倒；放下较干净 | 严重失败：机器人摔倒/趴在箱子和地面附近，箱子与身体/腿强干涉 |

视觉结论：

- **E075B 明显优于 E075A，更接近成功**。
- E075B 的主要残余问题是 f145-f166 放箱阶段手仍持续贴箱，释放不够干净；但 f180 箱子落地、机器人站稳、箱子与腿基本分离。
- E075A 是明显失败：f166 开始压箱/腿箱强干涉，f180 摔倒。

## 分析

### 1. 限时 hold_contact 是有效方向

E074C 的主要副作用是 1.8-3.0s 持续吸引手靠近箱子，导致放下后仍贴/压箱并靠近腿部。E075B 将窗口截到 2.5s 后：

- post2 contact 从 E074A 的 54.3% 提到 67.9%。
- frame100-145 contact 从 E074A 的 54.3% 提到 76.1%。
- post2 obj_err max 0.287m，未复现 E074C 的 0.324m 退化。
- 视觉 f180 能分离站稳。

这说明 E074C 的 proximity surrogate 不是完全错误；错误主要在于过强/过长，特别是不应覆盖放置释放阶段。

### 2. scale=0.5 并不更安全

E075A 权重更弱，但视觉和 pelvis 指标更差：

- first robot ctrl Linf>0.5 提前到 f100。
- first pelvis_z<45cm 出现在 f168。
- f166-f180 出现明显身体/腿/箱干涉并摔倒。

所以 E075A 不是“更弱更安全”，更可能是 hold support 不足，CEM 转而用更差的 body/leg interaction 追踪目标。后续不建议沿 scale=0.5 继续。

### 3. E075B 仍未解决最终任务

E075B 仍有两个明确缺口：

- first obj_err >25cm 仍是 f100，说明 2.0s 的 object tracking 偏离没有根治。
- f145-f166 手仍贴/扶箱，释放不够干净；当前 reward 还没有表达“落地后脱离/clearance”。

因此 E075B 可作为新的主线 base，但下一步必须补 placement/release 约束，而不是继续单纯增大 hold_contact。

## 决策

下一步建议：

1. **主线采用 E075B**：保留 `ctrl_ref_guard_scale=0.5` + `hold_contact_rew_scale=1.0` + hold window 1.8-2.5s。
2. **不要采用 E075A**：scale=0.5 发生稳定性回归。
3. **E076 方向**：在 E075B 基础上增加放置后 release/clearance 设计，重点是 2.5s 后手/腿与箱子的距离或接触质量，而不是继续增强 proximity。

候选 E076：

- E076D: 先做 replay 诊断，量化 f145-f180 hand/leg/box SDF/contact，确认 release 和 leg interference 的先后顺序。
- E076A: E075B + 2.5-3.3s hand-object release clearance reward，鼓励手离开箱子但不推倒箱子。
- E076B: E075B + leg/box clearance penalty，避免箱子靠近脚/小腿。

优先级：E076D diagnostic -> E076A/B 小规模并行。理由是 E075B 已接近可用，下一步要避免把 reward 写成新的错目标。
