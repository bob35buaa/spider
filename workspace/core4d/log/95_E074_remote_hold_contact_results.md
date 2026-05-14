# E074 结果: ctrl guard 与 hold-contact reward 首轮远程并行

## 状态

E074 已完成远程 2-GPU 并行训练、scp 回收、本地评估与关键帧视觉复核。

结论简述：

- **E074A ctrl guard**: 数值小幅改善 contact/object/stability，视觉上更接近成功，最后箱子能与机器人分离，但 frame145-166 仍有手扶/拖拽和箱体倾斜。
- **E074C hold contact**: contact 指标明显提高，但 object error 变差，视觉上后段箱子紧贴腿部，有腿/箱异常接触或干涉；不是干净 hold/place。
- 两者都没有 early drift 或摔倒回归。

## 实验配置

| Run | Base | 单一变量 | 远程 GPU |
|-----|------|----------|----------|
| E074A | `core4d_e073_box023` | `ctrl_ref_guard_scale=0.5` | GPU0 |
| E074C | `core4d_e073_box023` | `hold_contact_rew_scale=2.0` | GPU1 |

Base 继承链：

```text
core4d_e073_box023
  -> core4d_e071w02_box023
  -> core4d_e062_box023
  -> core4d_e041c
```

远程信息：

| 项 | 值 |
|----|----|
| Host | `spider-remote` |
| Repo | `/home/xiayb/pHRI_workspace/spider` |
| Training HEAD | `b112eac` |
| Session | `tmux E074` |
| 启动时间 | 2026-05-14 21:02 |
| 完成时间 | 2026-05-14 21:21 |

## 结果路径

| 类型 | 路径 |
|------|------|
| E074A npz | `workspace/core4d/results/E074/E074A_box023.npz` |
| E074A video | `workspace/core4d/results/E074/E074A_box023.mp4` |
| E074C npz | `workspace/core4d/results/E074/E074C_box023.npz` |
| E074C video | `workspace/core4d/results/E074/E074C_box023.mp4` |
| 对比表 | `workspace/core4d/results/E074/comparison.csv` |
| timeseries | `workspace/core4d/results/E074/timeseries_E074{A,C}.csv` |
| keyframes | `workspace/core4d/results/E074/keyframes/{E074A,E074C}/` |
| logs | `logs/E074/` |

## 运行日志

两组均完成 272 sim steps，并保存 npz/mp4。

| Run | Total time | Final obj pos err | Final obj quat err | 备注 |
|-----|-----------:|------------------:|-------------------:|------|
| E074A | 1151.1s | 0.1095 | 0.4527 | 末尾 EGL destructor ignored exception，无害 |
| E074C | 1165.1s | 0.1279 | 0.4368 | 正常结束 |

启动日志确认关键路径：

- `Preserving raw ctrl reference for contact guidance (ctrl dims: 29 -> 35)` 生效。
- `E040 dynamic target ... uses_eef_offset=True` 生效。
- rotated-SDF mask active ratio = 41.9%。

## 量化指标

| 指标 | E073 | E074A ctrl guard | E074C hold contact | 判断 |
|------|-----:|-----------------:|-------------------:|------|
| yaw err 0.017/0.033 deg | 0.574 / 1.075 | 0.574 / 1.075 | 0.574 / 1.075 | 无 early drift 回归 |
| B1 pre-contact max foot z | 0.080m | 0.083m | 0.085m | 合格，轻微变高 |
| first obj_err >25cm | f100 | f100 | f100 | 未解决 2.0s 物体偏离 |
| first zero contact | f108 | f110 | f101 | A 小幅推迟，C 反而提前 |
| frame100-145 contact | 45.7% | 54.3% | 63.0% | C 最接近 65% 目标 |
| post2 contact | 49.4% | 54.3% | 64.2% | C 明显改善 |
| post2 obj_err max | 0.293m | 0.289m | 0.324m | A 小幅改善，C 退化 |
| post2 obj_err mean | 0.155m | 0.177m | 0.197m | 两者均退化，C 更明显 |
| post2 pelvis_z min | 0.663m | 0.692m | 0.701m | 稳定性改善 |
| first robot ctrl Linf >0.5 | f101 | f122 | f114 | A 符合 trust-region 预期 |
| post2 robot ctrl Linf max | 0.778 | 0.740 | 0.897 | C 使 robot ctrl 更激进 |
| post2 min hand SDF mean | 0.066m | 0.073m | 0.043m | C 确实让手更贴近 |
| post2 min hand SDF max | 0.222m | 0.324m | 0.167m | C 降低最大脱离距离 |

## Claims 验证

### E074A: robot ctrl trust-region guard

| Claim | 标准 | 结果 | 判定 |
|------|------|------|------|
| 不破坏 early drift | yaw <2deg, B1<=0.10m | yaw 0.574/1.075, B1=0.083m | ✅ |
| 推迟 ctrl 大偏离 | first robot ctrl Linf>0.5 晚于 E073 f101 | f122 | ✅ |
| 改善 contact | frame100-145 >=65% 或 post2 >=60% | 54.3% / 54.3% | ❌ |
| 降低 obj_err | post2 max <0.25m | 0.289m | ❌ |
| 保持稳定 | pelvis_z min >=0.55m | 0.692m | ✅ |
| 视觉 f130/f145 仍持箱/托箱 | 不能落地 | f130 尚可，f145 接近/刚接触地面 | ⚠️ |

E074A 是 **partial positive**：证明 ctrl guard 可以延后 robot ctrl 偏离并小幅改善整体行为，但强度不足以解决 hold。

### E074C: hold/contact continuity reward

| Claim | 标准 | 结果 | 判定 |
|------|------|------|------|
| 不破坏 early drift | yaw <2deg, B1<=0.10m | yaw 0.574/1.075, B1=0.085m | ✅ |
| 提升 contact | frame100-145 >=65% 或 post2 >=60% | 63.0% / 64.2% | ⚠️ 接近/通过一项 |
| 推迟 zero contact | first zero contact > f115，最好 > f130 | f101 | ❌ |
| 降低 obj_err | post2 max <0.25m | 0.324m | ❌ |
| 保持稳定 | pelvis_z min >=0.55m | 0.701m | ✅ |
| 视觉 f130/f145 仍持箱/托箱 | 不能落地/腿部干涉 | f145 基本落地，f160 后腿/箱干涉 | ❌ |

E074C 是 **metric-improving but visually unsafe**：contact/SDF surrogate 生效，但它优化的是“手更贴近箱”，不是“箱被稳定搬放并干净脱离”。

## 可视化复核

按用户要求，关键帧观察交给 subagent Erdos 完成。观察对象：

- `workspace/core4d/results/E074/keyframes/E074A/f100.jpg ... f180.jpg`
- `workspace/core4d/results/E074/keyframes/E074C/f100.jpg ... f180.jpg`

逐帧总结：

| 帧 | E074A | E074C |
|----|-------|-------|
| f100 | 箱子悬空，双手仍贴近/托住；未落地；机器人未摔倒 | 箱子悬空，双手托住；未落地；脚底接触标记明显 |
| f115 | 箱子仍悬空、被手持住；姿态稳定 | 箱子仍悬空、被手持住；脚底接触标记更明显 |
| f130 | 箱子悬空偏低，双手还在箱上方/侧面；机器人前倾但未倒 | 箱子悬空偏低，仍被手控制；双脚接触标记明显 |
| f145 | 箱子接近或刚接触地面，手仍压/扶在箱上；机器人未摔倒 | 箱子已基本落地，手仍在箱顶/侧面；机器人前倾较大 |
| f160 | 箱子已落地，手仍贴近/可能还在扶；箱体靠近腿部 | 箱子已落地，手臂压在箱内/箱顶附近；腿和箱很近，有干涉风险 |
| f166 | 箱子落地但姿态有倾斜/滑移，手仍靠近；机器人未倒 | 箱子落地，手仍伸入/贴近箱体；腿/箱非常近，疑似异常接触 |
| f180 | 箱子已放开并落地，与机器人有间距；机器人未摔倒，姿态较稳 | 箱子落地但紧贴机器人腿部，疑似腿/箱接触或干涉 |

视觉结论：

- **E074A 更接近成功**。最后箱子能落地并与机器人分离，机器人保持站立；主要问题是放置过程仍有手扶/拖拽，箱体中途略倾斜。
- **E074C 主要失败模式** 是落地后箱子与机器人腿部过近，后段出现腿/箱异常接触或干涉，放箱没有干净脱离。

## 分析

### 1. E074A 验证了 ctrl guard 的方向，但强度不够

E074A 将 first robot ctrl Linf>0.5 从 E073 的 f101 推迟到 f122，post2 robot ctrl Linf max 从 0.778 降到 0.740。说明 trust-region reward 的确约束了 CEM 控制偏离。

但 contact 只从 45.7%/49.4% 提升到 54.3%/54.3%，first zero contact 只从 f108 到 f110，未达到 f115/f130 目标。单独 ctrl guard 不足以维持 hold。

### 2. E074C 的 surrogate 生效，但目标错位

E074C 把 post2 min hand SDF mean 从 6.6cm 降到 4.3cm，post2 contact 从 49.4% 提到 64.2%，说明 hold_contact_rew 确实让手更贴近箱。

但 first zero contact 提前到 f101，post2 obj_err max 变差到 32.4cm，robot ctrl Linf max 增至 0.897。视觉上它更像“继续贴/压箱”，并把箱子留在腿附近，而不是完成稳定搬放。

这说明当前 hold_contact reward 还缺少两个要素：

- 接触方向/接触面质量约束：不是所有“贴近箱”的接触都对搬放有用。
- 放置后脱离/clearance 约束：不能让箱子落地后继续挤在腿边。

### 3. 当前成功标准仍未达到

两组都未解决核心标准：

- first obj_err >25cm 仍是 f100。
- f145 箱子仍接近/已经落地。
- E074C 的 contact 指标接近目标，但视觉不可接受。

所以 E074 不是最终成功，而是给下一步提供方向：**ctrl guard 是安全组件；hold-contact 需要改成更精确的“有用接触 + 放置脱离”目标。**

## 决策

下一步不建议直接上 E075 “E074A + E074C 原样组合”。原因：E074C 的副作用会把箱子吸到腿附近，组合后可能只是让这个失败模式更稳定。

推荐下一步：

1. **E075A = E074A + weaker hold_contact**  
   将 `hold_contact_rew_scale` 从 2.0 降到 0.5/1.0，并保留 ctrl guard，验证能否吃到 contact 提升而不引入腿/箱干涉。

2. **E075B = E074A + time-limited hold_contact**  
   只在 1.8-2.5s 激活 hold_contact，不覆盖放下后的 2.5-3.0s，避免落地后继续吸箱。

3. **E076 diagnostic = E074C leg/box clearance replay**  
   对 f145-f180 计算 box 与 foot/shank 的距离/接触，确认 E074C 后段是否是腿/箱物理干涉驱动。

优先级：E075B > E075A > E076。理由是当前失败集中在放下阶段，先缩短 hold_contact 时间窗最小侵入。

## Tracker 更新建议

E074 状态应标为 ⚠️ partial/mixed：

- A: partial positive, safest visual.
- C: metric positive but visually unsafe.
- 主线保留 ctrl guard；hold_contact 需降权或缩短时间窗后再组合。
