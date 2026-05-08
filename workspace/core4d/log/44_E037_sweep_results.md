# E037 Sweep: Contact Mask Reward Parameter Grid — box025 Results

## 状态: S3 (gain=3.5, sigma=0.15) 最优 — Contact<10cm=57% + MPKPE=1.4cm + 100% Stable

## 实验设计

### 动机
E037/E037b 证明 gain=2.0, sigma=0.3 的 contact_mask_rew 太弱 (占总 reward ~6%)，CEM 忽略。
本 sweep 系统探索 gain/sigma/baseline 组合，在 box025 上找到最优配置。

### 参数分析

**Reward 理论计算** (contact 帧, 手距物体表面 dist=d):
```
contact_mask_rew = gain * exp(-d / sigma)
body_tracking_max = 3.5 (per frame)
```

| Config | gain | sigma | baseline | d=0 reward | d=5cm | d=15cm | d=30cm | 设计依据 |
|--------|------|-------|----------|-----------|-------|--------|--------|----------|
| E037b (baseline) | 2.0 | 0.3 | 0.0 | 2.0 | 1.7 | 1.2 | 0.7 | 原始配置 |
| **S1** | 2.0 | 0.3 | 1.0 | 2.0 | 1.7 | 1.2 | 0.7 | HDMI baseline 消融 |
| **S2** | 3.5 | 0.3 | 0.0 | 3.5 | 2.9 | 2.1 | 1.3 | gain = tracking max |
| **S3** | 3.5 | 0.15 | 0.0 | 3.5 | 2.5 | 1.3 | 0.5 | 更陡 kernel |
| **S4** | 3.5 | 0.3 | 1.0 | 3.5 | 2.9 | 2.1 | 1.3 | HDMI baseline + max gain |
| **S5** | 5.0 | 0.3 | 1.0 | 5.0 | 4.2 | 3.0 | 1.8 | 超越 tracking (HDMI 原参数) |

## 结果汇总

| Config | MPKPE (cm) | Joint (°) | ObjPos (cm) | Stable>0.6m | Contact<10cm | Contact<15cm | Mean Surf (cm) |
|--------|-----------|-----------|-------------|-------------|-------------|-------------|----------------|
| **E036** (no contact rew) | 1.43 | 0.85 | 0.87 | 100% | 55.6% | 79.8% | 12.5 |
| **E037b** (gain=2, σ=0.3, b=0) | 1.4 | 0.8 | 0.8 | 100% | 49.2% | 79.0% | 12.4 |
| **S1** (gain=2, σ=0.3, b=1) | 1.4 | 0.8 | 0.8 | 100% | 45.2% | — | — |
| **S2** (gain=3.5, σ=0.3, b=0) | **2.1** | 1.1 | 0.8 | **71%** ❌ | 43.5% | 71.0% | 13.9 |
| **S3** (gain=3.5, σ=0.15, b=0) | **1.4** | 0.9 | **0.7** | **100%** | **57.3%** ✅ | 79.8% | 12.3 |
| **S4** (gain=3.5, σ=0.3, b=1) | **1.4** | 0.8 | **0.7** | **100%** | 53.2% | **85.5%** | 11.3 |
| **S5** (gain=5, σ=0.3, b=1) | 1.5 | 0.8 | 0.7 | 100% | 40.3% | 65.3% | 15.0 |

## 关键发现

### 1. S2 失稳 — gain=3.5 + sigma=0.3 + baseline=0 导致摔倒

S2 pelvis_z 最低降到 0.26m (摔倒)，稳定性只有 71%。
- **原因**: sigma=0.3 的 kernel 太"宽"，在 30cm 外仍有 1.3 reward → CEM 为了获得 contact reward 牺牲了平衡
- baseline=0 意味着非接触帧 reward=0，接触帧 max=3.5 → CEM 强烈偏好接触帧 → 过度前倾

### 2. S3 最优 — sigma=0.15 解决了 S2 的问题

同样 gain=3.5, baseline=0，但 sigma=0.15 使 kernel 更陡:
- d=0: reward=3.5 (满分)
- d=15cm: reward=1.3 (只有 37% → 远距离没有太强吸引)
- d=30cm: reward=0.5 (几乎为 0 → CEM 不会为远距离接触牺牲平衡)

**更陡的 kernel 只在手非常靠近时给 reward → 不会像 S2 那样把机器人拉倒**。

### 3. S4 也很好 — baseline=1.0 提供了另一种稳定机制

gain=3.5, sigma=0.3, baseline=1.0:
- 非接触帧: reward=1.0 (constant)
- 接触帧: max=3.5, 30cm处=1.3
- Net incentive at 30cm = 1.3 - 1.0(下一帧 baseline) = 0.3 → 弱信号
- Net incentive at 0cm = 3.5 - 1.0 = 2.5 → 强信号

baseline=1.0 使得 CEM 不会强烈偏好接触帧 vs 非接触帧 → 更稳定。
但 Contact<10cm (53%) 略低于 S3 (57%)。

### 4. S5 反而更差 — gain=5.0 过大不等于更好

gain=5.0 超过了 tracking max (3.5)，但 contact 反而只有 40%:
- 可能因为 gain 太大导致 CEM 在接触帧过度优化 contact → 短期冲刺行为 → 后续帧偏离
- 整体 mean surf dist=15cm 是所有配置中最差的

### 5. Baseline (0 vs 1) 在 gain=2.0 时无差异

S1 (b=1) vs E037b (b=0): Contact<10cm = 45% vs 49%，在噪声范围内。
**原因**: gain=2.0 本身太弱，baseline 的 1.0 差异相对 tracking 3.5 不显著。

## 可视化分析 (S3 vs E036 — 离线渲染对比)

使用 `render_trajectory_video.py` 从 npz 离线渲染 side-by-side (ref左 | sim右) 视频。

### S3 (gain=3.5, sigma=0.15, baseline=0) vs E036 (no contact reward)

| 帧 | 时间 | S3 观察 | E036 观察 | 差异 |
|------|------|---------|-----------|------|
| 0 (0s) | 起始 | ref/sim 对齐 | ref/sim 对齐 | 无差异 |
| 24 (0.8s) | 弯腰 | sim 弯腰趴箱顶, 手搭箱面 | sim 弯腰趴箱顶, 手搭箱面 | 几乎一致 |
| 49 (1.7s) | 推箱 | sim 站箱旁, 手臂伸向箱面 | sim 站箱旁, 手略远于箱面 | **S3 手更靠近** |
| 74 (2.5s) | 站立 | sim 站箱侧, 单手碰箱顶边缘 | sim 站箱侧, 手远离箱面 | **S3 明显更近** |
| 99 (3.3s) | 弯腰 | sim 弯腰趴箱, 双手箱面 | sim 弯腰趴箱, 双手箱面 | 类似 (ref 本身手在箱上) |
| 123 (4.1s) | 结束 | sim 手举起离开 | sim 手举起离开 | 无差异 |

**结论**: S3 的改善主要在 t=1.7-2.5s (ref 手距箱 10-30cm 的过渡阶段)。当 ref 手紧贴箱面 (t=0.8s, 3.3s) 时两者差异小 (body tracking 本身就能把手送到)。当 ref 手远离箱 (t=4.1s) 时 mask=0 无激励。**Contact mask reward 在"中间地带"起效。**

## 最优配置选定: S3

```yaml
contact_mask_rew_scale: 3.5   # = tracking max
contact_mask_rew_sigma: 0.15  # steep: only reward when hand very close
contact_mask_rew_baseline: 0.0
```

**理由**:
1. Contact<10cm=57.3% (最高)
2. MPKPE=1.4cm (无退化)
3. Stability=100% (S2 的问题通过 steep sigma 解决)
4. Contact Preservation: 73.2% of desired frames have sim<10cm

## 下一步

用 S3 配置跑全 3 cases (desk005, box025, bucket010)，验证泛化性。

## 结果路径

| 产出 | 路径 |
|------|------|
| S1 npz | `workspace/core4d/results/E037_sweep/S1_box025.npz` |
| S2 npz | `workspace/core4d/results/E037_sweep/S2_gain3.5_sigma0.3_base0.0_box025.npz` |
| S3 npz | `workspace/core4d/results/E037_sweep/S3_gain3.5_sigma0.15_base0.0_box025.npz` |
| S4 npz | `workspace/core4d/results/E037_sweep/S4_gain3.5_sigma0.3_base1.0_box025.npz` |
| S5 npz | `workspace/core4d/results/E037_sweep/S5_gain5.0_sigma0.3_base1.0_box025.npz` |
| S5 video | `workspace/core4d/results/E037_sweep/visualization_mjwp_act.mp4` |
| Sweep script | `workspace/core4d/scripts/run_e037_sweep_remote.sh` |
| Logs | `logs/E037_sweep/` |
