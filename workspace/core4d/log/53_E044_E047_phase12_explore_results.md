# E044-E047: Phase 12 广泛探索 — 初步结果

## 状态: 进行中 — E045 box025 结果已有，其余运行中

## Direction 1: Body Tracking 改进

### E045: Sigma Sweep (box025 初步结果)

| 实验 | pos_sigma | MPKPE | Contact<10cm | Preservation | Stability>0.6 | 对比 E041c |
|------|-----------|-------|-------------|-------------|--------------|-----------|
| E041c (baseline) | 0.5 | 1.4cm | **66%** | 88.2% | **100%** | — |
| E045a | 0.3 | 1.6cm | 38% | 47.9% | 88% | ❌ 全面退化 |
| E045b | 0.15 | 1.4cm | 46% | 64.0% | 100% | ❌ contact 下降 |

**分析**: 收紧 sigma 不仅没有改善 contact，反而显著恶化：
- E045a (σ=0.3): stability 从 100% 降到 88%，contact 从 66% 降到 38%
- E045b (σ=0.15): stability 保持 100%，但 contact 仍从 66% 降到 46%

**原因分析**: 
- 更紧的 sigma 使 body tracking reward 梯度更陡 → CEM 更保守地优化 body pose
- 这导致手部更严格跟踪 ref 的 body 位姿，但 ref 中手本身不在物体表面
- contact reward (gain=5.0) 相对于更陡的 tracking reward 变得更弱 → CEM 选择 tracking 而非 contact
- 这证实了 **contact 质量的瓶颈不在于 tracking 精度，而在于 ref 中手与物体的距离**

**结论**: 方向 1 的 sigma 收紧路线 **无效**，不继续。等待 E044 wrist weight 结果。

### E044: Wrist Weight (待远程运行)

E044a (weight=2.0) 和 E044b (weight=3.0) 代码和配置已完成，等 E045 跑完后部署远程。

---

## Direction 3: CEM/SBTO 算法优化

### E047a: SBTO 对齐 DynaRetarget (box025 初步结果)

| 指标 | E041c (MPC) | E047a (SBTO) | 变化 |
|------|------------|-------------|------|
| MPKPE | 1.4cm | **155cm** | ❌❌ 完全失败 |
| Contact<10cm | 66% | 0% | ❌ |
| Stability>0.6 | 100% | 31% | ❌ 机器人摔倒 |
| 运行时间 | ~14min | ~8min | ✅ 更快 |

**SBTO 运行日志**:
```
SBTO: knot  1/16, h=0.50s, iter=50, max_σ=0.0445, rew=3.194
SBTO: knot  5/16, h=1.50s, iter=50, max_σ=0.0458, rew=2.071
SBTO: knot 10/16, h=2.75s, iter=50, max_σ=0.0745, rew=1.354
SBTO: knot 15/16, h=4.00s, iter=50, max_σ=0.0223, rew=1.255
Total: 481.8s
```

**失败分析**:
1. **mean_momentum=0.95 太保守**: 每步仅更新 5% 的 mean → 50 轮迭代不足以收敛到好解
2. **所有 knot 都跑满 50 次迭代**: σ_min=0.01 太紧，noise 从未收缩到该阈值
3. **Reward 持续下降** (3.19→1.26): 随 horizon 增长，每步 mean reward 自然降低，但绝对质量也在恶化
4. **SBTO 不兼容当前 reward 结构**: DynaRetarget 用 joint-space cost (penalty)，SPIDER 用 exp-kernel reward → 优化 landscape 不同

**下一步**:
- 尝试 E047b: 降低 mean_momentum 到 0.5 或 0.0
- 尝试 E047c: 放宽 σ_min 到 0.05
- 考虑 SBTO + 不同 reward 权重

---

## Direction 2: 更多 Case (E046)

### 已完成的数据转换

| Case | 帧数 | 源 NPZ |
|------|------|--------|
| box025_person2 | 124 | V2-replace trimmed |
| bucket005_person2 | 159 | V2-replace trimmed |
| bucket010_person2 | 126 | V2-replace trimmed |

### 新物体候选 (已识别，待处理)

| 物体 | Session | 动作 | 帧数 |
|------|---------|------|------|
| Board020 | 20231008/010 | move2_obs0 | 157 |
| bucket001 | 20231030/094 | move2_obs0 | 154 |
| Box021 | 20231018/030 | move2_obs0 | 172 |
| Desk021 | 20231023/037 | move2_obs0 | 153 |
| desk007 | 20231030/028 | move2_obs0 | 149 |

这些需要在 holosoma 项目中运行完整的 OmniRetarget V2-replace pipeline。

---

## 结果路径

| 产出 | 路径 |
|------|------|
| E045a box025 | `workspace/core4d/results/E045/E045a_box025.{npz,mp4}` |
| E045b box025 | `workspace/core4d/results/E045/E045b_box025.{npz,mp4}` |
| E047a box025 | `workspace/core4d/results/E047/E047a_box025.{npz,mp4}` |
