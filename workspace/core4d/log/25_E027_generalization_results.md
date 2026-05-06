# E027: Partner Force Sweep + Multi-Case Generalization — 结果

## 状态: 突破确认 (跨 case 泛化 + 低 partner force 仍有效)

## 核心发现

1. **Hand approach reward 在不同 partner force 水平下都产生接触** — 64% contact rate 不依赖 partner force 强度
2. **desk005 表现优于 bucket010**: lift=11.7cm, 80% contact, 8 consecutive contact frames
3. **50% partner force (物体 1.5kg 有效重量) 仍然有效** — approach reward 真正在驱动接触

## Partner Force Sweep (bucket010, 4096 samples, 24 iter)

| Run | pf | effective_weight | stable% | contact% | lift_max | consecutive>5cm |
|-----|-----|-----------------|---------|----------|---------|----------------|
| E026-b | 0.85 | 0.45kg | 100% | 64% | 0.093m | 4 |
| E027-a | 0.70 | 0.90kg | 100% | 64% | 0.091m | 2 |
| E027-b | 0.50 | 1.50kg | 100% | 64% | 0.089m | 1 |

**关键洞察**: contact% = 64% 不随 pf 变化 → approach reward 驱动接触, pf 只影响 lift 持续性
- 降低 pf: lift_max 几乎不变 (~9cm), 但 sustained >5cm 减少 (4→2→1)
- 原因: 更重的物体需要更大力才能维持 altitude, CEM 能产生瞬间力但难以持续

## Multi-Case 泛化 (desk005)

| Run | case | samples | pf | contact% | lift_max | consecutive>5cm |
|-----|------|---------|-----|----------|---------|----------------|
| E025-g | desk005 | 2048 | 0.85 | 40% | 0.057m | 2 |
| **E027-c** | **desk005** | **4096** | **0.85** | **80%** | **0.117m** | **2** |

**desk005 大幅提升**: 4096 samples 使 contact% 翻倍 (40%→80%), lift 翻倍 (5.7cm→11.7cm)

### E027-c 时间线 (desk005, best overall)
```
t=0: hand=0.230m (远离)
t=1: hand=0.080m (接近中)
t=2: hand=0.040m **CONTACT** → lift +2.0cm
t=3: hand=0.004m **CONTACT** → lift +8.2cm **LIFT**
t=4: hand=0.012m **CONTACT** → lift +11.7cm **PEAK**
t=5: hand=0.000m **CONTACT** → lift +2.2cm (下降)
t=6-9: hand=0.000m **SUSTAINED CONTACT** → lift 2.8-4.7cm
```

## Phase 6 整体总结 (E025-E027)

| 实验 | 关键贡献 |
|------|---------|
| E025 | 证明 hand approach reward 能突破 "CEM 不产生接触" 限制 |
| E026 | 4096 samples + 24 iter 使 lift 持续 4 steps; 长 horizon 反而保守 |
| E027 | 接触率不依赖 partner force; desk005 11.7cm lift 是本项目最佳物理交互 |

## Phase 6 总结论

**SPIDER CEM + hand approach reward + partner force = 可以做 contact-aware retargeting**

条件:
1. `hand_approach_rew_scale=5.0, sigma=5.0` — 必须
2. `anchored trajectory` (去除行走) — 必须
3. `partner_force_scale≥0.50` — 帮助持续性, 非必须
4. `num_samples=4096, max_num_iterations=24` — 帮助质量

局限:
1. Lift 非沿 ref 轨迹 (推向不一定正确的方向)
2. 不是抓取 (grasping) — 是推/碰/压
3. box025 仍不可解 (臂展限制, 非算法限制)
4. 需要 partner force 来减轻物体重量以实现持续 lift

## 结果路径

| 产出 | 路径 |
|------|------|
| E027-a (bucket010, pf=0.70) | `workspace/core4d/results/E026_sustained_contact/bucket010/e027a.{npz,mp4}` |
| E027-b (bucket010, pf=0.50) | `workspace/core4d/results/E026_sustained_contact/bucket010/e027b.{npz,mp4}` |
| E027-c (desk005, best) | `workspace/core4d/results/E026_sustained_contact/desk005/e027c.{npz,mp4}` |
