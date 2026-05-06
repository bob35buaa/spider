# E026: Sustained Contact Optimization — 结果

## 状态: 成功 (E026-b: 4 consecutive >5cm lift, bucket upright)

## 核心发现

1. **4096 samples + 24 fixed iterations (1.6s horizon)** = 最佳配置
2. **4 consecutive MPC steps (≈1.9s) 持续 >5cm lift** — E025 最好只有 1 step
3. **Bucket 保持直立** — E025-f bucket 被推倒, E026-b 保持 upright
4. **长 horizon (2.4s) 导致过于保守** — 接触但不推 (contact=45% 但 lift=0)

## 实验矩阵

| Run | samples | horizon | maxiter | imp_thresh | pelvis_min | contact% | lift_max | sustained>5cm | 质量 |
|-----|---------|---------|---------|------------|-----------|---------|---------|--------------|------|
| E025-f (baseline) | 2048 | 1.6s | ~10avg | 0.01 | 0.691 | 55% | 0.095m | 1 step | 翻倒 |
| E026-a | 4096 | **2.4s** | 24 | 0.0 | 0.727 | 45% | 0.000m | 0 | 只碰不推 |
| **E026-b** | **4096** | **1.6s** | **24** | **0.0** | **0.725** | **64%** | **0.093m** | **4 steps** | **直立** |

## E026-b 时间线 (最佳)

```
t= 0: hand=0.347m (远离)
t= 1: hand=0.231m (接近中)
t= 2: hand=0.000m **CONTACT** → obj +1.1cm (开始推)
t= 3: hand=0.078m (近距离) → obj +4.0cm (蓄力)
t= 4: hand=0.000m **CONTACT** → obj +6.8cm **LIFT** (突破 5cm)
t= 5: hand=0.000m **CONTACT** → obj +9.3cm **PEAK LIFT**
t= 6: hand=0.025m **CONTACT** → obj +9.3cm **SUSTAINED**
t= 7: hand=0.000m **CONTACT** → obj +7.9cm **LIFT**
t= 8: hand=0.000m **CONTACT** → obj +1.7cm (下降)
t= 9: hand=0.015m **CONTACT** → obj +1.5cm
t=10: hand=0.164m (失去接触)
```

**关键特征**: 接触→蓄力→lift→sustained→下降 的完整物理搬运序列

## Claims 验证

| Claim | 结果 | 通过? |
|-------|------|------|
| C1: sustained >5cm lift ≥3 consecutive steps | 4 consecutive steps (t=4-7) | **PASS** ✅ |
| C2: 视频确认手接触+bucket 直立 | frame_03: 机器人弯腰, 手在 bucket 两侧, bucket 直立 | **PASS** ✅ |
| C3: lift 与 contact 时间相关 | contact(t=2) → lift(t=4-7), release(t=10) → drop | **PASS** ✅ |
| C4: 稳定性 | pelvis ≥ 0.50m 100% | **PASS** ✅ |

## 关键洞察: 为什么长 horizon 反而更差?

```
E026-a (horizon=2.4s):
  CEM sees: "如果我现在推 bucket, 1.5s 后 bucket 会飞走, 我的 body tracking 会变差"
  → 选择: 碰触 bucket 但不用力 → 获得 approach reward, 不损失 body reward
  → 结果: 100% contact, 0% lift

E026-b (horizon=1.6s):
  CEM sees: "在接下来 1.6s 内, 推 bucket 能获得 obj_rew + approach_rew"
  → 短视到看不见"推完后的后果" → 更激进
  → 结果: 64% contact, 9.3cm lift
```

**这与 SBMPC 文献一致**: MPC 的短视 (receding horizon) 既是限制也是优势。对接触任务, 短 horizon 产生更积极的力, 长 horizon 过于保守。

## 视频关键帧 (E026-b)

### frame_02 (t=45%): First contact phase
- ref: 双手抱住 bucket 上方
- sim: 右手伸向 bucket 侧面, bucket 轻微偏移

### frame_03 (t=64%): Peak lift phase
- ref: 搬运中
- sim: **机器人弯腰, 手在 bucket 两侧, bucket 直立** ← 最佳时刻

### frame_04 (t=91%): Release phase
- ref: 放下动作
- sim: 机器人站在 bucket 后方, 手在 bucket 顶部, bucket 直立

## 后续: E027 方向

E025-E026 证明了 hand approach reward 在 bucket010 上有效。关键限制:
1. 仍需 85% partner force (物体实际只有 15% 重量)
2. Lift 不跟踪 ref 轨迹 (只是被推起, 非沿 ref 搬运)
3. 泛化到 desk005 有效但 lift 更小 (5.7cm)

**E027 优先方向**:
1. **降低 partner force 渐进**: 85% → 70% → 50% → 验证最小必需支撑
2. **desk005 优化**: 更强 approach + 调整 sigma (desk005 手-物体距离更大)
3. **添加 contact_rew**: approach 引导手到位后, contact_rew 维持接触
4. **导出 hybrid 轨迹**: sim_robot + ref_object → Holosoma RL 训练初始化

## 结果路径

| 产出 | 路径 |
|------|------|
| E026-a (long horizon) | `workspace/core4d/results/E026_sustained_contact/bucket010/e026a.{npz,mp4}` |
| **E026-b (best)** | `workspace/core4d/results/E026_sustained_contact/bucket010/e026b.{npz,mp4}` |
| 关键帧 | `workspace/core4d/results/E026_sustained_contact/bucket010/frames_e026b/` |
| 配置 | `examples/config/override/core4d_bucket010_e026.yaml` |
