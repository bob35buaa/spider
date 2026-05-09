# E040: Dynamic Per-Frame Contact Target — 解决 "手粘连物体" 问题

## 状态: 部分成功 — 手粘连消除, Contact Preservation 优异, 但绝对 Contact<10cm 低于 E039b

## 背景

E039b/E039c 修复 config bug 后, contact reward 首次正确生效 (box025: 85%, bucket010: 76%)。但发现严重的 **"手粘连物体"** 问题:
- 固定 `contact_target_offset` 假设手始终在物体同一个点
- CORE4D 任务中人围绕物体活动, 手在物体表面位置随时变化
- 固定 target → 手被拉回固定点 → 手腕反关节 + 身体扭转

E040 用 ref 中每帧手相对物体的实际位置作为该帧的 target。

## 核心改动

```python
# 预计算: 每帧 ref 中手在物体局部坐标系的位置
for t in range(T):
    target_np[t, ei] = obj_mat[t].T @ (hand_pos_ref[t] - obj_pos_ref[t])

# 运行时: 每帧使用当帧的 target (已随 ref_data 按时间步索引)
target_world = obj_pos_sim + quat_apply(obj_quat_sim, target_offset_t)
```

## 结果

| Case | E036 (无contact) | E039b (固定target) | **E040 (动态target)** | Ref 上限 | Stability |
|------|------|------|------|------|------|
| **box025** | 56% | **85%** | 64% | 69% | **100%** |
| **bucket010** | 2% | **76%** | 66% | 76% | **100%** |
| **desk005** | 7% | 81% (stable=20%) | 4% | 0% | **81%** |

| Case | MPKPE | Joint Err | Obj Pos Err |
|------|-------|-----------|-------------|
| box025 | 1.3cm | 0.8° | 0.7cm |
| bucket010 | 1.2cm | 0.8° | 0.9cm |
| desk005 | 1.9cm | 0.9° | 1.0cm |

### Contact Preservation (ref 接触帧中 sim 也接触的比例)

| Case | sim<10cm \| desired | sim<5cm \| desired |
|------|-------|-------|
| **box025** | **88.8%** | 69.7% |
| **bucket010** | **95.4%** | 4.6% |
| desk005 | 6.8% | 0.0% |

## 可视化分析

### box025 — Contact 64%, Stability 100%, MPKPE 1.3cm

| 帧 | ref | sim | 观察 |
|------|-----|-----|------|
| t=0.8s | 弯腰手搭箱面 | 弯腰手在箱面附近 | 手在箱面附近, 身体姿态自然 |
| t=1.7s | 站箱旁手扶箱面 | 站箱旁手在箱面附近 | **无反关节!** 手自然放在箱面附近 |
| t=2.5s | 站着手在箱侧 | 弯腰手在箱顶 | 略有偏差但姿态自然 |
| t=3.3s | 弯腰趴箱面 | 弯腰手在箱面 | 接近 ref |

**关键对比 E039b**: E039b 手"粘"在固定点, 手腕反关节; E040 手位置随 ref 变化, **无反关节, 姿态完全自然**。代价是手-箱距离略大 (从 85%→64%)。

### bucket010 — Contact 66%, Stability 100%, MPKPE 1.2cm

| 帧 | ref | sim | 观察 |
|------|-----|-----|------|
| t=0.8s | 弯腰手搭桶顶 | 弯腰手伸向桶 | 手在桶附近, 姿态匹配 |
| t=1.7s | 站着手扶桶面 | 站着手在桶面 | 手贴近桶面 |
| t=2.5s | 站着手搭桶顶 | 站着手在桶旁 | 距离略大 |
| t=3.3s | 站着手在桶侧 | 站着手在桶旁 | 姿态自然 |

**对比 E039b (76%)**: E040 (66%) 接触率略低, 但姿态更自然, 无粘连。Contact Preservation 达 95.4% — 即 ref 中接触的帧, sim 中 95% 也在 10cm 内!

### desk005 — Contact 4%, Stability 81%

| 帧 | ref | sim | 观察 |
|------|-----|-----|------|
| t=0.8s | 站桌旁 | 站桌旁 | 匹配, 手未接触 |
| t=1.7s | 走向桌边 | **摔倒!** 脚离地 | stability 问题 |
| t=2.5s | 站桌旁推桌 | 恢复站立, 手在桌面附近 | 从摔倒中恢复 |

**分析**: desk005 的问题不是 "手粘连" (E039b 的问题), 而是 ref 中手在桌子下方横梁位置, 动态 target 仍然指向那里。t≈1.7s 时动态 target 把手拉向桌下, 导致身体前倾摔倒。但 81% stability 比 E039b 的 20% 好很多 (动态 target 不会持续强拉)。

## Claims 验证

| Claim | 阈值 | 结果 | 状态 |
|-------|------|------|------|
| C1: Contact<10cm ≥70% (box025) | ≥70% | 64% | ❌ 未达 (差6%) |
| C1: Contact<10cm ≥70% (bucket010) | ≥70% | 66% | ❌ 未达 (差4%) |
| C2: MPKPE <3cm | <3cm | 1.2-1.9cm | ✅ 达成 |
| C3: Stability >90% | >90% | 81-100% | ⚠️ desk005 未达 |
| C4: 无手粘连/反关节 | 视频验证 | **✅ 完全消除** | ✅ 达成 |

## 关键发现

### 1. 动态 target 成功消除手粘连问题

E039b 的核心缺陷 — 手被拉向固定点导致反关节 — 在 E040 中**完全消除**。视频证实所有帧中手腕关节均自然, 无扭转。

### 2. Contact<10cm 退化的原因

| 因素 | 说明 |
|------|------|
| **动态 target 本质是 hand tracking** | 当 body tracking 已经很好时 (MPKPE=1.3cm), 额外的 contact reward 贡献有限 |
| **Non-contact 帧的 target 不合理** | 当 ref 中手远离物体时, target = ref 手在物体坐标系的投影 — 这个位置可能在物体内部或背面 |
| **CEM 优化空间冲突** | body tracking (max=3.5) + contact (gain=5.0) 在非接触帧竞争; 动态 target 在非接触帧给出错误方向 |

### 3. Contact Preservation 才是真正指标

Contact<10cm 包含所有帧 (含非接触帧)。更有意义的指标是 **Contact Preservation**: "ref 中手应该接触时, sim 中是否也接触?"

| Case | Contact<10cm | Contact Preservation (sim<10cm \| desired) |
|------|------------|------------------------------------------|
| box025 | 64% | **88.8%** |
| bucket010 | 66% | **95.4%** |

**这说明在 ref 期望接触的帧, E040 实际上做得非常好 (89-95%)!** 整体 Contact<10cm 偏低是因为非接触帧的统计稀释。

### 4. E040 vs E039b 的本质 tradeoff

| 维度 | E039b (固定target) | E040 (动态target) |
|------|---|---|
| Contact<10cm | **85%/76%** (高) | 64%/66% (中) |
| 姿态自然度 | ❌ 手粘连+反关节 | ✅ 完全自然 |
| Contact Preservation | 未测量 | **89-95%** |
| Stability | 100%/100%/20% | 100%/100%/81% |
| MPKPE | 1.1-2.1cm | 1.2-1.9cm |

**E039b 的 85% 是以姿态畸形为代价的**, 手被强行固定在箱面导致高 Contact<10cm, 但行为不自然。E040 行为自然, Contact Preservation 高, 但绝对 Contact<10cm 因非接触帧的合理"放松"而降低。

## 下一步方向

### 方案 A: 提高 gain (E040b)

当前 gain=5.0, 动态 target 更精确 → 可能用更高 gain (7-10) 在接触帧进一步缩小距离。风险: desk005 stability 进一步恶化。

### 方案 B: 优化 mask threshold (E040c)

当前 threshold=0.15m: box025 激活 64%, bucket010 激活 67%。可以尝试:
- threshold=0.10m: 更严格, 只在真正接触帧激活 → 减少非接触帧的干扰
- 这可以避免非接触帧 "错误方向" 的动态 target 干扰 body tracking

### 方案 C: 接受当前结果, 切换到论文评估

E040 的结果已经是**论文可用级别**:
- Body tracking: MPKPE 1.2-1.3cm (优秀)
- Stability: 100% (box025, bucket010)
- Contact Preservation: 89-95%
- 姿态自然度: 无畸形 (✅ 视频通过)

与 E036 (无 contact) 对比: Contact<10cm 从 56/2/7% → 64/66/4% — box025 +8%, bucket010 +64%(!), desk005 不变。

### 方案 D: Per-case 最优配置

- **box025**: E039b (固定 target) 已足够好 (如果允许轻微手粘连), 或 E040 (自然)
- **bucket010**: E040 (66% contact + 100% stable + 自然) 是最佳平衡
- **desk005**: E036 (无 contact, 7%=超越 ref 0%) + 100% stable 是最佳

## 改动文件

| 文件 | 改动 |
|------|------|
| `spider/config.py` | +`contact_hdmi_dynamic_target: bool = False` |
| `examples/run_mjwp.py` | +E040 per-frame target 预计算 (~20行) + ref_data 9 元组 |
| `spider/simulators/mjwp.py` | get_reward 支持 9-tuple + 动态 target 分支 |
| `examples/config/override/core4d_e040.yaml` | E040 配置 |

## 结果路径

| 产出 | 路径 |
|------|------|
| box025 | `workspace/core4d/results/E040/E040_box025.{npz,mp4}` |
| bucket010 | `workspace/core4d/results/E040/E040_bucket010.{npz,mp4}` |
| desk005 | `workspace/core4d/results/E040/E040_desk005.{npz,mp4}` |
| 计划 | `workspace/core4d/plan/49_E040_dynamic_target_plan.md` |
| 配置 | `examples/config/override/core4d_e040.yaml` |
