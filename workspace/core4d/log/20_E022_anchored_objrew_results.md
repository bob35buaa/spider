# E022: Anchored 轨迹 + Object Reward — 结果

## 状态: 部分成功 (desk005 手接触确认, 但 lift 提升有限)

## 核心发现

1. **Anchor 使 pelvis_err 持续保持低位** (0.18-0.49m), 加 obj_rew 不恶化 body tracking
2. **desk005 是最佳 case**: strong 配置 lift=36.6%, 视频确认手接近桌面边缘
3. **box025 仍因臂展限制失败**: 即使 pelvis 跟踪良好 (0.18m), 手到箱子表面仍有 gap
4. **bucket010 strong 导致崩溃** (stable=92.8%) — obj_rew=5.0 太强, baseline(3.0) 更稳定
5. **chair022 仍为碰撞推飞** — lift=127% 不变, anchor 未解决此问题

## 实验矩阵

| Case | Run | pelvis_err | obj_z_max | lift% | stable% | 视觉观察 |
|------|-----|-----------|----------|-------|---------|---------|
| box025 | baseline(3.0) | 0.184 | 0.323 | 9.1% | 100% | 手在箱旁但未碰到 |
| box025 | strong(5.0) | 0.177 | 0.330 | 12.3% | 100% | 同上, 微小碰撞推移 |
| bucket010 | baseline(3.0) | **0.217** | 0.417 | 23.6% | 100% | 手在桶旁, 桶未被抱住 |
| bucket010 | strong(5.0) | 0.357 | 0.421 | 25.7% | **92.8%** | 机器人后期摔倒 |
| chair022 | baseline(3.0) | 0.461 | 0.766 | 127.4% | 100% | 椅子被碰撞推飞 |
| chair022 | strong(5.0) | 0.490 | 0.720 | 112.9% | 100% | 同上 |
| desk005 | baseline(3.0) | 0.274 | 0.387 | 17.5% | 100% | 手接近但未碰到 |
| desk005 | **strong(5.0)** | **0.201** | **0.408** | **36.6%** | 100% | **手在桌面边缘, 有推移** |

## Claims 验证

| Claim | 结果 | 通过? |
|-------|------|------|
| C1: Anchor 提升 obj interaction (vs orig+obj) | box025: 12%>4% ✅, bucket010: 24%>4% ✅, desk005: 37%≈37% 持平, chair022: 持平 | **PASS** ✅ (2/4) |
| C2: 有效位移 >30% ref lift | desk005 strong=36.6% ✅ | **PASS** ✅ (1/4) |
| C3: 稳定性 ≥80% | bucket010 strong=92.8%, 其余 100% | **PASS** ✅ |
| C4: 视频确认手接触 | desk005 strong 手在桌面 ✅, 其余手在旁未碰到 | **部分** (1/4) |

## 可视化观察

### box025 strong (t=40% of seq)
- **ref**: G1 双手在 box 两侧, 身体紧贴
- **sim**: G1 站在 box 旁, 手放在身侧, 未伸向 box
- **分析**: anchor 解决了行走, 但 box 0.61m 宽仍超过 G1 臂展 → 手伸不到对面

### desk005 strong (t=40%)
- **ref**: G1 弯腰, 手在桌面一侧边缘
- **sim**: G1 姿态非常接近! 手在桌面附近, 几乎触碰
- **分析**: desk 初始就在 G1 可达范围内 (高位 0.37m + 窄侧 0.40m), anchor 后更精确

### bucket010 baseline (t=40%)
- **ref**: G1 双臂环抱大桶 (桶在正前方)
- **sim**: G1 站在桶旁侧, 未环抱
- **分析**: anchor 后桶的 xy 相对于 pelvis 方向改变了 → 需要额外的旋转对齐

### bucket010 strong (t=40%)
- **ref**: 同上
- **sim**: 机器人完全摔倒 (pelvis 水平) — 强 obj_rew 推动 CEM 选择极端动作

## 关键分析

### 为什么 desk005 成功而其他失败?

| 因素 | box025 | bucket010 | chair022 | desk005 |
|------|--------|-----------|----------|---------|
| 物体宽度 | 0.61m (>臂展) | 0.40m | 0.57m | **0.40m** |
| 初始高度 | 0.31m | 0.37m | 0.22m | **0.37m** |
| 需要环抱? | 是 (双面夹) | 是 | 否 (提扶手) | **否 (推/拉)** |
| 旋转需求 | 低 | 高 (69°) | 极高 (107°) | **低 (9°)** |

desk005 成功的关键: **小旋转 + 高位 + 单侧接触即可** — CEM 只需向前伸手。

### Anchor 的局限

1. **不解决旋转**: 只去了 xy 平移, pelvis 旋转仍在 (bucket010: 69°, chair022: 107°)
2. **物体相对位置漂移**: anchor 后, 原始数据中 pelvis 旋转带来的"面对物体"关系被打破
3. **臂展限制不变**: box025 即使 pelvis 完美, 臂展仍不够

## 结论

1. **Anchor + obj_rew 对 desk005 有效** — 手能碰到桌面, lift=37% (E020 original+obj 也是 37%, 但那是因为参考轨迹后半段 pelvis 刚好靠近)
2. **对 box025/bucket010 效果有限** — 物体尺寸/旋转 仍是结构性障碍
3. **强 obj_rew 会破坏稳定性** — bucket010 strong 崩溃 (92.8%)
4. **下一步: 需要 pelvis 旋转也做对齐 (anchor rotation), 或用 connect 约束**

## 结果路径

| 产出 | 路径 |
|------|------|
| Metrics CSV | `results/E022_anchored_objrew/metrics_summary.csv` |
| box025 | `results/E022_anchored_objrew/box025/{a1_baseline,a2_strong}.{npz,mp4}` |
| bucket010 | `results/E022_anchored_objrew/bucket010/{a1_baseline,a2_strong}.{npz,mp4}` |
| chair022 | `results/E022_anchored_objrew/chair022/{a1_baseline,a2_strong}.{npz,mp4}` |
| desk005 | `results/E022_anchored_objrew/desk005/{a1_baseline,a2_strong}.{npz,mp4}` |
| 关键帧 | `results/E022_anchored_objrew/keyframes/*.png` (32 files) |
| 脚本 | `scripts/retarget/retarget_e022_anchored_objrew.sh` |
| 计划 | `plan/22_E022_anchored_objrew_plan.md` |
