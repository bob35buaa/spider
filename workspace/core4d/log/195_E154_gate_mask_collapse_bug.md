# E154 补充分析 — hand_support_gate mask 路径追踪 + release_false 根因

> 计划: `plan/162_E154_masked_tracking_eval_plan.md`（追加分析）
> 状态: **分析完成 — 代码无 bug，release_false 是 b1 reward 强度导致的行为惯性**
> 关联: E151/E152/E153 的 b1 reward

## 0. 一句话结论（修正）

~~初始假设：`_sample_gate_from_ref_mask` 把 mask 坍缩为 1.0 → gate 失效。~~

**Stage0 诊断推翻了此假设**：通过完整的代码链路追踪 + 数值模拟，确认：
1. mask 正确从 3cm npz 加载、resize 到 qpos_ref 长度（含放手段 mask=0 的尾部 111 帧）
2. `get_slice` 正确按时间切片，`ref[6] = approach_mask_t[S+t]` = 当前步 mask 值 `(2,)`
3. `_sample_gate_from_ref_mask` 收到 `(2,)` 输入 → `max(L,R)` → 放手段返回 0 ✅
4. **gate 在运行时正确关闭**（放手段 `hand_support_rew` = 0）

**release_false 高的真正原因**：b1 在搬运段给强力贴合奖励（3.0 × score），CEM 把手"压"在
物体表面。放手段 reward 突然归零时手已在物体上，body tracking 的 qpos_rew 力度不够快把手拉开
→ 放手段前段仍有物理接触 → release_false > 0。这是**行为惯性**，不是 code bug。

## 1. 影响路径全链

```
trajectory_kinematic.npz
  contact 字段 = all-1 (contact_detection_mode="one")
  → io.py:load_data() → contact_ref (全 1)
  → mjwp.py L843: contact_dist * contact_ref  [路径1]
  → 但 contact_rew_scale = 0.0 (E017后关闭)
  → ★ 路径1 对 E098-E153 无影响

contact_hdmi_mask_path (3cm npz)
  → run_mjwp.py L883-913: 加载 → approach_mask_t (T,2) per-hand
  → 插入 ref tuple 第7位 → approach_mask_val
  → mjwp.py L1343: support_gate("contact_mask",...) 调用 _sample_gate_from_ref_mask
  → L1219: ref_gate[0].max().expand_as(time_gate)  ← BUG在此
  → 第0帧有接触 → max=1.0 → gate=全1 → hand_support_rew 全程开启
```

## 2. 代码链路追踪结论（Stage0 诊断）

### 调用链

```
run_mjwp.py: approach_mask_t shape (T_padded, 2) ← resize from eval_contact_mask_3cm
  → ref_data tuple[6] = approach_mask_t
  → get_slice(ref_data, S, S+H) → ref_slice[6] shape (H, 2)
  → sampling.py L235: ref = [r[t] for r in ref_slice] → ref[6] shape (2,)
  → get_reward(config, env, ref) → approach_mask_val shape (2,) = 当前步的 per-eef mask
  → support_gate("contact_mask") → _sample_gate_from_ref_mask(mask=(2,), N=512)
  → L179-182: mask.ndim==1, shape[0]=2 != N → mask.max() → max(L,R)
  → 放手段 (0,0) → max=0 → gate=0 ✅
```

### 数值验证（box023_person2_legobj）

```
T_ref=136 (30Hz) → interp ×2 → T_interp=272 → +pad(4+1) → T_padded=277
eval_mask(227,2) resize → (277,2), contact window: [34, 165]
Release window: frames 166-276 (111 frames)
CEM runs 272 steps → release from step 166

Step 165: mask=(0,1) → gate=1 (搬运最后一步，R手仍接触)
Step 166: mask=(0,0) → gate=0 (放手开始)  ✅
Step 267: mask=(0,0) → gate=0                ✅
```

### 结论

**`_sample_gate_from_ref_mask` 没有 bug**。它收到的 input 已经是正确的当前步值 `(2,)`。
放手段确实返回 gate=0 → `hand_support_rew` 在放手段被正确关闭。

~~之前的"gate 坍缩为 1.0"假设是错误的~~——那个分析基于静态代码审阅，假设
`approach_mask_val` 是完整的 `(T,2)` tensor 而非经过 `get_slice + [t]` 索引后的 `(2,)` 值。
实际运行时，`sampling.py L235` 的 per-step indexing 已经把时间维度解析掉了。

## 3. 哪些实验受影响

| 实验 | 是否传入 3cm mask | gate 是否生效 | 影响程度 |
|------|------------------|-------------|---------|
| E143 (raw_mask) | ✅ 通过 contact_hdmi_mask_path | ❌ 被坍缩为 1.0 | hold_contact_rew_scale=0 → 路径2a不生效; b1未开 |
| E148 (rubber) | ✅ 继承 E143 | ❌ 同上 | 同 E143，无 hand_support_rew |
| E151 (b1) | ✅ 继承 E148→E143 | ❌ gate 坍缩 | **hand_support_rew 全程开启** |
| E152 (gateA+b1) | ✅ 继承 E151 | ❌ gate 坍缩 | **同上** |
| E153 (sweep) | ✅ 继承 E151 | ❌ gate 坍缩 | **同上** |

## 4. 路径 1 (`contact_rew`) 状态

- `contact_rew_scale` 默认 = 0.0（config.py L518）
- E098-E153 的 override 链中无任何 yaml 将其设为 >0
- **结论: 路径 1 不生效，全 1 mask 对近期实验无实际影响**

## 5. 路径 2a (`hold_contact_rew`) 状态

- `hold_contact_rew_scale` 默认 = 0.0（config.py L237）
- E143 显式设 `hold_contact_rew_scale: 0.0`
- E151/E152/E153 继承，未覆盖
- **结论: 路径 2a 不生效**

## 6. 路径 2b (`hand_support_rew` / b1) — 唯一生效路径

- `hand_support_rew_scale: 3.0`（E151 开启）
- `hand_support_gate_source: contact_mask`（读 approach_mask_val）
- 但 gate 被坍缩 → 等效于 `gate_source: always`
- **b1 全程奖励手贴近物体 = 放手段也在奖励接触 = release_false 高的根因**

## 7. release_false 高的真正根因 + 可能的改进方向

### 根因

b1 (`hand_support_rew` scale=3.0, sigma=0.015) 在搬运段给**非常强的贴合奖励**：
当手 SDF ≈ 0 时 score ≈ 1.0 → reward ≈ 3.0。CEM 会选择把手"压"在物体表面。

放手段开始后 gate 正确归零 → `hand_support_rew` = 0，但：
- 手已经在物体表面（搬运段末尾的最优解 = 手贴物体）
- 放手段的 body tracking (`qpos_rew`) 要求手跟随 reference 离开
- **但 qpos_rew 力度相比 b1 的 3.0 scale 弱**（通常 scale=1.0）
- 加上物理仿真的惯性 → 手需要若干帧才能离开物体 → 这些帧仍有物理接触

### 对比 baseline

baseline 只有 `contact_hdmi_rew`（gain=5.0），它在搬运段奖励手靠近 target 点（不直接奖励贴合），
手不会被"压"在物体表面，所以放手段过渡更自然 → release_false = 0。

### 改进方向（不是 fix bug，而是 reward 设计改进）

| 方案 | 描述 | 预期效果 |
|------|------|---------|
| A. 放手段加 release reward | mask=0 时正向奖励手远离物体（而非仅 neutral） | 主动拉手离开 |
| B. 降低 b1 scale / sigma | 减弱搬运段贴合力度 → 手不被压那么紧 | 放手更容易，但搬运段接触可能降 |
| C. mask 边界 ramp-down | 在搬运→放手过渡处用 0.5→0 渐变 mask | 平滑过渡，减少突变 |
| D. 接受现状 | release_false 作为诊断指标保留，不进门控 | E154 已确认：tracking 门控才是行为判据 |

**推荐 D**：E154 已经证明 tracking 门控（pz_term < 0.08）是真正有区分力的行为指标。
release_false 高但 tracking pass 的 run（如 E153 sdf010_v10, box021 release_false=0.75 但
pz_term=0.035）说明机器人起身正常但手滞留——这对 RL 下游不致命（RL 可以自己学放手）。

## 8. 对现有结论的修正

- **E152/E153 "b1 提升接触"结论有效** — gate 正确工作，搬运段接触确实被 b1 增强
- **release_false 高不是 bug**，是 b1 reward 强度的副作用（行为惯性）
- **E153 最优配置 `(-0.010, 0.10)` tracking 3/3 结论不受影响**
- **无需重训即可进入下游 RL** — tracking pass 的轨迹已经满足 Holosoma 输入要求

## 9. Baseline (E148) 的 mask 使用分析

E148 baseline 配置链: `E089A → E143 → E148`

| reward 通路 | scale | mask 使用 | 是否正确 |
|---|---|---|---|
| `contact_rew` (路径1) | 0.0 | — | 不生效 |
| `hold_contact_rew` (路径2a) | 0.0 | — | 不生效 |
| `hand_support_rew` (路径2b/b1) | 0.0 | — | 不生效（E151 才开启） |
| **`contact_hdmi_rew`** (E039b) | **5.0** | `approach_mask_val` | **✅ 正确** |

**`contact_hdmi_rew` 的 mask 处理是正确的**（mjwp.py L1108-1146）：
- mask shape `(horizon, 2)` → 走 L1131 分支 `mask[0].unsqueeze(0).expand_as(rew_stack)`
- `mask[0]` = horizon 切片的第一个步 = **当前仿真时间步**（因为 `get_slice(ref_data, t, t+horizon)` 已经按时间切好了）
- 公式: `mask=1 → gain*pos_rew` (奖励靠近 target), `mask=0 → 1.0` (不罚也不奖)
- 所以 baseline 在放手段确实不奖励接触 → 这就是为什么 baseline `release_false = 0`

**结论: baseline 的 mask 正确生效**。bug 只在 b1 的 `_sample_gate_from_ref_mask` 坍缩路径。

## 10. 3cm Mask 质量质检 (3 cases)

### 10.1 总览

| case | T | L active | R active | Either | Both | 窗口内gap |
|------|---|----------|----------|--------|------|-----------|
| box021_029_p2 | 75 | 51 (68%) | 55 (73%) | 55 (73%) | 51 (68%) | **0** |
| box004_083_p2 | 105 | 61 (58%) | 62 (59%) | 62 (59%) | 61 (58%) | **0** |
| box023_person2 | 136 | 61 (45%) | 65 (48%) | 65 (48%) | 61 (45%) | **0** |

**好消息: 三个 case 的接触窗口内部都没有 gap（断裂）**。

### 10.2 不对称 (L≠R) 分析

| case | 不对称帧数 | 位置 | 原因 |
|------|-----------|------|------|
| box021 | 4 帧 (49-52) | 窗口中段 | L手暂时远离(5.5~9.6cm)，R手仍接触(1~18mm) |
| box004 | 1 帧 (88) | 窗口末帧 | L手刚松(>3cm)，R手仍在(1.8cm) |
| box023 | 4 帧 (17-18, 80-81) | 首尾各2帧 | 接触渐进/渐退，L手比R手迟到/先走 |

### 10.3 box021 中段不对称深入分析

```
frame 48: L=0.0101m R=0.0018m  (L 快到 3cm 阈值)
frame 49: L=0.0551m R=0.0014m  ← L 突然跳到 5.5cm
frame 50: L=0.0877m R=0.0010m
frame 51: L=0.0963m R=0.0034m  ← L 最远 9.6cm
frame 52: L=0.0857m R=0.0183m
frame 53: L=0.0008m R=0.0010m  ← L 突然回到 <1mm
```

这是**动捕数据的瞬时跳变**（4帧 = 0.13s @30Hz），不是真实松手。
在 30Hz 下一个手从 1cm 跳到 9.6cm 再跳回 <1mm，物理上不可能 → **动捕质量问题**。

### 10.4 边界处的双手不对称

**box023 onset (frame 17-18)**:
```
frame 17: L=0.106m R=0.011m → R先到(渐进), L还在靠近
frame 18: L=0.062m R=0.001m → R完全贴合, L距离3cm
frame 19: L=0.028m R=0.001m → 双手均接触
```

**box023 offset (frame 80-81)**:
```
frame 79: L=0.008m R=0.001m → 双手贴合
frame 80: L=0.032m R=0.005m → L刚超 3cm, R仍近
frame 81: L=0.062m R=0.025m → 双手渐离
```

边界处的不对称是**真实的物理现象**——搬运时双手不会完全同时接触/松开。

### 10.5 mask 处理建议

| 问题 | 处理方案 | 理由 |
|------|---------|------|
| box021 中段 4 帧 L 跳变 | **用双手 max 补全** | 搬运任务中一手仍接触=两手都应接触；且该跳变是动捕 artifact |
| 边界 L≠R (box023 2帧) | **用双手 max 扩展** | 搬运渐进/渐退时，任一手到位即视为接触开始 |
| 无 gap (3 cases 均无) | 不需要填充 | mask 质量基本合格 |

**推荐后处理**: 对搬运任务（双手协作），生成 `contact_mask_carry` = `max(L, R)` per-frame 作为统一 gate，
替代 per-hand 独立 mask。这样：
1. 修复了中段动捕跳变（R=1 时 L 被补为 1）
2. 统一了边界定义（"首次有手到位" = 开始，"最后一只手离开" = 结束）
3. 对 `hand_support_rew` 而言语义正确（搬运期间**两手都应该**在物体表面）

## 11. E148 之后 contact_hdmi_rew 是否仍在使用

**是，一直在用。** 完整继承链：

```
E143: contact_hdmi_gain: 5.0, contact_hdmi_mask_path: ✓ (3cm npz)
  → E148: 继承 (只换 scene_act → rubber_hull)
    → E151 b1: 继承 (追加 hand_support_rew_scale: 3.0)
      → E152 gateA+b1: 继承 (追加 cem_hand_gate)
        → E153 sweep: 继承 (CLI 覆盖 gate 参数)
```

所以 **E148-E153 全部同时开着两个接触 reward**：

| Reward | Scale | Mask 处理 | 放手段行为 |
|--------|-------|-----------|-----------|
| `contact_hdmi_rew` | 5.0 | ✅ 正确 (per-step per-eef) | mask=0 → 给中性值 1.0（不奖励） |
| `hand_support_rew` (b1) | 3.0 | ❌ gate 坍缩为 1.0 | 全程奖励贴近（含放手段） |

### 两个 reward 的作用分工

- **`contact_hdmi_rew`** (E039b): 奖励手的 eef 点接近物体表面 target 位置（object local frame），
  mask=1 时 reward = `gain * exp(-dist/sigma)`，mask=0 时 reward = 1.0（中性 baseline）。
  **这是主接触驱动力**，且 mask 正确生效。

- **`hand_support_rew`** (b1): 奖励手几何体(rubber mesh)与物体 box SDF 接近零
  （贴在表面 ≈ SDF=0），用 `exp(-|sdf|/sigma)` 计分。
  **这是补充接触精修**（几何贴合度），本意是配合 mask gate 只在搬运段激活。
  但 gate 坍缩 → 全程激活 → 放手段也在拉手回去。

### release_false 根因完整解释

放手段时：
- `contact_hdmi_rew`: mask=0 → reward=1.0（CEM 不在乎手位置，优化其他 reward）
- `hand_support_rew`: gate 应为 0 但实为 1 → reward = 3.0 × score（**仍在激励贴合**）
- CEM 发现"贴着物体"能多拿 3.0 × score 的 reward → 选择不松手 → release_false 高

**修复 `_sample_gate_from_ref_mask` 后**:
- 放手段 `hand_support_rew` gate=0 → reward=0 → 只剩 `contact_hdmi_rew`=1.0
- CEM 无接触激励 → 自然松手（跟 baseline 行为一致）
- 预期 release_false 从 0.24-0.75 降至接近 0

## 12. 总结: reward 路径全景图

```
E148-E153 CEM 每步 reward 构成:
┌────────────────────────────────────────────────────────────┐
│ qpos_rew (body tracking)                      scale=1.0   │ 始终
│ + contact_hdmi_rew (E039b, 手→target)         scale=5.0   │ mask ✅ 正确
│ + hand_support_rew (b1, 手SDF→0)             scale=3.0   │ mask ❌ 坍缩 [E151+]
│ + cem_hand_gate (gateA, 过滤穿透 sample)       filter     │ [E152+]
│ + robot_object_penalty                        scale=var   │
│ + leg_object_penalty                          scale=var   │
│ + contact_rew (路径1)                         scale=0.0   │ 关闭
│ + hold_contact_rew (路径2a)                   scale=0.0   │ 关闭
└────────────────────────────────────────────────────────────┘
```

**Bug 只影响 `hand_support_rew` 的 gate 逻辑**。
`contact_hdmi_rew`（主接触驱动）的 mask 处理完全正确。

---

*记录日期: 2026-06-11*

