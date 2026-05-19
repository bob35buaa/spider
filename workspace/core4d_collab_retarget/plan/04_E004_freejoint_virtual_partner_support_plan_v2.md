# E004 Plan v2: true-freejoint virtual partner support

日期：2026-05-17 (revised)

## Context

E002 showed true-freejoint fails (obj_err 0.7-0.8m). E003 showed passive mass/friction tuning insufficient.

The code already has `partner_force_*` (Mode D) applying `xfrc_applied` to the object body while preserving freejoint physics (`nu=29`).

### 历史教训（必须回避的死路）

| 实验 | 方案 | 失败原因 |
|------|------|----------|
| E024 (core4d) | gravity comp 50-90% only | CEM 从不主动接近物体，手始终在体侧 |
| E028-E030 (core4d) | xfrc_applied rotation torque kp_rot=1-10 | **结构性 NaN**：MJWarp batch 中 3880/4096 environments 发散。position spring + orientation torque 正反馈环导致能量注入 |
| E009 (core4d) | gravity comp on box025 | 几何不匹配（G1 臂展 0.5m < 箱长 0.61m），即使无重力也无法搬运 |
| E025-E027 (core4d) | hand_approach + gravity comp | 手到达物体但只是推/撞静止物体，partner 横向动力学缺失 |

### 与 E028-E030 的关键区别

本轮有以下改进使结果可能不同：
1. E070/E071 ctrl mapping bug 已修复（yaw drift 12.4° → 0.6°）
2. E048 碰撞盒已修正
3. 使用 person2（更好的 ref 质量，E078 确认）
4. E081 leg-object 碰撞对已添加
5. **不使用 rotation torque**（回避 E028-E030 的结构性 NaN）

## Claims

| Claim | Evidence |
|-------|----------|
| C1 virtual partner translational spring enables cooperative carrying | box025_p2 obj_mean < 0.30m, pelvis_min > 0.50m |
| C2 remains physically meaningful | `contact_guidance=false`, `nu=29`, `nq_obj=7`, no object actuators |
| C3 guard stable | box023_p2 leg_intf ≤ 5%, no fallover |
| C4 robot actively participates (not just spring carrying) | hand_contact > 50% in contact-reward variants |

## 力量标定

5kg 物体，重力 49N。人类搬运时单手贡献约 25-50N。

| kp | 典型位移 0.3m | 典型位移 0.7m | 对应人力 |
|----|--------------|--------------|----------|
| 10 | 3N (6% gravity) | 7N (14%) | 轻触 |
| 20 | 6N (12%) | 14N (28%) | 中等支撑 |
| 40 | 12N (24%) | 28N (57%) | 强支撑 |

gravity_comp=0.5 提供 24.5N 垂直力。kp=20 + gravity=0.5 在 0.5m 位移下总力 ~34.5N，接近人类单手贡献。

## Variants

**原则：不使用 rotation torque（kp_rot=0），分离 position tracking 和 orientation 问题。**

| Variant | gravity | spring_kp | kp_rot | contact_rew | Role | 目的 |
|---------|---------|-----------|--------|-------------|------|------|
| `E004_box025_p2_s20` | 0.5 | 20 | 0 | off | main | 中等弹簧，测试位置跟踪 |
| `E004_box025_p2_s40` | 0.5 | 40 | 0 | off | main | 强弹簧，测试物理上限 |
| `E004_box025_p2_s20_cr` | 0.5 | 20 | 0 | on (E078 stack) | main | 弹簧 + contact reward，测试机器人是否主动参与 |
| `E004_box023_p2_s10` | 0.5 | 10 | 0 | off | guard | 弱弹簧 guard，稳定性验证 |

### 变体设计理由

1. **砍掉 gravity-only (g05)**：E024 已证明纯 gravity comp CEM 不接近物体
2. **砍掉 kp_rot**：E028-E030 证明 xfrc torque 在 MJWarp batch 结构性 NaN
3. **新增 s40 强弹簧**：如果 s20 位移过大物体仍 floor-supported，s40 提供更强恢复力
4. **新增 s20_cr (contact reward)**：E024 教训 — 弹簧移动物体但 CEM 可能忽略物体。contact reward 确保机器人主动参与。使用 E078 的 per-EEF contact mask reward stack（已验证有效）
5. **guard 用更低 kp=10**：box023 更小更轻，弹簧可以更弱

### Orientation 策略（如果位置跟踪成功但 orientation 不行）

不在 E004 中解决。如果 E004 位置跟踪成功但物体翻转：
- 后续实验用 **低摩擦 floor + 高摩擦 hand**（E003 思路）让物体不易翻
- 或用 **contact reward 引导手到特定面**（E062 palm_normal 思路）抵抗翻转
- 最后手段：尝试极低 kp_rot=0.5 + 现有 NaN guard，但需要单独验证

## Implementation

复用 E002 的 freejoint_legobj task（5kg, 标准碰撞），只生成 Hydra overrides。

### 新增 / 修改文件

- `workspace/core4d_collab_retarget/scripts/E004/variants.tsv`
- `workspace/core4d_collab_retarget/scripts/E004/generate_e004_overrides.py`
- `workspace/core4d_collab_retarget/scripts/train/train_E004.sh`
- `workspace/core4d_collab_retarget/scripts/eval/eval_E004.py`
- `workspace/core4d_collab_retarget/log/04_E004_freejoint_virtual_partner_support_results.md`

### Hydra override 要点

```yaml
# 共通
partner_force_scale: 0.5
partner_force_spring_kd: -1.0  # auto critical damping
partner_force_spring_kp_rot: 0.0  # 不使用旋转力矩
contact_guidance: false
scene_name: ""
# s20
partner_force_spring_kp: 20.0
# s40
partner_force_spring_kp: 40.0
# s20_cr (额外)
contact_mask_rew_scale: 1.0
contact_hdmi_rew_scale: 0.5
hand_approach_rew_scale: 0.0  # 不用 hand_approach（E036 教训）
```

### ref_dt 不一致问题

`_apply_partner_force` 用 `dt=1/30` 查找参考帧，但 `ref_dt` 实际 50Hz。对于本实验（验证性质），此问题可接受（误差 <1 frame）。如果 E004 成功，后续修复为 `dt=ref_dt`。

## Commands

```bash
bash workspace/core4d_collab_retarget/scripts/train/train_E004.sh smoke 0
bash workspace/core4d_collab_retarget/scripts/train/train_E004.sh full 0  # or remote
```

## Success Criteria

| Metric | Threshold |
|--------|-----------|
| box025_p2 case-window obj mean | < 0.30m useful, < 0.20m strong |
| box025_p2 pelvis_min | > 0.50m |
| box025_p2 floor contact | < E003 best 76.9% |
| box025_p2 hand contact (s20_cr) | > 50% |
| box023_p2 leg interference | ≤ 5% |
| model/config parity | `contact_guidance=false`, `nu=29`, `nq_obj=7` |

## Decision Rule

| E004 结果 | 下一步 |
|-----------|--------|
| s20 或 s40 obj_mean < 0.30m + stable | 成功 → position 弹簧可行。后续解决 orientation（低 kp_rot / 接触引导） |
| s20_cr 比 s20 手部接触率高 >20pp | contact reward 有效 → 后续保留 |
| 所有变体 obj_mean > 0.50m | partner translational spring 不足 → 转 H002 (virtual grasp) 或 H004 (SBTO) |
| guard 倒或 leg_intf > 10% | 弹簧对小物体过强 → 降低 guard kp 或分开处理 |
