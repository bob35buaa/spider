# E013: Intra-rollout Mocap Partner + Reward Sweep — 结果

## 状态: 部分成功 (C1/C2 同时通过，但高方差)

## 核心发现

1. **`wp.to_torch` 共享内存写入在 CUDA graph launch 前生效** — 技术修复成功
2. **Intra-rollout mocap 使 CEM 优化更难** — partner 运动增加了 rollout 内的动力学复杂度
3. **高 contact_rew + intra-rollout 是最佳组合** — 但 CEM 高方差
4. **C1/C2 trade-off 被部分打破** — 最佳配置偶尔同时通过两个阈值

## 实验矩阵

### E013a: Intra-rollout 技术验证

| 测试 | 结果 |
|------|------|
| `wp.copy(data_wp.mocap_pos, ...)` before graph | **失败**: `dest.ptr` is None after graph capture |
| `wp.to_torch(data_wp.mocap_pos)` in-place write | **成功**: 共享内存，graph 读到新值 |
| 独立验证: 写入后 graph 是否使用新值 | **确认**: mocap_pos 写入 (1,2,0.5) → graph step 后仍为 (1,2,0.5) |

### E013c: Reward 权重扫描

| Run | intra | pos_rew | base_pos | contact | obj_z_max | pelvis_min | 评价 |
|-----|-------|---------|----------|---------|-----------|------------|------|
| ctrl | no | 3.0 | 3.0 | 1.0 | 0.463 | 0.687 | 接近 E011, 偶尔 C1+C2 |
| r1 | yes | 3.0 | 3.0 | 1.0 | 0.311 | 0.161 | intra 恶化 (CEM 更难) |
| r2 | no | 0.0 | 5.0 | 1.0 | 0.389 | 0.136 | body only → 无 lift |
| r3 | no | 1.5 | 5.0 | 2.0 | 0.372 | **0.697** | C2 pass, C1 fail |
| r4 | no | 2.0 | 7.0 | 1.0 | 0.345 | 0.160 | 失败 |
| r5 | no | 3.0 | 10.0 | 1.0 | 0.445 | 0.111 | 极端 base 无效 |
| r6 | yes | 3.0 | 10.0 | 1.0 | 0.337 | **0.701** | intra 帮助 body |
| **r7** | **yes** | **2.0** | **5.0** | **3.0** | **0.477** | **0.712** | **最佳 — C1+C2 同时通过** |
| r7b | no | 2.0 | 5.0 | 3.0 | 0.368 | 0.691 | 无 intra → 更差 |
| r8 | yes | 2.0 | 5.0 | 5.0 | 0.338 | **0.738** | contact 过强 → body only |
| r9 | yes | 3.0 | 5.0 | 3.0 | 0.456 | 0.115 | 高 pos → pelvis 崩溃 |

### E013-r7 稳定性 (5 次运行)

| Run | obj_z_max | pelvis_min | C1 (≥0.40) | C2 (≥0.50) |
|-----|-----------|------------|-----------|-----------|
| 1 (原始) | 0.477 | 0.712 | PASS | PASS |
| 2 | 0.316 | 0.634 | FAIL | PASS |
| 3 | 0.466 | 0.161 | PASS | FAIL |
| 4 | 0.349 | 0.683 | FAIL | PASS |
| 5 | 0.445 | 0.692 | PASS | PASS |

**通过率**: C1 3/5 (60%), C2 4/5 (80%), C1+C2 同时 2/5 (40%)

### E013-ctrl 代表运行

| obj_z_max | pelvis_min | C1 | C2 | 物体 lift 帧数 |
|-----------|------------|----|----|--------------|
| 0.463 | 0.687 | PASS | PASS | MPC step 4-5 (0.410, 0.408) |

## Claims 验证

| Claim | 阈值 | 最佳单次 | 稳定性 | 通过? |
|-------|------|---------|--------|------|
| C1: obj z_max ≥ 0.40m | 0.40 | 0.477 (r7) | 3/5 次 | **部分** |
| C2: pelvis_min ≥ 0.50m | 0.50 | 0.712 (r7) | 4/5 次 | **部分** |
| C3: obj z>0.35m ≥ 3帧 | 3帧 | 2 MPC步 (ctrl) | 不稳定 | **FAIL** |
| C4: 视频确认物体离地 | 视觉 | 箱子被推开/翻转 | — | **FAIL** |
| C5: 方案A技术可行 | 无crash | 成功 | — | **PASS** |

## 可视化观察

### E013-r7 (intra, contact=3.0, 此次运行 obj_z=0.316)

| 时间 | ref | sim |
|------|-----|-----|
| t=0s | G1 站在箱前直立 | G1 站在箱前直立，橙色 partner 胶囊可见于箱子右下角 |
| t=1s | 弯腰手伸向箱子 | 弯腰姿态相似，手接近箱面，partner 胶囊在箱底 |
| t=2s | 半蹲抱箱抬起 | G1 站直走向箱子侧面，箱子被推向右侧，partner 胶囊在地上 |
| t=3s | 持箱站立 | G1 独自站立，箱子在远处。partner 胶囊滞后于箱子 |
| t=4s | 弯腰放下箱子 | G1 弯腰，箱子在远处地面 |

**视觉结论**: 箱子被推开而非抬起。Partner 胶囊体太小（r=0.04m），在箱底提供的力不足以阻止箱子被推走。

### E013-ctrl (nointra, E011 config, 此次运行 obj_z=0.463)

| 时间 | ref | sim |
|------|-----|-----|
| t=0s | G1 站在箱前直立 | G1 直立，partner 胶囊可见于箱右下角 |
| t=1s | 弯腰手伸向箱子 | G1 弯腰，手臂伸向箱面，可见手接触箱顶 partner 胶囊贴在箱前 |
| t=2s | 半蹲抱箱 | G1 站立于箱旁，箱子向右偏移但仍在附近，partner 可见 |
| t=3s | 持箱站立 | G1 站立，箱子被推到右侧但距离不远 |
| t=4s | 弯腰放下 | G1 弯腰，箱子在右侧地面，partner 胶囊在箱下 |

**视觉结论**: t=1s 手臂与箱面有接触，MPC step 4-5 的 obj_z=0.41 可能是箱子被短暂翘起（非整体抬升）。机器人始终站立稳定 (pelvis≥0.69)。

## 关键 Insight

### 1. Intra-rollout 是双刃剑

| 场景 | intra 效果 | 原因 |
|------|-----------|------|
| 低 contact_rew (1.0) + 高 pos_rew (3.0) | **负面** | CEM 追 moving object → 过度前倾 |
| 高 contact_rew (3.0) + 中 pos_rew (2.0) | **正面** | CEM 关注 hand proximity → 配合 partner |
| 极高 base_pos (10.0) | **正面** (body stable) | 但 obj 完全不动 |

**规律**: Intra-rollout 只在 contact_rew 主导时有帮助，因为 CEM "看到" partner 移动后能计划手跟随 partner 接近箱子。

### 2. obj_z 的"假抬起"

qpos 中 obj_z_max=0.46 的真实含义:
- **不是整体抬升**: 视频确认箱子从未完全离地
- **是翻转/翘起**: 机器人前臂推到箱子一角 → 箱子绕底边旋转 → 对角 z 升高
- 与 E006/E008 的诊断一致: obj_z > 0.35 不等于"搬起"

### 3. Partner 胶囊体太小

当前 mocap partner: capsule size=(0.04, 0.08)，即半径 4cm、半长 8cm。总高度 ≈ 24cm。
箱子尺寸: 61 × 61 × 89cm。
Partner 胶囊位于箱底附近，**无法提供有效的侧面支撑力**。

### 4. C1/C2 trade-off 的本质

| 物体跟踪 (C1) | 需要 CEM 找到 "手推箱子" 的控制 → 身体前倾 |
| 身体稳定 (C2) | 需要 CEM 保持 "站直" 的姿态 → 不接触箱子 |

这不是 reward 调参能解决的——是 CEM 采样 MPC 的根本限制:
- CEM 一次只优化 0.4s horizon
- 在 0.4s 内，"稳定站立" 和 "前倾推箱" 是互斥的局部最优
- 没有长期规划能力来找到 "先弯腰→稳步推→再直立" 的序列

## 结果路径

| 产出 | 路径 |
|------|------|
| E013-r7 最佳轨迹 | `workspace/core4d/results/E013_box025_p1_intra_mocap.npz` |
| E013-r7 视频 | `workspace/core4d/results/E013_box025_p1_intra_mocap.mp4` |
| E013-ctrl 视频 | `workspace/core4d/results/E013_ctrl_nointra.mp4` |
| 配置 | `examples/config/override/core4d_box025_e013.yaml` |
| 脚本 | `workspace/core4d/scripts/retarget/retarget_core4d_e013.sh` |
| 计划 | `workspace/core4d/plan/11_phase3_retarget_roadmap.md` |

## 代码改动

| 文件 | 改动 |
|------|------|
| `spider/config.py` | +`mocap_partner_intra_step: bool = True` |
| `spider/simulators/mjwp.py::_update_mocap_partner` | 改用 `wp.to_torch` 共享内存写入 (修复 `wp.copy` ptr=None 问题) |
| `spider/simulators/mjwp.py::step_env` | 在 `wp.capture_launch` 前调用 `_update_mocap_partner` (条件性) |

## 运行命令

```bash
# E013-r7 最佳配置
uv run examples/run_mjwp.py +override=core4d_box025_e013 \
    mocap_partner_intra_step=true \
    pos_rew_scale=2.0 rot_rew_scale=0.5 \
    base_pos_rew_scale=5.0 base_rot_rew_scale=2.0 \
    contact_rew_scale=3.0 \
    task=box025_person1 data_id=0 viewer=none

# E013-ctrl (复现 E011)
uv run examples/run_mjwp.py +override=core4d_box025_e013 \
    mocap_partner_intra_step=false \
    task=box025_person1 data_id=0 viewer=none
```

## E013b (mjwp_eq) 跳过原因

mjwp_eq 的 per-step mocap 更新等价于 E013a 的 intra-rollout 方案。E013a 已验证该功能技术可行但效果有限，mjwp_eq 不会带来额外改善。核心瓶颈不是 mocap 更新频率，而是 CEM 采样 MPC 的规划能力。

## 下一步建议

### 短期 (E014): 增大 partner 碰撞体

当前 partner capsule r=0.04m 太小。增大到:
- size=(0.08, 0.12) → 半径 8cm, 长 24cm
- 放在箱子侧面而非底部
- 增大 friction
- 这可能让 partner 提供有效的侧面支撑力

### 中期: Weld 约束 + Mocap Partner (原 E014 方案)

如果增大 partner 碰撞体仍不够:
- mjwp_eq 约束退火保证 G1 手在箱面
- partner 提供另一侧物理力
- CEM 只需优化 body

### 长期: 接受 body-only 导出 (E012 路线)

E012 已证明 body-only retargeting (pelvis_err=0.083m) 质量优秀。
物体交互留给 Holosoma RL 的 interaction reward。
这可能是最务实的路线。
