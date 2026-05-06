# Phase 3 实验路线图：死磕重定向（续）

## 1. Phase 1-2 全景复盘

### 1.1 实验时间线（E001-E012）

```
Phase 1 (E001-E009): 单人重定向探索 → 全部失败，发现几何根因
Phase 2 (E010-E012): 辅助方案验证 → 运动学可行 + 部分物理成功 + 导出OK
```

| Run | 方法 | obj z_max (实测) | pelvis_min | 核心发现 |
|-----|------|-----------------|-----------|---------|
| E002 | 基线 (无引导) | 0.305 (落地) | 0.81 | 物体完全不动 |
| E006f | 3-box前臂+contact_rew | 0.307 (推开) | 0.584 | **虚假突破 — 视频证实只是推** |
| E007a | physics_dt=0.005 | 0.313 | 0.715 | sim 对齐破坏接触时机 |
| E008a | 衰减PD引导 | 0.375 | 0.635 | 引导期OK撤除即落 |
| E009c | gravcomp | 0.381 (底面仅+8mm) | 0.278 | **问题不是力，是接触几何** |
| E010-kinobj | PD驱动物体+body CEM | PD超调 | **0.733** | **G1运动学可行** |
| **E011** | **mocap partner (full rew)** | **0.460** | 0.116 | **最佳物理结果，但机器人摔倒** |
| E011b | mocap partner (body only) | 0.320 | **0.690** | 稳定但物体不动 |
| E012 | body only (无partner) | 0.305 | 0.750 | 身体跟踪最佳 (err=0.083m) |

### 1.2 三层根因更新

```
表层: "物体抬不起来"
  ↓
中层: G1 手/前臂几何无法形成力闭合
      + CEM搜索空间中"正确协作姿态"被"推开箱子"局部最优淹没
  ↓
深层 (Phase 1): CORE4D box025 协作任务 → 单人几何不可解
深层 (Phase 2 新): E011 架构限制 → mocap partner 在 rollout 内静止
                    → CEM 在错误的物理假设下优化 → 找不到协作策略
```

### 1.3 关键代码发现 (本次新发现)

| 发现 | 影响 |
|------|------|
| `mjwp_eq.py` 在 `step_env` 内调用 `update_mocap_pos` (line 668) | **可以在 rollout 内更新 mocap** — 直接修复 E011 限制 |
| `mjwp_eq.py` 支持约束退火 (soft→stiff, lines 203-254) | weld/connect 约束可以配合 CEM 退火 |
| `wp.copy(data_wp.ctrl, ...)` 在 graph launch 前执行是安全的 | `wp.copy(data_wp.mocap_pos, ...)` 在 graph launch 前也应该安全 |
| Gibbs sampling (run_mjwp.py:371-451) 已有交替优化框架 | 可扩展为双机器人交替优化 |

### 1.4 已验证的死路（不再重复）

| 方向 | 实验依据 | 为什么不行 |
|------|---------|-----------|
| 调 reward 权重 | E006a-f | 接触方向错 |
| 调 PD 增益 | E006c, E007b | 高kp振荡/低kp不足 |
| 改 physics_dt | E007a | 破坏接触时机 |
| 衰减PD引导 | E008a | 撤除即落 |
| 减质量/gravcomp | E009b/c | 问题不是力 |
| 残余 actuator | E009a/a2/a3 | 虚拟弹簧不解决几何 |
| 静态 connect 约束 (mjwp.py) | E010-connect | 太软/太硬两极 |
| **mocap 仅在 sync_env 更新** | **E011** | **partner 在 rollout 内静止** |

---

## 2. Phase 3 核心策略

### 2.1 问题重新定义

Phase 2 问题: "让重定向产生物体真实离地的轨迹"
Phase 3 问题: **"让 G1 + partner 在物理仿真中协作搬运 box025"**

关键转变:
- **修复 E011 的架构限制**（rollout 内 mocap 更新）
- **利用 mjwp_eq 的约束退火能力**
- **验证必须用 qpos + 视频**（Phase 1 教训）
- **结果写入 workspace/core4d/results/**

### 2.2 实验路线图

```
E013: Intra-rollout Mocap Partner (修复E011架构限制)
  "partner手在rollout内跟随轨迹运动, C1/C2能否同时通过?"
  │
  ├─ E013a: mjwp.py step_env 内 wp.copy mocap (最小改动)
  ├─ E013b: 切换到 mjwp_eq 作为 simulator (利用已有mocap更新)
  ├─ E013c: 奖励权重扫描 (在修复后的架构上)
  │
  ├─ 成功 (C1+C2) → E015: 质量优化 + 多物体泛化
  │                  → 导出到 results/ + 尝试 bucket/chair
  │
  └─ 失败 → E014: mjwp_eq Weld 退火 + Mocap Partner
            "约束退火保证抓握 + mocap提供协作力"
            │
            ├─ 成功 → E015: 导出 + 泛化
            │
            └─ 失败 → E015: 双机器人交替优化
                      "两个G1联合重定向, 扩展Gibbs框架"
```

---

## 3. E013 详细计划: Intra-rollout Mocap Partner

### 3.1 Context

E011 是所有非kinobj方案中物体抬起最高的 (z=0.460)。但 C1/C2 存在 trade-off:
- full reward: 物体抬起但机器人摔倒
- body only: 机器人稳定但物体不动

**核心假设**: 这个 trade-off 主要来自**架构限制**而非物理不可行:
- mocap partner 在 rollout 内静止 → CEM "看到"的物理是错误的
- CEM 认为 partner 不会移动 → 不会规划"配合 partner 一起抬"的策略
- 结果: CEM 要么忽略 partner (body only), 要么过度前倾去够箱子 (full reward → 摔倒)

如果 partner 在 rollout 内正确运动:
- CEM 能"看到" partner 同步抬箱的物理过程
- CEM 可以规划"我这边也抬，partner 那边也抬"的协作策略
- C1 (物体抬起) 和 C2 (身体稳定) 不再矛盾

### 3.2 技术方案

#### 方案 A (优先): mjwp.py step_env 内 wp.copy

```python
# spider/simulators/mjwp.py::step_env 修改
def step_env(config, env, ctrl_mujoco):
    apply_perturbation(...)
    wp.copy(env.data_wp.ctrl, ...)

    # NEW: 更新 partner mocap (在 graph launch 前)
    if hasattr(env, 'mocap_partner_pos') and env.mocap_partner_pos is not None:
        _update_partner_mocap_pre_step(env)

    wp.capture_launch(env.graph)
```

```python
def _update_partner_mocap_pre_step(env):
    """在 CUDA graph launch 前更新 partner mocap 位置。
    
    关键: wp.copy 写入的是 graph capture 时同一个 buffer 地址，
    所以 graph 内的 mjwarp.step 会读到更新后的值。
    """
    # 读当前时间 (graph launch 前, 是当前步的时间)
    time_arr = wp.to_torch(env.data_wp.time)  # (N,)
    t = time_arr[0].item()
    
    dt = env.mocap_partner_dt
    T = env.mocap_partner_pos.shape[0]
    idx = min(int(t / dt), T - 1)
    
    # 取当前帧 partner 姿态
    pos = env.mocap_partner_pos[idx]    # (2, 3)
    quat = env.mocap_partner_quat[idx]  # (2, 4)
    
    # 扩展到所有 worlds
    N = env.num_worlds
    mocap_pos_all = wp.to_torch(env.data_wp.mocap_pos)   # (N, n_mocap, 3)
    mocap_quat_all = wp.to_torch(env.data_wp.mocap_quat) # (N, n_mocap, 4)
    
    if mocap_pos_all.shape[1] >= 2:
        # 直接写入 (in-place, 不需要 wp.copy)
        mocap_pos_all[:, :2] = pos.unsqueeze(0).expand(N, -1, -1)
        mocap_quat_all[:, :2] = quat.unsqueeze(0).expand(N, -1, -1)
```

**风险**: `wp.to_torch` 返回的 tensor 是否与 `data_wp.mocap_pos` 共享内存？
- 如果共享 → in-place 写入直接生效 → 方案 A 可行
- 如果不共享 → 需要 `wp.copy` → 可能需要方案 B

#### 方案 B (备选): 使用 mjwp_eq simulator

`mjwp_eq.py` 已经在 `step_env` (line 668) 调用 `update_mocap_pos`。其工作方式:

1. `_copy_state(env.data_wp, dwp)` — 复制主 env 到 DR group 的 data
2. `update_mocap_pos(config, env)` — 更新 mocap
3. `wp.capture_launch(graph)` — 执行 graph
4. `_copy_state(dwp, env.data_wp)` — 复制结果回主 env

这避免了 CUDA graph 约束（每步都做完整的 state 拷贝），代价是更多的内存拷贝。

**改动**:
1. 修改 `core4d_box025_mocap_partner.yaml` 添加 `simulator: mjwp_eq`
2. 在 `mjwp_eq.py` 的 `update_mocap_pos` 中添加 partner 轨迹支持
3. 生成 `scene_eq_mocap_partner.xml`（适配 mjwp_eq 格式）

### 3.3 奖励配置扫描

在修复架构后，重新扫描 E011 的奖励权重空间:

| Sub-run | pos_rew | base_pos_rew | contact_rew | 假设 |
|---------|---------|-------------|-------------|------|
| E013-r1 | 3.0 | 3.0 | 1.0 | 复现E011 (应该更好因为架构修复) |
| E013-r2 | 0.0 | 5.0 | 1.0 | body only + contact (不直接追object) |
| E013-r3 | 1.5 | 5.0 | 2.0 | 弱object + 强body + 强contact |
| E013-r4 | 2.0 | 7.0 | 1.0 | 比E011c更强body |

### 3.4 Claims

| Claim | 阈值 | 验证方法 |
|-------|------|---------|
| C1: obj z_max ≥ 0.40m (qpos实测) | 0.40m | `npz['qpos'][:, obj_z_idx]` |
| C2: pelvis_min ≥ 0.50m | 0.50m | `npz['qpos'][:, pelvis_z_idx]` |
| C3: obj z>0.35m 持续 ≥ 3帧 | 3帧 | 帧计数 |
| C4: 视频确认物体离地+partner持续接触 | 视频 | 关键帧分析 |
| C5: 方案A或B至少有一个成功 | 技术可行 | 代码运行无crash |

### 3.5 需要修改的文件

| # | 文件 | 改动 |
|---|------|------|
| 1 | `spider/simulators/mjwp.py::step_env` | 在 graph launch 前添加 partner mocap 更新 |
| 2 | `spider/simulators/mjwp_eq.py::update_mocap_pos` | (方案B) 添加 partner 轨迹支持 |
| 3 | `examples/config/override/core4d_box025_mocap_partner_v2.yaml` | E013 配置 |
| 4 | `workspace/core4d/scripts/retarget/retarget_core4d_e013.sh` | 运行脚本 |

### 3.6 成功标准

| 指标 | E011 (当前最佳) | E013 目标 |
|------|----------------|----------|
| obj z_max | 0.460m (pelvis崩溃) | **≥ 0.40m + pelvis ≥ 0.50m** |
| pelvis_min | 0.116 (E011) / 0.690 (E011b) | **≥ 0.50m** |
| 视频 | 机器人摔倒 | **站立 + 物体离地** |

---

## 4. E014 详细计划: mjwp_eq Weld 退火 + Mocap Partner

### 4.1 Context (依赖 E013 结果)

如果 E013 修复架构后 C1/C2 仍然 trade-off，说明问题不仅是架构限制，还有:
- G1 手部几何无法形成力闭合（即使 partner 正确运动）
- CEM 无法在大搜索空间中同时找到"正确接触"+"稳定身体"

E014 用**约束退火**消除"手部接触"的搜索维度:
- 早期迭代: 弱 weld 约束 → CEM 可以自由探索身体姿态
- 晚期迭代: 强 weld 约束 → 手被锁在物体上 → 只需优化身体运动
- 同时 mocap partner 提供协作力 → 物体有物理支撑

### 4.2 方法

使用 `mjwp_eq` simulator 的约束退火框架:

```yaml
simulator: mjwp_eq
# 约束退火参数 (mjwp_eq 已有支持)
eq_solref_start: [-100, -10]     # 软约束 (CEM探索期)
eq_solref_end: [-1000, -100]     # 硬约束 (CEM收敛期)
num_dyn: 4                        # 退火阶段数
```

场景需要同时包含:
- 手-物体 connect 约束 (左/右 wrist → object 接触点)
- mocap partner 碰撞体 (person2 双手)

### 4.3 Claims

| Claim | 阈值 |
|-------|------|
| C1: 约束退火+mocap, obj z_max ≥ 0.40m | 0.40m |
| C2: pelvis_min ≥ 0.50m | 0.50m |
| C3: 最终迭代约束锁定后, 手-物体距离 < 0.05m | 0.05m |
| C4: 视频确认物体被搬起（不是PD假象） | 视频 |

### 4.4 需要修改的文件

| # | 文件 | 改动 |
|---|------|------|
| 1 | `workspace/core4d/scripts/generate_scene_eq_mocap_partner.py` | 新增: 结合 eq 约束 + mocap partner 的场景生成 |
| 2 | `spider/simulators/mjwp_eq.py` | 适配 partner 轨迹 + connect 约束退火 |
| 3 | `examples/config/override/core4d_box025_eq_mocap.yaml` | E014 配置 |

---

## 5. E015 详细计划: 导出 + 泛化 (或 Dual-Robot 兜底)

### 5.1 如果 E013/E014 成功

导出最佳重定向结果:
1. **导出格式**: Holosoma RL compatible NPZ (沿用 E012 格式)
2. **结果路径**: `workspace/core4d/results/E0XX_box025_person1_retarget.npz`
3. **包含字段**: body_pos_w, body_quat_w, joint_pos, object_pos_w, partner_hand_pos_w 等

泛化验证:
- 尝试 CORE4D 其他物体 (需要转换数据)
- 调整 scene XML 和碰撞几何适配不同物体形状

### 5.2 如果 E013+E014 都失败 (双机器人兜底)

**方法**: 两个 G1 联合重定向
- 创建 scene 含 2 个 G1 (person1 + person2) + 共享 box
- 利用已有 Gibbs sampling 框架 (run_mjwp.py:371-451) 扩展:
  - Phase 1: 固定 robot2 (mocap), 优化 robot1
  - Phase 2: 固定 robot1, 优化 robot2
  - 交替迭代直到收敛

**需要**:
- person2 重定向数据 (当前 box025_person2 目录已存在但无轨迹)
- 双机器人场景 XML
- Gibbs 框架扩展到多机器人

---

## 6. 结果管理

### 6.1 结果路径规范

所有重定向产出写入 `workspace/core4d/results/`:

| 文件 | 内容 |
|------|------|
| `E013_box025_p1_mocap_intra.npz` | E013 最佳结果 (intra-rollout mocap) |
| `E014_box025_p1_eq_mocap.npz` | E014 最佳结果 (weld退火+mocap) |
| `E0XX_box025_p1_final.npz` | 最终最佳结果 |
| `E0XX_box025_p1_visualization.mp4` | 最佳结果的可视化视频 |

### 6.2 验证清单 (每次实验必做)

- [ ] qpos 实测 obj z_max (不看 reward 字段)
- [ ] qpos 实测 pelvis z_min
- [ ] 生成可视化视频
- [ ] 视频关键帧分析 (写入 log)
- [ ] 结果 NPZ 复制到 workspace/core4d/results/

---

## 7. 时间预算

| 实验 | 预估 | 依赖 |
|------|------|------|
| E013 方案A (mjwp step_env 修改) | 0.5天 | 无 |
| E013 方案B (mjwp_eq 适配) | 1天 | A失败 |
| E013 奖励扫描 (4个变体) | 0.5天 | A或B成功 |
| E014 (weld退火+mocap) | 1天 | E013失败 |
| E015 (导出/泛化 or 双机器人) | 1-2天 | E013或E014成功 |
| **Total** | **~3-4天** | |

## 8. 优先级

**立即执行**: E013 方案A（最小改动 + 最高信息价值）

**原因**:
1. E011 是非 kinobj 方案中物体抬起最高的 (0.460m) — 说明 mocap partner 确实有效
2. 架构限制是可修复的工程问题，不是物理限制
3. 方案A改动极小（step_env 加几行 mocap 更新）
4. 如果 `wp.to_torch` 返回共享内存 tensor → 修复只需 10 行代码
5. 即使方案A失败，方案B (mjwp_eq) 是已有代码的适配
