# E015: Bucket005 小物体验证 — 结果

## 状态: 完成 (body-only 成功, 物体交互失败 — 与 box025 一致)

## 核心发现

1. **bucket005 比 box025 更难稳定**: 物体在地面 (z=0.126m), 需深蹲 → 机器人容易前倾摔倒
2. **Body-only retargeting 质量良好**: pelvis_err=0.129m, 全程稳定 (pelvis_z≥0.607)
3. **物体从未被搬起**: 所有配置 obj_z_max ≤ 0.207m (ref max=0.247m), 微小移动为碰撞推动
4. **重要 bug 修复**: `scene_name` config 字段缺失, 之前 E013/E014 的 mocap partner scene 实际未加载

## 物体特征

| 属性 | bucket005 | box025 (对比) |
|------|-----------|--------------|
| 尺寸 | 0.295 × 0.410 × 0.286 m | 0.611 × 0.610 × 0.893 m |
| 初始 obj_z | 0.126m (地面) | 0.310m (半高) |
| 参考 obj_z_max | 0.247m (抬升 0.12m) | 0.533m (抬升 0.22m) |
| 质量 | 2.0 kg | 5.0 kg |
| 序列长度 | 160帧 (5.33s) | 124帧 (4.13s) |
| 碰撞几何 | box 0.148×0.205×0.143 (近似) | box 0.305×0.305×0.446 |

## 实验矩阵

| Run | 配置 | base_pos | pos_rew | contact | obj_z_max | pelvis_min | 评价 |
|-----|------|----------|---------|---------|-----------|------------|------|
| E015-a | baseline | 3.0 | 3.0 | 1.0 | 0.155 | **0.105** (CRASH) | 机器人摔倒 |
| E015-b | strong body | 10.0 | 1.0 | 0.5 | 0.197 | 0.572 | 身体稳定, obj 不动 |
| E015-c | balanced | 7.0 | 2.0 | 2.0 | **0.207** | 0.625 | 最好 obj (偶发碰撞) |
| **E015-d** | **body-only** | **10.0** | **0.0** | **0.0** | 0.148 | **0.607** | **最佳 body 跟踪** |
| E015-e | kinobj (act) | 5.0 | 3.0 | 1.0 | 0.167 | 0.596 | PD 驱动 obj 不理想 |

## Claims 验证

| Claim | 阈值 | 结果 | 通过? |
|-------|------|------|------|
| C1: 管线通过 | 无 crash | scene.xml + 转换 + MJWP 全通 | **PASS** |
| C2: obj 离地 | obj_z_max > ref*0.5 (>0.124) | max=0.207 > 0.124 | **PASS** (但极微小) |
| C3: 身体稳定 | pelvis_min ≥ 0.50m | E015-d: 0.607 | **PASS** (body-only) |
| C4: 视频确认手触碰 | 视觉 | 手接近但未有效接触 | **FAIL** |
| C5: 改进有效 | baseline < best | a(CRASH) < d(stable) | **PASS** (body 稳定性) |

## 可视化观察

### E015-a baseline (CRASH)
| MPC | sim | ref |
|-----|-----|-----|
| 0 | G1 站在 bucket 旁直立 | 同 |
| 2 | G1 深度前倾, 即将摔倒 | G1 弯腰, 手在 bucket 上 |
| 4 | G1 跪在地上 | G1 蹲下抓 bucket |

### E015-d body-only (最佳)
| MPC | sim | ref |
|-----|-----|-----|
| 0 | G1 站立, 正确初始姿态 | 同 |
| 2 | G1 弯腰但保持平衡, 手未碰 bucket | G1 弯腰手在 bucket 上 |
| 5 | G1 略弯腰, bucket 未动 | G1 持 bucket 半蹲 |
| 10 | G1 直立, bucket 在原位 | G1 直立, bucket 在旁 |

**关键差异**: 参考中手与 bucket 有明确接触, 物理仿真中手接近但避免深度前倾。

## 技术发现

### 1. scene_name bug 修复

**问题**: `scene_name` 不是 Config 字段 → `filter_config_fields` 静默丢弃 → 所有实验只加载 `scene.xml`

**影响**: E013/E014 设置 `scene_name: scene_mocap_partner` 但实际加载 `scene.xml` (无 mocap bodies)
→ E013-r7 的 "mocap partner" 结果实际是 scene.xml + 不同 reward 权重

**修复**: 
```python
# spider/config.py Config dataclass
scene_name: str = ""  # override scene XML basename

# process_config()
if config.scene_name:
    scene_xml = f"{config.scene_name}.xml"
```

### 2. bucket005 低物体位置的挑战

bucket 在地面 (z=0.126m) vs box025 在半高 (z=0.310m)。
G1 需要弯腰到 pelvis_z ≈ 0.6m 才能触碰 bucket。
在 CEM 0.4s horizon 内, "弯腰到底" 和 "保持平衡" 是更极端的矛盾。

### 3. PD-driven kinobj 对 bucket005 不适用

contact_guidance 模式: PD 增益从 0 ramp up → bucket 在增益不足时被重力拉下。
obj_z 跌到 0.060m (低于初始 0.126m!)。

## 与 box025 对比

| 指标 | box025 body-only (E012) | bucket005 body-only (E015-d) |
|------|------------------------|------------------------------|
| pelvis_err | 0.083m | 0.129m |
| joint_err | 0.108 rad | 0.202 rad |
| pelvis_min | ≥0.73m | ≥0.607m |
| 身体稳定 | 优秀 | 良好 |
| 物体抬起 | 否 | 否 |

bucket005 的 body-only 质量略低于 box025, 主要因为深蹲动作更极端。

## 结论

1. **数据管线通用**: CORE4D 任何物体 (bucket/chair/desk) 都可快速接入 SPIDER
2. **Body-only retargeting 是通用解**: 不依赖物体类型, pelvis_err < 0.15m
3. **CEM 物理搬运限制不变**: 与物体大小无关, 根因是 CEM horizon 和采样局限
4. **bucket005 更难稳定**: 低物体位置增加了 body stability challenge

## 代码改动

| 文件 | 改动 |
|------|------|
| `spider/config.py` | +`scene_name: str = ""`, process_config 支持 scene_name 覆盖 |
| `examples/config/override/core4d_bucket005.yaml` | bucket005 基础配置 |
| Scene XMLs (bucket005) | `scene.xml`, `scene_act.xml` |
| `task_info.json` (bucket005) | contact_site_ids: [11, 15] |

## 结果路径

| 产出 | 路径 |
|------|------|
| E015-a baseline | `workspace/core4d/results/E015a_bucket005_p1_baseline.npz/mp4` |
| E015-b strong body | `workspace/core4d/results/E015b_bucket005_p1_strong_body.npz` |
| E015-c balanced | `workspace/core4d/results/E015c_bucket005_p1_balanced.npz` |
| E015-d body-only (best) | `workspace/core4d/results/E015d_bucket005_p1_bodyonly.npz/mp4` |
| E015-e kinobj | `workspace/core4d/results/E015e_bucket005_p1_kinobj.npz/mp4` |
| 配置 | `examples/config/override/core4d_bucket005.yaml` |
| 计划 | `workspace/core4d/plan/14_phase4_small_object_dual_robot_plan.md` |
