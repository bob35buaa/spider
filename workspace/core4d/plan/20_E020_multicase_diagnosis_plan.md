# E020: SPIDER 单人重定向能力多 Case 诊断

## Context

**问题**: 前 18 个实验集中在 box025 (大物体、双人夹、G1 臂展不够) 和 bucket005 (小但在地面、需深蹲)。两个 case 都失败后归因为 "case 太难 + CEM 短视"。但:
- E015 (bucket005) 在加入 mocap partner 的基础上测试 → partner 本身可能引入了干扰
- 没有在任何 **难度适中** 的 case 上做过纯净的 SPIDER baseline 测试
- 无法区分: 是 **所有 case 都不行** (SPIDER 算法瓶颈), 还是 **只有这两个极端 case 不行** (case 选择偏差)

**可用数据 (holosoma v2 retarget output)**:

| Case | 物体尺寸 (m) | 初始 obj_z | ref lift | pelvis 变化 | 帧数 | 难度预判 |
|------|------------|----------|----------|------------|------|---------|
| box025_p1 | 0.61×0.61×0.89 | 0.310 | 0.223m | min=0.747 | 124 | **极难** (双人、大物体) |
| bucket005_p1 | 0.30×0.41×0.29 | 0.126 | 0.121m | min=0.604 | 160 | **难** (地面、深蹲) |
| bucket010_p1 | 0.40×0.74×0.40 | 0.373 | 0.195m | min=0.768 | 125 | **中等** (桌面高度、站姿) |
| chair022_p1 | 0.57×0.86×0.53 | 0.221 | 0.313m | min=0.663 | 127 | **中等** (需弯腰但幅度小) |
| desk005_p2 | 0.40×0.74×0.80 | 0.368 | 0.105m | min=0.776 | 116 | **较易** (高位、小 lift) |

## 核心假设

**H0 (SPIDER 能力假设)**: SPIDER SBMPC 在难度适中的 case (bucket010, desk005) 上, body-only 重定向质量 pelvis_err ≤ 0.10m, 且物理环境中物体至少有碰撞位移 (obj_Δz > ref_lift * 20%)。

**H1 (Case 难度假设)**: box025/bucket005 的失败是 case 特异性问题 (臂展不够 / 深蹲极端), 不代表 SPIDER 对所有 CORE4D case 无能。

## Claims (成功标准)

| Claim | 定义 | 量化阈值 |
|-------|------|---------|
| C1: 数据管线通用 | 所有新 case scene.xml 生成 + SPIDER 加载成功 | 0 crash, nq=43 |
| C2: Body 跟踪质量 | body-only 模式 pelvis_err | ≤ 0.15m (至少 2/3 case 通过) |
| C3: Body 稳定性 | pelvis_z_min | ≥ 0.50m (至少 2/3 case) |
| C4: 物体交互信号 | obj 有物理位移 (非仅碰撞推动) | obj_Δz > ref_lift * 20% (至少 1/3 case) |
| C5: 难度梯度验证 | 简单 case 明显优于 box025/bucket005 | pelvis_err 排序符合难度预判 |

## 实验矩阵

### Phase A: 数据管线 (E020-a)
为 bucket010, chair022, desk005 各生成:
1. scene.xml (G1 + object + ground)
2. trajectory_kinematic.npz (via core4d.py)
3. 验证 SPIDER 加载无 crash

### Phase B: Body-Only Baseline (E020-b)
对所有 5 个 case 跑纯净的 body-only retargeting (pos_rew=0, contact_rew=0, base_pos=10):
- 只优化 body tracking, 不加任何 object reward
- 统一参数, 对比纯 body 能力

### Phase C: 有 Object Reward (E020-c)
对所有 5 个 case 加 pos_rew=2.0, contact_rew=1.0:
- 看哪些 case 能产生有效 object 交互
- 不加 mocap partner / connect, 纯 SPIDER baseline

### Phase D: 可视化对比 (E020-d)
- 每个 case 生成 mp4 视频
- 提取关键帧, 制作对比图
- 生成综合 metrics 表

## 代码改动清单

| 步骤 | 文件 | 改动 |
|------|------|------|
| A1 | `workspace/core4d/scripts/convert/setup_new_cases.py` | 新脚本: 批量生成 scene.xml + 运行 core4d.py |
| A2 | `examples/config/override/core4d_bucket010.yaml` | bucket010 配置 |
| A3 | `examples/config/override/core4d_chair022.yaml` | chair022 配置 |
| A4 | `examples/config/override/core4d_desk005.yaml` | desk005 配置 |
| B1 | `workspace/core4d/scripts/retarget/retarget_multicase_bodyonly.sh` | 批量 body-only 脚本 |
| C1 | `workspace/core4d/scripts/retarget/retarget_multicase_withobj.sh` | 批量 obj-reward 脚本 |
| D1 | `workspace/core4d/scripts/eval/eval_multicase_metrics.py` | 统一指标提取 + 对比表 |

## Results 目录组织

```
workspace/core4d/results/
├── E020_multicase_diagnosis/
│   ├── bucket010/
│   │   ├── bodyonly.npz
│   │   ├── bodyonly.mp4
│   │   ├── withobj.npz
│   │   └── withobj.mp4
│   ├── chair022/
│   │   ├── bodyonly.npz
│   │   ├── bodyonly.mp4
│   │   ├── withobj.npz
│   │   └── withobj.mp4
│   ├── desk005/
│   │   ├── bodyonly.npz
│   │   ├── bodyonly.mp4
│   │   ├── withobj.npz
│   │   └── withobj.mp4
│   ├── box025/ (复用已有结果)
│   │   └── bodyonly.npz → symlink
│   ├── bucket005/ (复用已有结果)
│   │   └── bodyonly.npz → symlink
│   ├── metrics_summary.csv
│   ├── comparison_chart.png
│   └── keyframes/
│       ├── bucket010_mpc{0,3,5,8}.png
│       ├── chair022_mpc{0,3,5,8}.png
│       └── desk005_mpc{0,3,5,8}.png
```

## 执行顺序

```
Step 1: 生成 scene.xml (bucket010, chair022, desk005)
  → 验证: mj.load_xml 无 crash, nq=43
Step 2: 转换 trajectory_kinematic.npz (3 个新 case)
  → 验证: npz shape 正确, 播放无异常
Step 3: 跑 body-only retargeting (5 cases)
  → 验证: pelvis_err, pelvis_z_min
Step 4: 跑 with-obj retargeting (5 cases)
  → 验证: obj_Δz, obj_pos_err
Step 5: 提取 metrics + 生成可视化
  → 验证: 5 case 对比表 + 关键帧
```

## 风险

1. **新 case 的 scene.xml 生成可能需要物体碰撞几何调整** — 用 bounding box 近似
2. **desk005 只有 person2 数据 (非 person1)** — 直接用 person2, task name 改为 `desk005_person2`
3. **chair022 形状不规则** — 可能需要 convex decomposition (先用 box 近似)
