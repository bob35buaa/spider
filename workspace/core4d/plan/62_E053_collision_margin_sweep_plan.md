# E053: Collision Box Margin Sweep — 寻找最优碰撞盒大小

## 状态: 计划中

## Context

### 碰撞盒修复回顾

`fix_collision_boxes.py` (2026-05-11) 发现并修复了 21 个 CORE4D case 的碰撞盒模板 bug:
- 原始碰撞盒从模板 (box025/bucket005/desk005) 复制, 未按实际 mesh 调整
- 最严重: box023 碰撞盒 1.8x 视觉 mesh (手在距表面 ~13cm 就碰到隐形墙)
- bucket010 碰撞盒 Y/Z 互换 (物理形状完全错误)

修复方案: `collision_size = mesh_AABB × 1.05` (5% margin)

### 问题: 1.05x margin 使 box025 结果退化

E041c (pre-fix, box025 碰撞盒 0.85x mesh) vs E048 (post-fix, 1.05x mesh) 视频对比:

| | Pre-fix (E041c, 0.85x) | Post-fix (E048, 1.05x) |
|---|---|---|
| t=2s | 机器人站着弯腰, 手在箱子侧面 | 机器人已完全趴在箱子上 |
| t=3s | 箱子倾斜但机器人还站着 | 姿态更差, 手完全没碰到箱子 |
| t=4s | 最终趴在箱子上, 但前几秒合理 | 比修复前更早崩溃 |
| 碰撞体积 | 0.332 m³ | **0.534 m³ (+61%)** |

**根因分析**:
1. 1.05x margin 使碰撞盒比视觉 mesh **大 5%** → 手在距视觉表面 ~2cm 处就被弹开
2. 大碰撞盒 → 更大碰撞力矩 → 机器人更容易被推倒
3. 0.85x (旧值) 虽偏小但对 CEM 有利: 手可穿过视觉表面少许 → 更深 contact → 更大摩擦力

### 各 case 的碰撞盒变化

| Case | Mesh AABB | Old (template) | 1.05x (当前) | Old/Mesh 比 | 模板问题类型 |
|------|-----------|---------------|-------------|------------|-----------|
| box025 | 0.359×0.360×0.447 | 0.305×0.305×0.446 | 0.377×0.378×0.469 | **0.85x** (偏小) | 偏小但形状基本正确 |
| bucket010 | 0.199×0.199×0.372 | 0.201×0.371×0.202 | 0.209×0.209×0.390 | **Y/Z 互换** | 形状完全错误 |
| desk005 | 0.255×0.553×0.403 | 0.200×0.370×0.400 | 0.268×0.580×0.423 | **0.67-0.78x** (过小) | 偏小且形状不匹配 |

**关键区别**: box025 的旧碰撞盒虽然偏小但形状合理 (比例一致), bucket010 和 desk005 的旧碰撞盒形状完全错误.
所以 box025 上 "修复" 实际上是从一个 "虽偏小但能用" 的状态变到了 "过大反而有害" 的状态.

### 各 margin 值的碰撞体积对比

| Case | 0.90x vol | 0.95x vol | 1.00x vol | 1.05x vol (当前) |
|------|-----------|-----------|-----------|-----------------|
| box025 | 0.337 m³ | 0.396 m³ | 0.462 m³ | 0.534 m³ |
| bucket010 | 0.086 m³ | 0.101 m³ | 0.118 m³ | 0.136 m³ |
| desk005 | 0.332 m³ | 0.390 m³ | 0.455 m³ | 0.527 m³ |

---

## Claims

| ID | 描述 | 量化标准 | 依据 |
|----|------|---------|------|
| C1 | 存在某个 margin ≤ 1.0 在 box025 上 **视觉表现** 优于当前 1.05x | 视频逐帧对比: 机器人站立时间更长 AND/OR 手-物体接触更合理 | E041c(0.85x) 视频已证明更小碰撞盒行为更好 |
| C2 | bucket010 在新 margin 下不退化 | Stability ≥ 当前值 AND 无 Y/Z 轴互换问题 | bucket010 旧碰撞盒 Y/Z 互换, 任何正确形状的 margin 都应优于旧值 |
| C3 | desk005 在新 margin 下不退化 | Stability ≥ 当前值 AND 碰撞盒形状正确 | desk005 旧碰撞盒过小(0.67x), 正确形状但更小的 margin 应仍优于旧值 |

**注意**: 基于 E048-E052 的视觉重新评估, **所有 case 都没有实现真正的搬运**.
本实验的目标不是实现搬运, 而是在 body tracking + stability 框架下找到最优碰撞盒大小,
使机器人行为尽可能物理合理 (站立、手接触物体、不穿模).

---

## 实验设计

### E053: Margin Sweep — 3 个 margin × 3 个 case = 9 次实验

| Margin | box025 | bucket010 | desk005 |
|--------|--------|-----------|---------|
| 0.90 | E053a_box025_m090 | E053a_bucket010_m090 | E053a_desk005_m090 |
| 0.95 | E053b_box025_m095 | E053b_bucket010_m095 | E053b_desk005_m095 |
| 1.00 | E053c_box025_m100 | E053c_bucket010_m100 | E053c_desk005_m100 |

**对照组** (已有结果, 无需重跑):
- 1.05x: E048 结果 (box025_baseline, bucket010_baseline, desk005_baseline)
- 旧模板: E041c 结果 (box025, bucket010 — 仅这两个有 pre-fix 结果)

### 碰撞盒应用策略

**重要**: 不能直接用 `fix_collision_boxes.py --margin X` 因为:
1. 脚本有 8% tolerance — 如果当前碰撞盒已在 margin 附近 8%, 会被跳过
2. 需要先回退当前 1.05x, 然后按新 margin 重新计算
3. 对 bucket010 和 desk005, 无论什么 margin 都需要修正形状 (不能回退到旧的错误形状)

**正确做法**: 修改脚本支持 `--force` 参数, 跳过 tolerance check, 强制覆盖.
或者直接写一个新脚本 `set_collision_margin.py` 对指定 case 设置精确 margin.

### 运行计划

每个实验使用 `+override=core4d_e041c` (与 E041c/E048 完全相同的配置).

**本地 (GPU0)**: E053c_box025_m100 (最重要: 1.0x 是否比 1.05x 好)
**远程 GPU0**: E053a_box025_m090, E053b_box025_m095, E053a_bucket010_m090
**远程 GPU1**: E053b_bucket010_m095, E053c_bucket010_m100, E053a_desk005_m090

按优先级:
1. box025 × 3 margins (直接与 E041c pre-fix + E048 post-fix 对比)
2. bucket010 × 3 margins (验证形状修正后 margin 影响)
3. desk005 × 3 margins (验证形状修正后 margin 影响)

9 个实验, 每个 ~30min, 3 GPU 并行 → ~1.5h 全部完成.

---

## 实现步骤

### Step 1: 创建 `set_collision_margin.py` 工具脚本

```python
# workspace/core4d/scripts/convert/set_collision_margin.py
# 为指定 case 的 scene.xml + scene_act.xml 设置碰撞盒 = mesh_AABB × margin
# 支持 --case, --margin, --force 参数
```

与 `fix_collision_boxes.py` 的区别:
- 无 8% tolerance — 精确设置到指定 margin
- 只处理指定 case (不是全部 21 个)
- 同时更新 scene.xml 和 scene_act.xml
- 打印前后对比 (旧值 → 新值)

### Step 2: 创建运行脚本 `run_E053_remote.sh`

```bash
# workspace/core4d/scripts/run_E053_remote.sh
# 远程 2 GPU 并行: box025 × 3 margins + bucket010 × 3 margins + desk005 × 1
```

### Step 3: 执行

1. 修改碰撞盒 (set_collision_margin.py) → 验证 MuJoCo 加载 → git commit
2. 本地跑 E053c_box025_m100
3. `git push` + 远程跑 6 个实验
4. 收集结果 + 视频分析

### Step 4: 评估

**必须包含**:
1. **视频逐帧分析** (每个实验提取 t=0s, 2s, 4s, 6s 关键帧)
2. 数值指标 (Stability, MPKPE, ObjPos, Contact) — 仅作参考, 不作为主要判断依据
3. 与 E041c pre-fix (0.85x) 和 E048 post-fix (1.05x) 的直接对比
4. 各 margin 的行为差异描述 (站立时间, 手-物体接触质量, 穿模情况)

---

## 评估标准 (诚实客观)

基于 E048-E052 的教训, **数值指标不可信**, 必须视觉验证:

| 指标 | 含义 | 陷阱 |
|------|------|------|
| Stability | pelvis_z > threshold 的帧比例 | 机器人可以稳定站立但完全不搬运 (desk005) |
| ObjPos | 物体位置误差 | 物体没被搬 = 低误差 (box023 假象) |
| Contact<10cm | 手距物体 < 10cm 的帧比例 | 手背接触 / 不自然姿势也算 |
| MPKPE | 关节角度追踪误差 | sim vs drifted ref 的 bug (已修正但需警惕) |

**本实验的真实评估标准**:
1. 机器人是否能站立? (视频确认, 不仅看 Stability 数字)
2. 手是否接触物体? (视频确认接触方式是否自然)
3. 有无穿模? (碰撞盒过小 → 手穿过物体; 过大 → 隐形墙)
4. 比较不同 margin: 哪个 margin 下机器人行为最物理合理?

---

## 风险分析

### 风险 1: 修改碰撞盒影响 scene_act.xml 的 actuator 工作

**分析**: scene_act.xml 的 actuator 是 object PD controller, 与碰撞盒大小无关.
碰撞盒只影响 contact detection 和 collision dynamics.

**影响**: 无

### 风险 2: 0.90x margin 导致手穿模

**分析**: 0.90x 碰撞盒比视觉 mesh 小 10%, 手可能穿入视觉表面 ~1-3cm.
对于 CEM 这可能反而有利 (更深的 contact = 更大摩擦力), 但视觉上不美观.

**应对**: 如果 0.90x 行为最好但穿模明显, 考虑 0.95x 作为折中.

### 风险 3: bucket010/desk005 在小 margin 下退化

**分析**: 这两个 case 的旧碰撞盒形状完全错误 (Y/Z 互换 / 过小).
即使 margin=0.90, 形状修正后仍比旧值合理. 退化风险低.

**应对**: 如果某个 case 在某个 margin 下明显退化, 允许 per-case 设置不同 margin.

---

## 成功标准

| 场景 | 如果是 | 结论 |
|------|--------|------|
| margin=1.00 在 box025 上视觉明显优于 1.05x | 1.05x margin 过大, 1.00x 是更好的默认值 |
| margin=0.95 最佳 (允许轻微穿透) | 0.95x 作为新默认值, 重跑所有 case |
| margin=0.90 最佳 | 需要权衡穿模 vs 行为质量 |
| 所有 margin 在 bucket010/desk005 上持平 | 这两个 case 主要改善来自形状修正, 不是 margin |
| 无明显差异 | 碰撞盒大小不是瓶颈, 排除此因素后探索其他方向 |

---

## 执行顺序

```
1. ✏️ 创建 set_collision_margin.py
2. ✏️ 创建 run_E053_remote.sh  
3. 🏃 本地 E053c_box025_m100 (margin=1.00)           [~30min]
4. 🚀 远程 6 实验并行                                 [~1.5h]
5. 📊 收集结果 + 视频逐帧分析
6. 📝 记录实验日志 (log/62_E053_collision_margin_sweep.md)
7. 📝 更新 EXPERIMENT_TRACKER.md
8. 🔧 如果找到更优 margin → 用 set_collision_margin.py 全局应用
```

---

## 改动文件

| 文件 | 操作 | 内容 |
|------|------|------|
| `workspace/core4d/scripts/convert/set_collision_margin.py` | 新建 | 精确设置碰撞盒 margin |
| `workspace/core4d/scripts/run_E053_remote.sh` | 新建 | 远程并行运行脚本 |
| `example_datasets/.../*/scene.xml` | 修改 | 碰撞盒大小 (逐轮修改+重跑) |
| `example_datasets/.../*/scene_act.xml` | 修改 | 碰撞盒大小同步 |
| `workspace/core4d/log/62_E053_collision_margin_sweep.md` | 新建 | 实验日志 |
| `workspace/core4d/plan/62_E053_collision_margin_sweep_plan.md` | 本文件 | 实验计划 |
