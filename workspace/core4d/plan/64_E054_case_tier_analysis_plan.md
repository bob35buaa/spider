# E054 实验计划：CORE4D Case 几何可达性 + Mocap 数据质量联合分析

## 状态: 计划中

## Context

### 上游路线图

详见 `workspace/core4d/plan/63_phase17_post_E053_roadmap_plan.md`：
E001-E053 共 53 个实验后，Phase 17 进入 4 条新路径（A 收口 / B warmstart / C force-closure / D 差分物理）。E054 是 Path A 的第一个具体实验，**不跑物理仿真，纯数据分析**，目的是为 E055+ 锁定首批验证 case 与方法路径。

### 前置发现（E001-E053 总结，详见 `log/63_E001_E053_stage_summary.md`）

53 个实验**视觉合格率 = 0**，5 类方法触顶。两个根因：

1. **几何根因**：CORE4D box025 (75 cm) > G1 双臂夹持极限 (~70 cm)，单人结构性不可解（E009 已确认）；但之前把 box023 (35 cm 视觉外形 / 17 cm mesh AABB) 错误归为同类
2. **数据根因**：mocap 本身不准 — ref 里 SMPL 手 mesh 与物体 mesh 距离很多 case 从未 < 1 cm，"找接触帧"是循环假设（动力学重定向存在的理由就是修正 mocap，不能反过来依赖 mocap 当 ground truth）

### 根因分析

之前所有实验把所有 case 一视同仁地丢进 CEM，混淆了三类失败：
- 几何不可解（box025 单人）→ 永远不会成功
- 几何可解但 mocap 数据质量差 → 需要主动修正 mocap（Path B/C）
- 几何可解且 mocap 质量好 → 应该能跑出来但没跑出来 → 真正的算法问题

**只有把这三类拆开，下一阶段实验才能定向**。E054 就是做这次拆分。

### 关键 insight

**用两个独立信号给每个 case 打两个标签**：

1. **几何标签 (Tier 1/2/3)**：物体最大维度 vs 单/双臂展，partner force 占比
2. **数据质量标签 (B-friendly / C-only / drop)**：物体运动信噪比 + 手意图窗口可识别性

两个标签正交，组合决定下一步用什么方法跑。

## Claims

| Claim | 最低证据 |
|-------|---------|
| **C1**: box023 是 Tier 1（推翻之前 Tier 3 的误判） | object_dim_max < 50 cm（单臂展），且 partner force 贡献 < 30% |
| **C2**: box025 确认是 Tier 3（结构性不可解） | object_dim_max > 70 cm，或 partner force > 50% |
| **C3**: 至少 3 个 case 标注为 "B-friendly"，可作为 Path B 验证集 | obj_z_amplitude > 5 cm AND intent_window_count ≥ 1 AND tier ≤ 2 |
| **C4**: 至少识别出 1 个 "C-only" case（物体几乎不动 / mocap 太差），用于 Path C 单独验证 | obj_z_amplitude < 2 cm OR intent_window_count = 0 |
| **C5**: 输出可复现的 csv，下游 E055+ 实验直接消费 | csv 字段完整 + 每个 case 有 recommended_method 标注 |

## 设计：分析脚本

### 输入

CORE4D 已转换的 SPIDER 格式数据：
- `workspace/core4d/results/` 下已有的 21 个 case × 2 person 的 motion npz（含 SMPL body + 物体 6D pose）
- 物体 mesh：`workspace/core4d/object_models/`
- 机器人 kinematic 参数：G1 双臂展 ~1.0 m，单臂展 ~0.5 m，可达高度 ~1.5 m

### 计算流程（每个 case × 每个 person 独立计算）

```
1. 物体几何
   - 加载物体 mesh，计算 AABB
   - object_dim_max = max(AABB.x, AABB.y, AABB.z)
   - object_height = AABB.z
   - object_mass = (从 CORE4D 元数据读取，缺失时 fallback = volume × 0.5 kg/L)

2. 物体运动（决定 obj-motion 锚定可行性）
   - obj_pos_w(t): T × 3 序列
   - obj_z_amplitude = max(obj_z) - min(obj_z) over t
   - obj_pos_traveled = sum(|obj_pos[t+1] - obj_pos[t]|)
   - 数值微分得 obj_acc_z(t)，低通滤波 (cutoff = 5 Hz)
   - obj_acc_z_snr = 10·log10(var(filtered) / var(raw - filtered))

3. 双人协作度（决定 Tier）
   - 对 person1 与 person2 分别计算手到物体距离
   - person2_hand_obj_dist_min: person2 双手与物体的最小距离
   - 如果 person2 手长时间 (>50% 帧) 距物体 < 30 cm → 视为参与
   - partner_force_contribution: 用 quasi-static 假设估计
     - 假设双人对夹时，每人承担约 50% 重量
     - 用 person1/person2 手位置在物体两侧的对称性近似 + 物体加速度 z 分量
     - (这是粗估，用于初判，不要求精确)

4. Mocap 接触意图（决定 B-friendly / C-only）
   - person1_hand_obj_dist(t): person1 双手到物体最近距离
   - hand_obj_dist_min = min(t)
   - hand_obj_dist_p20 = 20th percentile
   - intent windows = 满足三个条件的连续帧段:
     a) hand_obj_dist(t) < hand_obj_dist_p20 + 0.05 m
     b) |hand_velocity(t)| < 0.3 m/s
     c) 持续 ≥ 10 帧 (≈ 0.33 s @ 30fps)
   - intent_window_count = 满足条件的窗口数

5. 几何可达性（决定 Tier）
   - person1 移动起止距离: pelvis_xy_traveled
   - 物体高度 vs G1 抬手范围 (0.5-1.5 m)
   - 物体 dim_max vs 单/双臂展

6. 分级与方法推荐
   - tier:
     - 1 if object_dim_max < 0.5 AND partner_force_contribution < 0.3
     - 2 if object_dim_max < 0.7 AND partner_force_contribution < 0.5
     - 3 otherwise
   - recommended_method:
     - "drop" if tier == 3 AND object_dim_max > 0.7
     - "dual-robot" if tier == 3 AND object_dim_max <= 0.7
     - "B+C" if tier <= 2 AND obj_z_amplitude > 0.05 AND intent_window_count >= 1
     - "C-only" otherwise
```

### 输出

`workspace/core4d/results/E054/case_tier_classification.csv`，列：

```
case_name, person_id, n_frames,
object_dim_max, object_height, object_mass,
robot_two_arm_span, robot_max_reach_height,
obj_z_amplitude, obj_pos_traveled, obj_acc_z_snr,
person2_hand_obj_dist_min, partner_force_contribution,
hand_obj_dist_min, hand_obj_dist_p20,
intent_window_count, intent_window_total_duration,
pelvis_xy_traveled,
tier, recommended_method,
notes
```

并附加 `workspace/core4d/results/E054/E054_case_tier_summary.md`：
- 每个 case 的一行总结
- 按 recommended_method 分组的 case 列表
- B-friendly 首批验证 case 推荐（含理由）

## 改动

### 1. 新建 case 分析脚本

**文件**: `workspace/core4d/scripts/analyze/case_tier_analysis.py`（新建）

主要函数：
```python
def analyze_case(case_name: str, person_id: int) -> dict:
    motion = load_motion(case_name, person_id)        # SPIDER NPZ
    obj_mesh = load_object_mesh(case_name)
    geom = compute_object_geometry(obj_mesh)
    motion_stats = compute_object_motion(motion)
    intent = detect_intent_windows(motion)
    partner = compute_partner_contribution(motion, person_id)
    tier, method = classify(geom, motion_stats, intent, partner)
    return {...}

def main():
    cases = list_all_cases()
    rows = [analyze_case(c, p) for c, p in cases]
    write_csv(rows, "workspace/core4d/results/E054/case_tier_classification.csv")
    write_summary_md(rows, "workspace/core4d/results/E054/E054_case_tier_summary.md")
```

### 2. 复用现有数据加载

**复用**: `spider/process_datasets/core4d.py`、`spider/io.py`

不改动 spider 核心代码，只在 `workspace/core4d/scripts/analyze/` 新增分析脚本。

## 需要修改的文件

| # | 文件 | 改动 |
|---|------|------|
| 1 | `workspace/core4d/scripts/analyze/case_tier_analysis.py` | **新建**主分析脚本 |
| 2 | `workspace/core4d/results/E054/` | **新建**结果目录 |
| 3 | (无 spider 核心代码改动) | — |

## 运行命令

```bash
uv run python workspace/core4d/scripts/analyze/case_tier_analysis.py \
    --cases all \
    --robot g1 \
    --output-dir workspace/core4d/results/E054/
```

CPU-only，不需要 GPU。预期 < 5 min 完成。

## 成功标准

| 指标 | 当前状态 | E054 目标 |
|------|---------|----------|
| 已分级 case 数 | 0 / 21 | **21 / 21** |
| Tier 分布 | 未知 | 明确给出 (Tier1, Tier2, Tier3) 数量 |
| **C1 (box023 → Tier 1)** | 错判为 Tier 3 | **修正为 Tier 1** |
| **C2 (box025 → Tier 3)** | 待确认 | **量化确认 Tier 3** |
| B-friendly case 数 | 0 | **≥ 3** |
| C-only case 数 | 0 | **≥ 1** |
| 输出 csv | 无 | **case_tier_classification.csv 完整** |
| 输出 summary | 无 | **E054_case_tier_summary.md 包含首批验证 case 推荐** |

## 验证步骤

完成后立即手工核对：
1. 打开 csv，逐行检查 box023 / box025 / bucket005 / bucket010 / desk005 / chair022 的 tier 是否符合直觉
2. 抽样 1-2 个标注为 B-friendly 的 case，加载 motion npz 用 matplotlib 画 hand-obj distance + obj_z 时间序列，目检 intent window 是否合理
3. 把 csv 与 EXPERIMENT_TRACKER.md 已有的"哪些 case 试过"列表交叉对比，确保有方法标签的 case 数 ≥ 5

## 后续

- E055（Path B）首批验证 case：从 C1 通过的 box023 + bucket005 中选 1 个，做 hand-snap warmstart + IK 投影
- E056（Path C）首批验证 case：从 C-only 类中选 1 个，做 force-closure reward
- 若 C1/C2 失败 → 整个 Path A 假设需要重审
