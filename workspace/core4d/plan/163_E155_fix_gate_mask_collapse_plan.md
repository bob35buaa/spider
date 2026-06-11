# E155 — ~~修复 hand_support_gate mask 坍缩 bug~~ → 诊断推翻：gate 正确，无需重跑

## 0. 背景与动机 + Stage0 诊断结论

### 初始假设（已推翻）

E154 初步分析认为 `_sample_gate_from_ref_mask` 把 mask 坍缩为 1.0 → gate 失效。

### Stage0 诊断结论

通过完整的代码链路追踪 + 数值模拟，**推翻了 bug 假设**：

1. mask 正确加载（`eval_contact_mask_3cm` → resize → `approach_mask_t` shape `(T_padded, 2)`）
2. `get_slice` 按时间切片 → `sampling.py L235: ref=[r[t] for r in ref_slice]` → shape `(2,)`
3. `_sample_gate_from_ref_mask` 收到 `(2,)` → `max(L,R)` → 放手段 `(0,0)` → gate=0 ✅
4. **gate 在运行时正确关闭**

### release_false 高的真正原因

**行为惯性**：b1 (`hand_support_rew` scale=3.0) 在搬运段把手"压"在物体表面。
放手段 reward 归零后，qpos_rew (scale=1.0) 力度不够快拉手离开 → 前几帧仍有物理接触。
这不是 code bug，是 reward 强度设计的副作用。

### 决策

**无需重跑 12 个实验**。E153 最优配置 tracking 3/3 结论不受影响。
release_false 作为诊断指标保留，不进门控（E154 已确认）。

如果后续想降低 release_false，可选方向：放手段加 release reward / 降 b1 scale / mask ramp-down。
但当前优先级是把 tracking-pass 的轨迹送入 Holosoma RL。

## 1. Claims（事前定义，可证伪）

- **C1（bug 修复正确性）**：修复后的 `_sample_gate_from_ref_mask` 在放手段返回 gate=0
  （通过单元测试验证：输入 `mask[t]=(0,0)` 时 gate=0，`mask[t]=(1,1)` 时 gate=1）。
- **C2（release_false 下降）**：修复后 b1 系列的 `release_false` 从 0.24-0.75 降至 ≤0.10
  （与 baseline 的 release_false≈0 趋势一致）。
- **C3（搬运段接触保持）**：修复后 b1 系列在搬运段（mask=1）的 `inmaskC` 不显著低于修复前
  （Δ ≤ −0.10，允许小幅下降因为放手段"拽回来"的假接触消失了）。
- **C4（tracking 仍 pass）**：E153 最优配置 `(-0.010, 0.10)` 修复后仍 3/3 tracking pass
  （pz_term_worst < 0.08）。

## 2. 改动

### 2.1 修复 `_sample_gate_from_ref_mask`

文件: `spider/simulators/mjwp.py`

**当前问题逻辑** (L183-188):
```python
if mask.ndim == 2:
    if mask.shape[0] == num_samples:
        return mask.max(dim=1).values
    if mask.shape[1] == num_samples:
        return mask.max(dim=0).values
    return mask[0].max().view(1).expand(num_samples)  # ← BUG: fallback 坍缩
```

**修复逻辑**: 当 mask shape 为 `(horizon, n_eef)`（既不是 `(N,...)` 也不是 `(...,N)`）时，
取 `mask[0]`（当前仿真步）的 per-eef max 作为该步 gate（任一手在接触窗口内 → gate=1）。
这与 `contact_hdmi_rew`（L1131-1133）的行为一致。

```python
if mask.ndim == 2:
    if mask.shape[0] == num_samples:
        return mask.max(dim=1).values
    if mask.shape[1] == num_samples:
        return mask.max(dim=0).values
    # (horizon, n_eef) — take current timestep [0], max across eef
    return mask[0].max().view(1).expand(num_samples)
```

等等——这和现在一样！问题是**语义正确的**：`mask[0]` 经过 `get_slice` 后就是**当前步**的值。

让我重新确认 bug 是否在 `support_gate` 内的 `hold_contact_rew` 分支（L1210-1224），
那里直接访问 `approach_mask_val` 而**不经过 `_sample_gate_from_ref_mask`**：

实际 bug 位于 `hold_contact_rew` 内（L1210-1224）：
```python
if config.hold_contact_require_ref_contact:
    ref_gate = approach_mask_val  # (horizon, 2) 整个 horizon 片段
    ...
    if ref_gate.ndim == 2 and ref_gate.shape[1] == hand_pos.shape[1]:
        ref_gate = ref_gate[0].max().expand_as(time_gate)  # 取 horizon[0] max
```

以及 `support_gate()` 内调用 `_sample_gate_from_ref_mask(approach_mask_val, N, ...)` 时，
`approach_mask_val` 是整个 horizon 切片 `(horizon, 2)` → 走 fallback → `mask[0].max()`。

**关键问题**: `mask[0]` 是 horizon 的第一帧 = 当前仿真步。如果当前步在搬运段（mask=1），
gate=1 是正确的；如果当前步在放手段（mask=0），gate=0 也是正确的。

**那 bug 到底在哪？** → 需要验证 `approach_mask_val` 在放手段时 `mask[0]` 是否确实 =0。
如果 get_slice 正确，mask[0] 应该 = 当前步的值。可能的 bug 是 mask 的**时间轴对齐**问题
（mask length vs qpos_ref length），或者是 horizon repeat padding 让放手段全被 1 填满了。

**→ 需要在 Stage0 先做诊断性插桩 log，确认 gate 在运行时的实际值。**

### 2.2 Mask 后处理: max(L,R) 统一

新增简单后处理：加载 3cm mask npz 后，per-frame 取 `max(L, R)` 作为搬运统一 gate。

位置: 在 `run_mjwp.py` L906 `per_eef_mask_np = raw_mask[:, person_idx, :]` 后追加：
```python
if config.contact_hdmi_mask_carry_union:
    # 搬运任务: 任一手在接触窗口 → 两手都视为接触期
    union = per_eef_mask_np.max(axis=1, keepdims=True)  # (T, 1)
    per_eef_mask_np = np.broadcast_to(union, per_eef_mask_np.shape).copy()
```

Config 新增: `contact_hdmi_mask_carry_union: bool = False`（默认关，只在搬运 case 显式开启）。

### 2.3 Config 新增字段

```python
# config.py
contact_hdmi_mask_carry_union: bool = False  # 搬运任务 mask L/R 取 union
```

## 3. 实验矩阵 (4 方法 × 3 case = 12 runs)

| # | 方法 | gate | b1 reward | 对比 |
|---|------|------|-----------|------|
| 1 | b1_fixed | off | ✅ (gate bug fixed) | vs E151 b1 (gate bugged) |
| 2 | gateA_b1_sdf010_v05 | sdf=-0.010, viol=0.050 | ✅ (fixed) | vs E152 gateA_b1 |
| 3 | gateA_b1_sdf010_v10 | sdf=-0.010, viol=0.100 | ✅ (fixed) | vs E153 最优 |
| 4 | gateA_b1_sdf005_v10 | sdf=-0.005, viol=0.100 | ✅ (fixed) | vs E153 次优 |

3 cases:
- `box021_029_p2` (task: `d003_box021_20231018_029_p2_e107_clean`)
- `box004_083_p2` (task: `e091_box004_20231003_2_083_p2_e092_dyn`)
- `box023_person2` (task: `box023_person2_legobj`)

全部启用 `contact_hdmi_mask_carry_union: true`。

## 4. GPU 分配与并行

每个 case 4 个实验串行 → 1 case ≈ 4 × 6min = 24min。

| GPU | case | 预计耗时 |
|-----|------|---------|
| **local-gpu0** | box021_029_p2 (4 runs serial) | ~24 min |
| **remote-gpu0** | box004_083_p2 (4 runs serial) | ~24 min |
| **remote-gpu1** | box023_person2 (4 runs serial) | ~24 min |

## 5. 脚本文件

| 文件 | 用途 |
|------|------|
| `workspace/core4d/scripts/train/train_E155_gate_mask_fix.sh` | 本地 box021 (4 runs) |
| `workspace/core4d/scripts/run_E155_remote.sh` | 远程 box004+box023 (GPU0/GPU1 各 4 runs) |
| `workspace/core4d/scripts/pull_E155_remote_results.sh` | 回收远程产物 |
| `workspace/core4d/scripts/watch_and_pull_E155.sh` | 自动监控+回收 |

## 6. Override 策略

复用 E152 的 override yaml（已继承完整链 E143→E148→E151→E152），
gate 参数和新字段通过 Hydra CLI 覆盖：

```bash
uv run examples/run_mjwp.py \
    +override=core4d_E152_${case}_gateA_b1 \
    cem_hand_gate_enabled=true \
    cem_hand_gate_min_sdf_m=${MIN_SDF} \
    cem_hand_gate_max_violation_pct=${MAX_VIOL} \
    cem_hand_gate_hard_floor_m=-0.020 \
    contact_hdmi_mask_carry_union=true \
    video_output_path="${RESULTS_DIR}/${VARIANT}.mp4"
```

对于 b1_fixed（无 gate）：
```bash
uv run examples/run_mjwp.py \
    +override=core4d_E151_${case}_b1_mesh \
    cem_hand_gate_enabled=false \
    contact_hdmi_mask_carry_union=true \
    video_output_path="${RESULTS_DIR}/${VARIANT}.mp4"
```

## 7. 评估

复用 E154 评测框架（真实 3cm mask + body tracking）：
```bash
python workspace/core4d/scripts/eval/eval_E152_axis1_hand_object_physics_gate.py \
    --manifest workspace/core4d/results/E155/manifest.tsv \
    --output-dir workspace/core4d/results/E155/eval
```

**关键指标**:
- `release_false` (C2): 期望 ≤0.10
- `inmaskC` (C3): 期望 ≥0.70
- `track_pelvis_z_err_terminal_m` (C4): 期望 worst < 0.08
- `success_tracked` (综合): 期望 3/3

## 8. 结果路径

| 类型 | 路径 |
|------|------|
| 计划 | `workspace/core4d/plan/163_E155_fix_gate_mask_collapse_plan.md` |
| 结果 NPZ/MP4 | `workspace/core4d/results/E155/cem/full/` |
| 评测输出 | `workspace/core4d/results/E155/eval/` |
| 远程日志 | `logs/E155/` |

## 9. 执行顺序

```
Stage0: 诊断 — 确认 mask 在放手段运行时的 gate 实际值
  → 插桩 log 或 unit test
  → 定位确切的坍缩位置（是 _sample_gate_from_ref_mask 还是时间轴对齐问题）
  → verify: 放手段 gate ≠ 0 (bug 复现)

Stage1: 修复
  → 修 _sample_gate_from_ref_mask 或修时间轴对齐
  → 新增 contact_hdmi_mask_carry_union config
  → verify: 单元测试 pass + smoke run 放手段 gate=0

Stage2: 跑实验
  → 本地 box021 + 远程 box004/box023
  → verify: 12 NPZ + 12 MP4 产出

Stage3: 评估
  → E154 框架重评
  → verify: C1-C4 逐条判定

Stage4: 记录
  → 更新 log/、EXPERIMENT_TRACKER.md、progress.md
```

## 10. 风险与回退

| 风险 | 应对 |
|------|------|
| 修复后 inmaskC 大幅下降（放手段不再拽回来，搬运段也变弱） | 先只跑 1 case smoke 确认趋势再全量 |
| mask 时间轴对齐问题（horizon repeat padding 填充了 1） | 需要确认 contact_ref_interp 的 tail padding 逻辑 |
| 修复后 gateA_b1 fallback 升高（gate 更严了） | 可微调 hard_floor 或 max_viol |
| box004 起身在 0.05~0.08 边界（单 seed 不稳定） | 多 seed 复核留作后续 |

---

*计划撰写: 2026-06-11*
