# E003 Results: freejoint physics feasibility sweep

日期：2026-05-17

## Status

E003 implementation and GPU smoke completed. Full CEM pending.

## Setup

四个 variant 均基于 E002 true-freejoint + leg-object collision task 派生，只修改派生 `scene.xml` 中的 object mass/inertia 和接触 friction：

| Variant | Source | Mass | Hand-object friction | Object-floor friction | Role |
|---------|--------|------|----------------------|-----------------------|------|
| `E003_box025_p2_m1` | `box025_person2_freejoint_legobj` | `1.0kg` | `2.0` | `1.0` | main |
| `E003_box025_p2_m1_f4` | `box025_person2_freejoint_legobj` | `1.0kg` | `4.0` | `0.5` | main |
| `E003_box023_p2_m1` | `box023_person2_freejoint_legobj` | `1.0kg` | `2.0` | `1.0` | guard |
| `E003_box023_p2_m1_f4` | `box023_person2_freejoint_legobj` | `1.0kg` | `4.0` | `0.5` | guard |

## Commands

```bash
bash workspace/core4d_collab_retarget/scripts/run_E003_preprocess.sh
bash workspace/core4d_collab_retarget/scripts/train/train_E003.sh smoke 0
bash workspace/core4d_collab_retarget/scripts/train/train_E003.sh local 0
```

## Results

### Preprocess / config audit

`bash workspace/core4d_collab_retarget/scripts/run_E003_preprocess.sh` 已生成四个派生 task 和四个 override。

| Variant | Scene check | Config check |
|---------|-------------|--------------|
| `E003_box025_p2_m1` | object mass `1kg`, inertia `0.03 0.03 0.02`, hand/object friction `2`, object/floor friction `1` | `contact_guidance=False`, `scene_name=""`, `nq/nv/nu/nq_obj=43/41/29/7`, `ctrl_ref=29` |
| `E003_box025_p2_m1_f4` | object mass `1kg`, inertia `0.03 0.03 0.02`, hand/object friction `4`, object/floor friction `0.5` | `contact_guidance=False`, `scene_name=""`, `nq/nv/nu/nq_obj=43/41/29/7`, `ctrl_ref=29` |
| `E003_box023_p2_m1` | object mass `1kg`, inertia `0.03 0.03 0.02`, hand/object friction `2`, object/floor friction `1` | `contact_guidance=False`, `scene_name=""`, `nq/nv/nu/nq_obj=43/41/29/7`, `ctrl_ref=29` |
| `E003_box023_p2_m1_f4` | object mass `1kg`, inertia `0.03 0.03 0.02`, hand/object friction `4`, object/floor friction `0.5` | `contact_guidance=False`, `scene_name=""`, `nq/nv/nu/nq_obj=43/41/29/7`, `ctrl_ref=29` |

### Smoke

GPU smoke passed:

```bash
bash workspace/core4d_collab_retarget/scripts/train/train_E003.sh smoke 0
.venv/bin/python workspace/core4d_collab_retarget/scripts/eval/eval_E003.py
```

Smoke uses `max_sim_steps=4`, so the table below only verifies execution/eval plumbing.

| Variant | T | nu | nq_obj | obj mean/max | hand contact | leg intf | floor contact |
|---------|---|----|--------|--------------|--------------|----------|---------------|
| `E003_box025_p2_m1` | 4 | 29 | 7 | `0.048/0.061m` | `0.0%` | `0.0%` | `100.0%` |
| `E003_box025_p2_m1_f4` | 4 | 29 | 7 | `0.049/0.062m` | `0.0%` | `0.0%` | `100.0%` |
| `E003_box023_p2_m1` | 4 | 29 | 7 | `0.010/0.013m` | `0.0%` | `0.0%` | `100.0%` |
| `E003_box023_p2_m1_f4` | 4 | 29 | 7 | `0.010/0.013m` | `0.0%` | `0.0%` | `100.0%` |

Full CEM 待运行后填写。

## Claims

| Claim | Status |
|-------|--------|
| C1 物理参数是主要瓶颈 | 待评估 |
| C2 当前 reward/optimizer 是主要瓶颈 | 待评估 |
| C3 接触增强没有通过腿/地板作弊 | 待评估 |

## Next

1. 运行 full CEM 并用 E003 evaluator 对齐 E002/E081 指标。
2. 抽取/检查关键帧 f100/f125/f160。
3. 根据 mass/friction sweep 结果决定是否进入 virtual grasp/contact constraint 或 dual-agent support。
