# E003 Plan: freejoint physics feasibility sweep

日期：2026-05-17

## Context

E002 证明：关闭 `scene_act/contact_guidance/object actuator` 后，当前 E081-style reward/control 在真 freejoint 物体上失败。main `box025_p2` 和 guard `box023_p2` 都能保持较高手部接触比例，但 object trajectory error 变成 `0.7-0.8m` case-window mean；物体主要保持 floor-supported 或倾倒/滑落。

E002 派生 `scene.xml` 中两个物体都使用：

```xml
<inertial pos="0 0 0" mass="5.0" diaginertia="0.15 0.15 0.1" />
```

这对 `box025` 可能还算可接受，但对 `box023` 小箱也同样是 5kg。下一步需要先判断失败是否主要来自物理参数/接触可行性，而不是直接引入虚拟 grasp。

## Claims

| Claim | 最低证据 |
|-------|----------|
| C1 物理参数是主要瓶颈 | 降低 object mass / 提高手-object friction 后，obj mean/max 和 floor contact 显著改善，至少 guard `box023` 恢复接近 E081。 |
| C2 当前 reward/optimizer 是主要瓶颈 | mass/friction 改善后仍然保持 `>0.5m` obj error 或大量 floor contact。 |
| C3 接触增强没有通过腿/地板作弊 | leg interference 保持低；关键帧显示主要由手部接触支撑/移动物体。 |

## Variants

基于 E002 的 true-freejoint 派生 task 再派生四个物理参数 case：

| Variant | Source | 物理改动 | Role |
|---------|--------|----------|------|
| `E003_box025_p2_m1` | `box025_person2_freejoint_legobj` | object mass `5.0 -> 1.0kg`，惯量按比例缩放 | main |
| `E003_box025_p2_m1_f4` | `box025_person2_freejoint_legobj` | mass `1.0kg`；hand-object pair friction `2 -> 4`；object-floor sliding friction `1 -> 0.5` | main |
| `E003_box023_p2_m1` | `box023_person2_freejoint_legobj` | object mass `5.0 -> 1.0kg`，惯量按比例缩放 | guard |
| `E003_box023_p2_m1_f4` | `box023_person2_freejoint_legobj` | mass `1.0kg`；hand-object pair friction `2 -> 4`；object-floor sliding friction `1 -> 0.5` | guard |

## Implementation

新增：

- `workspace/core4d_collab_retarget/scripts/E003/variants.tsv`
- `workspace/core4d_collab_retarget/scripts/E003/create_physics_sweep_cases.py`
- `workspace/core4d_collab_retarget/scripts/E003/generate_e003_overrides.py`
- `workspace/core4d_collab_retarget/scripts/run_E003_preprocess.sh`
- `workspace/core4d_collab_retarget/scripts/train/train_E003.sh`
- `workspace/core4d_collab_retarget/scripts/run_E003_remote.sh`
- `workspace/core4d_collab_retarget/scripts/eval/eval_E003.py`
- `workspace/core4d_collab_retarget/log/03_E003_freejoint_physics_feasibility_sweep_results.md`

E003 eval 复用 E002 freejoint evaluator，并额外对比 E002/E081。

## Remote execution

按 `.codex/skills/experiment-planning-zh/remote-execution.md` 使用 `spider-remote` 双卡：

- GPU0: `E003_box025_p2_m1` -> `E003_box025_p2_m1_f4`
- GPU1: `E003_box023_p2_m1` -> `E003_box023_p2_m1_f4`

启动前必须先处理远程 dirty state。当前远程 `git switch` 后显示已有修改的 dataset/mesh/`uv.lock` 文件；不能直接假设远程工作区 clean。

## Success criteria

| 指标 | 解释 |
|------|------|
| guard `box023` obj mean `<=0.25m` and floor contact significantly below E002 | 说明物理参数是 E002 failure 的主要原因，可继续做 true-freejoint 方法开发 |
| both `m1` and `m1_f4` still `>0.5m` obj mean | 说明单机器人 CEM reward/接触建模不足，下一步应转向 virtual grasp/contact constraint 或 dual-agent support |
| leg interference `<=5%` | 防止靠腿/脚错误支撑 |
| keyframes f100/f125/f160 | 必须肉眼确认不是 floor/穿模/腿部假支撑 |
