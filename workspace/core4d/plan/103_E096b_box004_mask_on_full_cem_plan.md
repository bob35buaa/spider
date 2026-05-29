# E096b Plan: box004 mask-on full CEM rerun

日期：2026-05-29

## Context

E096 在两条 preprocess-ready box004 case 上得到 full CEM `WORK`：

- `e091_box004_20231003_2_083_p1`
- `e091_box004_20231003_2_082_p1`

但复核配置后发现：E096 为避免继承 `core4d_E089A_box021_person1_upperobj` 的旧 box021 mask，把 `contact_hdmi_mask_source/path` 清空了。实际运行没有接入 CORE4D raw 3cm contact mask，而是回退到 `run_mjwp.py` 的 rotated-SDF per-EEF mask。

本实验按用户要求命名为 **E096b**，只补上正确 `core4d_3cm` mask 后重跑 full CEM。P3 `e091_box004_20231003_2_082_p2` 仍然没有 SPIDER trajectory，保持 preprocess blocked，不纳入本次 rerun。

## Claims

| Claim | 验证方式 |
|---|---|
| C1: E096b 只改变 mask source，不改变 CEM/safety/scene 主配置 | override/config_act diff：同 E096 使用 `core4d_E089A` base、ref_fk wrist5cm、E083-style leg+upper object pairs、full CEM 32 iter |
| C2: P1/P2 都实际加载 CORE4D 3cm per-EEF mask | CEM log 出现 `E078 core4d_3cm per-EEF mask`，active L/R 与 mask NPZ 统计一致 |
| C3: P1/P2 mask-on full CEM 仍能达到 `WORK` 或暴露差异 | full eval summary：object/contact/pelvis/head/upper/hand-floor |
| C4: 并行执行不干扰已有 RL | 本地 GPU0 + remote GPU0 叠加运行，不 kill/stop 其他进程 |
| C5: 可视化可支撑结论 | full CEM MP4/keyframes、frame sheet、high subagent review |

## Cases

| id | source task | split | mask |
|---|---|---|---|
| P1 | `e091_box004_20231003_2_083_p1` | local GPU0 | `.../contact_masks/e091_box004_20231003_2_083_p1/raw_contact_mask_3cm.npz` |
| P2 | `e091_box004_20231003_2_082_p1` | remote GPU0 | `.../contact_masks/e091_box004_20231003_2_082_p1/raw_contact_mask_3cm.npz` |

## Execution Plan

1. Create E096b builder and scripts:
   - `workspace/core4d/scripts/E096b/build_mask_on_cem_tasks.py`
   - `workspace/core4d/scripts/train/train_E096b_mask_cem.sh`
   - `workspace/core4d/scripts/run_E096b_remote.sh`
   - `workspace/core4d/scripts/pull_E096b_remote_results.sh`
2. Build derived tasks:
   - Copy P1/P2 source tasks to `*_e096b_mask_cem`.
   - Add the same E083-style leg/foot-object and upper-body-object contact pairs as E096.
   - Copy raw 3cm mask NPZ/audit JSON into `workspace/core4d/results/E096b/contact_masks/` so local and remote use repo-relative paths.
   - Generate overrides with `contact_hdmi_mask_source=core4d_3cm`.
3. Static checks and commit/push scaffold.
4. Run full CEM:
   - local GPU0: P1.
   - remote GPU0: P2.
   - no remote GPU1 job for E096b.
5. Pull remote result and run unified eval.
6. Generate frame sheets and request high subagent visual review.
7. Write E096b log, update tracker/progress, commit/push.

## Success Criteria

| Item | Standard |
|---|---|
| Mask loading | logs show `E078 core4d_3cm per-EEF mask` for both P1/P2 |
| CEM output | both variants have `trajectory_mjwp_act.npz`, MP4, keyframes, eval summary |
| Comparison | log compares E096b against E096 full CEM metrics |
| Visualization | frame sheets and high subagent review saved |
| Reproducibility | mask copies, overrides, variants, scripts, active scenes are committed or force-added |

## Stop Rules

- Do not kill existing RL jobs.
- Do not rerun P3 unless upstream OmniRetarget/preprocess trajectory exists.
- Do not advance any non-WORK sequence to Holosoma RL.
