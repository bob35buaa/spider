# E096 Plan: box004 three-case contact semantics + full CEM

日期：2026-05-29

## Context

E095 从 `data_construction_v2` 选出三条 first-batch worklike case（历史文件名仍叫 box004 priority）：

- `e091_box004_20231003_2_083_p1`
- `e091_box004_20231003_2_082_p1`
- `e091_box004_20231003_2_082_p2`

其中 `083_p1` 和 `082_p1` 已通过 no-fingertip Stage2b / SPIDER verify / D005b；`082_p2` 在 OmniRetarget 阶段报 `RuntimeError: CVXPY solve failed: infeasible`，目前没有 retargeted / trimmed / SPIDER trajectory。

用户要求把这三条当作新实验：

1. 做接触语义分析：E093/E094 口径，即 raw contact、`wrist_yaw_link + 5cm`、support/inside、handbox/proxy。
2. 都跑 full CEM。
3. 三卡并行：local GPU0 + remote GPU0/GPU1；现有 RL 进程不 kill，CEM 叠加运行。
4. 强制可视化，并用 high subagent 做可视化分析。

## Claims

| Claim | 验证方式 |
|---|---|
| C1: 新 box004 p1 cases 的 contact semantics 与 known WORK `083_p2` 同 pattern | E096 contact geometry summary：`wrist5->raw`、support、inside、projection delta 与 E093/E094 box004 guard 对比 |
| C2: `083_p1` 和 `082_p1` 可完成 SPIDER full CEM | full CEM NPZ/MP4/eval summary；WORK/PASS/FAIL 由 E094/E092 full gate 判定 |
| C3: `082_p2` 的 full CEM 可行性由 preprocess 轨迹决定 | 先重试/核查 retarget；若仍无 `trajectory_kinematic.npz`，记录为 preprocess blocker，而不是伪造 CEM |
| C4: 三卡并行不干扰现有 RL 进程 | 启动前记录 `nvidia-smi`；脚本只启动新增 CEM，不 kill/stop 任何进程 |
| C5: 可视化足够支持结论 | contact semantic MuJoCo videos、CEM autocam videos/keyframes、high subagent review |

## Case Manifest

| case | source task | person_idx | split | current readiness |
|---|---|---:|---|---|
| P1 | `e091_box004_20231003_2_083_p1` | 0 | local GPU0 | ready: scene/trajectory/mask exist |
| P2 | `e091_box004_20231003_2_082_p1` | 0 | remote GPU0 | ready: scene/trajectory/mask exist |
| P3 | `e091_box004_20231003_2_082_p2` | 1 | remote GPU1 | not ready: OmniRetarget infeasible, no SPIDER trajectory |

## Execution Plan

1. Create E096 manifest and scripts:
   - `workspace/core4d/scripts/E096/build_contact_manifest.py`
   - `workspace/core4d/scripts/E096/build_cem_tasks.py`
   - `workspace/core4d/scripts/train/train_E096_box004_cem.sh`
   - `workspace/core4d/scripts/run_E096_remote.sh`
   - `workspace/core4d/scripts/pull_E096_remote_results.sh`
   - `workspace/core4d/scripts/eval/eval_E096_cem.py`
2. Run contact semantic analysis:
   - E093 audit with E096 manifest.
   - E094 `adaptive_support` projection with E096 ready cases.
   - MuJoCo full-body videos for contact markers and projection markers.
3. Build CEM tasks:
   - Add E083 upper-body + leg/foot object collision pairs.
   - Use base no-fingertip `ref_fk + wrist5cm` target for the first full CEM pass; projection analysis is diagnostic, not automatically applied unless the gate shows a clear need.
4. Run full CEM:
   - Local GPU0: P1.
   - Remote GPU0: P2.
   - Remote GPU1: P3 only if preprocess trajectory exists; otherwise P3 records `preprocess_blocked`.
   - Do not kill existing RL processes; CEM is launched in new shell/tmux sessions.
5. Pull remote results and run unified eval.
6. Re-render corrected autocam videos if needed; extract keyframes/contact sheets.
7. Spawn high subagent to review contact semantic videos + CEM videos.
8. Write E096 log, update tracker/progress, commit/push.

## Success Criteria

| Item | Standard |
|---|---|
| Contact semantics | `geometry_summary`, `projection_summary`, object-local/timeline/dashboard PNGs, MuJoCo videos exist and nonblank |
| CEM execution | Every ready case has full CEM `trajectory_mjwp_act.npz`, MP4, eval summary |
| P3 handling | If no trajectory exists, log exact preprocess blocker with path and stack evidence |
| Visual review | high subagent report saved under `workspace/core4d/results/E096/.../visual_review/` |
| Reproducibility | active scenes and E096 `scene_snapshot/` saved; scripts pass static checks |

## Stop Rules

- Do not rerun or kill any existing RL jobs.
- Do not claim P3 full CEM if no SPIDER trajectory exists.
- Do not advance any full-CEM `FAIL` or `preprocess_blocked` sequence to Holosoma RL.
