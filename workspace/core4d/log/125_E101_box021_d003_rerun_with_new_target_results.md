# E101 Results: box021 D003 + box004 Phase 1 full CEM with E100 fingertip target

日期：2026-05-31
对应 plan：`workspace/core4d/plan/108_E101_box021_d003_rerun_with_new_target_plan.md`
上游：E098 replay gate / E099 fingertip face vote / E100 fingertip-aware target NPZ

## TL;DR

E101 Phase 1 已补齐并触发 stop-loss：**box004 guard 2/2 WORK；box021 D003 0/4 WORK**。

- ✅ Phase 0：`+use_torch_compile=false` 解决 torch.compile/triton 路径，E100/E101 target 均能加载。
- ✅ box004 guard：`e091_box004_083_p2` seed0/1 均 PASS，pelvis_min `0.656/0.660m`，tilt_end `43.0/43.4°`，lie_on_box `0.0`。
- ❌ box021 face_changed 主测：`030_p1` seed0/1 均 FAIL，pelvis 高度不低 (`0.690m`) 但 tilt_end `78.1° > 75°`，最终物体误差 `0.835m`，视频表现为倾斜/未完成搬运。
- ❌ box021 H1 控制：`18029_p2` FAIL_TILT (`tilt_end=96.4°`)；`11035/035_p2` FAIL_PELVIS_LIE (`pelvis_end=0.159m`, `lie=0.541`)。
- 决策：Phase 1 没有任何 D003 转 WORK，按 plan stop-loss **不启动 Phase 2 推广**。结论是 E100 fingertip target/face 修复对 box004 不退化，但**不足以救 box021 D003**；主因转向 motion-level/H2、posture/upright、reward hacking/valid-carry 约束。

## 1. 改动文件

| 类别 | 文件 | 说明 |
|---|---|---|
| 修改 | `workspace/core4d/scripts/E101/replay_gate_cem_results.py` | 支持 `qpos` shape `(T,2,nq)`，取 SPIDER sim channel `[:,0,:]` |
| 修改 | `workspace/core4d/scripts/train/train_E101_phase1.sh` | 加 `030_p1` seed0/1，已有产物自动 skip |
| 新增 | `workspace/core4d/results/E101/phase1_gate_summary.tsv` | Phase 1 replay gate 汇总 |
| 新增 | `workspace/core4d/results/E101/cem_outcome_matrix.tsv` | Phase 1 outcome matrix |
| 新增 | `workspace/core4d/results/E101/visuals/phase1_sheets/*.jpg` | 6 条 rollout 的 1fps sheet |

## 2. Phase 1 结果

| variant | role | gate | pelvis_min | pelvis_end | tilt_end | lie | final obj | 视觉分类 |
|---|---|---|---:|---:|---:|---:|---:|---|
| box004_083_p2 seed0 | guard face_changed | PASS | 0.656 | 0.781 | 43.0 | 0.000 | n/a | standing_hold_box |
| box004_083_p2 seed1 | guard face_changed | PASS | 0.660 | 0.781 | 43.4 | 0.000 | n/a | standing_hold_box |
| box021_030_p1 seed0 | D003 face_changed | FAIL_TILT_OBJECT_MISS | 0.690 | 0.698 | 78.1 | 0.000 | 0.835m | tilted_no_transport |
| box021_030_p1 seed1 | D003 face_changed | FAIL_TILT_OBJECT_MISS | 0.690 | 0.698 | 78.1 | 0.000 | 0.835m | tilted_no_transport |
| box021_11035/035_p2 seed0 | D003 H1 control | FAIL_PELVIS_LIE | 0.154 | 0.159 | 18.6 | 0.541 | n/a | pelvis_collapse_lie_on_box |
| box021_18029_p2 seed0 | D003 H1 control | FAIL_TILT | 0.521 | 0.561 | 96.4 | 0.000 | n/a | upperbody_lean_tilt |

Key files:
- `workspace/core4d/results/E101/phase1_gate_summary.tsv`
- `workspace/core4d/results/E101/cem_outcome_matrix.tsv`
- `workspace/core4d/results/E101/phase1/*.npz`
- `workspace/core4d/results/E101/phase1/*.mp4`
- `workspace/core4d/results/E101/visuals/phase1_sheets/*.jpg`

## 3. Claims

| Claim | 判定 | 证据 |
|---|---|---|
| C1 Phase 0 torch.compile 环境 | PASS | E101 runs 全部使用 `+use_torch_compile=false`，target NPZ 正常加载 |
| C2 Phase 1 typical validation | FAIL for D003 | box021 D003 0/4 WORK；box004 guard 2/2 WORK |
| C3 Phase 2 推广 | SKIP | C2 未满足 “至少 1 个 D003/typical case 达成 gate PASS + valid visual” |
| C4 失败分类 | PASS for Phase 1 | 分类为 `FAIL_TILT_OBJECT_MISS` / `FAIL_TILT` / `FAIL_PELVIS_LIE`，均有 mp4 + sheet |

## 4. 解释

E101 把两个关键分支分开了：

1. **face_changed 对 box004 无害**：box004_083_p2 的 R hand target 从 palm `-z` 换到 fingertip `-x` 后，两个 seed 均保持 WORK。这说明 E100 target 生成器不会破坏已知 positive guard。
2. **face_changed 不能救 box021 D003**：030_p1 是 E100 中真正 face_changed 的 D003 case（R: `+z -> +x`），两个 seed 都失败，且最终物体误差约 `0.835m`。所以 “修 fingertip face/target 就能让 D003 转 WORK” 被当前 Phase 1 否定。
3. **H1 target-source 不是主导充分条件**：18029_p2 / 11035(035)_p2 在 E100 audit 里 face_changed=False，E101 仍失败，失败形态分别是 torso tilt 与 pelvis/lie-on-box。结合 030_p1 负例，box021 D003 的剩余瓶颈更像 posture/upright/motion-level binding，而不是单纯 target 面错。

## 5. 执行备注

- `train_E101_phase1.sh` 初次前台运行被会话中断在 `030_p1 seed0` 中段，未产出完整文件；随后用 `setsid bash -c ...` 分别跑完 `030_p1` seed0/1。
- `030_p1` 两个 seed 全 CEM 耗时约 25-27 分钟/seed。
- `replay_gate_cem_results.py` 原先不能处理 `(T,2,nq)` qpos，已修复；否则会报 broadcast shape error。

## 6. 下一步

不建议直接进入 E101 Phase 2 全量推广。当前最低成本下一步是把 E101 作为 negative stop-loss 收尾，然后进入 E102 的数据扩张/Box022 preflight；若仍要救 box021 D003，应单独开新计划，目标从 contact target 转向 posture/upright gate、motion-level H2 或 dual-G1。
