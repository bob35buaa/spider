# E101 — Stage 3: box021 D003 + Box026 + box004 full CEM 重跑 + 失败模式重新分类

日期：2026-05-30
分支：`exp/core4d-collab-retarget`
上游：E098 (face_utils + replay_gate) / E099 (fingertip vote + quat audit) / E100 (fingertip-aware target NPZ + yaml override)
下游：E102 (新 WORK case 进 RL-ready set)

## Context

整体计划 Stage 3 的目标：在 E098 新 gate + E100 新 fingertip target 下重跑历史 FAIL case，验证 H1 (target source 错位) 是否是 box021 D003 失败的主导原因；如果 typical case 转 WORK → 推广 Phase 2；如果 typical case 全 FAIL → stop-loss + 失败模式重新分类。

**Phase 0 (排查 torch.compile 环境)**：已在 E100 smoke test 中完成 ✅。修复方法：CLI 加 `+use_torch_compile=false`（E085 / E094 历史 train script 已用此参数，本来就是 spider 仓库的实践默认）。Smoke test 用 box023_p2 fingertip target + max_num_iterations=2 + num_samples=128 在 209s 内成功完成（target NPZ 加载 OK，trajectory NPZ + mp4 输出 OK）。

## 约束

1. **不动 OmniRetarget IK**（继续保持）；
2. **不动 spider/simulators/mjwp.py**（reward/gate/elite filter 不动）；
3. **GPU 紧张**：Phase 1 双卡 2 case × 2 seed = 4 run（不是 3 seed，节省时间）；Phase 2 视 Phase 1 结果决定是否启动；
4. **Phase 1 stop-loss**：≥1 case 达成 gate 全 PASS + pelvis_min ≥ 0.55m + contact ≥ 50% → 启动 Phase 2；否则 stop, log 中分析根因再与用户对齐；
5. **守门反向**：box004 三个已 WORK case + box023/025 守门 case 在新 target 下不得退化（contact frac Δ ≤ 5pp、pelvis_min Δ ≤ 0.02m）。

## Claims

### C1 — Phase 0 (torch.compile 环境)

**判据**：CLI `+use_torch_compile=false` 让 spider/run_mjwp.py 跑通 fingertip target；smoke test exit 0 + video + traj NPZ 生成 + target NPZ 加载日志清晰。

**状态**：✅ 已完成 (E100 smoke test 验证)。本 Claim 不需额外动作。

### C2 — Phase 1 typical 2 case 双卡 full CEM (4 run)

**判据**：
- 2 case：`d003_box021_20231018_029_p2` (H1 主战场) + `d003_box021_20231011_035_p2` (E090 S1 full FAIL 案，pelvis 0.134m)
- 配置：base = `core4d_E100_*_fingertip` (E100 yaml override)，full CEM default (`max_num_iterations=32, num_samples=1024`)，2 seed (0, 1)，双卡 GPU0/GPU1 各 1 case 2 seed 串行
- 至少 1/2 case 达成：
  - pelvis_min ≥ 0.55m (E098 replay_gate)
  - pelvis_tilt_end ≤ 75° (E098 replay_gate)
  - lie_on_box_frac ≤ 0.30 (E098 replay_gate)
  - head/upper/floor pen 全 ≤ 5%
  - contact frac ≥ 50%
  - 视频签收：robot 双手贴 box 顶面 / 投票面 + pelvis 站直 + 物体被举起
- 若 1/2 → Phase 2 启动；若 0/2 → stop-loss，进 Phase 3 失败归因
- 若 2/2 → 同上 + 在 log 中标记 H1 主导假设成立

### C3 — Phase 2 推广（视 Phase 1 结果，硬阻塞由 C2 决定）

**条件性判据**（仅 Phase 1 ≥ 1/2 时启动）：
- 推广到 box021 D003 剩余 4 case (20019_p1 / 030_p1 / 020_p2 / 028_p2) + Box026 全 2 case (039_p2 / 135_p2) + 2 守门 (box023_p2 / box025_p2) + box004 三 WORK case 对照 = 共 11 case
- 每 case 1 seed (节省时间)，双卡 GPU0/GPU1 串行
- 输出 `results/E101/cem_outcome_matrix.tsv`：每 case 输出 contact/obj_err/pelvis_min/pelvis_tilt_end/head/upper/hand_floor/lie_on_box/new_gate_pass
- 与 E082-E088 / E089A / E090 / E094 同口径对比表 commit；至少 1 个 D003 case 在 Phase 1+2 合计转 WORK；box004 守门 3 case 不退化

### C4 — Phase 3 失败模式重新分类

**判据**（无论 Phase 1/2 结果都做）：
- 所有 FAIL case 各出 1 段 mp4（关键帧叠加 replay_gate 触发标签）
- 归类到 {motion-level H2 binding / pelvis collapse residual / lie-on-box / reward hacking residual / IK 过拟合 (来自 E099 audit) / other}
- 每类至少 1 case 的代表性视频 + 单帧叠加图
- 不允许 "原因不明"

## 改动文件

### spider 主仓库（本期）

| 类别 | 文件 | 改动 |
|---|---|---|
| 新增 | `workspace/core4d/plan/108_E101_*.md` | 本计划 |
| 新增 | `examples/config/override/core4d_E101_d003_box021_20231011_035_p2_fingertip.yaml` | 035_p2 fingertip target override |
| 新增 | `examples/config/override/core4d_E101_d003_box021_20231011_035_p2_palmbase.yaml` | 035_p2 baseline (E090 S1 既有 target，若 NPZ 缺则 ref_fk) |
| 新增 | `workspace/core4d/scripts/train/train_E101_phase1.sh` | Phase 1 双卡 4 run |
| 新增 | `workspace/core4d/scripts/E101/replay_gate_cem_results.py` | 跑 E098 replay_gate 对 CEM output |
| 新增 | `workspace/core4d/scripts/E101/build_phase2_overrides.py` | Phase 2 yaml 自动派生（仅 Phase 1 通过时调用）|
| 新增 | `workspace/core4d/scripts/run_E101_remote.sh` | 远程双卡（备用，若本机 GPU 占用切远程）|
| 新增 | `workspace/core4d/results/E101/phase1/{variant}_outdir/` | Phase 1 CEM 产出 |
| 新增 | `workspace/core4d/results/E101/cem_outcome_matrix.tsv` | C3 |
| 新增 | `workspace/core4d/results/E101/visuals/` | C4 视频 + 帧 |
| 新增 | `workspace/core4d/log/125_E101_*.md` | 本日志 |

### holosoma 仓库

E101 阶段**无 holosoma 改动**（不动 IK 算法 + 不动 RL framework）。E102 RL-ready handoff 才会动 holosoma side。

## 流程

1. **P1**：写本 plan ✅
2. **P2**：派生 035_p2 yaml override（E100 fingertip + baseline）
3. **P3**：写 train_E101_phase1.sh + 启动 Phase 1（GPU0=18029_p2, GPU1=035_p2，各 2 seed = 4 run）
4. **P4**：等待 Phase 1 完成 → 跑 replay_gate + 视觉签收 → 决定 Phase 2 / stop-loss
5. **P5**：(条件性) Phase 2 推广 + Phase 3 失败归因
6. **P6**：写 log + commit + push

## 验证命令

```bash
# Phase 1: 双卡 4 run
bash workspace/core4d/scripts/train/train_E101_phase1.sh

# Phase 1 eval (replay_gate + 指标汇总)
.venv/bin/python workspace/core4d/scripts/E101/replay_gate_cem_results.py \
  --phase1-dir workspace/core4d/results/E101/phase1 \
  --out workspace/core4d/results/E101/phase1_gate_summary.tsv
```

## 风险 / 应对

| 风险 | 应对 |
|---|---|
| GPU 突然被抢占 | Phase 1 用 nohup + tee log；如中断，从该 seed 重启；最多 4 run，重试成本可接受 |
| Phase 1 typical 2 case 全 FAIL | stop-loss 不投 Phase 2 GPU；在 log 里分析（H2 binding？pelvis 失败？lie-on-box？），与用户对齐再决定 dual-G1 / 改 reward 方向 |
| Phase 1 视频生成失败 (rerun/mujoco viewer issue) | spider 默认会出 visualization_mjwp_act.mp4（已在 smoke test 验证）；如失败用 trajectory NPZ 离线渲染 |
| 18029_p2 fingertip target 让 IK 过拟合的 R hand 出现奇怪行为 | E100 audit 已记录 18029_p2 R fingertip vote = +z 与 palm vote 同向（face_changed=False），target 不变；这点不阻塞 |
| 035_p2 fingertip target 与 E085 历史 raw target 差异大 | 11035_p2 L: +x = palm +x，R: -x = palm -x，face_changed_L/R 全 False；target 与原 palm-based 一致 |

实际上 18029_p2 + 11035_p2 两个 case 在 E100 audit 里 face_changed_L/R **全 False**。这意味着 fingertip vote face 与 palm vote face 完全一致；E100 fingertip target 等效于 palm-based target。所以**严格意义上 Phase 1 不会因 face 翻转受益**——这两个 case 的失败原因不在 face vote bug 上，而在其它（H2 binding / pelvis collapse / IK 过拟合）。

**重新解读 Phase 1**：
- 在 18029_p2 / 11035_p2 上跑 fingertip target 等于复现 E085 / E090 历史结果（target 没变化）。
- 若 Phase 1 仍 FAIL，证实 H1 (target source) **不是** 主导原因，进 Phase 3 归因到 H2 / pelvis / reward。
- 这是有价值的负向结果：把"E082-E088 失败是因为 target 错"这条假设干净排除。

**Phase 1 真正能受益的 case** 是 face_changed=True 的：030_p1 / 20020_p2 / 028_p2 / box021_person1 / box023_p1 / box023_p2 / box004 三 case。如果想验"face 翻转的实际 reward 影响"，应该把 typical case 改为 18030_p1 + box004_083_p2（或 box023_person2 加 RL flag）。

**决策**：**Phase 1 typical case 改为 `d003_box021_20231018_030_p1` (face_changed_R, E101_phase2 中本来排第二) + `e091_box004_20231003_2_083_p2` (box004 R face changed, 守门兼新 target test)**。这样 Phase 1 真正能验"face 翻转对 reward 是否有正向 impact"。

如果 Phase 1 通过，再加 18029_p2 / 035_p2 跑控制实验（H1 排除）。

## Phase 1 final 决定

- **GPU0**: `d003_box021_20231018_030_p1` (E101_phase2 typical 1; face_changed_R: +z→+x)
- **GPU1**: `e091_box004_20231003_2_083_p2` (box004 守门 + face_changed_R: -z→-x)
- 各 2 seed (seed=0, seed=1)，full CEM default 参数
- 预计时间 ~1-2h per case per seed × 2 seed = ~2-4h per case；双卡并行 ~2-4h total

## 时间预算

- P1: 0.5 h
- P2: 0.3 h
- P3: 4 h (双卡 Phase 1 含等待)
- P4: 1 h (eval + 视觉)
- P5: (条件) 8-12 h Phase 2 + 2-3 h Phase 3
- P6: 1 h log + commit

**本期目标**：完成 Phase 0/1，最低限度 commit Phase 1 结果 (无论 PASS/FAIL)，Phase 2/3 视 Phase 1 结果决定。
