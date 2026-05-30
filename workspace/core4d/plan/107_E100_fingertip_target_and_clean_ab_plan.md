# E100 — Stage 2: contact target 重做 + 干净 A/B

日期：2026-05-30
分支：`exp/core4d-collab-retarget`
上游：E098（face_utils + replay_gate）/ E099（fingertip vote API + quat audit + audit 报告 Tier 1-3）
下游：E101（full CEM 用新 target）、E102（mining 用 face 兼容性）

## Context

E099 audit 已暴露 9/33 hand palm-vote ≠ fingertip-vote 主面差异（B6 验证）+ 17/17 case obj quat > 30°（必须 disable world-up 投影）。E100 任务是把 E094 / E085 已有的 external contact target NPZ 生成器升级为 fingertip-aware 版本：

- **不动 IK 算法**（继续不改 OmniRetarget）；
- **不动 spider/simulators/mjwp.py**（CEM elite filter / reward 不动）；
- 只动 `workspace/core4d/scripts/E100/build_fingertip_aware_target.py`（新 target 生成器）+ 派生 yaml override（新 target 路径）。

E100 阶段 GPU 紧张（监控显示两张卡 100% util，可能正在跑别的实验）；**plan 已写明 CEM 实际触发可推迟到 E101 Phase 1 联合验证**（同 case 18029_p2 + 035_p2 既验 H1 又验 full 收敛）。本期 commit C1 + C3 offline 部分，C2 干净 A/B 的 config 文件 + run 脚本就绪但 CEM run 视 GPU 状态推迟。

## 约束（沿用 E098/E099）

1. **不动 OmniRetarget**；
2. **不动 spider/simulators/mjwp.py reward / gate / elite filter**（只动 target 输入）；
3. **新 target 必须默认 use_world_up=False**（quat_audit 17/17 都 disable）；
4. **守门反向**：box023_person2 / box025_person2 / box004 WORK 3 case 在 new target 下的 offline gap 变化必须 ≤ 1 cm（不动已 work 的 case）。

## Claims

### C1 — Fingertip-aware target 生成器 + 全 17 case target NPZ

**判据**：
- 提供 `build_fingertip_aware_target.py`：
  - 输入：`workspace/core4d/results/E099/fingertip_vote_per_case/*.json`（vote face）+ `quat_audit.tsv` + spider 仓库 trajectory_kinematic.npz（每帧 obj pose 与 contact_pos）；
  - 逻辑：每帧每只手把 spider FK palm 投到 E099 vote face 上（in-plane 用 palm 当前 xy 投影并 clip 到 face 内 ± half - 1cm 内）；若 fingertip vote face 与 palm vote face 同向，target ≈ palm；若不同向，target 翻到 vote face；
  - 输出：`workspace/core4d/results/E100/fingertip_targets/{case}/spider_contact_target_object_local.npz`（key=`spider_contact_target_object_local`, shape (T, 2, 3)）；
- ≥ 14/17 case 成功生成 NPZ；
- 单元测试 3/3 PASS（box023_p2 target 应在 +z 面上, box026_039 target 应在 -z 面上, 18029_p2 R target 应在 +z 面上）。

### C2 — 干净 A/B 配置就绪（CEM 实际运行可推迟到 E101 Phase 1）

**判据**：
- `examples/config/override/core4d_E100_*_fingertip_target.yaml` 覆盖 2 个典型 case（18029_p2 / box023_p2）+ 2 个 source variant（new fingertip target / 既有 raw E085 target）= 4 override；
- `workspace/core4d/scripts/run_E100_remote.sh` 写好（GPU0/GPU1 双卡分配，3 seed × 4 variant = 12 run，24-step mini CEM）；
- **CEM 实际触发推迟到 E101 Phase 1**：因 GPU 100% util，且 E101 Phase 1 同样跑 18029_p2 + 035_p2 的 full CEM，把 24-step mini 当作 E101 的"first 24 steps"前向输出即可（同 case 同 config，节省一轮 GPU 时间）；
- log 里明示 mitigation 路径 + GPU 排队监控。

### C3 — 全 17 case target gap audit + 守门反向

**判据**：
- 输出 `workspace/core4d/results/E100/target_gap_summary.tsv`：每 case 每只手 `palm_to_new_target_mean_m / max_m`（按 raw mocap fingertip 中心为参照算 palm vs new target gap）；
- 守门 case (box023_person2 / box025_person2 / box004 三 WORK case) 的 `palm_to_new_target` 与 `palm_to_palm`（即 palm 不动）差异 ≤ 1 cm；
- Tier 1 case (E099 列出的 9 个 DIFFER hand) 出对比 PNG 4 张/case（raw vs new target 在 obj local frame 的投影散点叠加 box wireframe）。

## 改动文件

### spider 主仓库（本期）

| 类别 | 文件 | 改动 |
|---|---|---|
| 新增 | `workspace/core4d/plan/107_E100_fingertip_target_and_clean_ab_plan.md` | 本计划 |
| 新增 | `workspace/core4d/scripts/E100/build_fingertip_aware_target.py` | target 生成器 |
| 新增 | `workspace/core4d/scripts/E100/audit_target_gap.py` | 全 17 case gap audit |
| 新增 | `workspace/core4d/scripts/E100/render_target_compare.py` | Tier 1 case 对比 PNG |
| 新增 | `workspace/core4d/scripts/E100/test_build_fingertip_aware_target.py` | 单测 3/3 |
| 新增 | `workspace/core4d/scripts/E100/run_all_E100.sh` | 一键运行 (offline 部分) |
| 新增 | `workspace/core4d/scripts/run_E100_remote.sh` | CEM A/B 远程双卡脚本 |
| 新增 | `examples/config/override/core4d_E100_d003_box021_20231018_029_p2_fingertip.yaml` | 18029_p2 new target |
| 新增 | `examples/config/override/core4d_E100_box023_person2_fingertip.yaml` | box023_p2 new target |
| 新增 | `workspace/core4d/results/E100/fingertip_targets/{case}/spider_contact_target_object_local.npz` | 14-17 个 NPZ |
| 新增 | `workspace/core4d/results/E100/target_gap_summary.tsv` | C3 |
| 新增 | `workspace/core4d/results/E100/visuals/target_compare/*.png` | Tier 1 case 对比 |
| 新增 | `workspace/core4d/log/124_E100_*_results.md` | 实验日志 |

### holosoma 仓库

E100 阶段**无 holosoma 改动**（不动 OmniRetarget IK）。

### 不动的文件

- `spider/simulators/mjwp.py` / `examples/run_mjwp.py`（已经支持 `contact_hdmi_target_source=external`，本期无需改）；
- `spider/process_datasets/core4d.py`（B4 deprecation comment E098 已加）；
- `workspace/core4d_collab_retarget/*`（B1-B3 E098 已修）。

## 流程

1. **P1**：写本 plan ✅（当前步骤）
2. **P2**：写 build_fingertip_aware_target.py + 单测 + 全 17 case 跑 NPZ
3. **P3**：写 audit_target_gap.py + render_target_compare.py + 跑全 17 case + Tier 1 case PNG
4. **P4**：写 4 个 yaml override + run_E100_remote.sh；GPU 空闲时再触发实际 CEM 或推迟到 E101
5. **P5**：写 log + EXPERIMENT_TRACKER + commit + push

## 验证命令

```bash
# C1 单测
.venv/bin/python workspace/core4d/scripts/E100/test_build_fingertip_aware_target.py

# C1 全 17 case 跑
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 .venv/bin/python workspace/core4d/scripts/E100/build_fingertip_aware_target.py \
  --manifest workspace/core4d/scripts/E098/historical_case_manifest.tsv \
  --fingertip-dir workspace/core4d/results/E099/fingertip_vote_per_case \
  --out workspace/core4d/results/E100/fingertip_targets

# C3 gap audit
.venv/bin/python workspace/core4d/scripts/E100/audit_target_gap.py \
  --targets workspace/core4d/results/E100/fingertip_targets \
  --out workspace/core4d/results/E100/target_gap_summary.tsv

# C3 Tier 1 PNG
.venv/bin/python workspace/core4d/scripts/E100/render_target_compare.py \
  --tier1 029_p2 030_p1 020_p2 028_p2 box021_person1 box023_person1 box023_person2 box004_083_p2 box004_083_p1 box004_082_p1

# C2 (CEM, GPU 空闲时跑或推迟到 E101)
bash workspace/core4d/scripts/run_E100_remote.sh
```

## 风险

| 风险 | 应对 |
|---|---|
| GPU 100% util，CEM 跑不了 | C2 mitigation：把 24-step mini CEM 合并到 E101 Phase 1 的 full CEM（同 case 同 config，前 24 step 等于 mini）。本期 C2 只完成 config + 脚本 ready |
| 守门 case (box023/025/004) 反退化 | C3 反向硬阻塞，反退化即停 (gap 变化 > 1 cm)，回 helper 调试；若不可解，记录 Tier 3 守门 case 保留 raw target，不替换 |
| spider FK palm 与 raw mocap fingertip 的 obj pose 不同时基 | target NPZ 用 spider 仓库 dt 时基 (与 trajectory_kinematic.npz 一致)；vote face 用 raw mocap 但是 per-case 常数（不需要逐帧对齐）|
| target 的 in-plane 坐标如果 clip 太狠会导致 reward 远离真实接触 | clip 内边距设 1 cm，留余地；audit 报告里观察 in-plane gap 分布 |
| Tier 1 中 028_p2 / box023_p1 是 IK 过拟合 case (palm 说 contact, fingertip 说 no_contact) | 这种 case 的 R target 应当置为 NaN 或保留 palm 位置但 active=False；E101 CEM elite filter 可读 active 列；本期 audit 报告里专门标记 |

## 下游影响

| 下游 | 依赖 | 阻塞性 |
|---|---|---|
| E101 Phase 1 (18029_p2 + 035_p2 full CEM) | E100 yaml override + fingertip target NPZ | 硬阻塞 |
| E101 Phase 2 (box021 D003 剩余 4 case + box026 全 case + 守门) | E100 全 17 case NPZ | 硬阻塞 |
| E101 失败归因视频 | E100 visuals/target_compare 提供 G1 rollout vs target 对比 | 软阻塞 |
| E102 mining target 生成 | build_fingertip_aware_target.py（同一脚本能跑新候选）| 软阻塞 |

## 时间预算

- P1: 0.5 h（已用）
- P2: 2 h（写 + 单测 + 全 17 case）
- P3: 1.5 h（audit + 10 case PNG）
- P4: 1 h（config + run 脚本）
- P5: 1 h（log + commit）
- **本期合计：~6 h（不含 CEM 实际运行）**
- CEM 实际运行：约 2-4 h（推迟到 E101）
