# E074+ Strategy Progress — 2026-05-14

## 当前状态: Plan 已写入，等待用户审核 E074+ 总路线

## 完成步骤

- [x] 按 `experiment-planning-zh` 恢复 `EXPERIMENT_TRACKER.md`、最新 plan/log、`progress.md`。
- [x] 启用 subagent Halley 复盘 HDMI workflow 与 E071/E073 后剩余差异。
- [x] 启用 subagent Euler 复盘 E037-E067 contact/reward 历史改动。
- [x] 本地复读 log 82-93、E041c/E062/E065-E067 yaml、`run_hdmi.py`/`run_mjwp.py`/`mjwp.py` reward 与优化循环。
- [x] 明确 E071 后结论重置：旧 pre-contact lunge 归因大多被 ctrl mapping bug 污染；当前主失败面是 post-2s hold/contact。
- [x] 写入总计划: `workspace/core4d/plan/79_E074_plus_post_E071_hold_strategy_plan.md`。
- [x] 写入 E074 前置分析日志: `workspace/core4d/log/94_E074_preflight_base_palm_normal_analysis.md`，覆盖 E060-E067 代码影响、E074 base、E062 palm normal 含义和影响。
- [x] 更新 `EXPERIMENT_TRACKER.md`，加入 E074 preflight 索引。
- [x] 写入 E074 实施与远程调度计划: `workspace/core4d/plan/80_E074_remote_execution_plan.md`。
- [x] 修改 `spider/config.py` / `spider/simulators/mjwp.py`，新增默认关闭的 E074A ctrl guard 与 E074C hold contact reward。
- [x] 新增 E074A/E074C override、训练脚本、评估脚本、远程启动与结果回收脚本。
- [x] 验证: `py_compile` 通过；`bash -n` 通过；Hydra compose 确认 E074A/E074C override 生效；`git diff --check` 通过。
- [x] 按 `experiment-planning-zh/remote-execution.md` 修正远程默认配置: `REMOTE_HOST=spider-remote`, `REMOTE_REPO=/home/xiayb/pHRI_workspace/spider`，并补充 tmux capture-pane/结果计数提示。
- [x] 首次远程启动时 SSH 网络超时且旧脚本无 timeout，已终止挂起进程，并给远程启动/回收脚本加入 BatchMode、ConnectTimeout 和 ServerAlive 参数。

## 遇到的错误

| 错误 | 尝试次数 | 解决方案 |
|------|---------|----------|
| 本地沙箱内 `git push` 触发 DNS 失败 | 1 | 用已批准的 escalated `run_E074_remote.sh` 重试，push 显示 up-to-date |
| 远程 SSH 间歇超时，旧启动脚本无 `ConnectTimeout` 导致挂起 | 1 | 终止挂起进程；脚本增加 `BatchMode=yes`、`ConnectTimeout=20`、`ServerAlive*` |

## 核心结论

- E073 是下一步可信 base：`contact_hdmi_target_uses_eef_offset=true` 小幅改善 contact 并消除摔倒，但没有解决 f130/f145 脱手。
- E060-E067 的 task_obj/actuator/body-partition 结论暂不复用；保留代码开关但不纳入第一批。
- E074 第一批建议只做两个单变量方向：
  - E074A: E073 + robot ctrl trust-region guard。
  - E074C: E073 + hold/contact continuity reward。
- 第一波调度: 远程 A6000 GPU0 跑 E074A，GPU1 跑 E074C；本机只做编译/smoke 和结果回收后的评估。

## 下一步

- 等用户审核 `plan/79`。
- 若批准，写 E074 具体实施 plan，再开始代码与训练脚本实现。

## 追加分析: E062 palm normal 对 E074 base 的含义

- E062 的 `contact_hdmi_palm_normal_left/right` 不是接触点位置，而是 contact_hdmi orientation reward 使用的 wrist-local 朝向向量。
- E041c 默认 box025 指纹是 L=`[0,-1,0]`, R=`[0,+1,0]`; E062 对 box023 自动计算后改成双手 `[+1,0,0]`。
- 该向量进入 `mjwp.py` 的 E041 orientation block: `palm_world = quat_apply(eef_quat, palm_local)`, 再与 `target_world-contact_point` 做 dot，作为 additive ori reward 的方向项。
- 因 E073 -> E071W02 -> E062 -> E041c，E074 默认继承 E062 palm normal。它已经是 E071/E073 结果的一部分，后续不应默认移除；若要验证影响，应作为单独 ablation。

---

# E073 Progress — 2026-05-14

## 当前状态: ✅ E073 完成，target eef_offset 修正部分有效但未解决 hold

## 完成步骤

- [x] 按 `experiment-planning-zh` 读取 `EXPERIMENT_TRACKER.md`、E072 plan/log、`progress.md`。
- [x] 审查 `examples/run_mjwp.py` 与 `spider/simulators/mjwp.py` contact_hdmi dynamic target 实现。
- [x] 发现 E040 dynamic target 口径不一致：target 使用 ref wrist body origin，reward 使用 sim `wrist + eef_offset` contact point。
- [x] 写 E073 plan: `workspace/core4d/plan/78_E073_contact_target_offset_consistency_plan.md`。
- [x] 修改 `spider/config.py`，新增 `contact_hdmi_target_uses_eef_offset`，默认 false。
- [x] 修改 `examples/run_mjwp.py`，E073 打开字段时 dynamic target 改用 ref `wrist + eef_offset`。
- [x] 新增 override: `examples/config/override/core4d_e073_box023.yaml`。
- [x] 新增 E073 eval: `workspace/core4d/scripts/eval/eval_E073.py`。
- [x] 新增 E073 train: `workspace/core4d/scripts/train/train_E073.sh`。
- [x] `py_compile` 通过。
- [x] 运行 `bash workspace/core4d/scripts/train/train_E073.sh 0`；日志确认 RTX 5090 可见，且 `E040 dynamic target ... uses_eef_offset=True` 生效。
- [x] 输出 `eval_summary.json`、`timeseries.csv`、timeline plot、frame100-180 keyframes。
- [x] 按用户要求将关键帧视觉复核交给 subagent Ampere，主线程未直接 `view_image`。
- [x] 写 E073 结果 log: `workspace/core4d/log/93_E073_contact_target_offset_consistency_results.md`。

## 当前实验

- **Run ID**: E073
- **阶段**: Complete
- **目标**: 修正 dynamic target 的 eef_offset 口径，验证 frame100-145 hand-object contact 是否改善。

## 关键结果

| 指标 | E071/E072 | E073 |
|------|----------:|-----:|
| yaw err t=0.017/0.033 | 0.574 / 1.075 deg | 0.574 / 1.075 deg |
| B1 pre-contact max foot z | 0.069m | 0.080m |
| first obj_err >25cm | frame100 / 2.00s | frame100 / 2.00s |
| first sim zero contact | frame100 / 2.00s | frame108 / 2.16s |
| post2 sim contact frames | 44.4% | 49.4% |
| post2 obj_err max | 0.308m | 0.293m |
| first pelvis_z <45cm | frame166 / 3.32s | none |
| post2 pelvis_z min | 0.207m | 0.663m |

**视觉结论**: f100/f115 手和箱还较近；f130 起拿持质量明显变差；f145 箱子已明显落地/接触地面；f166/f168 未像 E071/E072 那样摔倒，但有脚/腿与箱体异常接触。

**结论**: eef_offset target 口径修正改善了接触连续性和稳定性，但没有解决 2s 后真实 hold。E074 应继承 E073，并增加 robot ctrl trust-region guard，重点压 frame100-145 的断触和 robot ctrl 快速偏离。

- 下一步: 规划 E074。

---

# E072 Progress — 2026-05-14

## 当前状态: ✅ E072 完成，hold/contact 先失效已定位

## 完成步骤

- [x] 按 `experiment-planning-zh` 读取 `EXPERIMENT_TRACKER.md`、E071 plan、E071 log、`progress.md`。
- [x] 确认 E071 结论：0-2s init/early drift 修复；2.0s 后物体跟踪误差先升至 >25cm，约 3.32s robot pelvis/body z <45cm 后摔倒。
- [x] 检查 E071 结果结构：`E071W02_box023.npz` 含 `qpos/qvel/ctrl/time/trace_ref`，scene snapshot 含 `scene_act.xml`，可以做 replay 诊断。
- [x] 写入 E072 plan: `workspace/core4d/plan/77_E072_post2_hold_place_diagnosis_plan.md`。
- [x] 新增 E072 eval 脚本: `workspace/core4d/scripts/eval/eval_E072.py`。
- [x] 新增 E072 入口脚本: `workspace/core4d/scripts/train/train_E072.sh`。
- [x] 运行 `bash workspace/core4d/scripts/train/train_E072.sh`，输出 `timeseries.csv`、`diagnosis_summary.json`、`contact_summary.csv`、timeline plot、frame-index keyframes。
- [x] 复核关键帧 f100/f115/f130/f145/f166/f180。
- [x] 写 E072 结果 log: `workspace/core4d/log/92_E072_post2_hold_place_diagnosis_results.md`。
- [x] 更新 `EXPERIMENT_TRACKER.md`。

## 当前实验

- **Run ID**: E072
- **阶段**: Complete
- **目标**: 区分 post-2s failure 是 hand-object hold/contact 先丢，还是 putdown 阶段稳定性先崩。

## 关键结果

| 指标 | 结果 |
|------|------|
| first obj_err >25cm | frame 100 / eval 2.00s / 0.308m |
| first sim hand-object zero contact | frame 100 / eval 2.00s, ref 同帧仍 contact=1 |
| first robot ctrl Linf >0.5 | frame 109 / eval 2.18s |
| first sim min hand SDF >10cm | frame 134 / eval 2.68s |
| first pelvis_z <45cm | frame 166 / eval 3.32s |
| post2 contact frames | sim 44.4% vs ref 80.2% |

**结论**: E071 post-2s 是 hold/contact 先失效，随后 CEM 追 object/body target 导致前扑和摔倒。object ctrl diff max 仅 0.01，不是新 mapping 问题。

## 下一步

- E073 优先做 hold/contact consistency 或 contact-preserving target，不先做单纯 stability weight。
- 同时考虑 robot ctrl trust-region guard，限制 frame 109 后的 robot ctrl Linf 级偏移。

---

# E068 Progress — 2026-05-14

## 当前状态: Plan 已写入，开始 init drift 诊断

## 完成步骤

- [x] 读取 `EXPERIMENT_TRACKER.md`、最新 plan/log/progress，确认最新问题来自 log 87 的 MJWP init pose mismatch。
- [x] 审查 `spider/simulators/mjwp.py::setup_env()`、`examples/run_mjwp.py` 初始化路径，发现 qpos/qvel/ctrl 写入后立即 `mj_step()` 的可疑路径。
- [x] 写 plan → `workspace/core4d/plan/73_E068_mjwp_init_drift_plan.md`
- [x] 新建 E068 诊断脚本 → `workspace/core4d/scripts/debug/diagnose_E068_init_drift.py`
- [x] 新建 E068 入口脚本 → `workspace/core4d/scripts/train/train_E068.sh`
- [x] 读取同步后的 E062-E067 真实结果，确认 E062/E063 第二个 substep 已出现约 22 deg yaw drift。
- [x] 发现 drift 不是 init `mj_step` 单独造成：CPU `mj_step` yaw drift 仅 0.22 deg；真实轨迹第一个 committed ctrl 相对 ref 有 1.56 rad 级别 robot joint 偏差。
- [x] 新建 first-commit 分析脚本 → `workspace/core4d/scripts/debug/analyze_E068_first_commit.py`
- [x] 运行 first-commit 分析，输出 `workspace/core4d/results/E068/first_commit_trace.csv` 和 `first_ctrl_delta.csv`
- [x] 视频/关键帧复查 E062/E063/E067，确认 lunge / handstand 与数值诊断一致。
- [x] 写 E068 结果 log → `workspace/core4d/log/88_E068_mjwp_init_drift_results.md`
- [x] 更新 `EXPERIMENT_TRACKER.md` 的 E068 行和 Logs 引用。
- [x] 写 E069 plan → `workspace/core4d/plan/74_E069_first_tick_warmup_plan.md`
- [x] 新建 E069 yaml 变体: `core4d_e069w02_box023.yaml`, `core4d_e069w05_box023.yaml`
- [x] 新建 E069 train/eval 脚本。
- [x] 空结果运行 `train_E069.sh eval` 验证脚本可执行，当前结果缺失为预期。
- [x] 修正 `train_E069.sh`，避免 eval-only 模式触发 scene snapshot。

## 当前实验

- **Run ID**: E068
- **Version**: R5 / Phase 18
- **阶段**: Plan → Diagnose
- **开始时间**: 2026-05-14

## 下一步

- 等待 GPU 机器运行 `bash workspace/core4d/scripts/train/train_E069.sh parallel 0 1`。
- 运行完成后检查 `workspace/core4d/results/E069/eval_summary.csv`，再写 log 89。

## 创建/修改的文件

- `workspace/core4d/plan/73_E068_mjwp_init_drift_plan.md`
- `workspace/core4d/scripts/debug/diagnose_E068_init_drift.py`
- `workspace/core4d/scripts/debug/analyze_E068_first_commit.py`
- `workspace/core4d/scripts/train/train_E068.sh`
- `workspace/core4d/log/88_E068_mjwp_init_drift_results.md`
- `workspace/core4d/results/E068/first_commit_trace.csv`
- `workspace/core4d/results/E068/first_ctrl_delta.csv`
- `workspace/core4d/EXPERIMENT_TRACKER.md`
- `workspace/core4d/plan/74_E069_first_tick_warmup_plan.md`
- `examples/config/override/core4d_e069w02_box023.yaml`
- `examples/config/override/core4d_e069w05_box023.yaml`
- `workspace/core4d/scripts/train/train_E069.sh`
- `workspace/core4d/scripts/eval/eval_E069.py`
- `workspace/core4d/progress.md`

## 遇到的错误

| 错误 | 尝试次数 | 解决方案 |
|------|---------|---------|
| 当前沙箱无 CUDA 设备，`device=cuda:0` 触发 `RuntimeError: No CUDA GPUs are available` | 1 | 将 E068 诊断入口默认改为 `E068_DEVICE=cpu`，只做 init state 诊断 |
| CPU device 无法执行 `setup_env()` 的 CUDA graph capture，触发 `RuntimeError: Must be a CUDA device` | 1 | 诊断脚本改为先保存 CPU `mj_forward`/`mj_step` 证据，并尝试手动 `mjwarp.put_data` 不 capture graph |
| `train_E069.sh eval` 初版会先 snapshot scene | 1 | 调整脚本，仅 `parallel/single_*` 训练模式执行 snapshot |

---

# E057 Progress — 2026-05-13

## 当前状态: ✅ 完成 (bucket005_s2 hand-snap, 6/6 Claims 通过)

## 完成步骤

- [x] 写 plan → `plan/67_E057_bucket005_s2_hand_snap_plan.md`
- [x] 克隆 E055 三件套 (snap / visualize / extract_keyframes), 改 CASE 路径
- [x] 新建 `verify_snap_face.py` (C6 — snap 后 main face vs E056 诊断一致性)
- [x] 写一键脚本 `run_E057_snap.sh` (snap → viz → keyframes → face 验证)
- [x] 跑流水线 — 单次跑通 4 步, 约 2 分钟
- [x] 视频核实 5 keyframe (front+side, top=ref/bot=snap)
- [x] 写 log → `log/67_E057_bucket005_s2_hand_snap_results.md`
- [x] 更新 EXPERIMENT_TRACKER (E057 行 + log/plan/scripts 引用)

## Claims 验证

| ID | 标准 | 实际 | 通过 |
|---|---|---|---|
| C1 | npz + csv + mp4 三件齐 | 101K npz + 11K csv + 2.3M mp4 | ✅ |
| C2 | palm-to-surface final ≤ 5cm | mean 4.93cm, max 5.10cm | ✅ |
| C3 | 关节限位 100% | 176/176 | ✅ |
| C4 | ≥ 4/5 keyframe 视觉合格 | **5/5 通过**, mid frame 教科书对侧握 ⭐ | ✅ |
| C5 | 一键脚本 | run_E057_snap.sh 跑通 | ✅ |
| C6 | snap face = E056 (L=-yz, R=+yz) | L=-yz 77%, R=+yz 100% | ✅ |

**6/6 通过 ✅**

## 关键结果

```
intent: (20, 107) = 88 frames @ 30fps (与 E056 诊断一致)
snap statistics (intent 内 88f × 2 hands = 176 rows):
  init  cm: mean=5.04 max=7.57   ← mocap 原始, 已经接近 5cm offset
  final cm: mean=4.93 max=5.10   ← 收敛到 target (表面外 5cm)
  ik_residual cm: mean=0.11 max=1.12
  ik_iters: mean=8.5 max=50
  joints in_limits: 176/176 (100%)
  L final: mean=4.99 max=5.09
  R final: mean=4.87 max=5.10

C6 face verification (intent 内):
  REF  L=-yz (med 0.58cm, 100%) | R=+yz (med 3.24cm, 100%)
  SNAP L=-yz (med 2.90cm,  77%) | R=+yz (med 1.87cm, 100%)
  → 双手 main face 完全保留 E056 诊断, 没漂移到错面
```

## 重要发现

1. **case 选对了, snap 几乎自由**: bucket005_s2 mocap 原始就 init=5cm, IK 几乎不需要努力。E055 box023 init~10cm, snap 后才 5cm。**E056 case 排序的物理意义在 E057 上兑现**。

2. **C6 face 验证是必要 guard**: 没这条 claim, snap 把 L 投到错面 (e.g. +yz) 数值上仍报"成功"。**后续任何 IK-to-surface 都应该 verify 接触面**。

3. **frame warm-start 持续有效**: max iter=50 偶发 (intent boundary), mean 8.5 iter, 大多数帧前一帧 qpos 就近。

## 下一步: E058 (Path B-CEM, bucket005_s2)

- 把 `warmstart_qpos.npz` 喂入 MJWP CEM 作为初始 mean trajectory
- body tracking ref 也换成 qpos_snap (让 reward "信" 修正后的 ref)
- 对比有/无 warmstart 的 contact / stability / pelvis_z
- Claims (草案):
  - contact (palm 距 bucket < 5cm 帧占比) ≥ 50%
  - stability (pelvis_z ≥ 0.5m) ≥ 90%
  - 视频: snap 阶段 CEM 没把 ±yz 两侧握姿"破坏"

## 改动文件

| 类型 | 路径 |
|---|---|
| 新建 plan | `workspace/core4d/plan/67_E057_bucket005_s2_hand_snap_plan.md` |
| 新建脚本 (×4) | `workspace/core4d/scripts/E057/{snap_bucket005_s2,visualize_snap,verify_snap_face}.py` + `extract_snap_keyframes.sh` |
| 新建一键脚本 | `workspace/core4d/scripts/run_E057_snap.sh` |
| 输出 npz | `workspace/core4d/results/E057/bucket005_s2_person1/warmstart_qpos.npz` |
| 输出 csv (×2) | `snap_diagnostics.csv` + `face_verification.csv` |
| 输出 mp4 | `snap_visualization.mp4` (2.3M, 148 帧) |
| 输出 png | `face_dist_snap.png` (2×2 时序) |
| 输出 jpg (×5) | `keyframes/frame_0[0-4]_*.jpg` |
| 新建 log | `workspace/core4d/log/67_E057_bucket005_s2_hand_snap_results.md` |
| EXPERIMENT_TRACKER | 添加 E057 行 + log/plan/scripts 引用 |

**`spider/preprocess/hand_snap_ik.py` 没动** — E055 实现 case-agnostic 已被 E057 验证。

---

## E069 进展: first-tick warmup 运行与保存修复

- [x] 已确认本机 GPU 在非 sandbox 命令下可见: RTX 5090 / Driver 580.126.09 / CUDA 13.0。
- [x] 已运行 `bash workspace/core4d/scripts/train/train_E069.sh parallel 0 0`。
- [x] W02/W05 均跑到 `sim_steps: 272/272`，不是 CUDA 或中途优化失败。
- [x] 失败点: `examples/run_mjwp.py` 结束保存 `info_list` 时 `np.stack` 遇到跨 tick shape 不一致的诊断字段，导致 `.npz`/`.mp4` 没有落盘。
- [x] 已修复保存逻辑: `qpos/qvel/time/ctrl` 等 shape 一致字段继续保存；shape 不一致或缺失的诊断字段跳过并写 warning。

### 遇到的错误

| 错误 | 尝试次数 | 解决方案 |
|------|---------|---------|
| E069 W02/W05 完整跑完后 `ValueError: all input arrays must have the same shape` | 1 | 修改 `examples/run_mjwp.py` 的 info 聚合逻辑，跳过 shape 不稳定的诊断字段，保留轨迹核心字段 |

### 下一步

- [x] 重跑 E069 W02/W05。
- [x] 修正 `eval_E069.py` 的 ctrl_ref 口径并重新生成 `eval_summary.csv`。
- [x] 从视频提取关键帧并写 `log/89_E069_first_tick_warmup_results.md`。
- [x] 更新 `EXPERIMENT_TRACKER.md`。

## E069 结果摘要

| 指标 | E069-W02 | E069-W05 | 结论 |
|------|----------|----------|------|
| warmup robot ctrl diff | 0.00 rad | 0.00 rad | ref ctrl 确实提交 |
| yaw err t=0.017/0.033s | 12.40 / 22.16 deg | 12.40 / 22.16 deg | warmup 无法压住 early yaw |
| B1 max foot z [0,2s] | 0.222m | 0.428m | 仍单脚/lunge |
| pelvis_min_intent | 0.578m | 0.197m | W02 数值通过但视频仍不可用 |

**新结论**: first CEM override 不是主因。即使 warmup 内提交 `ctrl_ref`，MJWarp commit step 仍复现 12/22 deg early yaw drift。下一步应做 E070 ref-control parity: MuJoCo `mj_step(ctrl_ref)` vs MJWarp `step_env(ctrl_ref)`。

## E070 计划

- [x] 写 plan → `workspace/core4d/plan/75_E070_mjwarp_ref_control_parity_plan.md`
- [x] 用户确认继续后，实现 parity 诊断脚本 → `workspace/core4d/scripts/debug/diagnose_E070_ref_control_parity.py`
- [x] 实现入口脚本 → `workspace/core4d/scripts/train/train_E070.sh`
- [x] `py_compile` 通过。
- [x] 首轮 E070 GPU parity 诊断完成。
- [x] 根据首轮结果扩展脚本: 增加 `qpos_ctrl` vs `orig_ctrl` 对照，区分 run_mjwp 当前 qpos-as-ctrl 映射与原始 29-dim robot ctrl 映射。
- [x] 重跑 E070 GPU parity 诊断。

### E070 实现补充

计划原本只比较 CPU vs MJWarp；实际脚本增加了两条控制变量：

- `zero_gains`: 保持 object actuator gains 为 0，对应 setup/start 状态。
- `restored_gains`: 按 `run_mjwp.py` commit 阶段恢复 object actuator gains，再提交 `ctrl_ref`。

这样能区分 drift 来自 MJWarp step 本身，还是来自 commit 阶段 object actuator gain 恢复后的物体反作用。

### E070 首轮发现

- `qpos_ctrl` 口径下，CPU MuJoCo 和 MJWarp 完全一致，并且都精确复现 E069: t=0.017/0.033 yaw err = 12.403/22.156 deg，`qpos_max_abs_diff_vs_e069 ≈ 0`。
- `zero_gains` 与 `restored_gains` 几乎一致，object actuator gain 恢复不是主因。
- 新疑点: `run_mjwp.py` 的 `ctrl_ref = qpos_ref[:, :config.nu]` 可能把 floating base pos/quat 当成 robot actuator ctrl；E068 的小漂移使用的是原始 29-dim robot ctrl + scene_act object ctrl。需要 `orig_ctrl` 对照验证。

### E070 最终发现

| ctrl 口径 | CPU yaw err t=0.017/0.033 | MJWarp yaw err t=0.017/0.033 | vs E069 | 结论 |
|-----------|----------------------------|-------------------------------|---------|------|
| `qpos_ctrl` (`qpos_ref[:, :nu]`) | 12.403 / 22.156 deg | 12.403 / 22.156 deg | `qpos_max_abs_diff≈0` | 精确复现 E069 错误 |
| `orig_ctrl` (原始 29-dim robot ctrl + scene_act object ctrl) | 0.574 / 1.075 deg | 0.574 / 1.075 deg | 明显不同 | early drift 基本消失 |

**根因修正**: 不是 MJWarp physics mismatch，也不是 object actuator gain 恢复。`examples/run_mjwp.py` 在 contact guidance 下的 `ctrl_ref = qpos_ref[:, :config.nu]` 把 floating-base qpos 前 7 维混入 robot actuator ctrl，导致 ref-control 本身就是错的。E071 应修 scene_act ctrl 映射: 保留原始 29-dim robot ctrl，只把 object 6DOF ctrl 从转换后的 qpos 填入末 6 维。

## E071 结果摘要

- [x] 已写 plan: `workspace/core4d/plan/76_E071_scene_act_ctrl_mapping_fix_plan.md`
- [x] 已修 `examples/run_mjwp.py`: 删除 qpos-as-ctrl fallback，保留原始 29-dim robot ctrl。
- [x] 已新增配置: `examples/config/override/core4d_e071w02_box023.yaml`
- [x] 已新增训练脚本: `workspace/core4d/scripts/train/train_E071.sh`
- [x] 已新增评估脚本: `workspace/core4d/scripts/eval/eval_E071.py`
- [x] 已运行 `bash workspace/core4d/scripts/train/train_E071.sh 0`。
- [x] 已生成 `.npz`、`.mp4`、`eval_summary.csv` 和关键帧。

| 指标 | E069-W02 | E071-W02 | 结论 |
|------|----------|----------|------|
| yaw err t=0.017/0.033s | 12.40 / 22.16 deg | 0.574 / 1.075 deg | early drift 消失 |
| vs E070 orig parity | N/A | -0.0004 / +0.0005 deg | 与正确 ctrl 口径一致 |
| warmup ctrl diff | 0.00 / 0.00 | 0.00 / 0.00 | ref ctrl 提交正确 |
| B1 max foot z [0,2s] | 0.222m | 0.069m | pre-contact lunge 消失 |
| pelvis_min_intent | 0.578m | 0.674m | 更稳定 |
| post-2s obj_err max/mean | 未统计 | 0.308 / 0.133m | post-contact FAIL |
| first post-2s obj_err > 25cm | 未统计 | 2.00s | 没有稳定拿住箱子 |
| post-2s pelvis body z min | 未统计 | 0.200m | 摔倒 |
| first post-2s pelvis z < 45cm | 未统计 | 3.32s | 摔倒开始 |

**修正结论**: E070 根因只对 early yaw/lunge 完全确认。scene_act 下 `qpos_ref[:, :nu]` fallback 是 box023 0-2s 初始漂移的主因；保留 raw robot ctrl 并由 scene_act conversion 补 object ctrl 后，初始漂移消失。但 E071 不是整体成功：2s 后 robot 没有稳定拿住箱子，约 3.3s 开始摔倒。下一步 E072 应聚焦 post-2s hold/place failure 诊断，而不是先做泛化 regression。
