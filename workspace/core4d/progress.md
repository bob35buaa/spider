# E078 Progress — 2026-05-15

## 当前状态: E078 计划已写入，正在实现 3cm per-EEF contact mask 的 CEM 实验脚本

## 完成步骤

- [x] 按 `experiment-planning-zh` 恢复实验上下文，确认最新计划为 `workspace/core4d/plan/83_E078_3cm_per_eef_contact_mask_cem_plan.md`。
- [x] 在 `spider/config.py` 增加默认关闭的 3cm contact mask source 配置项，旧实验默认仍走 `rotated_sdf`。
- [x] 修改 `examples/run_mjwp.py`，支持从 E077 `raw_contact_mask_3cm.npz` 读取 `(T, person, hand)` mask，并转换为 HDMI-style per-EEF `(T,2)` gating。
- [x] 修改 `spider/simulators/mjwp.py`，使 `contact_hdmi_rew` 支持 scalar、legacy `(N,)` 和 per-EEF `(N,2)` mask；`hold_contact` 只用 per-EEF mask 的 max 作为旧式 ref gate，避免形状污染。
- [x] 新增 E078 override：
  - `examples/config/override/core4d_e078a_box023_p1_3cm.yaml`
  - `examples/config/override/core4d_e078b_box023_p2_3cm.yaml`
- [x] 新增 E078 train/eval/remote/pull 脚本：
  - `workspace/core4d/scripts/train/train_E078.sh`
  - `workspace/core4d/scripts/eval/eval_E078.py`
  - `workspace/core4d/scripts/run_E078_remote.sh`
  - `workspace/core4d/scripts/pull_E078_remote_results.sh`

## 待完成

- [x] 静态验证通过：`py_compile` 覆盖 `spider/config.py`、`spider/simulators/mjwp.py`、`examples/run_mjwp.py`、`eval_E078.py`；`bash -n` 覆盖 E078 train/remote/pull。
- [x] Hydra/_build_config 验证：
  - E078A: `task=box023_person1`, `mask_person_idx=0`, E077 mask keys `(178,2,2)/(136,2,2)/(227,2,2)`。
  - E078B: `task=box023_person2`, `mask_person_idx=1`, `data_path/model_path` 均存在。
- [x] 本地短 horizon GPU smoke test 通过：
  - E078A: RTX 5090, `max_sim_steps=4`, 3cm mask 选 `eval_contact_mask_3cm`, `person_idx=0`, `len 227→322`, active L/R=46.3%/45.0%。
  - E078B: RTX 5090, `max_sim_steps=4`, 3cm mask 选 `eval_contact_mask_3cm`, `person_idx=1`, `len 227→322`, active L/R=45.0%/47.5%。
  - 两者均生成 `/tmp/e078_smoke_{a,b}/trajectory_mjwp_act.npz`，未触发 reward mask shape error。
- [x] 强制纳入远程必需数据：E077 3cm mask 与 `box023_person2` SPIDER case；未纳入 `.codex/config.toml` / `__pycache__`。
- [ ] commit + push 后启动远程 E078A/E078B 并行。

## 遇到的错误

| 错误 | 尝试次数 | 解决方案 |
|------|---------|----------|
| 沙箱内 `uv run` 无法访问 CUDA，报 `No CUDA GPUs are available` | 1 | 使用批准的 escalated GPU smoke run；`nvidia-smi` 与 Warp 均确认 RTX 5090 可用 |
| `git diff --cached --check` 报 E077 CSV CRLF / E076 log trailing whitespace | 1 | 转为 LF 并移除末尾空格后通过 |

---

# E075 Progress — 2026-05-14

## 当前状态: Plan 已写入，开始 E075 限时/弱化 hold_contact 远程并行实现

## 完成步骤

- [x] 按 `experiment-planning-zh` 恢复 `EXPERIMENT_TRACKER.md`、最新 plan/log、`progress.md`。
- [x] 读取远程执行指南 `.codex/skills/experiment-planning-zh/remote-execution.md`，确认 `spider-remote` 与 `/home/xiayb/pHRI_workspace/spider`。
- [x] 复读 E074 计划/脚本/config，确认 E075 不需要改 reward 实现，只需新增两个组合 override 与 E075 脚本。
- [x] 写入 E075 计划: `workspace/core4d/plan/81_E075_limited_hold_contact_remote_plan.md`。
- [x] 新增 E075 override:
  - `examples/config/override/core4d_e075b_box023.yaml` = E074A + hold_contact scale 1.0, window 1.8-2.5s。
  - `examples/config/override/core4d_e075a_box023.yaml` = E074A + hold_contact scale 0.5, window 1.8-2.5s。
- [x] 新增 E075 train/eval/remote/pull 脚本:
  - `workspace/core4d/scripts/train/train_E075.sh`
  - `workspace/core4d/scripts/eval/eval_E075.py`
  - `workspace/core4d/scripts/run_E075_remote.sh`
  - `workspace/core4d/scripts/pull_E075_remote_results.sh`
- [x] 静态验证通过: `py_compile eval_E075.py`; `bash -n` 三个 shell 脚本。
- [x] Hydra compose 验证:
  - E075B: `ctrl_ref_guard_scale=0.5`, `hold_contact_rew_scale=1.0`, window 1.8-2.5, `contact_hdmi_target_uses_eef_offset=True`。
  - E075A: `ctrl_ref_guard_scale=0.5`, `hold_contact_rew_scale=0.5`, window 1.8-2.5, `contact_hdmi_target_uses_eef_offset=True`。
- [x] 提交并推送 E075 实验脚本/config: `90d5b34 exp(core4d): E075 limited hold-contact remote sweep`。
- [x] 远程 `spider-remote:/home/xiayb/pHRI_workspace/spider` 已 fast-forward 到 `90d5b34`。
- [x] 启动远程 tmux session `E075`:
  - E075B -> GPU0, PID 1238653。
  - E075A -> GPU1, PID 1238654。
  - 远程 scene snapshot: `workspace/core4d/results/E075/scene_snapshot/`。

## 远程运行状态

- 2026-05-14 21:57: tmux 输出确认两个 run 已启动。
- 初始结果计数: `0` 个 `.npz`，符合刚启动状态。
- 2026-05-14 22:19: 远程 E075 完成，已 scp 回收并完成本地 `eval_E075.py`。

### 初步数值结果

| 指标 | E074A | E074C | E075B scale1.0 limited | E075A scale0.5 limited |
|------|------:|------:|-----------------------:|-----------------------:|
| yaw 0.017/0.033 deg | 0.574 / 1.075 | 0.574 / 1.075 | 0.574 / 1.075 | 0.574 / 1.075 |
| B1 pre-contact foot z | 0.083m | 0.085m | 0.083m | 0.085m |
| first zero contact | f110 | f101 | f110 | f111 |
| frame100-145 contact | 54.3% | 63.0% | 76.1% | 60.9% |
| post2 contact | 54.3% | 64.2% | 67.9% | 61.7% |
| post2 obj_err max | 0.289m | 0.324m | 0.287m | 0.290m |
| post2 obj_err mean | 0.177m | 0.197m | 0.158m | 0.155m |
| post2 pelvis_z min | 0.692m | 0.701m | 0.660m | 0.147m |
| first robot ctrl Linf >0.5 | f122 | f114 | f117 | f100 |
| post2 robot ctrl Linf max | 0.740 | 0.897 | 0.690 | 0.662 |
| post2 min hand SDF mean | 0.073m | 0.043m | 0.057m | 0.054m |

初步判断:

- E075B 是当前数值最好的组合：contact 高于 E074C，object error 接近/略优 E074A，robot ctrl Linf 更低，pelvis 没有摔倒。
- E075A 接触也改善，但 first robot ctrl Linf 在 f100 即超阈值，且 pelvis_z min=0.147m，稳定性明显回归，不宜作为主线。
- 需要等待 subagent 视觉复核 E075B/E075A 关键帧后写正式 log。
- [x] subagent Erdos 完成 E075A/E075B 关键帧视觉复核。
- [x] 写入正式结果日志: `workspace/core4d/log/96_E075_limited_hold_contact_results.md`。
- [x] 更新 `EXPERIMENT_TRACKER.md` E075 行。

正式结论:

- E075B 是 best-so-far partial positive：contact 大幅提升，object error 不退化，f180 站稳且箱子分离；但 f145-f166 释放仍不干净，first obj_err>25cm 仍 f100。
- E075A 失败：f166-f180 出现身体/腿/箱强干涉并摔倒。
- 下一步应以 E075B 为 base，先做 release/leg clearance replay 诊断，再设计放置后脱离/clearance reward。

### 追加诊断: E075B f115-f130 右腿相位偏差

- [x] 用户指出 E075B f120-f125 右腿相对 ref 突然前跨；已补抽 f115-f130 连续帧并交给 subagent Erdos 视觉复核。
- [x] 视觉结论: 不是单帧视觉错觉。ref 在 f120 后进入停步/弯腰/准备放箱，右脚接近地面且趋于稳定；sim 仍在继续向前走一步，右腿从后摆连续前跨，到 f125-f130 与 ref 姿态明显分歧。
- [x] 数值结论: f119-f125 sim right_foot XY 每帧位移约 7.5-9.7cm，而 ref 约 1.9-4.0cm；right_hip_pitch ctrl diff 在 f120-f124 成为主导偏离，约 -0.50rad。该段是真实步态/任务相位偏差，不只是 release 问题。
- [x] 将用户关于 hand-crafted `hold_contact_start/end_eval_time` 泛化风险、HDMI contact label vs core4d estimated mask 差异、以及 E076 应优先 audit/fix contact mask 的讨论写入 `workspace/core4d/log/96_E075_limited_hold_contact_results.md`。

## 当前实验

- **Run ID**: E075
- **阶段**: Implement
- **目标**: 在 E074A ctrl guard 基础上，对比 `hold_contact` 的限时中等强度(scale=1.0, 1.8-2.5s) 与限时弱强度(scale=0.5, 1.8-2.5s)，验证能否提升接触而不复现 E074C 的腿/箱干涉。

---

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

## 远程运行状态

- 2026-05-14 21:02: 远程 `spider-remote:/home/xiayb/pHRI_workspace/spider` 已 fast-forward 到 `b112eac`。
- tmux session: `E074`。
- 远程启动命令: `bash workspace/core4d/scripts/train/train_E074.sh parallel 0 1`。
- GPU 分配: E074A -> GPU0, E074C -> GPU1。
- tmux 输出确认:
  - `[21:02:26] launched E074A PID=1219883, E074C PID=1219884`
  - `[21:02:26] === E074A_box023 override=core4d_e074a_box023 GPU=0 ===`
  - `[21:02:26] === E074C_box023 override=core4d_e074c_box023 GPU=1 ===`
- 初始结果计数: `0` 个 `.npz`，符合刚启动状态。

## E074 回收状态

- 2026-05-14 21:36: 已从 `spider-remote:/home/xiayb/pHRI_workspace/spider` scp 回收 E074 结果与日志。
- 本地结果:
  - `workspace/core4d/results/E074/E074A_box023.npz`
  - `workspace/core4d/results/E074/E074A_box023.mp4`
  - `workspace/core4d/results/E074/E074C_box023.npz`
  - `workspace/core4d/results/E074/E074C_box023.mp4`
  - `workspace/core4d/results/E074/comparison.csv`
  - `workspace/core4d/results/E074/keyframes/{E074A,E074C}/f100..f180.jpg`
- 本地日志:
  - `logs/E074/E074A_box023.log`
  - `logs/E074/E074C_box023.log`
  - `logs/E074/eval_E074_local_after_pull.log`

### 初步数值结果

| 指标 | E073 | E074A ctrl guard | E074C hold contact |
|------|-----:|-----------------:|-------------------:|
| yaw 0.017/0.033 deg | 0.574 / 1.075 | 0.574 / 1.075 | 0.574 / 1.075 |
| B1 pre-contact foot z | 0.080m | 0.083m | 0.085m |
| first zero contact | f108 | f110 | f101 |
| frame100-145 contact | 45.7% | 54.3% | 63.0% |
| post2 contact | 49.4% | 54.3% | 64.2% |
| post2 obj_err max | 0.293m | 0.289m | 0.324m |
| post2 pelvis_z min | 0.663m | 0.692m | 0.701m |
| post2 robot ctrl Linf max | 0.778(E073 prior) | 0.740 | 0.897 |

初步判断:

- E074A 小幅改善 contact/object/stability，符合“更保守”的预期，但 contact 仍不足。
- E074C 明显提高 contact 与 hand SDF，但 first zero contact 反而提前到 f101，post2 obj_err max 变差到 0.324m，说明接触 reward 可能让手更贴近但没有改善物体跟随。
- 两者均未造成 early drift 或摔倒回归。
- 下一步需要按用户要求用 subagent 复核 E074A/E074C 视频关键帧后写正式 log 95。

## E074 正式分析

- [x] subagent Erdos 复核 E074A/E074C 关键帧。
- [x] 写入正式结果日志: `workspace/core4d/log/95_E074_remote_hold_contact_results.md`。
- [x] 更新 `EXPERIMENT_TRACKER.md` E074 行。

正式结论:

- E074A 是 partial positive：ctrl guard 将 robot ctrl 大偏离延后到 f122，视觉更接近成功，但 contact 提升不足。
- E074C 是 metric positive / visually unsafe：post2 contact 达 64.2%，但 obj_err 变差，后段腿/箱干涉明显。
- 下一步不应直接原样组合 E074A+C；推荐 E075B = E074A + time-limited weaker hold_contact。

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

---

## 2026-05-15 E076 contact source audit 进展

- [x] 使用 `experiment-planning-zh` 恢复 `EXPERIMENT_TRACKER.md`、E075 log 和 `progress.md`。
- [x] 回收 subagent 结果：
  - Fermat 确认 CORE4D raw 没有人手接触人工真值；官方 contact 是 SMPL-X/object 几何生成，`prepare_hho.py` 默认 2cm，visualization runtime contact 用 3cm。
  - Laplace 确认当前 `box023_person1` 进入 SPIDER 后是单 G1 + object，`contact=(136,2)` 全帧 `[1,1]`，不是 CORE4D/HDMI label；E039/E075 mask 仍是单 G1 ref 的 scalar SDF proxy。
- [x] 本地确认源序列：
  - `box023_person1` 对应 raw `/mnt/.../CORE4D_Real/human_object_motions/20231008/045`，object=`Box023`，action=`move2_obs0`。
  - Holosoma `trimmed` qpos 与 `retargeted[42:178]` 精确一致，因此 SPIDER 30Hz ref frame `k` 对应 raw frame `k+42`。
- [x] 修正视频帧对齐：
  - E075 视频/eval 是 50Hz，`f115-f130` 是 `2.30-2.60s`。
  - 对应 30Hz ref frame 约 `69-78`，raw frame `111-120`，不是 raw/ref 的 `115-130`。
- [x] raw 几何核验结果：
  - 在 E075 `f115-f130` 对应 raw `111-120`，person1 左手强接触：min dist mean/min/max = `0.74/0.42/1.08cm`，2cm/3cm/5cm 均 16/16 帧。
  - person1 右手是边界接触：min dist mean/min/max = `2.12/1.77/2.58cm`；2cm 阈值 5/16 帧，3cm 阈值 16/16 帧。
  - person2 双手强接触：左手 `0.17/0.03/0.53cm`，右手 `0.11/0.04/0.17cm`，2cm/3cm/5cm 均 16/16 帧。
- [x] 写入新诊断日志：`workspace/core4d/log/97_E076_contact_source_audit.md`。

### 修正后的判断

此前“ref 右手不一定应该继续强接触”的说法过强。当前证据只支持：

- robot retarget / MuJoCo ref 里右手几何接触弱；
- raw SMPL-X 里 person1 右手是 2cm 阈值边界、3cm 阈值持续接触；
- person2 双手在同一阶段强接触，所以双人支撑必须纳入解释；
- 当前 SPIDER contact/mask 不是 raw/HDMI label，下一步应先做 contact source alignment 和 per-hand mask 修复，而不是继续手写 hold/release 时间窗。

---

## E077 进展: 3cm contact mask + box023_person2 计划

- [x] 已按 `experiment-planning-zh` 写入计划：`workspace/core4d/plan/82_E077_core4d_3cm_contact_mask_and_box023_person2_plan.md`。
- [x] 初步检查本地没有现成的 `20231008-045-person2-Box023` Holosoma retarget 输出；person2 需要从 raw `20231008/045` 重新跑 `convert_core4d_to_omniretarget.py` + `robot_retarget.py`，不能直接复制 person1。
- [x] 计划将 3cm mask 分成 raw/spider/eval 三个时间轴，避免再次混淆 30Hz ref frame 和 50Hz eval frame。

### 当前 E077 决策

- 先生成 3cm raw contact proxy，作为临时 contact mask “真值”。
- 同时保存 min distance 与 contact vertex count，避免二值 mask 抹掉 person1 右手的边界接触信息。
- person2 构造必须核验 `trimmed == retargeted[42:178]`、object qpos 与 person1 同窗口，以及 `scene.xml/scene_act.xml` MuJoCo load。

### E077 实现进展

- [x] 新增脚本：
  - `workspace/core4d/scripts/E077/generate_core4d_contact_masks.py`
  - `workspace/core4d/scripts/E077/trim_box023_person2.py`
  - `workspace/core4d/scripts/E077/create_box023_person2_scene.py`
  - `workspace/core4d/scripts/E077/verify_box023_person2.py`
  - `workspace/core4d/scripts/E077/build_box023_person2.sh`
- [x] `py_compile` 与 `bash -n` 通过。
- [x] 已生成 3cm contact mask：
  - `workspace/core4d/results/E077/contact_masks/box023/raw_contact_mask_3cm.npz`
  - `workspace/core4d/results/E077/contact_masks/box023/raw_contact_mask_3cm.csv`
  - `workspace/core4d/results/E077/contact_masks/box023/audit_summary_3cm.json`

### E077 3cm mask 关键结果

| eval window | raw window | p1 L | p1 R | p2 L | p2 R |
|-------------|------------|------|------|------|------|
| 100-114 | 102-110 | 15/15, mean 0.81cm | 13/15, mean 2.42cm | 15/15, mean 0.11cm | 15/15, mean 0.11cm |
| 115-130 | 111-120 | 16/16, mean 0.74cm | 16/16, mean 2.12cm | 16/16, mean 0.17cm | 16/16, mean 0.11cm |
| 131-145 | 121-129 | 5/15, mean 13.46cm | 4/15, mean 11.14cm | 2/15, mean 25.04cm | 5/15, mean 15.26cm |

这复现并固化了 E076 结论：3cm 口径下 f115-f130 person1 右手为持续接触，但距离接近阈值边界；person2 双手强接触。

### E077 person2 构造结果

- [x] 修复 build 脚本环境：`convert_core4d_to_omniretarget.py` 需要先 source Holosoma `hsretargeting` conda 环境，否则找不到 `smplx`。
- [x] person2 retarget 完成：
  - `workspace/core4d/results/E077/holosoma_box023_person2/retargeted/20231008-045-person2-Box023_with_obj_original.npz`
  - qpos shape `(178,43)`，final cost 约 `0.574`。
- [x] person2 trim 完成：
  - `workspace/core4d/results/E077/holosoma_box023_person2/trimmed/20231008-045-person2-Box023_with_obj_original.npz`
  - qpos shape `(136,43)`，且 `trimmed == retargeted[42:178]`。
- [x] SPIDER case 完成：
  - `example_datasets/processed/core4d/unitree_g1/humanoid_object/box023_person2/scene.xml`
  - `example_datasets/processed/core4d/unitree_g1/humanoid_object/box023_person2/0/trajectory_kinematic.npz`
  - `example_datasets/processed/core4d/unitree_g1/humanoid_object/box023_person2/scene_act.xml`
- [x] 核验完成：
  - `scene.xml`: `nq=43,nv=41,nu=29`
  - `scene_act.xml`: `nq=42,nv=41,nu=35`, euler `XZY`
  - `trajectory_kinematic.npz`: qpos `(136,43)`, qvel `(136,41)`, ctrl `(136,29)`, contact `(136,2)` 全 1。
- [x] 写入结果日志：`workspace/core4d/log/98_E077_3cm_contact_mask_and_person2_results.md`。

### E077 关键 caveat

converted 层 `person1/person2` 的 object pose 完全一致，但 retarget/SPIDER 层 object qpos 不完全一致：

- max abs diff `0.0621m`
- position diff mean `[0.00018, 0.02700, -0.00697]`
- quat diff max `0`

原因是 Holosoma preprocess 按每个人的 `smpl_scale` 缩放 object xy/z 轨迹。结论：`box023_person2` 可以作为单人 case 使用，但不能和现有 `box023_person1` retarget qpos 直接合并成双机器人同场景；双人合成前必须做 common-scale/common-world alignment。

---

## E078 计划

- [x] 用户确认下一轮方向：修改 3cm contact mask 并对齐 HDMI，在 `box023_person1` 和 `box023_person2` 两个单人 case 做 CEM 动力学重定向。
- [x] 按用户要求先写计划、不执行实现。
- [x] 已写入计划：`workspace/core4d/plan/83_E078_3cm_per_eef_contact_mask_cem_plan.md`。

### E078 计划摘要

- E078A: `box023_person1`，基于 E075B，读取 E077 3cm mask 的 `person_idx=0`。
- E078B: `box023_person2`，构造 E075B-like p2 配置，读取 E077 3cm mask 的 `person_idx=1`。
- 主改动：
  - MJWP contact_hdmi mask 从 scalar `(T,)` 改成 HDMI-style per-EEF `(T,2)`。
  - 增加 `contact_hdmi_mask_source="core4d_3cm"`，从 E077 npz 读取 mask。
  - 保持旧 `rotated_sdf` 默认行为，避免影响其他实验。
- 执行前待用户确认；当前未修改代码、未启动训练。
