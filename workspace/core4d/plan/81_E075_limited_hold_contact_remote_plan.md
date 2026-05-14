# E075 计划: ctrl guard + 限时/弱化 hold_contact 远程并行

## Context

E074 已完成远程并行实验:

- E074A (`core4d_e073_box023 + ctrl_ref_guard_scale=0.5`) 是 partial positive: robot ctrl 大偏离从 f101 推迟到 f122，视觉最接近成功，但 contact/object 仍未达标。
- E074C (`hold_contact_rew_scale=2.0`, 1.8-3.0s) 提高了 hand SDF/contact，但 object error 变差，后段出现腿/箱干涉；说明 surrogate 生效但目标过粗。

因此 E075 不直接原样组合 E074A+E074C，而是验证两个更保守组合：

| Run | Base | 变量 | 远程 GPU |
|-----|------|------|----------|
| E075B | E074A | `hold_contact_rew_scale=1.0`, window 1.8-2.5s | GPU0 |
| E075A | E074A | `hold_contact_rew_scale=0.5`, window 1.8-2.5s | GPU1 |

命名上保留 E075B 为主优先级，因为 E074 log 95 的推荐优先级是 "time-limited hold_contact"。

## Claims

### C1: 不破坏 E071/E073 之后的 early drift 修复

成功标准：

- yaw err 0.017/0.033s < 2 deg。
- B1 pre-contact max foot z <= 0.10m。

### C2: 保留 E074A 的 robot ctrl 安全性

成功标准：

- first robot ctrl Linf > 0.5 不早于 E073 的 f101，目标接近或晚于 E074A 的 f122。
- post2 robot ctrl Linf max <= 0.80。

### C3: 吃到 hold_contact 的接触收益，但不复现 E074C 腿/箱干涉

成功标准：

- frame100-145 contact >= 60%，目标 >= 65%。
- post2 contact >= 60%。
- post2 sim_min_hand_sdf_mean <= E074A 的 0.073m。
- 视觉 f145-f180 不出现 E074C 那种箱子紧贴腿部/手臂持续压箱的模式。

### C4: object tracking 必须不比 E074C 坏

成功标准：

- post2 obj_err max < 0.30m，目标 < 0.25m。
- post2 obj_err mean <= E074A 的 0.177m。
- first obj_err >25cm 尽量晚于 f100；若仍是 f100，则必须在视觉上保持持箱/托箱关系更久。

## 改动

### 1. YAML

新增两个 override，均继承 `core4d_e074a_box023`：

- `examples/config/override/core4d_e075b_box023.yaml`
  - `hold_contact_rew_scale: 1.0`
  - `hold_contact_start_eval_time: 1.8`
  - `hold_contact_end_eval_time: 2.5`
  - `hold_contact_require_ref_contact: true`

- `examples/config/override/core4d_e075a_box023.yaml`
  - `hold_contact_rew_scale: 0.5`
  - 同样使用 1.8-2.5s window。

不改 `spider/simulators/mjwp.py`，确保 E075 是纯 config 组合实验。

### 2. 脚本

新增：

- `workspace/core4d/scripts/train/train_E075.sh`
- `workspace/core4d/scripts/eval/eval_E075.py`
- `workspace/core4d/scripts/run_E075_remote.sh`
- `workspace/core4d/scripts/pull_E075_remote_results.sh`

训练脚本需要先快照 scene：

```bash
bash workspace/core4d/scripts/convert/snapshot_scenes.sh E075 box023_person1
```

远程执行遵循 `.codex/skills/experiment-planning-zh/remote-execution.md`：

- host: `spider-remote`
- repo: `/home/xiayb/pHRI_workspace/spider`
- session: `E075`
- GPU0: E075B
- GPU1: E075A

## 成功/失败判定

| 判定 | 解释 | 下一步 |
|------|------|--------|
| E075B 优于 E075A 且视觉干净 | time window 是主因，保留 scale=1.0 | 下一步做接触质量/脱离 clearance |
| E075A 优于 E075B | 权重仍偏强，保留弱 hold_contact | 下一步在 0.25-0.5 之间细调或引入方向约束 |
| 两者 contact 改善但 obj_err/视觉仍差 | proximity surrogate 仍目标错位 | 转 E076 leg/box clearance + contact direction 诊断 |
| 两者均退化 | hold_contact 与当前 CEM 目标不兼容 | 回退到 E074A，优先做 contact target/placement phase 设计 |

## 执行命令

本地验证：

```bash
.venv/bin/python -m py_compile workspace/core4d/scripts/eval/eval_E075.py
bash -n workspace/core4d/scripts/train/train_E075.sh
bash -n workspace/core4d/scripts/run_E075_remote.sh
bash -n workspace/core4d/scripts/pull_E075_remote_results.sh
```

提交与远程启动：

```bash
git add examples/config/override/core4d_e075*.yaml workspace/core4d/scripts/train/train_E075.sh workspace/core4d/scripts/eval/eval_E075.py workspace/core4d/scripts/run_E075_remote.sh workspace/core4d/scripts/pull_E075_remote_results.sh workspace/core4d/plan/81_E075_limited_hold_contact_remote_plan.md workspace/core4d/progress.md
git commit -m "exp(core4d): E075 limited hold-contact remote sweep"
git push
bash workspace/core4d/scripts/run_E075_remote.sh
```

监控：

```bash
ssh spider-remote "tmux capture-pane -t E075 -p | tail -40"
ssh spider-remote "ls /home/xiayb/pHRI_workspace/spider/workspace/core4d/results/E075/*.npz 2>/dev/null | wc -l"
```

回收与评估：

```bash
bash workspace/core4d/scripts/pull_E075_remote_results.sh
```

## 视觉复核

训练完成并回收视频后，按用户要求如果需要 `view_image`/关键帧观察，由 subagent 观察 E075A/E075B 的 f100/f115/f130/f145/f160/f166/f180 关键帧，主线程汇总到 E075 结果日志。
