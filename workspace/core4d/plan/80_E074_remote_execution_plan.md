# E074 实施与远程调度计划

## Context

E073 已确认：

- 0-2s init/early drift 已由 E071 ctrl mapping 修复。
- E073 target eef_offset 口径修正改善了接触和稳定性。
- 当前主失败仍是 frame100-145 的 hold/contact discontinuity：f130 后脱手，f145 箱已落地。

用户说明本机只有 1 张 GPU，远程服务器可并行运行 2 个实验。按 `experiment-planning-zh/remote-execution.md`，远程机器配置为 `spider-remote`，项目路径 `/home/xiayb/pHRI_workspace/spider`，2x RTX 6000 Ada 48GB。远程只需要 git 同步代码/config，运行完成后 scp 回收。

## 目标

准备 E074 第一波实验的代码、配置、训练脚本、评估脚本和远程调度脚本。

第一波只跑两个单变量实验：

| Run | Base | 单一变量 | GPU 分配 |
|-----|------|----------|----------|
| E074A | E073 | robot ctrl trust-region guard | remote GPU0 |
| E074C | E073 | hold/contact continuity reward | remote GPU1 |

本机用途：

- 做 `py_compile`/配置 smoke。
- 必要时单独跑某个小规模复现或评估。
- 不参与第一波正式训练，避免串行拖慢节奏。

## 改动

### 1. Config

在 `spider/config.py` 增加默认关闭的 E074 字段：

```yaml
ctrl_ref_guard_scale: 0.0
ctrl_ref_guard_robot_only: true
ctrl_ref_guard_sigma: 0.25
ctrl_ref_guard_start_eval_time: 1.8
ctrl_ref_guard_end_eval_time: 3.0

hold_contact_rew_scale: 0.0
hold_contact_sigma: 0.05
hold_contact_start_eval_time: 1.8
hold_contact_end_eval_time: 3.0
hold_contact_require_ref_contact: true
```

默认值必须保持 E073 行为不变。

### 2. Reward

在 `spider/simulators/mjwp.py::get_reward()` 中增加：

- `ctrl_ref_guard_rew`: 当前 rollout ctrl 与 ref ctrl 的 robot actuator 维度 Huber penalty。
- `hold_contact_rew`: ref 接触窗口内，sim hand 到 rotated object box 的 min SDF proximity reward。

两者都用 `env.data_wp.time` 做 1.8-3.0s 窗口 gate。

### 3. YAML

新增：

- `examples/config/override/core4d_e074a_box023.yaml`
- `examples/config/override/core4d_e074c_box023.yaml`

均继承 `core4d_e073_box023`，只改一个变量。

### 4. 脚本

新增：

- `workspace/core4d/scripts/train/train_E074.sh`
- `workspace/core4d/scripts/eval/eval_E074.py`
- `workspace/core4d/scripts/run_E074_remote.sh`
- `workspace/core4d/scripts/pull_E074_remote_results.sh`

`train_E074.sh` 支持：

```bash
bash workspace/core4d/scripts/train/train_E074.sh single E074A 0
bash workspace/core4d/scripts/train/train_E074.sh single E074C 1
bash workspace/core4d/scripts/train/train_E074.sh parallel 0 1
```

远程脚本默认：

- host: `${REMOTE_HOST:-spider-remote}`
- repo: `${REMOTE_REPO:-/home/xiayb/pHRI_workspace/spider}`

如远程路径不同，运行前用环境变量覆盖。

## 评估指标

复用 E073/E072 口径，输出：

- `eval_summary.csv/json`
- `timeseries_<variant>.csv`
- `comparison.csv`
- `plots/<variant>_post2_failure_timeline.png`
- `keyframes/<variant>_f100/f115/f130/f145/f160/f166/f180.jpg`

核心对照：

| 指标 | E073 | E074 目标 |
|------|-----:|----------:|
| yaw 0.017/0.033 | 0.574/1.075 deg | <2 deg |
| B1 foot z | 0.080m | <=0.10m |
| first zero contact | frame108 | > frame115，最好 > frame130 |
| frame100-145 contact | 45.7% | >=65% |
| post2 contact | 49.4% | >=60% |
| post2 obj_err max | 0.293m | <0.25m |
| post2 pelvis_z min | 0.663m | >=0.55m |

视觉成功标准：f130/f145 必须仍是持箱/托箱关系，不能只是机器人站稳但箱子落地。

## 执行顺序

1. 本地实现代码/config/scripts。
2. 本地 `py_compile`。
3. git commit + push。
4. 远程 `git pull --ff-only`，tmux session `E074` 中执行 `train_E074.sh parallel 0 1`。
5. 监控：`ssh spider-remote "tmux capture-pane -t E074 -p | tail -40"`；结果计数：`ssh spider-remote "ls /home/xiayb/pHRI_workspace/spider/workspace/core4d/results/E074/*.npz 2>/dev/null | wc -l"`。
6. 完成后 scp 回收 `workspace/core4d/results/E074` 和 `logs/E074`。
7. 本地运行/复查 `eval_E074.py`，写 log 95。

## 需要用户配合

如果默认 `spider-remote` 或远程 repo 路径不对，需要用户提供：

- SSH host alias。
- 远程 repo 绝对路径。
