# 远程实验执行指南

## 概述

当本地 GPU 资源不足以并行运行多个实验时，使用远程多卡机器并行执行。
本文档定义了标准流程，确保代码同步、日志收集、结果回传的可复现性。

## 远程机器配置

| 字段 | 值 |
|------|------|
| SSH alias | `spider-remote` |
| Host | 10.100.71.70 |
| Port | 58122 |
| User | xiayb |
| GPUs | 2x NVIDIA RTX 6000 Ada (48GB) |
| 项目路径 | `/home/xiayb/pHRI_workspace/spider` |
| SSH config | `~/.ssh/config` 中配置为 `spider-remote` |

## 标准流程

### 1. 代码同步 (本地 → 远程)

```bash
# 本地提交并推送
git add <files> && git commit -m "..." && git push

# 远程拉取
ssh spider-remote "cd /home/xiayb/pHRI_workspace/spider && git pull"
```

**注意**: gitignore 的文件 (npz 结果、视频) 用 scp 传输。

### 2. 编写远程运行脚本

在 `workspace/{exp_name}/scripts/` 下创建脚本，模板如下：

```bash
#!/bin/bash
# E{NNN}: <实验描述>
# GPU0: <任务列表> | GPU1: <任务列表>
set -e

OVERRIDE=core4d_e0XX
RESULTS_DIR=workspace/core4d/results/E0XX
LOGS_DIR=logs/E0XX
mkdir -p "$RESULTS_DIR" "$LOGS_DIR"

run_experiment() {
    local name=$1 gpu=$2 task=$3
    # ... 其余参数 ...
    echo "[$(date '+%H:%M:%S')] Starting $name on GPU $gpu"
    CUDA_VISIBLE_DEVICES=$gpu MUJOCO_GL=egl uv run examples/run_mjwp.py \
        +override=$OVERRIDE task=$task \
        video_output_path="$RESULTS_DIR/${name}.mp4" \
        > "$LOGS_DIR/${name}.log" 2>&1
    cp "<output_dir>/trajectory_mjwp_act.npz" "$RESULTS_DIR/${name}.npz"
    echo "[$(date '+%H:%M:%S')] Finished $name"
}

# GPU 0: 实验 A (sequential)
( run_experiment "A1" 0 task1; run_experiment "A2" 0 task2 ) &
PID0=$!

# GPU 1: 实验 B (sequential)
( run_experiment "B1" 1 task1; run_experiment "B2" 1 task2 ) &
PID1=$!

echo "=== Launched: GPU0 PID=$PID0 | GPU1 PID=$PID1 ==="
wait $PID0; echo "GPU0 done"
wait $PID1; echo "GPU1 done"
echo "=== All complete ==="
```

### 3. 部署并启动 (tmux)

```bash
# 同步代码
ssh spider-remote "cd /home/xiayb/pHRI_workspace/spider && git pull"

# 在 tmux 中启动 (不会因 SSH 断开而终止)
ssh spider-remote "cd /home/xiayb/pHRI_workspace/spider && \
    tmux new-session -d -s <session_name> && \
    tmux send-keys -t <session_name> 'bash <script_path>' Enter"
```

### 4. 监控进度

```bash
# 查看 tmux 输出
ssh spider-remote "tmux capture-pane -t <session_name> -p | tail -10"

# 检查已完成的结果数量
ssh spider-remote "ls /home/xiayb/pHRI_workspace/spider/<results_dir>/*.npz | wc -l"

# 查看某个实验的进度
ssh spider-remote "tail -1 /home/xiayb/pHRI_workspace/spider/<logs_dir>/<name>.log"
```

### 5. 回收结果

```bash
# SCP 结果 (npz + mp4)
scp spider-remote:/home/xiayb/pHRI_workspace/spider/<results_dir>/*.npz <local_results_dir>/
scp spider-remote:/home/xiayb/pHRI_workspace/spider/<results_dir>/*.mp4 <local_results_dir>/

# SCP 运行日志
scp spider-remote:/home/xiayb/pHRI_workspace/spider/<logs_dir>/*.log <local_logs_dir>/
```

### 6. 本地评估

```bash
# 对每个结果运行 eval
uv run workspace/core4d/scripts/eval/eval_comprehensive.py <task> <result.npz>
```

### 7. 离线可视化 (不需要重跑实验)

```bash
MUJOCO_GL=egl python workspace/hdmi_reproduce/scripts/render_trajectory_video.py \
    --scene <scene.xml> \
    --kin <kinematic.npz> \
    --phys <result.npz> \
    --output <output.mp4> --fps 30
```

## 并行策略

| 场景 | GPU 分配 | 预期时间 |
|------|---------|---------|
| 2 个 case, 2 GPU | 每 GPU 1 个 | ~6 min |
| 4 个 case, 2 GPU | 每 GPU 2 个 (串行) | ~12 min |
| 6 个 case, 2 GPU | 每 GPU 3 个 (串行) | ~18 min |
| Sweep 5 configs, 1 case | GPU0: 2+GPU1: 3 | ~18 min |

## 关键约束

1. **同一 GPU 上的实验必须串行** — MuJoCo Warp 占满显存
2. **video_output_path 必须不同** — 否则后跑的覆盖先跑的视频
3. **trajectory npz 也会互相覆盖** — 每个实验完成后立即 cp 到结果目录
4. **tmux session 不会主动结束** — 所有实验完成后手动 `tmux kill-session`

## 故障排除

| 问题 | 解决 |
|------|------|
| SSH connection refused | 检查 VPN/网络, `ssh -v spider-remote` |
| Permission denied | 重新 `ssh-copy-id` |
| CUDA OOM | 减少 num_samples (1024→512) |
| EGL error on exit | 无害,忽略 |
| tmux session 找不到 | `ssh spider-remote "tmux list-sessions"` |
| 脚本在远程报错 | 查看 `<logs_dir>/<name>.log` |
